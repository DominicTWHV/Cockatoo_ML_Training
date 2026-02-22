import torch
import json

from typing import Dict, Union
from pathlib import Path

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, CLIPProcessor

from PIL import Image

from cockatoo_ml.registry import InferenceConfig, ModelConfig, ModelType
from cockatoo_ml.registry.column_mapping import DatasetColumnMapping

from cockatoo_ml.logger.context import inference_api_server_logger as logger


class ThreatClassifier:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = InferenceConfig.DEFAULT_MODEL_PATH
            
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Loading model on device: {'cuda:0' if self.device == 0 else 'cpu'}")

        # detect model type from config
        self.model_type = self._detect_model_type(model_path)
        logger.info(f"Detected model type: {self.model_type}")

        if self.model_type == ModelType.CLIP_VIT:
            # load CLIP model
            self._load_clip_model(model_path)
        else:
            # Load DeBERTa or other text-only models with pipeline
            self._load_text_model(model_path)

        # get label mapping from config
        self.id2label = self._get_id2label(model_path)
        logger.info(f"Detected labels: {list(self.id2label.values())}")

    def _detect_model_type(self, model_path: str) -> str:
        config_path = Path(model_path) / "config.json"

        if config_path.exists():

            with open(config_path) as f:
                config = json.load(f)
                if 'model_type' in config and 'clip' in config['model_type'].lower():
                    return ModelType.CLIP_VIT
                
        return ModelType.DEBERTA

    def _load_clip_model(self, model_path: str):
        #load clip model
        from train.model_setup import CLIPClassifier
        
        try:
            self.processor = CLIPProcessor.from_pretrained(ModelConfig.get_base_model_name())
            self.model = CLIPClassifier(ModelConfig.get_base_model_name(), ModelConfig.NUM_LABELS)
            
            # load trained weights
            state_dict_path = Path(model_path) / "pytorch_model.bin"
            if state_dict_path.exists():
                self.model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
            
            if self.device == 0:
                self.model = self.model.cuda()

            self.model.eval()
            self.classifier = None
            logger.info("CLIP model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

    def _load_text_model(self, model_path: str):
        
        # must use manual inference (not pipeline) because the text-classification pipeline uses softmax (single-label), not sigmoid (multi-label)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        ModelConfig.validate_attention_precision_compatibility()

        def _is_flash_attention_error(exc: Exception) -> bool:
            text = str(exc).lower()
            flash_markers = (
                "flash attention 2 only supports",
                "flash_attn",
                "flash_attn_2_cuda",
                "undefined symbol",
                "attn_implementation",
            )
            return any(marker in text for marker in flash_markers)

        attn_implementation = ModelConfig.ATTENTION_IMPLEMENTATION
        model_dtype = ModelConfig.get_dtype()

        if attn_implementation == 'flash_attention_2' and self.device == -1:
            logger.warning("flash_attention_2 requested on CPU; falling back to sdpa for inference.")
            attn_implementation = 'sdpa'

        if attn_implementation == 'flash_attention_2' and model_dtype == 'auto':
            if self.device == 0 and torch.cuda.is_bf16_supported():
                model_dtype = torch.bfloat16

            elif self.device == 0:
                model_dtype = torch.float16

            else:
                model_dtype = torch.float32
            logger.info(f"Resolved dtype='auto' to {model_dtype} for flash_attention_2 compatibility.")

        if self.device == -1 and model_dtype in {torch.float16, torch.bfloat16}:
            logger.warning("Half-precision dtype requested on CPU; falling back to float32 for inference.")
            model_dtype = torch.float32

        model_kwargs = {
            'attn_implementation': attn_implementation,
            'torch_dtype': model_dtype,
        }

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                **model_kwargs,
            )

        except TypeError:
            model_kwargs.pop('torch_dtype', None)
            model_kwargs['dtype'] = model_dtype
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                **model_kwargs,
            )

        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            if attn_implementation == 'flash_attention_2' and _is_flash_attention_error(exc):
                logger.warning(f"flash_attention_2 failed ({exc}); retrying with sdpa attention.")
                model_kwargs['attn_implementation'] = 'sdpa'
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    **model_kwargs,
                )

            else:
                raise

        if self.device == 0:
            self.model = self.model.cuda()

        self.model.eval()
        self.classifier = None
        logger.info("Text model loaded with manual inference for multi-label classification")

    def _get_id2label(self, model_path: str) -> Dict[int, str]:
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                return config.get("id2label", {})
        return {}

    def get_label_thresholds(self) -> Dict[str, float]:
        # grab label thresholds for inferencing
        thresholds = {}
        for label in self.id2label.values():
            thresholds[label] = DatasetColumnMapping.get_label_threshold(label, default=0.5)
        return thresholds

    def predict(self, text: str, image: Union[str, Image.Image, None] = None) -> Dict:
        # predict endpoint call with optional image support if clip
        # CLIP models: supports both text-only and text+image
        # DeBERTa: text-only (image is ignored)
        try:
            if self.model_type == ModelType.CLIP_VIT:
                return self._predict_clip(text, image)
            
            else:
                return self._predict_text(text)
            
        except Exception as e:
            return {"error": str(e)}

    def _predict_clip(self, text: str, image: Union[str, Image.Image, None] = None) -> Dict:
        # predict pipeline for clip model (has image support)
        inputs = self.processor(
            text=[text],
            images=[image] if image is not None else None,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        if self.device == 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # get probabilities and map to labels
        probs_raw = torch.sigmoid(logits).cpu().numpy()[0]
        probs = {self.id2label.get(str(i), f"LABEL_{i}"): round(float(score), 4) 
                for i, score in enumerate(probs_raw)}
        
        # construct response
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_label = sorted_probs[0][0] if sorted_probs else None
        max_score = sorted_probs[0][1] if sorted_probs else None
        
        return {
            "predictions": probs,
            "top_label": top_label,
            "max_score": max_score,
            "model_type": "clip-vit"
        }

    def _predict_text(self, text: str) -> Dict:
        # prediction inference for multi-label text classification using manual inference
        # this uses sigmoid activation (not softmax) to return independent scores for each label

        # tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=InferenceConfig.TRUNCATION, 
            max_length=InferenceConfig.INFERENCE_MAX_LENGTH
        )
        
        if self.device == 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # run inference
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # apply sigmoid for multi-label classification
        probs_raw = torch.sigmoid(logits).cpu().numpy()[0]
        probs = {self.id2label.get(str(i), f"LABEL_{i}"): round(float(score), 4) 
                for i, score in enumerate(probs_raw)}

        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_label = sorted_probs[0][0] if sorted_probs else None
        max_score = sorted_probs[0][1] if sorted_probs else None

        return {
            "predictions": probs,
            "top_label": top_label,
            "max_score": max_score,
            "model_type": "deberta"
        }