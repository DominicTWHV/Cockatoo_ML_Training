import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict
from pathlib import Path

from cockatoo_ml.registry import InferenceConfig, ModelConfig
from cockatoo_ml.logger.context import inference_api_server_logger as logger


class ThreatClassifier:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = InferenceConfig.DEFAULT_MODEL_PATH
            
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Loading model on device: {'cuda:0' if self.device == 0 else 'cpu'}")

        # try loading data with pipeline
        try:
            self.classifier = pipeline(
                "text-classification",
                model=model_path,
                device=self.device,
                torch_dtype=torch.float16 if self.device == 0 else None,  # memory saving on GPU
                return_all_scores=True,
                batch_size=InferenceConfig.BATCH_SIZE,
            )
            logger.info("Pipeline loaded successfully")

        except Exception as e:
            logger.warning(f"Pipeline load failed: {e} | Falling back to manual loading")
            # attempt manual loading files if pipeline fails
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            if self.device == 0:
                self.model = self.model.cuda()

            self.model.eval()
            self.classifier = None
            logger.info("Fallback manual model loaded")

        # Get label mapping from config
        self.id2label = self._get_id2label(model_path)
        logger.info(f"Detected labels: {list(self.id2label.values())}")

    def _get_id2label(self, model_path: str) -> Dict[int, str]:
        import json

        # read id2label from config if exists, otherwise just return {}
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                return config.get("id2label", {})
            
        return {}

    def predict(self, text: str) -> Dict:
        try:
            if self.classifier is not None:
                # predict results with model pipeline
                raw_results = self.classifier(text, truncation=InferenceConfig.TRUNCATION, max_length=InferenceConfig.INFERENCE_MAX_LENGTH)
                
                # extract scores and map to labels
                if isinstance(raw_results, list) and len(raw_results) > 0:
                    scores_list = raw_results[0] if isinstance(raw_results[0], list) else raw_results
                    probs = {item['label']: round(item['score'], 4) for item in scores_list}

                else:
                    return {"error": "Unexpected pipeline output format"}
                
            else:
                # manually handle tokenization and inference if pipeline failed to load
                inputs = self.tokenizer(text, return_tensors="pt", truncation=InferenceConfig.TRUNCATION, max_length=InferenceConfig.INFERENCE_MAX_LENGTH)
                if self.device == 0:
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    logits = self.model(**inputs).logits

                probs_raw = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                probs = {self.id2label.get(i, f"LABEL_{i}"): round(float(score), 4) for i, score in enumerate(probs_raw)}

            # construct response
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            top_label = sorted_probs[0][0] if sorted_probs else None
            max_score = sorted_probs[0][1] if sorted_probs else None

            return {
                "predictions": probs,
                "top_label": top_label,
                "max_score": max_score
            }

        except Exception as e:
            return {"error": str(e)}