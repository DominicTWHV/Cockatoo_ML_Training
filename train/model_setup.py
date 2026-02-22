import torch
import torch.nn as nn

from transformers import AutoModelForSequenceClassification, CLIPModel

from train.losses import BCEWithLogitsLoss

from cockatoo_ml.registry import ModelConfig, LabelConfig, ModelType, DataSplitConfig
from cockatoo_ml.logger.context import model_training_logger as logger


class CLIPClassifier(nn.Module):
    # CLIP-based multi-label classifier
    # uses CLIP's vision and text encoders with a custom classification head

    def __init__(self, clip_model_name, num_labels):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        # get embedding dimension from config
        self.embedding_dim = self.clip.config.projection_dim
        
        # classification head for multi-label
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, ModelConfig.CLIP_PROJECTION_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ModelConfig.CLIP_PROJECTION_DIM, num_labels)
        )
        
        # store num_labels for compatibility with Trainer
        self.num_labels = num_labels
        self.config = self.clip.config
        self.config.num_labels = num_labels
        self.config.problem_type = ModelConfig.PROBLEM_TYPE
        
    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None, **kwargs):
        # forward pass with image/text support
        # Get CLIP outputs
        if pixel_values is not None:
            outputs = self.clip(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_dict=True
            )
            # add the image and text embeddings together
            embeddings = (outputs.image_embeds + outputs.text_embeds) / 2

        else:
            # use text encoder only if no images (pixel_values)
            outputs = self.clip.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            embeddings = outputs
        
        # classification
        logits = self.classifier(embeddings)
        
        # compute loss if labels provided
        loss = None
        if labels is not None:
            # multi label BCE loss
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        
        # return in format expected by trainer
        return type('Output', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': None,
            'attentions': None
        })()


def compute_pos_weight(dataset, class_weights=None):
    # compute class weights for imbalanced sets for BCE loss

    # if weights already computed during rebalancing, use those
    if class_weights is not None:
        logger.info(f"Using pre-computed class weights from rebalancing")
        return class_weights
    
    # fallback: compute weights from dataset
    logger.info(f"Computing class weights from dataset (no pre-computed weights provided)")
    labels = torch.tensor(dataset['labels'], dtype=torch.float32)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)

    num_samples = labels.shape[0]
    pos_counts = labels.sum(dim=0)
    neg_counts = num_samples - pos_counts

    pos_weight = neg_counts / (pos_counts + ModelConfig.EPSILON)

    # handle labels with zero positives to avoid exploding weights
    pos_weight = torch.where(pos_counts > 0, pos_weight, torch.ones_like(pos_weight))

    # optionally scale extreme weights to avoid unstable gradients while preserving ratios
    max_pos_weight = DataSplitConfig.POS_WEIGHT_MAX
    if max_pos_weight is not None and max_pos_weight > 0:
        current_max = float(pos_weight.max().item()) if pos_weight.numel() > 0 else 0.0
        if current_max > max_pos_weight:
            scale = max_pos_weight / current_max
            logger.warning(
                f"Scaling pos_weight by {scale:.6f} to cap max at {max_pos_weight}."
            )
            pos_weight = pos_weight * scale

    logger.info(f"Positive weights (per label): {pos_weight.tolist()}")
    return pos_weight


def load_model(model_name=None, num_labels=None, model_type=None):
    # load model based on config
    # use config defaults if not provided
    if model_type is None:
        model_type = ModelConfig.MODEL_TYPE
        
    if model_name is None:
        model_name = ModelConfig.get_base_model_name()
        
    if num_labels is None:
        num_labels = ModelConfig.NUM_LABELS
    
    label2id = {label: idx for idx, label in enumerate(LabelConfig.ACTIVE_LABELS)}
    id2label = {idx: label for label, idx in label2id.items()}
    model_dtype = ModelConfig.get_dtype()
    ModelConfig.validate_attention_precision_compatibility()
    
    logger.info(f"Loading model type: {model_type}")
    logger.info(f"Model name: {model_name}")

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

    def _load_transformer_model():
        attn_implementation = ModelConfig.ATTENTION_IMPLEMENTATION
        effective_dtype = model_dtype

        if attn_implementation == 'flash_attention_2' and not torch.cuda.is_available():
            logger.warning("flash_attention_2 requested but CUDA is unavailable; falling back to sdpa.")
            attn_implementation = 'sdpa'

        if attn_implementation == 'flash_attention_2' and model_dtype == 'auto':
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                effective_dtype = torch.bfloat16
            else:
                effective_dtype = torch.float16
            logger.info(
                f"Resolved dtype='auto' to {effective_dtype} for flash_attention_2 compatibility."
            )

        # prefer torch_dtype for broad transformers compatibility.
        # keep a fallback to dtype for forward-compat with newer APIs if needed.
        model_kwargs = {
            'num_labels': num_labels,
            'problem_type': ModelConfig.PROBLEM_TYPE,
            'label2id': label2id,
            'id2label': id2label,
            'attn_implementation': attn_implementation,
            'torch_dtype': effective_dtype,
        }

        try:
            return AutoModelForSequenceClassification.from_pretrained(
                model_name,
                **model_kwargs,
            )
        except TypeError:
            # fallback path for potential API changes where torch_dtype is renamed
            model_kwargs.pop('torch_dtype', None)
            model_kwargs['dtype'] = effective_dtype
            return AutoModelForSequenceClassification.from_pretrained(
                model_name,
                **model_kwargs,
            )
        
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            if attn_implementation == 'flash_attention_2' and _is_flash_attention_error(exc):
                logger.warning(f"flash_attention_2 failed ({exc}); retrying with sdpa attention.")

                fallback_kwargs = dict(model_kwargs)
                fallback_kwargs['attn_implementation'] = 'sdpa'
                return AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    **fallback_kwargs,
                )
            
            raise
    
    if model_type == ModelType.CLIP_VIT:
        # load clip classifier
        model = CLIPClassifier(model_name, num_labels)
        model.config.label2id = label2id
        model.config.id2label = id2label
        logger.info(f"Loaded CLIP model with {num_labels} labels")
        
    elif model_type == ModelType.DEBERTA:
        # load deberta classifier
        model = _load_transformer_model()
        logger.info(f"Loaded DeBERTa model with {num_labels} labels")

    elif model_type == ModelType.MODERNBERT:
        # load modernbert classifier (uses sdpa attention via ATTENTION_IMPLEMENTATION)
        model = _load_transformer_model()
        logger.info(f"Loaded ModernBERT model with {num_labels} labels")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0
    logger.info(
        f"Model parameters: total={total_params:,}, trainable={trainable_params:,} ({trainable_pct:.2f}%)"
    )
    if trainable_pct < 90.0:
        logger.warning(
            f"Only {trainable_pct:.2f}% of parameters are trainable. This may cause unusually low VRAM usage and weak fine-tuning."
        )

    return model
