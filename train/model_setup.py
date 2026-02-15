import torch
import torch.nn as nn

from transformers import AutoModelForSequenceClassification, CLIPModel

from cockatoo_ml.registry import ModelConfig, LabelConfig, ModelType
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
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        
        # return in format expected by trainer
        return type('Output', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': None,
            'attentions': None
        })()


def compute_pos_weight(dataset):
    # compute pos weight for BCE loss based on label distribution
    labels = torch.tensor(dataset['labels'], dtype=torch.float32)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)

    num_samples = labels.shape[0]
    pos_counts = labels.sum(dim=0)
    neg_counts = num_samples - pos_counts

    pos_weight = neg_counts / (pos_counts + ModelConfig.EPSILON)

    # handle labels with zero positives to avoid exploding weights
    pos_weight = torch.where(pos_counts > 0, pos_weight, torch.ones_like(pos_weight))

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
    
    logger.info(f"Loading model type: {model_type}")
    logger.info(f"Model name: {model_name}")
    
    if model_type == ModelType.CLIP_VIT:
        # load clip classifier
        model = CLIPClassifier(model_name, num_labels)
        model.config.label2id = label2id
        model.config.id2label = id2label
        logger.info(f"Loaded CLIP model with {num_labels} labels")
        
    elif model_type == ModelType.DEBERTA:
        # load deberta classifier
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type=ModelConfig.PROBLEM_TYPE,
            torch_dtype=torch.float32,
            label2id=label2id,
            id2label=id2label,
        )
        logger.info(f"Loaded DeBERTa model with {num_labels} labels")
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
