import torch

from transformers import AutoModelForSequenceClassification

from cockatoo_ml.registry import ModelConfig, LabelConfig
from cockatoo_ml.logger.context import model_training_logger as logger


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


def load_model(model_name=None, num_labels=None):
    # load base model with torch float32 dtype
    if model_name is None:
        model_name = ModelConfig.BASE_MODEL_NAME
        
    if num_labels is None:
        num_labels = ModelConfig.NUM_LABELS
        
    label2id = {label: idx for idx, label in enumerate(LabelConfig.ACTIVE_LABELS)}
    id2label = {idx: label for label, idx in label2id.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=ModelConfig.PROBLEM_TYPE,
        torch_dtype=torch.float32,
        label2id=label2id,
        id2label=id2label,
    )
    return model
