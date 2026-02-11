import torch

from transformers import AutoModelForSequenceClassification

from cockatoo_ml.registry import ModelConfig
from logger.context import model_training_logger as logger


def compute_pos_weight(dataset):
    # compute pos weight for BCE loss based on label distribution
    all_labels = torch.cat([torch.tensor(l) for l in dataset['labels']])
    pos_weight = (all_labels.numel() - all_labels.sum()) / (all_labels.sum() + ModelConfig.EPSILON)
    logger.info(f"Positive weight (inverse frequency): {pos_weight:.2f}")
    return pos_weight


def load_model(model_name=None, num_labels=None):
    # load base model with torch float32 dtype
    if model_name is None:
        model_name = ModelConfig.BASE_MODEL_NAME
    if num_labels is None:
        num_labels = ModelConfig.NUM_LABELS
        
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=ModelConfig.PROBLEM_TYPE,
        torch_dtype=torch.float32
    )
    return model
