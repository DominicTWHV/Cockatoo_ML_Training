import torch

from transformers import AutoModelForSequenceClassification

from logger.context import model_training_logger as logger


def compute_pos_weight(dataset):
    # compute pos weight for BCE loss based on label distribution
    all_labels = torch.cat([torch.tensor(l) for l in dataset['labels']])
    pos_weight = (all_labels.numel() - all_labels.sum()) / (all_labels.sum() + 1e-6)  # avoid div by zero
    logger.info(f"Positive weight (inverse frequency): {pos_weight:.2f}")
    return pos_weight


def load_model(model_name="microsoft/deberta-v3-base", num_labels=4):
    # load base model with torch float32 dtype
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        torch_dtype=torch.float32
    )
    return model
