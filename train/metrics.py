import torch

from sklearn.metrics import precision_recall_fscore_support

from cockatoo_ml.registry import MetricsConfig

def compute_metrics(eval_pred):
    # compute metrics from eval predictions
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > MetricsConfig.PREDICTION_THRESHOLD).numpy().astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average=MetricsConfig.AVERAGE_STRATEGY, 
        zero_division=MetricsConfig.ZERO_DIVISION
    )
    return {
        'precision': p,
        'recall': r,
        'f1': f1
    }
