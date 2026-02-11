import torch

from sklearn.metrics import precision_recall_fscore_support

def compute_metrics(eval_pred):
    # compute metrics from eval predictions
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy().astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    return {
        'precision': p,
        'recall': r,
        'f1': f1
    }
