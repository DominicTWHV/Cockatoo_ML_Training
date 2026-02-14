import torch

from sklearn.metrics import precision_recall_fscore_support

from cockatoo_ml.registry import MetricsConfig, LabelConfig

def compute_metrics(eval_pred):
    # compute metrics from eval predictions
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > MetricsConfig.PREDICTION_THRESHOLD).numpy().astype(int)

    # per-label metrics
    p_all, r_all, f1_all, support = precision_recall_fscore_support(
        labels,
        preds,
        average=None,
        zero_division=MetricsConfig.ZERO_DIVISION
    )

    if MetricsConfig.INCLUDE_EMPTY_LABELS:
        valid_idx = list(range(len(f1_all)))
    else:
        valid_idx = [i for i, s in enumerate(support) if s > 0]

    if valid_idx:
        p = float(torch.tensor([p_all[i] for i in valid_idx]).mean().item())
        r = float(torch.tensor([r_all[i] for i in valid_idx]).mean().item())
        f1 = float(torch.tensor([f1_all[i] for i in valid_idx]).mean().item())
    else:
        p, r, f1 = 0.0, 0.0, 0.0

    metrics = {
        'precision': p,
        'recall': r,
        'f1': f1
    }

    # include per-label metrics for visibility
    for i, label in enumerate(LabelConfig.ACTIVE_LABELS):
        metrics[f'precision_{label}'] = float(p_all[i]) if i < len(p_all) else 0.0
        metrics[f'recall_{label}'] = float(r_all[i]) if i < len(r_all) else 0.0
        metrics[f'f1_{label}'] = float(f1_all[i]) if i < len(f1_all) else 0.0

    return metrics
