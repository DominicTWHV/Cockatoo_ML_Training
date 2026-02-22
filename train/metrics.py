import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support

from cockatoo_ml.registry import MetricsConfig, LabelConfig
from cockatoo_ml.registry.column_mapping import DatasetColumnMapping

def compute_metrics(eval_pred, thresholds=None):
    # compute metrics from eval predictions
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    
    # always apply per-label thresholds if available, otherwise use default threshold for all labels (static single value)
    preds = np.zeros_like(probs, dtype=int)
    for i, label in enumerate(LabelConfig.ACTIVE_LABELS):
        # custom thresh overrides label thresh which overrides default thresh
        if thresholds is not None and label in thresholds:
            threshold = thresholds[label]

        else:
            threshold = DatasetColumnMapping.LABEL_THRESHOLDS.get(label, MetricsConfig.PREDICTION_THRESHOLD)
        preds[:, i] = (probs[:, i] > threshold).astype(int)

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
        strategy = MetricsConfig.AVERAGE_STRATEGY
        if strategy == 'macro':
            # unweighted mean over valid labels only
            p = float(np.mean([p_all[i] for i in valid_idx]))
            r = float(np.mean([r_all[i] for i in valid_idx]))
            f1 = float(np.mean([f1_all[i] for i in valid_idx]))
            
        else:
            # micro / weighted: delegate to sklearn on the valid label columns
            p, r, f1, _ = precision_recall_fscore_support(
                labels[:, valid_idx],
                preds[:, valid_idx],
                average=strategy,
                zero_division=MetricsConfig.ZERO_DIVISION
            )
            p, r, f1 = float(p), float(r), float(f1)
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
