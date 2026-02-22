import torch
import numpy as np

from sklearn.metrics import f1_score, precision_recall_fscore_support

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


def find_best_thresholds(logits, labels):
    # run a sweep over thresholds for each label and find the threshold that maximises F1 on the validation set
    # returns a dict of {label: {threshold: x, f1: y}}
    
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    steps = getattr(MetricsConfig, 'THRESHOLD_SEARCH_STEPS', 100)
    candidates = np.linspace(0.0, 1.0, steps + 2)[1:-1]  # exclude 0 and 1
    zero_division = getattr(MetricsConfig, 'ZERO_DIVISION', 0)

    results = {}
    for i, label in enumerate(LabelConfig.ACTIVE_LABELS):
        label_probs = probs[:, i]
        label_true = labels[:, i]

        best_threshold = candidates[0]
        best_f1 = -1.0

        for threshold in candidates:
            preds = (label_probs > threshold).astype(int)
            score = f1_score(label_true, preds, zero_division=zero_division)
            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        results[label] = {'threshold': float(best_threshold), 'f1': float(best_f1)}

    return results
