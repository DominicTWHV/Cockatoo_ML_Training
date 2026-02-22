import pandas as pd

import torch

import numpy as np

from cockatoo_ml.registry import DataSplitConfig, DatasetColumns, RebalancingPolicy, WeightCheckingPolicy, WeightRatioThresholds
from cockatoo_ml.logger.context import data_processing_logger as logger


def calculate_class_weights(labels_column, method="inverse_frequency", smoothing=None):
    # inverse_frequency: weights = total_samples / (num_classes * class_count)
    # effective_num: weights = (1 - beta) / (1 - beta^n) where beta = (total_samples - 1) / total_samples
    # sqrt_inverse: weights = sqrt(total_samples / class_count)

    # smoothing applies a small constant to class counts to prevent division by zero and excessively large weights for rare classes
    
    if smoothing is None:
        smoothing = DataSplitConfig.WEIGHT_SMOOTHING
    
    # convert labels to matrix format (num_samples x num_classes)
    label_matrix = pd.DataFrame(labels_column.tolist()).values
    
    num_samples, num_classes = label_matrix.shape
    class_counts = label_matrix.sum(axis=0)  # count positives per class
    
    logger.info(f"Class counts: {class_counts.tolist()}")
    
    if method == "inverse_frequency":
        # weights = total_samples / (num_classes * class_count)
        # this gives higher weight to underrepresented classes
        weights = (num_samples + smoothing) / (num_classes * (class_counts + smoothing))
        
    elif method == "effective_num":
        # effective number of samples: (1 - beta) / (1 - beta^n)
        # where beta controls how much to reduce weights for overrepresented classes
        beta = (num_samples - 1) / num_samples  # typical choice
        weights = (1 - beta) / (1 - np.power(beta, class_counts + smoothing))
        
    elif method == "sqrt_inverse":
        # gentler version of inverse frequency using sqrt to prevent excessively large weights
        weights = np.sqrt((num_samples + smoothing) / (class_counts + smoothing))
        
    else:
        raise ValueError(f"Unknown weight calculation method: {method}")
    
    # normalize weights to have mean of 1.0 for better scaling with loss
    weights = weights / weights.mean()
    
    logger.info(f"Weight calculation method: {method}")
    logger.info(f"Per-class weights: {weights.tolist()}")
    
    return torch.tensor(weights, dtype=torch.float32)


def rebalance_dataset(df, policy=None, random_state=None):
    # apply rebalancing policy to the df
    
    if policy is None:
        policy = DataSplitConfig.REBALANCING_POLICY
    
    if random_state is None:
        random_state = DataSplitConfig.RANDOM_STATE
    
    if policy is None or policy == "none":
        logger.info("No rebalancing policy applied")
        return df, None
    
    logger.info(f"Applying rebalancing policy: {policy}")
    
    if policy == RebalancingPolicy.OVERSAMPLING:
        df = _apply_oversampling(df, random_state)
        # recalculate weights after oversampling to reflect the new balanced distribution
        labels_series = df[DatasetColumns.LABELS_COL]
        class_weights = calculate_class_weights(
            labels_series,
            method=DataSplitConfig.WEIGHT_CALCULATION,
            smoothing=DataSplitConfig.WEIGHT_SMOOTHING
        )
        return df, class_weights

    elif policy == RebalancingPolicy.UNDERSAMPLING:
        df = _apply_undersampling(df, random_state)
        # calculate weights on the undersampled distribution for consistency
        labels_series = df[DatasetColumns.LABELS_COL]
        class_weights = calculate_class_weights(
            labels_series,
            method=DataSplitConfig.WEIGHT_CALCULATION,
            smoothing=DataSplitConfig.WEIGHT_SMOOTHING
        )
        return df, class_weights
    
    elif policy == RebalancingPolicy.REWEIGHTING:
        # calculate class weights on the original imbalanced distribution
        labels_series = df[DatasetColumns.LABELS_COL]
        class_weights = calculate_class_weights(
            labels_series,
            method=DataSplitConfig.WEIGHT_CALCULATION,
            smoothing=DataSplitConfig.WEIGHT_SMOOTHING
        )
        # for reweighting, we just return the weights without modifying the dataset
        return df, class_weights
    
    elif policy == RebalancingPolicy.COMBINED:
        df = _apply_oversampling(df, random_state)
        # recalculate weights after oversampling to reflect the new balanced distribution
        labels_series = df[DatasetColumns.LABELS_COL]
        class_weights = calculate_class_weights(
            labels_series,
            method=DataSplitConfig.WEIGHT_CALCULATION,
            smoothing=DataSplitConfig.WEIGHT_SMOOTHING
        )
        return df, class_weights
    
    else:
        raise ValueError(f"Unknown rebalancing policy: {policy}")


def _apply_oversampling(df, random_state):
    # upsamples minority classes by duplicating samples until all classes have the same number of samples as the majority class
    
    logger.info("Rebalancing training split via oversampling...")
    
    labels_tuple = df[DatasetColumns.LABELS_COL].map(tuple)
    df = df.copy()
    df["_labels_tuple"] = labels_tuple
    
    counts = df["_labels_tuple"].value_counts()
    if counts.empty:
        return df.drop(columns=["_labels_tuple"])
    
    max_count = counts.max()
    pre_rebalance_total = len(df)
    logger.info("Label-combo distribution before oversampling:")
    logger.info(f"{counts.to_dict()}")
    
    rebalanced_parts = []
    
    for label_combo, combo_count in counts.items():
        combo_df = df[df["_labels_tuple"] == label_combo]
        
        if combo_count < max_count:
            # upsample this class to match the max
            extra = combo_df.sample(
                n=max_count - combo_count,
                replace=True,
                random_state=random_state
            )
            combo_df = pd.concat([combo_df, extra], ignore_index=True)
        
        rebalanced_parts.append(combo_df)
    
    rebalanced_df = pd.concat(rebalanced_parts, ignore_index=True)
    rebalanced_df = rebalanced_df.drop(columns=["_labels_tuple"]).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)
    
    post_rebalance_total = len(rebalanced_df)
    added = post_rebalance_total - pre_rebalance_total
    
    # compute new distribution after rebalance
    post_counts = rebalanced_df[DatasetColumns.LABELS_COL].map(tuple).value_counts()
    
    logger.info(f"Oversampled: added {added} samples (total: {pre_rebalance_total} → {post_rebalance_total})")
    logger.info("Label-combo distribution after oversampling:")
    logger.info(f"{post_counts.to_dict()}")
    
    return rebalanced_df


def _apply_undersampling(df, random_state):
    # downsamples majority classes by removing samples until all classes have the same number of samples as the minority class

    logger.info("Rebalancing training split via undersampling...")

    labels_tuple = df[DatasetColumns.LABELS_COL].map(tuple)
    df = df.copy()
    df["_labels_tuple"] = labels_tuple

    counts = df["_labels_tuple"].value_counts()
    if counts.empty:
        return df.drop(columns=["_labels_tuple"])

    min_count = counts.min()
    pre_rebalance_total = len(df)
    logger.info("Label-combo distribution before undersampling:")
    logger.info(f"{counts.to_dict()}")

    rebalanced_parts = []

    for label_combo, combo_count in counts.items():
        combo_df = df[df["_labels_tuple"] == label_combo]

        if combo_count > min_count:
            combo_df = combo_df.sample(
                n=min_count,
                replace=False,
                random_state=random_state
            )

        rebalanced_parts.append(combo_df)

    rebalanced_df = pd.concat(rebalanced_parts, ignore_index=True)
    rebalanced_df = rebalanced_df.drop(columns=["_labels_tuple"]).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)

    post_rebalance_total = len(rebalanced_df)
    removed = pre_rebalance_total - post_rebalance_total

    # compute new distribution after rebalance
    post_counts = rebalanced_df[DatasetColumns.LABELS_COL].map(tuple).value_counts()

    logger.info(f"Undersampled: removed {removed} samples (total: {pre_rebalance_total} → {post_rebalance_total})")
    logger.info("Label-combo distribution after undersampling:")
    logger.info(f"{post_counts.to_dict()}")

    return rebalanced_df


def calculate_weight_ratio(class_weights):
    """
    Calculate the weight ratio (max_weight / min_weight) from class weights.
    
    Args:
        class_weights: torch tensor of shape [num_classes] with per-class weights
        
    Returns:
        float: ratio of max weight to min weight
    """
    weights_np = class_weights.numpy() if isinstance(class_weights, torch.Tensor) else class_weights
    max_weight = np.max(weights_np)
    min_weight = np.min(weights_np)
    
    # avoid division by zero
    if min_weight == 0:
        min_weight = 1e-8
    
    ratio = max_weight / min_weight
    return float(ratio)


def check_and_compensate_weight_ratio(class_weights, base_learning_rate=None, labels=None):
    
    result = {
        'weight_ratio': None,
        'policy': None,
        'adjusted_learning_rate': None,
        'warnings': [],
        'compensations_applied': [],
    }
    
    # calc weight ratio
    weight_ratio = calculate_weight_ratio(class_weights)
    result['weight_ratio'] = weight_ratio
    
    # get policy for this ratio
    policy = WeightCheckingPolicy.get_policy(weight_ratio)
    result['policy'] = policy
    
    # log weight information
    weights_np = class_weights.numpy() if isinstance(class_weights, torch.Tensor) else class_weights
    logger.info("-" * 50)
    logger.info(f"WEIGHT RATIO ANALYSIS - Status: {policy['status']}")
    logger.info(f"Weight Ratio (max/min): {weight_ratio:.2f}:1")
    logger.info(f"Description: {policy['description']}")
    logger.info(f"Min weight: {np.min(weights_np):.4f} | Max weight: {np.max(weights_np):.4f}")
    
    if labels is not None and len(labels) == len(weights_np):
        logger.info("Per-class weights:")
        for label, weight in zip(labels, weights_np):
            logger.info(f"  {label}: {weight:.4f}")
    
    # apply compensations based on policy
    if policy['status'] == 'SAFE':
        logger.info("No intervention needed - ratio is within safe range.")
        
    elif policy['status'] == 'CAUTION':
        # apply LR adjustment
        if base_learning_rate is not None and policy['apply_lr_adjustment']:
            adjusted_lr = base_learning_rate * policy.get('lr_multiplier', WeightRatioThresholds.CAUTION_LR_MULTIPLIER)
            result['adjusted_learning_rate'] = adjusted_lr
            result['compensations_applied'].append('learning_rate_adjustment')
            
            warning_msg = (
                f"⚠️  CAUTION: Weight ratio {weight_ratio:.2f}:1 is in caution range. "
                f"Automatically adjusting learning rate from {base_learning_rate:.2e} to {adjusted_lr:.2e}. "
                f"Consider using gradient clipping and monitoring loss closely."
            )
            logger.warning(warning_msg)
            result['warnings'].append(warning_msg)

        else:
            warning_msg = (
                f"⚠️  CAUTION: Weight ratio {weight_ratio:.2f}:1 is in caution range. "
                f"Requires lower learning rates (e.g., 1×10⁻⁵) and gradient clipping to stay stable."
            )
            logger.warning(warning_msg)
            result['warnings'].append(warning_msg)
    
    elif policy['status'] == 'DANGEROUS':
        # raise warnings only, no automatic compensation
        warning_msg = (
            f"⚠️  DANGEROUS: Weight ratio {weight_ratio:.2f}:1 is in dangerous range (1:20 to 1:50). "
            f"High risk of training collapse. Model likely to predict only majority/minority classes. "
            f"STRONGLY RECOMMENDED: Review dataset balance, consider different rebalancing strategy, "
            f"or collect more data for minority classes. No automatic compensation applied."
        )

        logger.warning(warning_msg)
        result['warnings'].append(warning_msg)
    
    elif policy['status'] == 'CRITICAL':
        warning_msg = (
            f"🔴 CRITICAL: Weight ratio {weight_ratio:.2f}:1 is in critical range (1:50 to 1:100). "
            f"Very high risk of divergence and NaN loss. Model may predict only majority class."
            f"URGENT: Consider data collection, different rebalancing strategy (e.g., combined oversampling + reweighting), "
            f"or adjust class weight calculation method. Training may fail or produce poor results."
        )
        logger.error(warning_msg)
        result['warnings'].append(warning_msg)
    
    elif policy['status'] == 'EXTREME':
        warning_msg = (
            f"🔴 EXTREME: Weight ratio {weight_ratio:.2f}:1 is in extreme range (>1:100). "
            f"CRITICAL RISK: Often results in NaN loss or complete divergence within the first few hundred steps. "
            f"URGENT INTERVENTION REQUIRED: Do NOT proceed with training without substantial dataset rebalancing or "
            f"alternative approaches (e.g., focal loss, class-balanced sampling). "
            f"Current configuration is almost certain to fail."
        )
        logger.error(warning_msg)
        result['warnings'].append(warning_msg)
    
    logger.info("-" * 50)
    
    return result

