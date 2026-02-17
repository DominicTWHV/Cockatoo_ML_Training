import pandas as pd

import torch

import numpy as np

from cockatoo_ml.registry import DataSplitConfig, DatasetColumns, RebalancingPolicy
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


def calculate_per_sample_weights(labels_df, class_weights):
    # labels_df: a dataframe where each column corresponds to a label and contains binary indicators (0/1) for whether that label is present in the sample
    # class_weights: a tensor of shape [num_classes] containing the weight for each class (computed by calculate_class_weights)

    label_matrix = labels_df.values.astype(float)
    class_weights_np = class_weights.numpy()
    
    # for each sample, get the maximum weight among its positive classes
    # if a sample has no positive labels, assign weight 1.0 (median)
    sample_weights = []
    for row in label_matrix:
        positive_indices = np.where(row > 0)[0]
        if len(positive_indices) > 0:
            weight = class_weights_np[positive_indices].max()
        else:
            weight = 1.0
        sample_weights.append(weight)
    
    weights = torch.tensor(sample_weights, dtype=torch.float32)
    logger.info(f"Per-sample weight stats - min: {weights.min():.4f}, max: {weights.max():.4f}, mean: {weights.mean():.4f}")
    
    return weights


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
