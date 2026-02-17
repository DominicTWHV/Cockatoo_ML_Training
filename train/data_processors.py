import pandas as pd

from datasets import Dataset, DatasetDict

from sklearn.model_selection import train_test_split

from cockatoo_ml.registry import LabelConfig, DatasetTypeConfig, DataSplitConfig, DatasetColumns, DataDedupConfig, RebalancingPolicy
from cockatoo_ml.logger.context import data_processing_logger as logger

from train.rebalancing import rebalance_dataset

# the data processor takes in the loaded datasets, applies appropriate labels, combines them, and splits into train/val/test sets for training
def apply_labels_by_type(df, dataset_type):
    # apply appropriate labeling logic based on dataset type and return dataframe with text and labels columns
    
    if dataset_type == DatasetTypeConfig.PHISHING:
        df[DatasetColumns.LABELS_COL] = [LabelConfig.scam_label() for _ in range(len(df))]
        cols_to_keep = [DatasetColumns.TEXT_COL, DatasetColumns.LABELS_COL]
        # include image column if present

        if 'image' in df.columns:
            cols_to_keep.append('image')
            
        return df[cols_to_keep]
    
    elif dataset_type == DatasetTypeConfig.HATE_SPEECH:
        df['harassment'] = (df[DatasetColumns.HATE_SPEECH_SCORE_COL] > LabelConfig.HATE_SPEECH_THRESHOLD).astype(int)
        df[DatasetColumns.LABELS_COL] = df['harassment'].apply(lambda x: LabelConfig.make_labels(harassment=x))
        cols_to_keep = [DatasetColumns.TEXT_COL, DatasetColumns.LABELS_COL]
        if 'image' in df.columns:
            cols_to_keep.append('image')

        return df[cols_to_keep]
    
    elif dataset_type == DatasetTypeConfig.TWEET_HATE:
        df[DatasetColumns.LABELS_COL] = df[DatasetColumns.LABEL_COL].apply(lambda x: LabelConfig.make_labels(harassment=x))
        cols_to_keep = [DatasetColumns.TEXT_COL, DatasetColumns.LABELS_COL]
        if 'image' in df.columns:
            cols_to_keep.append('image')

        return df[cols_to_keep]
    
    elif dataset_type == DatasetTypeConfig.TOXICCHAT:
        df[DatasetColumns.LABELS_COL] = df[DatasetColumns.TOXICITY_COL].apply(lambda x: LabelConfig.make_labels(harassment=x))
        cols_to_keep = [DatasetColumns.TEXT_COL, DatasetColumns.LABELS_COL]
        if 'image' in df.columns:
            cols_to_keep.append('image')

        return df[cols_to_keep]
    
    elif dataset_type == DatasetTypeConfig.JIGSAW:
        df['harassment'] = (df[DatasetColumns.TOXICITY_COL] > LabelConfig.TOXICITY_THRESHOLD).astype(int)
        df[DatasetColumns.LABELS_COL] = df['harassment'].apply(lambda x: LabelConfig.make_labels(harassment=x))
        cols_to_keep = [DatasetColumns.TEXT_COL, DatasetColumns.LABELS_COL]
        if 'image' in df.columns:
            cols_to_keep.append('image')
            
        return df[cols_to_keep]
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def combine_datasets(datasets):
    # combine the datasets into one dataframe, applying appropriate labels and handling duplicates

    labeled_dfs = []
    
    # collect and label each dataframe
    for df, dataset_type in datasets:
        labeled_df = apply_labels_by_type(df, dataset_type)
        labeled_dfs.append(labeled_df)
    
    if not labeled_dfs:
        return None
    
    # concat and cleanup the combined dataframe
    combined_df = pd.concat(labeled_dfs, ignore_index=True).dropna(subset=[DatasetColumns.TEXT_COL])
    
    # normalize text for deduplication (lowercase and strip)
    combined_df["_temp_text_norm"] = combined_df[DatasetColumns.TEXT_COL].str.lower().str.strip()

    # sample the number of entries before actually removing duplicates for logging later on
    pre_dedupe_count = len(combined_df)

    # Create a helper column of sorted tuples to handle label order variations
    # ie: ["Action", "Comedy"] becomes ("Action", "Comedy") and ["Comedy", "Action"] also becomes ("Action", "Comedy") | (agnostic to order + tuple conversion for hashability)
    combined_df["_temp_labels_tuple"] = combined_df[DatasetColumns.LABELS_COL].apply(
        lambda x: tuple(sorted(x)) if isinstance(x, (list, set, tuple)) else (x,)
    )

    policy = DataDedupConfig.SAME_TEXT_DIFFERENT_LABELS
    logger.info(f"Deduplication policy for same text/different labels: {policy}")

    # policy based deduplication

    if policy == "keep_all":
        # keep separate rows for different label combinations
        combined_df = combined_df.drop_duplicates(
            subset=["_temp_text_norm", "_temp_labels_tuple"],
            keep='first'
        )

    elif policy == "keep_first":
        # keep first occurrence per text, discard other labels (reduces noise, model is only learning one label combo per text)
        combined_df = combined_df.drop_duplicates(
            subset=["_temp_text_norm"],
            keep='first'
        )

    elif policy == "merge_labels":
        # merge labels into a single row per text
        merged_rows = []
        for _, group in combined_df.groupby("_temp_text_norm", sort=False):
            merged_labels = sorted({label for labels in group["_temp_labels_tuple"] for label in labels})
            row = group.iloc[0].copy()
            row[DatasetColumns.LABELS_COL] = merged_labels
            merged_rows.append(row)

        combined_df = pd.DataFrame(merged_rows)

    elif policy == "drop_conflicts":
        # drop any text that appears with multiple distinct label sets
        label_variety = combined_df.groupby("_temp_text_norm")["_temp_labels_tuple"].nunique()
        conflict_texts = label_variety[label_variety > 1].index
        
        if len(conflict_texts) > 0:
            logger.info(f"Dropping {len(conflict_texts)} conflicting texts with multiple label sets")

        combined_df = combined_df[~combined_df["_temp_text_norm"].isin(conflict_texts)]
        combined_df = combined_df.drop_duplicates(
            subset=["_temp_text_norm", "_temp_labels_tuple"],
            keep='first'
        )

    else:
        raise ValueError(f"Unknown deduplication policy: {policy}")

    # clean up helper column
    combined_df = combined_df.drop(columns=["_temp_text_norm", "_temp_labels_tuple"]).reset_index(drop=True)
    
    post_dedupe_count = len(combined_df)

    # log the number of duplicates found and removed
    diff = pre_dedupe_count - post_dedupe_count
    if diff > 0:
        logger.info(f"Deduplicated {diff} entries (kept {post_dedupe_count}/{pre_dedupe_count})")
        
    else:
        logger.info(f"No duplicates found (kept {post_dedupe_count}/{pre_dedupe_count})")
    
    return combined_df


def split_dataset(combined_df, test_size=None, val_size=None, random_state=None):
    # split the dataset into train/val/test sets with stratification
    if test_size is None:
        test_size = DataSplitConfig.TEST_SIZE
    if val_size is None:
        val_size = DataSplitConfig.VAL_SIZE
    if random_state is None:
        random_state = DataSplitConfig.RANDOM_STATE

    train_df, temp_df = train_test_split(
        combined_df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=combined_df[DatasetColumns.LABELS_COL].apply(tuple)
    )

    # apply rebalancing policy to training split
    policy = DataSplitConfig.REBALANCING_POLICY
    if policy and (policy == RebalancingPolicy.OVERSAMPLING or policy == RebalancingPolicy.REWEIGHTING or policy == RebalancingPolicy.COMBINED):
        train_df, class_weights = rebalance_dataset(train_df, policy=policy, random_state=random_state)
    
    else:
        class_weights = None
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=val_size, 
        random_state=random_state, 
        stratify=temp_df[DatasetColumns.LABELS_COL].apply(tuple)
    )
    
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
        'validation': Dataset.from_pandas(val_df.reset_index(drop=True)),
        'test': Dataset.from_pandas(test_df.reset_index(drop=True))
    })
    
    # store class weights in dataset metadata for later access
    dataset.class_weights = class_weights
    
    return dataset


def print_dataset_stats(combined_df, dataset):
    # a function to show the user some stats about the combined dataset and the splits
    logger.info("="*30)
    logger.info("DATASET SUMMARY")
    logger.info("="*30)
    logger.info(f"Combined total: {len(combined_df)} samples")
    logger.info(f"Train: {len(dataset['train'])} | Validation: {len(dataset['validation'])} | Test: {len(dataset['test'])}")
    
    # show distribution of labels
    train_df = dataset['train'].to_pandas()
    labels_flat = pd.Series([l for sublist in train_df[DatasetColumns.LABELS_COL] for l in sublist])
    logger.info(f"Train class distribution ({'/'.join(LabelConfig.ACTIVE_LABELS)}):")
    logger.info(f"{labels_flat.value_counts()}")

    # per-label positive counts and rates
    label_matrix = pd.DataFrame(train_df[DatasetColumns.LABELS_COL].tolist(), columns=LabelConfig.ACTIVE_LABELS)
    positives = label_matrix.sum().astype(int)
    rates = (label_matrix.mean() * 100).round(2)
    logger.info(f"Train positive counts by label: {positives.to_dict()}")
    logger.info(f"Train positive rates by label (%): {rates.to_dict()}")
    
    # show adjusted class weights if rebalancing was applied
    class_weights = getattr(dataset, 'class_weights', None)
    if class_weights is not None:
        logger.info("-" * 30)
        logger.info("REBALANCING APPLIED - Adjusted Class Weights:")
        weights_dict = {label: float(weight) for label, weight in zip(LabelConfig.ACTIVE_LABELS, class_weights.tolist())}
        
        for label, weight in weights_dict.items():
            logger.info(f"  {label}: {weight:.4f}")

            if weight > DataSplitConfig.SAFETY_MAXIMUM_WEIGHT:
                logger.warning(f"  WARNING: Weight for label '{label}' exceeds safety maximum of {DataSplitConfig.SAFETY_MAXIMUM_WEIGHT} | It is no longer recommended to use rebalancing with this dataset without further adjustments | Consider changing the rebalancing policy or adjusting the weight calculation method")
        
        logger.info(f"Weight calculation method: {DataSplitConfig.WEIGHT_CALCULATION}")
        logger.info(f"Rebalancing policy: {DataSplitConfig.REBALANCING_POLICY}")

    else:
        logger.info("-" * 30)
        logger.info("No rebalancing applied - using default weights")

    logger.info("="*60)