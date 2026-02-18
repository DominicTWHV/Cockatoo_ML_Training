import pandas as pd

from datasets import Dataset, DatasetDict

from sklearn.model_selection import train_test_split

from cockatoo_ml.registry import LabelConfig, DataSplitConfig, DatasetColumns, DataDedupConfig, RebalancingPolicy
from cockatoo_ml.registry.dataset_label_mapping import get_dataset_label_mapping

from cockatoo_ml.logger.context import data_processing_logger as logger

from train.rebalancing import rebalance_dataset, check_and_compensate_weight_ratio

# the data processor takes in the loaded datasets, applies appropriate labels, combines them, and splits into train/val/test sets for training
def apply_labels_by_type(df, dataset_type):
    # expected raw label columns depend on dataset
    # these are extracted by data_loaders and need to be mapped to active labels
    
    cols_to_keep = [DatasetColumns.TEXT_COL]
    
    # get available label columns anything not text column
    label_cols = [c for c in df.columns if c != DatasetColumns.TEXT_COL]
    
    if not label_cols:
        logger.warning(f"No label columns found for {dataset_type}")
        df[DatasetColumns.LABELS_COL] = [[0] * len(LabelConfig.ACTIVE_LABELS) for _ in range(len(df))]
    else:
        # build label vector for each row using available labels
        def build_label_vector(row):
            label_dict = {}
            for label_col in label_cols:
                # label_col name is the label ('scam', 'toxicity', etc.)
                label_dict[label_col] = row[label_col]
            return LabelConfig.make_labels(**label_dict)
        
        df[DatasetColumns.LABELS_COL] = df.apply(build_label_vector, axis=1)
    
    cols_to_keep.append(DatasetColumns.LABELS_COL)
    
    # include image column if present
    if 'image' in df.columns:
        cols_to_keep.append('image')
    
    # track dataset source for label masking
    df['_dataset_source'] = dataset_type
    cols_to_keep.append('_dataset_source')
    
    return df[cols_to_keep]

def combine_datasets(datasets):
    # combine multiple datasets into one dataframe, applying labels and handling duplicates according to policy

    labeled_dfs = []
    mapping = get_dataset_label_mapping()
    
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
    # splits dataset into train/validation/test sets with stratification and masking
    if test_size is None:
        test_size = DataSplitConfig.TEST_SIZE
    if val_size is None:
        val_size = DataSplitConfig.VAL_SIZE
    if random_state is None:
        random_state = DataSplitConfig.RANDOM_STATE
    
    mapping = get_dataset_label_mapping()
    
    # add label mask column based on dataset source
    # this tells the trainer which labels should be evaluated for this sample
    def get_label_mask(dataset_source):
        try:
            dataset_labels = mapping.get_labels(dataset_source)
            return [1 if label in dataset_labels else 0 for label in LabelConfig.ACTIVE_LABELS]
        
        except KeyError:
            # if dataset source not registered, assume all labels are valid
            logger.warning(f"Unknown dataset source '{dataset_source}' - using all labels for masking")
            return [1] * len(LabelConfig.ACTIVE_LABELS)
    
    combined_df['label_mask'] = combined_df['_dataset_source'].apply(get_label_mask)
    
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
        
        logger.info(f"Weight calculation method: {DataSplitConfig.WEIGHT_CALCULATION}")
        logger.info(f"Rebalancing policy: {DataSplitConfig.REBALANCING_POLICY}")
        
        # apply dynamic weight ratio checking and compensation recommendations
        check_result = check_and_compensate_weight_ratio(
            class_weights=class_weights,
            base_learning_rate=None,  # LR not available in this function, just check status
            labels=LabelConfig.ACTIVE_LABELS
        )

    else:
        logger.info("-" * 30)
        logger.info("No rebalancing applied - using default weights")

    logger.info("="*60)