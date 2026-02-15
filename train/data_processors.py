import pandas as pd

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from cockatoo_ml.registry import LabelConfig, DatasetTypeConfig, DataSplitConfig, DatasetColumns
from cockatoo_ml.logger.context import data_processing_logger as logger

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
    # combine all datasets into a single dataframe
    labeled_dfs = []
    
    for df, dataset_type in datasets:
        labeled_df = apply_labels_by_type(df, dataset_type)
        labeled_dfs.append(labeled_df)
    
    if not labeled_dfs:
        return None
    
    combined_df = pd.concat(labeled_dfs, ignore_index=True).dropna(subset=[DatasetColumns.TEXT_COL])
    combined_df[DatasetColumns.TEXT_COL] = combined_df[DatasetColumns.TEXT_COL].astype(str).str.strip()
    combined_df["_labels_tuple"] = combined_df[DatasetColumns.LABELS_COL].apply(tuple)
    combined_df = combined_df.drop_duplicates(subset=[DatasetColumns.TEXT_COL, "_labels_tuple"]).drop(columns=["_labels_tuple"])
    
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
    
    return dataset


def print_dataset_stats(combined_df, dataset):
    # a function to show the user some stats about the combined dataset and the splits
    logger.info(f"Combined total: {len(combined_df)} samples")
    logger.info(f"Train: {len(dataset['train'])} | Validation: {len(dataset['validation'])} | Test: {len(dataset['test'])}")
    
    # show distribution of labels
    train_df = dataset['train'].to_pandas()
    labels_flat = pd.Series([l for sublist in train_df[DatasetColumns.LABELS_COL] for l in sublist])
    logger.info(f"Train class distribution ({'/'.join(LabelConfig.ACTIVE_LABELS)}):")
    logger.info(f"{labels_flat.value_counts()}")