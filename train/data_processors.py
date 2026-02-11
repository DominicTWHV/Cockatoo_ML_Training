import pandas as pd

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from logger.context import data_processing_logger as logger

# the data processor takes in the loaded datasets, applies appropriate labels, combines them, and splits into train/val/test sets for training
def make_labels(scam=0, violence=0, nsfw=0, harassment=0):
    # make 4 labels
    return [scam, violence, nsfw, harassment]


def apply_labels_by_type(df, dataset_type):
    # apply appropriate labeling logic based on dataset type and return dataframe with text and labels columns
    if dataset_type == 'phishing':
        df['labels'] = [make_labels(scam=1) for _ in range(len(df))]
        return df[['text', 'labels']]
    
    elif dataset_type == 'hate_speech':
        df['harassment'] = (df['hate_speech_score'] > 0.5).astype(int)
        df['labels'] = df['harassment'].apply(lambda x: make_labels(harassment=x))
        return df[['text', 'labels']]
    
    elif dataset_type == 'tweet_hate':
        df['labels'] = df['label'].apply(lambda x: make_labels(harassment=x))
        return df[['text', 'labels']]
    
    elif dataset_type == 'toxicchat':
        df['labels'] = df['toxicity'].apply(lambda x: make_labels(harassment=x))
        return df[['text', 'labels']]
    
    elif dataset_type == 'jigsaw':
        df['harassment'] = (df['toxicity'] > 0.5).astype(int)
        df['labels'] = df['harassment'].apply(lambda x: make_labels(harassment=x))
        return df[['text', 'labels']]
    
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
    
    combined_df = pd.concat(labeled_dfs, ignore_index=True).dropna(subset=['text'])
    combined_df['text'] = combined_df['text'].astype(str).str.strip()
    
    return combined_df


def split_dataset(combined_df, test_size=0.2, val_size=0.5, random_state=42):
    # split the dataset into train/val/test sets with stratification

    train_df, temp_df = train_test_split(
        combined_df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=combined_df['labels'].apply(tuple)
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=val_size, 
        random_state=random_state, 
        stratify=temp_df['labels'].apply(tuple)
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
    labels_flat = pd.Series([l for sublist in train_df['labels'] for l in sublist])
    logger.info("Train class distribution (scam/violence/nsfw/harassment):")
    logger.info(f"{labels_flat.value_counts()}")
