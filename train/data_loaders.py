"""Dataset loading utilities for Constellation One."""

import os
import pandas as pd

from cockatoo_ml.registry import DatasetPaths, DatasetColumns, PathConfig
from cockatoo_ml.logger.context import data_processing_logger as logger


def find_text_column(df):
    # find the first matching text candidate column in dframe
    for col in DatasetColumns.TEXT_CANDIDATES:
        if col in df.columns:
            logger.info(f"Found text column: {col}")
            return col
        
    return None


def load_phishing_dataset(base_dir=None):
    # load the phishing dataset from json file (combined_reduced.json) and find text col
    if base_dir is None:
        base_dir = PathConfig.BASE_DATA_DIR

    phish_path = os.path.join(base_dir, DatasetPaths.PHISHING_DIR, DatasetPaths.PHISHING_FILE)
    if os.path.exists(phish_path):
        df_phish = pd.read_json(phish_path)
        text_col = find_text_column(df_phish)

        if text_col:
            df_phish = df_phish[[text_col]].dropna().rename(columns={text_col: DatasetColumns.TEXT_COL})
            logger.info(f"Phishing loaded: {len(df_phish)} samples (text col: {text_col})")
            return df_phish, 'phishing'
        
        else:
            logger.warning("Phishing: No text column found - skipping")

    else:
        logger.warning("Phishing file not found")

    return None, None


def load_hate_speech_dataset(base_dir=None):
    # load the measuring hate speech dataset from parquet and find text and hate score cols
    if base_dir is None:
        base_dir = PathConfig.BASE_DATA_DIR
        
    hate_path = os.path.join(base_dir, DatasetPaths.HATE_SPEECH_DIR, DatasetPaths.HATE_SPEECH_SUBDIR, DatasetPaths.HATE_SPEECH_FILE)
    if not os.path.exists(hate_path):
        hate_path = os.path.join(base_dir, DatasetPaths.HATE_SPEECH_DIR, DatasetPaths.HATE_SPEECH_SUBDIR, DatasetPaths.HATE_SPEECH_FALLBACK_FILE)

    if os.path.exists(hate_path):
        df_hate = pd.read_parquet(hate_path)

        if DatasetColumns.TEXT_COL in df_hate.columns and DatasetColumns.HATE_SPEECH_SCORE_COL in df_hate.columns:
            df_hate = df_hate[[DatasetColumns.TEXT_COL, DatasetColumns.HATE_SPEECH_SCORE_COL]].dropna()
            logger.info(f"Measuring hate loaded: {len(df_hate)} samples")
            return df_hate, 'hate_speech'
        
        else:
            logger.warning(f"Measuring hate: Expected columns '{DatasetColumns.TEXT_COL}' and '{DatasetColumns.HATE_SPEECH_SCORE_COL}' not found - skipping")

    else:
        logger.warning("Measuring hate parquet not found")

    return None, None


def load_tweet_hate_dataset(base_dir=None):
    # load the tweet_eval hate speech dataset from parquet and find text and label cols
    if base_dir is None:
        base_dir = PathConfig.BASE_DATA_DIR

    tweet_hate_path = os.path.join(base_dir, DatasetPaths.TWEET_EVAL_DIR, DatasetPaths.TWEET_EVAL_HATE_SUBDIR, DatasetPaths.TWEET_EVAL_FILE)
    if os.path.exists(tweet_hate_path):
        df_tweet = pd.read_parquet(tweet_hate_path)

        if DatasetColumns.TEXT_COL in df_tweet.columns and DatasetColumns.LABEL_COL in df_tweet.columns:
            df_tweet = df_tweet[[DatasetColumns.TEXT_COL, DatasetColumns.LABEL_COL]].dropna()
            logger.info(f"Tweet hate loaded: {len(df_tweet)} samples")
            return df_tweet, 'tweet_hate'
        
        else:
            logger.warning(f"Tweet hate: Expected '{DatasetColumns.TEXT_COL}' and '{DatasetColumns.LABEL_COL}' not found - skipping")

    return None, None


def load_toxicchat_dataset(base_dir=None):
    # load the toxic chat dataset from csv and find text and toxicity cols
    if base_dir is None:
        base_dir = PathConfig.BASE_DATA_DIR

    toxic_path = os.path.join(base_dir, DatasetPaths.TOXICCHAT_DIR, DatasetPaths.HATE_SPEECH_SUBDIR, DatasetPaths.TOXICCHAT_VERSION, DatasetPaths.TOXICCHAT_FILE)
    if os.path.exists(toxic_path):
        df_toxic = pd.read_csv(toxic_path)
        text_col = find_text_column(df_toxic)

        if text_col and DatasetColumns.TOXICITY_COL in df_toxic.columns:
            df_toxic = df_toxic[[text_col, DatasetColumns.TOXICITY_COL]].dropna()
            df_toxic = df_toxic.rename(columns={text_col: DatasetColumns.TEXT_COL})
            logger.info(f"ToxicChat loaded: {len(df_toxic)} samples (text col: {text_col})")
            return df_toxic, 'toxicchat'
        
        else:
            logger.warning(f"ToxicChat: No suitable text column or '{DatasetColumns.TOXICITY_COL}' missing - skipping. Columns were: {df_toxic.columns.tolist()}")

    else:
        logger.warning("ToxicChat train csv not found")

    return None, None


def load_jigsaw_dataset(base_dir=None):
    # load the jigsaw bias dataset
    if base_dir is None:
        base_dir = PathConfig.BASE_DATA_DIR
        
    jigsaw_path = os.path.join(base_dir, DatasetPaths.JIGSAW_DIR, DatasetPaths.JIGSAW_FILE)
    if os.path.exists(jigsaw_path):
        df_jig = pd.read_csv(jigsaw_path)
        text_col = find_text_column(df_jig)
        toxicity_col = next((c for c in df_jig.columns if 'toxic' in c.lower() or 'target' in c.lower()), None)

        if text_col and toxicity_col:
            df_jig = df_jig[[text_col, toxicity_col]].dropna()
            df_jig = df_jig.rename(columns={text_col: DatasetColumns.TEXT_COL, toxicity_col: DatasetColumns.TOXICITY_COL})
            logger.info(f"Jigsaw loaded: {len(df_jig)} samples")
            return df_jig, 'jigsaw'
        
        else:
            logger.warning(f"Jigsaw: No text/toxicity column found - skipping. Columns: {df_jig.columns.tolist()}")

    return None, None


def load_all_datasets(base_dir=None):
    # unifed function to all datasets
    if base_dir is None:
        base_dir = PathConfig.BASE_DATA_DIR
        
    logger.info("Loading and preprocessing datasets...")
    
    loaders = [
        load_phishing_dataset,
        load_hate_speech_dataset,
        load_tweet_hate_dataset,
        load_toxicchat_dataset,
        load_jigsaw_dataset,
    ]
    
    datasets = []
    for loader in loaders:
        df, dataset_type = loader(base_dir)
        if df is not None:
            datasets.append((df, dataset_type))
    
    return datasets
