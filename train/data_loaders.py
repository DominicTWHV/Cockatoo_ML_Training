"""Dataset loading utilities for Constellation One."""

import os
import pandas as pd

from logger.context import data_processing_logger as logger


# text column in different sets
TEXT_CANDIDATES = ['text', 'comment_text', 'user_input', 'conversation', 'content', 'message']


def find_text_column(df):
    # find the first matching text candidate column in dframe
    for col in TEXT_CANDIDATES:
        if col in df.columns:
            logger.info(f"Found text column: {col}")
            return col
        
    return None


def load_phishing_dataset(base_dir="./data"):
    # load the phishing dataset from json file (combined_reduced.json) and find text col

    phish_path = os.path.join(base_dir, "phishing", "combined_reduced.json")
    if os.path.exists(phish_path):
        df_phish = pd.read_json(phish_path)
        text_col = find_text_column(df_phish)

        if text_col:
            df_phish = df_phish[[text_col]].dropna().rename(columns={text_col: 'text'})
            logger.info(f"Phishing loaded: {len(df_phish)} samples (text col: {text_col})")
            return df_phish, 'phishing'
        
        else:
            logger.warning("Phishing: No text column found - skipping")

    else:
        logger.warning("Phishing file not found")

    return None, None


def load_hate_speech_dataset(base_dir="./data"):
    # load the measuring hate speech dataset from parquet and find text and hate score cols
    hate_path = os.path.join(base_dir, "hate_speech_measuring", "data", "measuring-hate-speech.parquet")
    if not os.path.exists(hate_path):
        hate_path = os.path.join(base_dir, "hate_speech_measuring", "data", "train-00000-of-00001.parquet")

    if os.path.exists(hate_path):
        df_hate = pd.read_parquet(hate_path)

        if 'text' in df_hate.columns and 'hate_speech_score' in df_hate.columns:
            df_hate = df_hate[['text', 'hate_speech_score']].dropna()
            logger.info(f"Measuring hate loaded: {len(df_hate)} samples")
            return df_hate, 'hate_speech'
        
        else:
            logger.warning("Measuring hate: Expected columns 'text' and 'hate_speech_score' not found - skipping")

    else:
        logger.warning("Measuring hate parquet not found")

    return None, None


def load_tweet_hate_dataset(base_dir="./data"):
    # load the tweet_eval hate speech dataset from parquet and find text and label cols

    tweet_hate_path = os.path.join(base_dir, "tweet_eval", "hate", "train-00000-of-00001.parquet")
    if os.path.exists(tweet_hate_path):
        df_tweet = pd.read_parquet(tweet_hate_path)

        if 'text' in df_tweet.columns and 'label' in df_tweet.columns:
            df_tweet = df_tweet[['text', 'label']].dropna()
            logger.info(f"Tweet hate loaded: {len(df_tweet)} samples")
            return df_tweet, 'tweet_hate'
        
        else:
            logger.warning("Tweet hate: Expected 'text' and 'label' not found - skipping")

    return None, None


def load_toxicchat_dataset(base_dir="./data"):
    # load the toxic chat dataset from csv and find text and toxicity cols

    toxic_path = os.path.join(base_dir, "toxicchat0124", "data", "0124", "toxic-chat_annotation_train.csv") #latest toxic chat data is 0124
    if os.path.exists(toxic_path):
        df_toxic = pd.read_csv(toxic_path)
        text_col = find_text_column(df_toxic)

        if text_col and 'toxicity' in df_toxic.columns:
            df_toxic = df_toxic[[text_col, 'toxicity']].dropna()
            df_toxic = df_toxic.rename(columns={text_col: 'text'})
            logger.info(f"ToxicChat loaded: {len(df_toxic)} samples (text col: {text_col})")
            return df_toxic, 'toxicchat'
        
        else:
            logger.warning(f"ToxicChat: No suitable text column or 'toxicity' missing - skipping. Columns were: {df_toxic.columns.tolist()}")

    else:
        logger.warning("ToxicChat train csv not found")

    return None, None


def load_jigsaw_dataset(base_dir="./data"):
    # load the jigsaw bias dataset
    jigsaw_path = os.path.join(base_dir, "jigsaw_bias_mitigation", "train.csv")
    if os.path.exists(jigsaw_path):
        df_jig = pd.read_csv(jigsaw_path)
        text_col = find_text_column(df_jig)
        toxicity_col = next((c for c in df_jig.columns if 'toxic' in c.lower() or 'target' in c.lower()), None)

        if text_col and toxicity_col:
            df_jig = df_jig[[text_col, toxicity_col]].dropna()
            df_jig = df_jig.rename(columns={text_col: 'text', toxicity_col: 'toxicity'})
            logger.info(f"Jigsaw loaded: {len(df_jig)} samples")
            return df_jig, 'jigsaw'
        
        else:
            logger.warning(f"Jigsaw: No text/toxicity column found - skipping. Columns: {df_jig.columns.tolist()}")

    return None, None


def load_all_datasets(base_dir="./data"):
    # unifed function to all datasets
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
