import os
import pandas as pd

from cockatoo_ml.registry import DatasetPaths, DatasetColumns, PathConfig
from cockatoo_ml.registry.column_mapping import DatasetColumnMapping, merge_multi_column_labels, apply_threshold

from cockatoo_ml.logger.context import data_processing_logger as logger


def extract_labels_from_df(df, mapping, dataset_name):
    # extract and merge labels from dataframe using column mapping

    if mapping is None:
        logger.error(f"No mapping found for {dataset_name}")
        return None
    
    text_col = mapping.get('text_col')
    label_specs = mapping.get('labels', {})
    
    # check text column exists
    if text_col not in df.columns:
        logger.warning(f"{dataset_name}: Text column '{text_col}' not found. Available: {df.columns.tolist()}")
        return None
    
    # start with text column
    result_df = df[[text_col]].copy()
    result_df = result_df.rename(columns={text_col: DatasetColumns.TEXT_COL})
    
    # extract each label
    for label_name, col_spec in label_specs.items():
        if isinstance(col_spec, list):

            # multiple columns to merge
            missing_cols = [c for c in col_spec if c not in df.columns]
            available_cols = [c for c in col_spec if c in df.columns]
            
            if not available_cols:
                logger.warning(f"{dataset_name}: Label '{label_name}' columns {col_spec} not found")
                result_df[label_name] = 0

            else:
                if missing_cols:
                    logger.info(f"{dataset_name}: Using available columns for '{label_name}': {available_cols}")
                
                # merge with OR strategy (any column = 1 means label is 1) | or is default
                result_df[label_name] = merge_multi_column_labels(df, available_cols, 'or')
        else:
            # single column
            if col_spec not in df.columns:
                logger.warning(f"{dataset_name}: Label column '{col_spec}' for '{label_name}' not found")
                result_df[label_name] = 0

            else:
                # apply threshold for continuous columns, otherwise convert to int
                if df[col_spec].dtype in [float, 'float32', 'float64']:
                    threshold = DatasetColumnMapping.get_label_threshold(label_name)
                    result_df[label_name] = apply_threshold(df[col_spec], threshold)

                else:
                    result_df[label_name] = df[col_spec].astype(int)
    
    return result_df


def load_phishing_dataset(base_dir=None):
    if base_dir is None:
        base_dir = PathConfig.BASE_DATA_DIR

    phish_path = os.path.join(base_dir, DatasetPaths.PHISHING_DIR, DatasetPaths.PHISHING_FILE)
    if os.path.exists(phish_path):
        df_phish = pd.read_json(phish_path)
        logger.info(f"Phishing raw columns: {df_phish.columns.tolist()}")
        
        mapping = DatasetColumnMapping.get_mapping('phishing')
        df_phish = extract_labels_from_df(df_phish, mapping, 'phishing')
        
        if df_phish is not None:
            df_phish = df_phish.dropna(subset=[DatasetColumns.TEXT_COL])
            logger.info(f"Phishing loaded: {len(df_phish)} samples")
            return df_phish, 'phishing'
    
    else:
        logger.warning("Phishing file not found")

    return None, None


def load_hate_speech_dataset(base_dir=None):
    if base_dir is None:
        base_dir = PathConfig.BASE_DATA_DIR
        
    hate_path = os.path.join(base_dir, DatasetPaths.HATE_SPEECH_DIR, DatasetPaths.HATE_SPEECH_SUBDIR, DatasetPaths.HATE_SPEECH_FILE)
    if not os.path.exists(hate_path):
        hate_path = os.path.join(base_dir, DatasetPaths.HATE_SPEECH_DIR, DatasetPaths.HATE_SPEECH_SUBDIR, DatasetPaths.HATE_SPEECH_FALLBACK_FILE)

    if os.path.exists(hate_path):
        df_hate = pd.read_parquet(hate_path)
        logger.info(f"Measuring hate raw columns: {df_hate.columns.tolist()}")
        
        mapping = DatasetColumnMapping.get_mapping('hate_speech')
        df_hate = extract_labels_from_df(df_hate, mapping, 'hate_speech')
        
        if df_hate is not None:
            df_hate = df_hate.dropna(subset=[DatasetColumns.TEXT_COL])
            logger.info(f"Measuring hate loaded: {len(df_hate)} samples")
            return df_hate, 'hate_speech'

    else:
        logger.warning("Measuring hate parquet not found")

    return None, None


def load_tweet_hate_dataset(base_dir=None):
    if base_dir is None:
        base_dir = PathConfig.BASE_DATA_DIR

    tweet_hate_path = os.path.join(base_dir, DatasetPaths.TWEET_EVAL_DIR, DatasetPaths.TWEET_EVAL_HATE_SUBDIR, DatasetPaths.TWEET_EVAL_FILE)
    if os.path.exists(tweet_hate_path):
        df_tweet = pd.read_parquet(tweet_hate_path)
        logger.info(f"Tweet hate raw columns: {df_tweet.columns.tolist()}")
        
        mapping = DatasetColumnMapping.get_mapping('tweet_hate')
        df_tweet = extract_labels_from_df(df_tweet, mapping, 'tweet_hate')
        
        if df_tweet is not None:
            df_tweet = df_tweet.dropna(subset=[DatasetColumns.TEXT_COL])
            logger.info(f"Tweet hate loaded: {len(df_tweet)} samples")
            return df_tweet, 'tweet_hate'

    return None, None


def load_tweet_emotion_dataset(base_dir=None):
    if base_dir is None:
        base_dir = PathConfig.BASE_DATA_DIR

    tweet_emotion_path = os.path.join(base_dir, DatasetPaths.TWEET_EVAL_DIR, DatasetPaths.TWEET_EVAL_EMOTION_SUBDIR, DatasetPaths.TWEET_EVAL_FILE)
    if os.path.exists(tweet_emotion_path):
        df_emotion = pd.read_parquet(tweet_emotion_path)
        logger.info(f"Tweet emotion raw columns: {df_emotion.columns.tolist()}")

        # emotion config uses a multiclass label: 0=anger, 1=joy, 2=optimism, 3=sadness
        # convert to boolean: 1 if anger (label==0), 0 otherwise
        if 'label' in df_emotion.columns:
            df_emotion['anger'] = (df_emotion['label'] == 0).astype(int)
            
        else:
            logger.warning("Tweet emotion: 'label' column not found, skipping")
            return None, None

        mapping = DatasetColumnMapping.get_mapping('tweet_emotion')
        df_emotion = extract_labels_from_df(df_emotion, mapping, 'tweet_emotion')

        if df_emotion is not None:
            df_emotion = df_emotion.dropna(subset=[DatasetColumns.TEXT_COL])
            logger.info(f"Tweet emotion loaded: {len(df_emotion)} samples")
            return df_emotion, 'tweet_emotion'

    else:
        logger.warning("Tweet emotion parquet not found")

    return None, None


def load_toxicchat_dataset(base_dir=None):
    if base_dir is None:
        base_dir = PathConfig.BASE_DATA_DIR

    toxic_path = os.path.join(base_dir, DatasetPaths.TOXICCHAT_DIR, DatasetPaths.HATE_SPEECH_SUBDIR, DatasetPaths.TOXICCHAT_VERSION, DatasetPaths.TOXICCHAT_FILE)
    if os.path.exists(toxic_path):
        df_toxic = pd.read_csv(toxic_path)
        logger.info(f"ToxicChat raw columns: {df_toxic.columns.tolist()}")
        
        mapping = DatasetColumnMapping.get_mapping('toxicchat')
        df_toxic = extract_labels_from_df(df_toxic, mapping, 'toxicchat')
        
        if df_toxic is not None:
            df_toxic = df_toxic.dropna(subset=[DatasetColumns.TEXT_COL])
            logger.info(f"ToxicChat loaded: {len(df_toxic)} samples")
            return df_toxic, 'toxicchat'
        
        else:
            logger.warning(f"ToxicChat: Label extraction failed")

    else:
        logger.warning("ToxicChat train csv not found")

    return None, None


def load_jigsaw_dataset(base_dir=None):
    if base_dir is None:
        base_dir = PathConfig.BASE_DATA_DIR
        
    jigsaw_path = os.path.join(base_dir, DatasetPaths.JIGSAW_DIR, DatasetPaths.JIGSAW_FILE)
    if os.path.exists(jigsaw_path):
        df_jig = pd.read_csv(jigsaw_path)
        logger.info(f"Jigsaw raw columns: {df_jig.columns.tolist()}")
        
        mapping = DatasetColumnMapping.get_mapping('jigsaw')
        df_jig = extract_labels_from_df(df_jig, mapping, 'jigsaw')
        
        if df_jig is not None:
            df_jig = df_jig.dropna(subset=[DatasetColumns.TEXT_COL])
            logger.info(f"Jigsaw loaded: {len(df_jig)} samples")
            return df_jig, 'jigsaw'
        
        else:
            logger.warning(f"Jigsaw: Label extraction failed")

    return None, None


def load_all_datasets(base_dir=None):
    
    loaders = [
        load_phishing_dataset,
        load_hate_speech_dataset,
        load_tweet_hate_dataset,
        load_tweet_emotion_dataset,
        load_toxicchat_dataset,
        load_jigsaw_dataset,
    ]
    
    datasets = []
    for loader in loaders:
        df, dataset_type = loader(base_dir)
        if df is not None:
            datasets.append((df, dataset_type))
    
    return datasets
