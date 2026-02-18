import pandas as pd

from cockatoo_ml.logger.context import data_processing_logger as logger

class DatasetColumnMapping:
    # maps label categories to their corresponding columns in each dataset, along with thresholds for continuous labels

    # Phishing dataset column mapping
    PHISHING = {
        'text_col': 'text',
        'labels': {
            'scam': 'label'  # binary column - 1 if phishing, 0 otherwise
        }
    }
    
    # Measuring Hate Speech dataset column mapping
    HATE_SPEECH = {
        'text_col': 'text',
        'labels': {
            'hate_speech': 'hate_speech_score',  # continuous score, will threshold
            'violence': 'violence',  # binary: 1 if mentions violence
            'harassment': ['insult', 'humiliate', 'dehumanize'],  # multiple columns, OR them
            'dehumanization': 'dehumanize',  # direct binary column
            'status': 'status',  # binary: attacks person's status
        }
    }
    
    # Tweet Hate Evaluation dataset column mapping
    TWEET_HATE = {
        'text_col': 'text',
        'labels': {
            'hate_speech': 'label'  # 1 = hate, 0 = not hate
        }
    }
    
    # Toxic Chat dataset column mapping
    TOXICCHAT = {
        'text_col': 'user_input',
        'labels': {
            'toxicity': 'toxicity',  # continuous, will threshold
            'jailbreaking': 'jailbreaking'  # binary or continuous
        }
    }
    
    # Jigsaw Toxicity dataset column mapping
    JIGSAW = {
        'text_col': 'comment_text',
        'labels': {
            'toxicity': ['toxic', 'severe_toxic'],  # multiple columns, OR them
            'obscenity': 'obscene',  # binary: 1 if obscene
            'violence': 'threat',  # map 'threat' to violence category
            'harassment': 'insult',  # insults -> harassment
            'hate_speech': 'identity_hate'  # identity-based hate speech
        }
    }
    
    # unified label categories and their thresholds
    LABEL_THRESHOLDS = {
        'scam': 0.5,
        'violence': 0.5,
        'nsfw': 0.5,
        'harassment': 0.5,
        'hate_speech': 0.5,
        'toxicity': 0.5,
        'jailbreaking': 0.5,
        'dehumanization': 0.5,
        'obscenity': 0.5,
        'status': 0.5,
    }
    
    # Map dataset names to their column mapping
    DATASET_MAPPINGS = {
        'phishing': PHISHING,
        'hate_speech': HATE_SPEECH,
        'hate_speech_measuring': HATE_SPEECH,
        'tweet_hate': TWEET_HATE,
        'tweet_eval': TWEET_HATE,
        'toxicchat': TOXICCHAT,
        'jigsaw': JIGSAW,
    }

    # or: if any column indicates label is present, label is 1
    # and: all columns must indicate label is present to label as 1
    # max: take max value across columns (useful for continuous scores)
    # mean: take mean value across columns (useful for continuous scores)

    DATASET_MERGING_STRATEGY = 'or'
    
    @classmethod
    def get_mapping(cls, dataset_name):
        # get column mapping for a dataset
        normalized_name = dataset_name.lower().replace('_', '').replace(' ', '')
        
        for key, mapping in cls.DATASET_MAPPINGS.items():
            if key.lower().replace('_', '') == normalized_name:
                return mapping
        
        logger.warning(f"No mapping found for dataset '{dataset_name}'")
        return None
    
    @classmethod
    def list_mappings(cls):
        #list all mapping keys
        return list(cls.DATASET_MAPPINGS.keys())
    
    @classmethod
    def get_label_threshold(cls, label_name, default=0.5):
        # get threshold for a label
        return cls.LABEL_THRESHOLDS.get(label_name, default)


def merge_multi_column_labels(df, columns, merge_strategy=DatasetColumnMapping.DATASET_MERGING_STRATEGY):
    # strategy based merging multiple columns for a label:
    # 'or' (default): label is 1 if any column is 1
    # 'and': label is 1 only if all columns are 1
    # 'max': take max value across columns (useful for continuous scores)
    # 'mean': take mean value across columns (useful for continuous scores)

    if isinstance(columns, str):
        return df[columns].astype(int)
    
    cols = [df[col].astype(float) for col in columns if col in df.columns]
    
    if not cols:
        logger.warning(f"None of columns {columns} found in dataframe")
        return None
    
    if merge_strategy == 'or':
        # any column being 1 means label is 1
        return (sum(cols) > 0).astype(int)
    
    elif merge_strategy == 'and':
        # all columns must be 1 to label as 1
        return (sum(cols) == len(cols)).astype(int)
    
    elif merge_strategy == 'max':
        # take max value across columns (useful for continuous scores)
        return pd.concat(cols, axis=1).max(axis=1)
    
    elif merge_strategy == 'mean':
        # take mean value across columns (useful for continuous scores)
        return pd.concat(cols, axis=1).mean(axis=1)
    
    else:
        raise ValueError(f"Unknown merge strategy: {merge_strategy}")


def apply_threshold(series, threshold=0.5):
    # convert continuous scores to binary labels based on threshold
    return (series > threshold).astype(int)
