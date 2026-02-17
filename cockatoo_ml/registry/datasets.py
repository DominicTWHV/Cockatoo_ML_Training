import random

class DatasetSources:
    # huggingface datasets we download, for more info, see datasets.md
    
    DATASETS = [
        ("ealvaradob/phishing-dataset", "phishing"),
        ("ucberkeley-dlab/measuring-hate-speech", "hate_speech_measuring"),
        ("cardiffnlp/tweet_eval", "tweet_eval"),
        ("deepghs/nsfw_detect", "nsfw_detect"),
        ("lmsys/toxic-chat", "toxicchat0124"),
        ("tasksource/jigsaw_toxicity", "jigsaw_bias_mitigation"),
        ("KoalaAI/Text-Moderation-Multilingual", "multilingual_moderation"),
    ]
    
    # repo type for huggingface datasets
    REPO_TYPE = "dataset"
    
    # ignore these files when downloading from hf hub
    IGNORE_PATTERNS = ["*.gitattributes", "README.md"]


class DatasetPaths:
    # paths to download sets to and load from, relative to base data dir
    # generally there is no need to edit these, but they are provided here for easy reference and updates if needed
    
    # Phishing dataset
    PHISHING_FILE = "combined_reduced.json"
    PHISHING_DIR = "phishing"
    
    # Hate speech dataset
    HATE_SPEECH_FILE = "measuring-hate-speech.parquet"
    HATE_SPEECH_FALLBACK_FILE = "train-00000-of-00001.parquet"
    HATE_SPEECH_DIR = "hate_speech_measuring"
    HATE_SPEECH_SUBDIR = "data"
    
    # Tweet evaluation dataset
    TWEET_EVAL_DIR = "tweet_eval"
    TWEET_EVAL_HATE_SUBDIR = "hate"
    TWEET_EVAL_FILE = "train-00000-of-00001.parquet"
    
    # Toxic chat dataset
    TOXICCHAT_DIR = "toxicchat0124"
    TOXICCHAT_VERSION = "0124"
    TOXICCHAT_FILE = "toxic-chat_annotation_train.csv"
    
    # Jigsaw dataset
    JIGSAW_DIR = "jigsaw_bias_mitigation"
    JIGSAW_FILE = "train.csv"


class DatasetColumns:
    # text column names to search for
    TEXT_CANDIDATES = ['text', 'comment_text', 'user_input', 'conversation', 'content', 'message']
    
    # std column names
    TEXT_COL = 'text'
    LABELS_COL = 'labels'
    LABEL_COL = 'label'
    TOXICITY_COL = 'toxicity'
    HATE_SPEECH_SCORE_COL = 'hate_speech_score'


class RebalancingPolicy:
    # this is a mapping to map config policy names to config values
    # do not change unless you know what you are doing

    # in most cases you are looking to change the policy in DataSplitConfig.REBALANCING_POLICY, not here

    # rebalancing strategies for handling class imbalance
    OVERSAMPLING = "oversampling"      # upsample minority classes to match majority
    UNDERSAMPLING = "undersampling"    # downsample majority classes to match minority
    REWEIGHTING = "reweighting"        # assign higher loss weights to minority classes
    COMBINED = "combined"              # both oversampling and reweighting


class DataSplitConfig:
    # dataset split ratios
    TEST_SIZE = 0.2
    VAL_SIZE = 0.5  # Split from the test portion
    
    # random seed for reproducibility
    RANDOM_STATE = 42 #or use random.randint(0, 10000) for non-deterministic splits

    # class rebalancing policy (applies to training split only)
    # options: RebalancingPolicy.OVERSAMPLING, RebalancingPolicy.UNDERSAMPLING, RebalancingPolicy.REWEIGHTING, RebalancingPolicy.COMBINED, None
    
    # oversampling: upsamples minority label combinations to match the majority (increases dataset size)
    #   pros: leverages all data, relatively simple
    #   cons: can lead to overfitting on minority samples, increases training time
    #   when: when you have sufficient GPU resources and want to maximize use of available data, or when imbalance is severe and reweighting alone doesn't help

    # reweighting: calculates loss weights automatically based on class frequency
    #   pros: doesn't increase dataset size, mathematically principled, more efficient
    #   cons: may not fully address imbalance if some classes are extremely rare, can be sensitive to weight calculation method
    #   when: most cases

    # undersampling: downsamples majority label combinations to match the minority (reduces dataset size)
    #   pros: faster training, reduces dominance of majority label combos
    #   cons: discards data, may reduce model performance if data is scarce
    #   when: datasets are large and imbalance is high, or when training speed is critical

    # combined: applies both oversampling and reweighting
    #   pros: addresses imbalance from multiple angles
    #   cons: most complex, may be overkill for some use cases
    #   when: severe imbalance and you have resources to handle larger dataset

    # None: disables rebalancing (not recommended for imbalanced datasets)
    #   when: dataset is already (relatively) balanced, or as a baseline to compare against rebalancing strategies

    REBALANCING_POLICY = RebalancingPolicy.REWEIGHTING
    
    # weight calculation method for reweighting (when policy is REWEIGHTING or COMBINED)
    # options: "inverse_frequency", "effective_num", "sqrt_inverse"

    # inverse_frequency: weights = total_samples / (num_classes * class_count)
    #   pros: simple, intuitive, widely used
    #   cons: can produce extreme weights for very rare classes, may lead to instability during training
    
    # effective_num: weights = (1 - beta) / (1 - beta^n) where beta = (total_samples - 1) / total_samples
    #   pros: smoother scaling, handles rare classes better
    #   cons: may require tuning of beta param, does not account for instance-level difficulty (all samples of a class get same weight)

    # sqrt_inverse: weights = sqrt(total_samples / class_count)
    #   pros: gentler than inverse frequency, still emphasizes imbalance
    #   cons: may underemphasize rare classes compared to inverse frequency

    WEIGHT_CALCULATION = "inverse_frequency"
    
    # smoothing factor for weight calculation (prevents infinite weights for zero-count classes)
    # higher values make weights more uniform, lower values emphasize imbalance more
    WEIGHT_SMOOTHING = 1e-6

class DataDedupConfig:
    # policy for handling same normalized text appearing with different labels
    # options: "keep_all", "keep_first", "merge_labels", "drop_conflicts"

    # when to use:
    # keep_all: preserves all label combos for same text across different sets
    #   pros: retains all data
    #   cons: may introduce noise/conflicts, model sees same text with different labels and can be confused (but may also learn more nuanced patterns)

    # keep_first: keeps first occurrence of text and its labels, drops other combos
    #   pros: simple, reduces noise, model learns one label combo per text
    #   cons: may lose valuable data, model doesn't learn that same text can have different meanings

    # merge_labels: merges all labels for same text into a single row with combined labels
    #   pros: retains all data, model learns all label combos for same text
    #   cons: may create very complex label combinations that are hard for model to learn, can also cause problems if the labels are mutually exclusive (e.g. both toxic and non-toxic)

    # drop_conflicts: drops any text that appears with multiple distinct label sets
    #   pros: eliminates all label conflicts, model learns only clear-cut cases, zero ambiguity
    #   cons: may lose a lot of data
    
    SAME_TEXT_DIFFERENT_LABELS = "keep_first"
