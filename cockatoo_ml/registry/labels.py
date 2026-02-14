class LabelConfig:

    # label indices
    SCAM_IDX = 0
    VIOLENCE_IDX = 1
    NSFW_IDX = 2
    HARASSMENT_IDX = 3
    
    # label names (full set)
    LABEL_NAMES = ['scam', 'violence', 'nsfw', 'harassment']

    # active labels for this training run
    # only these labels are used for training/eval/output ordering.
    ACTIVE_LABELS = ['scam', 'harassment']
    
    # num of labels (active)
    NUM_LABELS = len(ACTIVE_LABELS)
    
    # Label thresholds for binary classification (these can be adjusted based on validation results or specific use case requirements)
    HATE_SPEECH_THRESHOLD = 0.5
    TOXICITY_THRESHOLD = 0.5
    
    @classmethod
    def make_labels(cls, scam=0, violence=0, nsfw=0, harassment=0):
        # build labels in the configured active-label order
        value_map = {
            'scam': scam,
            'violence': violence,
            'nsfw': nsfw,
            'harassment': harassment
        }
        return [value_map.get(label, 0) for label in cls.ACTIVE_LABELS]
    
    @classmethod
    def scam_label(cls):
        return cls.make_labels(scam=1)
    
    @classmethod
    def violence_label(cls):
        return cls.make_labels(violence=1)
    
    @classmethod
    def nsfw_label(cls):
        return cls.make_labels(nsfw=1)
    
    @classmethod
    def harassment_label(cls):
        return cls.make_labels(harassment=1)


class DatasetTypeConfig:
    # a map to convert dataset names into str for flexibility
    
    PHISHING = 'phishing'
    HATE_SPEECH = 'hate_speech'
    TWEET_HATE = 'tweet_hate'
    TOXICCHAT = 'toxicchat'
    JIGSAW = 'jigsaw'
