class LabelConfig:
    # all available labels in the system
    ALL_LABELS = ['scam', 'violence', 'nsfw', 'harassment', 'hate_speech', 'toxicity', 'obscenity', 'jailbreaking', 'genocide']
    
    # active labels for current training run (can be dynamically set and can be the same as ALL_LABELS or a subset)
    ACTIVE_LABELS = ['scam', 'violence', 'harassment', 'hate_speech', 'toxicity', 'obscenity', 'jailbreaking', 'genocide']
    
    @classmethod
    def set_active_labels(cls, labels):
        invalid_labels = set(labels) - set(cls.ALL_LABELS)
        if invalid_labels:
            raise ValueError(f"Invalid labels: {invalid_labels}. Must be from {cls.ALL_LABELS}")
        
        cls.ACTIVE_LABELS = labels
    
    @classmethod
    def get_num_labels(cls):
        return len(cls.ACTIVE_LABELS)
    
    @classmethod
    def make_labels(cls, **kwargs):
        return [kwargs.get(label, 0) for label in cls.ACTIVE_LABELS]
