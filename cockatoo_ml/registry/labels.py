class LabelConfig:
    # all available labels in the system
    ALL_LABELS = ['scam', 'violence', 'nsfw', 'harassment', 'hate_speech', 'toxicity', "obscenity"]
    
    # active labels for current training run (can be dynamically set and can be the same as ALL_LABELS or a subset)
    ACTIVE_LABELS = ['scam', 'violence', 'harassment', 'hate_speech', 'toxicity', "obscenity"]
    
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
    
    @classmethod
    def make_labels_dict(cls, **kwargs):
        # build label dictionary for active labels, defaulting to 0 if not provided
        return {label: kwargs.get(label, 0) for label in cls.ACTIVE_LABELS}
    
    @classmethod
    def label_to_idx(cls, label_name):
        # get idx of a label in ACTIVE_LABELS, raise error if not found
        try:
            return cls.ACTIVE_LABELS.index(label_name)
        
        except ValueError:
            raise ValueError(f"Label '{label_name}' not in ACTIVE_LABELS: {cls.ACTIVE_LABELS}")
    
    @classmethod
    def idx_to_label(cls, idx):
        # get label by idx in ACTIVE_LABELS, raise error if idx out of range
        if idx >= len(cls.ACTIVE_LABELS):
            raise ValueError(f"Index {idx} out of range for {len(cls.ACTIVE_LABELS)} labels")
        
        return cls.ACTIVE_LABELS[idx]
