from cockatoo_ml.registry.labels import LabelConfig

from cockatoo_ml.logger.context import data_processing_logger as logger

# this is a helper module to get the dataset label mapping instance and provide some helper functions for working with it

class DatasetLabelMapping:
    
    def __init__(self):
        self._mappings = {}
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        #set default mappings
        self._mappings = {
            'phishing': {
                'labels': ['scam'],
                'description': 'Phishing dataset - contains scam labels'
            },
            'hate_speech': {
                'labels': ['hate_speech', 'violence', 'harassment', 'dehumanization', 'status'],
                'description': 'Measuring hate speech - multiple hate speech related labels'
            },
            'tweet_hate': {
                'labels': ['hate_speech'],
                'description': 'Tweet hate eval - contains hate_speech labels'
            },
            'toxicchat': {
                'labels': ['toxicity', 'jailbreaking'],
                'description': 'Toxic chat - contains toxicity and jailbreaking labels'
            },
            'jigsaw': {
                'labels': ['toxicity', 'obscenity', 'violence', 'harassment', 'hate_speech'],
                'description': 'Jigsaw toxicity - multi-label toxic content classification'
            },
        }
    
    def register(self, dataset_name, labels, description=""):
        # register dataset with labels and optional description
        # validate labels
        invalid = set(labels) - set(LabelConfig.ALL_LABELS)
        if invalid:
            raise ValueError(f"Invalid labels {invalid} for dataset '{dataset_name}'. "
                           f"Must be from {LabelConfig.ALL_LABELS}")
        
        self._mappings[dataset_name] = {
            'labels': labels,
            'description': description
        }

        logger.info(f"Registered dataset '{dataset_name}' with labels: {labels}")
    
    def get_labels(self, dataset_name):
        # get labels for a dataset (name)
        if dataset_name not in self._mappings:
            raise KeyError(f"Dataset '{dataset_name}' not registered. Available: {list(self._mappings.keys())}")
        
        return self._mappings[dataset_name]['labels']
    
    def get_mask(self, dataset_name):
        # get binary mask for dataset labels based on ACTIVE_LABELS

        dataset_labels = self.get_labels(dataset_name)
        return [1 if label in dataset_labels else 0 for label in LabelConfig.ACTIVE_LABELS]
    
    def get_all_labels_from_datasets(self, dataset_names):
        # get all the labels from a list of datasets (union)
        all_labels = set()
        for name in dataset_names:
            all_labels.update(self.get_labels(name))

        return sorted(list(all_labels))
    
    def list_datasets(self):
        # list datasets from self._mappings
        return self._mappings
    
    def reset_to_defaults(self):
        # reinitialize the mappings to defaults
        self._initialize_defaults()
        logger.info("Dataset label mappings reset to defaults")

#global instance for application-wide use
_global_mapping = DatasetLabelMapping()


# global mapping based functions

def get_dataset_label_mapping():
    return _global_mapping

def register_dataset(dataset_name, labels, description=""):
    _global_mapping.register(dataset_name, labels, description)

def get_dataset_labels(dataset_name):
    return _global_mapping.get_labels(dataset_name)

def get_dataset_label_mask(dataset_name):
    return _global_mapping.get_mask(dataset_name)
