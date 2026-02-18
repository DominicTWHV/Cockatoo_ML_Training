from cockatoo_ml.registry.labels import LabelConfig
from cockatoo_ml.registry.dataset_label_mapping import get_dataset_label_mapping

from cockatoo_ml.logger.context import data_processing_logger as logger

# helper function to get the dataset label mapping instance

def configure_labels_for_datasets(dataset_names, custom_labels=None):
    mapping = get_dataset_label_mapping()
    
    if custom_labels is not None:
        # Use custom labels if provided
        logger.info(f"Using custom labels: {custom_labels}")
        LabelConfig.set_active_labels(custom_labels)
        return custom_labels
    
    # Get union of all labels from datasets
    all_labels = mapping.get_all_labels_from_datasets(dataset_names)
    
    logger.info(f"Configuring labels from datasets {dataset_names}")
    logger.info(f"Union of labels: {all_labels}")
    
    # Set the active labels
    LabelConfig.set_active_labels(all_labels)
    
    return all_labels


def register_custom_datasets(datasets_config):
    mapping = get_dataset_label_mapping()
    
    for dataset_name, config in datasets_config.items():
        labels = config.get('labels', [])
        description = config.get('description', '')
        
        try:
            mapping.register(dataset_name, labels, description)
        except ValueError as e:
            logger.error(f"Failed to register '{dataset_name}': {e}")
            raise


def print_dataset_label_summary():
    mapping = get_dataset_label_mapping()
    datasets = mapping.list_datasets()
    
    logger.info("=" * 60)
    logger.info("DATASET LABEL MAPPING SUMMARY")
    logger.info("=" * 60)
    
    for dataset_name, config in datasets.items():
        labels = config.get('labels', [])
        description = config.get('description', '')
        logger.info(f"\n{dataset_name}:")
        logger.info(f"  Labels: {labels}")
        if description:
            logger.info(f"  Description: {description}")
    
    logger.info(f"\nActive labels: {LabelConfig.ACTIVE_LABELS}")
    logger.info(f"Number of active labels: {LabelConfig.get_num_labels()}")
    logger.info("=" * 60)


def validate_dataset_labels(dataset_names):
    mapping = get_dataset_label_mapping()
    valid = True
    
    for dataset_name in dataset_names:
        try:
            labels = mapping.get_labels(dataset_name)
            logger.info(f"✓ {dataset_name}: {labels}")
            
        except KeyError:
            logger.error(f"✗ {dataset_name}: NOT REGISTERED")
            valid = False
    
    return valid
