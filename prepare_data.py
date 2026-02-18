import os
from train.data_loaders import load_all_datasets
from train.data_processors import combine_datasets, split_dataset, print_dataset_stats
from cockatoo_ml.registry import PathConfig, LabelConfig, DataSplitConfig
from cockatoo_ml.logger.context import data_processing_logger as logger

# entrypoint to prepare data for training
os.makedirs(PathConfig.BASE_DATA_DIR, exist_ok=True)


def main():
    # load all available datasets
    datasets = load_all_datasets()
    
    if not datasets:
        logger.error("No data loaded - check file paths and column names above.")
        return
    
    # combine and label datasets
    combined_df = combine_datasets(datasets)
    
    if combined_df is None or len(combined_df) == 0:
        logger.error("Failed to combine datasets.")
        return
    
    # split dataset into training/val/test
    dataset = split_dataset(combined_df)
    
    # save processed dataset into data folder
    save_path = PathConfig.get_processed_data_path()
    dataset.save_to_disk(save_path)
    logger.info(f"Saved to: {save_path}")

    # print a few samples of the merged dataset
    logger.info("Sample rows from merged dataset (processed_text):")
    sample_count = min(5, len(combined_df))
    sample_df = combined_df.sample(
        n=sample_count,
        random_state=DataSplitConfig.RANDOM_STATE
    ).reset_index(drop=True)
    for i, row in sample_df.iterrows():
        text_preview = str(row.get('text', ''))
        if len(text_preview) > 200:
            text_preview = text_preview[:200] + "..."
        label_vector = row.get('labels') or []
        active_labels = [
            label
            for label, is_on in zip(LabelConfig.ACTIVE_LABELS, label_vector)
            if is_on == 1
        ]
        if not active_labels:
            active_labels = ["<none>"]
        logger.info(
            f"Sample {i + 1}: text='{text_preview}' | labels={active_labels} | source={row.get('_dataset_source')}"
        )
    
    # print stats
    print_dataset_stats(combined_df, dataset)
    
    logger.info(f"Done! If more datasets loaded, update train.py with '{save_path}'")


if __name__ == "__main__":
    main()