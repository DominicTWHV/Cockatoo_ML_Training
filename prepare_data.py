import os
from train.data_loaders import load_all_datasets
from train.data_processors import combine_datasets, split_dataset, print_dataset_stats
from logger.context import data_processing_logger as logger

# entrypoint to prepare data for training
BASE_DIR = "./data"
os.makedirs(BASE_DIR, exist_ok=True)


def main():
    # load all available datasets
    datasets = load_all_datasets(BASE_DIR)
    
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
    save_path = os.path.join(BASE_DIR, "processed_text")
    dataset.save_to_disk(save_path)
    logger.info(f"Saved to: {save_path}")
    
    # print stats
    print_dataset_stats(combined_df, dataset)
    
    logger.info("Done! If more datasets loaded, update train_text.py with 'data/processed_text'")


if __name__ == "__main__":
    main()