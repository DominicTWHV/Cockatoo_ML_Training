import os


class PathConfig:
    # the directory to download subfolders containing datasets to, relative to project root
    BASE_DATA_DIR = "./data"
    
    # data/model output directories, relative to project root.

    # model output dir: model checkpoints and final model will be saved here
    # logging dir: tensorflow logs, you can run tensorboard to visualize training progress using this path
    MODEL_OUTPUT_DIR = "constellation_one_text"
    LOGGING_DIR = "constellation_one_logs"
    
    # processed dataset folder name, within the BASE_DATA_DIR
    PROCESSED_DATA_DIR = "processed_text"
    
    @classmethod
    def get_processed_data_path(cls):
        return os.path.join(cls.BASE_DATA_DIR, cls.PROCESSED_DATA_DIR)
    
    @classmethod
    def get_dataset_path(cls, dataset_name):
        return os.path.join(cls.BASE_DATA_DIR, dataset_name)
