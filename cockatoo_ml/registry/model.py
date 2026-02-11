class ModelConfig:
    
    # base model identifier from HuggingFace
    BASE_MODEL_NAME = "microsoft/deberta-v3-base"
    
    # number of output labels for multi-label classification
    NUM_LABELS = 4
    
    # problem type for model
    PROBLEM_TYPE = "multi_label_classification"
    
    # max sequence length for tokenization
    MAX_TOKEN_LENGTH = 256
    
    # max sequence length for inference
    INFERENCE_MAX_LENGTH = 512
    
    # epsilon for numerical stability
    EPSILON = 1e-6


class InferenceConfig:

    # default model path for inference
    DEFAULT_MODEL_PATH = "constellation_one_text"
    
    # batch size for inference pipeline
    BATCH_SIZE = 8
    
    # truncation setting
    TRUNCATION = True
