from .labels import LabelConfig

class ModelType:
    CLIP_VIT = "clip-vit"  # CLIP ViT-L-14 for vision-text
    DEBERTA = "deberta"    # DeBERTa V3 for text-only

class ModelConfig:
    
    # model type selection (choose between CLIP_VIT and DEBERTA)
    MODEL_TYPE = ModelType.CLIP_VIT
    
    # CLIP ViT-L-14 configuration
    CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
    CLIP_IMAGE_SIZE = 224
    CLIP_EMBEDDING_DIM = 768
    CLIP_PROJECTION_DIM = 768  # for classification head
    CLIP_MAX_TOKEN_LENGTH = 77  # CLIP's max token length for text inputs
    
    # DeBERTa V3 configuration
    DEBERTA_MODEL_NAME = "microsoft/deberta-v3-base"
    DEBERTA_MAX_TOKEN_LENGTH = 256
    
    # dynamic base model name based on model type
    @classmethod
    def get_base_model_name(cls):
        if cls.MODEL_TYPE == ModelType.CLIP_VIT:
            return cls.CLIP_MODEL_NAME
        
        elif cls.MODEL_TYPE == ModelType.DEBERTA:
            return cls.DEBERTA_MODEL_NAME
        
        else:
            raise ValueError(f"Unknown model type: {cls.MODEL_TYPE}")
    
    # Static default for backwards compatibility with code that accesses BASE_MODEL_NAME directly
    # generally there is no need to modify this line, but its provided for ease of access
    BASE_MODEL_NAME = CLIP_MODEL_NAME if MODEL_TYPE == ModelType.CLIP_VIT else DEBERTA_MODEL_NAME
    
    # number of output labels for multi-label classification
    # derived from active labels for this use case
    NUM_LABELS = len(LabelConfig.ACTIVE_LABELS)
    
    # problem type for model
    PROBLEM_TYPE = "multi_label_classification"
    
    # max sequence length - depends on model type
    @classmethod
    def get_max_token_length(cls):
        if cls.MODEL_TYPE == ModelType.CLIP_VIT:
            return cls.CLIP_MAX_TOKEN_LENGTH
        
        elif cls.MODEL_TYPE == ModelType.DEBERTA:
            return cls.DEBERTA_MAX_TOKEN_LENGTH
        
        else:
            return 256
    
    # Static default for backwards compatibility
    MAX_TOKEN_LENGTH = 77  # CLIP default

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

    INFERENCE_MAX_LENGTH = ModelConfig.INFERENCE_MAX_LENGTH
