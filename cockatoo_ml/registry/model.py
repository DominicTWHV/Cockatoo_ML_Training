from .labels import LabelConfig

class ModelType:
    CLIP_VIT = "clip-vit"  # CLIP ViT-L-14 for vision-text
    DEBERTA = "deberta"    # DeBERTa V3 for text-only
    MODERNBERT = "modernbert"  # ModernBERT for text-only

class ModelConfig:
    
    # model type selection (choose between CLIP_VIT and DEBERTA)
    MODEL_TYPE = ModelType.CLIP_VIT
    
    # CLIP ViT-L-14 configuration
    CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
    CLIP_IMAGE_SIZE = 224
    CLIP_PROJECTION_DIM = 768  # for classification head
    CLIP_MAX_TOKEN_LENGTH = 77  # CLIP's max token length for text inputs
    CLIP_MAX_INFERENCING_TOKEN_LENGTH = 77  # allow longer inputs for inferencing? (experimental)
    
    # DeBERTa V3 configuration
    DEBERTA_MODEL_NAME = "microsoft/deberta-v3-base"
    DEBERTA_MAX_TOKEN_LENGTH = 256
    DEBERTA_MAX_INFERENCING_TOKEN_LENGTH = 256  # allow longer inputs when inferencing? (experimental)

    # ModernBERT configuration
    MODERNBERT_MODEL_NAME = "answerdotai/ModernBERT-large" #modernbert large
    MODERNBERT_MAX_TOKEN_LENGTH = 512 # to save memory
    MODERNBERT_MAX_INFERENCING_TOKEN_LENGTH = 8192  # allow longer inputs for inferencing? (experimental)


    # epsilon for numerical stability
    EPSILON = 1e-6

    # ---------- Automatically derived settings below ---------
    # Generally there should be no need to go below this line (within this class). The config options below are automatically derived from inputs above.
    
    # dynamic base model name based on model type
    @classmethod
    def get_base_model_name(cls):
        if cls.MODEL_TYPE == ModelType.CLIP_VIT:
            return cls.CLIP_MODEL_NAME
        
        elif cls.MODEL_TYPE == ModelType.DEBERTA:
            return cls.DEBERTA_MODEL_NAME
        
        elif cls.MODEL_TYPE == ModelType.MODERNBERT:
            return cls.MODERNBERT_MODEL_NAME
        
        else:
            raise ValueError(f"Unknown model type: {cls.MODEL_TYPE}")
    
    # number of output labels for multi-label classification
    # derived from active labels for this use case
    NUM_LABELS = len(LabelConfig.ACTIVE_LABELS)
    
    # problem type for model
    PROBLEM_TYPE = "multi_label_classification"

    # attention implementation for transformer model

    ATTENTION_IMPLEMENTATION = "sdpa" if MODEL_TYPE == ModelType.MODERNBERT else "default"  # use standard attention for CLIP and DeBERTa, but switch to sdpa for ModernBERT which supports it for better efficiency on long inputs
    
    # max sequence length - depends on model type
    @classmethod
    def get_max_token_length(cls):
        if cls.MODEL_TYPE == ModelType.CLIP_VIT:
            return cls.CLIP_MAX_TOKEN_LENGTH
        
        elif cls.MODEL_TYPE == ModelType.DEBERTA:
            return cls.DEBERTA_MAX_TOKEN_LENGTH
        
        elif cls.MODEL_TYPE == ModelType.MODERNBERT:
            return cls.MODERNBERT_MAX_TOKEN_LENGTH
        
        else:
            return 256 # default fallback (should never be used if the model is properly configured)

    # max sequence length for inference
    INFERENCE_MAX_LENGTH = CLIP_MAX_INFERENCING_TOKEN_LENGTH if MODEL_TYPE == ModelType.CLIP_VIT else DEBERTA_MAX_INFERENCING_TOKEN_LENGTH if MODEL_TYPE == ModelType.DEBERTA else MODERNBERT_MAX_INFERENCING_TOKEN_LENGTH

class InferenceConfig:

    # default model path for inference
    DEFAULT_MODEL_PATH = "constellation_one_text"

    # truncation setting
    TRUNCATION = True

    INFERENCE_MAX_LENGTH = ModelConfig.INFERENCE_MAX_LENGTH
