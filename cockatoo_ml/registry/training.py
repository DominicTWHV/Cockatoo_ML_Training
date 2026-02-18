from .model import ModelConfig, ModelType

class ModelTrainingConfig:
    # hyperparameters for training, enables bulk value swapping for different models
    # configure which set to use in cockatoo_ml/registry/model.py ModelConfig.MODEL_TYPE

    # --- CLIP hyperparams
    
    CLIP_NUM_EPOCHS = 3
    CLIP_BATCH_SIZE = 16  # CLIP is a larger model
    CLIP_GRADIENT_ACCUMULATION_STEPS = 6 
    CLIP_LEARNING_RATE = 1e-5  # lower LR for CLIP
    CLIP_WEIGHT_DECAY = 0.01
    CLIP_WARMUP_RATIO = 0.1

    CLIP_USE_FP16 = True
    CLIP_USE_BF16 = False  # older hardware, no modern api support
    CLIP_USE_TF32 = False

    # --- DeBERTa hyperparams

    DEBERTA_NUM_EPOCHS = 3
    DEBERTA_BATCH_SIZE = 24
    DEBERTA_GRADIENT_ACCUMULATION_STEPS = 4
    DEBERTA_LEARNING_RATE = 2e-5
    DEBERTA_WEIGHT_DECAY = 0.01
    DEBERTA_WARMUP_RATIO = 0.1

    DEBERTA_USE_FP16 = True
    DEBERTA_USE_BF16 = False  # older hardware, no modern api support
    DEBERTA_USE_TF32 = False

class TrainingConfig:
    
    # training hyperparameters (derived from ModelTrainingConfig and ModelConfig)
    NUM_EPOCHS = ModelTrainingConfig.CLIP_NUM_EPOCHS if ModelConfig.MODEL_TYPE == ModelType.CLIP_VIT else ModelTrainingConfig.DEBERTA_NUM_EPOCHS
    BATCH_SIZE = ModelTrainingConfig.CLIP_BATCH_SIZE if ModelConfig.MODEL_TYPE == ModelType.CLIP_VIT else ModelTrainingConfig.DEBERTA_BATCH_SIZE
    GRADIENT_ACCUMULATION_STEPS = ModelTrainingConfig.CLIP_GRADIENT_ACCUMULATION_STEPS if ModelConfig.MODEL_TYPE == ModelType.CLIP_VIT else ModelTrainingConfig.DEBERTA_GRADIENT_ACCUMULATION_STEPS
    LEARNING_RATE = ModelTrainingConfig.CLIP_LEARNING_RATE if ModelConfig.MODEL_TYPE == ModelType.CLIP_VIT else ModelTrainingConfig.DEBERTA_LEARNING_RATE
    WEIGHT_DECAY = ModelTrainingConfig.CLIP_WEIGHT_DECAY if ModelConfig.MODEL_TYPE == ModelType.CLIP_VIT else ModelTrainingConfig.DEBERTA_WEIGHT_DECAY
    WARMUP_RATIO = ModelTrainingConfig.CLIP_WARMUP_RATIO if ModelConfig.MODEL_TYPE == ModelType.CLIP_VIT else ModelTrainingConfig.DEBERTA_WARMUP_RATIO
    
    # precision
    USE_FP16 = ModelTrainingConfig.CLIP_USE_FP16 if ModelConfig.MODEL_TYPE == ModelType.CLIP_VIT else ModelTrainingConfig.DEBERTA_USE_FP16
    USE_BF16 = ModelTrainingConfig.CLIP_USE_BF16 if ModelConfig.MODEL_TYPE == ModelType.CLIP_VIT else ModelTrainingConfig.DEBERTA_USE_BF16
    USE_TF32 = ModelTrainingConfig.CLIP_USE_TF32 if ModelConfig.MODEL_TYPE == ModelType.CLIP_VIT else ModelTrainingConfig.DEBERTA_USE_TF32
    
    # eval and logging
    EVAL_STRATEGY = 'epoch'
    SAVE_STRATEGY = 'epoch'
    LOAD_BEST_MODEL_AT_END = True
    METRIC_FOR_BEST_MODEL = 'f1'
    GREATER_IS_BETTER = True
    LOGGING_STEPS = 12
    
    # dataloader settings
    DATALOADER_NUM_WORKERS = 4
    SAVE_TOTAL_LIMIT = 2
    
    # distributed training
    DDP_FIND_UNUSED_PARAMETERS = False
    
    # reporting
    REPORT_TO = 'tensorboard'


class MetricsConfig: 
    # threshold for binary classification (used if no per-label thresholds provided for evaluation)
    # per-label thresholds override this value, and can be configured in cockatoo_ml/registry/column_mapping.py DatasetColumnMapping.LABEL_THRESHOLDS
    PREDICTION_THRESHOLD = 0.5
    
    # averaging strategy for multi-label metrics
    AVERAGE_STRATEGY = 'macro'

    # exclude labels with zero support from macro averaging
    INCLUDE_EMPTY_LABELS = False
    
    # zero division handling

    # set to 0 to penalize the model for not making decisions
    # set to 1 to give the model the benefit of the doubt when it fails to predict any positives
    # set to 'warn' to raise warnings and return 0 for precision/recall when this occurs
    ZERO_DIVISION = 0


class CallbackConfig: 
    #these are used in conjunction with cockatoo_ml/registry/api.py WebhookConfig for reporting training telemetry to external services

    # default experiment identifier
    DEFAULT_EXPERIMENT_ID = "constellation-one-text-001"
    
    # request timeout for webhook callbacks
    WEBHOOK_TIMEOUT = 5
