class TrainingConfig:
    
    # training hyperparameters (currently optimized for CLIP)
    NUM_EPOCHS = 3
    BATCH_SIZE = 16  # increase for DeBERTa as CLIP is a bigger model
    GRADIENT_ACCUMULATION_STEPS = 6  # reduce to something like 4 for DeBERTa
    LEARNING_RATE = 1e-5  # lower LR for CLIP (was 2e-5 for DeBERTa)
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    
    # precision
    USE_FP16 = True
    USE_BF16 = False  # older hardware, no modern api support
    USE_TF32 = False
    
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
    # threshold for binary classification
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
