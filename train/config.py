from transformers import TrainingArguments

from cockatoo_ml.registry import TrainingConfig, PathConfig

# this file contains training configs for the model
def get_training_args(
    output_dir=None,
    logging_dir=None,
    num_train_epochs=None,
    batch_size=None,
    gradient_accumulation_steps=None,
    learning_rate=None,
    use_fp16=None
):
    # Use registry defaults if not provided
    if output_dir is None:
        output_dir = PathConfig.MODEL_OUTPUT_DIR

    if logging_dir is None:
        logging_dir = PathConfig.LOGGING_DIR

    if num_train_epochs is None:
        num_train_epochs = TrainingConfig.NUM_EPOCHS

    if batch_size is None:
        batch_size = TrainingConfig.BATCH_SIZE

    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = TrainingConfig.GRADIENT_ACCUMULATION_STEPS

    if learning_rate is None:
        learning_rate = TrainingConfig.LEARNING_RATE
        
    if use_fp16 is None:
        use_fp16 = TrainingConfig.USE_FP16

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        ddp_find_unused_parameters=TrainingConfig.DDP_FIND_UNUSED_PARAMETERS,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=TrainingConfig.WEIGHT_DECAY,
        warmup_ratio=TrainingConfig.WARMUP_RATIO,
        eval_strategy=TrainingConfig.EVAL_STRATEGY,
        save_strategy=TrainingConfig.SAVE_STRATEGY,
        load_best_model_at_end=TrainingConfig.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=TrainingConfig.METRIC_FOR_BEST_MODEL,
        greater_is_better=TrainingConfig.GREATER_IS_BETTER,
        fp16=use_fp16,
        bf16=TrainingConfig.USE_BF16,
        tf32=TrainingConfig.USE_TF32,
        report_to=TrainingConfig.REPORT_TO,
        logging_dir=logging_dir,
        logging_steps=TrainingConfig.LOGGING_STEPS,
        dataloader_num_workers=TrainingConfig.DATALOADER_NUM_WORKERS,
        save_total_limit=TrainingConfig.SAVE_TOTAL_LIMIT,
    )
    
    return training_args
