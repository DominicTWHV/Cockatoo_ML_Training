from transformers import TrainingArguments

# this file contains training configs for the model
def get_training_args(
    output_dir='constellation_one_text',
    logging_dir='constellation_one_logs',
    num_train_epochs=3,
    batch_size=24,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    use_fp16=True
):

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        ddp_find_unused_parameters=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        fp16=use_fp16,
        bf16=False,  # older hardware, no modern api support
        tf32=False,
        report_to='tensorboard',
        logging_dir=logging_dir,
        logging_steps=12,
        dataloader_num_workers=4,
        save_total_limit=2,
    )
    
    return training_args
