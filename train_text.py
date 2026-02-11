from dotenv import load_dotenv
import os

from datasets import load_from_disk
from train.tokenization import get_tokenizer, tokenize_dataset
from train.model_setup import load_model, compute_pos_weight
from train.metrics import compute_metrics
from train.config import get_training_args
from train.trainer import CustomTrainer
from train.callbacks import LiveMetricsWebhookCallback
from cockatoo_ml.registry import PathConfig, ModelConfig, WebhookConfig, CallbackConfig
from cockatoo_ml.logger.context import model_training_logger as logger


load_dotenv()

logger.info("Starting training script...")

# entrypoint for model training
def main():
    # load the preprocessed dataset
    dataset = load_from_disk(PathConfig.get_processed_data_path())
    logger.info(f"Dataset loaded. Train size: {len(dataset['train'])}")
    
    # init the tokenizer to tokenize the dataset
    tokenizer = get_tokenizer()
    
    # tokenize here
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # load model base for tuning
    model = load_model()
    
    # compute pos weight for BCE loss
    pos_weight = compute_pos_weight(dataset['train'])
    
    # get training args
    training_args = get_training_args()
    
    # define trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        pos_weight=pos_weight
    )
    
    # hook to metrics server for posting metrics
    if WebhookConfig.enable: # only add callback if enabled in config
        trainer.add_callback(LiveMetricsWebhookCallback(
            endpoint_url=WebhookConfig.get_webhook_url(),
            auth_token=WebhookConfig.get_api_key(),
            experiment_id=CallbackConfig.DEFAULT_EXPERIMENT_ID
        ))
    
    # start training
    logger.info("Starting training...")
    trainer.train()
    
    # save the trained model
    trainer.save_model(PathConfig.MODEL_OUTPUT_DIR)
    logger.info(f"Training finished. Model saved to: {PathConfig.MODEL_OUTPUT_DIR}")
    
    # evaluate the test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_dataset['test'])
    logger.info(f"Test results: {test_results}")

    #done!

if __name__ == "__main__":
    main()