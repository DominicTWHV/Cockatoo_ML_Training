import argparse

from dotenv import load_dotenv

from datasets import load_from_disk

from train.tokenization import get_tokenizer, tokenize_dataset
from train.model_setup import load_model, compute_pos_weight
from train.metrics import compute_metrics
from train.config import get_training_args
from train.trainer import CustomTrainer
from train.callbacks import LiveMetricsWebhookCallback

from cockatoo_ml.registry import PathConfig, WebhookConfig, CallbackConfig, DataSplitConfig, RebalancingPolicy
from cockatoo_ml.registry.column_mapping import DatasetColumnMapping

from cockatoo_ml.logger.context import model_training_logger as logger


load_dotenv()

logger.info("Starting training script...")

# entrypoint for model training
def main():
    parser = argparse.ArgumentParser(description="Train or evaluate the model")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only run evaluation using the saved model.",
    )
    parser.add_argument(
        "--eval-split",
        default="test",
        choices=["test", "validation"],
        help="Dataset split to eval model on when --eval-only is set (default: test).",
    )
    args = parser.parse_args()

    # load the preprocessed dataset
    dataset = load_from_disk(PathConfig.get_processed_data_path())
    logger.info(f"Dataset loaded. Train size: {len(dataset['train'])}")
    
    # check for and log modified weights from rebalancing
    class_weights_metadata = getattr(dataset, 'class_weights', None)
    if class_weights_metadata is not None:
        logger.info("-"*30)
        logger.info("Detected modified weights from rebalancing:")
        logger.info("-"*30)
        from cockatoo_ml.registry import LabelConfig
        weights_dict = {label: float(weight) for label, weight in zip(LabelConfig.ACTIVE_LABELS, class_weights_metadata.tolist())}
        for label, weight in weights_dict.items():
            logger.info(f"  {label}: {weight:.4f}")
        logger.info("These weights will be applied during training to handle class imbalance.")
        logger.info("="*60)
    
    # init the tokenizer to tokenize the dataset
    tokenizer = get_tokenizer()
    
    # tokenize here
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # load model base for tuning or the saved fine-tuned model for eval-only
    if args.eval_only:
        logger.info(f"Loading saved model from: {PathConfig.MODEL_OUTPUT_DIR}")
        model = load_model(model_name=PathConfig.MODEL_OUTPUT_DIR)

    else:
        model = load_model()
    
    # compute pos_weight for BCE loss only when using reweighting policies
    class_weights = getattr(dataset, 'class_weights', None)
    pos_weight = None
    if DataSplitConfig.REBALANCING_POLICY in (RebalancingPolicy.REWEIGHTING, RebalancingPolicy.COMBINED):
        # use pre-computed class weights if available from rebalancing, otherwise compute from dataset
        pos_weight = compute_pos_weight(dataset['train'], class_weights=class_weights)
        
    else:
        if class_weights is not None:
            logger.info("Dataset rebalancing applied; skipping pos_weight in loss.")
    
    # get training args
    training_args = get_training_args()
    
    # determine which threshold to use (LABEL_THRESHOLDS for eval-only mode)
    eval_thresholds = DatasetColumnMapping.LABEL_THRESHOLDS if args.eval_only else None
    
    # define trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        pos_weight=pos_weight,
        eval_thresholds=eval_thresholds
    )
    
    # hook to metrics server for posting metrics
    if WebhookConfig.enable or WebhookConfig.enable_validation:  # only add callback if at least one is enabled
        trainer.add_callback(LiveMetricsWebhookCallback(
            training_endpoint_url=WebhookConfig.get_webhook_url(),
            validation_endpoint_url=WebhookConfig.get_validation_webhook_url(),

            auth_token=WebhookConfig.get_api_key(),

            experiment_id=CallbackConfig.DEFAULT_EXPERIMENT_ID,
            
            enable_training=WebhookConfig.enable,
            enable_validation=WebhookConfig.enable_validation
        ))
    
    if not args.eval_only:
        # start training
        logger.info("Starting training...")
        trainer.train()
        
        # save the trained model
        trainer.save_model(PathConfig.MODEL_OUTPUT_DIR)
        logger.info(f"Training finished. Model saved to: {PathConfig.MODEL_OUTPUT_DIR}")
        
    else:
        logger.info("Eval-only mode enabled. Skipping training.")
    
    # evaluate the requested split
    eval_split = args.eval_split
    if args.eval_only:
        logger.info(f"Evaluating on {eval_split} set with custom thresholds: {DatasetColumnMapping.LABEL_THRESHOLDS}")
    else:
        logger.info(f"Evaluating on {eval_split} set...")
    test_results = trainer.evaluate(tokenized_dataset[eval_split])
    logger.info(f"{eval_split.capitalize()} results: {test_results}")

    #done!

if __name__ == "__main__":
    main()