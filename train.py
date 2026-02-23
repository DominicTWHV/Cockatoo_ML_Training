import argparse
import json
import os
import torch

from pathlib import Path

from dotenv import load_dotenv

from datasets import load_from_disk

from train.tokenization import get_tokenizer, tokenize_dataset
from train.model_setup import load_model, compute_pos_weight
from train.metrics import compute_metrics, find_best_thresholds
from train.config import get_training_args
from train.trainer import CustomTrainer
from train.callbacks import LiveMetricsWebhookCallback

from analysis.plotting import plot_eval_metrics

from cockatoo_ml.registry import PathConfig, WebhookConfig, CallbackConfig, DataSplitConfig, RebalancingPolicy, ModelConfig, MetricsConfig
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
    parser.add_argument(
        "--eval-data-json",
        type=str,
        default=None,
        help="Path to JSON file containing evaluation metrics. If provided, plots will be generated from this data instead of running evaluation.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="graph",
        help="Directory to save evaluation plots (default: graph).",
    )
    args = parser.parse_args()

    # warn if distributed env vars were injected by the container runtime but torchrun was not used.
    # when these are set without a proper DDP init, HuggingFace Trainer enters a broken distributed
    # mode: it thinks it is a non-primary rank, skips moving the model to GPU, and effectively loads
    # nothing — producing the "1 GB VRAM with a huge batch" symptom.
    _DIST_VARS = ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT')
    _detected_dist = {k: os.environ[k] for k in _DIST_VARS if k in os.environ}
    if _detected_dist and not torch.distributed.is_initialized():
        logger.warning(
            f"Distributed env vars detected but torch.distributed is NOT initialized: {_detected_dist}. "
            "If you are NOT using torchrun/DDP, unset these variables to prevent the Trainer from "
            "entering a broken distributed mode (near-zero VRAM, model not loaded on GPU). "
            "See docs/containers.md for details."
        )

    # if eval data json arg is passed, just generate plots
    if args.eval_data_json:
        logger.info(f"Loading evaluation data from: {args.eval_data_json}")
        json_path = Path(args.eval_data_json)
        
        if not json_path.exists():
            raise FileNotFoundError(f"Eval data JSON file not found: {args.eval_data_json}")
        
        with open(json_path, 'r') as f:
            test_results = json.load(f)
        
        logger.info(f"Loaded eval data from JSON: {test_results}")
        
        # generate plots from json data
        try:
            experiment_name = CallbackConfig.DEFAULT_EXPERIMENT_ID
            model_name = ModelConfig.MODEL_TYPE
            
            plot_dir = plot_eval_metrics(eval_data=test_results, experiment_name=experiment_name, model_name=model_name, epoch_num=None, output_dir=args.plot_dir)
            logger.info(f"Evaluation plots saved to: {plot_dir.absolute()}")

        except Exception as e:
            logger.warning(f"Failed to generate evaluation plots: {e}")
        
        return

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

    # basic token-length diagnostics to catch silent under-utilization
    try:
        train_sample_size = min(512, len(tokenized_dataset['train']))
        train_sample = tokenized_dataset['train'].select(range(train_sample_size))
        lengths = train_sample['attention_mask'].sum(dim=1).tolist()
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            max_seen = max(lengths)
            logger.info(
                f"Token length stats (train sample n={len(lengths)}): avg_nonpad={avg_len:.1f}, max_nonpad={max_seen}, configured_max={ModelConfig.get_max_token_length()}"
            )
            if avg_len < 64:
                logger.warning(
                    "Average non-padding token length is very low. VRAM usage may stay low even with larger batch sizes."
                )
    except Exception as e:
        logger.warning(f"Failed to compute token-length diagnostics: {e}")
    
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
    logger.info(
        f"Training setup: model_type={ModelConfig.MODEL_TYPE}, batch_size={training_args.per_device_train_batch_size}, grad_accum={training_args.gradient_accumulation_steps}, fp16={training_args.fp16}, bf16={training_args.bf16}, grad_ckpt={training_args.gradient_checkpointing}"
    )

    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    if cuda_available:
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA device[0]: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available. Training will run on CPU, which can explain low GPU VRAM usage.")
    
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
    logger.info(f"Trainer device: {trainer.args.device} | n_gpu={trainer.args.n_gpu}")
    
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

        # optionally sweep per-label thresholds on the validation set to find the best F1 score
        if getattr(MetricsConfig, 'FIND_BEST_THRESHOLDS', False):
            logger.info("Searching for best per-label thresholds on the validation set...")
            val_preds = trainer.predict(tokenized_dataset['validation'])
            best_thresholds = find_best_thresholds(val_preds.predictions, val_preds.label_ids)
            logger.info("-" * 20)
            logger.info("Best per-label thresholds (maximising F1):")

            for label, result in best_thresholds.items():
                logger.info(f"  {label:<20} threshold={result['threshold']:.4f}  f1={result['f1']:.4f}")

            logger.info("-" * 20)

    else:
        logger.info("Eval-only mode enabled. Skipping training.")

        # optionally sweep per-label thresholds on the validation set to find the best F1 score
        if getattr(MetricsConfig, 'FIND_BEST_THRESHOLDS', False):
            logger.info("Searching for best per-label thresholds on the validation set...")
            val_preds = trainer.predict(tokenized_dataset['validation'])
            best_thresholds = find_best_thresholds(val_preds.predictions, val_preds.label_ids)
            logger.info("-" * 20)
            logger.info("Best per-label thresholds (maximising F1):")

            for label, result in best_thresholds.items():
                logger.info(f"  {label:<20} threshold={result['threshold']:.4f}  f1={result['f1']:.4f}")

            logger.info("-" * 20)

    # evaluate the requested split
    eval_split = args.eval_split
    
    # run evaluation
    if args.eval_only:
        logger.info(f"Evaluating on {eval_split} set with custom thresholds: {DatasetColumnMapping.LABEL_THRESHOLDS}")
    else:
        logger.info(f"Evaluating on {eval_split} set...")
    
    test_results = trainer.evaluate(tokenized_dataset[eval_split])
    
    logger.info(f"{eval_split.capitalize()} results: {test_results}")
    
    # generate and save evaluation plots
    try:
        experiment_name = CallbackConfig.DEFAULT_EXPERIMENT_ID
        model_name = ModelConfig.MODEL_TYPE
        # extract epoch number if available from training args
        epoch_num = None
        if hasattr(trainer, 'state') and trainer.state is not None:
            epoch_num = int(trainer.state.epoch) if trainer.state.epoch else None
        
        plot_dir = plot_eval_metrics(eval_data=test_results, experiment_name=experiment_name, model_name=model_name, epoch_num=epoch_num, output_dir=args.plot_dir)
        logger.info(f"Evaluation plots saved to: {plot_dir.absolute()}")

    except Exception as e:
        logger.warning(f"Failed to generate evaluation plots: {e}")

    #done!

if __name__ == "__main__":
    main()