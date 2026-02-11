from dotenv import load_dotenv
import os

from datasets import load_from_disk
from train.tokenization import get_tokenizer, tokenize_dataset
from train.model_setup import load_model, compute_pos_weight
from train.metrics import compute_metrics
from train.config import get_training_args
from train.trainer import CustomTrainer
from train.callbacks import LiveMetricsWebhookCallback


load_dotenv()

print("Starting training script...")

# entrypoint for model training
def main():
    # load the preprocessed dataset
    dataset = load_from_disk('data/processed_text')
    print("Dataset loaded. Train size:", len(dataset['train']))
    
    # init the tokenizer to tokenize the dataset
    tokenizer = get_tokenizer("microsoft/deberta-v3-base")
    
    # tokenize here
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length=256)
    
    # load model base for tuning
    model = load_model("microsoft/deberta-v3-base", num_labels=4)
    
    # compute pos weight for BCE loss
    pos_weight = compute_pos_weight(dataset['train'])
    
    # get training args
    training_args = get_training_args(
        output_dir='constellation_one_text',
        logging_dir='constellation_one_logs',
        num_train_epochs=3,
        batch_size=24,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        use_fp16=True
    )
    
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
    webhook_url = os.getenv("METRICS_WEBHOOK_URL", "https://api.cockatoo.dev/api/training/data")
    trainer.add_callback(LiveMetricsWebhookCallback(
        endpoint_url=webhook_url,
        auth_token=os.getenv("METRICS_API_KEY"),
        experiment_id="constellation-one-text-001"
    ))
    
    # start training
    print("Starting training...")
    trainer.train()
    
    # save the trained model
    trainer.save_model('constellation_one_text')
    print("Training finished. Model saved to: constellation_one_text")
    
    # evaluate the test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_dataset['test'])
    print("Test results:", test_results)

    #done!

if __name__ == "__main__":
    main()