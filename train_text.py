from dotenv import load_dotenv

import os
import torch
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerControl, TrainerCallback, TrainerState
from datasets import load_from_disk
from sklearn.metrics import precision_recall_fscore_support

from datetime import datetime

load_dotenv()

print("Starting training script...")

# load the processed dataset
dataset = load_from_disk('data/processed_text')
print("Dataset loaded. Train size:", len(dataset['train']))

# tokenizer (ms deberta v3 base)
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

def preprocess(examples):
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    )
    tokenized['labels'] = torch.tensor(examples['labels'], dtype=torch.float)
    return tokenized

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=['text'])
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# select model (ms deberta v3 base)
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=4,
    problem_type="multi_label_classification",
    torch_dtype=torch.float32
)

# compute positive weight for BCE loss (helps with imbalance)
all_labels = torch.cat([torch.tensor(l) for l in dataset['train']['labels']])
pos_weight = (all_labels.numel() - all_labels.sum()) / (all_labels.sum() + 1e-6)  # avoid div by zero
print(f"Positive weight (inverse frequency): {pos_weight:.2f}")

# custom loss function
class WeightedBCE(torch.nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, labels):
        return self.loss_fn(logits, labels.float())

# metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy().astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    return {
        'precision': p,
        'recall': r,
        'f1': f1
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='constellation_one_text',
    num_train_epochs=3,
    per_device_train_batch_size=24, # reduce if OOM
    per_device_eval_batch_size=24,
    ddp_find_unused_parameters=False,
    gradient_accumulation_steps=4, # effective batch size = 32
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    fp16=True,
    bf16=False, #no hardware support for bf16 or tf32 
    tf32=False,
    report_to='tensorboard',
    logging_dir='constellation_one_logs',
    logging_steps=12,
    dataloader_num_workers=4,
    save_total_limit=2,
)

# remote server metrics hook call
class LiveMetricsWebhookCallback(TrainerCallback):
    def __init__(self, endpoint_url: str, auth_token: str = None, experiment_id: str = None):
        self.endpoint_url = endpoint_url
        self.headers = {"Content-Type": "application/json"}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
        self.experiment_id = experiment_id or "default-run"

    def _send_metrics(self, payload: dict):
        try:
            # Short timeout to prevent training stalls if the endpoint is slow
            r = requests.post(self.endpoint_url, json=payload, headers=self.headers, timeout=5)
            r.raise_for_status()
        except Exception as e:
            print(f"⚠️ Failed to send metrics: {e}")

    def on_log(self, args, state, control, **kwargs):
        """Sends all logged training metrics immediately."""
        if state.log_history:
            # Grab the most recent log entry
            latest_metrics = state.log_history[-1].copy()
            
            payload = {
                "experiment_id": self.experiment_id,
                "global_step": state.global_step,
                "epoch": state.epoch,
                "metrics": latest_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self._send_metrics(payload)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Sends evaluation metrics immediately when they become available."""
        if metrics:
            payload = {
                "experiment_id": self.experiment_id,
                "global_step": state.global_step,
                "epoch": state.epoch,
                "metrics": metrics,
                "timestamp": datetime.utcnow().isoformat(),
                "is_eval": True
            }
            self._send_metrics(payload)

# torch trainer class

class CustomTrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        #BCEWithLogitsLoss handles the internal conversion to half if needed
        self.pos_weight = pos_weight.to(self.model.device)
        self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels") # Hide from model's internal loss logic
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        device = logits.device
        dtype = logits.dtype
        
        loss_fct = torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.to(device=device, dtype=dtype)
        )
        
        loss = loss_fct(logits, labels.to(dtype))
        return (loss, outputs) if return_outputs else loss

# make sure to pass in pos_weight to the trainer so it can be used in compute_loss
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
    pos_weight=pos_weight
)


webhook_url = "https://api.cockatoo.dev/api/training/data"
trainer.add_callback(LiveMetricsWebhookCallback(
    endpoint_url=webhook_url,
    auth_token=os.getenv("METRICS_API_KEY"),
    experiment_id="constellation-one-text-001"
))

print("Starting training...")
trainer.train()

# save model
trainer.save_model('constellation_one_text')
print("Training finished. Model saved to: constellation_one_text")

# eval on test set
print("Evaluating on test set...")
test_results = trainer.evaluate(tokenized_dataset['test'])
print("Test results:", test_results)