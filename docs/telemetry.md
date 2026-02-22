# Telemetry

We have embedded an internal http client to push training data live to an API server. You can modify the settings in `cockatoo_ml/registry/api.py`. If your remote API server expects an authentication token, edit the `.env` file to include the token.

The telemetry hook can be enabled independently for training and validation:

- Training telemetry: `WebhookConfig.enable` (uses `METRICS_WEBHOOK_URL`)
- Validation telemetry: `WebhookConfig.enable_validation` (uses `METRICS_VALIDATION_WEBHOOK_URL`)

This is the telemetry system we use at [cockatoo.dev](https://cockatoo.dev/ml-training.html) to monitor and publish training data across all our models in one place.

*Note: This telemetry is not something that sends us information. It is a system that you can use to send live training data to your own servers. It can be leveraged if you want to create the same metrics display we have at cockatoo.dev*

**Telemetry API Output**

Training endpoint payload (sample):

```json
{
  "experiment_id": "constellation-one-text-001",
  "global_step": null,
  "epoch": 1.0,
  "metrics": {
    "loss": 0.4312,
    "learning_rate": 1.5e-05,
    "step": 120
  },
  "timestamp": "2026-02-13 18:12:03.123456"
}
```

Validation endpoint payload (sample):

```json
{
  "experiment_id": "constellation-one-text-001",
  "global_step": 120,
  "epoch": 1.0,
  "metrics": {
    "eval_loss":0.16034406423568726,
    "eval_precision":0.6059971310039647,
    "eval_recall":0.9138250950483955,
    "eval_f1":0.7164361696270752,
    "eval_precision_scam":0.9117559964465501,
    "eval_recall_scam":0.9532507739938081,
    "eval_f1_scam":0.9320417738761919,
    "eval_precision_violence":0.42734150795721365,
    "eval_recall_violence":0.8970427163198248,
    "eval_f1_violence":0.5789008658773634,
    "eval_precision_harassment":0.7726063829787234,
    "eval_recall_harassment":0.9423076923076923,
    "eval_f1_harassment":0.8490605427974948,
    "eval_precision_hate_speech":0.429821819318537,
    "eval_recall_hate_speech":0.8969341161121983,
    "eval_f1_hate_speech":0.5811496196111581,
    "eval_precision_toxicity":0.5737432488574989,
    "eval_recall_toxicity":0.8712933753943217,
    "eval_f1_toxicity":0.6918837675350702,
    "eval_precision_obscenity":0.5207138304652645,
    "eval_recall_obscenity":0.9221218961625283,
    "eval_f1_obscenity":0.6655804480651731,
    "eval_runtime":247.1414,
    "eval_samples_per_second":117.512,
    "eval_steps_per_second":2.452
  },
  "timestamp": "2026-02-13 18:12:03.654321",
  "is_eval": true
}
```