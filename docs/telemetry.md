# Telemetry

We have embedded an internal http client to push training data live to an API server. You can modify the settings in `cockatoo_ml/registry/api.py`. If your remote API server expects an authentication token, edit the `.env` file to include the token.

The telemetry hook can be enabled independently for training and validation:

- Training telemetry: `WebhookConfig.enable` (uses `METRICS_WEBHOOK_URL`)
- Validation telemetry: `WebhookConfig.enable_validation` (uses `METRICS_VALIDATION_WEBHOOK_URL`)

This is the telemetry system we use at [cockatoo.dev](https://cockatoo.dev/ml-training.html) to monitor and publish training data across all our models in one place.

*Note: This telemetry is not something that sends us information. It is a system that you can use to send live training data to your own servers. It can be leveraged if you want to create the same metrics display we have at cockatoo.dev*

**Telemetry API Output**

Training endpoint payload:

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

Validation endpoint payload:

```json
{
  "experiment_id": "constellation-one-text-001",
  "global_step": 120,
  "epoch": 1.0,
  "metrics": {
    "eval_loss": 0.3891,
    "eval_f1": 0.9123,
    "eval_precision": 0.9011,
    "eval_recall": 0.9239
  },
  "timestamp": "2026-02-13 18:12:03.654321",
  "is_eval": true
}
```