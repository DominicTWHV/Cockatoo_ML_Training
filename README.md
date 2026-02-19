# Cockatoo ML Training/Inferencing server

This repository is provided as a reference implementation for the training/inferencing server component of Cockatoo. We have designed it to fit our specific use case, but it was also designed with flexibility in mind.

[![CodeQL](https://github.com/DominicTWHV/Cockatoo_ML_Training/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/DominicTWHV/Cockatoo_ML_Training/actions/workflows/github-code-scanning/codeql)


> [!Important]
> This repository is highly experimental and is constantly being updated. It does not yet produce production-quality models. We are tinkering with different ideas and approaches, so expect breaking changes and refactors. We will eventually stabilize the codebase and produce production-ready models, but for now, this is a playground for us to experiment and iterate quickly.

---

## Model Architecture

**Current Default: CLIP ViT-L-14**

This codebase now defaults to training **CLIP ViT-L-14** for multi-modal content classification. The model can process:
- Text-only inputs
- Text + image inputs

**Also Supported: DeBERTa V3**

DeBERTa V3 text classification is still supported through configuration. To switch to DeBERTa:

1. Edit `cockatoo_ml/registry/model.py`
2. Change `MODEL_TYPE = ModelType.CLIP_VIT` to `MODEL_TYPE = ModelType.DEBERTA`

The codebase will automatically adapt to use the appropriate model architecture, tokenization, and training parameters.

---

## Training

The training code is built around torch with a pipeline pulling datasets from Hugging Face and pushing metrics to a custom API server.

We recommend running the training loop with a GPU-enabled device. Although training on CPU is possible without config changes, it will be significantly slower and would not be practical in this case (CLIP ViT-L-14 is a large model, CPU-based training can take weeks).

You may follow these steps to run the training loop on your local machine:

**Install Deps**

*Note: You may want to use a newer version of torch than the one specified in requirements.txt, so you can install it separately with the appropriate CUDA version for your system. We are using an older version due to hardware constrains.*

*Newer versions of torch may induce breakages, so if you do want to upgrade, you may have to fix some code in the training loop.*

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

#install torch separately
pip uninstall torch torchvision torchaudio -y # just in case torch exists in your old environment
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Tensorboard requires `pkg_resources`. This is buggy in newer Python versions. If you have problems getting Tensorboard running, try running:

```bash
pip uninstall setuptools
pip install setuptools==81.0
```

And then run the Tensorboard entrypoint as normal.

Verify that torch is installed correctly and can access your GPU:

```bash
python3 test_gpu.py
```

You should see your GPU being listed in the output along with the number of CUDA devices available.

**Run Training**

Load dataset from Hugging Face:

*If it fails, you may have to log into the Hugging Face CLI with `huggingface-cli login` or `hf auth login` (newer)*

```bash
python3 download_data.py
```

Preprocess dataset:

*Tip: review the rebalancing configs in `cockatoo_ml/registry/datasets.py` before running this step. You can adjust the rebalancing policy, weight calculation method, and other parameters to see how they affect the training process and model performance. See [rebalancing docs](docs/rebalancing.md) for more details on the rebalancing options and their implications.*

```bash
python3 prepare_data.py
```

> [!Important]
> Before you start the training loop, please review the trainer configs at `cockatoo_ml/registry/training.py` and adjust them as needed. 
> 
> **CLIP Training Notes:**
> - Default batch size is 16 (reduced from 24 for DeBERTa due to larger model size)
> - Learning rate is 1e-5 (lower than DeBERTa's 2e-5 for stability)
> - Gradient accumulation steps increased to 6 to maintain effective batch size
> - If you run into OOM errors, reduce batch size or increase gradient accumulation
> - CLIP benefits from mixed precision training (FP16 enabled by default)

Run training loop:

```bash
python3 train.py
```

**Run Evaluation**

*Note: The evaluation process is automatic when running a full training loop. The steps below are for evaluating a pre-existing model.*

```bash
python3 train.py --eval-only
```

This bypasses training and skips directly to loading a model from `PathConfig.MODEL_OUTPUT_DIR` and evaluating it against either the `test` (default) or `validation` dataset obtained earlier from pre-processing.

This behavior can be manually defined with

```bash
python3 train.py --eval-only --eval-split test #for testing against the test split

# or

python3 train.py --eval-only --eval-split validation #for testing against the validation split
```

**Metrics Telemetry**

We have integrated a hook to push training data live to an API server. You can modify the settings in `cockatoo_ml/registry/api.py`. If your remote API server expects an authentication token, edit the `.env` file to include the token.

The telemetry hook can be enabled independently for training and validation:

- Training telemetry: `WebhookConfig.enable` (uses `METRICS_WEBHOOK_URL`)
- Validation telemetry: `WebhookConfig.enable_validation` (uses `METRICS_VALIDATION_WEBHOOK_URL`)

This is the telemetry system we use at [cockatoo.dev](https://cockatoo.dev/ml-training.html) to monitor and publish training data across all our models in one place.

*Note: This telemetry is not something that sends us information. It is a system that you can use to send live training data to your own servers.*

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

**Evaluation Generation**

The training loop automatically plots evaluation results once available. You can also manualy pass in an evaluation JSON file to generate the evaluation plots:

```bash
python3 train.py --eval-data-json path/to/eval_results.json
```

Or with a custom output directory for the generated plots:

```bash
python3 train.py --plot-dir ./visualizations --eval-data-json path/to/eval_results.json
```

---

## Inferencing Server:

We have constructed our inferencing server around Quart + Hypercorn, which provides a simple and efficient way to serve our model. The server is designed to handle incoming requests, process them using the trained model, and return predictions in a timely manner.

**Run Inferencing Server**

*Alternatively, use the provided `start_api.sh` script to start the server*

```bash
source venv/bin/activate
hypercorn app:app --bind 0.0.0.0:8000
```

## Endpoints:

**Endpoint:** `/health`
**Method:** `GET`
**Description:** Checks if the server is responding and healthy. Returns a simple JSON response indicating the status and the active model version.

```json
{
  "status": "ok",
  "model": "constellation_one_text"
}
```

---

**Endpoint:** `/predict`
**Method:** `POST`
**Description:** Classifies a single string. Supports a global threshold or a per-label mapping.
**Request Body:**

```json
{
  "text": "The text to classify",
  "threshold": 0.5 
}

```

> **Note:** `threshold` is optional. It can be a **float** (0.5), a **dictionary** (`{"LABEL_1": 0.8, "LABEL_2": 0.4}`), or `null`. If `null`, the system defaults to per-label thresholds defined in the classifier configuration.

**Response Body:**

```json
{
  "text": "The text to classify",
  "predictions": {
    "LABEL_1": 0.12,
    "LABEL_2": 0.05,
    "LABEL_3": 0.9944
  },
  "positive_labels": ["LABEL_3"],
  "top_label": "LABEL_3",
  "max_score": 0.9944
}

```

> **Note:** The `predictions` field contains **all labels** from the model with their respective confidence scores, regardless of the threshold. The `positive_labels` field is a convenience filter showing only labels that meet the threshold criteria.

---

**Endpoint:** `/batch`
**Method:** `POST`
**Description:** Classifies a list of strings in a single request.
**Request Body:**

```json
{
  "texts": [
    "First text to classify",
    "Second text to classify"
  ],
  "threshold": {
    "LABEL_3": 0.90
  }
}

```

**Response Body:**

```json
{
  "count": 2,
  "results": [
    {
      "text": "First text to classify",
      "predictions": { "LABEL_1": 0.05, "LABEL_2": 0.02, "LABEL_3": 0.9944 },
      "positive_labels": ["LABEL_3"],
      "top_label": "LABEL_3",
      "max_score": 0.9944
    },
    {
      "text": "Second text to classify",
      "predictions": { "LABEL_1": 0.10, "LABEL_2": 0.05, "LABEL_3": 0.85 },
      "positive_labels": [],
      "top_label": "LABEL_3",
      "max_score": 0.85
    }
  ]
}

```

> **Note:** The `predictions` field in each result contains **all labels** from the model with their respective confidence scores, regardless of the threshold. The `positive_labels` field is a convenience filter showing only labels that meet the threshold criteria.

## Licensing:

This code is provided under the MIT License. Datasets are subject to their original licenses, which you should review before use:

[Datasets Used](docs/datasets.md)

*We have included links to a few datasets that are not used here. They will be used in future models when we revamp our image classification models.*

We greatly appreciate the open source community and the creators of these datasets for making their work available to the public. Please ensure that you comply with the terms of the original dataset licenses when using them in your projects.