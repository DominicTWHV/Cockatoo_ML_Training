# Cockatoo ML Training/Inferencing server

This repository is provided as a reference implementation for the training/inferencing server component of Cockatoo. We have designed it to fit our specific use case, but it was also designed with flexibility in mind.

[![CodeQL](https://github.com/DominicTWHV/Cockatoo_ML_Training/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/DominicTWHV/Cockatoo_ML_Training/actions/workflows/github-code-scanning/codeql)

---

## Training

The training code is built around torch with a pipeline pulling datasets from Hugging Face and pushing metrics to a custom API server.

We recommend running the training loop with a GPU-enabled device. Although training on CPU is possible without config changes, it will be significantly slower and would not be practical in this case (processed dataset size ~493 MB, CPU-based training can easily take multiple days to complete).

You may follow these steps to run the training loop on your local machine:

**Install Deps**

*Note: You may want to use a newer version of torch than the one specified in requirements.txt, so you can install it separately with the appropriate CUDA version for your system. We are using an older version due to hardware constrains.*

*Newer versions of torch may induce breakages, so if you do want to upgrade, you may have to fix some code in the training loop.*

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

#install torch separately
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Verify that torch is installed correctly and can access your GPU:

```bash
python3 test_gpu.py
```

You should see your GPU being listed in the output along with the number of CUDA devices available.

**Run Training**

Load dataset from Hugging Face:

*If it fails, you may have to log into the Hugging Face CLI with `huggingface-cli login`*

```bash
python3 download_data.py
```

Preprocess dataset:

```bash
python3 prepare_data.py
```

> [!Important]
> Before you start the training loop, please review the trainer configs at `cockatoo_ml/registry/training.py` and adjust them as needed. For example, if you run into OOM errors, you may want to reduce the batch size, or increase it if you desire a faster training speed and have the hardware to support it.

Run training loop:

```bash
python3 train_text.py
```

**Metrics Telemetry**

We have integrated a hook to push training data live to an API server. You can modify the settings in `cockatoo_ml/registry/api.py`. If your remote API server expects an authentication token, edit the `.env` file to include the token.

The metrics hook can also be disabled entirely by setting `WebhookConfig.enable` to `False` in `cockatoo_ml/registry/api.py`.

This is the telemetry system we use at [cockatoo.dev](https://cockatoo.dev/ml-training.html) to monitor and publish training data across all our models in one place.

---

## Inferencing Server:

We have constructed our inferencing server around Quart + Hypercorn, which provides a simple and efficient way to serve our model. The server is designed to handle incoming requests, process them using the trained model, and return predictions in a timely manner.

**Run Inferencing Server**

*Alternatively, use the provided `start_api.sh` script to start the server*

```bash
source venv/bin/activate
hypercorn app:app --bind 0.0.0.0:8000
```

**Endpoints:**

/health
- Method: GET
- Description: Checks if the server is responding and healthy. Returns a simple JSON response indicating the status.
- Response Body:
```json
{
  "status": "ok",
  "model": "constellation_one_text"
}
```

/predict
- Method: POST
- Description: Accepts a JSON payload containing the text to classify and an optional confidence threshold.
- Request Body:
```json
{
  "text": "The text to classify",
  "threshold": 0.5
}
```

- Response Body (sample, not exhaustive):

```json
{
  "error": null,
  "max_score": 0.9944,
  "positive_labels": [
    "LABEL_3"
  ],
  "predictions": {
    "LABEL_3": 0.9944
  },
  "text": "input text here",
  "top_label": "LABEL_3"
}
```

/batch
- Method: POST
- Description: Accepts a JSON payload containing a list of texts to classify and an optional confidence threshold.
- Request Body:
```json
{
  "texts": [
    "First text to classify",
    "Second text to classify"
  ],
  "threshold": 0.5
}
```

 - Response Body (sample, not exhaustive):

```json
{
  "count": 2,
  "results": [
    {
      "error": null,
      "max_score": 0.9944,
      "positive_labels": ["LABEL_3"],
      "predictions": {
        "LABEL_3": 0.9944
      },
      "text": "first text to classify",
      "top_label": "LABEL_3"
    },
    {
      "error": null,
      "max_score": 0.9944,
      "positive_labels": ["LABEL_3"],
      "predictions": {
        "LABEL_3": 0.9944
      },
      "text": "second text to classify",
      "top_label": "LABEL_3"
    }
  ]
}
```

## Licensing:

This code is provided under the MIT License. Datasets are subject to their original licenses, which you should review before use:

[Datasets Used](datasets.md)

*We have included links to a few datasets that are not used here. They will be used in future models when we revamp our image classification models.*

We greatly appreciate the open source community and the creators of these datasets for making their work available to the public. Please ensure that you comply with the terms of the original dataset licenses when using them in your projects.