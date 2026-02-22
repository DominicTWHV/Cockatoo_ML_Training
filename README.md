# Cockatoo ML Training/Inferencing server

This repository is what we plan to use for the training/inferencing server component of Cockatoo. We have designed it to fit our specific use case, but it was also designed with flexibility in mind.

[![CodeQL](https://github.com/DominicTWHV/Cockatoo_ML_Training/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/DominicTWHV/Cockatoo_ML_Training/actions/workflows/github-code-scanning/codeql)


> [!Important]
> This repository is highly experimental and is constantly being updated. It does not yet produce production-quality models. We are tinkering with different ideas and approaches, so expect breaking changes and refactors. We will eventually stabilize the codebase and produce production-ready models, but for now, this is a playground for us to experiment and iterate quickly.

---

## Model Architecture (currently supported):

**CLIP ViT-L-14** (HIGHLY EXPERIMENTAL - Untested)

**DeBERTa V3** (Stable)

**ModernBERT** (Experimental)

For config changes needed to switch between models, reference the model switching guide [here](/docs/model_switching.md).

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

Moved! Read the telemetry docs [here](/docs/telemetry.md)!

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

Read about the inference server API docs [here](/docs/api.md).

## Licensing:

This code is provided under the MIT License. Datasets are subject to their original licenses, which you should review before use:

[Datasets Used](docs/datasets.md)

*We have included links to a few datasets that are not used here. They will be used in future models when we revamp our image classification models.*

We greatly appreciate the open source community and the creators of these datasets for making their work available to the public. Please ensure that you comply with the terms of the original dataset licenses when using them in your projects.