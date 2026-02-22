# Model switching: config guide

## Overview

All three supported model architectures share the same training pipeline. Switching between them requires **one primary change** in the model registry, after which all derived training settings resolve automatically.

---

## The single switch

**File:** [cockatoo_ml/registry/model.py](/cockatoo_ml/registry/model.py)

```python
class ModelConfig:
    MODEL_TYPE = ModelType.CLIP_VIT      # change this
```

Available values:

| Value | Architecture | Model |
|---|---|---|
| `ModelType.CLIP_VIT` | Vision-text (CLIP) | `openai/clip-vit-large-patch14` |
| `ModelType.DEBERTA` | Text-only (DeBERTa) | `microsoft/deberta-v3-base` |
| `ModelType.MODERNBERT` | Text-only (ModernBERT) | `answerdotai/ModernBERT-large` |

Because `TrainingConfig`, `ModelConfig.ATTENTION_IMPLEMENTATION`, and `ModelConfig.INFERENCE_MAX_LENGTH` are all evaluated at class-definition time (module import), the correct per-model values are picked up automatically. There is no need to touch `training.py` or `paths.py` for a standard model switch.

---

## Auto-derived values

When `MODEL_TYPE` is changed, the following values resolve automatically from `ModelTrainingConfig` and `ModelConfig`:

| Setting | Resolved in |
|---|---|
| `TrainingConfig.BATCH_SIZE` | `cockatoo_ml/registry/training.py` |
| `TrainingConfig.NUM_EPOCHS` | `cockatoo_ml/registry/training.py` |
| `TrainingConfig.GRADIENT_ACCUMULATION_STEPS` | `cockatoo_ml/registry/training.py` |
| `TrainingConfig.LEARNING_RATE` | `cockatoo_ml/registry/training.py` |
| `TrainingConfig.WEIGHT_DECAY` | `cockatoo_ml/registry/training.py` |
| `TrainingConfig.WARMUP_RATIO` | `cockatoo_ml/registry/training.py` |
| `TrainingConfig.USE_FP16/BF16/TF32` | `cockatoo_ml/registry/training.py` |
| `ModelConfig.ATTENTION_IMPLEMENTATION` | `cockatoo_ml/registry/model.py` |
| `ModelConfig.INFERENCE_MAX_LENGTH` | `cockatoo_ml/registry/model.py` |
| `ModelConfig.get_base_model_name()` | `cockatoo_ml/registry/model.py` |
| `ModelConfig.get_max_token_length()` | `cockatoo_ml/registry/model.py` |

---

## Per-model optimal configs

### ModernBERT Large

**Model ID:** `answerdotai/ModernBERT-large`

```python
MODEL_TYPE = ModelType.MODERNBERT
```

| Parameter | Value | Set in |
|---|---|---|
| `BATCH_SIZE` | `8` | `ModelTrainingConfig.MODERNBERT_BATCH_SIZE` |
| `GRADIENT_ACCUMULATION_STEPS` | `12` | `ModelTrainingConfig.MODERNBERT_GRADIENT_ACCUMULATION_STEPS` |
| Effective batch size | **96** | (8 × 12) |
| `LEARNING_RATE` | `1e-5` | `ModelTrainingConfig.MODERNBERT_LEARNING_RATE` |
| `WEIGHT_DECAY` | `0.01` | `ModelTrainingConfig.MODERNBERT_WEIGHT_DECAY` |
| `WARMUP_RATIO` | `0.1` | `ModelTrainingConfig.MODERNBERT_WARMUP_RATIO` |
| `NUM_EPOCHS` | `3` | `ModelTrainingConfig.MODERNBERT_NUM_EPOCHS` |
| `USE_FP16` | `True` | `ModelTrainingConfig.MODERNBERT_USE_FP16` |
| `ATTENTION_IMPLEMENTATION` | `sdpa` | auto-derived in `ModelConfig` |
| Max token length (train) | `512` | `ModelConfig.MODERNBERT_MAX_TOKEN_LENGTH` |
| Max token length (inference) | `8192` | `ModelConfig.MODERNBERT_MAX_INFERENCING_TOKEN_LENGTH` |
| `OPTIMIZER` | `adamw_8bit` | `TrainingConfig.OPTIMIZER` (shared) |
| `LOSS_FUNCTION` | `asl` | `TrainingConfig.LOSS_FUNCTION` (shared) |

**Notes:**
- `sdpa` (scaled dot-product attention) is auto-selected; this is required for efficient long-context handling and is only supported by ModernBERT in this project.
- The low batch size (8) is intentional — ModernBERT-large is memory-intensive. Gradient accumulation is set high to compensate and maintain an effective batch of 96.
- Training token length is capped at 512 to save memory. Inference can use up to 8192 tokens.
- The `adamw_8bit` optimizer is strongly recommended for large models to reduce VRAM use during optimiser state storage.

---

### CLIP ViT-L/14

**Model ID:** `openai/clip-vit-large-patch14`

```python
MODEL_TYPE = ModelType.CLIP_VIT
```

| Parameter | Value | Set in |
|---|---|---|
| `BATCH_SIZE` | `16` | `ModelTrainingConfig.CLIP_BATCH_SIZE` |
| `GRADIENT_ACCUMULATION_STEPS` | `6` | `ModelTrainingConfig.CLIP_GRADIENT_ACCUMULATION_STEPS` |
| Effective batch size | **96** | (16 × 6) |
| `LEARNING_RATE` | `1e-5` | `ModelTrainingConfig.CLIP_LEARNING_RATE` |
| `WEIGHT_DECAY` | `0.01` | `ModelTrainingConfig.CLIP_WEIGHT_DECAY` |
| `WARMUP_RATIO` | `0.1` | `ModelTrainingConfig.CLIP_WARMUP_RATIO` |
| `NUM_EPOCHS` | `3` | `ModelTrainingConfig.CLIP_NUM_EPOCHS` |
| `USE_FP16` | `True` | `ModelTrainingConfig.CLIP_USE_FP16` |
| `ATTENTION_IMPLEMENTATION` | `default` | auto-derived in `ModelConfig` |
| Max token length (train) | `77` | `ModelConfig.CLIP_MAX_TOKEN_LENGTH` |
| Max token length (inference) | `77` | `ModelConfig.CLIP_MAX_INFERENCING_TOKEN_LENGTH` |
| `CLIP_PROJECTION_DIM` | `768` | `ModelConfig.CLIP_PROJECTION_DIM` |
| `OPTIMIZER` | `adamw_8bit` | `TrainingConfig.OPTIMIZER` (shared) |
| `LOSS_FUNCTION` | `asl` | `TrainingConfig.LOSS_FUNCTION` (shared) |

**Notes:**
- CLIP uses a custom `CLIPClassifier` wrapper (defined in `train/model_setup.py`) instead of `AutoModelForSequenceClassification`. This adds a two-layer classification head on top of the CLIP text/image embeddings.
- The token length limit of **77 is a hard constraint** imposed by CLIP's positional embedding table. Text beyond 77 tokens is truncated silently. If your inputs are typically longer, consider DeBERTa or ModernBERT.
- CLIP is the only multimodal option. If your dataset includes an `image` column, pixel values are extracted and averaged with the text embeddings automatically in `CLIPClassifier.forward()`.
- Lower learning rate (1e-5) because CLIP's pretrained weights are sensitive to large gradient updates.

---

### DeBERTa V3 Base

**Model ID:** `microsoft/deberta-v3-base`

```python
MODEL_TYPE = ModelType.DEBERTA
```

| Parameter | Value | Set in |
|---|---|---|
| `BATCH_SIZE` | `24` | `ModelTrainingConfig.DEBERTA_BATCH_SIZE` |
| `GRADIENT_ACCUMULATION_STEPS` | `4` | `ModelTrainingConfig.DEBERTA_GRADIENT_ACCUMULATION_STEPS` |
| Effective batch size | **96** | (24 × 4) |
| `LEARNING_RATE` | `2e-5` | `ModelTrainingConfig.DEBERTA_LEARNING_RATE` |
| `WEIGHT_DECAY` | `0.01` | `ModelTrainingConfig.DEBERTA_WEIGHT_DECAY` |
| `WARMUP_RATIO` | `0.1` | `ModelTrainingConfig.DEBERTA_WARMUP_RATIO` |
| `NUM_EPOCHS` | `3` | `ModelTrainingConfig.DEBERTA_NUM_EPOCHS` |
| `USE_FP16` | `True` | `ModelTrainingConfig.DEBERTA_USE_FP16` |
| `ATTENTION_IMPLEMENTATION` | `default` | auto-derived in `ModelConfig` |
| Max token length (train) | `256` | `ModelConfig.DEBERTA_MAX_TOKEN_LENGTH` |
| Max token length (inference) | `256` | `ModelConfig.DEBERTA_MAX_INFERENCING_TOKEN_LENGTH` |
| `OPTIMIZER` | `adamw_8bit` | `TrainingConfig.OPTIMIZER` (shared) |
| `LOSS_FUNCTION` | `asl` | `TrainingConfig.LOSS_FUNCTION` (shared) |

**Notes:**
- DeBERTa V3 Base is the lightest text model in the registry. It supports the largest per-device batch (24) and the fewest accumulation steps, making it the fastest to train.
- Token length of 256 covers the majority of short-to-medium text inputs (social media posts, comments). Extend `DEBERTA_MAX_TOKEN_LENGTH` and `DEBERTA_MAX_INFERENCING_TOKEN_LENGTH` if your data includes longer documents, at the cost of higher memory and training time.

---

## Step-by-step: switching models

1. **Open** [`cockatoo_ml/registry/model.py`](/cockatoo_ml/registry/model.py)
2. **Change** `MODEL_TYPE` to the desired value:
   ```python
   MODEL_TYPE = ModelType.MODERNBERT   # or CLIP_VIT, DEBERTA
   ```
3. **Verify** the per-model hyperparameters in [`cockatoo_ml/registry/training.py`](/cockatoo_ml/registry/training.py) under `ModelTrainingConfig`. Adjust any values (e.g. `MODERNBERT_BATCH_SIZE`) to match your hardware.
4. **Update** `PathConfig.MODEL_OUTPUT_DIR` and `PathConfig.LOGGING_DIR` in [`cockatoo_ml/registry/paths.py`](/cockatoo_ml/registry/paths.py) to avoid overwriting previous runs.
5. **Run** `prepare_data.py` if tokenization has not been done for the new model (tokenized cache is model-specific).
6. **Run** `train.py`.

---

## Shared settings that apply to all models

The following `TrainingConfig` values in [`cockatoo_ml/registry/training.py`](/cockatoo_ml/registry/training.py) are **not** model-specific and apply regardless of `MODEL_TYPE`:

| Setting | Value | Notes |
|---|---|---|
| `OPTIMIZER` | `adamw_8bit` | 8-bit AdamW; reduces VRAM for optimiser states |
| `LOSS_FUNCTION` | `asl` | Asymmetric Loss, good for multi-label classification (what we are doing here) |
| `ASL_GAMMA_NEG` | `4` | Suppresses easy-negative gradients |
| `ASL_GAMMA_POS` | `1` | Standard positive focusing |
| `ASL_CLIP` | `0.05` | Probability margin for negative shifting |
| `EVAL_STRATEGY` | `epoch` | Eval after every epoch |
| `METRIC_FOR_BEST_MODEL` | `f1` | Best checkpoint selected by macro-F1 |
| `DATALOADER_NUM_WORKERS` | `4` | Parallel data loading |
| `SAVE_TOTAL_LIMIT` | `2` | Only keep 2 most recent checkpoints |

To tune these for a specific model, you can override the value conditionally in `training.py` following the same pattern already used for batch size and learning rate.
