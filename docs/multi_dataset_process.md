# Labels & Datasets Configuration Guide

This guide explains how labels are defined, how datasets are mapped to those labels, and how to add new labels or new datasets.

---

## Overview: How the Pipeline Works

Data flows through four layers before reaching the trainer. Understanding them makes configuration straightforward:

```
1. LabelConfig           – declares every label that can exist (and labels that are active, i.e. trained on)
        ↓
2. DatasetColumnMapping  – maps raw dataset columns → label names
        ↓
3. DatasetLabelMapping   – declares which labels each dataset provides
        ↓
4. Data loaders          – load files, run column extraction, return (df, dataset_type)
```

The trainer uses the label mask from step 3 to ensure that loss is only computed on labels a given dataset actually annotates. For example, the phishing dataset only annotates `scam`, so all other label positions in its rows are masked out during training.

---

## Step 1 — Label Registry (`labels.py`)

**File:** [`cockatoo_ml/registry/labels.py`](../cockatoo_ml/registry/labels.py)

```python
class LabelConfig:
    ALL_LABELS    = ['scam', 'violence', 'nsfw', 'harassment',
                     'hate_speech', 'toxicity', 'obscenity', 'jailbreaking']
    ACTIVE_LABELS = ['scam', 'violence', 'harassment',
                     'hate_speech', 'toxicity', 'obscenity', 'jailbreaking']
```

| Field | Purpose |
|---|---|
| `ALL_LABELS` | Every label the system is aware of. Used for validation — nothing outside this list can be referenced elsewhere. |
| `ACTIVE_LABELS` | The labels actually used in the current training run. Controls the size of the output head and the order of every label vector. |

`ACTIVE_LABELS` can be a subset of `ALL_LABELS`. Labels in `ALL_LABELS` but not in `ACTIVE_LABELS` are simply never trained on; they don't need to be removed from column mappings or dataset registrations.

> **Order matters.** The position of each label in `ACTIVE_LABELS` defines its index in the output vector `[scam, violence, harassment, ...]`. Changing the order is a breaking change for saved checkpoints.

---

## Step 2 — Column Mapping (`column_mapping.py`)

**File:** [`cockatoo_ml/registry/column_mapping.py`](../cockatoo_ml/registry/column_mapping.py)

This class describes how to extract a normalised label value from the raw columns of each dataset file. Each dataset has a dict with two keys:

```python
SOME_DATASET = {
    'text_col': '<name of the text column in the raw file>',
    'labels': {
        '<label_name>': '<raw_column>'           # single column
        '<label_name>': ['<col_a>', '<col_b>'],  # merge multiple columns
    }
}
```

### Single column

```python
'toxicity': 'toxicity'
```

The raw column is read as-is. If it is a float column it is binarised using the threshold from `LABEL_THRESHOLDS` (default 0.5). If it is already int/bool it is cast to int directly.

### Multiple columns (merged)

```python
'harassment': ['insult', 'humiliate', 'dehumanize', 'attack_defend']
```

All listed columns are combined using `DATASET_MERGING_STRATEGY` (default `'or'`). With `'or'`, the resulting label is 1 if **any** column is 1. See the merge strategies table below.

### Merge strategies

| Strategy | Behaviour | When to use |
|---|---|---|
| `'or'` | 1 if any column ≥ 1 | Boolean signals, any-positive is sufficient |
| `'and'` | 1 only if all columns ≥ 1 | Require consensus across signals |
| `'max'` | max value across columns | Continuous scores, want the highest |
| `'mean'` | mean value across columns | Continuous scores, want the average |

`DATASET_MERGING_STRATEGY` sets the global default. You can also pass a strategy explicitly to `merge_multi_column_labels(df, cols, strategy)` if you need per-label control in a custom loader.

### Thresholds

`LABEL_THRESHOLDS` provides a per-label cutoff for continuous float columns:

```python
LABEL_THRESHOLDS = {
    'toxicity': 0.5,
    'hate_speech': 0.5,
    ...
}
```

If a label does not appear here, `get_label_threshold()` returns 0.5 as a default.

### Registering a dataset mapping

Add the dict as a class attribute and add it to `DATASET_MAPPINGS`:

```python
MY_DATASET = {
    'text_col': 'body',
    'labels': {
        'toxicity': 'tox_score',           # float column, will be thresholded
        'harassment': ['rude', 'abusive'],  # OR-merged
    }
}

DATASET_MAPPINGS = {
    ...
    'my_dataset': MY_DATASET,
}
```

The key in `DATASET_MAPPINGS` must match the `dataset_type` string returned by the data loader (see step 4).

---

## Step 3 — Dataset Label Mapping (`dataset_label_mapping.py`)

**File:** [`cockatoo_ml/registry/dataset_label_mapping.py`](../cockatoo_ml/registry/dataset_label_mapping.py)

This registry declares **which labels each dataset provides**. It drives the label mask used during training — columns for labels a dataset does not annotate are masked out so they don't contribute to the loss.

```python
_mappings = {
    'phishing': {
        'labels': ['scam'],
        'description': 'Phishing dataset - contains scam labels'
    },
    'jigsaw': {
        'labels': ['toxicity', 'obscenity', 'violence', 'harassment', 'hate_speech'],
        'description': 'Jigsaw toxicity - multi-label toxic content classification'
    },
    ...
}
```

The `labels` list must be a subset of `LabelConfig.ALL_LABELS`. If you include a label that isn't in `ALL_LABELS`, `register()` will raise a `ValueError` at startup.

> **Labels here must match what the column mapping actually extracts.** If `column_mapping.py` extracts a `toxicity` column but the dataset is registered here with `labels: ['hate_speech']`, the `toxicity` column will be silently ignored.

To add a new dataset registration:

```python
mapping = get_dataset_label_mapping()
mapping.register(
    'my_dataset',
    labels=['toxicity', 'harassment'],
    description='My custom dataset'
)
```

Or add it directly to `_initialize_defaults()` for it to be available on every run without any extra call.

---

## Step 4 — Data Loaders (`data_loaders.py`)

**File:** [`train/data_loaders.py`](../train/data_loaders.py)

Each loader:
1. Reads the raw file from disk.
2. Optionally pre-processes columns (e.g. converts a multiclass label to binary).
3. Calls `extract_labels_from_df(df, mapping, dataset_name)` which applies the column mapping.
4. Returns `(df, dataset_type)` where `dataset_type` is the key used in steps 2 and 3.

`load_all_datasets()` calls every registered loader in order and collects the results.

### Standard loader

Most datasets need only a few lines:

```python
def load_my_dataset(base_dir=None):
    if base_dir is None:
        base_dir = PathConfig.BASE_DATA_DIR

    path = os.path.join(base_dir, 'my_dataset', 'train.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        logger.info(f"My dataset raw columns: {df.columns.tolist()}")

        mapping = DatasetColumnMapping.get_mapping('my_dataset')
        df = extract_labels_from_df(df, mapping, 'my_dataset')

        if df is not None:
            df = df.dropna(subset=[DatasetColumns.TEXT_COL])
            logger.info(f"My dataset loaded: {len(df)} samples")
            return df, 'my_dataset'
    else:
        logger.warning("My dataset file not found")

    return None, None
```

Then add it to `load_all_datasets`:

```python
loaders = [
    ...
    load_my_dataset,
]
```

### Loader with pre-processing

If the raw dataset doesn't directly expose boolean labels (e.g. a multiclass column), transform the column before handing off to `extract_labels_from_df`. The tweet_eval emotion dataset is an example — it has a `label` column with values `0=anger, 1=joy, 2=optimism, 3=sadness`:

```python
# convert multiclass → binary before extraction
df['anger'] = (df['label'] == 0).astype(int)

mapping = DatasetColumnMapping.get_mapping('tweet_emotion')
df = extract_labels_from_df(df, mapping, 'tweet_emotion')
```

The column mapping then routes the synthetic `anger` column to the `hate_speech` label.

---

## Complete Walkthrough: Adding a New Label

### Example: adding a `spam` label

**1. Add to `LabelConfig`**

```python
# cockatoo_ml/registry/labels.py
ALL_LABELS    = [..., 'spam']
ACTIVE_LABELS = [..., 'spam']   # include only when ready to train. Labels defined in ALL_LABELS but not in ACTIVE_LABELS are ignored by the system, so you can add to ALL_LABELS early without breaking anything.

# ACTIVE_LABELS exists to control what the model is trained on, and ALL_LABELS exists to verify ACTIVE_LABELS are valid.
```

**2. Add a threshold (optional)**

```python
# cockatoo_ml/registry/column_mapping.py — LABEL_THRESHOLDS
'spam': 0.5,
```

**3. Map it in each relevant dataset**

For any dataset whose raw columns can indicate spam, add the mapping:

```python
MY_DATASET = {
    'text_col': 'text',
    'labels': {
        ...
        'spam': 'is_spam', # raw boolean column
    }
}
```

**4. Register it in `DatasetLabelMapping`**

```python
'my_dataset': {
    'labels': [..., 'spam'],
    'description': '...'
},
```

That's all. The label vector builder (`LabelConfig.make_labels`) and mask logic pick up the new label automatically.

---

## Complete Walkthrough: Adding a New Dataset

### Example with the phishing dataset

**1. Add the download source** *(if from Hugging Face)*

This configures where to download the data from (HF only)

```python
# cockatoo_ml/registry/datasets.py — DatasetSources.DATASETS
("ealvaradob/phishing-dataset", "phishing"),
```

**2. Add path constants**

```python
# cockatoo_ml/registry/datasets.py — DatasetPaths
PHISHING_FILE = "combined_reduced.json"
PHISHING_DIR = "phishing"
```

**3. Add a column mapping**

```python
# cockatoo_ml/registry/column_mapping.py
PHISHING = {
        'text_col': 'text',
        'labels': {
            'scam': 'label'  # binary column - 1 if phishing, 0 otherwise
        }
    }

DATASET_MAPPINGS = {
    'phishing': PHISHING,
    ...
}
```

**4. Register labels**

```python
# cockatoo_ml/registry/dataset_label_mapping.py — _initialize_defaults
'phishing': {
    'labels': ['scam'],
    'description': 'Phishing dataset - contains scam labels'
},
```

**5. Write the loader**

```python
# train/data_loaders.py
def load_phishing_dataset(base_dir=None):
    if base_dir is None:
        base_dir = PathConfig.BASE_DATA_DIR

    phish_path = os.path.join(base_dir, DatasetPaths.PHISHING_DIR, DatasetPaths.PHISHING_FILE)
    if os.path.exists(phish_path):
        df_phish = pd.read_json(phish_path)
        logger.info(f"Phishing raw columns: {df_phish.columns.tolist()}")
        
        mapping = DatasetColumnMapping.get_mapping('phishing')
        df_phish = extract_labels_from_df(df_phish, mapping, 'phishing')
        
        if df_phish is not None:
            df_phish = df_phish.dropna(subset=[DatasetColumns.TEXT_COL])
            logger.info(f"Phishing loaded: {len(df_phish)} samples")
            return df_phish, 'phishing'
    
    else:
        logger.warning("Phishing file not found")

    return None, None
```

**6. Register the loader**

```python
# train/data_loaders.py — load_all_datasets
loaders = [
    load_phishing_dataset,
    ...
]
```

---

## Reference: Currently Configured Datasets

| Dataset key | Labels annotated | Text column | Notes |
|---|---|---|---|
| `phishing` | `scam` | `text` | Binary `label` column |
| `hate_speech` | `hate_speech`, `violence`, `harassment` | `text` | Boolean columns; violence+genocide merged; harassment is OR of insult/humiliate/dehumanize/attack_defend |
| `tweet_hate` | `hate_speech` | `text` | Binary `label` column (1=hate) |
| `tweet_emotion` | `hate_speech` | `text` | Multiclass `label` pre-processed: anger (0) → hate_speech |
| `toxicchat` | `toxicity`, `jailbreaking` | `user_input` | Continuous toxicity, thresholded at 0.5 |
| `jigsaw` | `toxicity`, `obscenity`, `violence`, `harassment`, `hate_speech` | `comment_text` | toxic+severe_toxic OR-merged; threat→violence; insult→harassment; identity_hate→hate_speech |

## Reference: Currently Defined Labels

| Label | Included in `ACTIVE_LABELS` | Notes |
|---|---|---|
| `scam` | ✅ | Phishing / fraud content |
| `violence` | ✅ | Threats, violent language, genocide references |
| `harassment` | ✅ | Insults, humiliation, targeted abuse |
| `hate_speech` | ✅ | Identity-based hate |
| `toxicity` | ✅ | General toxic language |
| `obscenity` | ✅ | Obscene/explicit language |
| `jailbreaking` | ✅ | LLM jailbreak attempts |
| `nsfw` | ❌ | In `ALL_LABELS` but not active — no dataset currently annotates this | (this is meant for future vision support, so although CLIP-ViT is "supported", its untested and experimental.)
