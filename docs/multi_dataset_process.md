# Multi-dataset label merging: process and config changes

## Where to change things

### A) Set active labels
File: [labels.py](/cockatoo_ml/registry/labels.py)
- Use `LabelConfig.set_active_labels([...])` to choose a unified label list.
- Example labels: `scam`, `hate_speech`, `violence`, `harassment`, `toxicity`, `jailbreaking`, `dehumanization`, `status`, `obscenity`.

### B) Dataset label membership
File: [dataset_label_mapping.py](/cockatoo_ml/registry/dataset_label_mapping.py)
- Update `_initialize_defaults()` to specify which labels exist in each dataset.
- This powers label masking in training.

### C) Dataset column mappings
File: [column_mapping.py](/cockatoo_ml/registry/column_mapping.py)
- Map dataset columns to unified labels.
- Merge multi-column labels and apply thresholds.

### D) Data loading & processing
File: [data_loaders.py](/train/data_loaders.py)
- Extract columns using `DatasetColumnMapping`.

File: [data_processors.py](/train/data_processors.py)
- Convert extracted columns into unified label vectors.
- Add `_dataset_source` and `label_mask` for each sample.

### E) Training with masking
File: [trainer.py](/train/trainer.py)
- `CustomTrainer.compute_loss()` uses `label_mask` to avoid penalizing missing labels.

## End-to-end flow

1) Configure active labels.
2) Map each dataset’s columns to the unified labels.
3) Load datasets and extract label columns.
4) Build label vectors and add `label_mask`.
5) Train with masked loss so only valid labels contribute.

## Example config steps

1) Choose unified labels:
- Update active labels to the union of labels you want to train.

2) Map dataset columns to labels:
- Ensure each dataset’s raw columns are mapped to unified labels.
- Merge duplicate concepts (e.g., `toxic` + `severe_toxic` → `toxicity`).

3) Register dataset label membership:
- Make sure each dataset declares which unified labels it actually provides.
- This is required for correct masking.

## Notes

- Label masking is automatic once `label_mask` is present in the dataset.
- If a dataset does not provide a label, that label is masked for its samples.
- This prevents negative training signals for labels the dataset never contains.
