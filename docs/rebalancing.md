# Dataset Rebalancing Guide

This document describes the policy-based dataset rebalancing system implemented in the training loop.

## Overview

The rebalancing system provides multiple strategies to handle class imbalance in training data:

1. **Oversampling**: Duplicates minority samples to match the majority class size
2. **Undersampling**: Removes majority samples to match the minority class size
3. **Reweighting**: Assigns higher loss weights to minority classes
4. **Combined**: Applies both oversampling and reweighting

## Configuration

Rebalancing is configured in [cockatoo_ml/registry/datasets.py](cockatoo_ml/registry/datasets.py) via the `DataSplitConfig` class:

### Rebalancing Policy

```python
REBALANCING_POLICY = RebalancingPolicy.REWEIGHTING
```

Choose from:
- `RebalancingPolicy.OVERSAMPLING`: Upsample minority classes
- `RebalancingPolicy.UNDERSAMPLING`: Downsample majority classes
- `RebalancingPolicy.REWEIGHTING`: Calculate loss weights (recommended)
- `RebalancingPolicy.COMBINED`: Both strategies
- `None`: No rebalancing

### Weight Calculation Method

```python
WEIGHT_CALCULATION = "inverse_frequency"
```

Available methods for reweighting policy:

#### 1. Inverse Frequency (Default)
```
weights = total_samples / (num_classes * class_count)
```

Assigns weight proportional to how rare each class is. Most intuitive approach.

**Pros:**
- Simple and mathematically principled
- Linearly scales with class frequency
- Good general-purpose choice

**Cons:**
- Can produce extreme weights for very rare classes

#### 2. Effective Number
```
weights = (1 - beta) / (1 - beta^n)
```

Uses the effective number of samples formula (common in long-tail recognition).

**Pros:**
- Smoother weight scaling for rare classes
- Better for severely imbalanced data
- Prevents extreme weight values

**Cons:**
- Does not account for instance-level difficulty (all samples of a class get same weight)
- May require tuning of beta parameter for best results

#### 3. Square Root Inverse (Gentler)
```
weights = sqrt(total_samples / class_count)
```

A gentler version of inverse frequency.

**Pros:**
- Moderate emphasis on rare classes
- More stable than pure inverse frequency
- Good middle ground

**Cons:**
- May not sufficiently penalize missing common classes

### Weight Smoothing

```python
WEIGHT_SMOOTHING = 1e-6
```

Small constant added to prevent:
- Division by zero for classes with zero samples
- Infinite weights for empty classes
- Numerical instability

## How It Works

### Training Pipeline

1. **Data Loading**: Load and combine datasets from multiple sources
2. **Deduplication**: Remove duplicate texts with conflict handling
3. **Rebalancing** (applied to training split only):
    - **Oversampling**: Upsample minority label combinations
    - **Undersampling**: Downsample majority label combinations
    - **Reweighting**: Calculate per-class loss weights
    - **Combined**: Do both

4. **Dataset Split**: Split into train/val/test with stratification
5. **Training**: Use calculated weights in loss function

### Class Weight Calculation

For multi-label classification:

1. Convert label vectors to a binary matrix (num_samples × num_classes)
2. Count positive samples per class
3. Calculate weights based on chosen method
4. Normalize to have mean of 1.0 for proper loss scaling

## Usage Examples

### Example 1: Reweighting Only (Recommended)

```python
# In cockatoo_ml/registry/datasets.py
class DataSplitConfig:
    REBALANCING_POLICY = RebalancingPolicy.REWEIGHTING
    WEIGHT_CALCULATION = "inverse_frequency"
    WEIGHT_SMOOTHING = 1e-6
```

**When to use:**
- You want to keep your dataset size unchanged
- Your model supports weighted loss (our BCEWithLogitsLoss does)
- You want maximum training efficiency

### Example 2: Oversampling Only

```python
class DataSplitConfig:
    REBALANCING_POLICY = RebalancingPolicy.OVERSAMPLING
```

**When to use:**
- You want to increase training signal for minority classes
- You have GPU memory to spare
- You're willing to accept longer training times
- Your model doesn't support class weights well

### Example 3: Combined Strategy

```python
class DataSplitConfig:
    REBALANCING_POLICY = RebalancingPolicy.COMBINED
```

**When to use:**
- Severe class imbalance (e.g., 1:100 ratio)
- You have sufficient GPU memory
- You want maximum emphasis on minority classes

### Example 4: Undersampling Only

```python
class DataSplitConfig:
    REBALANCING_POLICY = RebalancingPolicy.UNDERSAMPLING
```

**When to use:**
- You want to reduce the training set size for faster iterations
- The dataset is large and highly imbalanced
- You can afford to discard majority samples

### Example 5: No Rebalancing

```python
class DataSplitConfig:
    REBALANCING_POLICY = None
```

**When to use:**
- Data is already well-balanced

## Performance Considerations

### Oversampling
- **Data size increase**: Proportional to imbalance ratio
- **Training time**: Increases with dataset size
- **Memory**: May require more GPU memory (bigger dataset size, more memory required)

### Undersampling
- **Data size decrease**: Down to the minority class size
- **Training time**: Decreases with dataset size
- **Memory**: Lower requirements
- **When**: Large datasets or rapid iteration needs

### Reweighting
- **Data size**: No increase
- **Training time**: Slightly faster (same dataset size)
- **Memory**: Minimal overhead
- **When**: Most use cases

### Combined
- **Data size**: Increases like oversampling
- **Training time**: Increases due to both dataset expansion and weight scaling
- **When**: Severe imbalance with sufficient resources

## Debugging Class Imbalance

Check logs during training:

```
[INFO] Class counts: [100, 50, 75]
[INFO] Weight calculation method: inverse_frequency
[INFO] Per-class weights: [0.83, 1.66, 1.10]
[INFO] Per-sample weight stats - min: 0.83, max: 1.66, mean: 1.00
```

The minority class (50 samples) gets the highest weight (1.66), while the majority class (100 samples) gets the lowest weight (0.83).

*This is a sample log output for a dataset with three classes.*