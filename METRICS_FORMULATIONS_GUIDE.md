# REFACTORED evaluation_metrics.py - Integration Guide

## ✅ What Changed

### Input Format (CRITICAL)
**BEFORE:**
```python
evaluator.compute_all_metrics(
    X_train_real, X_test_real, X_synth,
    y_train_real, y_test_real, y_synth
)
```

**NOW:**
```python
evaluator.compute_all_metrics(X_test, X_synth)
# Where:
# X_test shape: (n_samples, n_features + 1 target)
# X_synth shape: (m_samples, n_features + 1 target)
# Target is LAST column (index -1)
```

### Data Handling
✅ **Automatically separates features from target** (last column)
✅ **Utility metrics internally split real data 80/20**
✅ **Trains XGBoost on real_train, synth_train**
✅ **Tests on real_test ONLY (synth_test not used)**
✅ **Other metrics use FULL datasets (no split)**

---

## 🎯 Quick Start

```python
import numpy as np
from evaluation_metrics import MetricsEvaluator

# Load your data
X_test = np.load('california_housing_test.npy')  # shape: (n, p+1)
X_synth = np.load('synthetic_data.npy')           # shape: (m, p+1)

# Create evaluator
evaluator = MetricsEvaluator(include_authenticity=True)

# Compute all metrics
metrics = evaluator.compute_all_metrics(X_test, X_synth)

# Access results
print(f"KS: {metrics['ks_statistic_mean']:.4f}")
print(f"ML Gap: {metrics['ml_efficiency_gap_percent']:.2f}%")
print(f"Authenticity: {metrics['authenticity']:.4f}")
```

---

## 📊 Metrics Computation

### Distribution Metrics (Use FULL data)
```
✅ KS Statistic
✅ Wasserstein Distance
✅ KL Divergence
✅ JS Divergence
✅ MMD
✅ SWD
```
Process: Separate features, compute on features only (not target)

### Correlation Metrics (Use FULL data)
```
✅ Correlation Similarity
✅ Pairwise Correlation Difference
✅ Correlation Distance
```
Process: Separate features, compute correlation on features only

### Utility Metrics (INTERNAL 80/20 SPLIT)
```
✅ R² Score (Real)
✅ R² Score (Synthetic)
✅ RMSE (Real & Synthetic)
✅ ML Efficiency Gap %
```
Process:
1. Split X_test 80/20 internally
2. Train XGBoost on X_test_train and X_synth
3. Test on X_test_test ONLY
4. Compute metrics

### Privacy Metrics (Use FULL data)
```
✅ DCR Mean
✅ DCR Median
✅ DCR Share
✅ Identical Matches
✅ Authenticity
```
Process: Use full data for distance computations

### Detection Metrics (Use FULL data)
```
✅ C2ST Accuracy
✅ C2ST AUC
✅ C2ST P-value
```
Process: Use full data (features + target)

---

## 🔧 5-Fold CV Integration

```python
from sklearn.model_selection import KFold
import json

# Setup
evaluator = MetricsEvaluator(include_authenticity=True)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
all_results = []

# Load full data
X_real_full = np.load('california_housing_full.npy')  # (N, p+1)

# 5-Fold CV
for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_real_full)):
    print(f"Fold {fold_idx + 1}/5...")
    
    # Split real data
    X_test = X_real_full[test_idx]  # Real test data
    
    # Generate synthetic data from train set (your DSBM/ASBM code)
    X_train_real = X_real_full[train_idx]
    X_synth = your_model.generate(X_train_real)  # Shape: (N', p+1)
    
    # Evaluate
    metrics = evaluator.compute_all_metrics(X_test, X_synth)
    metrics['fold'] = fold_idx + 1
    all_results.append(metrics)
    
    print(f"  KS: {metrics['ks_statistic_mean']:.4f}")
    print(f"  ML Gap: {metrics['ml_efficiency_gap_percent']:.2f}%")

# Aggregate results
results = {
    "dataset": "california_housing",
    "model": "DSBM",
    "folds": all_results,
    "summary": {
        "metrics_mean": {k: np.mean([f[k] for f in all_results])
                         for k in all_results[0].keys() if k != 'fold'},
        "metrics_std": {k: np.std([f[k] for f in all_results])
                        for k in all_results[0].keys() if k != 'fold'}
    }
}

# Save
with open('results_dsbm_5fold.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
```

---

## ⚙️ Configuration Options

```python
# Default (recommended)
evaluator = MetricsEvaluator()

# Fast (skip expensive metrics)
evaluator = MetricsEvaluator(
    include_alpha_precision=False,  # Skip (expensive, ~1 min)
    include_c2st=False              # Skip (expensive, ~2 min)
)

# Full evaluation
evaluator = MetricsEvaluator(
    include_alpha_precision=True,   # Include
    include_c2st=True               # Include
)

# Custom
evaluator = MetricsEvaluator(
    include_alpha_precision=False,
    include_c2st=False,
    include_authenticity=True,      # Privacy metric
    verbose=True                    # Print warnings
)
```

---

## 📋 Enabled Metrics (by default)

```python
evaluator = MetricsEvaluator()
enabled = evaluator.get_enabled_metrics()
print(f"Computing {len(enabled)} metrics:")
for m in enabled:
    print(f"  - {m}")
```

Output:
```
Computing 20 metrics:
  - ks_statistic_mean
  - ks_statistic_per_feature_mean
  - wasserstein_distance
  - kl_divergence
  - js_divergence
  - mmd
  - swd
  - correlation_similarity
  - pairwise_correlation_diff
  - correlation_distance
  - r2_score_real
  - r2_score_synth
  - rmse_real
  - rmse_synth
  - ml_efficiency_gap_percent
  - dcr_mean
  - dcr_median
  - dcr_share
  - identical_matches_fraction
  - authenticity
```

---

## 🔍 Data Format Examples

### California Housing Example
```python
# X_test (real test data)
X_test.shape = (3819, 14)  # 13 features + 1 target
X_test[:5, :] =
  [13.55   1.  58.  1077.  17.8   26.28   27.75  12. ...]
  [13.16   1.  37. 1078.  18.3   23.75   27.75  12. ...]
  [14.31   1.  49. 1099.  17.4   24.84   27.75  12. ...]
  ...

# X_synth (generated synthetic data)
X_synth.shape = (3819, 14)  # Same structure
X_synth[:5, :] =
  [13.42   1.05  59.1 1055.2  18.1   26.15  27.82  11.9 ...]
  [13.28   0.98  36.8 1089.3  18.2   23.62  27.71  12.1 ...]
  [14.25   1.02  50.2 1102.5  17.3   24.91  27.78  12.0 ...]
  ...

# Last column is target (price in this case)
y_test = X_test[:, -1]
y_synth = X_synth[:, -1]
```

---

## ✅ What the Refactored Code Does

### For Distribution/Correlation/Privacy/Detection Metrics:
```python
X_test_feat, y_test = separate_features_and_target(X_test)
X_synth_feat, y_synth = separate_features_and_target(X_synth)
# Compute metrics on full datasets (X_test_feat, X_synth_feat)
```

### For Utility Metrics:
```python
X_test_feat, y_test = separate_features_and_target(X_test)
X_synth_feat, y_synth = separate_features_and_target(X_synth)

# Split X_test 80/20
X_train_real, X_test_real, y_train_real, y_test_real = \
    train_test_split(X_test_feat, y_test, test_size=0.2)

# Train on real and synthetic
model_real = XGBRegressor().fit(X_train_real, y_train_real)
model_synth = XGBRegressor().fit(X_synth_feat, y_synth)

# Test on real test set only
y_pred_real = model_real.predict(X_test_real)
y_pred_synth = model_synth.predict(X_test_real)  # <-- Uses real test

# Compute R2, RMSE on real test set
r2_real = r2_score(y_test_real, y_pred_real)
r2_synth = r2_score(y_test_real, y_pred_synth)
```

---

## 🚀 Migration Checklist

- [ ] Replace `evaluation_metrics.py` with `evaluation_metrics_refactored.py`
- [ ] Update imports (if needed, should be same)
- [ ] Change function call signature from 6 args to 2 args
- [ ] Ensure data has target in last column
- [ ] Test on sample data first
- [ ] Run full 5-fold CV
- [ ] Verify results match expected ranges
- [ ] Save JSON output

---

## 📊 Expected Output

```python
metrics = evaluator.compute_all_metrics(X_test, X_synth)

# Example output
{
    'ks_statistic_mean': 0.0523,
    'ks_statistic_per_feature_mean': 0.0487,
    'wasserstein_distance': 0.1234,
    'kl_divergence': 0.3456,
    'js_divergence': 0.0789,
    'mmd': 0.2134,
    'swd': 0.0945,
    'correlation_similarity': 0.8723,
    'pairwise_correlation_diff': 0.4567,
    'correlation_distance': 0.4567,
    'r2_score_real': 0.5678,
    'r2_score_synth': 0.5234,
    'rmse_real': 4.3210,
    'rmse_synth': 4.6789,
    'ml_efficiency_gap_percent': 7.8123,
    'dcr_mean': 0.7234,
    'dcr_median': 0.6789,
    'dcr_share': 0.4567,
    'identical_matches_fraction': 0.0012,
    'authenticity': 0.8901
}
```

---

## 🔧 Troubleshooting

### Issue: Target column not last
**Fix:** Ensure your data has target as LAST column
```python
# If target is first column
X = np.column_stack([X_features, X_target])  # target last
```

### Issue: Wrong number of features
**Fix:** Check shape matches
```python
assert X_test.shape[1] == X_synth.shape[1], "Feature count mismatch"
assert X_test.shape[1] > 1, "Need at least features + target"
```

### Issue: NaN in results
**Fix:** Check data quality
```python
assert not np.isnan(X_test).any(), "NaN in X_test"
assert not np.isnan(X_synth).any(), "NaN in X_synth"
```

### Issue: Memory error
**Fix:** Reduce sample size for expensive metrics
```python
# Sample data if too large
if X_test.shape[0] > 10000:
    idx = np.random.choice(len(X_test), 10000, replace=False)
    X_test = X_test[idx]
```

---

## 📚 File Locations

```
your_project/
├── evaluation_metrics.py         ← Use refactored version
├── dsbm_eval_cv_5fold.py        ← Update to use new API
├── asbm_eval_cv_5fold.py        ← Update to use new API
├── xgboostdsbm_eval_cv_5fold.py ← Update to use new API
└── data/
    ├── X_test_*.npy             ← Real test data
    └── X_synth_*.npy            ← Synthetic data
```

---

## ✅ You're Ready!

1. Use `evaluation_metrics_refactored.py` (rename to `evaluation_metrics.py`)
2. Call: `evaluator.compute_all_metrics(X_test, X_synth)`
3. Get back dict with 20+ metrics
4. That's it!

---

**Version**: 2.0 (Refactored)
**Status**: Ready for Use
**Last Updated**: December 28, 2025
