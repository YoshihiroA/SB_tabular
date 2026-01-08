# 5-Fold Evaluation: Aggregation & Visualization

## Overview

After running 5-fold cross-validation for each model across multiple datasets, two complementary aggregation approaches are used to visualize and compare results:

1. **Grouped by Dataset**: Compare all models on each dataset (per-metric subplots)
2. **Grouped by Metric**: Compare all models across all datasets (per-metric full comparison)

---

## Data Structure: Input to Aggregation

### Output from 5-Fold Evaluation

Each model evaluation produces JSON files with structure:

```json
{
  "metadata": {
    "dataset": "california_housing",
    "model": "DSBM",
    "n_folds": 5,
    "n_metrics": 20
  },
  "folds": [
    { "fold": 1, "ks_statistic_mean": 0.0456, "wasserstein_distance": 0.1234, ... },
    { "fold": 2, "ks_statistic_mean": 0.0412, "wasserstein_distance": 0.1189, ... },
    ...
  ],
  "summary": {
    "metrics_mean": {
      "ks_statistic_mean": 0.0423,
      "wasserstein_distance": 0.1210,
      "r2_score_real": 0.8945,
      "r2_score_synth": 0.8234,
      "rmse_real": 0.1456,
      "rmse_synth": 0.1678,
      ...
    },
    "metrics_std": {
      "ks_statistic_mean": 0.0018,
      ...
    },
    "metrics_min": { ... },
    "metrics_max": { ... }
  }
}
```

### Directory Organization

```
results/5fold_metrics/
├── dsbm/
│   ├── results_DSBM_california_housing.json
│   ├── results_DSBM_diabetes.json
│   ├── results_DSBM_king_county_housing.json
│   └── ...
├── asbm/
│   ├── results_ASBM_california_housing.json
│   └── ...
├── dsbmxgboost/
│   ├── results_XGBOOST_DSBM_california_housing.json
│   └── ...
├── dsbm_vae/
│   ├── results_DSBM_VAE_california_housing.json
│   └── ...
└── tabsyn/
    ├── results_TABSYN_california_housing.json
    └── ...
```

**Usage**: Aggregation scripts discover all model directories automatically and load all JSON files.

---

## Aggregation Approach 1: Grouped by Dataset

**File**: `aggregate_by_dataset_group.py`

**Purpose**: For each dataset, create comparison plots showing all models side-by-side across metric groups (Correlation, Utility, Distribution, Privacy).

### Data Flow

```
results/5fold_metrics/
  ├── dsbm/results_DSBM_california_housing.json
  ├── asbm/results_ASBM_california_housing.json
  ├── dsbmxgboost/results_XGBOOST_DSBM_california_housing.json
  ├── dsbm_vae/results_DSBM_VAE_california_housing.json
  └── tabsyn/results_TABSYN_california_housing.json
        ↓
    Load all 5 JSONs for california_housing
        ↓
    Extract metrics by group (Correlation, Utility, Distribution, Privacy)
        ↓
    For each metric group:
      ├─ Create subplot grid (metrics × rows)
      ├─ Plot all models as grouped bars
      ├─ Detect outliers (IQR-based with flags)
      ├─ Clamp extreme values (preserve visualization)
      └─ Add reference lines (utility: 0% = baseline)
        ↓
    Output: california_housing_correlation.png
           california_housing_utility.png
           california_housing_distribution.png
           california_housing_privacy.png
```

### Algorithm: Per-Dataset Aggregation

```python
for dataset_name in DATASETS:
    
    # Load all models for this dataset
    for model_name in MODELS:
        load_json(f"results/5fold_metrics/{model_name}/results_{MODEL}_{dataset_name}.json")
        → Extract metrics_mean, metrics_std
    
    # Group metrics by category
    for group_name in ["Correlation", "Utility", "Distribution", "Privacy"]:
        metrics = METRIC_GROUPS[group_name]  # e.g., ["r2_score_synth", "rmse_synth", ...]
        
        # Create subplot grid
        n_rows = ceil(len(metrics) / 3)
        fig, axes = plt.subplots(n_rows, 3)
        
        # Plot each metric
        for metric in metrics:
            
            # Collect data from all models
            for model_name in MODELS:
                if metric in ["r2_score_synth", "rmse_synth"]:
                    # Calculate relative improvement
                    value = calculate_utility_improvement(model_data, metric)
                else:
                    # Use raw value
                    value = model_data[metric]['mean']
            
            # Detect outliers using IQR
            lower, upper = detect_outlier_bounds(all_values, is_distribution=True)
            
            # Clamp values (show original as yellow label)
            clamped_values = np.clip(all_values, lower, upper)
            
            # Plot grouped bars with error bars
            ax.bar(models, clamped_values, yerr=stds, color=colors, alpha=0.8)
            
            # Add outlier labels
            for outlier in detected_outliers:
                ax.text(x, y, f"{original_value:.2f}", 
                       bbox=yellow_box)  # Yellow highlight
        
        # Save subplot grid
        plt.savefig(f"figures/model_comparison_by_dataset/{dataset_name}_{group_name}.png")
```

### Outlier Detection & Visualization

**Problem**: Different metrics have vastly different scales (KS: 0.01-0.1, Wasserstein: 0.001-1.0)

**Solution**: Per-metric outlier detection with value clamping

```python
def detect_outlier_bounds(values, is_distribution=False):
    """IQR-based outlier detection with relative scaling."""
    
    q1 = percentile(values, 25)
    q3 = percentile(values, 75)
    median = percentile(values, 50)
    iqr = q3 - q1
    
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    # For distribution metrics: scale relative to median
    if is_distribution and median > 0:
        max_reasonable = median * 3.0
        upper = min(upper, max_reasonable)
    
    if is_distribution:
        lower = max(0.0, lower)  # No negative values
    
    return lower, upper

def clamp_value_with_flag(value, lower, upper):
    """Clamp to bounds and return outlier flag."""
    is_outlier = value < lower or value > upper
    clamped = np.clip(value, lower, upper)
    return clamped, is_outlier
```

**Visualization**:
- Clamped value displayed as bar
- Original value shown as yellow label if outlier
- Red dashed lines show outlier thresholds
- Prevents extreme values from crushing the scale

### Output Structure

```
figures/model_comparison_by_dataset/
├── california_housing_correlation.png       [3 metrics × 5 models]
├── california_housing_utility.png           [3 metrics × 5 models]
├── california_housing_distribution.png      [6 metrics × 5 models]
├── california_housing_privacy.png           [4 metrics × 5 models]
├── diabetes_correlation.png
├── diabetes_utility.png
├── diabetes_distribution.png
├── diabetes_privacy.png
└── ...                                       [8 datasets × 4 groups = 32 files]
```

**Figure Format**: 
- Size: 18 × 6*n_rows inches @ 300 DPI
- Layout: 3 columns, n_rows rows (up to 6 metrics per group)
- Legend: Model colors with labels
- Footnote: Explains outlier handling and reference lines

---

## Aggregation Approach 2: Grouped by Metric

**File**: `aggregate_all_models_comparison.py`

**Purpose**: For each metric, create comparison plot showing all models across all datasets (8 datasets × 5 models = 40 bars per metric).

### Data Flow

```
results/5fold_metrics/
  ├── dsbm/results_DSBM_*.json     (for each dataset)
  ├── asbm/results_ASBM_*.json
  ├── dsbmxgboost/...
  ├── dsbm_vae/...
  └── tabsyn/...
        ↓
    Load all models for all datasets
        ↓
    For each metric (e.g., "ks_statistic_mean"):
      
      ├─ Create 2D array: [n_datasets × n_models]
      ├─ Extract means and stds
      │
      ├─ Detect outliers (metric-specific IQR)
      ├─ Clamp values
      │
      ├─ Create grouped bar plot:
      │   X-axis: 8 datasets
      │   Groups: 5 models per dataset
      │   Color: Per-model consistent color
      │   Error bars: ± std across 5 folds
      │
      ├─ Add reference lines:
      │   - Utility metrics: y=0 (green dashed)
      │   - Others: y=0 (black solid)
      │   - Outlier bounds (gray dashed)
      │
      └─ Output: comparison_{metric_name}.png
        ↓
    Output: 15 per-metric comparison plots (utility + dist + corr + privacy)
```

### Algorithm: Per-Metric Aggregation

```python
for metric in METRICS_TO_PLOT:
    
    # Prepare 2D data matrix
    means_df = pd.DataFrame(index=DATASETS, columns=MODELS)
    stds_df = pd.DataFrame(index=DATASETS, columns=MODELS)
    
    for model_name in MODELS:
        for dataset_name in DATASETS:
            
            # Load JSON
            json_data = load(f"results/5fold_metrics/{model_name}/{dataset_name}.json")
            
            # Extract metric (handle utility specially)
            if metric in ["r2_score_synth", "rmse_synth"]:
                # Calculate improvement
                r2_real = json_data['metrics_mean']['r2_score_real']
                r2_synth = json_data['metrics_mean']['r2_score_synth']
                
                # Formula 1: R² improvement (% change from real)
                r2_improvement = (r2_synth - r2_real) / abs(r2_real) * 100
                
                # Formula 2: RMSE improvement (% reduction)
                rmse_real = json_data['metrics_mean']['rmse_real']
                rmse_synth = json_data['metrics_mean']['rmse_synth']
                rmse_improvement = (rmse_real - rmse_synth) / rmse_real * 100
                
                means_df.loc[dataset_name, model_name] = r2_improvement  # or rmse_improvement
            else:
                # Raw metric value
                means_df.loc[dataset_name, model_name] = json_data['metrics_mean'][metric]
            
            # Store std deviation
            stds_df.loc[dataset_name, model_name] = json_data['metrics_std'][metric]
    
    # Detect outliers (metric-wide across all cells)
    all_values = means_df.values.flatten()
    lower, upper = detect_outlier_bounds(all_values, is_distribution=(metric in DISTRIBUTION_METRICS))
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(16, 9))
    
    n_datasets = len(DATASETS)
    n_models = len(MODELS)
    bar_width = 0.15
    x = np.arange(n_datasets)
    
    for idx, model_name in enumerate(MODELS):
        offset = (idx - n_models/2 + 0.5) * bar_width
        x_pos = x + offset
        
        means = means_df[model_name].values
        stds = stds_df[model_name].values
        
        # Clamp values
        clamped_means = []
        for mean_val in means:
            clamped, is_outlier = clamp_value_with_flag(mean_val, lower, upper)
            clamped_means.append(clamped)
        
        # Plot bars
        ax.bar(x_pos, clamped_means, bar_width, 
               yerr=stds,
               label=MODELS[model_name]['label'],
               color=MODELS[model_name]['color'],
               alpha=0.8)
    
    # Add reference lines
    if metric in ["r2_score_synth", "rmse_synth"]:
        ax.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Baseline')
    
    # Add outlier bounds
    ax.axhline(y=lower, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=upper, color='gray', linestyle='--', alpha=0.3)
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=45, ha='right')
    ax.set_title(f"{metric.replace('_', ' ').title()} - Model Comparison")
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.2)
    
    # Save
    plt.savefig(f"figures/model_comparison_by_metric/comparison_{metric}.png", dpi=300)
```

### Output Structure

```
figures/model_comparison_by_metric/
├── comparison_r2_score_synth.png              [Utility]
├── comparison_rmse_synth.png                  [Utility]
├── comparison_ks_statistic_mean.png           [Distribution]
├── comparison_wasserstein_distance.png        [Distribution]
├── comparison_mmd.png                         [Distribution]
├── comparison_kl_divergence.png               [Distribution]
├── comparison_js_divergence.png               [Distribution]
├── comparison_swd.png                         [Distribution]
├── comparison_correlation_similarity.png      [Correlation]
├── comparison_pairwise_correlation_diff.png   [Correlation]
├── comparison_correlation_distance.png        [Correlation]
├── comparison_dcr_mean.png                    [Privacy]
├── comparison_dcr_median.png                  [Privacy]
├── comparison_dcr_share.png                   [Privacy]
└── comparison_authenticity.png                [Privacy]
                                               [15 total files]
```

**Figure Format**:
- Size: 16 × 9 inches @ 300 DPI (landscape)
- X-axis: 8 datasets (all datasets in one plot)
- Grouped bars: 5 models per dataset
- Legend: Model names with colors
- Error bars: ± 1 std across 5 folds

---

## Utility Metrics: Relative Improvement Calculation

### Why Convert to Improvement %?

**Problem**: Raw values are not comparable across datasets
```
Dataset A:  r2_real = 0.95, r2_synth = 0.92  (close)
Dataset B:  r2_real = 0.45, r2_synth = 0.44  (close, but lower baseline)
```

**Solution**: Calculate relative improvement
```
Dataset A: (0.92 - 0.95) / 0.95 × 100 = -3.2%   (3.2% worse)
Dataset B: (0.44 - 0.45) / 0.45 × 100 = -2.2%   (2.2% worse, but same relative gap)
```

### Formulas (CORRECT SIGNED VALUES)

**R² Improvement**:
```
improvement = (R²_synth - R²_real) / |R²_real| × 100%

Positive value: Synthetic model performs BETTER (higher R²)
Negative value: Synthetic model performs WORSE (lower R²)
Example:  +5% means R²_synth is 5% higher than R²_real
```

**RMSE Improvement**:
```
improvement = (RMSE_real - RMSE_synth) / RMSE_real × 100%

Positive value: Synthetic model performs BETTER (lower RMSE)
Negative value: Synthetic model performs WORSE (higher RMSE)
Example:  +10% means RMSE_synth is 10% lower than RMSE_real
```

### Implementation

```python
def calculate_relative_improvement(all_models, dataset_name):
    """Calculate utility metric improvements with correct sign."""
    
    improvements = {}
    
    for model_name in all_models.keys():
        metrics = all_models[model_name][dataset_name]['mean']
        
        # R² improvement
        r2_real = metrics.get('r2_score_real', np.nan)
        r2_synth = metrics.get('r2_score_synth', np.nan)
        
        if not np.isnan(r2_real) and r2_real != 0:
            # Positive: synth better, Negative: synth worse
            r2_improvement = (r2_synth - r2_real) / abs(r2_real) * 100
        else:
            r2_improvement = np.nan
        
        # RMSE improvement
        rmse_real = metrics.get('rmse_real', np.nan)
        rmse_synth = metrics.get('rmse_synth', np.nan)
        
        if not np.isnan(rmse_real) and rmse_real != 0:
            # Positive: synth better (lower error), Negative: synth worse
            rmse_improvement = (rmse_real - rmse_synth) / rmse_real * 100
        else:
            rmse_improvement = np.nan
        
        improvements[model_name] = {
            'r2_improvement': r2_improvement,
            'rmse_improvement': rmse_improvement
        }
    
    return improvements
```

---

## Metric Classification

### Metric Groups (Used in Dataset-Grouped Approach)

```python
METRIC_GROUPS = {
    "Correlation": [
        "correlation_similarity",      # 1 - ||Corr_real - Corr_synth||_F
        "pairwise_correlation_diff",   # Mean pairwise correlation difference
        "correlation_distance",         # Feature correlation distance
    ],
    "Utility": [
        "r2_score_synth",              # (R²_synth - R²_real) / R²_real × 100%
        "rmse_synth",                  # (RMSE_real - RMSE_synth) / RMSE_real × 100%
        "ml_efficiency_gap_percent",   # |R²_real - R²_synth| / |R²_real| × 100%
    ],
    "Distribution": [
        "ks_statistic_mean",           # Max|F_real - F_synth| per feature
        "wasserstein_distance",        # Mean 1D Wasserstein per feature
        "mmd",                         # Maximum Mean Discrepancy
        "kl_divergence",               # Kullback-Leibler divergence
        "js_divergence",               # Jensen-Shannon divergence
        "swd",                         # Sliced Wasserstein Distance
    ],
    "Privacy": [
        "dcr_mean",                    # Distance to Closest Record (mean)
        "dcr_median",                  # Distance to Closest Record (median)
        "dcr_share",                   # Share of perfect matches
        "authenticity",                # Authenticity (synthetic is authentic)
    ]
}
```

### Metrics to Plot (Metric-Grouped Approach)

```python
METRICS_TO_PLOT = [
    # Utility (2)
    "r2_score_synth",
    "rmse_synth",
    
    # Distribution (6)
    "ks_statistic_mean",
    "wasserstein_distance",
    "mmd",
    "kl_divergence",
    "js_divergence",
    "swd",
    
    # Correlation (3)
    "correlation_similarity",
    "pairwise_correlation_diff",
    "correlation_distance",
    
    # Privacy (4)
    "dcr_mean",
    "dcr_median",
    "dcr_share",
    "authenticity",
]

# Note: ml_efficiency_gap_percent EXCLUDED (redundant with r2_score_synth, rmse_synth)
```

---

## Quick Reference: Which Aggregation to Use?

| Question | Use Approach |
|----------|-------------|
| **"How do all models compare on dataset X?"** | Grouped by Dataset |
| **"Which metric is most variable across models?"** | Grouped by Metric |
| **"Is model A better on all datasets?"** | Grouped by Metric |
| **"How does dataset X affect metric performance?"** | Grouped by Dataset |
| **"Which model wins each dataset?"** | Grouped by Dataset |
| **"What's the overall ranking for metric Y?"** | Grouped by Metric |

---

## Reproducibility

Both scripts use:
- **Consistent model colors**: #FF6B6B (DSBM), #4ECDC4 (ASBM), etc.
- **Fixed seeds**: random_state=42 for all operations
- **Deterministic outlier detection**: IQR-based, reproducible
- **DPI=300**: High-resolution PNG output
- **Automatic discovery**: Both scripts auto-find all model directories

---

## Summary: Data Pipeline

```
5-Fold Evaluation Results (JSON per model per dataset)
    ↓
    ├─→ Aggregation Approach 1: By Dataset
    │    └─→ For each dataset: 4 multi-metric subplots
    │        (Correlation, Utility, Distribution, Privacy)
    │        └─→ 32 figures (8 datasets × 4 groups)
    │
    └─→ Aggregation Approach 2: By Metric
         └─→ For each metric: 1 full comparison plot
             (All 8 datasets, all 5 models)
             └─→ 15 figures (Utility + Distribution + Correlation + Privacy)

Total: 47 visualization files
Ready for: thesis, paper, presentation slides
```

---
