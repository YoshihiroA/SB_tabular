# Schrödinger Bridge Models for Tabular Data Synthesis

Comprehensive implementation and evaluation framework for three distinct Schrödinger Bridge approaches to synthetic tabular data generation: DSBM (neural network-based), XGBoost-DSBM (tree-based), and ASBM (adversarial GAN-based).

---

## Overview

This repository provides:

1. **Three Production-Ready Models**
   - DSBM: Continuous-time diffusion with neural drift networks
   - XGBoost-DSBM: Discrete-time with XGBoost tree models
   - ASBM: Discrete-time with adversarial GAN training

2. **Comprehensive Evaluation Framework**
   - 5-fold cross-validation with stratified splits
   - 20+ evaluation metrics (distribution, utility, privacy, correlation)
   - Automatic Optuna hyperparameter optimization
   - Camparison with real data baseline, TabDDPM, CTGAN, TabSyn
   - Aggregation with visualization
   

3. **Research Extensions**
   - VAE preprocessing for categorical data handling
   - Dimensionality reduction (compression dimensions)

---

## Repository Structure

```
repo/
├── models/
│   ├── DSBM/
│   │   └── dsbm_tabular_bridge.py          # Neural network DSBM
│   ├── XGBoostDSBM/
│   │   └── xgboostdsbm_tabular_bridge.py   # Tree-based DSBM
│   └── ASBM/
│       └── asbm_tabular_bridge.py          # Adversarial ASBM
│
├── experiments/
│   ├── optuna_dsbm.py                      # Hyperparameter tuning
│   ├── optuna_asbm.py
│   ├── optuna_xgboost_dsbm.py
│   ├── dsbm_eval_cv_5fold.py               # 5-fold evaluation
│   ├── dsbm_eval_cv_5fold_with_vae.py      # VAE extension
│   ├── evaluation_metrics.py               # 20+ metrics
│   └── vae_preprocessing_attention.py      # VAE for categorical
│
├── results/
│   ├── optuna_results_dsbm/SUMMARY.json
│   ├── optuna_results_asbm/SUMMARY.json
│   ├── optuna_results_xgboost_dsbm/SUMMARY.json
│   ├── 5fold_metrics/
│   │   ├── dsbm/
│   │   ├── asbm/
│   │   ├── dsbmxgboost/
│   │   ├── dsbm_vae/
│   │   └── tabsyn/
│   └── figures/
│       ├── model_comparison_by_dataset/    # 32 figures
│       └── model_comparison_by_metric/     # 15 figures
│
├── datasets/
│   └── datasets_numeric_merged.pkl         # Input datasets
│
└── docs/
    ├── models.md                         # API documentation
    ├── classes_methods.md                # Specifics of realizations
    ├── 5fold_evaluation.md               # Evaluation pipeline
    ├── optuna.md                         # Optuna opimization
    ├── metrics.md
    └── aggregation_visualization.md      # Visualization guide
```

---

## Quick Start

### 1. Model Training & Inference

```python
from models.DSBM.dsbm_tabular_bridge import DSBMTabularBridge

# Initialize with data
bridge = DSBMTabularBridge(
    x0_train=np.random.randn(n_train, d),    # Source: Gaussian noise
    x1_train=X_train_scaled,                  # Target: real data
    num_timesteps=200,
    learning_rate=1e-4,
    sigma=0.5
)

# Train using Iterative Markovian Fitting
history = bridge.fit(
    imf_iters=15,
    inner_iters=500,
    batch_size=256
)

# Generate synthetic data
X_synth = bridge.generate(n_samples=10000, direction='forward')
```

All three models (DSBM, XGBoost-DSBM, ASBM) implement the same interface:
- `fit()`: Train with Optuna best parameters
- `generate()`: Create synthetic samples
- `evaluate()`: Compute quality metrics

### 2. Hyperparameter Optimization

```bash
python experiments/optuna_dsbm.py  # Bayesian optimization on StandardScaled data
```

Optimizes MMD + SWD objective on StandardScaled data for comparable magnitudes across datasets.

Output: `results/optuna_results_dsbm/SUMMARY.json` with best hyperparameters per dataset.

### 3. 5-Fold Evaluation

```bash
python experiments/dsbm_eval_cv_5fold.py
```

Process:
1. Load Optuna best parameters
2. For each of 5 stratified folds:
   - Split data with stratification on target
   - Fit StandardScaler on training only
   - Train model on scaled data
   - Generate synthetic samples
   - Compute 20+ metrics
3. Aggregate across folds: mean ± std per metric
4. Output: `results_DSBM_{dataset}.json` with fold-level and aggregate statistics

### 4. Evaluation Metrics

**Distribution Metrics** (full data, no split):
- Kolmogorov-Smirnov: max|F_real(x) - F_synth(x)| per feature
- Wasserstein Distance: mean 1D Wasserstein per feature
- MMD & SWD: multi-dimensional distance measures
- KL/JS Divergence: information-theoretic distances

**Utility Metrics** (80/20 split on real only):
- Train XGBoost on real train → evaluate on real test (baseline R², RMSE)
- Train XGBoost on synthetic (all) → evaluate on real test (synthetic R², RMSE)
- Gap = relative degradation: |(R²_synth - R²_real) / R²_real| × 100%

**Correlation Metrics** (full data):
- Pairwise correlation similarity between real and synthetic
- Frobenius norm of correlation matrix difference

**Privacy Metrics** (full data):
- DCR (Distance to Closest Record): min distance to real data
- Authenticity: fraction where distance_to_real > distance_within_real
- Identical matches: fraction with distance < 1e-6

### 5. VAE Preprocessing (Categorical Data)

```bash
python experiments/dsbm_eval_cv_5fold_with_vae.py
```

Pipeline:
1. One-hot encode categorical features (if present)
2. Fit VAE on training data:
   - Learns dense latent representation (23d → 8d compression)
   - Multi-head attention with auto-determined heads
3. StandardScaler fitted on VAE-encoded training latent space
4. Train DSBM in compressed latent space (3-10x faster)
5. Generate synthetic → decode → evaluate in original space

Benefits: 13-15x dimensionality reduction, 3-10x training speedup, preserves feature correlations.

### 6. Aggregation & Visualization

```bash
# All models grouped by dataset
python experiments/aggregate_by_dataset_group.py
# Output: 32 figures (8 datasets × 4 metric groups)

# All models grouped by metric
python experiments/aggregate_all_models_comparison.py
# Output: 15 figures (per-metric comparison across datasets)
```

**Visualization Features**:
- Per-metric outlier detection (IQR-based)
- Value clamping with outlier labels (gray boxes)
- Relative scaling for distribution metrics (prevents scale crushing)
- Error bars: ± std across 5 folds

---

## Models: Design & Trade-offs

### DSBM (Continuous-Time, Neural)

**Architecture**: 
- Score networks learn drift from Gaussian to data distribution
- Continuous time discretization (200 steps)
- Multi-layer perceptron with time embedding

**Training**: Iterative Markovian Fitting (IMF)
- Alternate forward/backward direction learning
- Each IMF iteration: 2 reciprocal + 2 markovian projections using learned drift

**Pros**: Flexible, good for research, natural continuous dynamics
**Cons**: Slowest training, higher memory, high quality

### XGBoost-DSBM (Discrete-Time, Trees)

**Architecture**:
- One XGBoost model per feature dimension
- Discrete time bins (K=20 default)
- Per-dimension regression targets

**Training**: IMF but with tree-based drift approximation
- Explicit marginal projection: nearest-neighbor resampling
- Per-iteration coupling improvement

**Pros**: Fast training (3-5x), low memory, scalable, adaptable for catgorical features
**Cons**: Tree-specific hyperparameters, limited interpolation between bins

### ASBM (Discrete-Time, Adversarial)

**Architecture**:
- Forward/backward generator-discriminator pairs (4 networks)
- Brownian posterior sampling for training data generation
- GAN-based loss (softplus discriminator)

**Training**: IMF with adversarial learning
- Alternating generator/discriminator updates
- Exponential moving average for generator stability

**Pros**: moderate quality, best fidelity, well-studied
**Cons**: Slowest training, highest memory, complex tuning

---

## Preprocessing Pipeline

### StandardScaler Integration

Applied **per-fold** to prevent data leakage:
- Fitted on training data only → compute mean, std
- Applied identically to both train and test
- Ensures all features have comparable scales
- Enables fair metric comparison across features and datasets

### VAE Extension (Categorical Handling)

Handles mixed categorical + continuous data:
1. One-hot encode categorical features
2. Learn VAE latent representation (e.g., 100d → 8d)
3. Multi-head attention encoder with auto-determined heads
4. Apply StandardScaler on VAE latent space
5. Train DSBM in compressed space
6. Reverse: decode → inverse StandardScaler → original space

Benefits:
- Categorical embeddings learned instead of sparse one-hot
- Significant speedup (18x compression, 5-10x training speedup)
- Preserves feature correlations in latent space
- Prepares for future work on arbitrary mixed datasets

---

## Cross-Validation & Reproducibility

### Stratified K-Fold Strategy

Ensures target distribution preserved:
- Uses target column (last column) for stratification
- For continuous targets: discretize into percentile bins
- Prevents lucky/unlucky random splits
- Applies equally to classification and regression

### Per-Fold Processing

1. **Split**: Stratified train/test (80/20)
2. **Preprocess**: Fit StandardScaler on train only
3. **Train**: DSBM on scaled training data
4. **Generate**: Synthetic equal to training set size
5. **Evaluate**: Metrics comparing synthetic vs real test
6. **Aggregate**: Mean ± Std across 5 folds

### Reproducibility Guarantees

- All random seeds: `random_state=42`
- Stratification: Target-based with discretization
- StandardScaler: Per-fold fitting (no leakage)
- Optuna: TPESampler with seed=42+dataset_index
- Metrics: Deterministic on fixed data

---

## Hyperparameter Optimization

### Optuna Integration

**Optimization Objective**: MMD + SWD on StandardScaled data
- Ensures metric magnitudes comparable (both ~0.02-0.1 range)
- Dataset-agnostic hyperparameter search
- Prevents scale-dependent overfitting

**Search Space** (per model):
- `learning_rate`: 1e-5 to 1e-2 (log scale)
- `sigma`: 0.1 to 2.0 (diffusion noise)
- `num_timesteps`: 100-300 (discretization steps)
- Model-specific: `imf_iters`, `inner_iters`, `layers` for DSBM

**Trial Process**:
1. Fast training (1 IMF iteration, 30 inner iterations)
2. Generate synthetic data
3. Compute MMD + SWD on StandardScaled space
4. Return combined loss

**Output Format**:
- Per-dataset: Best parameters and best loss value
- SUMMARY.json: Aggregated across all datasets
- Used in 5-fold evaluation with full training

---

## Evaluation Results Structure

### Input: Per-Model JSON

```json
{
  "metadata": {
    "dataset": "california_housing",
    "model": "DSBM",
    "n_folds": 5,
    "n_metrics": 20
  },
  "folds": [
    { "fold": 1, "ks_statistic_mean": 0.0456, ... },
    { "fold": 2, "ks_statistic_mean": 0.0412, ... },
    ...
  ],
  "summary": {
    "metrics_mean": { ... },
    "metrics_std": { ... },
    "metrics_min": { ... },
    "metrics_max": { ... }
  }
}
```

### Aggregation Approach 1: By Dataset

**For each dataset**: Create 4 subplots (Correlation, Utility, Distribution, Privacy)
- X-axis: Models (DSBM, ASBM, XGBoost-DSBM, DSBM-VAE, TabSyn, CTGAN, TabDDPM)
- Y-axis: Metric values
- Grouped bars with error bars (± std across folds)
- Outliers: Yellow labels show clamped values
- Reference lines: Utility metrics show y=0 baseline

Output: 32 figures (8 datasets × 4 metric groups)

### Aggregation Approach 2: By Metric

**For each metric**: Create 1 comparison plot across all datasets
- X-axis: Datasets (8 total)
- Grouped bars: 5 models per dataset
- Y-axis: Metric values
- Per-metric outlier detection with clamping
- Reference lines consistent with dataset-grouped approach

Output: 15 figures (Utility + Distribution + Correlation + Privacy)

### Outlier Handling

**Problem**: Different metrics have vastly different scales (KS: 0.01-0.1, Wasserstein: 0.001-1.0)

**Solution**: Per-metric IQR-based detection with value clamping
1. Calculate Q1, Q3, IQR for all metric values
2. Define bounds: [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
3. Clamp values to bounds for visualization
4. Display original value as yellow label if outside bounds
5. Prevents extreme outliers from crushing scale

---

## Utility Metrics: Signed Improvement

### R² Improvement

```
(R²_synth - R²_real) / |R²_real| × 100%

Positive: Synthetic performs BETTER (higher R²)
Negative: Synthetic performs WORSE (lower R²)
```

### RMSE Improvement

```
(RMSE_real - RMSE_synth) / RMSE_real × 100%

Positive: Synthetic performs BETTER (lower RMSE)
Negative: Synthetic performs WORSE (higher RMSE)
```

These signed values enable direct comparison: baseline (0%) vs improvement/degradation.

---

## Key Design Decisions

### Why StandardScale Everything?

1. **Fair Comparison**: All features contribute equally to metrics
2. **Stable Training**: N(0,1) initialization matches scaled data
3. **Cross-Dataset**: Enables hyperparameter reuse (Optuna on StandardScaled)
4. **Metric Interpretation**: Distribution distances comparable across features

### Why Per-Fold Fitting?

1. **Realistic**: Each fold gets independent preprocessing (true blind test)
2. **No Leakage**: Test statistics never affect training
3. **Variance**: Report std across folds (uncertainty estimate)
4. **Standard Practice**: Matches sklearn cross-validation conventions

---

## Research Contributions

### Implemented Models

1. **DSBM**: Neural score-based continuous-time bridge (Shi et al., NeurIPS 2023)
2. **XGBoost-DSBM**: Tree-based discrete-time variant with explicit marginal projection
3. **ASBM**: Adversarial GAN-based discrete-time synthesis (Gushchin et al., NeurIPS 2024)

### Evaluation Framework

1. **20+ Metrics**: Distribution, utility, correlation, privacy in unified evaluator
2. **5-Fold CV**: Stratified cross-validation with per-fold StandardScaler
3. **Optuna Integration**: Bayesian hyperparameter optimization on StandardScaled data
   
### Research Extensions

1. **VAE Preprocessing**: Handles categorical data, enables compression
3. **Attention Mechanism**: Auto-determined multi-head attention for tabular data
4. **Reproducible Evaluation**: Deterministic pipeline with fixed seeds throughout

---

## File Reference

### Model Classes

- `dsbm_tabular_bridge.py`: DSBMTabularBridge class
  - Methods: `fit()`, `generate()`, `evaluate()`, `save()`, `load()`
  - Components: MLP, ScoreNetwork, DSBM (low-level)

- `xgboostdsbm_tabular_bridge.py`: XGBoostDriftNetwork, XGBoostDSBMTabularBridge
  - Methods: `fit()`, `generate()`, `train()`, `sample()`
  - Static method: `get_training_data()` for pair generation

- `asbm_tabular_bridge.py`: ASBMTabularBridge
  - Methods: `fit()`, `generate()`, `evaluate()`, `save()`, `load()`, `_plot_gan_losses()`
  - Components: MyGenerator, MyDiscriminator, BrownianPosterior_Coefficients

### Evaluation

- `evaluation_metrics.py`: MetricsEvaluator
  - 20+ metrics computed in single call: `compute_all_metrics(X_real, X_synth)`
  - Includes: KS, Wasserstein, MMD, SWD, correlation, DCR, authenticity, etc.

- `dsbm_eval_cv_5fold.py`: 5-fold evaluation pipeline
  - Stratified splits, per-fold StandardScaler, aggregation with mean/std

- `dsbm_eval_cv_5fold_with_vae.py`: VAE preprocessing extension
  - DSBMVAEPipeline class: `fit()`, `transform()`, `inverse_transform()`
  - Auto-determined attention heads for categorical handling

### Optimization & Aggregation

- `optuna_dsbm.py`, `optuna_asbm.py`, `optuna_xgboost_dsbm.py`: Hyperparameter tuning
  - Objective: MMD + SWD on StandardScaled data
  - Output: best_params per dataset in SUMMARY.json

- `aggregate_by_dataset_group.py`: Dataset-grouped visualization
  - Output: 4 subplots per dataset × 8 datasets = 32 figures

- `aggregate_all_models_comparison.py`: Metric-grouped visualization
  - Output: 1 comparison per metric × 15 metrics = 15 figures
  - Both with outlier detection and relative scaling

---

## Citation Guide

**When Referencing Components**:

- DSBM model: Cite Shi et al. (NeurIPS 2023)
- ASBM model: Cite Gushchin et al. (NeurIPS 2024)

**Key Metrics**:

- Distribution: KS, Wasserstein, MMD standard definitions
- Utility: XGBoost-based predictive modeling gap
- Privacy: DCR and authenticity from synthetic data evaluation literature
- Correlation: Frobenius norm of correlation matrix difference

---

## Future Work

1. **Categorical Handling**: Full native categorical support without one-hot encoding
2. **Larger Datasets**: Extend to million-row tabular data
3. **Mixed Types**: Direct handling of categorical, continuous, and ordinal features

---

## License

Research project from ITMO University. See LICENSE file for details.

---
