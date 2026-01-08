# Hyperparameter Optimization with Optuna

## Overview

Optuna-based hyperparameter optimization for all Schrödinger Bridge models. All scripts follow identical pipeline structure.

---

## Workflow

### 1. Search Space Definition

Define hyperparameters using Optuna's suggestion methods:
- Continuous: `trial.suggest_float(name, low, high, log=True/False)`
- Integer: `trial.suggest_int(name, low, high, step=...)`
- Categorical: `trial.suggest_categorical(name, choices=[...])`

### 2. Data Pipeline

```python
# Load merged pickle
with open('datasets_numeric_merged.pkl', 'rb') as f:
    raw_data = pickle.load(f)

# For each dataset
X = raw_data[dataset_name]
X_train, X_test = train_test_split(X, ...)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train).astype(np.float32)

# Create inputs
x0_train = randn(n_train, n_features)  # Noise
x1_train = X_train_scaled              # Real data
```

### 3. Optimization

```python
sampler = TPESampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=20)
best_params = study.best_trial.params
```

**Objective Function**:
```python
def objective(trial, X_train, ...):
    params = suggest_hyperparameters(trial)
    model = ModelTabularBridge(x0_train, x1_train, **params)
    model.fit(...)
    X_synth = model.generate(n_samples)
    metric = compute_metric(X_train, X_synth)
    return metric  # Lower is better
```

### 4. Results Storage

```
optuna_results_{model}/
├── dataset1_best_params.json
├── dataset2_best_params.json
├── ...
└── SUMMARY.json
```

**JSON Format**:
```json
{
  "dataset": "name",
  "n_trials": 20,
  "n_samples": 10000,
  "n_features": 8,
  "best_trial": 14,
  "best_value": 0.023,
  "best_params": { ... }
}
```

---

## Evaluation Metrics

**Objective**: Minimize `MMD + SWD`

**MMD (Maximum Mean Discrepancy)**:
```
MMD²(P, Q) = E[k(x,x')] + E[k(y,y')] - 2·E[k(x,y)]
k(x,y) = exp(-||x-y||² / (2σ²))
```

**SWD (Sliced Wasserstein Distance)**:
```
SWD(P, Q) = ∫ W_1(proj_θ(P), proj_θ(Q)) dθ
```

---

## Usage

```bash
# Tune all datasets
python optuna_{model}.py

# Specific datasets
python optuna_{model}.py --datasets wine diabetes --n-trials 50

# Custom configuration
python optuna_{model}.py --pkl-file custom.pkl --results-dir my_results
```

**Load Results**:
```python
import json
with open('optuna_results_{model}/dataset_best_params.json', 'r') as f:
    params = json.load(f)['best_params']

model = ModelTabularBridge(x0_train, x1_train, **params)
model.fit(**params)
```

---

## Configuration

**Sampler**: TPESampler (Tree-structured Parzen Estimator)
- Bayesian optimization
- Balances exploration/exploitation

**Study Direction**: Minimize (lower metric = better quality)

---
