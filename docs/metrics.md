# Evaluation Metrics for Synthetic Tabular Data

## Input Format

```python
X_real:  [n_samples, n_features + 1]  # Last column = target
X_synth: [m_samples, n_features + 1]  # Last column = target
```

---

## Metrics

### 1. Distribution Similarity

**Kolmogorov-Smirnov (KS)**:
```
D_KS = max_x |F_real(x) - F_synth(x)|
```

**Wasserstein Distance**:
```
W_1 = ∫ |F_real^{-1}(u) - F_synth^{-1}(u)| du
```

**KL Divergence**:
```
D_KL(P || Q) = ∫ p(x) log(p(x)/q(x)) dx
```

**JS Divergence**:
```
D_JS(P || Q) = 0.5·D_KL(P||M) + 0.5·D_KL(Q||M), M=(P+Q)/2
```

**MMD (Maximum Mean Discrepancy)**:
```
MMD²(P,Q) = E[k(x,x')] + E[k(y,y')] - 2·E[k(x,y)]
k(x,y) = exp(-||x-y||²/(2σ²))
```

**SWD (Sliced Wasserstein)**:
```
SWD(P,Q) = ∫ W_1(proj_θ(P), proj_θ(Q)) dθ
```

---

### 2. Correlation Preservation

**Correlation Similarity**:
```
S = 1 - ||Corr_real - Corr_synth||_F / ||max||_F
```

**Pairwise Correlation Difference**:
```
PCD = ||Corr_real - Corr_synth||_F
```

---

### 3. Fidelity & Diversity

**Alpha-Precision**:
```
IP_α = ∫ |P(X_synth ∈ S_α(X_real)) - α| dα
```

**Beta-Recall**:
```
IR_β = ∫ |P(X_real ∈ S_β(X_synth)) - β| dβ
```

---

### 4. Utility

**R² Score**:
```
R² = 1 - SS_res/SS_tot
```

**RMSE**:
```
RMSE = √((1/n)Σ(y_true - y_pred)²)
```

**ML Efficiency Gap**:
```
Gap = |R²_real - R²_synth| / |R²_real| × 100%
```

Training: XGBoost on real train (80%) + synthetic, test on real test (20%)

---

### 5. Privacy

**DCR (Distance to Closest Record)**:
```
DCR(x_synth) = min_{x_real} ||x_synth - x_real||_2
```

**DCR Share**:
```
Fraction of X_synth closer to X_real_part1 than X_real_part2
```

**Identical Matches**:
```
Fraction with min(||x_synth - x_real||) < tol
```

**Authenticity**:
```
Fraction where d(x_synth, closest_real) > d(closest_real, its_neighbor)
```

---

### 6. Detection

**C2ST (Classifier Two-Sample Test)**:
```
Train classifier: X_real (label=0) vs X_synth (label=1)
Report: accuracy, AUC, p-value
```

---

## Usage

```python
from evaluation_metrics import MetricsEvaluator

evaluator = MetricsEvaluator(
    include_alpha_precision=False,
    include_c2st=False,
    include_authenticity=True
)

metrics = evaluator.compute_all_metrics(X_real, X_synth)
```

**Available Metrics**: 26 total
- Distribution: 6
- Correlation: 3
- Fidelity/Diversity: 2
- Utility: 5
- Privacy: 5
- Detection: 3

---
