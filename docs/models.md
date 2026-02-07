# Schrödinger Bridge Models for Tabular Data Synthesis

## Overview

Three Schrödinger Bridge implementations for synthetic tabular data generation, sharing common mathematical foundation (Iterative Markovian Fitting) but differing in time discretization, network architecture, and training methodology.

---

## Common Foundation

### Schrödinger Bridge Problem

Find optimal path measure P^SB minimizing KL divergence from reference process while satisfying endpoint constraints:

```
P^SB = argmin_P { KL(P || Q) : P_0 = π_0, P_T = π_T }
```

where Q is reference Brownian motion, π_0, π_T are source/target marginals.

---

### Unified Interface

All models implement `<MODEL>TabularBridge` with identical API:

```python
model = <MODEL>TabularBridge(x0_train, x1_train, x0_test, x1_test, **hyperparams)
history = model.fit(imf_iters, inner_iters, batch_size, **training_params)
synthetic = model.generate(n_samples, direction='forward')
```

---

### General IMF Algorithm (All Models)

```
Algorithm: Iterative Markovian Fitting (IMF)

Input: Joint distribution Π⁰_{0,T}, reference process Q, iterations N

1:  Initialize Π⁰ = Π⁰_{0,T} Q_{|0,T}
2:  for n = 0 to N-1 do
3:      # BACKWARD MARKOVIAN PROJECTION
4:      Train backward network v_φ using Π_{2n}
5:      Compute backward Markovian kernel M_{2n+1}
6:      # RECIPROCAL PROJECTION
7:      Set Π_{2n+1} = M^{2n+1}_{0,T} Q_{|0,T}
8:      
9:      # FORWARD MARKOVIAN PROJECTION
10:     Train forward network v_θ using Π_{2n+1}
11:     Compute forward Markovian kernel M_{2n+2}
12:     # RECIPROCAL PROJECTION
13:     Set Π_{2n+2} = M^{2n+2}_{0,T} Q_{|0,T}
14: end for

Output: Trained networks v_θ (forward), v_φ (backward)
```

**Key Difference Across Models**: How drift networks v_θ, v_φ are defined and trained (see below).

---

## Model-Specific Formulations

### Model 1: Standard DSBM

**Reference**: Shi et al., "Diffusion Schrödinger Bridge Matching", NeurIPS 2023

**Forward SDE** (Equation 1):
```
dX_t = {f_t(X_t) + v_θ*(t, X_t)} dt + σ_t dB_t,    X_0 ~ π_0
```

**Forward Loss** (Equation 2):
```
θ* = argmin_θ ∫_0^T E_{Π_{t,T}}[ ||σ_t² ∇log Q_{T|t}(X_T|X_t) - v_θ(t, X_t)||² / σ_t² ] dt
```

**Backward SDE** (Equation 3):
```
dY_t = {-f_{T-t}(Y_t) + v_φ*(T-t, Y_t)} dt + σ_{T-t} dB_t,    Y_0 ~ π_T
```

**Backward Loss** (Equation 4):
```
φ* = argmin_φ ∫_0^T E_{Π_{0,t}}[ ||σ_t² ∇log Q_{t|0}(X_t|X_0) - v_φ(t, X_t)||² / σ_t² ] dt
```

**Drift Network Training**: 
- Continuous time t ∈ [ε, 1-ε]
- Regression on Brownian bridge targets
- Time-conditioned MLPs

**Inference**: Euler-Maruyama SDE integration

**Marginal Preservation**: Implicit (alternating anchoring)

---

### Model 2: XGBoost-DSBM

**Reference**: Extension of DSBM with gradient boosting

**Formulation**: Same SDEs and losses as DSBM (Equations 1-4), but with discrete time binning:

```
T = {t_1, t_2, ..., t_K}
```

**Drift Network Training**:
- Discrete K time bins
- Per-dimension XGBoost regressors: v_θ^{(i,k)}(x) ≈ v_θ*(t_k, x)_i
- Gradient boosting on regression targets

**Inference**: Euler-Maruyama with tree-based drift

**Marginal Preservation**: **Explicit** nearest-neighbor projection

```
After sampling: z_endpoints = SDE_sample(z_start)
Project onto real: perm_idx = argmin(distance(z_endpoints, z1_base))
Pair with real: (z0_base, z1_base[perm_idx])
```

---

### Model 3: ASBM

**Reference**: Gushchin et al., "Adversarial Schrödinger Bridge Matching", NeurIPS 2024

**Discrete Time**: N steps (typically N=4)

**Brownian Bridge Interpolation** (Equation 15):
```
X_t | X_0 = x_0, X_1 = x_1 ~ N(μ_t, Σ_t)

where:
  μ_t = (1-t)·x_0 + t·x_1
  Σ_t = ε·t·(1-t)·I
```

**Posterior Distribution** (Equation 16):
```
X_t | X_0 = x_0, X_{t+1} = x_{t+1} ~ N(μ_post, Σ_post)

where:
  μ_post = ((1-t)/(1-(t+Δt)))·x_0 + (t/(1-(t+Δt)))·x_{t+1}
  Σ_post = ε·t·(1-t-Δt)/(1-(t+Δt))·I
```

**Drift Network Training**:
- Discrete N steps
- Adversarial GAN: 4 networks (G_fw, G_bw, D_fw, D_bw)
- Generator loss: L_G = E[softplus(-D(G(x_{t+1}, t, z), t, x_{t+1}))]
- Discriminator loss: L_D = E[softplus(-D(x_t^real, t, x_{t+1}))] + E[softplus(D(x_t^fake, t, x_{t+1}))]
- Training pairs from posterior sampling (Eq. 16)

**Inference**: Sequential posterior sampling (N steps)

**Marginal Preservation**: Implicit (alternating anchoring)

