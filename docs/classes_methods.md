# Model Classes and Methods Reference

## 1. DSBMTabularBridge

**File**: `dsbm_tabular_bridge.py`

**Purpose**: Continuous-time Schrödinger Bridge with neural network drift learning (DSBM from Shi et al., NeurIPS 2023)

### Class: `DSBMTabularBridge`

**Initialization**:
```python
__init__(
    x0_train: np.ndarray,           # [n_train, d] source distribution
    x1_train: np.ndarray,           # [n_train, d] target distribution  
    x0_test: Optional[np.ndarray],  # [n_test, d] optional test source
    x1_test: Optional[np.ndarray],  # [n_test, d] optional test target
    num_timesteps: int = 200,       # continuous time discretization steps
    epsilon: float = 0.664,         # time boundary clipping
    sigma: float = 0.5,             # diffusion noise level (tunable)
    learning_rate: float = 1e-4,    # training learning rate (tunable)
    device: Optional[torch.device],
    **kwargs
)
```

**Attributes**:
- `x0_train, x1_train`: Training data stored as float32 numpy
- `x0_test, x1_test`: Optional test data
- `D`: Data dimension (inferred from input)
- `model`: Trained DSBM instance (initialized during fit)
- `model_list`: List of checkpointed models per IMF iteration
- `is_trained`: Boolean flag

**Methods**:

#### `fit(imf_iters, inner_iters, batch_size, learning_rate, layers, verbose) → Dict`

Train DSBM using Iterative Markovian Fitting (IMF).

**Args**:
- `imf_iters`: Number of outer IMF iterations
- `inner_iters`: Training iterations per IMF iteration
- `batch_size`: Batch size for gradient descent
- `learning_rate`: Optional override of self.learning_rate
- `layers`: MLP hidden layer widths list
- `verbose`: Print training progress

**IMF Loop**:
1. First iteration: Train backward network on random coupling
2. Subsequent: Alternately train forward/backward networks
3. After each training: Save checkpointed model for next reciprocal projection

**Returns**: `Dict` with training history (loss curves per direction)

#### `generate(n_samples, direction) → np.ndarray`

Generate synthetic samples via trained SDE.

**Args**:
- `n_samples`: Number of samples to generate
- `direction`: 'forward' (x0→x1) or 'backward' (x1→x0)

**Process**:
1. Initialize z from x0_train (forward) or x1_train (backward)
2. Apply latest trained model's SDE for N steps
3. Use Euler-Maruyama integration

**Returns**: `np.ndarray` [n_samples, d] in float32

#### `traindata` property
Returns x0_train (compatibility)

#### `testdata` property
Returns x1_train (compatibility)

---

## 2. XGBoostDriftNetwork + XGBoostDSBMTabularBridge

**File**: `xgboostdsbm_tabular_bridge.py`

**Purpose**: Discrete-time Schrödinger Bridge with XGBoost tree-based drift learning

### Class: `XGBoostDriftNetwork`

**Initialization**:
```python
__init__(
    input_dim: int,                 # data dimension d
    n_timesteps: int = 15,          # number of discrete time bins K
    xgb_params: Optional[Dict]      # XGBoost hyperparameters dict
)
```

**Attributes**:
- `input_dim, output_dim`: Data dimension
- `n_timesteps`: Discrete time bins
- `xgb_models`: Dict of per-dimension XGBoost regressors {f"dim_0": model, ...}
- `feature_importance`: Dict of feature importances per dimension

**Methods**:

#### `get_time_bins(eps) → np.ndarray`
Generate K discrete uniform time points in [eps, 1-eps].

#### `get_time_bin_index(t) → np.ndarray`
Map continuous time t ∈ [0,1] to nearest bin index ∈ [0,K-1].

#### `predict(x_input, t) → np.ndarray`
Predict drift at positions x and times t.

**Args**:
- `x_input`: [batch_size, d] positions
- `t`: [batch_size] or [batch_size, 1] times in [0,1]

**Process**:
1. Normalize time to [0, K-1]
2. Create features [x, t_normalized]
3. For each dimension: call xgb_model.predict()
4. Stack outputs [batch_size, d]

#### `get_training_data(z_pairs, n_timesteps, sig, eps, fb) → Tuple[X, y]`

Generate training data across all discrete time bins.

**Static method**

**Args**:
- `z_pairs`: [N, 2, d] paired samples [z0, z1]
- `n_timesteps`: Number of time bins K
- `sig`: Diffusion noise σ
- `eps`: Time clipping margin
- `fb`: 'f' for forward or 'b' for backward

**Process** (per time point t):
1. Interpolate: z_t = (1-t)·z0 + t·z1 + σ·√(t(1-t))·ε
2. Compute drift target: (z1-z0) - σ·√(t/(1-t))·ε (forward)
3. Stack features [z_t, t]

**Returns**: Tuple (X [N·K, d+1], y [N·K, d])

#### `train(x_pairs, n_timesteps, sig, eps, fb, verbose) → self`

Train XGBoost models for one direction.

**Process**:
1. Call `get_training_data` to generate regression pairs
2. For each dimension d: fit XGBRegressor on (X, y[:, d])
3. Store model in `xgb_models[f"dim_{d}"]`
4. Compute and cache feature importances

#### `sample(zstart, n_euler_steps, sig, fb) → List[np.ndarray]`

Sample from learned SDE using Euler-Maruyama.

**Args**:
- `zstart`: [batch_size, d] starting positions
- `n_euler_steps`: Number of Euler steps
- `sig`: Diffusion noise σ
- `fb`: 'f' (forward: t 0→1) or 'b' (backward: t 1→0)

**Returns**: List of trajectory states, each [batch_size, d]

---

### Class: `XGBoostDSBMTabularBridge`

**Initialization**:
```python
__init__(
    x0_train: np.ndarray,           # [m_train, n] source distribution
    x1_train: np.ndarray,           # [m_train, n] target distribution
    x0_test: Optional[np.ndarray],  # [m_test, n] optional
    x1_test: Optional[np.ndarray],  # [m_test, n] optional
    n_timesteps: int = 15,          # discrete time bins K
    sig: float = 0.05,              # diffusion noise σ
    eps: float = 0.05,              # time clipping margin
    xgb_params: Optional[Dict],     # XGBoost hyperparameters
    **kwargs
)
```

**Attributes**:
- `x0_train, x1_train`: Training data stored as float32
- `x0_test, x1_test`: Optional test data
- `m_train, n_features`: Dimensions
- `drift_networks`: Dict {'f': XGBoostDriftNetwork, 'b': XGBoostDriftNetwork}
- `is_trained`: Boolean flag
- `history`: Training history dict

**Methods**:

#### `fit(n_iterations, imf_batch_size, verbose) → Dict`

Train using Iterative Marginal Fitting (IMF) loop.

**IMF Algorithm**:
```
Fixed: z0_base = x0_train, z1_base = x1_train
for iteration 0 to n_iterations-1:
    fb = 'b' if even else 'f'
    
    if iteration == 0:
        coupling = random permutation [z0_base, z1_base[perm]]
    else if fb available:
        coupling = resample via learned bridge
    
    Train XGBoostDriftNetwork(fb) on coupling
    Store in drift_networks[fb]
```

**Args**:
- `n_iterations`: Number of IMF outer iterations
- `imf_batch_size`: Batch size (default: full data)
- `verbose`: Print progress

**Returns**: `Dict` with iteration history

#### `_resample_coupling(z0_base, z1_base, fb, n_euler_steps, verbose) → np.ndarray`

Compute better coupling via learned bridge.

**Strategy**:
1. Sample trajectories: `net.sample(z0_base if fb=='f' else z1_base, ...)`
2. Get endpoints zT
3. Compute pairwise distances to target distribution
4. Find nearest-neighbor optimal matching
5. Return new paired samples

#### `generate(n_samples, direction, n_euler_steps) → np.ndarray`

Generate synthetic samples.

**Args**:
- `n_samples`: Number of samples
- `direction`: 'forward' or 'backward'
- `n_euler_steps`: Euler steps for SDE

**Returns**: [n_samples, n_features] float32 numpy array

#### `traindata` property
Returns x0_train

#### `testdata` property
Returns x1_train

#### `D` property
Returns n_features

---

## 3. ASBMTabularBridge

**File**: `asbm_tabular_bridge.py`

**Purpose**: Discrete-time Schrödinger Bridge with adversarial GAN-based learning (ASBM from Gushchin et al., NeurIPS 2024)

### Helper Classes

**`MyGenerator`**: GAN generator network
- Input: [x_t, t_embedding, z_latent]
- Output: x_{t+1} prediction

**`MyDiscriminator`**: GAN discriminator network  
- Input: [x_t, t_embedding, x_{t+1}]
- Output: real/fake score

**`BrownianPosterior_Coefficients`**: Posterior distribution coefficients
- Precomputes mean/variance coefficients for posterior sampling

---

### Class: `ASBMTabularBridge`

**Initialization**:
```python
__init__(
    x0_train: np.ndarray,                      # [n_train, d] source
    x1_train: np.ndarray,                      # [n_train, d] target
    x0_test: np.ndarray,                       # [n_test, d] test source
    x1_test: np.ndarray,                       # [n_test, d] test target
    global_scaler: Optional[StandardScaler],   # normalization
    categorical_columns: Optional[List[int]],  # feature grouping
    continuous_columns: Optional[List[int]],   # feature grouping
    num_timesteps: int = 4,                    # discrete time steps N
    beta_min: float = 0.1,                     # diffusion schedule
    beta_max: float = 20.0,                    # diffusion schedule
    epsilon: float = 1.0,                      # posterior variance
    device: Optional[torch.device],
    **kwargs
)
```

**Attributes**:
- `x0_train, x1_train, x0_test, x1_test`: Data stored as float32 numpy
- `D`: Data dimension
- `netG_fw, netG_bw`: Forward/backward generators
- `netD_fw, netD_bw`: Forward/backward discriminators
- `pos_coeff`: BrownianPosterior_Coefficients instance
- `config_obj`: dotdict for compatibility
- `is_trained`: Boolean flag

**Methods**:

#### `fit(imf_iters, inner_iters, batch_size, lr_g, lr_d, use_ema, ema_decay, verbose, save_dir) → Dict`

Train ASBM using Iterative Markovian Fitting with adversarial learning.

**IMF Loop with GAN**:
```
for iteration 0 to imf_iters-1:
    # Backward direction
    for inner_iter 0 to inner_iters-1:
        Sample (x_t^real, x_{t+1}) from posterior (Eq. 16)
        x_t^fake = G_bw(x_{t+1}, t, z)
        Update D_bw: L_D = softplus(-D(...real)) + softplus(D(...fake))
        Update G_bw: L_G = softplus(-D(...fake))
    
    # Forward direction (same with G_fw, D_fw)
```

**Args**:
- `imf_iters`: IMF outer iterations
- `inner_iters`: GAN training iterations per IMF
- `batch_size`: Mini-batch size
- `lr_g, lr_d`: Generator/Discriminator learning rates (tunable)
- `use_ema`: Exponential moving average for generators
- `ema_decay`: EMA decay rate
- `verbose`: Print progress
- `save_dir`: Directory for loss plots

**Training Data**: Generated via `q_sample_supervised_pairs_brownian()` using posterior (Eq. 16)

**Returns**: `Dict` with G_loss and D_loss curves

#### `generate(n_samples, direction) → np.ndarray`

Generate synthetic samples via posterior sampling.

**Args**:
- `n_samples`: Number of samples (default: x0_test size)
- `direction`: 'forward' (x0_test→synthetic x1) or 'backward'

**Process**:
1. Initialize from source (x0_train for forward, x1_train for backward)
2. For each time step t ∈ {N-1, N-2, ..., 0}:
   - Generate x_0 prediction: `netG(x_t, t, z_latent)`
   - Sample from posterior: `sample_posterior(...)`
3. Return final trajectory endpoint

**Returns**: [n_samples, d] float32 numpy array

#### `evaluate(metrics, nproj, sigma_mmd, max_eval_samples) → Dict`

Evaluate quality of generated samples.

**Basic stats** computed:
- Mean/std differences between real and synthetic

#### `save(path)` / `load(path)`

Save/load trained generator and discriminator state dicts.

#### `_plot_gan_losses(loss_history_fw, loss_history_bw, imf_iters, inner_iters, smoothing_factor, save_path)`

Plot smoothed GAN losses with IMF iteration markers (vertical lines).

---

## Unified Interface Summary

All three models implement the same public interface:

```python
model = ModelTabularBridge(x0_train, x1_train, x0_test, x1_test, **params)
history = model.fit(imf_iters, inner_iters, batch_size, **training_args)
synthetic = model.generate(n_samples, direction='forward')
```

**Drop-in compatibility**: Change only the import statement to switch models.

---

## Key Internal Classes

### DSBM Internals

**`MLP`**: Multi-layer perceptron
- Configurable layer widths and activation

**`ScoreNetwork`**: Time-conditioned MLP
- Concatenates [x, t] and passes through MLP
- Input: [batch_size, d+1]
- Output: [batch_size, d]

**`DSBM` (low-level)**: Core diffusion model
- Methods: `get_train_tuple()`, `generate_new_dataset()`, `sample_sde()`

**`train_dsbm()`: Standalone training function**
- Trains backward or forward network
- Manages data generation and optimization

---

### XGBoost Internals

**`XGBoostDriftNetwork`**: Encapsulates per-dimension XGBoost models

**Static method `get_training_data()`**: Generates Brownian bridge pairs

---

### ASBM Internals

**`q_sample_supervised_pairs_brownian()`**: Generates training pairs from posterior

**`sample_posterior()`**: Samples from posterior using learned coefficient

**`sample_from_model()`**: Full trajectory sampling via reversed posterior sampling

---
