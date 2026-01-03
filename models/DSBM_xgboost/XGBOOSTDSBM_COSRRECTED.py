import numpy as np
import xgboost as xgb


class XGBoostBridgeNet:
    """
    XGBoost-based score network for DSBM drift learning.

    Trains separate XGBoost models for each output dimension.
    Enables discrete time interval learning with tree-based models.
    """

    def __init__(self, input_dim, n_timesteps=10, xgb_params=None):
        """
        Initialize XGBoost bridge network.

        Args:
            input_dim (int): Data dimension d
            n_timesteps (int): Number of discrete time bins K (default: 10)
            xgb_params (dict): XGBoost hyperparameters.
        """
        self.input_dim = input_dim
        self.n_timesteps = n_timesteps
        self.output_dim = input_dim

        # One model per output dimension
        self.xgb_models = {f"dim_{d}": None for d in range(input_dim)}

        # Default XGBoost hyperparameters
        if xgb_params is None:
            self.xgb_params = {
                "max_depth": 5,
                "learning_rate": 0.1,
                "n_estimators": 300,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "gamma": 0.0,
                "min_child_weight": 1,
                "objective": "reg:squarederror",
                "random_state": 42,
                "n_jobs": -1,
            }
        else:
            self.xgb_params = xgb_params

        self.feature_importance = {}

    # ------------------------------------------------------------------ #
    # Time handling
    # ------------------------------------------------------------------ #

    def get_time_bins(self, eps: float = 0.1) -> np.ndarray:
        """Generate K discrete time bins in [eps, 1-eps]."""
        return np.linspace(eps, 1.0 - eps, self.n_timesteps)

    def get_time_bin_index(self, t: np.ndarray) -> np.ndarray:
        """
        Map continuous time t to nearest discrete bin index.

        t: shape (batch_size,) or (batch_size,1)
        """
        t_flat = np.atleast_1d(t).flatten()
        k = np.floor(t_flat * self.n_timesteps).astype(int)
        k = np.clip(k, 0, self.n_timesteps - 1)
        return k

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(self, x_input: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Predict drift at given position and time.

        Args:
            x_input: (batch_size, d)
            t: (batch_size,) or (batch_size,1)

        Returns:
            drift: (batch_size, d)
        """
        batch_size = x_input.shape[0]
        drift = np.zeros((batch_size, self.output_dim), dtype=float)

        t_flat = np.atleast_1d(t).flatten()
        time_bins = self.get_time_bin_index(t_flat)
        t_normalized = time_bins / self.n_timesteps

        features = np.column_stack([x_input, t_normalized])

        for d in range(self.output_dim):
            model = self.xgb_models[f"dim_{d}"]
            if model is not None:
                drift[:, d] = model.predict(features)

        return drift


# ---------------------------------------------------------------------- #
# Training data generation
# ---------------------------------------------------------------------- #


def get_train_tuple_xgboost(
    z_pairs: np.ndarray,
    n_timesteps: int = 10,
    sig: float = 0.1,
    eps: float = 0.1,
    fb: str = "f",
):
    """
    Generate training data across all discrete time bins.

    Args:
        z_pairs: (batch_size, 2, d) paired samples (z0, z1)
        n_timesteps: number of time bins K
        sig: noise level σ
        eps: time clipping
        fb: 'f' (forward) or 'b' (backward)

    Returns:
        X: (batch_size*K, d+1)  features [z_t, t_bin]
        y: (batch_size*K, d)    drift targets
    """
    assert fb in ("f", "b")
    batch_size, _, d = z_pairs.shape

    z0 = z_pairs[:, 0]  # (batch_size, d)
    z1 = z_pairs[:, 1]  # (batch_size, d)

    time_points = np.linspace(eps, 1.0 - eps, n_timesteps)

    X_list = []
    y_list = []

    for t in time_points:
        z = np.random.randn(batch_size, d)

        z_t = (1.0 - t) * z0 + t * z1 + sig * np.sqrt(t * (1.0 - t)) * z

        if fb == "f":
            drift_target = z1 - z0 - sig * np.sqrt(t / (1.0 - t)) * z
        else:
            drift_target = -(z1 - z0) - sig * np.sqrt((1.0 - t) / t) * z

        t_feature = np.full((batch_size, 1), t)
        features = np.column_stack([z_t, t_feature])

        X_list.append(features)
        y_list.append(drift_target)

    X = np.vstack(X_list)
    y = np.vstack(y_list)

    return X, y


# ---------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------- #


def train_xgboost_drift(
    xgb_net: XGBoostBridgeNet,
    x_pairs: np.ndarray,
    n_timesteps: int = 10,
    sig: float = 0.1,
    eps: float = 0.1,
    fb: str = "f",
    verbose: bool = 0,
) -> XGBoostBridgeNet:
    """
    Train XGBoost models for drift prediction at discrete time intervals.

    x_pairs: (N, 2, d)
    """
    assert fb in ("f", "b")
    batch_size, _, d = x_pairs.shape

    if verbose:
        print(f"Generating training data: {batch_size} samples × {n_timesteps} bins...")

    X, y = get_train_tuple_xgboost(x_pairs, n_timesteps, sig, eps, fb)

    if verbose:
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")

    for dim in range(d):
        if verbose:
            print(f"Training XGBoost for dimension {dim}/{d}...")

        y_dim = y[:, dim]

        model = xgb.XGBRegressor(**xgb_net.xgb_params)
        model.fit(X, y_dim, verbose=False)

        xgb_net.xgb_models[f"dim_{dim}"] = model
        xgb_net.feature_importance[f"dim_{dim}"] = model.feature_importances_

        if verbose:
            train_r2 = model.score(X, y_dim)
            print(f"  ✓ R² score: {train_r2:.4f}")

    if verbose:
        print("✓ Training complete.\n")

    return xgb_net


# ---------------------------------------------------------------------- #
# Sampling
# ---------------------------------------------------------------------- #


def sample_sde_xgboost(
    xgb_net: XGBoostBridgeNet,
    zstart: np.ndarray,
    n_euler_steps: int = 100,
    sig: float = 0.1,
    fb: str = "f",
):
    """
    Sample from learned SDE using Euler discretization.

    zstart: (batch_size, d)
    """
    assert fb in ("f", "b")
    dt = 1.0 / n_euler_steps

    z = zstart.copy()
    batch_size = z.shape[0]
    d = z.shape[1]

    traj = [z.copy()]

    time_steps = np.arange(n_euler_steps) / n_euler_steps
    if fb == "b":
        time_steps = 1.0 - time_steps

    for t in time_steps:
        t_batch = np.full((batch_size, 1), t)

        drift = xgb_net.predict(z, t_batch)
        z = z + drift * dt
        z = z + sig * np.random.randn(batch_size, d) * np.sqrt(dt)

        traj.append(z.copy())

    return traj


# ---------------------------------------------------------------------- #
# High-level DSBM-IMF training
# ---------------------------------------------------------------------- #


class DSBMContainer:
    def __init__(self):
        self.xgb_dict = {}


def train_dsbm_xgboost(
    dsbm_ipf: DSBMContainer,
    x_pairs: np.ndarray,
    n_timesteps: int = 10,
    sig: float = 0.1,
    eps: float = 0.1,
    fb: str = "f",
    verbose: bool = True,
):
    """
    Train one direction (forward or backward) of DSBM using XGBoost.
    """
    assert fb in ("f", "b")
    _, _, d = x_pairs.shape

    if verbose:
        print("\n" + "=" * 60)
        print(f"Training {fb.upper()} direction (XGBoost, K={n_timesteps})")
        print("=" * 60)

    xgb_net = XGBoostBridgeNet(
        input_dim=d,
        n_timesteps=n_timesteps,
    )

    xgb_net = train_xgboost_drift(
        xgb_net, x_pairs, n_timesteps=n_timesteps, sig=sig, eps=eps, fb=fb
    )

    dsbm_ipf.xgb_dict[fb] = xgb_net
    return dsbm_ipf, xgb_net


# def train_dsbm_xgboost_imf(
#     x_pairs: np.ndarray,
#     n_iterations: int = 10,
#     n_timesteps: int = 10,
#     sig: float = 0.1,
#     eps: float = 0.1,
#     verbose: bool = True,
# ) -> DSBMContainer:
#     """
#     Complete Iterative Markovian Fitting (IMF) loop using XGBoost.

#     x_pairs: (N, 2, d)
#     """
#     dsbm_ipf = DSBMContainer()

#     N, _, d = x_pairs.shape
#     z0_base = x_pairs[:, 0]
#     z1_base = x_pairs[:, 1]

#     if verbose:
#         print("\n" + "=" * 60)
#         print("DSBM-IMF with XGBoost")
#         print("=" * 60)
#         print(f"Data: N={N}, d={d}")
#         print(f"Parameters: K={n_timesteps}, σ={sig}, ε={eps}")
#         print(f"Iterations: {n_iterations} (alternating f/b)")
#         print("=" * 60 + "\n")

#     for iter_num in range(n_iterations):
#         fb = "b" if iter_num % 2 == 0 else "f"

#         perm_idx = np.random.permutation(N)
#         x_pairs_iter = np.stack([z0_base, z1_base[perm_idx]], axis=1)

#         dsbm_ipf, _ = train_dsbm_xgboost(
#             dsbm_ipf,
#             x_pairs_iter,
#             n_timesteps=n_timesteps,
#             sig=sig,
#             eps=eps,
#             fb=fb,
#             verbose=verbose,
#         )

#     return dsbm_ipf

# def train_dsbm_xgboost_imf_bridge(
#     x_pairs,
#     n_iterations=10,
#     n_timesteps=10,
#     sig=0.05,
#     eps=0.05,
#     verbose=True,
# ):
#     class DSBMContainer:
#         def __init__(self):
#             self.xgb_dict = {}

#     dsbm_ipf = DSBMContainer()
#     N, _, d = x_pairs.shape
#     z0_base = x_pairs[:, 0]
#     z1_base = x_pairs[:, 1]

#     for iter_num in range(n_iterations):
#         fb = "b" if iter_num % 2 == 0 else "f"

#         # 1) For the very first iteration, still use independent coupling
#         if iter_num == 0 or ("f" not in dsbm_ipf.xgb_dict) or ("b" not in dsbm_ipf.xgb_dict):
#             perm_idx = np.random.permutation(N)
#             x_pairs_iter = np.stack([z0_base, z1_base[perm_idx]], axis=1)

#         # 2) From iteration 1 onward, resample coupling using learned bridge
#         else:
#             x_pairs_iter = resample_coupling_from_bridge(
#                 xgb_net_f=dsbm_ipf.xgb_dict["f"],
#                 xgb_net_b=dsbm_ipf.xgb_dict["b"],
#                 z0_base=z0_base,
#                 z1_base=z1_base,
#                 n_euler_steps=50,
#                 sig=sig,
#             )

#         # 3) Train current direction on this updated coupling
#         dsbm_ipf, _ = train_dsbm_xgboost(
#             dsbm_ipf,
#             x_pairs_iter,
#             n_timesteps=n_timesteps,
#             sig=sig,
#             eps=eps,
#             fb=fb,
#             verbose=verbose,
#         )

#     return dsbm_ipf



def generate_samples_xgboost(
    dsbm_ipf: DSBMContainer,
    n_samples: int,
    d: int,
    n_euler_steps: int = 100,
    sig: float = 0.1,
) -> np.ndarray:
    """
    Generate synthetic samples using the trained forward SDE.
    """
    z_start = np.random.randn(n_samples, d)
    xgb_net_f = dsbm_ipf.xgb_dict["f"]
    traj = sample_sde_xgboost(xgb_net_f, z_start, n_euler_steps=n_euler_steps, sig=sig, fb="f")
    return traj[-1]



import numpy as np

def resample_coupling_from_bridge(
    xgb_net_f,
    xgb_net_b,
    z0_base,
    z1_base,
    n_euler_steps=50,
    sig=0.05,
):
    """
    Use learned forward/backward drifts to produce a new endpoint coupling.

    Args:
        xgb_net_f: forward XGBoostBridgeNet (Gaussian -> data)
        xgb_net_b: backward XGBoostBridgeNet (data -> Gaussian)
        z0_base: (N, d) current source samples
        z1_base: (N, d) current target samples
        n_euler_steps: Euler steps for intermediate sampling
        sig: diffusion noise

    Returns:
        x_pairs_new: (N, 2, d) new paired endpoints (X0^n, XT^n)
    """
    N, d = z0_base.shape

    # 1) Sample forward paths from z0_base

    traj_f = sample_sde_xgboost(
        xgb_net_f,
        zstart=z0_base,
        n_euler_steps=n_euler_steps,
        sig=sig,
        fb="f",
    )
    zT_f = traj_f[-1]  # (N, d) forward endpoints

    # 2) Sample backward paths from z1_base
    traj_b = sample_sde_xgboost(
        xgb_net_b,
        zstart=z1_base,
        n_euler_steps=n_euler_steps,
        sig=sig,
        fb="b",
    )
    z0_b = traj_b[-1]  # (N, d) backward endpoints

    # 3) Build new coupling:
    #    pair forward-start with backward-end (symmetrized heuristic)
    x_pairs_new = np.stack([z0_b, zT_f], axis=1)  # (N, 2, d)

    return x_pairs_new






"""
CORRECTED IMF Loop with Marginal Projection

Key fix: After resampling with learned bridge, we must:
1. Keep z0 ~ π₀ (Gaussian) for forward passes
2. Keep z1 ~ π₁ (Swiss roll) for backward passes
3. Only update the COUPLING (which z0 pairs with which z1)

This prevents distribution collapse.
"""


# ========================================================================
# PROPER IMF WITH MARGINAL PROJECTION
# ========================================================================

def train_dsbm_xgboost_imf_correct(
    x_pairs: np.ndarray,
    n_iterations: int = 12,
    n_timesteps: int = 15,
    sig: float = 0.05,
    eps: float = 0.05,
    verbose: bool = True,
    xgb_params: dict = None,
):
    """
    Corrected IMF loop that preserves marginals while updating coupling.

    Key insight:
    - Fixed marginals: z0_base ~ π₀ (Gaussian), z1_base ~ π₁ (Swiss Roll)
    - What changes: the PAIRING between z0 and z1
    - How: Use learned bridge to find which z0 should pair with which z1

    Args:
        x_pairs: (N, 2, d) initial paired data [z0_base, z1_base]
        n_iterations: Number of IMF iterations
        n_timesteps: Discrete time bins for XGBoost
        sig: Diffusion noise level
        eps: Time clipping
        verbose: Print progress
        xgb_params: XGBoost hyperparameters

    Returns:
        dsbm_ipf: Trained model container
    """

    class DSBMContainer:
        def __init__(self):
            self.xgb_dict = {}

    dsbm_ipf = DSBMContainer()
    N, _, d = x_pairs.shape

    # Fixed marginals (NEVER CHANGE THESE)
    z0_base = x_pairs[:, 0].copy()  # (N, d) π₀ (Gaussian)
    z1_base = x_pairs[:, 1].copy()  # (N, d) π₁ (Swiss Roll)

    # Default XGBoost params
    if xgb_params is None:
        xgb_params = {
            "max_depth": 7,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.7,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "gamma": 0,
            "min_child_weight": 1,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }

    if verbose:
        print("\n" + "=" * 80)
        print("CORRECTED DSBM-IMF WITH MARGINAL PROJECTION")
        print("=" * 80)
        print(f"Data: N={N}, d={d}")
        print(f"Parameters: K={n_timesteps}, σ={sig}, ε={eps}")
        print(f"Iterations: {n_iterations}")
        print(f"XGBoost: max_depth={xgb_params['max_depth']}, "
              f"n_estimators={xgb_params['n_estimators']}")
        print("=" * 80 + "\n")

    # ====================================================================
    # IMF LOOP
    # ====================================================================

    for iter_num in range(n_iterations):
        fb = "b" if iter_num % 2 == 0 else "f"

        if verbose:
            print(f"\n{'='*70}")
            print(f"IMF Iteration {iter_num + 1}/{n_iterations} - Direction: {fb.upper()}")
            print(f"{'='*70}")

        # ----------------------------------------------------------------
        # STEP 1: Compute optimal coupling using current bridge
        # ----------------------------------------------------------------

        if iter_num == 0:
            # First iteration: random independent coupling
            if verbose:
                print("  Strategy: Random independent coupling (initialization)")
            perm_idx = np.random.permutation(N)

        else:
            # Later iterations: use learned bridge to find optimal pairing
            if verbose:
                print("  Strategy: Recompute coupling via learned bridge")

            if fb == "f":
                # Forward: learn π₀ → π₁
                # Sample forward from each z0, find closest z1
                perm_idx = compute_optimal_coupling_forward(
                    dsbm_ipf.xgb_dict.get("f"),
                    z0_base,
                    z1_base,
                    n_euler_steps=50,
                    sig=sig,
                    verbose=verbose,
                )

            else:  # fb == "b"
                # Backward: learn π₁ → π₀
                # Sample backward from each z1, find closest z0
                perm_idx = compute_optimal_coupling_backward(
                    dsbm_ipf.xgb_dict.get("b"),
                    z0_base,
                    z1_base,
                    n_euler_steps=50,
                    sig=sig,
                    verbose=verbose,
                )

        # ----------------------------------------------------------------
        # STEP 2: Create paired data with new coupling BUT FIXED MARGINALS
        # ----------------------------------------------------------------

        if fb == "f":
            # Forward: pair z0[i] with z1[perm_idx[i]]
            x_pairs_iter = np.stack([z0_base, z1_base[perm_idx]], axis=1)
        else:
            # Backward: pair z0[perm_idx[i]] with z1[i]
            x_pairs_iter = np.stack([z0_base[perm_idx], z1_base], axis=1)

        # ----------------------------------------------------------------
        # STEP 3: Train XGBoost on this coupling
        # ----------------------------------------------------------------

        xgb_net = XGBoostBridgeNet(
            input_dim=d,
            n_timesteps=n_timesteps,
            xgb_params=xgb_params,
        )

        xgb_net = train_xgboost_drift(
            xgb_net,
            x_pairs_iter,
            n_timesteps=n_timesteps,
            sig=sig,
            eps=eps,
            fb=fb,
            verbose=verbose,
        )

        dsbm_ipf.xgb_dict[fb] = xgb_net

        if verbose:
            print(f"✓ Iteration {iter_num + 1} complete.\n")

    if verbose:
        print("\n" + "=" * 80)
        print("✓ IMF TRAINING COMPLETE")
        print("=" * 80)

    return dsbm_ipf


# ========================================================================
# COUPLING COMPUTATION FUNCTIONS
# ========================================================================

def compute_optimal_coupling_forward(
    xgb_net_f,
    z0_base: np.ndarray,
    z1_base: np.ndarray,
    n_euler_steps: int = 50,
    sig: float = 0.05,
    verbose: bool = False,
):
    """
    Compute optimal coupling for FORWARD pass.

    Strategy:
    1. Sample forward from each z0 using learned drift
    2. Match each endpoint to closest z1 (Hungarian/greedy)
    3. Return permutation indices

    Args:
        xgb_net_f: Learned forward XGBoost network
        z0_base: (N, d) fixed π₀ samples (Gaussian)
        z1_base: (N, d) fixed π₁ samples (Swiss Roll)
        n_euler_steps: Euler integration steps
        sig: Diffusion noise
        verbose: Print progress

    Returns:
        perm_idx: (N,) permutation indices for z1_base
    """
    N, d = z0_base.shape

    if xgb_net_f is None:
        # Fallback: random permutation
        if verbose:
            print("    Warning: No forward model, using random coupling")
        return np.random.permutation(N)

    # Sample forward paths from z0_base
    if verbose:
        print(f"    Sampling {N} forward paths from π₀...")

    traj_f = sample_sde_xgboost(
        xgb_net_f,
        zstart=z0_base,
        n_euler_steps=n_euler_steps,
        sig=sig,
        fb="f",
    )
    z_endpoints = traj_f[-1]  # (N, d) sampled endpoints

    # Match endpoints to z1_base (greedy nearest neighbor)
    if verbose:
        print(f"    Computing optimal coupling (greedy matching)...")

    perm_idx = greedy_nearest_neighbor_matching(z_endpoints, z1_base)

    if verbose:
        # Measure coupling quality
        distances = np.linalg.norm(z_endpoints - z1_base[perm_idx], axis=1)
        print(f"    ✓ Mean coupling distance: {distances.mean():.4f}")

    return perm_idx


def compute_optimal_coupling_backward(
    xgb_net_b,
    z0_base: np.ndarray,
    z1_base: np.ndarray,
    n_euler_steps: int = 50,
    sig: float = 0.05,
    verbose: bool = False,
):
    """
    Compute optimal coupling for BACKWARD pass.

    Strategy:
    1. Sample backward from each z1 using learned drift
    2. Match each endpoint to closest z0
    3. Return permutation indices

    Args:
        xgb_net_b: Learned backward XGBoost network
        z0_base: (N, d) fixed π₀ samples (Gaussian)
        z1_base: (N, d) fixed π₁ samples (Swiss Roll)
        n_euler_steps: Euler integration steps
        sig: Diffusion noise
        verbose: Print progress

    Returns:
        perm_idx: (N,) permutation indices for z0_base
    """
    N, d = z1_base.shape

    if xgb_net_b is None:
        if verbose:
            print("    Warning: No backward model, using random coupling")
        return np.random.permutation(N)

    # Sample backward paths from z1_base
    if verbose:
        print(f"    Sampling {N} backward paths from π₁...")

    traj_b = sample_sde_xgboost(
        xgb_net_b,
        zstart=z1_base,
        n_euler_steps=n_euler_steps,
        sig=sig,
        fb="b",
    )
    z_endpoints = traj_b[-1]  # (N, d) sampled endpoints

    # Match endpoints to z0_base
    if verbose:
        print(f"    Computing optimal coupling (greedy matching)...")

    perm_idx = greedy_nearest_neighbor_matching(z_endpoints, z0_base)

    if verbose:
        distances = np.linalg.norm(z_endpoints - z0_base[perm_idx], axis=1)
        print(f"    ✓ Mean coupling distance: {distances.mean():.4f}")

    return perm_idx


def greedy_nearest_neighbor_matching(
    sources: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Greedy nearest-neighbor matching (fast approximation of optimal transport).

    For each source[i], find the nearest unused target[j].

    Args:
        sources: (N, d) source points
        targets: (N, d) target points

    Returns:
        perm_idx: (N,) indices such that sources[i] → targets[perm_idx[i]]
    """
    N = len(sources)
    perm_idx = np.zeros(N, dtype=int)
    used = np.zeros(N, dtype=bool)

    for i in range(N):
        # Compute distances to all unused targets
        dists = np.linalg.norm(targets - sources[i], axis=1)
        dists[used] = np.inf  # Mask already-used targets

        # Find nearest
        j = np.argmin(dists)
        perm_idx[i] = j
        used[j] = True

    return perm_idx


