"""
Generalized IMF with Blockwise Feature Transfer

Instead of:
- Per-dimension transfer (1 feature at a time)
- Full transfer (all features at once)

Now supports:
- Arbitrary feature groupings (blocks)
- Blockwise parallel transfer
- Flexible scheduling

Example:
    For 10D data:
    - blocks=[[0,1,2], [3,4,5], [6,7,8], [9]]  → 4 blocks
    - blocks=[[0,1,2,3,4,5,6,7,8,9]]           → 1 block (all at once)
    - blocks=[[i] for i in range(10)]          → 10 blocks (per-dimension)
"""

import numpy as np
from typing import List, Optional
from models.DSBM_xgboost.XGBOOSTDSBM_COSRRECTED import (
    train_xgboost_drift,
    sample_sde_xgboost,
    XGBoostBridgeNet,
)


# ========================================================================
# BLOCKWISE IMF WITH FLEXIBLE FEATURE GROUPING
# ========================================================================

def train_dsbm_xgboost_imf_blockwise(
    x_pairs: np.ndarray,
    feature_blocks: Optional[List[List[int]]] = None,
    n_iterations: int = 12,
    n_timesteps: int = 15,
    sig: float = 0.05,
    eps: float = 0.05,
    verbose: bool = True,
    xgb_params: dict = None,
):
    """
    Corrected IMF with blockwise/grouped feature transfer.

    Allows flexible grouping of features for parallel learning:
    - Per-dimension: [[0], [1], [2], ...] (current default)
    - Blockwise: [[0,1], [2,3], ...] (groups)
    - Full: [[0,1,2,...,d]] (all features at once)
    - Custom: any list of feature groups

    Key insight:
    - Fixed marginals: z0_base ~ π₀, z1_base ~ π₁
    - Updated coupling computed per feature block
    - Blocks trained independently or sequentially

    Args:
        x_pairs: (N, 2, d) initial paired data
        feature_blocks: List[List[int]] feature grouping
            If None: defaults to per-dimension [[0], [1], ..., [d-1]]
            Examples:
              [[0,1,2], [3,4,5]]  → 2 blocks of 3 features each
              [[0,1,2,3,4,5]]     → 1 block (all features)
              [[i] for i in range(10)]  → per-dimension (10 blocks)
        n_iterations: Number of IMF iterations
        n_timesteps: Discrete time bins for XGBoost
        sig: Diffusion noise level
        eps: Time clipping
        verbose: Print progress
        xgb_params: XGBoost hyperparameters

    Returns:
        dsbm_ipf: Trained model container with 'f' and 'b' networks
    """

    class DSBMContainer:
        def __init__(self):
            self.xgb_dict = {}

    dsbm_ipf = DSBMContainer()
    N, _, d = x_pairs.shape

    # Fixed marginals (NEVER CHANGE THESE)
    z0_base = x_pairs[:, 0].copy()  # (N, d) π₀
    z1_base = x_pairs[:, 1].copy()  # (N, d) π₁

    # Default feature blocks: per-dimension
    if feature_blocks is None:
        feature_blocks = [[i] for i in range(d)]

    # Validate blocks
    all_features = set()
    for block in feature_blocks:
        all_features.update(block)
    assert all_features == set(range(d)), f"Blocks must cover all features 0..{d-1}"

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
        print("BLOCKWISE DSBM-IMF WITH FEATURE GROUPING")
        print("=" * 80)
        print(f"Data: N={N}, d={d}")
        print(f"Feature blocks: {len(feature_blocks)} groups")
        for i, block in enumerate(feature_blocks):
            print(f"  Block {i}: features {block}")
        print(f"Parameters: K={n_timesteps}, σ={sig}, ε={eps}")
        print(f"Iterations: {n_iterations}")
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
        # STEP 1: Compute couplings for each feature block
        # ----------------------------------------------------------------

        if iter_num == 0:
            # First iteration: random independent coupling
            if verbose:
                print("  Strategy: Random independent coupling (initialization)")
            all_perm_idx = np.random.permutation(N)

        else:
            # Later iterations: compute per-block couplings
            if verbose:
                print(f"  Strategy: Blockwise coupling ({len(feature_blocks)} blocks)")

            all_perm_idx = compute_blockwise_coupling(
                dsbm_ipf.xgb_dict.get(fb),
                z0_base,
                z1_base,
                feature_blocks,
                n_euler_steps=50,
                sig=sig,
                fb=fb,
                verbose=verbose,
            )

        # ----------------------------------------------------------------
        # STEP 2: Create paired data with updated coupling
        # ----------------------------------------------------------------

        if fb == "f":
            x_pairs_iter = np.stack([z0_base, z1_base[all_perm_idx]], axis=1)
        else:
            x_pairs_iter = np.stack([z0_base[all_perm_idx], z1_base], axis=1)

        # ----------------------------------------------------------------
        # STEP 3: Train XGBoost on this coupling (blockwise or all-at-once)
        # ----------------------------------------------------------------

        if len(feature_blocks) == 1:
            # Single block: train normally on all features
            if verbose:
                print(f"  Training single block on all {d} features...")

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

        else:
            # Multiple blocks: train each independently, then combine
            if verbose:
                print(f"  Training {len(feature_blocks)} feature blocks...")

            xgb_nets = []
            for block_idx, block_features in enumerate(feature_blocks):
                if verbose:
                    print(f"    Block {block_idx + 1}/{len(feature_blocks)}: "
                          f"features {block_features}")

                # Extract only this block's features
                x_pairs_block = x_pairs_iter[:, :, block_features]

                # Train XGBoost for this block
                block_dim = len(block_features)
                xgb_net_block = XGBoostBridgeNet(
                    input_dim=block_dim,
                    n_timesteps=n_timesteps,
                    xgb_params=xgb_params,
                )

                xgb_net_block = train_xgboost_drift(
                    xgb_net_block,
                    x_pairs_block,
                    n_timesteps=n_timesteps,
                    sig=sig,
                    eps=eps,
                    fb=fb,
                    verbose=False,  # Suppress per-dimension output
                )

                xgb_nets.append((block_features, xgb_net_block))

            # Store combined network
            dsbm_ipf.xgb_dict[fb] = BlockwiseXGBoostNetwork(xgb_nets)

            if verbose:
                print(f"  ✓ All {len(feature_blocks)} blocks trained")

        if verbose:
            print(f"✓ Iteration {iter_num + 1} complete.\n")

    if verbose:
        print("\n" + "=" * 80)
        print("✓ IMF TRAINING COMPLETE")
        print("=" * 80)

    return dsbm_ipf


# ========================================================================
# BLOCKWISE COUPLING COMPUTATION
# ========================================================================

def compute_blockwise_coupling(
    xgb_net,
    z0_base: np.ndarray,
    z1_base: np.ndarray,
    feature_blocks: List[List[int]],
    n_euler_steps: int = 50,
    sig: float = 0.05,
    fb: str = "f",
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute optimal coupling by aggregating per-block couplings.

    Strategy:
    1. For each feature block, compute coupling independently
    2. Aggregate votes via majority/consensus
    3. Return final permutation

    Args:
        xgb_net: Network (single or blockwise)
        z0_base: (N, d) fixed π₀ samples
        z1_base: (N, d) fixed π₁ samples
        feature_blocks: Feature grouping
        n_euler_steps: Euler steps
        sig: Diffusion noise
        fb: 'f' or 'b'
        verbose: Print progress

    Returns:
        perm_idx: (N,) consensus permutation
    """
    N, d = z0_base.shape

    if xgb_net is None:
        if verbose:
            print("    Warning: No model, using random coupling")
        return np.random.permutation(N)

    # Collect per-block permutations
    block_perms = []

    if isinstance(xgb_net, BlockwiseXGBoostNetwork):
        # Blockwise network: get coupling from each block
        for block_idx, (block_features, xgb_net_block) in enumerate(xgb_net.networks):
            # Extract block features
            z0_block = z0_base[:, block_features]
            z1_block = z1_base[:, block_features]

            # Sample and match for this block
            perm = compute_single_block_coupling(
                xgb_net_block,
                z0_block,
                z1_block,
                n_euler_steps=n_euler_steps,
                sig=sig,
                fb=fb,
                verbose=False,
            )

            block_perms.append(perm)

        # Aggregate via voting
        consensus_perm = aggregate_permutations(block_perms, method="majority")

    else:
        # Single network: compute full coupling
        z0 = z0_base
        z1 = z1_base

        if fb == "f":
            traj = sample_sde_xgboost(
                xgb_net,
                zstart=z0,
                n_euler_steps=n_euler_steps,
                sig=sig,
                fb="f",
            )
        else:
            traj = sample_sde_xgboost(
                xgb_net,
                zstart=z1,
                n_euler_steps=n_euler_steps,
                sig=sig,
                fb="b",
            )

        z_endpoints = traj[-1]
        consensus_perm = greedy_nearest_neighbor_matching(z_endpoints, z1 if fb == "f" else z0)

    if verbose:
        print(f"    ✓ Coupling computed ({len(block_perms)} blocks aggregated)")

    return consensus_perm


def compute_single_block_coupling(
    xgb_net_block,
    z0_block: np.ndarray,
    z1_block: np.ndarray,
    n_euler_steps: int = 50,
    sig: float = 0.05,
    fb: str = "f",
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute coupling for a single feature block.

    Args:
        xgb_net_block: XGBoost network for this block
        z0_block: (N, d_block) source features
        z1_block: (N, d_block) target features
        n_euler_steps: Euler steps
        sig: Diffusion noise
        fb: 'f' or 'b'
        verbose: Print progress

    Returns:
        perm_idx: (N,) permutation for this block
    """
    traj = sample_sde_xgboost(
        xgb_net_block,
        zstart=z0_block if fb == "f" else z1_block,
        n_euler_steps=n_euler_steps,
        sig=sig,
        fb=fb,
    )

    z_endpoints = traj[-1]
    target = z1_block if fb == "f" else z0_block

    perm = greedy_nearest_neighbor_matching(z_endpoints, target)
    return perm


def aggregate_permutations(
    perms: List[np.ndarray],
    method: str = "majority",
) -> np.ndarray:
    """
    Aggregate multiple permutations via voting.

    Args:
        perms: List of (N,) permutation arrays
        method: 'majority', 'first', or 'consensus'

    Returns:
        aggregated_perm: (N,) consensus permutation
    """
    N = len(perms[0])
    num_blocks = len(perms)

    if method == "majority":
        # Majority vote on each position
        aggregated = np.zeros(N, dtype=int)
        for i in range(N):
            votes = [p[i] for p in perms]
            # Most common vote (or first if tie)
            aggregated[i] = max(set(votes), key=votes.count)
        return aggregated

    elif method == "first":
        # Use first permutation
        return perms[0]

    elif method == "consensus":
        # Return only if all agree; else random
        aggregated = np.zeros(N, dtype=int)
        for i in range(N):
            votes = [p[i] for p in perms]
            if len(set(votes)) == 1:
                aggregated[i] = votes[0]
            else:
                aggregated[i] = np.random.choice(votes)
        return aggregated

    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# ========================================================================
# BLOCKWISE NETWORK WRAPPER
# ========================================================================

class BlockwiseXGBoostNetwork:
    """
    Wrapper for multiple XGBoost networks, one per feature block.

    Enables:
    - Independent training per block
    - Parallel predictions
    - Modular updates
    """

    def __init__(self, networks: List[tuple]):
        """
        Args:
            networks: List of (feature_indices, XGBoostBridgeNet) tuples
        """
        self.networks = networks  # List[(feature_indices, xgb_net), ...]

    def predict(self, x_input: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Predict drift by combining per-block predictions.

        Args:
            x_input: (N, d) input features
            t: (N,) or (N, 1) time values

        Returns:
            drift: (N, d) combined drift predictions
        """
        N, d = x_input.shape
        drift = np.zeros((N, d))

        for block_features, xgb_net_block in self.networks:
            # Extract block features
            x_block = x_input[:, block_features]

            # Predict drift for this block
            drift_block = xgb_net_block.predict(x_block, t)

            # Place in full drift
            drift[:, block_features] = drift_block

        return drift


def greedy_nearest_neighbor_matching(
    sources: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Greedy nearest-neighbor matching."""
    N = len(sources)
    perm_idx = np.zeros(N, dtype=int)
    used = np.zeros(N, dtype=bool)

    for i in range(N):
        dists = np.linalg.norm(targets - sources[i], axis=1)
        dists[used] = np.inf
        j = np.argmin(dists)
        perm_idx[i] = j
        used[j] = True

    return perm_idx


# ========================================================================
# USAGE EXAMPLES
# ========================================================================

if __name__ == "__main__":
    print("""
BLOCKWISE IMF WITH FLEXIBLE FEATURE GROUPING
=============================================

Example 1: Per-dimension (original)
    blocks = [[0], [1], [2]]  → Train 3 separate 1D models
    
Example 2: Blockwise
    blocks = [[0, 1], [2, 3]]  → Train 2 separate 2D models
    
Example 3: All-at-once
    blocks = [[0, 1, 2, 3]]  → Train 1 single 4D model
    
Example 4: Mixed
    blocks = [[0, 1], [2], [3, 4, 5]]  → 3 blocks of varying sizes

USAGE:
------
from imf_marginal_projection import train_dsbm_xgboost_imf_blockwise

# Option 1: Full transfer (all features at once)
dsbm_ipf = train_dsbm_xgboost_imf_blockwise(
    x_pairs,
    feature_blocks=[[0, 1]],  # One block with both features
    n_iterations=12,
    n_timesteps=15,
    sig=0.05,
)

# Option 2: Per-dimension (default)
dsbm_ipf = train_dsbm_xgboost_imf_blockwise(
    x_pairs,
    feature_blocks=None,  # Defaults to [[0], [1], ...]
    n_iterations=12,
)

# Option 3: Custom blockwise
dsbm_ipf = train_dsbm_xgboost_imf_blockwise(
    x_pairs,
    feature_blocks=[[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]],
    n_iterations=12,
)
""")
