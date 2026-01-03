"""
XGBoostDSBMTabularBridge: DSBM with XGBoost for Tabular Data

A generalized, production-ready class for training discrete Schrödinger Bridge models
on arbitrary m×n tabular datasets using XGBoost as the drift function approximator.

Provides the same interface as DSBMTabularBridge but with:
  - XGBoost-based tree models (vs neural networks)
  - Discrete time binning (more efficient for tabular data)
  - Proper IMF loop with marginal projection
  - Full hyperparameter control
  - Scalable to high-dimensional tabular data
"""

import numpy as np
import xgboost as xgb
from typing import Dict, Any, Optional, List, Tuple
import copy


class XGBoostDriftNetwork:
    """
    XGBoost-based score network for learning drift across discrete time bins.
    
    Trains one XGBoost model per output dimension, each predicting drift at
    timestep t given position x.
    
    Attributes:
        input_dim (int): Data dimension d
        n_timesteps (int): Number of discrete time bins K
        xgb_models (dict): Per-dimension XGBoost regressors
        xgb_params (dict): XGBoost hyperparameters
        feature_importance (dict): Learned feature importances per dimension
    """
    
    def __init__(self, input_dim: int, n_timesteps: int = 15, xgb_params: Optional[Dict] = None):
        """
        Initialize XGBoost drift network.
        
        Args:
            input_dim (int): Data dimension d
            n_timesteps (int): Number of discrete time bins K (default: 15)
            xgb_params (dict): XGBoost hyperparameters, or None for defaults
        """
        self.input_dim = input_dim
        self.n_timesteps = n_timesteps
        self.output_dim = input_dim
        
        # One model per output dimension
        self.xgb_models = {f"dim_{d}": None for d in range(input_dim)}
        
        # Default XGBoost hyperparameters
        if xgb_params is None:
            self.xgb_params = {
                "max_depth": 7,
                "learning_rate": 0.05,
                "n_estimators": 300,
                "subsample": 0.7,
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
    
    # ================================================================
    # Time handling
    # ================================================================
    
    def get_time_bins(self, eps: float = 0.05) -> np.ndarray:
        """
        Generate K discrete time bins uniformly in [eps, 1-eps].
        
        Args:
            eps (float): Time clipping margin
            
        Returns:
            np.ndarray: Shape (K,) with time points
        """
        return np.linspace(eps, 1.0 - eps, self.n_timesteps)
    
    def get_time_bin_index(self, t: np.ndarray) -> np.ndarray:
        """
        Map continuous time t ∈ [0,1] to nearest discrete bin index ∈ [0,K-1].
        
        Args:
            t: Shape (batch_size,) or (batch_size, 1)
            
        Returns:
            np.ndarray: Bin indices, shape (batch_size,)
        """
        t_flat = np.atleast_1d(t).flatten()
        k = np.floor(t_flat * self.n_timesteps).astype(int)
        k = np.clip(k, 0, self.n_timesteps - 1)
        return k
    
    # ================================================================
    # Prediction
    # ================================================================
    
    def predict(self, x_input: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Predict drift at given position and time.
        
        Args:
            x_input: Shape (batch_size, d) - positions
            t: Shape (batch_size,) or (batch_size, 1) - times in [0,1]
            
        Returns:
            np.ndarray: Drift predictions, shape (batch_size, d)
        """
        batch_size = x_input.shape[0]
        drift = np.zeros((batch_size, self.output_dim), dtype=np.float32)
        
        t_flat = np.atleast_1d(t).flatten()
        time_bins = self.get_time_bin_index(t_flat)
        t_normalized = time_bins / self.n_timesteps
        
        # Features: [x_0, x_1, ..., x_{d-1}, t_normalized]
        features = np.column_stack([x_input, t_normalized]).astype(np.float32)
        
        for d in range(self.output_dim):
            model = self.xgb_models[f"dim_{d}"]
            if model is not None:
                drift[:, d] = model.predict(features)
        
        return drift
    
    # ================================================================
    # Training data generation
    # ================================================================
    
    @staticmethod
    def get_training_data(
        z_pairs: np.ndarray,
        n_timesteps: int = 15,
        sig: float = 0.05,
        eps: float = 0.05,
        fb: str = "f",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate XGBoost training data across all discrete time bins.
        
        For each time point t in [eps, 1-eps]:
          - Interpolate: z_t = (1-t)·z0 + t·z1 + σ·√(t(1-t))·noise
          - Compute drift target based on forward/backward direction
          - Create feature vector [z_t, t_normalized]
        
        Args:
            z_pairs: Shape (N, 2, d) paired samples [z0, z1]
            n_timesteps (int): Number of time bins K
            sig (float): Diffusion noise level σ
            eps (float): Time clipping margin
            fb (str): 'f' for forward or 'b' for backward
            
        Returns:
            Tuple[X, y] where:
              X: Shape (N*K, d+1) - features [z_t, t_normalized]
              y: Shape (N*K, d) - drift targets
        """
        assert fb in ("f", "b"), f"fb must be 'f' or 'b', got {fb}"
        
        N, _, d = z_pairs.shape
        z0 = z_pairs[:, 0]  # (N, d)
        z1 = z_pairs[:, 1]  # (N, d)
        
        time_points = np.linspace(eps, 1.0 - eps, n_timesteps)
        
        X_list = []
        y_list = []
        
        for t in time_points:
            # Sample noise
            noise = np.random.randn(N, d).astype(np.float32)
            
            # Interpolate path: z_t = (1-t)z0 + t·z1 + σ√(t(1-t))·noise
            z_t = (1.0 - t) * z0 + t * z1 + sig * np.sqrt(t * (1.0 - t)) * noise
            z_t = z_t.astype(np.float32)
            
            # Compute drift target
            if fb == "f":
                # Forward: drift toward z1
                drift_target = z1 - z0 - sig * np.sqrt(t / (1.0 - t)) * noise
            else:
                # Backward: drift toward z0
                drift_target = -(z1 - z0) - sig * np.sqrt((1.0 - t) / t) * noise
            
            drift_target = drift_target.astype(np.float32)
            
            # Time feature
            t_feature = np.full((N, 1), t, dtype=np.float32)
            
            # Stack features: [z_t, t_feature]
            features = np.column_stack([z_t, t_feature]).astype(np.float32)
            
            X_list.append(features)
            y_list.append(drift_target)
        
        X = np.vstack(X_list)
        y = np.vstack(y_list)
        
        return X.astype(np.float32), y.astype(np.float32)
    
    # ================================================================
    # Training
    # ================================================================
    
    def train(
        self,
        x_pairs: np.ndarray,
        n_timesteps: int = 15,
        sig: float = 0.05,
        eps: float = 0.05,
        fb: str = "f",
        verbose: bool = False,
    ) -> "XGBoostDriftNetwork":
        """
        Train XGBoost drift models for forward or backward direction.
        
        Args:
            x_pairs: Shape (N, 2, d) paired samples [z0, z1]
            n_timesteps (int): Number of time bins K
            sig (float): Diffusion noise level σ
            eps (float): Time clipping margin
            fb (str): 'f' for forward, 'b' for backward
            verbose (bool): Print training progress
            
        Returns:
            self (for chaining)
        """
        assert fb in ("f", "b"), f"fb must be 'f' or 'b', got {fb}"
        
        N, _, d = x_pairs.shape
        
        if verbose:
            print(f"\nGenerating {fb.upper()} training data: {N} samples × {n_timesteps} bins...")
        
        X, y = self.get_training_data(x_pairs, n_timesteps, sig, eps, fb)
        
        if verbose:
            print(f"  X shape: {X.shape}, y shape: {y.shape}")
            print(f"\nTraining XGBoost models ({d} dimensions):")
        
        for dim in range(d):
            if verbose:
                print(f"  Dimension {dim+1}/{d}...", end=" ")
            
            y_dim = y[:, dim]
            
            # Create and train XGBoost regressor
            model = xgb.XGBRegressor(**self.xgb_params)
            model.fit(X, y_dim, verbose=False)
            
            self.xgb_models[f"dim_{dim}"] = model
            self.feature_importance[f"dim_{dim}"] = model.feature_importances_
            
            if verbose:
                train_r2 = model.score(X, y_dim)
                print(f"R² = {train_r2:.4f}")
        
        if verbose:
            print()
        
        return self
    
    # ================================================================
    # Sampling
    # ================================================================
    
    def sample(
        self,
        zstart: np.ndarray,
        n_euler_steps: int = 100,
        sig: float = 0.05,
        fb: str = "f",
    ) -> List[np.ndarray]:
        """
        Sample from learned SDE using Euler-Maruyama discretization.
        
        Args:
            zstart: Shape (batch_size, d) - starting positions
            n_euler_steps (int): Number of Euler steps
            sig (float): Diffusion noise level σ
            fb (str): 'f' for forward (t: 0→1), 'b' for backward (t: 1→0)
            
        Returns:
            List[np.ndarray]: Trajectory of length n_euler_steps+1,
                              each element shape (batch_size, d)
        """
        assert fb in ("f", "b"), f"fb must be 'f' or 'b', got {fb}"
        
        dt = 1.0 / n_euler_steps
        z = zstart.copy().astype(np.float32)
        batch_size, d = z.shape
        
        trajectory = [z.copy()]
        
        # Time steps
        time_steps = np.arange(n_euler_steps) / n_euler_steps
        if fb == "b":
            # Backward: go from t=1 to t=0
            time_steps = 1.0 - time_steps
        
        for t in time_steps:
            # Get drift at current position and time
            t_batch = np.full((batch_size,), t, dtype=np.float32)
            drift = self.predict(z, t_batch)
            
            # Euler step: z_{k+1} = z_k + drift(z_k, t_k)·dt + σ·dW
            z = z + drift * dt
            z = z + sig * np.random.randn(batch_size, d).astype(np.float32) * np.sqrt(dt)
        
        trajectory.append(z.copy())
        
        return trajectory


class XGBoostDSBMTabularBridge:
    """
    DSBM solver for tabular data using XGBoost with proper IMF loop.
    
    Generalizes to arbitrary m×n datasets by:
      1. Treating data as (m, n) samples with n features
      2. Learning joint distributions on all n features together
      3. Using XGBoost for scalable tree-based drift learning
      4. Implementing proper Iterative Marginal Fitting (IMF)
    
    Attributes:
        x0_train: Shape (m_train, n) - source distribution
        x1_train: Shape (m_train, n) - target distribution
        x0_test: Optional, shape (m_test, n)
        x1_test: Optional, shape (m_test, n)
        drift_networks: Dict of forward/backward XGBoostDriftNetwork models
    """
    
    def __init__(
        self,
        x0_train: np.ndarray,
        x1_train: np.ndarray,
        x0_test: Optional[np.ndarray] = None,
        x1_test: Optional[np.ndarray] = None,
        n_timesteps: int = 15,
        sig: float = 0.05,
        eps: float = 0.05,
        xgb_params: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initialize XGBoost DSBM bridge.
        
        Args:
            x0_train: Shape (m_train, n) - source samples (e.g., Gaussian noise)
            x1_train: Shape (m_train, n) - target samples (e.g., real data)
            x0_test: Optional, shape (m_test, n) - source test samples
            x1_test: Optional, shape (m_test, n) - target test samples
            n_timesteps (int): Number of time bins K for XGBoost
            sig (float): Diffusion noise level σ
            eps (float): Time clipping margin
            xgb_params (dict): XGBoost hyperparameters, or None for defaults
            **kwargs: Additional arguments (ignored for compatibility)
        """
        # Store training data as-is
        self.x0_train = np.asarray(x0_train, dtype=np.float32)
        self.x1_train = np.asarray(x1_train, dtype=np.float32)
        
        # Optional test data
        self.x0_test = np.asarray(x0_test, dtype=np.float32) if x0_test is not None else None
        self.x1_test = np.asarray(x1_test, dtype=np.float32) if x1_test is not None else None
        
        # Data dimensions
        self.m_train = self.x0_train.shape[0]
        self.n_features = self.x0_train.shape[1]
        
        # DSBM parameters
        self.n_timesteps = n_timesteps
        self.sig = sig
        self.eps = eps
        self.xgb_params = xgb_params
        
        # Drift networks (trained during fit)
        self.drift_networks = {}  # Keys: 'f', 'b'
        self.is_trained = False
        
        # Training history
        self.history = {
            'iteration': [],
            'direction': [],
            'forward_r2': [],
            'backward_r2': [],
        }
    
    # ================================================================
    # Training
    # ================================================================
    
    def fit(
        self,
        n_iterations: int = 12,
        imf_batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Train DSBM using Iterative Marginal Fitting (IMF) loop.
        
        Key idea:
          - Fix marginals: z0_base ~ π₀ (source), z1_base ~ π₁ (target)
          - Iteratively improve coupling by learning forward/backward bridges
          - On each iteration, use learned drift to resample optimal coupling
        
        Args:
            n_iterations (int): Number of IMF iterations
            imf_batch_size (int): Batch size within each iteration (default: full)
            verbose (bool): Print training progress
            
        Returns:
            Dict with training history
        """
        if verbose:
            print("\n" + "=" * 80)
            print("XGBoost DSBM Tabular Bridge - Iterative Marginal Fitting (IMF)")
            print("=" * 80)
            print(f"Data shape: ({self.m_train}, {self.n_features})")
            print(f"Parameters: K={self.n_timesteps}, σ={self.sig}, ε={self.eps}")
            print(f"Iterations: {n_iterations} (alternating forward/backward)")
            if self.xgb_params:
                print(f"XGBoost: max_depth={self.xgb_params.get('max_depth', 7)}, "
                      f"n_estimators={self.xgb_params.get('n_estimators', 300)}")
            print("=" * 80 + "\n")
        
        # Fixed marginals (NEVER change these)
        z0_base = self.x0_train.copy()  # (m, n) source
        z1_base = self.x1_train.copy()  # (m, n) target
        
        # Current coupling
        current_coupling = np.stack([z0_base, z1_base], axis=1)  # (m, 2, n)
        
        # ====================================================================
        # IMF LOOP
        # ====================================================================
        
        for iter_num in range(n_iterations):
            fb = "b" if iter_num % 2 == 0 else "f"
            
            if verbose:
                print(f"{'='*70}")
                print(f"IMF Iteration {iter_num + 1}/{n_iterations} - Direction: {fb.upper()}")
                print(f"{'='*70}")
            
            # ================================================================
            # STEP 1: Compute coupling (first iter: random; later: via bridge)
            # ================================================================
            
            if iter_num == 0:
                # First iteration: random independent coupling
                if verbose:
                    print("  Strategy: Random independent coupling (initialization)")
                perm_idx = np.random.permutation(self.m_train)
                x_pairs_iter = np.stack([z0_base, z1_base[perm_idx]], axis=1)
            
            elif fb in self.drift_networks:
                # Use learned drift to resample coupling
                if verbose:
                    print("  Strategy: Resample coupling via learned bridge")
                
                x_pairs_iter = self._resample_coupling(
                    z0_base, z1_base, fb, verbose=verbose
                )
            
            else:
                # Fallback: random coupling
                if verbose:
                    print("  Strategy: Learned bridge not yet available, using random coupling")
                perm_idx = np.random.permutation(self.m_train)
                x_pairs_iter = np.stack([z0_base, z1_base[perm_idx]], axis=1)
            
            # ================================================================
            # STEP 2: Train drift network on current coupling
            # ================================================================
            
            xgb_net = XGBoostDriftNetwork(
                input_dim=self.n_features,
                n_timesteps=self.n_timesteps,
                xgb_params=self.xgb_params,
            )
            
            xgb_net.train(
                x_pairs_iter,
                n_timesteps=self.n_timesteps,
                sig=self.sig,
                eps=self.eps,
                fb=fb,
                verbose=verbose,
            )
            
            self.drift_networks[fb] = xgb_net
            
            # ================================================================
            # STEP 3: Evaluate and record history
            # ================================================================
            
            self.history['iteration'].append(iter_num + 1)
            self.history['direction'].append(fb)
            
            if verbose:
                print()
        
        if verbose:
            print("=" * 80)
            print("✓ Training complete!")
            print("=" * 80 + "\n")
        
        self.is_trained = True
        return self.history
    
    def _resample_coupling(
        self,
        z0_base: np.ndarray,
        z1_base: np.ndarray,
        fb: str,
        n_euler_steps: int = 50,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Use learned drift to find better coupling between z0 and z1.
        
        Strategy:
          1. Sample forward paths from z0_base (using forward network if available)
          2. Sample backward paths from z1_base (using backward network if available)
          3. Compute distances and find optimal matching
          4. Return new pairing
        
        Args:
            z0_base: Shape (m, n) - source samples
            z1_base: Shape (m, n) - target samples
            fb (str): 'f' or 'b' (which network to use)
            n_euler_steps (int): Euler steps for sampling
            verbose (bool): Print debug info
            
        Returns:
            np.ndarray: Shape (m, 2, n) - new paired samples
        """
        m, n = z0_base.shape
        
        if fb == "f":
            # Forward direction: use forward network if available
            if "f" in self.drift_networks:
                net = self.drift_networks["f"]
                traj_f = net.sample(z0_base, n_euler_steps=n_euler_steps, sig=self.sig, fb="f")
                zT_f = traj_f[-1]  # (m, n)
                
                # Match z0_base with zT_f using nearest neighbor
                from sklearn.metrics.pairwise import euclidean_distances
                dist = euclidean_distances(zT_f, z1_base)
                perm_idx = np.argmin(dist, axis=1)
                
                return np.stack([z0_base, z1_base[perm_idx]], axis=1)
            else:
                # Fallback
                perm_idx = np.random.permutation(m)
                return np.stack([z0_base, z1_base[perm_idx]], axis=1)
        
        else:  # fb == "b"
            # Backward direction: use backward network if available
            if "b" in self.drift_networks:
                net = self.drift_networks["b"]
                traj_b = net.sample(z1_base, n_euler_steps=n_euler_steps, sig=self.sig, fb="b")
                z0_b = traj_b[-1]  # (m, n)
                
                # Match z0_b with z0_base using nearest neighbor
                from sklearn.metrics.pairwise import euclidean_distances
                dist = euclidean_distances(z0_b, z0_base)
                perm_idx = np.argmin(dist, axis=1)
                
                return np.stack([z0_base[perm_idx], z1_base], axis=1)
            else:
                # Fallback
                perm_idx = np.random.permutation(m)
                return np.stack([z0_base, z1_base[perm_idx]], axis=1)
    
    # ================================================================
    # Generation
    # ================================================================
    
    def generate(
        self,
        n_samples: int,
        direction: str = "forward",
        n_euler_steps: int = 100,
    ) -> np.ndarray:
        """
        Generate synthetic samples using trained DSBM.
        
        Args:
            n_samples (int): Number of samples to generate
            direction (str): 'forward' (π₀ → π₁) or 'backward' (π₁ → π₀)
            n_euler_steps (int): Number of Euler steps
            
        Returns:
            np.ndarray: Generated samples, shape (n_samples, n_features)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before generating samples")
        
        assert direction in ("forward", "backward"), f"direction must be 'forward' or 'backward'"
        
        fb = "f" if direction == "forward" else "b"
        
        if fb not in self.drift_networks:
            raise RuntimeError(f"No trained network for direction {direction}")
        
        net = self.drift_networks[fb]
        
        # Choose starting distribution
        if direction == "forward":
            zstart = np.random.randn(n_samples, self.n_features).astype(np.float32)
        else:
            # Backward: sample from target
            idx = np.random.choice(self.m_train, n_samples, replace=True)
            zstart = self.x1_train[idx].copy()
        
        # Sample trajectory
        trajectory = net.sample(zstart, n_euler_steps=n_euler_steps, sig=self.sig, fb=fb)
        
        return trajectory[-1].astype(np.float32)
    
    # ================================================================
    # Properties for compatibility
    # ================================================================
    
    @property
    def traindata(self) -> np.ndarray:
        """For compatibility with older interfaces."""
        return self.x0_train
    
    @property
    def testdata(self) -> np.ndarray:
        """For compatibility with older interfaces."""
        return self.x1_train
    
    @property
    def D(self) -> int:
        """Dimension (number of features)."""
        return self.n_features

