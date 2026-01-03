#!/usr/bin/env python3
"""
Comprehensive metrics for synthetic data evaluation.

Metrics computed:
1. MMD (Maximum Mean Discrepancy) - kernel-based distribution distance
2. KL Divergence (Kullback-Leibler) - statistical divergence
3. Wasserstein Distance - optimal transport distance
4. Frobenius Norm - correlation matrix difference
5. Marginal KL/JS - per-feature KL and Jensen-Shannon divergence
6. Histogram-based KL/JS - per-feature histogram comparison
7. Utility Metrics:
   - R² (Coefficient of Determination)
   - RMSE (Root Mean Squared Error)
   - using CatBoost model trained on real data

All metrics accept: data_real (n_samples, n_features), data_synthetic (n_samples, n_features)
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not installed. Utility metrics will not be computed.")


# ============================================================================
# 1. MAXIMUM MEAN DISCREPANCY (MMD)
# ============================================================================

def compute_mmd(X_real: np.ndarray, X_synth: np.ndarray, kernel: str = 'rbf', 
                sigma: float = 1.0) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between real and synthetic data.
    
    MMD measures distance between two distributions using kernel methods.
    Lower MMD = more similar distributions.
    
    Formula: MMD² = E[K(x,x')] + E[K(y,y')] - 2*E[K(x,y)]
    
    Args:
        X_real: Real data (n_samples, n_features)
        X_synth: Synthetic data (n_samples, n_features)
        kernel: 'rbf' or 'linear'
        sigma: Bandwidth for RBF kernel
    
    Returns:
        MMD value (float, non-negative)
    """
    
    # Compute kernel matrices
    if kernel == 'rbf':
        # RBF kernel: K(x, y) = exp(-||x-y||^2 / (2*sigma^2))
        K_real = np.exp(-cdist(X_real, X_real, 'sqeuclidean') / (2 * sigma ** 2))
        K_synth = np.exp(-cdist(X_synth, X_synth, 'sqeuclidean') / (2 * sigma ** 2))
        K_cross = np.exp(-cdist(X_real, X_synth, 'sqeuclidean') / (2 * sigma ** 2))
    
    elif kernel == 'linear':
        # Linear kernel: K(x, y) = x · y
        K_real = linear_kernel(X_real, X_real)
        K_synth = linear_kernel(X_synth, X_synth)
        K_cross = linear_kernel(X_real, X_synth)
    
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    # MMD^2 = E[K(x,x')] + E[K(y,y')] - 2*E[K(x,y)]
    mmd_real = np.mean(K_real)
    mmd_synth = np.mean(K_synth)
    mmd_cross = np.mean(K_cross)
    
    mmd_squared = mmd_real + mmd_synth - 2 * mmd_cross
    mmd = np.sqrt(np.maximum(mmd_squared, 0))
    
    return mmd


# ============================================================================
# 2. KULLBACK-LEIBLER DIVERGENCE (KL)
# ============================================================================

def compute_kl_divergence(X_real: np.ndarray, X_synth: np.ndarray, 
                         n_bins: int = 50) -> float:
    """
    Compute KL divergence between real and synthetic data using histograms.
    
    KL(P||Q) = Σ P(x) * log(P(x) / Q(x))
    Lower KL = more similar distributions.
    
    Args:
        X_real: Real data (n_samples, n_features)
        X_synth: Synthetic data (n_samples, n_features)
        n_bins: Number of bins for histogram
    
    Returns:
        Average KL divergence across features
    """
    
    n_features = X_real.shape[1]
    kl_divs = []
    
    for feat_idx in range(n_features):
        real_feat = X_real[:, feat_idx]
        synth_feat = X_synth[:, feat_idx]
        
        # Create combined range for consistent binning
        min_val = min(real_feat.min(), synth_feat.min())
        max_val = max(real_feat.max(), synth_feat.max())
        bins = np.linspace(min_val, max_val, n_bins)
        
        # Compute histograms
        p, _ = np.histogram(real_feat, bins=bins)
        q, _ = np.histogram(synth_feat, bins=bins)
        
        # Normalize to probabilities
        p = p / (p.sum() + 1e-10)
        q = q / (q.sum() + 1e-10)
        
        # Add small epsilon to avoid log(0)
        p = p + 1e-10
        q = q + 1e-10
        
        # KL divergence
        kl = np.sum(p * np.log(p / q))
        kl_divs.append(kl)
    
    return np.mean(kl_divs)


# ============================================================================
# 3. WASSERSTEIN DISTANCE
# ============================================================================

def compute_wasserstein_distance(X_real: np.ndarray, X_synth: np.ndarray) -> float:
    """
    Compute Wasserstein distance (optimal transport) between distributions.
    
    Computes 1D Wasserstein distance for each feature and averages.
    Lower Wasserstein = more similar distributions.
    
    Args:
        X_real: Real data (n_samples, n_features)
        X_synth: Synthetic data (n_samples, n_features)
    
    Returns:
        Average Wasserstein distance across features
    """
    
    n_features = X_real.shape[1]
    n_samples = min(X_real.shape[0], X_synth.shape[0])
    
    wasserstein_dists = []
    
    for feat_idx in range(n_features):
        real_feat = np.sort(X_real[:n_samples, feat_idx])
        synth_feat = np.sort(X_synth[:n_samples, feat_idx])
        
        # 1D Wasserstein distance
        wd = wasserstein_distance(real_feat, synth_feat)
        wasserstein_dists.append(wd)
    
    return np.mean(wasserstein_dists)


# ============================================================================
# 4. FROBENIUS NORM OF CORRELATION MATRICES
# ============================================================================

def compute_correlation_frobenius_norm(X_real: np.ndarray, X_synth: np.ndarray) -> float:
    """
    Compute Frobenius norm of difference between correlation matrices.
    
    Measures how well the synthetic data preserves correlations.
    Lower Frobenius = better correlation preservation.
    
    Formula: ||Corr(real) - Corr(synth)||_F
    
    Args:
        X_real: Real data (n_samples, n_features)
        X_synth: Synthetic data (n_samples, n_features)
    
    Returns:
        Frobenius norm (float)
    """
    
    # Compute correlation matrices
    corr_real = np.corrcoef(X_real.T)
    corr_synth = np.corrcoef(X_synth.T)
    
    # Handle NaN values (can occur with constant features)
    corr_real = np.nan_to_num(corr_real, nan=0.0)
    corr_synth = np.nan_to_num(corr_synth, nan=0.0)
    
    # Frobenius norm
    frobenius = np.linalg.norm(corr_real - corr_synth, 'fro')
    
    return frobenius


# ============================================================================
# 5. MARGINAL KL/JS DIVERGENCE (PER-FEATURE)
# ============================================================================

def compute_marginal_kl_js(X_real: np.ndarray, X_synth: np.ndarray, 
                          n_bins: int = 50) -> tuple:
    """
    Compute per-feature KL and Jensen-Shannon divergence (histogram-based).
    
    Args:
        X_real: Real data (n_samples, n_features)
        X_synth: Synthetic data (n_samples, n_features)
        n_bins: Number of bins for histograms
    
    Returns:
        Tuple of (mean_kl, mean_js, kl_per_feature, js_per_feature)
    """
    
    n_features = X_real.shape[1]
    kl_per_feature = []
    js_per_feature = []
    
    for feat_idx in range(n_features):
        real_feat = X_real[:, feat_idx]
        synth_feat = X_synth[:, feat_idx]
        
        # Create combined range for consistent binning
        min_val = min(real_feat.min(), synth_feat.min())
        max_val = max(real_feat.max(), synth_feat.max())
        bins = np.linspace(min_val, max_val, n_bins)
        
        # Compute histograms
        p, _ = np.histogram(real_feat, bins=bins)
        q, _ = np.histogram(synth_feat, bins=bins)
        
        # Normalize to probabilities
        p = p / (p.sum() + 1e-10)
        q = q / (q.sum() + 1e-10)
        
        # Add small epsilon to avoid log(0)
        p = p + 1e-10
        q = q + 1e-10
        
        # KL divergence: KL(P||Q)
        kl = np.sum(p * np.log(p / q))
        kl_per_feature.append(kl)
        
        # Jensen-Shannon divergence: JS(P||Q)
        m = 0.5 * (p + q)
        js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
        js = np.sqrt(js)  # Symmetric form
        js_per_feature.append(js)
    
    mean_kl = np.mean(kl_per_feature)
    mean_js = np.mean(js_per_feature)
    
    return mean_kl, mean_js, np.array(kl_per_feature), np.array(js_per_feature)


# ============================================================================
# 6. CATBOOST UTILITY METRICS (R², RMSE)
# ============================================================================

def compute_catboost_utility_metrics(X_real: np.ndarray, y_real: np.ndarray,
                                     X_synth: np.ndarray, y_synth: np.ndarray,
                                     test_size: float = 0.2,
                                     random_state: int = 42) -> dict:
    """
    Train CatBoost on real data, evaluate on synthetic data.
    Compute R² and RMSE as utility metrics.
    
    Higher R² = better utility (closer to 1.0 is perfect)
    Lower RMSE = better utility
    
    Args:
        X_real: Real features (n_samples, n_features)
        y_real: Real target (n_samples,)
        X_synth: Synthetic features (n_samples, n_features)
        y_synth: Synthetic target (n_samples,)
        test_size: Fraction for test set
        random_state: Random seed
    
    Returns:
        Dictionary with 'r2', 'rmse', 'mse'
    """
    
    if not CATBOOST_AVAILABLE:
        return {'r2': None, 'rmse': None, 'mse': None, 'error': 'CatBoost not installed'}
    
    try:
        # Split real data for training
        X_train, X_test, y_train, y_test = train_test_split(
            X_real, y_real, test_size=test_size, random_state=random_state
        )
        
        # Train CatBoost on real data
        model = CatBoostRegressor(
            iterations=100,
            verbose=False,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Evaluate on synthetic data
        y_synth_pred = model.predict(X_synth)
        
        # Compute metrics
        r2 = r2_score(y_synth, y_synth_pred)
        rmse = np.sqrt(mean_squared_error(y_synth, y_synth_pred))
        mse = mean_squared_error(y_synth, y_synth_pred)
        
        return {
            'r2': float(r2),
            'rmse': float(rmse),
            'mse': float(mse),
        }
    
    except Exception as e:
        return {'r2': None, 'rmse': None, 'mse': None, 'error': str(e)}


# ============================================================================
# 7. COMPREHENSIVE EVALUATION FUNCTION
# ============================================================================

def evaluate_synthetic_data(X_real: np.ndarray, X_synth: np.ndarray,
                           y_real: np.ndarray = None, y_synth: np.ndarray = None,
                           compute_utility: bool = True,
                           verbose: bool = True) -> dict:
    """
    Comprehensive evaluation of synthetic data quality.
    
    Computes all metrics and returns dictionary with results.
    
    Args:
        X_real: Real features (n_samples, n_features)
        X_synth: Synthetic features (n_samples, n_features)
        y_real: Real target (n_samples,) - optional for utility metrics
        y_synth: Synthetic target (n_samples,) - optional for utility metrics
        compute_utility: Whether to compute CatBoost utility metrics
        verbose: Print results
    
    Returns:
        Dictionary with all computed metrics
    """
    
    # Ensure same shape
    n_samples = min(X_real.shape[0], X_synth.shape[0])
    X_real = X_real[:n_samples].astype(np.float32)
    X_synth = X_synth[:n_samples].astype(np.float32)
    
    results = {}
    
    # 1. MMD
    if verbose:
        print("Computing MMD...", end=" ")
    mmd = compute_mmd(X_real, X_synth, kernel='rbf', sigma=1.0)
    results['mmd'] = float(mmd)
    if verbose:
        print(f"✓ {mmd:.6f}")
    
    # 2. KL Divergence
    if verbose:
        print("Computing KL Divergence...", end=" ")
    kl = compute_kl_divergence(X_real, X_synth, n_bins=50)
    results['kl_divergence'] = float(kl)
    if verbose:
        print(f"✓ {kl:.6f}")
    
    # 3. Wasserstein Distance
    if verbose:
        print("Computing Wasserstein Distance...", end=" ")
    wd = compute_wasserstein_distance(X_real, X_synth)
    results['wasserstein_distance'] = float(wd)
    if verbose:
        print(f"✓ {wd:.6f}")
    
    # 4. Frobenius Norm
    if verbose:
        print("Computing Frobenius Norm (Correlation)...", end=" ")
    frob = compute_correlation_frobenius_norm(X_real, X_synth)
    results['frobenius_norm'] = float(frob)
    if verbose:
        print(f"✓ {frob:.6f}")
    
    # 5. Marginal KL/JS
    if verbose:
        print("Computing Marginal KL/JS...", end=" ")
    mean_kl, mean_js, kl_per_feat, js_per_feat = compute_marginal_kl_js(
        X_real, X_synth, n_bins=50
    )
    results['marginal_kl_mean'] = float(mean_kl)
    results['marginal_js_mean'] = float(mean_js)
    results['marginal_kl_per_feature'] = kl_per_feat.tolist()
    results['marginal_js_per_feature'] = js_per_feat.tolist()
    if verbose:
        print(f"✓ KL={mean_kl:.6f}, JS={mean_js:.6f}")
    
    # 6. CatBoost Utility Metrics
    if compute_utility and y_real is not None and y_synth is not None:
        if verbose:
            print("Computing CatBoost Utility Metrics...", end=" ")
        
        # Ensure same target size
        n_samples_y = min(len(y_real), len(y_synth))
        y_real = y_real[:n_samples_y].astype(np.float32)
        y_synth = y_synth[:n_samples_y].astype(np.float32)
        
        utility = compute_catboost_utility_metrics(X_real, y_real, X_synth, y_synth)
        results['utility_r2'] = utility.get('r2')
        results['utility_rmse'] = utility.get('rmse')
        results['utility_mse'] = utility.get('mse')
        
        if verbose:
            if utility.get('r2') is not None:
                print(f"✓ R²={utility['r2']:.6f}, RMSE={utility['rmse']:.6f}")
            else:
                print(f"⚠️  {utility.get('error', 'Unknown error')}")
    else:
        results['utility_r2'] = None
        results['utility_rmse'] = None
        results['utility_mse'] = None
    
    return results


# ============================================================================
# 8. FORMATTED REPORT
# ============================================================================

def print_evaluation_report(results: dict, dataset_name: str = "Synthetic Data"):
    """Print formatted evaluation report."""
    
    print("\n" + "="*80)
    print(f"SYNTHETIC DATA EVALUATION REPORT: {dataset_name}")
    print("="*80 + "\n")
    
    print("Distribution Similarity Metrics:")
    print(f"  MMD (Maximum Mean Discrepancy):        {results['mmd']:.6f}")
    print(f"  KL Divergence (Kullback-Leibler):     {results['kl_divergence']:.6f}")
    print(f"  Wasserstein Distance:                  {results['wasserstein_distance']:.6f}")
    print(f"  Frobenius Norm (Correlation Matrix):   {results['frobenius_norm']:.6f}")
    print(f"  Marginal KL (Mean):                    {results['marginal_kl_mean']:.6f}")
    print(f"  Marginal JS (Mean):                    {results['marginal_js_mean']:.6f}")
    
    print("\nUtility Metrics (CatBoost):")
    if results['utility_r2'] is not None:
        print(f"  R² Score:                              {results['utility_r2']:.6f}")
        print(f"  RMSE:                                  {results['utility_rmse']:.6f}")
        print(f"  MSE:                                   {results['utility_mse']:.6f}")
    else:
        print(f"  (Not computed)")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    """
    Example usage and testing
    """
    
    print("\n" + "="*80)
    print("SYNTHETIC DATA EVALUATION METRICS - EXAMPLE")
    print("="*80 + "\n")
    
    # Generate example data
    np.random.seed(42)
    
    # Real data: from a specific distribution
    n_samples = 1000
    n_features = 5
    X_real = np.random.randn(n_samples, n_features)
    y_real = X_real[:, 0] + 0.5 * X_real[:, 1] + np.random.randn(n_samples) * 0.1
    
    # Synthetic data: slightly different distribution
    X_synth = np.random.randn(n_samples, n_features) + 0.1
    y_synth = X_synth[:, 0] + 0.5 * X_synth[:, 1] + np.random.randn(n_samples) * 0.1
    
    print(f"Data shapes:")
    print(f"  X_real:  {X_real.shape}")
    print(f"  X_synth: {X_synth.shape}")
    print(f"  y_real:  {y_real.shape}")
    print(f"  y_synth: {y_synth.shape}\n")
    
    # Evaluate
    results = evaluate_synthetic_data(
        X_real, X_synth, y_real, y_synth,
        compute_utility=True,
        verbose=True
    )
    
    # Print report
    print_evaluation_report(results, dataset_name="Example Synthetic Dataset")
    
    # Print detailed per-feature results
    print("Per-Feature Marginal Divergences:")
    for feat_idx, (kl, js) in enumerate(zip(results['marginal_kl_per_feature'], 
                                             results['marginal_js_per_feature'])):
        print(f"  Feature {feat_idx}: KL={kl:.6f}, JS={js:.6f}")
    
    print("\n✅ Evaluation complete!")
