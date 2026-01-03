"""
evaluation_metrics.py - REFACTORED

Comprehensive evaluation metrics library for synthetic tabular data.

KEY CHANGES FROM PREVIOUS VERSION:
  ✅ Input: Single datasets X_test (real), X_synth (synthetic)
     Both have shape: (n_samples, n_features + 1 target)
  ✅ Utility metrics: Automatically split into 80/20 for training
  ✅ XGBoost: Train on real/synth, test on real test set only
  ✅ Other metrics: Evaluated on WHOLE datasets (no split)
  ✅ Target column: Last column (index -1) separated automatically

Implements:
- Distribution metrics (KS, Wasserstein, KL, JS, MMD, SWD)
- Correlation metrics (Similarity, PCD, Distance)
- Fidelity & Diversity (Alpha-Precision, Beta-Recall)
- Utility metrics (R2, RMSE, ML Efficiency Gap) using XGBoost
- Privacy metrics (DCR, Identical Matches, Authenticity)
- Detection metrics (C2ST, Detection Score)

Usage:
    from evaluation_metrics import MetricsEvaluator
    
    # X_test shape: (n_samples, n_features + 1 target)
    # X_synth shape: (m_samples, n_features + 1 target)
    evaluator = MetricsEvaluator()
    metrics = evaluator.compute_all_metrics(X_test, X_synth)
"""

import numpy as np
from scipy.stats import entropy, ks_2samp
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import OneClassSVM
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# UTILITY FUNCTIONS FOR DATA HANDLING
# ============================================================================

def separate_features_and_target(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate features from target (last column).
    
    Args:
        X: Data array (n_samples, n_features + 1 target)
    
    Returns:
        Tuple of (features, target) where target is last column
    """
    return X[:, :-1], X[:, -1]


def split_train_test_80_20(X: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split dataset 80% train / 20% test.
    
    Args:
        X: Data array (n_samples, n_features + 1 target)
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_test)
    """
    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.2,
        random_state=random_state
    )
    return X[train_idx], X[test_idx]


# ============================================================================
# SECTION 1: DISTRIBUTION METRICS
# ============================================================================

def compute_ks_statistic(X_real: np.ndarray, X_synth: np.ndarray) -> Tuple[float, List[float]]:
    """
    Compute Kolmogorov-Smirnov (KS) statistic for each feature.
    
    Mathematical Definition:
        D = max_x |F_real(x) - F_synth(x)|
    
    Inputs: Full datasets (n_samples, n_features + 1 target)
    Process: Separates features, computes KS on features only
    
    Args:
        X_real: Real data (n_samples, n_features + 1 target)
        X_synth: Synthetic data (m_samples, n_features + 1 target)
    
    Returns:
        Tuple of (mean_ks, per_feature_ks)
    """
    X_real_feat, _ = separate_features_and_target(X_real)
    X_synth_feat, _ = separate_features_and_target(X_synth)
    
    n_features = min(X_real_feat.shape[1], X_synth_feat.shape[1])
    ks_values = []
    
    for col in range(n_features):
        try:
            stat, _ = ks_2samp(X_real_feat[:, col], X_synth_feat[:, col])
            ks_values.append(float(stat))
        except Exception:
            ks_values.append(np.nan)
    
    ks_mean = float(np.nanmean(ks_values))
    return ks_mean, ks_values


def compute_wasserstein_distance(X_real: np.ndarray, X_synth: np.ndarray, 
                                  n_samples: Optional[int] = None) -> float:
    """
    Compute 1-Wasserstein distance.
    
    Inputs: Full datasets (n_samples, n_features + 1 target)
    Process: Separates features, computes Wasserstein on features only
    """
    X_real_feat, _ = separate_features_and_target(X_real)
    X_synth_feat, _ = separate_features_and_target(X_synth)
    
    n_features = min(X_real_feat.shape[1], X_synth_feat.shape[1])
    if n_samples is None:
        n_samples = min(X_real_feat.shape[0], X_synth_feat.shape[0])
    
    wd_values = []
    
    for col in range(n_features):
        try:
            real_sorted = np.sort(X_real_feat[:n_samples, col])
            synth_sorted = np.sort(X_synth_feat[:n_samples, col])
            wd = np.mean(np.abs(real_sorted - synth_sorted))
            wd_values.append(float(wd))
        except Exception:
            wd_values.append(np.nan)
    
    return float(np.nanmean(wd_values))


def compute_kl_divergence(X_real: np.ndarray, X_synth: np.ndarray, 
                          bins: int = 10) -> float:
    """
    Compute KL divergence between real and synthetic distributions.
    
    Inputs: Full datasets (n_samples, n_features + 1 target)
    Process: Separates features, computes KL on features only
    """
    X_real_feat, _ = separate_features_and_target(X_real)
    X_synth_feat, _ = separate_features_and_target(X_synth)
    
    kl_values = []
    n_features = min(X_real_feat.shape[1], X_synth_feat.shape[1])
    
    for col in range(n_features):
        try:
            real_col = X_real_feat[:, col]
            synth_col = X_synth_feat[:, col]
            
            hist_real, bin_edges = np.histogram(real_col, bins=bins, density=True)
            hist_synth, _ = np.histogram(synth_col, bins=bin_edges, density=True)
            
            hist_real = hist_real / (hist_real.sum() + 1e-10)
            hist_synth = hist_synth / (hist_synth.sum() + 1e-10)
            
            kl = entropy(hist_real + 1e-10, hist_synth + 1e-10)
            kl_values.append(float(kl))
        except Exception:
            kl_values.append(np.nan)
    
    return float(np.nanmean(kl_values))


def compute_js_divergence(X_real: np.ndarray, X_synth: np.ndarray, 
                          bins: int = 10) -> float:
    """
    Compute Jensen-Shannon divergence (symmetric KL).
    
    Inputs: Full datasets (n_samples, n_features + 1 target)
    Process: Separates features, computes JS on features only
    """
    X_real_feat, _ = separate_features_and_target(X_real)
    X_synth_feat, _ = separate_features_and_target(X_synth)
    
    js_values = []
    n_features = min(X_real_feat.shape[1], X_synth_feat.shape[1])
    
    for col in range(n_features):
        try:
            real_col = X_real_feat[:, col]
            synth_col = X_synth_feat[:, col]
            
            hist_real, bin_edges = np.histogram(real_col, bins=bins, density=True)
            hist_synth, _ = np.histogram(synth_col, bins=bin_edges, density=True)
            
            hist_real = hist_real / (hist_real.sum() + 1e-10)
            hist_synth = hist_synth / (hist_synth.sum() + 1e-10)
            
            m = (hist_real + hist_synth) / 2.0
            js = 0.5 * entropy(hist_real + 1e-10, m + 1e-10) + \
                 0.5 * entropy(hist_synth + 1e-10, m + 1e-10)
            js_values.append(float(js))
        except Exception:
            js_values.append(np.nan)
    
    return float(np.nanmean(js_values))


def compute_mmd(X_real: np.ndarray, X_synth: np.ndarray, 
                kernel: str = 'rbf', sigma: float = 1.0) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD).
    
    Inputs: Full datasets (n_samples, n_features + 1 target)
    Process: Separates features, computes MMD on features only
    """
    X_real_feat, _ = separate_features_and_target(X_real)
    X_synth_feat, _ = separate_features_and_target(X_synth)
    
    try:
        if kernel == 'rbf':
            K_real = np.exp(-cdist(X_real_feat, X_real_feat, 'sqeuclidean') / (2 * sigma ** 2))
            K_synth = np.exp(-cdist(X_synth_feat, X_synth_feat, 'sqeuclidean') / (2 * sigma ** 2))
            K_cross = np.exp(-cdist(X_real_feat, X_synth_feat, 'sqeuclidean') / (2 * sigma ** 2))
        elif kernel == 'linear':
            K_real = linear_kernel(X_real_feat, X_real_feat)
            K_synth = linear_kernel(X_synth_feat, X_synth_feat)
            K_cross = linear_kernel(X_real_feat, X_synth_feat)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
        
        mmd_real = np.mean(K_real)
        mmd_synth = np.mean(K_synth)
        mmd_cross = np.mean(K_cross)
        mmd_squared = mmd_real + mmd_synth - 2 * mmd_cross
        
        return float(np.sqrt(np.maximum(mmd_squared, 0)))
    except Exception:
        return np.nan


def compute_swd(X_real: np.ndarray, X_synth: np.ndarray, 
                n_projections: int = 100, random_state: int = 42) -> float:
    """
    Compute Sliced Wasserstein Distance (SWD).
    
    Inputs: Full datasets (n_samples, n_features + 1 target)
    Process: Separates features, computes SWD on features only
    """
    X_real_feat, _ = separate_features_and_target(X_real)
    X_synth_feat, _ = separate_features_and_target(X_synth)
    
    try:
        n_features = X_real_feat.shape[1]
        n_samples = min(X_real_feat.shape[0], X_synth_feat.shape[0])
        
        rng = np.random.RandomState(random_state)
        swd_values = []
        
        for _ in range(n_projections):
            theta = rng.randn(n_features)
            theta = theta / np.linalg.norm(theta)
            
            real_proj = X_real_feat @ theta
            synth_proj = X_synth_feat @ theta
            
            real_proj_sorted = np.sort(real_proj[:n_samples])
            synth_proj_sorted = np.sort(synth_proj[:n_samples])
            
            wd = np.sqrt(np.mean((real_proj_sorted - synth_proj_sorted) ** 2))
            swd_values.append(wd)
        
        return float(np.mean(swd_values))
    except Exception:
        return np.nan


# ============================================================================
# SECTION 2: CORRELATION METRICS
# ============================================================================

def compute_correlation_similarity(X_real: np.ndarray, X_synth: np.ndarray) -> float:
    """
    Compute similarity of correlation matrices.
    
    Inputs: Full datasets (n_samples, n_features + 1 target)
    Process: Separates features, computes correlation on features only
    """
    X_real_feat, _ = separate_features_and_target(X_real)
    X_synth_feat, _ = separate_features_and_target(X_synth)
    
    try:
        corr_real = np.corrcoef(X_real_feat.T)
        corr_synth = np.corrcoef(X_synth_feat.T)
        
        corr_real = np.nan_to_num(corr_real, nan=0.0)
        corr_synth = np.nan_to_num(corr_synth, nan=0.0)
        
        diff = np.linalg.norm(corr_real - corr_synth, 'fro')
        max_diff = np.linalg.norm(np.ones_like(corr_real), 'fro')
        
        similarity = 1.0 - (diff / (max_diff + 1e-10))
        return float(np.clip(similarity, 0, 1))
    except Exception:
        return np.nan


def compute_pairwise_correlation_diff(X_real: np.ndarray, X_synth: np.ndarray) -> float:
    """
    Compute Pairwise Correlation Difference (PCD).
    
    Inputs: Full datasets (n_samples, n_features + 1 target)
    Process: Separates features, computes PCD on features only
    """
    X_real_feat, _ = separate_features_and_target(X_real)
    X_synth_feat, _ = separate_features_and_target(X_synth)
    
    try:
        corr_real = np.corrcoef(X_real_feat.T)
        corr_synth = np.corrcoef(X_synth_feat.T)
        
        corr_real = np.nan_to_num(corr_real, nan=0.0)
        corr_synth = np.nan_to_num(corr_synth, nan=0.0)
        
        pcd = np.linalg.norm(corr_real - corr_synth, 'fro')
        return float(pcd)
    except Exception:
        return np.nan


def compute_correlation_distance(X_real: np.ndarray, X_synth: np.ndarray) -> float:
    """
    Compute Frobenius norm distance between correlation matrices.
    Alias for compute_pairwise_correlation_diff.
    """
    return compute_pairwise_correlation_diff(X_real, X_synth)


# ============================================================================
# SECTION 3: FIDELITY & DIVERSITY (ALPHA-PRECISION / BETA-RECALL)
# ============================================================================

class OneClassEmbedding:
    """One-class SVM based embedding for alpha-precision/beta-recall metrics."""
    
    def __init__(self, nu: float = 0.05, kernel: str = 'rbf', gamma: str = 'auto'):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.svm = None
        self.center = None
        self.radius_quantiles = {}
    
    def fit(self, X: np.ndarray, alphas: np.ndarray = np.linspace(0.1, 0.95, 10)):
        """Fit one-class SVM and compute alpha-support radii."""
        try:
            self.svm = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
            self.svm.fit(X)
            
            scores = self.svm.decision_function(X)
            
            for alpha in alphas:
                quantile = np.percentile(scores, (1 - alpha) * 100)
                self.radius_quantiles[alpha] = quantile
            
            self.center = X.mean(axis=0)
        except Exception:
            pass
    
    def get_alpha_support_scores(self, X: np.ndarray) -> np.ndarray:
        """Get decision function scores."""
        if self.svm is None:
            raise ValueError("Must call fit() first")
        return self.svm.decision_function(X)
    
    def get_alpha_precision(self, X_synth: np.ndarray, alpha: float) -> float:
        """Compute alpha-precision: P(X_synth in S_alpha_real)"""
        try:
            scores = self.get_alpha_support_scores(X_synth)
            threshold = self.radius_quantiles.get(alpha, 0)
            precision = np.mean(scores >= threshold)
            return float(precision)
        except Exception:
            return np.nan


def compute_alpha_precision_and_beta_recall(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    alphas: np.ndarray = np.linspace(0.1, 0.95, 10)
) -> Tuple[float, float]:
    """
    Compute integrated alpha-precision and beta-recall.
    
    Inputs: Full datasets (n_samples, n_features + 1 target)
    Process: Separates features, computes metrics on features only
    """
    try:
        X_real_feat, _ = separate_features_and_target(X_real)
        X_synth_feat, _ = separate_features_and_target(X_synth)
        
        embedder = OneClassEmbedding(nu=0.05)
        embedder.fit(X_real_feat, alphas=alphas)
        
        precision_curve = []
        for alpha in alphas:
            p_alpha = embedder.get_alpha_precision(X_synth_feat, alpha)
            precision_curve.append(p_alpha)
        precision_curve = np.array(precision_curve)
        
        embedder_synth = OneClassEmbedding(nu=0.05)
        embedder_synth.fit(X_synth_feat, alphas=alphas)
        
        recall_curve = []
        for beta in alphas:
            r_beta = embedder_synth.get_alpha_precision(X_real_feat, beta)
            recall_curve.append(r_beta)
        recall_curve = np.array(recall_curve)
        
        ip_alpha = np.nanmean(np.abs(precision_curve - alphas))
        ir_beta = np.nanmean(np.abs(recall_curve - alphas))
        
        return float(ip_alpha), float(ir_beta)
    except Exception:
        return np.nan, np.nan


# ============================================================================
# SECTION 4: UTILITY METRICS (ML EFFICIENCY) - XGBOOST BASED
# ============================================================================

def compute_utility_metrics(X_real: np.ndarray, X_synth: np.ndarray) -> Dict[str, float]:
    """
    Compute utility metrics using XGBoost.
    
    KEY DIFFERENCES FROM PREVIOUS:
    1. Input: Full datasets (n_samples, n_features + 1 target)
    2. Automatically separates features from target (last column)
    3. Internally splits REAL data 80/20 for train/test
    4. Trains XGBoost on REAL train and SYNTHETIC train
    5. Tests on REAL test set only (synthetic test not used)
    6. Computes R², RMSE, ML Efficiency Gap
    
    Args:
        X_real: Real data (n_samples, n_features + 1 target)
        X_synth: Synthetic data (m_samples, n_features + 1 target)
    
    Returns:
        Dict with 'r2_score_real', 'r2_score_synth', 'rmse_real', 'rmse_synth', 'ml_efficiency_gap_percent'
    """
    try:
        import xgboost as xgb
        
        # Separate features and target from real data
        X_real_feat, y_real = separate_features_and_target(X_real)
        
        # Split REAL data 80/20 for training XGBoost
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real_feat, y_real, test_size=0.2, random_state=42
        )
        
        # Separate features and target from synthetic data
        X_synth_feat, y_synth = separate_features_and_target(X_synth)
        
        # Train XGBoost on REAL train data
        model_real = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        model_real.fit(X_train_real, y_train_real, verbose=False)
        y_pred_real = model_real.predict(X_test_real)
        
        # Train XGBoost on SYNTHETIC train data
        model_synth = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        model_synth.fit(X_synth_feat, y_synth, verbose=False)
        # Test on REAL test set (NOT synthetic test)
        y_pred_synth = model_synth.predict(X_test_real)
        
        # Compute metrics on REAL test set
        r2_real = r2_score(y_test_real, y_pred_real)
        rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        
        r2_synth = r2_score(y_test_real, y_pred_synth)
        rmse_synth = np.sqrt(mean_squared_error(y_test_real, y_pred_synth))
        
        # Compute ML Efficiency Gap
        if r2_real == 0 or np.isnan(r2_real):
            gap = 0.0
        else:
            gap = abs(r2_real - r2_synth) / abs(r2_real) * 100.0
        
        return {
            'r2_score_real': float(r2_real),
            'r2_score_synth': float(r2_synth),
            'rmse_real': float(rmse_real),
            'rmse_synth': float(rmse_synth),
            'ml_efficiency_gap_percent': float(gap)
        }
    
    except ImportError:
        return {
            'r2_score_real': np.nan,
            'r2_score_synth': np.nan,
            'rmse_real': np.nan,
            'rmse_synth': np.nan,
            'ml_efficiency_gap_percent': np.nan
        }
    except Exception as e:
        return {
            'r2_score_real': np.nan,
            'r2_score_synth': np.nan,
            'rmse_real': np.nan,
            'rmse_synth': np.nan,
            'ml_efficiency_gap_percent': np.nan
        }


# ============================================================================
# SECTION 5: PRIVACY METRICS
# ============================================================================

def compute_distance_to_closest_record(X_synth: np.ndarray, X_real: np.ndarray) -> Tuple[float, float]:
    """
    Compute DCR: Distance to Closest Record.
    
    Inputs: Full datasets (n_samples, n_features + 1 target)
    Process: Uses FULL data (features + target) for distance computation
    """
    try:
        dcr_values = []
        
        for synth_sample in X_synth:
            distances = np.linalg.norm(X_real - synth_sample, axis=1)
            min_distance = np.min(distances)
            dcr_values.append(min_distance)
        
        dcr_values = np.array(dcr_values)
        return float(np.mean(dcr_values)), float(np.median(dcr_values))
    except Exception:
        return np.nan, np.nan


def compute_dcr_share(X_synth: np.ndarray, X_real: np.ndarray) -> float:
    """
    Compute DCR Share: Fraction of synthetic samples closer to real than to test.
    
    NOTE: With input X_real and X_synth only, this computes:
    Fraction closer to first half than second half of X_real
    """
    try:
        # Split X_real into two halves (pseudo train/test)
        split_point = len(X_real) // 2
        X_real_part1 = X_real[:split_point]
        X_real_part2 = X_real[split_point:]
        
        count_closer_to_part1 = 0
        
        for synth_sample in X_synth:
            dist_to_part1 = np.min(np.linalg.norm(X_real_part1 - synth_sample, axis=1))
            dist_to_part2 = np.min(np.linalg.norm(X_real_part2 - synth_sample, axis=1))
            
            if dist_to_part1 < dist_to_part2:
                count_closer_to_part1 += 1
        
        share = count_closer_to_part1 / len(X_synth)
        return float(share)
    except Exception:
        return np.nan


def compute_identical_matches(X_synth: np.ndarray, X_real: np.ndarray, 
                              tol: float = 1e-6) -> float:
    """
    Compute fraction of synthetic samples that are exact copies of real data.
    
    Inputs: Full datasets (n_samples, n_features + 1 target)
    Process: Uses FULL data for matching
    """
    try:
        identical_count = 0
        
        for synth_sample in X_synth:
            distances = np.linalg.norm(X_real - synth_sample, axis=1)
            if np.min(distances) < tol:
                identical_count += 1
        
        return float(identical_count / len(X_synth))
    except Exception:
        return np.nan


def compute_authenticity(X_synth: np.ndarray, X_real: np.ndarray) -> float:
    """
    Compute authenticity score: Fraction of synthetic samples NOT copied.
    
    Inputs: Full datasets (n_samples, n_features + 1 target)
    Process: Uses FULL data for computation
    """
    try:
        authentic_count = 0
        
        for synth_sample in X_synth:
            distances_to_real = np.linalg.norm(X_real - synth_sample, axis=1)
            closest_real_idx = np.argmin(distances_to_real)
            d_synth_to_real = distances_to_real[closest_real_idx]
            
            closest_real = X_real[closest_real_idx]
            other_real = np.delete(X_real, closest_real_idx, axis=0)
            
            if len(other_real) > 0:
                distances_within_real = np.linalg.norm(other_real - closest_real, axis=1)
                d_real_to_neighbor = np.min(distances_within_real)
            else:
                d_real_to_neighbor = 1.0
            
            if d_synth_to_real > d_real_to_neighbor:
                authentic_count += 1
        
        return float(authentic_count / len(X_synth))
    except Exception:
        return np.nan


# ============================================================================
# SECTION 6: DETECTION METRICS
# ============================================================================

def compute_c2st_score(X_real: np.ndarray,
                       X_synth: np.ndarray,
                       classifier_type: str = 'rf',
                       cv: int = 5,
                       verbose: bool = False) -> Dict[str, float]:
    """
    Compute Classifier Two-Sample Test (C2ST) score.
    
    Inputs: Full datasets (n_samples, n_features + 1 target)
    Process: Uses FULL data (features + target) for classification
    """
    try:
        from sklearn.metrics import roc_auc_score
        
        # Use FULL data (features + target)
        X_combined = np.vstack([X_real, X_synth])
        y_labels = np.hstack([np.zeros(len(X_real)), np.ones(len(X_synth))])
        
        if classifier_type == 'rf':
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(random_state=42, max_iter=1000)
        
        cv_scores = cross_val_score(clf, X_combined, y_labels, cv=cv, scoring='accuracy')
        accuracy = float(np.mean(cv_scores))
        
        clf.fit(X_combined, y_labels)
        y_proba = clf.predict_proba(X_combined)[:, 1]
        auc = float(roc_auc_score(y_labels, y_proba))
        
        # Simple permutation p-value
        n_permutations = 50
        perm_accuracies = []
        for _ in range(n_permutations):
            y_perm = np.random.permutation(y_labels)
            perm_score = cross_val_score(clf, X_combined, y_perm, cv=cv, scoring='accuracy')
            perm_accuracies.append(np.mean(perm_score))
        
        p_value = float(np.mean(np.array(perm_accuracies) >= accuracy))
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'p_value': p_value,
        }
    
    except Exception as e:
        if verbose:
            warnings.warn(f"C2ST computation failed: {str(e)}")
        return {'accuracy': np.nan, 'auc': np.nan, 'p_value': np.nan}


# ============================================================================
# SECTION 7: METRIC AGGREGATOR (ORCHESTRATOR)
# ============================================================================

class MetricsEvaluator:
    """
    Centralized evaluator for all synthetic data quality metrics.
    
    REFACTORED FOR NEW INPUT FORMAT:
    ✅ Input: X_test (real), X_synth (synthetic)
       Both shape: (n_samples, n_features + 1 target)
    ✅ Automatically separates features and target
    ✅ Distribution/Correlation/Privacy/Detection: Use FULL data (no split)
    ✅ Utility metrics: Internally split real data 80/20
    ✅ Test on real test set only
    
    Usage:
        evaluator = MetricsEvaluator()
        metrics = evaluator.compute_all_metrics(X_test, X_synth)
    """
    
    def __init__(self,
                 include_alpha_precision: bool = False,
                 include_c2st: bool = False,
                 include_authenticity: bool = True,
                 c2st_classifier_type: str = 'rf',
                 verbose: bool = False):
        """
        Args:
            include_alpha_precision: Compute expensive alpha-precision/beta-recall
            include_c2st: Compute C2ST detection metric
            include_authenticity: Compute authenticity/memorization metric
            c2st_classifier_type: 'rf' or 'lr'
            verbose: Print warnings
        """
        self.include_alpha_precision = include_alpha_precision
        self.include_c2st = include_c2st
        self.include_authenticity = include_authenticity
        self.c2st_classifier_type = c2st_classifier_type
        self.verbose = verbose
    
    def compute_all_metrics(self,
                            X_test: np.ndarray,
                            X_synth: np.ndarray) -> Dict[str, float]:
        """
        Compute all enabled metrics.
        
        REFACTORED INPUTS:
        Args:
            X_test: Real test data (n_samples, n_features + 1 target)
            X_synth: Synthetic data (m_samples, n_features + 1 target)
        
        Returns:
            Dict with 18+ metrics
        """
        metrics = {}
        
        try:
            # -------- DISTRIBUTION METRICS (use full data) --------
            ks_mean, ks_per_feature = compute_ks_statistic(X_test, X_synth)
            metrics['ks_statistic_mean'] = ks_mean
            metrics['ks_statistic_per_feature_mean'] = float(np.nanmean(ks_per_feature))
            
            metrics['wasserstein_distance'] = compute_wasserstein_distance(X_test, X_synth)
            metrics['kl_divergence'] = compute_kl_divergence(X_test, X_synth)
            metrics['js_divergence'] = compute_js_divergence(X_test, X_synth)
            metrics['mmd'] = compute_mmd(X_test, X_synth)
            metrics['swd'] = compute_swd(X_test, X_synth)
            
            # -------- CORRELATION METRICS (use full data) --------
            metrics['correlation_similarity'] = compute_correlation_similarity(X_test, X_synth)
            metrics['pairwise_correlation_diff'] = compute_pairwise_correlation_diff(X_test, X_synth)
            metrics['correlation_distance'] = compute_correlation_distance(X_test, X_synth)
            
            # -------- FIDELITY & DIVERSITY (optional, expensive) --------
            if self.include_alpha_precision:
                ip_alpha, ir_beta = compute_alpha_precision_and_beta_recall(X_test, X_synth)
                metrics['ip_alpha'] = ip_alpha
                metrics['ir_beta'] = ir_beta
            
            # -------- UTILITY METRICS (XGBoost, internally split real 80/20) --------
            utility_metrics = compute_utility_metrics(X_test, X_synth)
            metrics.update(utility_metrics)
            
            # -------- PRIVACY METRICS (use full data) --------
            dcr_mean, dcr_median = compute_distance_to_closest_record(X_synth, X_test)
            metrics['dcr_mean'] = dcr_mean
            metrics['dcr_median'] = dcr_median
            metrics['dcr_share'] = compute_dcr_share(X_synth, X_test)
            metrics['identical_matches_fraction'] = compute_identical_matches(X_synth, X_test)
            
            # -------- DETECTION METRICS --------
            if self.include_c2st:
                c2st_results = compute_c2st_score(X_test, X_synth, 
                                                  classifier_type=self.c2st_classifier_type)
                metrics.update({'c2st_' + k: v for k, v in c2st_results.items()})
            
            if self.include_authenticity:
                metrics['authenticity'] = compute_authenticity(X_synth, X_test)
        
        except Exception as e:
            if self.verbose:
                warnings.warn(f"Error computing metrics: {str(e)}")
        
        return metrics
    
    def get_enabled_metrics(self) -> List[str]:
        """Return list of metrics being computed."""
        metrics = [
            'ks_statistic_mean',
            'ks_statistic_per_feature_mean',
            'wasserstein_distance',
            'kl_divergence',
            'js_divergence',
            'mmd',
            'swd',
            'correlation_similarity',
            'pairwise_correlation_diff',
            'correlation_distance',
            'r2_score_real',
            'r2_score_synth',
            'rmse_real',
            'rmse_synth',
            'ml_efficiency_gap_percent',
            'dcr_mean',
            'dcr_median',
            'dcr_share',
            'identical_matches_fraction',
            'authenticity',
        ]
        
        if self.include_alpha_precision:
            metrics.extend(['ip_alpha', 'ir_beta'])
        if self.include_c2st:
            metrics.extend(['c2st_accuracy', 'c2st_auc', 'c2st_p_value'])
        
        return metrics


if __name__ == "__main__":
    print("✅ Metrics library loaded successfully")
    evaluator = MetricsEvaluator()
    print(f"Available metrics: {len(evaluator.get_enabled_metrics())}")
    print(f"Metrics: {evaluator.get_enabled_metrics()}")
