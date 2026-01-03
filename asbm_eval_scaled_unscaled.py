#!/usr/bin/env python3
"""
ASBM Complete Evaluation Pipeline (SCALED + UNSCALED UTILITY METRICS)

Workflow:
1. Load dataset (features + target as last column)
2. Split: (X_train, y_train) | (X_test, y_test)
3. Standardize (train set fits scaler)
4. Train ASBM on SCALED full data
5. Generate FULL synthetic data (scaled)
6. Evaluate utility metrics on BOTH scaled and unscaled data:
   - SCALED: All data scaled by StandardScaler
   - UNSCALED: Inverse-transform synthetic back to original scale

Output:
- asbm_evaluation_results.json (15 utility metrics: 6 scaled + 6 unscaled + 3 distribution)
- asbm_evaluation_plots/ (12 PNG files):
  * 3 scaled utility metrics (R², RMSE, MAE on scaled data)
  * 3 unscaled utility metrics (R², RMSE, MAE on unscaled data)
  * 6 distribution metrics
  * 3x3 grid
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# Import ASBM and metrics
try:
    from asbm_tabular_bridge import ASBMTabularBridge
except ImportError:
    print("❌ Error: asbm_tabular_bridge.py not found")
    exit(1)

try:
    from synthetic_data_metrics import evaluate_synthetic_data
except ImportError:
    print("❌ Error: synthetic_data_metrics.py not found")
    exit(1)


# ============================================================================
# 1. LOAD AND PARSE OPTUNA SUMMARY
# ============================================================================

def load_optuna_summary(summary_file: str = "optuna_results_asbm/SUMMARY.json") -> Dict:
    """Load best hyperparameters from Optuna SUMMARY.json."""
    summary_path = Path(summary_file)
    if not summary_path.exists():
        raise FileNotFoundError(f"SUMMARY.json not found: {summary_file}")
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print(f"✓ Loaded Optuna SUMMARY: {summary_file}")
    print(f"  Found {len(summary)} datasets\n")
    return summary


# ============================================================================
# 2. LOAD AND PREPARE DATASET
# ============================================================================

def load_and_prepare_dataset(
    pkl_file: str,
    dataset_name: str,
    train_size: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, StandardScaler]:
    """
    Load dataset and prepare full data (features + target).
    
    Returns: (X_train_full_scaled, X_test_full_scaled, X_train_full_unscaled, X_test_full_unscaled, n_features, scaler)
    """
    pkl_path = Path(pkl_file)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pkl_file}")
    
    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    if dataset_name not in raw_data:
        raise ValueError(f"Dataset '{dataset_name}' not found in pickle")
    
    # Convert to numpy
    X_full = np.asarray(raw_data[dataset_name], dtype=np.float32)
    if hasattr(raw_data[dataset_name], 'to_numpy'):
        X_full = raw_data[dataset_name].to_numpy().astype(np.float32)
    
    # Train/test split on FULL data (BEFORE scaling)
    X_train_full_unscaled, X_test_full_unscaled = train_test_split(
        X_full, test_size=1.0 - train_size, random_state=42
    )
    
    # Standardize (fit on train, apply to both)
    scaler = StandardScaler().fit(X_train_full_unscaled)
    X_train_full_scaled = scaler.transform(X_train_full_unscaled).astype(np.float32)
    X_test_full_scaled = scaler.transform(X_test_full_unscaled).astype(np.float32)
    
    n_features = X_train_full_scaled.shape[1] - 1  # Exclude target column
    return X_train_full_scaled, X_test_full_scaled, X_train_full_unscaled, X_test_full_unscaled, n_features, scaler


# ============================================================================
# 3. CREATE AND TRAIN ASBM BRIDGE (ON FULL DATA)
# ============================================================================

def create_and_train_asbm_full(
    X_train_full: np.ndarray,
    X_test_full: np.ndarray,
    best_params: Dict[str, Any],
    dataset_name: str,
    verbose: bool = True
) -> ASBMTabularBridge:
    """Create ASBMTabularBridge and train on FULL data (features + target)."""
    m_train, n_cols = X_train_full.shape
    m_test = X_test_full.shape[0]
    
    if verbose:
        print(f"✓ Creating ASBM for {dataset_name} (FULL data: features + target)")
        print(f"  X_train shape: {X_train_full.shape}")
    
    # Create x0 (random noise source)
    rng = np.random.RandomState(42)
    x0_train = rng.randn(m_train, n_cols).astype(np.float32)
    x1_train = X_train_full.astype(np.float32)
    
    x0_test = rng.randn(m_test, n_cols).astype(np.float32)
    x1_test = X_test_full.astype(np.float32)
    
    # Initialize ASBM with best params
    bridge = ASBMTabularBridge(
        x0_train=x0_train,
        x1_train=x1_train,
        x0_test=x0_test,
        x1_test=x1_test,
        num_timesteps=best_params['num_timesteps'],
        epsilon=0.1,
    )
    
    batch_size = min(256, m_train)
    
    if verbose:
        print(f"✓ Training ASBM (timesteps={best_params['num_timesteps']})...")
    
    # Train with best params
    history = bridge.fit(
        imf_iters=5,
        inner_iters=3000,
        batch_size=batch_size,
        lr_g=best_params['lr_g'],
        lr_d=best_params['lr_d'],
        layers_G_fw=best_params['layers_G'],
        layers_G_bw=best_params['layers_G'],
        layers_D_fw=best_params['layers_D'],
        layers_D_bw=best_params['layers_D'],
        plot_losses=True,
        save_dir="./GAN_losses_real/"+dataset_name,
        verbose=False,
    )
    
    return bridge


# ============================================================================
# 4. COMPUTE CATBOOST UTILITY METRICS (SCALED + UNSCALED)
# ============================================================================

def compute_catboost_utility_metrics(
    bridge: ASBMTabularBridge,
    X_train_full_scaled: np.ndarray,
    X_test_full_scaled: np.ndarray,
    X_train_full_unscaled: np.ndarray,
    X_test_full_unscaled: np.ndarray,
    scaler: StandardScaler,
    dataset_name: str,
    verbose: bool = True
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluate utility metrics on BOTH scaled and unscaled data.
    
    Workflow:
    1. Generate X_synth_full from bridge (on scaled data)
    2. Evaluate on SCALED data (both data in StandardScaler space)
    3. Inverse-transform synthetic to get UNSCALED versions
    4. Evaluate on UNSCALED data (original scale)
    """
    
    if verbose:
        print(f"✓ Generating FULL synthetic data (features + SYNTHETIC target)...")
    
    # Generate full synthetic data (on scaled space)
    X_synth_full_scaled = bridge.generate(n_samples=len(X_train_full_scaled), direction='forward')
    
    # Inverse transform synthetic to get unscaled versions
    X_synth_full_unscaled = scaler.inverse_transform(X_synth_full_scaled)
    
    # Extract components (SCALED)
    X_train_real_scaled = X_train_full_scaled[:, :-1]
    y_train_real_scaled = X_train_full_scaled[:, -1]
    X_test_real_scaled = X_test_full_scaled[:, :-1]
    y_test_real_scaled = X_test_full_scaled[:, -1]
    X_synth_scaled = X_synth_full_scaled[:, :-1]
    y_synth_scaled = X_synth_full_scaled[:, -1]
    
    # Extract components (UNSCALED)
    X_train_real_unscaled = X_train_full_unscaled[:, :-1]
    y_train_real_unscaled = X_train_full_unscaled[:, -1]
    X_test_real_unscaled = X_test_full_unscaled[:, :-1]
    y_test_real_unscaled = X_test_full_unscaled[:, -1]
    X_synth_unscaled = X_synth_full_unscaled[:, :-1]
    y_synth_unscaled = X_synth_full_unscaled[:, -1]
    
    if verbose:
        print(f"  Real train (scaled): X={X_train_real_scaled.shape}, y={y_train_real_scaled.shape}")
        print(f"  Synthetic (scaled): X={X_synth_scaled.shape}, y={y_synth_scaled.shape}")
        print(f"  Real test (scaled): X={X_test_real_scaled.shape}, y={y_test_real_scaled.shape}")
    
    # Import CatBoost
    try:
        from catboost import CatBoostRegressor
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    except ImportError:
        if verbose:
            print("  ❌ CatBoost not available")
        return {}, X_synth_full_scaled
    
    # === SCALED DATA EVALUATION ===
    if verbose:
        print(f"✓ Training CatBoost models on SCALED data...")
    
    model_real_scaled = CatBoostRegressor(verbose=False, iterations=100, random_state=42)
    model_real_scaled.fit(X_train_real_scaled, y_train_real_scaled)
    y_pred_real_scaled = model_real_scaled.predict(X_test_real_scaled)
    
    model_synth_scaled = CatBoostRegressor(verbose=False, iterations=100, random_state=42)
    model_synth_scaled.fit(X_synth_scaled, y_synth_scaled)
    y_pred_synth_scaled = model_synth_scaled.predict(X_test_real_scaled)
    
    # Compute metrics (SCALED)
    r2_real_scaled = float(r2_score(y_test_real_scaled, y_pred_real_scaled))
    rmse_real_scaled = float(np.sqrt(mean_squared_error(y_test_real_scaled, y_pred_real_scaled)))
    mae_real_scaled = float(mean_absolute_error(y_test_real_scaled, y_pred_real_scaled))
    
    r2_synth_scaled = float(r2_score(y_test_real_scaled, y_pred_synth_scaled))
    rmse_synth_scaled = float(np.sqrt(mean_squared_error(y_test_real_scaled, y_pred_synth_scaled)))
    mae_synth_scaled = float(mean_absolute_error(y_test_real_scaled, y_pred_synth_scaled))
    
    if verbose:
        print(f"  [SCALED] Real → R2: {r2_real_scaled:.4f}, RMSE: {rmse_real_scaled:.4f}")
        print(f"  [SCALED] Synth → R2: {r2_synth_scaled:.4f}, RMSE: {rmse_synth_scaled:.4f}")
    
    # === UNSCALED DATA EVALUATION ===
    if verbose:
        print(f"✓ Training CatBoost models on UNSCALED data...")
    
    model_real_unscaled = CatBoostRegressor(verbose=False, iterations=100, random_state=42)
    model_real_unscaled.fit(X_train_real_unscaled, y_train_real_unscaled)
    y_pred_real_unscaled = model_real_unscaled.predict(X_test_real_unscaled)
    
    model_synth_unscaled = CatBoostRegressor(verbose=False, iterations=100, random_state=42)
    model_synth_unscaled.fit(X_synth_unscaled, y_synth_unscaled)
    y_pred_synth_unscaled = model_synth_unscaled.predict(X_test_real_unscaled)
    
    # Compute metrics (UNSCALED)
    r2_real_unscaled = float(r2_score(y_test_real_unscaled, y_pred_real_unscaled))
    rmse_real_unscaled = float(np.sqrt(mean_squared_error(y_test_real_unscaled, y_pred_real_unscaled)))
    mae_real_unscaled = float(mean_absolute_error(y_test_real_unscaled, y_pred_real_unscaled))
    
    r2_synth_unscaled = float(r2_score(y_test_real_unscaled, y_pred_synth_unscaled))
    rmse_synth_unscaled = float(np.sqrt(mean_squared_error(y_test_real_unscaled, y_pred_synth_unscaled)))
    mae_synth_unscaled = float(mean_absolute_error(y_test_real_unscaled, y_pred_synth_unscaled))
    
    if verbose:
        print(f"  [UNSCALED] Real → R2: {r2_real_unscaled:.4f}, RMSE: {rmse_real_unscaled:.4f}")
        print(f"  [UNSCALED] Synth → R2: {r2_synth_unscaled:.4f}, RMSE: {rmse_synth_unscaled:.4f}")
    
    metrics = {
        # Scaled utility metrics
        'r2_score_real_scaled': r2_real_scaled,
        'r2_score_synth_scaled': r2_synth_scaled,
        'rmse_real_scaled': rmse_real_scaled,
        'rmse_synth_scaled': rmse_synth_scaled,
        'mae_real_scaled': mae_real_scaled,
        'mae_synth_scaled': mae_synth_scaled,
        
        # Unscaled utility metrics
        'r2_score_real_unscaled': r2_real_unscaled,
        'r2_score_synth_unscaled': r2_synth_unscaled,
        'rmse_real_unscaled': rmse_real_unscaled,
        'rmse_synth_unscaled': rmse_synth_unscaled,
        'mae_real_unscaled': mae_real_unscaled,
        'mae_synth_unscaled': mae_synth_unscaled,
    }
    
    return metrics, X_synth_full_scaled


# ============================================================================
# 5. EVALUATE DISTRIBUTION METRICS
# ============================================================================

def evaluate_distribution_metrics(
    X_train_full_real: np.ndarray,
    X_synth_full: np.ndarray,
    dataset_name: str,
    verbose: bool = True
) -> Dict[str, float]:
    """Compute distribution metrics between real and synthetic FULL data."""
    
    if verbose:
        print(f"✓ Computing distribution metrics...")
    
    results = evaluate_synthetic_data(X_train_full_real, X_synth_full, verbose=False)
    
    def get_val(key):
        return float(results.get(key, 0.0))
    
    metrics = {
        'mmd': get_val('mmd'),
        'kl_divergence': get_val('kl_divergence'),
        'wasserstein_distance': get_val('wasserstein_distance'),
        'frobenius_norm': get_val('frobenius_norm'),
        'marginal_kl_mean': get_val('marginal_kl_mean'),
        'marginal_js_mean': get_val('marginal_js_mean'),
    }
    
    if verbose:
        print(f"  MMD: {metrics['mmd']:.4f}, KL: {metrics['kl_divergence']:.4f}")
    
    return metrics


# ============================================================================
# 6. MAIN EVALUATION PIPELINE
# ============================================================================

def evaluate_all_asbm_models(
    summary_file: str = "optuna_results_asbm/SUMMARY.json",
    pkl_file: str = "datasets_numeric_merged.pkl",
    output_json: str = "asbm_evaluation_results.json",
    output_plots_dir: str = "asbm_evaluation_plots",
    dataset_names: list = None,
    verbose: bool = True
) -> Dict[str, Dict]:
    """Complete evaluation pipeline with scaled + unscaled utility metrics."""
    
    print("\n" + "="*80)
    print("ASBM EVALUATION (SCALED + UNSCALED UTILITY METRICS)")
    print("="*80 + "\n")
    
    # Load Optuna summary
    summary = load_optuna_summary(summary_file)
    
    if dataset_names is None:
        dataset_names = list(summary.keys())
    
    print(f"Evaluating {len(dataset_names)} datasets\n")
    
    all_results = {}
    
    for i, dataset_name in enumerate(dataset_names, 1):
        print(f"{'='*80}")
        print(f"[{i}/{len(dataset_names)}] {dataset_name.upper()}")
        print(f"{'='*80}\n")
        
        try:
            if dataset_name not in summary:
                print(f"⚠️  No Optuna results for {dataset_name}\n")
                continue
            
            optuna_result = summary[dataset_name]
            if 'error' in optuna_result:
                print(f"⚠️  Optuna failed: {optuna_result['error']}\n")
                continue
            
            best_params = optuna_result['best_params']
            
            # Load and prepare full data (scaled + unscaled)
            X_train_scaled, X_test_scaled, X_train_unscaled, X_test_unscaled, n_features, scaler = load_and_prepare_dataset(
                pkl_file, dataset_name
            )
            
            # Create and train ASBM on scaled data
            bridge = create_and_train_asbm_full(
                X_train_scaled, X_test_scaled, best_params, dataset_name, verbose=True
            )
            
            # Evaluate utility metrics (scaled + unscaled)
            utility_metrics, X_synth_full_scaled = compute_catboost_utility_metrics(
                bridge, X_train_scaled, X_test_scaled, X_train_unscaled, X_test_unscaled, scaler, dataset_name, verbose=True
            )
            
            # Evaluate distribution metrics
            dist_metrics = evaluate_distribution_metrics(
                X_train_scaled, X_synth_full_scaled, dataset_name, verbose=True
            )
            
            # Combine metrics
            all_metrics = {**dist_metrics, **utility_metrics}
            
            # Store results
            all_results[dataset_name] = {
                'dataset': dataset_name,
                'n_samples': len(X_train_scaled),
                'n_features': n_features,
                'best_params': best_params,
                'metrics': all_metrics
            }
            
            print(f"✅ {dataset_name} COMPLETE\n")
        
        except Exception as e:
            print(f"❌ ERROR: {str(e)}\n")
            import traceback
            traceback.print_exc()
            all_results[dataset_name] = {'error': str(e)}
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80 + "\n")
    
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✓ Results saved: {output_json}\n")
    
    # Create plots
    create_metric_plots(all_results, output_plots_dir)
    
    # Print summary
    print_evaluation_summary(all_results)
    
    return all_results


# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================

def create_metric_plots(results: Dict[str, Dict], output_dir: str):
    """Create comparison plots for all metrics (12 total)."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    successful = {k: v for k, v in results.items() if 'error' not in v}
    if not successful:
        return
    
    datasets = list(successful.keys())
    
    # 12 metric pairs: 6 distribution + 3 scaled utility + 3 unscaled utility
    metrics_pairs = [
        # Distribution (6)
        ('mmd', None),
        ('kl_divergence', None),
        ('wasserstein_distance', None),
        ('frobenius_norm', None),
        ('marginal_kl_mean', None),
        ('marginal_js_mean', None),
        # Scaled utility (3)
        ('r2_score_real_scaled', 'r2_score_synth_scaled'),
        ('rmse_real_scaled', 'rmse_synth_scaled'),
        ('mae_real_scaled', 'mae_synth_scaled'),
        # Unscaled utility (3)
        ('r2_score_real_unscaled', 'r2_score_synth_unscaled'),
        ('rmse_real_unscaled', 'rmse_synth_unscaled'),
        ('mae_real_unscaled', 'mae_synth_unscaled'),
    ]
    
    sns.set_style("whitegrid")
    
    print("Creating 12 comparison plots...\n")
    
    for metric_real, metric_synth in metrics_pairs:
        real_values = []
        synth_values = []
        valid_datasets = []
        
        for ds in datasets:
            real_val = successful[ds]['metrics'].get(metric_real)
            
            if metric_synth is None:
                if real_val is not None:
                    real_values.append(real_val)
                    synth_values.append(real_val)
                    valid_datasets.append(ds)
            else:
                synth_val = successful[ds]['metrics'].get(metric_synth)
                if real_val is not None and synth_val is not None:
                    real_values.append(real_val)
                    synth_values.append(synth_val)
                    valid_datasets.append(ds)
        
        if not real_values:
            continue
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(valid_datasets))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, real_values, width, label='Trained on Real Data',
                       color='#2ecc71', edgecolor='black', linewidth=1.5, alpha=0.85)
        bars2 = ax.bar(x + width/2, synth_values, width, label='Trained on Synthetic Data',
                       color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.85)
        
        ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        
        title = metric_real.replace('_real', '').replace('_scaled', '').replace('_unscaled', '').replace('_', ' ').upper()
        
        if metric_synth is None:
            title += " (Distribution - Lower is Better)"
        elif 'scaled' in metric_real:
            title += " (Scaled Data - "
            title += "Higher is Better)" if 'r2' in metric_real else "Lower is Better)"
        else:
            title += " (Unscaled Data - "
            title += "Higher is Better)" if 'r2' in metric_real else "Lower is Better)"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(valid_datasets, rotation=45, ha='right')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plot_name = metric_real if metric_synth is None else metric_real.replace('_real', '')
        plt.savefig(output_path / f"asbm_comparison_{plot_name}.png", dpi=300)
        plt.close()
        print(f"  ✓ {plot_name} plot saved")
    
    # 3x3 grid
    print(f"\n  Creating combined 3x3 comparison grid...")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metric_real, metric_synth) in enumerate(metrics_pairs[:9]):
        ax = axes[idx]
        
        real_values = []
        synth_values = []
        valid_datasets = []
        
        for ds in datasets:
            real_val = successful[ds]['metrics'].get(metric_real)
            if metric_synth is None:
                if real_val is not None:
                    real_values.append(real_val)
                    synth_values.append(real_val)
                    valid_datasets.append(ds)
            else:
                synth_val = successful[ds]['metrics'].get(metric_synth)
                if real_val is not None and synth_val is not None:
                    real_values.append(real_val)
                    synth_values.append(synth_val)
                    valid_datasets.append(ds)
        
        if not real_values:
            continue
        
        x = np.arange(len(valid_datasets))
        width = 0.35
        
        ax.bar(x - width/2, real_values, width, label='Real',
               color='#2ecc71', edgecolor='black', linewidth=1, alpha=0.85)
        ax.bar(x + width/2, synth_values, width, label='Synthetic',
               color='#e74c3c', edgecolor='black', linewidth=1, alpha=0.85)
        
        title = metric_real.replace('_real', '').replace('_scaled', '').replace('_unscaled', '').replace('_', ' ')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(valid_datasets, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        if idx == 0:
            ax.legend(fontsize=9, loc='upper right')
    
    plt.suptitle('ASBM Evaluation: Distribution + Scaled Utility (3x3 Grid)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path / "asbm_comparison_grid_3x3.png", dpi=300)
    plt.close()
    print(f"  ✓ Combined 3x3 grid saved")
    
    print(f"\n✅ All 12 comparison plots created!\n")


# ============================================================================
# 8. SUMMARY
# ============================================================================

def print_evaluation_summary(results: Dict[str, Dict]):
    """Print summary."""
    print("="*80)
    print("EVALUATION SUMMARY")
    print("="*80 + "\n")
    
    successful = [r for r in results.values() if 'error' not in r]
    
    if successful:
        print(f"{'Dataset':<25} {'MMD':<10} {'R2_Real_Sc':<12} {'R2_Synth_Sc':<12} {'R2_Real_Un':<12} {'R2_Synth_Un':<12}")
        print("-" * 115)
        for ds_name, res in results.items():
            if 'error' not in res:
                m = res['metrics']
                print(f"{ds_name:<25} {m['mmd']:<10.4f} {m['r2_score_real_scaled']:<12.4f} "
                      f"{m['r2_score_synth_scaled']:<12.4f} {m['r2_score_real_unscaled']:<12.4f} "
                      f"{m['r2_score_synth_unscaled']:<12.4f}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-file", default="optuna_results_asbm/SUMMARY.json")
    parser.add_argument("--pkl-file", default="datasets_numeric_merged.pkl")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    
    evaluate_all_asbm_models(
        summary_file=args.summary_file,
        pkl_file=args.pkl_file,
        dataset_names=args.datasets,
        output_plots_dir="asbm_evaluation_plots" if not args.no_plots else None
    )
