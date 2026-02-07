#!/usr/bin/env python3
"""
ASBM Complete Evaluation Pipeline (FULL DATA WITH SYNTHETIC TARGET)

Workflow:
1. Load dataset (features + target as last column)
2. Split: (X_train, y_train) | (X_test, y_test)
3. Merge: X_train_full = [X_train | y_train] for ASBM input
4. Train ASBM on FULL data (features + target together)
5. Generate FULL synthetic data (synthetic features + synthetic target)
6. Train CatBoost Model 1: on REAL (X_train, y_train) → test on REAL (X_test, y_test)
7. Train CatBoost Model 2: on SYNTHETIC (X_synth, y_synth) → test on REAL (X_test, y_test)
8. Compare metrics on REAL test set

Output:
- asbm_evaluation_results.json (9 metrics)
- asbm_evaluation_plots/ (9 pairs + 3x3 grid)
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
    from models.ASBM.asbm_tabular_bridge import ASBMTabularBridge
except ImportError:
    print("❌ Error: asbm_tabular_bridge.py not found")
    exit(1)

try:
    from experiments.synthetic_data_metrics import evaluate_synthetic_data
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
) -> Tuple[np.ndarray, np.ndarray, int, StandardScaler]:
    """
    Load dataset and prepare full data (features + target).
    
    Returns: (X_train_full_w, X_test_full_w, n_features, scaler)
    where X_full = [features | target] (last column is target)
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
    
    # Train/test split on FULL data
    X_train_full, X_test_full = train_test_split(
        X_full, test_size=1.0 - train_size, random_state=42
    )
    
    # Standardize (fit on train, apply to both)
    # Standardize all columns including target
    scaler = StandardScaler().fit(X_train_full)
    X_train_full_w = scaler.transform(X_train_full).astype(np.float32)
    X_test_full_w = scaler.transform(X_test_full).astype(np.float32)
    
    n_features = X_train_full_w.shape[1] - 1  # Exclude target column
    return X_train_full_w, X_test_full_w, n_features, scaler


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
        imf_iters=8,
        inner_iters=5000,
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
# 4. COMPUTE CATBOOST UTILITY METRICS (NEW STRATEGY)
# ============================================================================

def compute_catboost_utility_metrics(
    bridge: ASBMTabularBridge,
    X_train_full: np.ndarray,
    X_test_full: np.ndarray,
    dataset_name: str,
    verbose: bool = True
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    NEW STRATEGY: Generate full synthetic data (features + target).
    
    Workflow:
    1. Generate X_synth_full from bridge (features + synthetic target)
    2. Extract: X_synth, y_synth from synthetic
    3. Extract: X_train, y_train, X_test, y_test from real
    4. Train Model 1: CatBoost(X_train_real, y_train_real)
    5. Train Model 2: CatBoost(X_synth, y_synth)  # BOTH synthetic!
    6. Test both on: X_test_real, y_test_real
    7. Compare metrics on real test set
    """
    
    if verbose:
        print(f"✓ Generating FULL synthetic data (features + SYNTHETIC target)...")
    
    # Generate full synthetic data
    X_synth_full = bridge.generate(n_samples=len(X_train_full), direction='forward')
    
    # Extract components
    X_train_real = X_train_full[:, :-1]
    y_train_real = X_train_full[:, -1]
    
    X_test_real = X_test_full[:, :-1]
    y_test_real = X_test_full[:, -1]
    
    X_synth = X_synth_full[:, :-1]
    y_synth = X_synth_full[:, -1]
    
    if verbose:
        print(f"  Real train: X={X_train_real.shape}, y={y_train_real.shape}")
        print(f"  Synthetic: X={X_synth.shape}, y={y_synth.shape}")
        print(f"  Real test: X={X_test_real.shape}, y={y_test_real.shape}")
    
    # Import CatBoost
    try:
        from catboost import CatBoostRegressor
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    except ImportError:
        if verbose:
            print("  ❌ CatBoost not available")
        return {}, X_synth_full
    
    if verbose:
        print(f"✓ Training two CatBoost models...")
    
    # Model 1: Train on REAL, test on REAL
    if verbose:
        print(f"  [Model 1] Training on REAL data...")
    model_real = CatBoostRegressor(verbose=False, iterations=100, random_state=42)
    model_real.fit(X_train_real, y_train_real)
    y_pred_real = model_real.predict(X_test_real)
    
    # Model 2: Train on SYNTHETIC, test on REAL
    if verbose:
        print(f"  [Model 2] Training on SYNTHETIC data...")
    model_synth = CatBoostRegressor(verbose=False, iterations=100, random_state=42)
    model_synth.fit(X_synth, y_synth)  # Train on BOTH synthetic features AND target
    y_pred_synth = model_synth.predict(X_test_real)  # Test on REAL features
    
    if verbose:
        print(f"✓ Computing metrics on REAL test set...")
    
    # Compute metrics
    r2_real = float(r2_score(y_test_real, y_pred_real))
    rmse_real = float(np.sqrt(mean_squared_error(y_test_real, y_pred_real)))
    mse_real = float(mean_squared_error(y_test_real, y_pred_real))
    mae_real = float(mean_absolute_error(y_test_real, y_pred_real))
    
    r2_synth = float(r2_score(y_test_real, y_pred_synth))
    rmse_synth = float(np.sqrt(mean_squared_error(y_test_real, y_pred_synth)))
    mse_synth = float(mean_squared_error(y_test_real, y_pred_synth))
    mae_synth = float(mean_absolute_error(y_test_real, y_pred_synth))
    
    metrics = {
        'r2_score_real': r2_real,
        'r2_score_synth': r2_synth,
        'rmse_real': rmse_real,
        'rmse_synth': rmse_synth,
        'mse_real': mse_real,
        'mse_synth': mse_synth,
        'mae_real': mae_real,
        'mae_synth': mae_synth,
    }
    
    if verbose:
        print(f"  Model trained on REAL → R2: {r2_real:.4f}, RMSE: {rmse_real:.4f}")
        print(f"  Model trained on SYNTHETIC → R2: {r2_synth:.4f}, RMSE: {rmse_synth:.4f}")
    
    return metrics, X_synth_full


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
    """Complete evaluation pipeline with synthetic target."""
    
    print("\n" + "="*80)
    print("ASBM EVALUATION (WITH SYNTHETIC TARGET)")
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
            
            # Load and prepare full data
            X_train_full, X_test_full, n_features, scaler = load_and_prepare_dataset(
                pkl_file, dataset_name
            )
            
            # Create and train ASBM on full data
            bridge = create_and_train_asbm_full(
                X_train_full, X_test_full, best_params, dataset_name, verbose=True
            )
            
            # Evaluate utility metrics with new strategy
            utility_metrics, X_synth_full = compute_catboost_utility_metrics(
                bridge, X_train_full, X_test_full, dataset_name, verbose=True
            )
            
            # Evaluate distribution metrics
            dist_metrics = evaluate_distribution_metrics(
                X_train_full, X_synth_full, dataset_name, verbose=True
            )
            
            # Combine metrics
            all_metrics = {**dist_metrics, **utility_metrics}
            
            # Store results
            all_results[dataset_name] = {
                'dataset': dataset_name,
                'n_samples': len(X_train_full),
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
    """Create side-by-side comparison plots for 9 metrics."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    successful = {k: v for k, v in results.items() if 'error' not in v}
    if not successful:
        return
    
    datasets = list(successful.keys())
    
    # 9 metric pairs
    metrics_pairs = [
        ('mmd', None),
        ('kl_divergence', None),
        ('wasserstein_distance', None),
        ('frobenius_norm', None),
        ('marginal_kl_mean', None),
        ('marginal_js_mean', None),
        ('r2_score_real', 'r2_score_synth'),
        ('rmse_real', 'rmse_synth'),
        ('mae_real', 'mae_synth'),
    ]
    
    sns.set_style("whitegrid")
    
    print("Creating 9 side-by-side comparison plots...\n")
    
    for metric_real, metric_synth in metrics_pairs:
        real_values = []
        synth_values = []
        valid_datasets = []
        
        for ds in datasets:
            real_val = successful[ds]['metrics'].get(metric_real)
            
            if metric_synth is None:
                # Distribution: same value for both
                if real_val is not None:
                    real_values.append(real_val)
                    synth_values.append(real_val)
                    valid_datasets.append(ds)
            else:
                # Utility: real model vs synth model
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
        
        title = metric_real.replace('_real', '').replace('_', ' ').upper()
        if metric_synth is None:
            title += " (Distribution - Lower is Better)"
        elif 'r2' in metric_real:
            title += " (Utility - Higher is Better)"
        else:
            title += " (Utility - Lower is Better)"
        
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
    
    for idx, (metric_real, metric_synth) in enumerate(metrics_pairs):
        ax = axes[idx]
        
        real_values = []
        synth_values = []
        valid_datasets = []
        
        for ds in datasets:
            real_val = successful[ds]['metrics'].get(metric_real)
            if metric_synth is None:
                if real_val is not None:
                    # real_values.append(real_val)
                    # synth_values.append(real_val)
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
        
        title = metric_real.replace('_real', '').replace('_', ' ')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(valid_datasets, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        if idx == 0:
            ax.legend(fontsize=9, loc='upper right')
    
    plt.suptitle('ASBM Evaluation: Real vs Synthetic Models (3x3 Grid)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path / "asbm_comparison_grid_3x3.png", dpi=300)
    plt.close()
    print(f"  ✓ Combined 3x3 grid saved")
    
    print(f"\n✅ All 9 comparison plots created!\n")


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
        print(f"{'Dataset':<25} {'MMD':<10} {'R2_Real':<12} {'R2_Synth':<12} {'RMSE_Real':<12} {'RMSE_Synth':<12}")
        print("-" * 95)
        for ds_name, res in results.items():
            if 'error' not in res:
                m = res['metrics']
                print(f"{ds_name:<25} {m['mmd']:<10.4f} {m['r2_score_real']:<12.4f} "
                      f"{m['r2_score_synth']:<12.4f} {m['rmse_real']:<12.4f} {m['rmse_synth']:<12.4f}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-file", default="results/optuna_results/optuna_results_asbm/SUMMARY.json")
    parser.add_argument("--pkl-file", default="datasets/datasets_numeric_merged.pkl")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    
    evaluate_all_asbm_models(
        summary_file=args.summary_file,
        pkl_file=args.pkl_file,
        dataset_names=args.datasets,
        output_plots_dir="asbm_evaluation_plots" if not args.no_plots else None
    )
