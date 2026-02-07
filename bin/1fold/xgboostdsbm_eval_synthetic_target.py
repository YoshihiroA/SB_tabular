#!/usr/bin/env python3

"""
XGBoost DSBM Evaluation Script (CatBoost Regressor for Comparability)

Supports BOTH JSON formats with professional plotting and summary output.
Uses CatBoost regressors for utility metrics (matching ASBM and original DSBM).

Features:
1. Auto-detects OLD vs NEW JSON format
2. Creates 9 individual comparison plots (like ASBM)
3. Creates 3x3 grid comparison plot
4. Prints professional summary table in terminal
5. Updates SUMMARY.json with evaluation results
6. Uses CatBoost regressors for consistency with ASBM/DSBM

Usage:
    python xgboost_dsbm_eval_refactored.py
    python xgboost_dsbm_eval_refactored.py --datasets california_housing diabetes
    python xgboost_dsbm_eval_refactored.py --no-plots
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# Import XGBoost DSBM and metrics
try:
    from models.DSBM_xgboost.xgboostdsbm_tabular_bridge import XGBoostDSBMTabularBridge
except ImportError:
    print("❌ Error: xgboostdsbm_tabular_bridge.py not found")
    exit(1)

try:
    from experiments.synthetic_data_metrics import evaluate_synthetic_data
except ImportError:
    print("❌ Error: synthetic_data_metrics.py not found")
    exit(1)

# CatBoost for consistent evaluation
try:
    from catboost import CatBoostRegressor
except ImportError:
    print("❌ Error: CatBoost not installed")
    exit(1)


# ============================================================================
# BACKWARD-COMPATIBLE PARAMETER EXTRACTION
# ============================================================================

def extract_params(params_result: Dict) -> Tuple[Dict, Dict]:
    """
    Extract best_params and fixed_params with backward compatibility.
    
    Supports two formats:
    1. NEW: Separate best_params + fixed_params
    2. OLD: All params in best_params (auto-splits them)
    
    Returns: (best_params_dict, fixed_params_dict)
    """
    
    # NEW FORMAT: Has both best_params and fixed_params
    if 'best_params' in params_result and 'fixed_params' in params_result:
        return params_result['best_params'], params_result['fixed_params']
    
    # OLD FORMAT: All params in best_params, need to split
    if 'best_params' in params_result:
        all_params = params_result['best_params']
        
        # Extract 3 optimized params
        best_params = {
            'sig': all_params.get('sig', 0.1),
            'max_depth': all_params.get('max_depth', 5),
            'learning_rate': all_params.get('learning_rate', 0.1),
        }
        
        # Extract 10+ fixed params
        fixed_params = {
            'n_timesteps': all_params.get('n_timesteps', 20),
            'eps': all_params.get('eps', 0.05),
            'n_iterations': 20,
            'n_estimators': all_params.get('n_estimators', 300),
            'subsample': all_params.get('subsample', 0.8),
            'colsample_bytree': all_params.get('colsample_bytree', 0.8),
            'reg_alpha': all_params.get('reg_alpha', 0.1),
            'reg_lambda': all_params.get('reg_lambda', 0.1),
            'objective': all_params.get('objective', 'reg:squarederror'),
            'random_state': all_params.get('random_state', 42),
            'n_jobs': all_params.get('n_jobs', -1),
        }
        
        return best_params, fixed_params
    
    # ERROR: No params found
    raise ValueError("No best_params found in params_result")


# ============================================================================
# 1. LOAD SUMMARY.JSON
# ============================================================================

def load_summary_json(params_file: str = "optuna_results_xgboost_dsbm/SUMMARY.json") -> Dict:
    """Load SUMMARY.json (any format)."""
    params_path = Path(params_file)
    if not params_path.exists():
        raise FileNotFoundError(f"SUMMARY.json file not found: {params_file}")
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    print(f"✓ Loaded SUMMARY.json: {params_file}")
    print(f" Found {len(params)} datasets\n")
    
    return params


# ============================================================================
# 2. LOAD AND PREPARE DATASET
# ============================================================================

def load_and_prepare_dataset(
    pkl_file: str,
    dataset_name: str,
    train_size: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, int, StandardScaler]:
    """Load dataset and prepare full data (features + target)."""
    
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
    
    # Standardize
    scaler = StandardScaler().fit(X_train_full)
    X_train_full_w = scaler.transform(X_train_full).astype(np.float32)
    X_test_full_w = scaler.transform(X_test_full).astype(np.float32)
    
    n_features = X_train_full_w.shape[1] - 1
    
    return X_train_full_w, X_test_full_w, n_features, scaler


# ============================================================================
# 3. CREATE AND TRAIN XGBoost DSBM
# ============================================================================

def create_and_train_xgboost_dsbm_full(
    X_train_full: np.ndarray,
    X_test_full: np.ndarray,
    best_params: Dict[str, Any],
    fixed_params: Dict[str, Any],
    dataset_name: str,
    verbose: bool = True
) -> XGBoostDSBMTabularBridge:
    """Create and train XGBoostDSBMTabularBridge."""
    
    m_train, n_cols = X_train_full.shape
    m_test = X_test_full.shape[0]
    
    if verbose:
        print(f"✓ Creating XGBoost DSBM for {dataset_name}")
        print(f" X_train shape: {X_train_full.shape}")
    
    # Create x0 (random noise)
    rng = np.random.RandomState(42)
    x0_train = rng.randn(m_train, n_cols).astype(np.float32)
    x1_train = X_train_full.astype(np.float32)
    
    x0_test = rng.randn(m_test, n_cols).astype(np.float32)
    x1_test = X_test_full.astype(np.float32)
    
    # Build XGBoost params
    xgb_params = {
        "max_depth": best_params['max_depth'],
        "learning_rate": best_params['learning_rate'],
        "n_estimators": fixed_params['n_estimators'],
        "subsample": fixed_params['subsample'],
        "colsample_bytree": fixed_params['colsample_bytree'],
        "reg_alpha": fixed_params['reg_alpha'],
        "reg_lambda": fixed_params['reg_lambda'],
        "objective": fixed_params['objective'],
        "random_state": fixed_params['random_state'],
        "n_jobs": fixed_params['n_jobs'],
    }
    
    if verbose:
        print(f"✓ Params: sig={best_params['sig']:.4f}, "
              f"max_depth={best_params['max_depth']}, "
              f"lr={best_params['learning_rate']:.4f}")
    
    bridge = XGBoostDSBMTabularBridge(
        x0_train=x0_train,
        x1_train=x1_train,
        x0_test=x0_test,
        x1_test=x1_test,
        n_timesteps=fixed_params['n_timesteps'],
        sig=best_params['sig'],
        eps=fixed_params['eps'],
        xgb_params=xgb_params,
    )
    
    if verbose:
        print(f"✓ Training XGBoost DSBM...")
    
    n_iterations = fixed_params.get('n_iterations', 12)
    history = bridge.fit(n_iterations=n_iterations, verbose=False)
    
    return bridge


# ============================================================================
# 4. COMPUTE METRICS (CatBoost for consistency with ASBM/DSBM)
# ============================================================================

def compute_catboost_utility_metrics(
    bridge: XGBoostDSBMTabularBridge,
    X_train_full: np.ndarray,
    X_test_full: np.ndarray,
    dataset_name: str,
    verbose: bool = True
) -> Tuple[Dict[str, float], np.ndarray]:
    """Generate synthetic data and compute utility metrics using CatBoost (for comparability)."""
    
    if verbose:
        print(f"✓ Generating synthetic data...")
    
    X_synth_full = bridge.generate(n_samples=len(X_train_full), direction="forward", n_euler_steps=100)
    
    X_train_real = X_train_full[:, :-1]
    y_train_real = X_train_full[:, -1]
    
    X_test_real = X_test_full[:, :-1]
    y_test_real = X_test_full[:, -1]
    
    X_synth = X_synth_full[:, :-1]
    y_synth = X_synth_full[:, -1]
    
    if verbose:
        print(f" Real: X={X_train_real.shape}, Synth: X={X_synth.shape}")
    
    try:
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    except ImportError:
        if verbose:
            print(" ❌ sklearn metrics not available")
        return {}, X_synth_full
    
    if verbose:
        print(f"✓ Training CatBoost regressors (consistent with ASBM/DSBM)...")
    
    # Model 1: Train on REAL, test on REAL (baseline)
    if verbose:
        print(f" [Model 1] Training on REAL data...")
    model_real = CatBoostRegressor(verbose=False, iterations=100, random_state=42)
    model_real.fit(X_train_real, y_train_real)
    y_pred_real = model_real.predict(X_test_real)
    
    # Model 2: Train on SYNTHETIC, test on REAL (utility measure)
    if verbose:
        print(f" [Model 2] Training on SYNTHETIC data...")
    model_synth = CatBoostRegressor(verbose=False, iterations=100, random_state=42)
    model_synth.fit(X_synth, y_synth)
    y_pred_synth = model_synth.predict(X_test_real)
    
    if verbose:
        print(f"✓ Computing metrics on REAL test set...")
    
    metrics = {
        'r2_score_real': float(r2_score(y_test_real, y_pred_real)),
        'r2_score_synth': float(r2_score(y_test_real, y_pred_synth)),
        'rmse_real': float(np.sqrt(mean_squared_error(y_test_real, y_pred_real))),
        'rmse_synth': float(np.sqrt(mean_squared_error(y_test_real, y_pred_synth))),
        'mse_real': float(mean_squared_error(y_test_real, y_pred_real)),
        'mse_synth': float(mean_squared_error(y_test_real, y_pred_synth)),
        'mae_real': float(mean_absolute_error(y_test_real, y_pred_real)),
        'mae_synth': float(mean_absolute_error(y_test_real, y_pred_synth)),
    }
    
    if verbose:
        print(f" Real model (trained on real) R2: {metrics['r2_score_real']:.4f}")
        print(f" Synth model (trained on synthetic) R2: {metrics['r2_score_synth']:.4f}")
    
    return metrics, X_synth_full


def evaluate_distribution_metrics(
    X_train_full_real: np.ndarray,
    X_synth_full: np.ndarray,
    dataset_name: str,
    verbose: bool = True
) -> Dict[str, float]:
    """Compute distribution metrics."""
    
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
        print(f" MMD: {metrics['mmd']:.4f}")
    
    return metrics


# ============================================================================
# 5. MAIN EVALUATION
# ============================================================================

def evaluate_all_xgboost_dsbm_models(
    params_file: str = "optuna_results_xgboost_dsbm/SUMMARY.json",
    pkl_file: str = "datasets_numeric_merged.pkl",
    dataset_names: list = None,
    no_plots: bool = False,
    verbose: bool = True
) -> Dict[str, Dict]:
    """Evaluation pipeline with CatBoost for consistency."""
    
    print("\n" + "="*80)
    print("XGBoost DSBM EVALUATION (CatBoost Regressor for Comparability)")
    print("="*80 + "\n")
    
    params_dict = load_summary_json(params_file)
    
    if dataset_names is None:
        dataset_names = list(params_dict.keys())
    
    print(f"Evaluating {len(dataset_names)} datasets\n")
    
    evaluated_results = {}
    
    for i, dataset_name in enumerate(dataset_names, 1):
        print(f"{'='*80}")
        print(f"[{i}/{len(dataset_names)}] {dataset_name.upper()}")
        print(f"{'='*80}\n")
        
        try:
            if dataset_name not in params_dict:
                print(f"⚠️ No parameters for {dataset_name}\n")
                continue
            
            params_result = params_dict[dataset_name]
            
            # ✅ BACKWARD-COMPATIBLE PARAMETER EXTRACTION
            try:
                best_params, fixed_params = extract_params(params_result)
                fmt = 'NEW' if 'fixed_params' in params_result else 'OLD'
                print(f"✓ Extracted params (format: {fmt})")
            except ValueError as e:
                print(f"⚠️ {str(e)}\n")
                continue
            
            # Load data
            X_train_full, X_test_full, n_features, scaler = load_and_prepare_dataset(
                pkl_file, dataset_name
            )
            
            # Train bridge
            bridge = create_and_train_xgboost_dsbm_full(
                X_train_full, X_test_full, best_params, fixed_params, dataset_name, verbose=True
            )
            
            # Evaluate with CatBoost (consistency with ASBM/DSBM)
            utility_metrics, X_synth_full = compute_catboost_utility_metrics(
                bridge, X_train_full, X_test_full, dataset_name, verbose=True
            )
            
            dist_metrics = evaluate_distribution_metrics(
                X_train_full, X_synth_full, dataset_name, verbose=True
            )
            
            all_metrics = {**dist_metrics, **utility_metrics}
            
            # Store in results dict
            evaluated_results[dataset_name] = {
                'dataset': dataset_name,
                'n_samples': len(X_train_full),
                'n_features': n_features,
                'best_params': best_params,
                'metrics': all_metrics
            }
            
            # Update JSON
            params_dict[dataset_name]['evaluation'] = {
                **all_metrics,
                'evaluation_timestamp': datetime.now().isoformat(),
                'evaluation_status': 'completed'
            }
            
            print(f"✅ {dataset_name} COMPLETE\n")
        
        except Exception as e:
            print(f"❌ ERROR: {str(e)}\n")
            import traceback
            traceback.print_exc()
            
            params_dict[dataset_name]['evaluation'] = {
                'evaluation_timestamp': datetime.now().isoformat(),
                'evaluation_status': 'failed',
                'error': str(e)
            }
    
    # Save updated JSON
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80 + "\n")
    
    summary_path = Path(params_file)
    with open(summary_path, 'w') as f:
        json.dump(params_dict, f, indent=2)
    
    print(f"✓ Updated {params_file}\n")
    
    # Create plots
    if not no_plots and evaluated_results:
        create_metric_plots(evaluated_results, "xgboost_dsbm_evaluation_plots")
    
    # Print summary
    print_evaluation_summary(evaluated_results)
    
    return evaluated_results


# ============================================================================
# 6. VISUALIZATIONS (Like ASBM)
# ============================================================================

def create_metric_plots(results: Dict[str, Dict], output_dir: str):
    """Create side-by-side comparison plots (same as ASBM)."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if not results:
        print("⚠️  No completed evaluations to plot\n")
        return
    
    datasets = list(results.keys())
    
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
            real_val = results[ds]['metrics'].get(metric_real)
            
            if metric_synth is None:
                # Distribution: same value for both
                if real_val is not None:
                    real_values.append(real_val)
                    synth_values.append(real_val)
                    valid_datasets.append(ds)
            else:
                # Utility: real model vs synth model
                synth_val = results[ds]['metrics'].get(metric_synth)
                
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
        plt.savefig(output_path / f"xgboost_dsbm_comparison_{plot_name}.png", dpi=300)
        plt.close()
        
        print(f" ✓ {plot_name} plot saved")
    
    # 3x3 grid
    print(f"\n Creating combined 3x3 comparison grid...")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metric_real, metric_synth) in enumerate(metrics_pairs):
        ax = axes[idx]
        
        real_values = []
        synth_values = []
        valid_datasets = []
        
        for ds in datasets:
            real_val = results[ds]['metrics'].get(metric_real)
            
            if metric_synth is None:
                if real_val is not None:
                    real_values.append(real_val)
                    synth_values.append(real_val)
                    valid_datasets.append(ds)
            else:
                synth_val = results[ds]['metrics'].get(metric_synth)
                
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
    
    plt.suptitle('XGBoost DSBM Evaluation: Real vs Synthetic Models (3x3 Grid)',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_path / "xgboost_dsbm_comparison_grid_3x3.png", dpi=300)
    plt.close()
    
    print(f" ✓ Combined 3x3 grid saved\n")
    print(f"✅ All 9 comparison plots created!\n")


# ============================================================================
# 7. SUMMARY TABLE (Like ASBM)
# ============================================================================

def print_evaluation_summary(results: Dict[str, Dict]):
    """Print professional summary table (same as ASBM)."""
    
    print("="*80)
    print("EVALUATION SUMMARY")
    print("="*80 + "\n")
    
    if not results:
        print("No completed evaluations.\n")
        return
    
    # Print header
    print(f"{'Dataset':<25} {'MMD':<10} {'R2_Real':<12} {'R2_Synth':<12} {'RMSE_Real':<12} {'RMSE_Synth':<12}")
    print("-" * 95)
    
    # Print each dataset
    for ds_name, res in results.items():
        m = res['metrics']
        print(f"{ds_name:<25} {m['mmd']:<10.4f} {m['r2_score_real']:<12.4f} "
              f"{m['r2_score_synth']:<12.4f} {m['rmse_real']:<12.4f} {m['rmse_synth']:<12.4f}")
    
    print("\n" + "="*80 + "\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="XGBoost DSBM Evaluation with CatBoost for Consistency"
    )
    
    parser.add_argument("--params-file", default="results/optuna_results/optuna_results_xgboost_dsbm/SUMMARY.json",
                        help="Path to SUMMARY.json (any format)")
    parser.add_argument("--pkl-file", default="datasets/datasets_numeric_merged.pkl",
                        help="Path to pickle file")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets to evaluate")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    
    args = parser.parse_args()
    
    evaluate_all_xgboost_dsbm_models(
        params_file=args.params_file,
        pkl_file=args.pkl_file,
        dataset_names=args.datasets,
        no_plots=args.no_plots
    )
    
    print("✅ Done!\n")
