#!/usr/bin/env python3
"""
asbm_eval_cv_5fold.py

ASBM 5-Fold Cross-Validation Evaluation Pipeline

Correctly uses ASBMTabularBridge with:
- Proper initialization: x0_train, x1_train, x0_test, x1_test, num_timesteps, epsilon
- Correct fit() method with imf_iters, inner_iters, batch_size, lr_g, lr_d, layer configs
- Correct generate() method

Computes 20+ metrics per fold and saves complete results to JSON.

Usage:
python asbm_eval_cv_5fold.py
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import warnings
import sys

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root))
warnings.filterwarnings('ignore')

from experiments.evaluation_metrics import MetricsEvaluator
from models.ASBM.asbm_tabular_bridge import ASBMTabularBridge

def load_optuna_summary(summary_file: str = "results/optuna_results/optuna_results_asbm/SUMMARY.json") -> Dict:
    """Load best hyperparameters from Optuna SUMMARY.json."""
    summary_path = Path(summary_file)
    if not summary_path.exists():
        raise FileNotFoundError(f"SUMMARY.json not found: {summary_file}")
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    print(f"✓ Loaded Optuna SUMMARY: {summary_file}")
    print(f" Found {len(summary)} datasets\n")
    return summary

def load_all_datasets(pkl_file: str = "datasets/datasets_numeric_merged.pkl") -> Dict[str, np.ndarray]:
    """Load all datasets from pickle file."""
    pkl_path = Path(pkl_file)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pkl_file}")
    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    datasets = {}
    for dataset_name, data in raw_data.items():
        X_full = np.asarray(data, dtype=np.float32)
        if hasattr(data, 'to_numpy'):
            X_full = data.to_numpy().astype(np.float32)
        datasets[dataset_name] = X_full
    print(f"✓ Loaded all datasets from pickle file: {pkl_file}")
    print(f" Found {len(datasets)} datasets:")
    for name, data in datasets.items():
        print(f" - {name:30s}: {data.shape}")
    print()
    return datasets

def create_and_train_asbm(X_train_full: np.ndarray, best_params: Dict[str, Any], verbose: bool = False) -> ASBMTabularBridge:
    """Create and train ASBM bridge on full training data."""
    try:
        m_train, n_features = X_train_full.shape
        rng = np.random.RandomState(42)
        x0_train = rng.randn(m_train, n_features).astype(np.float32)
        x1_train = X_train_full.astype(np.float32)
        m_test = max(1, m_train // 10)
        x0_test = rng.randn(m_test, n_features).astype(np.float32)
        x1_test = rng.randn(m_test, n_features).astype(np.float32)
        
        lr_g = best_params.get('lr_g', 1e-4)
        lr_d = best_params.get('lr_d', 1e-4)
        num_timesteps = best_params.get('num_timesteps', 6)
        layers_G = [best_params.get('G_0', 256), best_params.get('G_1', 256), best_params.get('G_2', 256)]
        layers_D = [best_params.get('D_0', 128), best_params.get('D_1', 128), best_params.get('D_2', 128)]
        
        if verbose:
            print(f"[ASBM] Training with: lr_g={lr_g:.2e}, lr_d={lr_d:.2e}, num_timesteps={num_timesteps}")
        
        bridge = ASBMTabularBridge(x0_train=x0_train, x1_train=x1_train, x0_test=x0_test, x1_test=x1_test, num_timesteps=num_timesteps, epsilon=0.1)
        
        if verbose:
            print(f"[ASBM] Bridge initialized, D={bridge.D}, timesteps={bridge.num_timesteps}")
        
        history = bridge.fit(imf_iters=10, inner_iters=3000, batch_size=min(256, m_train), lr_g=lr_g, lr_d=lr_d, layers_G_fw=layers_G, layers_G_bw=layers_G, layers_D_fw=layers_D, layers_D_bw=layers_D, verbose=False)
        
        if verbose:
            print(f"[ASBM] Training complete")
        
        return bridge
    except Exception as e:
        if verbose:
            print(f"❌ ASBM training failed: {str(e)}")
        return None

def generate_synthetic_data(bridge: ASBMTabularBridge, n_samples: int, direction: str = "forward") -> np.ndarray:
    """Generate synthetic data from trained ASBM bridge."""
    try:
        synth = bridge.generate(n_samples=n_samples, direction=direction)
        return synth.astype(np.float32)
    except Exception as e:
        print(f"❌ Generation failed: {str(e)}")
        return None

def evaluate_fold(bridge: ASBMTabularBridge, X_train_full: np.ndarray, X_test_full: np.ndarray, fold: int, evaluator: MetricsEvaluator) -> Dict[str, Any]:
    """Evaluate a single fold with comprehensive metrics."""
    try:
        X_synth_full = generate_synthetic_data(bridge, n_samples=len(X_train_full), direction="forward")
        if X_synth_full is None:
            print(f" ✗ Fold {fold}: Synthetic generation failed")
            return {}
        metrics = evaluator.compute_all_metrics(X_test_full, X_synth_full)
        metrics['fold'] = fold
        metrics['test_size'] = len(X_test_full)
        metrics['synth_size'] = len(X_synth_full)
        return metrics
    except Exception as e:
        print(f" ✗ Fold {fold}: Evaluation failed - {str(e)}")
        return {}

def evaluate_single_dataset(dataset_name: str, X_full: np.ndarray, best_params: Dict[str, Any], evaluator: MetricsEvaluator, n_splits: int = 5) -> Tuple[bool, str]:
    """Evaluate a single dataset with 5-fold cross-validation."""
    print(f"\n{'='*80}\nDATASET: {dataset_name}\n{'='*80}\n Shape: {X_full.shape}\n")
    print(f"✓ Best params:")
    for key, val in sorted(best_params.items()):
        print(f" {key}: {val}")
    print()
    
    y_strat = X_full[:, -1].astype(int) if X_full.shape[1] > 0 else np.arange(len(X_full))
    if len(np.unique(y_strat)) > 20:
        y_strat = np.digitize(X_full[:, -1], np.percentile(X_full[:, -1], np.linspace(0, 100, 6)))
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    print(f"Running {n_splits}-fold evaluation:\n" + "-" * 80)
    
    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X_full, y_strat), 1):
        print(f"\nFold {fold_num}/{n_splits}:")
        X_train_full_fold = X_full[train_idx].copy()
        X_test_full_fold = X_full[test_idx].copy()
        print(f" Training: {len(X_train_full_fold)}, Test: {len(X_test_full_fold)}")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_full_fold).astype(np.float32)
        X_test_scaled = scaler.transform(X_test_full_fold).astype(np.float32)
        
        bridge = create_and_train_asbm(X_train_scaled, best_params, verbose=False)
        if bridge is None:
            print(f" ✗ Model training failed")
            continue
        print(f" ✓ ASBM trained")
        
        fold_result = evaluate_fold(bridge, X_train_scaled, X_test_scaled, fold_num, evaluator)
        if fold_result:
            fold_results.append(fold_result)
            n_metrics = len([k for k in fold_result.keys() if k not in ['fold', 'test_size', 'synth_size']])
            print(f" ✓ Metrics: {n_metrics}")
            if 'ks_statistic_mean' in fold_result:
                print(f"   KS: {fold_result['ks_statistic_mean']:.4f}")
            if 'ml_efficiency_gap_percent' in fold_result:
                print(f"   Gap: {fold_result['ml_efficiency_gap_percent']:.2f}%")
    
    if not fold_results:
        print(f"\n✗ No successful folds for {dataset_name}!")
        return False, ""
    
    print("\n" + "=" * 80 + "\nAGGREGATING RESULTS\n" + "=" * 80)
    summary_stats = {'metrics_mean': {}, 'metrics_std': {}, 'metrics_min': {}, 'metrics_max': {}}
    exclude_keys = {'fold', 'test_size', 'synth_size'}
    metric_keys = [k for k in fold_results[0].keys() if k not in exclude_keys]
    
    for metric in metric_keys:
        values = [f[metric] for f in fold_results if metric in f]
        if values:
            summary_stats['metrics_mean'][metric] = float(np.mean(values))
            summary_stats['metrics_std'][metric] = float(np.std(values))
            summary_stats['metrics_min'][metric] = float(np.min(values))
            summary_stats['metrics_max'][metric] = float(np.max(values))
    
    output = {'metadata': {'dataset': dataset_name, 'model': 'ASBM', 'n_folds': len(fold_results), 'timestamp': datetime.now().isoformat(), 'n_metrics': len(metric_keys)}, 'folds': fold_results, 'summary': summary_stats}
    output_file = f"results_ASBM_{dataset_name}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f" Folds: {len(fold_results)}, Metrics: {len(metric_keys)}")
    return True, output_file

def main():
    """Run 5-fold CV evaluation for ALL datasets with ASBM."""
    print("=" * 80 + "\nASBM 5-FOLD CV EVALUATION - ALL DATASETS\n" + "=" * 80 + "\n")
    
    optuna_file = "results/optuna_results/optuna_results_asbm/SUMMARY.json"
    dataset_file = "datasets/datasets_numeric_merged.pkl"
    
    try:
        summary = load_optuna_summary(optuna_file)
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return
    
    try:
        all_datasets = load_all_datasets(dataset_file)
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return
    
    results_summary = {'successful': [], 'failed': [], 'total': len(all_datasets)}
    evaluator = MetricsEvaluator(include_authenticity=True, include_c2st=False)
    
    for dataset_name, X_full in all_datasets.items():
        if dataset_name not in summary:
            print(f"\n⚠️ No params for '{dataset_name}'")
            results_summary['failed'].append({'dataset': dataset_name, 'reason': 'No Optuna params'})
            continue
        
        try:
            best_params = summary[dataset_name]['best_params']
        except Exception as e:
            print(f"\n✗ Error extracting params for {dataset_name}: {str(e)}")
            results_summary['failed'].append({'dataset': dataset_name, 'reason': f'Param extraction error'})
            continue
        
        try:
            success, output_file = evaluate_single_dataset(dataset_name, X_full, best_params, evaluator)
            if success:
                results_summary['successful'].append({'dataset': dataset_name, 'output_file': output_file})
            else:
                results_summary['failed'].append({'dataset': dataset_name, 'reason': 'Evaluation failed'})
        except Exception as e:
            print(f"\n✗ Error evaluating {dataset_name}: {str(e)}")
            results_summary['failed'].append({'dataset': dataset_name, 'reason': f'Evaluation error'})
    
    print("\n\n" + "=" * 80 + "\nFINAL SUMMARY\n" + "=" * 80)
    print(f"\nTotal datasets: {results_summary['total']}")
    print(f"Successful: {len(results_summary['successful'])}")
    print(f"Failed: {len(results_summary['failed'])}")
    
    if results_summary['successful']:
        print("\n✅ Successful:")
        for item in results_summary['successful']:
            print(f" ✓ {item['dataset']:30s} → {item['output_file']}")
    
    if results_summary['failed']:
        print("\n❌ Failed:")
        for item in results_summary['failed']:
            print(f" ✗ {item['dataset']:30s}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
