#!/usr/bin/env python3

"""

dsbm_eval_cv_5fold.py

DSBM 5-Fold Cross-Validation Evaluation Pipeline

Uses MetricsEvaluator for comprehensive metric computation.

Computes 20+ metrics per fold and saves complete results to JSON.

Evaluates ALL datasets from the pickle file automatically.

Requires:

- evaluation_metrics.py in same directory

- Optuna SUMMARY.json with best parameters for all datasets

- Dataset pickle file with all training data

Usage:

python dsbm_eval_cv_5fold.py

Output:

results_DSBM_[dataset].json for each dataset

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

# Import the MetricsEvaluator

from experiments.evaluation_metrics import MetricsEvaluator

# ============================================================================

# LOAD OPTUNA SUMMARY

# ============================================================================

def load_optuna_summary(summary_file: str = "results/optuna_results/optuna_results_dsbm/SUMMARY.json") -> Dict:

    """Load best hyperparameters from Optuna SUMMARY.json."""

    summary_path = Path(summary_file)

    if not summary_path.exists():

        raise FileNotFoundError(f"SUMMARY.json not found: {summary_file}")

    with open(summary_path, 'r') as f:

        summary = json.load(f)

    print(f"✓ Loaded Optuna SUMMARY: {summary_file}")

    print(f" Found {len(summary)} datasets\n")

    return summary

# ============================================================================

# LOAD DATASET

# ============================================================================

def load_dataset(pkl_file: str, dataset_name: str) -> np.ndarray:

    """Load dataset from pickle file."""

    pkl_path = Path(pkl_file)

    if not pkl_path.exists():

        raise FileNotFoundError(f"Pickle file not found: {pkl_file}")

    with open(pkl_path, 'rb') as f:

        raw_data = pickle.load(f)

    if dataset_name not in raw_data:

        raise ValueError(f"Dataset {dataset_name} not found in pickle")

    X_full = np.asarray(raw_data[dataset_name], dtype=np.float32)

    if hasattr(raw_data[dataset_name], 'to_numpy'):

        X_full = raw_data[dataset_name].to_numpy().astype(np.float32)

    if dataset_name in ["adult_numeric", "king_county_housing"]:
        X_full = X_full[:, [-1, 2]]

    return X_full

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

# ============================================================================

# CREATE AND TRAIN DSBM

# ============================================================================

def create_and_train_dsbm(

    X_train_full: np.ndarray,

    best_params: Dict[str, Any],

    verbose: bool = False

):

    """Create and train DSBM on full data."""

    try:

        from models.DSBM.dsbm_tabular_bridge import DSBMTabularBridge

    except ImportError:

        if verbose:

            print("❌ Error: dsbm_tabular_bridge not found")

        return None

    try:

        m_train, n_cols = X_train_full.shape

        rng = np.random.RandomState(42)

        x0_train = rng.randn(m_train, n_cols).astype(np.float32)

        x1_train = X_train_full.astype(np.float32)

        m_test = max(1, m_train // 10)

        x0_test = rng.randn(m_test, n_cols).astype(np.float32)

        x1_test = rng.randn(m_test, n_cols).astype(np.float32)

        bridge = DSBMTabularBridge(

            x0_train=x0_train, x1_train=x1_train,

            x0_test=x0_test, x1_test=x1_test,

            num_timesteps=best_params.get('num_timesteps', 100),

            learning_rate=best_params.get('learning_rate', 1e-4),

            sigma=best_params.get('sigma', 0.5),

            batch_size=min(256, m_train)

        )

        history = bridge.fit(

            imf_iters=best_params.get('imf_iters', 15),

            inner_iters=best_params.get('inner_iters', 500),

            batch_size=min(256, m_train),

            learning_rate=best_params.get('learning_rate', 1e-4),

            layers=best_params.get('layers', [256, 256, 256]),

            verbose=False

        )

        return bridge

    except Exception as e:

        if verbose:

            print(f"❌ Training failed: {str(e)}")

        return None

# ============================================================================

# GENERATE SYNTHETIC DATA

# ============================================================================

def generate_synthetic_data(bridge, n_samples: int) -> np.ndarray:

    """Generate synthetic data from trained DSBM."""

    try:

        X_synth = bridge.generate(n_samples=n_samples, direction='forward')

        return X_synth.astype(np.float32)

    except Exception as e:

        return None

# ============================================================================

# EVALUATE FOLD WITH METRICS EVALUATOR

# ============================================================================

def evaluate_fold(

    bridge,

    X_train_full: np.ndarray,

    X_test_full: np.ndarray,

    fold: int,

    evaluator: MetricsEvaluator

) -> Dict[str, Any]:

    """Evaluate a single fold with comprehensive metrics."""

    try:

        X_synth_full = generate_synthetic_data(bridge, len(X_train_full))

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

def evaluate_single_dataset(

    dataset_name: str,

    X_full: np.ndarray,

    best_params: Dict[str, Any],

    evaluator: MetricsEvaluator,

    n_splits: int = 5

) -> Tuple[bool, str]:

    """Evaluate a single dataset with 5-fold CV."""

    print(f"\n{'='*80}")

    print(f"DATASET: {dataset_name}")

    print(f"{'='*80}")

    print(f" Shape: {X_full.shape}")

    print(f" Features: {X_full.shape[1] - 1}, Target: 1 (last column)\n")

    print(f"✓ Best params:")

    for key, val in sorted(best_params.items()):

        print(f" {key}: {val}")

    print()

    y_strat = X_full[:, -1].astype(int) if X_full.shape[1] > 0 else np.arange(len(X_full))

    if len(np.unique(y_strat)) > 20:

        y_strat = np.digitize(X_full[:, -1], np.percentile(X_full[:, -1], np.linspace(0, 100, 6)))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []

    print(f"Running {n_splits}-fold evaluation:")

    print("-" * 80)

    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X_full, y_strat), 1):

        print(f"\nFold {fold_num}/{n_splits}:")

        X_train_full_fold = X_full[train_idx].copy()

        X_test_full_fold = X_full[test_idx].copy()

        print(f" Training samples: {len(X_train_full_fold)}, Test samples: {len(X_test_full_fold)}")

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train_full_fold).astype(np.float32)

        X_test_scaled = scaler.transform(X_test_full_fold).astype(np.float32)

        bridge = create_and_train_dsbm(X_train_scaled, best_params)

        if bridge is None:

            print(f" ✗ Model training failed")

            continue

        print(f" ✓ Model trained")

        fold_result = evaluate_fold(bridge, X_train_scaled, X_test_scaled, fold_num, evaluator)

        if fold_result:

            fold_results.append(fold_result)

            n_metrics = len([k for k in fold_result.keys() if k not in ['fold', 'test_size', 'synth_size']])

            print(f" ✓ Metrics computed: {n_metrics} metrics")

            if 'ks_statistic_mean' in fold_result:

                print(f" KS: {fold_result['ks_statistic_mean']:.4f}")

            if 'ml_efficiency_gap_percent' in fold_result:

                print(f" Gap: {fold_result['ml_efficiency_gap_percent']:.2f}%")

            if 'authenticity' in fold_result:

                print(f" Auth:{fold_result['authenticity']:.4f}")

        else:

            print(f" ✗ Fold evaluation failed")

    if not fold_results:

        print(f"\n✗ No successful folds for {dataset_name}! Skipping...")

        return False, ""

    print("\n" + "=" * 80)

    print("AGGREGATING RESULTS ACROSS ALL FOLDS")

    print("=" * 80)

    summary_stats = {

        'metrics_mean': {},

        'metrics_std': {},

        'metrics_min': {},

        'metrics_max': {}

    }

    exclude_keys = {'fold', 'test_size', 'synth_size'}

    metric_keys = [k for k in fold_results[0].keys() if k not in exclude_keys]

    print(f"\nAggregating {len(metric_keys)} metrics across {len(fold_results)} folds...")

    for metric in metric_keys:

        values = [f[metric] for f in fold_results if metric in f]

        if values:

            summary_stats['metrics_mean'][metric] = float(np.mean(values))

            summary_stats['metrics_std'][metric] = float(np.std(values))

            summary_stats['metrics_min'][metric] = float(np.min(values))

            summary_stats['metrics_max'][metric] = float(np.max(values))

    print(f"✓ Aggregated {len(summary_stats['metrics_mean'])} metrics")

    output = {

        'metadata': {

            'dataset': dataset_name,

            'model': 'DSBM',

            'n_folds': len(fold_results),

            'timestamp': datetime.now().isoformat(),

            'evaluator_version': '2.0',

            'n_metrics': len(metric_keys),

            'metrics_list': metric_keys

        },

        'folds': fold_results,

        'summary': summary_stats

    }

    output_file = f"results_DSBM_{dataset_name}.json"

    with open(output_file, 'w') as f:

        json.dump(output, f, indent=2, default=str)

    file_size_kb = Path(output_file).stat().st_size / 1024

    print(f"\n✓ Results saved to: {output_file}")

    print(f" Total folds: {len(fold_results)}")

    print(f" Total metrics: {len(metric_keys)}")

    print(f" File size: {file_size_kb:.1f} KB")

    print("\n" + "=" * 80)

    print("KEY METRICS (MEAN ± STD)")

    print("=" * 80)

    key_metrics = [

        'ks_statistic_mean',

        'wasserstein_distance',

        'correlation_similarity',

        'ml_efficiency_gap_percent',

        'authenticity'

    ]

    for metric in key_metrics:

        if metric in summary_stats['metrics_mean']:

            mean = summary_stats['metrics_mean'][metric]

            std = summary_stats['metrics_std'][metric]

            print(f"{metric:35s}: {mean:8.4f} ± {std:.4f}")

    return True, output_file

# ============================================================================

# MAIN EVALUATION LOOP

# ============================================================================

def main():

    """Run 5-fold CV evaluation for ALL datasets with comprehensive metrics."""

    print("=" * 80)

    print("DSBM 5-FOLD CV EVALUATION - ALL DATASETS")

    print("=" * 80)

    print()

    optuna_file = "results/optuna_results/optuna_results_dsbm/SUMMARY.json"

    dataset_file = "datasets/datasets_numeric_merged.pkl"

    try:

        summary = load_optuna_summary(optuna_file)

    except FileNotFoundError as e:

        print(f"✗ Error: {e}")

        print(f" Make sure Optuna summary file exists at: {optuna_file}")

        return

    try:

        all_datasets = load_all_datasets(dataset_file)

    except FileNotFoundError as e:

        print(f"✗ Error: {e}")

        print(f" Make sure dataset file exists at: {dataset_file}")

        return

    results_summary = {

        'successful': [],

        'failed': [],

        'total': len(all_datasets)

    }

    evaluator = MetricsEvaluator(

        include_authenticity=True,

        include_c2st=False

    )

    for dataset_name, X_full in all_datasets.items():

        if dataset_name not in summary:

            print(f"\n⚠️ WARNING: No Optuna params for dataset '{dataset_name}'")

            print(f" Available datasets: {list(summary.keys())}")

            results_summary['failed'].append({

                'dataset': dataset_name,

                'reason': 'No Optuna params'

            })

            continue

        try:

            best_params = summary[dataset_name]['best_params']

        except Exception as e:

            print(f"\n✗ Error extracting params for {dataset_name}: {str(e)}")

            results_summary['failed'].append({

                'dataset': dataset_name,

                'reason': f'Param extraction error: {str(e)}'

            })

            continue

        try:

            success, output_file = evaluate_single_dataset(

                dataset_name, X_full, best_params, evaluator

            )

            if success:

                results_summary['successful'].append({

                    'dataset': dataset_name,

                    'output_file': output_file

                })

            else:

                results_summary['failed'].append({

                    'dataset': dataset_name,

                    'reason': 'Evaluation failed'

                })

        except Exception as e:

            print(f"\n✗ Error evaluating {dataset_name}: {str(e)}")

            results_summary['failed'].append({

                'dataset': dataset_name,

                'reason': f'Evaluation error: {str(e)}'

            })

    print("\n\n" + "=" * 80)

    print("EVALUATION SUMMARY")

    print("=" * 80)

    print(f"\nTotal datasets: {results_summary['total']}")

    print(f"Successful: {len(results_summary['successful'])}")

    print(f"Failed: {len(results_summary['failed'])}")

    if results_summary['successful']:

        print("\n✅ Successfully evaluated datasets:")

        for item in results_summary['successful']:

            print(f" ✓ {item['dataset']:30s} → {item['output_file']}")

    if results_summary['failed']:

        print("\n❌ Failed datasets:")

        for item in results_summary['failed']:

            print(f" ✗ {item['dataset']:30s} → {item['reason']}")

    print("\n" + "=" * 80)

    print("✅ ALL EVALUATIONS COMPLETE!")

    print("=" * 80)

if __name__ == "__main__":

    main()
