#!/usr/bin/env python3
"""
optuna_tune_all_datasets_save_individual.py

Hyperparameter tuning for ASBM on ALL datasets from merged pickle file.
Saves best params for EACH dataset to separate JSON files in optuna_results_asbm/ folder.

Usage:
    python optuna_tune_all_datasets_save_individual.py                  # Tune all, 20 trials each
    python optuna_tune_all_datasets_save_individual.py --n-trials 50   # More trials
    python optuna_tune_all_datasets_save_individual.py --pkl-file custom.pkl
    python optuna_tune_all_datasets_save_individual.py --datasets wine california_housing
"""

import numpy as np
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import optuna
from optuna.samplers import TPESampler

try:
    from asbm_tabular_bridge import ASBMTabularBridge
except ImportError:
    print("❌ Error: asbm_tabular_bridge.py not found")
    print("   Make sure it's in the same directory")
    exit(1)

from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import linear_kernel


def compute_mmd(X_real: np.ndarray, X_synth: np.ndarray, kernel='rbf', sigma=1.0) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between real and synthetic data.
    
    MMD measures the distance between two distributions using kernel methods.
    Lower MMD = more similar distributions.
    
    Args:
        X_real: Real data (n_samples, n_features)
        X_synth: Synthetic data (n_samples, n_features)
        kernel: 'rbf' or 'linear'
        sigma: Bandwidth for RBF kernel (default: 1.0)
    
    Returns:
        MMD value (float, non-negative)
    """
    
    n_real = X_real.shape[0]
    n_synth = X_synth.shape[0]
    
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
    # where x, x' ~ P_real and y, y' ~ P_synth
    
    mmd_real = np.mean(K_real)  # E[K(x, x')]
    mmd_synth = np.mean(K_synth)  # E[K(y, y')]
    mmd_cross = np.mean(K_cross)  # E[K(x, y)]
    
    mmd_squared = mmd_real + mmd_synth - 2 * mmd_cross
    mmd = np.sqrt(np.maximum(mmd_squared, 0))  # Ensure non-negative
    
    return mmd


def compute_swd(X_real: np.ndarray, X_synth: np.ndarray, n_projections: int = 100, 
                random_state: int = 42) -> float:
    """
    Compute Sliced Wasserstein Distance (SWD) between real and synthetic data.
    
    SWD approximates Wasserstein distance by averaging 1D Wasserstein distances
    along random projections.
    
    Args:
        X_real: Real data (n_samples, n_features)
        X_synth: Synthetic data (n_samples, n_features)
        n_projections: Number of random projections (default: 100)
        random_state: Random seed for reproducibility
    
    Returns:
        SWD value (float, non-negative)
    """
    
    n_features = X_real.shape[1]
    n_samples = min(X_real.shape[0], X_synth.shape[0])
    
    rng = np.random.RandomState(random_state)
    
    swd_values = []
    
    # Generate random projection directions
    for _ in range(n_projections):
        # Random direction on unit sphere
        theta = rng.randn(n_features)
        theta = theta / np.linalg.norm(theta)
        
        # Project data onto this direction
        real_proj = X_real @ theta
        synth_proj = X_synth @ theta
        
        # Sort projections
        real_proj_sorted = np.sort(real_proj[:n_samples])
        synth_proj_sorted = np.sort(synth_proj[:n_samples])
        
        # 1D Wasserstein distance (L2 norm of sorted differences)
        wd = np.sqrt(np.mean((real_proj_sorted - synth_proj_sorted) ** 2))
        swd_values.append(wd)
    
    # Average over all projections
    swd = np.mean(swd_values)
    
    return swd


def validate_dimensions(X_train: np.ndarray, X_test: np.ndarray, dataset_name: str, batch_size: int) -> None:
    """Validate training data dimensions before ASBM initialization."""
    
    m_train, n_features = X_train.shape
    m_test, _ = X_test.shape
    
    print(f"\n✅ Data validation for '{dataset_name}':")
    print(f"   X_train: {X_train.shape} (m={m_train}, n={n_features})")
    print(f"   X_test: {X_test.shape}")
    print(f"   batch_size: {batch_size} (max allowed: {m_train})")
    
    # Critical checks
    assert len(X_train.shape) == 2, f"X_train must be 2D, got {X_train.shape}"
    assert X_test.shape[1] == n_features, f"Feature mismatch: {n_features} vs {X_test.shape[1]}"
    assert X_train.dtype == np.float32, f"X_train must be float32, got {X_train.dtype}"
    assert X_test.dtype == np.float32, f"X_test must be float32, got {X_test.dtype}"
    assert batch_size <= m_train, f"❌ batch_size ({batch_size}) > m_train ({m_train})"
    assert not np.any(np.isnan(X_train)), "X_train contains NaN"
    assert not np.any(np.isinf(X_train)), "X_train contains inf"
    
    print(f"   ✓ All checks passed\n")


def objective(
    trial: optuna.Trial,
    X_train_w: np.ndarray,
    dataset_name: str,
    batch_size: int,
    verbose: bool = False
) -> float:
    """
    Optuna objective function for ASBM tuning.
    
    Suggests hyperparameters and trains ASBM, returns metric.
    """
    
    # Suggest hyperparameters
    lr_g = trial.suggest_float('lr_g', 1e-6, 1e-3, log=True)
    lr_d = trial.suggest_float('lr_d', 1e-6, 1e-3, log=True)
    num_timesteps = trial.suggest_int('num_timesteps', 4, 8, step=2)
    layers_G = [
    trial.suggest_int('G_0', 256, 512, step=128),  
    trial.suggest_int('G_1', 256, 512, step=128),  
    trial.suggest_int('G_2', 256, 512, step=128),  
]
    layers_D = [
    trial.suggest_int('D_0', 128, 384, step=128), 
    trial.suggest_int('D_1', 128, 384, step=128),  
    trial.suggest_int('D_2', 128, 384, step=128),   
]
    if verbose and trial.number % 5 == 0:
        print(f"\n  Trial {trial.number}:")
        print(f"    lr_g={lr_g:.2e}, lr_d={lr_d:.2e}, timesteps={num_timesteps}")
    
    try:
        # Prepare training data
        m_train, n_features = X_train_w.shape
        
        # x0 is random noise, x1 is real data
        rng = np.random.RandomState(trial.number)
        x0_train = rng.randn(m_train, n_features).astype(np.float32)
        x1_train = X_train_w.astype(np.float32)
        
        # Test data (small)
        m_test = max(1, m_train // 10)
        x0_test = rng.randn(m_test, n_features).astype(np.float32)
        x1_test = rng.randn(m_test, n_features).astype(np.float32)
        
        # Initialize ASBM
        bridge = ASBMTabularBridge(
            x0_train=x0_train,
            x1_train=x1_train,
            x0_test=x0_test,
            x1_test=x1_test,
            num_timesteps=num_timesteps,
            epsilon=0.1,
        )
        
        # Verify dimensions
        assert bridge.D == n_features, f"Dimension mismatch: {bridge.D} vs {n_features}"
        
        # Adaptive batch size
        actual_batch_size = min(batch_size, m_train)
        
        # Train ASBM
        history = bridge.fit(
            imf_iters=1,
            inner_iters=30,  # Reduced for speed
            batch_size=actual_batch_size,
            lr_g=lr_g,
            lr_d=lr_d,
            layers_G_fw=layers_G,
            layers_G_bw=layers_G,
            layers_D_fw=layers_D,
            layers_D_bw=layers_D,  
            verbose=False,
        )
        
        # Generate synthetic data
        synth_w = bridge.generate(n_samples=m_train, direction='forward')
        
        # Mean error
        mmd = compute_mmd(x1_train ,synth_w)
        
        # Covariance error (Frobenius norm)
        swd = compute_swd(x1_train ,synth_w)
        
        # Metric: negative error (higher is better for maximization)
        metric = (mmd + swd)
        
        return metric
    
    except Exception as e:
        if verbose:
            print(f"    ❌ Failed: {str(e)[:50]}")
        return float('-inf')


def save_dataset_results(
    results_dir: Path,
    dataset_name: str,
    best_trial: int,
    best_value: float,
    best_params: Dict,
    n_trials: int,
    n_samples: int,
    n_features: int,
) -> None:
    """Save results for single dataset to individual JSON file."""
    filename = results_dir / f"{dataset_name}_best_params.json"

    # Reconstruct layer lists from individual G_* / D_* entries
    layers_G = [
        int(best_params["G_0"]),
        int(best_params["G_1"]),
        int(best_params["G_2"]),
    ]
    layers_D = [
        int(best_params["D_0"]),
        int(best_params["D_1"]),
        int(best_params["D_2"]),
    ]

    result_data = {
        "dataset": dataset_name,
        "n_trials": n_trials,
        "n_samples": n_samples,
        "n_features": n_features,
        "best_trial": int(best_trial),
        "best_value": float(best_value),
        "best_params": {
            "lr_g": float(best_params["lr_g"]),
            "lr_d": float(best_params["lr_d"]),
            "num_timesteps": int(best_params["num_timesteps"]),
            "layers_G": layers_G,
            "layers_D": layers_D,
        },
    }

    with open(filename, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"   ✓ Saved to: {filename}")



def tune_all_datasets(
    pkl_file: str = "datasets_numeric_merged.pkl",
    dataset_names: list = None,
    n_trials: int = 20,
    train_size: float = 0.8,
    results_dir: str = "optuna_results_asbm",
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Run Optuna tuning on all datasets from merged pickle file.
    Saves individual JSON files for each dataset.
    
    Args:
        pkl_file: Path to merged pickle file
        dataset_names: List of datasets to tune. If None, use all
        n_trials: Number of trials per dataset
        train_size: Train/test split ratio
        results_dir: Directory to save individual result JSON files
        verbose: Print progress
    
    Returns:
        Dictionary of results for each dataset
    """
    
    # Create results directory
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    
    # Load pickle file
    pkl_path = Path(pkl_file)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pkl_file}")
    
    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    # Determine which datasets to tune
    if dataset_names is None:
        dataset_names = list(raw_data.keys())
    
    print("\n" + "="*80)
    print("OPTUNA HYPERPARAMETER TUNING FOR ALL DATASETS")
    print("="*80)
    print(f"Pickle file: {pkl_file}")
    print(f"Datasets: {len(dataset_names)}")
    print(f"Trials per dataset: {n_trials}")
    print(f"Results directory: {results_dir}/")
    print("="*80 + "\n")
    
    all_results = {}
    
    # Tune each dataset
    for i, dataset_name in enumerate(dataset_names, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(dataset_names)}] TUNING: {dataset_name.upper()}")
        print(f"{'='*80}\n")
        
        try:
            # Load and prepare data
            X = np.asarray(raw_data[dataset_name], dtype=np.float32)
            
            if hasattr(raw_data[dataset_name], 'to_numpy'):
                X = raw_data[dataset_name].to_numpy().astype(np.float32)
            
            print(f"Data shape: {X.shape}")
            
            # Train/test split
            X_train, X_test = train_test_split(
                X, test_size=1.0 - train_size, random_state=42
            )
            
            # Standardize
            scaler = StandardScaler().fit(X_train)
            X_train_w = scaler.transform(X_train).astype(np.float32)
            X_test_w = scaler.transform(X_test).astype(np.float32)
            
            # Validate
            m_train = X_train_w.shape[0]
            n_features = X_train_w.shape[1]
            batch_size = min(256, m_train)
            validate_dimensions(X_train_w, X_test_w, dataset_name, batch_size)
            
            # Create Optuna study
            sampler = TPESampler(seed=42 + i)
            study = optuna.create_study(
                direction='minimize',
                sampler=sampler,
                study_name=f"asbm_{dataset_name}"
            )
            
            # Run optimization
            print(f"Running {n_trials} trials...\n")
            study.optimize(
                lambda trial: objective(trial, X_train_w, dataset_name, batch_size, verbose),
                n_trials=n_trials,
                show_progress_bar=True,
            )
            
            # Get best trial
            best_trial = study.best_trial
            best_params = best_trial.params
            
            print(f"\n✅ BEST TRIAL: {best_trial.number}")
            print(f"   Metric: {best_trial.value:.6f}")
            print(f"   Parameters:")
            for key, value in best_params.items():
                # if isinstance(value, float):
                #     print(f"     {key}: {value:.2e}" if abs(value) < 0.001 or abs(value) > 1000 else f"     {key}: {value:.6f}")
                # else:
                    print(f"     {key}: {value}")
            

            # Store results
            result = {
                "dataset": dataset_name,
                "n_trials": n_trials,
                "n_features": n_features,
                "best_trial": int(best_trial.number),
                "best_value": float(best_trial.value),
                "best_params": {
                    "lr_g": float(best_params["lr_g"]),
                    "lr_d": float(best_params["lr_d"]),
                    "num_timesteps": int(best_params["num_timesteps"]),
                    "layers_G": [
                        int(best_params["G_0"]),
                        int(best_params["G_1"]),
                        int(best_params["G_2"]),
                    ],
                    "layers_D": [
                        int(best_params["D_0"]),
                        int(best_params["D_1"]),
                        int(best_params["D_2"]),
                    ],
                },
            }

            
            all_results[dataset_name] = result
        

            # Save individual JSON file
            print(f"\nSaving results...")
            save_dataset_results(
                results_path,
                dataset_name,
                best_trial.number,
                best_trial.value,
                best_params,
                n_trials,
                m_train,
                n_features,
            )
        
        except Exception as e:
            print(f"\n❌ ERROR tuning {dataset_name}:")
            print(f"   {str(e)}")
            all_results[dataset_name] = {'error': str(e)}
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")
    
    successful = [d for d in all_results.values() if 'error' not in d]
    print(f"Successfully tuned {len(successful)}/{len(dataset_names)} datasets\n")
    
    print("Results:")
    for name, result in all_results.items():
        if 'error' not in result:
            print(f"  ✓ {name:<30} best={result['best_value']:>8.6f} trial={result['best_trial']}")
        else:
            print(f"  ❌ {name:<30} error")
    
    # Create summary file
    summary_file = results_path / "SUMMARY.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ All results saved to: {results_dir}/")
    print(f"   ├─ {len(successful)} individual JSON files (one per dataset)")
    print(f"   └─ SUMMARY.json (combined results)")
    print("="*80 + "\n")
    
    return all_results


def load_dataset_params(results_dir: str, dataset_name: str) -> Dict:
    """Load best params for single dataset from JSON file."""
    
    filename = Path(results_dir) / f"{dataset_name}_best_params.json"
    
    if not filename.exists():
        raise FileNotFoundError(f"Results file not found: {filename}")
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data['best_params']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune ASBM on all datasets, save individual results")
    
    parser.add_argument(
        "--pkl-file",
        type=str,
        default="datasets_numeric_merged.pkl",
        help="Path to merged pickle file"
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Specific datasets to tune (default: all)"
    )
    
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1,
        help="Number of trials per dataset (default: 20)"
    )
    
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.8,
        help="Train/test split ratio (default: 0.75)"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="optuna_results_asbm_83",
        help="Directory to save individual result JSON files"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    
    args = parser.parse_args()
    
    # Run tuning
    results = tune_all_datasets(
        pkl_file=args.pkl_file,
        dataset_names=args.datasets,
        n_trials=args.n_trials,
        train_size=args.train_size,
        results_dir=args.results_dir,
        verbose=args.verbose,
    )
    
    print("✅ Done! Results saved to:", args.results_dir)
    print("\nDirectory structure:")
    print(f"""
{args.results_dir}/
├── california_housing_best_params.json
├── wine_best_params.json
├── iris_best_params.json
├── diabetes_best_params.json
└── SUMMARY.json
""")
    
    print("\nUsage in code:")
    print(f"""
from optuna_tune_all_datasets_save_individual import load_dataset_params
import json

# Load best params for specific dataset
params = load_dataset_params('{args.results_dir}', 'california_housing')
print(f"lr_g: {{params['lr_g']}}")
print(f"sigma: {{params['sigma']}}")

# Or load all results from summary
with open('{args.results_dir}/SUMMARY.json', 'r') as f:
    all_results = json.load(f)

for dataset_name, result in all_results.items():
    if 'error' not in result:
        print(f"{{dataset_name}}: {{result['best_value']}}")
""")
