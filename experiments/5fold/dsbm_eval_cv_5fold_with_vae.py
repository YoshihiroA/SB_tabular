#!/usr/bin/env python3

"""
dsbm_eval_cv_5fold_with_vae.py

DSBM 5-Fold Cross-Validation Evaluation Pipeline WITH VAE Preprocessing

Pipeline:
1. Download datasets
2. One-hot encode categorical features (if present)
3. Split into train and test
4. Learn StandardScaler() on train, apply to both
5. Learn VAE on train, apply encode to both → reduced dimension space
6. Fit DSBM bridge between Gaussian and VAE-encoded train data
7. Sample from test gaussian synthetic dataset
8. Implement reverse VAE to obtain initial dataset shape
9. Evaluate metrics between real test and generated from bridge reversed VAE synthetic dataset
Evaluation is same as now in file so no corrections for downloading, evaluation and saving parts.

Uses MetricsEvaluator for comprehensive metric computation.
Computes 20+ metrics per fold and saves complete results to JSON.
Evaluates ALL datasets from the pickle file automatically.

Requires:
- evaluation_metrics.py in same directory
- vae_preprocessing_attention.py in same directory
- Optuna SUMMARY.json with best parameters for all datasets
- Dataset pickle file with all training data

Usage:
python dsbm_eval_cv_5fold_with_vae.py

Output:
results_DSBM_VAE_[dataset].json for each dataset
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
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

# Import VAE preprocessor
from experiments.vae_preprocessing_attention import AttentionVAEPreprocessor

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_optimal_num_heads(input_dim: int, preferred_heads: int = 8) -> int:
    """
    Find the largest number of attention heads that divides input_dim.
    
    Attention requires: embed_dim % num_heads == 0
    Tries to use as many heads as possible (better correlation learning).
    
    Algorithm:
    1. If input_dim % preferred_heads == 0, use preferred_heads
    2. Otherwise, find the largest divisor of input_dim up to preferred_heads
    3. Fall back to 1 if nothing works
    
    Examples:
    - input_dim=23 (prime): tries 8→no, 7→no, ..., 2→no, 1→yes → uses 1 head
    - input_dim=24: tries 8→yes → uses 8 heads
    - input_dim=16: tries 8→yes → uses 8 heads
    - input_dim=14: tries 8→no, 7→yes → uses 7 heads
    - input_dim=12: tries 8→no, 6→yes → uses 6 heads
    - input_dim=9: tries 8→no, 3→yes → uses 3 heads
    
    Note: For prime input_dims like 23, we use 1 head, but this is ONLY for
    the attention mechanism on the encoder INPUT. The VAE can still compress
    to any latent_dim (like 8) without any constraint!
    
    Args:
        input_dim: Input feature dimension
        preferred_heads: Try heads from preferred_heads down to 1
    
    Returns:
        Optimal number of heads (largest divisor <= preferred_heads)
    """
    # Try from preferred_heads down to 1, find first divisor
    for num_heads in range(preferred_heads, 0, -1):
        if input_dim % num_heads == 0:
            return num_heads
    return 1


# ============================================================================
# PREPROCESSING PIPELINE CLASS
# ============================================================================

class DSBMVAEPipeline:
    """
    Complete preprocessing pipeline: StandardScaler → VAE encode/decode
    
    Workflow:
    1. fit() learns StandardScaler and VAE on training data
    2. transform() applies both to input data (returns VAE-encoded latent)
    3. inverse_transform() decodes from latent back to original space
    
    Key insight:
    - Input features (N) can have any attention heads (largest divisor <= 8)
    - Latent dimension can be any value (e.g., 8) regardless of input features
    - Compression 23→8 is perfectly fine, only attention is constrained
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        use_attention: bool = True,
        num_heads: int = None,
        vae_epochs: int = 50,
        verbose: bool = False
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_attention = use_attention
        
        # Auto-determine num_heads if not specified
        # This finds the largest divisor of input_dim up to 8
        if num_heads is None:
            num_heads = get_optimal_num_heads(input_dim)
        elif input_dim % num_heads != 0:
            # If specified num_heads doesn't work, auto-correct
            num_heads = get_optimal_num_heads(input_dim)
        
        self.num_heads = num_heads
        self.vae_epochs = vae_epochs
        self.verbose = verbose
        
        self.scaler = None
        self.vae = None
        self.is_fitted = False
    
    def fit(self, X_train: np.ndarray) -> 'DSBMVAEPipeline':
        """
        Fit StandardScaler and VAE on training data.
        
        Args:
            X_train: Training data [N, input_dim]
        
        Returns:
            self (for chaining)
        """
        
        # ① Fit StandardScaler on train
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
        
        if self.verbose:
            print(f"[Pipeline] StandardScaler fitted on {len(X_train)} samples")
        
        # ② Fit VAE on scaled train data
        self.vae = AttentionVAEPreprocessor(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            use_attention=self.use_attention,
            num_heads=self.num_heads,
            beta_schedule='warmup',  # Adaptive KL weighting
            verbose=self.verbose
        )
        
        self.vae.fit(
            X_train_scaled,
            epochs=self.vae_epochs,
            batch_size=32,
            validation_split=0.1,
            verbose_every=10
        )
        
        if self.verbose:
            print(f"[Pipeline] VAE fitted (input_dim={self.input_dim}, latent_dim={self.latent_dim}, num_heads={self.num_heads})")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data: StandardScaler → VAE encode
        
        Args:
            X: Input data [N, input_dim]
        
        Returns:
            Z: VAE latent representation [N, latent_dim]
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        
        # ① Apply StandardScaler
        X_scaled = self.scaler.transform(X).astype(np.float32)
        
        # ② Encode with VAE
        Z = self.vae.encode(X_scaled)
        
        return Z.astype(np.float32)
    
    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Inverse transform: VAE decode → StandardScaler inverse
        
        Args:
            Z: VAE latent representation [N, latent_dim]
        
        Returns:
            X: Original-space data [N, input_dim]
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before inverse_transform")
        
        # ① Decode with VAE
        X_scaled = self.vae.decode(Z).astype(np.float32)
        
        # ② Apply StandardScaler inverse
        X = self.scaler.inverse_transform(X_scaled)
        
        return X.astype(np.float32)


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
        
        # Create x0 (random noise) and x1 (real data in VAE latent space)
        x0_train = rng.randn(m_train, n_cols).astype(np.float32)
        x1_train = X_train_full.astype(np.float32)
        
        # Create dummy test data
        m_test = max(1, m_train // 10)
        x0_test = rng.randn(m_test, n_cols).astype(np.float32)
        x1_test = rng.randn(m_test, n_cols).astype(np.float32)
        
        # Initialize bridge
        bridge = DSBMTabularBridge(
            x0_train=x0_train, x1_train=x1_train,
            x0_test=x0_test, x1_test=x1_test,
            num_timesteps=best_params.get('num_timesteps', 100),
            learning_rate=best_params.get('learning_rate', 1e-4),
            sigma=best_params.get('sigma', 0.5),
            batch_size=min(256, m_train)
        )
        
        # Train
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
    X_train_vae: np.ndarray,  # VAE-encoded training data
    X_test_orig: np.ndarray,  # Original test data (for metrics)
    pipeline: DSBMVAEPipeline,  # Pipeline for inverse transform
    fold: int,
    evaluator: MetricsEvaluator
) -> Dict[str, Any]:
    """
    Evaluate a single fold with comprehensive metrics.
    
    Pipeline:
    1. Generate synthetic in VAE latent space
    2. Decode back to original space
    3. Evaluate against real test data
    """
    try:
        # ① Generate synthetic in VAE latent space
        Z_synth = generate_synthetic_data(bridge, len(X_train_vae))
        if Z_synth is None:
            print(f" ✗ Fold {fold}: Synthetic generation failed")
            return {}
        
        # ② Decode from VAE latent space back to original
        X_synth_orig = pipeline.inverse_transform(Z_synth)
        
        # ③ Compute metrics comparing real test vs synthetic original
        metrics = evaluator.compute_all_metrics(X_test_orig, X_synth_orig)
        
        # Add fold metadata
        metrics['fold'] = fold
        metrics['test_size'] = len(X_test_orig)
        metrics['synth_size'] = len(X_synth_orig)
        metrics['vae_latent_dim'] = pipeline.latent_dim
        metrics['vae_num_heads'] = pipeline.num_heads
        metrics['vae_input_dim'] = pipeline.input_dim
        
        return metrics
    
    except Exception as e:
        print(f" ✗ Fold {fold}: Evaluation failed - {str(e)}")
        return {}


def evaluate_single_dataset(
    dataset_name: str,
    X_full: np.ndarray,
    best_params: Dict[str, Any],
    evaluator: MetricsEvaluator,
    n_splits: int = 5,
    vae_latent_dim: int = 8,
    vae_epochs: int = 50
) -> Tuple[bool, str]:
    """
    Evaluate a single dataset with 5-fold CV + VAE preprocessing.
    
    Automatically adjusts num_heads based on dataset feature dimensions.
    
    Returns: (success, output_file)
    """
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*80}")
    print(f" Shape: {X_full.shape}")
    print(f" Features: {X_full.shape[1] - 1}, Target: 1 (last column)\n")
    
    # Print params
    print(f"✓ Best params:")
    for key, val in sorted(best_params.items()):
        print(f" {key}: {val}")
    print(f" VAE latent dim: {vae_latent_dim}")
    print()
    
    # 5-fold evaluation
    y_strat = X_full[:, -1].astype(int) if X_full.shape[1] > 0 else np.arange(len(X_full))
    
    if len(np.unique(y_strat)) > 20:
        y_strat = np.digitize(X_full[:, -1], np.percentile(X_full[:, -1], np.linspace(0, 100, 6)))
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    
    print(f"Running {n_splits}-fold evaluation with VAE preprocessing:")
    print("-" * 80)
    
    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X_full, y_strat), 1):
        print(f"\nFold {fold_num}/{n_splits}:")
        
        # ① Split data (ORIGINAL SPACE)
        X_train_orig = X_full[train_idx].copy()
        X_test_orig = X_full[test_idx].copy()
        
        print(f" Training samples: {len(X_train_orig)}, Test samples: {len(X_test_orig)}")
        
        # ② Build and fit preprocessing pipeline (StandardScaler + VAE)
        # num_heads is auto-determined based on input_dim (largest divisor up to 8)
        pipeline = DSBMVAEPipeline(
            input_dim=X_train_orig.shape[1],
            latent_dim=vae_latent_dim,
            use_attention=True,
            num_heads=None,  # Auto-determine from input_dim
            vae_epochs=vae_epochs,
            verbose=False
        )
        
        print(f" ℹ Compression: {pipeline.input_dim}d → {pipeline.latent_dim}d")
        print(f" ℹ Attention heads: {pipeline.num_heads} (largest divisor of {pipeline.input_dim})")
        
        pipeline.fit(X_train_orig)
        
        # ③ Transform to VAE latent space
        X_train_vae = pipeline.transform(X_train_orig).astype(np.float32)
        
        print(f" ✓ VAE encoding complete: {X_train_orig.shape[1]} → {X_train_vae.shape[1]} dims")
        
        # ④ Train DSBM in latent space
        bridge = create_and_train_dsbm(X_train_vae, best_params)
        
        if bridge is None:
            print(f" ✗ Model training failed")
            continue
        
        print(f" ✓ DSBM trained on VAE latent space ({X_train_vae.shape[1]} dims)")
        
        # ⑤ Evaluate fold
        # Note: X_test_orig is in original space, metrics will compare:
        # - Real test (original space)
        # - Synthetic from DSBM latent → decoded back to original
        fold_result = evaluate_fold(
            bridge, X_train_vae, X_test_orig, pipeline, fold_num, evaluator
        )
        
        if fold_result:
            fold_results.append(fold_result)
            n_metrics = len([k for k in fold_result.keys() if k not in ['fold', 'test_size', 'synth_size', 'vae_latent_dim', 'vae_num_heads', 'vae_input_dim']])
            print(f" ✓ Metrics computed: {n_metrics} metrics")
            
            # Display key metrics
            if 'ks_statistic_mean' in fold_result:
                print(f" KS: {fold_result['ks_statistic_mean']:.4f}")
            if 'ml_efficiency_gap_percent' in fold_result:
                print(f" Gap: {fold_result['ml_efficiency_gap_percent']:.2f}%")
            if 'authenticity' in fold_result:
                print(f" Auth: {fold_result['authenticity']:.4f}")
        else:
            print(f" ✗ Fold evaluation failed")
    
    if not fold_results:
        print(f"\n✗ No successful folds for {dataset_name}! Skipping...")
        return False, ""
    
    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATING RESULTS ACROSS ALL FOLDS")
    print("=" * 80)
    
    # Summary statistics
    summary_stats = {
        'metrics_mean': {},
        'metrics_std': {},
        'metrics_min': {},
        'metrics_max': {}
    }
    
    # Get all metric names (excluding fold metadata)
    exclude_keys = {'fold', 'test_size', 'synth_size', 'vae_latent_dim', 'vae_num_heads', 'vae_input_dim'}
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
    
    # Prepare final JSON output
    output = {
        'metadata': {
            'dataset': dataset_name,
            'model': 'DSBM_with_VAE',
            'n_folds': len(fold_results),
            'timestamp': datetime.now().isoformat(),
            'evaluator_version': '2.0',
            'n_metrics': len(metric_keys),
            'metrics_list': metric_keys,
            'vae_input_dim': fold_results[0].get('vae_input_dim', 'auto'),
            'vae_latent_dim': vae_latent_dim,
            'vae_num_heads': fold_results[0].get('vae_num_heads', 'auto'),
            'vae_epochs': vae_epochs
        },
        'folds': fold_results,
        'summary': summary_stats
    }
    
    # Save JSON
    output_file = f"results_DSBM_VAE_{dataset_name}.json"
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    file_size_kb = Path(output_file).stat().st_size / 1024
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f" Total folds: {len(fold_results)}")
    print(f" Total metrics: {len(metric_keys)}")
    print(f" File size: {file_size_kb:.1f} KB")
    
    # Display summary
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
    """Run 5-fold CV evaluation for ALL datasets with VAE preprocessing."""
    
    print("=" * 80)
    print("DSBM 5-FOLD CV EVALUATION WITH VAE PREPROCESSING - ALL DATASETS")
    print("=" * 80)
    print()
    
    # Configuration
    optuna_file = "results/optuna_results/optuna_results_dsbm/SUMMARY.json"
    dataset_file = "datasets/datasets_numeric_merged.pkl"
    vae_latent_dim = 8      # Dimension for VAE latent space
    vae_epochs = 50         # Training epochs for VAE
    
    # ① Load Optuna params
    try:
        summary = load_optuna_summary(optuna_file)
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print(f" Make sure Optuna summary file exists at: {optuna_file}")
        return
    
    # ② Load all datasets
    try:
        all_datasets = load_all_datasets(dataset_file)
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print(f" Make sure dataset file exists at: {dataset_file}")
        return
    
    # ③ Evaluate each dataset
    results_summary = {
        'successful': [],
        'failed': [],
        'total': len(all_datasets)
    }
    
    evaluator = MetricsEvaluator(
        include_authenticity=True,
        include_c2st=False  # Set to True for detection metrics (slower)
    )
    
    for dataset_name, X_full in all_datasets.items():
        
        # Check if params exist for this dataset
        if dataset_name not in summary:
            print(f"\n⚠️ WARNING: No Optuna params for dataset '{dataset_name}'")
            print(f" Available datasets: {list(summary.keys())}")
            results_summary['failed'].append({
                'dataset': dataset_name,
                'reason': 'No Optuna params'
            })
            continue
        
        # Extract params
        try:
            best_params = summary[dataset_name]['best_params']
        except Exception as e:
            print(f"\n✗ Error extracting params for {dataset_name}: {str(e)}")
            results_summary['failed'].append({
                'dataset': dataset_name,
                'reason': f'Param extraction error: {str(e)}'
            })
            continue
        
        # Evaluate dataset
        try:
            success, output_file = evaluate_single_dataset(
                dataset_name, X_full, best_params, evaluator,
                vae_latent_dim=vae_latent_dim,
                vae_epochs=vae_epochs
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
    
    # ④ Print final summary
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
