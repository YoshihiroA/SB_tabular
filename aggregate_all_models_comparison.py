#!/usr/bin/env python3
"""
aggregate_all_models_comparison_UPDATED.py

Aggregate 5-fold CV results from ALL MODELS and create per-metric visualizations.

Key improvements vs original:
1. Distinct y-axis scaling per metric (IQR-based with outlier detection)
2. Relative outlier thresholds to prevent scale crushing
3. Utility metrics as SEPARATE plots with CORRECT formulas:
   - R² Improvement: (R²_synth - R²_real) / R²_real * 100%
   - RMSE Improvement: (RMSE_real - RMSE_synth) / RMSE_real * 100%
4. EXCLUDE ml_efficiency_gap_percent from plotting
5. Handle outliers exactly as FINAL version for datasets

Structure expected:
results/5fold_metrics/
├── dsbm/          → results_*.json
├── asbm/          → results_*.json
├── dsbmxgboost/   → results_*.json
├── dsbm_vae/      → results_*.json
└── tabsyn/        → results_*.json

Usage:
python aggregate_all_models_comparison_UPDATED.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_ROOT = Path("results/5fold_metrics")
OUTPUT_DIR = Path("results")
FIGURES_DIR = Path("figures/model_comparison_by_metric")

DPI = 300
FIGSIZE = (16, 9)

DATASETS = [
    "california_housing",
    "diabetes",
    "king_county_housing",
    "adult_numeric",
    "bank_marketing",
    "online_shoppers",
    "covertype",
    "german_credit"
]

MODELS = {
    "dsbm": {"color": "#FF6B6B", "label": "DSBM"},
    "asbm": {"color": "#4ECDC4", "label": "ASBM"},
    "dsbmxgboost": {"color": "#45B7D1", "label": "XGBoost-DSBM"},
    "dsbm_vae": {"color": "#FFA07A", "label": "DSBM-VAE"},
    "tabsyn": {"color": "#95E1D3", "label": "TabSyn"},
}

# Metrics to plot - EXCLUDING ml_efficiency_gap_percent
METRICS_TO_PLOT = [
    "r2_score_synth",
    "rmse_synth",
    "ks_statistic_mean",
    "wasserstein_distance",
    "mmd",
    "correlation_similarity",
    "pairwise_correlation_diff",
    "correlation_distance",
    "kl_divergence",
    "js_divergence",
    "swd",
    "dcr_mean",
    "dcr_median",
    "dcr_share",
    "authenticity",
]

# Utility metrics (for special handling)
UTILITY_REAL_SYNTH_PAIRS = {
    "r2_score": ("r2_score_real", "r2_score_synth"),
    "rmse": ("rmse_real", "rmse_synth"),
}

# Metrics that should have non-negative y-axis
NON_NEGATIVE_METRICS = {
    "ks_statistic_mean", "wasserstein_distance", "mmd", "kl_divergence",
    "js_divergence", "swd", "correlation_similarity", "pairwise_correlation_diff",
    "correlation_distance", "dcr_mean", "dcr_median", "dcr_share", "authenticity"
}

# Distribution metrics (for relative scaling)
DISTRIBUTION_METRICS = {
    "ks_statistic_mean", "wasserstein_distance", "mmd", "kl_divergence",
    "js_divergence", "swd"
}

# ============================================================================
# OUTLIER DETECTION
# ============================================================================

def detect_outlier_bounds(values: np.ndarray, is_distribution: bool = False) -> Tuple[float, float]:
    """
    Detect outlier bounds using IQR method with relative scaling for distribution metrics.
    """
    valid_values = values[~np.isnan(values)]
    
    if len(valid_values) == 0:
        return 0.0, 1.0
    
    q1 = np.percentile(valid_values, 25)
    q3 = np.percentile(valid_values, 75)
    median = np.percentile(valid_values, 50)
    iqr = q3 - q1
    
    if iqr == 0:
        lower = q1 - 1.0
        upper = q1 + 1.0
    else:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
    
    # For distribution metrics: scale thresholds relative to median
    if is_distribution and median > 0:
        relative_upper = q3 + 1.5 * iqr if iqr > 0 else q3 + 1.0
        max_reasonable = median * 3.0
        upper = min(relative_upper, max_reasonable)
    
    if is_distribution:
        lower = max(0.0, lower)
    
    return float(lower), float(upper)

def clamp_value_with_flag(value: float, lower: float, upper: float) -> Tuple[float, bool]:
    """Clamp value to bounds and return outlier flag."""
    if np.isnan(value):
        return 0.0, False
    
    is_outlier = value < lower or value > upper
    clamped = np.clip(value, lower, upper)
    
    return clamped, is_outlier

# ============================================================================
# LOADING FUNCTIONS
# ============================================================================

def find_model_dirs() -> Dict[str, Path]:
    """Find all model directories in RESULTS_ROOT."""
    model_dirs = {}
    for model_name in MODELS.keys():
        model_path = RESULTS_ROOT / model_name
        if model_path.exists():
            model_dirs[model_name] = model_path
            print(f"✓ Found model: {model_name}")
        else:
            print(f"⚠️  Model dir not found: {model_path}")
    return model_dirs

def load_results_from_json(json_path: Path) -> Tuple[str, Dict]:
    """Load single JSON result file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        dataset_name = data['metadata']['dataset']
        summary = data['summary']['metrics_mean']
        std_dev = data['summary']['metrics_std']
        
        return dataset_name, {'mean': summary, 'std': std_dev}
    except Exception as e:
        return None, None

def load_model_results(model_dir: Path) -> Dict[str, Dict]:
    """Load all JSON results for a single model."""
    results = {}
    json_files = sorted(model_dir.glob("*.json"))
    
    for json_file in json_files:
        dataset_name, metrics = load_results_from_json(json_file)
        if dataset_name and metrics:
            results[dataset_name] = metrics
    
    return results

def load_all_models_results(model_dirs: Dict[str, Path]) -> Dict[str, Dict[str, Dict]]:
    """Load results for all models."""
    all_models = {}
    
    for model_name, model_dir in model_dirs.items():
        results = load_model_results(model_dir)
        all_models[model_name] = results
        print(f"  Loaded {len(results)} datasets")
    
    return all_models

# ============================================================================
# UTILITY METRICS TRANSFORMATION
# ============================================================================

def calculate_utility_improvements(all_models: Dict[str, Dict[str, Dict]]) -> Dict[str, Dict[str, Dict]]:
    """
    Calculate R² and RMSE improvements for all models and datasets.
    
    Returns: {model_name: {dataset_name: {"r2_improvement": value, "rmse_improvement": value, ...}}}
    """
    improvements = {}
    
    for model_name in all_models.keys():
        improvements[model_name] = {}
        
        for dataset_name, metrics_dict in all_models[model_name].items():
            means = metrics_dict['mean']
            stds = metrics_dict['std']
            
            dataset_improvements = {}
            
            # R² improvement
            r2_real = means.get("r2_score_real", np.nan)
            r2_synth = means.get("r2_score_synth", np.nan)
            
            if not np.isnan(r2_real) and not np.isnan(r2_synth) and r2_real != 0:
                r2_improvement = (r2_synth - r2_real) / abs(r2_real) * 100
                r2_std = stds.get("r2_score_synth", 0.0)
            else:
                r2_improvement = np.nan
                r2_std = 0.0
            
            dataset_improvements["r2_score_synth"] = r2_improvement
            dataset_improvements["r2_score_synth_std"] = r2_std
            
            # RMSE improvement
            rmse_real = means.get("rmse_real", np.nan)
            rmse_synth = means.get("rmse_synth", np.nan)
            
            if not np.isnan(rmse_real) and not np.isnan(rmse_synth) and rmse_real != 0:
                rmse_improvement = (rmse_real - rmse_synth) / rmse_real * 100
                rmse_std = stds.get("rmse_synth", 0.0)
            else:
                rmse_improvement = np.nan
                rmse_std = 0.0
            
            dataset_improvements["rmse_synth"] = rmse_improvement
            dataset_improvements["rmse_synth_std"] = rmse_std
            
            improvements[model_name][dataset_name] = dataset_improvements
    
    return improvements

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_metric_data(all_models: Dict[str, Dict[str, Dict]], metric: str, 
                       utility_improvements: Dict = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Prepare data for plotting a single metric across all models and datasets.
    
    Returns:
    - means_df: DataFrame with shape (n_datasets, n_models)
    - stds_df: DataFrame with shape (n_datasets, n_models)
    - outlier_info: Dict with outlier bounds and flags
    """
    means_data = {}
    stds_data = {}
    all_values = []
    
    for model_name in MODELS.keys():
        means_col = []
        stds_col = []
        
        for dataset_name in DATASETS:
            mean_val = np.nan
            std_val = 0.0
            
            if model_name in all_models and dataset_name in all_models[model_name]:
                # Handle utility metrics with improvements
                if metric in ["r2_score_synth", "rmse_synth"] and utility_improvements:
                    if dataset_name in utility_improvements[model_name]:
                        mean_val = utility_improvements[model_name][dataset_name].get(metric, np.nan)
                        std_val = utility_improvements[model_name][dataset_name].get(f"{metric}_std", 0.0)
                else:
                    # Regular metrics
                    metrics_dict = all_models[model_name][dataset_name]
                    mean_val = metrics_dict['mean'].get(metric, np.nan)
                    std_val = metrics_dict['std'].get(metric, 0.0)
            
            if np.isnan(mean_val):
                means_col.append(np.nan)
                stds_col.append(0.0)
            else:
                means_col.append(mean_val)
                stds_col.append(std_val)
                all_values.append(mean_val)
        
        means_data[model_name] = means_col
        stds_data[model_name] = stds_col
    
    means_df = pd.DataFrame(means_data, index=DATASETS)
    stds_df = pd.DataFrame(stds_data, index=DATASETS)
    
    # Detect outliers
    valid_values = np.array(all_values)
    is_distribution = metric in DISTRIBUTION_METRICS
    lower_bound, upper_bound = detect_outlier_bounds(valid_values, is_distribution=is_distribution)
    
    outlier_info = {
        'lower': lower_bound,
        'upper': upper_bound,
        'is_distribution': is_distribution,
        'all_values': all_values
    }
    
    return means_df, stds_df, outlier_info

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_metric_comparison(metric: str, means_df: pd.DataFrame, stds_df: pd.DataFrame,
                          outlier_info: Dict, output_path: Path = None) -> None:
    """
    Create grouped bar plot comparing a metric across all models and datasets.
    
    Each dataset has grouped bars for all models with outlier detection.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    
    n_datasets = len(DATASETS)
    n_models = len([m for m in MODELS.keys() if m in means_df.columns])
    
    bar_width = 0.15
    x = np.arange(n_datasets)
    
    lower_bound = outlier_info['lower']
    upper_bound = outlier_info['upper']
    
    # Plot bars for each model
    model_order = [m for m in MODELS.keys() if m in means_df.columns]
    
    for idx, model_name in enumerate(model_order):
        offset = (idx - n_models/2 + 0.5) * bar_width
        x_pos = x + offset
        
        means = means_df[model_name].values
        stds = stds_df[model_name].values
        
        color = MODELS[model_name]['color']
        label = MODELS[model_name]['label']
        
        # Clamp values and detect outliers
        clamped_means = []
        outlier_flags = []
        clamped_stds = []
        
        for mean_val, std_val in zip(means, stds):
            if np.isnan(mean_val):
                clamped_means.append(0.0)
                outlier_flags.append(False)
                clamped_stds.append(0.0)
            else:
                clamped, is_outlier = clamp_value_with_flag(mean_val, lower_bound, upper_bound)
                clamped_means.append(clamped)
                outlier_flags.append(is_outlier)
                max_std = abs(upper_bound - lower_bound) * 0.1
                clamped_stds.append(min(std_val, max_std))
        
        # Plot bars
        bars = ax.bar(x_pos, clamped_means, bar_width,
                     yerr=clamped_stds,
                     label=label,
                     color=color,
                     alpha=0.8,
                     capsize=4,
                     edgecolor='black',
                     linewidth=1,
                     error_kw={'elinewidth': 1, 'capthick': 3})
        
        # Add outlier labels
        for i, (is_outlier, original_val) in enumerate(zip(outlier_flags, means)):
            if is_outlier and not np.isnan(original_val):
                y_pos = clamped_means[i] + clamped_stds[i] + (upper_bound - lower_bound) * 0.02
                ax.text(i, y_pos, f'{original_val:.2f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7,
                                edgecolor='red', linewidth=1))
    
    # Formatting
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    
    # Custom title for utility metrics
    if metric == "r2_score_synth":
        metric_label = "R² Improvement (%) - Model Comparison\n+: better, -: worse"
    elif metric == "rmse_synth":
        metric_label = "RMSE Improvement (%) - Model Comparison\n+: better, -: worse"
    else:
        metric_label = f"{metric.replace('_', ' ').title()} - Model Comparison"
    
    ax.set_title(metric_label, fontsize=13, fontweight='bold', pad=15)
    
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=45, ha='right', fontsize=10)
    
    ax.grid(axis='y', alpha=0.2, linestyle=':')
    
    # Reference lines
    if metric in ["r2_score_synth", "rmse_synth"]:
        ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')
    else:
        ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Outlier bounds
    if lower_bound != 0 or upper_bound != 0:
        ax.axhline(y=lower_bound, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axhline(y=upper_bound, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Set y-axis limits
    if metric in NON_NEGATIVE_METRICS and metric not in ["r2_score_synth", "rmse_synth"]:
        ax.set_ylim(bottom=0)
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, title='Models', title_fontsize=12)
    
    # Footnote
    if outlier_info['is_distribution']:
        footnote = ('Yellow labels: Values outside IQR (relative-scaled for balanced visualization). '
                   'Thresholds prevent extreme outliers from dominating scale. Bars show mean ± std across 5 folds.')
    elif metric in ["r2_score_synth", "rmse_synth"]:
        footnote = ('Yellow labels: Values outside IQR. Green line: baseline (0% difference). '
                   'Positive = synth better, Negative = synth worse. Bars show mean ± std across 5 folds.')
    else:
        footnote = ('Yellow labels: Outlier values (outside IQR bounds). '
                   'Bars show mean ± std across 5 folds.')
    
    fig.text(0.5, 0.01, footnote, ha='center', fontsize=9, style='italic', color='#555555')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main aggregation pipeline."""
    print("=" * 80)
    print("AGGREGATE ALL MODELS - PER-METRIC COMPARISON (UPDATED)")
    print("=" * 80)
    print(f"\nResults root: {RESULTS_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Figures directory: {FIGURES_DIR}\n")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Finding model directories...")
    model_dirs = find_model_dirs()
    
    if not model_dirs:
        print("❌ No model directories found!")
        return
    
    print("\n" + "=" * 80)
    print("LOADING RESULTS")
    print("=" * 80 + "\n")
    
    all_models = load_all_models_results(model_dirs)
    
    if not all_models:
        print("❌ No results loaded!")
        return
    
    # Calculate utility improvements
    print("\nCalculating utility metric improvements...")
    utility_improvements = calculate_utility_improvements(all_models)
    
    # Generate plots for each metric
    print("\n" + "=" * 80)
    print("GENERATING PER-METRIC COMPARISON PLOTS")
    print("=" * 80)
    
    for metric in METRICS_TO_PLOT:
        print(f"\n📊 Plotting {metric}...", end=" ")
        
        means_df, stds_df, outlier_info = prepare_metric_data(
            all_models, metric, utility_improvements
        )
        
        plot_path = FIGURES_DIR / f"comparison_{metric}.png"
        plot_metric_comparison(metric, means_df, stds_df, outlier_info, plot_path)
    
    # Final summary
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE ✅")
    print("=" * 80)
    
    print(f"\nGenerated outputs:")
    print(f" 📊 {len(METRICS_TO_PLOT)} per-metric comparison plots")
    print(f"    - Utility metrics (2): R² Improvement, RMSE Improvement")
    print(f"    - Distribution metrics (6): KS, Wasserstein, MMD, KL, JS, SWD")
    print(f"    - Correlation metrics (3): Similarity, Pairwise Diff, Distance")
    print(f"    - Privacy metrics (4): DCR Mean/Median/Share, Authenticity")
    
    print(f"\nModels included: {len(all_models)}")
    for model_name, model_data in all_models.items():
        print(f"  • {MODELS[model_name]['label']:20s}: {len(model_data)} datasets")
    
    print(f"\nDatasets: {len(DATASETS)}")
    print(f"\nFigure location: {FIGURES_DIR}/")
    
    print(f"\nKey Features:")
    print(f" • Per-metric y-axis with IQR-based outlier detection")
    print(f" • Relative scaling for distribution metrics (prevents crushing)")
    print(f" • Utility metrics show signed improvement vs baseline")
    print(f" • Outliers highlighted with yellow boxes")
    print(f" • ml_efficiency_gap_percent EXCLUDED")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
