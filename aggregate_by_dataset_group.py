#!/usr/bin/env python3
"""
aggregate_all_models_by_dataset_and_group_FINAL.py

FINAL VERSION with correct utility formulas.

Key features:
1. Utility metrics: CORRECT signed value formulas
   - R²: (R²_synth - R²_real) / R²_real * 100%
     → Positive = synth better (higher R²)
     → Negative = synth worse (lower R²)
   - RMSE: (RMSE_real - RMSE_synth) / RMSE_real * 100%
     → Positive = synth better (lower RMSE)
     → Negative = synth worse (higher RMSE)

2. Distribution metrics: Relative outlier thresholds
   - Prevents scale crushing from extreme outliers
   - Keeps small values visible

Usage:
python aggregate_all_models_by_dataset_and_group_FINAL.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_ROOT = Path("results/5fold_metrics")
OUTPUT_DIR = Path("results")
FIGURES_DIR = Path("figures/model_comparison_by_dataset")

DPI = 300

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

METRIC_GROUPS = {
    "Correlation": [
        "correlation_similarity",
        "pairwise_correlation_diff",
        "correlation_distance",
    ],
    "Utility": [
        "r2_score_synth",
        "rmse_synth",
        "ml_efficiency_gap_percent",
    ],
    "Distribution": [
        "ks_statistic_mean",
        "wasserstein_distance",
        "mmd",
        "kl_divergence",
        "js_divergence",
        "swd",
    ],
    "Privacy": [
        "dcr_mean",
        "dcr_median",
        "dcr_share",
        "authenticity",
    ]
}

UTILITY_REAL_SYNTH_PAIRS = {
    "r2_score": ("r2_score_real", "r2_score_synth"),
    "rmse": ("rmse_real", "rmse_synth"),
}

NON_NEGATIVE_METRICS = {
    "correlation_similarity", "pairwise_correlation_diff", "correlation_distance",
    "ks_statistic_mean", "wasserstein_distance", "mmd", "kl_divergence",
    "js_divergence", "swd",
    "dcr_mean", "dcr_median", "dcr_share", "authenticity",
    "rmse_synth", "ml_efficiency_gap_percent"
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

def find_model_dirs() -> Dict[str, Path]:
    """Find all model directories in RESULTS_ROOT."""
    model_dirs = {}
    for model_name in MODELS.keys():
        model_path = RESULTS_ROOT / model_name
        if model_path.exists():
            model_dirs[model_name] = model_path
        else:
            print(f"⚠️  Model dir not found: {model_path}")
    return model_dirs

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
    
    return all_models

# ============================================================================
# UTILITY METRICS TRANSFORMATION
# ============================================================================

def calculate_relative_improvement(all_models: Dict[str, Dict[str, Dict]], dataset_name: str) -> Dict[str, Dict]:
    """
    Calculate relative improvement metrics for utility metrics.
    
    CORRECT FORMULAS:
    
    For R²_synth: (R²_synth - R²_real) / |R²_real| * 100%
      - Positive = R²_synth > R²_real (better)
      - Negative = R²_synth < R²_real (worse)
    
    For RMSE_synth: (RMSE_real - RMSE_synth) / RMSE_real * 100%
      - Positive = RMSE_synth < RMSE_real (better, lower error)
      - Negative = RMSE_synth > RMSE_real (worse, higher error)
    
    Returns: {model_name: {"r2_improvement": value, "rmse_improvement": value}}
    """
    improvements = {}
    
    for model_name in all_models.keys():
        if dataset_name not in all_models[model_name]:
            continue
        
        metrics_dict = all_models[model_name][dataset_name]
        means = metrics_dict['mean']
        stds = metrics_dict['std']
        
        # Calculate R² improvement
        r2_real = means.get("r2_score_real", np.nan)
        r2_synth = means.get("r2_score_synth", np.nan)
        r2_real_std = stds.get("r2_score_real", 0.0)
        r2_synth_std = stds.get("r2_score_synth", 0.0)
        
        if not np.isnan(r2_real) and not np.isnan(r2_synth):
            if r2_real != 0:
                # CORRECTED FORMULA: (synth - real) for proper direction
                r2_improvement = (r2_synth - r2_real) / abs(r2_real) * 100
                r2_improvement_std = np.sqrt((r2_real_std**2 + r2_synth_std**2)) / abs(r2_real) * 100
            else:
                r2_improvement = np.nan
                r2_improvement_std = 0.0
        else:
            r2_improvement = np.nan
            r2_improvement_std = 0.0
        
        # Calculate RMSE improvement
        rmse_real = means.get("rmse_real", np.nan)
        rmse_synth = means.get("rmse_synth", np.nan)
        rmse_real_std = stds.get("rmse_real", 0.0)
        rmse_synth_std = stds.get("rmse_synth", 0.0)
        
        if not np.isnan(rmse_real) and not np.isnan(rmse_synth):
            if rmse_real != 0:
                # Positive = lower RMSE (better), Negative = higher RMSE (worse)
                rmse_improvement = (rmse_real - rmse_synth) / rmse_real * 100
                rmse_improvement_std = np.sqrt((rmse_real_std**2 + rmse_synth_std**2)) / rmse_real * 100
            else:
                rmse_improvement = np.nan
                rmse_improvement_std = 0.0
        else:
            rmse_improvement = np.nan
            rmse_improvement_std = 0.0
        
        improvements[model_name] = {
            "r2_improvement": r2_improvement,
            "r2_improvement_std": r2_improvement_std,
            "rmse_improvement": rmse_improvement,
            "rmse_improvement_std": rmse_improvement_std,
        }
    
    return improvements

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_metrics_group_for_dataset(dataset_name: str, group_name: str, metrics: List[str],
                                   all_models: Dict[str, Dict[str, Dict]],
                                   output_path: Path = None) -> None:
    """Create subplot plot for a metric group and dataset."""
    n_metrics = len(metrics)
    
    # Create subplots
    n_rows = (n_metrics + 2) // 3
    n_cols = min(3, n_metrics)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows), dpi=DPI)
    
    # Ensure axes is always array
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    axes = axes.flatten()
    
    # Calculate relative improvements for utility metrics
    relative_improvements = {}
    if group_name == "Utility":
        relative_improvements = calculate_relative_improvement(all_models, dataset_name)
    
    is_distribution_group = group_name == "Distribution"
    
    # Prepare data for each metric
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        model_names_with_data = []
        model_colors = []
        model_means = []
        model_stds = []
        
        # For utility metrics, use relative improvement
        if group_name == "Utility":
            for model_name in sorted(MODELS.keys()):
                if model_name not in relative_improvements:
                    continue
                
                improv_data = relative_improvements[model_name]
                
                if metric == "r2_score_synth":
                    mean_val = improv_data.get("r2_improvement", np.nan)
                    std_val = improv_data.get("r2_improvement_std", 0.0)
                elif metric == "rmse_synth":
                    mean_val = improv_data.get("rmse_improvement", np.nan)
                    std_val = improv_data.get("rmse_improvement_std", 0.0)
                else:
                    if dataset_name in all_models[model_name]:
                        metrics_dict = all_models[model_name][dataset_name]
                        mean_val = metrics_dict['mean'].get(metric, np.nan)
                        std_val = metrics_dict['std'].get(metric, 0.0)
                    else:
                        continue
                
                if not np.isnan(mean_val):
                    model_names_with_data.append(model_name)
                    model_colors.append(MODELS[model_name]['color'])
                    model_means.append(mean_val)
                    model_stds.append(std_val)
        else:
            # Non-utility metrics: use raw values
            for model_name in sorted(MODELS.keys()):
                if model_name in all_models and dataset_name in all_models[model_name]:
                    metrics_dict = all_models[model_name][dataset_name]
                    mean_val = metrics_dict['mean'].get(metric, np.nan)
                    std_val = metrics_dict['std'].get(metric, 0.0)
                    
                    model_names_with_data.append(model_name)
                    model_colors.append(MODELS[model_name]['color'])
                    model_means.append(mean_val)
                    model_stds.append(std_val)
        
        if not model_names_with_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
            continue
        
        model_means = np.array(model_means)
        model_stds = np.array(model_stds)
        
        # Detect outliers
        lower_bound, upper_bound = detect_outlier_bounds(model_means, is_distribution=is_distribution_group)
        
        if metric in NON_NEGATIVE_METRICS and not is_distribution_group:
            lower_bound = max(lower_bound, 0.0)
        
        # Clamp values
        clamped_means = []
        outlier_flags = []
        clamped_stds = []
        
        for mean_val, std_val in zip(model_means, model_stds):
            clamped, is_outlier = clamp_value_with_flag(mean_val, lower_bound, upper_bound)
            clamped_means.append(clamped)
            outlier_flags.append(is_outlier)
            
            max_std = abs(upper_bound - lower_bound) * 0.1
            clamped_stds.append(min(std_val, max_std))
        
        clamped_means = np.array(clamped_means)
        clamped_stds = np.array(clamped_stds)
        
        # Plot bars
        x_pos = np.arange(len(model_names_with_data))
        bar_width = 0.6 / len(model_names_with_data)
        
        bars = ax.bar(x_pos, clamped_means, bar_width,
                     yerr=clamped_stds,
                     color=model_colors,
                     alpha=0.8,
                     capsize=4,
                     edgecolor='black',
                     linewidth=1,
                     error_kw={'elinewidth': 1, 'capthick': 3})
        
        # Add outlier labels
        for i, (is_outlier, original_val) in enumerate(zip(outlier_flags, model_means)):
            if is_outlier and not np.isnan(original_val):
                y_pos = clamped_means[i] + clamped_stds[i] + (upper_bound - lower_bound) * 0.02
                ax.text(i, y_pos, f'{original_val:.2f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7,
                                edgecolor='red', linewidth=1))
        
        # Add missing data markers
        for i, model_name in enumerate(model_names_with_data):
            if np.isnan(model_means[i]):
                ax.text(i, 0.01, '✗', ha='center', va='bottom', fontsize=12, color='red')
        
        # Reference line for utility metrics
        if group_name == "Utility" and metric in ["r2_score_synth", "rmse_synth"]:
            ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7,
                      label='Baseline (0% difference)')
            ax.legend(loc='upper right', fontsize=8)
        
        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels([MODELS[m]['label'] for m in model_names_with_data],
                           rotation=0, fontsize=9)
        
        # Custom title for utility metrics
        if group_name == "Utility":
            if metric == "r2_score_synth":
                metric_label = "R² Improvement (%)\n+: better, -: worse"
            elif metric == "rmse_synth":
                metric_label = "RMSE Improvement (%)\n+: better, -: worse"
            else:
                metric_label = metric.replace('_', ' ').title()
        else:
            metric_label = metric.replace('_', ' ').title()
        
        ax.set_ylabel('Value', fontsize=9)
        ax.set_title(metric_label, fontweight='bold', fontsize=10)
        
        # Add outlier bounds
        if not np.all(np.isnan(model_means)):
            ax.axhline(y=lower_bound, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            ax.axhline(y=upper_bound, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        
        # Set y-axis limits
        if metric in NON_NEGATIVE_METRICS and not is_distribution_group:
            ax.set_ylim(bottom=0)
        
        if is_distribution_group:
            ax.set_ylim(bottom=0)
        
        ax.grid(axis='y', alpha=0.2, linestyle=':')
        
        if group_name == "Utility":
            ax.axhline(y=0, color='black', linewidth=0.5)
        elif not is_distribution_group:
            ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    # Add title and legend
    fig.suptitle(f'{dataset_name.replace("_", " ").title()} - {group_name} Metrics',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Create legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=MODELS[m]['color'], alpha=0.8,
                                    edgecolor='black', linewidth=1, label=MODELS[m]['label'])
                      for m in sorted(MODELS.keys()) if m in all_models]
    
    fig.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95,
              title='Models', title_fontsize=12, bbox_to_anchor=(0.98, 0.98))
    
    # Footnote
    if group_name == "Utility":
        footnote = ('Yellow labels: Values outside IQR. + positive = synth better than real, - negative = synth worse. '
                   'Green line at 0: baseline. Bars show mean ± std across 5 folds.')
    elif group_name == "Distribution":
        footnote = ('Yellow labels: Values outside IQR bounds (relative-scaled for balanced visualization). '
                   'Thresholds prevent extreme outliers from dominating scale. Bars show mean ± std across 5 folds.')
    else:
        footnote = ('Yellow labels: Outlier values (clamped to IQR bounds for visualization). '
                   'Red dashed lines: Outlier detection thresholds. Bars show mean ± std across 5 folds.')
    
    fig.text(0.5, 0.01, footnote,
            ha='center', fontsize=9, style='italic', color='#555555')
    
    plt.tight_layout(rect=[0, 0.02, 0.97, 0.99])
    
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
    print("AGGREGATE MODELS BY DATASET AND METRIC GROUP (FINAL)")
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
    
    print(f"✓ Found {len(model_dirs)} model(s):\n")
    for model_name in sorted(model_dirs.keys()):
        print(f"  • {MODELS[model_name]['label']:20s} ({model_name})")
    
    print("\n" + "=" * 80)
    print("LOADING RESULTS")
    print("=" * 80)
    
    all_models = load_all_models_results(model_dirs)
    
    if not all_models:
        print("❌ No results loaded!")
        return
    
    print(f"\n✓ Loaded results for {len(all_models)} model(s)")
    for model_name, model_data in all_models.items():
        print(f"  • {MODELS[model_name]['label']:20s}: {len(model_data)} dataset(s)")
    
    print("\n" + "=" * 80)
    print("GENERATING METRIC GROUP PLOTS")
    print("=" * 80)
    
    total_plots = 0
    
    for dataset_name in DATASETS:
        print(f"\n{dataset_name.upper()}")
        print("-" * 80)
        
        for group_name, metrics in METRIC_GROUPS.items():
            has_data = False
            for model_name in all_models.keys():
                if dataset_name in all_models[model_name]:
                    has_data = True
                    break
            
            if not has_data:
                print(f"  ⚠️  {group_name:15s}: No data for dataset")
                continue
            
            output_path = FIGURES_DIR / f"{dataset_name}_{group_name.lower()}.png"
            
            print(f"  📊 {group_name:15s}: {len(metrics)} metrics", end=" ... ")
            
            plot_metrics_group_for_dataset(dataset_name, group_name, metrics, all_models, output_path)
            
            total_plots += 1
    
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE ✅")
    print("=" * 80)
    
    n_metric_groups = len(METRIC_GROUPS)
    n_datasets_with_data = sum(1 for ds in DATASETS 
                              if any(ds in all_models[m] for m in all_models.keys()))
    
    print(f"\nGenerated outputs:")
    print(f" 📊 {total_plots} figures")
    print(f"    - {n_metric_groups} metric groups: {', '.join(METRIC_GROUPS.keys())}")
    print(f"    - {n_datasets_with_data} datasets with data")
    
    print(f"\nFigure location: {FIGURES_DIR}/")
    
    print(f"\nKey Features:")
    print(f" • Utility R²: (R²_synth - R²_real) / R²_real * 100%")
    print(f"   Positive = synth better (higher R²)")
    print(f" • Utility RMSE: (RMSE_real - RMSE_synth) / RMSE_real * 100%")
    print(f"   Positive = synth better (lower RMSE)")
    print(f" • Distribution: Relative outlier thresholds (prevents scale crushing)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
