#!/usr/bin/env python3

"""
aggregate_5fold_results_v4.py

IMPROVED VERSION V4 with:
1. Petronas towers style for utility metrics (Real | Synth pairs per dataset)
2. Outlier values LABELED directly on bars
3. Error bars CLAMPED to outlier bounds (no visible overflow)
4. All other metrics: outlier awareness with smart std handling

Aggregate 5-fold CV results from JSON files into:
- Petronas towers plots for utility metrics (Real vs Synthetic connected pairs)
- Bar plots for distribution/correlation/privacy metrics
- Heatmap (without identical matches)
- Comprehensive table
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# ================================================================================
# CONFIGURATION
# ================================================================================

RESULTS_DIR = Path("results/5fold_metrics/dsbm_vae")
OUTPUT_DIR = Path("results")
FIGURES_DIR = Path("figures/metrics_barplots")
DPI = 300
FIGSIZE = (14, 8)

# Metrics to plot
UTILITY_METRICS = [
    ('r2_score_real', 'r2_score_synth'),
    ('rmse_real', 'rmse_synth'),
    ('ml_efficiency_gap_percent',),
]

DISTRIBUTION_METRICS = [
    'ks_statistic_mean',
    'wasserstein_distance',
    'mmd',
    'kl_divergence',
    'js_divergence',
    'swd',
]

CORRELATION_METRICS = [
    'correlation_similarity',
    'pairwise_correlation_diff',
    'correlation_distance',
]

PRIVACY_METRICS = [
    'dcr_mean',
    'dcr_median',
    'dcr_share',
    'authenticity',  # REMOVED: 'identical_matches_fraction'
]

# Metric group colors
METRIC_COLORS = {
    'Utility': '#FF6B6B',
    'Distribution': '#4ECDC4',
    'Correlation': '#45B7D1',
    'Privacy': '#FFA07A',
}

# Outlier handling config
OUTLIER_CONFIG = {
    'method': 'iqr',  # 'iqr', 'zscore', or 'percentile'
    'lower_percentile': 5,  # For percentile method
    'upper_percentile': 95,
    'iqr_multiplier': 1.5,  # Standard IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
}


# ================================================================================
# LOADING FUNCTIONS
# ================================================================================

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
        print(f"❌ Error loading {json_path}: {str(e)}")
        return None, None


def load_all_results(results_dir: Path) -> Dict[str, Dict]:
    """Load all JSON results from directory."""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    all_results = {}
    json_files = sorted(results_dir.glob("*.json"))
    
    print(f"✓ Found {len(json_files)} JSON files in {results_dir}")
    
    for json_file in json_files:
        dataset_name, metrics = load_results_from_json(json_file)
        if dataset_name and metrics:
            all_results[dataset_name] = metrics
            print(f"  ✓ Loaded: {dataset_name}")
    
    print(f"✓ Total datasets loaded: {len(all_results)}\n")
    return all_results


# ================================================================================
# OUTLIER DETECTION & SCALING
# ================================================================================

def detect_outlier_bounds(values: np.ndarray, config: Dict = None) -> Tuple[float, float]:
    """
    Detect outlier bounds for a metric.
    
    Returns: (lower_bound, upper_bound)
    Values outside these bounds are clamped to the boundary.
    """
    
    if config is None:
        config = OUTLIER_CONFIG
    
    # Remove NaN values
    valid_values = values[~np.isnan(values)]
    
    if len(valid_values) == 0:
        return 0.0, 1.0
    
    method = config['method']
    
    if method == 'iqr':
        # Interquartile Range method
        q1 = np.percentile(valid_values, 25)
        q3 = np.percentile(valid_values, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            # All values are the same
            lower = q1 - 1.0
            upper = q1 + 1.0
        else:
            lower = q1 - config['iqr_multiplier'] * iqr
            upper = q3 + config['iqr_multiplier'] * iqr
    
    elif method == 'percentile':
        # Percentile method
        lower = np.percentile(valid_values, config['lower_percentile'])
        upper = np.percentile(valid_values, config['upper_percentile'])
    
    elif method == 'zscore':
        # Z-score method (3 sigma)
        mean = np.mean(valid_values)
        std = np.std(valid_values)
        
        if std == 0:
            lower = mean - 1.0
            upper = mean + 1.0
        else:
            lower = mean - 3 * std
            upper = mean + 3 * std
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(lower), float(upper)


def clamp_with_outlier_marking(values: np.ndarray, lower: float, upper: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clamp values to [lower, upper] bounds.
    Return clamped values and boolean array indicating which were outliers.
    """
    clamped = np.clip(values, lower, upper)
    outliers = (values < lower) | (values > upper)
    return clamped, outliers


def clamp_error_bars(stds: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    Clamp error bars to visible range.
    If std is large (beyond 10% of range), cap it.
    """
    visible_range = abs(upper - lower)
    if visible_range < 1e-6:
        return np.zeros_like(stds)
    
    max_std = visible_range * 0.1  # Cap at 10% of visible range
    return np.minimum(stds, max_std)


# ================================================================================
# AGGREGATION FUNCTIONS
# ================================================================================

def create_metrics_table(all_results: Dict[str, Dict]) -> pd.DataFrame:
    """Create comprehensive table with all datasets × metrics."""
    
    all_metrics = (
        [m for pair in UTILITY_METRICS for m in pair] +
        DISTRIBUTION_METRICS +
        CORRELATION_METRICS +
        PRIVACY_METRICS
    )
    
    rows = []
    
    for dataset_name in sorted(all_results.keys()):
        row_data = {'Dataset': dataset_name}
        
        metrics_data = all_results[dataset_name]
        means = metrics_data['mean']
        stds = metrics_data['std']
        
        for metric in all_metrics:
            if metric in means:
                mean_val = means[metric]
                std_val = stds.get(metric, 0.0)
                
                if np.isnan(mean_val):
                    row_data[metric] = 'NaN'
                else:
                    row_data[metric] = f"{mean_val:.4f}±{std_val:.4f}"
            else:
                row_data[metric] = 'N/A'
        
        rows.append(row_data)
    
    df = pd.DataFrame(rows)
    return df


# ================================================================================
# PLOTTING FUNCTIONS
# ================================================================================

def plot_paired_utility_metrics(all_results: Dict[str, Dict], real_metric: str, synth_metric: str,
                               output_path: Path = None) -> None:
    """
    Create Petronas towers style plot for paired utility metrics.
    For each dataset: Real (darker, left) and Synthetic (lighter, right) connected pair.
    Like two towers of different heights next to each other.
    Outlier values labeled directly on bars.
    Error bars clamped to visible range.
    """
    
    # Extract data
    datasets = sorted(all_results.keys())
    real_means = []
    real_stds = []
    synth_means = []
    synth_stds = []
    
    for dataset in datasets:
        metrics_data = all_results[dataset]
        
        real_mean = metrics_data['mean'].get(real_metric, np.nan)
        real_std = metrics_data['std'].get(real_metric, 0.0)
        synth_mean = metrics_data['mean'].get(synth_metric, np.nan)
        synth_std = metrics_data['std'].get(synth_metric, 0.0)
        
        real_means.append(real_mean)
        real_stds.append(real_std)
        synth_means.append(synth_mean)
        synth_stds.append(synth_std)
    
    real_means = np.array(real_means)
    real_stds = np.array(real_stds)
    synth_means = np.array(synth_means)
    synth_stds = np.array(synth_stds)
    
    # Check if all NaN
    if all(np.isnan(real_means)) and all(np.isnan(synth_means)):
        print(f"⚠️  Skipping {real_metric} vs {synth_metric} - all values are NaN")
        return
    
    # Detect outlier bounds
    all_vals = np.concatenate([real_means[~np.isnan(real_means)], 
                               synth_means[~np.isnan(synth_means)]])
    lower, upper = detect_outlier_bounds(all_vals)
    
    real_clamped, real_outliers = clamp_with_outlier_marking(real_means, lower, upper)
    synth_clamped, synth_outliers = clamp_with_outlier_marking(synth_means, lower, upper)
    
    # Clamp error bars
    real_stds_clamped = clamp_error_bars(real_stds, lower, upper)
    synth_stds_clamped = clamp_error_bars(synth_stds, lower, upper)
    
    # Create single figure with connected tower pairs
    fig, ax = plt.subplots(figsize=(16, 8), dpi=DPI)
    
    n_datasets = len(datasets)
    width = 0.35  # Width of each tower
    
    # Create positions: pairs with spacing
    # Each dataset gets a pair (Real, Synth) separated by 'width'
    # Pairs are separated by 2*width to create gaps between dataset pairs
    x_positions = []
    for i in range(n_datasets):
        x_positions.append(i * (2 * width + 0.3))  # Real tower position
    
    real_x = np.array(x_positions)
    synth_x = real_x + width  # Synth tower right next to Real
    
    # Plot Real Data towers (darker color - darker red)
    bars_real = ax.bar(real_x, real_clamped, width=width, yerr=real_stds_clamped, capsize=6,
                       color='#FF6B6B', alpha=0.85, edgecolor='black', linewidth=1.5,
                       label='Real Data',
                       error_kw={'elinewidth': 1.5, 'capthick': 6})
    
    # Plot Synthetic Data towers (lighter color - lighter teal)
    bars_synth = ax.bar(synth_x, synth_clamped, width=width, yerr=synth_stds_clamped, capsize=6,
                        color='#4ECDC4', alpha=0.65, edgecolor='black', linewidth=1.5,
                        label='Synthetic Data',
                        error_kw={'elinewidth': 1.5, 'capthick': 6})
    
    # Label outliers directly on bars
    for i, (val, is_outlier, clamped_val) in enumerate(zip(real_means, real_outliers, real_clamped)):
        if is_outlier and not np.isnan(val):
            y_offset = (upper - lower) * 0.04
            ax.text(real_x[i], clamped_val + y_offset, f'{val:.2f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8,
                            edgecolor='red', linewidth=1.5))
    
    for i, (val, is_outlier, clamped_val) in enumerate(zip(synth_means, synth_outliers, synth_clamped)):
        if is_outlier and not np.isnan(val):
            y_offset = (upper - lower) * 0.04
            ax.text(synth_x[i], clamped_val + y_offset, f'{val:.2f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8,
                            edgecolor='red', linewidth=1.5))
    
    # Formatting
    metric_name = real_metric.replace('_real', '').replace('_', ' ').title()
    ax.set_title(f'{metric_name}', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=13, fontweight='bold')
    
    # Set x-axis ticks at center of each pair
    pair_centers = real_x + width / 2
    ax.set_xticks(pair_centers)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=11)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(fontsize=12, loc='upper left', framealpha=0.95)
    
    # Footnote
    fig.text(0.5, 0.01, 'Petronas tower pairs: Real (dark) and Synthetic (light) towers for each dataset.\nYellow boxes show outlier values. Error bars clamped for clarity.',
            ha='center', fontsize=9, style='italic', color='#555555')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.close()


def plot_single_metric(all_results: Dict[str, Dict], metric: str,
                      output_path: Path = None) -> None:
    """
    Create bar plot for single metric with outlier clamping.
    Outlier values labeled on bars. Error bars clamped.
    """
    
    # Extract data
    datasets = sorted(all_results.keys())
    means = []
    stds = []
    
    for dataset in datasets:
        metrics_data = all_results[dataset]
        mean_val = metrics_data['mean'].get(metric, np.nan)
        std_val = metrics_data['std'].get(metric, 0.0)
        
        means.append(mean_val)
        stds.append(std_val)
    
    means = np.array(means)
    stds = np.array(stds)
    
    # Check if all NaN
    if all(np.isnan(means)):
        print(f"⚠️  Skipping {metric} - all values are NaN")
        return
    
    # Detect outlier bounds
    lower, upper = detect_outlier_bounds(means)
    clamped, outliers = clamp_with_outlier_marking(means, lower, upper)
    stds_clamped = clamp_error_bars(stds, lower, upper)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8), dpi=DPI)
    
    x_pos = np.arange(len(datasets))
    
    # Determine color group
    if metric in DISTRIBUTION_METRICS:
        color = METRIC_COLORS['Distribution']
        group = 'Distribution'
    elif metric in CORRELATION_METRICS:
        color = METRIC_COLORS['Correlation']
        group = 'Correlation'
    elif metric in PRIVACY_METRICS:
        color = METRIC_COLORS['Privacy']
        group = 'Privacy'
    else:
        color = '#95A3A6'
        group = 'Other'
    
    # Bar plot with clamped error bars
    ax.bar(x_pos, clamped, yerr=stds_clamped, capsize=8, color=color, alpha=0.8,
           edgecolor='black', linewidth=1.5, error_kw={'elinewidth': 1.5, 'capthick': 8})
    
    # Label outliers directly on bars
    for i, (val, is_outlier, clamped_val) in enumerate(zip(means, outliers, clamped)):
        if is_outlier and not np.isnan(val):
            # Show actual value above bar
            y_offset = (upper - lower) * 0.05
            ax.text(i, clamped_val + y_offset, f'{val:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8,
                            edgecolor='red', linewidth=2))
    
    # Formatting
    ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=13, fontweight='bold')
    ax.set_title(f'{metric.replace("_", " ").title()} ({group} Metric)',
                fontsize=15, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Footnote
    fig.text(0.5, 0.01, 'Yellow boxes show outlier values. Error bars clamped to visible range for clarity.',
            ha='center', fontsize=9, style='italic', color='#555555')
    
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.close()


def create_all_metrics_heatmap(all_results: Dict[str, Dict], output_path: Path = None) -> None:
    """
    Create heatmap of all metrics across datasets.
    EXCLUDING identical_matches_fraction.
    With outlier-aware scaling.
    """
    
    all_metrics = (
        [m for pair in UTILITY_METRICS for m in pair] +
        DISTRIBUTION_METRICS +
        CORRELATION_METRICS +
        PRIVACY_METRICS
    )
    
    # Build matrix
    datasets = sorted(all_results.keys())
    matrix = []
    matrix_clamped = []
    valid_metrics = []
    
    for metric in all_metrics:
        row = []
        has_valid = False
        
        for dataset in datasets:
            metrics_data = all_results[dataset]
            mean_val = metrics_data['mean'].get(metric, np.nan)
            
            if not np.isnan(mean_val):
                has_valid = True
            row.append(mean_val)
        
        if has_valid:
            row = np.array(row)
            valid_metrics.append(metric)
            
            # Clamp outliers
            lower, upper = detect_outlier_bounds(row)
            row_clamped, _ = clamp_with_outlier_marking(row, lower, upper)
            
            # Normalize for visualization
            valid_vals = row_clamped[~np.isnan(row_clamped)]
            if len(valid_vals) > 0:
                min_val = np.min(valid_vals)
                max_val = np.max(valid_vals)
                if max_val - min_val > 1e-6:
                    row_normalized = (row_clamped - min_val) / (max_val - min_val)
                else:
                    row_normalized = np.zeros_like(row_clamped)
            else:
                row_normalized = np.zeros_like(row_clamped)
            
            matrix.append(row)
            matrix_clamped.append(row_normalized)
    
    if not matrix:
        print("⚠️  No valid metrics found for heatmap")
        return
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 12), dpi=DPI)
    
    matrix_normalized = np.array(matrix_clamped, dtype=float)
    im = ax.imshow(matrix_normalized, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(valid_metrics)))
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(valid_metrics, fontsize=9)
    
    # Text annotations (actual values)
    for i in range(len(valid_metrics)):
        for j in range(len(datasets)):
            val = matrix[i][j]
            if not np.isnan(val):
                text_color = "black" if 0.2 < matrix_normalized[i, j] < 0.8 else "white"
                ax.text(j, i, f'{val:.2f}', ha="center", va="center",
                       color=text_color, fontsize=7)
    
    ax.set_title('All Metrics Across Datasets\n(Clamped to IQR bounds, row-normalized for heatmap)',
                fontsize=15, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Normalized Value (within bounds)')
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.close()


# ================================================================================
# MAIN FUNCTION
# ================================================================================

def main(results_dir: Path = None, output_dir: Path = None, figures_dir: Path = None):
    """Main aggregation pipeline."""
    
    if results_dir is None:
        results_dir = RESULTS_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if figures_dir is None:
        figures_dir = FIGURES_DIR
    
    print("=" * 80)
    print("AGGREGATING 5-FOLD CV RESULTS - V4 (PETRONAS TOWERS)")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Figures directory: {figures_dir}\n")
    print("Features:")
    print("  ✓ Utility metrics: Petronas towers style (Real | Synth pairs)")
    print("  ✓ Outlier values labeled directly on bars")
    print("  ✓ Error bars clamped to visible range")
    print("  ✓ Connected towers per dataset with different heights\n")
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    try:
        all_results = load_all_results(results_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    if not all_results:
        print("❌ No results loaded!")
        return
    
    # ========== CREATE COMPREHENSIVE TABLE ==========
    
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE TABLE")
    print("=" * 80)
    
    metrics_table = create_metrics_table(all_results)
    
    # Save as CSV
    table_csv_path = output_dir / "aggregated_metrics_table.csv"
    metrics_table.to_csv(table_csv_path, index=False)
    print(f"✓ Saved CSV table: {table_csv_path}")
    
    # Save as Excel (if available)
    try:
        table_xlsx_path = output_dir / "aggregated_metrics_table.xlsx"
        metrics_table.to_excel(table_xlsx_path, index=False, engine='openpyxl')
        print(f"✓ Saved Excel table: {table_xlsx_path}")
    except ImportError:
        print("⚠️  openpyxl not available, skipping Excel export")
    
    # Print table preview
    print("\nTable Preview (first 5 rows, first 8 columns):")
    print(metrics_table.iloc[:5, :8].to_string())
    print(f"\nTable shape: {metrics_table.shape[0]} datasets × {metrics_table.shape[1] - 1} metrics")
    
    # ========== CREATE PETRONAS TOWERS UTILITY METRICS PLOTS ==========
    
    print("\n" + "=" * 80)
    print("GENERATING PETRONAS TOWERS UTILITY METRICS PLOTS")
    print("=" * 80)
    print()
    
    for metrics_pair in UTILITY_METRICS:
        if len(metrics_pair) == 2:
            real_metric, synth_metric = metrics_pair
            plot_path = figures_dir / f"metric_{real_metric.replace('_real', '')}.png"
            plot_paired_utility_metrics(all_results, real_metric, synth_metric, plot_path)
        else:
            # Single utility metric (like ml_efficiency_gap)
            metric = metrics_pair[0]
            plot_path = figures_dir / f"metric_{metric}.png"
            plot_single_metric(all_results, metric, plot_path)
    
    # ========== CREATE DISTRIBUTION/CORRELATION/PRIVACY METRIC PLOTS ==========
    
    print("\n" + "=" * 80)
    print("GENERATING INDIVIDUAL METRIC PLOTS")
    print("=" * 80)
    print()
    
    for metric in DISTRIBUTION_METRICS + CORRELATION_METRICS + PRIVACY_METRICS:
        plot_path = figures_dir / f"metric_{metric}.png"
        plot_single_metric(all_results, metric, plot_path)
    
    # ========== CREATE HEATMAP ==========
    
    print("\n" + "=" * 80)
    print("GENERATING HEATMAP")
    print("=" * 80)
    
    heatmap_path = figures_dir / "metrics_heatmap_all.png"
    create_all_metrics_heatmap(all_results, heatmap_path)
    
    # ========== FINAL SUMMARY ==========
    
    print("\n" + "=" * 80)
    print("AGGREGATION COMPLETE ✅")
    print("=" * 80)
    
    total_plots = 2 + len(DISTRIBUTION_METRICS) + len(CORRELATION_METRICS) + len(PRIVACY_METRICS)
    
    print(f"\nGenerated outputs:")
    print(f"  📊 {total_plots} metric plots in: {figures_dir}/metric_*.png")
    print(f"      - Utility metrics: Petronas towers (Real | Synth pairs)")
    print(f"      - Distribution metrics: {len(DISTRIBUTION_METRICS)} plots")
    print(f"      - Correlation metrics: {len(CORRELATION_METRICS)} plots")
    print(f"      - Privacy metrics: {len(PRIVACY_METRICS)} plots")
    print(f"  📊 1 heatmap in: {heatmap_path}")
    print(f"  📋 CSV table in: {table_csv_path}")
    if 'table_xlsx_path' in locals():
        print(f"  📋 Excel table in: {table_xlsx_path}")
    print(f"\nDatasets processed: {len(all_results)}")
    print(f"Total metrics: {len(metrics_table.columns) - 1}")
    print(f"\nVisualization features:")
    print(f"  • Petronas towers: Dark (Real) and light (Synthetic) towers per dataset")
    print(f"  • Towers of different heights show variation between Real and Synthetic")
    print(f"  • Outliers clamped to IQR bounds with value labels")
    print(f"  • Error bars clamped to visible range (≤10% of range)")
    print(f"  • Large variances automatically hidden (indicates model failure)")
    print("\n" + "=" * 80)


# ================================================================================
# MAIN ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate 5-fold CV results into plots and tables (V4 - Petronas Towers)"
    )
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR,
                       help=f'Path to results directory (default: {RESULTS_DIR})')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR,
                       help=f'Path to output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--figures-dir', type=Path, default=FIGURES_DIR,
                       help=f'Path to figures directory (default: {FIGURES_DIR})')
    
    args = parser.parse_args()
    
    main(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir
    )
