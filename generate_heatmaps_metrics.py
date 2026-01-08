#!/usr/bin/env python3
"""
Generate Heatmaps from Evaluation Results

Creates two main visualizations:
1. METRICS VALUES HEATMAP: Rows = Models, Columns = Metrics (averaged across datasets)
2. METRICS STD HEATMAP: Rows = Models, Columns = Metrics (averaged stds across datasets)

Plus per-dataset bar plots showing each metric value across models.

Color scheme:
- Green: Excellent (best values)
- Yellow: Good
- Red: Poor (worst values)

Usage:
    python generate_presentation_heatmaps.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = Path("results/5fold_metrics")
OUTPUT_DIR = Path("figures/presentation_metrics")

DATASETS = [
    "california_housing",
    "diabetes",
    "king_county_housing",
    "adult_numeric",
    "bank_marketing",
    "online_shoppers",
    # "covertype",
    "german_credit"
]

MODELS = ["dsbm", "asbm", "dsbmxgboost"]
MODEL_LABELS = {
    "dsbm": "DSBM",
    "asbm": "ASBM",
    "dsbmxgboost": "XGBoost-DSBM"
}

METRICS_TO_PLOT = {
    "r2_score_synth": {
        "label": "R²",
        "direction": "higher_is_better",
        "bounds": (0.5, 1.0)
    },
    "rmse_synth": {
        "label": "RMSE",
        "direction": "lower_is_better",
        "bounds": (0, 1)
    },
    "correlation_distance": {
        "label": "Corr. Dist.",
        "direction": "lower_is_better",
        "bounds": (0, 2)
    },
    "kl_divergence": {
        "label": "KL Div.",
        "direction": "lower_is_better",
        "bounds": (0, 0.2)
    },
    "wasserstein_distance": {
        "label": "Wasserstein",
        "direction": "lower_is_better",
        "bounds": (0, 0.2)
    }
}

DPI = 300
FIGSIZE_HEATMAP = (14, 6)
FIGSIZE_BARS = (14, 7)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_json_results(model: str, dataset: str) -> Dict:
    """Load JSON results for a model and dataset."""
    file_paths = [
        RESULTS_DIR / model / f"results_{model.upper()}_{dataset}.json",
        RESULTS_DIR / model / f"results_DSBM_{dataset}.json",
        RESULTS_DIR / model / f"results_ASBM_{dataset}.json",
        RESULTS_DIR / model / f"results_XGBOOSTDSBM_{dataset}.json",
        RESULTS_DIR / model / f"results_{dataset}.json"
    ]
    
    for path in file_paths:
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
    
    return None

def normalize_score(value: float, direction: str, bounds: Tuple[float, float]) -> float:
    """
    Normalize score to [0, 1] range for color mapping.
    
    direction: "higher_is_better" or "lower_is_better"
    bounds: (min, max) for clipping
    """
    lower, upper = bounds
    value = np.clip(value, lower, upper)
    
    if direction == "higher_is_better":
        # Higher is better: map [lower, upper] → [0, 1]
        normalized = (value - lower) / (upper - lower)
    else:
        # Lower is better: map [lower, upper] → [1, 0]
        normalized = (upper - value) / (upper - lower)
    
    return normalized

def get_cmap():
    """Get colormap: red (bad) → yellow (ok) → green (good)"""
    return sns.color_palette("RdYlGn", as_cmap=True)

# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_all_metrics_for_models() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract all metric values and stds for all models.
    
    Averages each metric across all datasets.
    
    Returns:
    - means_df: shape (n_models, n_metrics) - averaged metric values
    - stds_df: shape (n_models, n_metrics) - averaged metric stds
    """
    means_data = {model: {} for model in MODEL_LABELS.values()}
    stds_data = {model: {} for model in MODEL_LABELS.values()}
    
    print("Extracting metrics...")
    for metric_name in METRICS_TO_PLOT.keys():
        for model in MODELS:
            mean_values = []
            std_values = []
            
            for dataset in DATASETS:
                results = load_json_results(model, dataset)
                
                if results is None:
                    continue
                
                metric_mean = results['summary']['metrics_mean'].get(metric_name, np.nan)
                metric_std = results['summary']['metrics_std'].get(metric_name, np.nan)
                mean_values.append(metric_mean)
                std_values.append(metric_std)
            
            # Average across datasets
            mean_avg = np.nanmean(mean_values) if mean_values else np.nan
            std_avg = np.nanmean(std_values) if std_values else np.nan
            
            means_data[MODEL_LABELS[model]][METRICS_TO_PLOT[metric_name]["label"]] = mean_avg
            stds_data[MODEL_LABELS[model]][METRICS_TO_PLOT[metric_name]["label"]] = std_avg
    
    means_df = pd.DataFrame(means_data).T
    stds_df = pd.DataFrame(stds_data).T
    
    return means_df, stds_df

def extract_metrics_per_dataset() -> Dict[str, pd.DataFrame]:
    """
    Extract metrics for each dataset separately.
    
    Returns dict: dataset_name → DataFrame(models × metrics)
    """
    data = {}
    
    for dataset in DATASETS:
        dataset_data = {}
        
        for model in MODELS:
            model_metrics = {}
            
            for metric_name in METRICS_TO_PLOT.keys():
                results = load_json_results(model, dataset)
                
                if results is None:
                    model_metrics[METRICS_TO_PLOT[metric_name]["label"]] = np.nan
                else:
                    metric_value = results['summary']['metrics_mean'].get(metric_name, np.nan)
                    model_metrics[METRICS_TO_PLOT[metric_name]["label"]] = metric_value
            
            dataset_data[MODEL_LABELS[model]] = model_metrics
        
        data[dataset] = pd.DataFrame(dataset_data).T
    
    return data

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_main_heatmap_means() -> Path:
    """
    Create main heatmap: Rows = Models, Columns = Metrics (values).
    
    Shows averaged metric values across all datasets.
    """
    means_df, _ = extract_all_metrics_for_models()
    
    # Normalize for color mapping
    normalized_df = means_df.copy()
    for col_idx, metric_name in enumerate(METRICS_TO_PLOT.keys()):
        col = METRICS_TO_PLOT[metric_name]["label"]
        metric_info = METRICS_TO_PLOT[metric_name]
        normalized_df[col] = means_df[col].apply(
            lambda x: normalize_score(x, metric_info["direction"], metric_info["bounds"])
            if not np.isnan(x) else np.nan
        )
    
    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP, dpi=DPI)
    
    # Create heatmap with values displayed
    sns.heatmap(means_df, annot=True, fmt=".4f", cmap=get_cmap(),
               cbar_kws={"label": "Normalized Score (Green=Best, Red=Worst)"},
               linewidths=2, linecolor="black",
               ax=ax, vmin=0, vmax=1, square=False,
               cbar=True, annot_kws={"fontsize": 11, "fontweight": "bold"})
    
    ax.set_title("Metrics Values Heatmap (Averaged Across 8 Datasets)\nRows: Models | Columns: Metrics",
                fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Models", fontsize=12, fontweight="bold")
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=11)
    
    # Add legend
    fig.text(0.02, 0.98, "Green: Better | Red: Worse", fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "heatmap_metrics_values.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    return output_path

def plot_main_heatmap_stds() -> Path:
    """
    Create main heatmap: Rows = Models, Columns = Metrics (stds).
    
    Shows averaged metric standard deviations across all datasets.
    """
    _, stds_df = extract_all_metrics_for_models()
    
    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP, dpi=DPI)
    
    # For stds: lower is better (less variance)
    # Normalize: [0, max_std] → [1, 0] (so lower stds are greener)
    normalized_stds = stds_df.copy()
    for col in stds_df.columns:
        max_std = stds_df[col].max()
        if max_std > 0:
            normalized_stds[col] = (max_std - stds_df[col]) / max_std
        else:
            normalized_stds[col] = 0
    
    # Create heatmap
    sns.heatmap(stds_df, annot=True, fmt=".5f", cmap=get_cmap(),
               cbar_kws={"label": "Normalized Score (Green=Lower Std, Red=Higher Std)"},
               linewidths=2, linecolor="black",
               ax=ax, vmin=0, vmax=1, square=False,
               cbar=True, annot_kws={"fontsize": 11, "fontweight": "bold"})
    
    ax.set_title("Metrics Standard Deviation Heatmap (Averaged Across 8 Datasets)\nRows: Models | Columns: Metrics | Green: Lower Variance (More Stable)",
                fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Models", fontsize=12, fontweight="bold")
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=11)
    
    # Add legend
    fig.text(0.02, 0.98, "Green: Lower Variance | Red: Higher Variance", fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "heatmap_metrics_std.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    return output_path

def plot_per_dataset_bars() -> list:
    """
    Create bar plots for each metric, showing all models across all datasets.
    """
    dataset_metrics = extract_metrics_per_dataset()
    output_paths = []
    
    for metric_name, metric_info in METRICS_TO_PLOT.items():
        metric_label = metric_info["label"]
        fig, ax = plt.subplots(figsize=FIGSIZE_BARS, dpi=DPI)
        
        x = np.arange(len(DATASETS))
        width = 0.25
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        
        for idx, model in enumerate(MODEL_LABELS.values()):
            offset = (idx - 1) * width
            values = [dataset_metrics[ds].loc[model, metric_label] if model in dataset_metrics[ds].index else np.nan 
                     for ds in DATASETS]
            
            bars = ax.bar(x + offset, values, width, label=model, color=colors[idx], alpha=0.8, 
                         edgecolor="black", linewidth=1)
            
            # Add value labels
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{metric_label} Value", fontsize=12, fontweight="bold")
        ax.set_title(f"{metric_label} - Values Across All Datasets",
                    fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS, rotation=45, ha="right", fontsize=9)
        ax.legend(loc="best", fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        
        plt.tight_layout()
        
        output_path = OUTPUT_DIR / f"bar_plot_{metric_name}_all_datasets.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        plt.close()
        
        output_paths.append(output_path)
    
    return output_paths

def create_summary_csv() -> Path:
    """
    Create CSV summary tables for both means and stds.
    """
    means_df, stds_df = extract_all_metrics_for_models()
    
    # Create combined CSV with both means and stds
    output_path = OUTPUT_DIR / "metrics_summary_tables.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("METRICS VALUES (Averaged Across 8 Datasets)\n")
        f.write("Rows: Models | Columns: Metrics\n\n")
        f.write(means_df.to_csv())
        f.write("\n\n")
        
        f.write("METRICS STANDARD DEVIATIONS (Averaged Across 8 Datasets)\n")
        f.write("Rows: Models | Columns: Metrics\n")
        f.write("(Lower STD = More Stable Model)\n\n")
        f.write(stds_df.to_csv())
    
    print(f"✓ Saved: {output_path}")
    return output_path

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    print("=" * 90)
    print("GENERATE HEATMAPS FROM EVALUATION RESULTS")
    print("=" * 90)
    print(f"\nResults root: {RESULTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Main heatmap - Metrics VALUES
    print("Generating main heatmap (Metrics Values)...")
    print("-" * 90)
    try:
        plot_main_heatmap_means()
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 2. Main heatmap - Metrics STD
    print("\nGenerating main heatmap (Metrics Standard Deviation)...")
    print("-" * 90)
    try:
        plot_main_heatmap_stds()
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 3. Per-dataset bar plots
    print("\nGenerating per-metric bar plots (all datasets)...")
    print("-" * 90)
    try:
        plot_per_dataset_bars()
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 4. Summary CSV
    print("\nCreating CSV summary tables...")
    print("-" * 90)
    try:
        create_summary_csv()
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 90)
    print("GENERATION COMPLETE ✅")
    print("=" * 90)
    print(f"\nGenerated outputs:")
    print(f"  🔥 1 Heatmap: Metrics VALUES (Models × Metrics)")
    print(f"  🔥 1 Heatmap: Metrics STD (Models × Metrics)")
    print(f"  📊 5 Bar plots: One per metric (showing all datasets)")
    print(f"  📋 1 CSV summary: Both tables (values & stds)")
    print(f"\nTotal: 8 files")
    print(f"\nLocation: {OUTPUT_DIR}/")
    print(f"\nHeatmap Structure:")
    print(f"  Rows: {list(MODEL_LABELS.values())}")
    print(f"  Columns: {list(METRICS_TO_PLOT[k]['label'] for k in METRICS_TO_PLOT.keys())}")
    print(f"  Values: Averaged across {len(DATASETS)} datasets")
    print(f"\nColor Scheme:")
    print(f"  🟢 Green: Better performance (best values)")
    print(f"  🟡 Yellow: Average performance")
    print(f"  🔴 Red: Worse performance (worst values)")
    print(f"\nMetrics Legend:")
    for metric_name, info in METRICS_TO_PLOT.items():
        direction = "↑ Higher Better" if info["direction"] == "higher_is_better" else "↓ Lower Better"
        print(f"  • {info['label']:15s} {direction}")
    print("\n" + "=" * 90)

if __name__ == "__main__":
    main()
