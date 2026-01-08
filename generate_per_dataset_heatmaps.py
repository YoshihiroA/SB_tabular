#!/usr/bin/env python3
"""
Generate Per-Dataset Heatmaps from Evaluation Results

Creates heatmaps for EACH dataset:
- Rows: Models (5 total: DSBM, ASBM, XGBoost-DSBM, DSBM-VAE, TabSyn)
- Columns: Metrics (5 total: R², RMSE, Corr. Dist, KL Div, Wasserstein)

Outlier handling:
- Outliers marked with GRAY color
- Main color scale defined by non-outlier values (IQR-based)
- Prevents extreme values from dominating the scale

Creates two types for each dataset:
1. VALUES heatmap: Actual metric values
2. STD heatmap: Standard deviations

Color scheme:
- Green: Excellent (best)
- Yellow: Good
- Red: Poor (worst)
- Gray: Outlier (outside IQR bounds)

Usage:
    python generate_per_dataset_heatmaps.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = Path("results/5fold_metrics")
OUTPUT_DIR = Path("figures/presentation_per_dataset_heatmaps")

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

MODELS = ["dsbm", "asbm", "dsbmxgboost", "dsbm_vae", "tabsyn"]
MODEL_LABELS = {
    "dsbm": "DSBM",
    "asbm": "ASBM",
    "dsbmxgboost": "XGBOOSTDSBM",
    "dsbm_vae": "DSBM-VAE",
    "tabsyn": "TabSyn"
}

METRICS_TO_PLOT = {
    "r2_score_synth": {
        "label": "R²",
        "direction": "higher_is_better",
    },
    "rmse_synth": {
        "label": "RMSE",
        "direction": "lower_is_better",
    },
    "correlation_distance": {
        "label": "Corr. Dist.",
        "direction": "lower_is_better",
    },
    "kl_divergence": {
        "label": "KL Div.",
        "direction": "lower_is_better",
    },
    "wasserstein_distance": {
        "label": "Wasserstein",
        "direction": "lower_is_better",
    }
}

DPI = 300
FIGSIZE_HEATMAP = (12, 8)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_json_results(model: str, dataset: str) -> Dict:
    """Load JSON results for a model and dataset."""
    file_patterns = [
        f"results_{model.upper()}_{dataset}.json",
        f"results_DSBM_{dataset}.json",
        f"results_ASBM_{dataset}.json",
        f"results_XGBOOST_DSBM_{dataset}.json",
        f"results_XGBOOSTDSBM_{dataset}.json",
        f"results_DSBM_VAE_{dataset}.json",
        f"results_TABSYN_VAE_{dataset}.json",
        f"results_{dataset}.json"
    ]
    
    for pattern in file_patterns:
        path = RESULTS_DIR / model / pattern
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
    
    return None

def detect_outliers_iqr(values: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Detect outliers using IQR method.
    
    Returns:
    - lower_bound: Q1 - 1.5*IQR
    - upper_bound: Q3 + 1.5*IQR
    - outlier_mask: boolean array marking outliers
    """
    valid_values = values[~np.isnan(values)]
    
    if len(valid_values) == 0:
        return 0.0, 1.0, np.zeros_like(values, dtype=bool)
    
    q1 = np.percentile(valid_values, 25)
    q3 = np.percentile(valid_values, 75)
    iqr = q3 - q1
    
    if iqr == 0:
        lower_bound = q1 - 1.0
        upper_bound = q1 + 1.0
    else:
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
    
    outlier_mask = (values < lower_bound) | (values > upper_bound)
    
    return lower_bound, upper_bound, outlier_mask

def normalize_score(value: float, direction: str, 
                   vmin: float, vmax: float) -> float:
    """
    Normalize score to [0, 1] range for color mapping.
    
    direction: "higher_is_better" or "lower_is_better"
    vmin, vmax: bounds computed from non-outlier values
    """
    if np.isnan(value):
        return np.nan
    
    value = np.clip(value, vmin, vmax)
    
    if direction == "higher_is_better":
        # Higher is better: map [vmin, vmax] → [0, 1]
        if vmax > vmin:
            normalized = (value - vmin) / (vmax - vmin)
        else:
            normalized = 0.5
    else:
        # Lower is better: map [vmin, vmax] → [1, 0]
        if vmax > vmin:
            normalized = (vmax - value) / (vmax - vmin)
        else:
            normalized = 0.5
    
    return normalized

def get_cmap():
    """Get colormap: red (bad) → yellow (ok) → green (good)"""
    return sns.color_palette("RdYlGn", as_cmap=True)

# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_metrics_for_dataset(dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract all metrics for a single dataset across all models.
    
    Returns:
    - means_df: shape (n_models, n_metrics)
    - stds_df: shape (n_models, n_metrics)
    """
    means_data = {}
    stds_data = {}
    
    for model in MODELS:
        model_means = {}
        model_stds = {}
        
        results = load_json_results(model, dataset)
        
        if results is None:
            print(f"⚠️  Missing: {model:15s} / {dataset}")
            for metric_name in METRICS_TO_PLOT.keys():
                model_means[METRICS_TO_PLOT[metric_name]["label"]] = np.nan
                model_stds[METRICS_TO_PLOT[metric_name]["label"]] = np.nan
        else:
            for metric_name in METRICS_TO_PLOT.keys():
                metric_label = METRICS_TO_PLOT[metric_name]["label"]
                model_means[metric_label] = results['summary']['metrics_mean'].get(metric_name, np.nan)
                model_stds[metric_label] = results['summary']['metrics_std'].get(metric_name, np.nan)
        
        means_data[MODEL_LABELS[model]] = model_means
        stds_data[MODEL_LABELS[model]] = model_stds
    
    means_df = pd.DataFrame(means_data).T
    stds_df = pd.DataFrame(stds_data).T
    
    return means_df, stds_df

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_dataset_heatmap(dataset: str, df: pd.DataFrame, 
                        value_type: str = "values") -> Path:
    """
    Create heatmap for a single dataset with outlier handling.
    
    Parameters:
    - dataset: dataset name
    - df: DataFrame with shape (n_models, n_metrics)
    - value_type: "values" or "std"
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP, dpi=DPI)
    
    # Prepare normalized data and outlier masks
    normalized_df = df.copy()
    outlier_mask = np.zeros_like(df.values, dtype=bool)
    
    for col_idx, metric_name in enumerate(METRICS_TO_PLOT.keys()):
        col = METRICS_TO_PLOT[metric_name]["label"]
        metric_info = METRICS_TO_PLOT[metric_name]
        
        values = df[col].values
        
        # Detect outliers
        lower, upper, col_outliers = detect_outliers_iqr(values)
        outlier_mask[:, col_idx] = col_outliers
        
        # Get non-outlier values for normalization bounds
        non_outlier_values = values[~col_outliers & ~np.isnan(values)]
        
        if len(non_outlier_values) > 0:
            vmin = non_outlier_values.min()
            vmax = non_outlier_values.max()
        else:
            vmin, vmax = 0, 1
        
        # Normalize using non-outlier bounds
        normalized_df[col] = [
            normalize_score(v, metric_info["direction"], vmin, vmax)
            for v in values
        ]
    
    # Create custom colormap data: outliers → gray, others → RdYlGn
    color_data = normalized_df.values.copy()
    
    # Create annotation matrix with outlier markers
    annot_data = df.copy()
    for i in range(outlier_mask.shape[0]):
        for j in range(outlier_mask.shape[1]):
            if outlier_mask[i, j]:
                # Mark outlier with asterisk
                original_val = df.iloc[i, j]
                annot_data.iloc[i, j] = f"{original_val:.4f}*"
            else:
                annot_data.iloc[i, j] = f"{df.iloc[i, j]:.4f}"
    
    # Create heatmap
    cmap = get_cmap()
    
    # Mask outliers for color mapping
    masked_color_data = np.ma.masked_where(outlier_mask, color_data)
    
    # Plot heatmap with custom colors
    sns.heatmap(masked_color_data, annot=annot_data, fmt="",
               cmap=cmap,
               cbar_kws={"label": "Normalized Score (Green=Best, Red=Worst)"},
               linewidths=2, linecolor="black",
               ax=ax, vmin=0, vmax=1, square=False,
               cbar=True, annot_kws={"fontsize": 10, "fontweight": "bold"})
    
    # Overlay gray color for outliers
    for i in range(outlier_mask.shape[0]):
        for j in range(outlier_mask.shape[1]):
            if outlier_mask[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, 
                                          fill=True, facecolor='gray', 
                                          edgecolor='black', linewidth=2))
                # Re-add text annotation
                ax.text(j + 0.5, i + 0.5, annot_data.iloc[i, j],
                       ha='center', va='center', fontsize=10, 
                       fontweight='bold', color='white')
    
    # Title
    title_suffix = "Values" if value_type == "values" else "Standard Deviations"
    ax.set_title(f"{dataset.replace('_', ' ').title()} - {title_suffix}\n"
                f"Rows: Models | Columns: Metrics | Gray: Outliers (IQR)",
                fontsize=13, fontweight="bold", pad=20)
    ax.set_xlabel("Metrics", fontsize=11, fontweight="bold")
    ax.set_ylabel("Models", fontsize=11, fontweight="bold")
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='green', edgecolor='black', label='Best'),
        mpatches.Patch(facecolor='yellow', edgecolor='black', label='Average'),
        mpatches.Patch(facecolor='red', edgecolor='black', label='Worst'),
        mpatches.Patch(facecolor='gray', edgecolor='black', label='Outlier (IQR)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(1.15, 1), fontsize=10)
    
    # Add footnote
    footnote = "* Outliers detected via IQR method (Q1-1.5×IQR, Q3+1.5×IQR) and marked gray"
    fig.text(0.5, 0.01, footnote, ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    
    # Save
    suffix = "values" if value_type == "values" else "std"
    output_path = OUTPUT_DIR / f"heatmap_{dataset}_{suffix}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    return output_path

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    print("=" * 90)
    print("GENERATE PER-DATASET HEATMAPS WITH OUTLIER HANDLING")
    print("=" * 90)
    print(f"\nResults root: {RESULTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    print(f"Models: {list(MODEL_LABELS.values())}")
    print(f"Metrics: {[METRICS_TO_PLOT[k]['label'] for k in METRICS_TO_PLOT.keys()]}")
    print(f"Datasets: {len(DATASETS)}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_plots = 0
    
    for dataset in DATASETS:
        print(f"\n{'='*90}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*90}")
        
        # Extract metrics
        means_df, stds_df = extract_metrics_for_dataset(dataset)
        
        # Plot VALUES heatmap
        print(f"  📊 Generating VALUES heatmap...", end=" ")
        try:
            plot_dataset_heatmap(dataset, means_df, value_type="values")
            total_plots += 1
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Plot STD heatmap
        print(f"  📊 Generating STD heatmap...", end=" ")
        try:
            plot_dataset_heatmap(dataset, stds_df, value_type="std")
            total_plots += 1
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 90)
    print("GENERATION COMPLETE ✅")
    print("=" * 90)
    print(f"\nGenerated outputs:")
    print(f"  🔥 {total_plots} heatmaps total")
    print(f"     - {len(DATASETS)} datasets")
    print(f"     - 2 heatmaps per dataset (values + std)")
    print(f"\nTotal files: {total_plots}")
    print(f"\nLocation: {OUTPUT_DIR}/")
    print(f"\nHeatmap Structure:")
    print(f"  Rows: {len(MODEL_LABELS)} models - {list(MODEL_LABELS.values())}")
    print(f"  Columns: {len(METRICS_TO_PLOT)} metrics - {[METRICS_TO_PLOT[k]['label'] for k in METRICS_TO_PLOT.keys()]}")
    print(f"\nOutlier Handling:")
    print(f"  • IQR-based detection: Q1 - 1.5×IQR to Q3 + 1.5×IQR")
    print(f"  • Outliers marked with GRAY color and asterisk (*)")
    print(f"  • Color scale defined by non-outlier values only")
    print(f"  • Prevents extreme values from dominating visualization")
    print(f"\nColor Scheme:")
    print(f"  🟢 Green: Best performance")
    print(f"  🟡 Yellow: Average performance")
    print(f"  🔴 Red: Worst performance")
    print(f"  ⬜ Gray: Outlier (outside IQR bounds)")
    print(f"\nMetrics Legend:")
    for metric_name, info in METRICS_TO_PLOT.items():
        direction = "↑ Higher Better" if info["direction"] == "higher_is_better" else "↓ Lower Better"
        print(f"  • {info['label']:15s} {direction}")
    print("\n" + "=" * 90)

if __name__ == "__main__":
    main()
