#!/usr/bin/env python3
"""
Extract metrics from PDF, create ranking tables per dataset, and perform Wilcoxon test

This script:
1. Extracts metrics from PDF data
2. For EACH dataset: ranks models (1=best, 8=worst) for each metric
3. Creates a ranking table PER DATASET (models as rows, metrics as columns)
4. Generates separate PNG for each dataset
5. Performs Wilcoxon signed-rank test on each dataset's rankings
6. Creates SEPARATE TIER LISTS for each metric (Mean_KL, Mean_WD, Corr_distance)
7. Each metric ranked independently across datasets
8. Handles ties: models with equal rank get same tier position
9. Uses only 3 metrics: Mean_KL, Mean_WD, Corr_distance
10. Excludes: TabPFN (no data), online_news_popularity, covertype
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Tuple, List

# ============================================================================
# DATA FROM PDF (extracted)
# ============================================================================

DATASETS = {
    "california_housing": {
        "CTGAN": {"Mean_KL": 0.087, "Mean_WD": 0.0917, "Corr_distance": 1.01},
        "TabDDPM": {"Mean_KL": 0.011, "Mean_WD": 0.0501, "Corr_distance": 0.94},
        "TabSyn": {"Mean_KL": 0.08, "Mean_WD": 30.8, "Corr_distance": 1.68},
        "Statsy": {"Mean_KL": 0.039, "Mean_WD": 0.859, "Corr_distance": 0.785},
        "DSB": {"Mean_KL": 0.285, "Mean_WD": 0.3009, "Corr_distance": 0.940},
        "DSBM": {"Mean_KL": 0.05, "Mean_WD": 0.08, "Corr_distance": 0.89},
        "AdvSB": {"Mean_KL": 0.11, "Mean_WD": 0.11, "Corr_distance": 1.45},
        "LightSB": {"Mean_KL": 0.182, "Mean_WD": 0.030, "Corr_distance": 0.731},
    },
    
    "diabetes": {
        "CTGAN": {"Mean_KL": 0.286, "Mean_WD": 0.5911, "Corr_distance": 0.071},
        "TabDDPM": {"Mean_KL": 0.450, "Mean_WD": 1.5518, "Corr_distance": 2.56},
        "TabSyn": {"Mean_KL": 6.99, "Mean_WD": 0.026, "Corr_distance": 5.85},
        "Statsy": {"Mean_KL": 2.722, "Mean_WD": 2.459, "Corr_distance": 1.721},
        "DSB": {"Mean_KL": 0.175, "Mean_WD": 0.2705, "Corr_distance": 0.99},
        "DSBM": {"Mean_KL": 0.14, "Mean_WD": 0.18, "Corr_distance": 1.01},
        "AdvSB": {"Mean_KL": 0.28, "Mean_WD": 0.23, "Corr_distance": 0.98},
        "LightSB": {"Mean_KL": 0.179, "Mean_WD": 0.143, "Corr_distance": 1.242},
    },
    
    "king_county_housing": {
        "CTGAN": {"Mean_KL": 0.279, "Mean_WD": 0.0738, "Corr_distance": 0.75},
        "TabDDPM": {"Mean_KL": 0.3719, "Mean_WD": 0.040, "Corr_distance": 3.16},
        "TabSyn": {"Mean_KL": 0.33, "Mean_WD": 378, "Corr_distance": 2.82},
        "Statsy": {"Mean_KL": 0.289, "Mean_WD": 2.166, "Corr_distance": 1.217},
        "DSB": {"Mean_KL": 0.381, "Mean_WD": 0.3949, "Corr_distance": 2.03},
        "DSBM": {"Mean_KL": 0.32, "Mean_WD": 0.25, "Corr_distance": 1.23},
        "AdvSB": {"Mean_KL": 0.19, "Mean_WD": 0.32, "Corr_distance": 1.57},
        "LightSB": {"Mean_KL": 0.611, "Mean_WD": 0.252, "Corr_distance": 0.896},
    },
    
    "adult_numeric": {
        "CTGAN": {"Mean_KL": 0.315, "Mean_WD": 0.1351, "Corr_distance": 1.12},
        "TabDDPM": {"Mean_KL": 0.020, "Mean_WD": 0.0351, "Corr_distance": 0.36},
        "TabSyn": {"Mean_KL": 4.3, "Mean_WD": 8.74, "Corr_distance": 4.01},
        "Statsy": {"Mean_KL": 0.788, "Mean_WD": 1.649, "Corr_distance": 0.198},
        "DSB": {"Mean_KL": 0.339, "Mean_WD": 0.3573, "Corr_distance": 0.57},
        "DSBM": {"Mean_KL": 0.57, "Mean_WD": 0.29, "Corr_distance": 0.58},
        "AdvSB": {"Mean_KL": 0.65, "Mean_WD": 0.29, "Corr_distance": 0.94},
        "LightSB": {"Mean_KL": 0.641, "Mean_WD": 0.374, "Corr_distance": 0.414},
    },
    
    "bank_marketing": {
        "CTGAN": {"Mean_KL": 0.250, "Mean_WD": 0.0921, "Corr_distance": 0.35},
        "TabDDPM": {"Mean_KL": 0.016, "Mean_WD": 0.0244, "Corr_distance": 0.13},
        "TabSyn": {"Mean_KL": 1.09, "Mean_WD": 278, "Corr_distance": 2.21},
        "Statsy": {"Mean_KL": 0.076, "Mean_WD": 1.312, "Corr_distance": 0.019},
        "DSB": {"Mean_KL": 0.305, "Mean_WD": 0.3180, "Corr_distance": 0.20},
        "DSBM": {"Mean_KL": 0.06, "Mean_WD": 0.12, "Corr_distance": 0.16},
        "AdvSB": {"Mean_KL": 0.07, "Mean_WD": 0.17, "Corr_distance": 0.58},
        "LightSB": {"Mean_KL": 0.214, "Mean_WD": 0.178, "Corr_distance": 0.303},
    },
    
    "online_shoppers": {
        "CTGAN": {"Mean_KL": 0.391, "Mean_WD": 0.0775, "Corr_distance": 0.85},
        "TabDDPM": {"Mean_KL": 0.098, "Mean_WD": 1.1357, "Corr_distance": 3.34},
        "TabSyn": {"Mean_KL": 0.92, "Mean_WD": 68, "Corr_distance": 4.33},
        "Statsy": {"Mean_KL": 0.198, "Mean_WD": 2.161, "Corr_distance": 0.056},
        "DSB": {"Mean_KL": 0.355, "Mean_WD": 0.3539, "Corr_distance": 0.86},
        "DSBM": {"Mean_KL": 0.14, "Mean_WD": 0.2, "Corr_distance": 0.71},
        "AdvSB": {"Mean_KL": 0.11, "Mean_WD": 0.24, "Corr_distance": 0.9},
        "LightSB": {"Mean_KL": 0.508, "Mean_WD": 0.121, "Corr_distance": 0.775},
    },
    
    "german_credit": {
        "CTGAN": {"Mean_KL": 0.308, "Mean_WD": 0.1795, "Corr_distance": 0.86},
        "TabDDPM": {"Mean_KL": 0.270, "Mean_WD": 0.8204, "Corr_distance": 0.96},
        "TabSyn": {"Mean_KL": 7.69, "Mean_WD": 212, "Corr_distance": 3.22},
        "Statsy": {"Mean_KL": 1.733, "Mean_WD": 0.2834, "Corr_distance": 0.283},
        "DSB": {"Mean_KL": 0.314, "Mean_WD": 0.3365, "Corr_distance": 0.38},
        "DSBM": {"Mean_KL": 0.33, "Mean_WD": 0.18, "Corr_distance": 0.47},
        "AdvSB": {"Mean_KL": 0.44, "Mean_WD": 0.36, "Corr_distance": 0.74},
        "LightSB": {"Mean_KL": 0.442, "Mean_WD": 0.195, "Corr_distance": 0.533},
    },
}

MODELS = ["CTGAN", "TabDDPM", "TabSyn", "Statsy", "DSB", "DSBM", "AdvSB", "LightSB"]
METRICS = ["Mean_KL", "Mean_WD", "Corr_distance"]

# ============================================================================
# RANKING FUNCTION
# ============================================================================

def rank_metric(values_dict: Dict[str, float], metric_name: str, direction: str = "lower") -> Dict[str, int]:
    """
    Rank models for a metric.
    
    Args:
        values_dict: Dictionary of model_name -> value
        metric_name: Name of metric
        direction: "lower" if lower is better, "higher" if higher is better
    
    Returns:
        Dictionary of model_name -> rank (1=best, 8=worst)
    """
    # Get non-None values
    valid_models = {m: v for m, v in values_dict.items() if v is not None}
    
    if not valid_models:
        return {m: np.nan for m in MODELS}
    
    # Rank based on direction
    if direction == "lower":
        # Lower values are better
        sorted_models = sorted(valid_models.items(), key=lambda x: x[1])
    else:
        # Higher values are better
        sorted_models = sorted(valid_models.items(), key=lambda x: x[1], reverse=True)
    
    # Assign ranks
    ranks = {}
    for rank, (model, value) in enumerate(sorted_models, 1):
        ranks[model] = rank
    
    # Assign NaN for missing values
    for model in MODELS:
        if model not in ranks:
            ranks[model] = np.nan
    
    return ranks

# ============================================================================
# CREATE RANK TABLE FOR EACH DATASET
# ============================================================================

def create_rank_table_for_dataset(dataset_name: str, dataset_data: Dict) -> pd.DataFrame:
    """
    Create ranking table for a single dataset.
    Models as rows, metrics as columns.
    Ranks: 1=best, 8=worst (for each metric within this dataset)
    
    Returns:
        DataFrame with models as rows, metrics as columns
    """
    
    rank_table = {}
    
    for metric in METRICS:
        # Extract values for this metric across all models
        metric_values = {}
        for model in MODELS:
            if model in dataset_data:
                val = dataset_data[model].get(metric)
                if val is not None:
                    metric_values[model] = val
        
        # Determine direction (lower is always better for these metrics)
        direction = "lower"  # Lower is better for KL, WD, Corr_distance
        
        # Rank
        ranks = rank_metric(metric_values, metric, direction)
        rank_table[metric] = ranks
    
    return pd.DataFrame(rank_table)

# ============================================================================
# WILCOXON TEST FOR EACH DATASET
# ============================================================================

def perform_wilcoxon_test_for_dataset(dataset_name: str, rank_table: pd.DataFrame) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test for a single dataset.
    Tests if metrics have significantly different average ranks.
    
    Returns:
        (stat, p_value)
    """
    print(f"\n{'='*80}")
    print(f"WILCOXON TEST: {dataset_name.upper()}")
    print('='*80)
    print("\nH₀: All metrics have equal average rank")
    print("H₁: Metrics have different average ranks")
    
    metrics = rank_table.columns.tolist()
    n_metrics = len(metrics)
    
    # Get average rank per metric
    avg_ranks = rank_table.mean(axis=0)
    print(f"\nAverage ranks per metric:")
    for metric, rank in avg_ranks.items():
        valid_count = rank_table[metric].notna().sum()
        print(f"  {metric:20s}: {rank:.3f} (n={int(valid_count)} models)")
    
    # Perform Friedman test
    # Remove rows with any NaN
    clean_data = rank_table.dropna()
    if len(clean_data) < 2:
        print(f"\nWarning: Not enough complete data for Friedman test (n={len(clean_data)})")
        return np.nan, np.nan
    
    from scipy.stats import friedmanchisquare
    stat, p_value = friedmanchisquare(*[clean_data[col].values for col in clean_data.columns])
    
    print(f"\nFriedman Test (χ² test for multiple treatments):")
    print(f"  Test statistic (χ²): {stat:.6f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Significance level (α): 0.05")
    
    if p_value < 0.05:
        print(f"\n  Result: REJECT H₀ (p = {p_value:.6f} < 0.05) ✓")
        print("  Conclusion: Metrics have SIGNIFICANTLY DIFFERENT average ranks")
    else:
        print(f"\n  Result: FAIL TO REJECT H₀ (p = {p_value:.6f} ≥ 0.05)")
        print("  Conclusion: Metrics do NOT have significantly different average ranks")
    
    # Compute total rank sum per model (equally weighted metrics)
    print(f"\n{'─'*80}")
    print(f"TOTAL RANK SUM per Model (equally weighted metrics, lower = better):")
    print(f"{'─'*80}")
    rank_sums = rank_table.sum(axis=1).sort_values()
    for i, (model, rank_sum) in enumerate(rank_sums.items(), 1):
        print(f"  {i}. {model:15s}: {rank_sum:.1f}")
    
    return stat, p_value

# ============================================================================
# CREATE VISUALIZATION FOR EACH DATASET
# ============================================================================

def create_visualization(dataset_name: str, rank_table: pd.DataFrame, stat: float, p_value: float):
    """Create heatmap of rank table for a dataset."""
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Create heatmap
    sns.heatmap(rank_table, annot=True, fmt=".1f", cmap="RdYlGn_r",
                cbar_kws={"label": "Rank (1=Best, 8=Worst)"},
                linewidths=2, linecolor="black", ax=ax,
                vmin=1, vmax=8, annot_kws={"fontsize": 12, "fontweight": "bold"},
                cbar=True)
    
    # Title with Wilcoxon results
    if not np.isnan(p_value):
        title_str = f"{dataset_name}\nModels × Metrics Ranking Table\n(χ²={stat:.4f}, p={p_value:.6f})"
    else:
        title_str = f"{dataset_name}\nModels × Metrics Ranking Table"
    
    ax.set_title(title_str, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Metrics (Lower Rank = Better)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Models", fontsize=12, fontweight="bold")
    
    # Create output directory if needed
    os.makedirs("ranking_tables", exist_ok=True)
    
    # Save figure
    filename = f"ranking_tables/{dataset_name}_rank_table.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved: {filename}")
    plt.close()

# ============================================================================
# CREATE FINAL TIER LISTS (ONE PER METRIC)
# ============================================================================

def create_final_tier_lists_per_metric(all_results: Dict) -> Dict[str, pd.DataFrame]:
    """
    Create final tier lists with average model rank FOR EACH METRIC SEPARATELY.
    Models ranked independently for each metric (Mean_KL, Mean_WD, Corr_distance).
    
    Returns:
        Dictionary of metric -> DataFrame with tier list
    """
    print(f"\n\n{'='*80}")
    print("FINAL TIER LISTS - MODEL RANKINGS PER METRIC (ACROSS ALL DATASETS)")
    print('='*80)
    
    all_tier_dfs = {}
    
    # Process each metric separately
    for metric in METRICS:
        print(f"\n\n{'='*80}")
        print(f"METRIC: {metric.upper()}")
        print('='*80)
        
        # Collect all ranks per model for this metric
        model_ranks = {model: [] for model in MODELS}
        
        for dataset_name in sorted(all_results.keys()):
            rank_table = all_results[dataset_name]["rank_table"]
            metric_ranks = rank_table[metric]
            for model in MODELS:
                if model in metric_ranks.index:
                    val = metric_ranks[model]
                    if not np.isnan(val):
                        model_ranks[model].append(val)
        
        # Calculate average rank per model for this metric
        avg_ranks = {}
        for model, ranks in model_ranks.items():
            if ranks:
                avg_ranks[model] = np.mean(ranks)
            else:
                avg_ranks[model] = np.nan
        
        # Sort by average rank (ascending - lower is better)
        sorted_models = sorted(
            [(m, r) for m, r in avg_ranks.items() if not np.isnan(r)],
            key=lambda x: x[1]
        )
        
        # Apply dense ranking with ties handling
        tier_data = []
        current_tier = 1
        prev_rank = None
        
        for idx, (model, avg_rank) in enumerate(sorted_models):
            # If rank changed, increment tier (dense ranking)
            if prev_rank is not None and avg_rank != prev_rank:
                current_tier = idx + 1
            
            num_datasets = len(model_ranks[model])
            status = "✓ BEST" if current_tier == 1 else ("✓" if current_tier <= 3 else "")
            
            tier_data.append({
                "Tier": current_tier,
                "Model": model,
                f"Avg {metric}": f"{avg_rank:.2f}",
                "Datasets": num_datasets,
                "Status": status
            })
            
            prev_rank = avg_rank
        
        tier_df = pd.DataFrame(tier_data)
        all_tier_dfs[metric] = tier_df
        
        print("\n" + tier_df.to_string(index=False))
        
        # Legend
        print(f"\n{'─'*80}")
        print(f"Tier Assignment for {metric} (Dense Ranking with Ties):")
        print("  - Models with equal average rank get same tier")
        print("  - Next tier number follows sequentially (no gaps)")
        print("  - Lower average rank = Better performance")
        print(f"{'─'*80}\n")
    
    return all_tier_dfs

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    
    print("\n" + "="*80)
    print("RANKING TABLES WITH WILCOXON TEST (PER DATASET) + FINAL TIER LISTS (PER METRIC)")
    print("="*80)
    print("\nStructure: Models as rows, Metrics as columns")
    print("Metrics: Mean_KL, Mean_WD, Corr_distance (lower rank = better)")
    print("Models: CTGAN, TabDDPM, TabSyn, Statsy, DSB, DSBM, AdvSB, LightSB")
    print(f"\nDatasets (excluding online_news_popularity, covertype):")
    for i, ds in enumerate(sorted(DATASETS.keys()), 1):
        print(f"  {i}. {ds}")
    print(f"\nDatasets: {len(DATASETS)}")
    print(f"Models: {len(MODELS)}")
    print(f"Metrics: {len(METRICS)}")
    
    # Create output directory
    os.makedirs("ranking_tables", exist_ok=True)
    
    # Process each dataset
    all_results = {}
    
    for dataset_name in sorted(DATASETS.keys()):
        dataset_data = DATASETS[dataset_name]
        
        print(f"\n\nProcessing: {dataset_name}")
        print("-" * 80)
        
        # Create rank table
        rank_table = create_rank_table_for_dataset(dataset_name, dataset_data)
        print("\nRank Table (Models × Metrics):")
        print(rank_table.to_string())
        
        # Perform Wilcoxon test
        stat, p_value = perform_wilcoxon_test_for_dataset(dataset_name, rank_table)
        
        # Create visualization
        create_visualization(dataset_name, rank_table, stat, p_value)
        
        # Store results
        all_results[dataset_name] = {
            "rank_table": rank_table,
            "stat": stat,
            "p_value": p_value
        }
    
    # Create final tier lists (one per metric)
    tier_dfs = create_final_tier_lists_per_metric(all_results)
    
    # Save tier lists
    for metric, tier_df in tier_dfs.items():
        filename = f"ranking_tables/tier_list_{metric}.csv"
        tier_df.to_csv(filename, index=False)
        print(f"✓ Tier list saved: {filename}")
    
    print(f"\n{'='*80}")
    print(f"✓ All {len(DATASETS)} ranking table PNGs created in 'ranking_tables/' directory")
    print(f"✓ {len(METRICS)} separate tier lists created (one per metric)")
    print(f"✓ Each PNG shows: Models as rows, 3 metrics as columns")
    print(f"✓ Metrics used: Mean_KL, Mean_WD, Corr_distance (all lower=better)")
    print(f"✓ Tier positions: Equal ranks get same tier, next model gets next tier number")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
