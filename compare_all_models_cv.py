#!/usr/bin/env python3

"""
compare_all_models_cv.py

Compare results from 5-fold CV evaluation of all 4 models:
- TabSyn
- ASBM
- DSBM
- DSBM + XGBoost

Aggregates results from individual JSON files and creates comparative summary.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_model_results(json_file: str) -> Dict:
    """Load evaluation results from model JSON file."""
    path = Path(json_file)
    
    if not path.exists():
        print(f"⚠️  File not found: {json_file}")
        return {}
    
    with open(path, 'r') as f:
        return json.load(f)


def create_comparison_dataframe(
    models_dict: Dict[str, Dict],
    metric_name: str
) -> pd.DataFrame:
    """Create comparison DataFrame for a specific metric across all models."""
    
    data = {}
    
    for model_name, results in models_dict.items():
        model_data = {}
        
        for dataset_name, dataset_result in results.items():
            if 'error' not in dataset_result and 'metrics' in dataset_result:
                metrics = dataset_result['metrics']
                
                if metric_name in metrics:
                    metric_val = metrics[metric_name]
                    mean_val = metric_val.get('mean', np.nan)
                    std_val = metric_val.get('std', np.nan)
                    
                    model_data[dataset_name] = {
                        'mean': mean_val,
                        'std': std_val,
                        'formatted': f"{mean_val:.4f}±{std_val:.4f}" if not np.isnan(mean_val) else "N/A"
                    }
        
        data[model_name] = model_data
    
    df = pd.DataFrame({
        model: [data[model].get(ds, {}).get('formatted', 'N/A') 
                for ds in sorted(set(d for model_data in data.values() for d in model_data.keys()))]
        for model in data.keys()
    })
    
    df.index = sorted(set(d for model_data in data.values() for d in model_data.keys()))
    
    return df


def create_numeric_comparison(
    models_dict: Dict[str, Dict],
    metric_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create numeric comparison DataFrames (means and stds separately)."""
    
    datasets = sorted(set(
        ds for results in models_dict.values() 
        for ds in results.keys() 
        if 'error' not in results[ds]
    ))
    
    means_data = {}
    stds_data = {}
    
    for model_name, results in models_dict.items():
        means = []
        stds = []
        
        for dataset_name in datasets:
            if dataset_name in results and 'error' not in results[dataset_name]:
                metrics = results[dataset_name].get('metrics', {})
                if metric_name in metrics:
                    means.append(metrics[metric_name].get('mean', np.nan))
                    stds.append(metrics[metric_name].get('std', np.nan))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            else:
                means.append(np.nan)
                stds.append(np.nan)
        
        means_data[model_name] = means
        stds_data[model_name] = stds
    
    df_means = pd.DataFrame(means_data, index=datasets)
    df_stds = pd.DataFrame(stds_data, index=datasets)
    
    return df_means, df_stds


def print_metric_comparison(
    models_dict: Dict[str, Dict],
    metric_name: str,
    metric_label: str = None,
    lower_is_better: bool = True
):
    """Print formatted comparison for a specific metric."""
    
    if metric_label is None:
        metric_label = metric_name.replace('_', ' ').upper()
    
    direction = "(Lower is Better)" if lower_is_better else "(Higher is Better)"
    
    print(f"\n{'='*100}")
    print(f"{metric_label} {direction}")
    print(f"{'='*100}\n")
    
    datasets = sorted(set(
        ds for results in models_dict.values() 
        for ds in results.keys() 
        if 'error' not in results[ds]
    ))
    
    models = list(models_dict.keys())
    
    # Print header
    print(f"{'Dataset':<25}", end='')
    for model in models:
        print(f"{model:>20}", end='')
    print()
    print("-" * (25 + len(models) * 20))
    
    # Print data
    for dataset in datasets:
        print(f"{dataset:<25}", end='')
        
        for model in models:
            if dataset in models_dict[model] and 'error' not in models_dict[model][dataset]:
                metrics = models_dict[model][dataset].get('metrics', {})
                if metric_name in metrics:
                    m = metrics[metric_name]['mean']
                    s = metrics[metric_name]['std']
                    print(f"{m:>8.4f}±{s:<8.4f}", end='')
                else:
                    print(f"{'N/A':>20}", end='')
            else:
                print(f"{'N/A':>20}", end='')
        
        print()


def print_ranking(
    models_dict: Dict[str, Dict],
    metric_name: str,
    lower_is_better: bool = True
):
    """Print ranking of models for a specific metric."""
    
    print(f"\n{'='*80}")
    print(f"RANKING FOR {metric_name.replace('_', ' ').upper()}")
    print(f"{'='*80}\n")
    
    rankings = {}
    
    for model_name in models_dict.keys():
        values = []
        
        for dataset_result in models_dict[model_name].values():
            if 'error' not in dataset_result and 'metrics' in dataset_result:
                metrics = dataset_result['metrics']
                if metric_name in metrics:
                    values.append(metrics[metric_name]['mean'])
        
        if values:
            avg = np.mean(values)
            rankings[model_name] = avg
    
    sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=not lower_is_better)
    
    print(f"{'Rank':<6} {'Model':<25} {'Mean Value':>15}")
    print("-" * 50)
    
    for rank, (model, value) in enumerate(sorted_rankings, 1):
        print(f"{rank:<6} {model:<25} {value:>15.4f}")


def generate_summary_report(models_dict: Dict[str, Dict]) -> str:
    """Generate comprehensive summary report."""
    
    report = "\n" + "="*100 + "\n"
    report += "COMPREHENSIVE 5-FOLD CROSS-VALIDATION COMPARISON\n"
    report += "="*100 + "\n\n"
    
    # Get all metrics
    all_metrics = set()
    for results in models_dict.values():
        for dataset_result in results.values():
            if 'error' not in dataset_result:
                all_metrics.update(dataset_result.get('metrics', {}).keys())
    
    all_metrics = sorted(list(all_metrics))
    
    # Metric configurations
    metric_configs = {
        'mmd': ('MMD - Distribution Similarity', True),
        'swd': ('SWD - Distribution Similarity', True),
        'kl_divergence': ('KL Divergence', True),
        'correlation_distance': ('Correlation Matrix Distance', True),
        'r2_score_real': ('R² Score (Real Model)', False),
        'r2_score_synth': ('R² Score (Synthetic Model)', False),
        'rmse_real': ('RMSE (Real Model)', True),
        'rmse_synth': ('RMSE (Synthetic Model)', True),
        'mse_real': ('MSE (Real Model)', True),
        'mse_synth': ('MSE (Synthetic Model)', True),
        'mae_real': ('MAE (Real Model)', True),
        'mae_synth': ('MAE (Synthetic Model)', True),
        'r2_difference_percent': ('R² Difference %', True),
        'rmse_difference_percent': ('RMSE Difference %', True),
    }
    
    for metric in all_metrics:
        if metric in metric_configs:
            label, lower_is_better = metric_configs[metric]
            
            datasets = sorted(set(
                ds for results in models_dict.values() 
                for ds in results.keys() 
                if 'error' not in results[ds]
            ))
            
            models = list(models_dict.keys())
            
            report += f"\n{'-'*100}\n"
            report += f"{label} {'(Lower is Better)' if lower_is_better else '(Higher is Better)'}\n"
            report += f"{'-'*100}\n"
            report += f"{'Dataset':<25}"
            
            for model in models:
                report += f"{model:>20}"
            
            report += "\n" + "-" * (25 + len(models) * 20) + "\n"
            
            for dataset in datasets:
                report += f"{dataset:<25}"
                
                for model in models:
                    if dataset in models_dict[model] and 'error' not in models_dict[model][dataset]:
                        metrics = models_dict[model][dataset].get('metrics', {})
                        if metric in metrics:
                            m = metrics[metric]['mean']
                            s = metrics[metric]['std']
                            report += f"{m:>8.4f}±{s:<8.4f}"
                        else:
                            report += f"{'N/A':>20}"
                    else:
                        report += f"{'N/A':>20}"
                
                report += "\n"
    
    return report


def main():
    """Main comparison function."""
    
    print("\n" + "="*100)
    print("LOADING ALL MODEL RESULTS")
    print("="*100 + "\n")
    
    models_dict = {
        'TabSyn': load_model_results('tabsyn_cv_results.json'),
        'ASBM': load_model_results('asbm_cv_results.json'),
        'DSBM': load_model_results('dsbm_cv_results.json'),
        'DSBM+XGBoost': load_model_results('xgboost_dsbm_cv_results.json'),
    }
    
    # Check what we loaded
    for model_name, results in models_dict.items():
        n_datasets = len([d for d in results.keys() if 'error' not in results[d]])
        print(f"✓ {model_name:<20} {n_datasets} datasets loaded")
    
    # Generate comparisons
    print("\n" + "="*100)
    print("GENERATING COMPARISONS")
    print("="*100)
    
    # Print detailed comparisons
    metrics_to_compare = [
        ('mmd', 'Maximum Mean Discrepancy', True),
        ('swd', 'Sliced Wasserstein Distance', True),
        ('kl_divergence', 'KL Divergence', True),
        ('correlation_distance', 'Correlation Distance', True),
        ('r2_score_real', 'R² Score (Real)', False),
        ('r2_score_synth', 'R² Score (Synthetic)', False),
        ('rmse_real', 'RMSE (Real)', True),
        ('rmse_synth', 'RMSE (Synthetic)', True),
        ('r2_difference_percent', 'R² Difference %', True),
        ('rmse_difference_percent', 'RMSE Difference %', True),
    ]
    
    for metric_name, label, lower_is_better in metrics_to_compare:
        print_metric_comparison(models_dict, metric_name, label, lower_is_better)
        print_ranking(models_dict, metric_name, lower_is_better)
    
    # Generate and save summary report
    report = generate_summary_report(models_dict)
    
    report_file = 'model_comparison_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Summary report saved: {report_file}")
    
    # Create aggregate CSV for easy analysis
    print("\n" + "="*100)
    print("GENERATING AGGREGATE CSV")
    print("="*100 + "\n")
    
    all_metrics = set()
    for results in models_dict.values():
        for dataset_result in results.values():
            if 'error' not in dataset_result:
                all_metrics.update(dataset_result.get('metrics', {}).keys())
    
    csv_rows = []
    
    for metric_name in sorted(all_metrics):
        means_df, stds_df = create_numeric_comparison(models_dict, metric_name)
        
        for dataset in means_df.index:
            row = {'Metric': metric_name, 'Dataset': dataset}
            
            for model in means_df.columns:
                if not np.isnan(means_df.loc[dataset, model]):
                    row[f'{model}_mean'] = means_df.loc[dataset, model]
                    row[f'{model}_std'] = stds_df.loc[dataset, model]
                else:
                    row[f'{model}_mean'] = None
                    row[f'{model}_std'] = None
            
            csv_rows.append(row)
    
    csv_df = pd.DataFrame(csv_rows)
    csv_file = 'model_comparison_aggregate.csv'
    csv_df.to_csv(csv_file, index=False)
    
    print(f"✓ Aggregate CSV saved: {csv_file}")
    print(f"  Shape: {csv_df.shape}")


if __name__ == "__main__":
    main()
    print("\n✅ Comparison complete!")
