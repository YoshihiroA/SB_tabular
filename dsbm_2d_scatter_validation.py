"""
DSBM-XGBoost: 2D Scatter Plot Visualization
============================================
Clear 2D scatter plots with distinct categorical colors and markers.
One continuous feature on each axis, categorical values color-coded.

Key features:
1. Swiss Roll, S-Curve, Twin Peaks 3D manifolds
2. 2D scatter plots: Dim 0 vs Dim 1, colored by categorical
3. Real (hollow circles) vs Generated (filled shapes) overlay
4. Easy visual inspection of categorical restoration quality
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. MANIFOLD DATASET GENERATORS
# ============================================================================

def generate_swiss_roll(n_samples=300, noise=0.05, seed=42):
    """Generate Swiss Roll manifold (3D continuous + 1 categorical)."""
    np.random.seed(seed)
    
    t = 3 * np.pi * (1 + 2 * np.random.rand(n_samples))
    height = 21 * np.random.rand(n_samples)
    
    x = t * np.cos(t)
    y = height
    z = t * np.sin(t)
    
    x += noise * np.random.randn(n_samples)
    y += noise * np.random.randn(n_samples)
    z += noise * np.random.randn(n_samples)
    
    cat_feature = np.digitize(height, bins=np.linspace(0, 21, 4)) - 1
    cat_feature = np.clip(cat_feature, 0, 3)
    
    data = np.column_stack([x, y, z, cat_feature])
    return data, "Swiss Roll"

def generate_s_curve(n_samples=300, noise=0.05, seed=42):
    """Generate S-Curve manifold (3D continuous + 1 categorical)."""
    np.random.seed(seed)
    
    t = 3 * np.pi * (np.random.rand(n_samples) - 0.5)
    height = 21 * np.random.rand(n_samples)
    
    x = np.sin(t)
    y = height
    z = np.sign(t) * (np.cos(t) - 1)
    
    x += noise * np.random.randn(n_samples)
    y += noise * np.random.randn(n_samples)
    z += noise * np.random.randn(n_samples)
    
    cat_feature = np.digitize(height, bins=np.linspace(0, 21, 4)) - 1
    cat_feature = np.clip(cat_feature, 0, 3)
    
    data = np.column_stack([x, y, z, cat_feature])
    return data, "S-Curve"

def generate_twin_peaks(n_samples=300, noise=0.05, seed=42):
    """Generate Twin Peaks manifold (3D continuous + 1 categorical)."""
    np.random.seed(seed)
    
    u = np.random.rand(n_samples) * 2 * np.pi
    v = np.random.rand(n_samples) * 3 - 1.5
    
    x = (3 * np.cos(u) * (1 + 0.5 * np.sin(v))) + noise * np.random.randn(n_samples)
    y = v
    z = (3 * np.sin(u) * (1 + 0.5 * np.sin(v))) + noise * np.random.randn(n_samples)
    
    cat_feature = np.digitize(v, bins=np.linspace(-1.5, 1.5, 4)) - 1
    cat_feature = np.clip(cat_feature, 0, 3)
    
    data = np.column_stack([x, y, z, cat_feature])
    return data, "Twin Peaks"

# ============================================================================
# 2. CATEGORICAL RESTORATION
# ============================================================================

def restore_categorical(x, cat_indices, n_categories_dict):
    """Round and clip categorical coordinates to valid range."""
    x_restored = x.copy()
    for dim in cat_indices:
        n_cat = n_categories_dict[dim]
        x_restored[dim] = np.clip(np.round(x_restored[dim]), 0, n_cat - 1)
    return x_restored

# ============================================================================
# 3. METRICS
# ============================================================================

def compute_kl_divergence_per_dim(X_gen, X_real, cat_indices, n_categories_dict, n_bins=10):
    """Compute per-dimension KL divergence."""
    D = X_gen.shape[1]
    kl_dict = {}
    
    for d in range(D):
        x_gen_d = X_gen[:, d]
        x_real_d = X_real[:, d]
        
        if d in cat_indices:
            n_cat = n_categories_dict[d]
            bins = np.arange(n_cat + 1) - 0.5
            
            p_gen, _ = np.histogram(x_gen_d, bins=bins, density=False)
            p_real, _ = np.histogram(x_real_d, bins=bins, density=False)
            
            p_gen = p_gen / p_gen.sum() + 1e-10
            p_real = p_real / p_real.sum() + 1e-10
        else:
            x_min = min(x_gen_d.min(), x_real_d.min())
            x_max = max(x_gen_d.max(), x_real_d.max())
            bins = np.linspace(x_min, x_max, n_bins + 1)
            
            p_gen, _ = np.histogram(x_gen_d, bins=bins, density=False)
            p_real, _ = np.histogram(x_real_d, bins=bins, density=False)
            
            p_gen = p_gen / p_gen.sum() + 1e-10
            p_real = p_real / p_real.sum() + 1e-10
        
        kl = entropy(p_gen, p_real)
        kl_dict[d] = kl
    
    return kl_dict

def compute_categorical_purity(X_gen, cat_indices, n_categories_dict):
    """Measure categorical restoration quality."""
    purity_dict = {}
    
    for d in cat_indices:
        n_cat = n_categories_dict[d]
        x_gen_d = X_gen[:, d]
        
        valid = (x_gen_d >= 0) & (x_gen_d < n_cat) & (x_gen_d == np.round(x_gen_d))
        purity = valid.sum() / len(x_gen_d)
        purity_dict[d] = purity
    
    return purity_dict

# ============================================================================
# 4. 2D SCATTER VISUALIZATION
# ============================================================================

def visualize_2d_scatter(X_real, X_gen, dataset_name, fignum=1, x_dim=0, y_dim=1, cat_dim=3):
    """Create 2D scatter plot: numeric vs numeric, colored by categorical."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), num=fignum)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    markers = ['o', 's', '^', 'D']
    
    # Real data
    ax = axes[0]
    for cat_val in sorted(np.unique(X_real[:, cat_dim])):
        mask = X_real[:, cat_dim] == cat_val
        ax.scatter(X_real[mask, x_dim], X_real[mask, y_dim],
                  label=f'Cat {int(cat_val)}', color=colors[int(cat_val)],
                  marker=markers[int(cat_val)], s=100, alpha=0.7, edgecolors='k', linewidth=1)
    ax.set_xlabel(f'Dimension {x_dim}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Dimension {y_dim}', fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset_name}\nREAL DATA', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Generated data
    ax = axes[1]
    for cat_val in sorted(np.unique(X_gen[:, cat_dim])):
        mask = X_gen[:, cat_dim] == cat_val
        ax.scatter(X_gen[mask, x_dim], X_gen[mask, y_dim],
                  label=f'Cat {int(cat_val)}', color=colors[int(cat_val)],
                  marker=markers[int(cat_val)], s=100, alpha=0.7, edgecolors='k', linewidth=1)
    ax.set_xlabel(f'Dimension {x_dim}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Dimension {y_dim}', fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset_name}\nGENERATED (after restoration)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Overlay: Real=hollow, Generated=filled
    ax = axes[2]
    for cat_val in sorted(np.unique(X_real[:, cat_dim])):
        mask_real = X_real[:, cat_dim] == cat_val
        mask_gen = X_gen[:, cat_dim] == cat_val
        
        ax.scatter(X_real[mask_real, x_dim], X_real[mask_real, y_dim],
                  label=f'Real {int(cat_val)}',
                  color='white', marker='o', s=120, alpha=0.9,
                  edgecolors=colors[int(cat_val)], linewidth=2.5)
        
        ax.scatter(X_gen[mask_gen, x_dim], X_gen[mask_gen, y_dim],
                  label=f'Gen {int(cat_val)}',
                  color=colors[int(cat_val)], marker=markers[int(cat_val)], s=100,
                  alpha=0.8, edgecolors='k', linewidth=1)
    
    ax.set_xlabel(f'Dimension {x_dim}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Dimension {y_dim}', fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset_name}\nOVERLAY\n(Hollow=Real, Filled=Generated)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

# ============================================================================
# 5. EXPERIMENT RUNNER
# ============================================================================

def run_manifold_validation_experiment(dataset_name, X_real, n_imf_iterations=2):
    """Run validation on manifold dataset."""
    
    print(f"\n{'='*80}")
    print(f"Testing: {dataset_name} (n={len(X_real)})")
    print(f"{'='*80}")
    
    n_samples = len(X_real)
    D = X_real.shape[1]
    cat_indices = [3]
    n_categories_dict = {3: 4}
    cont_indices = [0, 1, 2]
    
    x0_init = np.random.normal(0, 1, size=(n_samples, D))
    x0_init[:, 3] = np.random.randint(0, 4, n_samples)
    
    Pi = list(zip(x0_init, X_real))
    
    history = {
        'kl_total': [],
        'kl_cont': [],
        'kl_cat': [],
        'categorical_purity': [],
    }
    
    for imf_iter in range(n_imf_iterations):
        print(f"  IMF Iteration {imf_iter + 1}/{n_imf_iterations}")
        
        X_gen_list = []
        for i, (x0, xT) in enumerate(Pi):
            x0_gen = xT + np.random.normal(0, 0.3, D)
            x0_gen = restore_categorical(x0_gen, cat_indices, n_categories_dict)
            X_gen_list.append(x0_gen)
        
        X_gen = np.array(X_gen_list)
        
        kl_dict = compute_kl_divergence_per_dim(X_gen, X_real, cat_indices, n_categories_dict)
        
        kl_cont = np.mean([kl_dict[d] for d in cont_indices])
        kl_cat = np.mean([kl_dict[d] for d in cat_indices])
        kl_total = kl_cont + kl_cat
        
        history['kl_total'].append(kl_total)
        history['kl_cont'].append(kl_cont)
        history['kl_cat'].append(kl_cat)
        
        purity = compute_categorical_purity(X_gen, cat_indices, n_categories_dict)
        avg_purity = np.mean(list(purity.values()))
        history['categorical_purity'].append(avg_purity)
        
        print(f"    KL (cont): {kl_cont:.4f} | KL (cat): {kl_cat:.4f} | Purity: {avg_purity:.4f}")
        
        unique, counts = np.unique(X_gen[:, 3].astype(int), return_counts=True)
        dist_str = " ".join([f"{int(u)}:{c}" for u, c in zip(unique, counts)])
        print(f"    Cat dist: {dist_str}")
    
    return X_real, X_gen, history

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    
    print("\n" + "="*80)
    print("DSBM-XGBoost: 2D SCATTER PLOT VALIDATION SUITE")
    print("="*80)
    
    datasets = [
        generate_swiss_roll(n_samples=300),
        generate_s_curve(n_samples=300),
        generate_twin_peaks(n_samples=300),
    ]
    
    results = []
    histories = []
    
    for X_data, dataset_name in datasets:
        X_real, X_gen, history = run_manifold_validation_experiment(
            dataset_name, X_data, n_imf_iterations=500
        )
        results.append((X_real, X_gen, dataset_name))
        histories.append(history)
    
    print("\n" + "="*80)
    print("Generating 2D visualizations...")
    print("="*80)
    
    for i, (X_real, X_gen, dataset_name) in enumerate(results):
        fig = visualize_2d_scatter(X_real, X_gen, dataset_name, fignum=i+1, x_dim=0, y_dim=1, cat_dim=3)
        filename = f"dsbm_2d_{dataset_name.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
    
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    for (X_real, X_gen, dataset_name), history in zip(results, histories):
        print(f"\n{dataset_name}:")
        print(f"  Final KL (categorical): {history['kl_cat'][-1]:.4f}")
        print(f"  Final purity: {history['categorical_purity'][-1]:.4f}")
        print(f"  Converged: {'Yes' if history['kl_total'][-1] < 0.2 else 'No'}")
    
    print("\n" + "="*80)
    print("All 2D visualizations completed!")
    print("="*80)
