#!/usr/bin/env python3
"""
Grid-Based Markovian Projection with Letter S Target

Instead of Swiss roll (hard to discretize cleanly), 
we use an explicit Letter S shape that's crystal clear 
when discretized to a 10x10 grid.

Author: Yoshi
Date: December 30, 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import json
import os
from typing import Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# DATA GENERATION: GAUSSIAN GRID (unchanged)
# ============================================================================

def generate_gaussian_grid_2d(
    grid_size: int = 10,
    n_samples: int = 300,
    seed: int = 42
) -> np.ndarray:
    """
    Generate Gaussian grid directly as categorical data.
    
    Each cell probability follows 2D Gaussian.
    For each sample, independently sample each cell via Bernoulli.
    
    Args:
        grid_size: Grid dimension (10×10, 15×15, etc.)
        n_samples: Number of samples to generate
        seed: Random seed
    
    Returns:
        samples: (n_samples, grid_size²) binary array
    """
    np.random.seed(seed)
    
    # Create 2D probability grid
    x = np.linspace(-3, 3, grid_size)
    y = np.linspace(-3, 3, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # Gaussian density (unnormalized)
    gaussian_prob = np.exp(-(xx**2 + yy**2))
    
    # Normalize to [0, 1]
    gaussian_prob = gaussian_prob / gaussian_prob.max()
    gaussian_prob_flat = gaussian_prob.flatten()
    
    # Generate samples: each cell independently Bernoulli
    samples = np.zeros((n_samples, grid_size * grid_size), dtype=np.uint8)
    for i in range(n_samples):
        samples[i] = np.random.binomial(1, gaussian_prob_flat)
    
    return samples


# ============================================================================
# DATA GENERATION: LETTER S GRID (NEW!)
# ============================================================================

def point_to_line_segment_distance(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float
) -> float:
    """
    Compute shortest distance from point (px, py) to line segment.
    
    Args:
        px, py: Point coordinates
        x1, y1: Line segment start
        x2, y2: Line segment end
    
    Returns:
        Shortest distance
    """
    # Vector from start to end
    dx = x2 - x1
    dy = y2 - y1
    
    # Vector from start to point
    dpx = px - x1
    dpy = py - y1
    
    # Parameter t (0 = start, 1 = end)
    denom = dx**2 + dy**2
    if denom < 1e-10:
        # Segment is a point
        return np.sqrt(dpx**2 + dpy**2)
    
    t = max(0, min(1, (dpx * dx + dpy * dy) / denom))
    
    # Closest point on segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Distance to closest point
    dist = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    return dist


def generate_letter_s_grid_2d(
    grid_size: int = 10,
    n_samples: int = 300,
    seed: int = 42,
    sigma: float = 0.3
) -> np.ndarray:
    """
    Generate Letter S grid directly as categorical data.
    
    Letter S consists of two C-shaped curves:
    - Top curve: waves at y ≈ 0.8 (backward curve)
    - Bottom curve: waves at y ≈ 0.2 (forward curve)
    
    Each cell's probability = exp(-distance_to_S² / σ²)
    
    Args:
        grid_size: Grid dimension
        n_samples: Number of samples
        seed: Random seed
        sigma: Width of S curves (controls activation width)
    
    Returns:
        samples: (n_samples, grid_size²) binary array
    """
    np.random.seed(seed)
    
    # Create grid of cell centers
    grid_coords = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(grid_coords, grid_coords)
    
    # Initialize distance field (all cells start infinitely far)
    min_distances = np.ones((grid_size, grid_size)) * np.inf
    
    # Define top S curve (backward curve)
    # x goes 0→1, y oscillates around 0.75
    n_curve_points = 50
    x_top = np.linspace(0, 1, n_curve_points)
    y_top = 0.75 - 0.15 * np.sin(np.pi * x_top)  # Waves backward
    
    # Define bottom S curve (forward curve)
    # x goes 0→1, y oscillates around 0.25
    x_bottom = np.linspace(0, 1, n_curve_points)
    y_bottom = 0.25 + 0.15 * np.sin(np.pi * x_bottom)  # Waves forward
    
    # Compute distance from each cell to nearest S curve
    for i in range(grid_size):
        for j in range(grid_size):
            cell_x = xx[i, j]
            cell_y = yy[i, j]
            
            # Distance to top curve
            for k in range(n_curve_points - 1):
                dist = point_to_line_segment_distance(
                    cell_x, cell_y,
                    x_top[k], y_top[k],
                    x_top[k+1], y_top[k+1]
                )
                min_distances[i, j] = min(min_distances[i, j], dist)
            
            # Distance to bottom curve
            for k in range(n_curve_points - 1):
                dist = point_to_line_segment_distance(
                    cell_x, cell_y,
                    x_bottom[k], y_bottom[k],
                    x_bottom[k+1], y_bottom[k+1]
                )
                min_distances[i, j] = min(min_distances[i, j], dist)
    
    # Convert distance to probability
    s_prob = np.exp(-min_distances**2 / (sigma**2))
    s_prob = np.clip(s_prob, 0, 1)  # Ensure valid probabilities
    s_prob_flat = s_prob.flatten()
    
    # Generate samples: each cell independently Bernoulli
    samples = np.zeros((n_samples, grid_size * grid_size), dtype=np.uint8)
    for i in range(n_samples):
        samples[i] = np.random.binomial(1, s_prob_flat)
    
    return samples


# ============================================================================
# DIFFUSION MODEL (unchanged)
# ============================================================================

class SimpleD3PM(nn.Module):
    """Simple discrete diffusion model for categorical data."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x → [0, 1]."""
        h = self.encoder(x.float())
        out = self.decoder(h)
        return out
    
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Generate samples: output probabilities → binary."""
        probs = self.forward(x)
        samples = torch.bernoulli(probs)
        return samples


# ============================================================================
# PERMUTATION COUPLING (unchanged)
# ============================================================================

class PermutationCoupling:
    """Bijective coupling via permutations."""
    
    def __init__(self, n: int):
        self.n = n
        self.forward_perm = np.arange(n)
    
    def set_forward_perm(self, perm: np.ndarray):
        """Set permutation after validation."""
        assert len(perm) == self.n, f"Length mismatch: {len(perm)} vs {self.n}"
        assert len(np.unique(perm)) == self.n, f"Not a permutation: duplicates found"
        assert perm.min() == 0 and perm.max() == self.n - 1, f"Out of range: min={perm.min()}, max={perm.max()}"
        self.forward_perm = perm.copy()
    
    def apply_forward(self, X: np.ndarray) -> np.ndarray:
        """Apply permutation to data."""
        return X[self.forward_perm]


# ============================================================================
# TRAINER (FIXED!)
# ============================================================================

class DiscretizedTrainerPermutation:
    """Train discrete diffusion model with permutation coupling."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-3,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCELoss()
    
    def train_iteration(
        self,
        X0: np.ndarray,
        X1: np.ndarray,
        coupling: PermutationCoupling,
        iterations: int = 10,
    ) -> Dict[str, Any]:
        """
        Run Markovian projection iterations.
        
        Args:
            X0: Input samples (n, d)
            X1: Target samples (n, d)
            coupling: Permutation coupling object
            iterations: Number of iterations
        
        Returns:
            Dictionary with metrics
        """
        n, d = X0.shape
        X0_t = torch.from_numpy(X0).float().to(self.device)
        X1_t = torch.from_numpy(X1).float().to(self.device)
        
        metrics = {
            'iterations': [],
            'losses': [],
            'perm_changes': [],
        }
        
        for it in range(iterations):
            # Apply current coupling
            X0_coupled = coupling.apply_forward(X0)
            X0_coupled_t = torch.from_numpy(X0_coupled).float().to(self.device)
            
            # Forward pass through model
            with torch.no_grad():
                X1_proposed = self.model.sample(X0_coupled_t)
            
            # Optimal assignment via Hungarian
            X1_proposed_np = X1_proposed.cpu().numpy()
            cost_matrix = np.linalg.norm(
                X1_proposed_np[:, :, None] - X1_t.cpu().numpy()[:, None, :],
                axis=0
            )
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Build permutation: perm[i] tells us which target sample to pair with input sample i
            perm_new = np.empty(n, dtype=np.int64)
            for row, col in zip(row_ind, col_ind):
                perm_new[row] = col
            
            # Verify it's a valid permutation
            assert len(np.unique(perm_new)) == n, f"Permutation has duplicates: unique={len(np.unique(perm_new))}, n={n}"
            assert perm_new.min() == 0 and perm_new.max() == n - 1, f"Permutation out of range"
            
            # Compute permutation change
            perm_change = np.sum(perm_new != coupling.forward_perm) / n
            metrics['perm_changes'].append(perm_change)
            
            # Update coupling
            coupling.set_forward_perm(perm_new)
            
            # Train step
            self.model.train()
            self.optimizer.zero_grad()
            
            X1_coupled = coupling.apply_forward(X1)
            X1_coupled_t = torch.from_numpy(X1_coupled).float().to(self.device)
            
            output = self.model(X0_coupled_t)
            loss = self.loss_fn(output, X1_coupled_t)
            
            loss.backward()
            self.optimizer.step()
            
            metrics['iterations'].append(it + 1)
            metrics['losses'].append(loss.item())
            
            print(f"[Iteration {it+1:2d}] Loss: {loss.item():.6f}, "
                  f"Perm change: {perm_change:.2%}")
        
        return metrics


# ============================================================================
# VISUALIZATION (updated for Letter S)
# ============================================================================

def visualize_grid_structure(
    data: np.ndarray,
    grid_size: int,
    title: str,
    save_path: str = None,
    cmap: str = 'YlOrRd'
) -> np.ndarray:
    """Visualize average grid structure as heatmap."""
    heatmap = data.reshape(-1, grid_size, grid_size).mean(axis=0)
    
    if save_path:
        plt.figure(figsize=(6, 6))
        plt.imshow(heatmap, cmap=cmap, origin='upper')
        plt.colorbar(label='Activation Frequency')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return heatmap


def visualize_4panel(
    heatmap_input: np.ndarray,
    heatmap_target: np.ndarray,
    heatmap_prediction: np.ndarray,
    losses: list,
    save_path: str = 'discretized_markovian_viz_letter_s.png'
):
    """Create 4-panel visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Top-left: Input
    im1 = axes[0, 0].imshow(heatmap_input, cmap='Greens', origin='upper')
    axes[0, 0].set_title('Input: Gaussian Grid (Average)', fontweight='bold')
    axes[0, 0].set_xlabel('Grid X')
    axes[0, 0].set_ylabel('Grid Y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Top-right: Loss trajectory
    axes[0, 1].plot(losses, 'o-', color='green', linewidth=2, markersize=6)
    axes[0, 1].set_title('Loss Trajectory (BCE)', fontweight='bold')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bottom-left: Target
    im3 = axes[1, 0].imshow(heatmap_target, cmap='YlOrRd', origin='upper')
    axes[1, 0].set_title('Target: Letter S Grid (Average)', fontweight='bold')
    axes[1, 0].set_xlabel('Grid X')
    axes[1, 0].set_ylabel('Grid Y')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Bottom-right: Prediction
    im4 = axes[1, 1].imshow(heatmap_prediction, cmap='YlOrRd', origin='upper')
    axes[1, 1].set_title('Synthetic: Model Prediction (Average)', fontweight='bold')
    axes[1, 1].set_xlabel('Grid X')
    axes[1, 1].set_ylabel('Grid Y')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 4-panel visualization saved: {save_path}")


# ============================================================================
# MAIN (updated to use Letter S)
# ============================================================================

def main(
    grid_size: int = 10,
    n_samples: int = 300,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    iterations: int = 10,
    seed: int = 42,
):
    """Run complete experiment with Letter S target."""
    
    print("\n" + "="*70)
    print("GRID-BASED MARKOVIAN PROJECTION WITH LETTER S TARGET")
    print("="*70 + "\n")
    
    # ====== Data Generation ======
    print("[Data Generation]")
    print(f"  Grid size: {grid_size}×{grid_size} = {grid_size**2} cells per sample")
    
    X0 = generate_gaussian_grid_2d(grid_size, n_samples, seed)
    print(f"  ✅ Generated Gaussian grid: shape {X0.shape}")
    
    X1 = generate_letter_s_grid_2d(grid_size, n_samples, seed)
    print(f"  ✅ Generated Letter S grid: shape {X1.shape}")
    
    sparsity_x0 = 1 - X0.mean()
    sparsity_x1 = 1 - X1.mean()
    print(f"  Sparsity: X0={sparsity_x0:.1%}, X1={sparsity_x1:.1%}\n")
    
    # ====== Visualization: Individual Structures ======
    print("[Grid Structure Visualization]")
    heatmap_gaussian = visualize_grid_structure(
        X0, grid_size,
        "Input: Gaussian Grid (Average)",
        save_path='gaussian_grid_structure.png',
        cmap='Greens'
    )
    print("  ✅ Gaussian grid structure: gaussian_grid_structure.png")
    
    heatmap_letter_s = visualize_grid_structure(
        X1, grid_size,
        "Target: Letter S Grid (Average)",
        save_path='letter_s_grid_structure.png',
        cmap='YlOrRd'
    )
    print("  ✅ Letter S grid structure: letter_s_grid_structure.png\n")
    
    # ====== Model Setup ======
    print("[Model Setup]")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    model = SimpleD3PM(grid_size**2, hidden_dim)
    print(f"  Model: input_dim={grid_size**2}, hidden_dim={hidden_dim}\n")
    
    # ====== Trainer Setup ======
    print("[Training]")
    trainer = DiscretizedTrainerPermutation(
        model,
        device=device,
        learning_rate=learning_rate,
    )
    
    coupling = PermutationCoupling(n_samples)
    
    print(f"  Learning rate: {learning_rate}")
    print(f"  Iterations: {iterations}\n")
    
    # ====== Training ======
    metrics = trainer.train_iteration(X0, X1, coupling, iterations)
    
    # ====== Results ======
    print("\n[Results Summary]")
    print(f"  Initial loss: {metrics['losses'][0]:.6f}")
    print(f"  Final loss: {metrics['losses'][-1]:.6f}")
    reduction = (metrics['losses'][0] - metrics['losses'][-1]) / metrics['losses'][0]
    print(f"  Loss reduction: {reduction:.2%}\n")
    
    # ====== Final Prediction ======
    print("[Final Prediction]")
    model.eval()
    X0_t = torch.from_numpy(X0).float().to(device)
    with torch.no_grad():
        X_pred = model.sample(X0_t).cpu().numpy()
    
    heatmap_pred = X_pred.reshape(-1, grid_size, grid_size).mean(axis=0)
    print(f"  Generated {n_samples} synthetic samples from Gaussian")
    print(f"  Model transformed them toward Letter S\n")
    
    # ====== 4-Panel Visualization ======
    visualize_4panel(
        heatmap_gaussian,
        heatmap_letter_s,
        heatmap_pred,
        metrics['losses'],
        save_path='discretized_markovian_viz_letter_s.png'
    )
    
    # ====== Save Results ======
    print("[Saving Results]")
    results = {
        'grid_size': grid_size,
        'n_samples': n_samples,
        'hidden_dim': hidden_dim,
        'learning_rate': learning_rate,
        'iterations': iterations,
        'seed': seed,
        'losses': metrics['losses'],
        'perm_changes': metrics['perm_changes'],
        'initial_loss': float(metrics['losses'][0]),
        'final_loss': float(metrics['losses'][-1]),
        'loss_reduction': float(reduction),
        'sparsity_x0': float(sparsity_x0),
        'sparsity_x1': float(sparsity_x1),
    }
    
    with open('discretized_markovian_results_letter_s.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("  ✅ Results saved: discretized_markovian_results_letter_s.json\n")
    
    print("="*70)
    print("✅ EXPERIMENT COMPLETE!")
    print("="*70 + "\n")
    
    return model, metrics, results


if __name__ == '__main__':
    main(
        grid_size=10,
        n_samples=100,
        hidden_dim=64,
        learning_rate=1e-3,
        iterations=100,
        seed=42,
    )
