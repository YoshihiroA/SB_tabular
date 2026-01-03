# dsbm_tabular_bridge_fixed_v2.py
# ==========================
# DSBMTabularBridge wrapper for ASBM - WITH sigma AND learning_rate AS PARAMETERS
# FIXED: Use ASBM's native 4-argument API (x0_train, x1_train, x0_test, x1_test)
# No internal data modification - works with input data as-is

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from functools import partial
import copy
import ot as pot
from typing import List, Optional, Tuple, Any, Dict
from sklearn.preprocessing import StandardScaler
import math

device = 'cpu'
# Redefine MLP
class MLP(nn.Module):
  def __init__(self, input_dim, layer_widths=[100,100,2], activate_final = False, activation_fn=F.tanh):
    super(MLP, self).__init__()
    layers = []
    prev_width = input_dim
    for layer_width in layer_widths:
      layers.append(torch.nn.Linear(prev_width, layer_width))
      prev_width = layer_width
    self.input_dim = input_dim
    self.layer_widths = layer_widths
    self.layers = nn.ModuleList(layers)
    self.activate_final = activate_final
    self.activation_fn = activation_fn

  def forward(self, x):
    for i, layer in enumerate(self.layers[:-1]):
      x = self.activation_fn(layer(x))
    x = self.layers[-1](x)
    if self.activate_final:
      x = self.activation_fn(x)
    return x


# Redefine ScoreNetwork
class ScoreNetwork(nn.Module):
  def __init__(self, input_dim, layer_widths=[100,100,2], activate_final = False, activation_fn=F.tanh):
    super().__init__()
    self.net = MLP(input_dim, layer_widths=layer_widths, activate_final=activate_final, activation_fn=activation_fn)

  def forward(self, x_input, t):
    inputs = torch.cat([x_input, t], dim=1)
    return self.net(inputs)


# Redefine ASBM with corrected get_train_tuple
class DSBM(nn.Module):
  def __init__(self, net_fwd=None, net_bwd=None, num_steps=1000, sig=0, eps=1e-1, first_coupling="ind"): # Increased eps
    super().__init__()
    self.net_fwd = net_fwd
    self.net_bwd = net_bwd
    self.net_dict = {"f": self.net_fwd, "b": self.net_bwd}
    self.N = num_steps
    self.sig = sig
    self.eps = eps
    self.first_coupling = first_coupling

  @torch.no_grad()
  def get_train_tuple(self, x_pairs=None, fb='', **kwargs):
    z0, z1 = x_pairs[:, 0], x_pairs[:, 1]
    t = torch.rand((z1.shape[0], 1), device=device) * (1-2*self.eps) + self.eps
    z_t = t * z1 + (1.-t) * z0 # Linear interpolation
    z = torch.randn_like(z_t) # Standard normal noise (epsilon)
    eps_safe = 1e-6
    z_t = z_t + self.sig * torch.sqrt(t*(1.-t)) * z # Add noise based on DSBM formulation

    if fb == 'f':
      # Target from DSBM formulation
      target = z1 - z0
      sqrt_ratio = torch.sqrt(t / torch.clamp(1.0 - t, min=eps_safe))
      target = target - self.sig * sqrt_ratio * z
    else:
      # Target from DSBM formulation
      target = - (z1 - z0)
      sqrt_ratio = torch.sqrt(torch.clamp(1.0 - t, min=eps_safe) / torch.clamp(t, min=eps_safe))
      target = target - self.sig * sqrt_ratio * z
    return z_t, t, target

  @torch.no_grad()
  def generate_new_dataset(self, x_pairs, prev_model=None, fb='', first_it=False):
    assert fb in ['f', 'b']

    if prev_model is None:
      assert first_it
      assert fb == 'b'
      zstart = x_pairs[:, 0]
      if self.first_coupling == "ref":
        zend = zstart + torch.randn_like(zstart) * self.sig
      elif self.first_coupling == "ind":
        zend = x_pairs[:, 1].clone()
        zend = zend[torch.randperm(len(zend))]
      else:
        raise NotImplementedError
      z0, z1 = zstart, zend
    else:
      assert not first_it
      if prev_model.fb == 'f':
        zstart = x_pairs[:, 0]
      else:
        zstart = x_pairs[:, 1]
      zend = prev_model.sample_sde(zstart=zstart, fb=prev_model.fb)[-1]
      if prev_model.fb == 'f':
        z0, z1 = zstart, zend
      else:
        z0, z1 = zend, zstart
    return z0, z1

  @torch.no_grad()
  def sample_sde(self, zstart=None, N=None, fb='', first_it=False):
    assert fb in ['f', 'b']
    if N is None:
      N = self.N
    dt = 1./N
    traj = []
    z = zstart.detach().clone()
    batchsize = z.shape[0]

    traj.append(z.detach().clone())
    ts = np.arange(N) / N
    if fb == 'b':
      ts = 1 - ts

    for i in range(N):
      t = torch.ones((batchsize,1), device=device) * ts[i]
      pred = self.net_dict[fb](z, t)
      z = z.detach().clone() + pred * dt
      z = z + self.sig * torch.randn_like(z) * np.sqrt(dt)
      traj.append(z.detach().clone())

    return traj


# Redefine train_asbm function
def train_dsbm(asbm_ipf, x_pairs, batch_size, inner_iters, lr, prev_model=None, fb='', first_it=False):
  assert fb in ['f', 'b']
  asbm_ipf.fb = fb
  optimizer = torch.optim.Adam(asbm_ipf.net_dict[fb].parameters(), lr=lr)
  loss_curve = []

  dl = iter(DataLoader(TensorDataset(*asbm_ipf.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)),
                       batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))

  for i in range(inner_iters):
    try:
      z0, z1 = next(dl)
    except StopIteration:
      dl = iter(DataLoader(TensorDataset(*asbm_ipf.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)),
                           batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))
      z0, z1 = next(dl)

    z_pairs = torch.stack([z0, z1], dim=1)
    z_t, t, target = asbm_ipf.get_train_tuple(z_pairs, fb=fb, first_it=first_it)
    optimizer.zero_grad()
    pred = asbm_ipf.net_dict[fb](z_t, t)
    loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)

    loss = loss.mean()
    loss.backward()

    if torch.isnan(loss).any():
      raise ValueError("Loss is nan")

    optimizer.step()
    loss_curve.append(np.log(loss.item()))

  return asbm_ipf, loss_curve


class DSBMTabularBridge:
    """
    Wrapper for ASBM that matches DSBTabularBridge interface.
    
    Uses ASBM's native 4-argument initialization:
    - x0_train: Training data distribution 0 (e.g., Gaussian noise)
    - x1_train: Training data distribution 1 (e.g., real data)
    - x0_test: Test data distribution 0
    - x1_test: Test data distribution 1
    
    No internal data modification - uses inputs directly.
    
    OPTIMIZABLE PARAMETERS:
    - sigma: Diffusion noise level (float, [0.1, 2.0])
    - learning_rate: Training learning rate (float, [1e-5, 1e-2])
    """

    def __init__(
        self,
        x0_train: np.ndarray,
        x1_train: np.ndarray,
        x0_test: Optional[np.ndarray] = None,
        x1_test: Optional[np.ndarray] = None,
        num_timesteps: int = 200,
        epsilon: float = 0.664,
        sigma: float = 0.5,
        learning_rate: float = 1e-4,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """
        Initialize DSBMTabularBridge with DSBM-compatible API.
        
        Args:
            x0_train: Distribution 0 training data (n_train, n_features)
            x1_train: Distribution 1 training data (n_train, n_features)
            x0_test: Distribution 0 test data (n_test, n_features)
            x1_test: Distribution 1 test data (n_test, n_features)
            num_timesteps: Number of diffusion timesteps
            epsilon: Epsilon parameter for DSBM
            sigma: Diffusion noise level (OPTIMIZABLE) - default 0.5
            learning_rate: Learning rate for training (OPTIMIZABLE) - default 1e-4
            device: torch device (cuda or cpu)
            **kwargs: Additional arguments (ignored)
        """
        # Store data as-is (no modification)
        self.x0_train = np.asarray(x0_train, dtype=np.float32)
        self.x1_train = np.asarray(x1_train, dtype=np.float32)
        
        # Test data (optional)
        if x0_test is not None:
            self.x0_test = np.asarray(x0_test, dtype=np.float32)
        else:
            self.x0_test = None
        
        if x1_test is not None:
            self.x1_test = np.asarray(x1_test, dtype=np.float32)
        else:
            self.x1_test = None
        
        # Determine device
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data dimension (inferred from training data)
        self.D = self.x0_train.shape[1]
        
        # ASBM parameters
        self.num_timesteps = num_timesteps
        self.epsilon = epsilon
        
        # ===== OPTIMIZABLE PARAMETERS =====
        self.sigma = sigma
        self.learning_rate = learning_rate
        
        # Model will be initialized during fit()
        self.model = None
        self.is_trained = False
        self.model_list = []

    def fit(
        self,
        imf_iters: int = 5,
        inner_iters: int = 1000,
        batch_size: int = 256,
        learning_rate: Optional[float] = 1e-4,
        layers=[256, 256, 256],
        verbose: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train DSBM on the provided train/test data.
        
        Args:
            imf_iters: Number of outer iterations (IMF iterations)
            inner_iters: Number of inner iterations per outer iteration
            batch_size: Batch size for training
            learning_rate: Learning rate (uses self.learning_rate if None)
            verbose: Print training progress
            **kwargs: Additional arguments
            
        Returns:
            Training history/logs
        """
        # Use provided learning_rate or fall back to self.learning_rate
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        
        # Convert to torch tensors on device
        x0_train_t = torch.from_numpy(self.x0_train).float().to(self.device)
        x1_train_t = torch.from_numpy(self.x1_train).float().to(self.device)
        
        # Create paired training data for ASBM (stacked as [x0, x1])
        x_pairs = torch.stack([x0_train_t, x1_train_t], dim=1).to(self.device)
        
        if verbose:
            print(f"Training DSBM with:")
            print(f"  num_timesteps={self.num_timesteps}")
            print(f"  epsilon={self.epsilon}")
            print(f"  sigma={self.sigma}")
            print(f"  learning_rate={learning_rate}")
            print(f"  Training pairs shape: {x_pairs.shape}")
            print(f"  Batch size: {batch_size}, IMF iters: {imf_iters}, Inner iters: {inner_iters}")
        
        # Create ScoreNetwork for DSBM
        def create_network():
            return ScoreNetwork(
                input_dim=self.D + 1,  # data dimension + time dimension
                layer_widths=layers + [self.D],
                activation_fn=torch.nn.LeakyReLU(),
            ).to(self.device)
        
        # Initialize DSBM model (use self.sigma here)
        self.model = DSBM(
            net_fwd=create_network(),
            net_bwd=create_network(),
            num_steps=self.num_timesteps,
            sig=self.sigma,
            eps=1e-2,
            first_coupling='ind',
        ).to(self.device)
        
        # Train DSBM using IMF (Iterative Markovian Fitting)
        training_history = {'loss_f': [], 'loss_b': []}
        
        for it in range(1, imf_iters + 1):
            if verbose:
                print(f"Outer Iteration {it}/{imf_iters}")
            
            if it == 1:
                # First iteration: train backward network only
                if verbose:
                    print("  First iteration - training backward network...")
                
                self.model, loss_bwd = train_dsbm(
                    self.model,
                    x_pairs,
                    batch_size=batch_size,
                    inner_iters=inner_iters,
                    lr=learning_rate,
                    prev_model=None,
                    fb='b',
                    first_it=True,
                )
                
                training_history['loss_b'].extend(loss_bwd)
                self.model_list.append(copy.deepcopy(self.model).to(self.device))
            
            else:
                # Alternate between forward and backward
                if verbose:
                    print("  Training forward network...")
                
                self.model, loss_fwd = train_dsbm(
                    self.model,
                    x_pairs,
                    batch_size=batch_size,
                    inner_iters=inner_iters,
                    lr=learning_rate,
                    prev_model=self.model_list[-1],
                    fb='f',
                    first_it=False,
                )
                
                training_history['loss_f'].extend(loss_fwd)
                self.model_list.append(copy.deepcopy(self.model).to(self.device))
                
                if verbose:
                    print("  Training backward network...")
                
                self.model, loss_bwd = train_dsbm(
                    self.model,
                    x_pairs,
                    batch_size=batch_size,
                    inner_iters=inner_iters,
                    lr=learning_rate,
                    prev_model=self.model_list[-1],
                    fb='b',
                    first_it=False,
                )
                
                training_history['loss_b'].extend(loss_bwd)
                self.model_list.append(copy.deepcopy(self.model).to(self.device))
        
        if verbose:
            print("Training completed!")
        
        self.is_trained = True
        return training_history

    def generate(
        self,
        n_samples: int,
        direction: str = 'forward'
    ) -> np.ndarray:
        """
        Generate synthetic samples using trained ASBM.
        
        Args:
            n_samples: Number of samples to generate
            direction: 'forward' (x0->x1) or 'backward' (x1->x0)
            
        Returns:
            Generated synthetic data (n_samples, n_features)
        """
        if not self.is_trained or not self.model_list:
            raise RuntimeError("Model must be trained before generating samples")
        
        assert direction in ['forward', 'backward'], "direction must be 'forward' or 'backward'"
        
        # Use the latest trained model
        current_model = self.model_list[-1]
        fb = 'f' if direction == 'forward' else 'b'
        
        # Determine starting distribution
        z_start = torch.from_numpy(self.x0_train[:n_samples]).float().to(self.device)
        
        with torch.no_grad():
            trajectory = current_model.sample_sde(zstart=z_start, N=None, fb=fb, first_it=False)
            synthetic_data = trajectory[-1]  # Get final samples from trajectory
        
        return synthetic_data.cpu().numpy().astype(np.float32)

    @property
    def traindata(self):
        """Compatibility property for old interface."""
        return self.x0_train
    
    @property
    def testdata(self):
        """Compatibility property for old interface."""
        return self.x1_train
