import torch
from torch_ema import ExponentialMovingAverage
import torch.nn as nn
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import torch.nn.functional as F
from functools import partial
from sklearn.metrics.pairwise import pairwise_distances
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from torch.nn.functional import softmax

from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gumbel import Gumbel

from sklearn.decomposition import PCA
from scipy.stats import ortho_group
from scipy.linalg import sqrtm, inv
from zipfile import ZipFile
import gdown
from os import environ, listdir, makedirs
from os.path import expanduser, isdir, join, splitext

def smooth_losses(losses: List[float], smoothing_factor: float = 0.1) -> List[float]:
    """Smooth loss values using exponential moving average (EMA)."""
    if not losses or len(losses) == 0:
        return losses
    
    smoothed = []
    ema = losses[0]
    
    for loss in losses:
        ema = smoothing_factor * loss + (1 - smoothing_factor) * ema
        smoothed.append(ema)
    
    return smoothed

def get_data_home(data_home=None) -> str:
    if data_home is None:
        data_home = environ.get("EOT_BENCHMARK_DATA", join("~", "eot_benchmark_data"))
    data_home = expanduser(data_home)
    makedirs(data_home, exist_ok=True)
    return data_home

class GaussianMixture:
    def __init__(self, probs: torch.tensor,
                 mus: torch.tensor,
                 sigmas: torch.tensor,
                 device: str):
#         assert torch.allclose(probs.sum(), torch.ones(1))
        self.device = device
    
        probs = probs.to(device)
        mus = mus.to(device)
        sigmas = sigmas.to(device)
    
        self.probs = probs
        self.components_distirubtion = Categorical(probs)
        
        assert len(probs) == len(mus) and len(mus) == len(sigmas)
        self.gaussians_distributions = [
            MultivariateNormal(loc=mu, covariance_matrix=sigma) for mu, sigma in zip(mus, sigmas)
        ]
        self.dim = mus[0].shape[0]
        
    def sample(self, n_samples:int =1) -> torch.tensor:
        components = self.components_distirubtion.sample(sample_shape=torch.Size((n_samples,))).to(self.device)
        
        gaussian_samples = [
            gaussian_distribution.sample(sample_shape=torch.Size((n_samples,))) for gaussian_distribution in self.gaussians_distributions
        ]
        gaussian_samples = torch.stack(gaussian_samples, dim=1)
            
        gaussian_mixture_samples = gaussian_samples.gather(1, components[:, None, None].expand(components.shape[0], 1, self.dim)).squeeze()
        
        return gaussian_mixture_samples
    
    
class ConditionalCategoricalDistribution:
    def __init__(self):
        self.gumbel_distribution = Gumbel(0, 1)
        
    def sample(self, log_probs: torch.tensor) -> torch.tensor:
        gumbel_samples = self.gumbel_distribution.sample(log_probs.shape).to(log_probs.device)
        return torch.argmax(gumbel_samples + log_probs, dim=1)
    
    
class PotentialCategoricalDistribution:
    def __init__(self, potential_probs: torch.tensor,
                 potential_mus: torch.tensor,
                 potential_sigmas: torch.tensor,
                 eps: float):
        
        device = potential_probs.device
        self.dim = potential_mus[0].shape[0]
        
        self.log_probs = torch.log(potential_probs)
        
        eps = torch.tensor(eps).to(device)
        identity = torch.diag(torch.ones(self.dim)).to(device)
        
        self.potential_gaussians_distributions = [
            MultivariateNormal(loc=mu, covariance_matrix=sigma + eps*identity) for mu, sigma in zip(potential_mus, potential_sigmas)
        ]
        self.categorical_distribution = ConditionalCategoricalDistribution()
    
    
    def calculate_log_probs(self, x):
        log_probs = [
            log_prob + distribution.log_prob(x) for log_prob, distribution in zip(self.log_probs, self.potential_gaussians_distributions)
        ]
        log_probs = torch.stack(log_probs, dim=1)
        
        return log_probs
    
    
    def sample(self, x) -> torch.tensor:
        log_probs = self.calculate_log_probs(x)
        
        return self.categorical_distribution.sample(log_probs)
    

class ConditionalPlan:
    def __init__(self, potential_probs: torch.tensor, 
                 potential_mus: torch.tensor, 
                 potential_sigmas: torch.tensor,
                 eps: float,
                 device: str):
#         assert torch.allclose(potential_probs.sum(), torch.ones(1))
        assert len(potential_probs) == len(potential_mus) and len(potential_mus) == len(potential_sigmas)
        assert eps > 0
        
        self.device = device
        potential_probs = potential_probs.to(device)
        potential_mus = potential_mus.to(device)
        potential_sigmas = potential_sigmas.to(device)
        
        self.potential_probs = potential_probs
        self.potential_mus = potential_mus
        self.potential_sigmas = potential_sigmas
        
        self.dim = potential_mus[0].shape[0]
        
        self.components_distirubtion = PotentialCategoricalDistribution(
            potential_probs, potential_mus, potential_sigmas, eps
        )
        
        eps = torch.tensor(eps).to(device)
        identity = torch.diag(torch.ones(self.dim)).to(device)
        
        plan_sigmas = [torch.linalg.inv((1/eps) * identity + torch.linalg.inv(sigma)) for sigma in potential_sigmas]
        plan_mu_biases = [plan_sigma@torch.linalg.inv(sigma)@mu for mu, sigma, plan_sigma in zip(potential_mus, potential_sigmas, plan_sigmas)]
        self.plan_mu_weights = [plan_sigma/eps for plan_sigma in plan_sigmas]
        
        self.gaussians_distributions = [
            MultivariateNormal(loc=mu, covariance_matrix=sigma) for mu, sigma in zip(plan_mu_biases, plan_sigmas)
        ]
        
        
    def sample(self, x: torch.tensor) -> torch.tensor:
        batch_size = x.shape[0]
        components = self.components_distirubtion.sample(x)
        
        gaussian_samples = [
            gaussian_distribution.sample([batch_size]) + (plan_mu_weight@x.T).T for plan_mu_weight, gaussian_distribution in zip(self.plan_mu_weights, self.gaussians_distributions)
        ]
        gaussian_samples = torch.stack(gaussian_samples, dim=1)
            
        gaussian_mixture_samples = gaussian_samples.gather(1, components[:, None, None].expand(components.shape[0], 1, self.dim)).squeeze()
        
        return gaussian_mixture_samples

    
def download_gaussian_mixture_benchmark_files():    
    path = get_data_home()
    urls = {
        "gaussian_mixture_benchmark_data.zip": "https://drive.google.com/uc?id=1HNXbrkozARbz4r8fdFbjvPw8R74n1oiY",
    }
        
    for name, url in urls.items():
        gdown.download(url, os.path.join(path, f"{name}"), quiet=False)
        
    with ZipFile(os.path.join(path, "gaussian_mixture_benchmark_data.zip"), 'r') as zip_ref:
        zip_ref.extractall(path)
        
        
class OutputSampler:
    def __init__(self, gm, conditional_plan):
        self.gm = gm
        self.conditional_plan = conditional_plan
        
    def sample(self, n_samples: int =1):
        return self.conditional_plan.sample(self.gm.sample(n_samples))
    
    
class PlanSampler:
    def __init__(self, gm, conditional_plan):
        self.gm = gm
        self.conditional_plan = conditional_plan
        
    def sample(self, n_samples: int =1):
        input_samples = self.gm.sample(n_samples)
        output_samples = self.conditional_plan.sample(input_samples)
        return input_samples, output_samples
        
        
def get_guassian_mixture_benchmark_sampler(input_or_target: str, dim: int, eps: float,
                                           batch_size: int, device: str ="cpu", download: bool =False):
    assert input_or_target in ["input", "target"]
    assert dim in [2, 4, 8, 16, 32, 64, 128]
    assert eps in [0.01, 0.1, 1, 10]
    
    if download:
        download_gaussian_mixture_benchmark_files()
        
    benchmark_data_path = get_data_home()
        
    probs = torch.load(os.path.join(benchmark_data_path, f"input_probs_dim_{dim}.torch"))
    mus = torch.load(os.path.join(benchmark_data_path, f"input_mus_dim_{dim}.torch"))
    sigmas = torch.load(os.path.join(benchmark_data_path, f"input_sigmas_dim_{dim}.torch"))
    
    gm = GaussianMixture(probs, mus, sigmas, device=device)
        
    if input_or_target == "input":
        return gm
    else:
        probs = torch.load(os.path.join(benchmark_data_path, f"potential_probs_dim_{dim}_eps_{eps}.torch"))
        mus = torch.load(os.path.join(benchmark_data_path, f"potential_mus_dim_{dim}_eps_{eps}.torch"))
        sigmas = torch.load(os.path.join(benchmark_data_path, f"potential_sigmas_dim_{dim}_eps_{eps}.torch"))
        
        conditional_plan = ConditionalPlan(probs, mus, sigmas, eps, device=device)
        
        return OutputSampler(gm, conditional_plan)
    
    
def get_guassian_mixture_benchmark_ground_truth_sampler(dim: int, eps: float, batch_size: int,
                                                        device: str ="cpu", download: bool =False):
    assert dim in [2, 4, 8, 16, 32, 64, 128]
    assert eps in [0.01, 0.1, 1, 10]
    
    if download:
        download_gaussian_mixture_benchmark_files()
        
    benchmark_data_path = get_data_home()
        
    probs = torch.load(os.path.join(benchmark_data_path, f"input_probs_dim_{dim}.torch"))
    mus = torch.load(os.path.join(benchmark_data_path, f"input_mus_dim_{dim}.torch"))
    sigmas = torch.load(os.path.join(benchmark_data_path, f"input_sigmas_dim_{dim}.torch"))
    
    gm = GaussianMixture(probs, mus, sigmas, device=device)
    
    probs = torch.load(os.path.join(benchmark_data_path, f"potential_probs_dim_{dim}_eps_{eps}.torch"))
    mus = torch.load(os.path.join(benchmark_data_path, f"potential_mus_dim_{dim}_eps_{eps}.torch"))
    sigmas = torch.load(os.path.join(benchmark_data_path, f"potential_sigmas_dim_{dim}_eps_{eps}.torch"))

    conditional_plan = ConditionalPlan(probs, mus, sigmas, eps, device=device)
        
    return PlanSampler(gm, conditional_plan)


def get_test_input_samples(dim, device="cpu"):
    benchmark_data_path = get_data_home()
    return torch.load(os.path.join(benchmark_data_path, "test_inputs", f"test_data_dim_{dim}.torch")).to(device)


class LoaderFromSampler:
    def __init__(self, sampler, batch_size, num_batches):
        self.num_batches = num_batches
        self.sampler = sampler
        self.batch_size = batch_size
        self.current_batch = 0
    
    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
            
        self.current_batch += 1
        samples = self.sampler.sample(self.batch_size)
        return samples, torch.zeros(self.batch_size)
    

class GroundTruthSDE(nn.Module):

    def __init__(self, potential_params, eps, n_steps, is_diagonal=False):
        super().__init__()
        probs, mus, sigmas = potential_params
        self.eps = eps
        self.n_steps = n_steps
        self.is_diagonal = is_diagonal

        self.register_buffer("potential_probs", probs)
        self.register_buffer("potential_mus", mus)
        self.register_buffer("potential_sigmas", sigmas)

    def forward(self, x):
        t_storage = [torch.zeros([1])]
        trajectory = [x.cpu()]
        for i in range(self.n_steps):
            delta_t = 1 / self.n_steps
            t = torch.tensor([i / self.n_steps])
            drift = self.get_drift(x, t)

            rand = np.sqrt(self.eps) * np.sqrt(delta_t) * torch.randn(*x.shape).to(x.device)

            x = (x + drift * delta_t + rand).detach()

            trajectory.append(x.cpu())
            t_storage.append(t)

        return torch.stack(trajectory, dim=0).transpose(0, 1), torch.stack(t_storage, dim=0).unsqueeze(1).repeat(
            [1, x.shape[0], 1])

    def sample(self, x):
        return self.forward(x)


# class EOTGMMSampler(Sampler):
class EOTGMMSampler:
    def __init__(self, dim, eps, batch_size=64, download=False) -> None:
        super().__init__()
        eps = eps if int(eps) < 1 else int(eps)

        self.dim = dim
        self.eps = eps
        self.x_sampler = get_guassian_mixture_benchmark_sampler(input_or_target="input", dim=dim, eps=eps,
                                                                batch_size=batch_size, device=f"cpu", download=download)
        self.y_sampler = get_guassian_mixture_benchmark_sampler(input_or_target="target", dim=dim, eps=eps,
                                                                batch_size=batch_size, device=f"cpu", download=download)
        self.gt_sampler = get_guassian_mixture_benchmark_ground_truth_sampler(dim=dim, eps=eps,
                                                                              batch_size=batch_size, device=f"cpu",
                                                                              download=download)

    def x_sample(self, batch_size):
        return self.x_sampler.sample(batch_size)

    def y_sample(self, batch_size):
        return self.y_sampler.sample(batch_size)

    def gt_sample(self, batch_size):
        return self.gt_sampler.sample(batch_size)

    def conditional_y_sample(self, x):
        return self.gt_sampler.conditional_plan.sample(x)

    def gt_sde_path_sampler(self):
        mus = self.y_sampler.conditional_plan.potential_mus
        probs = self.y_sampler.conditional_plan.potential_probs
        sigmas = self.y_sampler.conditional_plan.potential_sigmas
        potential_params = (probs, mus, sigmas)

        n_em_steps = 99
        gt_sde = GroundTruthSDE(potential_params, self.eps, n_em_steps)

        def return_fn(batch_size):
            x_samples = self.x_sample(batch_size)
            x_t, t = gt_sde.sample(x_samples)
            t = t.transpose(0, 1)
            return x_t, t

        return return_fn

    def brownian_bridge_sampler(self):
        def return_fn(batch_size):
            x_0 = self.x_sample(batch_size)

            x_1 = self.gt_sampler.conditional_plan.sample(x_0)
            t_0 = 0
            t_1 = 1
            n_timesteps = 100

            t = torch.arange(n_timesteps).reshape([-1, 1]).repeat([1, batch_size]).transpose(0, 1) / n_timesteps

            x_0 = x_0.unsqueeze(1).repeat([1, n_timesteps, 1])
            x_1 = x_1.unsqueeze(1).repeat([1, n_timesteps, 1])

            # N x T x D

            mean = x_0 + ((t - t_0) / (t_1 - t_0)).reshape([x_0.shape[0], x_0.shape[1], 1]) * (x_1 - x_0)

            std = torch.sqrt(self.eps * (t - t_0) * (t_1 - t) / (t_1 - t_0))

            x_t = mean + std.reshape([std.shape[0], std.shape[1], 1]) * torch.randn_like(mean)

            return x_t, t

        return return_fn

    def path_sampler(self):
        mus = self.y_sampler.conditional_plan.potential_mus
        probs = self.y_sampler.conditional_plan.potential_probs
        sigmas = self.y_sampler.conditional_plan.potential_sigmas
        potential_params = (probs, mus, sigmas)

        n_em_steps = 99
        gt_sde = GroundTruthSDE(potential_params, self.eps, n_em_steps)

        def return_fn(batch_size):
            x_samples = self.x_sample(batch_size)
            x_t, t = gt_sde.sample(x_samples)
            t = t.transpose(0, 1)
            return x_t, t

        return return_fn

    def __str__(self) -> str:
        return f'EOTSampler_D_{self.dim}_eps_{self.eps}'

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
      noise = torch.randn_like(x_start)
      
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise
    
    return x_t


def sample_posterior(coefficients, x_0, x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos

def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t+1, x_start.shape) * noise
    
    return x_t, x_t_plus_one

def q_sample_supervised(pos_coeff, x_start, t, x_end, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
      noise = torch.randn_like(x_start)

    # T = len(coeff.a_s_cum)
    T = len(pos_coeff.a_s_cum)

    x_t = x_end
    for t_current in reversed(list(range(t[0], T))):
        t_tensor = torch.full((x_t.size(0),), t_current, dtype=torch.int64).to(x_t.device)
        x_t = sample_posterior(pos_coeff, x_start, x_t, t_tensor)
    
    return x_t


def q_sample_supervised_pairs(pos_coeff, x_start, t, x_end):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
#     noise = torch.randn_like(x_start)
    T = pos_coeff.posterior_mean_coef1.shape[0]

    x_t_plus_one = x_end
    t_current = T

    while t_current != t[0]:
        t_tensor = torch.full((x_end.size(0),), t_current-1, dtype=torch.int64).to(x_end.device)
        x_t_plus_one = sample_posterior(pos_coeff, x_start, x_t_plus_one, t_tensor)
        t_current -= 1

    t_tensor = torch.full((x_end.size(0),), t_current, dtype=torch.int64).to(x_end.device)
    x_t = sample_posterior(pos_coeff, x_start, x_t_plus_one, t_tensor)
    
    return x_t, x_t_plus_one


def q_sample_supervised_pairs_brownian(pos_coeff, x_start, t, x_end):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    num_steps = pos_coeff.posterior_mean_coef1.shape[0]
    t_plus_one_tensor = ((t+1)/num_steps)[:, None, None, None]

    x_t_plus_one = t_plus_one_tensor*x_end + (1.0 - t_plus_one_tensor)*x_start + torch.sqrt(pos_coeff.epsilon*t_plus_one_tensor*(1-t_plus_one_tensor))*noise
    
    x_t = sample_posterior(pos_coeff, x_start, x_t_plus_one, t)
    
    return x_t, x_t_plus_one


def q_sample_supervised_trajectory(pos_coeff, x_start, x_end):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
#     noise = torch.randn_like(x_start)
    trajectory = [x_end]
    T = pos_coeff.posterior_mean_coef1.shape[0]

    x_t_plus_one = x_end
    t_current = T

    while t_current != 0:
        t_tensor = torch.full((x_end.size(0),), t_current-1, dtype=torch.int64).to(x_end.device)
        x_t_plus_one = sample_posterior(pos_coeff, x_start, x_t_plus_one, t_tensor)
        t_current -= 1
        trajectory.append(x_t_plus_one)

    t_tensor = torch.full((x_end.size(0),), t_current, dtype=torch.int64).to(x_end.device)
    x_t = sample_posterior(pos_coeff, x_start, x_t_plus_one, t_tensor)
    trajectory.append(x_t)
    
    return trajectory


def sample_from_model(coefficients, generator, n_time, x_init, opt, return_trajectory=False):
    x = x_init
    trajectory = [x]
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
          
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
            trajectory.append(x)

    if return_trajectory:
        return x, trajectory
    
    return x


def sample_from_generator(coefficients, generator, n_time, x_init, opt, append_first=False):
    x = x_init
    if append_first:
        trajectory = [x]
    else:
        trajectory = []
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
          
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, latent_z)
            trajectory.append(x_0.detach())
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()

    trajectory = torch.stack(trajectory)

    return trajectory

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class MyGenerator(nn.Module):
    def __init__(
        self, x_dim=2, t_dim=2, n_t=4, z_dim=1, out_dim=2, layers=[128, 128, 128],
        active=partial(nn.LeakyReLU, 0.2),
    ):
        super().__init__()

        self.x_dim = x_dim
        self.t_dim = t_dim
        self.z_dim = z_dim

        self.model = []
        ch_prev = x_dim + t_dim + z_dim

        self.t_transform = nn.Embedding(n_t, t_dim,)

        for ch_next in layers:
            self.model.append(nn.Linear(ch_prev, ch_next))
            self.model.append(active())
            ch_prev = ch_next

        self.model.append(nn.Linear(ch_prev, out_dim))
        self.model = nn.Sequential(*self.model)

    def forward(self, x, t, z):
        batch_size = x.shape[0]

        if z.shape != (batch_size, self.z_dim):
            z = z.reshape((batch_size, self.z_dim))

        return self.model(
            torch.cat([
                x,
                self.t_transform(t),
                z,
            ], dim=1)
        )


class MyDiscriminator(nn.Module):
    def __init__(
        self, x_dim=2, t_dim=2, n_t=4, layers=[128, 128, 128],
        active=partial(nn.LeakyReLU, 0.2),
    ):
        super().__init__()

        self.x_dim = x_dim
        self.t_dim = t_dim

        self.model = []
        ch_prev = 2 * x_dim + t_dim

        self.t_transform = nn.Embedding(n_t, t_dim,)

        for ch_next in layers:
            self.model.append(nn.Linear(ch_prev, ch_next))
            self.model.append(active())
            ch_prev = ch_next

        self.model.append(nn.Linear(ch_prev, 1))
        self.model = nn.Sequential(*self.model)

    def forward(self, x_t, t, x_tp1,):
        transform_t = self.t_transform(t)
        # print(f"x_t.shape = {x_t.shape}, transform_t = {transform_t.shape}, x_tp1 = {x_tp1.shape}")

        return self.model(
            torch.cat([
                x_t,
                transform_t,
                x_tp1,
            ], dim=1)
        ).squeeze()


class MySampler:
    def __init__(self, batch_size, sample_func, precalc=None):
        self.precalc = precalc
        self.batch_size = batch_size
        self.sample_func = sample_func
        print(f"sample_func = {sample_func}")

        if self.precalc is not None:
            self.regenerate()

    def regenerate(self):
        self.generated = self.sample_func(self.precalc * self.batch_size,)
        self.idx = 0

    def sample(self):
        if self.precalc is None:
            return self.sample_func(self.batch_size,)
        if self.idx == self.precalc:
            self.regenerate()
        ret = self.generated[self.idx * self.batch_size: (self.idx + 1) * self.batch_size]
        self.idx += 1
        return ret


def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

class Diffusion_Coefficients():
    def __init__(self, args, device):
                
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)

#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


class BrownianPosterior_Coefficients():
    def __init__(self, args, device):
        epsilon = args.epsilon
        self.epsilon = epsilon
        print(f"BrownianPosterior with epsilon {epsilon} and num steps {args.num_timesteps}")
        num_timesteps = args.num_timesteps

        t = torch.linspace(0, 1, num_timesteps+1, device=device)
        self.posterior_mean_coef1 = 1 - t[:-1]/t[1:]
        self.posterior_mean_coef2 = t[:-1]/t[1:]

        self.posterior_variance = epsilon*t[:-1]*(t[1:] - t[:-1])/t[1:]
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

def sample_from_model(coefficients, generator, n_time, x_init, opt):
    x = x_init
    x_traj = [x_init]
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
            x_traj.append(x)

    traj = torch.stack(x_traj, dim=0)
    return x, traj

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class MyGenerator(nn.Module):
    def __init__(
        self, x_dim=2, t_dim=2, n_t=4, z_dim=1, out_dim=2, layers=[128, 128, 128],
        active=partial(nn.LeakyReLU, 0.2),
    ):
        super().__init__()

        self.x_dim = x_dim
        self.t_dim = t_dim
        self.z_dim = z_dim

        self.model = []
        ch_prev = x_dim + t_dim + z_dim

        self.t_transform = nn.Embedding(n_t, t_dim,)

        for ch_next in layers:
            self.model.append(nn.Linear(ch_prev, ch_next))
            self.model.append(active())
            ch_prev = ch_next

        self.model.append(nn.Linear(ch_prev, out_dim))
        self.model = nn.Sequential(*self.model)

    def forward(self, x, t, z):
        batch_size = x.shape[0]

        if z.shape != (batch_size, self.z_dim):
            z = z.reshape((batch_size, self.z_dim))

        return self.model(
            torch.cat([
                x,
                self.t_transform(t),
                z,
            ], dim=1)
        )


class MyDiscriminator(nn.Module):
    def __init__(
        self, x_dim=2, t_dim=2, n_t=4, layers=[128, 128, 128],
        active=partial(nn.LeakyReLU, 0.2),
    ):
        super().__init__()

        self.x_dim = x_dim
        self.t_dim = t_dim

        self.model = []
        ch_prev = 2 * x_dim + t_dim

        self.t_transform = nn.Embedding(n_t, t_dim,)

        for ch_next in layers:
            self.model.append(nn.Linear(ch_prev, ch_next))
            self.model.append(active())
            ch_prev = ch_next

        self.model.append(nn.Linear(ch_prev, 1))
        self.model = nn.Sequential(*self.model)

    def forward(self, x_t, t, x_tp1,):
        transform_t = self.t_transform(t)
        # print(f"x_t.shape = {x_t.shape}, transform_t = {transform_t.shape}, x_tp1 = {x_tp1.shape}")

        return self.model(
            torch.cat([
                x_t,
                transform_t,
                x_tp1,
            ], dim=1)
        ).squeeze()


class MySampler:
    def __init__(self, batch_size, sample_func, precalc=None):
        self.precalc = precalc
        self.batch_size = batch_size
        self.sample_func = sample_func
        print(f"sample_func = {sample_func}")

        if self.precalc is not None:
            self.regenerate()

    def regenerate(self):
        self.generated = self.sample_func(self.precalc * self.batch_size,)
        self.idx = 0

    def sample(self):
        if self.precalc is None:
            return self.sample_func(self.batch_size,)
        if self.idx == self.precalc:
            self.regenerate()
        ret = self.generated[self.idx * self.batch_size: (self.idx + 1) * self.batch_size]
        self.idx += 1
        return ret


def sample_posterior(coefficients, x_0,x_t, t):

    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped


    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))
        while len(nonzero_mask.shape) < len(mean.shape):
            nonzero_mask = nonzero_mask.unsqueeze(-1)

        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos


def sample_from_model(coefficients, generator, n_time, x_init, opt):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()

    return x


class OneDistSampler:
    def __init__(self) -> None:
        pass

    def __call__(self, batch_size):
        return self.sample(batch_size)

    def sample(self, batch_size):
        raise NotImplementedError('Abstract Class')

    def __str__(self):
        return self.__class__.__name__


class GaussianDist(OneDistSampler):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def sample(self, batch_size):
        return torch.randn([batch_size, self.dim])


def q_sample_supervised_pairs_brownian(pos_coeff, x_start, t, x_end):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    num_steps = pos_coeff.posterior_mean_coef1.shape[0]
    t_plus_one_tensor = ((t + 1) / num_steps)[:, None]

    x_t_plus_one = t_plus_one_tensor * x_end + (1.0 - t_plus_one_tensor) * x_start + torch.sqrt(
        pos_coeff.epsilon * t_plus_one_tensor * (1 - t_plus_one_tensor)) * noise

    x_t = sample_posterior(pos_coeff, x_start, x_t_plus_one, t)

    return x_t, x_t_plus_one


class ASBMTabularBridge:
    """
    Wrapper for ASBM that aligns with DSBTabularBridge interface.
    
    Key features:
    - Accepts pre-split x0_train, x1_train, x0_test, x1_test directly
    - Learns a bridge between x0_train and x1_train distributions
    - Generates synthetic x1 samples from x0_test
    - Evaluates quality using metrics (Wasserstein, MMD, etc.)
    - Compatible with DSB experimental pipeline
    """
    
    def __init__(
        self,
        x0_train: np.ndarray,
        x1_train: np.ndarray,
        x0_test: np.ndarray,
        x1_test: np.ndarray,
        global_scaler: Optional[StandardScaler] = None,
        categorical_columns: Optional[List[int]] = None,
        continuous_columns: Optional[List[int]] = None,
        num_timesteps: int = 4,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        epsilon: float = 1.0,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """
        Initialize ASBMTabularBridge.
        
        Args:
            x0_train: Training data for source distribution (n_train, n_features)
            x1_train: Training data for target distribution (n_train, n_features)
            x0_test: Test data for source distribution (n_test, n_features)
            x1_test: Test data for target distribution (n_test, n_features)
            global_scaler: Optional global scaler for normalization
            categorical_columns: Indices of categorical columns
            continuous_columns: Indices of continuous columns
            num_timesteps: Number of diffusion timesteps
            beta_min: Minimum beta for diffusion schedule
            beta_max: Maximum beta for diffusion schedule
            epsilon: Epsilon parameter for posteriors
            device: torch device (cuda or cpu)
            **kwargs: Additional arguments
        """
        # Convert to numpy arrays and store
        self.x0_train = np.asarray(x0_train, dtype=np.float32)
        self.x1_train = np.asarray(x1_train, dtype=np.float32)
        self.x0_test = np.asarray(x0_test, dtype=np.float32)
        self.x1_test = np.asarray(x1_test, dtype=np.float32)
        
        # Verify shapes match
        assert self.x0_train.shape[0] == self.x1_train.shape[0], \
            "x0_train and x1_train must have same number of samples"
        assert self.x0_test.shape[0] == self.x1_test.shape[0], \
            "x0_test and x1_test must have same number of samples"
        
        self.categorical_columns = categorical_columns or []
        self.continuous_columns = continuous_columns or []
        self.scaler = global_scaler
        
        # Determine device
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data dimension (inferred from input)
        self.D = self.x0_train.shape[1]
        assert self.x1_train.shape[1] == self.D, "x0 and x1 must have same feature dimension"
        
        # ASBM parameters
        self.num_timesteps = num_timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.epsilon = epsilon
        
        # Model components will be initialized during fit()
        self.netG_fw = None  # Forward generator: x1 -> x0
        self.netG_bw = None  # Backward generator: x0 -> x1
        self.netD_fw = None  # Forward discriminator
        self.netD_bw = None  # Backward discriminator
        self.pos_coeff = None
        self.config_obj = None  # Store config for later use
        self.is_trained = False
        
        # Metrics storage
        self.metrics = {}
        
    # def fit(
    #     self,
    #     imf_iters: int = 5,
    #     inner_iters: int = 1000,
    #     batch_size: int = 512,
    #     lr_g: float = 1e-4,
    #     lr_d: float = 1e-4,
    #     use_ema: bool = True,
    #     ema_decay: float = 0.999,
    #     verbose: bool = False,
    #     save_dir: Optional[str] = None,
    #     **kwargs
    # ) -> Dict[str, Any]:
    #     """
    #     Train ASBM on training data using IMF (Iterative Markovian Fitting).
        
    #     Args:
    #         imf_iters: Number of IMF outer iterations
    #         inner_iters: Number of inner training iterations per IMF iteration
    #         batch_size: Batch size for training
    #         lr_g: Learning rate for generators
    #         lr_d: Learning rate for discriminators
    #         use_ema: Whether to use exponential moving average
    #         ema_decay: EMA decay rate
    #         verbose: Print training progress
    #         save_dir: Directory to save training figures and models
    #         **kwargs: Additional arguments
            
    #     Returns:
    #         Training history/logs
    #     """
        
    #     if verbose:
    #         print(f"Starting ASBM training with {imf_iters} IMF iterations")
        
    #     # Convert training data to tensors
    #     x0_train = torch.from_numpy(self.x0_train).float().to(self.device)
    #     x1_train = torch.from_numpy(self.x1_train).float().to(self.device)
        
    #     # Create config-like object for compatibility
    #     config = {
    #         'num_timesteps': self.num_timesteps,
    #         't_dim': 2,
    #         'x_dim': self.D,
    #         'out_dim': self.D,
    #         'beta_min': self.beta_min,
    #         'beta_max': self.beta_max,
    #         'epsilon': self.epsilon,
    #         'batch_size': batch_size,
    #         'nz': 1,
    #         'lr_g': lr_g,
    #         'lr_d': lr_d,
    #         'beta_1': 0.5,
    #         'beta_2': 0.9,
    #         'r1_gamma': 0.01,
    #         'lazy_reg': 1,
    #         'use_ema': use_ema,
    #         'ema_decay': ema_decay,
    #         'print_every': 100,
    #     }
    #     config_obj = dotdict(config)
    #     self.config_obj = config_obj  # Store for use in generate()
        
    #     # Initialize time schedule FIRST (sets config.t)
    #     T = get_time_schedule(config_obj, self.device)
        
    #     # Initialize networks
    #     nz = config['nz']
    #     self.netG_fw = MyGenerator(
    #         x_dim=self.D, t_dim=2, n_t=self.num_timesteps,
    #         out_dim=self.D, z_dim=nz, layers=[256, 256, 256]
    #     ).to(self.device)
        
    #     self.netG_bw = MyGenerator(
    #         x_dim=self.D, t_dim=2, n_t=self.num_timesteps,
    #         out_dim=self.D, z_dim=nz, layers=[256, 256, 256]
    #     ).to(self.device)
        
    #     self.netD_fw = MyDiscriminator(
    #         x_dim=self.D, t_dim=2, n_t=self.num_timesteps,
    #         layers=[256, 256, 256]
    #     ).to(self.device)
        
    #     self.netD_bw = MyDiscriminator(
    #         x_dim=self.D, t_dim=2, n_t=self.num_timesteps,
    #         layers=[256, 256, 256]
    #     ).to(self.device)
        
    #     # Initialize optimizers
    #     opt_G_fw = optim.Adam(self.netG_fw.parameters(), lr=lr_g, betas=(0.5, 0.9))
    #     opt_D_fw = optim.Adam(self.netD_fw.parameters(), lr=lr_d, betas=(0.5, 0.9))
    #     opt_G_bw = optim.Adam(self.netG_bw.parameters(), lr=lr_g, betas=(0.5, 0.9))
    #     opt_D_bw = optim.Adam(self.netD_bw.parameters(), lr=lr_d, betas=(0.5, 0.9))
        
    #     # Initialize diffusion coefficients (AFTER time schedule creates config.t)
    #     self.pos_coeff = BrownianPosterior_Coefficients(config_obj, self.device)
        
    #     # Initialize EMA
    #     ema_g_fw = ExponentialMovingAverage(self.netG_fw.parameters(), decay=ema_decay)
    #     ema_g_bw = ExponentialMovingAverage(self.netG_bw.parameters(), decay=ema_decay)
        
    #     # Create save directory
    #     if save_dir:
    #         os.makedirs(save_dir, exist_ok=True)
        
    #     training_history = {'loss_fw': [], 'loss_bw': []}
        
    #     # IMF training loop
    #     for imf_iter in range(imf_iters):
    #         if verbose:
    #             print(f"\n=== IMF Iteration {imf_iter + 1}/{imf_iters} ===")
            
    #         # Forward pass: x1_train -> x0_train
    #         if verbose:
    #             print("Training forward network (x1 -> x0)...")
            
    #         loss_fw_list = self._train_one_direction(
    #             source_data=x1_train,
    #             target_data=x0_train,
    #             netG=self.netG_fw,
    #             netD=self.netD_fw,
    #             opt_G=opt_G_fw,
    #             opt_D=opt_D_fw,
    #             ema_g=ema_g_fw,
    #             config=config_obj,
    #             inner_iters=inner_iters,
    #             verbose=verbose
    #         )
    #         training_history['loss_fw'].extend(loss_fw_list)
            
    #         # Backward pass: x0_train -> x1_train
    #         if verbose:
    #             print("Training backward network (x0 -> x1)...")
            
    #         loss_bw_list = self._train_one_direction(
    #             source_data=x0_train,
    #             target_data=x1_train,
    #             netG=self.netG_bw,
    #             netD=self.netD_bw,
    #             opt_G=opt_G_bw,
    #             opt_D=opt_D_bw,
    #             ema_g=ema_g_bw,
    #             config=config_obj,
    #             inner_iters=inner_iters,
    #             verbose=verbose
    #         )
    #         training_history['loss_bw'].extend(loss_bw_list)
            
    #         if verbose:
    #             print(f"IMF {imf_iter}: FW Loss {loss_fw_list[-1]:.6f}, BW Loss {loss_bw_list[-1]:.6f}")
        
    #     if verbose:
    #         print("\nTraining completed!")
        
    #     self.is_trained = True
    #     return training_history

    def fit(
        self,
        imf_iters: int = 5,
        inner_iters: int = 1000,
        batch_size: int = 512,
        lr_g: float = 1e-4,
        lr_d: float = 1e-4,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        verbose: bool = False,
        smoothing_factor: float = 0.1,
        plot_losses: bool = False,
        save_dir: Optional[str] = None,
        layers_G_fw: list = [256, 256, 256],
        layers_G_bw: list = [256, 256, 256],
        layers_D_fw: list = [256, 256, 256],
        layers_D_bw: list = [256, 256, 256],
        **kwargs

    ) -> Dict[str, Any]:

        """
        Train ASBM on training data using IMF (Iterative Markovian Fitting).

        Args:
            imf_iters: Number of IMF outer iterations
            inner_iters: Number of inner training iterations per IMF iteration
            batch_size: Batch size for training
            lr_g: Learning rate for generators
            lr_d: Learning rate for discriminators
            use_ema: Whether to use exponential moving average
            ema_decay: EMA decay rate
            verbose: Print training progress
            plot_losses: Whether to plot and save GAN losses (default: False)
            save_dir: Directory to save training figures and models
            **kwargs: Additional arguments

        Returns:
            Training history/logs including loss histories
        """

        if verbose:
            print(f"Starting ASBM training with {imf_iters} IMF iterations")

        # Convert training data to tensors
        x0_train = torch.from_numpy(self.x0_train).float().to(self.device)
        x1_train = torch.from_numpy(self.x1_train).float().to(self.device)

        # Create config-like object for compatibility
        config = {
            'num_timesteps': self.num_timesteps,
            't_dim': 2,
            'x_dim': self.D,
            'out_dim': self.D,
            'beta_min': self.beta_min,
            'beta_max': self.beta_max,
            'epsilon': self.epsilon,
            'batch_size': batch_size,
            'nz': 1,
            'lr_g': lr_g,
            'lr_d': lr_d,
            'beta_1': 0.5,
            'beta_2': 0.9,
            'r1_gamma': 0.01,
            'lazy_reg': 1,
            'use_ema': use_ema,
            'ema_decay': ema_decay,
            'print_every': 100,
            'layers_G_fw': layers_G_fw,
            'layers_G_bw': layers_G_bw,
            'layers_D_fw': layers_D_fw,
            'layers_D_bw': layers_D_bw
        }

        config_obj = dotdict(config)
        self.config_obj = config_obj

        # Initialize time schedule FIRST (sets config.t)
        T = get_time_schedule(config_obj, self.device)

        # Initialize networks
        nz = config['nz']

        self.netG_fw = MyGenerator(
            x_dim=self.D, t_dim=2, n_t=self.num_timesteps,
            out_dim=self.D, z_dim=nz, layers=layers_G_fw
        ).to(self.device)

        self.netG_bw = MyGenerator(
            x_dim=self.D, t_dim=2, n_t=self.num_timesteps,
            out_dim=self.D, z_dim=nz, layers=layers_G_bw
        ).to(self.device)

        self.netD_fw = MyDiscriminator(
            x_dim=self.D, t_dim=2, n_t=self.num_timesteps,
            layers=layers_D_fw
        ).to(self.device)

        self.netD_bw = MyDiscriminator(
            x_dim=self.D, t_dim=2, n_t=self.num_timesteps,
            layers=layers_D_bw
        ).to(self.device)

        # Initialize optimizers
        opt_G_fw = optim.Adam(self.netG_fw.parameters(), lr=lr_g, betas=(0.5, 0.9))
        opt_D_fw = optim.Adam(self.netD_fw.parameters(), lr=lr_d, betas=(0.5, 0.9))
        opt_G_bw = optim.Adam(self.netG_bw.parameters(), lr=lr_g, betas=(0.5, 0.9))
        opt_D_bw = optim.Adam(self.netD_bw.parameters(), lr=lr_d, betas=(0.5, 0.9))

        # Initialize diffusion coefficients (AFTER time schedule creates config.t)
        self.pos_coeff = BrownianPosterior_Coefficients(config_obj, self.device)

        # Initialize EMA
        ema_g_fw = ExponentialMovingAverage(self.netG_fw.parameters(), decay=ema_decay)
        ema_g_bw = ExponentialMovingAverage(self.netG_bw.parameters(), decay=ema_decay)

        # Create save directory
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Initialize loss history tracking
        training_history = {
            'loss_fw': [],
            'loss_bw': [],
            'G_loss_fw': [],
            'D_loss_fw': [],
            'G_loss_bw': [],
            'D_loss_bw': [],
        }

        # IMF training loop
        for imf_iter in range(imf_iters):
            if verbose:
                print(f"\n{'='*70}")
                print(f"IMF Iteration {imf_iter + 1}/{imf_iters}")
                print(f"{'='*70}")

            # Forward pass: x1_train -> x0_train
            if verbose:
                print("Training forward network (x1 -> x0)...")

            fw_losses = self._train_one_direction(
                source_data=x1_train,
                target_data=x0_train,
                netG=self.netG_fw,
                netD=self.netD_fw,
                opt_G=opt_G_fw,
                opt_D=opt_D_fw,
                ema_g=ema_g_fw,
                config=config_obj,
                inner_iters=inner_iters,
                verbose=verbose,
            )

            # Extract losses from forward direction
            training_history['loss_fw'].extend(fw_losses['G_loss'])
            training_history['G_loss_fw'].extend(fw_losses['G_loss'])
            training_history['D_loss_fw'].extend(fw_losses['D_loss'])

            # Backward pass: x0_train -> x1_train
            if verbose:
                print("Training backward network (x0 -> x1)...")

            bw_losses = self._train_one_direction(
                source_data=x0_train,
                target_data=x1_train,
                netG=self.netG_bw,
                netD=self.netD_bw,
                opt_G=opt_G_bw,
                opt_D=opt_D_bw,
                ema_g=ema_g_bw,
                config=config_obj,
                inner_iters=inner_iters,
                verbose=verbose,
            )

            # Extract losses from backward direction
            training_history['loss_bw'].extend(bw_losses['G_loss'])
            training_history['G_loss_bw'].extend(bw_losses['G_loss'])
            training_history['D_loss_bw'].extend(bw_losses['D_loss'])

            if verbose:
                print(f"IMF {imf_iter + 1}: FW G Loss {fw_losses['G_loss'][-1]:.6f}, BW G Loss {bw_losses['G_loss'][-1]:.6f}")

        if verbose:
            print(f"\n{'='*70}")
            print("Training completed!")
            print(f"{'='*70}")

        self.is_trained = True

        # Plot losses if requested
        if plot_losses:
            self._plot_gan_losses(
                loss_history_fw={
                    'G_loss': training_history['G_loss_fw'],
                    'D_loss': training_history['D_loss_fw'],
                },
                loss_history_bw={
                    'G_loss': training_history['G_loss_bw'],
                    'D_loss': training_history['D_loss_bw'],
                },
                imf_iters=imf_iters,
                inner_iters=inner_iters,
                save_path=os.path.join(save_dir or '.', 'gan_losses.png'),
                smoothing_factor=smoothing_factor
            )

        return training_history
    
    # def _train_one_direction(
    #     self, source_data, target_data, netG, netD, opt_G, opt_D, ema_g,
    #     config, inner_iters, verbose
    # ):
        
    #     loss_list = []
        
    #     for iteration in tqdm.tqdm(range(inner_iters), disable=not verbose):
    #         # Sample batch
    #         idx = torch.randperm(source_data.shape[0])[:config.batch_size]
    #         x_src = source_data[idx]
    #         x_tgt = target_data[idx]
            
    #         # ===== Discriminator Update =====
    #         for p in netD.parameters():
    #             p.requires_grad = True
    #         netD.zero_grad()
            
    #         t = torch.randint(0, config.num_timesteps, (x_src.size(0),), device=self.device)
    #         x_t, x_tp1 = q_sample_supervised_pairs_brownian(
    #             self.pos_coeff, x_tgt, t, x_src
    #         )
    #         x_t.requires_grad = True
            
    #         # Real samples
    #         D_real = netD(x_t, t, x_tp1.detach()).view(-1)
    #         err_D_real = F.softplus(-D_real).mean()
    #         err_D_real.backward(retain_graph=True)
            
    #         # Gradient penalty
    #         if iteration % config.lazy_reg == 0:
    #             grad_real = torch.autograd.grad(
    #                 outputs=D_real.sum(), inputs=x_t, create_graph=True
    #             )[0]
    #             grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
    #             grad_penalty = (config.r1_gamma / 2) * grad_penalty
    #             grad_penalty.backward()
            
    #         # Fake samples
    #         latent_z = torch.randn(config.batch_size, config.nz, device=self.device)
    #         x_0_predict = netG(x_tp1.detach(), t, latent_z)
    #         x_pos_sample = sample_posterior(self.pos_coeff, x_0_predict, x_tp1, t)
    #         output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            
    #         err_D_fake = F.softplus(output).mean()
    #         err_D_fake.backward()
            
    #         opt_D.step()
            
    #         # ===== Generator Update =====
    #         for p in netD.parameters():
    #             p.requires_grad = False
    #         netG.zero_grad()
            
    #         t = torch.randint(0, config.num_timesteps, (x_src.size(0),), device=self.device)
    #         x_t, x_tp1 = q_sample_supervised_pairs_brownian(
    #             self.pos_coeff, x_tgt, t, x_src
    #         )
            
    #         # latent_z = torch.randn(x_t.shape[0], config.nz, device=self.device)
    #         latent_z = torch.randn(config.batch_size, config.nz, device=self.device)
    #         x_0_predict = netG(x_tp1.detach(), t, latent_z)
    #         x_pos_sample = sample_posterior(self.pos_coeff, x_0_predict, x_tp1, t)
    #         output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            
    #         err_G = F.softplus(-output).mean()
    #         err_G.backward()
            
    #         opt_G.step()
    #         ema_g.update()
            
    #         loss_list.append(err_G.item())
            
    #         if verbose and iteration % config.print_every == 0:
    #             print(f"Step {iteration:05d} | G Loss {err_G.item():.6f}")
        
    #     return loss_list
    def _train_one_direction(
        self, source_data, target_data, netG, netD, opt_G, opt_D, ema_g,
        config, inner_iters, verbose
    ):
        """
        Train one direction of ASBM with GAN loss tracking.
        
        Returns:
            Dictionary with 'G_loss' and 'D_loss' lists
        """
        
        G_loss_list = []
        D_loss_list = []

        for iteration in tqdm.tqdm(range(inner_iters), disable=not verbose):

            # Sample batch
            idx = torch.randperm(source_data.shape[0])[:config.batch_size]
            x_src = source_data[idx]
            x_tgt = target_data[idx]

            # ===== Discriminator Update =====
            for p in netD.parameters():
                p.requires_grad = True

            netD.zero_grad()

            t = torch.randint(0, config.num_timesteps, (x_src.size(0),), device=self.device)

            x_t, x_tp1 = q_sample_supervised_pairs_brownian(
                self.pos_coeff, x_tgt, t, x_src
            )

            x_t.requires_grad = True

            # Real samples
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)
            err_D_real = F.softplus(-D_real).mean()
            err_D_real.backward(retain_graph=True)

            # Gradient penalty
            if iteration % config.lazy_reg == 0:
                grad_real = torch.autograd.grad(
                    outputs=D_real.sum(), inputs=x_t, create_graph=True
                )[0]
                grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = (config.r1_gamma / 2) * grad_penalty
                grad_penalty.backward()

            # Fake samples
            latent_z = torch.randn(config.batch_size, config.nz, device=self.device)
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(self.pos_coeff, x_0_predict, x_tp1, t)

            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            err_D_fake = F.softplus(output).mean()
            err_D_fake.backward()

            opt_D.step()

            # TRACK DISCRIMINATOR LOSS
            D_loss = err_D_real.item() + err_D_fake.item()
            D_loss_list.append(D_loss)

            # ===== Generator Update =====
            for p in netD.parameters():
                p.requires_grad = False

            netG.zero_grad()

            t = torch.randint(0, config.num_timesteps, (x_src.size(0),), device=self.device)

            x_t, x_tp1 = q_sample_supervised_pairs_brownian(
                self.pos_coeff, x_tgt, t, x_src
            )

            latent_z = torch.randn(config.batch_size, config.nz, device=self.device)
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(self.pos_coeff, x_0_predict, x_tp1, t)

            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            err_G = F.softplus(-output).mean()

            err_G.backward()

            opt_G.step()

            ema_g.update()

            # TRACK GENERATOR LOSS
            G_loss_list.append(err_G.item())

            if verbose and iteration % config.print_every == 0:
                print(f"Step {iteration:05d} | G Loss {err_G.item():.6f} | D Loss {D_loss:.6f}")

        return {
            'G_loss': G_loss_list,
            'D_loss': D_loss_list,
        }
    
    def generate(self, n_samples: Optional[int] = None, direction: str = 'forward') -> np.ndarray:
        """
        Generate synthetic samples using trained ASBM.
        
        Args:
            n_samples: Number of samples to generate. If None, uses x0_test size.
            direction: 'forward' (x0_test -> synthetic x1) or 'backward' (x1_test -> synthetic x0)
            
        Returns:
            Generated synthetic data (n_samples, n_features)
        """
        if not self.is_trained or self.netG_fw is None or self.netG_bw is None:
            raise RuntimeError("Model must be trained before generating samples")
        
        assert direction in ['forward', 'backward'], "direction must be 'forward' or 'backward'"
        
        
        if n_samples is None:
            if direction == 'forward':
                n_samples = self.x0_test.shape[0]
            else:
                n_samples = self.x1_test.shape[0]
        
        # Get source samples
        # if direction == 'forward':
        #     source_data = torch.from_numpy(self.x0_test[:n_samples]).float().to(self.device)
        #     netG = self.netG_bw
        # else:
        #     source_data = torch.from_numpy(self.x1_test[:n_samples]).float().to(self.device)
        #     netG = self.netG_fw
        if direction == 'forward':
            source_data = torch.from_numpy(self.x0_train[:n_samples]).float().to(self.device)
            netG = self.netG_bw
        else:
            source_data = torch.from_numpy(self.x1_train[:n_samples]).float().to(self.device)
            netG = self.netG_fw
        
        # Use stored config object (dotdict with attribute access)
        with torch.no_grad():
            synthetic_data = sample_from_model(
                self.pos_coeff, netG, self.num_timesteps,
                source_data, self.config_obj
            )
        
        return synthetic_data.cpu().numpy()
    
    def evaluate(
        self,
        metrics: Optional[List[str]] = None,
        nproj: int = 256,
        sigma_mmd: float = 1.0,
        max_eval_samples: int = 5000,
    ) -> Dict[str, float]:
        """
        Evaluate synthetic data quality by comparing with test set.
        
        Args:
            metrics: List of metrics to compute (not yet implemented - placeholder)
            nproj: Number of projections for Sliced Wasserstein Distance
            sigma_mmd: Kernel bandwidth for MMD
            max_eval_samples: Maximum samples for evaluation
            
        Returns:
            Dictionary of metric values
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        if metrics is None:
            metrics = ['basic_stats']  # Placeholder for now
        
        results = {}
        
        # Generate synthetic x1 from x0_test
        synthetic_x1 = self.generate(n_samples=min(self.x0_test.shape[0], max_eval_samples), 
                                     direction='forward')
        real_x1 = self.x1_test[:synthetic_x1.shape[0]]
        
        # Compute basic statistics as placeholder
        if 'basic_stats' in metrics:
            results['x1_mean_diff'] = float(np.mean(np.abs(synthetic_x1.mean(axis=0) - real_x1.mean(axis=0))))
            results['x1_std_diff'] = float(np.mean(np.abs(synthetic_x1.std(axis=0) - real_x1.std(axis=0))))
        
        # Create scaler if not provided
        if self.scaler is None:
            scaler = StandardScaler()
            scaler.fit(self.x1_test)
        else:
            scaler = self.scaler
        
        # Scale data for evaluation
        synthetic_x1_scaled = scaler.transform(synthetic_x1)
        real_x1_scaled = scaler.transform(real_x1)
        
        results['synthetic_x1_shape'] = synthetic_x1.shape
        results['real_x1_shape'] = real_x1.shape
        
        self.metrics = results
        return results
    
    def save(self, path: str) -> None:
        """Save trained models."""
        if self.netG_fw is None or self.netG_bw is None:
            raise RuntimeError("No models to save")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'netG_fw': self.netG_fw.state_dict(),
            'netG_bw': self.netG_bw.state_dict(),
            'netD_fw': self.netD_fw.state_dict() if self.netD_fw else None,
            'netD_bw': self.netD_bw.state_dict() if self.netD_bw else None,
        }, path)
        
        print(f"Models saved to {path}")
    
    def load(self, path: str) -> None:
        """Load trained models."""
        if self.netG_fw is None or self.netG_bw is None:
            raise RuntimeError("Models must be initialized before loading")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.netG_fw.load_state_dict(checkpoint['netG_fw'])
        self.netG_bw.load_state_dict(checkpoint['netG_bw'])
        
        if checkpoint['netD_fw'] is not None:
            self.netD_fw.load_state_dict(checkpoint['netD_fw'])
        if checkpoint['netD_bw'] is not None:
            self.netD_bw.load_state_dict(checkpoint['netD_bw'])
        
        self.is_trained = True

    def _plot_gan_losses(
        self,
        loss_history_fw: Dict[str, List[float]],
        loss_history_bw: Dict[str, List[float]],
        imf_iters: int,
        inner_iters: int,
        smoothing_factor=0.1,
        save_path: str = "gan_losses.png",
    ) -> None:
        """
        Plot GAN losses over training iterations with IMF iteration markers (serifs).
        
        Args:
            loss_history_fw: Dictionary with keys 'G_loss' and 'D_loss' for forward direction
            loss_history_bw: Dictionary with keys 'G_loss' and 'D_loss' for backward direction
            imf_iters: Number of IMF iterations
            inner_iters: Number of inner iterations per IMF iteration
            save_path: Path to save PNG file
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # X-axis: inner iterations across all IMF iterations
        iterations = np.arange(len(loss_history_fw['G_loss']))
        
        # Calculate IMF iteration boundaries (serifs)
        imf_boundaries = [(i+1) * inner_iters for i in range(imf_iters)]
        
        # Serif styling
        serif_color = 'red'
        serif_style = '--'
        serif_alpha = 0.6
        serif_width = 2
        
        # ===== FORWARD DIRECTION - GENERATOR LOSS =====
        ax = axes[0, 0]
        # Apply smoothing
        G_loss_fw_smooth = smooth_losses(loss_history_fw['G_loss'], smoothing_factor=0.1)

        # Plot raw losses faintly
        ax.plot(iterations, loss_history_fw['G_loss'], 
                alpha=0.5, color='#1f77b4', linewidth=0.5, label='Raw Loss')

        # Plot smoothed losses prominently
        ax.plot(iterations, G_loss_fw_smooth, 
                label='Smoothed Loss (EMA)', linewidth=2.5, alpha=0.95, color='#1f77b4')

        ax.set_title('Forward Network - Generator Loss', fontsize=13, fontweight='bold')
        ax.set_xlabel('Inner Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        for i, boundary in enumerate(imf_boundaries):
            ax.axvline(x=boundary, color=serif_color, linestyle=serif_style, 
                    alpha=serif_alpha, linewidth=serif_width)
        
        # ===== FORWARD DIRECTION - DISCRIMINATOR LOSS =====
        ax = axes[0, 1]
        D_loss_fw_smooth = smooth_losses(loss_history_fw['D_loss'], smoothing_factor=0.1)

        # Plot raw losses faintly
        ax.plot(iterations, loss_history_fw['D_loss'], 
                alpha=0.5, color='#ff7f0e', linewidth=0.5, label='Raw Loss')

        # Plot smoothed losses prominently
        ax.plot(iterations, D_loss_fw_smooth, 
                label='Smoothed Loss (EMA)', linewidth=2.5, alpha=0.95, color='#ff7f0e')
        
        ax.set_title('Forward Network - Discriminator Loss', fontsize=13, fontweight='bold')
        ax.set_xlabel('Inner Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        for i, boundary in enumerate(imf_boundaries):
            ax.axvline(x=boundary, color=serif_color, linestyle=serif_style, 
                    alpha=serif_alpha, linewidth=serif_width)
        
        # ===== BACKWARD DIRECTION - GENERATOR LOSS =====
        ax = axes[1, 0]
        G_loss_bw_smooth = smooth_losses(loss_history_bw['G_loss'], smoothing_factor=0.1)

        # Plot raw losses faintly
        ax.plot(iterations, loss_history_bw['G_loss'], 
                alpha=0.5, color='#2ca02c', linewidth=0.5, label='Raw Loss')

        # Plot smoothed losses prominently
        ax.plot(iterations, G_loss_bw_smooth, 
                label='Smoothed Loss (EMA)', linewidth=2.5, alpha=0.95, color='#2ca02c')
        
        ax.set_title('Backward Network - Generator Loss', fontsize=13, fontweight='bold')
        ax.set_xlabel('Inner Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        for i, boundary in enumerate(imf_boundaries):
            ax.axvline(x=boundary, color=serif_color, linestyle=serif_style, 
                    alpha=serif_alpha, linewidth=serif_width)
        
        # ===== BACKWARD DIRECTION - DISCRIMINATOR LOSS =====
        ax = axes[1, 1]
        D_loss_bw_smooth = smooth_losses(loss_history_bw['D_loss'], smoothing_factor=0.1)

        # Plot raw losses faintly
        ax.plot(iterations, loss_history_bw['D_loss'], 
                alpha=0.5, color='#d62728', linewidth=0.5, label='Raw Loss')

        # Plot smoothed losses prominently
        ax.plot(iterations, D_loss_bw_smooth, 
                label='Smoothed Loss (EMA)', linewidth=2.5, alpha=0.95, color='#d62728')
        
        ax.set_title('Backward Network - Discriminator Loss', fontsize=13, fontweight='bold')
        ax.set_xlabel('Inner Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        for i, boundary in enumerate(imf_boundaries):
            ax.axvline(x=boundary, color=serif_color, linestyle=serif_style, 
                    alpha=serif_alpha, linewidth=serif_width)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

# def generate_swiss_roll(n_samples=1000, noise=0.2, seed=42):
#     """Generate 2D Swiss Roll distribution"""
#     np.random.seed(seed)
#     t = np.linspace(0, 4 * np.pi, n_samples)
#     x = t * np.cos(t) + np.random.randn(n_samples) * noise
#     y = t * np.sin(t) + np.random.randn(n_samples) * noise
    
#     data = np.column_stack([x, y])

#     return data

# # ============ EXAMPLE USAGE ============
# if __name__ == '__main__':
    
#     def generate_swiss_roll(n_samples=1600, noise=0.01, seed=42):
#         np.random.seed(seed)
#         t = np.linspace(0, 4 * np.pi, n_samples)
#         x = t * np.cos(t) + np.random.randn(n_samples) * 0.2
#         y = t * np.sin(t) + np.random.randn(n_samples) * 0.2
        
#         data = np.column_stack([x, y])
#         return (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

#     # Generate synthetic data
#     x0_train = np.random.randn(1000, 2)
#     x1_train = generate_swiss_roll(1000, noise = 0.01)
#     x0_test = np.random.randn(200, 2)
#     x1_test = generate_swiss_roll(200, noise = 0.01)
    
#     # Initialize bridge
#     bridge = ASBMTabularBridge(
#         x0_train, x1_train, x0_test, x1_test,
#         num_timesteps=4,
#     )
    
#     # Train
#     history = bridge.fit(imf_iters=5, inner_iters=5000, verbose=False, smoothing_factor=0.05, plot_losses=True, save_dir='./GAN_losses')
    
#     # Generate samples
#     synthetic_x1 = bridge.generate(n_samples=200, direction='forward')
#     import matplotlib.pyplot as plt
#     plt.scatter(synthetic_x1[:,0], synthetic_x1[:,1])
#     plt.scatter(x1_test[:,0], x1_test[:,1])
#     plt.show()
    
#     # Evaluate
#     metrics = bridge.evaluate()
#     print("Metrics:", metrics)
    
#     # Save
#     bridge.save('asbm_bridge_model.pt')
