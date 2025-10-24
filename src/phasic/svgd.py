import os
import platform
from time import time, sleep
import numpy as np
import pickle
import hashlib
import pathlib

# Note: JAX environment (XLA_FLAGS, device count) is configured by
# phasic.__init__.py before this module is imported.
# Users should configure via:
#   1. phasic.configure() before import, OR
#   2. PTDALG_CPUS environment variable
# See: src/phasic/__init__.py lines 101-133
import jax
# print(jax.devices())
import jax.numpy as jnp
from jax import grad, vmap, jit, pmap
from jax.scipy.stats import norm
import jax.nn as jnn
import jax.sharding as jsh
from jax.experimental import checkify
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.pjit import pjit

from functools import partial

# Import configuration system
from .config import get_config
from .exceptions import PTDConfigError

from .plot import black_white, Theme

## requires equinox dependency
# from .decoders import VariableDimPTDDecoder, LessThanOneDecoder, 
#     SumToOneDecoder, IndependentProbDecoder

# def string_to_class(s, suffix=''):
#     class_name = ''.join(x.capitalize() for x in s.split('_')) + suffix
#     if class_name not in globals():
#         raise ValueError(f"Cannot translate string to class name: {s}")
#     return globals()[class_name]


from tqdm import trange, tqdm
trange = partial(trange, bar_format="{bar}", leave=False)
tqdm = partial(tqdm, bar_format="{bar}", leave=False)


#from jax import random, vmap, grad, jit


# ============================================================================
# Schedule Classes for Step Size and Bandwidth Control
# ============================================================================

FIGSIZE = (4.5, 3.2)
class StepSizeSchedule:
    """
    Base class for step size schedules.

    Subclasses should implement __call__(iteration, particles) returning a scalar step size.
    """
    def __call__(self, iteration, particles=None):
        """
        Compute step size for given iteration.

        Parameters
        ----------
        iteration : int
            Current iteration number (0-indexed)
        particles : jnp.ndarray, optional
            Current particle positions, shape (n_particles, theta_dim)

        Returns
        -------
        float
            Step size for this iteration
        """
        raise NotImplementedError

    def plot(self, nr_iter, figsize=FIGSIZE, title=None, ax=None):
        """
        Plot the step size schedule over iterations.

        Parameters
        ----------
        nr_iter : int
            Number of iterations to plot
        figsize : tuple, default=(4, 3)
            Figure size (width, height) in inches
        title : str, optional
            Plot title. If None, uses class name
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        ax : matplotlib.axes.Axes
            Axes object

        Examples
        --------
        >>> schedule = ExponentialDecayStepSize(first_step=0.1, last_step=0.01, tau=500.0)
        >>> fig, ax = schedule.plot(nr_iter=2000)
        >>> plt.show()
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=FIGSIZE)
        else:
            fig = ax.get_figure()

        # Compute schedule values
        iterations = np.arange(nr_iter)
        values = np.array([self(i) for i in iterations])

        # Plot
        with Theme():
            ax.plot(iterations, values, 'C1')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Step Size')
            ax.set_title(title or f'{self.__class__.__name__}')
            # ax.grid(True, alpha=0.3)

            # Add horizontal lines for first and last values if they exist
            if hasattr(self, 'first_step') and hasattr(self, 'last_step'):
                ax.axhline(self.first_step, color=black_white(ax), linestyle='--', alpha=0.5,
                        label=f'first_step={self.first_step:.4f}')
                ax.axhline(self.last_step, color=black_white(ax), linestyle='--', alpha=0.5,
                        label=f'last_step={self.last_step:.4f}')

        return fig, ax


class ConstantStepSize(StepSizeSchedule):
    """
    Constant step size (default behavior).

    Parameters
    ----------
    step_size : float
        Fixed step size for all iterations

    Examples
    --------
    >>> schedule = ConstantStepSize(0.01)
    >>> schedule(0)  # iteration 0
    0.01
    >>> schedule(100)  # iteration 100
    0.01
    """
    def __init__(self, step_size=0.01):
        self.step_size = step_size

    def __call__(self, iteration, particles=None):
        return self.step_size


class ExponentialDecayStepSize(StepSizeSchedule):
    """
    Exponential decay schedule: step_size = first_step * exp(-iteration/tau) + last_step * (1 - exp(-iteration/tau)).

    This schedule helps prevent divergence with large datasets by gradually reducing
    the step size as optimization progresses.

    Parameters
    ----------
    first_step : float, default=0.01
        Initial (first) step size at iteration 0
    last_step : float, default=1e-6
        Final (last) step size as iteration → ∞
    tau : float, default=1000.0
        Decay time constant (larger = slower decay)

    Examples
    --------
    >>> schedule = ExponentialDecayStepSize(first_step=0.1, last_step=0.01, tau=500.0)
    >>> schedule(0)      # iteration 0
    0.1
    >>> schedule(500)    # iteration 500 (≈63% decay)
    0.037
    >>> schedule(5000)   # iteration 5000 (full decay)
    0.01
    """
    def __init__(self, first_step=0.01, last_step=1e-6, tau=1000.0):
        self.first_step = first_step
        self.last_step = last_step
        self.tau = tau

    def __call__(self, iteration, particles=None):
        decay = jnp.exp(-iteration / self.tau)
        return self.first_step * decay + self.last_step * (1 - decay)


class AdaptiveStepSize(StepSizeSchedule):
    """
    Adaptive step size based on particle spread (KL divergence proxy).

    Increases step size when particles are too concentrated (low KL),
    decreases when particles are too dispersed (high KL).

    Parameters
    ----------
    base_step : float, default=0.01
        Base step size
    kl_target : float, default=0.1
        Target KL divergence (in log-space particle spread)
    adjust_rate : float, default=0.1
        Rate of adjustment (0 = no adjustment, 1 = immediate)

    Examples
    --------
    >>> schedule = AdaptiveStepSize(base_step=0.01, kl_target=0.1)
    >>> particles = jnp.array([[1.0], [1.1], [0.9]])  # concentrated
    >>> schedule(10, particles)  # will increase step size
    0.011
    """
    def __init__(self, base_step=0.01, kl_target=0.1, adjust_rate=0.1):
        self.base_step = base_step
        self.kl_target = kl_target
        self.adjust_rate = adjust_rate
        self.current_step = base_step

    def __call__(self, iteration, particles=None):
        if particles is None:
            return self.current_step

        # Estimate KL divergence using particle spread
        particle_std = jnp.std(particles, axis=0)
        kl_estimate = jnp.mean(jnp.log(particle_std + 1e-8))

        # Adaptive adjustment
        if kl_estimate > self.kl_target:
            # Particles too spread out, reduce step size
            adjustment = 1.0 - self.adjust_rate
        else:
            # Particles too concentrated, increase step size
            adjustment = 1.0 + self.adjust_rate

        self.current_step = self.current_step * adjustment
        return self.current_step


# ============================================================================
# Regularization Schedule Classes
# ============================================================================

class RegularizationSchedule:
    """
    Base class for regularization schedules.

    Subclasses should implement __call__(iteration, particles) returning a scalar regularization value.
    """
    def __call__(self, iteration, particles=None):
        """
        Compute regularization strength for given iteration.

        Parameters
        ----------
        iteration : int
            Current iteration number (0-indexed)
        particles : jnp.ndarray, optional
            Current particle positions, shape (n_particles, theta_dim)

        Returns
        -------
        float
            Regularization strength for this iteration
        """
        raise NotImplementedError

    def plot(self, nr_iter, figsize=FIGSIZE, title=None, ax=None):
        """
        Plot the regularization schedule over iterations.

        Parameters
        ----------
        nr_iter : int
            Number of iterations to plot
        figsize : tuple, default=(4, 3)
            Figure size (width, height) in inches
        title : str, optional
            Plot title. If None, uses class name
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        ax : matplotlib.axes.Axes
            Axes object

        Examples
        --------
        >>> schedule = ExponentialDecayRegularization(first_reg=5.0, last_reg=0.1, tau=500.0)
        >>> fig, ax = schedule.plot(nr_iter=2000)
        >>> plt.show()
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=FIGSIZE)
        else:
            fig = ax.get_figure()

        # Compute schedule values
        iterations = np.arange(nr_iter)
        values = np.array([self(i) for i in iterations])

        # Plot
        with Theme():
            ax.plot(iterations, values, 'C2')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Regularization Strength')
            ax.set_title(title or f'{self.__class__.__name__}')
            # ax.grid(True, alpha=0.3)

            # Add horizontal lines for first and last values if they exist
            if hasattr(self, 'first_reg') and hasattr(self, 'last_reg'):
                ax.axhline(self.first_reg, color=black_white(ax), linestyle='--', alpha=0.5,
                        label=f'first_reg={self.first_reg:.4f}')
                ax.axhline(self.last_reg, color=black_white(ax), linestyle='--', alpha=0.5,
                        label=f'last_reg={self.last_reg:.4f}')

        return fig, ax


class ConstantRegularization(RegularizationSchedule):
    """
    Constant regularization (default behavior).

    Parameters
    ----------
    regularization : float, default=0.0
        Fixed regularization strength for all iterations

    Examples
    --------
    >>> schedule = ConstantRegularization(1.0)
    >>> schedule(0)  # iteration 0
    1.0
    >>> schedule(100)  # iteration 100
    1.0
    """
    def __init__(self, regularization=0.0):
        self.regularization = regularization

    def __call__(self, iteration, particles=None):
        return self.regularization


class ExponentialDecayRegularization(RegularizationSchedule):
    """
    Exponential decay schedule: reg = first_reg * exp(-iteration/tau) + last_reg * (1 - exp(-iteration/tau)).

    This schedule helps by starting with strong moment regularization to guide
    initial exploration, then gradually reducing regularization as optimization
    converges to allow fine-tuning.

    Parameters
    ----------
    first_reg : float, default=1.0
        Initial (first) regularization strength at iteration 0
    last_reg : float, default=0.0
        Final (last) regularization strength as iteration → ∞
    tau : float, default=1000.0
        Decay time constant (larger = slower decay)

    Examples
    --------
    >>> schedule = ExponentialDecayRegularization(first_reg=5.0, last_reg=0.1, tau=500.0)
    >>> schedule(0)      # iteration 0
    5.0
    >>> schedule(500)    # iteration 500 (≈63% decay)
    0.1925
    >>> schedule(5000)   # iteration 5000 (full decay)
    0.1
    """
    def __init__(self, first_reg=1.0, last_reg=0.0, tau=1000.0):
        self.first_reg = first_reg
        self.last_reg = last_reg
        self.tau = tau

    def __call__(self, iteration, particles=None):
        decay = jnp.exp(-iteration / self.tau)
        return self.first_reg * decay + self.last_reg * (1 - decay)


class ExponentialCDFRegularization(RegularizationSchedule):
    """
    Exponential CDF schedule: reg = first_reg + (last_reg - first_reg) * (1 - exp(-iteration/tau)).

    This schedule uses the exponential cumulative distribution function (CDF) to create
    a smooth S-curve transition between first_reg and last_reg. Unlike exponential decay,
    this is bidirectional and works naturally for both increasing and decreasing schedules.

    The CDF approach provides:
    - Smooth, continuous transitions
    - Fast initial change that gradually slows
    - Natural interpretation: tau is the "characteristic time" (63% transition at tau)
    - Works equally well for increasing or decreasing regularization

    Parameters
    ----------
    first_reg : float, default=0.0
        Initial (first) regularization strength at iteration 0
    last_reg : float, default=1.0
        Final (last) regularization strength as iteration → ∞
    tau : float, default=1000.0
        Transition time constant (larger = slower transition)

    Examples
    --------
    >>> # Increasing regularization (useful for progressive regularization)
    >>> schedule = ExponentialCDFRegularization(first_reg=0.0, last_reg=1.0, tau=500.0)
    >>> schedule(0)      # iteration 0
    0.0
    >>> schedule(500)    # iteration 500 (≈63% transition)
    0.632
    >>> schedule(5000)   # iteration 5000 (nearly complete)
    0.993

    >>> # Decreasing regularization (similar to exponential decay)
    >>> schedule = ExponentialCDFRegularization(first_reg=5.0, last_reg=0.1, tau=500.0)
    >>> schedule(0)      # iteration 0
    5.0
    >>> schedule(500)    # iteration 500 (≈63% transition)
    1.9
    >>> schedule(5000)   # iteration 5000 (nearly complete)
    0.1
    """
    def __init__(self, first_reg=0.0, last_reg=1.0, tau=1000.0):
        self.first_reg = first_reg
        self.last_reg = last_reg
        self.tau = tau

    def __call__(self, iteration, particles=None):
        cdf = 1.0 - jnp.exp(-iteration / self.tau)
        return self.first_reg + (self.last_reg - self.first_reg) * cdf


# class BandwidthSchedule:
#     """
#     Base class for bandwidth schedules.

#     Subclasses should implement __call__(particles) returning bandwidth(s).
#     """
#     def __call__(self, particles):
#         """
#         Compute bandwidth for current particle configuration.

#         Parameters
#         ----------
#         particles : jnp.ndarray
#             Current particle positions, shape (n_particles, theta_dim)

#         Returns
#         -------
#         float or jnp.ndarray
#             Bandwidth (scalar for global, array for local)
#         """
#         raise NotImplementedError


# class MedianBandwidth(BandwidthSchedule):
#     """
#     Median heuristic bandwidth (default behavior).

#     Sets bandwidth to median of pairwise distances between particles.

#     Examples
#     --------
#     >>> schedule = MedianBandwidth()
#     >>> particles = jnp.array([[0.0], [1.0], [2.0]])
#     >>> schedule(particles)
#     1.0
#     """
#     def __call__(self, particles):
#         n_particles = particles.shape[0]
#         pairwise_dists = jnp.array([
#             jnp.linalg.norm(particles[i] - particles[j])
#             for i in range(n_particles)
#             for j in range(i + 1, n_particles)
#         ])
#         return jnp.median(pairwise_dists)


# class FixedBandwidth(BandwidthSchedule):
#     """
#     Fixed bandwidth for all iterations.

#     Parameters
#     ----------
#     bandwidth : float
#         Fixed bandwidth value

#     Examples
#     --------
#     >>> schedule = FixedBandwidth(1.0)
#     >>> particles = jnp.array([[0.0], [1.0]])
#     >>> schedule(particles)
#     1.0
#     """
#     def __init__(self, bandwidth=1.0):
#         self.bandwidth = bandwidth

#     def __call__(self, particles):
#         return self.bandwidth


# class LocalAdaptiveBandwidth(BandwidthSchedule):
#     """
#     Local adaptive bandwidth using k-nearest neighbors.

#     Computes per-particle bandwidth based on distance to k-nearest neighbors.

#     Parameters
#     ----------
#     alpha : float, default=0.9
#         Scaling factor for local bandwidth
#     k_frac : float, default=0.1
#         Fraction of particles to use as k-nearest neighbors

#     Examples
#     --------
#     >>> schedule = LocalAdaptiveBandwidth(alpha=0.9, k_frac=0.1)
#     >>> particles = jnp.array([[0.0], [1.0], [10.0]])
#     >>> bandwidths = schedule(particles)
#     >>> bandwidths.shape
#     (3,)
#     """
#     def __init__(self, alpha=0.9, k_frac=0.1):
#         self.alpha = alpha
#         self.k_frac = k_frac

#     def __call__(self, particles):
#         n_particles = particles.shape[0]
#         k_nn = max(1, int(n_particles * self.k_frac))

#         bandwidths = []
#         for i in range(n_particles):
#             # Compute distances to all other particles
#             distances = jnp.array([
#                 jnp.linalg.norm(particles[i] - particles[j])
#                 for j in range(n_particles) if j != i
#             ])
#             # Take k-nearest neighbors
#             knn_distances = jnp.sort(distances)[:k_nn]
#             local_bw = jnp.mean(knn_distances) * self.alpha
#             bandwidths.append(local_bw)

#         return jnp.array(bandwidths)


# ============================================================================
# End of Schedule Classes
# ============================================================================


# def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
#     new_cmap = colors.LinearSegmentedColormap.from_list(
#         'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
#         cmap(np.linspace(minval, maxval, n)))
#     return new_cmap
# # "_iridis" color map (viridis without the deep purple):
# _iridis = truncate_colormap(plt.get_cmap('viridis'), 0.2, 1)


# @jit
# def calculate_param_dim(k, m):
#     """Calculate parameter dimension for discrete phase-type distribution
    
#     Parameters:
#     - k: number of dimensions (absorption states)  
#     - m: number of transient states
    
#     Returns:
#     - Total parameter dimension
#     """
#     # Initial distribution: m parameters (no constraint)
#     alpha_dim = m
    
#     # Sub-intensity matrix: m×m parameters with row-sum constraints
#     # Each row sums to <= 0, so m-1 free parameters per row
#     sub_Q_dim = m * (m - 1) 
    
#     # Exit rates: k×m parameters (all free)
#     exit_rates_dim = k * m
    
#     return alpha_dim + sub_Q_dim + exit_rates_dim

# def example_ptd_spec(key, k=1, m=2):
#     """Generate example discrete phase-type distribution parameters
    
#     Returns flattened parameter vector for the distribution with:
#     - k absorption states (dimensions)
#     - m transient states
#     """
#     # Generate initial distribution (normalized)
#     key, subkey = jax.random.split(key)
#     alpha_raw = jax.random.exponential(subkey, shape=(m,))
#     alpha = alpha_raw / jnp.sum(alpha_raw)
    
#     # Generate sub-intensity matrix Q (m×m)
#     key, subkey = jax.random.split(key)
#     # Off-diagonal elements (positive, will be made negative)
#     off_diag = jax.random.exponential(subkey, shape=(m, m))
#     off_diag = off_diag.at[jnp.diag_indices(m)].set(0)  # Zero diagonal
    
#     # Make off-diagonal negative and set diagonal to ensure row sums < 0
#     Q = -off_diag
#     row_sums = jnp.sum(Q, axis=1)
#     Q = Q.at[jnp.diag_indices(m)].set(-jnp.abs(row_sums) - 0.1)  # Ensure diagonal < row sum
    
#     # Generate exit rates (k×m, all positive)
#     key, subkey = jax.random.split(key)
#     exit_rates = jax.random.exponential(subkey, shape=(k, m))
    
#     # Flatten into parameter vector
#     # Structure: [alpha (m), Q off-diagonal (m*(m-1)), exit_rates (k*m)]
#     q_off_diag = jnp.concatenate([Q[i, :i].flatten() for i in range(m)] + 
#                                  [Q[i, i+1:].flatten() for i in range(m)])
    
#     params = jnp.concatenate([alpha, q_off_diag, exit_rates.flatten()])
#     return params

def unpack_theta(params, k, m):
    """Unpack flattened parameter vector into components using JAX operations"""
    # Calculate dimensions
    alpha_dim = m
    sub_Q_dim = m * (m - 1)
    
    # Extract components using standard slicing (will be handled by JAX)
    alpha = params[:alpha_dim]
    q_off_diag = params[alpha_dim:alpha_dim + sub_Q_dim]
    exit_rates_flat = params[alpha_dim + sub_Q_dim:alpha_dim + sub_Q_dim + k * m]
    
    # Reconstruct Q matrix - simplified approach for any m
    Q = jnp.zeros((m, m))
    
    # For general case, use a more systematic approach
    # Fill off-diagonal elements in order
    idx = 0
    for i in range(m):
        for j in range(m):
            if i != j:  # Skip diagonal
                Q = Q.at[i, j].set(q_off_diag[idx])
                idx += 1
    
    # Set diagonal elements to ensure valid sub-intensity matrix
    row_sums = jnp.sum(Q, axis=1)
    Q = Q.at[jnp.diag_indices(m)].set(-jnp.abs(row_sums) - 0.1)
    
    # Reshape exit rates
    exit_rates = exit_rates_flat.reshape(k, m)
    
    return alpha, Q, exit_rates

# def simulate_example_data(key, params, k, m, n_samples):
#     """Simulate data from discrete phase-type distribution"""
#     alpha, Q, exit_rates = unpack_theta(params, k, m)
    
#     # Simple simulation - generate random absorption times
#     # This is a placeholder - real DPH simulation would be more complex
#     key, subkey = jax.random.split(key)
    
#     # Generate samples using approximation
#     # Sample from geometric distributions and combine
#     samples = []
#     for _ in range(n_samples):
#         key, subkey = jax.random.split(key)
#         # Simple approximation: sample absorption times
#         absorption_times = jax.random.geometric(subkey, 0.3, shape=(k,))
#         samples.append(absorption_times)
    
#     return jnp.array(samples)

def log_pmf_dph(x, params, k, m):
    """Log probability mass function for discrete phase-type distribution"""
    alpha, Q, exit_rates = unpack_theta(params, k, m)
    
    # Simple approximation for discrete phase-type log-pmf
    # Real implementation would involve matrix exponentials
    
    # Ensure x is properly shaped
    x = jnp.atleast_1d(x)
    if x.shape[0] != k:
        # Pad or truncate to match k dimensions
        if x.shape[0] < k:
            x = jnp.concatenate([x, jnp.ones(k - x.shape[0])])
        else:
            x = x[:k]
    
    # Approximate log-pmf using geometric distribution mixture
    log_prob = 0.0
    for i in range(k):
        for j in range(m):
            rate = jnp.abs(exit_rates[i, j])
            # Geometric log-pmf approximation
            p = rate / (1.0 + rate)
            log_prob += jnp.log(p) + (x[i] - 1) * jnp.log(1 - p)
    
    # Add initial distribution contribution
    log_prob += jnp.sum(jnp.log(alpha + 1e-8))
    
    return log_prob

# Simpler approach: direct parameter mapping
@jit
def z_to_theta(z):
    """Convert latent variable to parameter space"""
    return z  # Direct mapping for simplicity

# SVGD functions
@jit
def rbf_kernel(x, y, bandwidth):
    """RBF kernel function"""
    diff = x - y
    return jnp.exp(-jnp.sum(diff**2) / (2 * bandwidth**2))

# @jit
# def median_heuristic(particles):
#     """Median heuristic for bandwidth selection"""
#     n_particles = particles.shape[0]
#     distances = []
#     for i in range(n_particles):
#         for j in range(i+1, n_particles):
#             dist = jnp.linalg.norm(particles[i] - particles[j])
#             distances.append(dist)
#     distances = jnp.array(distances)
#     median_dist = jnp.median(distances)
#     return median_dist / jnp.log(n_particles + 1)

@jit 
def batch_median_heuristic(particles):
    """Vectorized median heuristic"""
    n_particles = particles.shape[0]
    # Compute pairwise distances
    diff = particles[:, None, :] - particles[None, :, :]
    distances = jnp.linalg.norm(diff, axis=2)
    # Get upper triangular part (excluding diagonal)
    triu_indices = jnp.triu_indices(n_particles, k=1)
    pairwise_dists = distances[triu_indices]
    median_dist = jnp.median(pairwise_dists)
    return median_dist / jnp.log(n_particles + 1)

@jit
def rbf_kernel_median(particles):
    """RBF kernel with median heuristic bandwidth"""
    bandwidth = batch_median_heuristic(particles)
    n_particles = particles.shape[0]
    
    # Compute kernel matrix
    K = jnp.zeros((n_particles, n_particles))
    for i in range(n_particles):
        for j in range(n_particles):
            K = K.at[i, j].set(rbf_kernel(particles[i], particles[j], bandwidth))
    
    # Compute gradients
    grad_K = jnp.zeros((n_particles, n_particles, particles.shape[1]))
    for i in range(n_particles):
        for j in range(n_particles):
            diff = particles[i] - particles[j]
            grad_K = grad_K.at[i, j].set(-K[i, j] * diff / bandwidth**2)
    
    return K, grad_K

# Define log probability functions
@jit
def logp(theta, data, k, m):
    """Log probability of data given parameters"""
    return jnp.sum(vmap(lambda x: log_pmf_dph(x, theta, k, m))(data))

# @jit  
# def logp_z(z, k, m):
#     """Log probability function for latent variables"""
#     theta = z_to_theta(z)
#     # Add prior (standard normal on z)
#     log_prior = -0.5 * jnp.sum(z**2)
#     return log_prior

# # Adaptive step size functions
# @jit
# def decayed_kl_target(iteration, base=0.1, decay=0.01):
#     """Exponentially decaying KL target"""
#     return base * jnp.exp(-decay * iteration)

# @jit  
# def step_size_schedule(iteration, max_step=0.001, min_step=1e-6):
#     """Step size schedule"""
#     decay = jnp.exp(-iteration / 1000.0)
#     return max_step * decay + min_step * (1 - decay)

@jit
def local_adaptive_bandwidth(particles, alpha=0.9):
    """Local adaptive bandwidth selection"""
    n_particles = particles.shape[0]
    # Use k-nearest neighbors approach
    k_nn = max(1, n_particles // 10)
    
    bandwidths = []
    for i in range(n_particles):
        # Compute distances to all other particles
        distances = jnp.array([jnp.linalg.norm(particles[i] - particles[j]) 
                              for j in range(n_particles) if j != i])
        # Take k-nearest neighbors
        knn_distances = jnp.sort(distances)[:k_nn]
        local_bw = jnp.mean(knn_distances) * alpha
        bandwidths.append(local_bw)
    
    return jnp.array(bandwidths)

@jit
def kl_adaptive_step(particles, kl_target=0.1):
    """Adaptive step size based on KL divergence estimate"""
    # Estimate KL divergence using particle approximation
    n_particles = particles.shape[0]
    
    # Simple KL estimate based on particle spread
    particle_std = jnp.std(particles, axis=0)
    kl_estimate = jnp.mean(jnp.log(particle_std + 1e-8))
    
    # Adaptive step using JAX conditional
    step_factor = jnp.where(kl_estimate > kl_target, 0.9, 1.1)
    
    return step_factor

# # SVGD update functions
# def svgd_update_z(particles_z, data, k, m, step_size=0.001, kl_target=0.1):
#     """SVGD update for latent variables"""
#     n_particles = particles_z.shape[0]
    
#     # Convert to parameter space for likelihood evaluation
#     particles_theta = jnp.array([z_to_theta(z) for z in particles_z])
    
#     # Compute log probability gradients
#     def logp_single(theta):
#         return logp(theta, data, k, m)
    
#     grad_logp = vmap(grad(logp_single))(particles_theta)
    
#     # Compute kernels
#     K, grad_K = rbf_kernel_median(particles_z)
    
#     # SVGD update
#     phi = jnp.zeros_like(particles_z)
#     for i in range(n_particles):
#         # Positive term: weighted gradient
#         positive_term = jnp.sum(K[i, :, None] * grad_logp, axis=0) / n_particles
        
#         # Negative term: kernel gradient
#         negative_term = jnp.sum(grad_K[i, :, :], axis=0) / n_particles
        
#         phi = phi.at[i].set(positive_term + negative_term)
    
#     # Adaptive step size
#     step_factor = kl_adaptive_step(particles_z, kl_target)
#     adaptive_step = step_size * step_factor
    
#     return particles_z + adaptive_step * phi

# # More sophisticated SVGD updates
# @jit
# def update_median_bw_kl_step(particles_z, k, m, kl_target=0.1, max_step=0.001):
#     """SVGD update with median bandwidth and KL-adaptive step"""
#     n_particles = particles_z.shape[0]
    
#     # Gradients in latent space (prior only for now)
#     grad_logp_z = -particles_z  # Gradient of standard normal prior
    
#     # Compute kernel and its gradients
#     K, grad_K = rbf_kernel_median(particles_z)
    
#     # SVGD update
#     phi = jnp.zeros_like(particles_z)
#     for i in range(n_particles):
#         positive_term = jnp.sum(K[i, :, None] * grad_logp_z, axis=0) / n_particles
#         negative_term = jnp.sum(grad_K[i, :, :], axis=0) / n_particles
#         phi = phi.at[i].set(positive_term + negative_term)
    
#     # Adaptive step
#     step_factor = kl_adaptive_step(particles_z, kl_target)
#     step_size = jnp.clip(max_step * step_factor, 1e-7, max_step)
    
#     return particles_z + step_size * phi

# @jit
# def update_local_bw_kl_step(particles_z, k, m, kl_target=0.1, max_step=0.001):
#     """SVGD update with local bandwidth and KL-adaptive step"""
#     n_particles = particles_z.shape[0]
    
#     # Get local bandwidths
#     local_bws = local_adaptive_bandwidth(particles_z)
    
#     # Gradients  
#     grad_logp_z = -particles_z
    
#     # Compute updates with local bandwidths
#     phi = jnp.zeros_like(particles_z)
#     for i in range(n_particles):
#         # Local kernel computations
#         local_K = jnp.array([rbf_kernel(particles_z[i], particles_z[j], local_bws[i]) 
#                             for j in range(n_particles)])
        
#         # Local kernel gradients
#         local_grad_K = jnp.array([
#             -local_K[j] * (particles_z[i] - particles_z[j]) / (local_bws[i]**2)
#             for j in range(n_particles)
#         ])
        
#         positive_term = jnp.sum(local_K[:, None] * grad_logp_z, axis=0) / n_particles
#         negative_term = jnp.sum(local_grad_K, axis=0) / n_particles
#         phi = phi.at[i].set(positive_term + negative_term)
    
#     # Adaptive step
#     step_factor = kl_adaptive_step(particles_z, kl_target)
#     step_size = jnp.clip(max_step * step_factor, 1e-7, max_step)
    
#     return particles_z + step_size * phi

# # Distributed SVGD
# def distributed_svgd_step(particles_z, k, m, kl_target=0.1, max_step=0.001):
#     """Distributed SVGD step using pjit"""
#     return update_median_bw_kl_step(particles_z, k, m, kl_target, max_step)

# # Main SVGD function
# def run_variable_dim_svgd(key, data, k, m, n_particles=40, n_steps=100, lr=0.001):
#     """Run SVGD for variable-dimension discrete phase-type distributions"""
    
#     # Calculate parameter dimension
#     param_dim = calculate_param_dim(k, m)
#     print(f"Running SVGD for k={k}, m={m} (param_dim={param_dim})")
    
#     # Generate true parameters
#     key, subkey = jax.random.split(key)
#     true_params = example_ptd_spec(subkey, k, m)
    
#     # SVGD parameters
#     n_devices = min(8, n_particles)  # Don't exceed available devices
#     kl_target_base = 0.1
#     kl_target_decay = 0.01
#     max_step = lr
#     min_step = 1e-7
#     max_step_scaler = 0.1
    
#     if n_particles % n_devices != 0:
#         n_particles = (n_particles // n_devices) * n_devices
#         print(f"Adjusted n_particles to {n_particles} for even sharding")
    
#     # Initial particles
#     key, subkey = jax.random.split(key)
#     particles_z = jax.random.normal(subkey, shape=(n_particles, param_dim))
    
#     # Shard particles over devices
#     devices = mesh_utils.create_device_mesh((n_devices,))
#     mesh = Mesh(devices, axis_names=("i",))
#     sharding = NamedSharding(mesh, P("i", None))
#     particles_z = jax.device_put(particles_z, sharding)
    
#     # SVGD iterations
#     particle_z_history = [particles_z]
#     every = max(1, n_steps // 10)  # Save every 10% of iterations
#     prev = None
    
#     with mesh:
#         # for i in range(n_steps):
#         for i in trange(n_steps):
#             kl_target = decayed_kl_target(i, base=kl_target_base, decay=kl_target_decay)
#             particles_z = distributed_svgd_step(particles_z, k, m, kl_target=kl_target, max_step=max_step)
            
#             if not i % every:
#                 particle_z_history.append(particles_z)
    
#     # Extract final results
#     particles = jnp.array([z_to_theta(z) for z in particles_z])
    
#     print(f"\nResults for k={k}, m={m}:")
#     print(f"True parameters shape: {true_params.shape}")
#     print(f"Estimated parameters shape: {particles.shape}")
#     print(f"Parameter means: {jnp.mean(particles, axis=0)}")
#     print(f"True parameters: {true_params}")
    
#     return particles, particle_z_history, true_params

# ==============================================================================
# Main SVGD API for external use
# ==============================================================================

@jit
def _compute_kernel_grad_impl(particles, bandwidth):
    """
    JIT-compiled RBF kernel computation (core implementation)

    Parameters
    ----------
    particles : array (n_particles, theta_dim)
        Current particle positions
    bandwidth : float
        Kernel bandwidth

    Returns
    -------
    K : array (n_particles, n_particles)
        Kernel matrix
    grad_K : array (n_particles, n_particles, theta_dim)
        Gradient of kernel matrix
    """
    # Vectorized computation - no Python loops!
    # Shape: (n_particles, n_particles, theta_dim)
    diff = particles[:, None, :] - particles[None, :, :]

    # Squared distances: (n_particles, n_particles)
    sq_dist = jnp.sum(diff**2, axis=2)

    # Kernel matrix: K[i,j] = exp(-||x_i - x_j||^2 / (2*h^2))
    K = jnp.exp(-sq_dist / (2 * bandwidth**2))

    # Kernel gradient: ∇K[i,j] = -K[i,j] * (x_i - x_j) / h^2
    # Shape: (n_particles, n_particles, theta_dim)
    grad_K = -K[:, :, None] * diff / bandwidth**2

    return K, grad_K


class SVGDKernel:
    """RBF kernel for SVGD with automatic bandwidth selection"""

    def __init__(self, bandwidth='median'):
        """
        Parameters
        ----------
        bandwidth : str or float default='median'
            Bandwidth selection method. Options:
            - 'median': Median heuristic (default)
            - float: Fixed bandwidth value
        """
        self.bandwidth_method = bandwidth

    def compute_kernel_grad(self, particles):
        """
        Compute RBF kernel matrix and its gradient (JIT-compiled)

        Parameters
        ----------
        particles : array (n_particles, theta_dim)
            Current particle positions

        Returns
        -------
        K : array (n_particles, n_particles)
            Kernel matrix
        grad_K : array (n_particles, n_particles, theta_dim)
            Gradient of kernel matrix
        """
        # Compute bandwidth (not JIT-compiled due to conditional logic)
        if isinstance(self.bandwidth_method, str) and self.bandwidth_method == 'median':
            bandwidth = batch_median_heuristic(particles)
        else:
            bandwidth = self.bandwidth_method

        # Call JIT-compiled implementation
        return _compute_kernel_grad_impl(particles, bandwidth)


@jit
def _svgd_update_jitted(particles, K, grad_K, grad_log_p, step_size):
    """
    JIT-compiled SVGD update (core computation)

    Parameters
    ----------
    particles : array (n_particles, theta_dim)
        Current particle positions
    K : array (n_particles, n_particles)
        Kernel matrix
    grad_K : array (n_particles, n_particles, theta_dim)
        Kernel gradient
    grad_log_p : array (n_particles, theta_dim)
        Log probability gradients
    step_size : float
        Step size for update

    Returns
    -------
    array (n_particles, theta_dim)
        Updated particles
    """
    n_particles = particles.shape[0]

    # SVGD update: phi = (K @ grad_log_p + sum(grad_K)) / n
    # Vectorized computation - no Python loop!
    # K: (n_particles, n_particles)
    # grad_log_p: (n_particles, theta_dim)
    # K @ grad_log_p -> (n_particles, theta_dim)
    positive_term = jnp.einsum('ij,jk->ik', K, grad_log_p) / n_particles

    # grad_K: (n_particles, n_particles, theta_dim)
    # Sum over all particle interactions -> (n_particles, theta_dim)
    negative_term = jnp.sum(grad_K, axis=1) / n_particles

    phi = positive_term + negative_term

    return particles + step_size * phi


def svgd_step(particles, log_prob_fn, kernel, step_size, compiled_grad=None,
              parallel_mode='vmap', n_devices=None):
    """
    Perform single SVGD update step

    Parameters
    ----------
    particles : array (n_particles, theta_dim)
        Current particle positions
    log_prob_fn : callable
        Log probability function: theta -> scalar
    kernel : SVGDKernel
        Kernel object for computing K and grad_K
    step_size : float
        Step size for update
    compiled_grad : callable, optional
        Precompiled gradient function for faster execution
    parallel_mode : str, default='vmap'
        Parallelization strategy: 'vmap', 'pmap', or 'none'
    n_devices : int, optional
        Number of devices to use for pmap (only used if parallel_mode='pmap')

    Returns
    -------
    array (n_particles, theta_dim)
        Updated particles
    """
    n_particles = particles.shape[0]

    # Use provided parallelization strategy
    actual_parallel_mode = parallel_mode
    actual_n_devices = n_devices

    # Compute log probability gradients based on parallelization strategy
    if actual_parallel_mode == 'pmap' and actual_n_devices is not None:
        # Parallel gradient computation across devices (pmap)
        particles_per_device = n_particles // actual_n_devices
        particles_sharded = particles.reshape(actual_n_devices, particles_per_device, -1)

        # NOTE: JAX 0.8+ requires explicit device mesh to avoid conflicts
        # Create mesh for current pmap operation
        from jax.experimental import mesh_utils
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        devices = mesh_utils.create_device_mesh((actual_n_devices,))
        mesh = Mesh(devices, axis_names=("batch",))

        # Use explicit mesh context for pmap
        # pmap over devices, vmap over particles within each device
        with mesh:
            if compiled_grad is not None:
                grad_log_p_sharded = pmap(vmap(compiled_grad), axis_name="batch")(particles_sharded)
            else:
                grad_log_p_sharded = pmap(vmap(grad(log_prob_fn)), axis_name="batch")(particles_sharded)

        grad_log_p = grad_log_p_sharded.reshape(n_particles, -1)
    elif actual_parallel_mode == 'vmap':
        # Single device vectorization - use vmap only
        if compiled_grad is not None:
            grad_log_p = vmap(compiled_grad)(particles)
        else:
            grad_log_p = vmap(grad(log_prob_fn))(particles)
    elif actual_parallel_mode == 'none':
        # No parallelization - sequential computation (useful for debugging)
        if compiled_grad is not None:
            grad_log_p = jnp.array([compiled_grad(p) for p in particles])
        else:
            grad_fn = grad(log_prob_fn)
            grad_log_p = jnp.array([grad_fn(p) for p in particles])
    else:
        raise ValueError(f"Invalid parallel_mode: {actual_parallel_mode}")

    # Compute kernel and kernel gradient
    K, grad_K = kernel.compute_kernel_grad(particles)

    # ##############    

    # # phi = jnp.zeros_like(particles)
    # # for i in range(n_particles):
    # #     positive_term = jnp.sum(K[i, :, None] * grad_log_p, axis=0) / n_particles
    # #     negative_term = jnp.sum(grad_K[i, :, :], axis=0) / n_particles
    # #     phi = phi.at[i].set(positive_term + negative_term)
    
    # positive_term = jnp.sum(K[:, :, None] * grad_log_p, axis=1) / n_particles
    # negative_term = jnp.sum(grad_K, axis=1) / n_particles
    # phi = positive_term + negative_term

    # # Adaptive step
    # # step_factor = kl_adaptive_step(particles, kl_target)
    # # _step_size = jnp.clip(max_step * step_factor, 1e-7, max_step)  * phi 
    # _step_size = step_size * phi   

    #  # Call JIT-compiled update
    # return _svgd_update_jitted(particles, K, grad_K, grad_log_p, _step_size)

    # ##############


    # Call JIT-compiled update
    return _svgd_update_jitted(particles, K, grad_K, grad_log_p, step_size)


def run_svgd(log_prob_fn, theta_init, n_steps, learning_rate=0.001,
             kernel=None, return_history=False, verbose=True, compiled_grad=None,
             parallel_mode='vmap', n_devices=None,
             log_prob_fn_factory=None, regularization_schedule=None):
    """
    Run Stein Variational Gradient Descent

    Parameters
    ----------
    log_prob_fn : callable
        Log probability function: theta -> scalar
        Should return log p(data|theta) + log p(theta)
    theta_init : array (n_particles, theta_dim)
        Initial particle positions
    n_steps : int
        Number of SVGD iterations
    learning_rate : float or StepSizeSchedule
        Step size. Can be:
        - float: constant step size (backward compatible)
        - StepSizeSchedule object: dynamic schedule
    kernel : SVGDKernel
        Kernel specification.
    return_history : bool
        If True, return particle positions at each iteration
    verbose : bool
        Print progress information
    compiled_grad : callable, optional
        Precompiled gradient function for faster execution
    parallel_mode : str, default='vmap'
        Parallelization strategy: 'vmap', 'pmap', or 'none'
    n_devices : int, optional
        Number of devices for pmap (only used if parallel_mode='pmap')

    Returns
    -------
    dict
        Results dictionary containing:
        - 'particles': Final particles (n_particles, theta_dim)
        - 'history': Particle history if return_history=True
        - 'theta_mean': Posterior mean
        - 'theta_std': Posterior standard deviation
    """

    # Initialize
    particles = theta_init
    history = [particles] if return_history else None
    history_iterations = [0] if return_history else []  # Track iteration numbers for history snapshots

    # Handle step size schedule (backward compatible)
    if isinstance(learning_rate, StepSizeSchedule):
        step_schedule = learning_rate
        use_schedule = True
    elif isinstance(learning_rate, (int, float)):
        step_schedule = ConstantStepSize(float(learning_rate))
        use_schedule = False  # Can still use constant value
    else:
        raise TypeError(
            f"learning_rate must be float or StepSizeSchedule, got: {type(learning_rate)}"
        )

    # SVGD iterations
    if verbose:
        print(f"Running SVGD: {n_steps} steps, {len(particles)} particles")

    # for step in range(n_steps) if verbose else range(n_steps):
    for step in trange(n_steps) if verbose else range(n_steps):
        # Compute current step size from schedule
        if use_schedule:
            current_step_size = step_schedule(step, particles)
        else:
            current_step_size = learning_rate

        # Compute current regularization and create log_prob_fn if using schedule
        if regularization_schedule is not None:
            current_reg = regularization_schedule(step, particles)
            # Create log_prob_fn with current regularization
            log_prob_fn = log_prob_fn_factory(current_reg)
            # Gradient is computed on-the-fly (no precompilation benefit)
            compiled_grad_to_use = None
        else:
            compiled_grad_to_use = compiled_grad

        # Perform SVGD update
        particles = svgd_step(particles, log_prob_fn, kernel, current_step_size,
                             compiled_grad=compiled_grad_to_use,
                             parallel_mode=parallel_mode,
                             n_devices=n_devices)

        # Store history
        if return_history: # and (step % max(1, n_steps // 20) == 0):
            history.append(particles)
            history_iterations.append(step)

    # Final history
    if return_history:
        history.append(particles)
        history_iterations.append(n_steps)

    # Compute summary statistics
    theta_mean = jnp.mean(particles, axis=0)
    theta_std = jnp.std(particles, axis=0)

    results = {
        'particles': particles,
        'theta_mean': theta_mean,
        'theta_std': theta_std,
    }

    if return_history:
        results['history'] = history
        results['history_iterations'] = history_iterations

    # Note: Final summary is printed by SVGD.fit() with transformed values

    return results


# ============================================================================
# Helper Functions for Moment-Based Regularization
# ============================================================================

def compute_sample_moments(data, nr_moments):
    """
    Compute sample moments from observed data.

    Parameters
    ----------
    data : array_like
        Observed data points (e.g., waiting times, event times)
    nr_moments : int
        Number of moments to compute

    Returns
    -------
    jnp.array
        Sample moments [mean(data), mean(data^2), ..., mean(data^k)]
        Shape: (nr_moments,)

    Examples
    --------
    >>> data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> moments = compute_sample_moments(data, nr_moments=2)
    >>> print(moments)  # [3.0, 11.0] = [mean, mean of squares]
    >>> # Variance from moments: Var = E[X^2] - E[X]^2 = 11.0 - 3.0^2 = 2.0
    """
    data = jnp.array(data)
    moments = []
    for k in range(1, nr_moments + 1):
        moments.append(jnp.mean(data**k))
    return jnp.array(moments)


# ============================================================================
# SVGD Class for Object-Oriented Interface
# ============================================================================

class SVGD:
    """
    Stein Variational Gradient Descent (SVGD) for Bayesian parameter inference.

    This class provides an object-oriented interface for SVGD inference with
    automatic result storage and diagnostic plotting capabilities.

    Parameters
    ----------
    model : callable
        JAX-compatible parameterized model with signature: model(theta, data) -> values
    observed_data : array_like
        Observed data points
    prior : callable, optional
        Log prior function: prior(theta) -> scalar.
        If None, uses standard normal prior: log p(theta) = -0.5 * sum(theta^2)
    n_particles : int, default is 20 times length of theta
        Number of SVGD particles
    n_iterations : int, default=1000
        Number of SVGD optimization steps
    learning_rate : float or StepSizeSchedule, default=0.001
        SVGD step size. Can be:
        - float: constant step size (backward compatible)
        - StepSizeSchedule object: dynamic step size schedule
        Examples: ConstantStepSize(0.01), ExponentialDecayStepSize(0.1, 0.01, 500.0)
    bandwidth : str default='median'
        Kernel bandwidth selection. Can be:
        - str: 'median' for median heuristic (backward compatible)
    theta_init : array_like, optional
        Initial particle positions (n_particles, theta_dim)
    theta_dim : int, optional
        Dimension of theta parameter vector (required if theta_init is None)
    seed : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Print progress information
    jit : bool or None, default=None
        Enable JIT compilation. If None, uses value from phasic.get_config().jit.
        If True, requires JAX to be available (raises PTDConfigError otherwise).
        JIT compilation provides significant speedup but adds initial compilation overhead.
    parallel : str or None, default=None
        Parallelization strategy:
        - 'vmap': Vectorize across particles (single device)
        - 'pmap': Parallelize across devices (uses multiple CPUs/GPUs)
        - 'none': No parallelization (sequential, useful for debugging)
        - None: Auto-select (pmap if multiple devices, vmap otherwise)

        **Single-machine multi-CPU**: Auto-selection uses pmap for multi-core parallelization.
        **Multi-node SLURM**: Call initialize_distributed() then set parallel='pmap' explicitly.
    n_devices : int or None, default=None
        Number of devices to use for pmap. Only used when parallel='pmap'.
        If None, uses all available devices. Must be <= number of available JAX devices.
        See: jax.devices() to check available devices, or configure via PTDALG_CPUS
        environment variable before import.
    precompile : bool, default=True
        (Deprecated: use jit parameter instead)
        Precompile model and gradient functions for faster execution.
        Implies jit=True for backward compatibility.
        First run will take longer (compilation time) but subsequent
        iterations will be much faster. Compiled functions are cached
        in memory and on disk (~/.phasic_cache/).
    compilation_config : CompilationConfig, dict, str, or Path, optional
        JAX compilation optimization configuration. Can be:
        - CompilationConfig object from phasic.CompilationConfig
        - dict with CompilationConfig parameters
        - str/Path to JSON config file
        - None (uses default balanced configuration)

        The configuration controls JAX/XLA compilation behavior including:
        - Persistent cache directory for cross-session caching
        - Optimization level (0-3)
        - Parallel compilation settings

        Examples:
        - Use preset: CompilationConfig.fast_compile()
        - Load from file: 'my_config.json'
        - Custom dict: {'optimization_level': 2, 'cache_dir': '/tmp/cache'}
    positive_params : bool, default=True
        If True, constrains parameters to positive domain using softplus transformation.
        SVGD operates in unconstrained space (can be negative) but results are
        transformed to positive values.

        DEFAULT=True because phase-type distribution parameters (rates) must be positive.
        Set to False only if you have a specific reason to allow negative parameters
        (e.g., regression coefficients, log-space parameterization).
    param_transform : callable, optional
        Custom parameter transformation function. Overrides positive_params if provided.
        Should map unconstrained space to constrained space (e.g., lambda x: jax.nn.sigmoid(x)
        for parameters in [0,1]). Cannot be used together with positive_params=True.
    regularization : float or RegularizationSchedule, default=0.0
        Moment-based regularization strength. Can be:
        - float: constant regularization (0.0 = no regularization, >0.0 = regularized SVGD)
        - RegularizationSchedule object: dynamic regularization schedule
        Examples: ConstantRegularization(1.0), ExponentialDecayRegularization(5.0, 0.1, 500.0)

        If > 0.0, adds penalty term to match model moments to sample moments.
        Sample moments are computed from observed_data at initialization.

        **Note**: Using RegularizationSchedule disables gradient precompilation for flexibility,
        which may be slower than constant regularization but allows dynamic strategies.
    nr_moments : int, default=2
        Number of moments to use for regularization. Only used if regularization > 0.
        Typical values: 2 (mean and variance) or 3 (mean, variance, skewness).

    Attributes
    ----------
    particles : array
        Final posterior samples (n_particles, theta_dim)
    theta_mean : array
        Posterior mean estimate
    theta_std : array
        Posterior standard deviation
    history : list of arrays, optional
        Particle evolution over iterations (if fit was called with return_history=True)
    is_fitted : bool
        Whether fit() has been called

    Examples
    --------
    >>> # Basic usage with auto-configuration
    >>> svgd = SVGD(model, observed_data, theta_dim=1)
    >>> svgd.fit()
    >>>
    >>> # Explicit single-device configuration
    >>> svgd = SVGD(model, observed_data, theta_dim=1, jit=True, parallel='vmap')
    >>> svgd.fit()
    >>>
    >>> # Multi-device parallelization
    >>> svgd = SVGD(model, observed_data, theta_dim=1,
    ...             jit=True, parallel='pmap', n_devices=8)
    >>> svgd.fit()
    >>>
    >>> # No JIT (for debugging)
    >>> svgd = SVGD(model, observed_data, theta_dim=1, jit=False, parallel='none')
    >>> svgd.fit()
    >>>
    >>> # Multi-node SLURM (explicit distributed initialization)
    >>> from phasic import initialize_distributed
    >>> dist = initialize_distributed()  # Auto-detects SLURM environment
    >>> svgd = SVGD(model, observed_data, theta_dim=1,
    ...             jit=True, parallel='pmap', n_devices=dist.num_processes)
    >>> svgd.fit()
    >>>
    >>> # Using step size schedules to prevent divergence
    >>> from phasic import ExponentialDecayStepSize
    >>> schedule = ExponentialDecayStepSize(first_step=0.1, last_step=0.01, tau=500.0)
    >>> svgd = SVGD(model, observed_data, theta_dim=1, learning_rate=schedule)
    >>> svgd.fit()
    >>>
    >>> # Using adaptive step size based on particle spread
    >>> from phasic import AdaptiveStepSize
    >>> schedule = AdaptiveStepSize(base_step=0.01, kl_target=0.1, adjust_rate=0.1)
    >>> svgd = SVGD(model, observed_data, theta_dim=1, learning_rate=schedule)
    >>> svgd.fit()
    >>>
    >>> # Using regularization schedules for moment matching
    >>> from phasic import ExponentialDecayRegularization
    >>> reg_schedule = ExponentialDecayRegularization(first_reg=5.0, last_reg=0.1, tau=500.0)
    >>> svgd = SVGD(model, observed_data, theta_dim=1,
    ...             regularization=reg_schedule, nr_moments=2)
    >>> svgd.fit()  # Starts with strong regularization, gradually reduces
    >>>
    >>> # Using CDF-based regularization schedule (bidirectional)
    >>>
    >>> # Constant regularization (no schedule)
    >>> svgd = SVGD(model, observed_data, theta_dim=1, regularization=1.0, nr_moments=2)
    >>> svgd.fit()
    >>>
    >>> # Using custom bandwidth schedule
    >>> from phasic import LocalAdaptiveBandwidth
    >>> bandwidth = LocalAdaptiveBandwidth(alpha=0.9, k_frac=0.1)
    >>> svgd = SVGD(model, observed_data, theta_dim=1, kernel=bandwidth)
    >>> svgd.fit()
    >>>
    >>> # Access results
    >>> print(svgd.theta_mean)
    >>> print(svgd.theta_std)
    >>>
    >>> # Generate diagnostic plots
    >>> svgd.plot_posterior()
    >>> svgd.plot_trace()
    """

    # Class-level cache for compiled models (shared across instances)
    _compiled_cache = {}

    def __init__(self, model, observed_data, prior=None, n_particles=None,
                 n_iterations=1000, learning_rate=0.001, bandwidth='median',
                 theta_init=None, theta_dim=None, seed=42, verbose=True,
                 jit=None,              # NEW: explicit JIT control
                 parallel=None,         # NEW: 'vmap', 'pmap', 'none'
                 n_devices=None,        # NEW: explicit device count for pmap
                 precompile=True,       # Keep for backward compat
                 compilation_config=None, positive_params=True, param_transform=None,
                 regularization=0.0, nr_moments=2, rewards=None):

        if n_particles is None:
            n_particles = 20 * theta_dim

        # Get configuration
        config = get_config()

        # Validate JIT parameter against config
        if jit is None:
            jit = config.jit  # Use config default
        elif jit and not config.jax:
            raise PTDConfigError(
                "jit=True requires JAX.\n"
                "  Current config: jax=False\n"
                "  Fix: phasic.configure(jax=True)"
            )

        # Validate parallel parameter
        if parallel is None:
            # Default: use pmap if multiple devices, vmap otherwise
            # This enables multi-core parallelization on single machines
            # For multi-node SLURM: call initialize_distributed() + set parallel='pmap' explicitly
            parallel = 'pmap' if len(jax.devices()) > 1 else 'vmap'
            if verbose:
                print(f"Auto-selected parallel='{parallel}' ({len(jax.devices())} devices available)")
        elif parallel not in ['vmap', 'pmap', 'none']:
            raise ValueError(
                f"parallel must be 'vmap', 'pmap', or 'none', got: {parallel}"
            )

        # Validate n_devices parameter and check for misconfigurations
        available_devices = len(jax.devices())

        if parallel == 'pmap':
            if available_devices == 1:
                import warnings
                warnings.warn(
                    "parallel='pmap' requested but only 1 JAX device available. "
                    "Using 'vmap' instead. To use pmap, configure more devices via "
                    "PTDALG_CPUS environment variable or initialize_distributed().",
                    UserWarning,
                    stacklevel=2
                )
                parallel = 'vmap'
                n_devices = None
            else:
                if n_devices is None:
                    n_devices = available_devices
                    if verbose:
                        print(f"Using all {n_devices} devices for pmap")
                elif n_devices > available_devices:
                    raise PTDConfigError(
                        f"n_devices={n_devices} but only {available_devices} devices available.\n"
                        f"  JAX devices: {jax.devices()}\n"
                        f"  Fix: Set n_devices<={available_devices} or configure more devices\n"
                        f"  See: PTDALG_CPUS environment variable or phasic.configure()"
                    )
                elif n_devices < 1:
                    raise ValueError(f"n_devices must be >= 1, got: {n_devices}")
        elif n_devices is not None:
            if verbose:
                print(f"Warning: n_devices={n_devices} ignored (only used with parallel='pmap')")
            n_devices = None

        # if verbose:
        #     print("---------------------------------------------")
        #     print(f"SVGD Configuration:")
        #     print(f"  JIT compilation:        {jit}")
        #     print(f"  Parallelization mode:   {parallel}")
        #     if parallel == 'pmap':
        #         print(f"  Number of devices:      {n_devices} (available: {available_devices})")    
        #     print("---------------------------------------------")

        # Store configuration (parallel may have been modified by validation)
        self.jit_enabled = jit
        self.parallel_mode = parallel
        self.n_devices = n_devices

        # Backward compatibility: precompile implies jit (deprecated)
        if precompile is not None and not precompile:
            import warnings
            warnings.warn(
                "precompile parameter is deprecated and will be removed in v1.0. "
                "Use jit=True/False instead.",
                DeprecationWarning,
                stacklevel=2
            )
        if precompile and not jit:
            if verbose:
                print("Warning: precompile=True but jit=False. Setting jit=True for backward compatibility.")
            self.jit_enabled = True

        # Handle compilation configuration
        if compilation_config is not None:
            from pathlib import Path
            try:
                from .jax_config import CompilationConfig
            except ImportError:
                # If running from svgd.py directly without package import
                try:
                    from jax_config import CompilationConfig
                except ImportError:
                    CompilationConfig = None

            # Parse compilation_config
            if isinstance(compilation_config, str) or isinstance(compilation_config, Path):
                # Load from file
                if CompilationConfig:
                    config = CompilationConfig.load_from_file(compilation_config)
                    config.apply(force=False)
                    if verbose:
                        print(f"Loaded compilation config from: {compilation_config}")
            elif isinstance(compilation_config, dict):
                # Create from dictionary
                if CompilationConfig:
                    config = CompilationConfig(**compilation_config)
                    config.apply(force=False)
                    if verbose:
                        print(f"Applied compilation config from dict")
            elif CompilationConfig and isinstance(compilation_config, CompilationConfig):
                # Already a CompilationConfig object
                compilation_config.apply(force=False)
                if verbose:
                    print(f"Applied compilation config")
            else:
                if verbose:
                    print(f"Warning: Could not parse compilation_config, using defaults")

        self.model = model
        self.observed_data = jnp.array(observed_data)
        self.prior = prior
        self.n_particles = n_particles
        self.n_iterations = n_iterations

        # Handle step size schedule (backward compatible)
        if isinstance(learning_rate, StepSizeSchedule):
            self.step_schedule = learning_rate
            self.learning_rate = None  # Will be computed dynamically
        elif isinstance(learning_rate, (int, float)):
            self.step_schedule = ConstantStepSize(float(learning_rate))
            self.learning_rate = float(learning_rate)  # Keep for backward compat
        else:
            raise TypeError(
                f"learning_rate must be float or StepSizeSchedule, got: {type(learning_rate)}"
            )

        self.bandwidth = bandwidth
        self.theta_dim = theta_dim
        self.seed = seed
        self.verbose = verbose
        self.precompile = precompile
        self.compilation_config = compilation_config

        # Handle parameter transformation
        if positive_params and param_transform is not None:
            raise ValueError(
                "Cannot specify both positive_params=True and param_transform. "
                "Use positive_params=True for automatic softplus transformation, "
                "or provide a custom param_transform function."
            )

        if positive_params:
            self.param_transform = lambda theta: jax.nn.softplus(theta)
            if verbose:
                print("Using softplus transformation to constrain parameters to positive domain")
        elif param_transform is not None:
            if not callable(param_transform):
                raise ValueError("param_transform must be a callable function")
            self.param_transform = param_transform
            if verbose:
                print("Using custom parameter transformation")
        else:
            self.param_transform = None

        # Validate and initialize particles
        if theta_init is None and theta_dim is None:
            raise ValueError(
                "Either theta_init or theta_dim must be provided. "
                "If you don't have initial particles, specify theta_dim (the number of parameters)."
            )

        # Adjust n_particles for pmap if needed
        if self.parallel_mode == 'pmap' and self.n_devices is not None:
            if n_particles % self.n_devices != 0:
                adjusted_n_particles = ((n_particles + self.n_devices - 1) // self.n_devices) * self.n_devices
                if verbose:
                    print(f"Adjusted n_particles from {n_particles} to {adjusted_n_particles} "
                          f"for even distribution across {self.n_devices} devices")
                n_particles = adjusted_n_particles
                self.n_particles = n_particles

        # Initialize particles
        key = jax.random.PRNGKey(seed)
        if theta_init is None:
            if self.param_transform is not None:
                # For transformed parameters, initialize in a range that maps to reasonable positive values
                # softplus(x) ≈ x for x >> 0, and softplus(0) ≈ 0.69
                # Initialize around N(1, 1) so softplus gives values around 1-2
                self.theta_init = jax.random.normal(key, (n_particles, theta_dim)) + 1.0
                if verbose:
                    print(f"Initialized {n_particles} particles with theta_dim={theta_dim} from N(1,1)")
                    print(f"  (Transformed range: softplus(N(1,1)) ≈ [0.7, 3.5])")
            else:
                self.theta_init = jax.random.normal(key, (n_particles, theta_dim))
                if verbose:
                    print(f"Initialized {n_particles} particles with theta_dim={theta_dim} from N(0,1)")
        else:
            self.theta_init = jnp.array(theta_init)
            if self.theta_init.ndim != 2:
                raise ValueError(
                    f"theta_init must be 2D array (n_particles, theta_dim), "
                    f"got shape {self.theta_init.shape}"
                )
            self.n_particles = self.theta_init.shape[0]
            self.theta_dim = self.theta_init.shape[1]
            if verbose:
                print(f"Using provided initial particles: {self.theta_init.shape}")

        # Store regularization settings and handle regularization schedule (backward compatible)
        if isinstance(regularization, RegularizationSchedule):
            self.regularization_schedule = regularization
            self.use_regularization_schedule = True
            # Evaluate at iteration 0 to get initial value
            self.regularization = regularization(0)
            if verbose:
                print(f"Using regularization schedule (initial value: {self.regularization})")
        elif isinstance(regularization, (int, float)):
            self.regularization_schedule = ConstantRegularization(float(regularization))
            self.use_regularization_schedule = False
            self.regularization = float(regularization)
        else:
            raise TypeError(
                f"regularization must be float or RegularizationSchedule, got: {type(regularization)}"
            )

        self.nr_moments = nr_moments
        self.rewards = rewards  # Can be None, 1D (n_vertices,), or 2D (n_vertices, n_features)

        # Compute sample moments if initial regularization > 0 or using schedule
        # (schedule might start at 0 but increase later, so we need moments ready)
        if self.regularization > 0.0 or self.use_regularization_schedule:
            self.sample_moments = compute_sample_moments(self.observed_data, nr_moments)
            if verbose and self.regularization > 0.0:
                print(f"Computed {nr_moments} sample moments for regularization={self.regularization}")
        else:
            self.sample_moments = None

        # Validate that model returns (pmf, moments) tuple
        # All models must use Graph.pmf_and_moments_from_graph()
        try:
            test_theta = self.theta_init[0]
            test_times = self.observed_data[:min(2, len(self.observed_data))]

            # Test with rewards if provided
            if self.rewards is not None:
                # For 2D rewards, extract first 2 columns to match test_times
                if jnp.asarray(self.rewards).ndim == 2 and test_times.ndim == 2:
                    test_rewards = jnp.asarray(self.rewards)[:, :test_times.shape[1]]
                else:
                    test_rewards = self.rewards
                result = self.model(test_theta, test_times, rewards=test_rewards)
            else:
                result = self.model(test_theta, test_times, rewards=None)

            if not isinstance(result, tuple) or len(result) != 2:
                raise ValueError(
                    "Model must return (pmf, moments) tuple. "
                    f"Got: {type(result)}. "
                    "Use Graph.pmf_and_moments_from_graph() to create model, "
                    "not Graph.pmf_from_graph()."
                )

            # Validate number of moments matches nr_moments parameter (if using regularization)
            if self.nr_moments > 0 and (self.regularization > 0.0 or self.use_regularization_schedule):
                pmf_vals, model_moments = result

                # Handle 2D moments (multivariate case)
                if model_moments.ndim == 2:
                    # Check shape: (n_features, nr_moments)
                    actual_nr_moments = model_moments.shape[1]
                else:
                    # 1D moments
                    actual_nr_moments = len(model_moments)

                if actual_nr_moments < self.nr_moments:
                    raise ValueError(
                        f"Model returns {actual_nr_moments} moments but SVGD is configured to use {self.nr_moments} moments. "
                        f"Create model with: Graph.pmf_and_moments_from_graph(graph, nr_moments={self.nr_moments})"
                    )

            if verbose:
                print("Model validated: returns (pmf, moments) tuple")
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Other errors during model evaluation
            raise ValueError(
                f"Model validation failed. Error: {e}\n"
                "Ensure model has signature: model(theta, times, rewards=None) -> (pmf, moments)"
            )

        # Results (initialized after fit())
        self.particles = None
        self.theta_mean = None
        self.theta_std = None
        self.history = None
        self.history_iterations = None
        self.is_fitted = False

        # Compiled model and gradient (set by _precompile_model if jit_enabled=True)
        self.compiled_model = None
        self.compiled_grad = None

        # Precompilation now happens in optimize() based on regularization settings
        # This allows caching for both regularized and non-regularized cases
        # Old behavior: if self.jit_enabled: self._precompile_model()

    def _log_prob(self, theta):
        """
        Log probability function: log p(data|theta) + log p(theta)

        Parameters
        ----------
        theta : array
            Parameter vector (in unconstrained space if using transformation)

        Returns
        -------
        scalar
            Log probability
        """
        # Apply parameter transformation if specified
        if self.param_transform is not None:
            theta_transformed = self.param_transform(theta)
        else:
            theta_transformed = theta

        # Log-likelihood
        try:
            result = self.model(theta_transformed, self.observed_data)

            # Handle both (pmf, moments) and pmf-only models
            if isinstance(result, tuple):
                model_values = result[0]  # Extract PMF values
            else:
                model_values = result
        except Exception as e:
            raise ValueError(
                f"Model evaluation failed. Ensure model has signature model(theta, times). "
                f"Error: {e}"
            )

        # Prevent log(0) by adding small epsilon
        log_lik = jnp.sum(jnp.log(model_values + 1e-10))

        # Log-prior (evaluated in unconstrained space)
        if self.prior is not None:
            log_pri = self.prior(theta)
        else:
            # Default: standard normal prior on unconstrained parameters
            log_pri = -0.5 * jnp.sum(theta**2)

        return log_lik + log_pri

    def _log_prob_regularized(self, theta, sample_moments, nr_moments, regularization):
        """
        Regularized log probability with moment matching term.

        log p(theta | data, moments) = log p(data|theta) + log p(theta) - λ * ||E[T^k|theta] - sample_moments||^2

        Parameters
        ----------
        theta : array
            Parameter vector (in unconstrained space if using transformation)
        sample_moments : array
            Sample moments computed from observed data
        nr_moments : int
            Number of moments to use for regularization
        regularization : float
            Strength of moment regularization (λ in objective)

        Returns
        -------
        scalar
            Regularized log probability
        """
        # Apply parameter transformation if specified
        if self.param_transform is not None:
            theta_transformed = self.param_transform(theta)
        else:
            theta_transformed = theta

        # Evaluate model to get PMF and moments
        try:
            result = self.model(theta_transformed, self.observed_data)
            if isinstance(result, tuple) and len(result) == 2:
                pmf_vals, model_moments = result
            else:
                raise ValueError("Model must return (pmf, moments) tuple for regularized SVGD")
        except Exception as e:
            raise ValueError(
                f"Model evaluation failed. Ensure model signature is model(theta, times) -> (pmf, moments). "
                f"Error: {e}"
            )

        # Standard log-likelihood term
        log_lik = jnp.sum(jnp.log(pmf_vals + 1e-10))

        # Log-prior term (evaluated in unconstrained space)
        if self.prior is not None:
            log_pri = self.prior(theta)
        else:
            # Default: standard normal prior
            log_pri = -0.5 * jnp.sum(theta**2)

        # Moment regularization penalty
        # We want to minimize (model_moments - sample_moments)^2
        # So we subtract this from log probability
        moment_diff = model_moments[:nr_moments] - sample_moments
        moment_penalty = regularization * jnp.sum(moment_diff**2)

        return log_lik + log_pri - moment_penalty

    def _log_prob_unified(self, theta, nr_moments=0, sample_moments=None,
                         regularization=0.0, rewards=None):
        """
        Unified log probability with optional moment regularization.

        This replaces both _log_prob() and _log_prob_regularized() with a single
        implementation that handles both cases based on the regularization parameter.

        Parameters
        ----------
        theta : array
            Parameter vector (in unconstrained space if using transformation)
        nr_moments : int, default=0
            Number of moments to use for regularization (only used if regularization > 0)
        sample_moments : array or None
            Sample moments from observed data (required if regularization > 0)
        regularization : float, default=0.0
            Strength of moment regularization (λ)
            - 0.0: No regularization
            - > 0.0: Moment-based regularization penalty
        rewards : array or None
            Reward vector for reward-transformed likelihood (not yet implemented)

        Returns
        -------
        scalar
            Log probability (with or without moment regularization penalty)

        Raises
        ------
        ValueError
            If regularization > 0 but model doesn't return moments
        ValueError
            If regularization > 0 but sample_moments is None
        """
        # Apply parameter transformation if specified
        if self.param_transform is not None:
            theta_transformed = self.param_transform(theta)
        else:
            theta_transformed = theta

        # Evaluate model
        try:
            result = self.model(theta_transformed, self.observed_data, rewards=self.rewards)
        except Exception as e:
            raise ValueError(
                f"Model evaluation failed. Ensure model has signature model(theta, times, rewards=None). "
                f"Error: {e}"
            )

        # Always expect (pmf, moments) tuple
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(
                "Model must return (pmf, moments) tuple. "
                f"Got: {type(result)}. "
                "Use Graph.pmf_and_moments_from_graph() to create model."
            )

        pmf_vals, model_moments = result

        # Log-likelihood term
        log_lik = jnp.sum(jnp.log(pmf_vals + 1e-10))

        # Log-prior term (evaluated in unconstrained space)
        if self.prior is not None:
            log_pri = self.prior(theta)
        else:
            # Default: standard normal prior on unconstrained parameters
            log_pri = -0.5 * jnp.sum(theta**2)

        # Moment regularization penalty
        # Always compute penalty if moments available (but it's zero if regularization=0)
        # This avoids Python control flow on potentially-traced values
        if sample_moments is not None and nr_moments > 0:
            # Handle 2D moments (multivariate case)
            if model_moments.ndim == 2:
                # Aggregate moments across features by taking mean
                # Shape: (n_features, nr_moments) -> (nr_moments,)
                model_moments_agg = jnp.mean(model_moments, axis=0)
                moment_diff = model_moments_agg[:nr_moments] - sample_moments
            else:
                # 1D moments (standard case)
                moment_diff = model_moments[:nr_moments] - sample_moments

            moment_penalty = regularization * jnp.sum(moment_diff**2)
            return log_lik + log_pri - moment_penalty
        else:
            # No regularization: moments computed but not used
            return log_lik + log_pri

    def _get_cache_path(self):
        """Generate cache path for this model configuration"""
        # Create cache key from model id and shapes
        theta_shape = (self.theta_dim,)
        times_shape = self.observed_data.shape
        cache_key = f"{id(self.model)}_{theta_shape}_{times_shape}"
        cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()[:16]

        # Cache directory
        cache_dir = pathlib.Path.home() / '.phasic_cache'
        cache_dir.mkdir(exist_ok=True)

        return cache_dir / f"compiled_svgd_{cache_hash}.pkl"

    def _get_cache_key_unified(self, nr_moments, regularization, rewards=None):
        """
        Generate cache key including regularization parameters.

        Different regularization settings require different compiled gradients,
        so we include nr_moments, regularization, and rewards in the cache key.

        Parameters
        ----------
        nr_moments : int
            Number of moments for regularization
        regularization : float
            Regularization strength
        rewards : tuple or None, optional
            Reward vector as tuple for hashing

        Returns
        -------
        str
            Cache hash for this configuration
        """
        theta_shape = (self.theta_dim,)
        times_shape = self.observed_data.shape
        # Include nr_moments, regularization, and rewards in cache key
        rewards_str = str(rewards) if rewards is not None else "None"
        cache_key = f"{id(self.model)}_{theta_shape}_{times_shape}_{nr_moments}_{regularization}_{rewards_str}"
        cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
        return cache_hash

    def _save_compiled(self, cache_path):
        """Save compiled model and gradient to disk"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'model': self.compiled_model,
                    'grad': self.compiled_grad
                }, f)
            if self.verbose:
                print(f"  Saved compiled functions to cache: {cache_path.name}")
        except Exception as e:
            # Disk caching is best-effort; memory cache still works
            # Pickling JIT functions with closures often fails - this is expected
            pass

    def _load_compiled(self, cache_path):
        """Load compiled model and gradient from disk"""
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                self.compiled_model = cached['model']
                self.compiled_grad = cached['grad']
                if self.verbose:
                    print(f"  Loaded compiled functions from cache: {cache_path.name}")
                return True
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed to load cache: {e}")
                return False
        return False

    def _precompile_model(self):
        """Precompile model and gradient for known shapes"""
        # Generate cache key
        theta_shape = (self.theta_dim,)
        times_shape = self.observed_data.shape
        memory_cache_key = (id(self.model), theta_shape, times_shape)

        # Check memory cache first
        if memory_cache_key in SVGD._compiled_cache:
            cached = SVGD._compiled_cache[memory_cache_key]
            self.compiled_model = cached['model']
            self.compiled_grad = cached['grad']
            if self.verbose:
                print(f"  Using cached compiled functions from memory")
            return

        # Check disk cache
        cache_path = self._get_cache_path()
        if self._load_compiled(cache_path):
            # Store in memory cache for future instances
            SVGD._compiled_cache[memory_cache_key] = {
                'model': self.compiled_model,
                'grad': self.compiled_grad
            }
            return

        # Need to compile
        if self.verbose:
            print(f"\nPrecompiling gradient function...")
            print(f"  Theta shape: {theta_shape}, Times shape: {times_shape}")
            print(f"  This may take several minutes for large models...")

        # Create dummy inputs with correct shapes
        dummy_theta = jnp.zeros(theta_shape)

        # JIT compile gradient (use jit without lower/compile so it can be vmapped/pmapped)
        if self.verbose:
            print(f"  JIT compiling gradient...")
        start = time()
        grad_fn = jax.grad(self._log_prob)
        self.compiled_grad = jax.jit(grad_fn)
        # Trigger compilation with dummy call
        _ = self.compiled_grad(dummy_theta)
        if self.verbose:
            print(f"  Gradient JIT compiled in {time() - start:.1f}s")
            print(f"  Precompilation complete!")

        # Save to both caches
        SVGD._compiled_cache[memory_cache_key] = {
            'model': self.compiled_model,
            'grad': self.compiled_grad
        }
        self._save_compiled(cache_path)

    def _create_log_prob_fn_with_regularization(self, regularization_value):
        """
        Create log_prob function with specific regularization value.

        This factory method is used for regularization schedules, where the
        regularization value changes per iteration.

        Parameters
        ----------
        regularization_value : float
            Current regularization strength for this iteration

        Returns
        -------
        callable
            Log probability function with signature: theta -> scalar
        """
        return partial(
            self._log_prob_unified,
            nr_moments=self.nr_moments,
            sample_moments=self.sample_moments,
            regularization=regularization_value,
            rewards=None
        )

    def _precompile_unified(self, nr_moments, sample_moments, regularization, rewards=None):
        """
        Precompile gradient for unified log_prob with given regularization settings.

        Handles caching (both memory and disk) for compiled gradients with different
        regularization parameters.

        Parameters
        ----------
        nr_moments : int
            Number of moments for regularization
        sample_moments : array or None
            Sample moments from data
        regularization : float
            Regularization strength
        rewards : array or None, optional
            Optional reward vector for reward-transformed moments

        Returns
        -------
        compiled_grad : callable
            JIT-compiled gradient function
        """
        # Generate cache key including regularization params and rewards
        # Convert rewards to hashable tuple (JAX arrays aren't hashable)
        if rewards is not None:
            import numpy as np
            rewards_tuple = tuple(np.asarray(rewards).flatten())
        else:
            rewards_tuple = None
        cache_hash = self._get_cache_key_unified(nr_moments, regularization, rewards_tuple)
        memory_cache_key = (id(self.model), self.theta_dim, self.observed_data.shape,
                           nr_moments, regularization, rewards_tuple)

        # Check memory cache first
        if memory_cache_key in SVGD._compiled_cache:
            cached = SVGD._compiled_cache[memory_cache_key]
            compiled_grad = cached['grad']
            if self.verbose:
                print(f"  Using cached compiled gradient from memory")
            return compiled_grad

        # Check disk cache
        cache_dir = pathlib.Path.home() / '.phasic_cache'
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / f"compiled_svgd_{cache_hash}.pkl"

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                compiled_grad = cached['grad']
                if self.verbose:
                    print(f"  Loaded compiled gradient from disk cache: {cache_path.name}")
                # Store in memory cache
                SVGD._compiled_cache[memory_cache_key] = {'grad': compiled_grad}
                return compiled_grad
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed to load cache: {e}")

        # Need to compile
        if self.verbose:
            print(f"\nPrecompiling gradient function...")
            print(f"  Theta shape: {(self.theta_dim,)}, Times shape: {self.observed_data.shape}")
            if regularization > 0:
                print(f"  Moment regularization: λ={regularization}, nr_moments={nr_moments}")
            print(f"  This may take several minutes for large models...")

        # Create log_prob function using partial
        log_prob_fn = partial(
            self._log_prob_unified,
            nr_moments=nr_moments,
            sample_moments=sample_moments,
            regularization=regularization,
            rewards=rewards
        )

        # JIT compile gradient
        start = time()
        grad_fn = jax.grad(log_prob_fn)
        compiled_grad = jax.jit(grad_fn)
        # Trigger compilation with dummy call
        dummy_theta = jnp.zeros((self.theta_dim,))
        _ = compiled_grad(dummy_theta)
        if self.verbose:
            print(f"  Gradient JIT compiled in {time() - start:.1f}s")
            print(f"  Precompilation complete!")

        # Save to both caches
        SVGD._compiled_cache[memory_cache_key] = {'grad': compiled_grad}
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({'grad': compiled_grad}, f)
            if self.verbose:
                print(f"  Saved compiled gradient to cache: {cache_path.name}")
        except Exception:
            # Disk caching is best-effort
            pass

        return compiled_grad

    def optimize(self, rewards=None, return_history=True):
        """
        Run SVGD inference with optional moment-based regularization.

        Regularization settings are configured at SVGD initialization via
        regularization and nr_moments parameters.

        Parameters
        ----------
        rewards : array_like, optional
            Reward vector for reward-transformed likelihood (not yet implemented).
            Length must match number of vertices (excluding start vertex).
            Each reward serves as multiplier of vertex value in trace.
        return_history : bool, default=True
            If True, store particle positions throughout optimization

        Returns
        -------
        self
            Returns self for method chaining

        Raises
        ------
        NotImplementedError
            If rewards parameter is provided (not yet implemented)

        Examples
        --------
        >>> # Standard SVGD (no regularization)
        >>> model = Graph.pmf_and_moments_from_graph(graph)
        >>> svgd = SVGD(model, observed_data, theta_dim=1, regularization=0.0)
        >>> svgd.optimize()

        >>> # SVGD with moment regularization
        >>> model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2)
        >>> svgd = SVGD(model, observed_data, theta_dim=1, regularization=1.0, nr_moments=2)
        >>> svgd.optimize()

        >>> # With custom moments and strong regularization
        >>> svgd = SVGD(model, observed_data, theta_dim=1, regularization=5.0, nr_moments=3)
        >>> svgd.optimize()

        Notes
        -----
        - Supports all JAX transformations: jit, grad, vmap, pmap
        - Supports multi-core parallelization via parallel='vmap'/'pmap'
        - Supports multi-machine distribution via initialize_distributed()
        - Gradient compilation is cached (both memory and disk) for performance
        - All functionality from fit() and fit_regularized() is preserved
        """
        # Create kernel
        kernel = SVGDKernel(bandwidth=self.bandwidth)

        # Use regularization settings from __init__
        use_regularization = (self.regularization > 0.0 or self.use_regularization_schedule)

        # Run SVGD - split into two paths based on regularization type
        if self.use_regularization_schedule:
            # Dynamic regularization - cannot precompile gradient
            # Gradient is computed on-the-fly each iteration with current regularization
            if self.verbose:
                print(f"\nStarting SVGD inference with regularization schedule...")
                print(f"  Model: parameterized phase-type distribution")
                print(f"  Data points: {len(self.observed_data)}")
                print(f"  Prior: {'custom' if self.prior is not None else 'standard normal'}")
                print(f"  Regularization: dynamic schedule (initial λ = {self.regularization})")
                print(f"  Nr moments: {self.nr_moments}")
                print(f"  Note: Gradient precompilation disabled for schedule flexibility")

            # Create factory that captures rewards parameter
            def log_prob_factory(reg_value):
                return partial(
                    self._log_prob_unified,
                    nr_moments=self.nr_moments,
                    sample_moments=self.sample_moments,
                    regularization=reg_value,
                    rewards=rewards
                )

            results = run_svgd(
                log_prob_fn=None,  # Created dynamically per iteration
                theta_init=self.theta_init,
                n_steps=self.n_iterations,
                learning_rate=self.step_schedule,
                kernel=kernel,
                return_history=return_history,
                verbose=self.verbose,
                compiled_grad=None,  # Cannot precompile with dynamic regularization
                parallel_mode=self.parallel_mode,
                n_devices=self.n_devices,
                log_prob_fn_factory=log_prob_factory,
                regularization_schedule=self.regularization_schedule
            )
        else:
            # Static regularization - use current precompiled approach
            # Precompile gradient with caching (if JIT enabled)
            if self.jit_enabled:
                compiled_grad = self._precompile_unified(self.nr_moments, self.sample_moments, self.regularization, rewards)
            else:
                # Create log_prob function using partial (no JIT)
                log_prob_fn = partial(
                    self._log_prob_unified,
                    nr_moments=self.nr_moments,
                    sample_moments=self.sample_moments,
                    regularization=self.regularization,
                    rewards=rewards
                )
                compiled_grad = jax.grad(log_prob_fn)  # Not JIT compiled

            # Create log_prob function for run_svgd
            log_prob_fn = partial(
                self._log_prob_unified,
                nr_moments=self.nr_moments,
                sample_moments=self.sample_moments,
                regularization=self.regularization,
                rewards=rewards
            )

            # Print info
            if self.verbose:
                print(f"\nStarting SVGD inference...")
                print(f"  Model: parameterized phase-type distribution")
                print(f"  Data points: {len(self.observed_data)}")
                print(f"  Prior: {'custom' if self.prior is not None else 'standard normal'}")
                if use_regularization:
                    print(f"  Moment regularization: λ = {self.regularization}")
                    print(f"  Nr moments: {self.nr_moments}")
                else:
                    print(f"  Moment regularization: disabled")

            results = run_svgd(
                log_prob_fn=log_prob_fn,
                theta_init=self.theta_init,
                n_steps=self.n_iterations,
                learning_rate=self.step_schedule,
                kernel=kernel,
                return_history=return_history,
                verbose=self.verbose,
                compiled_grad=compiled_grad,
                parallel_mode=self.parallel_mode,
                n_devices=self.n_devices,
                log_prob_fn_factory=None,
                regularization_schedule=None
            )

        # Store results as attributes
        self.particles = results['particles']
        self.theta_mean = results['theta_mean']
        self.theta_std = results['theta_std']

        if return_history:
            self.history = results['history']
            self.history_iterations = results['history_iterations']

        self.is_fitted = True

        # Print summary with transformed values if verbose
        if self.verbose:
            print(f"\nSVGD complete!")
            transformed_results = self.get_results()
            print(f"Posterior mean: {transformed_results['theta_mean']}")
            print(f"Posterior std:  {transformed_results['theta_std']}")

        return self

    # def fit(self, return_history=True):
    #     """
    #     Run SVGD inference.

    #     Convenience wrapper for optimize(). Regularization settings are
    #     configured at SVGD initialization via regularization and nr_moments parameters.

    #     Parameters
    #     ----------
    #     return_history : bool, default=True
    #         If True, store particle positions throughout optimization

    #     Returns
    #     -------
    #     self
    #         Returns self for method chaining
    #     """
    #     return self.optimize(return_history=return_history)

    # def fit_regularized(self, observed_times=None, nr_moments=None,
    #                    regularization=None, return_history=True):
    #     """
    #     Run SVGD with moment-based regularization.

    #     .. deprecated::
    #         This method is deprecated. Configure regularization settings via
    #         SVGD(..., regularization=..., nr_moments=...) at initialization,
    #         then call fit() or optimize().

    #     Parameters
    #     ----------
    #     observed_times : array_like, optional
    #         (Ignored) Observed times are now set at SVGD initialization.
    #     nr_moments : int, optional
    #         (Ignored) Number of moments is now set at SVGD initialization.
    #     regularization : float, optional
    #         (Ignored) Regularization strength is now set at SVGD initialization.
    #     return_history : bool, default=True
    #         Whether to store particle history

    #     Returns
    #     -------
    #     self
    #         Returns self for method chaining
    #     """
    #     import warnings
    #     warnings.warn(
    #         "fit_regularized() is deprecated. Configure regularization settings at "
    #         "SVGD initialization: SVGD(..., regularization=1.0, nr_moments=2), "
    #         "then call fit() or optimize().",
    #         DeprecationWarning,
    #         stacklevel=2
    #     )

    #     # Warn if user tried to pass parameters
    #     if observed_times is not None:
    #         warnings.warn("observed_times parameter is ignored - data is set at SVGD initialization", UserWarning)
    #     if nr_moments is not None:
    #         warnings.warn("nr_moments parameter is ignored - set at SVGD initialization", UserWarning)
    #     if regularization is not None:
    #         warnings.warn("regularization parameter is ignored - set at SVGD initialization", UserWarning)

    #     return self.optimize(return_history=return_history)

    def get_results(self):
        """
        Get inference results as a dictionary.

        Returns
        -------
        dict
            Dictionary containing:
            - 'particles': Final posterior samples (in constrained space if using transformation)
            - 'theta_mean': Posterior mean (in constrained space if using transformation)
            - 'theta_std': Posterior standard deviation (in constrained space if using transformation)
            - 'history': Particle evolution (if available, in constrained space if using transformation)
            - 'particles_unconstrained': Particles in unconstrained space (only if using transformation)
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before accessing results")

        # Transform particles to constrained space if transformation is active
        if self.param_transform is not None:
            particles_constrained = jnp.array([self.param_transform(p) for p in self.particles])
            theta_mean = particles_constrained.mean(axis=0)
            theta_std = particles_constrained.std(axis=0)

            results = {
                'particles': particles_constrained,
                'theta_mean': theta_mean,
                'theta_std': theta_std,
                'particles_unconstrained': self.particles,  # Also return unconstrained
            }

            if self.history is not None:
                # Transform history as well
                history_constrained = jnp.array([[self.param_transform(p) for p in step] for step in self.history])
                results['history'] = history_constrained
                results['history_unconstrained'] = self.history
        else:
            results = {
                'particles': self.particles,
                'theta_mean': self.theta_mean,
                'theta_std': self.theta_std,
            }

            if self.history is not None:
                results['history'] = self.history

        return results

    def plot_posterior(self, true_theta=None, param_names=None, bins=20,
                      figsize=None, save_path=None, show_transformed=True):
        """
        Plot posterior distributions for each parameter.

        Parameters
        ----------
        true_theta : array_like, optional
            True parameter values (if known) to overlay on plot
        param_names : list of str, optional
            Names for each parameter dimension
        bins : int, default=20
            Number of histogram bins
        figsize : tuple, optional
            Figure size (width, height)
        save_path : str, optional
            Path to save the plot
        show_transformed : bool, default=True
            If True, show transformed (constrained) parameter values.
            If False, show untransformed (unconstrained) values.
            Only relevant when using parameter transformations.

        Returns
        -------
        fig, axes
            Matplotlib figure and axes objects
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before plotting")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        # Get appropriate particle representation
        results = self.get_results()
        if show_transformed or self.param_transform is None:
            if not show_transformed and self.param_transform is None:
                raise ValueError(
                    "show_transformed=False has no effect when no parameter transformation is used. "
                    "Either set show_transformed=True, or use positive_params=True / param_transform "
                    "to enable parameter transformation."
                )
            particles = results['particles']
            theta_mean = results['theta_mean']
            space_label = " (transformed)" if self.param_transform is not None else ""
        else:
            particles = results.get('particles_unconstrained', results['particles'])
            theta_mean = jnp.mean(particles, axis=0)
            space_label = " (untransformed)"

        n_params = self.theta_dim

        # Determine subplot layout
        if n_params == 1:
            nrows, ncols = 1, 1
            figsize = figsize or FIGSIZE
        elif n_params == 2:
            nrows, ncols = 1, 2
            figsize = figsize or (12, 4)
        else:
            ncols = min(3, n_params)
            nrows = (n_params + ncols - 1) // ncols
            figsize = figsize or (4 * ncols, 4 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for i in range(n_params):
            ax = axes[i]

            # Histogram of posterior samples
            ax.hist(particles[:, i], bins=bins, alpha=0.7, density=True,
                   edgecolor='black', label='Posterior')

            # Posterior mean
            ax.axvline(theta_mean[i], color=black_white(ax), linestyle='--',
                      linewidth=2, label=f'Mean = {theta_mean[i]:.3f}')

            # True value (if provided)
            if true_theta is not None:
                true_val = jnp.array(true_theta)[i]
                ax.axvline(true_val, color='red', linestyle='--',
                          linewidth=2, label=f'True = {true_val:.3f}')

            # Labels
            param_name = param_names[i] if param_names else f'θ_{i}'
            ax.set_xlabel(param_name + space_label)
            ax.set_ylabel('Density')
            ax.set_title(f'Posterior: {param_name}')
            ax.legend()
            # ax.grid(alpha=0.3)

        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")

        return fig, axes

    def plot_trace(self, param_names=None, figsize=FIGSIZE,
                   skip=0, max_particles=None, save_path=None, show_transformed=True,
                   ):
        """
        Plot trace plots showing particle evolution over iterations.

        Requires fit() to have been called with return_history=True.

        Parameters
        ----------
        param_names : list of str, optional
            Names for each parameter dimension
        figsize : tuple, optional
            Figure size (width, height)
        skip : int, optional
            Number of initial iterations to skip. Defaults to 0.
        max_particles : int, optional
            Max number of particles to plot. Defaults to all particles.
        save_path : str, optional
            Path to save the plot
        show_transformed : bool, default=True
            If True, show transformed (constrained) parameter values.
            If False, show untransformed (unconstrained) values.
            Only relevant when using parameter transformations.

        Returns
        -------
        fig, axes
            Matplotlib figure and axes objects
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before plotting")

        if self.history is None:
            raise RuntimeError("History not available. Call fit(return_history=True) first")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        # Get appropriate history representation
        results = self.get_results()
        if show_transformed or self.param_transform is None:
            if not show_transformed and self.param_transform is None:
                raise ValueError(
                    "show_transformed=False has no effect when no parameter transformation is used. "
                    "Either set show_transformed=True, or use positive_params=True / param_transform "
                    "to enable parameter transformation."
                )
            history = results.get('history', self.history)
            theta_mean = results['theta_mean']
            space_label = " (transformed)" if self.param_transform is not None else ""
        else:
            history = results.get('history_unconstrained', self.history)
            theta_mean = jnp.mean(history[-1], axis=0)
            space_label = " (untransformed)"

        n_params = self.theta_dim

        cols = int(n_params > 1) + 1
        rows = n_params // 2 + n_params % 2

        # Determine subplot layout
        if n_params == 1:
            figsize = figsize or FIGSIZE
        else:
            figsize = figsize or (min(14, 3.5 * cols), min(12, 2.7 * rows))

        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        # Convert history to array: (n_snapshots, n_particles, theta_dim)
        history_array = jnp.stack(history)
        n_snapshots = len(history)

        # for i in range(n_params):
            # ax = axes[i]
        for i, ax in enumerate(axes):

            if i >= n_params:
                ax.axis('off')
                continue

            # Plot each particle's trajectory
            max_plotted = self.n_particles if max_particles is None else max_particles
            for p in range(max_plotted):  # Plot first 10 particles
                y = history_array[:, p, i]
                x = np.arange(y.size)
                ax.plot(x[skip:], y[skip:], alpha=0.5, linewidth=1)

            # Plot mean trajectory
            mean_trajectory = jnp.mean(history_array[:, :, i], axis=1)
            y = mean_trajectory
            x = np.arange(y.size)

            ax.plot(x[skip:], y[skip:], color=black_white(ax), linewidth=2,
                    linestyle='dashed', label=f'Mean = {theta_mean[i]:.3f}')

            # Labels
            param_name = param_names[i] if param_names else f'θ_{i}'
            ax.set_xlabel('SVGD Iteration')
            ax.set_ylabel(param_name + space_label)
            ax.set_title(f'Trace: {param_name}')
            ax.legend()
            # ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")

        return fig, axes

    def plot_convergence(self, figsize=(7, 3), save_path=None, skip=0, show_transformed=True):
        """
        Plot convergence diagnostics showing mean and std over iterations.

        Requires fit() to have been called with return_history=True.

        Parameters
        ----------
        figsize : tuple, default=(7, 4)
            Figure size (width, height)
        save_path : str, optional
            Path to save the plot
        skip : int, optional
            Number of initial iterations to skip. Defaults to 0.
        show_transformed : bool, default=True
            If True, show transformed (constrained) parameter values.
            If False, show untransformed (unconstrained) values.
            Only relevant when using parameter transformations.

        Returns
        -------
        fig, axes
            Matplotlib figure and axes objects
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before plotting")

        if self.history is None:
            raise RuntimeError("History not available. Call fit(return_history=True) first")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        # Get appropriate history representation
        results = self.get_results()
        if show_transformed or self.param_transform is None:
            if not show_transformed and self.param_transform is None:
                raise ValueError(
                    "show_transformed=False has no effect when no parameter transformation is used. "
                    "Either set show_transformed=True, or use positive_params=True / param_transform "
                    "to enable parameter transformation."
                )
            history = results.get('history', self.history)
            space_label = " (transformed)" if self.param_transform is not None else ""
        else:
            history = results.get('history_unconstrained', self.history)
            space_label = " (untransformed)"

        # Convert history to array
        history_array = jnp.stack(history)

        # Compute mean and std at each snapshot
        mean_over_time = jnp.mean(history_array, axis=1)  # (n_snapshots, theta_dim)
        std_over_time = jnp.std(history_array, axis=1)    # (n_snapshots, theta_dim)

        # Get iteration numbers for x-axis
        iterations = self.history_iterations if self.history_iterations is not None else range(len(history))

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax1, ax2 = axes

        # Plot 1: Mean convergence
        for i in range(self.theta_dim):
            param_name = f'θ_{i}'
            x, y = iterations, mean_over_time[:, i]
            ax1.plot(x[skip:], y[skip:], label=param_name, linewidth=2)

        ax1.set_xlabel('SVGD Iteration')
        ax1.set_ylabel('Posterior Mean' + space_label)
        ax1.set_title('Mean Convergence')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Plot 2: Std convergence
        for i in range(self.theta_dim):
            param_name = f'θ_{i}'
            x, y = iterations, std_over_time[:, i]
            ax2.plot(x[skip:], y[skip:], label=param_name, linewidth=2)

        ax2.set_xlabel('SVGD Iteration')
        ax2.set_ylabel('Posterior Std' + space_label)
        ax2.set_title('Std Convergence')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")

        return fig, axes

    # ========================================================================
    # Convergence Analysis and Diagnostics
    # ========================================================================

    def _compute_particle_diversity(self, particles):
        """
        Compute particle diversity metrics.

        Parameters
        ----------
        particles : array (n_particles, theta_dim)
            Particle positions

        Returns
        -------
        dict
            'mean_distance': Mean pairwise distance
            'min_distance': Minimum pairwise distance
            'ess': Effective sample size estimate
        """
        n_particles = particles.shape[0]

        # Compute pairwise distances
        distances = jnp.array([
            jnp.linalg.norm(particles[i] - particles[j])
            for i in range(n_particles)
            for j in range(i + 1, n_particles)
        ])

        mean_dist = jnp.mean(distances)
        min_dist = jnp.min(distances)

        # Estimate ESS from particle weights (uniform weights for SVGD)
        # Use inverse participation ratio: ESS ≈ 1 / sum(w_i^2)
        # For SVGD, approximate based on particle spread
        particle_var = jnp.var(particles, axis=0)
        overall_var = jnp.mean(particle_var)

        # Rough ESS estimate: higher variance → better ESS
        # Normalize by expected variance for uniform particles
        ess_estimate = n_particles * (overall_var / (overall_var + 1e-10))

        return {
            'mean_distance': float(mean_dist),
            'min_distance': float(min_dist),
            'ess': float(ess_estimate),
            'ess_ratio': float(ess_estimate / n_particles)
        }

    def _detect_convergence_point(self, trajectory, window=50, threshold=0.01):
        """
        Detect iteration where trajectory converged.

        Parameters
        ----------
        trajectory : array (n_iterations,)
            Trajectory of mean or std over iterations
        window : int
            Window size for stability check
        threshold : float
            Relative change threshold for convergence

        Returns
        -------
        int or None
            Iteration where converged, or None if not converged
        """
        if len(trajectory) < window * 2:
            return None

        for i in range(window, len(trajectory) - window):
            # Check if trajectory is stable in window around this point
            window_vals = trajectory[i:i + window]
            mean_val = jnp.mean(window_vals)

            if abs(mean_val) < 1e-10:
                continue  # Skip if near zero

            # Compute relative variation
            rel_var = jnp.std(window_vals) / abs(mean_val)

            if rel_var < threshold:
                return i

        return None

    def _detect_variance_collapse(self, history_array):
        """
        Detect if particles collapsed to same value (variance collapse).

        Parameters
        ----------
        history_array : array (n_iterations, n_particles, theta_dim)
            Full particle history

        Returns
        -------
        dict
            'collapsed': bool
            'collapse_iteration': int or None
            'final_diversity': float
        """
        n_iterations = history_array.shape[0]

        # Check variance over time
        std_over_time = jnp.std(history_array, axis=1)  # (n_iterations, theta_dim)
        mean_std_over_time = jnp.mean(std_over_time, axis=1)  # (n_iterations,)

        # Check if std drops to near-zero
        final_std = mean_std_over_time[-1]
        max_std = jnp.max(mean_std_over_time)

        collapsed = final_std < 0.01 * max_std

        # Find when collapse happened
        collapse_iter = None
        if collapsed:
            threshold = 0.1 * max_std
            for i in range(len(mean_std_over_time)):
                if mean_std_over_time[i] < threshold:
                    collapse_iter = i
                    break

        return {
            'collapsed': bool(collapsed),
            'collapse_iteration': int(collapse_iter) if collapse_iter is not None else None,
            'final_diversity': float(final_std),
            'max_diversity': float(max_std)
        }

    def _suggest_learning_rate(self, diagnostics):
        """
        Suggest learning rate improvements based on diagnostics.

        Parameters
        ----------
        diagnostics : dict
            Diagnostics from analyze_trace

        Returns
        -------
        dict
            'recommended': schedule object or float
            'reason': str explaining suggestion
        """
        # Extract key metrics
        converged = diagnostics['converged']
        conv_point = diagnostics.get('convergence_point')
        n_iterations = diagnostics['n_iterations']
        variance_collapsed = diagnostics['variance_collapse']['collapsed']

        # Get current learning rate info
        current_schedule = self.step_schedule

        # Decision logic
        if variance_collapsed:
            return {
                'recommended': ExponentialDecayStepSize(
                    max_step=0.005, min_step=0.0005, tau=500.0
                ),
                'reason': 'Variance collapsed - reduce learning rate significantly'
            }
        elif not converged:
            # Not converged - might need more iterations or different schedule
            if isinstance(current_schedule, ConstantStepSize):
                return {
                    'recommended': ExponentialDecayStepSize(
                        max_step=current_schedule.step_size * 1.5,
                        min_step=current_schedule.step_size * 0.1,
                        tau=n_iterations * 0.5
                    ),
                    'reason': 'Not converged - use decay schedule for better convergence'
                }
            else:
                return {
                    'recommended': 'increase n_iterations',
                    'reason': 'Not converged within current iterations'
                }
        elif conv_point and conv_point < n_iterations * 0.5:
            # Converged very early - could use higher learning rate
            if isinstance(current_schedule, ConstantStepSize):
                return {
                    'recommended': current_schedule.step_size * 1.5,
                    'reason': f'Converged early (iteration {conv_point}) - could converge faster'
                }
            else:
                return {
                    'recommended': 'current schedule is good',
                    'reason': 'Converged efficiently'
                }
        else:
            return {
                'recommended': 'current learning rate is appropriate',
                'reason': 'Good convergence behavior'
            }

    def _suggest_particles(self, diagnostics):
        """
        Suggest particle count based on diagnostics.

        Parameters
        ----------
        diagnostics : dict
            Diagnostics from analyze_trace

        Returns
        -------
        dict
            'recommended': int
            'reason': str
        """
        current_n = self.n_particles
        ess_ratio = diagnostics['diversity']['ess_ratio']
        variance_collapsed = diagnostics['variance_collapse']['collapsed']

        if variance_collapsed:
            return {
                'recommended': current_n * 2,
                'reason': 'Variance collapse detected - increase particles for diversity'
            }
        elif ess_ratio < 0.5:
            return {
                'recommended': int(current_n * 1.5),
                'reason': f'Low ESS ratio ({ess_ratio:.2f}) - increase particles'
            }
        elif ess_ratio > 0.9:
            return {
                'recommended': max(20, int(current_n * 0.8)),
                'reason': f'High ESS ratio ({ess_ratio:.2f}) - could reduce particles'
            }
        else:
            return {
                'recommended': current_n,
                'reason': 'Particle count is appropriate'
            }

    def analyze_trace(self, burnin=None, verbose=True, return_dict=False):
        """
        Analyze SVGD convergence and suggest parameter improvements.

        Computes convergence diagnostics, detects issues, and recommends
        parameter updates for better performance.

        Parameters
        ----------
        burnin : int, optional
            Number of initial iterations to discard as burn-in.
            If None, auto-detects using convergence detection.
        verbose : bool, default=True
            Print detailed diagnostic report
        return_dict : bool, default=False
            Return full diagnostics dictionary

        Returns
        -------
        dict or None
            If return_dict=True, returns diagnostics dictionary with:
            - 'converged': bool - Whether SVGD converged
            - 'convergence_point': int or None - Iteration where converged
            - 'diversity': dict - Particle diversity metrics
            - 'variance_collapse': dict - Variance collapse diagnostics
            - 'suggestions': dict - Recommended parameter updates
            - 'issues': list - Detected problems

        Raises
        ------
        RuntimeError
            If fit() not called or history not available

        Examples
        --------
        >>> svgd = SVGD(model, data, theta_dim=1, n_iterations=1000)
        >>> svgd.fit(return_history=True)
        >>> svgd.analyze_trace()  # Prints diagnostic report

        >>> # Get full diagnostics
        >>> diag = svgd.analyze_trace(return_dict=True, verbose=False)
        >>> if not diag['converged']:
        >>>     print("Need more iterations!")
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before analyzing trace")

        if self.history is None:
            raise RuntimeError(
                "History not available. Call fit(return_history=True) first"
            )

        # Get history in appropriate space
        results = self.get_results()
        if self.param_transform is not None:
            history = results.get('history', self.history)
            space_label = " (transformed)"
        else:
            history = self.history
            space_label = ""

        # Convert to array
        history_array = jnp.stack(history)  # (n_iterations, n_particles, theta_dim)
        n_iterations, n_particles, theta_dim = history_array.shape

        # Compute trajectories
        mean_over_time = jnp.mean(history_array, axis=1)  # (n_iterations, theta_dim)
        std_over_time = jnp.std(history_array, axis=1)    # (n_iterations, theta_dim)

        # Average across dimensions for overall convergence
        mean_trajectory = jnp.mean(mean_over_time, axis=1)
        std_trajectory = jnp.mean(std_over_time, axis=1)

        # Detect convergence
        mean_conv_point = self._detect_convergence_point(mean_trajectory, window=50, threshold=0.01)
        std_conv_point = self._detect_convergence_point(std_trajectory, window=50, threshold=0.05)

        converged = mean_conv_point is not None

        # Auto-detect burnin if not provided
        if burnin is None:
            burnin = mean_conv_point if mean_conv_point is not None else int(n_iterations * 0.2)

        # Compute particle diversity
        final_particles = history_array[-1]
        diversity = self._compute_particle_diversity(final_particles)

        # Detect variance collapse
        variance_collapse = self._detect_variance_collapse(history_array)

        # Build diagnostics dict
        diagnostics = {
            'converged': converged,
            'convergence_point': mean_conv_point,
            'std_convergence_point': std_conv_point,
            'n_iterations': n_iterations,
            'n_particles': n_particles,
            'theta_dim': theta_dim,
            'diversity': diversity,
            'variance_collapse': variance_collapse,
            'burnin': burnin,
        }

        # Get suggestions
        lr_suggestion = self._suggest_learning_rate(diagnostics)
        particle_suggestion = self._suggest_particles(diagnostics)

        # Detect issues
        issues = []
        if variance_collapse['collapsed']:
            issues.append(f"⚠ Variance collapse at iteration {variance_collapse['collapse_iteration']}")
        if not converged:
            issues.append("⚠ Did not converge within n_iterations")
        if diversity['ess_ratio'] < 0.5:
            issues.append(f"⚠ Low effective sample size ({diversity['ess_ratio']:.1%})")
        if converged and mean_conv_point < n_iterations * 0.7:
            pct = mean_conv_point / n_iterations * 100
            issues.append(f"ℹ Converged at {pct:.1f}% of iterations - could reduce n_iterations")

        diagnostics['issues'] = issues
        diagnostics['suggestions'] = {
            'learning_rate': lr_suggestion,
            'n_particles': particle_suggestion
        }

        # Print report if verbose
        if verbose:
            self._print_analysis_report(diagnostics, space_label)

        if return_dict:
            return diagnostics

    def _print_analysis_report(self, diag, space_label=""):
        """Print formatted analysis report."""
        print("=" * 80)
        print("SVGD Convergence Analysis")
        print("=" * 80)
        print()

        # Convergence status
        if diag['converged']:
            print(f"✓ CONVERGED (iteration {diag['convergence_point']}/{diag['n_iterations']})")
            print(f"  Mean stabilized at iteration {diag['convergence_point']}")
            if diag['std_convergence_point']:
                print(f"  Std stabilized at iteration {diag['std_convergence_point']}")
        else:
            print(f"✗ NOT CONVERGED after {diag['n_iterations']} iterations")

        print()
        print("Particle Diversity:")
        div = diag['diversity']
        print(f"  Mean inter-particle distance: {div['mean_distance']:.3f}")
        print(f"  Effective sample size (ESS): {div['ess']:.1f} / {diag['n_particles']} particles ({div['ess_ratio']:.1%})")

        if div['ess_ratio'] > 0.7:
            print("  ✓ Good particle diversity")
        elif div['ess_ratio'] > 0.5:
            print("  ⚠ Moderate particle diversity")
        else:
            print("  ✗ Low particle diversity")

        if diag['variance_collapse']['collapsed']:
            print()
            print("Variance Collapse:")
            vc = diag['variance_collapse']
            print(f"  ✗ Particles collapsed at iteration {vc['collapse_iteration']}")
            print(f"  Final diversity: {vc['final_diversity']:.4f} (max was {vc['max_diversity']:.4f})")

        # Issues
        if diag['issues']:
            print()
            print("Detected Issues:")
            for issue in diag['issues']:
                print(f"  {issue}")

        # Suggestions
        print()
        print("=" * 80)
        print("Suggested Parameter Updates")
        print("=" * 80)
        print()

        print("Current Configuration:")
        print(f"  learning_rate={self.step_schedule}")
        print(f"  n_particles={self.n_particles}")
        print(f"  n_iterations={self.n_iterations}")
        print()

        # Learning rate suggestion
        lr_sug = diag['suggestions']['learning_rate']
        print(f"Learning Rate: {lr_sug['reason']}")
        if isinstance(lr_sug['recommended'], str):
            print(f"  Recommendation: {lr_sug['recommended']}")
        elif isinstance(lr_sug['recommended'], ExponentialDecayStepSize):
            sched = lr_sug['recommended']
            print(f"  Recommendation: ExponentialDecayStepSize(")
            print(f"      max_step={sched.max_step},")
            print(f"      min_step={sched.min_step},")
            print(f"      tau={sched.tau}")
            print(f"  )")
        else:
            print(f"  Recommendation: {lr_sug['recommended']}")

        print()

        # Particle suggestion
        part_sug = diag['suggestions']['n_particles']
        print(f"Particles: {part_sug['reason']}")
        if part_sug['recommended'] != self.n_particles:
            print(f"  Recommendation: n_particles={part_sug['recommended']}")
        else:
            print(f"  Recommendation: Keep n_particles={self.n_particles}")

        print()

        # Iteration suggestion
        if diag['converged'] and diag['convergence_point'] < diag['n_iterations'] * 0.8:
            suggested_iters = int(diag['convergence_point'] * 1.2)  # Add 20% buffer
            print(f"Iterations: Converged early")
            print(f"  Recommendation: Could reduce to n_iterations={suggested_iters}")
            print()
        elif not diag['converged']:
            suggested_iters = int(diag['n_iterations'] * 1.5)
            print(f"Iterations: Did not converge")
            print(f"  Recommendation: Increase to n_iterations={suggested_iters}")
            print()

        print("=" * 80)

    def plot_pairwise(self, true_theta=None, param_names=None,
                     figsize=None, save_path=None, show_transformed=True):
        """
        Plot pairwise scatter plots for all parameter pairs.

        Parameters
        ----------
        true_theta : array_like, optional
            True parameter values (if known) to overlay on plot
        param_names : list of str, optional
            Names for each parameter dimension
        figsize : tuple, optional
            Figure size (width, height)
        save_path : str, optional
            Path to save the plot
        show_transformed : bool, default=True
            If True, show transformed (constrained) parameter values.
            If False, show untransformed (unconstrained) values.
            Only relevant when using parameter transformations.

        Returns
        -------
        fig, axes
            Matplotlib figure and axes objects
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before plotting")

        if self.theta_dim < 2:
            raise ValueError("Pairwise plots require at least 2 parameters")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        # Get appropriate particle representation
        results = self.get_results()
        if show_transformed or self.param_transform is None:
            if not show_transformed and self.param_transform is None:
                raise ValueError(
                    "show_transformed=False has no effect when no parameter transformation is used. "
                    "Either set show_transformed=True, or use positive_params=True / param_transform "
                    "to enable parameter transformation."
                )
            particles = results['particles']
            space_label = " (transformed)" if self.param_transform is not None else ""
        else:
            particles = results.get('particles_unconstrained', results['particles'])
            space_label = " (untransformed)"

        n_params = self.theta_dim
        figsize = figsize or (min(14, 3 * n_params), min(12, 2.3 * n_params))

        fig, axes = plt.subplots(n_params, n_params, figsize=figsize)

        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j]

                if i == j:
                    # Diagonal: histogram
                    ax.hist(particles[:, i], bins=20, alpha=0.7,
                           edgecolor='black')
                    param_name = param_names[i] if param_names else f'θ_{i}'
                    ax.set_ylabel('Count')

                    if true_theta is not None:
                        true_val = jnp.array(true_theta)[i]
                        ax.axvline(true_val, color='red', linestyle='--', linewidth=2)
                else:
                    # Off-diagonal: scatter plot
                    ax.scatter(particles[:, j], particles[:, i],
                             alpha=0.5, s=20)

                    if true_theta is not None:
                        true_val_i = jnp.array(true_theta)[i]
                        true_val_j = jnp.array(true_theta)[j]
                        ax.scatter([true_val_j], [true_val_i], color='red',
                                 s=100, marker='x', linewidths=3)

                # Labels
                if i == n_params - 1:
                    param_name_j = param_names[j] if param_names else f'θ_{j}'
                    ax.set_xlabel(param_name_j + space_label)
                if j == 0:
                    param_name_i = param_names[i] if param_names else f'θ_{i}'
                    ax.set_ylabel(param_name_i + space_label)

                # ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")

        return fig, axes

    def _validate_animation_params(self, skip):
        """Validate common animation parameters and import dependencies."""
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before animating")

        if self.history is None:
            raise RuntimeError(
                "No history available. Call fit(return_history=True) to record particle evolution."
            )

        # Validate skip parameter
        n_iterations = len(self.history)
        if skip >= n_iterations:
            raise ValueError(
                f"skip ({skip}) must be less than number of iterations ({n_iterations})"
            )

        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            return plt, FuncAnimation
        except ImportError:
            raise ImportError(
                "matplotlib is required for animation. Install with: pip install matplotlib"
            )

    def _save_animation(self, anim, save_as_gif, save_as_mp4, interval):
        """Save animation to file if requested."""
        if save_as_gif:
            try:
                anim.save(save_as_gif, writer='pillow', fps=int(1000/interval))
                if self.verbose:
                    print(f"Animation saved as GIF: {save_as_gif}")
            except Exception as e:
                print(f"Warning: Could not save GIF: {e}")

        if save_as_mp4:
            try:
                anim.save(save_as_mp4, writer='ffmpeg', fps=int(1000/interval))
                if self.verbose:
                    print(f"Animation saved as MP4: {save_as_mp4}")
            except Exception as e:
                print(f"Warning: Could not save MP4: {e}")

    def _return_animation_html(self, anim):
        """Return animation as HTML for Jupyter display."""
        try:
            from IPython.display import HTML
            return HTML(anim.to_jshtml())
        except ImportError:
            print("Warning: IPython not available. Returning animation object.")
            return anim

    def animate(self, param_idx=0, true_theta=None, param_name=None,
                figsize=FIGSIZE, skip=0, thin=20, interval=100, bins=30,
                show_particles=True, max_particles=20,
                save_as_gif=None, save_as_mp4=None, show_transformed=True):
        """
        Create an animation showing the evolution of a single parameter's distribution.

        This method creates a side-by-side animation with:
        - Left panel: Histogram of current parameter distribution
        - Right panel: Particle trajectories over time

        Parameters
        ----------
        param_idx : int, default=0
            Index of the parameter to animate (0-indexed)
        true_theta : array_like, optional
            True parameter values. If provided, will overlay the true value for param_idx.
        param_name : str, optional
            Name for the parameter (e.g., 'jump rate'). If None, uses 'θ_{param_idx}'.
        figsize : tuple, default=(8, 6)
            Figure size (width, height)
        skip : int, default=0
            Number of initial iterations to skip in the animation
        thin : int, thin=20
            Interval of interations to plot/annimate.
        interval : int, default=100
            Delay between frames in milliseconds
        bins : int, default=30
            Number of histogram bins
        show_particles : bool, default=True
            If True, show individual particle trajectories in right panel
        max_particles : int, default=20
            Maximum number of particle trajectories to show (for clarity)
        save_as_gif : str, optional
            Path to save animation as GIF (requires pillow)
        save_as_mp4 : str, optional
            Path to save animation as MP4 (requires ffmpeg)
        show_transformed : bool, default=True
            If True, show transformed (constrained) parameter values.
            If False, show untransformed (unconstrained) values.
            Only relevant when using parameter transformations.

        Returns
        -------
        IPython.display.HTML
            Animation as HTML for Jupyter notebook display

        Examples
        --------
        >>> svgd = SVGD(model, data, theta_dim=3, n_iterations=100)
        >>> svgd.fit(return_history=True)
        >>> anim = svgd.animate(param_idx=0, true_theta=[2.0, 3.0, 2.0],
        ...                     param_name='jump rate')
        """
        plt, FuncAnimation = self._validate_animation_params(skip)

        if param_idx < 0 or param_idx >= self.theta_dim:
            raise ValueError(f"param_idx ({param_idx}) out of range [0, {self.theta_dim-1}]")

        # Get appropriate history representation
        results = self.get_results()
        if show_transformed or self.param_transform is None:
            if not show_transformed and self.param_transform is None:
                raise ValueError(
                    "show_transformed=False has no effect when no parameter transformation is used. "
                    "Either set show_transformed=True, or use positive_params=True / param_transform "
                    "to enable parameter transformation."
                )
            full_history = results.get('history', self.history)
            space_label = " (transformed)" if self.param_transform is not None else ""
        else:
            full_history = results.get('history_unconstrained', self.history)
            space_label = " (untransformed)"

        # Get history subset
        history = jnp.stack(full_history[skip::thin])
        param_history = history[:, :, param_idx]

        # Compute axis limits
        param_min = jnp.min(param_history)
        param_max = jnp.max(param_history)
        param_range = param_max - param_min
        param_lim = (param_min - 0.1 * param_range, param_max + 0.1 * param_range)

        param_name = param_name or f'θ_{param_idx}'
        true_val = jnp.array(true_theta)[param_idx] if true_theta is not None else None

        # Create figure
        fig, (ax_hist, ax_traj) = plt.subplots(1, 2, figsize=figsize)

        # Setup histogram panel
        ax_hist.set_xlim(param_lim)
        ax_hist.set_ylim(0, self.n_particles * 0.4)
        ax_hist.set_xlabel(param_name + space_label)
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('Current Distribution')
        ax_hist.grid(alpha=0.3)
        if true_val is not None:
            ax_hist.axvline(true_val, color='red', linestyle='--',
                          linewidth=2, label='True value', zorder=10)
            ax_hist.legend()

        # Setup trajectory panel
        ax_traj.set_xlim(0, len(history))
        ax_traj.set_ylim(param_lim)
        ax_traj.set_xlabel('Iteration')
        ax_traj.set_ylabel(param_name + space_label)
        ax_traj.set_title('Particle Trajectories')
        ax_traj.grid(alpha=0.3)
        if true_val is not None:
            ax_traj.axhline(true_val, color='red', linestyle='--',
                          linewidth=2, label='True value', zorder=10)

        # Initialize trajectory lines
        particle_lines = []
        if show_particles:
            n_show = min(max_particles, self.n_particles)
            for _ in range(n_show):
                line, = ax_traj.plot([], [], alpha=0.3, linewidth=1)
                particle_lines.append(line)

        mean_line, = ax_traj.plot([], [], color=black_white(ax), linewidth=2.5, label='Mean', zorder=5)
        current_marker = ax_traj.axvline(0, color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
        ax_traj.legend()

        iteration_text = fig.text(0.5, 0.98, '', ha='center', va='top')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        def init():
            for line in particle_lines:
                line.set_data([], [])
            mean_line.set_data([], [])
            iteration_text.set_text('')
            return particle_lines + [mean_line, iteration_text]

        def update(frame):
            particles_current = param_history[frame]

            # Update histogram
            ax_hist.clear()
            ax_hist.hist(particles_current, bins=bins, alpha=0.7,
                        edgecolor='black', range=param_lim)
            ax_hist.set_xlim(param_lim)
            ax_hist.set_xlabel(param_name)
            ax_hist.set_ylabel('Count')
            ax_hist.set_title('Current Distribution')
            ax_hist.grid(alpha=0.3)
            if true_val is not None:
                ax_hist.axvline(true_val, color='red', linestyle='--', linewidth=2, zorder=10)

            # Update trajectories
            iterations = jnp.arange(frame + 1)
            if show_particles:
                n_show = min(max_particles, self.n_particles)
                for p in range(n_show):
                    particle_lines[p].set_data(iterations, param_history[:frame+1, p])

            mean_trajectory = jnp.mean(param_history[:frame+1], axis=1)
            mean_line.set_data(iterations, mean_trajectory)
            current_marker.set_xdata([frame, frame])
            iteration_text.set_text(f'Iteration: {skip + frame}/{skip + len(history) - 1}')

            return particle_lines + [mean_line, iteration_text]

        anim = FuncAnimation(fig, update, frames=len(history),
                           init_func=init, blit=False, interval=interval)
        plt.close(fig)

        self._save_animation(anim, save_as_gif, save_as_mp4, interval)
        return self._return_animation_html(anim)

    def animate_pairwise(self, true_theta=None, param_names=None,
                        figsize=None, skip=0, thin=20, interval=100,
                        save_as_gif=None, save_as_mp4=None, show_transformed=True):
        """
        Create an animated pairwise scatter plot showing SVGD particle evolution.

        Parameters
        ----------
        true_theta : array_like, optional
            True parameter values (if known) to overlay as red 'x' markers
        param_names : list of str, optional
            Names for each parameter dimension (e.g., ['jump', 'flood_left', 'flood_right'])
        figsize : tuple, optional
            Figure size (width, height). Auto-sized based on parameter dimension if None.
        skip : int, default=0
            Number of initial iterations to skip in the animation
        thin : int, thin=20
            Interval of interations to plot/annimate.
        interval : int, default=100
            Delay between frames in milliseconds
        save_as_gif : str, optional
            Path to save animation as GIF (requires pillow)
        save_as_mp4 : str, optional
            Path to save animation as MP4 (requires ffmpeg)
        show_transformed : bool, default=True
            If True, show transformed (constrained) parameter values.
            If False, show untransformed (unconstrained) values.
            Only relevant when using parameter transformations.

        Returns
        -------
        IPython.display.HTML
            Animation as HTML for Jupyter notebook display

        Raises
        ------
        RuntimeError
            If fit() was not called with return_history=True
        ImportError
            If matplotlib or required animation backend is not installed

        Examples
        --------
        >>> svgd = SVGD(model, data, theta_dim=3, n_iterations=100)
        >>> svgd.fit(return_history=True)
        >>> anim = svgd.animate_pairwise(
        ...     true_theta=[2.0, 3.0, 2.0],
        ...     param_names=['jump', 'flood_left', 'flood_right'],
        ...     save_as_gif='svgd_evolution.gif'
        ... )
        """
        plt, FuncAnimation = self._validate_animation_params(skip)

        if self.theta_dim < 2:
            raise ValueError("Pairwise plots require at least 2 parameters")

        # Get appropriate history representation
        results = self.get_results()
        if show_transformed or self.param_transform is None:
            if not show_transformed and self.param_transform is None:
                raise ValueError(
                    "show_transformed=False has no effect when no parameter transformation is used. "
                    "Either set show_transformed=True, or use positive_params=True / param_transform "
                    "to enable parameter transformation."
                )
            full_history = results.get('history', self.history)
            space_label = " (transformed)" if self.param_transform is not None else ""
        else:
            full_history = results.get('history_unconstrained', self.history)
            space_label = " (untransformed)"

        n_params = self.theta_dim
        figsize = figsize or (min(14, 3 * n_params), min(12, 2.3 * n_params))

        # Get history subset
        history = full_history[skip::thin]

        # Compute global axis limits based on all history
        all_particles = jnp.concatenate(history, axis=0)
        param_mins = jnp.min(all_particles, axis=0)
        param_maxs = jnp.max(all_particles, axis=0)
        param_ranges = param_maxs - param_mins
        param_lims = [(param_mins[i] - 0.1 * param_ranges[i],
                       param_maxs[i] + 0.1 * param_ranges[i])
                      for i in range(n_params)]

        # Create figure and axes
        fig, axes = plt.subplots(n_params, n_params, figsize=figsize)

        # Initialize scatter plots and histograms
        scatter_plots = {}
        hist_data = {}

        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j]
                ax.set_xlim(param_lims[j])

                if i == j:
                    # Diagonal: histogram (will be updated each frame)
                    ax.set_ylim(0, self.n_particles * 0.3)  # Will adjust dynamically
                    param_name = param_names[i] if param_names else f'θ_{i}'
                    ax.set_ylabel('Count')

                    if true_theta is not None:
                        true_val = jnp.array(true_theta)[i]
                        ax.axvline(true_val, color='red', linestyle='--', linewidth=2, zorder=10)

                    hist_data[(i, j)] = None  # Placeholder for histogram artists
                else:
                    # Off-diagonal: scatter plot
                    ax.set_ylim(param_lims[i])
                    scatter = ax.scatter([], [], alpha=0.5, s=20)
                    scatter_plots[(i, j)] = scatter

                    if true_theta is not None:
                        true_val_i = jnp.array(true_theta)[i]
                        true_val_j = jnp.array(true_theta)[j]
                        ax.scatter([true_val_j], [true_val_i], color='red',
                                 s=100, marker='x', linewidths=3, zorder=10)

                # Labels
                if i == n_params - 1:
                    param_name_j = param_names[j] if param_names else f'θ_{j}'
                    ax.set_xlabel(param_name_j + space_label)
                if j == 0:
                    param_name_i = param_names[i] if param_names else f'θ_{i}'
                    ax.set_ylabel(param_name_i + space_label)

#                ax.grid(alpha=0.3)

        # Add iteration counter
        iteration_text = fig.text(0.5, 0.98, '', ha='center', va='top')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        def init():
            """Initialize animation."""
            for scatter in scatter_plots.values():
                scatter.set_offsets(jnp.empty((0, 2)))
            iteration_text.set_text('')
            return list(scatter_plots.values()) + [iteration_text]

        def update(frame):
            """Update function for each animation frame."""
            particles = history[frame]  # Shape: (n_particles, n_params)

            # Update scatter plots
            for (i, j), scatter in scatter_plots.items():
                scatter.set_offsets(jnp.column_stack([particles[:, j], particles[:, i]]))

            # Update histograms
            for i in range(n_params):
                ax = axes[i, i]
                ax.clear()
                ax.hist(particles[:, i], bins=20, alpha=0.7, edgecolor='black', range=param_lims[i])
                ax.set_xlim(param_lims[i])
                ax.set_ylabel('Count')

                param_name = param_names[i] if param_names else f'θ_{i}'
                if i == n_params - 1:
                    ax.set_xlabel(param_name)

                if true_theta is not None:
                    true_val = jnp.array(true_theta)[i]
                    ax.axvline(true_val, color='red', linestyle='--', linewidth=2, zorder=10)

                # ax.grid(alpha=0.3)

            # Update iteration counter
            iteration_text.set_text(f'Iteration: {skip + frame}/{skip + len(history) - 1}')

            return list(scatter_plots.values()) + [iteration_text]

        # Create animation
        anim = FuncAnimation(fig, update, frames=len(history),
                           init_func=init, blit=False, interval=interval)

        plt.close(fig)  # Prevent duplicate display in notebooks

        self._save_animation(anim, save_as_gif, save_as_mp4, interval)
        return self._return_animation_html(anim)

    def summary(self):
        """Print a summary of the inference results."""
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before getting summary")

        # Get transformed results if using parameter transformation
        results = self.get_results()
        particles = results['particles']
        theta_mean = results['theta_mean']
        theta_std = results['theta_std']

        print("=" * 70)
        print("SVGD Inference Summary")
        print("=" * 70)
        print(f"Number of particles: {self.n_particles}")
        print(f"Number of iterations: {self.n_iterations}")
        print(f"Parameter dimension: {self.theta_dim}")

        if self.param_transform is not None:
            print(f"Parameter transformation: active (positive constraint)")

        print(f"\nPosterior estimates:")
        for i in range(self.theta_dim):
            # Compute quantiles directly from particles for accurate CI
            ci_lower = jnp.percentile(particles[:, i], 2.5)
            ci_upper = jnp.percentile(particles[:, i], 97.5)

            print(f"  θ_{i}: {theta_mean[i]:.4f} ± {theta_std[i]:.4f}")
            print(f"       95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print("=" * 70)




