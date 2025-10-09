import os
import platform
from time import time, sleep
import numpy as np

# environment variables for JAX must be set before running any JAX code
if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
if platform.system() == "Linux" and os.environ.get('SLURM_JOB_CPUS_PER_NODE', ''):
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.environ['SLURM_JOB_CPUS_PER_NODE']}"
import jax
# print(jax.devices())
import jax.numpy as jnp
from jax import grad, vmap, jit
from jax.scipy.stats import norm
import jax.nn as jnn
import jax.sharding as jsh
from jax.experimental import checkify
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.pjit import pjit

from functools import partial

## requires equinox dependency
# from .decoders import VariableDimPTDDecoder, LessThanOneDecoder, 
#     SumToOneDecoder, IndependentProbDecoder

from tqdm import trange, tqdm
trange = partial(trange, bar_format="{bar}", leave=False)
tqdm = partial(tqdm, bar_format="{bar}", leave=False)


#from jax import random, vmap, grad, jit




# def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
#     new_cmap = colors.LinearSegmentedColormap.from_list(
#         'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
#         cmap(np.linspace(minval, maxval, n)))
#     return new_cmap
# # "_iridis" color map (viridis without the deep purple):
# _iridis = truncate_colormap(plt.get_cmap('viridis'), 0.2, 1)


@jit
def calculate_param_dim(k, m):
    """Calculate parameter dimension for discrete phase-type distribution
    
    Parameters:
    - k: number of dimensions (absorption states)  
    - m: number of transient states
    
    Returns:
    - Total parameter dimension
    """
    # Initial distribution: m parameters (no constraint)
    alpha_dim = m
    
    # Sub-intensity matrix: m×m parameters with row-sum constraints
    # Each row sums to <= 0, so m-1 free parameters per row
    sub_Q_dim = m * (m - 1) 
    
    # Exit rates: k×m parameters (all free)
    exit_rates_dim = k * m
    
    return alpha_dim + sub_Q_dim + exit_rates_dim

def example_ptd_spec(key, k=1, m=2):
    """Generate example discrete phase-type distribution parameters
    
    Returns flattened parameter vector for the distribution with:
    - k absorption states (dimensions)
    - m transient states
    """
    # Generate initial distribution (normalized)
    key, subkey = jax.random.split(key)
    alpha_raw = jax.random.exponential(subkey, shape=(m,))
    alpha = alpha_raw / jnp.sum(alpha_raw)
    
    # Generate sub-intensity matrix Q (m×m)
    key, subkey = jax.random.split(key)
    # Off-diagonal elements (positive, will be made negative)
    off_diag = jax.random.exponential(subkey, shape=(m, m))
    off_diag = off_diag.at[jnp.diag_indices(m)].set(0)  # Zero diagonal
    
    # Make off-diagonal negative and set diagonal to ensure row sums < 0
    Q = -off_diag
    row_sums = jnp.sum(Q, axis=1)
    Q = Q.at[jnp.diag_indices(m)].set(-jnp.abs(row_sums) - 0.1)  # Ensure diagonal < row sum
    
    # Generate exit rates (k×m, all positive)
    key, subkey = jax.random.split(key)
    exit_rates = jax.random.exponential(subkey, shape=(k, m))
    
    # Flatten into parameter vector
    # Structure: [alpha (m), Q off-diagonal (m*(m-1)), exit_rates (k*m)]
    q_off_diag = jnp.concatenate([Q[i, :i].flatten() for i in range(m)] + 
                                 [Q[i, i+1:].flatten() for i in range(m)])
    
    params = jnp.concatenate([alpha, q_off_diag, exit_rates.flatten()])
    return params

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

def simulate_example_data(key, params, k, m, n_samples):
    """Simulate data from discrete phase-type distribution"""
    alpha, Q, exit_rates = unpack_theta(params, k, m)
    
    # Simple simulation - generate random absorption times
    # This is a placeholder - real DPH simulation would be more complex
    key, subkey = jax.random.split(key)
    
    # Generate samples using approximation
    # Sample from geometric distributions and combine
    samples = []
    for _ in range(n_samples):
        key, subkey = jax.random.split(key)
        # Simple approximation: sample absorption times
        absorption_times = jax.random.geometric(subkey, 0.3, shape=(k,))
        samples.append(absorption_times)
    
    return jnp.array(samples)

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

@jit
def median_heuristic(particles):
    """Median heuristic for bandwidth selection"""
    n_particles = particles.shape[0]
    distances = []
    for i in range(n_particles):
        for j in range(i+1, n_particles):
            dist = jnp.linalg.norm(particles[i] - particles[j])
            distances.append(dist)
    distances = jnp.array(distances)
    median_dist = jnp.median(distances)
    return median_dist / jnp.log(n_particles + 1)

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

@jit  
def logp_z(z, k, m):
    """Log probability function for latent variables"""
    theta = z_to_theta(z)
    # Add prior (standard normal on z)
    log_prior = -0.5 * jnp.sum(z**2)
    return log_prior

# Adaptive step size functions
@jit
def decayed_kl_target(iteration, base=0.1, decay=0.01):
    """Exponentially decaying KL target"""
    return base * jnp.exp(-decay * iteration)

@jit  
def step_size_schedule(iteration, max_step=0.001, min_step=1e-6):
    """Step size schedule"""
    decay = jnp.exp(-iteration / 1000.0)
    return max_step * decay + min_step * (1 - decay)

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

# SVGD update functions
def svgd_update_z(particles_z, data, k, m, step_size=0.001, kl_target=0.1):
    """SVGD update for latent variables"""
    n_particles = particles_z.shape[0]
    
    # Convert to parameter space for likelihood evaluation
    particles_theta = jnp.array([z_to_theta(z) for z in particles_z])
    
    # Compute log probability gradients
    def logp_single(theta):
        return logp(theta, data, k, m)
    
    grad_logp = vmap(grad(logp_single))(particles_theta)
    
    # Compute kernels
    K, grad_K = rbf_kernel_median(particles_z)
    
    # SVGD update
    phi = jnp.zeros_like(particles_z)
    for i in range(n_particles):
        # Positive term: weighted gradient
        positive_term = jnp.sum(K[i, :, None] * grad_logp, axis=0) / n_particles
        
        # Negative term: kernel gradient
        negative_term = jnp.sum(grad_K[i, :, :], axis=0) / n_particles
        
        phi = phi.at[i].set(positive_term + negative_term)
    
    # Adaptive step size
    step_factor = kl_adaptive_step(particles_z, kl_target)
    adaptive_step = step_size * step_factor
    
    return particles_z + adaptive_step * phi

# More sophisticated SVGD updates
@jit
def update_median_bw_kl_step(particles_z, k, m, kl_target=0.1, max_step=0.001):
    """SVGD update with median bandwidth and KL-adaptive step"""
    n_particles = particles_z.shape[0]
    
    # Gradients in latent space (prior only for now)
    grad_logp_z = -particles_z  # Gradient of standard normal prior
    
    # Compute kernel and its gradients
    K, grad_K = rbf_kernel_median(particles_z)
    
    # SVGD update
    phi = jnp.zeros_like(particles_z)
    for i in range(n_particles):
        positive_term = jnp.sum(K[i, :, None] * grad_logp_z, axis=0) / n_particles
        negative_term = jnp.sum(grad_K[i, :, :], axis=0) / n_particles
        phi = phi.at[i].set(positive_term + negative_term)
    
    # Adaptive step
    step_factor = kl_adaptive_step(particles_z, kl_target)
    step_size = jnp.clip(max_step * step_factor, 1e-7, max_step)
    
    return particles_z + step_size * phi

@jit
def update_local_bw_kl_step(particles_z, k, m, kl_target=0.1, max_step=0.001):
    """SVGD update with local bandwidth and KL-adaptive step"""
    n_particles = particles_z.shape[0]
    
    # Get local bandwidths
    local_bws = local_adaptive_bandwidth(particles_z)
    
    # Gradients  
    grad_logp_z = -particles_z
    
    # Compute updates with local bandwidths
    phi = jnp.zeros_like(particles_z)
    for i in range(n_particles):
        # Local kernel computations
        local_K = jnp.array([rbf_kernel(particles_z[i], particles_z[j], local_bws[i]) 
                            for j in range(n_particles)])
        
        # Local kernel gradients
        local_grad_K = jnp.array([
            -local_K[j] * (particles_z[i] - particles_z[j]) / (local_bws[i]**2)
            for j in range(n_particles)
        ])
        
        positive_term = jnp.sum(local_K[:, None] * grad_logp_z, axis=0) / n_particles
        negative_term = jnp.sum(local_grad_K, axis=0) / n_particles
        phi = phi.at[i].set(positive_term + negative_term)
    
    # Adaptive step
    step_factor = kl_adaptive_step(particles_z, kl_target)
    step_size = jnp.clip(max_step * step_factor, 1e-7, max_step)
    
    return particles_z + step_size * phi

# Distributed SVGD
def distributed_svgd_step(particles_z, k, m, kl_target=0.1, max_step=0.001):
    """Distributed SVGD step using pjit"""
    return update_median_bw_kl_step(particles_z, k, m, kl_target, max_step)

# Main SVGD function
def run_variable_dim_svgd(key, data, k, m, n_particles=40, n_steps=100, lr=0.001):
    """Run SVGD for variable-dimension discrete phase-type distributions"""
    
    # Calculate parameter dimension
    param_dim = calculate_param_dim(k, m)
    print(f"Running SVGD for k={k}, m={m} (param_dim={param_dim})")
    
    # Generate true parameters
    key, subkey = jax.random.split(key)
    true_params = example_ptd_spec(subkey, k, m)
    
    # SVGD parameters
    n_devices = min(8, n_particles)  # Don't exceed available devices
    kl_target_base = 0.1
    kl_target_decay = 0.01
    max_step = lr
    min_step = 1e-7
    max_step_scaler = 0.1
    
    if n_particles % n_devices != 0:
        n_particles = (n_particles // n_devices) * n_devices
        print(f"Adjusted n_particles to {n_particles} for even sharding")
    
    # Initial particles
    key, subkey = jax.random.split(key)
    particles_z = jax.random.normal(subkey, shape=(n_particles, param_dim))
    
    # Shard particles over devices
    devices = mesh_utils.create_device_mesh((n_devices,))
    mesh = Mesh(devices, axis_names=("i",))
    sharding = NamedSharding(mesh, P("i", None))
    particles_z = jax.device_put(particles_z, sharding)
    
    # SVGD iterations
    particle_z_history = [particles_z]
    every = max(1, n_steps // 10)  # Save every 10% of iterations
    prev = None
    
    with mesh:
        for i in trange(n_steps):
            kl_target = decayed_kl_target(i, base=kl_target_base, decay=kl_target_decay)
            particles_z = distributed_svgd_step(particles_z, k, m, kl_target=kl_target, max_step=max_step)
            
            if not i % every:
                particle_z_history.append(particles_z)
    
    # Extract final results
    particles = jnp.array([z_to_theta(z) for z in particles_z])
    
    print(f"\nResults for k={k}, m={m}:")
    print(f"True parameters shape: {true_params.shape}")
    print(f"Estimated parameters shape: {particles.shape}")
    print(f"Parameter means: {jnp.mean(particles, axis=0)}")
    print(f"True parameters: {true_params}")
    
    return particles, particle_z_history, true_params

# ==============================================================================
# Main SVGD API for external use
# ==============================================================================

class SVGDKernel:
    """RBF kernel for SVGD with automatic bandwidth selection"""

    def __init__(self, bandwidth='median'):
        """
        Parameters
        ----------
        bandwidth : str or float
            Bandwidth selection method. Options:
            - 'median': Median heuristic (default)
            - float: Fixed bandwidth value
        """
        self.bandwidth_method = bandwidth

    def compute_kernel_grad(self, particles):
        """
        Compute RBF kernel matrix and its gradient

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
        if isinstance(self.bandwidth_method, str) and self.bandwidth_method in ['median', 'rbf_median']:
            bandwidth = batch_median_heuristic(particles)
        else:
            bandwidth = self.bandwidth_method

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


def svgd_step(particles, log_prob_fn, kernel, step_size):
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

    Returns
    -------
    array (n_particles, theta_dim)
        Updated particles
    """
    n_particles = particles.shape[0]

    # Compute log probability gradients for each particle
    grad_log_p = vmap(grad(log_prob_fn))(particles)

    # Compute kernel and kernel gradient
    K, grad_K = kernel.compute_kernel_grad(particles)

    # SVGD update: phi = (K @ grad_log_p + sum(grad_K)) / n
    phi = jnp.zeros_like(particles)
    for i in range(n_particles):
        # Positive term: weighted gradient
        positive_term = jnp.sum(K[i, :, None] * grad_log_p, axis=0) / n_particles

        # Negative term: kernel gradient (repulsive force)
        negative_term = jnp.sum(grad_K[i, :, :], axis=0) / n_particles

        phi = phi.at[i].set(positive_term + negative_term)

    return particles + step_size * phi


def run_svgd(log_prob_fn, theta_init, n_steps, learning_rate=0.001,
             kernel='rbf_median', return_history=False, verbose=True):
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
    learning_rate : float
        Step size (can be constant or use adaptive schedule)
    kernel : str or SVGDKernel
        Kernel specification. If string, creates SVGDKernel with that bandwidth method
    return_history : bool
        If True, return particle positions at each iteration
    verbose : bool
        Print progress information

    Returns
    -------
    dict
        Results dictionary containing:
        - 'particles': Final particles (n_particles, theta_dim)
        - 'history': Particle history if return_history=True
        - 'theta_mean': Posterior mean
        - 'theta_std': Posterior standard deviation
    """
    # Create kernel if needed
    if isinstance(kernel, str):
        kernel_obj = SVGDKernel(bandwidth=kernel)
    else:
        kernel_obj = kernel

    # Initialize
    particles = theta_init
    history = [particles] if return_history else None

    # SVGD iterations
    if verbose:
        print(f"Running SVGD: {n_steps} steps, {len(particles)} particles")

    for step in trange(n_steps) if verbose else range(n_steps):
        # Perform SVGD update
        particles = svgd_step(particles, log_prob_fn, kernel_obj, learning_rate)

        # Store history
        if return_history and (step % max(1, n_steps // 20) == 0):
            history.append(particles)

    # Final history
    if return_history:
        history.append(particles)

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

    if verbose:
        print(f"\nSVGD complete!")
        print(f"Posterior mean: {theta_mean}")
        print(f"Posterior std:  {theta_std}")

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
    n_particles : int, default=50
        Number of SVGD particles
    n_iterations : int, default=1000
        Number of SVGD optimization steps
    learning_rate : float, default=0.001
        SVGD step size
    kernel : str, default='rbf_median'
        Kernel bandwidth selection method
    theta_init : array_like, optional
        Initial particle positions (n_particles, theta_dim)
    theta_dim : int, optional
        Dimension of theta parameter vector (required if theta_init is None)
    seed : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Print progress information

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
    >>> # Build parameterized model
    >>> graph = Graph(callback=coalescent, parameterized=True, nr_samples=3)
    >>> model = Graph.pmf_from_graph(graph)
    >>>
    >>> # Create SVGD object and fit
    >>> svgd = SVGD(model, observed_data, theta_dim=1)
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

    def __init__(self, model, observed_data, prior=None, n_particles=50,
                 n_iterations=1000, learning_rate=0.001, kernel='rbf_median',
                 theta_init=None, theta_dim=None, seed=42, verbose=True):

        self.model = model
        self.observed_data = jnp.array(observed_data)
        self.prior = prior
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.kernel_str = kernel
        self.theta_dim = theta_dim
        self.seed = seed
        self.verbose = verbose

        # Validate and initialize particles
        if theta_init is None and theta_dim is None:
            raise ValueError(
                "Either theta_init or theta_dim must be provided. "
                "If you don't have initial particles, specify theta_dim (the number of parameters)."
            )

        # Initialize particles
        key = jax.random.PRNGKey(seed)
        if theta_init is None:
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

        # Detect model type: does it return (pmf, moments) or just pmf?
        self.model_returns_moments = False
        try:
            test_theta = self.theta_init[0]
            test_times = self.observed_data[:min(2, len(self.observed_data))]
            result = self.model(test_theta, test_times)
            if isinstance(result, tuple) and len(result) == 2:
                # Model returns (pmf, moments)
                self.model_returns_moments = True
                if verbose:
                    print("Detected model type: returns (pmf, moments)")
            else:
                if verbose:
                    print("Detected model type: returns pmf only")
        except Exception as e:
            # If detection fails, assume pmf only
            if verbose:
                print(f"Model type detection failed (assuming pmf only): {e}")
            pass

        # Results (initialized after fit())
        self.particles = None
        self.theta_mean = None
        self.theta_std = None
        self.history = None
        self.is_fitted = False

    def _log_prob(self, theta):
        """
        Log probability function: log p(data|theta) + log p(theta)

        Parameters
        ----------
        theta : array
            Parameter vector

        Returns
        -------
        scalar
            Log probability
        """
        # Log-likelihood
        try:
            result = self.model(theta, self.observed_data)
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

        # Log-prior
        if self.prior is not None:
            log_pri = self.prior(theta)
        else:
            # Default: standard normal prior
            log_pri = -0.5 * jnp.sum(theta**2)

        return log_lik + log_pri

    def fit(self, return_history=False):
        """
        Run SVGD inference to approximate the posterior distribution.

        Parameters
        ----------
        return_history : bool, default=False
            If True, store particle positions throughout optimization

        Returns
        -------
        self
            Returns self for method chaining
        """
        # Create kernel
        kernel_obj = SVGDKernel(bandwidth=self.kernel_str)

        # Run SVGD
        if self.verbose:
            print(f"\nStarting SVGD inference...")
            print(f"  Model: parameterized phase-type distribution")
            print(f"  Data points: {len(self.observed_data)}")
            print(f"  Prior: {'custom' if self.prior is not None else 'standard normal'}")

        results = run_svgd(
            log_prob_fn=self._log_prob,
            theta_init=self.theta_init,
            n_steps=self.n_iterations,
            learning_rate=self.learning_rate,
            kernel=kernel_obj,
            return_history=return_history,
            verbose=self.verbose
        )

        # Store results as attributes
        self.particles = results['particles']
        self.theta_mean = results['theta_mean']
        self.theta_std = results['theta_std']

        if return_history:
            self.history = results['history']

        self.is_fitted = True

        return self

    def fit_regularized(self, observed_times=None, nr_moments=2,
                       regularization=1.0, return_history=False):
        """
        Run SVGD with moment-based regularization.

        Adds regularization term that penalizes difference between model moments
        and sample moments, improving stability and convergence.

        The regularized objective is:
            log p(theta | data) = log p(data|theta) + log p(theta) - λ * Σ_k (E[T^k|theta] - mean(data^k))^2

        Parameters
        ----------
        observed_times : array_like, optional
            Actual observed data points (waiting times, not PMF values).
            Used for computing sample moments.
            If None, uses self.observed_data (assumes it contains times, not PMF values).
        nr_moments : int, default=2
            Number of moments to use for regularization.
            Higher moments provide stronger constraints but may be less stable.
            Example: nr_moments=2 uses E[T] and E[T^2]
        regularization : float, default=1.0
            Strength of moment regularization (λ in objective).
            - 0.0: No regularization (equivalent to standard SVGD)
            - 0.1-1.0: Mild regularization
            - 1.0-10.0: Strong regularization
            Higher values enforce moment matching more strongly.
        return_history : bool, default=False
            Whether to store particle history

        Returns
        -------
        self
            Returns self for method chaining

        Raises
        ------
        ValueError
            If model doesn't support moments (wasn't created with pmf_and_moments_from_graph)
        ValueError
            If observed_times is None and cannot determine sample moments

        Examples
        --------
        >>> # Create parameterized model with moments
        >>> graph = Graph(callback=coalescent, parameterized=True, nr_samples=4)
        >>> model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2)
        >>>
        >>> # Generate observed data
        >>> true_theta = jnp.array([0.8])
        >>> observed_times = jnp.array([0.5, 1.2, 0.8, 1.5, 2.0])
        >>> observed_pmf = model(true_theta, observed_times)[0]  # Extract PMF values
        >>>
        >>> # Run regularized SVGD
        >>> svgd = SVGD(model, observed_pmf, theta_dim=1)
        >>> svgd.fit_regularized(observed_times=observed_times, nr_moments=2, regularization=1.0)
        >>>
        >>> # Access results
        >>> print(f"Posterior mean: {svgd.theta_mean}")
        >>> print(f"Posterior std: {svgd.theta_std}")

        Notes
        -----
        - Requires model created with Graph.pmf_and_moments_from_graph()
        - The regularization term stabilizes inference by matching distribution moments
        - Particularly useful when observed data is sparse or noisy
        - Start with regularization=1.0 and adjust based on performance
        """
        # Validate that model supports moments
        if not self.model_returns_moments:
            raise ValueError(
                "Model must return both PMF and moments for regularized SVGD. "
                "Use Graph.pmf_and_moments_from_graph() to create model, not Graph.pmf_from_graph()."
            )

        # Determine observed times
        if observed_times is None:
            # Try to use self.observed_data as times
            observed_times = self.observed_data
            if self.verbose:
                print("Using self.observed_data as observed times for sample moment computation")
        else:
            observed_times = jnp.array(observed_times)

        # Compute sample moments from observed data
        sample_moments = compute_sample_moments(observed_times, nr_moments)
        if self.verbose:
            print(f"Sample moments from data: {sample_moments}")

        # Define regularized log-probability function
        def log_prob_regularized(theta):
            """
            Regularized log probability with moment matching term.

            log p(theta | data, moments) = log p(data|theta) + log p(theta) - λ * ||E[T^k|theta] - sample_moments||^2
            """
            try:
                result = self.model(theta, self.observed_data)
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

            # Log-prior term
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

        # Create kernel
        kernel_obj = SVGDKernel(bandwidth=self.kernel_str)

        # Run SVGD with regularized objective
        if self.verbose:
            print(f"\nStarting regularized SVGD inference...")
            print(f"  Model: parameterized phase-type distribution")
            print(f"  Data points: {len(self.observed_data)}")
            print(f"  Prior: {'custom' if self.prior is not None else 'standard normal'}")
            print(f"  Moment regularization: λ = {regularization}")
            print(f"  Nr moments: {nr_moments}")

        results = run_svgd(
            log_prob_fn=log_prob_regularized,
            theta_init=self.theta_init,
            n_steps=self.n_iterations,
            learning_rate=self.learning_rate,
            kernel=kernel_obj,
            return_history=return_history,
            verbose=self.verbose
        )

        # Store results as attributes
        self.particles = results['particles']
        self.theta_mean = results['theta_mean']
        self.theta_std = results['theta_std']

        if return_history:
            self.history = results['history']

        self.is_fitted = True

        # Store regularization info
        self.regularization = regularization
        self.nr_moments = nr_moments
        self.sample_moments = sample_moments

        return self

    def get_results(self):
        """
        Get inference results as a dictionary.

        Returns
        -------
        dict
            Dictionary containing:
            - 'particles': Final posterior samples
            - 'theta_mean': Posterior mean
            - 'theta_std': Posterior standard deviation
            - 'history': Particle evolution (if available)
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before accessing results")

        results = {
            'particles': self.particles,
            'theta_mean': self.theta_mean,
            'theta_std': self.theta_std,
        }

        if self.history is not None:
            results['history'] = self.history

        return results

    def plot_posterior(self, true_theta=None, param_names=None, bins=20,
                      figsize=None, save_path=None):
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

        n_params = self.theta_dim

        # Determine subplot layout
        if n_params == 1:
            nrows, ncols = 1, 1
            figsize = figsize or (6, 4)
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
            ax.hist(self.particles[:, i], bins=bins, alpha=0.7, density=True,
                   edgecolor='black', label='Posterior')

            # Posterior mean
            ax.axvline(self.theta_mean[i], color='blue', linestyle='--',
                      linewidth=2, label=f'Mean = {self.theta_mean[i]:.3f}')

            # True value (if provided)
            if true_theta is not None:
                true_val = jnp.array(true_theta)[i]
                ax.axvline(true_val, color='red', linestyle='--',
                          linewidth=2, label=f'True = {true_val:.3f}')

            # Labels
            param_name = param_names[i] if param_names else f'θ_{i}'
            ax.set_xlabel(param_name, fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title(f'Posterior: {param_name}', fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)

        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")

        return fig, axes

    def plot_trace(self, param_names=None, figsize=None, save_path=None):
        """
        Plot trace plots showing particle evolution over iterations.

        Requires fit() to have been called with return_history=True.

        Parameters
        ----------
        param_names : list of str, optional
            Names for each parameter dimension
        figsize : tuple, optional
            Figure size (width, height)
        save_path : str, optional
            Path to save the plot

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

        n_params = self.theta_dim

        # Determine subplot layout
        if n_params == 1:
            figsize = figsize or (10, 4)
        else:
            figsize = figsize or (10, 3 * n_params)

        fig, axes = plt.subplots(n_params, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        # Convert history to array: (n_snapshots, n_particles, theta_dim)
        history_array = jnp.stack(self.history)
        n_snapshots = len(self.history)

        for i in range(n_params):
            ax = axes[i]

            # Plot each particle's trajectory
            for p in range(min(10, self.n_particles)):  # Plot first 10 particles
                ax.plot(history_array[:, p, i], alpha=0.3, linewidth=1)

            # Plot mean trajectory
            mean_trajectory = jnp.mean(history_array[:, :, i], axis=1)
            ax.plot(mean_trajectory, color='red', linewidth=2,
                   label=f'Mean = {self.theta_mean[i]:.3f}')

            # Labels
            param_name = param_names[i] if param_names else f'θ_{i}'
            ax.set_xlabel('SVGD Iteration', fontsize=12)
            ax.set_ylabel(param_name, fontsize=12)
            ax.set_title(f'Trace: {param_name}', fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")

        return fig, axes

    def plot_convergence(self, figsize=(10, 4), save_path=None):
        """
        Plot convergence diagnostics showing mean and std over iterations.

        Requires fit() to have been called with return_history=True.

        Parameters
        ----------
        figsize : tuple, default=(10, 4)
            Figure size (width, height)
        save_path : str, optional
            Path to save the plot

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

        # Convert history to array
        history_array = jnp.stack(self.history)

        # Compute mean and std at each snapshot
        mean_over_time = jnp.mean(history_array, axis=1)  # (n_snapshots, theta_dim)
        std_over_time = jnp.std(history_array, axis=1)    # (n_snapshots, theta_dim)

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax1, ax2 = axes

        # Plot 1: Mean convergence
        for i in range(self.theta_dim):
            param_name = f'θ_{i}'
            ax1.plot(mean_over_time[:, i], label=param_name, linewidth=2)

        ax1.set_xlabel('SVGD Iteration', fontsize=12)
        ax1.set_ylabel('Posterior Mean', fontsize=12)
        ax1.set_title('Mean Convergence', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Plot 2: Std convergence
        for i in range(self.theta_dim):
            param_name = f'θ_{i}'
            ax2.plot(std_over_time[:, i], label=param_name, linewidth=2)

        ax2.set_xlabel('SVGD Iteration', fontsize=12)
        ax2.set_ylabel('Posterior Std', fontsize=12)
        ax2.set_title('Std Convergence', fontsize=14)
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")

        return fig, axes

    def plot_pairwise(self, true_theta=None, param_names=None,
                     figsize=None, save_path=None):
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

        n_params = self.theta_dim
        figsize = figsize or (3 * n_params, 3 * n_params)

        fig, axes = plt.subplots(n_params, n_params, figsize=figsize)

        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j]

                if i == j:
                    # Diagonal: histogram
                    ax.hist(self.particles[:, i], bins=20, alpha=0.7,
                           edgecolor='black')
                    param_name = param_names[i] if param_names else f'θ_{i}'
                    ax.set_ylabel('Count')

                    if true_theta is not None:
                        true_val = jnp.array(true_theta)[i]
                        ax.axvline(true_val, color='red', linestyle='--', linewidth=2)
                else:
                    # Off-diagonal: scatter plot
                    ax.scatter(self.particles[:, j], self.particles[:, i],
                             alpha=0.5, s=20)

                    if true_theta is not None:
                        true_val_i = jnp.array(true_theta)[i]
                        true_val_j = jnp.array(true_theta)[j]
                        ax.scatter([true_val_j], [true_val_i], color='red',
                                 s=100, marker='x', linewidths=3)

                # Labels
                if i == n_params - 1:
                    param_name_j = param_names[j] if param_names else f'θ_{j}'
                    ax.set_xlabel(param_name_j)
                if j == 0:
                    param_name_i = param_names[i] if param_names else f'θ_{i}'
                    ax.set_ylabel(param_name_i)

                ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")

        return fig, axes

    def summary(self):
        """Print a summary of the inference results."""
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before getting summary")

        print("=" * 70)
        print("SVGD Inference Summary")
        print("=" * 70)
        print(f"Number of particles: {self.n_particles}")
        print(f"Number of iterations: {self.n_iterations}")
        print(f"Parameter dimension: {self.theta_dim}")
        print(f"\nPosterior estimates:")
        for i in range(self.theta_dim):
            print(f"  θ_{i}: {self.theta_mean[i]:.4f} ± {self.theta_std[i]:.4f}")
            print(f"       95% CI: [{self.theta_mean[i] - 1.96*self.theta_std[i]:.4f}, "
                  f"{self.theta_mean[i] + 1.96*self.theta_std[i]:.4f}]")
        print("=" * 70)




