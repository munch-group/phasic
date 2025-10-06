import os
import platform
from time import time, sleep
import numpy as np

# environment variables for JAX must be set before running any JAX code
if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
if platform.system() == "Linux" and os.environ['SLURM_JOB_CPUS_PER_NODE']:
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.environ['SLURM_JOB_CPUS_PER_NODE']}"
import jax
print(jax.devices())
import jax.numpy as jnp
from jax import grad, vmap, jit
from jax.scipy.stats import norm
import equinox as eqx
import jax.nn as jnn
import jax.sharding as jsh
from jax.experimental import checkify
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.pjit import pjit

from functools import partial

from tqdm import trange, tqdm
trange = partial(trange, bar_format="{bar}", leave=False)
tqdm = partial(tqdm, bar_format="{bar}", leave=False)


#from jax import random, vmap, grad, jit

class LessThanOneDecoder(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, latent_dim: int, *, key):
        self.linear = eqx.nn.Linear(latent_dim, 3, key=key)

    def __call__(self, z: jnp.ndarray):
        probs = jax.nn.softmax(self.linear(z))
        return probs[0], probs[1]  # a, b ∈ (0,1), a + b < 1
    
class SumToOneDecoder(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, latent_dim: int, *, key):
        self.linear = eqx.nn.Linear(latent_dim, 1, key=key)

    def __call__(self, z: jnp.ndarray):
        s = jax.nn.sigmoid(self.linear(z)[0])
        return s, 1.0 - s

class IndependentProbDecoder(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, latent_dim: int, *, key):
        self.linear = eqx.nn.Linear(latent_dim, 2, key=key)

    def __call__(self, z: jnp.ndarray):
        logits = self.linear(z)  # shape (2,)
        a, b = jax.nn.sigmoid(logits[0]), jax.nn.sigmoid(logits[1])
        return a, b


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

# Neural network decoder for parameters
class VariableDimPTDDecoder(eqx.Module):
    """Neural network to decode latent variables to PTD parameters"""
    layers: list
    k: int
    m: int
    param_dim: int
    
    def __init__(self, key, latent_dim, k, m):
        self.k = k
        self.m = m
        self.param_dim = calculate_param_dim(k, m)
        
        # Simple MLP
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(latent_dim, 64, key=keys[0]),
            eqx.nn.Linear(64, 32, key=keys[1]), 
            eqx.nn.Linear(32, self.param_dim, key=keys[2])
        ]
    
    def __call__(self, z):
        x = z
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        x = self.layers[-1](x)
        return x

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

# Test the basic functionality
print("Testing basic SVGD functionality...")

key = jax.random.key(42)

# Generate test data for k=1, m=2
key, subkey = jax.random.split(key)
true_1d = example_ptd_spec(subkey, k=1, m=2)
data_1d = simulate_example_data(subkey, true_1d, k=1, m=2, n_samples=100)

print(f"Generated test data:")
print(f"True parameters shape: {true_1d.shape}")
print(f"Data shape: {data_1d.shape}")

# Test parameter unpacking
alpha, Q, exit_rates = unpack_theta(true_1d, k=1, m=2)
print(f"Unpacked shapes - alpha: {alpha.shape}, Q: {Q.shape}, exit_rates: {exit_rates.shape}")
print(f"Data shape: {data_1d.shape}")
# Test log probability
log_prob = log_pmf_dph(data_1d[0], true_1d, k=1, m=2)
print(f"Log probability test: {log_prob}")
print(f"Unpacked shapes - alpha: {alpha.shape}, Q: {Q.shape}, exit_rates: {exit_rates.shape}")
# Run SVGD
key, subkey = jax.random.split(key)
particles_1d, history_1d, true_1d_internal = run_variable_dim_svgd(
    subkey, data_1d, k=1, m=2, 
    n_particles=32, n_steps=50, lr=0.001
)

print(f"SVGD completed! Final particles shape: {particles_1d.shape}")






