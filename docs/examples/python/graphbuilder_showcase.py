#!/usr/bin/env python3
"""
Comprehensive GraphBuilder & JAX FFI Showcase

This script demonstrates all features of the new parameterized GraphBuilder
and JAX FFI integration for efficient phase-type distribution computation.

Features demonstrated:
1. Direct GraphBuilder usage (pybind11)
2. JAX FFI wrappers with pure_callback
3. JIT compilation
4. vmap batching (for multi-parameter inference)
5. Combined PMF + moments (for SVGD with regularization)
6. Real-world SVGD-like workflow
7. Performance comparison
8. Multi-CPU and multi-device parallelization with pmap
9. Summary and key takeaways

Requirements:
- JAX with x64 enabled
- GraphBuilder module compiled with parameterized support
- Multiple CPU devices for optimal pmap performance
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
import json
import time
from typing import Dict, Any

# Enable 64-bit types in JAX (required for C++ float64 compatibility)
config.update('jax_enable_x64', True)

print("=" * 80)
print("GraphBuilder & JAX FFI Comprehensive Showcase")
print("=" * 80)
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"JAX x64 enabled: {config.jax_enable_x64}")
print()


# ============================================================================
# Part 1: Create a Parameterized Phase-Type Distribution
# ============================================================================

def create_erlang_distribution(num_stages: int = 3) -> Dict[str, Any]:
    """
    Create a parameterized Erlang distribution.

    Erlang(n, λ) = sum of n independent Exp(λ) random variables

    Structure:
    - start -> s0 -> s1 -> ... -> s_{n-1} -> absorbing
    - Each transition has rate λ (parameterized by theta[0])

    Parameters
    ----------
    num_stages : int
        Number of exponential stages

    Returns
    -------
    dict
        JSON-serializable dictionary representing the graph structure
    """
    # State vectors: Each stage needs a unique state value
    # - Index 0: start state [0] (not absorbing)
    # - Indices 1 to num_stages: intermediate stages [2], [3], ..., [num_stages+1] (not absorbing, unique values)
    # - Index num_stages+1: final absorbing state [1]
    states = [[0]] + [[i+1] for i in range(1, num_stages + 1)] + [[1]]

    # Regular edges: none (all edges are parameterized)
    edges = []

    # Start edge: start -> s0 with weight 1.0
    start_edges = [[1, 1.0]]  # to state index 1 (s0)

    # Parameterized edges: s_i -> s_{i+1} with rate = theta[0]
    param_edges = []
    for i in range(1, num_stages + 1):
        param_edges.append([i, i+1, 1.0])  # from stage i to stage i+1

    # Final transition from s_{num_stages} to absorbing is included in the loop above

    # No parameterized start edges
    start_param_edges = []

    return {
        "states": states,
        "edges": edges,
        "start_edges": start_edges,
        "param_edges": param_edges,
        "start_param_edges": start_param_edges,
        "param_length": 1,
        "state_length": 1,
        "n_vertices": len(states)
    }


print("=" * 80)
print("Part 1: Creating Parameterized Erlang Distribution")
print("=" * 80)

erlang_structure = create_erlang_distribution(num_stages=3)
erlang_json = json.dumps(erlang_structure)

print(f"Distribution: Erlang(n=3, λ=theta[0])")
print(f"Number of states: {erlang_structure['n_vertices']}")
print(f"Number of parameters: {erlang_structure['param_length']}")
print(f"Structure JSON length: {len(erlang_json)} bytes")
print()


# ============================================================================
# Part 2: Direct GraphBuilder Usage (pybind11)
# ============================================================================

print("=" * 80)
print("Part 2: Direct GraphBuilder Usage (pybind11)")
print("=" * 80)

from ptdalgorithms.ptdalgorithmscpp_pybind import parameterized

# Create GraphBuilder
builder = parameterized.GraphBuilder(erlang_json)

print(f"GraphBuilder created:")
print(f"  - param_length: {builder.param_length}")
print(f"  - state_length: {builder.state_length}")
print(f"  - vertices_length: {builder.vertices_length}")
print()

# Compute PMF for different rate parameters
theta_values = [0.5, 1.0, 2.0]
times = np.linspace(0.1, 10.0, 100)

print("Computing PDF for different rate parameters...")
for theta_val in theta_values:
    theta = np.array([theta_val])

    # Time the computation
    start_time = time.time()
    pdf = builder.compute_pmf(theta, times, discrete=False, granularity=100)
    elapsed = (time.time() - start_time) * 1000

    # Find peak
    peak_idx = np.argmax(pdf)
    peak_time = times[peak_idx]
    peak_value = pdf[peak_idx]

    print(f"  λ={theta_val}: peak at t={peak_time:.2f} "
          f"(PDF={peak_value:.4f}), time={elapsed:.2f}ms")
print()

# Compute moments
print("Computing moments...")
theta = np.array([1.0])
moments = builder.compute_moments(theta, nr_moments=4)

print(f"  E[T]   = {moments[0]:.4f} (expected: {3/1.0:.4f})")
print(f"  E[T²]  = {moments[1]:.4f}")
print(f"  E[T³]  = {moments[2]:.4f}")
print(f"  E[T⁴]  = {moments[3]:.4f}")

# Compute mean and variance
mean = moments[0]
variance = moments[1] - moments[0]**2
print(f"  Mean = {mean:.4f}, Variance = {variance:.4f}")
print()


# ============================================================================
# Part 3: JAX FFI Wrappers (pure_callback)
# ============================================================================

print("=" * 80)
print("Part 3: JAX FFI Wrappers with pure_callback")
print("=" * 80)

from ptdalgorithms.ffi_wrappers import (
    compute_pmf_ffi,
    compute_moments_ffi,
    compute_pmf_and_moments_ffi,
)

# Convert to JAX arrays
theta_jax = jnp.array([1.0])
times_jax = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

print("Computing PDF with JAX FFI wrapper...")
pdf_jax = compute_pmf_ffi(erlang_json, theta_jax, times_jax, discrete=False)
print(f"  Times: {times_jax}")
print(f"  PDF:   {pdf_jax}")
print()

print("Computing moments with JAX FFI wrapper...")
moments_jax = compute_moments_ffi(erlang_json, theta_jax, nr_moments=3)
print(f"  Moments: {moments_jax}")
print()

print("Computing PMF + moments together (efficient for SVGD)...")
pdf_combined, moments_combined = compute_pmf_and_moments_ffi(
    erlang_json, theta_jax, times_jax, nr_moments=3, discrete=False
)
print(f"  PDF:     {pdf_combined}")
print(f"  Moments: {moments_combined}")
print()


# ============================================================================
# Part 4: JIT Compilation
# ============================================================================

print("=" * 80)
print("Part 4: JIT Compilation")
print("=" * 80)

# Create JIT-compiled version
compute_pmf_jit = jax.jit(compute_pmf_ffi, static_argnums=(0, 3, 4))

print("First call (compilation + execution)...")
start_time = time.time()
pdf_jit_1 = compute_pmf_jit(erlang_json, theta_jax, times_jax, False, 100)
time_1 = (time.time() - start_time) * 1000
print(f"  Time: {time_1:.2f}ms")
print(f"  Result: {pdf_jit_1}")
print()

print("Second call (cached, should be faster)...")
start_time = time.time()
pdf_jit_2 = compute_pmf_jit(erlang_json, theta_jax, times_jax, False, 100)
time_2 = (time.time() - start_time) * 1000
print(f"  Time: {time_2:.2f}ms")
print(f"  Result: {pdf_jit_2}")
print(f"  Speedup: {time_1/time_2:.1f}x")
print(f"  Results match: {jnp.allclose(pdf_jit_1, pdf_jit_2)}")
print()


# ============================================================================
# Part 5: vmap Batching (Multi-Parameter Inference)
# ============================================================================

print("=" * 80)
print("Part 5: vmap Batching for Multi-Parameter Inference")
print("=" * 80)

# Simulate SVGD particles (multiple rate parameters)
num_particles = 10
theta_batch = jnp.linspace(0.5, 2.0, num_particles).reshape(-1, 1)

print(f"Simulating SVGD with {num_particles} particles...")
print(f"Parameter values (λ): {theta_batch.flatten()[:5]}... {theta_batch.flatten()[-1]}")
print()

# Define function to vmap over
def compute_pdf_for_particle(theta):
    return compute_pmf_ffi(erlang_json, theta, times_jax, discrete=False)

# Apply vmap
print("Computing PDF for all particles with vmap...")
vmap_compute_pdf = jax.vmap(compute_pdf_for_particle)

start_time = time.time()
pdf_batch = vmap_compute_pdf(theta_batch)
elapsed = (time.time() - start_time) * 1000

print(f"  Batch computation time: {elapsed:.2f}ms")
print(f"  Result shape: {pdf_batch.shape} (particles × time_points)")
print(f"  First particle PDF: {pdf_batch[0]}")
print(f"  Last particle PDF:  {pdf_batch[-1]}")
print()

# Visualize batch statistics
pdf_mean = jnp.mean(pdf_batch, axis=0)
pdf_std = jnp.std(pdf_batch, axis=0)
print(f"Batch statistics across {num_particles} particles:")
print(f"  Mean PDF: {pdf_mean}")
print(f"  Std PDF:  {pdf_std}")
print()


# ============================================================================
# Part 6: SVGD-like Workflow (PMF + Moments)
# ============================================================================

print("=" * 80)
print("Part 6: SVGD-like Workflow with Moment Regularization")
print("=" * 80)

# Target moments (what we want to match)
target_moments = jnp.array([3.0, 12.0, 60.0])  # E[T]=3, E[T²]=12, E[T³]=60
print(f"Target moments: {target_moments}")
print()

def svgd_objective(theta, observations, target_moments_local):
    """
    SVGD objective: log-likelihood + moment matching penalty.

    This demonstrates the efficient combined computation.
    """
    # Compute PMF and moments in one call (efficient!)
    pdf, moments = compute_pmf_and_moments_ffi(
        erlang_json, theta, observations, nr_moments=3, discrete=False
    )

    # Log-likelihood (avoid log(0))
    log_likelihood = jnp.sum(jnp.log(pdf + 1e-10))

    # Moment matching penalty
    moment_penalty = jnp.sum((moments - target_moments_local)**2)

    # Combined objective
    return log_likelihood - 0.1 * moment_penalty

# Evaluate objective for different parameters
print("Evaluating SVGD objective for different parameters...")
observations = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

for theta_val in [0.5, 1.0, 1.5, 2.0]:
    theta_test = jnp.array([theta_val])
    objective_val = svgd_objective(theta_test, observations, target_moments)
    print(f"  λ={theta_val}: objective={objective_val:.4f}")
print()

# Batch evaluation with vmap
print(f"Batch evaluation for {num_particles} particles with vmap...")
def eval_objective(theta):
    return svgd_objective(theta, observations, target_moments)

vmap_eval = jax.vmap(eval_objective)
start_time = time.time()
objectives_batch = vmap_eval(theta_batch)
elapsed = (time.time() - start_time) * 1000

print(f"  Time: {elapsed:.2f}ms")
print(f"  Objectives: {objectives_batch}")
print(f"  Best particle: λ={theta_batch[jnp.argmax(objectives_batch)][0]:.4f}")
print()


# ============================================================================
# Part 7: Performance Comparison
# ============================================================================

print("=" * 80)
print("Part 7: Performance Comparison")
print("=" * 80)

# Compare different approaches
num_iterations = 100
theta_test = np.array([1.0])
times_test = np.linspace(0.1, 5.0, 50)

print(f"Running {num_iterations} iterations...")
print()

# 1. Direct pybind11 (numpy)
print("1. Direct pybind11 (numpy):")
start_time = time.time()
for _ in range(num_iterations):
    pdf_direct = builder.compute_pmf(theta_test, times_test, discrete=False)
time_direct = (time.time() - start_time) * 1000
print(f"   Total: {time_direct:.2f}ms, Per iteration: {time_direct/num_iterations:.3f}ms")

# 2. JAX FFI wrapper (pure_callback, no JIT)
print("2. JAX FFI wrapper (no JIT):")
theta_jax_test = jnp.array([1.0])
times_jax_test = jnp.array(times_test)
start_time = time.time()
for _ in range(num_iterations):
    pdf_ffi = compute_pmf_ffi(erlang_json, theta_jax_test, times_jax_test, discrete=False)
time_ffi = (time.time() - start_time) * 1000
print(f"   Total: {time_ffi:.2f}ms, Per iteration: {time_ffi/num_iterations:.3f}ms")

# 3. JAX FFI wrapper (with JIT)
print("3. JAX FFI wrapper (with JIT):")
compute_pmf_jit_test = jax.jit(compute_pmf_ffi, static_argnums=(0, 3, 4))
# Warm-up
_ = compute_pmf_jit_test(erlang_json, theta_jax_test, times_jax_test, False, 100)
start_time = time.time()
for _ in range(num_iterations):
    pdf_jit_test = compute_pmf_jit_test(erlang_json, theta_jax_test, times_jax_test, False, 100)
time_jit = (time.time() - start_time) * 1000
print(f"   Total: {time_jit:.2f}ms, Per iteration: {time_jit/num_iterations:.3f}ms")

# 4. Batch computation (vmap)
print(f"4. Batch computation (vmap, {num_particles} particles):")
def batch_compute(theta_single):
    return compute_pmf_ffi(erlang_json, theta_single, times_jax_test, discrete=False)
vmap_batch_compute = jax.vmap(batch_compute)
theta_batch_test = jnp.linspace(0.5, 2.0, num_particles).reshape(-1, 1)
# Warm-up
_ = vmap_batch_compute(theta_batch_test)
start_time = time.time()
for _ in range(num_iterations // num_particles):  # Fair comparison
    pdf_batch_test = vmap_batch_compute(theta_batch_test)
time_batch = (time.time() - start_time) * 1000
time_batch_per_particle = time_batch / (num_iterations // num_particles) / num_particles
print(f"   Total: {time_batch:.2f}ms, Per particle: {time_batch_per_particle:.3f}ms")

print()
print("Summary:")
print(f"  Direct pybind11:    {time_direct/num_iterations:.3f}ms per call (baseline)")
print(f"  FFI no JIT:         {time_ffi/num_iterations:.3f}ms per call ({time_direct/time_ffi:.2f}x)")
print(f"  FFI with JIT:       {time_jit/num_iterations:.3f}ms per call ({time_direct/time_jit:.2f}x)")
print(f"  Batch (vmap):       {time_batch_per_particle:.3f}ms per call ({time_direct/(time_batch_per_particle*num_iterations//num_particles):.2f}x)")
print()


# ============================================================================
# Part 8: Multi-CPU and Multi-Device Parallelization
# ============================================================================

print("=" * 80)
print("Part 8: Multi-CPU and Multi-Device Parallelization")
print("=" * 80)

# Check available devices
cpu_devices = jax.devices('cpu')
num_cpus = len(cpu_devices)
print(f"Available CPU devices: {num_cpus}")
print(f"Devices: {cpu_devices[:min(4, num_cpus)]}")  # Show first 4
print()

# ============================================================================
# 8.1: pmap - Parallel Map Across Devices
# ============================================================================

print("--- 8.1: pmap (Parallel Map) vs vmap (Sequential) ---")
print()

# Create a batch of particles that divides evenly across available CPUs
# For pmap, batch size should be divisible by number of devices
num_pmap_particles = num_cpus * 4  # 4 particles per CPU
theta_pmap_batch = jnp.linspace(0.5, 2.0, num_pmap_particles).reshape(-1, 1)
times_pmap = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

print(f"Testing with {num_pmap_particles} particles across {num_cpus} CPU devices")
print(f"  Particles per device: {num_pmap_particles // num_cpus}")
print()

# Define computation function for a single particle
def compute_pdf_single(theta):
    """Compute PDF for a single parameter value."""
    return compute_pmf_ffi(erlang_json, theta, times_pmap, discrete=False)

# 1. Sequential vmap (runs on single device)
print("1. Sequential vmap (single device):")
vmap_compute = jax.vmap(compute_pdf_single)

# Warm-up
_ = vmap_compute(theta_pmap_batch)

# Time it
start_time = time.time()
pdf_vmap = vmap_compute(theta_pmap_batch)
time_vmap = (time.time() - start_time) * 1000
print(f"   Time: {time_vmap:.2f}ms")
print(f"   Result shape: {pdf_vmap.shape}")
print()

# 2. Parallel pmap (distributes across devices)
print(f"2. Parallel pmap ({num_cpus} devices):")

# Reshape batch to split across devices: (n_devices, particles_per_device, n_params)
theta_pmap_split = theta_pmap_batch.reshape(num_cpus, num_pmap_particles // num_cpus, 1)
print(f"   Input shape for pmap: {theta_pmap_split.shape}")

# Define pmap function that maps over first axis (devices)
pmap_compute = jax.pmap(vmap_compute, in_axes=0)

# Warm-up
_ = pmap_compute(theta_pmap_split)

# Time it
start_time = time.time()
pdf_pmap = pmap_compute(theta_pmap_split)
time_pmap = (time.time() - start_time) * 1000
print(f"   Time: {time_pmap:.2f}ms")
print(f"   Result shape: {pdf_pmap.shape}")
print(f"   Speedup: {time_vmap / time_pmap:.2f}x")
print()

# Verify results are the same
pdf_pmap_flat = pdf_pmap.reshape(num_pmap_particles, -1)
difference = jnp.max(jnp.abs(pdf_vmap - pdf_pmap_flat))
print(f"   Max difference between vmap and pmap: {difference:.2e}")
print()

# ============================================================================
# 8.2: Parallel SVGD with pmap
# ============================================================================

print("--- 8.2: Parallel SVGD Objective Evaluation ---")
print()

# Use pmap to parallelize SVGD objective computation across particles
target_moments_parallel = jnp.array([3.0, 12.0, 60.0])
observations_parallel = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

def svgd_objective_single(theta):
    """SVGD objective for a single particle."""
    pdf, moments = compute_pmf_and_moments_ffi(
        erlang_json, theta, observations_parallel,
        nr_moments=3, discrete=False
    )
    log_likelihood = jnp.sum(jnp.log(pdf + 1e-10))
    moment_penalty = jnp.sum((moments - target_moments_parallel)**2)
    return log_likelihood - 0.1 * moment_penalty

# Sequential vmap evaluation
print(f"1. Sequential evaluation (vmap, {num_pmap_particles} particles):")
vmap_objective = jax.vmap(svgd_objective_single)
start_time = time.time()
objectives_vmap = vmap_objective(theta_pmap_batch)
time_vmap_obj = (time.time() - start_time) * 1000
print(f"   Time: {time_vmap_obj:.2f}ms")
print()

# Parallel pmap evaluation
print(f"2. Parallel evaluation (pmap, {num_cpus} devices):")
pmap_objective = jax.pmap(vmap_objective, in_axes=0)
start_time = time.time()
objectives_pmap = pmap_objective(theta_pmap_split)
time_pmap_obj = (time.time() - start_time) * 1000
print(f"   Time: {time_pmap_obj:.2f}ms")
print(f"   Speedup: {time_vmap_obj / time_pmap_obj:.2f}x")
print()

# Show best particle
objectives_pmap_flat = objectives_pmap.reshape(-1)
best_idx = jnp.argmax(objectives_pmap_flat)
print(f"   Best particle: λ={theta_pmap_batch[best_idx][0]:.4f}")
print(f"   Best objective: {objectives_pmap_flat[best_idx]:.2f}")
print()

# ============================================================================
# 8.3: Multi-Device Statistics and Load Balancing
# ============================================================================

print("--- 8.3: Device Load Balancing ---")
print()

# Demonstrate how pmap distributes work across devices
print(f"Work distribution across {num_cpus} CPUs:")
print(f"  Total particles: {num_pmap_particles}")
print(f"  Particles per device: {num_pmap_particles // num_cpus}")
print(f"  Input shape to pmap: {theta_pmap_split.shape}")
print(f"    - Axis 0 (devices): {theta_pmap_split.shape[0]}")
print(f"    - Axis 1 (particles per device): {theta_pmap_split.shape[1]}")
print(f"    - Axis 2 (parameters): {theta_pmap_split.shape[2]}")
print()

# Performance summary
print("Performance Summary:")
print(f"  PDF computation:")
print(f"    - vmap (sequential): {time_vmap:.2f}ms")
print(f"    - pmap (parallel):   {time_pmap:.2f}ms ({time_vmap/time_pmap:.2f}x speedup)")
print(f"  SVGD objective:")
print(f"    - vmap (sequential): {time_vmap_obj:.2f}ms")
print(f"    - pmap (parallel):   {time_pmap_obj:.2f}ms ({time_vmap_obj/time_pmap_obj:.2f}x speedup)")
print()

print("Key Insights:")
print(f"  • pmap distributes computation across {num_cpus} CPU cores")
print(f"  • Sequential vmap processes all particles on single device")
print(f"  • Parallel pmap provides ~{(time_vmap/time_pmap + time_vmap_obj/time_pmap_obj)/2:.1f}x average speedup")
print(f"  • Effective for SVGD with many particles (>{num_cpus*2})")
print(f"  • Batch size should be divisible by num_devices for best balance")
print()


# ============================================================================
# Part 9: Summary & Key Takeaways
# ============================================================================

print("=" * 80)
print("Part 9: Summary & Key Takeaways")
print("=" * 80)
print()
print("✓ GraphBuilder successfully separates structure from parameters")
print("✓ JAX FFI wrappers enable JIT compilation and vmap batching")
print("✓ Combined PMF + moments computation is efficient for SVGD")
print("✓ vmap enables batch processing of multiple parameter values")
print("✓ pmap enables parallel processing across multiple CPU devices")
print("✓ JIT compilation provides speedup after initial compilation")
print()
print("Key Features:")
print("  • Parameterized phase-type distributions with theta parameters")
print("  • JAX pure_callback integration for JIT compatibility")
print("  • Sequential vmap batching for multi-parameter inference")
print("  • Parallel pmap execution across multiple CPU devices")
print("  • Combined PMF+moments for moment-based regularization")
print("  • Thread-safe via JAX's automatic GIL management")
print()
print("Ready for:")
print("  • SVGD inference with moment-based regularization")
print("  • Multi-chain MCMC with parallel likelihood evaluation")
print("  • Multi-device parallel inference across CPU cores")
print("  • Parameter optimization with gradient-free methods")
print("  • Batch inference over multiple parameter sets")
print()
print("Next steps:")
print("  • Implement custom VJP rules for gradient-based optimization")
print("  • Expose native XLA FFI handlers for better performance")
print("  • Add GPU support for large-scale inference")
print()
print("=" * 80)
print("Showcase Complete!")
print("=" * 80)
