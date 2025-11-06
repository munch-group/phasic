#!/usr/bin/env python3
"""
GraphBuilder Correctness Test - 10,000 Observations
===================================================

Tests SVGD inference on coalescent model with 10,000 observations.
Target: < 2% relative error
Uses θ = 10 as specified by user
"""

import numpy as np
from phasic import Graph, configure
import jax.numpy as jnp

configure(ffi=True, openmp=True, jit=True)

print("="*80)
print("GraphBuilder Correctness Test - 10,000 Observations")
print("="*80)

# ============================================================================
# Coalescent Model Callback
# ============================================================================

def coalescent_callback(state, nr_samples=None):
    """
    Coalescent model: n lineages → n-1 lineages at rate n*(n-1)/2
    """
    if state.size == 0:
        # Initial state: nr_samples lineages
        return [([nr_samples], 0.0, [1.0])]

    n = state[0]
    if n <= 1:
        return []

    coalescence_rate = n * (n - 1) / 2
    return [([n - 1], 0.0, [coalescence_rate])]

# ============================================================================
# Test 1: Univariate SVGD - 10k Observations
# ============================================================================

print("\n" + "="*80)
print("Test 1: Univariate SVGD - 10,000 Observations")
print("="*80)

print("\n1. Building parameterized coalescent graph...")
graph = Graph(
    callback=coalescent_callback,
    parameterized=True,
    nr_samples=3
)

print(f"   Graph created with {graph.vertices_length()} vertices")

# Generate synthetic data from true parameter
print("\n2. Generating 10,000 synthetic observations...")
true_theta = 10.0
np.random.seed(42)

# Sample from the true model
n_obs = 10000
graph.update_parameterized_weights(np.array([true_theta]))
observed_times = graph.sample(n_obs)

print(f"   True parameter: θ = {true_theta}")
print(f"   Generated {n_obs} observations")
print(f"   Sample mean: {np.mean(observed_times):.4f}")
print(f"   Sample std:  {np.std(observed_times):.4f}")

# Run SVGD inference
print("\n3. Running SVGD inference...")
print("   Using GraphBuilder backend with FFI + OpenMP")

svgd = graph.svgd(
    observed_data=observed_times,
    theta_dim=1,
    n_particles=100,
    n_iterations=2000,  # More iterations for larger dataset
    learning_rate=0.01,  # Will be auto-scaled
    seed=42,
    verbose=True,
    positive_params=True,  # Constrain theta > 0
    jit=True,
    parallel='pmap'  # Use multi-core parallelization
)
results = svgd.get_results()

print(f"\n" + "="*80)
print("RESULTS - Univariate SVGD")
print("="*80)

posterior_mean = results['theta_mean'][0]
posterior_std = results['theta_std'][0]
error = abs(posterior_mean - true_theta)
rel_error = error / true_theta

print(f"\nTrue parameter:     θ = {true_theta:.4f}")
print(f"Posterior mean:     θ̂ = {posterior_mean:.4f}")
print(f"Posterior std:      σ = {posterior_std:.4f}")
print(f"Absolute error:     |θ̂ - θ| = {error:.4f}")
print(f"Relative error:     {rel_error*100:.2f}%")

# Check convergence
if rel_error < 0.02:
    print(f"\n✅ PASSED: Relative error {rel_error*100:.2f}% < 2%")
    exit_code_univar = 0
else:
    print(f"\n❌ FAILED: Relative error {rel_error*100:.2f}% >= 2%")
    exit_code_univar = 1

# ============================================================================
# Test 2: Multivariate SVGD - 10k Observations with 2D Rewards
# ============================================================================

print("\n" + "="*80)
print("Test 2: Multivariate SVGD - 10,000 Observations with 2D Rewards")
print("="*80)

print("\n1. Building coalescent graph for multivariate test...")
graph_mv = Graph(
    callback=coalescent_callback,
    parameterized=True,
    nr_samples=3
)

print(f"   Graph created with {graph_mv.vertices_length()} vertices")

# Generate sparse multivariate observations
print("\n2. Generating sparse 2D observations...")
n_features = 3
n_obs_per_feature = 3400  # ~10k total observations
n_obs_total = n_obs_per_feature * n_features

# Initialize with all NaNs: (n_obs_total, n_features)
observed_mv = np.full((n_obs_total, n_features), np.nan)

# Generate observations for each feature separately using graph.sample()
graph_mv.update_parameterized_weights(np.array([true_theta]))
for feature_idx in range(n_features):
    feature_times = graph_mv.sample(n_obs_per_feature)

    # Place each observation in its own row (round-robin)
    for i, time_val in enumerate(feature_times):
        row_idx = feature_idx + i * n_features
        observed_mv[row_idx, feature_idx] = time_val

print(f"   Observations shape: {observed_mv.shape}")
print(f"   Features: {n_features}")
print(f"   Total rows: {n_obs_total}")
print(f"   Non-NaN values: {np.sum(~np.isnan(observed_mv))} ({n_obs_per_feature} per feature)")
print(f"   Sparsity: {100 * np.sum(np.isnan(observed_mv)) / observed_mv.size:.1f}% NaN")

# Create 2D rewards
n_vertices = graph_mv.vertices_length()
rewards_2d = np.ones((n_vertices, n_features))
# Different emphasis for each feature
rewards_2d[1, 0] = 2.0
rewards_2d[1, 1] = 2.0
rewards_2d[2, 2] = 2.0

print(f"   Rewards shape: {rewards_2d.shape}")

# Run multivariate SVGD
print("\n3. Running multivariate SVGD...")

svgd_mv = graph_mv.svgd(
    observed_data=observed_mv,
    theta_dim=1,
    n_particles=100,
    n_iterations=2000,  # More iterations for larger dataset
    learning_rate=0.01,  # Will be auto-scaled
    seed=42,
    verbose=True,
    positive_params=True,
    jit=True,
    parallel='pmap',
    rewards=rewards_2d  # 2D rewards
)
results_mv = svgd_mv.get_results()

print(f"\n" + "="*80)
print("RESULTS - Multivariate SVGD")
print("="*80)

posterior_mean_mv = results_mv['theta_mean'][0]
posterior_std_mv = results_mv['theta_std'][0]
error_mv = abs(posterior_mean_mv - true_theta)
rel_error_mv = error_mv / true_theta

print(f"\nTrue parameter:     θ = {true_theta:.4f}")
print(f"Posterior mean:     θ̂ = {posterior_mean_mv:.4f}")
print(f"Posterior std:      σ = {posterior_std_mv:.4f}")
print(f"Absolute error:     |θ̂ - θ| = {error_mv:.4f}")
print(f"Relative error:     {rel_error_mv*100:.2f}%")

# Check convergence
if rel_error_mv < 0.02:
    print(f"\n✅ PASSED: Relative error {rel_error_mv*100:.2f}% < 2%")
    exit_code_multivar = 0
else:
    print(f"\n❌ FAILED: Relative error {rel_error_mv*100:.2f}% >= 2%")
    exit_code_multivar = 1

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nTrue parameter: θ = {true_theta}")
print(f"Observations: {n_obs} (univariate), {n_obs_total} (multivariate)")

print(f"\nUnivariate SVGD:")
print(f"  Error: {rel_error*100:.2f}% {'✅' if exit_code_univar == 0 else '❌'}")

print(f"\nMultivariate SVGD:")
print(f"  Error: {rel_error_mv*100:.2f}% {'✅' if exit_code_multivar == 0 else '❌'}")

exit_code = max(exit_code_univar, exit_code_multivar)

if exit_code == 0:
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED - GraphBuilder correctness verified!")
    print("="*80)
else:
    print("\n" + "="*80)
    print("❌ SOME TESTS FAILED - See results above")
    print("="*80)

import sys
sys.exit(exit_code)
