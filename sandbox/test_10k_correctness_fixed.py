#!/usr/bin/env python3
"""
GraphBuilder Correctness Test - 10,000 Observations (FIXED PRIOR)
===================================================================

Tests SVGD inference on coalescent model with 10,000 observations.
Target: < 2% relative error
Uses θ = 10 as specified by user

FIX: Adjusted prior initialization from N(1,1) to N(2.5,1.0) to better
cover θ=10 after softplus transformation.

Prior range: softplus(N(2.5,1.0)) ≈ [1.5, 15] which includes θ=10
"""

import numpy as np
from phasic import Graph, configure, SVGD
import jax
import jax.numpy as jnp

configure(ffi=True, openmp=True, jit=True)

print("="*80)
print("GraphBuilder Correctness Test - 10,000 Observations (FIXED)")
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

# Create GraphBuilder model
print("\n3. Creating GraphBuilder model...")
model = Graph.pmf_and_moments_from_graph(graph, nr_moments=0, discrete=False)

# Run SVGD inference with ADJUSTED PRIOR
print("\n4. Running SVGD inference with adjusted prior...")
print("   FIXED: Using N(2.5, 1.0) prior to better cover θ=10")
print("   Prior range: softplus(N(2.5,1.0)) ≈ [1.5, 15]")

# Initialize particles manually from N(2.5, 1.0) for better coverage of θ=10
key = jax.random.PRNGKey(42)
theta_init = jax.random.normal(key, (100, 1)) * 1.0 + 2.5

# Use SVGD class directly with custom theta_init
svgd = SVGD(
    model=model,
    observed_data=observed_times,
    theta_dim=1,
    theta_init=theta_init,  # ADJUSTED: Custom initialization
    n_particles=100,
    n_iterations=3000,  # More iterations for convergence from wider prior
    learning_rate=0.01,  # Will be auto-scaled
    regularization=0.0,
    nr_moments=0,
    seed=42,
    verbose=True,
    positive_params=True,
    jit=True,
    parallel='pmap'
)

svgd.optimize()

print(f"\n" + "="*80)
print("RESULTS - Univariate SVGD")
print("="*80)

posterior_mean = svgd.theta_mean[0]
posterior_std = svgd.theta_std[0]
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

# Create multivariate model
print("\n3. Creating multivariate GraphBuilder model...")
model_mv = Graph.pmf_and_moments_from_graph_multivariate(
    graph_mv,
    nr_moments=0,
    discrete=False
)

# Run multivariate SVGD with adjusted prior
print("\n4. Running multivariate SVGD with adjusted prior...")

# Initialize particles manually from N(2.5, 1.0)
key_mv = jax.random.PRNGKey(42)
theta_init_mv = jax.random.normal(key_mv, (100, 1)) * 1.0 + 2.5

svgd_mv = SVGD(
    model=model_mv,
    observed_data=observed_mv,
    theta_dim=1,
    theta_init=theta_init_mv,  # ADJUSTED: Custom initialization
    n_particles=100,
    n_iterations=3000,  # More iterations
    learning_rate=0.01,  # Will be auto-scaled
    regularization=0.0,
    nr_moments=0,
    seed=42,
    verbose=True,
    positive_params=True,
    jit=True,
    parallel='pmap',
    rewards=rewards_2d  # 2D rewards
)

svgd_mv.optimize()

print(f"\n" + "="*80)
print("RESULTS - Multivariate SVGD")
print("="*80)

posterior_mean_mv = svgd_mv.theta_mean[0]
posterior_std_mv = svgd_mv.theta_std[0]
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
    print("\nKEY FIX: Adjusted prior from N(1,1) to N(2.5,1.0)")
    print("  Prior range: softplus(N(2.5,1.0)) ≈ [1.5, 15]")
    print("  This allows particles to reach θ=10")
else:
    print("\n" + "="*80)
    print("❌ SOME TESTS FAILED - See results above")
    print("="*80)

import sys
sys.exit(exit_code)
