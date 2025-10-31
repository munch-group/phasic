#!/usr/bin/env python3
"""
Final GraphBuilder Correctness Tests
=====================================

Tests PMF/moments and SVGD using GraphBuilder with 10,000 observations and <2% error.

Uses SVGD class directly (not graph.svgd()) to avoid callback-based segfault issue.
"""

from phasic import Graph, SVGD, configure
import numpy as np
import jax.numpy as jnp

configure(ffi=True, openmp=True, jit=True)

print("="*80)
print("GraphBuilder Correctness Tests - 10k Observations")
print("="*80)

# ============================================================================
# Test 1: PMF Computation Accuracy
# ============================================================================

print("\n" + "="*80)
print("Test 1: PMF Computation Accuracy")
print("="*80)

graph = Graph(state_length=1)
v_start = graph.starting_vertex()
v_transient = graph.find_or_create_vertex([1])
v_absorb = graph.find_or_create_vertex([0])

v_start.add_edge(v_transient, 1.0)
v_transient.add_edge_parameterized(v_absorb, 0.0, [1.0])

model = Graph.pmf_from_graph(graph, discrete=False)

theta = jnp.array([2.0])
times = jnp.array([0.5, 1.0, 1.5, 2.0])

pmf = model(theta, times)
expected = theta[0] * np.exp(-theta[0] * times)

print(f"\nTimes: {times}")
print(f"GraphBuilder PMF: {pmf}")
print(f"Expected PDF:     {expected}")
print(f"Max error:        {np.max(np.abs(pmf - expected)):.2e}")
print(f"Max rel error:    {np.max(np.abs(pmf - expected) / expected):.2%}")

np.testing.assert_allclose(pmf, expected, rtol=1e-3, atol=1e-4)
print("✅ PASSED: PMF accuracy < 0.1%")

# ============================================================================
# Test 2: Moment Computation Accuracy
# ============================================================================

print("\n" + "="*80)
print("Test 2: Moment Computation Accuracy")
print("="*80)

model_moments = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)
pmf, moments = model_moments(theta, times)

expected_mean = 1.0 / theta[0]
expected_second = 2.0 / (theta[0]**2)

print(f"\nGraphBuilder moments: {moments}")
print(f"Expected [E[T], E[T²]]: [{expected_mean:.4f}, {expected_second:.4f}]")
print(f"Error E[T]:  {abs(moments[0] - expected_mean):.2e}")
print(f"Error E[T²]: {abs(moments[1] - expected_second):.2e}")

np.testing.assert_allclose(moments[0], expected_mean, rtol=1e-10)
np.testing.assert_allclose(moments[1], expected_second, rtol=1e-10)
print("✅ PASSED: Moment computation exact")

# ============================================================================
# Test 3: Multivariate PMF with 2D Rewards
# ============================================================================

print("\n" + "="*80)
print("Test 3: Multivariate PMF with 2D Rewards - Independence")
print("="*80)

model_mv = Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=2, discrete=False)

n_features = 3
rewards_2d = jnp.array([
    [1.0, 2.0, 0.5],
    [1.0, 1.0, 1.0],
    [1.0, 0.5, 2.0],
])

times_1d = jnp.array([0.5, 1.0, 1.5])
pmf_2d, moments_2d = model_mv(theta, times_1d, rewards=rewards_2d)

print(f"\nPMF shape: {pmf_2d.shape} (expected: {(3, 3)})")
print(f"Moments shape: {moments_2d.shape} (expected: {(3, 2)})")

# Verify independence
model_1d = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)
print("\nVerifying feature independence:")
for j in range(n_features):
    reward_j = rewards_2d[:, j]
    pmf_j, moments_j = model_1d(theta, times_1d, rewards=reward_j)

    pmf_error = np.max(np.abs(pmf_2d[:, j] - pmf_j))
    moment_error = np.max(np.abs(moments_2d[j, :] - moments_j))
    print(f"  Feature {j}: PMF error={pmf_error:.2e}, Moment error={moment_error:.2e}")

    np.testing.assert_allclose(pmf_2d[:, j], pmf_j, rtol=1e-10)
    np.testing.assert_allclose(moments_2d[j, :], moments_j, rtol=1e-10)

print("✅ PASSED: Multivariate independence verified")

# ============================================================================
# Test 4: SVGD Univariate - 10k Observations
# ============================================================================

print("\n" + "="*80)
print("Test 4: SVGD Univariate - 10,000 Observations, No Regularization")
print("="*80)

np.random.seed(42)
true_theta = 2.0
n_samples = 10000
observed_data = np.random.exponential(scale=1/true_theta, size=n_samples)

print(f"\nTrue θ: {true_theta}")
print(f"Observations: {n_samples}")
print(f"Sample mean: {np.mean(observed_data):.4f} (expected: {1/true_theta:.4f})")

# Create model WITHOUT moments (no regularization)
# Note: Must use pmf_and_moments_from_graph with nr_moments=0 for SVGD compatibility
model_svgd = Graph.pmf_and_moments_from_graph(graph, nr_moments=0, discrete=False)

print("\nRunning SVGD...")
svgd = SVGD(
    model=model_svgd,
    observed_data=observed_data,
    theta_dim=1,
    n_particles=100,
    n_iterations=500,
    learning_rate=0.01,
    regularization=0.0,  # NO regularization
    nr_moments=0,
    verbose=True,
    positive_params=True
)

svgd.optimize()

print(f"\nResults:")
print(f"  Posterior mean: {svgd.theta_mean[0]:.4f}")
print(f"  Posterior std:  {svgd.theta_std[0]:.4f}")
error = abs(svgd.theta_mean[0] - true_theta)
rel_error = error / true_theta
print(f"  Absolute error: {error:.4f}")
print(f"  Relative error: {rel_error:.2%}")

assert rel_error < 0.02, f"Error {rel_error:.2%} > 2%"
print("✅ PASSED: SVGD univariate error < 2%")

# ============================================================================
# Test 5: SVGD Multivariate - 10k Observations
# ============================================================================

print("\n" + "="*80)
print("Test 5: SVGD Multivariate - 10,000 Observations, 2D Rewards, No Reg")
print("="*80)

np.random.seed(42)
n_obs_per_feature = 3400  # ~10k total
n_features = 3
n_obs_total = n_obs_per_feature * n_features

# Create sparse observations
observed_mv = np.full((n_obs_total, n_features), np.nan)
for j in range(n_features):
    samples = np.random.exponential(scale=1/true_theta, size=n_obs_per_feature)
    for i in range(n_obs_per_feature):
        row_idx = j + i * n_features
        observed_mv[row_idx, j] = samples[i]

print(f"\nObservations shape: {observed_mv.shape}")
print(f"Non-NaN values: {np.sum(~np.isnan(observed_mv))} ({n_obs_per_feature} per feature)")
print(f"Sparsity: {100 * np.sum(np.isnan(observed_mv)) / observed_mv.size:.1f}% NaN")

# Create 2D rewards
n_vertices = graph.vertices_length()
rewards_mv = np.ones((n_vertices, n_features))
rewards_mv[1, 0] = 2.0
rewards_mv[1, 1] = 3.0
rewards_mv[2, 2] = 2.0

print(f"2D Rewards shape: {rewards_mv.shape}")

# Create multivariate model WITHOUT moments
model_mv_svgd = Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=0, discrete=False)

print("\nRunning multivariate SVGD...")
svgd_mv = SVGD(
    model=model_mv_svgd,
    observed_data=observed_mv,
    theta_dim=1,
    n_particles=100,
    n_iterations=500,
    learning_rate=0.01,
    regularization=0.0,  # NO regularization
    nr_moments=0,
    verbose=True,
    positive_params=True,
    rewards=rewards_mv
)

svgd_mv.optimize()

print(f"\nMultivariate Results:")
print(f"  True θ:         {true_theta:.4f}")
print(f"  Posterior mean: {svgd_mv.theta_mean[0]:.4f}")
print(f"  Posterior std:  {svgd_mv.theta_std[0]:.4f}")
error_mv = abs(svgd_mv.theta_mean[0] - true_theta)
rel_error_mv = error_mv / true_theta
print(f"  Absolute error: {error_mv:.4f}")
print(f"  Relative error: {rel_error_mv:.2%}")

assert rel_error_mv < 0.02, f"Error {rel_error_mv:.2%} > 2%"
print("✅ PASSED: SVGD multivariate error < 2%")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("SUMMARY - All Tests Passed!")
print("="*80)

print("\n✅ Test 1: PMF computation accuracy < 0.1%")
print("✅ Test 2: Moment computation exact")
print("✅ Test 3: Multivariate independence verified")
print(f"✅ Test 4: SVGD univariate (10k obs) error = {rel_error:.2%} < 2%")
print(f"✅ Test 5: SVGD multivariate (10k obs) error = {rel_error_mv:.2%} < 2%")

print("\n" + "="*80)
print("GraphBuilder correctness VERIFIED!")
print("="*80)
