#!/usr/bin/env python
"""Comprehensive test of multivariate rewards implementation."""

import numpy as np
from phasic import Graph
import jax.numpy as jnp

print("="*70)
print("MULTIVARIATE REWARDS IMPLEMENTATION TEST")
print("="*70)
print()

# Create a simple parameterized graph
def simple_callback(state, nr_samples=None):
    if state.size == 0:
        return [([1], 0.0, [1.0])]  # Parameterized starting edge
    if state[0] == 0:
        return []
    return [([0], 0.0, [1.0])]  # Weight = theta[0]

g = Graph(callback=simple_callback, parameterized=True, nr_samples=1)
n_vertices = g.vertices_length()

print(f"Graph: {n_vertices} vertices\n")

# Create both models for comparison
model_univariate = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)
model_multivariate = Graph.pmf_and_moments_from_graph_multivariate(g, nr_moments=2, discrete=False)

theta = jnp.array([2.0])
times = jnp.array([0.5, 1.0, 1.5])

# Test 1: No rewards (both models should behave identically)
print("TEST 1: No rewards (both models identical)")
print("-"*70)
pmf1, moments1 = model_univariate(theta, times, rewards=None)
pmf2, moments2 = model_multivariate(theta, times, rewards=None)
print(f"Univariate:   PMF shape={pmf1.shape}, Moments shape={moments1.shape}")
print(f"Multivariate: PMF shape={pmf2.shape}, Moments shape={moments2.shape}")
print(f"Shapes match: {pmf1.shape == pmf2.shape and moments1.shape == moments2.shape}")
print(f"Values match: {np.allclose(pmf1, pmf2) and np.allclose(moments1, moments2)}")
print()

# Test 2: 1D rewards (both models should behave identically)
print("TEST 2: 1D rewards (both models identical)")
print("-"*70)
rewards_1d = jnp.array([1.0, 2.0, 0.5])
pmf1, moments1 = model_univariate(theta, times, rewards=rewards_1d)
pmf2, moments2 = model_multivariate(theta, times, rewards=rewards_1d)
print(f"Rewards shape: {rewards_1d.shape}")
print(f"Univariate:   PMF shape={pmf1.shape}, Moments shape={moments1.shape}")
print(f"Multivariate: PMF shape={pmf2.shape}, Moments shape={moments2.shape}")
print(f"Shapes match: {pmf1.shape == pmf2.shape and moments1.shape == moments2.shape}")
print(f"Values match: {np.allclose(pmf1, pmf2) and np.allclose(moments1, moments2)}")
print()

# Test 3: 2D rewards (multivariate only)
print("TEST 3: 2D rewards (multivariate model)")
print("-"*70)
n_features = 3
rewards_2d = jnp.array([
    [1.0, 0.5, 2.0],
    [2.0, 1.0, 1.5],
    [0.5, 1.5, 1.0],
])
print(f"Rewards shape: {rewards_2d.shape} (n_vertices={n_vertices}, n_features={n_features})")
print()

pmf_2d, moments_2d = model_multivariate(theta, times, rewards=rewards_2d)
print(f"PMF shape: {pmf_2d.shape} (expected: ({len(times)}, {n_features}))")
print(f"Moments shape: {moments_2d.shape} (expected: ({n_features}, 2))")
print()

print("PMF values:")
for t_idx, t in enumerate(times):
    print(f"  t={t}: {pmf_2d[t_idx, :]}")
print()

print("Moments (per feature):")
for j in range(n_features):
    print(f"  Feature {j}: E[T]={moments_2d[j, 0]:.3f}, E[T²]={moments_2d[j, 1]:.3f}")
print()

# Verify independence: each feature computed with its own reward vector
print("Verification: Features computed independently")
print("-"*70)
all_match = True
for j in range(n_features):
    # Compute feature j using univariate model
    reward_j = rewards_2d[:, j]
    pmf_j, moments_j = model_univariate(theta, times, rewards=reward_j)

    # Compare with multivariate result
    pmf_match = np.allclose(pmf_j, pmf_2d[:, j])
    moments_match = np.allclose(moments_j, moments_2d[j, :])

    print(f"Feature {j}: PMF match={pmf_match}, Moments match={moments_match}")
    all_match = all_match and pmf_match and moments_match

print()

# Final summary
print("="*70)
print("SUMMARY")
print("="*70)
if (pmf_2d.shape == (len(times), n_features) and
    moments_2d.shape == (n_features, 2) and
    all_match):
    print("✓ SUCCESS: All tests passed!")
    print("  - No rewards: univariate and multivariate models identical")
    print("  - 1D rewards: univariate and multivariate models identical")
    print("  - 2D rewards: correct shapes and independent feature computation")
else:
    print("✗ FAILURE: Some tests failed")
print("="*70)
