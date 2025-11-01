#!/usr/bin/env python
"""Test multivariate reward transformation in compute_pmf_and_moments."""

import numpy as np
from phasic import Graph

# Create a simple parameterized graph (exponential with rate = theta[0])
def simple_callback(state, nr_samples=None):
    if state.size == 0:
        # Initial state: start with value 1
        return [([1], 0.0, [0.0])]  # Non-parameterized starting edge

    if state[0] == 0:  # Absorbing state
        return []

    # Single transition to absorbing state with parameterized rate
    return [([0], 0.0, [1.0])]  # Weight = 1.0 * theta[0]

g = Graph(callback=simple_callback, parameterized=True, nr_samples=1)

print("Graph constructed with vertices:", g.vertices_length())
print()

# Test 1: No rewards (baseline)
print("=" * 60)
print("TEST 1: No rewards (baseline)")
print("=" * 60)

model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)
theta = np.array([2.0])  # Rate = 2.0
times = np.array([0.5, 1.0, 1.5])

pmf, moments = model(theta, times, rewards=None)

print(f"Theta: {theta}")
print(f"Times: {times}")
print(f"PMF shape: {pmf.shape}")
print(f"PMF values: {pmf}")
print(f"Moments shape: {moments.shape}")
print(f"Moments: {moments}")
print(f"Expected mean (1/rate): {1.0/theta[0]}")
print()

# Test 2: 1D rewards (backward compatibility)
print("=" * 60)
print("TEST 2: 1D rewards (backward compatibility)")
print("=" * 60)

n_vertices = g.vertices_length()
rewards_1d = np.array([1.0, 2.0])  # Different rewards for the two vertices

pmf_1d, moments_1d = model(theta, times, rewards=rewards_1d)

print(f"Rewards shape: {rewards_1d.shape}")
print(f"Rewards: {rewards_1d}")
print(f"PMF shape: {pmf_1d.shape}")
print(f"PMF values: {pmf_1d}")
print(f"Moments shape: {moments_1d.shape}")
print(f"Moments: {moments_1d}")
print()

# Test 3: 2D rewards (multivariate case)
print("=" * 60)
print("TEST 3: 2D rewards (multivariate, 3 features)")
print("=" * 60)

n_features = 3
rewards_2d = np.array([
    [1.0, 2.0, 3.0],  # Vertex 0 rewards for 3 features
    [0.5, 1.5, 2.5],  # Vertex 1 rewards for 3 features
])

print(f"Rewards shape: {rewards_2d.shape} (n_vertices={n_vertices}, n_features={n_features})")
print(f"Rewards:\n{rewards_2d}")
print()

pmf_2d, moments_2d = model(theta, times, rewards=rewards_2d)

print(f"PMF shape: {pmf_2d.shape} (should be (n_times, n_features))")
print(f"PMF values:\n{pmf_2d}")
print(f"Moments shape: {moments_2d.shape} (should be (n_features, nr_moments))")
print(f"Moments:\n{moments_2d}")
print()

# Verify each feature is computed independently
print("Verification: Each column should be computed with different reward vector")
for j in range(n_features):
    print(f"Feature {j}: rewards={rewards_2d[:, j]}, PMF[0,{j}]={pmf_2d[0, j]:.6f}, mean={moments_2d[j, 0]:.6f}")

print()
print("=" * 60)
print("All tests completed successfully!")
print("=" * 60)
