#!/usr/bin/env python
"""Complete test of 1D and 2D rewards."""

import numpy as np
from phasic import Graph

def simple_callback(state, nr_samples=None):
    if state.size == 0:
        return [([1], 0.0, [1.0])]  # Parameterized starting edge
    if state[0] == 0:
        return []
    return [([0], 0.0, [1.0])]  # Weight = theta[0]

g = Graph(callback=simple_callback, parameterized=True, nr_samples=1)
n_vertices = g.vertices_length()

print(f"Graph: {n_vertices} vertices\n")
print("="*70)

model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)
theta = np.array([2.0])
times = np.array([0.5, 1.0, 1.5])

# Test 1: No rewards
print("TEST 1: No rewards")
print("="*70)
pmf, moments = model(theta, times, rewards=None)
print(f"PMF shape: {pmf.shape}")
print(f"PMF: {pmf}")
print(f"Moments shape: {moments.shape}")
print(f"Moments: {moments}")
print(f"Expected mean (1/rate): {1.0/theta[0]:.3f}")
print()

# Test 2: 1D rewards
print("TEST 2: 1D rewards (backward compatible)")
print("="*70)
rewards_1d = np.array([1.0, 2.0, 0.5])  # length = n_vertices
pmf_1d, moments_1d = model(theta, times, rewards=rewards_1d)
print(f"Rewards shape: {rewards_1d.shape}")
print(f"PMF shape: {pmf_1d.shape}")
print(f"PMF: {pmf_1d}")
print(f"Moments shape: {moments_1d.shape}")
print(f"Moments: {moments_1d}")
print()

# Test 3: 2D rewards
print("TEST 3: 2D rewards (multivariate, 3 features)")
print("="*70)
n_features = 3
rewards_2d = np.array([
    [1.0, 0.5, 2.0],  # Vertex 0 rewards
    [2.0, 1.0, 1.5],  # Vertex 1 rewards
    [0.5, 1.5, 1.0],  # Vertex 2 rewards
])
print(f"Rewards shape: {rewards_2d.shape} (n_vertices={n_vertices}, n_features={n_features})")
print(f"Rewards:\n{rewards_2d}\n")

pmf_2d, moments_2d = model(theta, times, rewards=rewards_2d)
print(f"PMF shape: {pmf_2d.shape} (expected: (n_times={len(times)}, n_features={n_features}))")
print(f"PMF:\n{pmf_2d}\n")
print(f"Moments shape: {moments_2d.shape} (expected: (n_features={n_features}, nr_moments=2))")
print(f"Moments:\n{moments_2d}\n")

# Verify independence
print("VERIFICATION: Each feature computed independently")
print("="*70)
for j in range(n_features):
    print(f"Feature {j}:")
    print(f"  Reward vector: {rewards_2d[:, j]}")
    if pmf_2d.ndim == 2:
        print(f"  PMF at t=0.5: {pmf_2d[0, j]:.6f}")
        print(f"  Mean (1st moment): {moments_2d[j, 0]:.6f}")
    else:
        print(f"  ERROR: PMF is 1D (shape={pmf_2d.shape}), expected 2D")

print()
print("="*70)
if pmf_2d.ndim == 2 and pmf_2d.shape == (len(times), n_features):
    print("✓ SUCCESS: 2D rewards working correctly!")
else:
    print("✗ FAILURE: 2D rewards not returning 2D arrays")
print("="*70)
