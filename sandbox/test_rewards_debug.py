#!/usr/bin/env python
"""Debug rewards shape issue."""

import numpy as np
from phasic import Graph

# Create a simple parameterized graph
def simple_callback(state, nr_samples=None):
    if state.size == 0:
        # Initial state: start with value 1
        return [([1], 0.0, [0.0])]  # Non-parameterized starting edge

    if state[0] == 0:  # Absorbing state
        return []

    # Single transition to absorbing state with parameterized rate
    return [([0], 0.0, [1.0])]  # Weight = 1.0 * theta[0]

g = Graph(callback=simple_callback, parameterized=True, nr_samples=1)

print(f"Graph has {g.vertices_length()} vertices")
print("Vertex states:")
for i in range(g.vertices_length()):
    v = g.vertex_at(i)
    print(f"  Vertex {i}: state = {v.state()}")

print()

# Try with correct number of vertices
n_vertices = g.vertices_length()
print(f"Testing with {n_vertices} vertices")

model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)
theta = np.array([2.0])
times = np.array([0.5, 1.0, 1.5])

# Test 1: No rewards
print("\n1. No rewards:")
pmf, moments = model(theta, times, rewards=None)
print(f"   PMF: {pmf}")
print(f"   Moments: {moments}")

# Test 2: 1D rewards with correct number of vertices
print(f"\n2. 1D rewards (length={n_vertices}):")
rewards_1d = np.ones(n_vertices)  # All ones
pmf_1d, moments_1d = model(theta, times, rewards=rewards_1d)
print(f"   PMF: {pmf_1d}")
print(f"   Moments: {moments_1d}")

# Test 3: 2D rewards
print(f"\n3. 2D rewards ({n_vertices}, 3):")
rewards_2d = np.ones((n_vertices, 3))  # 3 features
pmf_2d, moments_2d = model(theta, times, rewards=rewards_2d)
print(f"   PMF shape: {pmf_2d.shape}")
print(f"   PMF:\n{pmf_2d}")
print(f"   Moments shape: {moments_2d.shape}")
print(f"   Moments:\n{moments_2d}")
