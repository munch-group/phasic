#!/usr/bin/env python
"""Test using pmf_and_moments_from_graph_multivariate."""

import numpy as np
from phasic import Graph
import jax.numpy as jnp

def simple_callback(state, nr_samples=None):
    if state.size == 0:
        return [([1], 0.0, [1.0])]
    if state[0] == 0:
        return []
    return [([0], 0.0, [1.0])]

g = Graph(callback=simple_callback, parameterized=True, nr_samples=1)
n_vertices = g.vertices_length()

# Use the multivariate function
model = Graph.pmf_and_moments_from_graph_multivariate(g, nr_moments=2, discrete=False)

theta = jnp.array([2.0])
times = jnp.array([0.5, 1.0, 1.5])

# Test with 2D rewards
rewards_2d = jnp.array([
    [1.0, 0.5, 2.0],
    [2.0, 1.0, 1.5],
    [0.5, 1.5, 1.0],
])

print(f"Graph: {n_vertices} vertices")
print(f"Rewards shape: {rewards_2d.shape}")
print()

print("Testing pmf_and_moments_from_graph_multivariate:")
pmf, moments = model(theta, times, rewards=rewards_2d)
print(f"  PMF shape: {pmf.shape} (expected: (3, 3))")
print(f"  Moments shape: {moments.shape} (expected: (3, 2))")
print(f"  PMF:\n{pmf}")
print(f"  Moments:\n{moments}")

if pmf.shape == (3, 3) and moments.shape == (3, 2):
    print("\n✓ SUCCESS: Multivariate function works!")
else:
    print("\n✗ FAILURE")
