#!/usr/bin/env python
"""Test with FFI disabled."""

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

# Force use of fallback (no FFI)
model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False, use_ffi=False)

theta = jnp.array([2.0])
times = jnp.array([0.5, 1.0, 1.5])

# Test with 2D rewards
rewards_2d = jnp.array([
    [1.0, 0.5, 2.0],
    [2.0, 1.0, 1.5],
    [0.5, 1.5, 1.0],
])

print(f"Rewards shape: {rewards_2d.shape}")
print()

print("Testing with use_ffi=False:")
pmf, moments = model(theta, times, rewards=rewards_2d)
print(f"  PMF shape: {pmf.shape}")
print(f"  Moments shape: {moments.shape}")

if pmf.shape == (3, 3):
    print("\n✓ SUCCESS!")
else:
    print("\n✗ FAILURE")
