#!/usr/bin/env python
"""Debug shape detection."""

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

model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

theta = jnp.array([2.0])
times = jnp.array([0.5, 1.0, 1.5])

# Test with 2D rewards
rewards_2d = jnp.array([
    [1.0, 0.5, 2.0],
    [2.0, 1.0, 1.5],
    [0.5, 1.5, 1.0],
])

print(f"Rewards shape: {rewards_2d.shape}")
print(f"Rewards ndim: {rewards_2d.ndim}")
print(f"Rewards type: {type(rewards_2d)}")
print()

# Call model
print("Calling model...")
pmf, moments = model(theta, times, rewards=rewards_2d)

print(f"PMF shape: {pmf.shape}")
print(f"Moments shape: {moments.shape}")
