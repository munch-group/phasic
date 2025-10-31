#!/usr/bin/env python3
"""Simple SVGD test to isolate segfault"""

import numpy as np
from phasic import Graph, configure
import jax.numpy as jnp

configure(ffi=True, openmp=True, jit=True)

print("Creating simple exponential graph...")
graph = Graph(state_length=1)
v_start = graph.starting_vertex()
v_transient = graph.find_or_create_vertex([1])
v_absorb = graph.find_or_create_vertex([0])

v_start.add_edge(v_transient, 1.0)
v_transient.add_edge_parameterized(v_absorb, 0.0, [1.0])

print(f"Graph vertices: {graph.vertices_length()}")

print("\nGenerating synthetic data...")
np.random.seed(42)
true_theta = 2.0
n_samples = 100  # Start with small sample
observed_data = np.random.exponential(scale=1/true_theta, size=n_samples)

print(f"Sample mean: {np.mean(observed_data):.4f} (expected: 0.5000)")

print("\nRunning SVGD (small test)...")
svgd = graph.svgd(
    observed_data=observed_data,
    theta_dim=1,
    n_particles=10,
    n_iterations=10,
    learning_rate=0.01,
    seed=42,
    verbose=False,
    regularization=0.0,
    nr_moments=0,
    positive_params=True
)

results = svgd.get_results()
print(f"Posterior mean: {results['theta_mean'][0]:.4f}")
print("✅ SVGD test successful!")
