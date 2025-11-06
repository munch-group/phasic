#!/usr/bin/env python3
"""Simple PMF test to isolate segfault"""

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

print("\nCreating PMF model...")
model = Graph.pmf_from_graph(graph, discrete=False)

print("\nComputing PMF...")
theta = jnp.array([2.0])
times = jnp.array([0.5, 1.0, 1.5, 2.0])

pmf = model(theta, times)

print(f"PMF: {pmf}")
print("✅ PMF computation successful!")
