#!/usr/bin/env python
"""Test with fixed callback (parameterized starting edge)."""

import numpy as np
from phasic import Graph

def simple_callback_v2(state, nr_samples=None):
    if state.size == 0:
        # Initial state: start with value 1
        # Use parameterized edge with coeff=1, but theta will effectively ignore it
        return [([1], 0.0, [1.0])]  # Parameterized starting edge

    if state[0] == 0:  # Absorbing state
        return []

    # Single transition to absorbing state with parameterized rate
    return [([0], 0.0, [1.0])]  # Weight = 1.0 * theta[0]

print("Building graph with parameterized starting edge...")
g = Graph(callback=simple_callback_v2, parameterized=True, nr_samples=1)

print(f"\nGraph has {g.vertices_length()} vertices:")
for i in range(g.vertices_length()):
    v = g.vertex_at(i)
    print(f"  Vertex {i}: state = {v.state()}")

print("\nStarting vertex:")
start = g.starting_vertex()
print(f"  State: {start.state()}")

print("\nSerialization:")
ser = g.serialize()
print(f"  states: {ser['states']}")
print(f"  start_edges: {ser['start_edges']}")
print(f"  start_param_edges: {ser['start_param_edges']}")
print(f"  param_edges: {ser['param_edges']}")

# Test PDF
print("\nTesting model...")
model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)
theta = np.array([2.0])
times = np.array([0.5, 1.0, 1.5])

pmf, moments = model(theta, times, rewards=None)
print(f"  PMF: {pmf}")
print(f"  Moments: {moments}")
print(f"\nExpected for exponential(rate=2): PDF(0.5) = {2.0 * np.exp(-1.0)}")
