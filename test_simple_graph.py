#!/usr/bin/env python
"""Test simple graph construction."""

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

print("Graph structure:")
print(f"Vertices: {g.vertices_length()}")
print()

# Print some graph details
print("Serialization (structure):")
ser = g.serialize()
print(f"Type: {type(ser)}")
print(f"Length: {len(ser) if hasattr(ser, '__len__') else 'N/A'}")
print()

# Test PDF directly
theta = np.array([2.0])
graph_with_theta = g.instantiate(theta)
print("Testing PDF directly on instantiated graph:")
times = [0.5, 1.0, 1.5]
for t in times:
    pdf_val = graph_with_theta.pdf(t, granularity=0)  # Auto granularity
    print(f"  t={t}: PDF={pdf_val}")

print()
print("For exponential distribution with rate=2.0:")
print("  Expected mean: 0.5")
print("  Expected PDF at t=0.5: 2.0 * exp(-2.0*0.5) =", 2.0 * np.exp(-2.0*0.5))
