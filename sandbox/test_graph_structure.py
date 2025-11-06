#!/usr/bin/env python
"""Check graph structure in detail."""

import numpy as np
from phasic import Graph

def simple_callback(state, nr_samples=None):
    print(f"Callback called with state: {state}")
    if state.size == 0:
        # Initial state: start with value 1
        result = [([1], 0.0, [0.0])]
        print(f"  -> Initial: {result}")
        return result

    if state[0] == 0:  # Absorbing state
        print(f"  -> Absorbing state, returning []")
        return []

    # Single transition to absorbing state with parameterized rate
    result = [([0], 0.0, [1.0])]
    print(f"  -> Transition: {result}")
    return result

print("Building graph...")
g = Graph(callback=simple_callback, parameterized=True, nr_samples=1)

print(f"\nGraph has {g.vertices_length()} vertices:")
for i in range(g.vertices_length()):
    v = g.vertex_at(i)
    print(f"  Vertex {i}: state = {v.state()}")

print("\nStarting vertex:")
start = g.starting_vertex()
print(f"  State: {start.state()}")
