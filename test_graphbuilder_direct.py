#!/usr/bin/env python
"""Test GraphBuilder directly."""

import numpy as np
import json
from phasic import Graph
import phasic.phasic_pybind as cpp_module

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

print(f"Graph has {g.vertices_length()} vertices\n")

# Serialize
serialized = g.serialize()
print("Serialized structure:")
for key, value in serialized.items():
    if isinstance(value, (list, np.ndarray)) and len(str(value)) > 100:
        print(f"  {key}: {type(value)} (length={len(value)})")
    else:
        print(f"  {key}: {value}")
print()

# Create JSON string (convert numpy arrays to lists)
def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    else:
        return obj

serialized_clean = numpy_to_list(serialized)
structure_json_str = json.dumps(serialized_clean)
print(f"JSON length: {len(structure_json_str)} bytes\n")

# Create GraphBuilder
print("Creating GraphBuilder...")
builder = cpp_module.parameterized.GraphBuilder(structure_json_str)
print("GraphBuilder created successfully\n")

# Test compute_pmf_and_moments
theta = np.array([2.0])
times = np.array([0.5, 1.0, 1.5])
nr_moments = 2

print("Testing compute_pmf_and_moments...")
print(f"  theta: {theta}")
print(f"  times: {times}")
print(f"  nr_moments: {nr_moments}")
print()

try:
    pmf, moments = builder.compute_pmf_and_moments(
        theta,
        times,
        nr_moments=nr_moments,
        discrete=False,
        granularity=0,
        rewards=None
    )
    print(f"Success!")
    print(f"  PMF: {pmf}")
    print(f"  Moments: {moments}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
