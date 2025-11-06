#!/usr/bin/env python
"""Test with working coalescent example."""

import numpy as np
import json
from phasic import Graph
import phasic.phasic_pybind as cpp_module

def coalescent_callback(state, nr_samples=None):
    """Working coalescent model from examples."""
    if state.size == 0:
        # Initial state: nr_samples lineages
        return [([nr_samples], 0.0, [0.0])]  # Non-parameterized starting edge

    n = state[0]
    if n <= 1:
        # Absorbing state
        return []

    # Coalescence rate: n choose 2
    coalescence_rate = n * (n - 1) / 2

    # Return: (next_state, base_weight, [coefficient for theta])
    return [([n - 1], 0.0, [coalescence_rate])]

g = Graph(callback=coalescent_callback, parameterized=True, nr_samples=3)

print(f"Coalescent graph has {g.vertices_length()} vertices\n")

# Serialize
serialized = g.serialize()
print("Serialized structure:")
for key, value in serialized.items():
    if isinstance(value, (list, np.ndarray)) and len(str(value)) > 100:
        print(f"  {key}: {type(value)} (length={len(value)})")
    else:
        print(f"  {key}: {value}")
print()

# Create JSON string
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

# Create GraphBuilder
builder = cpp_module.parameterized.GraphBuilder(structure_json_str)

# Test compute_pmf_and_moments
theta = np.array([1.5])
times = np.array([0.5, 1.0, 1.5])
nr_moments = 2

print("Testing compute_pmf_and_moments with coalescent model...")
print(f"  theta: {theta}")
print(f"  times: {times}")
print()

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
print()
print(f"Expected: PMF should be non-zero, moments should be finite")
