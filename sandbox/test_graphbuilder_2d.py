#!/usr/bin/env python
"""Test GraphBuilder directly with 2D rewards."""

import numpy as np
import json
from phasic import Graph
import phasic.phasic_pybind as cpp_module

def simple_callback(state, nr_samples=None):
    if state.size == 0:
        return [([1], 0.0, [1.0])]
    if state[0] == 0:
        return []
    return [([0], 0.0, [1.0])]

g = Graph(callback=simple_callback, parameterized=True, nr_samples=1)
n_vertices = g.vertices_length()

# Serialize
serialized = g.serialize()

# Convert to JSON
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

theta = np.array([2.0])
times = np.array([0.5, 1.0, 1.5])
nr_moments = 2

print(f"Graph: {n_vertices} vertices")
print()

# Test 1: No rewards
print("TEST 1: No rewards")
pmf, moments = builder.compute_pmf_and_moments(
    theta, times, nr_moments=nr_moments, discrete=False, granularity=0, rewards=None
)
print(f"  PMF shape: {pmf.shape}, Moments shape: {moments.shape}")
print(f"  PMF: {pmf}")
print()

# Test 2: 1D rewards
print("TEST 2: 1D rewards")
rewards_1d = np.array([1.0, 2.0, 0.5])
pmf_1d, moments_1d = builder.compute_pmf_and_moments(
    theta, times, nr_moments=nr_moments, discrete=False, granularity=0, rewards=rewards_1d
)
print(f"  Rewards shape: {rewards_1d.shape}")
print(f"  PMF shape: {pmf_1d.shape}, Moments shape: {moments_1d.shape}")
print(f"  PMF: {pmf_1d}")
print()

# Test 3: 2D rewards
print("TEST 3: 2D rewards")
n_features = 3
rewards_2d = np.array([
    [1.0, 0.5, 2.0],
    [2.0, 1.0, 1.5],
    [0.5, 1.5, 1.0],
])
print(f"  Rewards shape: {rewards_2d.shape}")

pmf_2d, moments_2d = builder.compute_pmf_and_moments(
    theta, times, nr_moments=nr_moments, discrete=False, granularity=0, rewards=rewards_2d
)
print(f"  PMF shape: {pmf_2d.shape} (expected: ({len(times)}, {n_features}))")
print(f"  Moments shape: {moments_2d.shape} (expected: ({n_features}, {nr_moments}))")
print(f"  PMF:\n{pmf_2d}")
print(f"  Moments:\n{moments_2d}")
print()

if pmf_2d.shape == (len(times), n_features) and moments_2d.shape == (n_features, nr_moments):
    print("✓ SUCCESS: GraphBuilder returns correct 2D shapes!")
else:
    print("✗ FAILURE: GraphBuilder not returning 2D shapes")
