"""Test GraphBuilder directly"""

import numpy as np
from phasic import Graph
import json

# Create simple exponential
graph1 = Graph(state_length=1, parameterized=True)
v0 = graph1.starting_vertex()
v1 = graph1.find_or_create_vertex([1])
v0.add_edge_parameterized(v1, 0.0, [1.0])

# Serialize
structure = graph1.serialize()

# Convert numpy arrays to lists for JSON serialization
def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

structure_serializable = convert_to_json_serializable(structure)
structure_json = json.dumps(structure_serializable)

print("Serialized structure:")
print(f"param_length: {structure['param_length']}")
print(f"start_param_edges: {structure['start_param_edges']}")
print()

# Now use GraphBuilder
from phasic.phasic_pybind import parameterized

builder = parameterized.GraphBuilder(structure_json)
print(f"GraphBuilder param_length: {builder.param_length}")
print(f"GraphBuilder vertices_length: {builder.vertices_length}")
print()

# Compute moments with theta=[2.0]
theta = np.array([2.0])
moments = builder.compute_moments(theta, nr_moments=2)

print(f"Computed moments: {moments}")
print(f"Expected: [0.5, 0.5]")
print()

# Try computing PDF
times = np.array([0.5])
pmf = builder.compute_pmf(theta, times, discrete=False, granularity=100)
print(f"Computed PDF(0.5): {pmf[0]}")
print(f"Expected: {2*np.exp(-1):.6f}")
