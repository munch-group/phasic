"""Debug moment computation"""

import numpy as np
from phasic import Graph
import jax.numpy as jnp

print("Debugging moment computation...\n")

# Test 1: Single exponential
print("="*60)
print("Test 1: Single Exponential(rate=2)")
print("="*60)

graph1 = Graph(state_length=1, parameterized=True)
v0 = graph1.starting_vertex()
v1 = graph1.find_or_create_vertex([1])
v0.add_edge_parameterized(v1, 0.0, [1.0])

print(f"Graph1 has {graph1.vertices_length()} vertices")

# Check the structure
structure = graph1.serialize()
print(f"param_length: {structure['param_length']}")
print(f"Regular edges: {len(structure['edges'])}")
print(f"Parameterized edges: {len(structure['param_edges'])}")
print(f"Starting regular edges: {len(structure['start_edges'])}")
print(f"Starting parameterized edges: {len(structure['start_param_edges'])}")

if len(structure['param_edges']) > 0:
    print("\nParameterized edges:")
    for edge in structure['param_edges']:
        print(f"  from {edge[0]:.0f} to {edge[1]:.0f}: coeffs={edge[2:]}")

if len(structure['start_param_edges']) > 0:
    print("\nStarting parameterized edges:")
    for edge in structure['start_param_edges']:
        print(f"  to {edge[0]:.0f}: coeffs={edge[1:]}")

model1 = Graph.pmf_and_moments_from_graph(graph1, nr_moments=2, discrete=False)

theta = jnp.array([2.0])
times = jnp.array([0.5, 1.0])

pmf, moments = model1(theta, times)

print(f"\nComputed moments: {moments}")
print(f"Expected: [0.5, 0.5]")

# Test 2: Erlang
print("\n" + "="*60)
print("Test 2: Erlang(2, rate=2)")
print("="*60)

graph2 = Graph(state_length=1, parameterized=True)
v0 = graph2.starting_vertex()
v1 = graph2.find_or_create_vertex([1])
v2 = graph2.find_or_create_vertex([2])
v0.add_edge_parameterized(v1, 0.0, [1.0])
v1.add_edge_parameterized(v2, 0.0, [1.0])

print(f"Graph2 has {graph2.vertices_length()} vertices")

# Check the structure
structure2 = graph2.serialize()
print(f"param_length: {structure2['param_length']}")
print(f"Regular edges: {len(structure2['edges'])}")
print(f"Parameterized edges: {len(structure2['param_edges'])}")
print(f"Starting regular edges: {len(structure2['start_edges'])}")
print(f"Starting parameterized edges: {len(structure2['start_param_edges'])}")

if len(structure2['param_edges']) > 0:
    print("\nParameterized edges:")
    for edge in structure2['param_edges']:
        print(f"  from {edge[0]:.0f} to {edge[1]:.0f}: coeffs={edge[2:]}")

if len(structure2['start_param_edges']) > 0:
    print("\nStarting parameterized edges:")
    for edge in structure2['start_param_edges']:
        print(f"  to {edge[0]:.0f}: coeffs={edge[1:]}")

model2 = Graph.pmf_and_moments_from_graph(graph2, nr_moments=2, discrete=False)

pmf2, moments2 = model2(theta, times)

print(f"\nComputed moments: {moments2}")
print(f"Expected: [1.0, 1.5]")
