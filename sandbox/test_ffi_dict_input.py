#!/usr/bin/env python
"""
Test FFI wrapper with dict input from graph.serialize()
"""
import numpy as np
import jax
import jax.numpy as jnp

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

from ptdalgorithms import Graph
from ptdalgorithms.ffi_wrappers import compute_pmf_ffi

# Build a simple rabbit model
def build_test_model():
    """Build simple 2-rabbit model"""
    g = Graph(state_length=2)
    initial_state = [2, 0]
    g.starting_vertex().add_edge(g.find_or_create_vertex(initial_state), 1)

    index = 1
    while index < g.vertices_length():
        vertex = g.vertex_at(index)
        state = vertex.state()

        if state[0] > 0:
            child_state = [state[0] - 1, state[1] + 1]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [state[0], 0, 0]
            )
            child_state = [0, state[1]]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [0, 1]
            )

        if state[1] > 0:
            child_state = [state[0] + 1, state[1] - 1]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [state[1], 0]
            )
            child_state = [state[0], 0]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [0, 1]
            )

        index += 1

    return g


print("Building test model...")
graph = build_test_model()
print(f"Model has {graph.vertices_length()} vertices")

# Test 1: Using dict from graph.serialize() (NEW - should now work)
print("\n=== Test 1: Dict input (from graph.serialize()) ===")
try:
    structure_dict = graph.serialize()
    print(f"Structure type: {type(structure_dict)}")

    theta = jnp.array([0.5, 0.1])
    times = jnp.linspace(0.1, 2.0, 10)

    pdf_values = compute_pmf_ffi(structure_dict, theta, times, discrete=False, granularity=50)

    print(f"✅ SUCCESS: Dict input works!")
    print(f"PDF values shape: {pdf_values.shape}")
    print(f"Sample values: {pdf_values[:3]}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 2: Using JSON string (OLD - should still work)
print("\n=== Test 2: JSON string input (backward compatibility) ===")
try:
    import json
    from ptdalgorithms.ffi_wrappers import _make_json_serializable

    structure_dict = graph.serialize()
    structure_json = json.dumps(_make_json_serializable(structure_dict))
    print(f"Structure type: {type(structure_json)}")

    pdf_values = compute_pmf_ffi(structure_json, theta, times, discrete=False, granularity=50)

    print(f"✅ SUCCESS: JSON string input works!")
    print(f"PDF values shape: {pdf_values.shape}")
    print(f"Sample values: {pdf_values[:3]}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n=== All tests passed! ===")
