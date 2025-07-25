"""
Debug test for separated graph system
"""

import ctypes
import os

print("1. Testing library loading...")

# Test loading the main library
try:
    lib = ctypes.CDLL("/Users/kmt/PtDalgorithms/jax_extension/separated_graph_pmf.so")
    print("✓ Main library loaded successfully")
except Exception as e:
    print(f"✗ Failed to load main library: {e}")
    exit(1)

# Test if the function exists
try:
    func = lib.jax_separated_graph_pmf
    print("✓ Main function found")
except Exception as e:
    print(f"✗ Function not found: {e}")
    exit(1)

print("\n2. Testing imports...")

try:
    import jax
    import jax.numpy as jnp
    print("✓ JAX imported")
except Exception as e:
    print(f"✗ JAX import failed: {e}")
    exit(1)

try:
    from separated_graph_python import GraphConfig
    print("✓ GraphConfig imported")
except Exception as e:
    print(f"✗ GraphConfig import failed: {e}")
    exit(1)

print("\n3. Testing basic functionality...")

try:
    config = GraphConfig(nr_samples=2, mutation_rate=0.01)
    print(f"✓ GraphConfig created: {config}")
    print(f"✓ Serialized config: {config.to_string()}")
except Exception as e:
    print(f"✗ GraphConfig test failed: {e}")
    exit(1)

print("\n4. Testing user graph registration...")

try:
    from separated_graph_python import register_graph_builder
    print("✓ register_graph_builder imported")
except Exception as e:
    print(f"✗ register_graph_builder import failed: {e}")
    exit(1)

# Very simple test graph
simple_graph_code = """
    Graph graph;
    
    // Create a trivial 2-state graph
    std::vector<int> state1 = {1};
    std::vector<int> state2 = {0};
    
    int v1 = graph.add_vertex(state1);
    int v2 = graph.add_vertex(state2);
    
    graph.add_edge(v1, v2, 1.0);
    graph.set_absorption_rate(v2, 1.0);
    
    return graph;
"""

try:
    simple_pmf = register_graph_builder("simple", simple_graph_code)
    print("✓ Simple graph registered successfully")
except Exception as e:
    print(f"✗ Graph registration failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nAll basic tests passed!")