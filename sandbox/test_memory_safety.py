"""Test memory safety issues that caused notebook crashes

Tests for:
1. Double-wrapping bug in copy() method
2. vertex_at() correctly uses reference_internal policy (vertices tied to graph lifetime)
3. Graph cloning with @phasic.callback decorator
"""

import phasic
import numpy as np
import gc

print("=" * 70)
print("TEST: Memory Safety and Crash Scenarios")
print("=" * 70)

# Test 1: vertex_at() works while graph is alive
print("\nTest 1: vertex_at() works correctly with graph lifetime")
g = phasic.Graph(2)
v1 = g.find_or_create_vertex([1, 0])
v2 = g.find_or_create_vertex([2, 0])
v1.add_edge(v2, 1.0)
v = g.vertex_at(1)  # Gets vertex at index 1 (which is [1, 0])
state = v.state()
print(f"  Vertex state while graph alive: {state}")
assert np.array_equal(state, [1, 0]), "Vertex should be valid while graph is alive"
print("  ✅ Test 1 PASSED - vertex_at() works while graph is alive")

# Test 2: copy() doesn't double-wrap
print("\nTest 2: copy() method works correctly")
g1 = phasic.Graph(2)
v1 = g1.find_or_create_vertex([1, 0])
v2 = g1.find_or_create_vertex([2, 0])
v1.add_edge(v2, 2.0)

# Make a copy
g2 = g1.copy()
print(f"  Original graph vertices: {g1.vertices_length()}")
print(f"  Copied graph vertices: {g2.vertices_length()}")
assert g2.vertices_length() == g1.vertices_length(), "Copy should have same vertices"

# Verify copy is independent
v3 = g2.find_or_create_vertex([3, 0])
assert g2.vertices_length() == g1.vertices_length() + 1, "Copy should be independent"
print("  ✅ Test 2 PASSED - copy() creates independent graph")

# Test 3: clone() with @phasic.callback pattern (from rabbits notebook)
print("\nTest 3: clone() with @phasic.callback decorator")

@phasic.callback([3, 2, 1])
def rabbit_model(state):
    """Rabbit population model from notebook"""
    rabbits, carrots, month = state
    edges = []

    if carrots > 0 and month > 0:
        # Rabbit eats carrot
        edges.append((np.array([rabbits, carrots - 1, month - 1]), 1.0))

    if month > 0:
        # Month passes without eating
        edges.append((np.array([rabbits, carrots, month - 1]), 1.0))

    return edges

# Create graph with callback
rabbit_graph = phasic.Graph(rabbit_model)
print(f"  Original graph vertices: {rabbit_graph.vertices_length()}")

# Clone it (this was crashing in the notebook)
cloned_graph = rabbit_graph.clone()
print(f"  Cloned graph vertices: {cloned_graph.vertices_length()}")
assert cloned_graph.vertices_length() == rabbit_graph.vertices_length()

# Verify independence
original_count = rabbit_graph.vertices_length()
cloned_graph.find_or_create_vertex([5, 5, 5])
assert cloned_graph.vertices_length() == original_count + 1
assert rabbit_graph.vertices_length() == original_count
print("  ✅ Test 3 PASSED - clone() with @phasic.callback works")

# Test 4: Multiple copy() operations (stress test)
print("\nTest 4: Multiple copy operations")
g = phasic.Graph(1)
g.find_or_create_vertex([1])
g.find_or_create_vertex([2])

copies = []
for i in range(10):
    copies.append(g.copy())

# Check all copies are valid
for i, copy in enumerate(copies):
    assert copy.vertices_length() == 3, f"Copy {i} should have 3 vertices"

print(f"  Created {len(copies)} copies successfully")
print("  ✅ Test 4 PASSED - multiple copy() operations work")

# Test 5: vertex_at() with copy (vertices should work with copied graph)
print("\nTest 5: vertex_at() works with copied graph")
g_orig = phasic.Graph(2)
g_orig.find_or_create_vertex([1, 0])
g_orig.find_or_create_vertex([2, 0])

g_copy = g_orig.copy()
v = g_copy.vertex_at(1)  # Get vertex from copy (index 1 = [1, 0])
state = v.state()
print(f"  Vertex state from copied graph: {state}")
assert np.array_equal(state, [1, 0])
print("  ✅ Test 5 PASSED - vertex_at() works with copied graph")

# Test 6: Edge access through vertex
print("\nTest 6: Edge access through vertex")
g = phasic.Graph(2)
v1 = g.find_or_create_vertex([1, 0])
v2 = g.find_or_create_vertex([2, 0])
v1.add_edge(v2, 3.0)
v = g.vertex_at(1)  # Get [1, 0] vertex which has an edge
edges = list(v.edges())
print(f"  Number of edges: {len(edges)}")
assert len(edges) == 1
print(f"  Edge weight: {edges[0].weight()}")
assert abs(edges[0].weight() - 3.0) < 1e-10
print("  ✅ Test 6 PASSED - edges accessible through vertex")

print("\n" + "=" * 70)
print("ALL MEMORY SAFETY TESTS PASSED! ✅")
print("=" * 70)
