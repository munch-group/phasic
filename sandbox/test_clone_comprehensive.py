"""Comprehensive test for graph.clone() functionality"""

from phasic import Graph
import os

print("="*70)
print("COMPREHENSIVE GRAPH.CLONE() TESTS")
print("="*70)

# Test 1: Basic clone with starting vertex
print("\nTest 1: Basic clone with starting vertex")
g1 = Graph(1)
v0 = g1.starting_vertex()
v1 = g1.find_or_create_vertex([1])
v0.add_edge(v1, 2.0)

g1_clone = g1.clone()
print(f"  Original: {g1.vertices_length()} vertices, starting vertex has {len(v0.edges())} edge(s)")
print(f"  Clone: {g1_clone.vertices_length()} vertices, starting vertex has {len(g1_clone.starting_vertex().edges())} edge(s)")

# Modify original
v2 = g1.find_or_create_vertex([2])
assert g1.vertices_length() == 3, f"Original should have 3 vertices, has {g1.vertices_length()}"
assert g1_clone.vertices_length() == 2, f"Clone should still have 2 vertices, has {g1_clone.vertices_length()}"
print("  ✅ Test 1 PASSED: Basic clone is independent")

# Test 2: Clone with multiple vertices and edges
print("\nTest 2: Clone with multiple vertices and edges")
g2 = Graph(1)
v0 = g2.starting_vertex()
v1 = g2.find_or_create_vertex([1])
v2 = g2.find_or_create_vertex([2])
v3 = g2.find_or_create_vertex([3])
v0.add_edge(v1, 1.0)
v1.add_edge(v2, 2.0)
v2.add_edge(v3, 3.0)

original_count = g2.vertices_length()
g2_clone = g2.clone()

# Check edges are preserved
orig_v1 = list(g2.vertices())[1]  # Vertex [1]
clone_v1 = list(g2_clone.vertices())[1]
assert len(orig_v1.edges()) == len(clone_v1.edges()), "Edge counts should match"

# Modify original
v4 = g2.find_or_create_vertex([4])
assert g2.vertices_length() == original_count + 1, "Original should have one more vertex"
assert g2_clone.vertices_length() == original_count, "Clone should be unchanged"
print("  ✅ Test 2 PASSED: Multi-vertex clone preserves structure")

# Test 3: Clone with graph containing vertex [0] (besides starting vertex)
print("\nTest 3: Clone with multiple [0] state vertices")
g3 = Graph(1)
v0 = g3.starting_vertex()
v1 = g3.find_or_create_vertex([1])
v2 = g3.find_or_create_vertex([0])  # Another vertex with state [0]!
v0.add_edge(v1, 1.0)
v1.add_edge(v2, 2.0)

print(f"  Original has {g3.vertices_length()} vertices")
print(f"  Starting vertex has {len(v0.edges())} edge(s)")
print(f"  Vertex with state [0] at end has {len(v2.edges())} edge(s)")

g3_clone = g3.clone()
clone_start = g3_clone.starting_vertex()

print(f"  Clone has {g3_clone.vertices_length()} vertices")
print(f"  Clone starting vertex has {len(clone_start.edges())} edge(s)")

# Starting vertex should have 1 edge, not 2
assert len(clone_start.edges()) == 1, f"Starting vertex should have 1 edge, has {len(clone_start.edges())}"

# Modify original
v3 = g3.find_or_create_vertex([3])
assert g3.vertices_length() != g3_clone.vertices_length(), "Clone should be independent"
print("  ✅ Test 3 PASSED: Multiple [0] vertices handled correctly")

# Test 4: Python copy() method works
print("\nTest 4: Python copy() method")
g4 = Graph(1)
v0 = g4.starting_vertex()
v1 = g4.find_or_create_vertex([1])
v0.add_edge(v1, 1.0)

g4_copy = g4.copy()
v2 = g4.find_or_create_vertex([2])

assert g4.vertices_length() == 3, f"Original should have 3 vertices"
assert g4_copy.vertices_length() == 2, f"Copy should have 2 vertices"
print("  ✅ Test 4 PASSED: Python copy() method works")

print("\n" + "="*70)
print("ALL TESTS PASSED! ✅")
print("graph.clone() and graph.copy() create independent deep copies")
print("="*70)

os._exit(0)  # Clean exit
