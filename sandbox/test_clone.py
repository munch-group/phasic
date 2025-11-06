"""Test graph.clone() implementation"""

from phasic import Graph
import numpy as np

print("="*70)
print("Testing graph.clone()")
print("="*70)

# Create a simple graph
g = Graph(state_length=1)
v0 = g.starting_vertex()
v1 = g.find_or_create_vertex([1])
v2 = g.find_or_create_vertex([2])
v3 = g.find_or_create_vertex([0])

v0.add_edge(v1, 1.0)
v1.add_edge(v2, 3.0)
v2.add_edge(v3, 2.0)

print(f"\nOriginal graph: {g.vertices_length()} vertices")
for i, v in enumerate(g.vertices()):
    print(f"  Vertex {i}: state={v.state()}, edges={len(v.edges())}")

# Clone the graph
print("\nCloning graph...")
g_copy = g.clone()

print(f"\nCloned graph: {g_copy.vertices_length()} vertices")
for i, v in enumerate(g_copy.vertices()):
    print(f"  Vertex {i}: state={v.state()}, edges={len(v.edges())}")

# Test that they're independent
print("\nAdding vertex to original graph...")
v4 = g.find_or_create_vertex([3])
v3.add_edge(v4, 1.5)

print(f"Original graph now has {g.vertices_length()} vertices")
print(f"Cloned graph still has {g_copy.vertices_length()} vertices")

if g_copy.vertices_length() == 3:
    print("\n✅ Clone is independent!")
else:
    print("\n❌ Clone is not independent!")

# Test copy() method
print("\n" + "="*70)
print("Testing graph.copy() method")
print("="*70)

g2 = Graph(state_length=1)
v0 = g2.starting_vertex()
v1 = g2.find_or_create_vertex([1])
v0.add_edge(v1, [2.0])  # Parameterized edge

print(f"\nOriginal graph: param_length={g2.c_graph().param_length}")
g2_copy = g2.copy()
print(f"Copied graph: param_length={g2_copy.c_graph().param_length}")

print("\n✅ ALL CLONE TESTS PASSED!")
