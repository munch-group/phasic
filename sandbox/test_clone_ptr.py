"""Test if cloned graphs share C pointers"""

from phasic import Graph
import ctypes

# Access internal graph pointer (hack for testing)
g1 = Graph(state_length=1)
v0 = g1.starting_vertex()
v1 = g1.find_or_create_vertex([1])
v0.add_edge(v1, 1.0)

# Get internal pointer via Python C API
g1_ptr = id(g1)
print(f"Original Python object id: {g1_ptr}")

# Clone
g2 = g1.clone()
g2_ptr = id(g2)
print(f"Cloned Python object id: {g2_ptr}")
print(f"Different Python objects: {g1_ptr != g2_ptr}")

# Check vertices_length before modification
print(f"\nBefore modification:")
print(f"  g1.vertices_length() = {g1.vertices_length()}")
print(f"  g2.vertices_length() = {g2.vertices_length()}")

# Modify g1
v2 = g1.find_or_create_vertex([2])
print(f"\nAfter adding vertex to g1:")
print(f"  g1.vertices_length() = {g1.vertices_length()}")
print(f"  g2.vertices_length() = {g2.vertices_length()}")

if g1.vertices_length() != g2.vertices_length():
    print("\n✅ Graphs are independent!")
else:
    print("\n❌ Graphs share state!")

    # Try to get C++ rf_graph pointer for debugging
    try:
        # Access rf_graph (this is a hack and may not work)
        print("\nDEBUG: Trying to access rf_graph...")
        if hasattr(g1, 'rf_graph'):
            print(f"  g1.rf_graph = {g1.rf_graph}")
            print(f"  g2.rf_graph = {g2.rf_graph}")
    except Exception as e:
        print(f"  Can't access rf_graph: {e}")
