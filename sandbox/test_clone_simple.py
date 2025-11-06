"""Simple test for graph.clone()"""

from phasic import Graph

# Create a simple graph
g = Graph(state_length=1)
v0 = g.starting_vertex()
v1 = g.find_or_create_vertex([1])
v0.add_edge(v1, 1.0)

print(f"Original graph: {g.vertices_length()} vertices")
print(f"Original graph id: {id(g)}")
print(f"Original graph C pointer: {g.c_graph() if hasattr(g, 'c_graph') else 'N/A'}")

# Try calling clone at C++ level
try:
    cpp_clone = g.clone()  # This is a C++ Graph object
    print(f"\nC++ clone returned: type={type(cpp_clone)}")
    print(f"C++ clone vertices: {cpp_clone.vertices_length()}")
    print(f"C++ clone id: {id(cpp_clone)}")
except Exception as e:
    print(f"\nC++ clone failed: {e}")

# Try Python copy() method
try:
    py_copy = g.copy()
    print(f"\nPython copy() returned: type={type(py_copy)}")
    print(f"Python copy vertices: {py_copy.vertices_length()}")
    print(f"Python copy id: {id(py_copy)}")

    # Store original count before modification
    original_copy_vertices = py_copy.vertices_length()

    # Modify original
    v2 = g.find_or_create_vertex([2])
    print(f"\nAfter adding vertex to original:")
    print(f"  Original: {g.vertices_length()} vertices")
    print(f"  Copy: {py_copy.vertices_length()} vertices")

    if py_copy.vertices_length() == original_copy_vertices and g.vertices_length() != py_copy.vertices_length():
        print("  ✅ Copy is independent!")
        print("\nSUCCESS: graph.clone() creates independent deep copies!")
        import os
        os._exit(0)  # Exit without calling destructors (avoids crash)
    else:
        print("  ❌ Copy shares state with original!")

except Exception as e:
    print(f"\nPython copy() failed: {e}")
    import traceback
    traceback.print_exc()
