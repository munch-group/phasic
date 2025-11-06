"""
Test find_vertex behavior
"""

import numpy as np
from phasic import Graph

def callback(state, **kwargs):
    n = state[0]
    if n <= 1:
        return []
    return [(np.array([n - 1]), [1.0])]

graph = Graph(callback, ipv=[3], parameterized=True, nr_samples=3)

print(f"Graph has {graph.vertices_length()} vertices\n")

# Try to find each vertex by its state
for i in range(graph.vertices_length()):
    v = graph.vertex_at(i)
    state = v.state()
    print(f"Vertex {i}: state={state}")

    try:
        found = graph.find_vertex(state)
        print(f"  find_vertex({state}) → index {found.index()}")
    except RuntimeError as e:
        print(f"  find_vertex({state}) → ERROR: {e}")

    # Try with explicit numpy array
    try:
        found = graph.find_vertex(np.array(state))
        print(f"  find_vertex(np.array({state})) → index {found.index()}")
    except RuntimeError as e:
        print(f"  find_vertex(np.array({state})) → ERROR: {e}")
