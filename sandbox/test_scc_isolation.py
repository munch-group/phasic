"""
Test if starting vertex isolation works
"""

import numpy as np
from phasic import Graph

def callback(state, **kwargs):
    n = state[0]
    if n <= 1:
        return []
    return [(np.array([n - 1]), [1.0])]

# Build graph
graph = Graph(callback, ipv=[3], parameterized=True, nr_samples=3)

print(f"Original graph: {graph.vertices_length()} vertices")
print(f"Starting vertex index: {graph.starting_vertex().index()}")
print(f"Starting vertex state: {graph.starting_vertex().state()}\n")

# SCC decomposition
scc_graph = graph.scc_decomposition()
sccs = scc_graph.sccs_in_topo_order()

print(f"Number of SCCs: {len(sccs)}\n")

for scc_idx, scc in enumerate(sccs):
    print(f"--- SCC {scc_idx} ---")
    print(f"Size: {scc.size()}")

    # Check internal vertex indices
    internal_indices = scc.internal_vertex_indices()
    print(f"Internal vertex indices: {internal_indices}")

    # Get the actual vertices
    for idx in internal_indices:
        v = graph.vertex_at(idx)
        print(f"  Vertex {idx}: state={v.state()}")

    print()

# Check if SCC 0 is the starting vertex only
print("Expected: SCC 0 should contain only the starting vertex")
print(f"SCC 0 size: {sccs[0].size()}")
print(f"SCC 0 internal indices: {sccs[0].internal_vertex_indices()}")
starting_idx = graph.starting_vertex().index()
print(f"Starting vertex index: {starting_idx}")
print(f"Match: {sccs[0].internal_vertex_indices() == [starting_idx]}")
