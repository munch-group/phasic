"""
Debug boundary edges
"""

import numpy as np
from phasic import Graph
from phasic.trace_elimination import record_elimination_trace
from phasic.hierarchical_trace_cache import stitch_scc_traces, _add_boundary_edges

def callback(state, **kwargs):
    n = state[0]
    if n <= 1:
        return []
    return [(np.array([n - 1]), [n * (n - 1) / 2])]

# Build graph
graph = Graph(callback, ipv=[3], parameterized=True, nr_samples=3)
print(f"Original graph: {graph.vertices_length()} vertices")
print("Edges:")
for i in range(graph.vertices_length()):
    v = graph.vertex_at(i)
    edges = v.parameterized_edges()
    print(f"  Vertex {i} ({v.state()}): {len(edges)} edges")
    for edge in edges:
        print(f"    -> Vertex {edge.to().index()} ({edge.to().state()})")
print()

# SCC decomposition
scc_graph = graph.scc_decomposition()
sccs = scc_graph.sccs_in_topo_order()

# Check internal indices
print("SCC internal indices:")
for scc_idx, scc in enumerate(sccs):
    print(f"  SCC {scc_idx}: {scc.internal_vertex_indices()}")
print()
