"""
Debug SCC 0 structure
"""

import numpy as np
from phasic import Graph
from phasic.trace_elimination import record_elimination_trace

def callback(state, **kwargs):
    n = state[0]
    if n <= 1:
        return []
    return [(np.array([n - 1]), [n * (n - 1) / 2])]

# Build graph
graph = Graph(callback, ipv=[3], parameterized=True, nr_samples=3)

# SCC decomposition
scc_graph = graph.scc_decomposition()
sccs = scc_graph.sccs_in_topo_order()

# Check SCC 0
scc0 = sccs[0]
print(f"SCC 0 internal indices: {scc0.internal_vertex_indices()}")
print(f"SCC 0 size: {scc0.size()}")

scc0_subgraph = scc0.as_graph()
print(f"\nSCC 0 subgraph:")
print(f"  vertices_length: {scc0_subgraph.vertices_length()}")
for i in range(scc0_subgraph.vertices_length()):
    v = scc0_subgraph.vertex_at(i)
    edges = v.parameterized_edges() if scc0_subgraph.parameterized() else v.edges()
    print(f"  Vertex {i}: state={v.state()}, {len(edges)} edges")

# Record trace
scc0_trace = record_elimination_trace(scc0_subgraph, param_length=1)
print(f"\nSCC 0 trace:")
print(f"  n_vertices: {scc0_trace.n_vertices}")
print(f"  states: {[tuple(scc0_trace.states[i]) for i in range(scc0_trace.n_vertices)]}")
print(f"  vertex_rates: {scc0_trace.vertex_rates}")
