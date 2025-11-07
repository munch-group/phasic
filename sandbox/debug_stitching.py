"""
Debug trace stitching
"""

import numpy as np
from phasic import Graph
from phasic.trace_elimination import record_elimination_trace
from phasic.hierarchical_trace_cache import stitch_scc_traces

def callback(state, **kwargs):
    n = state[0]
    if n <= 1:
        return []
    return [(np.array([n - 1]), [n * (n - 1) / 2])]

# Build graph
graph = Graph(callback, ipv=[3], parameterized=True, nr_samples=3)
print(f"Original graph: {graph.vertices_length()} vertices\n")

# Direct trace
trace_direct = record_elimination_trace(graph, param_length=1)
print(f"Direct trace:")
print(f"  n_vertices: {trace_direct.n_vertices}")
print(f"  vertex_rates: {trace_direct.vertex_rates}")
print(f"  n_operations: {len(trace_direct.operations)}\n")

# SCC decomposition and stitching
scc_graph = graph.scc_decomposition()
sccs = scc_graph.sccs_in_topo_order()
print(f"Number of SCCs: {len(sccs)}\n")

scc_trace_dict = {}
for scc_idx, scc in enumerate(sccs):
    scc_hash = scc.hash()
    scc_subgraph = scc.as_graph()
    scc_trace = record_elimination_trace(scc_subgraph, param_length=1)
    scc_trace_dict[scc_hash] = scc_trace
    print(f"SCC {scc_idx}:")
    print(f"  n_vertices: {scc_trace.n_vertices}")
    print(f"  vertex_rates: {scc_trace.vertex_rates}")
    print(f"  n_operations: {len(scc_trace.operations)}")

trace_stitched = stitch_scc_traces(scc_graph, scc_trace_dict)
print(f"\nStitched trace:")
print(f"  n_vertices: {trace_stitched.n_vertices}")
print(f"  vertex_rates: {trace_stitched.vertex_rates}")
print(f"  n_operations: {len(trace_stitched.operations)}")
