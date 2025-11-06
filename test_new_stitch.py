"""
Test new trace stitching with enhanced subgraphs
"""

import numpy as np
from phasic import Graph
from phasic.trace_elimination import record_elimination_trace, evaluate_trace_jax
from phasic.hierarchical_trace_cache import record_enhanced_scc_traces, stitch_scc_traces
import jax.numpy as jnp

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
print(f"Direct trace: {trace_direct.n_vertices} vertices, {len(trace_direct.operations)} operations")

# SCC decomposition
scc_graph = graph.scc_decomposition()
sccs = scc_graph.sccs_in_topo_order()
print(f"Number of SCCs: {len(sccs)}\n")

# Record enhanced SCC traces
print("Recording enhanced SCC traces...")
scc_trace_dict = record_enhanced_scc_traces(scc_graph, param_length=1)

for scc_idx, scc in enumerate(sccs):
    scc_hash = scc.hash()
    scc_trace = scc_trace_dict[scc_hash]
    print(f"  SCC {scc_idx}: {scc_trace.n_vertices} vertices, {len(scc_trace.operations)} operations")

# Stitch traces
print("\nStitching traces...")
trace_stitched = stitch_scc_traces(scc_graph, scc_trace_dict)
print(f"Stitched trace: {trace_stitched.n_vertices} vertices, {len(trace_stitched.operations)} operations")

# Evaluate and compare
theta = jnp.array([1.0])
result_direct = evaluate_trace_jax(trace_direct, theta)
result_stitched = evaluate_trace_jax(trace_stitched, theta)

print(f"\nDirect vertex_rates: {result_direct['vertex_rates']}")
print(f"Stitched vertex_rates: {result_stitched['vertex_rates']}")

print(f"\nMatch: {np.allclose(result_direct['vertex_rates'], result_stitched['vertex_rates'])}")
