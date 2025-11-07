"""
Debug trace operations
"""

import numpy as np
from phasic import Graph
from phasic.trace_elimination import record_elimination_trace, evaluate_trace_jax
from phasic.hierarchical_trace_cache import stitch_scc_traces
import jax.numpy as jnp

def callback(state, **kwargs):
    n = state[0]
    if n <= 1:
        return []
    return [(np.array([n - 1]), [n * (n - 1) / 2])]

# Build graph
graph = Graph(callback, ipv=[3], parameterized=True, nr_samples=3)

# Direct trace
trace_direct = record_elimination_trace(graph, param_length=1)
print("Direct trace operations:")
for i, op in enumerate(trace_direct.operations[:10]):  # First 10
    print(f"  Op {i}: type={op.op_type.name if hasattr(op.op_type, 'name') else op.op_type}")

# SCC stitched
scc_graph = graph.scc_decomposition()
sccs = scc_graph.sccs_in_topo_order()
scc_trace_dict = {}
for scc in sccs:
    scc_hash = scc.hash()
    scc_subgraph = scc.as_graph()
    scc_trace = record_elimination_trace(scc_subgraph, param_length=1)
    scc_trace_dict[scc_hash] = scc_trace

trace_stitched = stitch_scc_traces(scc_graph, scc_trace_dict)
print(f"\nStitched trace operations ({len(trace_stitched.operations)} total):")
for i, op in enumerate(trace_stitched.operations):
    print(f"  Op {i}: type={op.op_type.name if hasattr(op.op_type, 'name') else op.op_type}")

# Evaluate
theta = jnp.array([1.0])
result_direct = evaluate_trace_jax(trace_direct, theta)
result_stitched = evaluate_trace_jax(trace_stitched, theta)

print(f"\nDirect vertex_rates: {result_direct['vertex_rates']}")
print(f"Stitched vertex_rates: {result_stitched['vertex_rates']}")
