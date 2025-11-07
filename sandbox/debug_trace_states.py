"""
Debug script to understand SCC trace states
"""

import numpy as np
from phasic import Graph
from phasic.trace_elimination import record_elimination_trace

def callback(state, **kwargs):
    n = state[0]
    if n <= 1:
        return []  # Absorbing
    return [(np.array([n - 1]), [1.0])]

# Build graph
graph = Graph(callback, ipv=[3], parameterized=True, nr_samples=3)

print(f"Original graph has {graph.vertices_length()} vertices")
print("Original graph vertex states:")
for i in range(graph.vertices_length()):
    v = graph.vertex_at(i)
    print(f"  Vertex {i}: state={v.state()}")

# Direct trace
print("\n--- Direct trace ---")
trace_direct = record_elimination_trace(graph, param_length=1)
print(f"Direct trace has {trace_direct.n_vertices} vertices")
print("Direct trace states:")
for i in range(trace_direct.n_vertices):
    print(f"  Vertex {i}: state={trace_direct.states[i]}")

# SCC decomposition
print("\n--- SCC decomposition ---")
scc_graph = graph.scc_decomposition()
sccs = scc_graph.sccs_in_topo_order()
print(f"Number of SCCs: {len(sccs)}")

for scc_idx, scc in enumerate(sccs):
    print(f"\n--- SCC {scc_idx} ---")
    scc_subgraph = scc.as_graph()
    print(f"SCC subgraph has {scc_subgraph.vertices_length()} vertices")
    print("SCC subgraph vertex states:")
    for i in range(scc_subgraph.vertices_length()):
        v = scc_subgraph.vertex_at(i)
        print(f"  Vertex {i}: state={v.state()}")

    # Record trace for this SCC
    scc_trace = record_elimination_trace(scc_subgraph, param_length=1)
    print(f"SCC trace has {scc_trace.n_vertices} vertices")
    print("SCC trace states:")
    for i in range(scc_trace.n_vertices):
        print(f"  Vertex {i}: state={scc_trace.states[i]}")

    # Try to find each trace state in original graph
    print("Trying to find trace states in original graph:")
    for i in range(scc_trace.n_vertices):
        state = scc_trace.states[i]
        try:
            orig_v = graph.find_vertex(state)
            print(f"  State {state} found at index {orig_v.index()}")
        except RuntimeError:
            print(f"  State {state} NOT FOUND in original graph")
