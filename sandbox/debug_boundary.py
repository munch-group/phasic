"""
Debug boundary edge detection
"""

import numpy as np
from phasic import Graph
from phasic.trace_elimination import record_elimination_trace
from phasic.hierarchical_trace_cache import _build_vertex_mappings

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

# Build traces
scc_trace_dict = {}
for scc in sccs:
    scc_hash = scc.hash()
    scc_subgraph = scc.as_graph()
    scc_trace = record_elimination_trace(scc_subgraph, param_length=1)
    scc_trace_dict[scc_hash] = scc_trace

# Build vertex mappings
vertex_to_original, original_to_merged = _build_vertex_mappings(scc_graph, scc_trace_dict)

print("Vertex mappings:")
for (scc_idx, scc_v_idx), orig_v_idx in sorted(vertex_to_original.items()):
    merged_idx = original_to_merged[orig_v_idx]
    print(f"  SCC {scc_idx}, trace vertex {scc_v_idx} -> orig {orig_v_idx} -> merged {merged_idx}")
print()

# Check SCC vertex sets
print("SCC vertex sets (from internal indices):")
for scc_idx, scc in enumerate(sccs):
    internal_states = set()
    internal_indices = scc.internal_vertex_indices()
    for orig_v_idx in internal_indices:
        v = graph.vertex_at(orig_v_idx)
        internal_states.add(tuple(v.state()))
    print(f"  SCC {scc_idx}: {internal_states}")
print()

# Check which edges are boundary edges
print("Checking edges:")
for scc_idx, scc in enumerate(sccs):
    internal_states = set()
    internal_indices = scc.internal_vertex_indices()
    for orig_v_idx in internal_indices:
        v = graph.vertex_at(orig_v_idx)
        internal_states.add(tuple(v.state()))
    
    print(f"SCC {scc_idx} internal states: {internal_states}")
    
    for (map_scc_idx, _), orig_v_idx in vertex_to_original.items():
        if map_scc_idx != scc_idx:
            continue
        
        orig_vertex = graph.vertex_at(orig_v_idx)
        edges = orig_vertex.parameterized_edges()
        print(f"  Vertex {orig_v_idx} has {len(edges)} edges:")
        
        for edge in edges:
            target_vertex = edge.to()
            target_state = tuple(target_vertex.state())
            is_boundary = target_state not in internal_states
            print(f"    -> {target_vertex.index()} (state={target_state}) boundary={is_boundary}")
