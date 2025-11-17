#!/usr/bin/env python3
"""Debug stitching to see where duplicate edges come from."""

import shutil, os
cache_dir = os.path.expanduser('~/.phasic_cache/traces')
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

import phasic
from phasic.logging_config import set_log_level
import numpy as np

set_log_level('WARNING')  # Reduce noise

# Build test graph
def scc_callback(state, **kwargs):
    s = state[0]
    if s == 0: return [[np.array([1]), [1.0]]]
    elif s == 1: return [[np.array([2]), [1.0]]]
    elif s == 2: return [[np.array([3]), [1.0]]]
    elif s == 3: return [[np.array([2]), [1.0]], [np.array([4]), [1.0]], [np.array([6]), [1.0]]]
    elif s == 4: return [[np.array([5]), [1.0]], [np.array([7]), [1.0]]]
    elif s == 5: return [[np.array([6]), [1.0]], [np.array([7]), [1.0]]]
    elif s == 6: return [[np.array([4]), [1.0]]]
    elif s == 7: return []

graph = phasic.Graph(scc_callback, ipv=[[np.array([1]), 1.0]], param_length=1)

print("="*80)
print("ORIGINAL GRAPH EDGES")
print("="*80)
for i in range(graph.vertices_length()):
    v = graph.vertex_at(i)
    edges = v.parameterized_edges()
    targets = [e.to().index() for e in edges]
    print(f"Vertex {i}: {len(edges)} edges → {targets}")

expected_edges = {
    0: [1],
    1: [2],
    2: [3],
    3: [2, 4, 5],  # 5 is vertex with state=[6], which is E
    4: [6, 7],     # 6 is vertex with state=[5], which is D
    5: [4],        # 5 is E, connects back to C (vertex 4)
    6: [5, 7],     # 6 is D, connects to E and F
    7: []
}

print("\nExpected structure:")
for v, targets in expected_edges.items():
    print(f"  Vertex {v} → {targets}")

# Get SCC decomposition
scc_decomp = graph.scc_decomposition()
sccs = list(scc_decomp.sccs_in_topo_order())

print(f"\n{'='*80}")
print("SCC DECOMPOSITION")
print("="*80)
for i, scc in enumerate(sccs):
    internal = scc.internal_vertex_indices()
    states = [tuple(graph.vertex_at(v).state()) for v in internal]
    print(f"SCC {i+1}: internal={internal}, states={states}")

# Manually build and inspect each SCC trace
from phasic.hierarchical_trace_cache import _build_first_scc_subgraph, _build_scc_subgraph
from phasic.trace_elimination import record_elimination_trace

print(f"\n{'='*80}")
print("INDIVIDUAL SCC TRACES")
print("="*80)

scc_traces = []
scc_metadatas = []

for i, scc in enumerate(sccs):
    print(f"\n--- SCC {i+1} ---")

    if i == 0:
        enhanced, metadata = _build_first_scc_subgraph(
            graph, graph.starting_vertex().index(), scc, scc_decomp
        )
    else:
        enhanced, metadata = _build_scc_subgraph(graph, scc, scc_decomp)

    trace = record_elimination_trace(enhanced, param_length=graph.param_length())

    ordered = metadata['ordered_vertices']
    print(f"ordered_vertices: {ordered}")
    print(f"Trace: {trace.n_vertices} vertices")

    for trace_idx in range(trace.n_vertices):
        if trace.vertex_rates[trace_idx] >= 0:
            targets = trace.vertex_targets[trace_idx]
            # Map to original indices
            orig_idx = ordered[trace_idx] if trace_idx < len(ordered) else None
            orig_targets = [ordered[t] if t < len(ordered) else None for t in targets]
            print(f"  trace[{trace_idx}] (orig={orig_idx}): {len(targets)} edges → trace{targets} (orig={orig_targets})")

    scc_traces.append(trace)
    scc_metadatas.append(metadata)

print(f"\n{'='*80}")
print("NOW STITCH AND SEE WHERE DUPLICATES COME FROM")
print("="*80)
print("\nLet's manually trace vertex 4 (C) through the stitching:")
print("Expected: 2 edges to [6, 7] (D and F)")

# Find vertex 4 in each SCC
for i, (trace, metadata) in enumerate(zip(scc_traces, scc_metadatas)):
    ordered = metadata['ordered_vertices']
    if 4 in ordered:
        trace_idx = ordered.index(4)
        targets = trace.vertex_targets[trace_idx]
        orig_targets = [ordered[t] if t < len(ordered) else None for t in targets]
        print(f"\nSCC {i+1}: vertex 4 appears at trace[{trace_idx}]")
        print(f"  Edges: {len(targets)} → trace{targets} (orig={orig_targets})")
        print(f"  Categories: upstream={4 in metadata.get('upstream', [])}, "
              f"internal={4 in metadata.get('internal', [])}, "
              f"downstream={4 in metadata.get('downstream', [])}")
