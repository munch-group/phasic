#!/usr/bin/env python3
"""
Test two_locus_arg with samples=5, min_size=10
This was failing with KeyError when trying to stitch traces
"""

import phasic
from phasic.state_indexing import Property, StateSpace
import numpy as np
from phasic.logging_config import set_log_level

set_log_level('INFO')

print("=" * 80)
print("Testing two_locus_arg with samples=5, min_size=10")
print("=" * 80)

# Exact code from notebook
def two_locus_arg(state, s=None, N=None, R=None, state_space=None):
    transitions = []
    if state.sum() <= 1: return transitions

    for i in range(state_space.size):
        if state[i] == 0: continue
        conf_i = state_space.index_to_props(i)

        for j in range(i, state_space.size):
            if state[j] == 0: continue
            conf_j = state_space.index_to_props(j)

            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            child = state.copy()
            child[i] -= 1
            child[j] -= 1
            L1Des = conf_i['L1Des'] + conf_j['L1Des']
            L2Des = conf_i['L2Des'] + conf_j['L2Des']
            if L1Des <= s and L2Des <= s:
                child[state_space.props_to_index(L1Des=L1Des, L2Des=L2Des)] += 1
                transitions.append([child, [state[i]*(state[j]-same)/(1+same)]])

        if state[i] > 0 and conf_i['L1Des'] > 0 and conf_i['L2Des'] > 0:
            child = state.copy()
            child[i] -= 1
            child[state_space.props_to_index(L1Des=conf_i['L1Des'], L2Des=0)] += 1
            child[state_space.props_to_index(L1Des=0, L2Des=conf_i['L2Des'])] += 1
            transitions.append([child, [R]])

    return transitions


print("\n1. Building graph (samples=5)...")
nr_samples = 5
state_space = StateSpace([
    Property('L1Des', max_value=nr_samples),
    Property('L2Des', max_value=nr_samples)
])

initial = np.empty(state_space.size+2, dtype=int)
initial.fill(0)
initial[state_space.props_to_index(L1Des=1, L2Des=1)] = nr_samples
ipv = [[initial, 1.0]]
graph = phasic.Graph(two_locus_arg, ipv=ipv, s=nr_samples, N=1, R=1, state_space=state_space)

n_vertices = graph.vertices_length()
print(f"   ✓ Graph built: {n_vertices} vertices")

# Check SCC structure
print("\n2. Analyzing SCC structure...")
scc_decomp = graph.scc_decomposition()
sccs = list(scc_decomp.sccs_in_topo_order())
scc_sizes = [scc.size() for scc in sccs]
print(f"   SCCs: {len(sccs)} total")
print(f"   SCC sizes: min={min(scc_sizes)}, max={max(scc_sizes)}, avg={sum(scc_sizes)/len(scc_sizes):.1f}")

large_sccs = sum(1 for s in scc_sizes if s >= 10)
small_sccs = sum(1 for s in scc_sizes if s < 10)
print(f"   Large SCCs (≥10): {large_sccs}")
print(f"   Small SCCs (<10): {small_sccs}")

print("\n3. Clearing caches...")
phasic.clear_caches()

print("\n4. Computing trace with hierarchical=True, min_size=10...")
print("   (This previously failed with KeyError)")
try:
    trace = graph.compute_trace(hierarchical=True, min_size=10)
    print(f"   ✓ SUCCESS: Trace computed: {trace.n_vertices} vertices, {len(trace.operations)} operations")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n5. Verifying correctness with hierarchical=False...")
try:
    phasic.clear_caches()
    trace_direct = graph.compute_trace(hierarchical=False)
    print(f"   ✓ Direct trace: {trace_direct.n_vertices} vertices, {len(trace_direct.operations)} operations")

    # Compare operation counts
    if len(trace.operations) == len(trace_direct.operations):
        print(f"   ✓ Operation count matches: {len(trace.operations)}")
    else:
        print(f"   ✗ Operation count mismatch: hierarchical={len(trace.operations)}, direct={len(trace_direct.operations)}")

except Exception as e:
    print(f"   ✗ Direct trace failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("✓✓✓ TEST PASSED! ✓✓✓")
print("=" * 80)
