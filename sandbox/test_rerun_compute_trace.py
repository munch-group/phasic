#!/usr/bin/env python3
"""
Test that graph.compute_trace() can be called multiple times without error.
This verifies the fix for the "ValueError: Graph has no vertices" issue.
"""

import phasic
from phasic.state_indexing import Property, StateSpace
import numpy as np

def coalescent(state, nr_samples=None):
    transitions = []
    n = int(nr_samples)
    for i in range(n):
        for j in range(i, n):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[i+j+1] += 1
            transitions.append([new, [state[i]*(state[j]-same)/(1+same)]])
    return transitions

print("=" * 80)
print("Testing graph.compute_trace() can be called multiple times")
print("=" * 80)

nr_samples = 4
ipv = [[[nr_samples] + [0] * (nr_samples+1), 1.0]]

print("\n1. Building graph...")
graph = phasic.Graph(coalescent, ipv=ipv, nr_samples=nr_samples)
print(f"   ✓ Graph: {graph.vertices_length()} vertices")

print("\n2. First compute_trace() call...")
trace1 = graph.compute_trace()
print(f"   ✓ Trace 1: {trace1.n_vertices} vertices, {len(trace1.operations)} operations")
print(f"   ✓ Graph still has {graph.vertices_length()} vertices (not destroyed)")

print("\n3. Second compute_trace() call (should work now)...")
try:
    trace2 = graph.compute_trace()
    print(f"   ✓ Trace 2: {trace2.n_vertices} vertices, {len(trace2.operations)} operations")
    print(f"   ✓ Graph still has {graph.vertices_length()} vertices")
except ValueError as e:
    print(f"   ✗ FAILED: {e}")
    exit(1)

print("\n4. Third compute_trace() call (just to be sure)...")
try:
    trace3 = graph.compute_trace()
    print(f"   ✓ Trace 3: {trace3.n_vertices} vertices, {len(trace3.operations)} operations")
    print(f"   ✓ Graph still has {graph.vertices_length()} vertices")
except ValueError as e:
    print(f"   ✗ FAILED: {e}")
    exit(1)

print("\n5. Verifying all traces are identical...")
assert trace1.n_vertices == trace2.n_vertices == trace3.n_vertices
assert len(trace1.operations) == len(trace2.operations) == len(trace3.operations)
print("   ✓ All traces match")

print("\n6. Testing with hierarchical=False...")
graph2 = phasic.Graph(coalescent, ipv=ipv, nr_samples=nr_samples)
try:
    trace_a = graph2.compute_trace(hierarchical=False)
    print(f"   ✓ First call: {trace_a.n_vertices} vertices")
    trace_b = graph2.compute_trace(hierarchical=False)
    print(f"   ✓ Second call: {trace_b.n_vertices} vertices")
    print(f"   ✓ Graph still has {graph2.vertices_length()} vertices")
except ValueError as e:
    print(f"   ✗ FAILED: {e}")
    exit(1)

print("\n" + "=" * 80)
print("✓✓✓ TEST PASSED: compute_trace() works multiple times! ✓✓✓")
print("=" * 80)
