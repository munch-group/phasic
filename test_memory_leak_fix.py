#!/usr/bin/env python3
"""
Test that memory doesn't accumulate when running compute_trace() multiple times.
This test verifies the fix for the Jupyter kernel crash issue.
"""

import phasic
from phasic.state_indexing import Property, StateSpace
import numpy as np
import gc
import psutil
import os

# Get current process
process = psutil.Process(os.getpid())

def get_memory_mb():
    """Get current memory usage in MB"""
    return process.memory_info().rss / 1024 / 1024

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

print("=" * 80)
print("Testing memory leak fix: Multiple compute_trace() calls")
print("=" * 80)

# Build graph once
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
print(f"   ✓ Graph built: {graph.vertices_length()} vertices")

# Clear caches before testing
print("\n2. Clearing caches...")
phasic.clear_caches()

# Force garbage collection
gc.collect()

# Get baseline memory
baseline_memory = get_memory_mb()
print(f"   Baseline memory: {baseline_memory:.1f} MB")

# Run compute_trace multiple times and monitor memory
print("\n3. Running compute_trace() 5 times to check for memory leaks...")
memory_readings = []

for i in range(5):
    print(f"\n   Run {i+1}/5:")

    # Clear caches before each run (simulates notebook re-running cells)
    phasic.clear_caches()

    # Compute trace
    trace = graph.compute_trace(hierarchical=True, min_size=10)
    print(f"      ✓ Trace computed: {trace.n_vertices} vertices, {len(trace.operations)} operations")

    # Force garbage collection
    gc.collect()

    # Measure memory
    current_memory = get_memory_mb()
    memory_increase = current_memory - baseline_memory
    memory_readings.append(memory_increase)
    print(f"      Memory: {current_memory:.1f} MB (Δ = {memory_increase:+.1f} MB)")

# Analyze results
print("\n" + "=" * 80)
print("Results:")
print("=" * 80)

print(f"\nMemory increase per run:")
for i, mem in enumerate(memory_readings):
    print(f"   Run {i+1}: {mem:+.1f} MB")

# Check if memory is accumulating
# Allow some tolerance (5 MB) for normal variation
max_increase = max(memory_readings)
min_increase = min(memory_readings)
variation = max_increase - min_increase

print(f"\nMemory variation: {variation:.1f} MB")

if variation > 20:  # More than 20 MB variation suggests a leak
    print("   ✗ MEMORY LEAK DETECTED: Memory increases significantly across runs")
    print(f"   Memory grew from {min_increase:.1f} MB to {max_increase:.1f} MB")
    exit(1)
else:
    print("   ✓ NO MEMORY LEAK: Memory stays relatively constant across runs")
    print(f"   Max variation: {variation:.1f} MB (acceptable)")

print("\n" + "=" * 80)
print("✓✓✓ MEMORY LEAK FIX VERIFIED! ✓✓✓")
print("=" * 80)
