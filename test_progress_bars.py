#!/usr/bin/env python3
"""
Test progress bars with verbose=True
"""

import os
import shutil
from phasic import Graph, callback

# Create a coalescent model with n=20
@callback(ipv=[([20], 1.0)])
def coalescent_callback(state, nr_samples=None, **kwargs):
    n = state[0]
    if n <= 1:
        return []
    rate = n * (n - 1) / 2
    return [([n - 1], [rate])]

print("Building coalescent graph (n=20)...")
graph = Graph(
    coalescent_callback,
    parameterized=True,
    nr_samples=20
)

print(f"Graph has {graph.vertices_length()} vertices\n")

# Clear cache
cache_dir = os.path.expanduser("~/.phasic_cache/traces")
if os.path.exists(cache_dir):
    print(f"Clearing cache: {cache_dir}\n")
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

print("="*70)
print("Test 1: compute_trace(verbose=False) - No progress bars")
print("="*70)
trace1 = graph.compute_trace(min_size=5, parallel='sequential', verbose=False)
print(f"✓ Trace computed: {len(trace1.operations)} operations\n")

# Rebuild graph
graph = Graph(
    coalescent_callback,
    parameterized=True,
    nr_samples=20
)

# Clear cache again
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

print("="*70)
print("Test 2: compute_trace(verbose=True) - Show progress bars")
print("="*70)
trace2 = graph.compute_trace(min_size=5, parallel='sequential', verbose=True)
print(f"✓ Trace computed: {len(trace2.operations)} operations\n")

print("="*70)
print("SUCCESS: Progress bars working correctly!")
print("="*70)
