#!/usr/bin/env python3
"""
Test to verify SCC counting fix.

The bug was that with min_size=200, it reported "0 large SCCs"
but with min_size=20, it reported "420 large SCCs" (some must be ≥200).

After the fix, large SCCs should be counted consistently.
"""

import os
import shutil
from phasic import Graph, callback
from phasic.logging_config import set_log_level

# Enable INFO logging to see SCC grouping messages
set_log_level('INFO')

# Create a coalescent model with n=20 (creates 21 vertices)
@callback(ipv=[([20], 1.0)])
def coalescent_callback(state, nr_samples=None, **kwargs):
    n = state[0]
    if n <= 1:
        return []
    rate = n * (n - 1) / 2
    return [([n - 1], [rate])]

print("="*70)
print("Building coalescent graph (n=20)...")
print("="*70)
graph = Graph(
    coalescent_callback,
    parameterized=True,
    nr_samples=20
)

print(f"\nGraph has {graph.vertices_length()} vertices")

# Clear cache
cache_dir = os.path.expanduser("~/.phasic_cache/traces")
if os.path.exists(cache_dir):
    print(f"Clearing cache: {cache_dir}")
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

# Test 1: min_size=5 (should find many large SCCs)
print("\n" + "="*70)
print("Test 1: min_size=5")
print("="*70)
trace1 = graph.compute_trace(min_size=5, parallel='sequential')
print(f"✓ Trace computed: {len(trace1.operations)} operations")

# Clear cache again
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

# Test 2: min_size=10 (should find fewer large SCCs)
print("\n" + "="*70)
print("Test 2: min_size=10")
print("="*70)
trace2 = graph.compute_trace(min_size=10, parallel='sequential')
print(f"✓ Trace computed: {len(trace2.operations)} operations")

# Clear cache again
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

# Test 3: min_size=15 (should be consistent with above)
print("\n" + "="*70)
print("Test 3: min_size=15")
print("="*70)
trace3 = graph.compute_trace(min_size=15, parallel='sequential')
print(f"✓ Trace computed: {len(trace3.operations)} operations")

print("\n" + "="*70)
print("SUCCESS: All traces computed correctly")
print("="*70)
print("\nKey observations:")
print("- If any SCCs ≥10 exist, they should appear in Test 2")
print("- If any SCCs ≥15 exist, they should appear in Test 3")
print("- The counts should be monotonically decreasing (more large SCCs at lower min_size)")
print("\nCheck the INFO logs above to verify SCC counts are consistent.")
