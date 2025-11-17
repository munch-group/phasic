#!/usr/bin/env python3
"""
Test that mimics the notebook use case more closely.
This creates a larger graph that would trigger hierarchical caching.
"""

import phasic
from phasic.logging_config import set_log_level

# Enable info logging to see what's happening
set_log_level('INFO')

print("Creating larger coalescent graph (will trigger hierarchical caching)...")
print("=" * 70)

# Create a coalescent model that's big enough to trigger SCCs
@phasic.callback([15])
def coalescent_large(state):
    n = state[0]
    if n <= 1:
        return []
    return [[[n-1], [n*(n-1)/2]]]

# Build graph
graph = phasic.Graph(coalescent_large)
print(f"✓ Graph built: {graph.vertices_length()} vertices")

# Test compute_trace (this is what the notebook calls)
print("\n" + "=" * 70)
print("Computing trace with hierarchical caching (this may trigger multiprocessing)...")
print("=" * 70)

try:
    trace = graph.compute_trace()
    print(f"\n✅ SUCCESS!")
    print(f"   Trace computed: {len(trace.operations)} operations")
    print(f"   Graph had {graph.vertices_length()} vertices")
    print(f"\nThis is exactly what your notebook does - graph.compute_trace()")
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n" + "=" * 70)
print("✅ NOTEBOOK USE CASE TEST PASSED")
print("=" * 70)
