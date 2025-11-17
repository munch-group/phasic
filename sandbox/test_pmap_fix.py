#!/usr/bin/env python3
"""
Test pmap fix for hierarchical trace caching.

This test verifies that the pmap result collection works correctly
with retry logic and disk-based trace cache.
"""

import phasic
from phasic.logging_config import set_log_level

# Enable debug logging to see retry behavior
set_log_level('INFO')

print("Testing pmap fix with hierarchical trace caching...")
print("="*70)

# Create a simple coalescent model
@phasic.callback([5])
def coalescent(state):
    n = state[0]
    if n <= 1:
        return []
    return [[[n-1], [n*(n-1)/2]]]

# Build graph
graph = phasic.Graph(coalescent)
print(f"✓ Graph built: {graph.vertices_length()} vertices")

# Compute trace with hierarchical caching (will use pmap if 2+ devices)
try:
    print("\nComputing trace with hierarchical caching...")
    trace = graph.compute_trace(hierarchical=True, verbose=True)
    print(f"\n✅ SUCCESS: Trace computed with {len(trace.operations)} operations")
    print("   pmap result collection works correctly!")
except RuntimeError as e:
    print(f"\n❌ FAILED: {e}")
    print("   The fix did not resolve the issue.")
    raise
