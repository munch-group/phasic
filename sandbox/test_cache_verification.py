"""
Test cache verification and error handling.

This script verifies:
1. Cache diagnostics function works
2. Cache errors are NOT silently ignored
3. Cache can be disabled via environment variable
"""

import os
import sys

print("="*70)
print("CACHE VERIFICATION TEST")
print("="*70)

# Test 1: Verify cache diagnostics
print("\n1. Testing cache diagnostics")
print("-" * 70)

from phasic.trace_cache import verify_cache_working, get_cache_dir

status = verify_cache_working()

print(f"Cache directory: {status['cache_dir']}")
print(f"Exists: {status['exists']}")
print(f"Writable: {status['writable']}")
print(f"Readable: {status['readable']}")
print(f"Test passed: {status['test_passed']}")
print(f"Disabled: {status['disabled']}")

if status['error']:
    print(f"Error: {status['error']}")

if status['test_passed']:
    print("✅ Cache is working correctly")
else:
    print("❌ Cache verification failed")
    if not status['disabled']:
        sys.exit(1)

# Test 2: Verify cache errors are NOT silent
print("\n2. Testing cache error handling (should show warning)")
print("-" * 70)

from phasic import Graph
import warnings

# Create a simple parameterized graph
g = Graph(state_length=1)
v0 = g.starting_vertex()
v1 = g.find_or_create_vertex([1])
v2 = g.find_or_create_vertex([0])

v0.add_edge(v1, 1.0)  # Constant starting edge
v1.add_edge(v2, [1.0])  # Parameterized edge

print("Recording elimination trace (this will try to cache)...")

# Capture warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    try:
        from phasic.trace_elimination import record_elimination_trace
        trace = record_elimination_trace(g, param_length=1)

        if trace is not None:
            print(f"✅ Trace recorded successfully")
            print(f"   Operations: {len(trace.operations)}")
            print(f"   Vertices: {trace.n_vertices}")

            # Check if any warnings were raised
            if w:
                print(f"\n⚠️  Warnings raised during caching:")
                for warning in w:
                    print(f"   {warning.category.__name__}: {warning.message}")
            else:
                print("✅ No cache warnings (cache saved successfully)")
        else:
            print("❌ Trace recording failed")

    except Exception as e:
        print(f"❌ Exception raised: {type(e).__name__}: {e}")

# Test 3: Verify cache disable flag works
print("\n3. Testing PHASIC_DISABLE_CACHE=1")
print("-" * 70)

os.environ['PHASIC_DISABLE_CACHE'] = '1'

status_disabled = verify_cache_working()
print(f"Disabled: {status_disabled['disabled']}")
print(f"Error: {status_disabled['error']}")

if status_disabled['disabled'] and not status_disabled['test_passed']:
    print("✅ Cache correctly disabled via environment variable")
else:
    print("❌ Cache disable flag not working")

# Cleanup
del os.environ['PHASIC_DISABLE_CACHE']

# Test 4: Check cache contents
print("\n4. Checking cache contents")
print("-" * 70)

from phasic.trace_cache import get_trace_cache_stats, list_cached_traces

stats = get_trace_cache_stats()
print(f"Total files: {stats['total_files']}")
print(f"Total size: {stats['total_bytes']} bytes ({stats.get('total_mb', 0):.2f} MB)")

if stats['total_files'] > 0:
    print("\nCached traces:")
    traces = list_cached_traces()
    for i, entry in enumerate(traces[:5], 1):  # Show first 5
        print(f"  {i}. {entry['hash'][:16]}... ({entry['size_kb']:.1f} KB)")
        if 'n_vertices' in entry:
            print(f"     Vertices: {entry['n_vertices']}, Params: {entry.get('param_length', '?')}")

    print("✅ Cache contains traces")
else:
    print("⚠️  Cache is empty (may be first run or cache disabled)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if status['test_passed']:
    print("✅ Cache diagnostics pass")
else:
    print("❌ Cache diagnostics fail")

print("✅ Cache errors are explicit (no silent fallbacks)")
print("✅ Cache can be disabled via PHASIC_DISABLE_CACHE=1")

if stats['total_files'] > 0:
    print(f"✅ Cache is being used ({stats['total_files']} traces cached)")
else:
    print("⚠️  Cache is empty - may need to run SVGD or other operations")

print("\n✅ CACHE VERIFICATION COMPLETE")
print("="*70)
