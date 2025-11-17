#!/usr/bin/env python3
"""
Test script for Phase 2: Hierarchical SCC-based caching

Tests:
1. Basic SCC decomposition works
2. hierarchical_trace_cache module imports correctly
3. compute_trace() method exists and has correct signature
4. get_trace_hierarchical() can be called (without full functionality)
"""

import sys
sys.path.insert(0, '/Users/kmt/phasic/src')

def test_scc_decomposition():
    """Test basic SCC decomposition"""
    print("Test 1: Basic SCC decomposition...")
    from phasic import Graph

    # Create simple cyclic graph
    g = Graph(2)
    v0 = g.starting_vertex()
    v1 = g.find_or_create_vertex([1, 0])
    v2 = g.find_or_create_vertex([0, 1])
    v0.add_edge(v1, 1.0)
    v1.add_edge(v2, 1.0)
    v2.add_edge(v1, 1.0)  # Cycle between v1 and v2

    # Decompose
    scc_graph = g.scc_decomposition()
    print(f"  ✓ Found {len(scc_graph)} SCCs")
    print(f"  ✓ SCC sizes: {scc_graph.scc_sizes()}")

    # Test hashing
    hashes = scc_graph.scc_hashes()
    print(f"  ✓ Computed {len(hashes)} hashes")

    return True

def test_hierarchical_cache_import():
    """Test hierarchical_trace_cache module imports"""
    print("\nTest 2: Hierarchical cache module import...")
    try:
        from phasic import hierarchical_trace_cache
        from phasic.hierarchical_trace_cache import (
            get_scc_graphs,
            collect_missing_traces_batch,
            get_trace_hierarchical
        )
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_compute_trace_method():
    """Test Graph.compute_trace() method exists"""
    print("\nTest 3: Graph.compute_trace() method...")
    from phasic import Graph

    g = Graph(1)
    assert hasattr(g, 'compute_trace'), "compute_trace method not found"
    print("  ✓ compute_trace() method exists")

    # Test signature
    import inspect
    sig = inspect.signature(g.compute_trace)
    params = list(sig.parameters.keys())
    expected = ['param_length', 'hierarchical', 'min_size', 'parallel']
    assert params == expected, f"Expected params {expected}, got {params}"
    print(f"  ✓ Method signature correct: {params}")

    return True

def test_get_trace_hierarchical_callable():
    """Test get_trace_hierarchical() is callable (may not fully work yet)"""
    print("\nTest 4: get_trace_hierarchical() callable...")
    from phasic.hierarchical_trace_cache import get_trace_hierarchical
    from phasic import Graph

    # Create tiny graph (should fall through to direct computation)
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v0.add_edge(v1, 1.0)

    try:
        # This may fail due to missing trace serialization, but should be callable
        trace = get_trace_hierarchical(g, min_size=100, parallel_strategy='sequential')
        print("  ✓ get_trace_hierarchical() callable and returned result")
        return True
    except NotImplementedError as e:
        if "Trace stitching" in str(e):
            print(f"  ✓ get_trace_hierarchical() callable (stitching not implemented yet)")
            return True
        raise
    except Exception as e:
        print(f"  ! Unexpected error: {e}")
        return False

def test_compute_trace_api():
    """Test Graph.compute_trace() API"""
    print("\nTest 5: Graph.compute_trace() API...")
    from phasic import Graph

    # Create tiny graph
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v0.add_edge(v1, 1.0)

    # Test non-hierarchical (should work with existing code)
    try:
        trace = g.compute_trace(hierarchical=False)
        print("  ✓ compute_trace(hierarchical=False) works")
    except Exception as e:
        print(f"  ! Non-hierarchical failed: {e}")
        return False

    # Test hierarchical (may not work yet due to stitching)
    try:
        trace = g.compute_trace(hierarchical=True, min_size=100, parallel='sequential')
        print("  ✓ compute_trace(hierarchical=True) works")
        return True
    except NotImplementedError as e:
        if "Trace stitching" in str(e):
            print("  ✓ compute_trace(hierarchical=True) callable (stitching not implemented yet)")
            return True
        raise
    except Exception as e:
        print(f"  ! Hierarchical failed: {e}")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("Phase 2: Hierarchical SCC Caching Tests")
    print("=" * 60)

    tests = [
        test_scc_decomposition,
        test_hierarchical_cache_import,
        test_compute_trace_method,
        test_get_trace_hierarchical_callable,
        test_compute_trace_api,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    sys.exit(0 if all(results) else 1)
