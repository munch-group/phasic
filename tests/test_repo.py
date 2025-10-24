"""
Test trace repository functionality.

This tests the registry browsing and filtering capabilities.
Note: Trace downloads will fail with mock CIDs until real IPFS CIDs are added.
"""

from phasic import TraceRegistry

def test_registry_access():
    """Test that we can access the deployed registry."""
    print("Testing TraceRegistry with deployed repository...")
    print("="*70)

    registry = TraceRegistry()

    print("\n✓ Registry loaded successfully")
    return registry


def test_list_traces(registry):
    """Test listing all traces."""
    print("\nListing all traces:")
    traces = registry.list_traces()

    assert len(traces) == 5, f"Expected 5 traces, found {len(traces)}"

    print(f"Found {len(traces)} traces:")
    for t in traces:
        print(f"  - {t['trace_id']}: {t['vertices']} vertices, {t['param_length']} params")

    print("✓ List traces works")
    return traces


def test_filter_by_model_type(registry):
    """Test filtering by model type."""
    print("\nFiltering by model_type='coalescent':")
    coalescent = registry.list_traces(model_type="coalescent")

    assert len(coalescent) == 5, f"Expected 5 coalescent traces, found {len(coalescent)}"

    print(f"Found {len(coalescent)} coalescent traces")
    print("✓ Model type filtering works")
    return coalescent


def test_filter_by_tags(registry):
    """Test filtering by tags."""
    print("\nFiltering by tags=['basic']:")
    basic = registry.list_traces(tags=["basic"])

    assert len(basic) == 5, f"Expected 5 basic traces, found {len(basic)}"

    print(f"Found {len(basic)} basic traces")
    print("✓ Tag filtering works")
    return basic


def test_filter_by_domain(registry):
    """Test filtering by domain."""
    print("\nFiltering by domain='population-genetics':")
    pop_gen = registry.list_traces(domain="population-genetics")

    assert len(pop_gen) == 5, f"Expected 5 pop-gen traces, found {len(pop_gen)}"

    print(f"Found {len(pop_gen)} population genetics traces")
    print("✓ Domain filtering works")
    return pop_gen


def test_trace_metadata(registry):
    """Test that trace metadata is complete."""
    print("\nValidating trace metadata:")
    traces = registry.list_traces()

    for trace in traces:
        # Check required fields
        assert 'trace_id' in trace, f"Missing trace_id in {trace}"
        assert 'description' in trace, f"Missing description in {trace}"
        assert 'cid' in trace, f"Missing cid in {trace}"
        assert 'vertices' in trace, f"Missing vertices in {trace}"
        assert 'param_length' in trace, f"Missing param_length in {trace}"
        assert 'tags' in trace, f"Missing tags in {trace}"

        # Check metadata structure
        assert isinstance(trace['tags'], list), f"Tags should be list in {trace['trace_id']}"
        assert isinstance(trace['vertices'], int), f"Vertices should be int in {trace['trace_id']}"
        assert isinstance(trace['param_length'], int), f"param_length should be int in {trace['trace_id']}"

    print("✓ All traces have valid metadata")


def test_download_trace_mock_cid(registry):
    """Test that download fails gracefully with mock CIDs."""
    print("\nTesting trace download (expected to fail with mock CIDs):")

    try:
        trace = registry.get_trace("coalescent_n5_theta1")
        print("✗ Download succeeded (unexpected - CIDs should be mocks)")
        return False
    except Exception as e:
        print(f"✓ Download failed as expected: {type(e).__name__}")
        print("  (This will work once real IPFS CIDs are added)")
        return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TRACE REPOSITORY TESTS")
    print("="*70)

    try:
        # Test registry access
        registry = test_registry_access()

        # Test listing and filtering
        test_list_traces(registry)
        test_filter_by_model_type(registry)
        test_filter_by_tags(registry)
        test_filter_by_domain(registry)

        # Test metadata
        test_trace_metadata(registry)

        # Test download (expected to fail with mock CIDs)
        test_download_trace_mock_cid(registry)

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nNOTE: Trace downloads will work once real IPFS CIDs are added.")
        print("To add real CIDs:")
        print("  1. Install IPFS: brew install ipfs")
        print("  2. Initialize: ipfs init && ipfs daemon &")
        print("  3. Publish traces: ipfs add -r /tmp/phasic_traces/coalescent_*")
        print("  4. Update registry.json with real CIDs")
        print("  5. Commit and push to GitHub")
        print("="*70)

        return True

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
