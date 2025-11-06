#!/usr/bin/env python3
"""
Test suite for hierarchical SCC-based caching implementation
Runs without pytest dependency
"""

import sys
import traceback
from phasic import Graph
from phasic.hierarchical_trace_cache import get_trace_hierarchical, get_scc_graphs
from phasic import hash as phasic_hash
from phasic.trace_elimination import record_elimination_trace


class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run_test(self, name, test_func):
        """Run a single test"""
        try:
            print(f"  {name}...", end=" ")
            test_func()
            print("✓")
            self.passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            self.failed += 1
            self.errors.append((name, str(e)))
        except Exception as e:
            print(f"✗ ERROR: {e}")
            self.failed += 1
            self.errors.append((name, f"Exception: {e}"))
            # traceback.print_exc()

    def summary(self):
        """Print summary"""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} tests passed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print(f"{'='*60}")
        return self.failed == 0


def test_scc_simple_acyclic():
    """Test SCC decomposition of simple acyclic graph"""
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v0.add_edge(v1, 1.0)

    scc = g.scc_decomposition()
    assert len(scc) >= 1, "Should have at least one SCC"


def test_scc_cyclic():
    """Test SCC decomposition of graph with cycle"""
    g = Graph(2)
    v0 = g.starting_vertex()
    v1 = g.find_or_create_vertex([1, 0])
    v2 = g.find_or_create_vertex([0, 1])
    v0.add_edge(v1, 1.0)
    v1.add_edge(v2, 1.0)
    v2.add_edge(v1, 1.0)

    scc = g.scc_decomposition()
    assert len(scc) >= 1, "Should have at least one SCC"
    sizes = scc.scc_sizes()
    assert 2 in sizes, "Should have SCC with 2 vertices"


def test_scc_hashing():
    """Test SCC hash computation"""
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v0.add_edge(v1, 1.0)

    scc_graph = g.scc_decomposition()
    hashes = scc_graph.scc_hashes()

    assert isinstance(hashes, list), "Should return list"
    assert len(hashes) == len(scc_graph), "One hash per SCC"
    assert all(len(h) == 64 for h in hashes), "Hashes should be 64 chars"


def test_scc_as_graph():
    """Test extracting SCC as standalone graph"""
    g = Graph(2)
    v0 = g.starting_vertex()
    v1 = g.find_or_create_vertex([1, 0])
    v2 = g.find_or_create_vertex([0, 1])
    v0.add_edge(v1, 1.0)
    v1.add_edge(v2, 1.0)

    scc_graph = g.scc_decomposition()
    for scc in scc_graph.sccs_in_topo_order():
        subgraph = scc.as_graph()
        assert isinstance(subgraph, Graph), "Should return Graph"
        assert subgraph.vertices_length() == scc.size(), "Size should match"


def test_compute_trace_exists():
    """Test that compute_trace method exists"""
    g = Graph(1)
    assert hasattr(g, 'compute_trace'), "compute_trace should exist"


def test_compute_trace_non_hierarchical():
    """Test compute_trace with hierarchical=False"""
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v0.add_edge(v1, 1.0)

    trace = g.compute_trace(hierarchical=False)
    assert trace is not None, "Should return trace"
    assert trace.n_vertices > 0, "Should have vertices"


def test_compute_trace_hierarchical():
    """Test compute_trace with hierarchical=True"""
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v0.add_edge(v1, 1.0)

    trace = g.compute_trace(hierarchical=True)
    assert trace is not None, "Should return trace"
    assert trace.n_vertices > 0, "Should have vertices"


def test_compute_trace_equivalence():
    """Test hierarchical and non-hierarchical produce same results"""
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v0.add_edge(v1, 1.0)

    trace1 = g.compute_trace(hierarchical=False)
    trace2 = g.compute_trace(hierarchical=True)

    assert trace1.n_vertices == trace2.n_vertices, "Same vertex count"
    assert trace1.state_length == trace2.state_length, "Same state length"


def test_forward_compatible_params():
    """Test that Phase 3b parameters are accepted"""
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v0.add_edge(v1, 1.0)

    trace = g.compute_trace(hierarchical=True, min_size=50, parallel='auto')
    assert trace is not None, "Should accept future params"


def test_graph_hashing():
    """Test graph content hashing"""
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v0.add_edge(v1, 1.0)

    hash_result = phasic_hash.compute_graph_hash(g)
    assert hash_result is not None, "Should return hash"
    assert len(hash_result.hash_hex) == 64, "Should be 64 hex chars"


def test_identical_graphs_same_hash():
    """Test that identical graphs have same hash"""
    g1 = Graph(1)
    v0_1 = g1.starting_vertex()
    v1_1 = g1.find_or_create_vertex([1])
    v0_1.add_edge(v1_1, 1.0)

    g2 = Graph(1)
    v0_2 = g2.starting_vertex()
    v1_2 = g2.find_or_create_vertex([1])
    v0_2.add_edge(v1_2, 1.0)

    hash1 = phasic_hash.compute_graph_hash(g1)
    hash2 = phasic_hash.compute_graph_hash(g2)

    assert hash1.hash_hex == hash2.hash_hex, "Same structure = same hash"


def test_get_scc_graphs():
    """Test get_scc_graphs function"""
    g = Graph(2)
    v0 = g.starting_vertex()
    v1 = g.find_or_create_vertex([1, 0])
    v0.add_edge(v1, 1.0)

    sccs = get_scc_graphs(g, min_size=1)
    assert isinstance(sccs, list), "Should return list"
    assert len(sccs) > 0, "Should have SCCs"


def test_backward_compatibility():
    """Test that default behavior is unchanged"""
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v0.add_edge(v1, 1.0)

    # Should work without parameters
    trace = g.compute_trace()
    assert trace is not None, "Default behavior works"

    # Existing API should work
    trace2 = record_elimination_trace(g)
    assert trace2 is not None, "Existing API works"


def main():
    print("="*60)
    print("Hierarchical SCC Caching - Test Suite")
    print("="*60)

    runner = TestRunner()

    print("\n## SCC API Tests")
    runner.run_test("Simple acyclic graph", test_scc_simple_acyclic)
    runner.run_test("Graph with cycle", test_scc_cyclic)
    runner.run_test("SCC hashing", test_scc_hashing)
    runner.run_test("SCC as_graph", test_scc_as_graph)

    print("\n## Compute Trace API Tests")
    runner.run_test("compute_trace exists", test_compute_trace_exists)
    runner.run_test("Non-hierarchical caching", test_compute_trace_non_hierarchical)
    runner.run_test("Hierarchical caching", test_compute_trace_hierarchical)
    runner.run_test("Hierarchical equivalence", test_compute_trace_equivalence)
    runner.run_test("Forward compatible params", test_forward_compatible_params)

    print("\n## Graph Hashing Tests")
    runner.run_test("Graph hash computation", test_graph_hashing)
    runner.run_test("Identical graphs same hash", test_identical_graphs_same_hash)

    print("\n## Module Integration Tests")
    runner.run_test("get_scc_graphs", test_get_scc_graphs)

    print("\n## Backward Compatibility Tests")
    runner.run_test("Default behavior unchanged", test_backward_compatibility)

    success = runner.summary()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
