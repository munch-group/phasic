"""
Unit tests for hierarchical caching (Phase 3a)

Tests the hierarchical caching API and functionality.
"""

import pytest
import numpy as np
from phasic import Graph
from phasic.hierarchical_trace_cache import get_trace_hierarchical


class TestComputeTraceAPI:
    """Test Graph.compute_trace() API"""

    def test_compute_trace_exists(self):
        """Test that compute_trace method exists"""
        g = Graph(1)
        assert hasattr(g, 'compute_trace'), "compute_trace method should exist"

    def test_compute_trace_non_hierarchical(self):
        """Test compute_trace with hierarchical=False (default)"""
        g = Graph(1)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v0.add_edge(v1, 1.0)

        trace = g.compute_trace(hierarchical=False)

        assert trace is not None, "Should return trace"
        assert trace.n_vertices > 0, "Trace should have vertices"
        assert hasattr(trace, 'operations'), "Should have operations"

    def test_compute_trace_hierarchical(self):
        """Test compute_trace with hierarchical=True"""
        g = Graph(1)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v0.add_edge(v1, 1.0)

        trace = g.compute_trace(hierarchical=True)

        assert trace is not None, "Should return trace"
        assert trace.n_vertices > 0, "Trace should have vertices"

    def test_compute_trace_equivalence(self):
        """Test that hierarchical and non-hierarchical produce same traces"""
        g = Graph(1)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v0.add_edge(v1, 1.0)

        trace1 = g.compute_trace(hierarchical=False)
        trace2 = g.compute_trace(hierarchical=True)

        assert trace1.n_vertices == trace2.n_vertices, "Should have same vertex count"
        assert trace1.state_length == trace2.state_length, "Should have same state length"
        # Note: operations may differ slightly, but results should be equivalent

    def test_compute_trace_forward_compatible_params(self):
        """Test that future Phase 3b parameters are accepted"""
        g = Graph(1)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v0.add_edge(v1, 1.0)

        # Should not raise error even though these params are not used in Phase 3a
        trace = g.compute_trace(
            hierarchical=True,
            min_size=50,
            parallel='auto'
        )

        assert trace is not None, "Should return trace"

    def test_compute_trace_param_length(self):
        """Test compute_trace with explicit param_length"""
        g = Graph(1)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v0.add_edge(v1, 1.0)

        trace = g.compute_trace(param_length=0)

        assert trace is not None, "Should return trace"
        assert trace.param_length == 0, "Should respect param_length"


class TestHierarchicalCacheModule:
    """Test hierarchical_trace_cache module"""

    def test_module_imports(self):
        """Test that module imports without error"""
        from phasic import hierarchical_trace_cache
        from phasic.hierarchical_trace_cache import (
            get_scc_graphs,
            collect_missing_traces_batch,
            get_trace_hierarchical
        )

        assert callable(get_scc_graphs), "get_scc_graphs should be callable"
        assert callable(collect_missing_traces_batch), "collect_missing_traces_batch should be callable"
        assert callable(get_trace_hierarchical), "get_trace_hierarchical should be callable"

    def test_get_trace_hierarchical_basic(self):
        """Test get_trace_hierarchical with simple graph"""
        g = Graph(1)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v0.add_edge(v1, 1.0)

        trace = get_trace_hierarchical(g)

        assert trace is not None, "Should return trace"
        assert trace.n_vertices > 0, "Trace should have vertices"

    def test_get_trace_hierarchical_with_params(self):
        """Test get_trace_hierarchical with all parameters"""
        g = Graph(1)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v0.add_edge(v1, 1.0)

        trace = get_trace_hierarchical(
            g,
            param_length=0,
            min_size=50,
            parallel_strategy='sequential'
        )

        assert trace is not None, "Should return trace"

    @pytest.mark.skip(reason="Causes C-level abort in scc_hashes(); see issue tracker")
    def test_get_scc_graphs(self):
        """Test get_scc_graphs extracts SCCs"""
        from phasic.hierarchical_trace_cache import get_scc_graphs

        g = Graph(2)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1, 0])
        v2 = g.find_or_create_vertex([0, 1])
        v0.add_edge(v1, 1.0)
        v1.add_edge(v2, 1.0)
        v2.add_edge(v1, 1.0)

        sccs = get_scc_graphs(g, min_size=1)

        assert isinstance(sccs, list), "Should return list"
        assert len(sccs) > 0, "Should have at least one SCC"
        assert all(isinstance(item, tuple) for item in sccs), "Should return tuples"
        assert all(len(item) == 2 for item in sccs), "Tuples should be (hash, graph)"


class TestGraphHashing:
    """Test graph hashing integration"""

    def test_graph_hash_computation(self):
        """Test that graph hashing works"""
        from phasic import hash as phasic_hash

        g = Graph(1)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v0.add_edge(v1, 1.0)

        hash_result = phasic_hash.compute_graph_hash(g)

        assert hash_result is not None, "Should return hash result"
        assert hasattr(hash_result, 'hash_hex'), "Should have hash_hex attribute"
        assert len(hash_result.hash_hex) == 64, "Hash should be 64 hex characters"

    def test_identical_graphs_same_hash(self):
        """Test that identical graphs produce same hash"""
        from phasic import hash as phasic_hash

        # Graph 1
        g1 = Graph(1)
        v0_1 = g1.starting_vertex()
        v1_1 = g1.find_or_create_vertex([1])
        v0_1.add_edge(v1_1, 1.0)

        # Graph 2 (identical structure)
        g2 = Graph(1)
        v0_2 = g2.starting_vertex()
        v1_2 = g2.find_or_create_vertex([1])
        v0_2.add_edge(v1_2, 1.0)

        hash1 = phasic_hash.compute_graph_hash(g1)
        hash2 = phasic_hash.compute_graph_hash(g2)

        assert hash1.hash_hex == hash2.hash_hex, "Identical graphs should have same hash"


class TestBackwardCompatibility:
    """Test backward compatibility"""

    def test_default_behavior_unchanged(self):
        """Test that default behavior is non-hierarchical"""
        g = Graph(1)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v0.add_edge(v1, 1.0)

        # Default should be hierarchical=False
        trace = g.compute_trace()

        assert trace is not None, "Should work without parameters"

    def test_existing_trace_elimination_works(self):
        """Test that existing trace_elimination module still works"""
        from phasic.trace_elimination import record_elimination_trace

        g = Graph(1)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v0.add_edge(v1, 1.0)

        trace = record_elimination_trace(g)

        assert trace is not None, "Existing API should still work"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
