"""
Unit tests for SCC API (Phase 1)

Tests the C++ SCC decomposition API and Python bindings.
"""

import pytest
import numpy as np
from phasic import Graph


class TestSCCDecomposition:
    """Test SCC decomposition functionality"""

    def test_simple_acyclic_graph(self):
        """Test SCC decomposition of simple acyclic graph"""
        g = Graph(1)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v0.add_edge(v1, 1.0)

        scc = g.scc_decomposition()
        assert len(scc) >= 1, "Should have at least one SCC"
        assert sum(scc.scc_sizes()) == g.vertices_length(), "All vertices accounted for"

    def test_simple_cyclic_graph(self):
        """Test SCC decomposition of graph with cycle"""
        g = Graph(2)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1, 0])
        v2 = g.find_or_create_vertex([0, 1])
        v0.add_edge(v1, 1.0)
        v1.add_edge(v2, 1.0)
        v2.add_edge(v1, 1.0)  # Creates cycle between v1 and v2

        scc = g.scc_decomposition()
        assert len(scc) >= 1, "Should have at least one SCC"

        # Should have one SCC with 2 vertices (the cycle)
        sizes = scc.scc_sizes()
        assert 2 in sizes, "Should have SCC with 2 vertices (the cycle)"

    def test_scc_sizes(self):
        """Test scc_sizes() returns correct values"""
        g = Graph(1)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])
        v0.add_edge(v1, 1.0)
        v1.add_edge(v2, 1.0)

        scc = g.scc_decomposition()
        sizes = scc.scc_sizes()

        assert isinstance(sizes, list), "Should return list"
        assert all(s > 0 for s in sizes), "All SCCs should have positive size"
        assert sum(sizes) == g.vertices_length(), "Sizes should sum to vertex count"

    def test_scc_iteration(self):
        """Test iteration over SCCs"""
        g = Graph(1)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v0.add_edge(v1, 1.0)

        scc_graph = g.scc_decomposition()

        # Test __len__
        n_sccs = len(scc_graph)
        assert n_sccs > 0, "Should have at least one SCC"

        # Test iteration
        scc_list = list(scc_graph.sccs_in_topo_order())
        assert len(scc_list) == n_sccs, "Iteration should yield all SCCs"

        # Test __getitem__
        for i in range(n_sccs):
            scc = scc_graph[i]
            assert scc.size() > 0, f"SCC {i} should have positive size"

    def test_scc_hashing(self):
        """Test SCC hash computation"""
        g = Graph(1)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v0.add_edge(v1, 1.0)

        scc_graph = g.scc_decomposition()
        hashes = scc_graph.scc_hashes()

        assert isinstance(hashes, list), "Should return list"
        assert len(hashes) == len(scc_graph), "One hash per SCC"
        assert all(isinstance(h, str) for h in hashes), "Hashes should be strings"
        assert all(len(h) == 64 for h in hashes), "Hashes should be 64 hex chars"

    def test_scc_as_graph(self):
        """Test extracting SCC as standalone graph"""
        g = Graph(2)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1, 0])
        v2 = g.find_or_create_vertex([0, 1])
        v0.add_edge(v1, 1.0)
        v1.add_edge(v2, 1.0)
        v2.add_edge(v1, 1.0)

        scc_graph = g.scc_decomposition()

        for scc in scc_graph.sccs_in_topo_order():
            subgraph = scc.as_graph()

            assert isinstance(subgraph, Graph), "Should return Graph object"
            assert subgraph.vertices_length() == scc.size(), "Subgraph size matches SCC size"
            assert subgraph.state_length() == g.state_length(), "State length preserved"


class TestSCCVertex:
    """Test SCCVertex functionality"""

    def test_scc_vertex_properties(self):
        """Test basic SCCVertex properties"""
        g = Graph(1)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v0.add_edge(v1, 1.0)

        scc_graph = g.scc_decomposition()
        scc = scc_graph[0]

        assert scc.size() > 0, "Size should be positive"
        assert scc.index() >= 0, "Index should be non-negative"
        assert len(scc) == scc.size(), "__len__ should match size()"

    def test_scc_vertex_hash(self):
        """Test individual SCC hash computation"""
        g = Graph(1)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v0.add_edge(v1, 1.0)

        scc_graph = g.scc_decomposition()
        scc = scc_graph[0]

        hash_str = scc.hash()
        assert isinstance(hash_str, str), "Hash should be string"
        assert len(hash_str) == 64, "Hash should be 64 hex chars"


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_single_vertex_graph(self):
        """Test graph with only starting vertex"""
        g = Graph(1)
        scc = g.scc_decomposition()

        assert len(scc) >= 1, "Should have at least one SCC"

    def test_disconnected_components(self):
        """Test graph with multiple disconnected components from starting vertex"""
        g = Graph(1)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])
        v0.add_edge(v1, 0.5)
        v0.add_edge(v2, 0.5)
        # v1 and v2 are not connected to each other

        scc = g.scc_decomposition()
        assert len(scc) >= 2, "Should have separate SCCs for disconnected branches"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
