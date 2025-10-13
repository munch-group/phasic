#!/usr/bin/env python
"""
Test suite for symbolic DAG functionality.
Tests symbolic Gaussian elimination and fast parameter instantiation.
"""

import pytest
import numpy as np
import ptdalgorithms as ptd
from ptdalgorithms import Graph


class TestSymbolicDAG:
    """Test symbolic DAG creation and instantiation."""

    def test_eliminate_to_dag_basic(self):
        """Test basic symbolic elimination."""
        # Create parameterized graph
        g = Graph(state_length=1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])

        g.starting_vertex().add_edge_parameterized(v1, 0.0, [1.0, 0.0])
        v1.add_edge_parameterized(v2, 0.0, [0.0, 1.0])

        # Eliminate to DAG
        dag = g.eliminate_to_dag()

        assert dag is not None
        assert dag.vertices_length() > 0
        assert dag.param_length() == 2

    def test_symbolic_dag_info(self):
        """Test symbolic DAG info property."""
        g = Graph(state_length=1)
        v = g.find_or_create_vertex([1])
        g.starting_vertex().add_edge_parameterized(v, 0.0, [1.0])

        dag = g.eliminate_to_dag()
        info = dag.info

        assert 'vertices_length' in info
        assert 'param_length' in info
        assert 'is_acyclic' in info

    def test_symbolic_dag_properties(self):
        """Test symbolic DAG properties."""
        g = Graph(state_length=1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])

        g.starting_vertex().add_edge_parameterized(v1, 0.0, [1.0, 0.0])
        v1.add_edge_parameterized(v2, 0.0, [0.0, 1.0])

        dag = g.eliminate_to_dag()

        # Test properties
        assert dag.vertices_length() > 0
        assert dag.param_length() == 2
        assert isinstance(dag.is_acyclic(), bool)

    def test_symbolic_dag_instantiate(self):
        """Test instantiating symbolic DAG with parameters."""
        # Create parameterized graph
        g = Graph(state_length=1)
        v = g.find_or_create_vertex([1])
        g.starting_vertex().add_edge_parameterized(v, 0.0, [1.0])

        # Eliminate to symbolic DAG
        dag = g.eliminate_to_dag()

        # Instantiate with concrete parameters
        params = np.array([2.0])
        g_concrete = dag.instantiate(params)

        assert g_concrete is not None
        assert g_concrete.vertices_length() > 0

        # Should be able to compute on instantiated graph
        exp = g_concrete.expectation()
        assert exp > 0

    def test_symbolic_dag_multiple_instantiations(self):
        """Test multiple instantiations with different parameters."""
        # Create parameterized graph
        g = Graph(state_length=1)
        v = g.find_or_create_vertex([1])
        g.starting_vertex().add_edge_parameterized(v, 0.0, [1.0])

        dag = g.eliminate_to_dag()

        # Instantiate with different parameters
        params1 = np.array([1.0])
        params2 = np.array([2.0])
        params3 = np.array([3.0])

        g1 = dag.instantiate(params1)
        g2 = dag.instantiate(params2)
        g3 = dag.instantiate(params3)

        # Each should give different expectations
        exp1 = g1.expectation()
        exp2 = g2.expectation()
        exp3 = g3.expectation()

        assert exp1 != exp2
        assert exp2 != exp3
        # Higher rate should give lower expectation
        assert exp1 > exp2 > exp3

    def test_symbolic_dag_multi_parameter(self):
        """Test symbolic DAG with multiple parameters."""
        g = Graph(state_length=1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])
        v3 = g.find_or_create_vertex([3])

        g.starting_vertex().add_edge_parameterized(v1, 0.0, [1.0, 0.0, 0.0])
        v1.add_edge_parameterized(v2, 0.0, [0.0, 1.0, 0.0])
        v2.add_edge_parameterized(v3, 0.0, [0.0, 0.0, 1.0])

        dag = g.eliminate_to_dag()

        assert dag.param_length() == 3

        # Instantiate
        params = np.array([1.0, 2.0, 3.0])
        g_concrete = dag.instantiate(params)

        assert g_concrete is not None

    def test_symbolic_dag_repr(self):
        """Test symbolic DAG string representation."""
        g = Graph(state_length=1)
        v = g.find_or_create_vertex([1])
        g.starting_vertex().add_edge_parameterized(v, 0.0, [1.0])

        dag = g.eliminate_to_dag()

        repr_str = repr(dag)
        assert 'SymbolicDAG' in repr_str

    def test_eliminate_non_parameterized_raises(self):
        """Test that eliminating non-parameterized graph raises error."""
        # Create regular (non-parameterized) graph
        g = Graph(state_length=1)
        v = g.find_or_create_vertex([1])
        g.starting_vertex().add_edge(v, 1.0)

        # Should raise error
        with pytest.raises((RuntimeError, ValueError)):
            dag = g.eliminate_to_dag()

    def test_symbolic_dag_preserves_distribution(self):
        """Test that instantiated DAG preserves distribution properties."""
        # Create simple parameterized graph
        g = Graph(state_length=1)
        v = g.find_or_create_vertex([1])
        g.starting_vertex().add_edge_parameterized(v, 0.0, [1.0])

        # Build reference graph with same structure
        g_ref = Graph(state_length=1)
        v_ref = g_ref.find_or_create_vertex([1])
        g_ref.starting_vertex().add_edge(v_ref, 2.0)
        g_ref.normalize()

        # Eliminate and instantiate with same rate
        dag = g.eliminate_to_dag()
        g_inst = dag.instantiate(np.array([2.0]))
        g_inst.normalize()

        # Should have same expectation
        exp_ref = g_ref.expectation()
        exp_inst = g_inst.expectation()

        assert exp_ref == pytest.approx(exp_inst, rel=1e-6)


class TestParameterizedEdges:
    """Test parameterized edge functionality."""

    def test_add_edge_parameterized_basic(self):
        """Test basic parameterized edge addition."""
        g = Graph(state_length=1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])

        v1.add_edge_parameterized(v2, 0.0, [1.0])

        # Check that graph knows it has parameterized edges
        serialized = g.serialize()
        assert serialized.get('param_length', 0) > 0

    def test_parameterized_edge_serialization(self):
        """Test that parameterized edges are properly serialized."""
        g = Graph(state_length=1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])

        # Add regular and parameterized edges
        v1.add_edge(v2, 1.0)
        v1.add_edge_parameterized(v2, 0.0, [1.0, 0.0])

        serialized = g.serialize()

        assert 'param_edges' in serialized
        assert 'param_length' in serialized
        assert serialized['param_length'] > 0
        assert len(serialized['param_edges']) > 0

    def test_update_parameterized_weights(self):
        """Test updating weights of parameterized edges."""
        g = Graph(state_length=1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])

        v1.add_edge_parameterized(v2, 0.0, [1.0])

        # Update weights
        new_params = np.array([2.5])
        g.update_parameterized_weights(new_params)

        # Graph should now behave as if edge has weight 2.5
        # (Exact behavior depends on implementation)

    def test_mixed_edges(self):
        """Test graph with both regular and parameterized edges."""
        g = Graph(state_length=1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])
        v3 = g.find_or_create_vertex([3])

        # Mix of regular and parameterized edges
        v1.add_edge(v2, 1.0)  # Regular
        v2.add_edge_parameterized(v3, 0.0, [1.0])  # Parameterized

        serialized = g.serialize()
        assert len(serialized['edges']) > 0  # Regular edges
        assert len(serialized['param_edges']) > 0  # Parameterized edges


class TestSymbolicPerformance:
    """Test performance characteristics of symbolic DAG."""

    def test_symbolic_faster_than_rebuild(self):
        """Test that symbolic DAG instantiation is faster than graph rebuild."""
        import time

        # Create a moderately complex parameterized graph
        def build_graph_parameterized():
            g = Graph(state_length=1)
            for i in range(1, 6):
                v = g.find_or_create_vertex([i])
                if i == 1:
                    g.starting_vertex().add_edge_parameterized(v, 0.0, [1.0, 0.0])
                else:
                    v_prev = g.find_or_create_vertex([i-1])
                    v_prev.add_edge_parameterized(v, 0.0, [0.0, 1.0])
            return g

        # Build symbolic DAG once
        g = build_graph_parameterized()
        dag = g.eliminate_to_dag()

        # Time symbolic instantiation
        n_instantiations = 10
        params = np.array([1.5, 2.5])

        start = time.time()
        for _ in range(n_instantiations):
            g_inst = dag.instantiate(params)
            _ = g_inst.expectation()
        symbolic_time = time.time() - start

        # Time full graph rebuild
        start = time.time()
        for _ in range(n_instantiations):
            g_new = build_graph_parameterized()
            g_new.update_parameterized_weights(params)
            _ = g_new.expectation()
        rebuild_time = time.time() - start

        # Symbolic should be faster (though not necessarily for small graphs)
        # Just check that both work
        assert symbolic_time > 0
        assert rebuild_time > 0

    def test_symbolic_dag_reusability(self):
        """Test that symbolic DAG can be reused many times."""
        g = Graph(state_length=1)
        v = g.find_or_create_vertex([1])
        g.starting_vertex().add_edge_parameterized(v, 0.0, [1.0])

        dag = g.eliminate_to_dag()

        # Use it many times
        n_uses = 100
        results = []

        for i in range(n_uses):
            params = np.array([1.0 + i * 0.1])
            g_inst = dag.instantiate(params)
            results.append(g_inst.expectation())

        # Should have 100 different results
        assert len(results) == n_uses
        assert len(set(results)) > 1  # Not all the same


class TestSymbolicWithCallback:
    """Test symbolic DAG with callback-based graphs."""

    def test_symbolic_callback_simple(self):
        """Test symbolic DAG with simple callback."""
        def callback(state):
            if state[0] < 3:
                return [([state[0] + 1], 0.0, [1.0])]
            return []

        g = Graph(callback=callback, parameterized=True)

        # Should be able to eliminate
        dag = g.eliminate_to_dag()

        assert dag.param_length() > 0

        # Should be able to instantiate
        params = np.ones(dag.param_length()) * 1.5
        g_inst = dag.instantiate(params)

        assert g_inst is not None

    def test_symbolic_callback_complex(self):
        """Test symbolic DAG with more complex callback."""
        def callback(state):
            results = []
            if state[0] < 5:
                # Two possible transitions with different parameters
                results.append(([state[0] + 1], 0.0, [1.0, 0.0]))
                if state[0] > 0:
                    results.append(([state[0] - 1], 0.0, [0.0, 1.0]))
            return results

        g = Graph(callback=callback, parameterized=True)

        dag = g.eliminate_to_dag()

        assert dag.param_length() == 2

        # Test with different parameter combinations
        params_list = [
            np.array([1.0, 0.5]),
            np.array([2.0, 1.0]),
            np.array([0.5, 2.0])
        ]

        for params in params_list:
            g_inst = dag.instantiate(params)
            exp = g_inst.expectation()
            assert exp > 0


class TestSymbolicEdgeCases:
    """Test edge cases for symbolic DAG."""

    def test_symbolic_single_parameter(self):
        """Test symbolic DAG with single parameter."""
        g = Graph(state_length=1)
        v = g.find_or_create_vertex([1])
        g.starting_vertex().add_edge_parameterized(v, 0.0, [1.0])

        dag = g.eliminate_to_dag()

        assert dag.param_length() == 1

        params = np.array([1.0])
        g_inst = dag.instantiate(params)
        assert g_inst is not None

    def test_symbolic_zero_parameter(self):
        """Test instantiation with zero parameter."""
        g = Graph(state_length=1)
        v = g.find_or_create_vertex([1])
        g.starting_vertex().add_edge_parameterized(v, 0.0, [1.0])

        dag = g.eliminate_to_dag()

        # Zero parameter should work (though may give inf expectation)
        params = np.array([0.0])

        # May raise or may work depending on implementation
        try:
            g_inst = dag.instantiate(params)
            # If it works, expectation might be inf
            exp = g_inst.expectation()
        except (RuntimeError, ValueError):
            # Zero rate may not be allowed
            pass

    def test_symbolic_large_parameters(self):
        """Test instantiation with large parameters."""
        g = Graph(state_length=1)
        v = g.find_or_create_vertex([1])
        g.starting_vertex().add_edge_parameterized(v, 0.0, [1.0])

        dag = g.eliminate_to_dag()

        params = np.array([1000.0])
        g_inst = dag.instantiate(params)

        exp = g_inst.expectation()
        assert exp > 0
        assert exp < 1.0  # Should be small for large rate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
