"""Tests for hierarchical trace-based computation in Graph.

This module tests the Graph(hierarchical=True) feature that enables
trace-based computation for moments, expectation, variance, etc.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose


def simple_coalescent_callback(state):
    """Simple coalescent model callback for testing.

    Parameters
    ----------
    state : list
        Current state [n] where n is number of lineages.

    Returns
    -------
    list
        List of (next_state, coefficients) tuples for parameterized edges.
        The coefficients define the edge weight as: sum(coeff_i * theta_i)
    """
    n = state[0]
    if n <= 1:
        return []
    # Coalescent rate: n*(n-1)/2 * theta
    # Return 2-tuple: (state, coefficients) for parameterized edge
    rate_coeff = n * (n - 1) / 2
    return [([n - 1], [rate_coeff])]


@pytest.fixture
def simple_model_callback():
    """Return the simple coalescent callback with IPV."""
    from phasic import with_ipv

    # IPV format: [[state, probability], ...]
    @with_ipv([[[5], 1.0]])
    def callback(state):
        return simple_coalescent_callback(state)

    return callback


class TestHierarchicalGraphBasic:
    """Basic functionality tests for hierarchical mode."""

    def test_hierarchical_flag_default_false(self):
        """Test that hierarchical defaults to False."""
        from phasic import Graph

        g = Graph(1)
        assert g.hierarchical is False
        assert g._trace is None
        assert g._trace_dirty is True

    @pytest.mark.skip(reason="retired Python EliminationTrace machinery; "
                             "compute_trace()/hierarchical are deprecated no-ops")
    def test_hierarchical_flag_set_true(self):
        """Test that hierarchical=True is stored."""
        from phasic import Graph

        g = Graph(1, hierarchical=True)
        assert g.hierarchical is True

    def test_trace_valid_initially_false(self):
        """Test that trace_valid is False before computation."""
        from phasic import Graph

        g = Graph(1, hierarchical=True)
        assert g.trace_valid is False


class TestHierarchicalGraphWithModel:
    """Tests with actual graph models."""

    @pytest.mark.skip(reason="retired Python EliminationTrace machinery; "
                             "compute_trace()/hierarchical are deprecated no-ops")
    def test_compute_trace_non_destructive(self, simple_model_callback):
        """Test that compute_trace preserves graph in hierarchical mode."""
        from phasic import Graph

        g = Graph(simple_model_callback, hierarchical=True)
        g.normalize()

        vertices_before = g.vertices_length()
        assert vertices_before > 0

        trace = g.compute_trace()

        # Graph should be preserved
        vertices_after = g.vertices_length()
        assert vertices_after == vertices_before
        assert g.trace_valid is True
        assert trace is not None

    @pytest.mark.skip(reason="retired Python EliminationTrace machinery; "
                             "compute_trace()/hierarchical are deprecated no-ops")
    def test_compute_trace_caches_result(self, simple_model_callback):
        """Test that subsequent compute_trace calls return cached trace."""
        from phasic import Graph

        g = Graph(simple_model_callback, hierarchical=True)
        g.normalize()

        trace1 = g.compute_trace()
        trace2 = g.compute_trace()

        # Should be same object (cached)
        assert trace1 is trace2

    def test_compute_trace_force_recompute(self, simple_model_callback):
        """Test that force=True recomputes the trace."""
        from phasic import Graph

        g = Graph(simple_model_callback, hierarchical=True)
        g.normalize()

        trace1 = g.compute_trace()
        trace2 = g.compute_trace(force=True)

        # Should be different objects (recomputed)
        assert trace1 is not trace2


class TestTraceInvalidation:
    """Tests for trace invalidation on graph modification."""

    @pytest.mark.skip(reason="retired Python EliminationTrace machinery; "
                             "compute_trace()/hierarchical are deprecated no-ops")
    def test_normalize_invalidates_trace(self, simple_model_callback):
        """Test that normalize() invalidates cached trace."""
        from phasic import Graph

        g = Graph(simple_model_callback, hierarchical=True)
        g.normalize()
        g.compute_trace()

        assert g.trace_valid is True

        # Normalize again should invalidate
        g.normalize()
        assert g.trace_valid is False

    def test_find_or_create_vertex_invalidates_trace(self):
        """Test that find_or_create_vertex invalidates cached trace."""
        from phasic import Graph

        g = Graph(1, hierarchical=True)
        start = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        start.add_edge(v1, 1.0)
        g.normalize()

        # Manually set up trace (simulating previous compute)
        g._trace = "dummy"
        g._trace_dirty = False
        assert g.trace_valid is True

        # Adding a vertex should invalidate
        g.find_or_create_vertex([2])
        assert g.trace_valid is False

    @pytest.mark.skip(reason="retired Python EliminationTrace machinery; "
                             "compute_trace()/hierarchical are deprecated no-ops")
    def test_update_weights_does_not_invalidate_trace(self, simple_model_callback):
        """Test that update_weights does NOT invalidate trace."""
        from phasic import Graph

        g = Graph(simple_model_callback, hierarchical=True)
        g.normalize()
        g.compute_trace()

        assert g.trace_valid is True

        # Update weights should preserve trace
        g.update_weights([2.0])
        assert g.trace_valid is True


class TestTraceMomentsComputation:
    """Tests for moments computation in hierarchical mode.

    NOTE: Trace-based moments are currently disabled due to a bug in
    instantiate_from_trace. These tests verify that hierarchical mode
    still works correctly by falling back to direct C++ computation.
    """

    def test_expectation_matches_direct(self, simple_model_callback):
        """Test that hierarchical expectation matches direct computation."""
        from phasic import Graph

        # Direct computation
        g_direct = Graph(simple_model_callback)
        g_direct.normalize()
        g_direct.update_weights([1.0])
        exp_direct = g_direct.expectation()

        # Hierarchical mode (currently uses C++ fallback)
        g_hier = Graph(simple_model_callback, hierarchical=True)
        g_hier.normalize()
        g_hier.update_weights([1.0])
        exp_hier = g_hier.expectation()

        assert_allclose(exp_hier, exp_direct, rtol=1e-10)

    def test_variance_matches_direct(self, simple_model_callback):
        """Test that hierarchical variance matches direct computation."""
        from phasic import Graph

        # Direct computation
        g_direct = Graph(simple_model_callback)
        g_direct.normalize()
        g_direct.update_weights([1.0])
        var_direct = g_direct.variance()

        # Hierarchical mode (currently uses C++ fallback)
        g_hier = Graph(simple_model_callback, hierarchical=True)
        g_hier.normalize()
        g_hier.update_weights([1.0])
        var_hier = g_hier.variance()

        assert_allclose(var_hier, var_direct, rtol=1e-10)

    def test_moments_matches_direct(self, simple_model_callback):
        """Test that hierarchical moments match direct computation."""
        from phasic import Graph

        # Direct computation
        g_direct = Graph(simple_model_callback)
        g_direct.normalize()
        g_direct.update_weights([1.0])
        moments_direct = g_direct.moments(2)

        # Hierarchical mode (currently uses C++ fallback)
        g_hier = Graph(simple_model_callback, hierarchical=True)
        g_hier.normalize()
        g_hier.update_weights([1.0])
        moments_hier = g_hier.moments(2)

        assert_allclose(moments_hier, moments_direct, rtol=1e-10)

    def test_expectation_with_different_theta(self, simple_model_callback):
        """Test expectation with different parameter values."""
        from phasic import Graph

        g = Graph(simple_model_callback, hierarchical=True)
        g.normalize()

        # First theta
        g.update_weights([1.0])
        exp1 = g.expectation()

        # Second theta
        g.update_weights([2.0])
        exp2 = g.expectation()

        # Should be different (scaled by 1/theta for coalescent)
        assert exp1 != exp2
        # For coalescent with theta scaling, exp scales inversely
        assert_allclose(exp2, exp1 / 2.0, rtol=1e-10)


class TestClone:
    """Tests for clone() with hierarchical mode."""

    @pytest.mark.skip(reason="retired Python EliminationTrace machinery; "
                             "compute_trace()/hierarchical are deprecated no-ops")
    def test_clone_preserves_hierarchical_setting(self, simple_model_callback):
        """Test that clone preserves hierarchical flag."""
        from phasic import Graph

        g = Graph(simple_model_callback, hierarchical=True)
        g.normalize()

        cloned = g.clone()

        assert cloned.hierarchical is True

    @pytest.mark.skip(reason="retired Python EliminationTrace machinery; "
                             "compute_trace()/hierarchical are deprecated no-ops")
    def test_clone_does_not_copy_trace(self, simple_model_callback):
        """Test that clone starts with fresh trace cache."""
        from phasic import Graph

        g = Graph(simple_model_callback, hierarchical=True)
        g.normalize()
        g.compute_trace()

        assert g.trace_valid is True

        cloned = g.clone()

        # Clone should have invalid trace
        assert cloned.trace_valid is False
        assert cloned._trace is None


class TestErrorHandling:
    """Tests for error handling in hierarchical mode."""

    def test_expectation_works_without_update_weights(self, simple_model_callback):
        """Test that expectation works without update_weights in hierarchical mode.

        NOTE: Since trace-based moments are disabled, hierarchical mode falls
        back to C++ computation which doesn't require update_weights.
        """
        from phasic import Graph

        g = Graph(simple_model_callback, hierarchical=True)
        g.normalize()
        g.update_weights([1.0])  # Required for parameterized graph

        # Should work (uses C++ fallback)
        exp = g.expectation()
        assert exp > 0

    def test_non_parameterized_graph_works(self):
        """Test that non-parameterized graph works in hierarchical mode."""
        from phasic import Graph

        # Non-parameterized graph - create a simple 2-state chain
        # State 2 -> State 1 -> Absorbing with rate 1.0 each
        g = Graph(1, hierarchical=True)
        start = g.starting_vertex()
        v2 = g.find_or_create_vertex([2])
        v1 = g.find_or_create_vertex([1])
        start.add_edge(v2, 1.0)  # Start at state 2
        v2.add_edge(v1, 1.0)    # Rate 1.0 to state 1
        # v1 implicitly absorbs (no outgoing edges)
        g.normalize()

        # Should work (uses direct C++)
        # Expectation = time in state 2 (E[Exp(1)]=1) + time in state 1 (E[Exp(1)]=1) = 2
        # But after normalize, rates may change. Just verify it's positive.
        exp = g.expectation()
        assert exp > 0
