"""
Test: Self-loop correction in trace elimination.

When eliminating a vertex that creates a back-edge to its parent,
the geometric series correction factor 1/(1-q) must be applied.
Without this correction, trace-based expectation produces NaN.
"""

import numpy as np
from pytest import approx
from phasic import Graph


def test_cyclic_graph_expectation_cache_trace_vs_direct():
    """
    Build a parameterized graph with a cycle (v1 -> v2 -> v1)
    and verify expectation() matches with and without cache_trace.
    """
    # Graph: start -> v1 -> v2 -> absorbing
    #                  v1 <- v2  (back-edge creating cycle)
    #
    # When v2 is eliminated, the edge v2->v1 creates a self-loop at v1.
    # The correction factor 1/(1-q) must be applied.

    def make_graph(cache_trace):
        g = Graph(1, cache_trace=cache_trace)
        v_start = g.starting_vertex()
        v1 = g.find_or_create_vertex([2])
        v2 = g.find_or_create_vertex([3])
        v_abs = g.find_or_create_vertex([1])

        # start -> v1 with probability 1
        v_start.add_edge_parameterized(v1, 0.0, [1.0])

        # v1 -> v2 with rate theta[0]
        v1.add_edge_parameterized(v2, 0.0, [0.5])

        # v2 -> v1 (back-edge: creates self-loop on elimination)
        v2.add_edge_parameterized(v1, 0.0, [0.3])

        # v2 -> absorbing
        v2.add_edge_parameterized(v_abs, 0.0, [0.7])

        return g

    theta = np.array([2.0])

    # Without cache_trace: uses direct C computation
    g_direct = make_graph(cache_trace=False)
    g_direct.update_weights(theta)
    exp_direct = g_direct.expectation()

    # With cache_trace: uses trace elimination -> instantiate -> C
    g_cached = make_graph(cache_trace=True)
    g_cached.update_weights(theta)
    exp_cached = g_cached.expectation()

    assert not np.isnan(exp_cached), (
        f"cache_trace=True produced NaN (direct gave {exp_direct})"
    )
    assert exp_cached == approx(exp_direct, rel=1e-10), (
        f"cache_trace mismatch: cached={exp_cached}, direct={exp_direct}"
    )


def test_high_self_loop_probability():
    """
    Test with parameter values that produce high self-loop probability
    (close to 1). This stresses the 1/(1-q) correction.
    """
    def make_graph(cache_trace):
        g = Graph(1, cache_trace=cache_trace)
        v_start = g.starting_vertex()
        v1 = g.find_or_create_vertex([2])
        v2 = g.find_or_create_vertex([3])
        v_abs = g.find_or_create_vertex([1])

        v_start.add_edge_parameterized(v1, 0.0, [1.0])
        # High back-edge coefficient relative to forward
        v1.add_edge_parameterized(v2, 0.0, [0.9])
        v2.add_edge_parameterized(v1, 0.0, [0.9])  # high back-edge
        v2.add_edge_parameterized(v_abs, 0.0, [0.1])

        return g

    theta = np.array([1.0])

    g_direct = make_graph(cache_trace=False)
    g_direct.update_weights(theta)
    exp_direct = g_direct.expectation()

    g_cached = make_graph(cache_trace=True)
    g_cached.update_weights(theta)
    exp_cached = g_cached.expectation()

    assert not np.isnan(exp_cached), (
        f"High self-loop probability produced NaN (direct gave {exp_direct})"
    )
    assert exp_cached == approx(exp_direct, rel=1e-10)


def test_variance_with_self_loop():
    """
    Verify variance() also works correctly with self-loops and cache_trace.
    """
    def make_graph(cache_trace):
        g = Graph(1, cache_trace=cache_trace)
        v_start = g.starting_vertex()
        v1 = g.find_or_create_vertex([2])
        v2 = g.find_or_create_vertex([3])
        v_abs = g.find_or_create_vertex([1])

        v_start.add_edge_parameterized(v1, 0.0, [1.0])
        v1.add_edge_parameterized(v2, 0.0, [0.5])
        v2.add_edge_parameterized(v1, 0.0, [0.3])
        v2.add_edge_parameterized(v_abs, 0.0, [0.7])

        return g

    theta = np.array([2.0])

    g_direct = make_graph(cache_trace=False)
    g_direct.update_weights(theta)
    var_direct = g_direct.variance()

    g_cached = make_graph(cache_trace=True)
    g_cached.update_weights(theta)
    var_cached = g_cached.variance()

    assert not np.isnan(var_cached), (
        f"variance with cache_trace produced NaN (direct gave {var_direct})"
    )
    assert var_cached == approx(var_direct, rel=1e-10)
