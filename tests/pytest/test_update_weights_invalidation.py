"""Stage 1 verification — weight_version cache invalidation.

These tests catch the regression mode where calling
``Graph.update_weights(theta)`` refreshes edge weights but leaves a
previously-computed forward-state context (``ph_context``,
``ph_context_markov``, ``dph_context``, ``dph_context_markov``) cached
inside the C++ wrapper. Before Stage 1 the C-side function did not
invalidate those caches; after Stage 1 every C-level edge-weight
mutation increments ``graph->weight_version`` and the wrapper guards
detect the mismatch.

Each test runs the same forward-state query twice on the *same* graph
object across an ``update_weights`` call, then compares against a
reference computed by building a fresh graph at the new theta. If the
cache invalidation is wrong, the second result reflects the *first*
theta and disagrees with the reference.
"""

import numpy as np
import pytest
from phasic import Graph


def _build_two_stage_param_graph():
    """Two-parameter Erlang-like: S -> [3] --(theta[0])--> [2] --(theta[1])--> [1]."""
    g = Graph(1)
    start = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v3, 1.0)
    v3.add_edge_parameterized(v2, 0.0, [1.0, 0.0])
    v2.add_edge_parameterized(v1, 0.0, [0.0, 1.0])
    return g


def _build_one_param_graph():
    """One-parameter exponential-ish: S -> [2] --(theta[0])--> [1]."""
    g = Graph(1)
    start = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v2, 1.0)
    v2.add_edge_parameterized(v1, 0.0, [1.0])
    return g


# ---------------------------------------------------------------------------
# accumulated_visiting_time: direct C++-wrapper cache (ph_context_markov)
# ---------------------------------------------------------------------------


def test_update_weights_invalidates_visiting_time():
    g = _build_two_stage_param_graph()
    g.update_weights([1.0, 1.0])
    v_a = np.asarray(g.accumulated_visiting_time(2.0, 10))

    g.update_weights([5.0, 0.5])
    v_b = np.asarray(g.accumulated_visiting_time(2.0, 10))

    g_ref = _build_two_stage_param_graph()
    g_ref.update_weights([5.0, 0.5])
    v_b_ref = np.asarray(g_ref.accumulated_visiting_time(2.0, 10))

    np.testing.assert_allclose(v_b, v_b_ref, rtol=1e-10)
    assert not np.allclose(v_a, v_b), (
        "Sanity: the two thetas must produce different visiting times"
    )


def test_update_weights_invalidates_stop_probability():
    g = _build_two_stage_param_graph()
    g.update_weights([1.0, 1.0])
    p_a = np.asarray(g.stop_probability(2.0, granularity=10))

    g.update_weights([5.0, 0.5])
    p_b = np.asarray(g.stop_probability(2.0, granularity=10))

    g_ref = _build_two_stage_param_graph()
    g_ref.update_weights([5.0, 0.5])
    p_b_ref = np.asarray(g_ref.stop_probability(2.0, granularity=10))

    np.testing.assert_allclose(p_b, p_b_ref, rtol=1e-10)
    assert not np.allclose(p_a, p_b)


# ---------------------------------------------------------------------------
# pdf: ph_context cache
# ---------------------------------------------------------------------------


def test_update_weights_invalidates_pdf():
    g = _build_one_param_graph()
    g.update_weights([1.0])
    pdf_a = float(g.pdf(1.5, granularity=100))

    g.update_weights([3.0])
    pdf_b = float(g.pdf(1.5, granularity=100))

    g_ref = _build_one_param_graph()
    g_ref.update_weights([3.0])
    pdf_b_ref = float(g_ref.pdf(1.5, granularity=100))

    np.testing.assert_allclose(pdf_b, pdf_b_ref, rtol=1e-10)
    assert pdf_a != pdf_b


# ---------------------------------------------------------------------------
# Sanity: reading a forward query, then update_weights, then reading again
# at a *smaller* time would have been wrong without invalidation. The cached
# ph_context_markov already advanced past the requested time so the existing
# ``time - 1/granularity > time`` guard rebuilds — but only because of that
# guard, not because of weight changes. The version counter makes correctness
# robust regardless of time ordering.
# ---------------------------------------------------------------------------


def test_update_weights_then_query_same_time():
    """Hits the same query twice with two different thetas; without the
    version counter the second result equals the first because all the
    other guard conditions (``ph_context != NULL``, ``granularity ==``,
    ``time within range``) are satisfied by the cached state."""
    g = _build_one_param_graph()

    g.update_weights([2.0])
    a = float(g.pdf(0.5, granularity=100))
    g.update_weights([4.0])
    b = float(g.pdf(0.5, granularity=100))  # same time, different theta

    g_ref = _build_one_param_graph()
    g_ref.update_weights([4.0])
    b_ref = float(g_ref.pdf(0.5, granularity=100))

    np.testing.assert_allclose(b, b_ref, rtol=1e-10)
    assert a != b
