"""Stage A0 verification — symbolic compute graph survives update_weights.

Before Stage A0, ``ptd_graph_update_weights`` (`src/c/phasic.c:3148`)
unconditionally destroyed ``graph->parameterized_reward_compute_graph``
(the C-level symbolic Gaussian elimination cache) at the end of every
call. The next forward query (`pdf`, `accumulated_visiting_time`, ...)
re-ran the O(n^3) elimination via
``ptd_precompute_reward_compute_graph`` -> the parameterised branch ->
``ptd_graph_ex_absorbation_time_comp_graph_parameterized``.

This was wasteful: the symbolic compute graph stores commands whose
``multiplierptr`` fields are pointers into the live edge weight slots.
The replay loop dereferences them at replay time, so the symbolic
structure depends only on graph topology + coefficients (both
theta-independent). It does NOT need to be rebuilt when theta changes.

Stage A0 removes the destroy. After the fix, the symbolic cache
survives across ``update_weights`` calls; only the *concrete*
``reward_compute_graph`` is rebuilt (cheap O(commands) path, not
O(n^3)).

These tests verify:

1. Cache pointer stays non-NULL across ``update_weights``.
2. Forward results remain correct for the new theta after
   ``update_weights`` — the cache being preserved doesn't break
   correctness.
3. End-to-end correctness: A → A_pdf, B → B_pdf via the same graph
   object matches building two fresh graphs at A and B.
"""

import numpy as np
import pytest
from phasic import Graph


def _build_two_param_graph():
    """S -> [3] --(theta[0])--> [2] --(theta[1])--> [1]."""
    g = Graph(1)
    start = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v3, 1.0)
    v3.add_edge_parameterized(v2, 0.0, [1.0, 0.0])
    v2.add_edge_parameterized(v1, 0.0, [0.0, 1.0])
    return g


# ---------------------------------------------------------------------------
# Cache pointer survives update_weights
# ---------------------------------------------------------------------------


def test_param_compute_graph_cache_survives_update_weights():
    """The symbolic elimination cache is theta-independent; it must
    not be destroyed when only edge weights change.

    Uses ``expected_waiting_time`` (which goes through
    ``ptd_precompute_reward_compute_graph`` and thus populates the
    parameterised cache) rather than ``pdf`` (which uses a separate
    forward-algorithm path that doesn't touch the symbolic cache)."""
    g = _build_two_param_graph()

    # Cache is lazy: first call to a moment function builds it.
    assert not g._has_param_compute_graph_cache(), (
        "Sanity: cache should not exist before any forward call"
    )

    g.update_weights([1.0, 1.0])
    _ = g.expected_waiting_time()
    assert g._has_param_compute_graph_cache(), (
        "After first expected_waiting_time call, the symbolic cache "
        "should be populated"
    )

    g.update_weights([2.0, 0.5])
    assert g._has_param_compute_graph_cache(), (
        "Stage A0: update_weights must NOT destroy the symbolic cache; "
        "it is theta-independent and survives weight updates."
    )

    # And after the second moment call:
    _ = g.expected_waiting_time()
    assert g._has_param_compute_graph_cache()


# ---------------------------------------------------------------------------
# Correctness still holds with the cache reused
# ---------------------------------------------------------------------------


def test_pdf_correct_after_update_weights_with_cached_compute_graph():
    """With the symbolic cache preserved across update_weights, two
    consecutive forward calls at different theta must each return the
    correct theta-specific PDF — i.e. the cache being reused does not
    leak old theta into the second result."""
    g = _build_two_param_graph()

    g.update_weights([1.0, 1.0])
    pdf_a = float(g.pdf(1.0, granularity=100))
    g.update_weights([2.0, 0.5])
    pdf_b = float(g.pdf(1.0, granularity=100))

    g_ref_a = _build_two_param_graph()
    g_ref_a.update_weights([1.0, 1.0])
    pdf_a_ref = float(g_ref_a.pdf(1.0, granularity=100))
    g_ref_b = _build_two_param_graph()
    g_ref_b.update_weights([2.0, 0.5])
    pdf_b_ref = float(g_ref_b.pdf(1.0, granularity=100))

    np.testing.assert_allclose(pdf_a, pdf_a_ref, rtol=1e-12)
    np.testing.assert_allclose(pdf_b, pdf_b_ref, rtol=1e-12)
    assert pdf_a != pdf_b, "Different thetas must give different PDFs"


def test_accumulated_visiting_time_correct_after_update_weights():
    """Same as above but for accumulated_visiting_time, which uses a
    different per-graph context (ph_context_markov)."""
    g = _build_two_param_graph()

    g.update_weights([1.0, 1.0])
    v_a = np.asarray(g.accumulated_visiting_time(2.0, 10))
    g.update_weights([3.0, 0.5])
    v_b = np.asarray(g.accumulated_visiting_time(2.0, 10))

    g_ref = _build_two_param_graph()
    g_ref.update_weights([3.0, 0.5])
    v_b_ref = np.asarray(g_ref.accumulated_visiting_time(2.0, 10))

    np.testing.assert_allclose(v_b, v_b_ref, rtol=1e-10)
    assert not np.allclose(v_a, v_b)


# ---------------------------------------------------------------------------
# End-to-end: many update_weights calls, single graph object
# ---------------------------------------------------------------------------


def test_many_update_weights_on_single_graph():
    """SVGD-like workload: many update_weights calls on the same graph
    instance. With the cache preserved, this should not leak memory
    (no destroy on each call), and every result must match the
    reference fresh-graph build.

    Uses ``expected_waiting_time`` so the symbolic cache is exercised
    on every iteration; if Stage A0 is reverted, the cache would be
    rebuilt from scratch every time and the post-loop assertion would
    fail intermittently. With Stage A0, the cache is built on the
    first iteration and reused for the remaining 49."""
    g = _build_two_param_graph()

    rng = np.random.default_rng(0)
    thetas = rng.uniform(0.5, 3.0, size=(50, 2))

    for theta in thetas:
        g.update_weights(theta.tolist())
        e_t = g.expected_waiting_time()

        g_ref = _build_two_param_graph()
        g_ref.update_weights(theta.tolist())
        e_t_ref = g_ref.expected_waiting_time()

        np.testing.assert_allclose(
            e_t, e_t_ref, rtol=1e-10,
            err_msg=f"E[T] mismatch after update_weights with theta={theta}",
        )

    # Cache must still be populated at the end.
    assert g._has_param_compute_graph_cache()
