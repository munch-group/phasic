"""End-to-end tests for Graph.update_ipv (C path).

Per update_ipv-plan.md. The plan describes Batch-1 checks as ".c
files in src/c/test/", but this repo's actual test discipline is
pytest-through-pybind, so the C primitive's behaviour is exercised
here via the Python surface that thinly wraps it. The four
plan-listed C checks (scalar round-trip, symbolic-cache survival,
non-IPV-edge skip, length mismatch) are mapped to corresponding
pytest cases below.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasic import Graph, with_ipv
import phasic.cache as cache


# ---------------------------------------------------------------------------
# Small helper graphs.
# ---------------------------------------------------------------------------


def _build_two_ipv_param_graph():
    """Param graph with two IPV edges (start -> v_a, start -> v_b) and a
    parameterised interior edge from each so update_weights / update_ipv
    are both meaningful.

    Topology:
        start --[ipv0]--> v_a --[theta_0]--> sink
        start --[ipv1]--> v_b --[theta_1]--> sink
    """
    g = Graph(1)
    start = g.starting_vertex()
    v_a = g.find_or_create_vertex([10])
    v_b = g.find_or_create_vertex([20])
    sink = g.find_or_create_vertex([0])
    # Interior parameterised edges first so param_length locks at 2.
    v_a.add_edge(sink, [1.0, 0.0])  # weight = theta[0]
    v_b.add_edge(sink, [0.0, 1.0])  # weight = theta[1]
    # IPV edges (scalar, length-1 coefficients).
    start.add_edge(v_a, 0.5)
    start.add_edge(v_b, 0.5)
    return g, v_a, v_b


def _build_cyclic_param_graph():
    """Small cyclic parameterised graph used to verify symbolic-cache
    survival across update_ipv.

    The two IPV edges target *different* vertices so the IPV split
    actually changes the expectation:

        start --[ipv0]--> v1 --[theta_0]--> v2 --[theta_1]--> v1   (cycle)
        start --[ipv1]--> v3 --[theta_0]--> sink                v2 --> sink
    """
    g = Graph(1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    v3 = g.find_or_create_vertex([3])
    sink = g.find_or_create_vertex([0])
    v1.add_edge(v2, [1.0, 0.0])
    v2.add_edge(v1, [0.0, 0.5])  # back-edge → cycle
    v2.add_edge(sink, [0.0, 0.5])
    v3.add_edge(sink, [1.0, 0.0])
    start.add_edge(v1, 0.7)
    start.add_edge(v3, 0.3)
    return g


# ---------------------------------------------------------------------------
# Batch 1: C-primitive coverage (via the Python binding).
# ---------------------------------------------------------------------------


class TestCPrimitive:
    """Mirrors the four plan-listed src/c/test/ checks."""

    def test_scalar_round_trip(self):
        # Plan: test_update_ipv_scalar.c
        g, _, _ = _build_two_ipv_param_graph()
        g.update_ipv([0.4, 0.6])
        weights = [e.weight() for e in g.starting_vertex().edges()]
        assert weights == [pytest.approx(0.4), pytest.approx(0.6)]

    def test_preserves_symbolic_cache(self, monkeypatch):
        # Plan: test_update_ipv_preserves_symbolic_cache.c
        # Stage A0 invariant: the symbolic compute graph survives IPV updates.
        # We assert via the on-disk cache file count: a single elimination
        # produces one file; subsequent update_ipv + expectation calls must
        # not produce additional cache entries. The reward-compute cache is
        # off by default, so opt in to observe the file count.
        monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")
        cache.clear_param_compute_cache()
        g = _build_cyclic_param_graph()
        g.update_weights([1.5, 2.5])
        g.expectation()  # populates the symbolic cache
        files_after_first = cache.param_compute_cache_info()['n_files']
        assert files_after_first == 1

        g.update_ipv([0.2, 0.8])
        g.expectation()
        g.update_ipv([0.9, 0.1])
        g.expectation()
        assert cache.param_compute_cache_info()['n_files'] == 1

    def test_skips_non_ipv_edges(self):
        # Plan: test_update_ipv_skips_non_ipv.c
        g, v_a, v_b = _build_two_ipv_param_graph()
        g.update_weights([5.0, 7.0])  # interior edges weighted via theta
        g.update_ipv([0.3, 0.7])

        ipv_weights = [e.weight() for e in g.starting_vertex().edges()]
        assert ipv_weights == [pytest.approx(0.3), pytest.approx(0.7)]

        # Interior edges must be untouched.
        assert v_a.edges()[0].weight() == pytest.approx(5.0)
        assert v_b.edges()[0].weight() == pytest.approx(7.0)

    def test_length_mismatch_raises(self):
        # Plan: test_update_ipv_length_mismatch.c
        g, _, _ = _build_two_ipv_param_graph()  # 2 IPV edges
        with pytest.raises(RuntimeError, match=r"ipv length 1 does not match"):
            g.update_ipv([0.5])
        with pytest.raises(RuntimeError, match=r"does not match"):
            g.update_ipv([0.3, 0.3, 0.4])


# ---------------------------------------------------------------------------
# Batch 3 gate: scalar round-trip via the pybind binding (subsumed by
# TestCPrimitive::test_scalar_round_trip above).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Batch 4: Python-method validation and round-trip.
# ---------------------------------------------------------------------------


class TestPythonMethod:
    def test_validation_shape(self):
        g, _, _ = _build_two_ipv_param_graph()
        with pytest.raises(ValueError, match="1-dimensional"):
            g.update_ipv(np.array([[0.4, 0.6]]))

    def test_validation_empty(self):
        g, _, _ = _build_two_ipv_param_graph()
        with pytest.raises(ValueError, match="non-empty"):
            g.update_ipv(np.array([]))

    def test_validation_nan(self):
        g, _, _ = _build_two_ipv_param_graph()
        with pytest.raises(ValueError, match="NaN"):
            g.update_ipv(np.array([0.4, math.nan]))

    def test_validation_inf(self):
        g, _, _ = _build_two_ipv_param_graph()
        with pytest.raises(ValueError, match="infinite"):
            g.update_ipv(np.array([0.4, math.inf]))

    def test_round_trip_python_to_c(self):
        g, _, _ = _build_two_ipv_param_graph()
        g.update_ipv(np.array([0.25, 0.75]))
        weights = [e.weight() for e in g.starting_vertex().edges()]
        assert weights == [pytest.approx(0.25), pytest.approx(0.75)]

    def test_last_ipv_stashed(self):
        g, _, _ = _build_two_ipv_param_graph()
        g.update_ipv(np.array([0.3, 0.7]))
        assert getattr(g, "_last_ipv", None) is not None
        np.testing.assert_array_equal(g._last_ipv, np.array([0.3, 0.7]))


# ---------------------------------------------------------------------------
# Batch 5: end-to-end on a cyclic model.
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_update_ipv_changes_expectation_on_cyclic_graph(self):
        g = _build_cyclic_param_graph()
        g.update_weights([1.5, 2.5])
        e0 = g.expectation()

        g.update_ipv([0.1, 0.9])
        e1 = g.expectation()

        # The IPVs differ; expectation should differ too. (If the graph
        # were symmetric in such a way that any two IPVs gave the same
        # expectation, this assertion would be wrong — but the asymmetric
        # IPV split [0.1, 0.9] vs the [0.7, 0.3] baked in at construction
        # time guarantees a real change here.)
        assert not math.isclose(e0, e1, rel_tol=1e-12)

    def test_update_ipv_matches_fresh_build(self):
        # Build with default IPV [0.7, 0.3], then update_ipv to [0.4, 0.6].
        g = _build_cyclic_param_graph()
        g.update_weights([1.5, 2.5])
        g.update_ipv([0.4, 0.6])
        e_updated = g.expectation()

        # Reference: a fresh graph with the same IPV baked in.
        g_ref = Graph(1)
        start_ref = g_ref.starting_vertex()
        v1_ref = g_ref.find_or_create_vertex([1])
        v2_ref = g_ref.find_or_create_vertex([2])
        v3_ref = g_ref.find_or_create_vertex([3])
        sink_ref = g_ref.find_or_create_vertex([0])
        v1_ref.add_edge(v2_ref, [1.0, 0.0])
        v2_ref.add_edge(v1_ref, [0.0, 0.5])
        v2_ref.add_edge(sink_ref, [0.0, 0.5])
        v3_ref.add_edge(sink_ref, [1.0, 0.0])
        start_ref.add_edge(v1_ref, 0.4)
        start_ref.add_edge(v3_ref, 0.6)
        g_ref.update_weights([1.5, 2.5])
        e_ref = g_ref.expectation()

        assert e_updated == pytest.approx(e_ref, rel=1e-10)

    def test_symbolic_cache_survives_full_cycle(self, monkeypatch):
        # Same shape as TestCPrimitive::test_preserves_symbolic_cache, but
        # via the Python surface — the Stage A0 invariant must hold all
        # the way through. Opt into the (default-off) reward-compute cache.
        monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")
        cache.clear_param_compute_cache()
        g = _build_cyclic_param_graph()
        g.update_weights([1.5, 2.5]); g.expectation()
        g.update_ipv([0.2, 0.8]);     g.expectation()
        g.update_weights([2.0, 0.5]); g.expectation()
        g.update_ipv([0.9, 0.1]);     g.expectation()
        assert cache.param_compute_cache_info()['n_files'] == 1
