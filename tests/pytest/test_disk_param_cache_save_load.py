"""Stage B1 verification — save/load of parameterized_reward_compute_graph.

These tests exercise the new C functions in isolation
(``ptd_save_parameterized_reward_compute_graph`` and
``ptd_load_parameterized_reward_compute_graph`` from
``api/c/phasic.h``) without going through the production cache
hook in ``ptd_precompute_reward_compute_graph`` — that hook is
Stage B2.

The test introspection bindings ``Graph._save_param_compute_graph``
and ``Graph._load_param_compute_graph`` are not part of the public
API; they exist purely for these tests.

Round-trip strategy:

1. Build a parameterised graph with several non-IPV parameter edges.
2. Trigger a forward operation that populates
   ``parameterized_reward_compute_graph`` (e.g. ``expected_waiting_time``).
3. Save the populated cache to a temp file.
4. Build a *fresh* graph with the same structure.
5. Load the cache file onto the fresh graph.
6. Run forward operations on the fresh graph and assert results
   match a reference computed via direct elimination on a third
   fresh graph (i.e. correctness equivalent to "no cache used").
"""

import os
import tempfile

import numpy as np
import pytest

from phasic import Graph


def _build_two_param_graph():
    """S -> [3] --(theta[0])--> [2] --(theta[1])--> [1] (absorbing)."""
    g = Graph(1)
    start = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v3, 1.0)
    v3.add_edge_parameterized(v2, 0.0, [1.0, 0.0])
    v2.add_edge_parameterized(v1, 0.0, [0.0, 1.0])
    return g


def _build_chain_graph(n_states):
    """N-state coalescent-style chain. Each non-absorbing vertex k has
    one parameterised outgoing edge to k-1 with coefficient k(k-1)/2."""

    def cb(state, **kwargs):
        n = state[0]
        if n <= 1:
            return []
        return [(np.array([n - 1]), [float(n * (n - 1) / 2)])]

    return Graph(cb, ipv=[n_states])


# ---------------------------------------------------------------------------
# Save -> load round trip on the same graph
# ---------------------------------------------------------------------------


def test_save_load_round_trip_small():
    g_src = _build_two_param_graph()
    g_src.update_weights([2.0, 1.5])
    # Populate the parameterised compute graph by computing moments.
    e_t_src = g_src.expected_waiting_time()
    assert g_src._has_param_compute_graph_cache(), "Sanity"

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "small.bin")
        g_src._save_param_compute_graph(path)
        assert os.path.getsize(path) > 0

        # Fresh graph with identical structure; load the cache onto it.
        g_dst = _build_two_param_graph()
        g_dst.update_weights([2.0, 1.5])
        # Loading installs the cache on the destination graph.
        g_dst._load_param_compute_graph(path)
        assert g_dst._has_param_compute_graph_cache()

        e_t_dst = g_dst.expected_waiting_time()
        np.testing.assert_allclose(
            e_t_dst, e_t_src, rtol=1e-12,
            err_msg="loaded graph produced different expected_waiting_time",
        )


def test_save_load_with_different_theta():
    """The cache is theta-independent (Stage A0). After load, calling
    update_weights with a *different* theta must still produce
    correct results — i.e. the cache references edge weights via
    pointer, and update_weights flowing into those pointers is
    correctly handled."""
    g_src = _build_two_param_graph()
    g_src.update_weights([1.0, 1.0])
    g_src.expected_waiting_time()  # populates cache

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "g.bin")
        g_src._save_param_compute_graph(path)

        g_dst = _build_two_param_graph()
        g_dst.update_weights([1.0, 1.0])
        g_dst._load_param_compute_graph(path)
        # Now change theta on the loaded graph.
        g_dst.update_weights([3.0, 0.5])
        e_t_dst = g_dst.expected_waiting_time()

        # Reference: fresh graph at theta=[3.0, 0.5] (no cache).
        g_ref = _build_two_param_graph()
        g_ref.update_weights([3.0, 0.5])
        e_t_ref = g_ref.expected_waiting_time()

        np.testing.assert_allclose(e_t_dst, e_t_ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# Save -> load on graphs of varying size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_states", [10, 50, 200])
def test_save_load_round_trip_chain(n_states):
    g_src = _build_chain_graph(n_states)
    g_src.update_weights([1.0])
    e_t_src = g_src.expected_waiting_time()
    assert g_src._has_param_compute_graph_cache()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, f"chain{n_states}.bin")
        g_src._save_param_compute_graph(path)

        g_dst = _build_chain_graph(n_states)
        g_dst.update_weights([1.0])
        g_dst._load_param_compute_graph(path)

        e_t_dst = g_dst.expected_waiting_time()
        np.testing.assert_allclose(
            e_t_dst, e_t_src, rtol=1e-12,
            err_msg=f"chain n={n_states} mismatch after load",
        )


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_load_missing_file():
    g = _build_two_param_graph()
    g.update_weights([1.0, 1.0])
    with pytest.raises(RuntimeError, match="cache miss|fopen"):
        g._load_param_compute_graph("/tmp/this-file-does-not-exist-12345.bin")


def test_load_corrupt_file_falls_through():
    g = _build_two_param_graph()
    g.update_weights([1.0, 1.0])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "garbage.bin")
        # Write enough random bytes to trip the magic/version check.
        with open(path, "wb") as f:
            f.write(b"NOTPHASIC" + b"\x00" * 200)
        with pytest.raises(RuntimeError, match="wrong magic|version|short"):
            g._load_param_compute_graph(path)


def test_save_with_no_cache_raises():
    g = _build_two_param_graph()
    # Don't trigger any forward call → cache is NULL.
    assert not g._has_param_compute_graph_cache()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "noop.bin")
        with pytest.raises(RuntimeError, match="no parameterized"):
            g._save_param_compute_graph(path)
