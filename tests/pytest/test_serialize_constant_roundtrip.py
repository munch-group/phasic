"""Constant graphs must stay constant through serialize() -> from_serialized().

Plan: graph-cache-constant-roundtrip-plan.md (Part 1 of
constant-parameterized-consistency-plan.md).

A constant edge is stored as a one-element coefficient array holding the
rate, so a constant graph has param_length() == 1. serialize() used to
default theta_dim to param_length() and export every such edge as a
parameterized edge, so a constant graph (including one loaded from the
on-disk graph cache) came back parameterized and silently theta-scaled.
"""
import numpy as np
import pytest

import phasic
import phasic.callback_hash as callback_hash
import phasic.graph_cache as graph_cache
from phasic import Graph, with_ipv

RTOL = 1e-14
N = 4


@with_ipv([N] + [0] * (N - 1))
def _coalescent_constant(state):
    """Kingman coalescent with FLOAT rates (a constant model)."""
    transitions = []
    for i in range(state.size):
        for j in range(i, state.size):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[i + j + 1] += 1
            transitions.append([new, state[i] * (state[j] - same) / (1 + same)])
    return transitions


@with_ipv([N] + [0] * (N - 1))
def _coalescent_param(state):
    """Same coalescent with coefficient-list rates (a 1-parameter model)."""
    transitions = []
    for i in range(state.size):
        for j in range(i, state.size):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[i + j + 1] += 1
            transitions.append([new, [state[i] * (state[j] - same) / (1 + same)]])
    return transitions


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Point the graph cache at an empty temporary directory."""
    d = tmp_path / "graphs"
    d.mkdir()
    monkeypatch.setattr(graph_cache, "DEFAULT_CACHE_DIR", d)
    monkeypatch.delenv("PHASIC_DISABLE_GRAPH_CACHE", raising=False)
    return d


def _constant_chain():
    """start -> [3] -> [2] -> [1] with scalar rates."""
    g = Graph(1)
    s = g.starting_vertex()
    a, b, c = (g.find_or_create_vertex([k]) for k in (3, 2, 1))
    s.add_edge(a, 1.0)
    a.add_edge(b, 2.0)
    b.add_edge(c, 0.5)
    return g


def _two_param_chain():
    g = Graph(1)
    s = g.starting_vertex()
    a, b, c = (g.find_or_create_vertex([k]) for k in (3, 2, 1))
    s.add_edge(a, 1.0)
    a.add_edge(b, [2.0, 0.0])
    b.add_edge(c, [0.0, 0.5])
    return g


def _assert_same_distribution(g1, g2, discrete=False):
    if discrete:
        for n in (1, 2, 3, 5, 8):
            np.testing.assert_allclose(g2.pdf(n), g1.pdf(n), rtol=RTOL)
        return
    np.testing.assert_allclose(np.asarray(g2.moments(3)),
                               np.asarray(g1.moments(3)), rtol=RTOL)
    for t in (0.1, 0.7, 2.0):
        np.testing.assert_allclose(g2.pdf(t), g1.pdf(t), rtol=RTOL)


# --------------------------------------------------------------- T1: the cache
def test_t1_constant_callback_graph_stays_constant_through_cache(tmp_cache):
    fresh = Graph(_coalescent_constant, graph_cache=True)   # miss: built, saved
    assert not fresh.parameterized()
    assert any(tmp_cache.iterdir()), "graph was not saved to the cache"
    # Load errors are swallowed (load_graph returns None, Graph rebuilds), so
    # check the hit directly: a silent miss must not let this test pass.
    hit = graph_cache.GraphCache().load_graph(fresh._callback)
    assert hit is not None and not hit.parameterized()
    loaded = Graph(_coalescent_constant, graph_cache=True)  # hit
    assert not loaded.parameterized()
    with pytest.raises(RuntimeError, match="constant graph"):
        loaded.update_weights([2.0])
    _assert_same_distribution(fresh, loaded)


# ------------------------------------------------- T2: default serialize()
def test_t2_constant_graph_default_serialize_has_no_param_edges():
    g = _constant_chain()
    ser = g.serialize()
    assert ser["param_length"] == 0
    assert len(ser["param_edges"]) == 0
    assert len(ser["edges"]) == 2
    g2 = Graph.from_serialized(ser)
    assert not g2.parameterized()
    _assert_same_distribution(g, g2)


# -------------------------------------------- T3: parameterized unchanged
@pytest.mark.parametrize("make", [
    lambda: Graph(_coalescent_param, graph_cache=True), _two_param_chain])
def test_t3_parameterized_default_serialize_unchanged(make, tmp_cache):
    g = make()
    assert g.parameterized()
    a, b = g.serialize(), g.serialize(theta_dim=g.param_length())
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], np.ndarray):
            np.testing.assert_array_equal(a[k], b[k])
        else:
            assert a[k] == b[k], k
    g2 = Graph.from_serialized(a)
    assert g2.parameterized() and g2.param_length() == g.param_length()


# --------------------------------------- T4: discretized / native DPH
def test_t4_discretized_constant_graph_roundtrip():
    d = _constant_chain().discretize(0.1)
    assert not d.parameterized() and d.get_was_dph()
    d2 = Graph.from_serialized(d.serialize())
    assert not d2.parameterized()
    assert d2.is_discrete and d2.get_was_dph()
    _assert_same_distribution(d, d2, discrete=True)


def test_t4_native_dph_constant_graph_roundtrip():
    g = Graph(1)
    s = g.starting_vertex()
    a, b = g.find_or_create_vertex([2]), g.find_or_create_vertex([1])
    s.add_edge(a, 1.0)
    a.add_edge(b, 0.3)
    g.is_discrete = True
    g2 = Graph.from_serialized(g.serialize())
    assert not g2.parameterized() and g2.is_discrete
    _assert_same_distribution(g, g2, discrete=True)


# ------------------------------------------------ T4b: UNLOCKED graphs
def test_t4b_unlocked_graphs_roundtrip_constant():
    p = _two_param_chain()
    p.update_weights([1.0, 1.0])
    rt = p.reward_transform(np.array([0.0, 2.0, 1.0, 0.0]))
    c = _constant_chain()
    lt = c.laplace_transform(0.5)
    for g in (rt, lt):
        assert not g.parameterized()
        g2 = Graph.from_serialized(g.serialize())
        assert not g2.parameterized()
        np.testing.assert_allclose(np.asarray(g2.moments(2)),
                                   np.asarray(g.moments(2)), rtol=RTOL)
        with pytest.raises(RuntimeError, match="constant graph"):
            g2.update_weights([2.0])


# ----------------------------------------- T4c: constant-edge variants
def test_t4c_constant_graph_with_add_edge_constant_roundtrip():
    g = _constant_chain()
    a = g.find_vertex([3])
    c = g.find_vertex([1])
    a.add_edge_constant(c, 0.25)
    g2 = Graph.from_serialized(g.serialize())
    assert not g2.parameterized()
    _assert_same_distribution(g, g2)


def test_t4c_set_param_length_one_then_constant_edges():
    g = Graph(1)
    g.set_param_length(1)
    s = g.starting_vertex()
    a, b = g.find_or_create_vertex([2]), g.find_or_create_vertex([1])
    s.add_edge(a, 1.0)
    a.add_edge(b, 3.0)
    g2 = Graph.from_serialized(g.serialize())
    assert g2.parameterized() == g.parameterized()
    _assert_same_distribution(g, g2)


# ---------------------------------------------- T5: explicit theta_dim
def test_t5_explicit_theta_dim_wins():
    ser = _constant_chain().serialize(theta_dim=1)
    assert ser["param_length"] == 1
    assert len(ser["param_edges"]) == 2


# -------------------------------------------- T6: stale cache entries
def test_t6_stale_entry_from_old_version_is_not_loaded(tmp_cache, monkeypatch):
    """An entry in the OLD (pre-fix) layout, saved under version 1.0, must
    not be loaded after the bump: the graph is rebuilt and is constant."""
    assert callback_hash.PHASIC_CALLBACK_VERSION != "1.0"
    new_version = callback_hash.PHASIC_CALLBACK_VERSION
    old = Graph(_coalescent_constant, graph_cache=False)
    # Write the stale layout: what serialize() produced before the fix.
    real_serialize = Graph.serialize
    monkeypatch.setattr(Graph, "serialize",
                        lambda self, theta_dim=None, **kw: real_serialize(self, theta_dim=1, **kw))
    monkeypatch.setattr(callback_hash, "PHASIC_CALLBACK_VERSION", "1.0")
    graph_cache.GraphCache().save_graph(old, old._callback)
    stale = graph_cache.GraphCache().load_graph(old._callback)
    assert stale is not None and stale.parameterized(), "stale entry not written"
    monkeypatch.setattr(Graph, "serialize", real_serialize)
    monkeypatch.setattr(callback_hash, "PHASIC_CALLBACK_VERSION", new_version)
    assert graph_cache.GraphCache().load_graph(old._callback) is None
    rebuilt = Graph(_coalescent_constant, graph_cache=True)
    assert not rebuilt.parameterized()
    with pytest.raises(RuntimeError, match="constant graph"):
        rebuilt.update_weights([2.0])


# -------------------------------- T7/T8: builders refuse constant graphs
def test_t7_moments_from_graph_refuses_constant_graph():
    with pytest.raises(ValueError, match="must have parameterized edges"):
        Graph.moments_from_graph(_constant_chain(), nr_moments=1)


@pytest.mark.parametrize("use_ffi", [False, True])
def test_t8_pmf_and_moments_from_graph_refuses_constant_graph(use_ffi):
    with pytest.raises(ValueError, match="must have parameterized edges"):
        Graph.pmf_and_moments_from_graph(_constant_chain(), nr_moments=1,
                                         use_ffi=use_ffi)
