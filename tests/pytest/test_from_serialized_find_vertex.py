"""Regression: ``Graph.from_serialized`` must register vertices in the graph's
AVL state-lookup tree so ``Graph.find_vertex`` works on a reloaded graph.

Provenance
----------
A ``from_serialized`` change (Q6b, commit ``2f60ea37``) rebuilt *every* vertex
with ``create_vertex`` -- which never inserts into the AVL lookup tree -- in
order to preserve the per-index identity of duplicate-state aux vertices (the
many all-zero-state ``discretize`` aux vertices must not be merged).

That silently broke ``Graph.find_vertex`` on ANY graph reloaded via
``from_serialized``: the lookup tree was effectively empty, so ``find_vertex``
raised ``RuntimeError: No such vertex found`` for every state. Because the
on-disk graph cache (``~/.phasic_cache/graphs``) loads graphs through
``from_serialized``, ``Graph.joint_prob_graph`` -- which does
``self.find_vertex(base_state)`` -- crashed for any user whose graph cache was
warm. It only "passed" on a cold cache (fresh ``find_or_create_vertex`` build).

The fix registers the FIRST occurrence of each state via
``find_or_create_vertex`` (so ``find_vertex`` works) while still using
``create_vertex`` for later duplicate-state occurrences (so the aux-vertex
identity fix is preserved).
"""
from functools import partial
from itertools import combinations_with_replacement

import numpy as np
import pytest

import phasic
from phasic import Graph


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
def _plain_chain() -> Graph:
    """S -> [3] -> [2] -> [1], all UNIQUE states, plain (non-parameterized)."""
    g = Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge(v2, 2.0)
    v2.add_edge(v1, 3.0)
    return g


def _dup_state_graph() -> Graph:
    """Two DISTINCT vertices sharing state [2] (like discretize aux vertices)."""
    g = Graph(1)
    s = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2a = g.create_vertex([2])        # create_vertex == no dedup
    v2b = g.create_vertex([2])        # second distinct vertex, same state [2]
    v3 = g.find_or_create_vertex([3])
    s.add_edge(v1, 1.0)
    v1.add_edge(v2a, 1.0)
    v1.add_edge(v2b, 1.0)
    v2a.add_edge(v3, 1.0)
    v2b.add_edge(v3, 1.0)
    return g


def _coalescent_graph_and_indexer():
    """The tied-vs-free LRT coalescent base graph (nr_samples=4)."""
    from phasic import with_ipv, StateIndexer, Property
    all_pairs = partial(combinations_with_replacement, r=2)
    nr_samples = 4
    indexer = StateIndexer(
        lineages=[Property("descendants", min_value=1, max_value=nr_samples)]
    )

    @with_ipv([nr_samples] + [0] * (nr_samples - 1))
    def coal(state, indexer=None):
        t = []
        for i, j in all_pairs(range(indexer.lineages.state_length)):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy(); new[i] -= 1; new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            t.append([new, [state[i] * (state[j] - same) / (1 + same)]])
        return t

    return Graph(coal, indexer=indexer), indexer


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_find_vertex_after_from_serialized_unique_states():
    """Core regression: every unique state is findable after a round-trip."""
    g = _plain_chain()
    reloaded = Graph.from_serialized(g.serialize())
    for st in ([3], [2], [1]):
        v = reloaded.find_vertex(st)          # must NOT raise "No such vertex"
        assert list(v.state()) == st


def test_duplicate_state_vertices_not_merged_after_roundtrip():
    """The fix must NOT re-introduce the Q6b merge: duplicate-state vertices
    stay distinct, and the unique states remain findable."""
    g = _dup_state_graph()
    n0 = g.vertices_length()
    reloaded = Graph.from_serialized(g.serialize())
    assert reloaded.vertices_length() == n0        # [2] duplicate NOT merged away
    # Unique states + the first [2] occurrence resolve via the lookup tree.
    assert list(reloaded.find_vertex([1]).state()) == [1]
    assert list(reloaded.find_vertex([3]).state()) == [3]
    assert list(reloaded.find_vertex([2]).state()) == [2]   # first occurrence


def test_joint_prob_graph_on_reloaded_graph_matches_original():
    """The exact operation that crashed in the wild: build a joint-prob graph
    from a ``from_serialized``-reloaded base graph. Must succeed AND be
    functionally identical to the non-reloaded build -- same vertex count and
    the same joint-probability distribution (not merely the same vertex count,
    which a mis-wiring that returned a wrong-but-existing base vertex could
    coincidentally match)."""
    graph, indexer = _coalescent_graph_and_indexer()
    kw = dict(reward_limit=5, mutation_rate=1e-4)

    jpg_orig = graph.joint_prob_graph(indexer, **kw)

    reloaded = Graph.from_serialized(graph.serialize())
    jpg_reload = reloaded.joint_prob_graph(indexer, **kw)   # was: "No such vertex"

    assert jpg_reload.vertices_length() == jpg_orig.vertices_length()

    # Functional equivalence: the two joint graphs must produce the SAME joint
    # distribution. Aligning on the outcome-state columns guards against any
    # row-order or wrong-vertex divergence that a bare count check would miss.
    theta = [1 / 10_000, 1e-4]
    jpg_orig.update_weights(theta)
    jpg_reload.update_weights(theta)
    cols = list(jpg_orig.joint_prob_table().columns)
    t_orig = jpg_orig.joint_prob_table().sort_values(cols).reset_index(drop=True)
    t_reload = jpg_reload.joint_prob_table().sort_values(cols).reset_index(drop=True)
    # Same set of outcomes...
    np.testing.assert_array_equal(
        t_orig[cols[:-1]].to_numpy(), t_reload[cols[:-1]].to_numpy()
    )
    # ...at the same probabilities.
    np.testing.assert_allclose(
        t_orig["prob"].to_numpy(), t_reload["prob"].to_numpy(),
        rtol=1e-12, atol=1e-14,
    )


def test_joint_prob_graph_via_warm_graph_cache(tmp_path, monkeypatch):
    """End-to-end trigger: a cache HIT reloads the base graph through
    ``from_serialized``; ``joint_prob_graph`` on that cache-hit graph must not
    crash. Mirrors the real warm-``~/.phasic_cache/graphs`` failure."""
    import phasic.graph_cache as gc
    monkeypatch.setattr(gc, "DEFAULT_CACHE_DIR", tmp_path / "graphs")

    # Spy on the cache load so we can PROVE the second build is a genuine HIT
    # (reload via from_serialized). Without this, a silent cache MISS -- e.g. an
    # unhashable callback, which is only a WARNING -- would fall through to a
    # fresh find_or_create build that also works now, so the test could pass
    # without ever exercising the reload path this regression is about.
    load_hits = {"non_none": 0}
    _orig_load = gc.GraphCache.load_graph

    def _spy_load(self, callback, **params):
        res = _orig_load(self, callback, **params)
        if res is not None:
            load_hits["non_none"] += 1
        return res

    monkeypatch.setattr(gc.GraphCache, "load_graph", _spy_load)

    from phasic import with_ipv, StateIndexer, Property
    all_pairs = partial(combinations_with_replacement, r=2)
    nr_samples = 4

    def build():
        indexer = StateIndexer(
            lineages=[Property("descendants", min_value=1, max_value=nr_samples)]
        )

        @with_ipv([nr_samples] + [0] * (nr_samples - 1))
        def coal(state, indexer=None):
            t = []
            for i, j in all_pairs(range(indexer.lineages.state_length)):
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                new = state.copy(); new[i] -= 1; new[j] -= 1
                new[min(i + j + 1, state.size - 1)] += 1
                t.append([new, [state[i] * (state[j] - same) / (1 + same)]])
            return t

        return Graph(coal, indexer=indexer, graph_cache=True), indexer

    kw = dict(reward_limit=5, mutation_rate=1e-4)
    g_miss, idx_miss = build()                       # cache MISS: fresh build + write
    jpg_miss = g_miss.joint_prob_graph(idx_miss, **kw)

    g_hit, idx_hit = build()                         # cache HIT: reload via from_serialized
    # Gate: the second build MUST have hit the cache (reloaded via
    # from_serialized), otherwise this test proves nothing about the reload path.
    assert load_hits["non_none"] >= 1, (
        "second build was not a cache hit; the from_serialized reload path was "
        "never exercised (did the callback fail to hash?)"
    )
    jpg_hit = g_hit.joint_prob_graph(idx_hit, **kw)  # regression: crashed here

    assert jpg_hit.vertices_length() == jpg_miss.vertices_length()
