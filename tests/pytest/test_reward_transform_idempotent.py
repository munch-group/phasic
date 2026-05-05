"""Stage 2 verification — reward_transform is non-mutating and idempotent.

The public ``ptd_graph_reward_transform`` and
``ptd_graph_dph_reward_transform`` must guarantee that the caller's
input graph is not mutated. Internally, ``_ptd_graph_reward_transform``
permutes ``internal_vertices[j]->index`` (SCC topological sort) and
divides edge weights by per-vertex outgoing rate. It then attempts to
restore the source by multiplying the rates back and writing
``vertices[i]->index = i`` sequentially — which is fragile: it relies
on the source-graph vertices already being in topological order so
the sequential reset reverses the reorder.

Stage 2 makes the non-mutation guarantee explicit by cloning the
input via ``ptd_clone_graph`` before invoking the mutating internal
function. The contract these tests lock in:

- The source graph survives ``reward_transform`` calls bit-identical
  (same vertex indices, same edge weights, same forward-state
  output). On master before Stage 2, the source graph might
  *coincidentally* survive for graphs where the sequential restore
  matches the original ordering — but the contract was implicit and
  fragile. Stage 2 makes it explicit and unconditional.
- ``A → B → A`` produces equivalent first-A and third-A transformed
  graphs.
"""

import numpy as np
import pytest
from phasic import Graph


def _build_acyclic_graph():
    """Five-vertex acyclic graph with non-trivial structure."""
    g = Graph(1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    v3 = g.find_or_create_vertex([3])
    v4 = g.find_or_create_vertex([4])
    start.add_edge(v1, 1.0)
    v1.add_edge(v2, 2.0)
    v1.add_edge(v3, 1.5)
    v2.add_edge(v4, 0.7)
    v3.add_edge(v4, 1.1)
    return g


def _build_cyclic_graph():
    """Graph with a non-trivial SCC: v1 ↔ v2, both feed into absorbing v3.

    The SCC containing v1, v2 forces _ptd_graph_reward_transform's
    topological sort to permute their indices in a way the
    restoration loop at phasic.c:4010 cannot undo (it just resets
    index = i sequentially), exposing the bug.
    """
    g = Graph(1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    v3 = g.find_or_create_vertex([3])
    start.add_edge(v1, 1.0)
    v1.add_edge(v2, 1.0)
    v2.add_edge(v1, 0.5)  # back-edge, creates {v1, v2} SCC
    v1.add_edge(v3, 0.5)  # exit edge to absorbing
    v2.add_edge(v3, 0.5)
    return g


def _snapshot_graph(g):
    """Capture vertex indices and per-edge (target_idx, weight)."""
    snap = []
    for v in g.vertices():
        edges = []
        for e in v.edges():
            edges.append((e.to().index(), float(e.weight())))
        snap.append((v.index(), tuple(sorted(edges))))
    return tuple(snap)


# ---------------------------------------------------------------------------
# Mutation: source graph survives unchanged
# ---------------------------------------------------------------------------


def test_reward_transform_does_not_mutate_source_acyclic():
    g = _build_acyclic_graph()
    rewards_a = np.array([0, 1, 1, 0, 1], dtype=np.float64)
    rewards_b = np.array([1, 0, 1, 1, 0], dtype=np.float64)

    snap_before = _snapshot_graph(g)

    g.reward_transform(rewards_a)
    g.reward_transform(rewards_b)
    g.reward_transform(rewards_a)

    snap_after = _snapshot_graph(g)
    assert snap_before == snap_after, (
        "Source graph mutated by reward_transform. "
        "Before:\n  " + repr(snap_before) +
        "\nAfter:\n  " + repr(snap_after)
    )


def test_reward_transform_does_not_mutate_source_cyclic():
    """Cyclic graph: SCC reordering in _ptd_graph_reward_transform
    permutes vertex indices in a way the restoration loop at
    phasic.c:4010 cannot undo. This is the test that fails on
    unpatched master."""
    g = _build_cyclic_graph()
    rewards_a = np.array([0, 1, 1, 0], dtype=np.float64)
    rewards_b = np.array([1, 0, 1, 1], dtype=np.float64)

    pdf_before = float(g.pdf(1.0, granularity=100))

    g.reward_transform(rewards_a)
    g.reward_transform(rewards_b)
    g.reward_transform(rewards_a)

    pdf_after = float(g.pdf(1.0, granularity=100))
    np.testing.assert_allclose(
        pdf_after, pdf_before, rtol=1e-12,
        err_msg="Source graph PDF changed after reward_transform calls "
                "(reward_transform mutated the source's vertex indices "
                "or edge weights)",
    )


def test_reward_transform_uses_current_edge_weights():
    """Regression for a latent bug in ptd_clone_graph: the regular-
    edge clone path constructed new edges via ptd_graph_add_edge,
    which initialises ``edge->weight`` from ``sum(coefficients * 1)``
    (default theta = 1) — overwriting the source's actual weight.
    The clone therefore reflected theta=1 weights, not whatever the
    source had been updated to via update_weights.

    This was masked on master before Stage 2 because the public
    reward_transform mutated the source directly. After Stage 2
    (clone-first), the bug surfaced as silently wrong PDFs from
    reward_transform when called after update_weights.

    Fix: copy ``new_edge->weight = old_edge->weight`` after each
    ptd_graph_add_edge in ptd_clone_graph (src/c/phasic.c:1262).
    """
    g = Graph(1)
    start = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v3, 1.0)
    v3.add_edge_parameterized(v2, 0.0, [3.0])
    v2.add_edge_parameterized(v1, 0.0, [2.0])
    v1.add_edge_parameterized(g.find_or_create_vertex([0]), 0.0, [1.0])
    g.update_weights([2.0])  # weights become [6, 4, 2]

    pdf_source_theta2 = float(g.pdf(0.5, granularity=100))

    rewards = np.ones(g.vertices_length())
    g_t = g.reward_transform(rewards)
    pdf_transformed = float(g_t.pdf(0.5, granularity=100))

    # The transformed graph (with all-1 rewards) should reproduce the
    # source's PDF closely. Without the clone-weight fix, the
    # transformed graph would reflect theta=1 weights instead.
    np.testing.assert_allclose(
        pdf_transformed, pdf_source_theta2, rtol=1e-10,
        err_msg="reward_transform produced a graph that doesn't reflect "
                "the source's update_weights theta — ptd_clone_graph "
                "weight-copy regression",
    )


# ---------------------------------------------------------------------------
# Repeatability: A → B → A produces the same first-A result
# ---------------------------------------------------------------------------


def test_reward_transform_repeatable():
    """The third reward_transform with the same rewards as the first
    must yield a graph that produces the same forward-state values.

    Forward queries are the cleanest equivalence check here because
    reward_transform is allowed to allocate vertices in any order so
    the resulting graphs need not be structurally identical, only
    behaviourally identical."""
    g = _build_acyclic_graph()
    rewards_a = np.array([0, 1, 1, 0, 1], dtype=np.float64)
    rewards_b = np.array([1, 0, 1, 1, 0], dtype=np.float64)

    g_a1 = g.reward_transform(rewards_a)
    _ = g.reward_transform(rewards_b)
    g_a2 = g.reward_transform(rewards_a)

    times = [0.5, 1.0, 1.5, 2.0]
    for t in times:
        v1 = float(g_a1.pdf(t, granularity=100))
        v2 = float(g_a2.pdf(t, granularity=100))
        np.testing.assert_allclose(
            v1, v2, rtol=1e-10,
            err_msg=f"reward_transform not repeatable at t={t}"
        )


# ---------------------------------------------------------------------------
# Multivariate end-to-end: compute_pmf_and_moments with two features
# ---------------------------------------------------------------------------


def test_multivariate_two_features_match_independent_builds():
    """The multivariate path in graph_builder.cpp calls
    ``g.reward_transform(rewards_2d[j])`` once per feature on the same
    graph instance ``g``. Without Stage 2, the second feature's
    transform sees a corrupted graph. Verify by building a reference
    where each feature is computed against a freshly-built graph."""

    pytest.importorskip("phasic")
    import json
    import phasic

    GraphBuilder = phasic.parameterized.GraphBuilder

    def callback(state, **kwargs):
        n = state[0]
        if n <= 0:
            return []
        return [(np.array([n - 1]), [float(n)])]

    graph = Graph(callback, ipv=[3])
    sd = graph.serialize()
    sj = json.dumps(
        {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in sd.items()}
    )

    builder = GraphBuilder(sj)
    n_vertices = builder.vertices_length

    theta = np.array([1.5])
    times = np.array([[1.0], [2.0]])  # (n_times=2, n_features=1)

    # One feature → easy reference
    rewards_a = np.ones((n_vertices, 1)) * 1.0
    rewards_b = np.ones((n_vertices, 1)) * 2.0

    pdf_a_alone = builder.compute_pmf_multivariate(
        theta, times, rewards_a,
        discrete=False, granularity=100, compute_joint=False,
    )
    pdf_b_alone = builder.compute_pmf_multivariate(
        theta, times, rewards_b,
        discrete=False, granularity=100, compute_joint=False,
    )

    # Two features in one call: shapes (n_times=2, n_features=2)
    times_2 = np.hstack([times, times])
    rewards_ab = np.hstack([rewards_a, rewards_b])

    pdf_combined = builder.compute_pmf_multivariate(
        theta, times_2, rewards_ab,
        discrete=False, granularity=100, compute_joint=False,
    )

    # Each combined column must match the corresponding alone result.
    np.testing.assert_allclose(
        pdf_combined[:, 0], pdf_a_alone[:, 0], rtol=1e-10,
        err_msg="Feature 0 PDF differs between combined and alone runs "
                "(reward_transform mutation likely)",
    )
    np.testing.assert_allclose(
        pdf_combined[:, 1], pdf_b_alone[:, 0], rtol=1e-10,
        err_msg="Feature 1 PDF differs between combined and alone runs "
                "(reward_transform mutation likely)",
    )
