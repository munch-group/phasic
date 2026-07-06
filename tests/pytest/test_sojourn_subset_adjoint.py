"""GATE — expected_sojourn_time(subset) adjoint equivalence + O(n) memory.

``ptd_expected_sojourn_time_subset`` computes the expected sojourn (residence)
time at ``k`` target vertices. It used to seed one one-hot reward column per
target and replay the elimination trace forward through an ``n x k`` dense
matrix — O(n*k) memory. On a joint-probability graph the targets are the
t-vertices (one per joint outcome), so ``k ~ n`` and the allocation is O(n^2):
a coalescent joint-prob graph over 8 samples (n=684226, k=279936) asked for a
1.5 PB matrix and crashed.

The routine now runs the equivalent **adjoint** (reverse-mode) pass: the sojourn
at v is exactly d results[0] / d seed[v], the gradient of the start-vertex read
w.r.t. the reward seed, which reverse-mode differentiation delivers for ALL v in
one O(len(trace)) pass with O(n) memory. These tests pin the adjoint against the
independent forward ``expected_sojourn_time()`` (no-arg, n x n, Kahan-summed)
reference and against the legacy forward subset (escape hatch), and check the
deficit-sink NaNs never leak into the finite t-vertex targets.
"""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
import pytest

from phasic import Graph, Property, StateIndexer, with_ipv


def _make_joint(n_samples: int, reward_limit: int, discrete: bool,
                mut: float = 1.0) -> Graph:
    indexer = StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=n_samples)]
    )
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = n_samples

    @with_ipv(ipv)
    def coal(state, indexer=None):
        tr = []
        for i, j in combinations_with_replacement(
            range(indexer.lineages.state_length), 2
        ):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            tr.append([new, [state[i] * (state[j] - same) / (1 + same)]])
        return tr

    g = Graph(coal, indexer=indexer).joint_prob_graph(
        indexer, mutation_rate=mut, reward_limit=reward_limit, discrete=discrete
    )
    g.update_weights([0.5, mut])
    return g


def _t_vertices(g: Graph) -> np.ndarray:
    return np.array(
        [v.index() for v in g.vertices()
         if v.edges_length() and any(len(e.to().edges()) == 0 for e in v.edges())],
        dtype=int,
    )


CONFIGS = [
    (3, 1, False), (3, 2, False), (4, 3, False), (5, 2, False),
    (4, 3, True),  (3, 2, True),
]


@pytest.mark.parametrize("n_samples,reward_limit,discrete", CONFIGS)
def test_subset_adjoint_matches_full_forward(n_samples, reward_limit, discrete):
    """Adjoint subset == independent no-arg forward (n x n) at every finite vertex."""
    g = _make_joint(n_samples, reward_limit, discrete)
    n = g.vertices_length()

    full = np.asarray(g.expected_sojourn_time())              # forward n x n, Kahan
    subset_all = np.asarray(g.expected_sojourn_time(list(range(n))))  # adjoint

    fin = np.isfinite(full)
    assert fin.sum() >= n - 2  # only the deficit-sink vertices may be non-finite
    np.testing.assert_allclose(
        subset_all[fin], full[fin], rtol=1e-9, atol=1e-12,
        err_msg="adjoint subset disagrees with forward full sojourn",
    )


@pytest.mark.parametrize("n_samples,reward_limit,discrete", CONFIGS)
def test_tvertex_targets_finite_and_correct(n_samples, reward_limit, discrete):
    """The t-vertices (joint-probability targets) are finite and match the ref —
    the trash/deficit-sink NaNs must not contaminate them via the reverse pass."""
    g = _make_joint(n_samples, reward_limit, discrete)
    tvs = _t_vertices(g)
    assert tvs.size > 0

    soj = np.asarray(g.expected_sojourn_time(tvs.tolist()))
    assert np.isfinite(soj).all(), "t-vertex sojourn leaked NaN/inf"

    full = np.asarray(g.expected_sojourn_time())
    np.testing.assert_allclose(soj, full[tvs], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("n_samples,reward_limit,discrete", CONFIGS)
def test_subset_gather_is_a_slice_of_full(n_samples, reward_limit, discrete):
    """subset(indices) is exactly full-adjoint indexed at `indices` (bit-identical
    gather — the single adjoint pass does not depend on which targets are asked)."""
    g = _make_joint(n_samples, reward_limit, discrete)
    n = g.vertices_length()
    tvs = _t_vertices(g)

    full_adj = np.asarray(g.expected_sojourn_time(list(range(n))))
    sub = np.asarray(g.expected_sojourn_time(tvs.tolist()))
    np.testing.assert_array_equal(sub, full_adj[tvs])


def test_forward_escape_hatch_matches_adjoint(monkeypatch):
    """PHASIC_SOJOURN_FORWARD=1 selects the legacy O(n*k) forward path; it must
    return the same numbers as the default adjoint (to summation-order rounding)."""
    g = _make_joint(5, 2, discrete=False)
    tvs = _t_vertices(g).tolist()

    monkeypatch.delenv("PHASIC_SOJOURN_FORWARD", raising=False)
    adjoint = np.asarray(g.expected_sojourn_time(tvs))

    monkeypatch.setenv("PHASIC_SOJOURN_FORWARD", "1")
    forward = np.asarray(g.expected_sojourn_time(tvs))

    np.testing.assert_allclose(adjoint, forward, rtol=1e-9, atol=1e-12)


def test_joint_prob_table_over_large_target_set():
    """joint_prob_table() on a graph with thousands of t-vertices completes and
    is a proper (sub-)distribution — the old forward path would have allocated an
    n*k matrix here (n=7959, k=3125 -> ~0.2 GB) instead of the adjoint's O(n)."""
    g = _make_joint(6, 4, discrete=False)
    tvs = _t_vertices(g)
    assert tvs.size > 3000  # genuinely large target set

    tbl = g.joint_prob_table()
    probs = tbl['prob'].to_numpy()
    assert np.isfinite(probs).all()
    assert (probs >= 0).all()
    total = probs.sum()
    assert 0.0 < total <= 1.0 + 1e-9
