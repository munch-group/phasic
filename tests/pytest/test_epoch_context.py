"""Tests for ``Graph.epoch_context`` / ``EpochContext``.

``EpochContext`` reads cumulative t-state absorption probabilities of a
continuous joint-prob graph at arbitrary times ``t``. As ``t → ∞`` the
read converges to the discrete asymptote ``Graph.joint_prob_table().prob``.

A previous notebook-side hand-rolled version of the same readout had a
typo (``st[i] += st_with_sisters[i]`` instead of indexing the aux
partner), which produced an overshoot at high mutation rates and small
``t``. These tests pin the contract that the new API is monotone and
matches the asymptote.
"""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import pytest

from phasic import EpochContext, Graph, Property, StateIndexer


def _coal_callback(state, indexer=None):
    transitions = []
    for i, j in combinations_with_replacement(range(indexer.lineages.state_length), 2):
        same = int(i == j)
        if same and state[i] < 2:
            continue
        if not same and (state[i] < 1 or state[j] < 1):
            continue
        new = state.copy()
        new[i] -= 1
        new[j] -= 1
        new[min(i + j + 1, state.size - 1)] += 1
        pair_count = state[i] * (state[j] - same) / (1 + same)
        transitions.append([new, [pair_count]])
    return transitions


def _build_graphs(mutation_rate: float, n_samples: int = 4, reward_limit: int = 1):
    """Return (source_graph, continuous_joint_prob_graph, discrete_joint_prob_graph)."""
    indexer = StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=n_samples)]
    )
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = n_samples
    graph = Graph(_coal_callback, ipv=ipv, indexer=indexer)
    cjpg = graph.joint_prob_graph(
        mutation_rate=mutation_rate, reward_limit=reward_limit, discrete=False
    )
    djg = graph.joint_prob_graph(
        mutation_rate=mutation_rate, reward_limit=reward_limit, discrete=True
    )
    return graph, cjpg, djg


def test_epoch_context_matches_joint_prob_table_at_infty():
    """At large t, cumulative_probs equals joint_prob_table().prob elementwise."""
    mutation_rate = 2.0
    theta = [2.0, mutation_rate]
    _, cjpg, djg = _build_graphs(mutation_rate)

    ctx = cjpg.epoch_context()
    ctx.update_weights(theta)
    djg.update_weights(theta)

    cont = ctx.cumulative_probs(50.0, table=False)
    asymp = djg.joint_prob_table().prob.reindex(ctx._t_vertex_indices).values

    assert cont.shape == asymp.shape
    np.testing.assert_allclose(cont, asymp, atol=1e-10, rtol=1e-10)


def test_epoch_context_array_t_shape_and_consistency():
    """Array-of-times call returns 2D output; each row equals the scalar call."""
    mutation_rate = 2.0
    theta = [2.0, mutation_rate]
    _, cjpg, _ = _build_graphs(mutation_rate)

    ctx = cjpg.epoch_context()
    ctx.update_weights(theta)

    ts = np.array([0.3, 1.0, 5.0])
    arr = ctx.cumulative_probs(ts, table=False)
    assert arr.shape == (3, len(ctx._t_vertex_indices))
    for k, tk in enumerate(ts):
        np.testing.assert_allclose(
            arr[k], ctx.cumulative_probs(float(tk), table=False)
        )


def test_epoch_context_monotone_no_overshoot_at_high_mutation():
    """Cumulative probs are non-decreasing in t — even at high mutation rate
    (where the deprecated notebook-side helper used to overshoot the
    asymptote)."""
    mutation_rate = 5.0
    theta = [2.0, mutation_rate]
    _, cjpg, djg = _build_graphs(mutation_rate)

    ctx = cjpg.epoch_context()
    ctx.update_weights(theta)
    djg.update_weights(theta)

    ts = np.linspace(0.001, 10.0, 200)
    arr = ctx.cumulative_probs(ts, table=False)

    # Monotone increase, allowing only floating-point noise.
    assert np.min(np.diff(arr, axis=0)) >= -1e-12

    # Never exceeds the asymptote per state.
    asymp = djg.joint_prob_table().prob.reindex(ctx._t_vertex_indices).values
    assert np.all(arr <= asymp + 1e-10)


def test_epoch_context_table_layout_matches_joint_prob_table():
    """cumulative_probs_table has the same columns/index as joint_prob_table."""
    mutation_rate = 2.0
    theta = [2.0, mutation_rate]
    _, cjpg, djg = _build_graphs(mutation_rate)

    ctx = cjpg.epoch_context()
    ctx.update_weights(theta)
    djg.update_weights(theta)

    table = ctx.cumulative_probs_table(50.0)
    asymp_table = djg.joint_prob_table()

    assert list(table.columns) == list(asymp_table.columns)
    assert list(table.index) == list(asymp_table.index)
    np.testing.assert_allclose(
        table['prob'].values, asymp_table['prob'].values, atol=1e-10, rtol=1e-10
    )


def test_epoch_context_rejects_discrete_joint_prob_graph():
    _, _, djg = _build_graphs(2.0)
    with pytest.raises(ValueError, match='continuous joint-prob graph'):
        djg.epoch_context()


def test_epoch_context_rejects_non_joint_prob_graph():
    """A vanilla parameterised graph (not built via joint_prob_graph) is rejected."""
    indexer = StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=4)]
    )
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = 4
    graph = Graph(_coal_callback, ipv=ipv, indexer=indexer)
    with pytest.raises(ValueError, match='joint_prob_graph'):
        graph.epoch_context()


def test_epoch_context_returns_correct_type():
    _, cjpg, _ = _build_graphs(2.0)
    ctx = cjpg.epoch_context()
    assert isinstance(ctx, EpochContext)
    # The underlying JSP graph is reachable for power users.
    assert getattr(ctx._graph, '_joint_stop_prob_graph', False) is True
    assert ctx._source_graph is cjpg


def test_epoch_context_inherits_theta_from_source_graph():
    """If update_weights has been called on the source graph,
    epoch_context() inherits theta — cumulative_probs is callable
    without an explicit context-side update_weights."""
    mutation_rate = 2.0
    theta = [2.0, mutation_rate]
    _, cjpg, djg = _build_graphs(mutation_rate)

    cjpg.update_weights(theta)               # set on source
    ctx = cjpg.epoch_context()               # should inherit theta
    inherited = ctx.cumulative_probs(50.0, table=False)   # works without ctx.update_weights

    # Confirm equivalence with the explicit-update path.
    cjpg2 = cjpg                              # same graph, theta unchanged
    ctx2 = cjpg2.epoch_context()
    ctx2.update_weights(theta)
    explicit = ctx2.cumulative_probs(50.0, table=False)
    np.testing.assert_allclose(inherited, explicit, atol=1e-12, rtol=1e-12)

    # And matches the discrete asymptote (sanity).
    djg.update_weights(theta)
    asymp = djg.joint_prob_table().prob.reindex(ctx._t_vertex_indices).values
    np.testing.assert_allclose(inherited, asymp, atol=1e-10, rtol=1e-10)


def test_next_ipv_shape_and_layout():
    """next_ipv returns a 1D vector of length len(_ipv_target_indices)."""
    _, cjpg, _ = _build_graphs(2.0)
    cjpg.update_weights([2.0, 2.0])
    ctx = cjpg.epoch_context()

    v = ctx.next_ipv(1.0)
    assert v.ndim == 1
    assert v.shape == (len(ctx._ipv_target_indices),)
    # All probabilities, so bounded.
    assert np.all(v >= 0.0) and np.all(v <= 1.0)


def test_next_ipv_matches_jsp_stop_probability_collapsed():
    """next_ipv(t) equals the t/aux-collapsed stop_probability(t) at
    _ipv_target_indices — the same collapse done inside
    _probe_daisy_t_eval and the C++ daisy-chain FFI handler."""
    _, cjpg, _ = _build_graphs(2.0)
    cjpg.update_weights([2.0, 2.0])
    ctx = cjpg.epoch_context()

    t = 0.7
    raw = np.asarray(ctx._graph.stop_probability(t))
    expected = np.array([
        raw[v] + (raw[ctx._t_aux_map[v]] if v in ctx._t_aux_map else 0.0)
        for v in ctx._ipv_target_indices
    ])
    np.testing.assert_allclose(ctx.next_ipv(t), expected, atol=1e-14, rtol=0)


def test_next_ipv_two_epoch_chain_matches_single_long_run():
    """Daisy-chaining two epochs by hand (epoch_a for time dt, then
    epoch_b for time t) using next_ipv as the bridge produces the same
    final cumulative_probs as a single epoch with thetas tied to be
    equal and t = dt + t."""
    _, cjpg, _ = _build_graphs(2.0)
    theta = [2.0, 2.0]

    # --- Single epoch baseline: theta active for time dt + t = 3.0 ---
    cjpg.update_weights(theta)
    ctx_single = cjpg.epoch_context()
    baseline = ctx_single.cumulative_probs(3.0, table=False)

    # --- Two-epoch hand chain: same theta, dt=1.0 then t=2.0 ---
    cjpg.update_weights(theta)
    ctx_a = cjpg.epoch_context()
    ipv_b = ctx_a.next_ipv(1.0)

    # Build a fresh epoch_b context — auto-IPV is overwritten with ipv_b.
    cjpg.update_weights(theta)
    ctx_b = cjpg.epoch_context()
    ctx_b.update_ipv(ipv_b)
    chained = ctx_b.cumulative_probs(2.0, table=False)

    np.testing.assert_allclose(baseline, chained, atol=1e-9, rtol=1e-9)


def test_next_ipv_rejects_array_t():
    _, cjpg, _ = _build_graphs(2.0)
    cjpg.update_weights([2.0, 2.0])
    ctx = cjpg.epoch_context()
    with pytest.raises(ValueError, match='scalar t or None'):
        ctx.next_ipv(np.array([0.5, 1.0]))


def test_cumulative_probs_default_t_matches_asymptote():
    """cumulative_probs() with no t picks t high enough that the read
    is within tol of the discrete asymptote — the same precision policy
    used by the C++ daisy-chain handler."""
    mutation_rate = 2.0
    theta = [2.0, mutation_rate]
    _, cjpg, djg = _build_graphs(mutation_rate)

    cjpg.update_weights(theta)
    ctx = cjpg.epoch_context()
    djg.update_weights(theta)
    asymp = djg.joint_prob_table().prob.reindex(ctx._t_vertex_indices).values

    auto = ctx.cumulative_probs(tol=1e-6, table=False)  # tighter than default
    np.testing.assert_allclose(auto, asymp, atol=1e-6, rtol=0)


def test_auto_t_is_monotone_in_tol():
    """Tighter tolerance → equal-or-larger chosen t."""
    _, cjpg, _ = _build_graphs(2.0)
    cjpg.update_weights([2.0, 2.0])
    ctx = cjpg.epoch_context()

    t_loose = ctx.auto_t(tol=1e-2)
    t_tight = ctx.auto_t(tol=1e-8)
    assert t_tight >= t_loose


def test_next_ipv_default_t_settles_residual_mass():
    """next_ipv() with no t auto-picks a t at which the returned IPV's
    non-t-state mass is below tol — i.e. the chain has settled."""
    _, cjpg, _ = _build_graphs(2.0)
    cjpg.update_weights([2.0, 2.0])
    ctx = cjpg.epoch_context()

    ipv = ctx.next_ipv(tol=1e-4)
    # Indices in the IPV layout that are NOT t-vertices.
    t_set = set(ctx._t_vertex_indices)
    non_t_positions = [
        pos for pos, v in enumerate(ctx._ipv_target_indices) if v not in t_set
    ]
    residual = float(np.sum(ipv[non_t_positions]))
    assert residual < 1e-4


def test_cumulative_probs_table_kwarg_matches_table_method():
    """cumulative_probs(t, table=True) equals cumulative_probs_table(t)."""
    _, cjpg, _ = _build_graphs(2.0)
    cjpg.update_weights([2.0, 2.0])
    ctx = cjpg.epoch_context()

    df_kwarg = ctx.cumulative_probs(0.7, table=True)
    df_alias = ctx.cumulative_probs_table(0.7)
    pd.testing.assert_frame_equal(df_kwarg, df_alias)


def test_cumulative_probs_table_default_t_matches_joint_prob_table():
    """cumulative_probs(table=True) with auto-t matches joint_prob_table()."""
    _, cjpg, djg = _build_graphs(2.0)
    cjpg.update_weights([2.0, 2.0])
    djg.update_weights([2.0, 2.0])

    ctx = cjpg.epoch_context()
    df = ctx.cumulative_probs(tol=1e-8, table=True)
    asymp = djg.joint_prob_table()

    assert list(df.columns) == list(asymp.columns)
    assert list(df.index) == list(asymp.index)
    np.testing.assert_allclose(df['prob'].values, asymp['prob'].values, atol=1e-8)


def test_cumulative_probs_table_rejects_array_t():
    _, cjpg, _ = _build_graphs(2.0)
    cjpg.update_weights([2.0, 2.0])
    ctx = cjpg.epoch_context()
    with pytest.raises(ValueError, match='table=True requires a scalar t'):
        ctx.cumulative_probs(np.array([0.5, 1.0]), table=True)


def test_epoch_context_can_override_inherited_theta():
    """ctx.update_weights overrides whatever was inherited from the source.

    Build the source graph once, inherit a first ``theta`` via
    ``epoch_context()``, then override it on the context and confirm the
    read changes to match the override (not the inherited value).
    """
    _, cjpg, _ = _build_graphs(2.0)

    theta_a = [2.0, 2.0]
    theta_b = [4.0, 2.0]   # different theta_0 → different overall rate scaling

    cjpg.update_weights(theta_a)
    ctx = cjpg.epoch_context()                  # inherits theta_a
    inherited_read = ctx.cumulative_probs(50.0, table=False).copy()

    ctx.update_weights(theta_b)                 # override on context
    overridden_read = ctx.cumulative_probs(50.0, table=False)

    # The reads must actually differ — i.e. the override took effect.
    assert not np.allclose(inherited_read, overridden_read, atol=1e-8)

    # And ctx.update_weights(theta_a) reverts to the original.
    ctx.update_weights(theta_a)
    reverted_read = ctx.cumulative_probs(50.0, table=False)
    np.testing.assert_allclose(reverted_read, inherited_read, atol=1e-12, rtol=1e-12)
