"""likelihood_ratio_test_at — nested LRT when full/nested are DIFFERENT model
callables (the tied-vs-free epoch daisy-chain case that likelihood_ratio_test's
same-model identity guard rejects), plus the SVGD helpful-error ergonomics.

Provenance: deferred-svgd-lr-bug.md, fixes C + D.
"""
from functools import partial
from itertools import combinations_with_replacement

import numpy as np
import pytest

import phasic
from phasic import Graph, SVGD
import phasic.model_selection as ms

jnp = pytest.importorskip("jax.numpy")
from phasic import ExpStepSize  # noqa: E402


def _flat_prior(phi):
    return -0.5 * jnp.sum((phi / 10.0) ** 2)


def _two_param_graph():
    """S -> [3] -θ0-> [2] -θ1-> [1]."""
    g = Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge_parameterized(v2, 0.0, [1.0, 0.0])
    v2.add_edge_parameterized(v1, 0.0, [0.0, 1.0])
    return g


def _fit(model, data, *, fixed=None, seed=0, n_iterations=500, n_particles=30):
    svgd = SVGD(
        model=model, observed_data=data, prior=_flat_prior, theta_dim=2,
        n_particles=n_particles, n_iterations=n_iterations,
        learning_rate=ExpStepSize(first_step=0.01, last_step=0.001, tau=500.0),
        parallel='vmap', seed=seed, verbose=False, fixed=fixed,
    )
    svgd.optimize()
    return svgd


# ---------------------------------------------------------------- C: shared model
@pytest.fixture(scope="module")
def shared_model_pair():
    np.random.seed(0)
    data = jnp.asarray(np.random.gamma(shape=2.0, scale=1.0, size=120))
    model = Graph.pmf_and_moments_from_graph(_two_param_graph(), nr_moments=2, theta_dim=2)
    full = _fit(model, data)                       # both rates free (dof 2)
    nested = _fit(model, data, fixed=[(1, 1.0)])   # θ1 fixed (dof 1)
    return full, nested


def test_lrt_at_agrees_with_lrt_on_shared_model(shared_model_pair):
    # On an ordinary shared-model fixed-nested pair, the new function must agree
    # with the established likelihood_ratio_test (which requires full.model is
    # nested.model). This anchors likelihood_ratio_test_at against a known-good.
    full, nested = shared_model_pair
    ref = ms.likelihood_ratio_test(full, nested)
    got = ms.likelihood_ratio_test_at(full, nested)
    assert got.df == ref.df == 1
    assert got.log_likelihood_full == pytest.approx(ref.log_likelihood_full, rel=1e-9)
    assert got.log_likelihood_nested == pytest.approx(ref.log_likelihood_nested, rel=1e-9)
    assert got.statistic == pytest.approx(ref.statistic, rel=1e-9)
    assert got.p_value == pytest.approx(ref.p_value, rel=1e-9)


def test_lrt_at_df_guard(shared_model_pair):
    full, nested = shared_model_pair
    # swapping full<->nested makes k_full <= k_nested -> df guard fires
    with pytest.raises(ValueError, match="k_full > k_nested"):
        ms.likelihood_ratio_test_at(nested, full)


def test_lrt_at_rejects_inconsistent_pair():
    # Two fits on DIFFERENT data with the same theta_dim: nested is NOT a
    # constrained sub-case of full, so the runtime consistency check (which
    # REPLACES the same-model identity guard) must reject the pair.
    np.random.seed(1)
    data_a = jnp.asarray(np.random.gamma(2.0, 1.0, size=120))
    data_b = jnp.asarray(np.random.gamma(2.0, 3.0, size=120))  # different scale
    model_a = Graph.pmf_and_moments_from_graph(_two_param_graph(), nr_moments=2, theta_dim=2)
    model_b = Graph.pmf_and_moments_from_graph(_two_param_graph(), nr_moments=2, theta_dim=2)
    full = _fit(model_a, data_a)
    nested = _fit(model_b, data_b, fixed=[(1, 1.0)])
    with pytest.raises(ValueError, match="constrained sub-case"):
        ms.likelihood_ratio_test_at(full, nested)


# ---------------------------------------------------------------- D: helpful errors
def test_svgd_rejects_epoch_starts_with_helpful_error():
    model = Graph.pmf_and_moments_from_graph(_two_param_graph(), nr_moments=2, theta_dim=2)
    data = jnp.asarray([1.0, 2.0, 1.5])
    with pytest.raises(TypeError) as ei:
        SVGD(model=model, observed_data=data, theta_dim=2, epoch_starts=[0, 0.5])
    msg = str(ei.value)
    assert "epoch_starts" in msg and "Graph.svgd" in msg


def test_svgd_rejects_unknown_kwarg():
    model = Graph.pmf_and_moments_from_graph(_two_param_graph(), nr_moments=2, theta_dim=2)
    data = jnp.asarray([1.0, 2.0, 1.5])
    with pytest.raises(TypeError, match="unexpected keyword"):
        SVGD(model=model, observed_data=data, theta_dim=2, definitely_not_a_kwarg=1)


# ---------------------------------------------------------------- C: tied-vs-free epoch
@pytest.fixture(scope="module")
def epoch_tied_free_pair():
    """A tied-vs-free epoch (daisy-chain) joint-prob pair — the motivating case:
    full.model (free) and nested.model (tied) are DIFFERENT callables."""
    from phasic import LogGaussPrior, Adamelia, with_ipv, StateIndexer, Property, set_log_level
    set_log_level("WARNING")
    np.random.seed(17)
    all_pairs = partial(combinations_with_replacement, r=2)
    nr_samples = 4
    indexer = StateIndexer(lineages=[Property("descendants", min_value=1, max_value=nr_samples)])

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

    graph = Graph(coal, indexer=indexer)
    rl, mu, th = 5, 1e-4, 1 / 10_000
    jpg_d = graph.joint_prob_graph(indexer, reward_limit=rl, mutation_rate=mu)
    jpg_d.update_weights([th, mu])
    jpt = jpg_d.joint_prob_table(); p = jpt['prob'] / jpt['prob'].sum()
    sample = np.random.choice(jpt.index.values, 1000, p=p.to_numpy())
    obs = jpt.loc[sample, jpt.columns[:-1]].to_numpy().tolist()
    jpg = graph.joint_prob_graph(indexer, reward_limit=rl, mutation_rate=mu, discrete=False)
    common = dict(fixed=[(1, mu)], prior=LogGaussPrior(ci=[1 / 50_000, 1 / 5000]),
                  n_iterations=40, n_particles=16, optimizer=Adamelia(learning_rate=0.2),
                  epoch_starts=[0, 0.5])
    full = jpg.svgd(obs, **common)                    # coal free per epoch
    nested = jpg.svgd(obs, tied=[(0, [0, 1])], **common)  # coal tied across epochs
    return full, nested


def test_lrt_at_tied_vs_free_epoch(epoch_tied_free_pair):
    full, nested = epoch_tied_free_pair
    # likelihood_ratio_test would reject this pair (different callables); the new
    # function must accept it and give a coherent result.
    assert full.model is not nested.model
    with pytest.raises(ValueError):
        ms.likelihood_ratio_test(full, nested)  # same-model identity guard fires
    res = ms.likelihood_ratio_test_at(full, nested, strict=False)
    assert res.df == 1                       # 2 free coal rates vs 1 tied
    assert res.statistic >= 0.0
    assert 0.0 <= res.p_value <= 1.0
    # nested is a genuine constrained sub-case -> no meaningful LL inversion
    assert res.log_likelihood_nested <= res.log_likelihood_full + 1e-4
