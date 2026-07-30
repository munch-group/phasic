"""Graph.epoch_model — the public reusable *free* daisy-chain epoch model (Batch 2).

Gates:
  - epoch_model().fit() reproduces Graph.svgd(epoch_starts=..., tied=None) to
    machine precision (the free model built out-of-band == the internal path);
  - one FreeEpochModel drives multiple fits sharing .model -> the same-model
    LRT fast path (free-vs-fixed nested LRT, df=1);
  - log_likelihood(theta=) works on a free fit;
  - .fit(fixed=extra) re-masks the prior and succeeds (flat index space);
  - exposure baked into the model matches Graph.svgd(exposure=...);
  - the observation mapping is deterministic under a seed.
"""
from functools import partial
from itertools import combinations_with_replacement

import numpy as np
import pytest

import phasic
from phasic import Graph, FreeEpochModel
import phasic.model_selection as ms

jnp = pytest.importorskip("jax.numpy")
from phasic import LogGaussPrior, Adamelia, with_ipv, StateIndexer, Property, set_log_level  # noqa: E402


MU = 1e-4
COMMON = dict(n_iterations=40, n_particles=12, seed=0)


def _optimizer():
    return Adamelia(learning_rate=0.2)


@pytest.fixture(scope="module")
def coal_setup():
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
    rl, th = 5, 1 / 10_000
    jpg_d = graph.joint_prob_graph(indexer, reward_limit=rl, mutation_rate=MU)
    jpg_d.update_weights([th, MU])
    jpt = jpg_d.joint_prob_table(); p = jpt['prob'] / jpt['prob'].sum()
    sample = np.random.choice(jpt.index.values, 1000, p=p.to_numpy())
    obs = jpt.loc[sample, jpt.columns[:-1]].to_numpy().tolist()
    jpg = graph.joint_prob_graph(indexer, reward_limit=rl, mutation_rate=MU, discrete=False)
    return dict(jpg=jpg, obs=obs, epoch_starts=[0, 0.5],
                prior=LogGaussPrior(ci=[1 / 50_000, 1 / 5000]))


@pytest.fixture(scope="module")
def free_model(coal_setup):
    return coal_setup['jpg'].epoch_model(
        coal_setup['obs'], epoch_starts=coal_setup['epoch_starts'],
        fixed=[(1, MU)], prior=coal_setup['prior'],
    )


@pytest.fixture(scope="module")
def free_fit(free_model):
    return free_model.fit(optimizer=_optimizer(), **COMMON)


# ------------------------------------------------------------------ reproduce
def test_epoch_model_reproduces_graph_svgd(coal_setup, free_fit):
    """epoch_model().fit() == Graph.svgd(epoch_starts=..., tied=None) exactly."""
    ref = coal_setup['jpg'].svgd(
        coal_setup['obs'], epoch_starts=coal_setup['epoch_starts'],
        fixed=[(1, MU)], prior=coal_setup['prior'],
        optimizer=_optimizer(), **COMMON,
    )
    assert free_fit.theta_dim == ref.theta_dim == 4
    assert free_fit.degrees_of_freedom == ref.degrees_of_freedom == 2
    map_ref = np.asarray(ref.map_estimate_from_particles()[0])
    map_free = np.asarray(free_fit.map_estimate_from_particles()[0])
    np.testing.assert_allclose(map_free, map_ref, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        float(free_fit.log_likelihood()), float(ref.log_likelihood()),
        rtol=1e-9, atol=1e-12,
    )


# ----------------------------------------------------------- same-model LRT
def test_same_model_lrt_fast_path(free_model, free_fit):
    """Two fits from the SAME FreeEpochModel share .model -> the fast path."""
    full = free_fit
    # nested: additionally fix one FLAT slot (epoch-0 coalescent rate) at full's
    # own MAP value.
    v = float(np.asarray(full.map_estimate_from_particles()[0])[0])
    nested = free_model.fit(fixed=[(0, v)], optimizer=_optimizer(), **COMMON)
    assert full.model is nested.model                      # same-model fast path
    # strict=False: two independent 40-iter fits don't guarantee the free fit
    # dominates the easier nested fit's point, so the LL-inversion guard may
    # legitimately fire — a convergence artifact orthogonal to the fast-path
    # identity + df this test pins (df comes from fixed-masks, not LL values).
    res = ms.likelihood_ratio_test(full, nested, strict=False)
    assert res.df == 1
    assert res.k_full == 2 and res.k_nested == 1
    assert res.statistic >= 0.0
    assert 0.0 <= res.p_value <= 1.0


# ------------------------------------------------------------ log_likelihood
def test_log_likelihood_at_theta(free_fit):
    theta = np.asarray(free_fit.map_estimate_from_particles()[0])
    ll_at = float(free_fit.log_likelihood(theta=theta))
    ll_map = float(free_fit.log_likelihood())
    np.testing.assert_allclose(ll_at, ll_map, rtol=1e-6, atol=1e-9)


# --------------------------------------------------- .fit extra fixed / prior
def test_fit_extra_fixed_remasks_prior(free_model):
    """.fit(fixed=extra) must merge with base fixed AND re-mask the prior
    (a raw shared prior at a fixed slot would raise); returns a valid fit."""
    fit = free_model.fit(fixed=[(0, 1e-4)], optimizer=_optimizer(), **COMMON)
    assert fit.degrees_of_freedom == 1            # base {1,3} + extra {0} fixed
    assert isinstance(fit.log_likelihood(), float)


def test_fit_rejects_exposure_kwarg(free_model):
    with pytest.raises(TypeError, match="exposure is baked"):
        free_model.fit(exposure=1.0, optimizer=_optimizer(), **COMMON)


# -------------------------------------------------------------- determinism
def test_mapping_deterministic_under_seed(coal_setup):
    # NOTE: this coal fixture has NO degenerate outcome groups (every outcome
    # maps to a unique index), so the mapping is deterministic regardless of
    # seed and the `idx.size > 1` / rng.choice tie-break branch is not exercised
    # here. This pins the wiring (helper == model's baked observed_data) and
    # seed-repro; a degenerate-group test for the sampling branch is a follow-up.
    jpg = coal_setup['jpg']
    a = jpg._map_joint_observations_to_indices(coal_setup['obs'], seed=123)
    b = jpg._map_joint_observations_to_indices(coal_setup['obs'], seed=123)
    assert a == b
    # and it agrees with the model's baked observed_data
    m = jpg.epoch_model(coal_setup['obs'], epoch_starts=coal_setup['epoch_starts'],
                        fixed=[(1, MU)], prior=coal_setup['prior'], seed=123)
    assert list(m.observed_data) == a


# ------------------------------------------------------------------ exposure
def test_exposure_parity(coal_setup):
    """epoch_model(exposure=...) fit matches Graph.svgd(exposure=...)."""
    jpg, obs, es = coal_setup['jpg'], coal_setup['obs'], coal_setup['epoch_starts']
    n = len(obs)
    # Only TWO distinct exposure values so the FFI dedup keeps this fast (1000
    # unique alphas would run 1000 chains and time out); still exercises the
    # per-obs exposure baking on both paths.
    alpha = np.where(np.arange(n) % 2 == 0, 0.9, 1.1).astype(float)
    ref = jpg.svgd(obs, epoch_starts=es, fixed=[(1, MU)], prior=coal_setup['prior'],
                   exposure=alpha, exposure_param_index=0,
                   optimizer=_optimizer(), **COMMON)
    m = jpg.epoch_model(obs, epoch_starts=es, fixed=[(1, MU)], prior=coal_setup['prior'],
                        exposure=alpha, exposure_param_index=0)
    fit = m.fit(optimizer=_optimizer(), **COMMON)
    np.testing.assert_allclose(
        np.asarray(fit.map_estimate_from_particles()[0]),
        np.asarray(ref.map_estimate_from_particles()[0]),
        rtol=1e-9, atol=1e-12,
    )
