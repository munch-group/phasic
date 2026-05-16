"""Unit tests for ``SVGD._log_lik_from_pmf`` after the Batch 1 refactor.

Per ``loglik-single-source-of-truth-plan.md`` Batch 1, this verifies:

1. With a model carrying ``_zero_inflated_p_fn`` and
   ``_n_zero_per_feature`` (the contract Graph.svgd sets up for
   partial-coverage rewards), ``_log_lik_from_pmf(pmf_vals, theta)``
   returns the hand-computed
   ``Σ log(pmf_i for r_i > 0) + Σ_j n_zero_j · log(1 − p_j(θ))``.

2. With a model that has *no* ``_zero_inflated_p_fn`` attribute (the
   standard path), ``_log_lik_from_pmf(pmf_vals, theta=None)`` returns
   the legacy expression — bit-identical to the pre-refactor formula
   for both sparse and dense observations.

3. When the model is zero-inflated but ``theta_transformed`` is None,
   ``_log_lik_from_pmf`` raises ValueError (the new contract).
"""
from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phasic import (
    Graph, dense_to_sparse, SparseObservations, with_ipv,
)
from phasic.svgd import SVGD


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_n4_coalescent() -> Graph:
    """N=4 coalescent fixture (mirrors test_reward_validation.py)."""
    nr_samples = 4

    @with_ipv([nr_samples] + [0] * (nr_samples - 1))
    def coalescent_1param(state):
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
                transitions.append(
                    [new, [state[i] * (state[j] - same) / (1 + same)]]
                )
        return transitions

    g = Graph(coalescent_1param)
    g.update_weights([10.0])
    return g


def _make_svgd_for_helper_only(model, observed_data):
    """Construct an SVGD with the minimum state needed for
    ``_log_lik_from_pmf`` to run.

    We bypass ``SVGD.__init__`` (which fits / allocates particles /
    validates priors) because the only thing under test here is the
    helper method.
    """
    svgd = SVGD.__new__(SVGD)
    svgd.model = model
    svgd.observed_data = observed_data
    svgd._sparse_format = isinstance(observed_data, SparseObservations)
    return svgd


# ---------------------------------------------------------------------------
# Case 1: zero-inflated model — helper matches reference formula
# ---------------------------------------------------------------------------


def test_zero_inflated_helper_matches_reference():
    """With ``_zero_inflated_p_fn`` attached to the model,
    _log_lik_from_pmf must equal Σ log(pmf_i for r_i > 0) plus the
    explicit point-mass term Σ_j n_zero_j · log(1 − p_j(θ))."""
    g = _build_n4_coalescent()
    rewards = g.states().T[:-1].astype(np.float64)
    n = 200
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per_feat = [g.sample(n, rewards=rewards[i]) for i in range(rewards.shape[0])]

    obs = np.full(
        (n * rewards.shape[0], rewards.shape[0]), np.nan, dtype=np.float64,
    )
    for i in range(rewards.shape[0]):
        obs[i*n:(i+1)*n, i] = per_feat[i]
    sparse_obs = dense_to_sparse(obs)

    g_model = _build_n4_coalescent()
    model = Graph.pmf_and_moments_from_graph_multivariate(
        g_model, nr_moments=2, discrete=False,
    )
    g_model._attach_zero_inflated_term(
        model, rewards=rewards, offenders=[2], observed_data=sparse_obs,
    )

    theta = jnp.array([10.0])
    pmf_vals, _ = model(theta, sparse_obs, rewards=jnp.asarray(rewards))

    svgd = _make_svgd_for_helper_only(model, sparse_obs)
    log_lik_helper = float(svgd._log_lik_from_pmf(pmf_vals, theta))

    # Reference: same expression the helper claims to compute.
    obs_values = jnp.asarray(sparse_obs.values)
    positive_mask = obs_values > 0.0
    log_lik_pos = float(jnp.sum(
        jnp.where(positive_mask, jnp.log(pmf_vals + 1e-10), 0.0)
    ))
    p_theta = model._zero_inflated_p_fn(theta)
    n_zero = jnp.asarray(model._n_zero_per_feature)
    p_clamped = jnp.clip(p_theta, 1e-10, 1.0 - 1e-10)
    zero_term = float(jnp.sum(n_zero * jnp.log1p(-p_clamped)))
    log_lik_ref = log_lik_pos + zero_term

    assert log_lik_helper == log_lik_ref, (
        f"Helper {log_lik_helper:.12f} differs from reference "
        f"{log_lik_ref:.12f}"
    )


def test_zero_inflated_helper_requires_theta():
    """When the model carries _zero_inflated_p_fn but the caller passes
    theta_transformed=None, the helper must raise (rather than silently
    drop the point-mass term)."""
    g = _build_n4_coalescent()
    rewards = g.states().T[:-1].astype(np.float64)
    n = 50
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per_feat = [g.sample(n, rewards=rewards[i]) for i in range(rewards.shape[0])]
    obs = np.full(
        (n * rewards.shape[0], rewards.shape[0]), np.nan, dtype=np.float64,
    )
    for i in range(rewards.shape[0]):
        obs[i*n:(i+1)*n, i] = per_feat[i]
    sparse_obs = dense_to_sparse(obs)

    g_model = _build_n4_coalescent()
    model = Graph.pmf_and_moments_from_graph_multivariate(
        g_model, nr_moments=2, discrete=False,
    )
    g_model._attach_zero_inflated_term(
        model, rewards=rewards, offenders=[2], observed_data=sparse_obs,
    )
    theta = jnp.array([10.0])
    pmf_vals, _ = model(theta, sparse_obs, rewards=jnp.asarray(rewards))

    svgd = _make_svgd_for_helper_only(model, sparse_obs)
    with pytest.raises(ValueError, match="theta_transformed is required"):
        svgd._log_lik_from_pmf(pmf_vals, None)


# ---------------------------------------------------------------------------
# Case 2: legacy (non-ZI) — helper matches pre-refactor expression
# ---------------------------------------------------------------------------


def _legacy_sparse_log_lik(pmf_vals: jnp.ndarray) -> float:
    """Pre-refactor sparse-format expression."""
    return float(jnp.sum(jnp.log(pmf_vals + 1e-10)))


def _legacy_dense_log_lik(pmf_vals: jnp.ndarray,
                           observed_data: jnp.ndarray) -> float:
    """Pre-refactor dense-format expression."""
    obs_mask = ~jnp.isnan(observed_data)
    return float(jnp.sum(jnp.where(obs_mask, jnp.log(pmf_vals + 1e-10), 0.0)))


def test_non_zi_sparse_matches_legacy():
    """Standard (non-ZI) sparse path must equal the pre-refactor sum.

    Bypasses the model entirely: we feed hand-crafted pmf_vals to a
    stub-model SVGD instance to isolate the helper's sparse code path.
    """
    n = 100
    rng = np.random.default_rng(0)
    sparse_obs = SparseObservations(
        values=jnp.asarray(rng.uniform(0.1, 5.0, n), dtype=jnp.float64),
        features=jnp.zeros(n, dtype=jnp.int32),
        n_features=1,
        slices=((0, n),),
    )

    # Stub model object with no _zero_inflated_p_fn attribute.
    class _StubModel:
        pass

    pmf_vals = jnp.asarray(
        rng.uniform(1e-4, 1.0, n), dtype=jnp.float64,
    )

    svgd = _make_svgd_for_helper_only(_StubModel(), sparse_obs)
    log_lik = float(svgd._log_lik_from_pmf(pmf_vals))
    ref = _legacy_sparse_log_lik(pmf_vals)
    assert log_lik == ref, (
        f"Sparse non-ZI helper {log_lik:.12f} differs from legacy "
        f"{ref:.12f}"
    )


def test_non_zi_dense_matches_legacy():
    """Standard (non-ZI) dense path must equal the pre-refactor sum,
    including across NaN-padded missing observations."""
    g = _build_n4_coalescent()
    n = 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = g.sample(n)
    # Dense 1D obs with a few NaN-padded slots.
    obs = np.concatenate([samples, np.array([np.nan, np.nan, np.nan])])
    obs = jnp.asarray(obs, dtype=jnp.float64)

    model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)
    assert not hasattr(model, "_zero_inflated_p_fn")

    theta = jnp.array([10.0])
    pmf_vals, _ = model(theta, obs)

    svgd = _make_svgd_for_helper_only(model, obs)
    log_lik = float(svgd._log_lik_from_pmf(pmf_vals))
    ref = _legacy_dense_log_lik(pmf_vals, obs)
    assert log_lik == ref, (
        f"Dense non-ZI helper {log_lik:.12f} differs from legacy "
        f"{ref:.12f}"
    )
