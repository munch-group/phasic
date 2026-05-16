"""Bit-identity gate for the Batch 2 refactor of ``_log_prob_unified``.

Per ``loglik-single-source-of-truth-plan.md`` Batch 2, this test
verifies that delegating the inlined sparse/dense + zero-inflation
block to ``_log_lik_from_pmf`` does not change the numerical value of
``_log_prob_unified`` at fixed thetas.

The reference value is computed by an in-test re-implementation that
mirrors the legacy (pre-Batch-2) inlined expression. ``==`` (bit
identity) is required because the helper performs the same ``jnp.sum``
in the same operand order.

Two configurations are exercised:

1. ZI sparse (partial-coverage rewards on the N=4 coalescent).
2. Non-ZI dense (standard univariate sample, no rewards).

Both at three fixed thetas, and with the standard-normal prior so the
prior term is purely a function of theta (no fitting side effects).
"""
from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from phasic import Graph, dense_to_sparse, with_ipv
from phasic.svgd import SVGD


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_n4_coalescent() -> Graph:
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


def _reference_log_prob_unified(svgd: SVGD, theta: jnp.ndarray,
                                  rewards=None) -> float:
    """Reproduces the legacy (pre-Batch-2) expression in
    ``_log_prob_unified``: continuous-density term, optional
    zero-inflation point mass, plus the prior. Applies
    svgd.param_transform exactly as the production code does."""
    # Apply parameter transformation (mirror _log_prob_unified).
    if svgd.param_transform is not None:
        theta_transformed = svgd.param_transform(theta)
    else:
        theta_transformed = theta

    # Evaluate the model at the same theta the production code does.
    pmf_vals, _ = svgd.model(theta_transformed, svgd.observed_data, rewards=rewards)

    has_zero_inflated = (
        getattr(svgd.model, "_zero_inflated_p_fn", None) is not None
    )

    if svgd._sparse_format:
        if has_zero_inflated:
            obs_values = jnp.asarray(svgd.observed_data.values)
            positive_mask = obs_values > 0.0
            log_lik = jnp.sum(
                jnp.where(positive_mask, jnp.log(pmf_vals + 1e-10), 0.0)
            )
        else:
            log_lik = jnp.sum(jnp.log(pmf_vals + 1e-10))
    else:
        obs_mask = ~jnp.isnan(svgd.observed_data)
        if has_zero_inflated:
            obs_positive_mask = obs_mask & (
                jnp.asarray(svgd.observed_data) > 0.0
            )
        else:
            obs_positive_mask = obs_mask
        log_lik = jnp.sum(
            jnp.where(obs_positive_mask, jnp.log(pmf_vals + 1e-10), 0.0)
        )

    if has_zero_inflated:
        p_theta = svgd.model._zero_inflated_p_fn(theta_transformed)
        n_zero = jnp.asarray(svgd.model._n_zero_per_feature)
        p_clamped = jnp.clip(p_theta, 1e-10, 1.0 - 1e-10)
        log_lik = log_lik + jnp.sum(n_zero * jnp.log1p(-p_clamped))

    # Mirror _log_prob_unified's prior selection.
    if getattr(svgd, "prior_list", None) is not None:
        log_pri = sum(
            svgd.prior_list[i](theta[i:i+1])
            for i in range(len(svgd.prior_list))
            if svgd.prior_list[i] is not None
        )
    elif getattr(svgd, "prior", None) is not None:
        log_pri = svgd.prior(theta)
    else:
        log_pri = -0.5 * jnp.sum(theta ** 2)
    return float(log_lik + log_pri)


def _fit_svgd(graph: Graph, observed_data, rewards=None):
    """Run a tiny SVGD just so the instance is fully constructed.

    We do not care about convergence — only that the SVGD object's
    state is populated so we can call ``_log_prob_unified`` directly.
    """
    from phasic import GaussPrior, ExpStepSize
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svgd = graph.svgd(
            observed_data=observed_data,
            rewards=rewards,
            prior=GaussPrior(ci=[1, 20]),
            n_iterations=2,
            n_particles=4,
            learning_rate=ExpStepSize(
                first_step=0.01, last_step=0.001, tau=30.0,
            ),
            progress=False,
        )
    return svgd


@pytest.fixture(scope="module")
def zi_svgd_and_pmf():
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
    svgd = _fit_svgd(g, sparse_obs, rewards=rewards)
    return svgd, sparse_obs, rewards


@pytest.fixture(scope="module")
def non_zi_svgd():
    g = _build_n4_coalescent()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = g.sample(150)
    obs = jnp.asarray(samples, dtype=jnp.float64)
    svgd = _fit_svgd(g, obs)
    return svgd, obs


# ---------------------------------------------------------------------------
# Bit-identity tests
# ---------------------------------------------------------------------------


THETAS = [
    jnp.array([1.5]),
    jnp.array([2.3]),
    jnp.array([0.7]),
]


@pytest.mark.parametrize("theta", THETAS)
def test_log_prob_unified_bit_identity_zi(zi_svgd_and_pmf, theta):
    """ZI sparse: _log_prob_unified must equal the legacy expression
    bit-identically."""
    svgd, sparse_obs, rewards = zi_svgd_and_pmf
    actual = float(svgd._log_prob_unified(theta, rewards=jnp.asarray(rewards)))
    expected = _reference_log_prob_unified(svgd, theta, rewards=jnp.asarray(rewards))

    assert actual == expected, (
        f"ZI: actual {actual!r} != expected {expected!r} (diff={actual-expected:g})"
    )


@pytest.mark.parametrize("theta", THETAS)
def test_log_prob_unified_bit_identity_non_zi(non_zi_svgd, theta):
    """Non-ZI dense: _log_prob_unified must equal the legacy
    expression bit-identically."""
    svgd, obs = non_zi_svgd
    actual = float(svgd._log_prob_unified(theta))
    expected = _reference_log_prob_unified(svgd, theta)

    assert actual == expected, (
        f"non-ZI: actual {actual!r} != expected {expected!r} "
        f"(diff={actual-expected:g})"
    )
