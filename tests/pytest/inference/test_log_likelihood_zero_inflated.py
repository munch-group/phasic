"""Batch 3 gate: log_likelihood() / AIC / BIC are correct for
partial-coverage zero-inflated models.

Per ``loglik-single-source-of-truth-plan.md`` Batch 3, the public
``SVGD.log_likelihood()`` method must include the zero-inflation
point-mass term ``Σ_j n_zero_j · log(1 − p_j(θ))`` when the model
carries ``_zero_inflated_p_fn``. Before this batch it omitted it,
silently producing wrong AIC/BIC for partial-coverage models.

This test asserts that ``log_likelihood(theta)`` equals the
hand-built reference for one fitted partial-coverage model at a
fixed theta in constrained space, and separately that the
no-argument form (LL at MAP) also matches the reference at the MAP.
"""
from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from phasic import (
    Graph, GaussPrior, ExpStepSize, dense_to_sparse, with_ipv,
)


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


def _reference_log_lik_zi(svgd, theta_constrained: jnp.ndarray,
                          rewards: jnp.ndarray) -> float:
    """Hand-built reference: Σ log(pmf_i for r_i > 0) + Σ_j n_zero_j · log(1 − p_j(θ))."""
    pmf_vals, _ = svgd.model(theta_constrained, svgd.observed_data, rewards=rewards)
    obs_values = jnp.asarray(svgd.observed_data.values)
    positive_mask = obs_values > 0.0
    log_lik_pos = jnp.sum(
        jnp.where(positive_mask, jnp.log(pmf_vals + 1e-10), 0.0)
    )
    p_theta = svgd.model._zero_inflated_p_fn(theta_constrained)
    n_zero = jnp.asarray(svgd.model._n_zero_per_feature)
    p_clamped = jnp.clip(p_theta, 1e-10, 1.0 - 1e-10)
    zero_term = jnp.sum(n_zero * jnp.log1p(-p_clamped))
    return float(log_lik_pos + zero_term)


@pytest.fixture(scope="module")
def fitted_zi_svgd():
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

    g_fit = _build_n4_coalescent()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svgd = g_fit.svgd(
            observed_data=sparse_obs,
            rewards=rewards,
            prior=GaussPrior(ci=[1, 20]),
            n_iterations=20,
            n_particles=8,
            learning_rate=ExpStepSize(
                first_step=0.01, last_step=0.001, tau=30.0,
            ),
            progress=False,
        )
    return svgd, rewards


def test_log_likelihood_at_supplied_theta_includes_zi_term(fitted_zi_svgd):
    """log_likelihood(theta) must include the ZI point-mass term.

    Before this batch the helper was called without theta_transformed,
    so the term was silently dropped and the returned LL was too
    high (closer to zero) by Σ_j n_zero_j · log(1 − p_j(θ)).
    """
    svgd, rewards = fitted_zi_svgd
    theta = jnp.array([10.0])

    ll_actual = svgd.log_likelihood(theta=theta)
    ll_ref = _reference_log_lik_zi(svgd, theta, jnp.asarray(rewards))

    # Tolerance is a few ULP — jnp.sum's reduction order in the helper
    # differs slightly from the per-step reference reduction above.
    # The point of this test is to catch the ZI term being dropped
    # entirely (a multi-unit error), not last-bit identity.
    assert abs(ll_actual - ll_ref) < 1e-9, (
        f"log_likelihood(theta) {ll_actual!r} != reference {ll_ref!r} "
        f"(diff={ll_actual - ll_ref:g}); ZI point-mass term may be missing."
    )

    # Bug-flavor cross-check: if the ZI term were dropped (the
    # pre-Batch-3 behavior), the result would be larger by
    # |Σ_j n_zero_j · log(1 − p_j(θ))|, which is non-zero for this
    # partial-coverage fixture. Compute the bug-flavor value and
    # require the actual to *not* match it.
    pmf_vals, _ = svgd.model(theta, svgd.observed_data, rewards=jnp.asarray(rewards))
    obs_values = jnp.asarray(svgd.observed_data.values)
    positive_mask = obs_values > 0.0
    ll_bug = float(jnp.sum(
        jnp.where(positive_mask, jnp.log(pmf_vals + 1e-10), 0.0)
    ))
    assert abs(ll_actual - ll_bug) > 1e-3, (
        f"log_likelihood(theta) {ll_actual!r} matches the pre-Batch-3 "
        f"bug-flavor value {ll_bug!r} — the ZI point-mass term is being "
        f"silently dropped."
    )


def test_log_likelihood_at_map_includes_zi_term(fitted_zi_svgd):
    """log_likelihood() (LL evaluated at the MAP) must also include
    the ZI point-mass term."""
    svgd, rewards = fitted_zi_svgd
    ll_actual = svgd.log_likelihood()

    # Reproduce the MAP locator to find the same theta the production
    # code uses, then build the reference at that theta.
    theta_unconstr_list, _ = svgd.map_estimate_from_particles(unconstrained=True)
    theta_unconstr = jnp.asarray(theta_unconstr_list, dtype=jnp.float64)
    if svgd.param_transform is not None:
        theta_map = svgd.param_transform(theta_unconstr)
    else:
        theta_map = theta_unconstr

    ll_ref = _reference_log_lik_zi(svgd, theta_map, jnp.asarray(rewards))
    assert abs(ll_actual - ll_ref) < 1e-9, (
        f"log_likelihood() {ll_actual!r} != reference {ll_ref!r} "
        f"(diff={ll_actual - ll_ref:g})"
    )


def test_aic_bic_use_zi_corrected_log_likelihood(fitted_zi_svgd):
    """AIC and BIC are simple algebraic transforms of log_likelihood(),
    so they automatically pick up the ZI correction. This test pins
    the relationship so a future regression of the helper plumbing
    fires here too."""
    from phasic.model_selection import aic, bic
    svgd, _ = fitted_zi_svgd

    ll = svgd.log_likelihood()
    k = svgd.degrees_of_freedom
    n = svgd.n_observations

    aic_result = aic(svgd)
    bic_result = bic(svgd)

    aic_expected = 2.0 * k - 2.0 * ll
    bic_expected = k * float(np.log(n)) - 2.0 * ll

    assert aic_result.aic == pytest.approx(aic_expected, rel=0, abs=1e-9)
    assert bic_result.bic == pytest.approx(bic_expected, rel=0, abs=1e-9)
    # And the log-likelihood field on each result mirrors svgd.log_likelihood().
    assert aic_result.log_likelihood == pytest.approx(ll, rel=0, abs=1e-12)
    assert bic_result.log_likelihood == pytest.approx(ll, rel=0, abs=1e-12)
