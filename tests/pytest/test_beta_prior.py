"""Tests for BetaPrior + per-dimension parameter transforms in SVGD.

BetaPrior constrains a parameter dimension to the open unit interval
(0, 1). Unlike GaussPrior / LogGaussPrior / HalfCauchyPrior — which
all assume softplus via the global ``positive_params=True`` path —
BetaPrior declares its own ``_natural_transform = sigmoid``. SVGD
detects this in a ``prior=[...]`` list and assembles a vector-
valued ``param_transform`` that applies the right transform to each
slice of θ.
"""
from __future__ import annotations

import phasic  # noqa: F401
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from phasic import BetaPrior, GaussPrior, Graph
from phasic.svgd import _sigmoid, _inverse_sigmoid


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_betaprior_alpha_beta():
    p = BetaPrior(alpha=2.0, beta=5.0)
    assert p.alpha == 2.0 and p.beta == 5.0


def test_betaprior_ci():
    p = BetaPrior(ci=(0.05, 0.5))
    # The solver returns (α, β) such that the 95% central mass falls
    # in (0.05, 0.5). Verify by checking the empirical quantiles match.
    from scipy.stats import beta as scipy_beta
    lo = scipy_beta.ppf(0.025, p.alpha, p.beta)
    hi = scipy_beta.ppf(0.975, p.alpha, p.beta)
    assert abs(lo - 0.05) < 1e-3
    assert abs(hi - 0.5) < 1e-3


def test_betaprior_mean_concentration():
    p = BetaPrior(mean=0.3, concentration=10.0)
    assert pytest.approx(p.alpha) == 3.0
    assert pytest.approx(p.beta) == 7.0


def test_betaprior_rejects_invalid_specs():
    with pytest.raises(ValueError, match="exactly one of"):
        BetaPrior()
    with pytest.raises(ValueError, match="exactly one of"):
        BetaPrior(alpha=1.0, beta=1.0, ci=(0.1, 0.5))
    with pytest.raises(ValueError, match="alpha and beta"):
        BetaPrior(alpha=1.0)
    with pytest.raises(ValueError, match="mean and concentration"):
        BetaPrior(mean=0.5)
    with pytest.raises(ValueError, match="mean must be in"):
        BetaPrior(mean=1.5, concentration=10.0)
    with pytest.raises(ValueError, match="ci must satisfy"):
        BetaPrior(ci=(0.5, 0.1))
    with pytest.raises(ValueError, match="ci must satisfy"):
        BetaPrior(ci=(-0.1, 0.5))


# ---------------------------------------------------------------------------
# Sampling and log-prob
# ---------------------------------------------------------------------------


def test_betaprior_samples_in_unit_interval():
    p = BetaPrior(alpha=2.0, beta=5.0)
    key = jax.random.PRNGKey(0)
    s = np.asarray(p.sample(key, (200,)))
    assert np.all((s > 0.0) & (s < 1.0))


def test_betaprior_log_prob_finite_inside_unit():
    p = BetaPrior(alpha=2.0, beta=5.0)
    # Without transform: phi treated as theta directly.
    lp = float(p(jnp.array([0.3])))
    assert np.isfinite(lp)


def test_betaprior_with_transform_roundtrips():
    """With ``_transform = sigmoid`` (the wiring SVGD does), samples
    come out in φ-space and converting back via sigmoid lands in
    (0, 1)."""
    p = BetaPrior(alpha=2.0, beta=5.0)
    p._transform = _sigmoid
    key = jax.random.PRNGKey(1)
    phi = np.asarray(p.sample(key, (200,)))
    theta_back = np.asarray(_sigmoid(jnp.asarray(phi)))
    assert np.all((theta_back > 0.0) & (theta_back < 1.0))


# ---------------------------------------------------------------------------
# Integration with SVGD
# ---------------------------------------------------------------------------


def _two_param_graph():
    """Build a simple 2-param Erlang-like graph with theta_dim=2."""
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    v0.add_edge(v1, [1.0, 0.0])  # weight = theta[0]
    v1.add_edge(v2, [0.0, 1.0])  # weight = theta[1]
    return g


def test_svgd_with_betaprior_in_mixed_list_keeps_param_in_unit():
    """Two-param fit with prior=[GaussPrior, BetaPrior]. After SVGD,
    theta_mean[0] must be positive (softplus) and theta_mean[1]
    must be in (0, 1) (sigmoid)."""
    g = _two_param_graph()
    g.update_weights(np.array([1.0, 0.5]))

    svgd = g.svgd(
        observed_data=jnp.array([1.5, 2.0, 2.5, 3.0]),
        prior=[GaussPrior(ci=[1, 5]), BetaPrior(ci=(0.05, 0.95))],
        theta_dim=2,
        n_iterations=5,
        n_particles=8,
        progress=False,
    )
    # `svgd.theta_mean` is the mean of φ-space particles directly
    # off the optimiser. To compare against the *constrained* (θ)
    # parameter space the model sees, use `get_results()` which
    # applies `param_transform` per particle and reports the mean
    # of the constrained sample.
    results = svgd.get_results()
    theta = np.asarray(results['theta_mean'])
    assert theta.shape == (2,)
    assert theta[0] > 0.0, (
        f"theta[0] under GaussPrior+softplus must be positive; "
        f"got {theta[0]}"
    )
    assert 0.0 < theta[1] < 1.0, (
        f"theta[1] under BetaPrior+sigmoid must lie in (0, 1); "
        f"got {theta[1]}"
    )


def test_svgd_param_transform_is_per_dim_when_betaprior_present():
    """The composite param_transform applied to a (2,)-φ produces
    one positive value and one value in (0, 1) at the expected
    indices."""
    g = _two_param_graph()
    g.update_weights(np.array([1.0, 0.5]))

    svgd = g.svgd(
        observed_data=jnp.array([1.5, 2.0]),
        prior=[GaussPrior(ci=[1, 5]), BetaPrior(ci=(0.05, 0.95))],
        theta_dim=2,
        n_iterations=1,
        n_particles=2,
        progress=False,
    )
    # Probe the assembled transform at a range of φ values.
    for phi_val in [-2.0, 0.0, 3.0, 10.0]:
        phi = jnp.array([phi_val, phi_val])
        theta = np.asarray(svgd.param_transform(phi))
        assert theta.shape == (2,)
        assert theta[0] > 0.0, (
            f"dim 0 should be positive under softplus; "
            f"got theta[0]={theta[0]} at phi={phi_val}"
        )
        assert 0.0 < theta[1] < 1.0, (
            f"dim 1 should be in (0, 1) under sigmoid; "
            f"got theta[1]={theta[1]} at phi={phi_val}"
        )


def test_svgd_callback_mode_with_betaprior_and_fixed_dim_runs_end_to_end():
    """End-to-end: a weight_callback-mode graph with `fixed=[(0, V)]`
    and `prior=[None, BetaPrior(...)]` runs through SVGD without
    raising and converges with the fixed parameter held exactly at
    V in θ-space.

    Pins three integrations that were originally broken:
      1. Smoke-validation applies `param_transform` so the callback
         sees θ-space rather than raw φ.
      2. The preconditioner's moment-matching sweep uses the right
         per-dim transforms when running on the learnable subset
         (uses `sigmoid` for dim 1, not the dim-0 softplus).
      3. JIT precompilation triggers compilation with a real prior
         particle (which has `fixed_values` baked in) rather than
         zeros (which would map to invalid θ values for dim 0).
    """
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v_abs = g.find_or_create_vertex([2])
    v0.add_edge(v1, [1.0, 0.0])
    v1.add_edge(v_abs, [0.0, 1.0])

    def cb(param, coef):
        coal_rate, s = param
        c0, c1 = coef
        return float(c0 * coal_rate + c1 * s)

    g.weight_callback = cb
    g.update_weights(np.array([0.5, 0.3]))

    svgd = g.svgd(
        observed_data=jnp.array([1.5, 2.0, 2.5, 3.0, 1.0]),
        callback=cb,
        theta_dim=2,
        fixed=[(0, 0.5)],
        prior=[None, BetaPrior(ci=(0.05, 0.95))],
        n_iterations=3,
        n_particles=4,
        progress=False,
    )
    theta = np.asarray(svgd.get_results()['theta_mean'])
    assert theta.shape == (2,)
    # Dim 0 was fixed at 0.5 — θ-space value must round-trip exactly.
    assert abs(theta[0] - 0.5) < 1e-6, (
        f"fixed dim 0 expected = 0.5; got {theta[0]}"
    )
    # Dim 1 must be in (0, 1) — BetaPrior constraint.
    assert 0.0 < theta[1] < 1.0, (
        f"BetaPrior dim 1 must be in (0, 1); got {theta[1]}"
    )


def test_svgd_callback_mode_partial_coverage_avoids_legacy_visit_prob():
    """When a callback-weight graph has partial-coverage 2D rewards
    (some features whose rewarded vertices aren't reached by every
    trajectory), `_attach_zero_inflated_term` would historically fall
    back to `compute_reward_visit_probability_ffi`, which sends the
    raw `weight_mode='callback'` JSON to the C++ parser — and the
    parser raises ``Unknown weight_mode: callback``. After the fix,
    the callback path exposes its own `_cdf_zero_fn` and the legacy
    visit-probability call is never invoked."""
    g = Graph(1)
    v0 = g.starting_vertex()
    v_a = g.find_or_create_vertex([1])
    v_b = g.find_or_create_vertex([2])
    v_abs = g.find_or_create_vertex([3])
    v0.add_edge(v_a, [1.0, 0.0])
    v0.add_edge(v_b, [1.0, 0.0])
    v_a.add_edge(v_abs, [0.0, 1.0])
    v_b.add_edge(v_abs, [0.0, 1.0])

    def cb(param, coef):
        rate1, rate2 = param
        c0, c1 = coef
        return float(c0 * rate1 + c1 * rate2)

    g.weight_callback = cb
    g.update_weights(np.array([1.0, 0.5]))

    # Partial-coverage rewards: feature 0 rewards only state [1]
    # (B-branch trajectories accumulate zero → atom at r=0). Feature 1
    # rewards both [1] and [2] (full coverage).
    rewards_2d = jnp.array([
        [0.0, 1.0, 0.0, 0.0],   # partial coverage
        [0.0, 1.0, 1.0, 0.0],   # full coverage
    ])
    from phasic import dense_to_sparse
    rng = np.random.default_rng(0)
    n = 4
    # 2D dense observations (n_obs, n_features) with NaN padding for
    # the partial-coverage feature, then convert to sparse format.
    dense = np.full((n, 2), np.nan, dtype=np.float64)
    dense[:, 0] = np.concatenate([np.zeros(2), rng.exponential(1.0, n - 2)])
    dense[:, 1] = rng.exponential(1.0, n)
    obs = dense_to_sparse(jnp.asarray(dense))

    # Patch the legacy entry point so a fallthrough triggers a clear
    # test failure rather than re-raising the C++ "Unknown weight_mode".
    import phasic.ffi_wrappers as fw
    sentinel = []
    original = fw.compute_reward_visit_probability_ffi

    def _trip(*a, **kw):
        sentinel.append(True)
        return original(*a, **kw)

    fw.compute_reward_visit_probability_ffi = _trip
    try:
        svgd = g.svgd(
            observed_data=obs,
            rewards=rewards_2d,
            callback=cb,
            theta_dim=2,
            prior=[GaussPrior(ci=[1, 5]), BetaPrior(ci=(0.05, 0.95))],
            n_iterations=2,
            n_particles=3,
            progress=False,
        )
    finally:
        fw.compute_reward_visit_probability_ffi = original

    assert not sentinel, (
        f"compute_reward_visit_probability_ffi was called "
        f"{len(sentinel)} time(s); the callback-mode `_cdf_zero_fn` "
        "should have routed the zero-inflated term through the "
        "fused pybind path instead."
    )
    assert 0 in svgd.zero_inflated_features


def test_svgd_without_betaprior_keeps_uniform_softplus():
    """Backwards compat: when no prior in the list declares a natural
    transform, the global softplus is used (single lambda, unchanged
    from pre-change behaviour)."""
    g = _two_param_graph()
    g.update_weights(np.array([1.0, 0.5]))
    svgd = g.svgd(
        observed_data=jnp.array([1.5, 2.0]),
        prior=[GaussPrior(ci=[1, 5]), GaussPrior(ci=[0.1, 1.0])],
        theta_dim=2,
        n_iterations=1,
        n_particles=2,
        progress=False,
    )
    # Both dimensions go through softplus → both positive at large φ.
    phi = jnp.array([3.0, 3.0])
    theta = np.asarray(svgd.param_transform(phi))
    # softplus(3.0) ≈ 3.05
    assert theta[0] > 2.5 and theta[1] > 2.5
