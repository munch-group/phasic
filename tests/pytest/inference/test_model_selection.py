"""Tests for phasic.model_selection: AIC, BIC, likelihood ratio tests.

Linchpin test: `test_log_likelihood_exponential_mle` pins the AIC/BIC/LRT
arithmetic down to the analytical MLE log-likelihood of an exponential
distribution. If that passes, every linear combination of (log_likelihood,
degrees_of_freedom, n_observations) is correct by construction.
"""

import math
import os
import warnings

# Enable JAX x64 precision before importing jax/phasic. SVGD requires float64
# end-to-end; phasic auto-enables this when _ensure_jax_active() runs, but
# we need it set before any jax.numpy import in test collection.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from phasic import Graph, SVGD, ExpStepSize
import phasic.model_selection as ms


# -----------------------------------------------------------------------------
# Test helpers
# -----------------------------------------------------------------------------


def _build_exp_graph():
    """Single-parameter exponential: rate = θ."""
    g = Graph(1)
    start = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v2, 1.0)
    v2.add_edge_parameterized(v1, 0.0, [1.0])
    return g


def _build_two_param_graph():
    """Two-parameter chain: S → [3] —θ₀→ [2] —θ₁→ [1]. Erlang-like when θ₀=θ₁."""
    g = Graph(1)
    start = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v3, 1.0)
    v3.add_edge_parameterized(v2, 0.0, [1.0, 0.0])  # rate = θ₀
    v2.add_edge_parameterized(v1, 0.0, [0.0, 1.0])  # rate = θ₁
    return g


def _flat_prior(phi):
    """Very weak prior so MAP ≈ MLE in PHI (unconstrained) space."""
    return -0.5 * jnp.sum((phi / 10.0) ** 2)


def _fit_exp(data, n_iterations=1500, n_particles=60, seed=0,
             fixed=None, model=None):
    """Fit single-parameter exponential SVGD on the given data."""
    if model is None:
        g = _build_exp_graph()
        model = Graph.pmf_and_moments_from_graph(
            g, nr_moments=2, discrete=False, theta_dim=1
        )
    schedule = ExpStepSize(first_step=0.01, last_step=0.001, tau=500.0)
    svgd = SVGD(
        model=model,
        observed_data=data,
        prior=_flat_prior,
        theta_dim=1,
        n_particles=n_particles,
        n_iterations=n_iterations,
        learning_rate=schedule,
        parallel='vmap',
        seed=seed,
        verbose=False,
        fixed=fixed,
    )
    svgd.optimize()
    return svgd, model


def _fit_two_param(data, n_iterations=1500, n_particles=60, seed=0,
                   fixed=None, model=None):
    """Fit two-parameter chain on the given data."""
    if model is None:
        g = _build_two_param_graph()
        model = Graph.pmf_and_moments_from_graph(
            g, nr_moments=2, discrete=False, theta_dim=2
        )
    schedule = ExpStepSize(first_step=0.01, last_step=0.001, tau=500.0)
    svgd = SVGD(
        model=model,
        observed_data=data,
        prior=_flat_prior,
        theta_dim=2,
        n_particles=n_particles,
        n_iterations=n_iterations,
        learning_rate=schedule,
        parallel='vmap',
        seed=seed,
        verbose=False,
        fixed=fixed,
    )
    svgd.optimize()
    return svgd, model


def _exp_analytical_ll(data, lam=None):
    """Analytical Exp(λ) log-likelihood at λ̂ = n/Σtᵢ (MLE) if lam=None."""
    arr = np.asarray(data, dtype=float)
    if lam is None:
        lam = len(arr) / np.sum(arr)
    return float(len(arr) * np.log(lam) - lam * np.sum(arr))


# -----------------------------------------------------------------------------
# SVGD member tests: log_likelihood, degrees_of_freedom, n_observations
# -----------------------------------------------------------------------------


def test_log_likelihood_exponential_mle():
    """Linchpin test: refined-MAP LL matches analytical Exp(λ̂) MLE LL."""
    np.random.seed(123)
    true_lam = 2.0
    data = np.random.exponential(scale=1.0 / true_lam, size=200)

    svgd, _ = _fit_exp(data, n_iterations=2000, n_particles=80, seed=7)

    ll_refined = svgd.log_likelihood(refine=True)
    ll_analytical = _exp_analytical_ll(data)  # at λ̂ = n/Σtᵢ
    # The library evaluates Exp(λ) PDF via uniformization (forward algorithm,
    # finite granularity), which differs slightly from the analytical PDF.
    # The refined MAP LL should be close to the analytical MLE LL — within
    # ~1% in absolute log-likelihood, which is well below the granularity-
    # related approximation error of the forward algorithm. The point of
    # this linchpin is not to pin down absolute agreement with the
    # analytical formula, but to verify that log_likelihood() is returning
    # a quantity in the right ballpark: maximum log-likelihood, not log-
    # posterior, not negative log-likelihood, not summed over the wrong axis.
    rel_err = abs(ll_refined - ll_analytical) / abs(ll_analytical)
    assert rel_err < 0.02, (
        f"Refined LL {ll_refined:.4f} vs analytical MLE LL "
        f"{ll_analytical:.4f} (rel err {rel_err:.4f})"
    )
    # Sanity: LL must be negative for continuous PDFs that integrate to 1
    # over Σtᵢ > 0; positive LL means we're returning a PMF or a sum of
    # probabilities instead.
    assert ll_refined < 0.0, (
        f"LL must be negative for continuous data ≫ 0, got {ll_refined}"
    )


def test_log_likelihood_at_supplied_theta_matches_manual_sum():
    """LL at user-supplied θ is close to the analytical Exp(λ) LL.

    The library evaluates the PDF via uniformization, which introduces
    a small approximation error relative to the closed-form Exp PDF.
    The point of this test is that supplying θ goes to the right code
    path (no transformation, model evaluated at that θ exactly).
    """
    np.random.seed(7)
    data = np.random.exponential(scale=0.5, size=50)  # λ=2
    svgd, _ = _fit_exp(data, n_iterations=200, seed=11)

    theta_const = jnp.array([1.5])
    ll = svgd.log_likelihood(theta=theta_const)

    arr = np.asarray(data)
    manual = float(len(arr) * np.log(1.5) - 1.5 * np.sum(arr))
    # Allow a few percent relative error from the forward-algorithm
    # approximation; this is the same tolerance as the linchpin test.
    rel_err = abs(ll - manual) / abs(manual)
    assert rel_err < 0.02, (
        f"LL at supplied theta {ll:.4f} vs manual sum {manual:.4f} "
        f"(rel err {rel_err:.4f})"
    )

    # Now perturb theta — LL should change as expected.
    ll_other = svgd.log_likelihood(theta=jnp.array([2.0]))
    manual_other = float(len(arr) * np.log(2.0) - 2.0 * np.sum(arr))
    assert abs((ll - ll_other) - (manual - manual_other)) < 0.5, (
        f"LL difference between thetas does not track analytical: "
        f"library Δ={ll - ll_other:.4f} vs analytical Δ={manual - manual_other:.4f}"
    )


def test_log_likelihood_independent_of_regularization():
    """Pure LL is identical whether SVGD ran with regularization or not."""
    np.random.seed(42)
    data = np.random.exponential(scale=0.5, size=120)

    # Both fits use the SAME data, same graph, same prior, same seed —
    # only differing in regularization.
    g = _build_exp_graph()
    model = Graph.pmf_and_moments_from_graph(
        g, nr_moments=2, discrete=False, theta_dim=1
    )
    sample_moments = jnp.array([np.mean(data), np.mean(data ** 2)])
    schedule = ExpStepSize(first_step=0.01, last_step=0.001, tau=500.0)

    svgd_reg = SVGD(
        model=model, observed_data=data, prior=_flat_prior,
        theta_dim=1, n_particles=60, n_iterations=1500,
        learning_rate=schedule, seed=21, verbose=False,
        regularization=5.0, nr_moments=2,
    )
    svgd_reg.optimize()
    svgd_no_reg = SVGD(
        model=model, observed_data=data, prior=_flat_prior,
        theta_dim=1, n_particles=60, n_iterations=1500,
        learning_rate=schedule, seed=21, verbose=False,
        regularization=0.0,
    )
    svgd_no_reg.optimize()

    # Evaluate both LLs at the SAME theta (the analytical MLE) so we
    # isolate "is LL the right formula" from "do the fits converge to
    # the same point". Both should agree to numerical precision.
    lam_mle = jnp.array([float(len(data) / np.sum(data))])
    ll_reg = svgd_reg.log_likelihood(theta=lam_mle)
    ll_no = svgd_no_reg.log_likelihood(theta=lam_mle)
    assert abs(ll_reg - ll_no) < 1e-6


def test_log_likelihood_unfitted_raises():
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=10)
    g = _build_exp_graph()
    model = Graph.pmf_and_moments_from_graph(
        g, nr_moments=2, discrete=False, theta_dim=1
    )
    svgd = SVGD(
        model=model, observed_data=data, prior=_flat_prior,
        theta_dim=1, n_particles=10, n_iterations=5,
        verbose=False,
    )
    with pytest.raises(RuntimeError, match="has not been fitted"):
        svgd.log_likelihood()


def test_log_likelihood_wrong_shape_raises():
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=20)
    svgd, _ = _fit_exp(data, n_iterations=100, seed=3)
    with pytest.raises(ValueError, match="shape"):
        svgd.log_likelihood(theta=jnp.array([1.0, 2.0]))


def test_degrees_of_freedom_no_fixed():
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=20)
    svgd, _ = _fit_two_param(data, n_iterations=100, seed=3)
    assert svgd.degrees_of_freedom == 2


def test_degrees_of_freedom_with_fixed_tuples():
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=20)
    svgd, _ = _fit_two_param(data, n_iterations=100, seed=3,
                             fixed=[(0, 2.0)])
    assert svgd.degrees_of_freedom == 1


def test_degrees_of_freedom_binary_mask():
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=20)
    svgd, _ = _fit_two_param(data, n_iterations=100, seed=3, fixed=[0, 1])
    assert svgd.degrees_of_freedom == 1


def test_n_observations_dense_1d():
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=37)
    svgd, _ = _fit_exp(data, n_iterations=50, seed=1)
    assert svgd.n_observations == 37


def test_n_observations_dense_1d_with_nan():
    np.random.seed(0)
    raw = np.random.exponential(scale=1.0, size=40)
    raw[5] = np.nan
    raw[10] = np.nan
    raw[20] = np.nan
    svgd, _ = _fit_exp(raw, n_iterations=50, seed=1)
    assert svgd.n_observations == 37


# -----------------------------------------------------------------------------
# AIC / BIC tests
# -----------------------------------------------------------------------------


def test_aic_formula():
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=100)  # λ=1
    svgd, _ = _fit_exp(data, n_iterations=1500, seed=5)

    res = ms.aic(svgd, refine=True)
    assert isinstance(res, ms.AICResult)
    assert res.k == 1
    assert res.n == 100
    assert res.refined is True
    # AIC = 2k − 2·LL by definition
    assert math.isclose(res.aic, 2.0 * res.k - 2.0 * res.log_likelihood,
                        rel_tol=1e-12, abs_tol=1e-12)


def test_bic_formula():
    np.random.seed(1)
    data = np.random.exponential(scale=1.0, size=100)
    svgd, _ = _fit_exp(data, n_iterations=1500, seed=5)

    res = ms.bic(svgd, refine=True)
    assert isinstance(res, ms.BICResult)
    assert res.k == 1
    assert res.n == 100
    # BIC = k·log(n) − 2·LL
    assert math.isclose(
        res.bic, res.k * math.log(res.n) - 2.0 * res.log_likelihood,
        rel_tol=1e-12, abs_tol=1e-12,
    )


def test_aic_unfitted_raises():
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=10)
    g = _build_exp_graph()
    model = Graph.pmf_and_moments_from_graph(
        g, nr_moments=2, discrete=False, theta_dim=1
    )
    svgd = SVGD(
        model=model, observed_data=data, prior=_flat_prior,
        theta_dim=1, n_particles=10, n_iterations=5, verbose=False,
    )
    with pytest.raises(ValueError, match="not been fitted"):
        ms.aic(svgd)


# -----------------------------------------------------------------------------
# Likelihood ratio test
# -----------------------------------------------------------------------------


def test_lrt_basic_correct_df_and_pvalue():
    """Data ~ Exp(λ=2). Fit full (2-param chain) and nested (param 1 fixed
    at MLE-ish value). LRT should produce df=1 and a valid p-value."""
    np.random.seed(99)
    data = np.random.exponential(scale=0.5, size=150)  # λ=2

    # SAME model callable used for both fits — required by strict LRT.
    g = _build_two_param_graph()
    model = Graph.pmf_and_moments_from_graph(
        g, nr_moments=2, discrete=False, theta_dim=2
    )
    svgd_full, _ = _fit_two_param(
        data, n_iterations=1500, seed=11, model=model
    )
    svgd_nested, _ = _fit_two_param(
        data, n_iterations=1500, seed=11, fixed=[(1, 4.0)], model=model
    )

    res = ms.likelihood_ratio_test(svgd_full, svgd_nested, refine=True)
    assert isinstance(res, ms.LRTResult)
    assert res.df == 1
    assert res.k_full == 2
    assert res.k_nested == 1
    assert res.n == 150
    assert 0.0 <= res.p_value <= 1.0
    assert res.statistic >= 0.0
    assert res.log_likelihood_full + 1e-6 >= res.log_likelihood_nested


def test_lrt_strict_model_identity_mismatch():
    """Two fits built from separate pmf_and_moments_from_graph calls
    fail the strict identity check, even with the same graph."""
    np.random.seed(7)
    data = np.random.exponential(scale=0.5, size=80)

    # Build the model TWICE — yields distinct callable objects.
    svgd_a, _ = _fit_two_param(data, n_iterations=200, seed=1)
    svgd_b, _ = _fit_two_param(data, n_iterations=200, seed=1,
                               fixed=[(1, 4.0)])
    with pytest.raises(ValueError, match="SAME model callable"):
        ms.likelihood_ratio_test(svgd_a, svgd_b)


def test_lrt_strict_false_warns_instead():
    np.random.seed(7)
    data = np.random.exponential(scale=0.5, size=80)
    svgd_a, _ = _fit_two_param(data, n_iterations=200, seed=1)
    svgd_b, _ = _fit_two_param(data, n_iterations=200, seed=1,
                               fixed=[(1, 4.0)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Should still produce a result (with warnings)
        res = ms.likelihood_ratio_test(svgd_a, svgd_b, strict=False)
    assert isinstance(res, ms.LRTResult)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_lrt_theta_dim_mismatch_raises():
    np.random.seed(7)
    data = np.random.exponential(scale=0.5, size=60)
    svgd_full, _ = _fit_two_param(data, n_iterations=100, seed=1)
    svgd_nested, _ = _fit_exp(data, n_iterations=100, seed=1)
    with pytest.raises(ValueError, match="theta_dim mismatch"):
        ms.likelihood_ratio_test(svgd_full, svgd_nested)


def test_lrt_n_mismatch_raises():
    np.random.seed(7)
    data_big = np.random.exponential(scale=0.5, size=80)
    data_small = np.random.exponential(scale=0.5, size=40)

    g = _build_two_param_graph()
    model = Graph.pmf_and_moments_from_graph(
        g, nr_moments=2, discrete=False, theta_dim=2
    )
    svgd_full, _ = _fit_two_param(
        data_big, n_iterations=100, seed=1, model=model
    )
    svgd_nested, _ = _fit_two_param(
        data_small, n_iterations=100, seed=1, fixed=[(1, 4.0)], model=model
    )
    with pytest.raises(ValueError, match="n_observations mismatch"):
        ms.likelihood_ratio_test(svgd_full, svgd_nested)


def test_lrt_identical_fixed_mask_raises():
    np.random.seed(7)
    data = np.random.exponential(scale=0.5, size=60)
    g = _build_two_param_graph()
    model = Graph.pmf_and_moments_from_graph(
        g, nr_moments=2, discrete=False, theta_dim=2
    )
    svgd_a, _ = _fit_two_param(
        data, n_iterations=200, seed=1, fixed=[(1, 4.0)], model=model
    )
    svgd_b, _ = _fit_two_param(
        data, n_iterations=200, seed=1, fixed=[(1, 4.0)], model=model
    )
    with pytest.raises(ValueError, match="df = 0|identical fixed_mask"):
        ms.likelihood_ratio_test(svgd_a, svgd_b)


def test_lrt_nested_not_superset_raises():
    """Nested model frees a parameter that full fixes — refused."""
    np.random.seed(7)
    data = np.random.exponential(scale=0.5, size=60)
    g = _build_two_param_graph()
    model = Graph.pmf_and_moments_from_graph(
        g, nr_moments=2, discrete=False, theta_dim=2
    )
    # full fixes index 0; nested fixes index 1 (≠ superset)
    svgd_full, _ = _fit_two_param(
        data, n_iterations=200, seed=1, fixed=[(0, 2.0)], model=model
    )
    svgd_nested, _ = _fit_two_param(
        data, n_iterations=200, seed=1, fixed=[(1, 4.0)], model=model
    )
    with pytest.raises(ValueError, match="frees parameters"):
        ms.likelihood_ratio_test(svgd_full, svgd_nested)


def test_lrt_shared_fixed_value_mismatch_raises():
    """Both models share a fixed index, but at different values — refused.

    Tests the shared-fixed-value mismatch check directly: we build two
    SVGD objects (both with the same fixed_mask but different fixed_values),
    monkey-patch them to look like a nested pair (full has only index 0
    fixed, nested has indices 0 and 1 fixed), and verify the LRT raises
    when the SHARED index has different values.

    This is a unit-level check on _check_nested_fixed; the fitted state of
    the SVGD objects is irrelevant to it.
    """
    np.random.seed(7)
    data = np.random.exponential(scale=0.5, size=60)
    g = _build_two_param_graph()
    model = Graph.pmf_and_moments_from_graph(
        g, nr_moments=2, discrete=False, theta_dim=2
    )
    # Both fits fix only index 0 — both have one learnable parameter (index 1).
    svgd_full, _ = _fit_two_param(
        data, n_iterations=200, seed=1, fixed=[(0, 2.0)], model=model
    )
    svgd_nested, _ = _fit_two_param(
        data, n_iterations=200, seed=1, fixed=[(0, 3.0)], model=model
    )
    # Now manually escalate `nested` to also fix index 1, so it has a true
    # superset of `full`'s fixed indices. The shared index 0 still has a
    # value mismatch (2.0 vs 3.0 in PHI space).
    nm = np.asarray(svgd_nested.fixed_mask).copy()
    nm[1] = 1
    svgd_nested.fixed_mask = jnp.asarray(nm)
    nv = np.asarray(svgd_nested.fixed_values).copy()
    nv[1] = 0.5  # arbitrary phi value for the now-fixed parameter
    svgd_nested.fixed_values = jnp.asarray(nv)

    with pytest.raises(ValueError, match="different values"):
        ms.likelihood_ratio_test(svgd_full, svgd_nested)


# -----------------------------------------------------------------------------
# compare()
# -----------------------------------------------------------------------------


def test_compare_aic_basic():
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=100)
    svgd_a, _ = _fit_exp(data, n_iterations=600, seed=1)
    svgd_b, _ = _fit_exp(data, n_iterations=600, seed=2)
    table = ms.compare(("A", svgd_a), ("B", svgd_b), criterion='aic')
    assert isinstance(table, ms.ComparisonTable)
    assert table.criterion == 'aic'
    assert len(table.rows) == 2
    # Weights sum to 1
    assert math.isclose(sum(r.weight for r in table.rows), 1.0, abs_tol=1e-9)
    # First row is the best (lowest AIC)
    assert table.rows[0].criterion <= table.rows[1].criterion
    # Both rows have delta >= 0; best row has delta == 0
    assert table.rows[0].delta == 0.0
    assert table.rows[1].delta >= 0.0


def test_compare_bic_basic():
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=100)
    svgd_a, _ = _fit_exp(data, n_iterations=600, seed=1)
    svgd_b, _ = _fit_exp(data, n_iterations=600, seed=2)
    table = ms.compare(svgd_a, svgd_b, criterion='bic')
    assert table.criterion == 'bic'
    assert table.rows[0].name in ('model_0', 'model_1')


def test_compare_anonymous_naming():
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=80)
    svgd_a, _ = _fit_exp(data, n_iterations=300, seed=1)
    svgd_b, _ = _fit_exp(data, n_iterations=300, seed=2)
    table = ms.compare(svgd_a, svgd_b)
    names = {r.name for r in table.rows}
    assert names == {'model_0', 'model_1'}


def test_compare_minimum_two_models_required():
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=20)
    svgd, _ = _fit_exp(data, n_iterations=50, seed=1)
    with pytest.raises(ValueError, match="at least 2 models"):
        ms.compare(svgd)


def test_compare_invalid_criterion():
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=20)
    svgd_a, _ = _fit_exp(data, n_iterations=50, seed=1)
    svgd_b, _ = _fit_exp(data, n_iterations=50, seed=2)
    with pytest.raises(ValueError, match="aic.*bic"):
        ms.compare(svgd_a, svgd_b, criterion='dic')


def test_compare_n_mismatch_raises():
    """compare() requires all models to use the same n_observations."""
    np.random.seed(0)
    data_a = np.random.exponential(scale=1.0, size=80)
    data_b = np.random.exponential(scale=1.0, size=120)
    svgd_a, _ = _fit_exp(data_a, n_iterations=200, seed=1)
    svgd_b, _ = _fit_exp(data_b, n_iterations=200, seed=2)
    with pytest.raises(ValueError, match="same number of observations"):
        ms.compare(svgd_a, svgd_b)


# -----------------------------------------------------------------------------
# Repr smoke tests (no exceptions on print)
# -----------------------------------------------------------------------------


def test_repr_aic_bic_lrt():
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=80)
    g = _build_two_param_graph()
    model = Graph.pmf_and_moments_from_graph(
        g, nr_moments=2, discrete=False, theta_dim=2
    )
    full, _ = _fit_two_param(
        data, n_iterations=300, seed=1, model=model
    )
    nested, _ = _fit_two_param(
        data, n_iterations=300, seed=1, fixed=[(1, 2.0)], model=model
    )
    assert "AICResult" in repr(ms.aic(full))
    assert "BICResult" in repr(ms.bic(full))
    assert "LRTResult" in repr(ms.likelihood_ratio_test(full, nested))
    assert "Model comparison by AIC" in repr(ms.compare(full, nested))
