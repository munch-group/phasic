"""Correctness of the likelihood and of what it estimates, against
CLOSED FORMS.

The requirement this file guards (user, 2026-08-16): inference must be
correct for any model, and that must be tracked by the test suite
rather than by anyone's judgement.

These tests use closed-form phase-type densities (exponential, Erlang,
hypoexponential) as oracles, so nothing here depends on another phasic
code path being right. Three properties are covered:

  1. the forward PDF matches the true density;
  2. the estimator built on that PDF lands where the true likelihood's
     estimator lands (i.e. the approximation does not bias the answer);
  3. the gradient of the log-likelihood matches the closed-form
     gradient, across parameter scales.

Where a property does NOT currently hold, the test is marked
strict-xfail with the measured value in the message. Strict means the
suite fails if the gap silently closes as well as if it widens — either
way, someone has to come and update the recorded number. Full analysis:
b3-mixed-scale-defect-settled.md.
"""
import numpy as np
import pytest

import phasic
from phasic import Graph, set_log_level

set_log_level("ERROR")


# --------------------------------------------------------------- fixtures
def expo_graph(rate):
    g = Graph(1)
    s = g.starting_vertex()
    v = g.find_or_create_vertex([2])
    a = g.find_or_create_vertex([1])
    s.add_edge(v, 1.0)
    v.add_edge(a, [1.0])
    g.update_weights([rate])
    return g


def erlang3_graph(rate):
    g = Graph(1)
    s = g.starting_vertex()
    v1 = g.find_or_create_vertex([4])
    v2 = g.find_or_create_vertex([3])
    v3 = g.find_or_create_vertex([2])
    a = g.find_or_create_vertex([1])
    s.add_edge(v1, 1.0)
    v1.add_edge(v2, [1.0])
    v2.add_edge(v3, [1.0])
    v3.add_edge(a, [1.0])
    g.update_weights([rate])
    return g


def hypo2_graph(l1, l2):
    g = Graph(1)
    s = g.starting_vertex()
    v1 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    a = g.find_or_create_vertex([1])
    s.add_edge(v1, 1.0)
    v1.add_edge(v2, [1.0, 0.0])
    v2.add_edge(a, [0.0, 1.0])
    g.update_weights([l1, l2])
    return g


def expo_pdf(t, r):
    return r * np.exp(-r * t)


def erlang3_pdf(t, r):
    return r ** 3 * t ** 2 * np.exp(-r * t) / 2.0


def hypo2_pdf(t, l1, l2):
    return l1 * l2 / (l2 - l1) * (np.exp(-l1 * t) - np.exp(-l2 * t))


TIMES = [0.25, 0.8, 1.5, 3.0]


# ------------------------------------------------- 1. forward PDF accuracy
@pytest.mark.parametrize("name,graph,truth", [
    ("exponential", expo_graph(1.3), lambda t: expo_pdf(t, 1.3)),
    ("erlang3", erlang3_graph(0.9), lambda t: erlang3_pdf(t, 0.9)),
    ("hypoexp2", hypo2_graph(1.0, 0.5), lambda t: hypo2_pdf(t, 1.0, 0.5)),
])
def test_pdf_matches_closed_form_at_default_granularity(name, graph, truth):
    """The DEFAULT pdf path (granularity=0 -> auto) against closed forms.

    This pins the CURRENT accuracy, which is ~2.5e-3 relative and is
    first-order in 1/granularity (auto resolves to 1024 for ordinary
    rates). It is deliberately a loose bound: its job is to catch a
    REGRESSION in forward accuracy. The aspiration is pinned separately
    by the strict-xfail below.
    """
    errs = [abs(graph.pdf(t, 0) - truth(t)) / abs(truth(t)) for t in TIMES]
    assert max(errs) < 1e-2, (
        f"{name}: default pdf accuracy regressed to {max(errs):.2e}")


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN GAP: the default pdf path uses auto granularity (=1024 for "
    "ordinary rates) and is first-order accurate, giving ~2.5e-3 "
    "relative error. Deferred 3's Poisson-mixture route measured "
    "1.63e-11 on the same fixtures and would close this. See "
    "b3-mixed-scale-defect-settled.md section 3."))
def test_pdf_is_accurate_to_1e8_at_default_granularity():
    g = hypo2_graph(1.0, 0.5)
    errs = [abs(g.pdf(t, 0) - hypo2_pdf(t, 1.0, 0.5)) / hypo2_pdf(t, 1.0, 0.5)
            for t in TIMES]
    assert max(errs) < 1e-8


def test_pdf_accuracy_improves_with_granularity():
    """Documents the mechanism: the error is first-order in 1/granularity.

    If this ever stops holding, the uniformization stepping changed and
    every accuracy number in b3-mixed-scale-defect-settled.md is stale.
    """
    g = hypo2_graph(1.0, 0.5)

    def err(gran):
        return max(abs(g.pdf(t, gran) - hypo2_pdf(t, 1.0, 0.5))
                   / hypo2_pdf(t, 1.0, 0.5) for t in TIMES)

    e1k, e16k = err(1024), err(16384)
    assert e1k > e16k, "higher granularity did not improve accuracy"
    ratio = e1k / e16k
    assert 4.0 < ratio < 64.0, (
        f"error scaling changed: 1024 vs 16384 gave a factor {ratio:.1f}, "
        f"expected roughly 16 (first order)")


# ------------------------------------- 2. does the approximation bias the estimate?
def _mle(nll, x0):
    from scipy.optimize import minimize
    r = minimize(nll, x0, method="Nelder-Mead",
                 options=dict(xatol=1e-10, fatol=1e-10, maxiter=8000))
    return r.x


@pytest.fixture(scope="module")
def hypo_sample():
    rng = np.random.default_rng(0)
    true = np.array([1.0, 0.35])
    n = 1500
    data = rng.exponential(1 / true[0], n) + rng.exponential(1 / true[1], n)
    return true, data


def test_estimator_bias_is_bounded(hypo_sample):
    """The estimate from phasic's likelihood versus the estimate from the
    TRUE likelihood on the same data.

    This is the inference-correctness property: an approximation that
    shifts the argmax biases every fit, and unlike sampling error the
    bias does NOT shrink as data grows. Measured at ~2.5e-3 with the
    default granularity; bounded loosely here to catch regressions.
    """
    true, data = hypo_sample

    def nll_closed(th):
        l1, l2 = th
        if l1 <= 0 or l2 <= 0 or abs(l1 - l2) < 1e-9:
            return 1e18
        f = hypo2_pdf(data, l1, l2)
        return 1e18 if np.any(f <= 0) else -np.sum(np.log(f))

    def nll_phasic(th):
        l1, l2 = th
        if l1 <= 0 or l2 <= 0 or abs(l1 - l2) < 1e-9:
            return 1e18
        g = hypo2_graph(l1, l2)
        f = np.array([g.pdf(float(t), 0) for t in data])
        return 1e18 if np.any(f <= 0) else -np.sum(np.log(f))

    x_closed = _mle(nll_closed, [0.8, 0.5])
    x_phasic = _mle(nll_phasic, [0.8, 0.5])
    shift = float(np.max(np.abs(x_phasic - x_closed) / np.abs(x_closed)))
    assert shift < 1e-2, (
        f"estimator bias regressed to {shift:.2e} (was ~2.5e-3); the "
        f"approximate likelihood's argmax moved away from the true one")


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN GAP: the default likelihood's argmax is displaced by ~2.5e-3 "
    "relative, tracking the forward pdf error exactly. The bias is "
    "independent of sample size, so it dominates once N is large. See "
    "b3-mixed-scale-defect-settled.md section 4."))
def test_estimator_is_unbiased_to_1e6(hypo_sample):
    true, data = hypo_sample

    def nll_closed(th):
        l1, l2 = th
        if l1 <= 0 or l2 <= 0 or abs(l1 - l2) < 1e-9:
            return 1e18
        f = hypo2_pdf(data, l1, l2)
        return 1e18 if np.any(f <= 0) else -np.sum(np.log(f))

    def nll_phasic(th):
        l1, l2 = th
        if l1 <= 0 or l2 <= 0 or abs(l1 - l2) < 1e-9:
            return 1e18
        g = hypo2_graph(l1, l2)
        f = np.array([g.pdf(float(t), 0) for t in data])
        return 1e18 if np.any(f <= 0) else -np.sum(np.log(f))

    x_closed = _mle(nll_closed, [0.8, 0.5])
    x_phasic = _mle(nll_phasic, [0.8, 0.5])
    assert float(np.max(np.abs(x_phasic - x_closed) / np.abs(x_closed))) < 1e-6


# --------------------------------- 3. gradient of the log-likelihood vs closed form
def _shipped_and_true_grad(l2):
    """Gradient of sum_i log pdf(t_i) at theta=(1.0, l2): the shipped
    inference path versus the closed form."""
    import jax
    import jax.numpy as jnp

    model = Graph.pmf_and_moments_from_graph(hypo2_graph(1.0, l2),
                                             nr_moments=2)
    times = jnp.asarray(TIMES)

    def shipped(th):
        pmf, _ = model(th, times)
        return jnp.sum(jnp.log(pmf + 1e-10))

    def closed(th):
        l1, ll2 = th[0], th[1]
        f = l1 * ll2 / (ll2 - l1) * (jnp.exp(-l1 * times) - jnp.exp(-ll2 * times))
        return jnp.sum(jnp.log(f))

    th = jnp.asarray([1.0, l2])
    return (np.asarray(jax.grad(shipped)(th)),
            np.asarray(jax.grad(closed)(th)))


@pytest.mark.parametrize("l2", [0.5, 1e-2, 1e-4])
def test_loglik_gradient_matches_closed_form_at_ordinary_scales(l2):
    """At ordinary parameter scales the gradient must be right to the
    accuracy the forward allows (~1e-3, the uniformization floor)."""
    g_ship, g_true = _shipped_and_true_grad(l2)
    rel = float(np.max(np.abs(g_ship - g_true) / np.maximum(np.abs(g_true), 1e-300)))
    assert rel < 1e-2, f"gradient error {rel:.2e} at theta=(1.0, {l2:g})"


def test_loglik_gradient_degrades_only_below_1e8():
    """Pins WHERE the mixed-scale defect actually bites.

    The absolute FD step is 1e-7, so a parameter far below that is
    perturbed by more than its own magnitude. Measured: 2.6e-2 relative
    error at 1e-8, 2.1e-1 at 1e-9. This test states the boundary so a
    change in the FD step or the dispatch shows up as a failure here
    rather than as silently wrong science.
    """
    _, _ = _shipped_and_true_grad(0.5)          # warm
    r8 = _shipped_and_true_grad(1e-8)
    rel8 = float(np.max(np.abs(r8[0] - r8[1]) / np.maximum(np.abs(r8[1]), 1e-300)))
    r9 = _shipped_and_true_grad(1e-9)
    rel9 = float(np.max(np.abs(r9[0] - r9[1]) / np.maximum(np.abs(r9[1]), 1e-300)))

    assert rel8 < 1e-1, (
        f"error at theta1=1e-8 grew to {rel8:.2e} (recorded 2.6e-2)")
    assert rel9 > 1e-2, (
        f"error at theta1=1e-9 is {rel9:.2e}; the recorded 2.1e-1 "
        f"degradation has changed — re-derive the boundary")
    assert rel9 > rel8, "degradation is no longer monotone in scale ratio"
