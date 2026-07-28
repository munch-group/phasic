"""SVGD divergence robustness: a diverged particle (θ implying an astronomically
large transition rate) must NOT abort the whole optimize() run.

The native distribution build raises
``RuntimeError: Maximum outgoing rate (...) is too large to build a phase-type
distribution ...`` for such a θ; unguarded, that error crosses the
``jax.pure_callback`` boundary and kills the entire fit — one bad particle out of
N. The fix (deferred issue #1, fix A) catches ONLY that rate-blowup error inside
the callback and returns a finite penalty (pmf = _PMF_FLOOR, moments = 0), so the
optimizer sees a terrible-but-finite loss and steps away.

These pin: the fail-soft is NARROW (a non-rate RuntimeError still propagates), the
penalty is finite and shaped correctly, and the happy path is untouched.

Provenance: deferred-svgd-divergece-fix.md, fix A.
"""
import numpy as np
import pytest

import phasic
from phasic import Graph

jnp = pytest.importorskip("jax.numpy")


def _coal(state, indexer=None):
    k = int(state[0])
    return [] if k <= 1 else [([k - 1], [k * (k - 1) / 2.0])]  # rate = C(k,2)*theta


def _dph_rowsum_break(state, indexer=None):
    # weight = 2.0*theta; at theta>0.5 the row sum exceeds 1 -> dph_pmf raises a
    # DIFFERENT (non-rate-blowup) RuntimeError, which must NOT be swallowed.
    k = int(state[0])
    return [] if k <= 1 else [([k - 1], [2.0])]


# --------------------------------------------------------------------------- unit
def test_is_rate_blowup_matches_only_rate_errors():
    # RuntimeError band (rate >~ 1e18)
    assert phasic._is_rate_blowup(RuntimeError(
        "Maximum outgoing rate (6.0e+31) is too large to build a phase-type distribution"))
    assert phasic._is_rate_blowup(RuntimeError(
        "Maximum outgoing rate (1e40) is too large to compute a phase-type distribution"))
    # ValueError band (rate ~2.5e8 .. 1e18) -- a diverging particle hits this FIRST
    assert phasic._is_rate_blowup(ValueError(
        "granularity * time = 6.0e+10 would require too many forward-algorithm steps (cap: 1e9)."))
    assert phasic._is_rate_blowup(RuntimeError(
        "... outgoing rate divided by granularity <= 1 ... Increase the granularity"))
    # NOT a rate blowup — a genuine modelling error that must surface:
    assert not phasic._is_rate_blowup(RuntimeError("some unrelated failure"))
    assert not phasic._is_rate_blowup(RuntimeError(
        "Expected vertex ... to have outgoing rate <= 1 ... discrete phase-type distribution?"))
    # ValueError type alone is not enough — the message must match
    assert not phasic._is_rate_blowup(ValueError("theta must have shape (3,), got (2,)"))


def test_rate_blowup_penalty_shapes():
    times = np.array([0.5, 1.0, 2.0])
    pmf, mom = phasic._rate_blowup_penalty(times, 2, None)
    assert pmf.shape == (3,) and mom.shape == (2,)
    assert np.all(pmf == phasic._PMF_FLOOR) and np.all(mom == 0.0)
    # multivariate: rewards 2D -> (n_times, n_features) / (n_features, nr_moments)
    rewards2d = np.ones((5, 4))  # (n_vertices, n_features=4)
    pmf2, mom2 = phasic._rate_blowup_penalty(times, 3, rewards2d)
    assert pmf2.shape == (3, 4) and mom2.shape == (4, 3)


# --------------------------------------------------------------------------- integration
def test_huge_theta_returns_finite_penalty_not_crash():
    g = Graph(_coal, ipv=[4])
    model = Graph.pmf_and_moments_from_graph(g, nr_moments=2)
    times = jnp.asarray([0.5, 1.0, 2.0])
    # sanity: a normal theta yields a REAL distribution (penalty NOT applied)
    pmf_ok, mom_ok = model(jnp.asarray([1.0]), times)
    assert np.all(np.asarray(pmf_ok) > phasic._PMF_FLOOR)
    # A diverged theta -> finite penalty, no XlaRuntimeError. Cover BOTH the
    # ValueError band (~2.5e8..1e18) that a diverging particle reaches first AND
    # the extreme RuntimeError band (>1e18).
    for th in (1e8, 1e10, 1e15, 1e31):
        pmf_bad, mom_bad = model(jnp.asarray([th]), times)
        pmf_bad = np.asarray(pmf_bad)
        assert np.all(np.isfinite(pmf_bad)) and np.all(pmf_bad == phasic._PMF_FLOOR), th
        assert np.all(np.isfinite(np.asarray(mom_bad))), th


def test_grad_at_diverged_theta_is_finite_not_nan():
    # The FD backward divides by actual_diff = (theta+eps)-(theta-eps), which
    # underflows to 0 at extreme theta -> nan, killing the prior's restoring
    # force and poisoning the coupled SVGD kernel. The guard returns a finite 0
    # gradient there.
    import jax
    g = Graph(_coal, ipv=[4])
    model = Graph.pmf_and_moments_from_graph(g, nr_moments=2)
    times = jnp.asarray([0.5, 1.0, 2.0])
    for th in (1e8, 1e10, 1e15, 1e31):
        grad = jax.grad(lambda t: jnp.sum(model(t, times)[0]))(jnp.asarray([th]))
        assert np.all(np.isfinite(np.asarray(grad))), (th, np.asarray(grad))
    # a normal theta still has a non-trivial (finite, non-zero) gradient
    gnorm = jax.grad(lambda t: jnp.sum(model(t, times)[0]))(jnp.asarray([1.0]))
    assert np.all(np.isfinite(np.asarray(gnorm))) and np.any(np.asarray(gnorm) != 0.0)


def test_batched_penalises_only_the_diverged_particle():
    import jax
    g = Graph(_coal, ipv=[4])
    model = Graph.pmf_and_moments_from_graph(g, nr_moments=2)
    times = jnp.asarray([0.5, 1.0, 2.0])
    thetas = jnp.asarray([[1.0], [1e10], [2.0]])  # middle particle diverges (ValueError band)
    pmf, _ = jax.vmap(lambda th: model(th, times))(thetas)
    pmf = np.asarray(pmf)
    assert np.all(pmf[0] > phasic._PMF_FLOOR)          # good particle unaffected
    assert np.all(pmf[1] == phasic._PMF_FLOOR)         # only the diverged one penalised
    assert np.all(pmf[2] > phasic._PMF_FLOOR)


def test_svgd_run_with_a_diverged_particle_completes():
    # End-to-end: an SVGD fit where one particle sits in the uncomputable band
    # must COMPLETE with finite results (no XlaRuntimeError crash, no nan cloud),
    # not abort the run. This is the property the whole fix exists to guarantee.
    from phasic import SVGD, ExpStepSize
    g = Graph(_coal, ipv=[4])
    model = Graph.pmf_and_moments_from_graph(g, nr_moments=2)
    np.random.seed(0)
    data = jnp.asarray(np.random.exponential(1.0, size=40))
    ti = np.random.normal(0.0, 0.5, size=(16, 1)).astype(np.float64)
    ti[0, 0] = 1e10  # one diverged particle in the ValueError band
    svgd = SVGD(model=model, observed_data=data,
                prior=lambda phi: -0.5 * jnp.sum((phi / 10.0) ** 2),
                theta_dim=1, n_particles=16, n_iterations=12,
                theta_init=jnp.asarray(ti),
                learning_rate=ExpStepSize(0.05, 0.005, 200),
                parallel='vmap', seed=0, verbose=False)
    svgd.optimize()  # must not raise
    res = svgd.get_results()
    assert np.all(np.isfinite(np.asarray(res['particles'])))
    assert np.all(np.isfinite(np.asarray(res['theta_mean'])))


def test_zero_inflation_cdf_zero_is_fail_soft():
    # The zero-inflation cdf_zero closure is AUTO-attached (no opt-in) on
    # partial-coverage reward models and evaluated every SVGD iteration, so a
    # diverged particle reaching it must be fail-soft too, not abort the run.
    import jax
    g = Graph(_coal, ipv=[4])
    nv = g.vertices_length()
    model = Graph.pmf_and_moments_from_graph(g, nr_moments=2)
    czf = getattr(model, "_cdf_zero_fn", None)
    assert czf is not None, "pybind pmf_and_moments model should expose _cdf_zero_fn"
    rew = jnp.asarray([1.0, 0.0, 1.0, 0.0, 1.0])[:nv]  # atom at r=0 -> zero-inflation
    assert np.all(np.isfinite(np.asarray(czf(jnp.asarray([1.0]), rew))))  # normal
    for th in (1e10, 1e30):
        cz = np.asarray(czf(jnp.asarray([th]), rew))
        assert np.all(np.isfinite(cz)) and np.all(cz == phasic._PMF_FLOOR), th
        grad = jax.grad(lambda t: jnp.sum(czf(t, rew)))(jnp.asarray([th]))
        assert np.all(np.isfinite(np.asarray(grad))), th


def test_non_rate_runtime_error_still_propagates():
    # A DPH row-sum violation is a DIFFERENT RuntimeError; the narrow fail-soft
    # must NOT swallow it (genuine modelling errors must surface).
    g = Graph(_dph_rowsum_break, ipv=[3])
    model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=True)
    jumps = jnp.asarray([2.0, 3.0, 4.0])
    with pytest.raises(Exception):
        _ = model(jnp.asarray([1.0]), jumps)  # weight 2.0 -> row sum 2 > 1
