"""Batch G.2 -- exact_moment_grad on the multivariate wrapper + the
uniform 2-D-rewards rejection on the 1-D model (master §16b item 10).

Two USER DECISIONS (2026-08-15) shipped here:
1. FULL SYMMETRY: `pmf_and_moments_from_graph_multivariate` gains
   `exact_moment_grad: bool = True`, forwarded to its SINGLE internal
   per-feature model build (the loops reuse the closure); svgd forwards
   explicit values (R29's 2-D arm retired; svgd cells live in
   test_svgd_exact_moment_grad_{kwarg,rewards}.py).
2. UNIFORM REJECTION: the 1-D model rejects ndim>=2 rewards on ALL
   THREE compute paths via the shared _check_rewards_len guard. Before:
   the default (pybind) path CRASHED with an obscure XLA wrong-axis
   shape error, the FFI path returned SILENT GARBAGE (features beyond
   the first dropped), and the callback path accidentally computed the
   correct multivariate sum (undocumented; retired by recorded decision
   -- the wrapper is probe-verified value-identical, migration
   lossless).
"""
import contextlib
import logging

import numpy as np
import pytest

import phasic
from phasic import Graph

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402


COEFFS = [(1.0, 0.5), (2.0, 0.25), (0.5, 1.5)]


def _chain():
    g = Graph(1)
    s = g.starting_vertex()
    v = [g.find_or_create_vertex([i + 1]) for i in range(3)]
    va = g.find_or_create_vertex([9])
    s.add_edge(v[0], 1.0)
    for (a, b), c in zip([(v[0], v[1]), (v[1], v[2]), (v[2], va)], COEFFS):
        a.add_edge(b, list(c))
    return g


THETA = jnp.asarray([1.1, 0.9])
TIMES2D = jnp.asarray([[0.5, 1.0], [1.0, 1.5], [1.5, 2.0]])
RW2D = jnp.asarray([[1.0, 0.0, 2.0, 0.5, 1.0], [1.0, 1.0, 2.0, 1.0, 0.5]])


def _mv(**kw):
    kw.setdefault("nr_moments", 2)
    kw.setdefault("discrete", False)
    kw.setdefault("theta_dim", 2)
    return Graph.pmf_and_moments_from_graph_multivariate(_chain(), **kw)


def test_wrapper_forwarding_discrimination(monkeypatch):
    """The kwarg reaches the wrapper's SINGLE internal build call (the
    per-feature loops reuse the closure -- refuter-verified one-site
    architecture): explicit False ARRIVES; omitted arrives as the
    signature default True (NOT absent -- differs from svgd's
    None-absent convention, asserted exactly)."""
    seen = []
    orig = Graph.pmf_and_moments_from_graph.__func__

    def spy(cls, *a, **kw):
        seen.append(dict(kw))
        return orig(cls, *a, **kw)

    monkeypatch.setattr(Graph, "pmf_and_moments_from_graph",
                        classmethod(spy))
    _mv(exact_moment_grad=False)
    assert len(seen) == 1 and seen[0].get("exact_moment_grad") is False
    seen.clear()
    _mv()
    assert len(seen) == 1 and seen[0].get("exact_moment_grad") is True


def test_wrapper_opt_out_changes_behavior(monkeypatch):
    """False => ZERO exact C calls under grad; default => >= n_features
    full-size successes (measured at I1 smoke: 0 and 2)."""
    calls = []
    orig = Graph._moments_grad_theta

    def spy(self, K, *a, **kw):
        r = orig(self, K, *a, **kw)
        calls.append(np.asarray(r).size)
        return r

    monkeypatch.setattr(Graph, "_moments_grad_theta", spy)
    m_off = _mv(exact_moment_grad=False)
    g_off = np.asarray(jax.grad(
        lambda th: jnp.sum(m_off(th, TIMES2D, RW2D)[1]))(THETA))
    assert np.all(np.isfinite(g_off))
    assert len(calls) == 0, f"opt-out leaked {len(calls)} exact calls"

    m_on = _mv()
    g_on = np.asarray(jax.grad(
        lambda th: jnp.sum(m_on(th, TIMES2D, RW2D)[1]))(THETA))
    full = [s for s in calls if s == 4]
    assert len(full) >= 2, f"{len(full)} full-size successes (want >= 2)"
    # parity: both paths correct (measured rel 7.2e-10 at I1 smoke)
    np.testing.assert_allclose(g_on, g_off, rtol=1e-6)


def test_wrapper_default_vs_explicit_true_bitwise():
    m_def = _mv()
    m_true = _mv(exact_moment_grad=True)
    g_def = np.asarray(jax.grad(
        lambda th: jnp.sum(m_def(th, TIMES2D, RW2D)[1]))(THETA))
    g_true = np.asarray(jax.grad(
        lambda th: jnp.sum(m_true(th, TIMES2D, RW2D)[1]))(THETA))
    np.testing.assert_array_equal(g_def, g_true)


def test_1d_model_rejects_2d_rewards_default_path():
    """§16b item 10 (uniform rejection, decision 2026-08-15): the
    default-path 1-D model raises the ACTIONABLE ValueError (previously
    an obscure XLA CpuCallback crash with a wrong-axis shape spec) --
    forward AND grad, naming the multivariate route and the correct
    (n_features, n_vertices) orientation."""
    m = Graph.pmf_and_moments_from_graph(_chain(), nr_moments=2,
                                         discrete=False, theta_dim=2)
    times = TIMES2D[:, 0]
    with pytest.raises(ValueError, match="multivariate"):
        m(THETA, times, RW2D)
    with pytest.raises(ValueError, match="n_features, n_vertices"):
        jax.grad(lambda th: jnp.sum(m(th, times, RW2D)[1]))(THETA)


def test_1d_model_rejects_2d_rewards_ffi_path():
    """The FFI path previously returned SILENT GARBAGE for 2-D rewards
    (correct values flattened into row 0, remaining feature rows zero)
    -- the uniform rejection replaces silent-wrong with loud."""
    m = Graph.pmf_and_moments_from_graph(_chain(), nr_moments=2,
                                         discrete=False, theta_dim=2,
                                         use_ffi=True)
    with pytest.raises(ValueError, match="multivariate"):
        m(THETA, TIMES2D[:, 0], RW2D)


def test_1d_model_rejects_2d_rewards_callback_path():
    """The callback path accidentally WORKED for 2-D rewards
    (value-identical to the wrapper, probe-verified) -- retired by
    RECORDED user decision for a uniform contract; the wrapper is the
    one blessed 2-D route (lossless migration)."""
    g = _chain()
    g.weight_callback = lambda t, c: jnp.dot(t, c[:2])
    m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False,
                                         theta_dim=2)
    with pytest.raises(ValueError, match="multivariate"):
        m(THETA, TIMES2D[:, 0], RW2D)


def test_wrapper_still_works_with_2d_rewards():
    """The wrapper (the blessed route) is untouched by the rejection --
    its per-feature 1-D slices never present 2-D rewards to the model.
    Value regression-guarded by the A/B/C multivariate cells; here just
    the composition."""
    m = _mv()
    pmf, mom = m(THETA, TIMES2D, RW2D)
    assert np.all(np.isfinite(np.asarray(pmf)))
    assert np.all(np.isfinite(np.asarray(mom)))


def test_exposure_with_2d_rewards_statically_impossible():
    """The plan-review worry: retiring R29's 2-D arm might expose an
    exposure x 2-D combination it silently blocked. Resolution (post
    R32-narrowing): the SPARSE+exposure route is rejected by the
    existing sparse-exposure rule, and the dead dense-1-D-observations
    route is rejected by R32 -- both crisp. The WORKING dense-2-D-obs +
    2-D-rewards + exposure route (which the plan review wrongly declared
    dead) remains, pinned by the pre-existing multivariate exposure
    tests with R11's warning."""
    from phasic.exceptions import SvgdConfigError
    from phasic.svgd import dense_to_sparse
    g = _chain()
    n = g.vertices_length()
    rw = np.ones((2, n))
    obs_sparse = dense_to_sparse(np.asarray([[0.5, 1.0], [1.0, 1.5]]))
    with pytest.raises(SvgdConfigError,
                       match="exposure is not supported with Sparse"):
        g.svgd(obs_sparse, rewards=rw, exposure=np.asarray([1.0, 2.0]),
               exposure_param_index=0, exact_moment_grad=False,
               n_iterations=1, n_particles=4)
    with pytest.raises(SvgdConfigError):
        g.svgd(np.asarray([0.5, 1.0]), rewards=rw,
               exposure=np.asarray([1.0, 2.0]), exposure_param_index=0,
               exact_moment_grad=False, n_iterations=1, n_particles=4)
