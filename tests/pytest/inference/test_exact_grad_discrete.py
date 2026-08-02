"""B3 discrete/was_dph exact gradient: Graph.pmf_and_moments_from_graph(...,
exact_moment_grad=True, discrete=True) on discretize()'d (was_dph=True,
renormalised/sibling-coupled) and native DPH (was_dph=False, plain linear)
graphs.

Extends the continuous-only exact reverse-mode theta-adjoint
(test_fd_gradient_mixed_scale.py) to discrete graphs via the new C function
ptd_moments_grad_theta_dph (src/c/phasic.c). See
b3-batch3-mpfr-and-discrete-derisk.md and b3-discrete-theta-adjoint-plan.md
for the math (renorm quotient-rule Jacobian + discrete moment correction).

Every assertion is anchored to an INDEPENDENT oracle: native central-difference
of graph.moments(K, discrete=True) (never the exact-grad code path itself).
"""
import contextlib
import logging

import numpy as np
import pytest

import phasic
from phasic import Graph

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402


# --------------------------------------------------------------------------- fixtures
def _erlang():
    """2-stage parameterized Erlang (continuous), to feed discretize()."""
    g = phasic.Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([2])
    b = g.find_or_create_vertex([1])
    s.add_edge(a, 1.0)
    a.add_edge(b, [1.0])
    return g


def _chain(n):
    """n-stage parameterized chain (continuous), to feed discretize(); exposes
    sibling coupling (each transient vertex gets 2 out-edges after
    discretize: advance + aux-return, sharing one renorm denominator)."""
    g = phasic.Graph(1)
    s = g.starting_vertex()
    vs = [g.find_or_create_vertex([n + 1 - i]) for i in range(n + 1)]
    s.add_edge(vs[0], 1.0)
    for i in range(n):
        vs[i].add_edge(vs[i + 1], [1.0 if j == i else 0.0 for j in range(n)])
    return g


def _dph_native(probs):
    """Native DPH: is_discrete=True, was_dph=False (edge weight IS c.theta
    directly -- plain linear contraction, no renormalisation)."""
    g = phasic.Graph(1)
    s = g.starting_vertex()
    n = len(probs)
    vs = [g.find_or_create_vertex([n + 1 - i]) for i in range(n + 1)]
    s.add_edge(vs[0], 1.0)
    for i in range(n):
        coeff = [0.0] * n
        coeff[i] = 1.0
        vs[i].add_edge(vs[i + 1], coeff)
    g.is_discrete = True
    return g


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-8)))


def _native_moments_grad_cd(build, theta, K, eps=1e-6):
    """Independent oracle: theta-perturbation central difference of the
    NATIVE graph.moments(K, discrete=True) (never the code path under test)."""
    theta = np.asarray(theta, dtype=float)
    P = len(theta)
    J = np.zeros((K, P))
    for j in range(P):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        gp = build(); gp.update_weights(list(tp))
        gm = build(); gm.update_weights(list(tm))
        mp = np.asarray(gp.moments(K, discrete=True))[:K]
        mm = np.asarray(gm.moments(K, discrete=True))[:K]
        J[:, j] = (mp - mm) / (2 * eps)
    return J


# --------------------------------------------------------------------------- was_dph (discretize())
@pytest.mark.parametrize("nr_moments", [2, 3])
def test_exact_grad_matches_native_cd_was_dph_erlang(nr_moments):
    theta = np.array([0.3, 0.4])
    model = Graph.pmf_and_moments_from_graph(
        _erlang().discretize(0.5), nr_moments=nr_moments, discrete=True,
        exact_moment_grad=True)

    def loss(th):
        _, moments = model(th, jnp.asarray([2.0, 3.0]))
        return moments  # vector-valued; use jacobian below

    J_exact = np.asarray(jax.jacobian(loss)(jnp.asarray(theta)))
    J_cd = _native_moments_grad_cd(lambda: _erlang().discretize(0.5), theta, nr_moments)
    assert _rel(J_exact, J_cd) < 1e-4


def test_exact_grad_matches_native_cd_was_dph_sibling_coupling():
    """_chain(2).discretize(0.3): 3 params, each transient vertex has 2
    out-edges (advance + aux) sharing one renorm denominator -- exercises the
    sibling-coupling quotient rule, not just a single-out-edge vertex."""
    theta = np.array([0.4, 0.6, 1.0])
    build = lambda: _chain(2).discretize(0.3)
    model = Graph.pmf_and_moments_from_graph(
        build(), nr_moments=2, discrete=True, exact_moment_grad=True)

    J_exact = np.asarray(jax.jacobian(lambda th: model(th, jnp.asarray([2.0, 3.0]))[1])(jnp.asarray(theta)))
    J_cd = _native_moments_grad_cd(build, theta, 2)
    assert _rel(J_exact, J_cd) < 1e-4


# --------------------------------------------------------------------------- native DPH (was_dph=False)
def test_exact_grad_matches_native_cd_native_dph():
    theta = np.array([0.3, 0.4])
    build = lambda: _dph_native((1.0, 1.0))
    model = Graph.pmf_and_moments_from_graph(
        build(), nr_moments=2, discrete=True, exact_moment_grad=True)

    J_exact = np.asarray(jax.jacobian(lambda th: model(th, jnp.asarray([2.0, 3.0]))[1])(jnp.asarray(theta)))
    J_cd = _native_moments_grad_cd(build, theta, 2)
    assert _rel(J_exact, J_cd) < 1e-4


def test_is_discrete_propagates_to_exact_grad_without_per_call_flag():
    """A graph flagged is_discrete produces the discrete gradient even when
    discrete=False is passed -- mirrors
    test_discrete_moments_and_reward.py::test_is_discrete_propagates_without_per_call_flag
    for the GRADIENT path (the exact-grad gate must follow the same effective
    discreteness as the forward, or it would silently apply the continuous
    Jacobian to a discrete forward)."""
    theta = np.array([0.3, 0.4])
    build = lambda: _dph_native((1.0, 1.0))
    model = Graph.pmf_and_moments_from_graph(
        build(), nr_moments=2, discrete=False, exact_moment_grad=True)

    J_exact = np.asarray(jax.jacobian(lambda th: model(th, jnp.asarray([2.0, 3.0]))[1])(jnp.asarray(theta)))
    J_cd = _native_moments_grad_cd(build, theta, 2)
    assert _rel(J_exact, J_cd) < 1e-4


# --------------------------------------------------------------------------- grad+vmap (SVGD-safety)
def test_exact_grad_vmap_matches_fd_was_dph():
    thetas = jnp.asarray([[0.3, 0.4], [0.5, 0.3], [0.2, 0.6]])
    model_exact = Graph.pmf_and_moments_from_graph(
        _erlang().discretize(0.5), nr_moments=2, discrete=True, exact_moment_grad=True)
    model_fd = Graph.pmf_and_moments_from_graph(
        _erlang().discretize(0.5), nr_moments=2, discrete=True, exact_moment_grad=False)
    times = jnp.asarray([2.0, 3.0])

    def loss(model, th):
        return jnp.sum(model(th, times)[1])

    g_exact = np.asarray(jax.vmap(jax.grad(lambda th: loss(model_exact, th)))(thetas))
    g_fd = np.asarray(jax.vmap(jax.grad(lambda th: loss(model_fd, th)))(thetas))
    assert _rel(g_exact, g_fd) < 1e-4


# --------------------------------------------------------------------------- default path unchanged
def test_default_path_byte_identical_to_explicit_exact_discrete():
    """exact_moment_grad now defaults to True: omitting the kwarg must be
    byte-identical to passing exact_moment_grad=True explicitly (same
    guarantee as the continuous default), and explicit exact_moment_grad=False
    must still give the (different, FD-based) result -- no silent behavior
    change depending on how the kwarg is spelled."""
    theta = jnp.asarray([0.3, 0.4])
    times = jnp.asarray([2.0, 3.0])
    model_default = Graph.pmf_and_moments_from_graph(
        _erlang().discretize(0.5), nr_moments=2, discrete=True)
    model_exact_explicit = Graph.pmf_and_moments_from_graph(
        _erlang().discretize(0.5), nr_moments=2, discrete=True, exact_moment_grad=True)
    model_fd_explicit = Graph.pmf_and_moments_from_graph(
        _erlang().discretize(0.5), nr_moments=2, discrete=True, exact_moment_grad=False)

    g_default = np.asarray(jax.grad(lambda th: jnp.sum(model_default(th, times)[1]))(theta))
    g_exact = np.asarray(jax.grad(lambda th: jnp.sum(model_exact_explicit(th, times)[1]))(theta))
    g_fd = np.asarray(jax.grad(lambda th: jnp.sum(model_fd_explicit(th, times)[1]))(theta))
    np.testing.assert_array_equal(g_default, g_exact)
    assert _rel(g_default, g_fd) < 1e-4  # FD is still a close approximation, just not identical
    assert not np.array_equal(g_default, g_fd)  # but genuinely a different code path


# --------------------------------------------------------------------------- MPFR decline -> FD fallback
def test_exact_grad_falls_back_to_fd_at_extreme_condition():
    """At an ill-conditioned theta the C path declines (empty Jacobian) and
    model_bwd falls back to FD -- the end-to-end gradient must stay finite,
    not NaN/crash (mirrors the continuous test_fd_gradient_mixed_scale.py /
    dr_mpfr_gate_test.py MPFR gate)."""
    model = Graph.pmf_and_moments_from_graph(
        _erlang().discretize(0.5), nr_moments=2, discrete=True, exact_moment_grad=True)
    theta = jnp.asarray([1.0, 1e-13])
    times = jnp.asarray([2.0, 3.0])
    grad = np.asarray(jax.grad(lambda th: jnp.sum(model(th, times)[1]))(theta))
    assert np.all(np.isfinite(grad))


# --------------------------------------------------------------------------- no silent fallback: INFO logging
class _CollectingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def _capture_phasic_info_logs():
    """The 'phasic' logger has propagate=False (logging_config.py), so
    pytest's caplog (root-based) never sees its records. Attach a handler
    directly and force the level to INFO for the duration of the test,
    restoring both afterward."""
    logger = logging.getLogger('phasic')
    handler = _CollectingHandler()
    handler.setLevel(logging.INFO)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def test_no_silent_fallback_logs_on_explicit_fd():
    """exact_moment_grad=False must log why FD is being used -- never silent."""
    with _capture_phasic_info_logs() as handler:
        model = Graph.pmf_and_moments_from_graph(
            _erlang().discretize(0.5), nr_moments=2, discrete=True, exact_moment_grad=False)
        jax.grad(lambda th: jnp.sum(model(th, jnp.asarray([2.0, 3.0]))[1]))(jnp.asarray([0.3, 0.4]))
    messages = [r.getMessage() for r in handler.records]
    assert any("exact_moment_grad=False" in m and "finite differences" in m for m in messages)


def test_no_silent_fallback_logs_on_out_of_scope_weight_mode():
    """A weight_mode outside {None,'linear','log'} must log why exact grad
    was skipped, even though exact_moment_grad defaults to True.
    weight_mode='callback' is permanently out of scope (an arbitrary Python
    function is not analytically differentiable in general) -- 'log' moved
    IN scope in the log-weight-mode batch, so it can no longer serve as the
    out-of-scope example here (see test_exact_grad_log_weight_mode.py for
    log-mode coverage)."""
    g = _erlang()
    g.weight_callback = lambda theta, coefficients: float(coefficients[0] * theta[0])
    with _capture_phasic_info_logs() as handler:
        Graph.pmf_and_moments_from_graph(g, nr_moments=2)
    messages = [r.getMessage() for r in handler.records]
    assert any("weight_mode" in m and "finite differences" in m for m in messages)


def test_no_silent_fallback_logs_on_dynamic_decline():
    """An ill-conditioned theta (MPFR decline) must log why FD was used for
    that step, in addition to staying finite (test above)."""
    model = Graph.pmf_and_moments_from_graph(
        _erlang().discretize(0.5), nr_moments=2, discrete=True, exact_moment_grad=True)
    with _capture_phasic_info_logs() as handler:
        jax.grad(lambda th: jnp.sum(model(th, jnp.asarray([2.0, 3.0]))[1]))(jnp.asarray([1.0, 1e-13]))
    messages = [r.getMessage() for r in handler.records]
    assert any("declined at theta" in m and "finite differences" in m for m in messages)


def test_default_is_silent_when_exact_succeeds():
    """No FD-related log line should appear when the exact path is used
    successfully (the default, well-conditioned case) -- logging should be
    informative about DEVIATIONS from the default, not routine noise."""
    model = Graph.pmf_and_moments_from_graph(_erlang().discretize(0.5), nr_moments=2, discrete=True)
    with _capture_phasic_info_logs() as handler:
        jax.grad(lambda th: jnp.sum(model(th, jnp.asarray([2.0, 3.0]))[1]))(jnp.asarray([0.3, 0.4]))
    messages = [r.getMessage() for r in handler.records]
    assert not any("finite differences" in m for m in messages)
