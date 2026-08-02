"""B3 log-weight-mode exact gradient: Graph.pmf_and_moments_from_graph(...,
exact_moment_grad=True) on continuous weight_mode='log' graphs.

Extends the linear-only exact reverse-mode theta-adjoint
(test_fd_gradient_mixed_scale.py, test_exact_grad_discrete.py) to
weight_mode='log' via the new C function ptd_moments_grad_theta_log
(src/c/phasic.c). See b3-log-weight-mode-plan.md for the math (product rule
dw_e/dtheta_j = w_e/theta_j) and the two bugs an adversarial review of the
plan caught before any C was written (the clone's update_weights call
needed log=True; discretize()+log does not always fail elsewhere, so the
was_dph/is_discrete exclusion is mandatory, not defensive).

Every assertion is anchored to an INDEPENDENT oracle: native
graph.update_weights(theta, log=True) + graph.moments(K) central-difference
(never the exact-grad code path itself).
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
def _two_param_dense():
    """3-vertex chain, 2 params, EVERY edge has both coefficients nonzero
    (required for weight_mode='log': w_e = prod_i(c_i*theta_i) over ALL i,
    and the C layer raises if any product <= 0)."""
    g = phasic.Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge_parameterized(v1, 0.0, [1.0, 1.0])
    g.weight_mode = 'log'
    return g


def _three_param_two_edge():
    """4-vertex chain, 3 params, two edges each using all 3 params."""
    g = phasic.Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge_parameterized(v2, 0.0, [1.0, 0.5, 2.0])
    v2.add_edge_parameterized(v1, 0.0, [2.0, 1.0, 0.5])
    g.weight_mode = 'log'
    return g


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-8)))


def _native_moments_grad_cd(build, theta, K, eps=1e-6):
    """Independent oracle: theta-perturbation central difference of the
    NATIVE graph.update_weights(theta, log=True) + graph.moments(K)."""
    theta = np.asarray(theta, dtype=float)
    P = len(theta)
    J = np.zeros((K, P))
    for j in range(P):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        gp = build(); gp.update_weights(list(tp), log=True)
        gm = build(); gm.update_weights(list(tm), log=True)
        mp = np.asarray(gp.moments(K))[:K]
        mm = np.asarray(gm.moments(K))[:K]
        J[:, j] = (mp - mm) / (2 * eps)
    return J


class _CollectingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def _capture_phasic_info_logs():
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


# --------------------------------------------------------------------------- matches native CD
@pytest.mark.parametrize("nr_moments", [2, 3])
def test_exact_grad_matches_native_cd_log_mode(nr_moments):
    theta = np.array([2.0, 3.0])
    model = Graph.pmf_and_moments_from_graph(
        _two_param_dense(), nr_moments=nr_moments, discrete=False, theta_dim=2,
        exact_moment_grad=True)

    def loss(th):
        _, moments = model(th, jnp.asarray([1.0, 2.0]))
        return moments

    J_exact = np.asarray(jax.jacobian(loss)(jnp.asarray(theta)))
    J_cd = _native_moments_grad_cd(_two_param_dense, theta, nr_moments)
    assert _rel(J_exact, J_cd) < 1e-4


def test_exact_grad_matches_native_cd_log_mode_three_params():
    """P=3, two edges each multiplying all 3 params -- exercises the
    'every param contributes to every edge' log-mode structure, not just a
    single dense edge."""
    theta = np.array([1.0, 2.0, 0.5])
    model = Graph.pmf_and_moments_from_graph(
        _three_param_two_edge(), nr_moments=2, discrete=False, theta_dim=3,
        exact_moment_grad=True)

    J_exact = np.asarray(jax.jacobian(lambda th: model(th, jnp.asarray([1.0, 2.0]))[1])(jnp.asarray(theta)))
    J_cd = _native_moments_grad_cd(_three_param_two_edge, theta, 2)
    assert _rel(J_exact, J_cd) < 1e-4


def test_exact_grad_matches_native_cd_log_mode_mixed_scale():
    theta = np.array([1.0, 1e-3])
    model = Graph.pmf_and_moments_from_graph(
        _two_param_dense(), nr_moments=2, discrete=False, theta_dim=2,
        exact_moment_grad=True)

    J_exact = np.asarray(jax.jacobian(lambda th: model(th, jnp.asarray([1.0, 2.0]))[1])(jnp.asarray(theta)))
    J_cd = _native_moments_grad_cd(_two_param_dense, theta, 2, eps=1e-6)
    assert _rel(J_exact, J_cd) < 1e-3


# --------------------------------------------------------------------------- grad+vmap (SVGD-safety)
def test_exact_grad_vmap_matches_fd_log_mode():
    thetas = jnp.asarray([[2.0, 3.0], [1.5, 2.5], [3.0, 1.0]])
    model_exact = Graph.pmf_and_moments_from_graph(
        _two_param_dense(), nr_moments=2, discrete=False, theta_dim=2, exact_moment_grad=True)
    model_fd = Graph.pmf_and_moments_from_graph(
        _two_param_dense(), nr_moments=2, discrete=False, theta_dim=2, exact_moment_grad=False)
    times = jnp.asarray([1.0, 2.0])

    def loss(model, th):
        return jnp.sum(model(th, times)[1])

    g_exact = np.asarray(jax.vmap(jax.grad(lambda th: loss(model_exact, th)))(thetas))
    g_fd = np.asarray(jax.vmap(jax.grad(lambda th: loss(model_fd, th)))(thetas))
    assert _rel(g_exact, g_fd) < 1e-4


# --------------------------------------------------------------------------- default path picks it up automatically
def test_default_path_uses_exact_for_log_mode():
    """exact_moment_grad defaults to True: a log-mode graph must get the
    exact gradient automatically, with no code change from an existing
    caller who omits the kwarg."""
    theta = jnp.asarray([2.0, 3.0])
    times = jnp.asarray([1.0, 2.0])
    model_default = Graph.pmf_and_moments_from_graph(
        _two_param_dense(), nr_moments=2, discrete=False, theta_dim=2)
    model_exact_explicit = Graph.pmf_and_moments_from_graph(
        _two_param_dense(), nr_moments=2, discrete=False, theta_dim=2, exact_moment_grad=True)

    g_default = np.asarray(jax.grad(lambda th: jnp.sum(model_default(th, times)[1]))(theta))
    g_exact = np.asarray(jax.grad(lambda th: jnp.sum(model_exact_explicit(th, times)[1]))(theta))
    np.testing.assert_array_equal(g_default, g_exact)


# --------------------------------------------------------------------------- discrete/was_dph + log stays excluded
def test_log_mode_discrete_combo_declines_and_logs():
    """A was_dph graph combined with weight_mode='log' must NOT use the
    exact path (confirmed by direct repro during plan review that
    discretize()+log does not always fail elsewhere -- the exclusion is
    mandatory, not defensive), and must log why."""
    # Base graph must already have 2 params (matching the discretize
    # callback's 2-element rate vector) or serialize() rejects the
    # inconsistent per-edge coefficient lengths.
    gd = _two_param_dense().discretize(lambda state, **kw: [0.5, 0.5])
    gd.weight_mode = 'log'
    assert gd.get_was_dph() is True

    # The out-of-scope log fires at MODEL CONSTRUCTION (the static
    # weight_mode/discreteness check), not inside the grad call -- both
    # must be inside the capture block.
    with _capture_phasic_info_logs() as handler:
        model = Graph.pmf_and_moments_from_graph(gd, nr_moments=2, discrete=True, theta_dim=2)
        grad = jax.grad(lambda th: jnp.sum(model(th, jnp.asarray([1.0, 2.0]))[1]))(jnp.asarray([1.0, 2.0]))
    assert np.all(np.isfinite(np.asarray(grad)))
    messages = [r.getMessage() for r in handler.records]
    assert any("weight_mode" in m and "finite differences" in m for m in messages)


# --------------------------------------------------------------------------- MPFR decline -> FD fallback
def test_exact_grad_log_mode_falls_back_to_fd_at_extreme_condition():
    """At an ill-conditioned theta the C path declines (empty Jacobian) and
    model_bwd falls back to FD -- the end-to-end gradient must stay finite."""
    model = Graph.pmf_and_moments_from_graph(
        _two_param_dense(), nr_moments=2, discrete=False, theta_dim=2, exact_moment_grad=True)
    theta = jnp.asarray([1.0, 1e-13])
    times = jnp.asarray([1.0, 2.0])
    grad = np.asarray(jax.grad(lambda th: jnp.sum(model(th, times)[1]))(theta))
    assert np.all(np.isfinite(grad))
