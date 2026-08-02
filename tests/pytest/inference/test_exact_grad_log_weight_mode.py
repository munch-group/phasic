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
    """4-vertex chain, 3 params, two edges each using all 3 params.

    NOTE: both edges' coefficients happen to have the SAME product
    (1.0*0.5*2.0 == 2.0*1.0*0.5 == 1.0), so in log mode (w_e =
    prod_i c_e[i]*theta_i) the two edges are numerically INDISTINGUISHABLE
    at every theta -- this fixture cannot detect a bug that scrambles which
    physical edge a tape input maps to (found via adversarial review; see
    _four_param_branching below for a fixture that CAN)."""
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


def _four_param_branching():
    """5-vertex graph with a BRANCHING vertex (v3 has two out-edges) and 4
    params, where every edge's coefficient PRODUCT is genuinely different
    (1.0, 10.5, 1.5, 0.5) -- so the four edges have distinguishable weight
    values at every theta, making an edge-index-mapping bug detectable
    (unlike _three_param_two_edge above).

    Verifiable against an EXACT closed form, no finite differences needed:
    in log mode w_e = prod_i(c_e[i]*theta_i) = (prod_i theta_i) * (prod_i
    c_e[i]) -- the theta-product factors out IDENTICALLY across every edge
    (log mode always multiplies ALL params, regardless of edge), so the
    generator matrix Q(theta) = f(theta)*Q_c for a theta-INDEPENDENT matrix
    Q_c and scalar f(theta) = prod_i theta_i. The raw moments are therefore
    m_r(theta) = m_r(1)*f(theta)^(-r) (homogeneous of degree -r), giving
    the EXACT identity dm_r/dtheta_j = -r*m_r(theta)/theta_j for ANY
    log-mode graph. Verified independently before use as a test oracle:
    matches both the shipped exact-grad C function (to ~1e-16) and central
    difference (to ~1e-10, FD's own truncation error) on this fixture."""
    g = phasic.Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    v0 = g.find_or_create_vertex([0])
    s.add_edge(v3, 1.0)
    v3.add_edge_parameterized(v2, 0.0, [2.0, 0.1, 5.0, 1.0])     # prod c = 1.0
    v3.add_edge_parameterized(v1, 0.0, [0.5, 3.0, 1.0, 7.0])     # prod c = 10.5 (branch)
    v2.add_edge_parameterized(v0, 0.0, [1.0, 2.0, 0.25, 3.0])    # prod c = 1.5
    v1.add_edge_parameterized(v0, 0.0, [4.0, 0.5, 2.0, 0.125])   # prod c = 0.5
    g.weight_mode = 'log'
    return g


def _closed_form_log_mode_jacobian(theta, m):
    """Exact (not approximate) oracle for ANY log-mode graph: dm_r/dtheta_j =
    -(r+1)*m_r/theta_j (the moment-homogeneity identity derived above)."""
    theta = np.asarray(theta, dtype=float)
    m = np.asarray(m, dtype=float)
    K, P = len(m), len(theta)
    return np.array([[-(k + 1) * m[k] / theta[j] for j in range(P)] for k in range(K)])


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


def test_exact_grad_matches_closed_form_branching_distinguishable_edges():
    """P=4, a BRANCHING vertex, and edges with genuinely different
    coefficient products (1.0, 10.5, 1.5, 0.5) -- distinguishes the edges
    numerically, so an edge-index-mapping bug in the C contraction would be
    caught here even though it wasn't detectable by the degenerate
    _three_param_two_edge fixture above (found via adversarial review).
    Checked against an EXACT closed form (moment homogeneity in log mode),
    not an approximate central difference -- see
    _closed_form_log_mode_jacobian's docstring for the derivation."""
    theta = [1e3, 1e-2, 5.0, 0.2]  # deliberately mixed-scale too
    K = 3
    build = _four_param_branching

    g_native = build()
    g_native.update_weights(theta, log=True)
    m = np.asarray(g_native.moments(K))
    J_closed = _closed_form_log_mode_jacobian(theta, m)

    model = Graph.pmf_and_moments_from_graph(
        build(), nr_moments=K, discrete=False, theta_dim=4, exact_moment_grad=True)
    J_exact = np.asarray(jax.jacobian(lambda th: model(th, jnp.asarray([1.0, 2.0]))[1])(jnp.asarray(theta)))
    assert _rel(J_exact, J_closed) < 1e-8  # exact vs exact: machine precision


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
    caller who omits the kwarg. Compares against explicit
    exact_moment_grad=True (must be byte-identical -- same code path) AND
    against explicit FD (must DIFFER -- proves the default is genuinely
    exact, not silently FD in both cases, which the first comparison alone
    cannot rule out; found via adversarial review)."""
    theta = jnp.asarray([2.0, 3.0])
    times = jnp.asarray([1.0, 2.0])
    model_default = Graph.pmf_and_moments_from_graph(
        _two_param_dense(), nr_moments=2, discrete=False, theta_dim=2)
    model_exact_explicit = Graph.pmf_and_moments_from_graph(
        _two_param_dense(), nr_moments=2, discrete=False, theta_dim=2, exact_moment_grad=True)
    model_fd_explicit = Graph.pmf_and_moments_from_graph(
        _two_param_dense(), nr_moments=2, discrete=False, theta_dim=2, exact_moment_grad=False)

    g_default = np.asarray(jax.grad(lambda th: jnp.sum(model_default(th, times)[1]))(theta))
    g_exact = np.asarray(jax.grad(lambda th: jnp.sum(model_exact_explicit(th, times)[1]))(theta))
    g_fd = np.asarray(jax.grad(lambda th: jnp.sum(model_fd_explicit(th, times)[1]))(theta))
    np.testing.assert_array_equal(g_default, g_exact)
    assert _rel(g_default, g_fd) < 1e-4  # FD is still a close approximation
    assert not np.array_equal(g_default, g_fd)  # but genuinely a different code path


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
    model_bwd falls back to FD -- the end-to-end gradient must stay finite
    AND the decline must be directly confirmed (checking finiteness alone,
    as a prior version of this test did, would also pass if the exact path
    had silently SUCCEEDED with a coincidentally-finite value; found via
    adversarial review). Confirmed two ways: the raw C function returns
    empty at this theta, and the end-to-end INFO log names the decline."""
    theta_np = [1.0, 1e-13]
    g_direct = _two_param_dense()
    g_direct.update_weights(theta_np, log=True)
    assert len(g_direct._moments_grad_theta_log(2, theta_np)) == 0, (
        "expected the C function to decline (empty) at this ill-conditioned theta")

    model = Graph.pmf_and_moments_from_graph(
        _two_param_dense(), nr_moments=2, discrete=False, theta_dim=2, exact_moment_grad=True)
    theta = jnp.asarray(theta_np)
    times = jnp.asarray([1.0, 2.0])
    with _capture_phasic_info_logs() as handler:
        grad = np.asarray(jax.grad(lambda th: jnp.sum(model(th, times)[1]))(theta))
    assert np.all(np.isfinite(grad))
    messages = [r.getMessage() for r in handler.records]
    assert any("declined at theta" in m and "finite differences" in m for m in messages)
