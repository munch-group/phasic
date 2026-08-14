"""B3 callback-weight-mode exact gradient (Batch C):
Graph.pmf_and_moments_from_graph(..., exact_moment_grad=True) on
continuous weight_mode='callback' graphs with JAX-NATIVE callbacks.

Architecture (b3-batchC-plan.md v1+v2): the C side exports the
PRE-CONTRACTION per-tape-input moment adjoint (ptd_moments_binp_exit,
the shared core's 5th consumer) plus per-input coefficient vectors and a
C-side frozen flag; Python contracts J = binp @ W where W's rows are
jax.jit(jax.grad(callback)) evaluated per engaged edge. The chain-rule
composition was validated BEFORE any C existed (plan review probe:
1.2e-10 vs the shipped linear path).

SCOPE: continuous + JAX-native-under-jit callbacks. Decoupled theta
dimensions are FULLY SUPPORTED (v2 SS-A: update_weights(theta, callback=)
accepts any theta length BY DESIGN -- phasiccpp.cpp:1879-1884 -- and the
exit is theta-blind); non-JAX-native callbacks decline PERMANENTLY (the
honest boundary for the arbitrary-Python escape hatch); discrete x
callback declines statically (was_dph silently computes -- probe-
confirmed load-bearing, dr_batchC_d5_derisk.py).

Every value assertion is anchored to an INDEPENDENT oracle: the LINEAR
exact path on a twin graph computing the identical weight function
(FD-independent -- the mixed-scale cells' oracle) or the central-diff of
the model's own forward. Engagement asserted POSITIVELY via spies paired
with LIVE log-text filters (the A/B-G4 discipline).
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


def _chain(coeffs_list):
    g = Graph(1)
    s = g.starting_vertex()
    v = [g.find_or_create_vertex([i + 1]) for i in range(3)]
    va = g.find_or_create_vertex([9])
    s.add_edge(v[0], 1.0)
    for (a, b), c in zip([(v[0], v[1]), (v[1], v[2]), (v[2], va)],
                         coeffs_list):
        a.add_edge(b, list(c))
    return g


CB_LIN = lambda t, c: jnp.dot(t, c[:2])  # noqa: E731  linear-equivalent
CB_NL = lambda t, c: jnp.exp(t[0] * c[0] * 0.3) + t[1] * c[1] / (t[0] + 0.5)  # noqa: E731
CB_FLOAT = lambda t, c: float(c[0] * t[0] + c[1] * t[1])  # noqa: E731  non-JAX-native


def _cb_graph(cb=CB_LIN, coeffs=None):
    g = _chain(coeffs or COEFFS)
    g.weight_callback = cb
    return g


TIMES = jnp.asarray([0.5, 1.0, 1.5])
THETA = jnp.asarray([1.1, 0.9])


def _model(g, **kw):
    kw.setdefault("nr_moments", 2)
    kw.setdefault("discrete", False)
    kw.setdefault("theta_dim", 2)
    return Graph.pmf_and_moments_from_graph(g, **kw)


def _central_diff(model, theta, times, rewards=None, eps=1e-6):
    theta = np.asarray(theta, dtype=float)
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        tp = theta.copy(); tp[i] += eps
        tm = theta.copy(); tm[i] -= eps
        ap = (jnp.asarray(tp), times) + (() if rewards is None else (rewards,))
        am = (jnp.asarray(tm), times) + (() if rewards is None else (rewards,))
        grad[i] = float(jnp.sum(model(*ap)[1]) - jnp.sum(model(*am)[1])) / (2 * eps)
    return grad


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


# --------------------------------------------------------------------------- cells
def test_callback_matches_linear_twin_and_central_diff(monkeypatch):
    """Linear-equivalent callback: exact grad must equal the LINEAR exact
    path on the twin graph (FD-independent; measured 0.0 at I3 smoke) and
    the central-diff of its own forward; the binp exit must actually run
    (spy on the pybind entry, full-tuple results)."""
    calls = []
    orig = Graph._moments_binp_exit

    def spy(self, K, *a, **kw):
        result = orig(self, K, *a, **kw)
        calls.append(len(np.asarray(result[0])))
        return result

    monkeypatch.setattr(Graph, "_moments_binp_exit", spy)
    mcb = _model(_cb_graph())
    mlin = _model(_chain(COEFFS))
    with _capture_phasic_info_logs() as handler:
        g_cb = np.asarray(jax.grad(
            lambda th: jnp.sum(mcb(th, TIMES)[1]))(THETA))
        g_lin = np.asarray(jax.grad(
            lambda th: jnp.sum(mlin(th, TIMES)[1]))(THETA))
    g_ref = _central_diff(mcb, THETA, TIMES)
    np.testing.assert_allclose(g_cb, g_lin, rtol=1e-12)
    np.testing.assert_allclose(g_cb, g_ref, rtol=1e-6)
    messages = [r.getMessage() for r in handler.records]
    assert not any("finite differences" in m for m in messages), messages
    assert len(calls) >= 1 and all(s > 0 for s in calls), calls


def test_callback_mixed_scale_vs_linear_twin():
    """The motivating cell: at theta=[1, 1e-8] FD is the defect; the
    oracle is the linear exact twin (gate (b1) measured rel exactly 0.0
    at this theta)."""
    theta = jnp.asarray([1.0, 1e-8])
    mcb = _model(_cb_graph())
    mlin = _model(_chain(COEFFS))
    g_cb = np.asarray(jax.grad(lambda th: jnp.sum(mcb(th, TIMES)[1]))(theta))
    g_lin = np.asarray(jax.grad(lambda th: jnp.sum(mlin(th, TIMES)[1]))(theta))
    assert np.all(np.isfinite(g_cb))
    np.testing.assert_allclose(g_cb, g_lin, rtol=1e-12)


def test_callback_nonlinear_matches_central_diff():
    theta = jnp.asarray([1.3, 0.7])
    # G4 fold: model construction INSIDE the capture -- the static decline
    # logs at BUILD time, so a dispatch regression is invisible otherwise
    # (proven by the refuter's forced-scope-False simulation)
    with _capture_phasic_info_logs() as handler:
        m = _model(_cb_graph(CB_NL))
        g = np.asarray(jax.grad(lambda th: jnp.sum(m(th, TIMES)[1]))(theta))
    g_ref = _central_diff(m, theta, TIMES)
    np.testing.assert_allclose(g, g_ref, rtol=1e-5)
    messages = [r.getMessage() for r in handler.records]
    assert not any("finite differences" in m_ for m_ in messages), messages


def test_callback_lazy_decoupled_supported():
    """The v2 SS-A headline (the ANTI-Batch-B cell): a lazily-built
    aux-coefficients graph (coeff len 3 locks C param_length=3; theta has
    len 2) ENGAGES the exact path -- decoupled theta dimensions are
    supported by design for callback mode (the exit is theta-blind;
    update_weights(callback=) accepts any theta length)."""
    cb_aux = lambda t, c: t[0] * c[0] + t[1] * c[1] + c[2] * 0.1  # noqa: E731
    g = _chain([(1.0, 0.5, 2.0), (2.0, 0.25, 1.0), (0.5, 1.5, 0.3)])
    g.weight_callback = cb_aux
    assert g.param_length() == 3  # locked by the coefficient length
    with _capture_phasic_info_logs() as handler:
        m = _model(g)  # theta_dim=2 != 3
        grad = np.asarray(jax.grad(
            lambda th: jnp.sum(m(th, TIMES)[1]))(THETA))
    g_ref = _central_diff(m, THETA, TIMES)
    np.testing.assert_allclose(grad, g_ref, rtol=1e-5)
    messages = [r.getMessage() for r in handler.records]
    assert not any("finite differences" in m_ for m_ in messages), messages


def test_callback_rewards_engage_all_ones_bitwise(monkeypatch):
    """1-D rewards thread through the exit (the core hooks scale the
    chain PRODUCING binp -- gated, not assumed): mixed rewards match
    central-diff; all-ones == rewardless BITWISE."""
    calls = []
    orig = Graph._moments_binp_exit

    def spy(self, K, *a, **kw):
        rw = kw.get("rewards")
        result = orig(self, K, *a, **kw)
        calls.append((0 if rw is None else len(rw),
                      len(np.asarray(result[0]))))
        return result

    monkeypatch.setattr(Graph, "_moments_binp_exit", spy)
    n = _cb_graph().vertices_length()
    rewards = jnp.asarray([1.0, 0.0, 2.0, 0.5, 1.0])
    assert n == 5
    m = _model(_cb_graph())
    with _capture_phasic_info_logs() as handler:
        g_rw = np.asarray(jax.grad(
            lambda th: jnp.sum(m(th, TIMES, rewards)[1]))(THETA))
    g_ref = _central_diff(m, THETA, TIMES, rewards=rewards)
    np.testing.assert_allclose(g_rw, g_ref, rtol=1e-5)
    messages = [r.getMessage() for r in handler.records]
    assert not any("finite differences" in m_ for m_ in messages), messages
    assert any(rw_len == n and size > 0 for rw_len, size in calls), calls

    m2 = _model(_cb_graph())
    g_ones = np.asarray(jax.grad(
        lambda th: jnp.sum(m2(th, TIMES, jnp.ones(n))[1]))(THETA))
    m3 = _model(_cb_graph())
    g_no = np.asarray(jax.grad(
        lambda th: jnp.sum(m3(th, TIMES)[1]))(THETA))
    np.testing.assert_array_equal(g_ones, g_no)


def test_callback_non_jax_native_declines_permanently():
    """The PERMANENT boundary: a float()-collapsing callback fails the
    jit(grad) probe -> static decline naming non-JAX-native, FD works."""
    g = _cb_graph(CB_FLOAT)
    with _capture_phasic_info_logs() as handler:
        m = _model(g)
        grad = np.asarray(jax.grad(
            lambda th: jnp.sum(m(th, TIMES)[1]))(THETA))
    assert np.all(np.isfinite(grad))
    messages = [r.getMessage() for r in handler.records]
    assert any("non-JAX-native" in m_ for m_ in messages), messages


def test_callback_branching_on_theta_declines_at_construction():
    """The v2 SS-B probe-design cell: data-dependent Python branching on
    theta passes PLAIN grad (concrete tracers) but fails the DEPLOYED
    jit(grad) -- the construction probe must use the deployed transform,
    so this callback declines STATICALLY (never a per-call raise)."""
    def cb_branch(t, c):
        if t[0] > 1.0:  # Python control flow on theta: not jittable
            return t[0] * c[0]
        return t[1] * c[1]

    g = _cb_graph(cb_branch)
    with _capture_phasic_info_logs() as handler:
        m = _model(g)
        grad = np.asarray(jax.grad(
            lambda th: jnp.sum(m(th, TIMES)[1]))(THETA))
    assert np.all(np.isfinite(grad))
    messages = [r.getMessage() for r in handler.records]
    assert any("non-JAX-native" in m_ for m_ in messages), messages


def test_callback_discrete_declines_with_log():
    g = _cb_graph()
    theta = jnp.asarray([0.3, 0.4])
    jumps = jnp.asarray([1.0, 2.0])
    with _capture_phasic_info_logs() as handler:
        m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=True,
                                             theta_dim=2)
        grad = np.asarray(jax.grad(
            lambda th: jnp.sum(m(th, jumps)[1]))(theta))
    assert np.all(np.isfinite(grad))
    messages = [r.getMessage() for r in handler.records]
    assert any("callback" in m_ and "continuous-only" in m_
               for m_ in messages), messages


def test_callback_out_of_scope_log_gone(monkeypatch):
    """The engagement flip: pre-C a JAX-native callback emitted the
    generic out-of-scope decline; post-C it must NOT (live filter), and
    the exit must run (spy)."""
    calls = []
    orig = Graph._moments_binp_exit

    def spy(self, K, *a, **kw):
        result = orig(self, K, *a, **kw)
        # G4 fold: record SUCCESS size, not just the call (a call-count
        # floor stays green under an always-decline regression)
        calls.append(len(np.asarray(result[0])))
        return result

    monkeypatch.setattr(Graph, "_moments_binp_exit", spy)
    with _capture_phasic_info_logs() as handler:
        m = _model(_cb_graph(), exact_moment_grad=True)
        g = np.asarray(jax.grad(lambda th: jnp.sum(m(th, TIMES)[1]))(THETA))
    assert np.all(np.isfinite(g))
    messages = [r.getMessage() for r in handler.records]
    assert not any("not available for weight_mode" in m_
                   for m_ in messages), messages
    assert any(s > 0 for s in calls), calls


def test_callback_vmap_jit_composition():
    m = _model(_cb_graph())
    loss = lambda th: jnp.sum(m(th, TIMES)[1])  # noqa: E731
    particles = jnp.asarray([[1.1, 0.9], [0.7, 1.3], [1.5, 0.5]])
    g_vmap = np.asarray(jax.vmap(jax.grad(loss))(particles))
    g_per = np.stack([np.asarray(jax.grad(loss)(p)) for p in particles])
    g_jit = np.asarray(jax.jit(jax.vmap(jax.grad(loss)))(particles))
    np.testing.assert_array_equal(g_vmap, g_per)
    np.testing.assert_array_equal(g_jit, g_vmap)


def test_callback_svgd_frontdoor_default_runs_exact(monkeypatch):
    """Front-door svgd with callback= (explicit theta_dim REQUIRED for
    callback mode -- _resolve_inference_theta_dim raises otherwise),
    default exact kwarg: fit completes finite, the exit SUCCEEDS >=
    n_particles times (full-tuple results -- the B-G4 success-floor
    discipline), no static-decline logs."""
    calls = []
    orig = Graph._moments_binp_exit

    def spy(self, K, *a, **kw):
        result = orig(self, K, *a, **kw)
        calls.append(len(np.asarray(result[0])))
        return result

    monkeypatch.setattr(Graph, "_moments_binp_exit", spy)
    g = _chain(COEFFS)
    lg = logging.getLogger("phasic")
    h = _CollectingHandler(); h.setLevel(logging.INFO)
    prev = lg.level
    lg.addHandler(h); lg.setLevel(logging.INFO)
    try:
        fit = g.svgd(np.asarray([0.5, 1.0, 1.5, 2.0]), callback=CB_LIN,
                     theta_dim=2, n_iterations=2, n_particles=6)
    finally:
        lg.removeHandler(h); lg.setLevel(prev)
    assert np.all(np.isfinite(np.asarray(fit.particles)))
    msgs = [r.getMessage() for r in h.records]
    assert not any("not available for weight_mode" in m_ for m_ in msgs), msgs
    assert not any("non-JAX-native" in m_ for m_ in msgs), msgs
    assert not any("continuous-only" in m_ for m_ in msgs), msgs
    full = [s for s in calls if s > 0]
    assert len(full) >= 6, (
        f"binp exit succeeded {len(full)}/{len(calls)} times "
        f"(expected >= n_particles=6)")


def test_callback_multivariate_per_feature_engages():
    """The multivariate wrapper's per-feature 1-D slices route through
    the callback exact path automatically -- value vs central-diff (the
    hard pin is measured and added at the G4 fold per A/B precedent)."""
    g = _cb_graph()
    times2d = jnp.asarray([[0.5, 1.0], [1.0, 1.5], [1.5, 2.0]])
    rewards2d = jnp.asarray([[1.0, 0.0, 2.0, 0.5, 1.0],
                             [1.0, 1.0, 2.0, 1.0, 0.5]])
    with _capture_phasic_info_logs() as handler:
        # G4 fold: construction inside the capture (the dispatch-regression
        # blindness fix, as cell 3)
        m = Graph.pmf_and_moments_from_graph_multivariate(
            g, nr_moments=2, discrete=False, theta_dim=2)
        g_mv = np.asarray(jax.grad(
            lambda th: jnp.sum(m(th, times2d, rewards2d)[1]))(THETA))
    g_ref = _central_diff(m, THETA, times2d, rewards=rewards2d)
    assert np.all(np.isfinite(g_mv))
    # G4 fold: the HARD absolute pin (A/B precedent; measured on the C
    # build -- equals B's pin, same underlying linear weight function)
    np.testing.assert_allclose(
        g_mv, [-11.508732477887918, -5.563485415672725], rtol=1e-9)
    np.testing.assert_allclose(g_mv, g_ref, rtol=1e-5)
    messages = [r.getMessage() for r in handler.records]
    assert not any("finite differences" in m_ for m_ in messages), messages


def test_callback_non_finite_gradient_declines_per_theta():
    """G4 fold (the mandated-but-swapped G1(e) cell, run first by the
    refuter): a callback whose GRADIENT is non-finite at a theta the
    forward accepts (d/dt1 sqrt(t1) = inf at t1=0) -- the probe at ones
    passes, the per-theta J-finiteness net declines with the new message,
    and FD takes over finitely."""
    cb = lambda t, c: c[0] * t[0] + c[1] * jnp.sqrt(t[1])  # noqa: E731
    g = _cb_graph(cb)
    theta0 = jnp.asarray([1.0, 0.0])
    with _capture_phasic_info_logs() as handler:
        m = _model(g)
        grad = np.asarray(jax.grad(
            lambda th: jnp.sum(m(th, TIMES)[1]))(theta0))
    assert np.all(np.isfinite(grad))
    messages = [r.getMessage() for r in handler.records]
    assert any("non-finite callback gradient" in m_ for m_ in messages), \
        messages
    # away from the kink the exact path engages cleanly
    theta1 = jnp.asarray([1.0, 0.5])
    with _capture_phasic_info_logs() as handler2:
        grad1 = np.asarray(jax.grad(
            lambda th: jnp.sum(m(th, TIMES)[1]))(theta1))
    g_ref = _central_diff(m, theta1, TIMES)
    np.testing.assert_allclose(grad1, g_ref, rtol=1e-5)
    messages2 = [r.getMessage() for r in handler2.records]
    assert not any("finite differences" in m_ for m_ in messages2), messages2


def test_callback_canonical_set_param_length_class_engages():
    """G4 fold (D-C4 coverage gap): the CANONICAL decoupled pattern --
    set_param_length(P) before longer-coefficient edges -- engages the
    exact path end-to-end (the third theta-dimension class; lazy and
    plain-aligned are covered by cells 4 and 1)."""
    cb = lambda t, c: t[0] * c[0] + t[1] * c[1] + c[2] * 0.1  # noqa: E731
    g = Graph(1)
    g.set_param_length(2)
    s = g.starting_vertex()
    v = [g.find_or_create_vertex([i + 1]) for i in range(3)]
    va = g.find_or_create_vertex([9])
    s.add_edge(v[0], 1.0)
    for (a, b), c in zip([(v[0], v[1]), (v[1], v[2]), (v[2], va)],
                         [(1.0, 0.5, 2.0), (2.0, 0.25, 1.0),
                          (0.5, 1.5, 0.3)]):
        a.add_edge(b, list(c))
    g.weight_callback = cb
    assert g.param_length() == 2
    with _capture_phasic_info_logs() as handler:
        m = _model(g)
        grad = np.asarray(jax.grad(
            lambda th: jnp.sum(m(th, TIMES)[1]))(THETA))
    g_ref = _central_diff(m, THETA, TIMES)
    np.testing.assert_allclose(grad, g_ref, rtol=1e-5)
    messages = [r.getMessage() for r in handler.records]
    assert not any("finite differences" in m_ for m_ in messages), messages
