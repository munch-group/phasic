"""B3 formula-weight-mode exact gradient: Graph.pmf_and_moments_from_graph(...,
exact_moment_grad=True) on continuous weight_mode='formula' graphs (Batch B).

Extends the linear/log/dph exact reverse-mode theta-adjoint family to
weight_mode='formula' via ptd_moments_grad_theta_formula (src/c/phasic.c):
a Wengert-list reverse-mode autodiff pass over the weight-formula tape
(POW two-term adjoint; zero-propagate through comparisons/booleans/
select-condition; skip set == the primal's frozen edge set) feeding the
shared ptd_b3_moments_core contraction. See b3-batchB-plan.md v1+v2 for
the full design record and the two-refuter plan review that shaped it.

SCOPE (plan v2 SS-A/SS-B): continuous ALIGNED graphs only -- the model's
theta dimension must equal the C graph's param_length (formula is the
only weight mode where they legally decouple; the lazily-built decoupled
class statically declines to FD with a log, tested here). Discrete x
formula declines statically (tested); callback mode remains out of scope
(Batch C).

Every value assertion is anchored to an INDEPENDENT oracle: the manual
central-difference of the model's own forward, or the LINEAR exact path
on a twin graph computing the identical weight function (FD-independent
-- the mixed-scale cell's oracle, where FD itself is the defect being
fixed). Engagement is asserted POSITIVELY via spies on the C entry point
paired with LIVE log-text filters (the A-G4 discipline: never grep a
deleted string).
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
COEFFS = [(1.0, 0.5), (2.0, 0.25), (0.5, 1.5)]


def _chain(coeffs_list, param_length=None):
    g = Graph(1)
    if param_length is not None:
        g.set_param_length(param_length)
    s = g.starting_vertex()
    v = [g.find_or_create_vertex([i + 1]) for i in range(3)]
    va = g.find_or_create_vertex([9])
    s.add_edge(v[0], 1.0)
    edges = [(v[0], v[1]), (v[1], v[2]), (v[2], va)]
    for (a, b), c in zip(edges, coeffs_list):
        a.add_edge(b, list(c))
    return g


def _formula_linear_equiv():
    """Formula computing EXACTLY the linear dot product -- its exact
    Jacobian must equal the linear kind's (the twin-oracle trick)."""
    g = _chain(COEFFS)
    g.weight_formula = "t0*c0 + t1*c1"
    return g


def _linear_twin():
    return _chain(COEFFS)


def _formula_pow():
    g = _chain([(1.0, 0.5, 2.0), (2.0, 0.25, 1.5), (0.5, 1.5, 3.0)],
               param_length=2)
    g.weight_formula = "(t0*c0)**c1 + t1*c2*0.1"
    return g


def _formula_texp():
    g = _chain([(1.2, 0.0), (0.8, 0.0), (1.5, 0.0)], param_length=2)
    g.weight_formula = "c0 * t0**t1 + 0.2"
    return g


def _formula_select():
    g = _chain([(2.0, 1.0), (1.0, 2.0), (3.0, 0.5)])
    g.weight_formula = "select(c0 > 1.5, t0*c1, t1*c1)"
    return g


def _formula_lazy_decoupled():
    """The lazy-build class: coeff length 2 locks C param_length=2, the
    formula uses only t0 -> model theta_dim resolves to 1 != 2."""
    g = _chain(COEFFS)
    g.weight_formula = "t0*c0 + c1"
    return g


TIMES = jnp.asarray([0.5, 1.0, 1.5])
THETA = jnp.asarray([1.1, 0.9])


def _central_diff(model, theta, times, rewards=None, eps=1e-6):
    theta = np.asarray(theta, dtype=float)
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        tp = theta.copy(); tp[i] += eps
        tm = theta.copy(); tm[i] -= eps
        args_p = (jnp.asarray(tp), times) + (() if rewards is None else (rewards,))
        args_m = (jnp.asarray(tm), times) + (() if rewards is None else (rewards,))
        grad[i] = float(jnp.sum(model(*args_p)[1]) - jnp.sum(model(*args_m)[1])) / (2 * eps)
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


def _model(g, **kw):
    kw.setdefault("nr_moments", 2)
    kw.setdefault("discrete", False)
    kw.setdefault("theta_dim", 2)
    return Graph.pmf_and_moments_from_graph(g, **kw)


# --------------------------------------------------------------------------- cells
def test_formula_matches_linear_twin_and_central_diff(monkeypatch):
    """The linear-equivalent formula's exact gradient must equal (1) the
    LINEAR exact path on the twin graph (FD-independent oracle) and
    (2) the central-diff of its own forward. Engagement asserted via a
    spy on the formula C entry (measured: 1 call per grad, full-size
    return)."""
    calls = []
    orig = Graph._moments_grad_theta_formula

    def spy(self, K, *a, **kw):
        result = orig(self, K, *a, **kw)
        calls.append(np.asarray(result).size)
        return result

    monkeypatch.setattr(Graph, "_moments_grad_theta_formula", spy)
    mf = _model(_formula_linear_equiv())
    ml = _model(_linear_twin())
    with _capture_phasic_info_logs() as handler:
        gf = np.asarray(jax.grad(lambda th: jnp.sum(mf(th, TIMES)[1]))(THETA))
        gl = np.asarray(jax.grad(lambda th: jnp.sum(ml(th, TIMES)[1]))(THETA))
    g_ref = _central_diff(mf, THETA, TIMES)
    # twin parity: both exact paths differentiate the identical weight
    # function -- measured identical to fp round-off (gate (b1): rel 0.0
    # at the WRAPPER level; model level adds identical plumbing)
    np.testing.assert_allclose(gf, gl, rtol=1e-12)
    np.testing.assert_allclose(gf, g_ref, rtol=1e-6)
    messages = [r.getMessage() for r in handler.records]
    assert not any("finite differences" in m for m in messages), messages
    assert len(calls) >= 1 and all(s == 4 for s in calls), calls


def test_formula_mixed_scale_exact_vs_linear_twin():
    """The MOTIVATING cell (D-B3 / plan v2 SS-G): at theta=[1, 1e-8] FD is
    the defect being fixed, so the oracle is the LINEAR exact path on the
    twin graph -- gate (b1) measured rel EXACTLY 0.0 at the wrapper level
    at this very theta."""
    theta = jnp.asarray([1.0, 1e-8])
    mf = _model(_formula_linear_equiv())
    ml = _model(_linear_twin())
    gf = np.asarray(jax.grad(lambda th: jnp.sum(mf(th, TIMES)[1]))(theta))
    gl = np.asarray(jax.grad(lambda th: jnp.sum(ml(th, TIMES)[1]))(theta))
    assert np.all(np.isfinite(gf))
    np.testing.assert_allclose(gf, gl, rtol=1e-12)


def test_formula_pow_theta_in_exponent_matches_central_diff():
    m = _model(_formula_texp())
    theta = jnp.asarray([1.4, 0.6])
    with _capture_phasic_info_logs() as handler:
        g = np.asarray(jax.grad(lambda th: jnp.sum(m(th, TIMES)[1]))(theta))
    g_ref = _central_diff(m, theta, TIMES)
    np.testing.assert_allclose(g, g_ref, rtol=1e-5)
    messages = [r.getMessage() for r in handler.records]
    assert not any("finite differences" in m_ for m_ in messages), messages


def test_formula_pow_mix_matches_central_diff():
    m = _model(_formula_pow())
    theta = jnp.asarray([1.3, 0.7])
    g = np.asarray(jax.grad(lambda th: jnp.sum(m(th, TIMES)[1]))(theta))
    g_ref = _central_diff(m, theta, TIMES)
    np.testing.assert_allclose(g, g_ref, rtol=1e-5)


def test_formula_select_matches_central_diff():
    """select() with edges resolving to BOTH branches (c0 in {2,1,3} vs
    the 1.5 threshold); finite-gradient arms only (an unchosen arm with
    an infinite local derivative lawfully NaNs -> decline, the JAX
    convention -- plan v2 SS-D)."""
    m = _model(_formula_select())
    theta = jnp.asarray([1.1, 0.9])
    with _capture_phasic_info_logs() as handler:
        g = np.asarray(jax.grad(lambda th: jnp.sum(m(th, TIMES)[1]))(theta))
    g_ref = _central_diff(m, theta, TIMES)
    np.testing.assert_allclose(g, g_ref, rtol=1e-5)
    messages = [r.getMessage() for r in handler.records]
    assert not any("finite differences" in m_ for m_ in messages), messages


def test_formula_rewards_engage_all_ones_bitwise(monkeypatch):
    """1-D rewards thread through the formula exact path (inherited via
    the shared core's hooks -- gated, not assumed): mixed rewards match
    central-diff; all-ones == rewardless BITWISE (the engagement guard:
    an FD fallback differs at ~1e-9)."""
    calls = []
    orig = Graph._moments_grad_theta_formula

    def spy(self, K, *a, **kw):
        rw = kw.get("rewards")
        result = orig(self, K, *a, **kw)
        calls.append((0 if rw is None else len(rw), np.asarray(result).size))
        return result

    monkeypatch.setattr(Graph, "_moments_grad_theta_formula", spy)
    rewards = jnp.asarray([1.0, 0.0, 2.0, 0.5, 1.0])
    m = _model(_formula_linear_equiv())
    n = 5
    assert _formula_linear_equiv().vertices_length() == n
    with _capture_phasic_info_logs() as handler:
        g_rw = np.asarray(jax.grad(
            lambda th: jnp.sum(m(th, TIMES, rewards)[1]))(THETA))
    g_ref = _central_diff(m, THETA, TIMES, rewards=rewards)
    np.testing.assert_allclose(g_rw, g_ref, rtol=1e-5)
    messages = [r.getMessage() for r in handler.records]
    assert not any("finite differences" in m_ for m_ in messages), messages
    assert any(rw_len == n and size == 4 for rw_len, size in calls), calls

    m2 = _model(_formula_linear_equiv())
    g_ones = np.asarray(jax.grad(
        lambda th: jnp.sum(m2(th, TIMES, jnp.ones(n))[1]))(THETA))
    m3 = _model(_formula_linear_equiv())
    g_no = np.asarray(jax.grad(
        lambda th: jnp.sum(m3(th, TIMES)[1]))(THETA))
    np.testing.assert_array_equal(g_ones, g_no)


def test_formula_discrete_declines_with_log():
    """discrete x formula: static decline, truthful message, FD stays
    correct (finite gradient)."""
    g = _formula_linear_equiv()
    theta = jnp.asarray([0.3, 0.4])
    jumps = jnp.asarray([1.0, 2.0])
    # the static decline logs at model CONSTRUCTION time -- build inside
    # the capture (first-run lesson: built outside, zero records)
    with _capture_phasic_info_logs() as handler:
        m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=True,
                                             theta_dim=2)
        grad = np.asarray(jax.grad(
            lambda th: jnp.sum(m(th, jumps)[1]))(theta))
    assert np.all(np.isfinite(grad))
    messages = [r.getMessage() for r in handler.records]
    assert any("formula" in m_ and "continuous-only" in m_
               for m_ in messages), messages


def test_formula_lazy_decoupled_declines_with_log():
    """The lazy-build decoupled class (plan v2 SS-A, D-B6): C param_length
    2, model theta_dim 1 -- static decline naming param_length, FD still
    works, NO crash (pre-B this class simply used FD; the exact path's
    clone update would RAISE inside pure_callback)."""
    g = _formula_lazy_decoupled()
    with _capture_phasic_info_logs() as handler:
        m = Graph.pmf_and_moments_from_graph(g, nr_moments=2,
                                             discrete=False, theta_dim=1)
        theta = jnp.asarray([0.7])
        grad = np.asarray(jax.grad(
            lambda th: jnp.sum(m(th, TIMES)[1]))(theta))
    assert np.all(np.isfinite(grad))
    messages = [r.getMessage() for r in handler.records]
    assert any("param_length" in m_ and "theta dimension" in m_
               for m_ in messages), messages


def test_formula_out_of_scope_log_gone(monkeypatch):
    """The engagement flip (plan SS-6 cell 7): pre-B, formula + exact True
    emitted the generic out-of-scope decline; post-B it must NOT (live
    filter on the CURRENT message text), and the C entry must actually
    run (spy)."""
    calls = []
    orig = Graph._moments_grad_theta_formula

    def spy(self, K, *a, **kw):
        result = orig(self, K, *a, **kw)
        calls.append(1)
        return result

    monkeypatch.setattr(Graph, "_moments_grad_theta_formula", spy)
    m = _model(_formula_linear_equiv(), exact_moment_grad=True)
    with _capture_phasic_info_logs() as handler:
        g = np.asarray(jax.grad(lambda th: jnp.sum(m(th, TIMES)[1]))(THETA))
    assert np.all(np.isfinite(g))
    messages = [r.getMessage() for r in handler.records]
    assert not any("not available for weight_mode" in m_
                   for m_ in messages), messages
    assert len(calls) >= 1


def test_formula_vmap_jit_composition():
    """SVGD's actual calling pattern: vmap(grad) == per-particle grads
    BITWISE; jit(vmap(grad)) == vmap(grad) bitwise (the A-batch measured
    precedent for the shared callback plumbing)."""
    m = _model(_formula_linear_equiv())
    loss = lambda th: jnp.sum(m(th, TIMES)[1])  # noqa: E731
    particles = jnp.asarray([[1.1, 0.9], [0.7, 1.3], [1.5, 0.5]])
    g_vmap = np.asarray(jax.vmap(jax.grad(loss))(particles))
    g_per = np.stack([np.asarray(jax.grad(loss)(p)) for p in particles])
    g_jit = np.asarray(jax.jit(jax.vmap(jax.grad(loss)))(particles))
    np.testing.assert_array_equal(g_vmap, g_per)
    np.testing.assert_array_equal(g_jit, g_vmap)


def test_formula_svgd_frontdoor_default_runs_exact(monkeypatch):
    """Front-door svgd on an ALIGNED formula graph, default kwarg (the E
    lesson: real particle-init distribution): fit completes finite, the
    formula C entry runs >= n_particles times, no formula static-decline
    logs (per-theta conditioning declines, if any, are allowed and
    surface through the standard NaN->FD path)."""
    calls = []
    orig = Graph._moments_grad_theta_formula

    def spy(self, K, *a, **kw):
        result = orig(self, K, *a, **kw)
        # record the RESULT SIZE, not just the call (G4 fold B-G4-8: a
        # call-count floor alone stays green under an always-decline
        # regression; full-size results prove the exact path SUCCEEDED)
        calls.append(np.asarray(result).size)
        return result

    monkeypatch.setattr(Graph, "_moments_grad_theta_formula", spy)
    g = _formula_linear_equiv()
    lg = logging.getLogger("phasic")
    h = _CollectingHandler(); h.setLevel(logging.INFO)
    prev = lg.level
    lg.addHandler(h); lg.setLevel(logging.INFO)
    try:
        fit = g.svgd(np.asarray([0.5, 1.0, 1.5, 2.0]),
                     n_iterations=2, n_particles=6)
    finally:
        lg.removeHandler(h); lg.setLevel(prev)
    assert np.all(np.isfinite(np.asarray(fit.particles)))
    msgs = [r.getMessage() for r in h.records]
    assert not any("not available for weight_mode" in m_ for m_ in msgs), msgs
    assert not any("continuous-only" in m_ for m_ in msgs), msgs
    assert not any("theta dimension" in m_ for m_ in msgs), msgs
    full = [s for s in calls if s == 4]  # K=2 x P=2 full-size successes
    assert len(full) >= 6, (
        f"formula C entry SUCCEEDED {len(full)}/{len(calls)} times "
        f"(expected >= n_particles=6)")


def test_formula_multivariate_per_feature_engages():
    """The multivariate wrapper's per-feature 1-D slices route through the
    formula exact path automatically (the Batch-A mechanism, now for
    formula graphs) -- value-pinned vs central-diff."""
    g = _formula_linear_equiv()
    m = Graph.pmf_and_moments_from_graph_multivariate(
        g, nr_moments=2, discrete=False, theta_dim=2)
    times2d = jnp.asarray([[0.5, 1.0], [1.0, 1.5], [1.5, 2.0]])
    rewards2d = jnp.asarray([[1.0, 0.0, 2.0, 0.5, 1.0],
                             [1.0, 1.0, 2.0, 1.0, 0.5]])
    with _capture_phasic_info_logs() as handler:
        g_mv = np.asarray(jax.grad(
            lambda th: jnp.sum(m(th, times2d, rewards2d)[1]))(THETA))
    g_ref = _central_diff(m, THETA, times2d, rewards=rewards2d)
    assert np.all(np.isfinite(g_mv))
    # G4 fold (B-G4-4): the HARD absolute pin (A's precedent) -- a forward
    # regression shifting both model and central-diff passes the relative
    # check but not this. Measured 2026-08-14 on the B build.
    np.testing.assert_allclose(
        g_mv, [-11.508732477887918, -5.563485415672725], rtol=1e-9)
    np.testing.assert_allclose(g_mv, g_ref, rtol=1e-5)
    messages = [r.getMessage() for r in handler.records]
    assert not any("finite differences" in m_ for m_ in messages), messages


def test_formula_div_exp_log_matches_central_diff():
    """G4 fold (B-G4-2): the DIV quotient rule and EXP/LOG chain rules had
    zero fixture coverage anywhere (the plan's op-list mandate) -- one
    formula exercising all three, vs central-diff. (The G4 wiring refuter
    probed these rules against an independent jax oracle at <=3.3e-15;
    this cell is the durable regression guard.)"""
    g = _chain(COEFFS)
    g.weight_formula = "c0*t0/(t1 + 0.5) + exp(t0*0.1)*0.2 + log(t1*c1 + 1.5)"
    m = _model(g)
    theta = jnp.asarray([1.2, 0.8])
    with _capture_phasic_info_logs() as handler:
        grad = np.asarray(jax.grad(lambda th: jnp.sum(m(th, TIMES)[1]))(theta))
    g_ref = _central_diff(m, theta, TIMES)
    np.testing.assert_allclose(grad, g_ref, rtol=1e-5)
    messages = [r.getMessage() for r in handler.records]
    assert not any("finite differences" in m_ for m_ in messages), messages


def test_formula_aux_constant_edge_matches_central_diff():
    """G4 fold (B-G4-1, the v2 SS-E pytest half): an add_aux_vertex_constant
    edge is a tape input the contraction MUST skip -- for formula the skip
    is correctness-load-bearing (an unskipped constant edge contributes a
    WRONG-but-FINITE term the isfinite sweep cannot catch; the formula
    here uses no c<j> beyond c0/c1 so the tape would evaluate fine on it).
    FD-of-the-forward discriminates: FD never moves the constant edge."""
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [1.0, 0.5])
    v2.add_aux_vertex_constant(0.7)
    g.weight_formula = "t0*c0 + t1*c1"
    m = _model(g)
    theta = jnp.asarray([1.2, 0.8])
    grad = np.asarray(jax.grad(lambda th: jnp.sum(m(th, TIMES)[1]))(theta))
    g_ref = _central_diff(m, theta, TIMES)
    np.testing.assert_allclose(grad, g_ref, rtol=1e-5)


def test_formula_pow_domain_edges():
    """G4 fold (B-G4-3, re-encoding plan v2 SS-C's probe cases): the POW
    two-term adjoint at its domain edges. (a) (t0-3)**2 at t0=3: the
    factored v*(b/a) form would NaN; the two-term rule gives an exact
    ZERO gradient. (b) a<0 with integer exponent: finite and correct
    (the ln(a) NaN dies at the theta-independent exponent leaf). (c) a
    theta-independent NON-LEAF exponent (2.0*c0): same quarantine through
    an intermediate MUL node."""
    # (a) exact zero at the parabola vertex
    ga = Graph(1)
    s = ga.starting_vertex()
    v2 = ga.find_or_create_vertex([2])
    v1 = ga.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [1.0])
    ga.weight_formula = "(t0 - 3.0)**2 + 1.0"
    ma = _model(ga, theta_dim=1)
    grad_a = np.asarray(jax.grad(
        lambda th: jnp.sum(ma(th, TIMES)[1]))(jnp.asarray([3.0])))
    np.testing.assert_array_equal(grad_a, [0.0])

    # (b) negative base, integer constant exponent
    gb = Graph(1)
    s = gb.starting_vertex()
    v2 = gb.find_or_create_vertex([2])
    v1 = gb.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [1.0])
    gb.weight_formula = "t0**2 + 10.0"
    mb = _model(gb, theta_dim=1)
    theta_b = jnp.asarray([-2.0])
    grad_b = np.asarray(jax.grad(
        lambda th: jnp.sum(mb(th, TIMES)[1]))(theta_b))
    ref_b = _central_diff(mb, theta_b, TIMES)
    assert np.all(np.isfinite(grad_b))
    np.testing.assert_allclose(grad_b, ref_b, rtol=1e-5)

    # (c) theta-independent non-leaf exponent, negative base
    gc = Graph(1)
    s = gc.starting_vertex()
    v2 = gc.find_or_create_vertex([2])
    v1 = gc.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [1.0])
    gc.weight_formula = "t0**(2.0*c0) + 10.0"
    mc = _model(gc, theta_dim=1)
    grad_c = np.asarray(jax.grad(
        lambda th: jnp.sum(mc(th, TIMES)[1]))(theta_b))
    ref_c = _central_diff(mc, theta_b, TIMES)
    assert np.all(np.isfinite(grad_c))
    np.testing.assert_allclose(grad_c, ref_c, rtol=1e-5)
