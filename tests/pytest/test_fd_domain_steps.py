"""Plan A (domain-aware finite-difference steps), Batch 1 tests.

plans-and-findings/fd-domain-steps-plan.md v2.10 §6 Batch 1; findings in
plans-and-findings/fd-domain-steps-findings.md. The FD step for declared
domains is s = 3e-5 (findings §4.3).

Sections:
 1. bit identity: undeclared slots outside the §2 band keep master's
    gradients bit for bit, per site, under eager / jit / vmap(grad);
 2. the probe arithmetic of _fd_steps (rules, cap, one-sided step, unit,
    validation, the joint-index host variant);
 3. the crash fallback (§4.4) on every refusing site and route;
 4. accuracy of declared domains (closed forms / reference differences);
 5. fixed and tied slots, and the probe-recording spy;
 6. degenerate values (§4.5): theta == 0, zero probe difference, ties,
    out-of-domain values;
 7. signed parameters, the discrete row-sum wall, domain plumbing.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import phasic  # noqa: E402,F401
from phasic import Graph, _fd_steps  # noqa: E402

import _fd_golden_cells as C  # noqa: E402

TIMES = jnp.asarray(C.TIMES)


# ===================================================================== 1
with open(C.GOLDEN_PATH) as _fh:
    _GOLD = json.load(_fh)
_PARAMS = [(name, tuple(theta), mode) for name, theta in C.CELLS for mode in C.modes_for(name)]


@pytest.mark.skipif(_GOLD["platform"] != C.platform_tag(),
                    reason="goldens were captured on " + _GOLD["platform"])
@pytest.mark.parametrize("name,theta,mode", _PARAMS,
                         ids=[f"{n}-{','.join(map(str, t))}-{m}" for n, t, m in _PARAMS])
def test_undeclared_gradients_are_bit_identical_to_master(name, theta, mode):
    """§2 goal 2 / §4.7: undeclared slots keep today's numbers, compared with
    master in the SAME mode (XLA rewrites x/(2 eps) under jit)."""
    want = np.array([float.fromhex(h) for h in _GOLD["grads"][C.key(name, theta, mode)]])
    got = np.ravel(C.grad_in_mode(name, list(theta), mode))
    if name in C.NON_REPRODUCIBLE:
        # master itself is not reproducible run to run here (OI-022)
        np.testing.assert_allclose(got, want, rtol=C.NON_REPRODUCIBLE_RTOL, atol=0)
    else:
        assert got.tobytes() == want.tobytes(), (got, want, got - want)


@pytest.mark.parametrize("mode", ["eager", "jitvmap"])
@pytest.mark.parametrize("name,theta", C.CELLS, ids=[f"{n}-{','.join(map(str, t))}" for n, t in C.CELLS])
def test_undeclared_gradients_equal_an_emulated_master(name, theta, mode, monkeypatch):
    """Platform-independent form of goal 2 (review m5): with every undeclared
    slot forced to today's rule -- master's traced program -- the gradient
    equals the real rules' outside the band, in the same process, including
    SVGD's jit(vmap(grad)) mode."""
    if name == "cpp" and mode == "jitvmap":
        pytest.skip("pmf_from_cpp cannot be vmapped (OI-123)")
    real = np.ravel(C.grad_in_mode(name, list(theta), mode))
    orig = _fd_steps.slot_rule
    monkeypatch.setattr(_fd_steps, "slot_rule",
                        lambda d, refuses: _fd_steps.TODAY if d is None else orig(d, refuses))
    master = np.ravel(C.grad_in_mode(name, list(theta), mode))
    if name in C.NON_REPRODUCIBLE:
        np.testing.assert_allclose(real, master, rtol=C.NON_REPRODUCIBLE_RTOL, atol=0)
    else:
        assert real.tobytes() == master.tobytes(), (real, master)


# ===================================================================== 2
def _probe(t, rule, family="capped", log_mode=False):
    th = jnp.asarray([0.5, t], dtype=jnp.float64)
    info = _fd_steps.BuildInfo(True, log_mode, family)
    tp, tm, use = _fd_steps.probes(th, 1, th.at[1].add(1e-7), th.at[1].add(-1e-7), rule, info)
    return float(tp[1]), float(tm[1]), (None if use is None else bool(use))


@pytest.mark.parametrize("t", [1e-300, 1e-13, 1e-12, 1e-9, 5e-8, 1e-7, 1e-3, 1.0, 1e6, 1e9])
def test_positive_probes_stay_inside_the_domain(t):
    for fam in ("capped", "full"):
        up, dn, use = _probe(t, _fd_steps.POSITIVE, family=fam)
        # the capped up-probe rounds onto theta from ~9e8 (findings §5)
        assert use is True and (up > t or (fam == "capped" and t > 9e8 and up == t)) and t > dn > 0
        if fam == "capped":
            # at most 1e-7 above theta, up to 2 ulp of rounding (plan v2.10 §4.2)
            assert up - t <= 1e-7 + 2 * np.spacing(t)
        else:
            assert up / t == pytest.approx(np.exp(_fd_steps.S), rel=1e-15)


@pytest.mark.parametrize("t", [1e-12, 1e-9, 0.3, 1 - 1e-9, 1 - 1e-12])
def test_unit_probes_stay_inside_zero_one(t):
    for fam in ("capped", "full"):
        up, dn, use = _probe(t, _fd_steps.UNIT, family=fam)
        assert use is True and 0 < dn < up < 1


def test_rule_selection():
    B, P = _fd_steps.BAND, _fd_steps.POSITIVE
    # undeclared on a refusing model: the capped positive rule exactly in [0, 1e-7]
    assert _probe(5e-8, B)[2] is True and _probe(1e-7, B)[2] is True
    assert _probe(2e-7, B)[2] is False and _probe(-1e-3, B)[2] is False
    assert _probe(0.0, B) == (1e-7, 0.0, True)                   # one-sided [0, 1e-7]
    assert _probe(0.0, B, log_mode=True)[2] is False             # log mode keeps today's error
    assert _probe(2e-7, B)[:2] == (2e-7 + 1e-7, 2e-7 - 1e-7)     # today's probes, unchanged
    # 'unconstrained': plain step; the band only on refusing models
    assert _probe(-5.0, _fd_steps.UNC_BAND) == (-5.0 + 1e-7, -5.0 - 1e-7, False)
    assert _probe(5e-8, _fd_steps.UNC_BAND)[2] is True
    up, dn, use = _probe(5e-8, _fd_steps.PLAIN)
    assert use is None and dn < 0                                # non-refusing may go negative
    # declared domains: theta == 0 one-sided, outside the range today's probe
    assert _probe(0.0, P) == (1e-7, 0.0, True)
    assert _probe(-1.0, P)[2] is False
    assert _probe(1.5, _fd_steps.UNIT)[2] is False and _probe(0.0, _fd_steps.UNIT)[2] is False
    # today's rule returns the site's own arrays (no traced selection at all)
    th = jnp.asarray([0.5, 0.3])
    a, b = th.at[1].add(1e-7), th.at[1].add(-1e-7)
    tp, tm, use = _fd_steps.probes(th, 1, a, b, _fd_steps.TODAY, _fd_steps.BuildInfo(False, False, "capped"))
    assert tp is a and tm is b and use is None


def test_slot_rules_validation_and_refusal():
    R = _fd_steps.slot_rule
    assert R(None, False) == _fd_steps.TODAY and R(None, True) == _fd_steps.BAND
    assert R("unconstrained", False) == _fd_steps.PLAIN
    assert R("unconstrained", True) == _fd_steps.UNC_BAND
    assert R("fixed", True) == _fd_steps.SKIP and R("unit", True) == "unit"
    assert _fd_steps.normalize_domains(None) is None
    assert _fd_steps.normalize_domains(["positive", None]) == ("positive", None)
    with pytest.raises(TypeError):
        _fd_steps.normalize_domains("positive")
    with pytest.raises(ValueError):
        _fd_steps.normalize_domains(["positiv"])
    with pytest.raises(ValueError, match="param_domains has 1 entries but theta has 2"):
        _fd_steps.check_length(("positive",), 2, "x")
    lin = _fd_steps.build_info({"weight_mode": "linear"})
    assert not _fd_steps.model_refuses(lin) and _fd_steps.model_refuses(lin, has_rewards=True)
    for s in ({"weight_mode": "log"}, {"weight_mode": "formula"}, {"weight_mode": "callback"},
              {"weight_mode": "linear", "is_discrete": True},
              {"weight_mode": "linear", "was_dph": True}):
        assert _fd_steps.model_refuses(_fd_steps.build_info(s)), s
    assert _fd_steps.model_refuses(_fd_steps.build_info(None, always_refuses=True))


def test_param_domains_length_is_checked_at_first_trace():
    m = Graph.pmf_from_graph(C.beside(), theta_dim=2, param_domains=["positive"])
    with pytest.raises(ValueError, match="param_domains has 1 entries"):
        jax.grad(lambda t: jnp.sum(m(t, TIMES)))(jnp.asarray([0.7, 1.3]))


def test_builders_attach_their_domains_and_refusal():
    d = ["positive", "fixed"]
    pam = Graph.pmf_and_moments_from_graph(C.beside(), theta_dim=2, param_domains=d)
    assert pam._param_domains == ("positive", "fixed") and not pam._fd_build_info.refuses
    mv = Graph.pmf_and_moments_from_graph_multivariate(C.beside(), theta_dim=2, param_domains=d)
    assert mv._param_domains == ("positive", "fixed")             # copied onto the wrapper
    assert Graph.pmf_from_graph(C.beside("log"), theta_dim=2)._fd_build_info.refuses
    assert Graph.pmf_from_graph(C.beside(), discrete=True, theta_dim=2)._fd_build_info.refuses
    assert not Graph.pmf_from_graph(C.beside(), theta_dim=2)._fd_build_info.refuses
    C._ensure_source_dir()
    # plan v2.10: moments_from_graph always refuses (generated C++)
    assert Graph.moments_from_graph(C.beside(), nr_moments=2)._fd_build_info.refuses


def test_host_probes_numpy_variant():
    info = _fd_steps.BuildInfo(True, False, "full")
    th = np.array([0.5, 0.3])
    h = max(abs(th[1]) * 1e-6, 1e-10)
    tp, tm, div, alt = _fd_steps.host_probes_np(th, 1, h, _fd_steps.BAND, info)
    assert not alt and (tp[1], tm[1], div) == (0.3 + h, 0.3 - h, 2 * h)   # today's relative step
    th0 = np.array([0.5, 0.0])
    tp, tm, div, alt = _fd_steps.host_probes_np(th0, 1, 1e-10, _fd_steps.BAND, info)
    assert alt and (tp[1], tm[1]) == (1e-7, 0.0)                          # refusing band at 0
    tp, tm, div, alt = _fd_steps.host_probes_np(th0, 1, 1e-10, _fd_steps.TODAY, info)
    assert not alt and tm[1] == -1e-10                                    # non-refusing: today's
    th5 = np.array([0.5, 5e-8])
    assert not _fd_steps.host_probes_np(th5, 1, 1e-10, _fd_steps.BAND, info)[3]  # band is [0, 1e-10)
    tp, tm, div, alt = _fd_steps.host_probes_np(th5, 1, 1e-10, _fd_steps.POSITIVE, info)
    assert alt and tp[1] / 5e-8 == pytest.approx(np.exp(3e-5)) and tm[1] > 0


# ===================================================================== 3
def _beside_model(builder, mode, disc, rewards=False, domains=None):
    """(forward, loss) on the 'beside' chain: slot 1 is an exit beside
    another, so slot 1 = 0 is a valid model."""
    g = C.beside(mode)
    x = jnp.asarray(C.JUMPS) if disc else TIMES
    rw = jnp.asarray(C.REWARDS)
    if builder == "pfg":
        m = Graph.pmf_from_graph(g, discrete=disc, theta_dim=2, param_domains=domains)
        f = lambda th: m(th, x)
    elif builder == "pam":
        # exact_moment_grad=False: the FD loop produces the whole gradient, so
        # the value checks see it (with the default the exact moments term
        # dominates the loss in log mode; review m5)
        m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=disc, theta_dim=2,
                                             param_domains=domains, exact_moment_grad=False)
        f = (lambda th: m(th, x, rewards=rw)) if rewards else (lambda th: m(th, x))
    elif builder == "rvp":
        from phasic.ffi_wrappers import compute_reward_visit_probability_ffi
        # reward only vertex b, visited with probability t0 / (t0 + t1): a
        # nonzero gradient in slot 1 (with a, always visited, p would be 1)
        tgt = jnp.asarray([2], dtype=jnp.int32)
        ipv = jnp.asarray(g._initial_probability_vector())
        sj = g.serialize()
        f = lambda th: compute_reward_visit_probability_ffi(sj, th, tgt, ipv, param_domains=domains)
    elif builder == "mfg":
        C._ensure_source_dir()
        f = Graph.moments_from_graph(g, nr_moments=2, param_domains=domains)
    else:
        raise KeyError(builder)
    return f, C.loss_of(f)


def _central(L, th, i, h):
    return float((L(th.at[i].add(h)) - L(th.at[i].add(-h))) / (2 * h))


def _forward_diff(L, th, i, h):
    return float((L(th.at[i].add(h)) - L(th)) / h)


_CRASH = []
for _mode in ("linear", "log", "formula", "callback"):
    for _disc in (False, True):
        for _b, _rw in (("pfg", False), ("pam", False), ("pam", True), ("rvp", False)):
            if _b == "rvp" and (_mode == "callback" or _disc):
                continue     # rvp: no callback graphs (OI-120)
            if _disc and _rw:
                continue
            _CRASH.append((_b, _mode, _disc, _rw))
_CRASH.append(("mfg", "linear", False, False))


@pytest.mark.parametrize("v", [5e-8, 0.0])
@pytest.mark.parametrize("builder,mode,disc,rw", _CRASH,
                         ids=[f"{b}-{m}-{'dph' if d else 'cont'}{'-rw' if r else ''}"
                              for b, m, d, r in _CRASH])
def test_no_crash_in_the_band_on_every_site(builder, mode, disc, rw, v):
    """§4.4: an undeclared slot at 5e-8 or 0 no longer makes a probe cross zero
    on any refusing site; log mode at exactly 0 keeps today's primal error
    (§4.5). The value matches a reference whose probes stay >= 0."""
    f, L = _beside_model(builder, mode, disc, rewards=rw)
    th = jnp.asarray([0.3 if disc else 0.7, v])
    if mode == "log" and v == 0.0:
        with pytest.raises(Exception, match="positive"):
            np.asarray(jax.grad(L)(th))
        return
    g = np.asarray(jax.grad(L)(th))
    assert np.all(np.isfinite(g)), g
    ref = _central(L, th, 1, 1e-9) if v else _forward_diff(L, th, 1, 1e-9)
    assert g[1] == pytest.approx(ref, rel=1e-3, abs=1e-6)
    if not (builder == "rvp" and mode == "log"):
        # (log mode: the visit probability is t0t1 / (t0t1 + 2t0t1) = 1/3 for
        # every theta, so a zero gradient is right there)
        assert abs(ref) > 1e-3, "the reference must not be trivially 0"


def test_no_crash_on_pmf_from_cpp_in_the_band():
    L = C.loss_of(C.build("cpp"))
    for v in (5e-8, 0.0):
        th = jnp.asarray([0.7, v])
        g = np.asarray(jax.grad(L)(th))
        assert np.all(np.isfinite(g))
        ref = _central(L, th, 1, 1e-9) if v else _forward_diff(L, th, 1, 1e-9)
        assert g[1] == pytest.approx(ref, rel=1e-3, abs=1e-6)


@pytest.mark.parametrize("v", [5e-8, 0.0])
def test_no_crash_on_the_discretize_slot(v):
    m = Graph.pmf_from_graph(C.beside().discretize(1.0), discrete=True, theta_dim=3)
    L = C.loss_of(lambda th: m(th, jnp.asarray([1.0, 2.0, 4.0])))
    th = jnp.asarray([0.7, 1.3, v])
    g = np.asarray(jax.grad(L)(th))
    assert np.all(np.isfinite(g))
    ref = _central(L, th, 2, 1e-9) if v else _forward_diff(L, th, 2, 1e-9)
    assert g[2] == pytest.approx(ref, rel=1e-3, abs=1e-6)


def _richardson_log(L, th, i, s=1e-2):
    """d L / d theta_i by Richardson extrapolation of log-coordinate central
    differences (probes theta * exp(+-h), inside the positive domain)."""
    def d(h):
        return float(L(th.at[i].multiply(np.exp(h))) - L(th.at[i].multiply(np.exp(-h)))) / (2 * h)
    a, b, c = d(s), d(s / 2), d(s / 4)
    r1, r2 = (4 * b - a) / 3, (4 * c - b) / 3
    return ((16 * r2 - r1) / 15) / float(th[i])


_EPOCH_CASES = [[1.0, 1e-8, 1.0, 1e-8], [1.0, 1e-9, 1.0, 1e-9], [1e-9, 1e-4, 1e-4, 1e-4],
                [1.0, 0.0, 1.0, 0.5]]


@pytest.mark.parametrize("theta", _EPOCH_CASES, ids=["1e-8", "1e-9", "first-1e-9", "rate-0"])
@pytest.mark.parametrize("cell", ["dc_noexp", "dc_exp", "dc_efg", "dcjp"])
def test_epoch_model_band_gradients_are_finite_and_right(cell, theta):
    """§1: these thetas raised 'NEGATIVE transition rate' on master (also with
    exact_final_grad=True, whose earlier-epoch slots use FD); now finite, and
    each band slot matches a reference whose probes stay >= 0 (a rate fixed
    at 0, the nested likelihood-ratio model, uses the one-sided step)."""
    f = C.build(cell)
    L = lambda t: jnp.sum(jnp.log(jax.tree_util.tree_leaves(f(t))[0]))
    th = jnp.asarray(theta)
    g = np.asarray(jax.grad(L)(th))
    assert np.all(np.isfinite(g)), g
    for i, t in enumerate(theta):
        if 0.0 < t <= 1e-7:
            # Measured (batch1/m1_epoch_band_errors.log): the epoch models
            # (dc_*) agree with Richardson to ~1e-10 relative on the mutation
            # slots, and to 2-6e-3 on a coalescence rate at 1e-9 (an effect
            # linear near 0: the §4.3 limit). daisy_chain_joint_probs' loss
            # barely depends on the slot, so any step that keeps the probes
            # in [0, 2e-7] is noise-dominated there (up to 4x relative at
            # 1e-9, < 1e-9 in loss units in theta * d/dtheta): finite only.
            if cell == "dcjp":
                continue
            ref = _richardson_log(L, th, i)
            rel = 2e-2 if i == 0 else 1e-6
            assert g[i] == pytest.approx(ref, rel=rel), (i, g[i], ref)
        elif t == 0.0:
            # one-sided [0, 1e-7] step: first order; reference second order
            ref = 2 * _forward_diff(L, th, i, 5e-8) - _forward_diff(L, th, i, 1e-7)
            assert g[i] == pytest.approx(ref, rel=2e-2, abs=1e-6)


def test_epoch_model_band_under_vmap_matches_per_particle():
    f = C.build("dc_noexp")
    L = lambda t: jnp.sum(jnp.log(f(t)[0]))
    P = jnp.asarray([[1.0, 1e-8, 1.0, 1e-8], [1.0, 0.5, 1.5, 0.3]])
    G = np.asarray(jax.vmap(jax.grad(L))(P))
    assert np.all(np.isfinite(G))
    np.testing.assert_allclose(G[0], np.asarray(jax.grad(L)(P[0])), rtol=1e-12)


def _small_prior():
    from phasic import LogGaussPrior
    return LogGaussPrior(ci=[2e-9, 5e-8])


def test_graph_svgd_epoch_route_runs_with_particles_in_the_band():
    """The reported crash on the Graph.svgd route: particles around 1e-8.
    Batch 1 declares nothing, so this is the §4.4 fallback at work."""
    jpg, _ = C.joint_graphs()
    s = jpg.svgd(C.joint_rows(), epoch_starts=[0.0, 0.5],
                 prior=[_small_prior(), _small_prior()], n_particles=4, n_iterations=2,
                 learning_rate=1e-12, progress=False, seed=1)
    assert np.all(np.isfinite(np.asarray(s.theta_mean)))


def test_epoch_model_and_direct_svgd_routes_run_with_particles_in_the_band():
    from phasic import SVGD
    jpg, _ = C.joint_graphs()
    em = jpg.epoch_model(C.joint_rows(), epoch_starts=[0.0, 0.5])
    r = em.fit(prior=[_small_prior()] * 4, n_particles=4, n_iterations=2,
               learning_rate=1e-12, progress=False, seed=1)
    assert np.all(np.isfinite(np.asarray(r.theta_mean)))
    s = SVGD(model=em.model, observed_data=em.observed_data, theta_dim=4,
             prior=[_small_prior()] * 4, n_particles=4, n_iterations=2,
             learning_rate=1e-12, progress=False, seed=1)
    s.optimize()
    assert np.all(np.isfinite(np.asarray(s.theta_mean)))


def test_mixed_scale_pin_now_passes():
    """§2 goal 6 is asserted by inference/test_fd_gradient_mixed_scale.py
    (its strict xfail marker is removed in this batch); here the same class of
    model: a slot at 1e-8 on a refusing model gives a finite gradient."""
    L = C.loss_of(C.build("dc_noexp"))
    assert np.all(np.isfinite(np.asarray(jax.grad(L)(jnp.asarray([1.0, 1e-8, 1.0, 1e-8])))))


# ===================================================================== 4
def _three_slot_chain():
    """s -> v1; v1 -> v2 (t0); v1 -> a (t1); v2 -> a (t2): distinct partials."""
    g = Graph(1)
    s = g.starting_vertex()
    v1 = g.find_or_create_vertex([1]); v2 = g.find_or_create_vertex([2])
    a = g.find_or_create_vertex([3])
    s.add_edge(v1, 1.0)
    v1.add_edge(v2, [1.0, 0.0, 0.0]); v1.add_edge(a, [0.0, 1.0, 0.0]); v2.add_edge(a, [0.0, 0.0, 1.0])
    return g


def _closed_moments(th):
    t0, t1, t2 = th[0], th[1], th[2]
    r1 = t0 + t1
    p = t0 / r1
    m1 = 1.0 / r1 + p / t2
    m2 = 2.0 / r1 ** 2 + p * (2.0 / (r1 * t2) + 2.0 / t2 ** 2)
    return jnp.stack([m1, m2])


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e3])
def test_declared_positive_moments_gradient_matches_closed_form(scale):
    """Elimination family, full rule (s = 3e-5), on moments_from_graph (no
    fixed-slot argument: slot 1 is 'fixed' through its domain): the FD
    gradient matches the closed form in the domain coordinate."""
    C._ensure_source_dir()
    mfg = Graph.moments_from_graph(_three_slot_chain(), nr_moments=2,
                                   param_domains=["positive", "fixed", "positive"])
    th = jnp.asarray([1.0, 0.5, 2.0]) * scale
    w = 1.0 / np.asarray(_closed_moments(th))
    L = lambda t: jnp.sum(mfg(t) * w)
    Lc = lambda t: jnp.sum(_closed_moments(t) * w)
    g = np.asarray(jax.grad(L)(th)); gc = np.asarray(jax.grad(Lc)(th))
    assert g[1] == 0.0                                    # fixed: not probed, zero
    for i in (0, 2):
        assert th[i] * g[i] == pytest.approx(float(th[i] * gc[i]), rel=1e-6, abs=1e-9)
    # today's absolute step at 1e-6 is a 10% relative step: visibly worse
    if scale == 1e-6:
        m0 = Graph.moments_from_graph(_three_slot_chain(), nr_moments=2)
        g0 = np.asarray(jax.grad(lambda t: jnp.sum(m0(t) * w))(th))
        assert abs(th[0] * g0[0] - th[0] * gc[0]) > 100 * abs(th[0] * g[0] - th[0] * gc[0])


@pytest.mark.parametrize("scale", [1e-6, 1.0])
def test_declared_positive_moments_on_the_capped_site(scale):
    """pmf_and_moments_from_graph is a capped site (its pdf half has jumps);
    with exact_moment_grad=False its moments inherit the cap (§4.1). Where the
    cap does not shrink the relative step (theta <= ~3e-3) the moments match
    the closed form; at large theta the capped step is today's 1e-7 absolute
    step, with today's noise (§4.3)."""
    m = Graph.pmf_and_moments_from_graph(
        _three_slot_chain(), nr_moments=2, theta_dim=3, exact_moment_grad=False,
        param_domains=["positive", "fixed", "positive"])
    th = jnp.asarray([1.0, 0.5, 2.0]) * scale
    w = 1.0 / np.asarray(_closed_moments(th))
    L = lambda t: jnp.sum(m(t, jnp.asarray([1.0]))[1] * w)
    g = np.asarray(jax.grad(L)(th))
    gc = np.asarray(jax.grad(lambda t: jnp.sum(_closed_moments(t) * w))(th))
    assert g[1] == 0.0
    for i in (0, 2):
        assert th[i] * g[i] == pytest.approx(float(th[i] * gc[i]), rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("scale", [1.0, 1e3])
def test_declared_positive_pdf_gradient_matches_a_reference(scale):
    """Euler pdf family (capped rule): against a Richardson extrapolation of
    the SAME forward in the log coordinate. The point avoids an integer
    2 x max exit rate: above a max rate of 512 the granularity int(2 x max
    rate) jumps there, and any 1e-7 probe (today's or capped) straddles the
    jump (plan §9; Plan B removes it)."""
    m = Graph.pmf_from_graph(_three_slot_chain(), theta_dim=3,
                             param_domains=["positive", "fixed", "positive"])
    th = jnp.asarray([1.0137, 0.5071, 2.0293]) * scale
    x = TIMES / scale
    L = lambda t: jnp.sum(jnp.log(m(t, x)))
    g = np.asarray(jax.grad(L)(th))
    assert g[1] == 0.0

    def rich(i, s=1e-2):
        def d(h):
            return float(L(th.at[i].multiply(np.exp(h))) - L(th.at[i].multiply(np.exp(-h)))) / (2 * h)
        a, b, c = d(s), d(s / 2), d(s / 4)
        r1, r2 = (4 * b - a) / 3, (4 * c - b) / 3
        return (16 * r2 - r1) / 15
    if scale == 1.0:
        # max exit rate < 512: fixed granularity, a smooth forward
        for i in (0, 2):
            assert float(th[i]) * g[i] == pytest.approx(rich(i), rel=1e-5)
    else:
        # max exit rate ~2029 > 512: the automatic granularity int(2 x max
        # rate) jumps every ~2.5e-4 relative in theta (so no wide-step
        # reference of the automatic forward is valid), and at ~2 Euler steps
        # per holding time the forward itself is far from the closed form
        # (Plan B's territory). The capped step stays on ONE granularity
        # plateau: the reference is the slope of the same forward at that
        # fixed granularity.
        gc = _chain_graph_at(th)
        gran = int(2 * max(512.0, float(th[0] + th[1]), float(th[2])))

        def Lfix(t):
            gg = _chain_graph_at(t)
            return float(np.sum(np.log([gg.pdf(float(u), gran) for u in np.asarray(x)])))
        for i in (0, 2):
            def d(h):
                return (Lfix(th.at[i].multiply(np.exp(h))) - Lfix(th.at[i].multiply(np.exp(-h)))) / (2 * h)
            a, b, c = d(1e-6), d(5e-7), d(2.5e-7)
            r1, r2 = (4 * b - a) / 3, (4 * c - b) / 3
            ref = (16 * r2 - r1) / 15
            assert float(th[i]) * g[i] == pytest.approx(ref, rel=1e-4)
        del gc


def _chain_graph_at(th):
    g = _three_slot_chain()
    g.update_weights([float(v) for v in np.asarray(th)])
    return g


# ===================================================================== 5
@pytest.fixture
def probe_spy():
    rec = []
    _fd_steps.set_probe_hook(lambda site, i, tp, tm: rec.append(
        (site, i, np.atleast_2d(np.asarray(tp)).copy(), np.atleast_2d(np.asarray(tm)).copy())))
    try:
        yield rec
    finally:
        _fd_steps.set_probe_hook(None)


def test_probe_spy_heterogeneous_domains_across_epochs(probe_spy):
    """§6 Batch 1 spy: every probe vector the loop passes to the forward
    changes only slot i, lies inside that slot's domain; a 'fixed' slot is
    never probed; a tied slave IS probed (its partial flows to the master).
    Exposure model, 2 epochs x 2 slots: [unit, positive (exposure slot),
    unit (tied slave of slot 0), fixed]."""
    jpg, oi = C.joint_graphs()
    m, _, _, _ = jpg._daisy_chain_svgd_model(
        observed_indices=list(oi), epoch_starts=np.array([0.0, 0.5]),
        user_prior=None, user_fixed=None, user_tied=[(0, [0, 1])],
        exposure_arr=np.linspace(0.5, 2.0, len(oi)), exposure_param_index=1,
        param_domains=["unit", "positive", "unit", "fixed"])
    th = jnp.asarray([0.3, 0.5, 0.3, 0.4])
    g = np.asarray(jax.grad(lambda t: jnp.sum(jnp.log(m(t, None)[0])))(th))
    assert np.all(np.isfinite(g)) and g[3] == 0.0
    seen = {}
    for site, i, tp, tm in probe_spy:
        assert site == "epoch model"
        for v in (tp, tm):
            changed = np.nonzero(np.any(v != np.asarray(th)[None, :], axis=0))[0].tolist()
            assert changed == [i], (i, v)
            x = v[:, i]
            if i in (0, 2):
                assert np.all((0 < x) & (x < 1))
            else:
                assert np.all(x > 0)
        seen[i] = seen.get(i, 0) + 1
    assert sorted(seen) == [0, 1, 2]                      # 3 ('fixed') never probed


def test_tied_master_gradient_is_the_sum_of_per_epoch_partials():
    """Declared domains do not break tying: d/d(master) of the tied model
    equals the sum of the free model's per-epoch partials (extends
    test_daisy_chain_c_path's load-bearing check to declared domains)."""
    jpg, oi = C.joint_graphs()
    kw = dict(observed_indices=list(oi), epoch_starts=np.array([0.0, 0.5]),
              user_prior=None, user_fixed=None, param_domains=["positive"] * 4)
    m_free, _, _, _ = jpg._daisy_chain_svgd_model(**kw)
    m_tied, _, _, _ = jpg._daisy_chain_svgd_model(user_tied=[(0, [0, 1])], **kw)
    th = jnp.asarray([0.8, 0.5, 0.8, 0.3])
    L = lambda m: (lambda t: jnp.sum(jnp.log(m(t, None)[0])))
    gf = np.asarray(jax.grad(L(m_free))(th)); gt = np.asarray(jax.grad(L(m_tied))(th))
    assert gt[0] == pytest.approx(gf[0] + gf[2], rel=1e-12)
    assert gt[2] == 0.0


def test_fixed_domain_is_never_probed_on_sites_without_fixed_information(probe_spy):
    """§4.6: pmf_from_graph, moments_from_graph, reward_visit_probability have
    no fixed-slot argument; the 'fixed' domain value skips them."""
    for builder in ("pfg", "mfg", "rvp"):
        probe_spy.clear()
        f, L = _beside_model(builder, "linear", False, domains=[None, "fixed"])
        g = np.asarray(jax.grad(L)(jnp.asarray([0.7, 1.3])))
        assert g[1] == 0.0 and np.isfinite(g[0])
        assert {i for _, i, _, _ in probe_spy} == {0}, builder


def test_formula_and_log_modes_with_fixed_slots():
    for mode in ("formula", "log"):
        f, L = _beside_model("pam", mode, False, domains=["positive", "fixed"])
        g = np.asarray(jax.grad(L)(jnp.asarray([0.7, 1.3])))
        assert g[1] == 0.0 and np.isfinite(g[0])
        ref = float(0.7 * _central(L, jnp.asarray([0.7, 1.3]), 0, 1e-6))
        assert 0.7 * g[0] == pytest.approx(ref, rel=1e-5)


# ===================================================================== 6
def test_declared_positive_at_zero_uses_the_one_sided_step():
    f, L = _beside_model("pfg", "linear", False, domains=["positive", "positive"])
    th = jnp.asarray([0.7, 0.0])
    g = np.asarray(jax.grad(L)(th))
    assert g[1] == pytest.approx(float((L(th.at[1].set(1e-7)) - L(th)) / 1e-7), rel=1e-8)


@pytest.fixture
def fd_warnings():
    rec = []
    h = logging.Handler()
    h.emit = lambda r: rec.append(r.getMessage())
    log = _fd_steps._logger()
    log.addHandler(h)
    try:
        yield rec
    finally:
        log.removeHandler(h)


def test_zero_probe_difference_is_counted_and_logged_once(fd_warnings):
    """§4.5: above ~1.8e9 the capped positive probes collapse onto theta; the
    slot gets a zero gradient, counted, logged once per model."""
    g = Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([3]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([1])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0, 0.0]); a.add_edge(c, [0.0, 1e-12]); b.add_edge(c, [1.0, 0.0])
    m = Graph.pmf_from_graph(g, theta_dim=2, param_domains=["positive", "positive"])
    L = lambda t: jnp.sum(m(t, TIMES))
    for _ in range(2):
        gr = np.asarray(jax.grad(L)(jnp.asarray([0.7, 3e9])))
        assert gr[1] == 0.0 and gr[0] != 0.0
    assert m._fd_stats.zero_difference == 2
    assert len([w for w in fd_warnings if "rounded to the same value" in w]) == 1


def test_forward_tie_is_counted_and_logged_once(fd_warnings):
    """§4.5: an offset slot at 1e-13 beside an O(1) rate on a refusing
    (formula) model: the capped probes do not change the forward at all, so
    the slot gets a counted, logged zero (may be a genuine zero)."""
    g = Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([2]); b = g.find_or_create_vertex([1])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0, 1.0])
    g.weight_formula = "c0*t0 + c1*t1"
    m = Graph.pmf_from_graph(g, theta_dim=2)
    gr = np.asarray(jax.grad(lambda t: jnp.sum(m(t, TIMES)))(jnp.asarray([1.0, 1e-13])))
    assert gr[1] == 0.0 and np.isfinite(gr[0])
    assert m._fd_stats.tie == 1
    assert len([w for w in fd_warnings if "forward tie" in w]) == 1


def test_out_of_domain_values_keep_todays_probe():
    """§4.5: a 'unit' slot outside (0, 1) and a 'positive' slot below 0 use
    today's probe and quotient (bit for bit the undeclared result)."""
    _, L0 = _beside_model("pfg", "linear", False)
    _, Lu = _beside_model("pfg", "linear", False, domains=[None, "unit"])
    th = jnp.asarray([0.7, 1.3])
    assert np.asarray(jax.grad(Lu)(th)).tobytes() == np.asarray(jax.grad(L0)(th)).tobytes()
    _, L0 = _beside_model("pam", "linear", False)
    _, Lp = _beside_model("pam", "linear", False, domains=[None, "positive"])
    th = jnp.asarray([0.7, -0.05])
    assert np.asarray(jax.grad(Lp)(th)).tobytes() == np.asarray(jax.grad(L0)(th)).tobytes()


# ===================================================================== 7
def _offset_graph():
    g = Graph(1)
    s = g.starting_vertex()
    v = g.find_or_create_vertex([2]); a = g.find_or_create_vertex([1])
    s.add_edge(v, 1.0); v.add_edge(a, [1.0, 1.0])
    return g


def test_signed_parameter_under_unconstrained():
    """§1: rate = t0 + t1 at theta = [6, -5]. Today's 1e-9 clamp gives t1 a
    secant over [theta, 1e-9] (-0.49382 vs -0.50049, review 2); the declared
    'unconstrained' plain step gives d/dt1 == d/dt0, as it must."""
    times = jnp.asarray([0.5, 1.0, 2.0])
    th = jnp.asarray([6.0, -5.0])
    m0 = Graph.pmf_and_moments_from_graph(_offset_graph(), nr_moments=2, exact_moment_grad=False)
    mu = Graph.pmf_and_moments_from_graph(_offset_graph(), nr_moments=2, exact_moment_grad=False,
                                          param_domains=["unconstrained", "unconstrained"])
    ll = lambda m: (lambda t: jnp.sum(jnp.log(m(t, times)[0])))
    g0 = np.asarray(jax.grad(ll(m0))(th)); gu = np.asarray(jax.grad(ll(mu))(th))
    assert gu[1] == pytest.approx(gu[0], rel=1e-6)
    assert gu[1] == pytest.approx(-0.50049, abs=1e-5)
    assert abs(g0[1] - g0[0]) > 1e-3                      # today's clamp, kept when undeclared


def _pgeom():
    """Native discrete: absorption probability theta[0] per step."""
    g = Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0])
    return g


def test_discrete_row_sum_wall():
    """Out of scope (§4.1): undeclared at 1 - 5e-8 keeps today's clear error;
    a declared 'unit' slot at 1 - 1e-9 stays inside the wall."""
    k = jnp.asarray([1.0, 2.0, 3.0])
    m0 = Graph.pmf_from_graph(_pgeom(), discrete=True)
    with pytest.raises(Exception, match="outgoing rate <= 1"):
        np.asarray(jax.grad(lambda t: jnp.sum(m0(t, k)))(jnp.asarray([1.0 - 5e-8])))
    mu = Graph.pmf_from_graph(_pgeom(), discrete=True, param_domains=["unit"])
    g = np.asarray(jax.grad(lambda t: jnp.sum(mu(t, k)))(jnp.asarray([1.0 - 1e-9])))
    assert np.all(np.isfinite(g))
    g = np.asarray(jax.grad(lambda t: jnp.sum(m0(t, k)))(jnp.asarray([5e-8])))
    assert np.all(np.isfinite(g))                         # the lower wall: §4.4 fallback


def test_exposure_band_with_small_alpha():
    """An exposure model with alpha below 1 and the exposure slot in the band."""
    jpg, oi = C.joint_graphs()
    m, _, _, _ = jpg._daisy_chain_svgd_model(
        observed_indices=list(oi), epoch_starts=np.array([0.0, 0.5]),
        user_prior=None, user_fixed=None,
        exposure_arr=np.linspace(1e-3, 0.5, len(oi)), exposure_param_index=1)
    L = lambda t: jnp.sum(jnp.log(m(t, None)[0]))
    th = jnp.asarray([1.0, 5e-8, 1.0, 5e-8])
    g = np.asarray(jax.grad(L)(th))
    assert np.all(np.isfinite(g))
    assert g[1] == pytest.approx(_central(L, th, 1, 1e-9), rel=1e-3)



# ===================================================================== review fixes
def test_joint_index_host_fallback_at_the_site(probe_spy, monkeypatch):
    """The exact_grad_decline='fd' host loop itself (review m5): force the C
    adjoint to decline AFTER construction (its probe succeeded), so the
    host-side FD runs; the 'fixed' slot is skipped there, the other matches
    the exact gradient taken before the patch."""
    jpg, oi = C.joint_graphs()
    m = Graph.pmf_from_graph_joint_index(jpg, theta_dim=2, exact_grad=True,
                                         exact_grad_decline="fd", param_domains=[None, "fixed"])
    th = jnp.asarray([1.0, 0.5])
    L = lambda t: jnp.sum(jnp.log(m(t, jnp.asarray(oi))[0]))
    g_exact = np.asarray(jax.grad(L)(th))
    monkeypatch.setattr(Graph, "_sojourn_grad_theta_subset", lambda self, idx: [])
    probe_spy.clear()
    g_host = np.asarray(jax.grad(lambda t: jnp.sum(jnp.log(m(t, jnp.asarray(oi))[0])))(th))
    sites = {(s, i) for s, i, _, _ in probe_spy}
    assert sites == {("pmf_from_graph_joint_index.host", 0)}, sites
    assert g_host[1] == 0.0 and g_exact[1] == 0.0
    assert g_host[0] == pytest.approx(g_exact[0], rel=1e-5)


def test_rvp_counts_ties_per_model():
    """§4.5 per model (review m4): two rvp callers with their own counters do
    not share counts."""
    from phasic import ffi_wrappers as fw
    g = Graph(1)
    s0 = g.starting_vertex()
    a = g.find_or_create_vertex([2]); b = g.find_or_create_vertex([1]); c = g.find_or_create_vertex([3])
    s0.add_edge(a, 1.0); a.add_edge(b, [1.0, 1.0]); a.add_edge(c, [1.0, 0.0])
    g.weight_formula = "c0*t0 + c1*t1"
    sj = g.serialize()
    ipv = jnp.asarray(g._initial_probability_vector())
    tgt = jnp.asarray([b.index()], dtype=jnp.int32)
    st1, st2 = _fd_steps.FDStats("m1"), _fd_steps.FDStats("m2")
    th = jnp.asarray([1.0, 1e-13])
    for st in (st1, st2):
        jax.grad(lambda t: fw.compute_reward_visit_probability_ffi(sj, t, tgt, ipv, fd_stats=st))(th)
    assert st1.tie == 1 and st2.tie == 1


def test_fixed_domain_is_zero_on_the_exact_final_epoch_path():
    """Review m2: with exact_final_grad=True a 'fixed'-domain final-epoch slot
    gets a zero gradient too (it is skipped like a truly fixed slot)."""
    jpg, oi = C.joint_graphs()
    m, _, _, _ = jpg._daisy_chain_svgd_model(
        observed_indices=list(oi), epoch_starts=np.array([0.0, 0.5]),
        user_prior=None, user_fixed=None, exact_final_grad=True,
        param_domains=[None, None, None, "fixed"])
    th = jnp.asarray([1.0, 0.5, 1.5, 0.3])
    g = np.asarray(jax.grad(lambda t: jnp.sum(jnp.log(m(t, None)[0])))(th))
    assert g[3] == 0.0 and np.all(np.isfinite(g)) and g[2] != 0.0
