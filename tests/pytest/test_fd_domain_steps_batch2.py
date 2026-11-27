"""Plan A (domain-aware finite-difference steps), Batch 2 tests.

plans-and-findings/fd-domain-steps-plan.md v2.10 §6 Batch 2; findings in
plans-and-findings/fd-domain-steps-findings.md §11. Batch 2 makes the fitters
DECLARE each slot's domain: Graph.svgd through the resolver
(svgd._resolve_param_domains), Graph.epoch_model by default 'positive', and
SVGD.__init__ / FreeEpochModel.fit check containment.

Sections:
 1. the resolver and the containment rule (no models);
 2. what Graph.svgd declares on each route;
 3. the containment check in SVGD.__init__ and FreeEpochModel.fit, and
    epoch_model's param_domains;
 4. gradients at fixed particles: declared vs today (master);
 5. one coarse fit-level check per route; the near-floor epoch fit;
 6. positive_params=False on refusing models; the D-NZ pins and the band
    trade-off; the joint-index host site;
 7. the probe spy through Graph.svgd (multivariate leaf, exposure fits);
 8. the D-NZ documentation deliverable.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import phasic  # noqa: E402
from phasic import Graph, SVGD, GaussPrior, BetaPrior, LogGaussPrior, _fd_steps  # noqa: E402
from phasic.svgd import (  # noqa: E402
    _check_param_domains, _domain_contains, _resolve_param_domains,
    _softplus_transform,
)
from phasic.epoch_model import FreeEpochModel  # noqa: E402

import _fd_golden_cells as C  # noqa: E402

TIMES = jnp.asarray(C.TIMES)
RW = jnp.asarray(C.REWARDS)
DATA = np.asarray([0.3, 0.5, 0.8, 1.1, 1.7, 2.4])
FIT = dict(n_particles=4, n_iterations=2, progress=False, seed=1, learning_rate=1e-6)
EPOCHS = [0.0, 0.5]


def _gp():
    return GaussPrior(1.0, 0.5, lower=0.0)


# ===================================================================== 1
def test_resolver_maps_transforms_to_domains():
    r = lambda prior, pp=True, pt=None, **kw: _resolve_param_domains(
        prior, pp, pt, theta_dim=kw.pop("n", 3), **kw)
    assert r(None) == ("positive",) * 3                      # softplus
    assert r(_gp()) == ("positive",) * 3
    assert r(BetaPrior(2, 2)) == ("unit",) * 3              # single BetaPrior: every slot
    assert r([BetaPrior(2, 2), _gp(), None]) == ("unit", "positive", "positive")
    assert r(GaussPrior(1, 1), pp=False) == ("unconstrained",) * 3
    assert r(None, pp=False, pt=lambda p: jnp.exp(p)) == (None,) * 3   # custom: unknown
    assert r([_gp(), _gp(), None], fixed_indices=[2]) == ("positive", "positive", "fixed")
    # the exposure slot is never 'unit' (§4.5)
    assert r([BetaPrior(2, 2), BetaPrior(2, 2), _gp()], exposure_slots=[1]) == \
        ("unit", "positive", "positive")
    # a prior list of the wrong length: the caller's own check raises
    assert r([_gp(), _gp()]) is None
    # the same errors as SVGD's transform decision
    with pytest.raises(ValueError, match="bounded interval"):
        r(BetaPrior(2, 2), pp=False)


def test_resolver_has_no_side_effects():
    p = BetaPrior(2, 2)
    before = getattr(p, "_transform", "unset")
    _resolve_param_domains(p, True, None, theta_dim=2)
    assert getattr(p, "_transform", "unset") is before


def test_containment_rule():
    D = ("positive", "unit", "unconstrained", "fixed", None)
    want = {   # (model, fit) -> allowed
        ("positive", "positive"): True, ("positive", "unit"): True,
        ("positive", "unconstrained"): False,
        ("unit", "unit"): True, ("unit", "positive"): False, ("unit", "unconstrained"): False,
        ("unconstrained", "positive"): True, ("unconstrained", "unit"): True,
        ("unconstrained", "unconstrained"): True,
        ("fixed", "positive"): False, ("fixed", "unit"): False, ("fixed", "unconstrained"): False,
    }
    for m in D:
        for f in D:
            if f == "fixed" or m is None or f is None:
                assert _domain_contains(m, f)
            else:
                assert _domain_contains(m, f) == want[(m, f)], (m, f)
    with pytest.raises(ValueError, match="2 parameter domains"):
        _check_param_domains(("positive",) * 2, ("positive",) * 3, "X", "")


def test_flat_epoch_domain_list_mixed_priors_tying_fixed_exposure():
    """§3: the epoch route broadcasts local domains over n_epochs x P, a tied
    slave takes its master's domain (even when a flat prior list gives the
    slave epoch another prior), local fixings are 'fixed' in every epoch, and
    the exposure slot is widened to 'positive' in every epoch."""
    from phasic import _epoch_fd_domains
    f = lambda prior, **kw: _epoch_fd_domains(
        prior, kw.pop("pp", True), None, n_epochs=3, param_length=3,
        user_fixed=kw.pop("fixed", None), user_tied=kw.pop("tied", None),
        exposure_param_index=kw.pop("exp", None))
    local = [BetaPrior(2, 2), _gp(), BetaPrior(2, 2)]
    assert f(local, fixed=[(1, 0.5)], tied=[(0, [0, 2])]) == (
        "unit", "fixed", "unit", "unit", "fixed", "unit", "unit", "fixed", "unit")
    assert f(local, exp=2) == ("unit", "positive", "positive") * 3
    # a flat list: the slave (epoch 1, slot 0) has a Gauss prior but takes
    # its master's 'unit'
    flat = [BetaPrior(2, 2), _gp(), _gp(), _gp(), _gp(), _gp(), _gp(), _gp(), _gp()]
    assert f(flat, tied=[(0, [0, 1])])[:4] == ("unit", "positive", "positive", "unit")
    assert f(None) == ("positive",) * 9                     # default prior: softplus
    assert f(None, pp=False, fixed=[(0, [1.0, 2.0, 3.0])]) == \
        ("fixed", "unconstrained", "unconstrained") * 3
    assert f([_gp(), _gp()]) is None                        # the builder raises


# ===================================================================== 2
def test_graph_svgd_declares_on_every_single_epoch_leaf():
    g = C.beside()
    s = g.svgd(DATA, prior=[BetaPrior(2, 2), _gp()], **FIT)
    assert s.model._param_domains == ("unit", "positive")
    s = g.svgd(DATA, prior=GaussPrior(1.0, 0.5), positive_params=False, fixed=[(0, 1.0)], **FIT)
    assert s.model._param_domains == ("fixed", "unconstrained")
    s = g.svgd(DATA, prior=None, **FIT)                     # DataPrior: softplus
    assert s.model._param_domains == ("positive", "positive")
    s = g.svgd(DATA, prior=GaussPrior(0.0, 1.0), positive_params=False,
               param_transform=lambda p: jnp.exp(p), **FIT)
    assert s.model._param_domains == (None, None)           # custom transform: unknown
    s = g.svgd(DATA, rewards=RW, prior=[_gp(), _gp()], **FIT)    # 1-D rewards leaf
    assert s.model._param_domains == ("positive", "positive")
    from phasic.svgd import dense_to_sparse
    obs = dense_to_sparse(jnp.column_stack([DATA, 2 * DATA]))
    s = g.svgd(obs, rewards=jnp.stack([RW, 2 * RW]), prior=[_gp(), _gp()], **FIT)
    assert s.model._param_domains == ("positive", "positive")    # multivariate leaf
    jpg, _ = C.joint_graphs()
    s = jpg.svgd(C.joint_rows(), prior=[_gp(), LogGaussPrior(ci=[1e-3, 1e-1])], **FIT)
    assert s.model._param_domains == ("positive", "positive")    # joint index


def test_graph_svgd_declares_on_the_epoch_route():
    jpg, _ = C.joint_graphs()
    s = jpg.svgd(C.joint_rows(), epoch_starts=EPOCHS,
                 prior=[BetaPrior(2, 2), LogGaussPrior(ci=[1e-3, 1e-1])],
                 tied=[(0, [0, 1])], **FIT)
    assert s.model._param_domains == ("unit", "positive", "unit", "positive")
    s = jpg.svgd(C.joint_rows(), epoch_starts=EPOCHS, fixed=[(1, 1e-2)], **FIT)
    assert s.model._param_domains == ("positive", "fixed", "positive", "fixed")
    s = jpg.svgd(C.joint_rows(), epoch_starts=EPOCHS, fixed=[(1, 1e-2)],
                 prior=[BetaPrior(2, 2), None], exposure=2.0, exposure_param_index=0, **FIT)
    assert s.model._param_domains == ("positive", "fixed", "positive", "fixed")


# ===================================================================== 3
def _pam(doms):
    return Graph.pmf_and_moments_from_graph(C.beside(), nr_moments=2, theta_dim=2,
                                            param_domains=doms)


def _svgd(model, **kw):
    kw.setdefault("prior", [_gp(), _gp()])
    return SVGD(model=model, observed_data=DATA, theta_dim=2, n_particles=4,
                n_iterations=1, progress=False, seed=1, **kw)


def test_svgd_init_containment_raises_on_a_widening():
    with pytest.raises(ValueError, match="built for 'unit', the fit uses 'positive'"):
        _svgd(_pam(["unit", "positive"]))
    with pytest.raises(ValueError, match="built for 'positive', the fit uses 'unconstrained'"):
        _svgd(_pam(["positive", "positive"]), prior=[GaussPrior(1, 1), GaussPrior(1, 1)],
              positive_params=False)
    with pytest.raises(ValueError, match="built for 'fixed'"):
        _svgd(_pam(["positive", "fixed"]))


def test_svgd_init_containment_passes():
    _svgd(_pam(["positive", "positive"]), prior=[BetaPrior(2, 2), _gp()])   # unit in positive
    _svgd(_pam(["positive", "fixed"]), prior=[_gp(), None], fixed=[(1, 0.5)])
    _svgd(_pam(["unit", "positive"]), prior=[None, _gp()], fixed=[(0, 0.3)])  # fit-fixed
    _svgd(_pam(None))                                                   # undeclared model
    _svgd(_pam(["unconstrained"] * 2))
    _svgd(_pam(["unit", "unit"]), prior=GaussPrior(0, 1), positive_params=False,
          param_transform=jax.nn.sigmoid)                              # custom: unknown
    # the exposure slot: a 'positive' model fitted with a sigmoid passes
    _svgd(_pam(["positive", "positive"]), prior=[BetaPrior(2, 2), _gp()],
          exposure=2.0, exposure_param_index=0)


def test_svgd_init_checks_before_the_exposure_wrapper():
    """Review m4: the check reads the model's domains BEFORE the exposure
    wrapper drops its attributes, so a widening with exposure still raises."""
    with pytest.raises(ValueError, match="built for 'unit', the fit uses 'positive'"):
        _svgd(_pam(["unit", "positive"]), exposure=2.0, exposure_param_index=1)


def test_graph_svgd_widens_a_beta_exposure_slot_on_the_real_path(monkeypatch):
    """Review m4: a BetaPrior on the exposure slot reaches the builder as
    'positive' through Graph.svgd itself (the SVGD exposure wrapper hides the
    model's attributes, so the builder call is spied)."""
    seen = []
    orig = Graph.pmf_and_moments_from_graph

    def spy(cls, *a, **k):
        seen.append(k.get("param_domains"))
        return orig(*a, **k)
    monkeypatch.setattr(Graph, "pmf_and_moments_from_graph", classmethod(spy))
    C.beside().svgd(DATA, prior=[BetaPrior(2, 2), BetaPrior(2, 2)], exposure=2.0,
                    exposure_param_index=1, **FIT)
    assert seen == [("unit", "positive")]


def test_fit_raises_before_the_default_prior_reference_fit():
    """Review m4: a containment failure is raised before the default prior's
    reference fit runs (the factory is never called)."""
    jpg, _ = C.joint_graphs()
    em = jpg.epoch_model(C.joint_rows(), EPOCHS, fixed=[(1, 1e-2)])
    assert em._default_prior_factory is not None

    def boom(*a, **k):
        raise AssertionError("default prior factory called before the check")
    em._default_prior_factory = boom
    with pytest.raises(ValueError, match="FreeEpochModel.fit"):
        em.fit(positive_params=False, **FIT)


def test_epoch_model_param_domains_forms():
    jpg, _ = C.joint_graphs()
    rows = C.joint_rows()
    em = jpg.epoch_model(rows, EPOCHS)
    assert em._param_domains == ("positive",) * 4 == em.model._param_domains
    assert jpg.epoch_model(rows, EPOCHS, param_domains=["unit", "positive"])._param_domains == \
        ("unit", "positive", "unit", "positive")
    assert jpg.epoch_model(rows, EPOCHS, param_domains=["positive", "unit", "fixed", "positive"]
                           )._param_domains == ("positive", "unit", "fixed", "positive")
    em0 = jpg.epoch_model(rows, EPOCHS, param_domains=None)
    assert em0._param_domains is None and em0.model._param_domains is None
    with pytest.raises(ValueError, match="param_domains has 3 entries"):
        jpg.epoch_model(rows, EPOCHS, param_domains=["positive"] * 3)
    with pytest.raises(ValueError, match="param_domains entries"):
        jpg.epoch_model(rows, EPOCHS, param_domains="rate")


def test_free_epoch_model_fit_checks_containment():
    jpg, _ = C.joint_graphs()
    rows = C.joint_rows()
    em = jpg.epoch_model(rows, EPOCHS, fixed=[(1, 1e-2)])
    # listed behaviour change (§7): positive_params=False on a default model
    with pytest.raises(ValueError, match="param_domains='unconstrained'"):
        em.fit(prior=[GaussPrior(1, 0.5), None, GaussPrior(1, 0.5), None],
               positive_params=False, **FIT)
    em.fit(prior=[BetaPrior(2, 2), None, BetaPrior(2, 2), None], **FIT)   # unit in positive
    emu = jpg.epoch_model(rows, EPOCHS, fixed=[(1, 1e-2)], param_domains="unconstrained")
    emu.fit(prior=[GaussPrior(1, 0.3), None, GaussPrior(1, 0.3), None],
            positive_params=False, **FIT)
    # a model slot declared 'fixed' must be fixed in the fit
    emf = jpg.epoch_model(rows, EPOCHS, param_domains=["positive", "fixed"])
    with pytest.raises(ValueError, match="built for 'fixed'"):
        emf.fit(prior=[_gp()] * 4, fixed=[(1, 1e-2)], **FIT)
    emf.fit(prior=[_gp(), None, _gp(), None], fixed=[(1, 1e-2), (3, 1e-2)], **FIT)


def test_fit_reads_domains_from_the_object_not_the_model():
    """§3: fit() reads FreeEpochModel._param_domains (tests replace .model
    with spies that carry no attributes)."""
    jpg, _ = C.joint_graphs()
    em = jpg.epoch_model(C.joint_rows(), EPOCHS, fixed=[(1, 1e-2)])
    real = em.model
    em.model = lambda th, obs, rewards=None: real(th, obs, rewards)
    with pytest.raises(ValueError, match="FreeEpochModel.fit"):
        em.fit(prior=[GaussPrior(1, 0.5), None, GaussPrior(1, 0.5), None],
               positive_params=False, **FIT)


# ===================================================================== 4
def _leaf(name, doms):
    g = C.beside()
    if name == "pam":
        m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2, param_domains=doms)
        return lambda th: m(th, TIMES)
    if name == "pam_rw":
        m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2, param_domains=doms)
        return lambda th: m(th, TIMES, rewards=RW)
    if name == "multivar":
        m = Graph.pmf_and_moments_from_graph_multivariate(g, nr_moments=2, theta_dim=2,
                                                          param_domains=doms)
        return lambda th: m(th, TIMES, rewards=jnp.stack([RW, 2 * RW]))
    if name == "jix":
        jpg, oi = C.joint_graphs()
        m = Graph.pmf_from_graph_joint_index(jpg, theta_dim=2, param_domains=doms)
        return lambda th: m(th, jnp.asarray(oi))
    jpg, oi = C.joint_graphs()
    m, _, _, _ = jpg._daisy_chain_svgd_model(
        observed_indices=list(oi), epoch_starts=np.array(EPOCHS),
        user_prior=None, user_fixed=None, param_domains=None if doms is None else doms * 2)
    return lambda th: m(th, None)


def _loglik(f):
    """SVGD's likelihood term (first output only)."""
    return lambda th: jnp.sum(jnp.log(f(th)[0]))


def _richardson(L, th, i, s=1e-2):
    def cd(h):
        tp = th.at[i].set(th[i] * np.exp(h)); tm = th.at[i].set(th[i] * np.exp(-h))
        return float((L(tp) - L(tm)) / (tp[i] - tm[i]))
    return (4 * cd(s / 2) - cd(s)) / 3


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e-4, 1e-5])
@pytest.mark.parametrize("name", ["pam", "pam_rw", "multivar", "jix", "epoch"])
def test_gradients_at_fixed_particles_match_master(name, scale):
    """§6 Batch 2: declared ('positive', what Graph.svgd declares under
    softplus) against undeclared (today's code, bit for bit = master outside
    the band) at fixed particles, per Graph.svgd leaf. theta >= 1e-3: within
    1e-6 relative (measured <= 1.7e-7). At 1e-4 and 1e-5 today's absolute
    step is itself up to 1.7e-4 off a Richardson reference (joint index,
    epoch; batch2/m1_measure_part1_loglik.log), so there the declared
    gradient must be within 1e-6 of the reference (tightest measured: epoch
    at 1e-5, 7.1e-7) and no worse than today's error + 1e-6 (review m3: a
    bound on |declared - today| would follow from the first by the triangle
    inequality)."""
    P = 4 if name == "epoch" else 2
    th = jnp.asarray(np.array([1.0, 0.6, 1.3, 0.8][:P]) * scale)
    Ld = _loglik(_leaf(name, ["positive"] * 2))
    Lu = _loglik(_leaf(name, None))
    gd = np.asarray(jax.grad(Ld)(th)); gu = np.asarray(jax.grad(Lu)(th))
    if scale >= 1e-3:
        np.testing.assert_allclose(gd, gu, rtol=1e-6)
    else:
        ref = np.array([_richardson(Lu, th, i) for i in range(P)])
        np.testing.assert_allclose(gd, ref, rtol=1e-6)
        assert np.all(np.abs(gd - ref) <= np.abs(gu - ref) + 1e-6 * np.abs(ref))


# ===================================================================== 5
def _rel(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.max(np.abs(a - b) / np.abs(b)))


FIT_LEVEL = dict(n_particles=8, n_iterations=40, progress=False, seed=3, learning_rate=0.05)
FIT_RTOL = 1e-3


def test_fit_level_graph_svgd_single_epoch(monkeypatch):
    g = C.beside()
    kw = dict(prior=[_gp(), _gp()], **FIT_LEVEL)
    sd = g.svgd(DATA, **kw)
    monkeypatch.setattr(phasic, "_svgd_fd_domains", lambda *a, **k: None)
    su = g.svgd(DATA, **kw)
    assert sd.model._param_domains == ("positive",) * 2 and su.model._param_domains is None
    print("fit-level drift, Graph.svgd single epoch:", _rel(sd.theta_mean, su.theta_mean))
    assert _rel(sd.theta_mean, su.theta_mean) < FIT_RTOL


def test_fit_level_graph_svgd_epoch(monkeypatch):
    jpg, _ = C.joint_graphs()
    kw = dict(epoch_starts=EPOCHS, fixed=[(1, 1e-2)], **FIT_LEVEL)
    sd = jpg.svgd(C.joint_rows(), **kw)
    monkeypatch.setattr(phasic, "_epoch_fd_domains", lambda *a, **k: None)
    su = jpg.svgd(C.joint_rows(), **kw)
    assert sd.model._param_domains == ("positive", "fixed") * 2
    assert su.model._param_domains is None
    print("fit-level drift, Graph.svgd epoch:", _rel(sd.theta_mean[::2], su.theta_mean[::2]))
    assert _rel(sd.theta_mean[::2], su.theta_mean[::2]) < FIT_RTOL


def test_fit_level_epoch_model_and_direct_svgd():
    jpg, _ = C.joint_graphs()
    rows = C.joint_rows()
    ed = jpg.epoch_model(rows, EPOCHS, fixed=[(1, 1e-2)])
    eu = jpg.epoch_model(rows, EPOCHS, fixed=[(1, 1e-2)], param_domains=None)
    rd = ed.fit(**FIT_LEVEL); ru = eu.fit(**FIT_LEVEL)
    print("fit-level drift, epoch_model:", _rel(rd.theta_mean[::2], ru.theta_mean[::2]))
    assert _rel(rd.theta_mean[::2], ru.theta_mean[::2]) < FIT_RTOL
    # direct SVGD(model=...) with a declared model
    kw = dict(observed_data=DATA, theta_dim=2, prior=[_gp(), _gp()], **FIT_LEVEL)
    sd = SVGD(model=_pam(["positive"] * 2), **kw); sd.optimize()
    su = SVGD(model=_pam(None), **kw); su.optimize()
    print("fit-level drift, direct SVGD:", _rel(sd.theta_mean, su.theta_mean))
    assert _rel(sd.theta_mean, su.theta_mean) < FIT_RTOL


def test_near_floor_epoch_fits_complete_with_declared_domains():
    """The reported crash (particles around 1e-8), now on DECLARED models:
    Graph.svgd's epoch route and epoch_model(...).fit."""
    small = LogGaussPrior(ci=[2e-9, 5e-8])
    jpg, _ = C.joint_graphs()
    s = jpg.svgd(C.joint_rows(), epoch_starts=EPOCHS, prior=[small, small],
                 n_particles=4, n_iterations=2, learning_rate=1e-12, progress=False, seed=1)
    assert s.model._param_domains == ("positive",) * 4
    assert np.all(np.isfinite(np.asarray(s.theta_mean)))
    em = jpg.epoch_model(C.joint_rows(), EPOCHS)
    r = em.fit(prior=[small] * 4, n_particles=4, n_iterations=2, learning_rate=1e-12,
               progress=False, seed=1)
    assert np.all(np.isfinite(np.asarray(r.theta_mean)))
    # and the gradient itself at the reported thetas
    L = _loglik(_leaf("epoch", ["positive"] * 2))
    for th in ([1.0, 1e-8, 1.0, 1e-8], [1.0, 1e-9, 1.0, 1e-9], [1e-9, 1e-4, 1e-4, 1e-4]):
        assert np.all(np.isfinite(np.asarray(jax.grad(L)(jnp.asarray(th)))))


# ===================================================================== 6
def _refusing_model(kind):
    if kind == "discretized":
        g = C.beside().discretize(1.0)
        m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=True, theta_dim=3,
                                             param_domains=["unconstrained"] * 3)
        return (lambda th: m(th, jnp.asarray([1.0, 2.0, 4.0]))), 3
    if kind == "linear_rewards":
        m = Graph.pmf_and_moments_from_graph(C.beside(), nr_moments=2, theta_dim=2,
                                             param_domains=["unconstrained"] * 2)
        return (lambda th: m(th, TIMES, rewards=RW)), 2
    m = Graph.pmf_and_moments_from_graph(C.beside(kind), nr_moments=2, theta_dim=2,
                                         param_domains=["unconstrained"] * 2)
    return (lambda th: m(th, TIMES)), 2


@pytest.mark.parametrize("kind", ["log", "formula", "callback", "discretized", "linear_rewards"])
def test_positive_params_false_refusing_models_at_5e8(kind):
    """§6 Batch 2: positive_params=False declares 'unconstrained', which drops
    today's 1e-9 clamp; on refusing models the §4.4 band rule replaces it, so
    a slot at 5e-8 does not crash. Checked against a central difference whose
    probes stay positive."""
    assert _resolve_param_domains(GaussPrior(1, 1), False, None, theta_dim=2) == \
        ("unconstrained",) * 2                              # what Graph.svgd declares
    f, P = _refusing_model(kind)
    L = _loglik(f)
    slots = [1, 2] if kind == "discretized" else [1]
    for i in slots:
        th = jnp.asarray([1.0] * P).at[i].set(5e-8)
        g = np.asarray(jax.grad(L)(th))
        assert np.all(np.isfinite(g))
        h = 1e-9
        ref = float((L(th.at[i].add(h)) - L(th.at[i].add(-h))) / (2 * h))
        assert g[i] == pytest.approx(ref, rel=1e-3, abs=1e-9)


def _two_stage():
    """s -> a; a -> b (t0); b -> absorb (t1): E[T] = 1/t0 + 1/t1."""
    g = Graph(1); s = g.starting_vertex()
    a = g.find_or_create_vertex([2]); b = g.find_or_create_vertex([1]); c = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0, 0.0]); b.add_edge(c, [0.0, 1.0])
    return g


def _mean_grad(g, th, doms):
    m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=len(th),
                                         exact_moment_grad=False, param_domains=doms)
    return np.asarray(jax.grad(lambda t: m(t, TIMES)[1][0])(jnp.asarray(th))), m


def test_dnz_pin_plain_step_has_the_documented_sign_error():
    """User decision D-NZ (plan §3), pinned so a later change is deliberate:
    positive_params=False, linear, no rewards, exact_moment_grad=False, a rate
    slot at 5e-8 beside a pole of E[T]. The declared 'unconstrained' slot
    uses the plain +-1e-7 step, which straddles the pole and gets the WRONG
    SIGN: +1.333e14 against the closed form -4.0e14 (review 8). Today's
    (undeclared) clamp gives -6.67e15 (right sign, wrong size)."""
    gu, _ = _mean_grad(_two_stage(), [1.0, 5e-8], ["unconstrained"] * 2)
    assert gu[1] == pytest.approx(4.0 / 3.0 * 1e14, rel=1e-6)
    assert -1.0 / (5e-8) ** 2 == pytest.approx(-4e14)
    g0, _ = _mean_grad(_two_stage(), [1.0, 5e-8], None)
    assert g0[1] == pytest.approx(-6.666666666e15, rel=1e-6)


def _log_pole():
    """Log mode, one slot: a -> b (c=1), b -> absorb (c=2): E[T] = 1.5 / t."""
    g = Graph(1); s = g.starting_vertex()
    a = g.find_or_create_vertex([2]); b = g.find_or_create_vertex([1]); c = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0]); b.add_edge(c, [2.0])
    g.weight_mode = "log"
    return g


def _offset(mode):
    """One exit at rate t0 + t1 (t1 only offsets t0 = 1): E[T] = 1/(t0 + t1)."""
    g = Graph(1); s = g.starting_vertex()
    v = g.find_or_create_vertex([2]); a = g.find_or_create_vertex([1])
    s.add_edge(v, 1.0); v.add_edge(a, [1.0, 1.0])
    if mode == "formula":
        g.weight_formula = "c0*t0 + c1*t1"
    return g


@pytest.mark.parametrize("v", [5e-8, 1e-9, 1e-13])
def test_band_rule_on_a_refusing_model_is_exact_at_a_pole(v):
    """§4.3: on a refusing (log-mode) model an 'unconstrained' slot in the
    band uses the capped positive rule, exact at a 1/t pole (closed form,
    not exact_moment_grad: OI-117)."""
    g, m = _mean_grad(_log_pole(), [v], ["unconstrained"])
    assert g[0] == pytest.approx(-1.5 / v ** 2, rel=1e-9)
    assert m._fd_stats.tie == 0


def test_band_rule_trade_off_on_an_offset_slot(caplog):
    """§4.3 / §4.5 trade-off, pinned (measured with s = 3e-5,
    batch2/m1_measure.log): on a refusing (formula) model an 'unconstrained'
    offset slot loses accuracy as it shrinks: 1.5e-5 relative at 5e-8,
    8e-4 at 1e-9, and at 1e-13 the probes tie: a zero gradient, counted in
    _fd_stats.tie and logged (log mode cannot express an additive offset,
    so the offset pin uses formula mode, also refusing)."""
    exact = lambda v: -1.0 / (1.0 + v) ** 2
    g, _ = _mean_grad(_offset("formula"), [1.0, 5e-8], ["unconstrained"] * 2)
    assert abs(g[1] / exact(5e-8) - 1) < 1e-4
    g, _ = _mean_grad(_offset("formula"), [1.0, 1e-9], ["unconstrained"] * 2)
    assert 1e-5 < abs(g[1] / exact(1e-9) - 1) < 5e-3
    with caplog.at_level("WARNING"):
        g, m = _mean_grad(_offset("formula"), [1.0, 1e-13], ["unconstrained"] * 2)
    assert g[1] == 0.0 and m._fd_stats.tie >= 1
    assert "forward tie" in caplog.text


@pytest.mark.parametrize("v", [5e-8, 1e-9, 1e-13])
def test_plain_step_keeps_offset_accuracy_on_a_non_refusing_model(v):
    """§6 Batch 2: linear, continuous, single-epoch, no rewards (non-refusing):
    an 'unconstrained' offset slot at tiny values keeps the plain +-1e-7 step,
    as accurate as today (here 1e-10 vs today's 1e-7, which clamps)."""
    exact = -1.0 / (1.0 + v) ** 2
    g, m = _mean_grad(_offset("linear"), [1.0, v], ["unconstrained"] * 2)
    assert g[1] == pytest.approx(exact, rel=1e-8)
    g0, _ = _mean_grad(_offset("linear"), [1.0, v], None)
    assert g0[1] == pytest.approx(exact, rel=1e-6)


@pytest.mark.parametrize("v", [5e-8, 1e-9])
def test_joint_index_host_site_keeps_todays_relative_step(v, monkeypatch):
    """§5 host-site row: 'unconstrained' slots keep today's RELATIVE step
    (correct near a pole, where the plain step would flip the sign, review
    11): bit-identical to undeclared, and equal to the exact adjoint."""
    jpg, oi = C.joint_graphs()
    th = jnp.asarray([1.0, v])
    mk = lambda d: Graph.pmf_from_graph_joint_index(
        jpg, theta_dim=2, exact_grad=True, exact_grad_decline="fd", param_domains=d)
    mu, m0 = mk(["unconstrained"] * 2), mk(None)
    L = lambda m: (lambda t: jnp.sum(jnp.log(m(t, jnp.asarray(oi))[0])))
    g_exact = np.asarray(jax.grad(L(mu))(th))
    monkeypatch.setattr(Graph, "_sojourn_grad_theta_subset", lambda self, idx: [])
    gu = np.asarray(jax.grad(L(mu))(th)); g0 = np.asarray(jax.grad(L(m0))(th))
    assert gu.tobytes() == g0.tobytes()
    np.testing.assert_allclose(gu, g_exact, rtol=1e-4)


# ===================================================================== 7
@pytest.fixture
def probe_spy():
    rec = []
    _fd_steps.set_probe_hook(lambda site, i, tp, tm: rec.append(
        (site, i, np.atleast_2d(np.asarray(tp)).copy(), np.atleast_2d(np.asarray(tm)).copy())))
    try:
        yield rec
    finally:
        _fd_steps.set_probe_hook(None)


def _assert_declared_positive_probes(rec, sites, free):
    """At theta ~ 1e-4 a declared 'positive' slot is probed at theta*e^(+-3e-5)
    (log ratio 6e-5); today's +-1e-7 would give a log ratio of ~2e-3."""
    assert rec, "no FD probe was recorded"
    seen = set()
    for site, i, tp, tm in rec:
        assert site in sites, site
        r = np.log(tp[:, i] / tm[:, i])
        np.testing.assert_allclose(r, 2 * _fd_steps.S, rtol=1e-6)
        seen.add(i)
    assert seen == set(free), seen


def _tiny_prior():
    return GaussPrior(1e-4, 1e-5, lower=0.0)


SPY_FIT = dict(n_particles=2, n_iterations=1, progress=False, seed=1, learning_rate=1e-12)


def test_spy_multivariate_leaf_carries_domains(probe_spy):
    from phasic.svgd import dense_to_sparse
    obs = dense_to_sparse(jnp.column_stack([DATA, 2 * DATA]))
    C.beside().svgd(obs, rewards=jnp.stack([RW, 2 * RW]),
                    prior=[_tiny_prior(), _tiny_prior()], **SPY_FIT)
    _assert_declared_positive_probes(probe_spy, {"pmf_and_moments_from_graph"}, [0, 1])


def test_spy_exposure_fit_single_epoch_carries_domains(probe_spy):
    C.beside().svgd(DATA, prior=[_tiny_prior(), _tiny_prior()], exposure=2.0,
                    exposure_param_index=1, **SPY_FIT)
    _assert_declared_positive_probes(probe_spy, {"pmf_and_moments_from_graph"}, [0, 1])


def test_spy_exposure_fit_epoch_carries_domains(probe_spy):
    jpg, _ = C.joint_graphs()
    jpg.svgd(C.joint_rows(), epoch_starts=EPOCHS, prior=[_tiny_prior(), None],
             fixed=[(1, 1e-2)], exposure=np.linspace(0.5, 2.0, len(C.joint_rows())),
             exposure_param_index=0, **SPY_FIT)
    _assert_declared_positive_probes(probe_spy, {"epoch model"}, [0, 2])


# ===================================================================== 8
def test_dnz_documentation_is_present():
    """§6 Batch 2 documentation deliverable (D-NZ): the positive_params
    docstrings of Graph.svgd and SVGD, epoch_model's param_domains and
    FreeEpochModel.fit state the near-zero limit, the counted/logged tie and
    the advice to fit rates with positive_params=True."""
    for doc in (Graph.svgd.__doc__, SVGD.__doc__, Graph.epoch_model.__doc__,
                FreeEpochModel.fit.__doc__):
        text = " ".join(doc.split())
        assert "D-NZ" in text
        assert "including its sign" in text
        assert "positive_params=True" in text
        assert "forward tie" in text and "logged once" in text
