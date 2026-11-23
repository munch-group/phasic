"""A fixed parameter reaches the model, and every reported result, as given.

Fixed values used to be stored in the unconstrained space and mapped back
through the parameter transform. That round trip is not the identity: a
value fixed at 0 reached the model as softplus(0) = 0.693, values below
1e-9 as 1e-9, values above ~700 as inf, a negative value as its softplus,
and under a custom transform the raw value was transformed (2.0 -> exp(2)).
Now the model-facing transform overwrites fixed slots with the user's
float, so nothing fixed passes through a transform.
Plan: bug2-fixed-values-plan.md.

Spy models record the theta they are called with; real models check that a
fit IS the fit at the fixed value. A fresh model object per fit unless a
test says otherwise.
"""
import random
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phasic
from phasic import Graph, GaussPrior, BetaPrior, SVGD
from phasic.mcmc import MCMC
from phasic.model_selection import _check_nested_fixed

from test_epoch_normalization import normalised, p_n2_two_epoch
from test_notebook_model_recovery import coalescent_indexed

DATA = np.array([0.5, 1.0, 1.5, 2.0])
VALUES = [0.0, 1e-12, 1e-10, 1e-9, 1e-8, 0.3, 5.0, 800.0, 1e6, -0.5]


def make_spy(dim=2):
    seen = []

    def spy(theta, times, rewards=None):
        jax.debug.callback(
            lambda t: seen.append(np.atleast_2d(np.asarray(t)).copy()), theta)
        return jnp.ones_like(times), jnp.zeros(2)

    return spy, seen


def seen_at(seen, slot):
    assert len(seen) >= 3, "the spy recorded too few model calls"
    return np.concatenate([s[:, slot] for s in seen])


def gauss():
    return GaussPrior(ci=[0.5, 5], lower=0.0)


def svgd_fit(fixed, n_particles=4, model=None, prior=None, **kw):
    spy, seen = make_spy()
    kw.setdefault('n_iterations', 2)
    s = SVGD(model=model or spy, observed_data=DATA, theta_dim=2,
             prior=prior if prior is not None else [gauss(), None],
             fixed=fixed, n_particles=n_particles, progress=False, seed=1, **kw)
    s.optimize()
    return s, seen


def assert_exports_exact(s, slot, v):
    r = s.get_results()
    np.testing.assert_array_equal(np.asarray(r['particles'])[:, slot], v)
    np.testing.assert_array_equal(np.asarray(r['history'])[..., slot], v)
    for src in (r, {'theta_mean': s.theta_mean, 'theta_std': s.theta_std}):
        assert float(np.asarray(src['theta_mean'])[slot]) == v
        assert float(np.asarray(src['theta_std'])[slot]) == 0.0
    assert r['hpd_lower'][slot] == v and r['hpd_upper'][slot] == v


# ------------------------------------------------- SVGD: model input, exports
# All values at 4 particles; 40 and 50 particles for two values whose mean
# over identical floats does not round back to the value.
@pytest.mark.parametrize("v, n_particles",
                         [(v, 4) for v in VALUES]
                         + [(v, n) for v in (0.3, 1e-10) for n in (40, 50)])
def test_svgd_positive_params_delivers_fixed_value(v, n_particles):
    s, seen = svgd_fit([(1, v)], n_particles=n_particles)
    np.testing.assert_array_equal(seen_at(seen, 1), v)
    assert_exports_exact(s, 1, v)


@pytest.mark.parametrize("v", [2.0, 0.0, 0.3])
def test_svgd_custom_transform_delivers_fixed_value(v):
    s, seen = svgd_fit([(1, v)], positive_params=False,
                       param_transform=lambda phi: jnp.exp(phi),
                       prior=[GaussPrior(ci=[0.5, 5]), None])
    np.testing.assert_array_equal(seen_at(seen, 1), v)
    assert_exports_exact(s, 1, v)


@pytest.mark.parametrize("v", [0.25, 0.3])
def test_svgd_per_dim_transform_delivers_fixed_value(v):
    s, seen = svgd_fit([(1, v)], prior=[BetaPrior(ci=(0.05, 0.95)), None])
    np.testing.assert_array_equal(seen_at(seen, 1), v)
    assert_exports_exact(s, 1, v)


@pytest.mark.parametrize("kw", [
    dict(),
    dict(positive_params=False, param_transform=lambda phi: jnp.exp(phi),
         prior=[GaussPrior(ci=[0.5, 5]), None]),
    dict(prior=[BetaPrior(ci=(0.05, 0.95)), None]),
], ids=["softplus", "exp", "beta"])
def test_binary_mask_means_exactly_one(kw):
    s, seen = svgd_fit([0, 1], **kw)
    np.testing.assert_array_equal(seen_at(seen, 1), 1.0)
    assert_exports_exact(s, 1, 1.0)


@pytest.mark.parametrize("v", [0.0, -0.5, 0.3])
def test_no_transform_regime_unchanged(v):
    s, seen = svgd_fit([(1, v)], positive_params=False,
                       prior=[GaussPrior(ci=[0.5, 5]), None])
    assert s.param_transform is None
    np.testing.assert_array_equal(seen_at(seen, 1), v)
    np.testing.assert_array_equal(np.asarray(s.fixed_values)[1], v)
    assert_exports_exact(s, 1, v)


def test_user_theta_init_fixed_column_is_overwritten():
    init = np.full((4, 2), 3.0)
    s, seen = svgd_fit([(1, 0.25)], theta_init=init)
    np.testing.assert_array_equal(seen_at(seen, 1), 0.25)
    np.testing.assert_array_equal(
        np.asarray(s.get_results()['history'])[0][:, 1], 0.25)


def test_exposure_scales_the_exact_fixed_value():
    alpha = np.array([1.0, 10.0, 100.0, 1000.0])
    for v in (1e-10, 0.0, 2.0):
        s, seen = svgd_fit([(1, v)], exposure=alpha, exposure_param_index=1)
        got = np.unique(np.concatenate([x[:, 1] for x in seen]))
        np.testing.assert_array_equal(np.sort(got), np.unique(alpha * v))


def test_pairwise_hdr_plot_evaluates_at_the_fixed_value():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spy3, seen = None, []

    def spy(theta, times, rewards=None):
        jax.debug.callback(
            lambda t: seen.append(np.atleast_2d(np.asarray(t)).copy()), theta)
        return jnp.exp(-0.1 * jnp.sum(theta[..., :2])) * jnp.ones_like(times), jnp.zeros(2)

    s = SVGD(model=spy, observed_data=DATA, theta_dim=3,
             prior=[gauss(), gauss(), None], fixed=[(2, 1e-8)], n_particles=8,
             n_iterations=3, progress=False, seed=1)
    s.optimize()
    seen.clear()
    try:
        s.plot_pairwise_hdr()
    finally:
        plt.close("all")
    np.testing.assert_array_equal(seen_at(seen, 2), 1e-8)


def test_same_model_object_fits_with_different_fixed_values_and_sets():
    # Needs fits to be independent of earlier fits on the same model object.
    spy, seen = make_spy()
    for fixed, slot, v in (([(1, 0.3)], 1, 0.3), ([(1, 2.0)], 1, 2.0),
                           ([(0, 0.7)], 0, 0.7)):
        seen.clear()
        prior = [gauss(), None] if slot == 1 else [None, gauss()]
        s = SVGD(model=spy, observed_data=DATA, theta_dim=2, prior=prior,
                 fixed=fixed, n_particles=4, n_iterations=2, progress=False,
                 seed=1)
        s.optimize()
        np.testing.assert_array_equal(seen_at(seen, slot), v)


# ------------------------------------------------------------------- MCMC
def mcmc_run(fixed, prior=None, **kw):
    seen = []

    def log_prob(theta):
        jax.debug.callback(
            lambda t: seen.append(np.atleast_2d(np.asarray(t)).copy()), theta)
        return -0.5 * jnp.sum((theta[0] - 1.0) ** 2)

    kw.setdefault('positive_params', True)
    m = MCMC(log_prob_fn=log_prob,
             prior=prior if prior is not None else [gauss(), None],
             theta_dim=2, fixed=fixed, n_samples=60, n_chains=2, burn_in=10,
             proposal_scale=0.5, verbose=False, progress=False, seed=0, **kw)
    m.run()
    return m, seen


def assert_mcmc_exports_exact(m, slot, v):
    r = m.get_results()
    np.testing.assert_array_equal(np.asarray(r['particles'])[:, slot], v)
    assert float(np.asarray(r['theta_mean'])[slot]) == v
    assert float(np.asarray(r['theta_std'])[slot]) == 0.0
    assert r['hpd_lower'][slot] == v and r['hpd_upper'][slot] == v


@pytest.mark.parametrize("v", VALUES)
def test_mcmc_positive_params_delivers_fixed_value(v):
    m, seen = mcmc_run([(1, v)])
    np.testing.assert_array_equal(seen_at(seen, 1), v)
    assert_mcmc_exports_exact(m, 1, v)


@pytest.mark.parametrize("v", [2.0, 0.0, 0.3])
def test_mcmc_custom_transform_delivers_fixed_value(v):
    m, seen = mcmc_run([(1, v)], positive_params=False,
                       param_transform=lambda phi: jnp.exp(phi),
                       prior=[GaussPrior(ci=[0.5, 5]), None])
    np.testing.assert_array_equal(seen_at(seen, 1), v)
    assert_mcmc_exports_exact(m, 1, v)


def test_mcmc_binary_mask_means_exactly_one():
    m, seen = mcmc_run([0, 1], positive_params=False,
                       param_transform=lambda phi: jnp.exp(phi),
                       prior=[GaussPrior(ci=[0.5, 5]), None])
    np.testing.assert_array_equal(seen_at(seen, 1), 1.0)
    assert_mcmc_exports_exact(m, 1, 1.0)


def _module_level_log_prob(theta):
    return -0.5 * jnp.sum((theta[0] - 1.0) ** 2)


@pytest.mark.parametrize("v", [1e-10, 0.3])
def test_mcmc_processes_mode_exports_fixed_value(v):
    # Chains run in child processes (a spy cannot record there): the chain
    # values themselves must hold the fixed value.
    m = MCMC(log_prob_fn=_module_level_log_prob, prior=[gauss(), None],
             theta_dim=2, fixed=[(1, v)], n_samples=40, n_chains=2, burn_in=5,
             proposal_scale=0.5, positive_params=True, parallel='processes',
             verbose=False, progress=False, seed=0)
    m.run()
    assert_mcmc_exports_exact(m, 1, v)


@pytest.mark.parametrize("v", [1e-10, 0.3])
def test_mcmc_model_mode_delivers_fixed_value(v):
    spy, seen = make_spy()
    m = MCMC(model=spy, observed_data=DATA, prior=[gauss(), None], theta_dim=2,
             fixed=[(1, v)], n_samples=40, n_chains=2, burn_in=5,
             proposal_scale=0.5, positive_params=True, verbose=False,
             progress=False, seed=0)
    m.run()
    np.testing.assert_array_equal(seen_at(seen, 1), v)
    assert_mcmc_exports_exact(m, 1, v)


# ------------------------------------------- the fit is the fit at that value
def chain2():
    """S -> a; a --theta0--> absorbing; a --theta1--> b --theta0--> absorbing."""
    g = Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([1])
    b = g.find_or_create_vertex([2])
    c = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0)
    a.add_edge(c, [1.0, 0.0])
    a.add_edge(b, [0.0, 1.0])
    b.add_edge(c, [1.0, 0.0])
    return g


def chain1():
    """The chain2 model with theta1 = 0: a single exponential stage."""
    g = Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([1])
    c = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0)
    a.add_edge(c, [1.0])
    return g


REAL_DATA = np.array([0.2, 0.5, 0.9, 1.4, 0.3, 0.7, 1.1, 0.6])
REAL_KW = dict(n_particles=6, n_iterations=12, learning_rate=0.01,
               progress=False, seed=1)


def test_fixed_at_zero_equals_the_reduced_model():
    init1 = np.linspace(-0.5, 1.5, 6).reshape(6, 1)
    init2 = np.concatenate([init1, np.zeros((6, 1))], axis=1)
    full = SVGD(model=Graph.pmf_and_moments_from_graph(chain2(), nr_moments=2, theta_dim=2),
                observed_data=REAL_DATA, theta_dim=2, prior=[gauss(), None],
                fixed=[(1, 0.0)], theta_init=init2, **REAL_KW)
    full.optimize()
    red = SVGD(model=Graph.pmf_and_moments_from_graph(chain1(), nr_moments=2, theta_dim=1),
               observed_data=REAL_DATA, theta_dim=1, prior=gauss(),
               theta_init=init1, **REAL_KW)
    red.optimize()
    pf = np.asarray(full.get_results()['particles'])
    pr = np.asarray(red.get_results()['particles'])
    assert np.all(np.isfinite(pf)) and np.all(pf[:, 1] == 0.0)
    # The extra zero-weight edge can change the arithmetic at rounding level.
    np.testing.assert_allclose(pf[:, 0], pr[:, 0], rtol=1e-9)


@pytest.mark.parametrize("v", [2.0, 0.3, 5.0])
def test_fixed_equals_hard_coded_value(v):
    def hard_coded():
        base = Graph.pmf_and_moments_from_graph(chain2(), nr_moments=2, theta_dim=2)

        def model(theta, times, rewards=None):
            return base(jnp.asarray(theta).at[..., 1].set(v), times)
        return model

    a = SVGD(model=Graph.pmf_and_moments_from_graph(chain2(), nr_moments=2, theta_dim=2),
             observed_data=REAL_DATA, theta_dim=2, prior=[gauss(), None],
             fixed=[(1, v)], **REAL_KW)
    a.optimize()
    b = SVGD(model=hard_coded(), observed_data=REAL_DATA, theta_dim=2,
             prior=[gauss(), None], fixed=[(1, 7.0)], **REAL_KW)
    b.optimize()
    np.testing.assert_array_equal(np.asarray(a.get_results()['particles'])[:, 0],
                                  np.asarray(b.get_results()['particles'])[:, 0])


# --------------------------------------------------------------- epoch models
L = 3
EPOCHS = [0.0, 0.4]


@pytest.fixture(scope="module")
def joint_rows():
    g, idx = coalescent_indexed(2)
    jc = g.joint_prob_graph(idx, discrete=False, reward_limit=L)
    p = normalised([p_n2_two_epoch(k, 2.0, 1.0, 6.0, 1.0, EPOCHS[1])
                    for k in range(L + 1)])
    rows = [[k, 0] for k in range(L + 1) for _ in range(int(round(p[k] * 200)))]
    return jc, rows


def spied_epoch_model(jc, rows, **kw):
    em = jc.epoch_model(rows, epoch_starts=EPOCHS, **kw)
    seen = []
    inner = em.model

    def spy(theta, *a, **k):
        jax.debug.callback(
            lambda t: seen.append(np.atleast_2d(np.asarray(t)).copy()), theta)
        return inner(theta, *a, **k)

    for name in ('_tying_info', '_precondition_output'):
        if hasattr(inner, name):
            setattr(spy, name, getattr(inner, name))
    em.model = spy
    return em, seen


EKW = dict(n_particles=4, n_iterations=2, learning_rate=0.01, progress=False,
           seed=1)


# 1e-6 does not round-trip through softplus (it failed bitwise before the
# fix). Values below the 1e-7 finite-difference step are not used here: the
# epoch model built by epoch_model() probes fixed slots too, and a probe
# below zero is a separate defect (gradient-step plan).
@pytest.mark.parametrize("v", [1e-6, 2.0])
def test_epoch_model_receives_the_fixed_value(joint_rows, v):
    jc, rows = joint_rows
    em, seen = spied_epoch_model(jc, rows)
    em.fit(fixed=[(1, v), (3, v)], prior=[gauss(), None] * 2, **EKW)
    np.testing.assert_array_equal(seen_at(seen, 1), v)
    np.testing.assert_array_equal(seen_at(seen, 3), v)


@pytest.mark.parametrize("v", [1e-6, 2.0])
def test_epoch_model_with_exposure_receives_the_fixed_value(joint_rows, v):
    # On the epoch path the exposure scaling happens inside the model, so the
    # model INPUT is the unscaled fixed value.
    jc, rows = joint_rows
    alpha = np.linspace(1.0, 3.0, len(rows))
    em, seen = spied_epoch_model(jc, rows, exposure=alpha, exposure_param_index=1)
    em.fit(fixed=[(1, v), (3, v)], prior=[gauss(), None] * 2, **EKW)
    np.testing.assert_array_equal(seen_at(seen, 1), v)
    np.testing.assert_array_equal(seen_at(seen, 3), v)


@pytest.fixture
def spy_on_svgd_models(monkeypatch):
    """Record the theta every SVGD-held model is called with (Graph.svgd
    builds its model internally, so the spy is installed on the model handed
    to SVGD.__init__)."""
    seen = []
    orig = SVGD.__init__

    def wrap(model):
        def spy(theta, *a, **k):
            jax.debug.callback(
                lambda t: seen.append(np.atleast_2d(np.asarray(t)).copy()), theta)
            return model(theta, *a, **k)
        spy.__dict__.update(getattr(model, '__dict__', {}))
        return spy

    def init(self, *a, **k):
        if k.get('model') is not None:
            k['model'] = wrap(k['model'])
        elif a:
            a = (wrap(a[0]),) + a[1:]
        return orig(self, *a, **k)

    monkeypatch.setattr(SVGD, '__init__', init)
    return seen


@pytest.mark.parametrize("exposure", [False, True], ids=["plain", "exposure"])
def test_graph_svgd_epoch_fit_below_the_old_floor(joint_rows, spy_on_svgd_models,
                                                  exposure):
    # Graph.svgd bakes the fixed slots into the epoch model's backward, so a
    # value below the 1e-7 finite-difference step works here (unlike a model
    # built by Graph.epoch_model, which probes fixed slots).
    jc, rows = joint_rows
    kw = {}
    if exposure:
        kw = dict(exposure=np.linspace(1.0, 3.0, len(rows)), exposure_param_index=1)
    s = jc.svgd(rows, epoch_starts=EPOCHS, fixed=[(1, 1e-10)],
                prior=[gauss(), None], n_particles=4, n_iterations=2,
                learning_rate=0.01, progress=False, seed=1, **kw)
    seen = spy_on_svgd_models
    for slot in (1, 3):
        np.testing.assert_array_equal(seen_at(seen, slot), 1e-10)
        assert float(np.asarray(s.theta_mean)[slot]) == 1e-10
        np.testing.assert_array_equal(
            np.asarray(s.get_results()['particles'])[:, slot], 1e-10)


def test_tied_epochs_still_export_master_values(joint_rows):
    jc, rows = joint_rows
    s = jc.svgd(rows, epoch_starts=EPOCHS, fixed=[(1, 0.3)], tied=[(0, [0, 1])],
                prior=[gauss(), None], n_particles=6, n_iterations=3,
                learning_rate=0.01, progress=False, seed=1)
    s2m = s.model._tying_info['slave_to_master']
    assert s2m
    r = s.get_results()
    P = np.asarray(r['particles'])
    for slave, master in s2m.items():
        np.testing.assert_array_equal(P[:, slave], P[:, master])
        assert float(np.asarray(r['theta_mean'])[slave]) == pytest.approx(
            float(np.asarray(r['theta_mean'])[master]), rel=1e-12)
        assert float(np.asarray(s.theta_mean)[slave]) == pytest.approx(
            float(np.asarray(s.theta_mean)[master]), rel=1e-12)
    for slot in (1, 3):                      # user-fixed in both epochs
        assert float(np.asarray(r['theta_mean'])[slot]) == 0.3
        np.testing.assert_array_equal(P[:, slot], 0.3)
    s.summary()


# ------------------------------------------------ preconditioner, MAP, refine
@pytest.mark.parametrize("precond", ['jacobian', 'fisher', 'none'])
@pytest.mark.parametrize("prior", ["softplus", "beta"])
def test_preconditioner_evaluations_see_the_fixed_value(precond, prior):
    pr = [gauss(), None] if prior == "softplus" else [BetaPrior(ci=(0.05, 0.95)), None]
    s, seen = svgd_fit([(1, 1e-10)], prior=pr, preconditioner=precond,
                       n_particles=6)
    np.testing.assert_array_equal(seen_at(seen, 1), 1e-10)


def test_map_and_refine_keep_fixed_values():
    base = Graph.pmf_and_moments_from_graph(chain2(), nr_moments=2, theta_dim=2)
    seen = []

    def spied(theta, times, rewards=None):
        jax.debug.callback(
            lambda t: seen.append(np.atleast_2d(np.asarray(t)).copy()), theta)
        return base(theta, times)

    s = SVGD(model=spied, observed_data=REAL_DATA, theta_dim=2,
             prior=[gauss(), None], fixed=[(1, 1e-10)], **REAL_KW)
    s.optimize()
    theta_map, _ = s.map_estimate_from_particles()
    assert float(theta_map[1]) == 1e-10
    seen.clear()
    assert np.isfinite(float(s.log_likelihood(refine=True)))
    np.testing.assert_array_equal(seen_at(seen, 1), 1e-10)


# ------------------------------------------------------ validation and guards
@pytest.mark.parametrize("bad", [float('nan'), float('inf'), -float('inf')])
def test_non_finite_fixed_value_raises(bad):
    spy, _ = make_spy()
    with pytest.raises(ValueError, match="must be finite"):
        SVGD(model=spy, observed_data=DATA, theta_dim=2, prior=[gauss(), None],
             fixed=[(1, bad)], n_particles=4, n_iterations=1, progress=False)
    with pytest.raises(ValueError, match="must be finite"):
        MCMC(log_prob_fn=lambda t: 0.0 * jnp.sum(t), prior=[gauss(), None],
             theta_dim=2, fixed=[(1, bad)], n_samples=10, n_chains=2, burn_in=2,
             positive_params=True, verbose=False, progress=False)


def test_bool_and_duplicate_indices_raise():
    spy, _ = make_spy()
    common = dict(model=spy, observed_data=DATA, theta_dim=2,
                  prior=[gauss(), None], n_particles=4, n_iterations=1,
                  progress=False)
    with pytest.raises(TypeError, match="must be integer"):
        SVGD(fixed=[(True, 0.3)], **common)
    with pytest.raises(ValueError, match="more than once"):
        SVGD(fixed=[(1, 0.3), (1, 0.4)], **common)


def test_all_parameters_fixed_raises_without_verbose():
    spy, _ = make_spy()
    with pytest.raises(ValueError, match="All parameters are fixed"):
        SVGD(model=spy, observed_data=DATA, theta_dim=2, prior=[None, None],
             fixed=[(0, 1.0), (1, 0.3)], n_particles=4, n_iterations=1,
             progress=False, verbose=False)


def test_coupled_custom_transform_with_fixed_raises():
    spy, _ = make_spy()
    with pytest.raises(ValueError, match="couples parameters"):
        SVGD(model=spy, observed_data=DATA, theta_dim=2,
             prior=[GaussPrior(ci=[0.5, 5]), None], fixed=[(1, 0.3)],
             positive_params=False,
             param_transform=lambda phi: jnp.cumsum(jnp.exp(phi), axis=-1)[..., ::-1],
             n_particles=4, n_iterations=1, progress=False)


def test_custom_transform_non_finite_at_placeholder_raises():
    spy, _ = make_spy()
    with pytest.raises(ValueError, match="not finite"):
        SVGD(model=spy, observed_data=DATA, theta_dim=2,
             prior=[GaussPrior(ci=[0.5, 5]), None], fixed=[(1, 0.3)],
             positive_params=False,
             param_transform=lambda phi: jnp.log(phi),
             n_particles=4, n_iterations=1, progress=False)


def _fit3(fixed):
    spy, _ = make_spy(3)
    prior = [None if i in dict(fixed) else gauss() for i in range(3)]
    s = SVGD(model=spy, observed_data=DATA, theta_dim=3, prior=prior,
             fixed=fixed, n_particles=4, n_iterations=1, progress=False, seed=1)
    s.optimize()
    return s


@pytest.mark.parametrize("a, b, ok", [(2.0, 2.0, True), (2.0, 3.0, False),
                                      (800.0, 900.0, False), (800.0, 800.0, True),
                                      (0.0, 5e-13, False)])
def test_nested_fixed_check_compares_parameter_values(a, b, ok):
    full = _fit3([(1, a)])
    nested = _fit3([(1, b), (2, 1.0)])       # a strict superset of fixed slots
    if ok:
        _check_nested_fixed(full, nested, strict=True)
    else:
        with pytest.raises(ValueError, match="different values"):
            _check_nested_fixed(full, nested, strict=True)


def test_placeholder_does_not_influence_free_slots(monkeypatch):
    import phasic.svgd as svgd_mod

    def run():
        base = Graph.pmf_and_moments_from_graph(chain2(), nr_moments=2, theta_dim=2)
        # A single (full-vector) prior is evaluated at the fixed slot's
        # placeholder too; being separable it must not affect the free slot.
        s = SVGD(model=base, observed_data=REAL_DATA, theta_dim=2,
                 prior=gauss(), fixed=[(1, 2.0)], **REAL_KW)
        s.optimize()
        return np.asarray(s.get_results()['particles'])[:, 0]

    ref = run()
    monkeypatch.setattr(svgd_mod, '_FIXED_PLACEHOLDER', 0.7)
    np.testing.assert_array_equal(run(), ref)
