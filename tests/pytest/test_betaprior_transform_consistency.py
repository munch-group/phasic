"""A BetaPrior's (0, 1) transform is applied consistently (bug 3).

A BetaPrior maps its parameter to (0, 1) with a sigmoid and hard-codes that
transform's Jacobian and inverse. SVGD's single-prior form gave the MODEL
softplus while the prior used its sigmoid (particles up to 3.75; a discrete
model was handed probabilities above 1), and MCMC handed the prior softplus
as well (NaN log-prior above 1, frozen chains, the wrong density). Now one
resolver (``svgd._resolve_param_transform``) decides each slot's transform,
and the model, the prior and the initial points all use it.
Plan: bug3-betaprior-transform-plan.md (test numbers refer to its section 3).

Spy models record the theta they are called with. MCMC tests call ``MCMC``
directly with pure-JAX models (``Graph.mcmc`` is not seed-reproducible).
"""
import math
import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from phasic import (BetaPrior, ExpStepSize, GammaPrior, GaussPrior, Graph,
                    HalfCauchyPrior, LogGaussPrior, SVGD)
from phasic.mcmc import MCMC
from phasic.svgd import _sigmoid, _softplus_transform

DATA = np.array([0.5, 1.0, 1.5, 2.0])
PHI_GRID = jnp.array([-40.0, -10.0, -1.0, 0.0, 0.3, 2.0, 10.0, 40.0])


# ----------------------------------------------------------------- helpers
def make_spy():
    seen = []

    def spy(theta, times, rewards=None):
        jax.debug.callback(
            lambda t: seen.append(np.atleast_2d(np.asarray(t)).copy()), theta)
        return jnp.ones_like(times), jnp.zeros(2)

    return spy, seen


def make_mcmc_spy():
    seen = []

    def spy(theta, times):
        jax.debug.callback(
            lambda t: seen.append(np.atleast_2d(np.asarray(t)).copy()), theta)
        return jnp.ones_like(times) + 0.0 * jnp.sum(theta)

    return spy, seen


def flat(theta, times):
    return jnp.ones_like(times) + 0.0 * jnp.sum(theta)


def softplus_ref(phi):
    """The default positive transform, written out (a reference, so the
    guards below also pass on code that predates ``_softplus_transform``)."""
    return jnp.maximum(jax.nn.softplus(phi), 1e-9)


def same_on_grid(f, g, dim):
    grid = jnp.stack([PHI_GRID] * dim, axis=-1)
    return np.array_equal(np.asarray(f(grid)), np.asarray(g(grid)))


def all_seen(seen, min_calls=3):
    assert len(seen) >= min_calls, "the spy recorded too few model calls"
    return np.concatenate(seen, axis=0)


def in_unit(x):
    x = np.asarray(x)
    return bool(np.all((x > 0.0) & (x < 1.0)))


def beta():
    return BetaPrior(ci=(0.05, 0.95))


def gauss():
    return GaussPrior(ci=[0.5, 5], lower=0.0)


def gauss_free():
    """For positive_params=False (no truncation allowed there)."""
    return GaussPrior(ci=[0.5, 5])


def svgd(prior, dim, model=None, n_iterations=3, n_particles=8, **kw):
    spy, seen = make_spy()
    s = SVGD(model=model or spy, observed_data=DATA, theta_dim=dim,
             prior=prior, n_particles=n_particles, n_iterations=n_iterations,
             progress=False, seed=1, learning_rate=0.01, **kw)
    s.optimize()
    return s, seen


def mcmc(prior, dim, model=flat, n_samples=2000, burn_in=500, seed=0, **kw):
    kw.setdefault('proposal_scale', 1.5)
    m = MCMC(model=model, observed_data=np.array([1.0, 2.0]), prior=prior,
             theta_dim=dim, n_samples=n_samples, n_chains=4, burn_in=burn_in,
             verbose=False, progress=False, seed=seed, **kw)
    return m


def ess_1d(x):
    """Effective sample size of one chain (Geyer initial positive sequence)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    var = x.var()
    if var == 0:
        return 1.0
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (n * var)
    s, k = 0.0, 1
    while k + 1 < n:
        pair = acf[k] + acf[k + 1]
        if pair <= 0:
            break
        s += pair
        k += 2
    return n / (1.0 + 2.0 * s)


def chains_theta(m):
    """(n_chains, n_samples, dim) on the parameter scale."""
    phi = np.asarray(m._chains_phi)
    if m.param_transform is None:
        return phi
    return np.asarray(m.param_transform(jnp.asarray(phi)))


def thinned_draws(m, slot):
    """Pool chains thinned by k = ceil(2N / ESS), N draws per chain."""
    th = chains_theta(m)[:, :, slot]
    n = th.shape[1]
    ess = sum(ess_1d(c) for c in th)
    k = max(1, math.ceil(2 * th.size / ess))
    return np.concatenate([c[::k] for c in th]), ess, n


def initial_log_probs(m):
    return np.array([float(m._log_prob(jnp.asarray(t))) for t in m.theta_init])


def geom():
    """Discrete geometric chain: per-step absorption probability theta[0]."""
    g = Graph(1)
    s = g.starting_vertex()
    v = g.find_or_create_vertex([1])
    a = g.find_or_create_vertex([0])
    s.add_edge(v, 1.0)
    v.add_edge(a, [1.0])
    return g


GEOM_TRUE = 0.3


@pytest.fixture(scope="module")
def geom_data():
    rng = np.random.default_rng(7)
    data = rng.geometric(GEOM_TRUE, size=200).astype(float)
    pr = BetaPrior(ci=(0.5, 0.95))
    # The prior excludes the truth.
    assert stats.beta(pr.alpha, pr.beta).cdf(GEOM_TRUE) < 1e-3
    post = stats.beta(pr.alpha + data.size, pr.beta + (data - 1).sum())
    return data, post


# --------------------------------------- test 1: single BetaPrior in SVGD
@pytest.mark.parametrize("dim", [1, 2])
def test_svgd_single_betaprior_stays_in_unit_interval(dim):
    pr = beta()
    s, seen = svgd(pr, dim)
    assert in_unit(all_seen(seen))
    assert in_unit(s.param_transform(s.theta_init))
    r = s.get_results()
    assert in_unit(r['particles'])
    assert in_unit(s.theta_mean)
    assert in_unit(r['hpd_lower']) and in_unit(r['hpd_upper'])
    grid = jnp.stack([PHI_GRID] * dim, axis=-1)
    np.testing.assert_array_equal(np.asarray(s.param_transform(grid)),
                                  np.asarray(pr._transform(grid)))


# ------------------------------------- test 2: single == list, same start
@pytest.mark.parametrize("dim", [1, 2])
def test_svgd_single_and_list_forms_agree(dim):
    theta_init = np.random.default_rng(3).normal(size=(8, dim))
    s1, _ = svgd(beta(), dim, theta_init=theta_init, n_iterations=10)
    s2, _ = svgd([beta() for _ in range(dim)], dim, theta_init=theta_init,
                 n_iterations=10)
    np.testing.assert_array_equal(np.asarray(s1.particles),
                                  np.asarray(s2.particles))
    np.testing.assert_array_equal(np.asarray(s1.get_results()['particles']),
                                  np.asarray(s2.get_results()['particles']))


# ----------------------------------- tests 3 and 4: a real discrete model
def test_svgd_geometric_model_single_betaprior(geom_data):
    data, post = geom_data
    np.random.seed(1)
    random.seed(1)
    s = geom().svgd(observed_data=jnp.asarray(data), discrete=True,
                    theta_dim=1, prior=BetaPrior(ci=(0.5, 0.95)),
                    n_particles=40, n_iterations=150,
                    learning_rate=ExpStepSize(first_step=0.05,
                                              last_step=0.005, tau=50.0),
                    progress=False, seed=1)
    P = np.asarray(s.get_results()['particles']).ravel()
    assert in_unit(P)
    # No truth-in-interval assertion: SVGD is known to be under-dispersed.
    assert abs(P.mean() - post.mean()) < post.std()


def test_mcmc_geometric_model_single_betaprior(geom_data):
    data, post = geom_data
    model = Graph.pmf_from_graph(geom(), discrete=True)
    m = MCMC(model=model, observed_data=data, prior=BetaPrior(ci=(0.5, 0.95)),
             theta_dim=1, n_samples=2000, n_chains=4, burn_in=500,
             verbose=False, progress=False, seed=0)
    assert np.all(np.isfinite(initial_log_probs(m)))
    m.run()
    th = chains_theta(m)[:, :, 0]
    assert in_unit(th)
    ess = sum(ess_1d(c) for c in th)
    assert ess > 50
    assert abs(th.mean() - post.mean()) < 4 * post.std() / math.sqrt(ess)


# ------------------------------- test 5: MCMC samples the Beta prior right
B135 = (1.35, 1.35)


def _prior_forms():
    p = BetaPrior(ci=(0.01, 0.99))
    return {
        "single": (lambda: BetaPrior(alpha=1.35, beta=1.35), 1, {0: B135}, ()),
        "list_beta25": (lambda: [BetaPrior(alpha=2.0, beta=5.0)], 1,
                        {0: (2.0, 5.0)}, ()),
        "gauss_beta": (lambda: [gauss(), BetaPrior(alpha=2.0, beta=5.0)], 2,
                       {1: (2.0, 5.0)}, (0,)),
        "alpha_below_1": (lambda: BetaPrior(ci=(0.01, 0.99)), 1,
                          {0: (p.alpha, p.beta)}, ()),
    }


def _ks_p(m, slot, cdf):
    draws, _, _ = thinned_draws(m, slot)
    return stats.kstest(draws, cdf).pvalue


@pytest.mark.parametrize("form", list(_prior_forms()))
def test_mcmc_bounded_prior_chains_move(form):
    mk, dim, beta_slots, _ = _prior_forms()[form]
    m = mcmc(mk(), dim, n_samples=500, burn_in=200)
    assert np.all(np.isfinite(initial_log_probs(m)))
    m.run()
    assert np.all(np.asarray(m.acceptance_rate) > 0)
    th = chains_theta(m)
    for slot in beta_slots:
        assert in_unit(th[:, :, slot])


def _distributional(form):
    mk, dim, beta_slots, gauss_slots = _prior_forms()[form]
    pvals = {slot: [] for slot in list(beta_slots) + list(gauss_slots)}
    for seed in range(5):
        m = mcmc(mk(), dim, seed=seed)
        m.run()
        for slot, (a, b) in beta_slots.items():
            pvals[slot].append(_ks_p(m, slot, stats.beta(a, b).cdf))
        for slot in gauss_slots:
            g = gauss()
            tn = stats.truncnorm((0 - g.mu) / g.sigma, np.inf, g.mu, g.sigma)
            pvals[slot].append(_ks_p(m, slot, tn.cdf))
    for slot, ps in pvals.items():
        assert np.median(ps) > 0.1, (form, slot, ps)


def test_mcmc_single_betaprior_samples_beta():
    _distributional("single")


@pytest.mark.slow
@pytest.mark.parametrize("form", ["list_beta25", "gauss_beta", "alpha_below_1"])
def test_mcmc_bounded_prior_samples_beta_slow(form):
    _distributional(form)


# ------------------------------------------------- test 6: fixed slots
FIXED_CASES = [
    ("single", lambda: beta(), [(0, 5.0)], 0, 5.0),
    ("none_beta", lambda: [None, beta()], [(0, 5.0)], 0, 5.0),
    ("beta_none", lambda: [beta(), None], [(1, 3.0)], 1, 3.0),
    ("single_mask", lambda: beta(), [1, 0], 0, 1.0),
]


@pytest.mark.parametrize("name, mk, fixed, slot, v", FIXED_CASES,
                         ids=[c[0] for c in FIXED_CASES])
def test_fixed_slot_exact_free_slot_in_unit_svgd(name, mk, fixed, slot, v):
    s, seen = svgd(mk(), 2, fixed=fixed)
    x = all_seen(seen)
    np.testing.assert_array_equal(x[:, slot], v)
    assert in_unit(x[:, 1 - slot])


@pytest.mark.parametrize("name, mk, fixed, slot, v", FIXED_CASES,
                         ids=[c[0] for c in FIXED_CASES])
def test_fixed_slot_exact_free_slot_in_unit_mcmc(name, mk, fixed, slot, v):
    spy, seen = make_mcmc_spy()
    m = mcmc(mk(), 2, model=spy, n_samples=50, burn_in=20, fixed=fixed)
    m.run()
    x = all_seen(seen)
    np.testing.assert_array_equal(x[:, slot], v)
    assert in_unit(x[:, 1 - slot])
    th = chains_theta(m)
    np.testing.assert_array_equal(th[:, :, slot], v)
    assert in_unit(th[:, :, 1 - slot])


# ------------------------------- test 7: SVGD refuses an unconfined Beta
@pytest.mark.parametrize("prior", [beta(), [gauss_free(), beta()]],
                         ids=["single", "list"])
def test_svgd_refuses_betaprior_without_transform(prior):
    spy, _ = make_spy()
    with pytest.raises(ValueError,
                       match=r"BetaPrior.*bounded interval.*positive_params=True"):
        SVGD(model=spy, observed_data=DATA, theta_dim=1 if not
             isinstance(prior, list) else 2, prior=prior,
             positive_params=False, n_particles=4, progress=False)


# -------------------------------------------- test 8: pairwise HDR plot
@pytest.fixture
def agg():
    import matplotlib
    import matplotlib.pyplot as plt
    old = matplotlib.get_backend()
    plt.switch_backend('Agg')
    yield plt
    plt.close('all')
    plt.switch_backend(old)


@pytest.mark.parametrize("prior", ["gauss_beta", "single"])
def test_pairwise_hdr_uses_per_slot_inverse(prior, agg):
    pr = [gauss(), beta()] if prior == "gauss_beta" else beta()
    s, seen = svgd(pr, 2, n_iterations=5, n_particles=12)
    # Only the hex grid may reach the spy: the MAP marker evaluates the
    # particles, whose values cover (0, 1) whatever the plot's inverse is.
    s.map_estimate_from_particles = (
        lambda unconstrained=False: (list(np.asarray(s.theta_mean)), 0.0))
    seen.clear()
    xlim = (0.5, 3.0) if prior == "gauss_beta" else (0.2, 0.8)
    s.plot_pairwise_hdr(xlim=xlim, ylim=(0.2, 0.8), n=6, center_map=False,
                        return_fig=True)
    x = all_seen(seen, min_calls=1)[:, 1]
    assert in_unit(x)
    assert x.max() >= 0.75 and x.min() <= 0.25


def test_pairwise_hdr_refuses_custom_transform(agg):
    s, _ = svgd([gauss_free(), gauss_free()], 2, positive_params=False,
                param_transform=lambda phi: jnp.exp(phi),
                n_iterations=3, n_particles=12)
    with pytest.raises(ValueError, match="unconstrained=True"):
        s.plot_pairwise_hdr(n=6, return_fig=True)
    s.plot_pairwise_hdr(n=6, unconstrained=True, return_fig=True)


# ------------------------------ test 9: _transform reassigned every time
def test_transform_reassigned_on_every_construction():
    pr = beta()
    svgd([gauss_free(), pr], 2, positive_params=False,
         param_transform=lambda phi: jnp.exp(phi))
    assert pr._transform is not None
    mcmc([gauss_free(), pr], 2, positive_params=False)
    assert pr._transform is None

    pr2 = beta()
    svgd(pr2, 1)
    assert pr2._transform is _sigmoid
    m = mcmc([gauss(), pr2], 2)          # MCMC's own rule: slot 1 sigmoid
    assert pr2._transform is _sigmoid
    assert m.param_transform._per_dim_fns == (_softplus_transform, _sigmoid)
    mcmc(pr2, 1, positive_params=False)
    assert pr2._transform is None


# --------------------------- test 10: list prior with a custom transform
def test_list_prior_custom_transform_wires_custom_function():
    custom = lambda phi: jax.nn.sigmoid(phi)
    b = beta()
    s, _ = svgd([gauss_free(), b], 2, positive_params=False,
                param_transform=custom)
    assert b._transform is custom
    assert s.param_transform is custom


# --------------------------------- test 10b: the log-density, pinned
@pytest.mark.parametrize("alpha", [0.5, 0.75, 2.0])
def test_betaprior_log_density_unclipped(alpha):
    b = BetaPrior(alpha=alpha, beta=1.3)
    b._transform = b._natural_transform
    phis = np.array([25.0, 40.0, 50.0, 500.0])
    for sign in (1.0, -1.0):
        vals = []
        for phi in sign * phis:
            got = float(b(jnp.array([phi])))
            want = (alpha * float(jax.nn.log_sigmoid(phi))
                    + 1.3 * float(jax.nn.log_sigmoid(-phi)))
            np.testing.assert_allclose(got, want, rtol=1e-12)
            vals.append(got)
        assert np.all(np.diff(vals) < 0), (sign, vals)
    v = jnp.array([0.3, -2.0])
    np.testing.assert_allclose(float(b(v)),
                               float(b(v[:1])) + float(b(v[1:])), rtol=1e-12)


# ======================================================== guards
# ---------------- guard 11: MCMC without a transform keeps its hard wall
def test_mcmc_no_transform_betaprior_hard_wall():
    a, bb = 2.0, 5.0
    pvals = []
    for seed in range(5):
        m = mcmc(BetaPrior(alpha=a, beta=bb), 1, positive_params=False,
                 proposal_scale=0.3, theta_init=np.full((4, 1), 0.3),
                 seed=seed)
        assert m.param_transform is None
        m.run()
        th = chains_theta(m)
        assert in_unit(th)
        pvals.append(_ks_p(m, 0, stats.beta(a, bb).cdf))
    assert np.median(pvals) > 0.1, pvals


# ------------------------------- guard 12: list forms keep their composite
@pytest.mark.parametrize("form", ["beta", "gauss_beta"])
def test_svgd_list_forms_keep_per_slot_composite(form):
    prior = [beta()] if form == "beta" else [gauss(), beta()]
    dim = len(prior)
    s, seen = svgd(prior, dim)
    fns = getattr(s.param_transform, '_per_dim_fns', None)
    assert fns is not None and len(fns) == dim
    expect = (_sigmoid,) if form == "beta" else (softplus_ref, _sigmoid)
    for f, e in zip(fns, expect):
        assert same_on_grid(f, e, 1)
    grid = jnp.stack([PHI_GRID] * dim, axis=-1)
    want = jnp.concatenate([f(grid[:, i:i + 1]) for i, f in enumerate(expect)],
                           axis=-1)
    np.testing.assert_array_equal(np.asarray(s.param_transform(grid)),
                                  np.asarray(want))
    x = all_seen(seen)
    assert in_unit(x[:, -1])
    if dim == 2:
        assert np.all(x[:, 0] > 0)


# --------------- guard 13: priors without a natural transform: softplus
UNBOUNDED = {
    "gauss": lambda: GaussPrior(ci=[0.5, 5], lower=0.0),
    "loggauss": lambda: LogGaussPrior(ci=[0.5, 5]),
    "gamma": lambda: GammaPrior(ci=[0.5, 5]),
    "halfcauchy": lambda: HalfCauchyPrior(scale=1.0),
}


@pytest.mark.parametrize("name", list(UNBOUNDED))
def test_unbounded_priors_get_plain_softplus(name):
    mk = UNBOUNDED[name]
    theta_init = np.random.default_rng(5).normal(size=(8, 2))
    p1 = mk()
    s1, _ = svgd(p1, 2, theta_init=theta_init, n_iterations=5)
    pl = [mk(), mk()]
    s2, _ = svgd(pl, 2, theta_init=theta_init, n_iterations=5)
    for s, priors in ((s1, [p1]), (s2, pl)):
        assert same_on_grid(s.param_transform, softplus_ref, 2)
        assert not hasattr(s.param_transform, '_per_dim_fns')
        assert all(same_on_grid(p._transform, softplus_ref, 2) for p in priors)
    np.testing.assert_array_equal(np.asarray(s1.particles),
                                  np.asarray(s2.particles))
    for prior in (mk(), [mk(), mk()]):
        m = mcmc(prior, 2, n_samples=10, burn_in=5)
        assert same_on_grid(m.param_transform, softplus_ref, 2)
        assert not hasattr(m.param_transform, '_per_dim_fns')
        for p in (prior if isinstance(prior, list) else [prior]):
            assert same_on_grid(p._transform, softplus_ref, 2)


def test_no_prior_gets_plain_softplus():
    s, _ = svgd(None, 2)
    assert same_on_grid(s.param_transform, softplus_ref, 2)
    assert not hasattr(s.param_transform, '_per_dim_fns')
    m = mcmc(None, 2, n_samples=10, burn_in=5)
    assert same_on_grid(m.param_transform, softplus_ref, 2)
    assert not hasattr(m.param_transform, '_per_dim_fns')


def test_data_prior_gets_plain_softplus():
    from phasic import DataPrior
    from test_notebook_model_recovery import coalescent_graph
    random.seed(1)
    np.random.seed(1)
    g = coalescent_graph(3)
    g.update_weights([2.0])
    data = g.sample(200)
    dp = DataPrior(g, data)
    s = g.svgd(data, prior=dp, n_particles=8, n_iterations=2,
               progress=False, learning_rate=0.001)
    assert same_on_grid(s.param_transform, softplus_ref, 1)
    assert not hasattr(s.param_transform, '_per_dim_fns')
    assert all(same_on_grid(p._transform, softplus_ref, 1)
               for p in s.prior_list if p is not None)
    dp2 = DataPrior(g, data)
    m = MCMC(model=Graph.pmf_from_graph(g), observed_data=data, prior=dp2,
             theta_dim=1, n_samples=10, n_chains=2, burn_in=5,
             verbose=False, progress=False, seed=0)
    assert same_on_grid(m.param_transform, softplus_ref, 1)
    assert not hasattr(m.param_transform, '_per_dim_fns')
    assert all(same_on_grid(p._transform, softplus_ref, 1)
               for p in m.prior_list if p is not None)


# ----------- guard 14: custom unclipped sigmoid, model and prior agree
@pytest.mark.parametrize("form", ["single", "list"])
def test_custom_unclipped_sigmoid_is_finite(form):
    custom = lambda x: jax.nn.sigmoid(x)
    b = beta()
    prior = b if form == "single" else [b]
    s, seen = svgd(prior, 1, positive_params=False, param_transform=custom)
    assert b._transform is custom
    grid = PHI_GRID[:, None]
    np.testing.assert_array_equal(np.asarray(s.param_transform(grid)),
                                  np.asarray(b._transform(grid)))
    for phi in (-40.0, 40.0):
        val = b(jnp.array([phi]))
        grad = jax.grad(lambda p: b(p))(jnp.array([phi]))
        assert np.isfinite(float(val))
        assert np.all(np.isfinite(np.asarray(grad)))


# ---------- guard 15: elementwise sigmoid with the preconditioner subset
def test_single_betaprior_preconditioner_with_fixed_slot():
    s, seen = svgd(beta(), 2, fixed=[(0, 2.0)], n_iterations=5)
    x = all_seen(seen)
    np.testing.assert_array_equal(x[:, 0], 2.0)
    assert in_unit(x[:, 1])
    assert in_unit(np.asarray(s.get_results()['particles'])[:, 1])
