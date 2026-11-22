"""GammaPrior: parameterisations, density, sampling, and use in SVGD/MCMC."""
import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import phasic
from phasic import GammaPrior
from phasic.mcmc import MCMC

from test_notebook_model_recovery import coalescent_graph

SOFTPLUS = lambda phi: jnp.maximum(jax.nn.softplus(phi), 1e-9)


def test_exported_from_package():
    assert phasic.GammaPrior is GammaPrior


@pytest.mark.parametrize("kw, shape, rate", [
    (dict(shape=2.0, rate=0.5), 2.0, 0.5),
    (dict(mean=4.0, std=2.0), 4.0, 1.0),
    (dict(shape=1, rate=3), 1.0, 3.0),
])
def test_parameterisations(kw, shape, rate):
    p = GammaPrior(**kw)
    assert p.shape == pytest.approx(shape) and p.rate == pytest.approx(rate)


@pytest.mark.parametrize("ci, prob", [((1.0, 10.0), 0.95), ((0.01, 0.02), 0.9),
                                      ((1e-5, 1.0), 0.95)])
def test_ci_reproduces_the_interval(ci, prob):
    p = GammaPrior(ci=ci, prob=prob)
    d = stats.gamma(p.shape, scale=1 / p.rate)
    np.testing.assert_allclose(d.ppf([(1 - prob) / 2, (1 + prob) / 2]), ci, rtol=1e-8)


@pytest.mark.parametrize("kw, exc", [
    (dict(), ValueError), (dict(shape=1.0), ValueError),
    (dict(shape=1.0, rate=1.0, mean=1.0), ValueError),
    (dict(shape=0.0, rate=1.0), ValueError), (dict(shape=1.0, rate=-1.0), ValueError),
    (dict(shape=np.inf, rate=1.0), ValueError), (dict(shape=True, rate=1.0), TypeError),
    (dict(mean=1.0, std=0.0), ValueError),
    (dict(ci=(2.0, 1.0)), ValueError), (dict(ci=(0.0, 1.0)), ValueError),
    (dict(ci=(1.0, 2.0), prob=1.0), ValueError),
    (dict(ci=(1.0, 1.0 + 1e-9)), ValueError),    # needs shape > 1e8
])
def test_invalid_specifications_rejected(kw, exc):
    with pytest.raises(exc, match="GammaPrior"):
        GammaPrior(**kw)


def test_log_density_matches_scipy_up_to_a_constant():
    p = GammaPrior(shape=2.5, rate=0.7)
    x = np.array([0.1, 1.0, 3.0, 9.0])
    ours = np.array([float(p(jnp.array([v]))) for v in x])     # no transform
    ref = stats.gamma(2.5, scale=1 / 0.7).logpdf(x)
    np.testing.assert_allclose(ours - ours[0], ref - ref[0], rtol=1e-12, atol=1e-12)


def test_gradient_under_transform_includes_jacobian():
    p = GammaPrior(shape=2.5, rate=0.7)
    p._transform = SOFTPLUS
    for phi in (-2.0, 0.3, 4.0):
        th = float(jax.nn.softplus(phi))
        dth = float(jax.nn.sigmoid(phi))
        # d/dphi [ (a-1) log th - b th + log dth/dphi ]
        want = ((2.5 - 1) / th - 0.7) * dth + (1 - dth)
        got = float(jax.grad(lambda f: p(f))(jnp.array([phi]))[0])
        assert got == pytest.approx(want, rel=1e-10)


def test_samples_follow_gamma():
    p = GammaPrior(shape=2.0, rate=0.5)
    pvals = [stats.kstest(np.asarray(p.sample(jax.random.PRNGKey(s), (20_000,))),
                          stats.gamma(2.0, scale=2.0).cdf).pvalue for s in range(10)]
    assert np.median(pvals) > 0.1 and min(pvals) > 1e-4, pvals


def test_transformed_samples_round_trip_to_positive_theta():
    p = GammaPrior(shape=3.0, rate=1.0)
    p._transform = SOFTPLUS
    key = jax.random.PRNGKey(0)
    theta = np.asarray(SOFTPLUS(p.sample(key, (5_000, 1)))).ravel()
    raw = np.asarray(jax.random.gamma(key, 3.0, (5_000, 1))).ravel()
    np.testing.assert_allclose(theta, raw, rtol=1e-9)


def test_plot_density_integrates_to_about_one():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = GammaPrior(shape=2.0, rate=0.5).plot(return_ax=True)
    x, y = ax.get_lines()[0].get_data()
    plt.close("all")
    assert abs(np.trapezoid(y, x) - (1 - 2e-4)) < 1e-3


@pytest.fixture(scope="module")
def coal_data():
    random.seed(2)
    np.random.seed(2)
    g = coalescent_graph(3)
    g.update_weights([2.0])
    return g, g.sample(300)


def test_svgd_fit_with_gamma_prior(coal_data):
    g, data = coal_data
    s = g.svgd(data, prior=GammaPrior(mean=2.0, std=1.5), n_particles=16,
               n_iterations=30, learning_rate=0.01, progress=False, seed=1)
    theta = np.asarray(s.particles).ravel()
    assert np.all(np.isfinite(theta)) and theta.min() > 0


def test_mcmc_with_gamma_prior_recovers_prior_mean():
    # Flat likelihood: the chain samples the prior (mean shape/rate = 4).
    m = MCMC(log_prob_fn=lambda t: 0.0 * jnp.sum(t), prior=GammaPrior(shape=4.0, rate=1.0),
             theta_dim=1, n_samples=4000, n_chains=4, burn_in=500, proposal_scale=0.5,
             positive_params=True, verbose=False, progress=False, seed=0)
    m.run()
    r = m.get_results()
    assert abs(float(np.asarray(r['theta_mean']).ravel()[0]) - 4.0) < 0.4


@pytest.mark.parametrize("make", [
    lambda: GammaPrior(shape=2.0, rate=1.0),
    lambda: phasic.LogGaussPrior(ci=(0.5, 5.0)),
    lambda: phasic.HalfCauchyPrior(scale=1.0),
], ids=["gamma", "loggauss", "halfcauchy"])
def test_positive_support_priors_need_positive_params(coal_data, make):
    g, data = coal_data
    with pytest.raises(ValueError, match="positive support.*positive_params=False"):
        g.svgd(data, prior=make(), positive_params=False,
               n_particles=4, n_iterations=1, progress=False)
    with pytest.raises(ValueError, match="positive support.*positive_params=False"):
        g.svgd(data, prior=[make()], positive_params=False,
               n_particles=4, n_iterations=1, progress=False)
    with pytest.raises(ValueError, match="positive support.*positive_params=False"):
        MCMC(log_prob_fn=lambda t: -jnp.sum(t ** 2), prior=make(),
             theta_dim=1, n_samples=10, n_chains=2, burn_in=2, positive_params=False,
             verbose=False, progress=False)
    # Under positive_params=True they are accepted.
    MCMC(log_prob_fn=lambda t: -jnp.sum(t ** 2), prior=make(),
         theta_dim=1, n_samples=10, n_chains=2, burn_in=2, positive_params=True,
         verbose=False, progress=False)
