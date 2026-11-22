"""GaussPrior(lower=...): truncation at zero (plan
epoch-default-prior-and-truncated-gauss-plan.md, Batch 1).

- lower=0.0 samples exactly from the Gaussian truncated to theta > 0: a
  non-positive Gaussian draw is mapped to a truncated draw by the inverse
  CDF (no clipping to 1e-6), and positive draws are bit-identical to the
  untruncated sampler.
- The log density and its gradient do not depend on lower.
- Under positive_params=True a GaussPrior must have lower=0.0; with
  positive_params=False it must have lower=None (checked by SVGD and MCMC
  at construction). The priors phasic builds itself follow the fit.
"""
import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from phasic import GaussPrior
from phasic.mcmc import MCMC
from phasic.svgd import _inverse_softplus

from test_notebook_model_recovery import coalescent_graph

SOFTPLUS = lambda phi: jnp.maximum(jax.nn.softplus(phi), 1e-9)


def _theta_samples(prior, n, seed=0):
    prior._transform = SOFTPLUS
    phi = prior.sample(jax.random.PRNGKey(seed), (n, 1))
    return np.asarray(SOFTPLUS(phi)).ravel(), np.asarray(phi).ravel()


# ---------------------------------------------------------------- sampling
def test_truncated_samples_are_positive_and_not_clipped():
    theta, _ = _theta_samples(GaussPrior(mean=0.5, std=1.0, lower=0.0), 50_000)
    assert theta.min() > 0
    # The legacy clip value must not appear as an atom.
    assert np.mean(np.isclose(theta, 1e-6, rtol=1e-3)) < 1e-3


def test_truncated_samples_follow_truncnorm():
    mu, sigma = 0.5, 1.0
    prior = GaussPrior(mean=mu, std=sigma, lower=0.0)
    prior._transform = None          # sample in theta space directly
    pvals = []
    for seed in range(10):
        x = np.asarray(prior.sample(jax.random.PRNGKey(seed), (20_000,)))
        pvals.append(stats.kstest(
            x, stats.truncnorm(-mu / sigma, np.inf, loc=mu, scale=sigma).cdf).pvalue)
    # Ten independent KS tests of a correct sampler: p-values ~ U(0,1).
    assert np.median(pvals) > 0.1 and min(pvals) > 1e-4, pvals


def test_positive_draws_bit_identical_to_untruncated():
    mu, sigma = 0.5, 1.0
    key = jax.random.PRNGKey(3)
    a = GaussPrior(mean=mu, std=sigma, lower=0.0)
    b = GaussPrior(mean=mu, std=sigma)
    a._transform = b._transform = None
    xa = np.asarray(a.sample(key, (10_000,)))
    xb = np.asarray(b.sample(key, (10_000,)))
    keep = xb > 0
    assert keep.sum() > 0 and (~keep).sum() > 0
    np.testing.assert_array_equal(xa[keep], xb[keep])


def test_all_positive_prior_identical_for_both_lower_values():
    _, phi_a = _theta_samples(GaussPrior(mean=10.5, std=1.2, lower=0.0), 2_000, seed=7)
    _, phi_b = _theta_samples(GaussPrior(mean=10.5, std=1.2), 2_000, seed=7)
    np.testing.assert_array_equal(phi_a, phi_b)


def test_untruncated_sampler_unchanged():
    # lower=None reproduces the pre-change sampler bit-for-bit, clip included.
    mu, sigma, key = 0.5, 1.0, jax.random.PRNGKey(11)
    prior = GaussPrior(mean=mu, std=sigma)
    prior._transform = SOFTPLUS
    got = np.asarray(prior.sample(key, (5_000, 1)))
    z = jax.random.normal(key, (5_000, 1)) * sigma + mu
    want = np.asarray(_inverse_softplus(jnp.maximum(z, 1e-6)))
    np.testing.assert_array_equal(got, want)
    assert np.sum(np.asarray(z) <= 0) > 0     # the clip was exercised


def test_deep_tail_prior_samples_correctly():
    # mu/sigma = -4.5: 3.4e-6 of the parent mass above zero (accepted).
    mu, sigma = -4.5, 1.0
    prior = GaussPrior(mean=mu, std=sigma, lower=0.0)
    prior._transform = None
    x = np.asarray(prior.sample(jax.random.PRNGKey(0), (20_000,)))
    assert np.all(np.isfinite(x)) and x.min() > 0
    p = stats.kstest(x, stats.truncnorm(4.5, np.inf, loc=mu, scale=sigma).cdf).pvalue
    assert p > 1e-4


def test_small_scale_parameter_floor_rate_documented():
    # _inverse_softplus still floors at 1e-6. For a rate near 1.2e-4 with
    # sigma 5e-5 the expected fraction of draws below 1e-6 is
    # (Phi(a + 1e-6/sigma) - Phi(a)) / (1 - Phi(a)), a = -mu/sigma (~4.6e-4):
    # small, and here measured rather than assumed.
    mu, sigma, n = 1.2e-4, 5e-5, 200_000
    theta, _ = _theta_samples(GaussPrior(mean=mu, std=sigma, lower=0.0), n, seed=5)
    a = -mu / sigma
    expected = (stats.norm.cdf(a + 1e-6 / sigma) - stats.norm.cdf(a)) / stats.norm.sf(a)
    floored = np.mean(theta <= 1e-6 * (1 + 1e-6))
    sd = np.sqrt(expected * (1 - expected) / n)
    assert abs(floored - expected) < 5 * sd, (floored, expected)


# ------------------------------------------------------ density / parameters
def test_log_density_and_gradient_independent_of_lower():
    a = GaussPrior(ci=[0.2, 3.0], lower=0.0)
    b = GaussPrior(ci=[0.2, 3.0])
    a._transform = b._transform = SOFTPLUS
    for phi in (jnp.array([-3.0]), jnp.array([0.1]), jnp.array([2.5])):
        assert float(a(phi)) == float(b(phi))
        assert float(jax.grad(a)(phi)[0]) == float(jax.grad(b)(phi)[0])


def test_parent_parameters_independent_of_lower():
    for kw in (dict(ci=[-1.0, 3.0]), dict(mean=0.3, std=2.0)):
        a, b = GaussPrior(lower=0.0, **kw), GaussPrior(**kw)
        assert (a.mu, a.sigma) == (b.mu, b.sigma)


@pytest.mark.parametrize("bad", [1.0, -1.0, 0.5, True, "0", np.inf])
def test_unsupported_lower_rejected(bad):
    with pytest.raises(ValueError, match="lower must be None"):
        GaussPrior(mean=1.0, std=1.0, lower=bad)


def test_integer_zero_accepted():
    assert GaussPrior(mean=1.0, std=1.0, lower=0).lower == 0.0


def test_prior_with_almost_no_positive_mass_rejected():
    with pytest.raises(ValueError, match="cannot describe a positive parameter"):
        GaussPrior(mean=-5.0, std=1.0, lower=0.0)
    with pytest.raises(ValueError, match="cannot describe a positive parameter"):
        GaussPrior(ci=[-20.0, -10.0], lower=0.0)
    GaussPrior(mean=-5.0, std=1.0)              # untruncated: fine


def test_plotted_density_integrates_to_one():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = GaussPrior(mean=0.5, std=1.0, lower=0.0).plot(return_ax=True)
    x, y = ax.get_lines()[0].get_data()
    plt.close("all")
    assert x[0] == 0.0
    mass = np.trapezoid(y, x)
    beyond = stats.norm.sf(4.0) / stats.norm.sf(-0.5)   # mass past mu+4sigma
    assert abs(mass + beyond - 1.0) < 2e-3, mass


def test_description_shows_truncation():
    assert "truncated" in GaussPrior(mean=1.0, std=1.0, lower=0.0)._describe()
    assert "truncated" not in GaussPrior(mean=1.0, std=1.0)._describe()


# ------------------------------------------------ construction-time checks
@pytest.fixture(scope="module")
def coal_data():
    random.seed(1)
    np.random.seed(1)
    g = coalescent_graph(3)
    g.update_weights([2.0])
    return g, g.sample(200)


def _svgd(g, data, **kw):
    return g.svgd(data, n_particles=8, n_iterations=2, progress=False,
                  learning_rate=0.001, **kw)


def test_svgd_rejects_untruncated_gauss_under_positive_params(coal_data):
    g, data = coal_data
    with pytest.raises(ValueError, match=r"lower=0\.0"):
        _svgd(g, data, prior=GaussPrior(ci=[0.5, 4.0]))
    with pytest.raises(ValueError, match=r"prior\[0\].*lower=0\.0"):
        _svgd(g, data, prior=[GaussPrior(ci=[0.5, 4.0])])


def test_svgd_rejects_truncated_gauss_without_positive_params(coal_data):
    g, data = coal_data
    with pytest.raises(ValueError, match="positive_params=False"):
        _svgd(g, data, prior=GaussPrior(ci=[0.5, 4.0], lower=0.0),
              positive_params=False)


def test_mcmc_checks_lower_against_positive_params():
    log_lik = lambda theta: -0.5 * jnp.sum((theta - 2.0) ** 2)
    common = dict(log_prob_fn=log_lik, theta_dim=1, n_samples=20, n_chains=2,
                  burn_in=5, verbose=False, progress=False, seed=0)
    with pytest.raises(ValueError, match=r"lower=0\.0"):
        MCMC(prior=GaussPrior(mean=1.0, std=1.0), positive_params=True, **common)
    with pytest.raises(ValueError, match="positive_params=False"):
        MCMC(prior=GaussPrior(mean=1.0, std=1.0, lower=0.0),
             positive_params=False, **common)
    MCMC(prior=GaussPrior(mean=1.0, std=1.0), positive_params=False, **common)


def test_mcmc_initialises_from_truncated_prior():
    log_lik = lambda theta: -0.5 * jnp.sum((theta - 2.0) ** 2)
    m = MCMC(log_prob_fn=log_lik, prior=GaussPrior(mean=0.2, std=1.0, lower=0.0),
             theta_dim=1, n_samples=10, n_chains=64, burn_in=2,
             positive_params=True, verbose=False, progress=False, seed=0)
    theta0 = np.asarray(SOFTPLUS(m.theta_init)).ravel()
    assert theta0.min() > 0
    assert not np.any(np.isclose(theta0, 1e-6, rtol=1e-3))


def test_standalone_evaluation_needs_no_fit():
    prior = GaussPrior(mean=1.0, std=1.0, lower=0.0)
    assert np.isfinite(float(prior(jnp.array([0.5]))))


def test_svgd_initialises_from_truncated_prior(coal_data):
    g, data = coal_data
    s = _svgd(g, data, prior=GaussPrior(mean=0.2, std=1.0, lower=0.0), seed=3)
    theta0 = np.asarray(SOFTPLUS(s.theta_init)).ravel()
    assert theta0.min() > 0
    assert not np.any(np.isclose(theta0, 1e-6, rtol=1e-3))


def test_default_prior_follows_positive_params(coal_data):
    g, data = coal_data
    s_pos = _svgd(g, data)
    assert all(p.lower == 0.0 for p in s_pos.prior_list if p is not None)
    s_free = _svgd(g, data, positive_params=False)
    assert all(p.lower is None for p in s_free.prior_list if p is not None)


def test_internal_prior_builders_take_lower(coal_data):
    from phasic import DataPrior
    g, data = coal_data
    assert all(p.lower == 0.0 for p in DataPrior(g, data) if p is not None)
    assert all(p.lower is None for p in DataPrior(g, data, lower=None) if p is not None)
    assert all(p.lower is None
               for p in g.method_of_moments(data, verbose=False, lower=None).prior
               if p is not None)
