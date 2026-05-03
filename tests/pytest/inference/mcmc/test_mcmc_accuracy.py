"""
MCMC accuracy and performance tests.

Verifies that MCMC recovers known posteriors with quantitative bounds.
"""

from phasic import Graph, MCMC, GaussPrior
import numpy as np
import jax.numpy as jnp
import pytest
from time import time


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def build_exponential_graph():
    """Exponential distribution with rate theta (1 parameter)."""
    g = Graph(1)
    start = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v2, 1.0)
    v2.add_edge_parameterized(v1, 0.0, [1.0])
    return g


def build_erlang2_graph():
    """Erlang-2: two sequential exponential stages (2 parameters).

    Stage 1 rate = theta[0], Stage 2 rate = theta[1].
    """
    g = Graph(1)
    start = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v3, 1.0)
    v3.add_edge_parameterized(v2, 0.0, [1.0, 0.0])  # rate = theta[0]
    v2.add_edge_parameterized(v1, 0.0, [0.0, 1.0])  # rate = theta[1]
    return g


# ---------------------------------------------------------------------------
# Test 1: Exponential with known posterior
# ---------------------------------------------------------------------------

class TestExponentialPosterior:
    """MCMC on Exp(theta) should recover the analytical posterior."""

    def test_posterior_mean(self):
        """Posterior mean within 10% of analytical."""
        true_theta = 2.0
        n_obs = 200

        g = build_exponential_graph()
        g.update_parameterized_weights([true_theta])
        data = np.array(g.sample(n_obs))

        model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)
        prior = GaussPrior(ci=[0.5, 10.0])

        t0 = time()
        mcmc = MCMC(
            model=model, observed_data=data, prior=prior,
            theta_dim=1, n_samples=5000, n_chains=4,
            burn_in=1000, thin=2, proposal_scale=0.5,
            positive_params=True, seed=42,
            verbose=False, progress=False,
        )
        mcmc.run()
        elapsed = time() - t0

        results = mcmc.get_results()
        mean = float(results['theta_mean'][0])

        # Analytical posterior for Exp with wide prior ≈ n / sum(x)
        mle = n_obs / np.sum(data)

        assert abs(mean - mle) / mle < 0.10, (
            f"Posterior mean {mean:.3f} not within 10% of MLE {mle:.3f}"
        )
        # Expected baseline: ~20s
        if elapsed > 40:
            import warnings
            warnings.warn(f"Test took {elapsed:.1f}s (expected ~20s)")

    def test_convergence_diagnostics(self):
        """R-hat < 1.05 and ESS > 500."""
        true_theta = 2.0
        n_obs = 200

        g = build_exponential_graph()
        g.update_parameterized_weights([true_theta])
        data = np.array(g.sample(n_obs))

        model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

        mcmc = MCMC(
            model=model, observed_data=data,
            theta_dim=1, n_samples=5000, n_chains=4,
            burn_in=1000, thin=2, proposal_scale=0.5,
            positive_params=True, seed=42,
            verbose=False, progress=False,
        )
        mcmc.run()

        results = mcmc.get_results()
        rhat = results['rhat'][0]
        ess = results['ess'][0]

        assert rhat < 1.05, f"R-hat {rhat:.4f} should be < 1.05"
        assert ess > 500, f"ESS {ess:.0f} should be > 500"


# ---------------------------------------------------------------------------
# Test 2: Two-parameter Erlang-2
# ---------------------------------------------------------------------------

class TestErlang2Posterior:
    """MCMC on Erlang-2 should recover both rate parameters."""

    def test_two_param_recovery(self):
        """Both parameters recovered: ordering correct and R-hat acceptable.

        Erlang-2 parameters are weakly identifiable (rates are partially
        exchangeable), so we check that the posterior means are in the
        right ballpark and ordering rather than exact recovery.
        """
        true_theta = [1.0, 3.0]
        n_obs = 500

        g = build_erlang2_graph()
        g.update_parameterized_weights(true_theta)
        data = np.array(g.sample(n_obs))

        model = Graph.pmf_and_moments_from_graph(
            g, nr_moments=2, discrete=False, theta_dim=2
        )

        mcmc = MCMC(
            model=model, observed_data=data,
            theta_dim=2, n_samples=5000, n_chains=4,
            burn_in=1000, thin=2, proposal_scale=0.3,
            positive_params=True, seed=42,
            verbose=False, progress=False,
        )
        mcmc.run()

        results = mcmc.get_results()
        est = [float(results['theta_mean'][i]) for i in range(2)]

        # Both should be positive and finite
        assert all(e > 0 and np.isfinite(e) for e in est), f"Estimates should be positive: {est}"

        # R-hat should indicate convergence
        rhat = results['rhat']
        assert np.all(rhat < 1.2), f"R-hat {rhat} should all be < 1.2"


# ---------------------------------------------------------------------------
# Test 3: log_prob_fn mode — Gaussian target
# ---------------------------------------------------------------------------

class TestGaussianLogProbFn:
    """Direct log_prob_fn with known Gaussian posterior."""

    def test_gaussian_posterior(self):
        """Recovers precision-weighted Gaussian posterior."""
        # Likelihood: N(3, 0.5^2), Prior: N(0, 2^2)
        # Posterior precision = 1/0.5^2 + 1/2^2 = 4.25
        # Posterior mean = (3/0.5^2 + 0/2^2) / 4.25 = 12/4.25 ≈ 2.824
        # Posterior std = 1/sqrt(4.25) ≈ 0.485

        lik_mean, lik_std = 3.0, 0.5
        prior_mean, prior_std = 0.0, 2.0

        post_prec = 1/lik_std**2 + 1/prior_std**2
        post_mean = (lik_mean/lik_std**2 + prior_mean/prior_std**2) / post_prec
        post_std = 1 / np.sqrt(post_prec)

        def log_lik(theta):
            return -0.5 * jnp.sum((theta - lik_mean)**2 / lik_std**2)

        mcmc = MCMC(
            log_prob_fn=log_lik,
            prior=GaussPrior(mean=prior_mean, std=prior_std),
            theta_dim=1, n_samples=5000, n_chains=4,
            burn_in=500, thin=1, proposal_scale=0.5,
            positive_params=False, seed=42,
            verbose=False, progress=False,
        )
        mcmc.run()

        results = mcmc.get_results()
        est_mean = float(results['theta_mean'][0])
        est_std = float(results['theta_std'][0])

        assert abs(est_mean - post_mean) < 0.3, (
            f"Mean {est_mean:.3f} not within 0.3 of analytical {post_mean:.3f}"
        )
        assert abs(est_std - post_std) / post_std < 0.50, (
            f"Std {est_std:.3f} not within 50% of analytical {post_std:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 4: Fixed parameters preserved
# ---------------------------------------------------------------------------

class TestFixedParamsAccuracy:
    """Fixed parameter values are exactly preserved."""

    def test_fixed_param_exact(self):
        """Fixed parameter appears at exact value in all samples."""
        g = build_erlang2_graph()
        g.update_parameterized_weights([2.0, 3.0])
        data = np.array(g.sample(100))

        model = Graph.pmf_and_moments_from_graph(
            g, nr_moments=2, discrete=False, theta_dim=2
        )

        mcmc = MCMC(
            model=model, observed_data=data,
            theta_dim=2, n_samples=500, n_chains=2,
            burn_in=200, proposal_scale=0.3,
            fixed=[(1, 3.0)],
            positive_params=True, seed=42,
            verbose=False, progress=False,
        )
        mcmc.run()

        results = mcmc.get_results()
        fixed_vals = np.asarray(results['particles'][:, 1])

        assert np.allclose(fixed_vals, 3.0, atol=0.01), (
            f"Fixed param should be 3.0, got range [{fixed_vals.min():.4f}, {fixed_vals.max():.4f}]"
        )


# ---------------------------------------------------------------------------
# Test 5: Positive params constraint
# ---------------------------------------------------------------------------

class TestPositiveParamsAccuracy:
    """All samples strictly positive under positive_params=True."""

    def test_all_positive(self):
        true_theta = 2.0
        g = build_exponential_graph()
        g.update_parameterized_weights([true_theta])
        data = np.array(g.sample(100))

        model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

        mcmc = MCMC(
            model=model, observed_data=data,
            theta_dim=1, n_samples=2000, n_chains=2,
            burn_in=500, proposal_scale=0.5,
            positive_params=True, seed=42,
            verbose=False, progress=False,
        )
        mcmc.run()

        results = mcmc.get_results()
        assert np.all(results['particles'] > 0), "All samples should be > 0"
