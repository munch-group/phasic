"""
MCMC Metropolis-Hastings Inference Tests

Tests MCMC inference on simple exponential distribution to verify:
1. Basic sampling and convergence
2. Parameter transformations (positive_params)
3. Fixed parameters
4. R-hat convergence diagnostic
5. Results format compatibility with SVGD
6. Graph.mcmc() convenience method
"""

from phasic import Graph, MCMC
import numpy as np
import jax.numpy as jnp
import pytest


def build_exponential_graph():
    """Build simple exponential distribution: rate theta."""
    g = Graph(1)
    start = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v2, 1.0)
    v2.add_edge_parameterized(v1, 0.0, [1.0])
    return g


def generate_data(true_theta, n_samples=200, seed=42):
    """Generate samples from Exponential(true_theta)."""
    np.random.seed(seed)
    g = build_exponential_graph()
    g.update_parameterized_weights([true_theta])
    return np.array(g.sample(n_samples))


class TestMCMCBasic:
    """Basic MCMC functionality."""

    def test_basic_sampling(self):
        """MCMC runs and produces samples."""
        true_theta = 2.0
        data = generate_data(true_theta, n_samples=100)
        g = build_exponential_graph()
        model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

        mcmc = MCMC(
            model=model,
            observed_data=data,
            theta_dim=1,
            n_samples=500,
            n_chains=2,
            burn_in=200,
            proposal_scale=0.1,
            verbose=False,
            progress=False,
            seed=42,
        )
        mcmc.run()

        assert mcmc.is_fitted
        results = mcmc.get_results()
        assert 'particles' in results
        assert 'theta_mean' in results
        assert 'theta_std' in results
        assert results['particles'].shape == (2 * 500, 1)

    def test_convergence(self):
        """MCMC recovers true parameter (within reasonable tolerance)."""
        true_theta = 2.0
        data = generate_data(true_theta, n_samples=200)
        g = build_exponential_graph()
        model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

        mcmc = MCMC(
            model=model,
            observed_data=data,
            theta_dim=1,
            n_samples=2000,
            n_chains=2,
            burn_in=500,
            proposal_scale=0.1,
            verbose=False,
            progress=False,
            seed=42,
        )
        mcmc.run()

        results = mcmc.get_results()
        theta_mean = float(results['theta_mean'][0])

        # Should be within 30% of true value (MCMC can be noisy with few samples)
        assert abs(theta_mean - true_theta) / true_theta < 0.3, (
            f"theta_mean={theta_mean:.3f}, true={true_theta}"
        )


class TestMCMCPositiveParams:
    """Parameter transformation tests."""

    def test_positive_params(self):
        """With positive_params=True, all samples should be positive."""
        true_theta = 2.0
        data = generate_data(true_theta, n_samples=100)
        g = build_exponential_graph()
        model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

        mcmc = MCMC(
            model=model,
            observed_data=data,
            theta_dim=1,
            n_samples=500,
            n_chains=2,
            burn_in=200,
            positive_params=True,
            proposal_scale=0.1,
            verbose=False,
            progress=False,
            seed=42,
        )
        mcmc.run()

        results = mcmc.get_results()
        assert np.all(results['particles'] > 0), "All samples should be positive"

    def test_positive_params_false_with_custom_transform(self):
        """With positive_params=False and custom exp transform, results are positive."""
        true_theta = 2.0
        data = generate_data(true_theta, n_samples=100)
        g = build_exponential_graph()
        model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

        mcmc = MCMC(
            model=model,
            observed_data=data,
            theta_dim=1,
            n_samples=500,
            n_chains=2,
            burn_in=200,
            positive_params=False,
            param_transform=lambda phi: jnp.exp(phi),
            proposal_scale=0.1,
            verbose=False,
            progress=False,
            seed=42,
        )
        mcmc.run()

        results = mcmc.get_results()
        assert np.all(results['particles'] > 0), "Custom exp transform should yield positive values"
        assert 'particles_unconstrained' in results


class TestMCMCFixedParams:
    """Fixed parameter tests."""

    def test_fixed_params_tuple_format(self):
        """Fixed parameters remain at their specified values."""
        true_theta = 2.0
        data = generate_data(true_theta, n_samples=100)

        # Use the standard 1-param exponential graph but tell MCMC theta_dim=2
        # and fix theta[1]. Only theta[0] is used by the model (coeff [1.0]).
        g = Graph(1)
        start = g.starting_vertex()
        v2 = g.find_or_create_vertex([2])
        v1 = g.find_or_create_vertex([1])
        start.add_edge(v2, 1.0)
        v2.add_edge_parameterized(v1, 0.0, [1.0, 0.0])

        model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False,
                                                  theta_dim=2)

        mcmc = MCMC(
            model=model,
            observed_data=data,
            theta_dim=2,
            n_samples=500,
            n_chains=2,
            burn_in=200,
            fixed=[(1, 1.0)],
            proposal_scale=0.1,
            verbose=False,
            progress=False,
            seed=42,
        )
        mcmc.run()

        results = mcmc.get_results()
        # theta[1] should be fixed at 1.0
        fixed_values = results['particles'][:, 1]
        assert np.allclose(fixed_values, 1.0, atol=0.01), (
            f"Fixed parameter should be ~1.0, got {np.unique(fixed_values)}"
        )


class TestMCMCDiagnostics:
    """Convergence diagnostics."""

    def test_rhat(self):
        """R-hat should be computable and close to 1 for converged chains."""
        true_theta = 2.0
        data = generate_data(true_theta, n_samples=200)
        g = build_exponential_graph()
        model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

        mcmc = MCMC(
            model=model,
            observed_data=data,
            theta_dim=1,
            n_samples=2000,
            n_chains=4,
            burn_in=500,
            proposal_scale=0.1,
            verbose=False,
            progress=False,
            seed=42,
        )
        mcmc.run()

        rhat = mcmc.rhat()
        assert rhat.shape == (1,)
        assert not np.isnan(rhat[0])
        # R-hat should be < 1.5 (loose bound for basic test)
        assert rhat[0] < 1.5, f"R-hat={rhat[0]:.3f}, expected < 1.5"

    def test_ess(self):
        """ESS should be computable and positive."""
        true_theta = 2.0
        data = generate_data(true_theta, n_samples=100)
        g = build_exponential_graph()
        model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

        mcmc = MCMC(
            model=model,
            observed_data=data,
            theta_dim=1,
            n_samples=500,
            n_chains=2,
            burn_in=200,
            proposal_scale=0.1,
            verbose=False,
            progress=False,
            seed=42,
        )
        mcmc.run()

        ess = mcmc.ess()
        assert ess.shape == (1,)
        assert ess[0] > 0, f"ESS should be positive, got {ess[0]}"

    def test_acceptance_rate(self):
        """Acceptance rate should be between 0 and 1."""
        true_theta = 2.0
        data = generate_data(true_theta, n_samples=100)
        g = build_exponential_graph()
        model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

        mcmc = MCMC(
            model=model,
            observed_data=data,
            theta_dim=1,
            n_samples=500,
            n_chains=2,
            burn_in=200,
            proposal_scale=0.1,
            verbose=False,
            progress=False,
            seed=42,
        )
        mcmc.run()

        assert mcmc.acceptance_rate.shape == (2,)
        assert np.all(mcmc.acceptance_rate >= 0)
        assert np.all(mcmc.acceptance_rate <= 1)


class TestMCMCResultsFormat:
    """Results format compatibility with SVGD."""

    def test_results_keys(self):
        """Results dict contains all expected keys."""
        true_theta = 2.0
        data = generate_data(true_theta, n_samples=100)
        g = build_exponential_graph()
        model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

        mcmc = MCMC(
            model=model,
            observed_data=data,
            theta_dim=1,
            n_samples=500,
            n_chains=2,
            burn_in=200,
            proposal_scale=0.1,
            verbose=False,
            progress=False,
            seed=42,
        )
        mcmc.run()

        results = mcmc.get_results()

        # Keys shared with SVGD
        assert 'particles' in results
        assert 'theta_mean' in results
        assert 'theta_std' in results
        assert 'hpd_lower' in results
        assert 'hpd_upper' in results
        assert 'history' in results

        # MCMC-specific keys
        assert 'chains' in results
        assert 'acceptance_rate' in results
        assert 'ess' in results
        assert 'rhat' in results  # n_chains >= 2

    def test_hpd_intervals(self):
        """HPD intervals should contain the posterior mean."""
        true_theta = 2.0
        data = generate_data(true_theta, n_samples=100)
        g = build_exponential_graph()
        model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

        mcmc = MCMC(
            model=model,
            observed_data=data,
            theta_dim=1,
            n_samples=1000,
            n_chains=2,
            burn_in=300,
            proposal_scale=0.1,
            verbose=False,
            progress=False,
            seed=42,
        )
        mcmc.run()

        results = mcmc.get_results()
        mean = float(results['theta_mean'][0])
        lower = float(results['hpd_lower'][0])
        upper = float(results['hpd_upper'][0])

        assert lower <= mean <= upper, (
            f"HPD [{lower:.3f}, {upper:.3f}] should contain mean {mean:.3f}"
        )


class TestGraphMCMC:
    """Graph.mcmc() convenience method."""

    def test_graph_mcmc(self):
        """Graph.mcmc() runs end-to-end."""
        true_theta = 2.0
        data = generate_data(true_theta, n_samples=100)
        g = build_exponential_graph()

        mcmc = g.mcmc(
            observed_data=data,
            n_samples=500,
            n_chains=2,
            burn_in=200,
            proposal_scale=0.1,
            verbose=False,
            progress=False,
            seed=42,
        )

        assert mcmc.is_fitted
        results = mcmc.get_results()
        assert results['particles'].shape[0] == 2 * 500

    def test_graph_mcmc_auto_theta_dim(self):
        """Graph.mcmc() infers theta_dim from graph."""
        true_theta = 2.0
        data = generate_data(true_theta, n_samples=100)
        g = build_exponential_graph()

        mcmc = g.mcmc(
            observed_data=data,
            n_samples=500,
            n_chains=2,
            burn_in=200,
            verbose=False,
            progress=False,
            seed=42,
        )

        assert mcmc.theta_dim == 1


class TestMCMCSummary:
    """Summary output."""

    def test_summary_runs(self):
        """summary() completes without error."""
        true_theta = 2.0
        data = generate_data(true_theta, n_samples=100)
        g = build_exponential_graph()
        model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

        mcmc = MCMC(
            model=model,
            observed_data=data,
            theta_dim=1,
            n_samples=500,
            n_chains=2,
            burn_in=200,
            proposal_scale=0.1,
            verbose=False,
            progress=False,
            seed=42,
        )
        mcmc.run()
        mcmc.summary()  # Should not raise


class TestMCMCErrors:
    """Error handling."""

    def test_results_before_run(self):
        """Accessing results before run() raises error."""
        data = np.array([1.0, 2.0])
        g = build_exponential_graph()
        model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

        mcmc = MCMC(
            model=model,
            observed_data=data,
            theta_dim=1,
            verbose=False,
        )
        with pytest.raises(RuntimeError, match="Must call run"):
            mcmc.get_results()

    def test_no_theta_dim_or_init(self):
        """Must provide theta_dim or theta_init."""
        data = np.array([1.0, 2.0])
        g = build_exponential_graph()
        model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

        with pytest.raises(ValueError, match="theta_init or theta_dim"):
            MCMC(model=model, observed_data=data, verbose=False)

    def test_positive_params_and_param_transform(self):
        """Cannot use both positive_params and param_transform."""
        data = np.array([1.0, 2.0])
        g = build_exponential_graph()
        model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

        with pytest.raises(ValueError, match="Cannot specify both"):
            MCMC(
                model=model,
                observed_data=data,
                theta_dim=1,
                positive_params=True,
                param_transform=lambda x: x,
                verbose=False,
            )
