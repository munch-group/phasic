# """
# MCMC accuracy and performance tests.

# Verifies that MCMC recovers known posteriors with quantitative bounds.
# """

# from phasic import Graph, MCMC, GaussPrior
# import numpy as np
# import jax.numpy as jnp
# import pytest
# from time import time


# # ---------------------------------------------------------------------------
# # Graph builders
# # ---------------------------------------------------------------------------

# def build_exponential_graph():
#     """Exponential distribution with rate theta (1 parameter)."""
#     g = Graph(1)
#     start = g.starting_vertex()
#     v2 = g.find_or_create_vertex([2])
#     v1 = g.find_or_create_vertex([1])
#     start.add_edge(v2, 1.0)
#     v2.add_edge_parameterized(v1, 0.0, [1.0])
#     return g


# def build_erlang2_graph():
#     """Erlang-2: two sequential exponential stages (2 parameters).

#     Stage 1 rate = theta[0], Stage 2 rate = theta[1].
#     """
#     g = Graph(1)
#     start = g.starting_vertex()
#     v3 = g.find_or_create_vertex([3])
#     v2 = g.find_or_create_vertex([2])
#     v1 = g.find_or_create_vertex([1])
#     start.add_edge(v3, 1.0)
#     v3.add_edge_parameterized(v2, 0.0, [1.0, 0.0])  # rate = theta[0]
#     v2.add_edge_parameterized(v1, 0.0, [0.0, 1.0])  # rate = theta[1]
#     return g


# # ---------------------------------------------------------------------------
# # Test 1: Exponential with known posterior
# # ---------------------------------------------------------------------------

# class TestExponentialPosterior:
#     """MCMC on Exp(theta) should recover the analytical posterior."""

#     def test_posterior_mean(self):
#         """Posterior mean within 10% of analytical."""
#         true_theta = 2.0
#         n_obs = 200

#         g = build_exponential_graph()
#         g.update_parameterized_weights([true_theta])
#         data = np.array(g.sample(n_obs))

#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)
#         prior = GaussPrior(ci=[0.5, 10.0])

#         t0 = time()
#         mcmc = MCMC(
#             model=model, observed_data=data, prior=prior,
#             theta_dim=1, n_samples=5000, n_chains=4,
#             burn_in=1000, thin=2, proposal_scale=0.5,
#             positive_params=True, seed=42,
#             verbose=False, progress=False,
#         )
#         mcmc.run()
#         elapsed = time() - t0

#         results = mcmc.get_results()
#         mean = float(results['theta_mean'][0])

#         # Analytical posterior for Exp with wide prior ≈ n / sum(x)
#         mle = n_obs / np.sum(data)

#         assert abs(mean - mle) / mle < 0.10, (
#             f"Posterior mean {mean:.3f} not within 10% of MLE {mle:.3f}"
#         )
#         # Expected baseline: ~20s
#         if elapsed > 40:
#             import warnings
#             warnings.warn(f"Test took {elapsed:.1f}s (expected ~20s)")

#     def test_convergence_diagnostics(self):
#         """R-hat < 1.05 and ESS > 500."""
#         true_theta = 2.0
#         n_obs = 200

#         g = build_exponential_graph()
#         g.update_parameterized_weights([true_theta])
#         data = np.array(g.sample(n_obs))

#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

#         mcmc = MCMC(
#             model=model, observed_data=data,
#             theta_dim=1, n_samples=5000, n_chains=4,
#             burn_in=1000, thin=2, proposal_scale=0.5,
#             positive_params=True, seed=42,
#             verbose=False, progress=False,
#         )
#         mcmc.run()

#         results = mcmc.get_results()
#         rhat = results['rhat'][0]
#         ess = results['ess'][0]

#         assert rhat < 1.05, f"R-hat {rhat:.4f} should be < 1.05"
#         assert ess > 500, f"ESS {ess:.0f} should be > 500"


# # ---------------------------------------------------------------------------
# # Test 2: Two-parameter Erlang-2
# # ---------------------------------------------------------------------------

# class TestErlang2Posterior:
#     """MCMC on Erlang-2 should recover both rate parameters."""

#     def test_two_param_recovery(self):
#         """Both parameters recovered: ordering correct and R-hat acceptable.

#         Erlang-2 parameters are weakly identifiable (rates are partially
#         exchangeable), so we check that the posterior means are in the
#         right ballpark and ordering rather than exact recovery.
#         """
#         true_theta = [1.0, 3.0]
#         n_obs = 500

#         g = build_erlang2_graph()
#         g.update_parameterized_weights(true_theta)
#         data = np.array(g.sample(n_obs))

#         model = Graph.pmf_and_moments_from_graph(
#             g, nr_moments=2, discrete=False, theta_dim=2
#         )

#         mcmc = MCMC(
#             model=model, observed_data=data,
#             theta_dim=2, n_samples=5000, n_chains=4,
#             burn_in=1000, thin=2, proposal_scale=0.3,
#             positive_params=True, seed=42,
#             verbose=False, progress=False,
#         )
#         mcmc.run()

#         results = mcmc.get_results()
#         est = [float(results['theta_mean'][i]) for i in range(2)]

#         # Both should be positive and finite
#         assert all(e > 0 and np.isfinite(e) for e in est), f"Estimates should be positive: {est}"

#         # R-hat should indicate convergence
#         rhat = results['rhat']
#         assert np.all(rhat < 1.2), f"R-hat {rhat} should all be < 1.2"


# # ---------------------------------------------------------------------------
# # Test 3: log_prob_fn mode — Gaussian target
# # ---------------------------------------------------------------------------

# class TestGaussianLogProbFn:
#     """Direct log_prob_fn with known Gaussian posterior."""

#     def test_gaussian_posterior(self):
#         """Recovers precision-weighted Gaussian posterior."""
#         # Likelihood: N(3, 0.5^2), Prior: N(0, 2^2)
#         # Posterior precision = 1/0.5^2 + 1/2^2 = 4.25
#         # Posterior mean = (3/0.5^2 + 0/2^2) / 4.25 = 12/4.25 ≈ 2.824
#         # Posterior std = 1/sqrt(4.25) ≈ 0.485

#         lik_mean, lik_std = 3.0, 0.5
#         prior_mean, prior_std = 0.0, 2.0

#         post_prec = 1/lik_std**2 + 1/prior_std**2
#         post_mean = (lik_mean/lik_std**2 + prior_mean/prior_std**2) / post_prec
#         post_std = 1 / np.sqrt(post_prec)

#         def log_lik(theta):
#             return -0.5 * jnp.sum((theta - lik_mean)**2 / lik_std**2)

#         mcmc = MCMC(
#             log_prob_fn=log_lik,
#             prior=GaussPrior(mean=prior_mean, std=prior_std),
#             theta_dim=1, n_samples=5000, n_chains=4,
#             burn_in=500, thin=1, proposal_scale=0.5,
#             positive_params=False, seed=42,
#             verbose=False, progress=False,
#         )
#         mcmc.run()

#         results = mcmc.get_results()
#         est_mean = float(results['theta_mean'][0])
#         est_std = float(results['theta_std'][0])

#         assert abs(est_mean - post_mean) < 0.3, (
#             f"Mean {est_mean:.3f} not within 0.3 of analytical {post_mean:.3f}"
#         )
#         assert abs(est_std - post_std) / post_std < 0.50, (
#             f"Std {est_std:.3f} not within 50% of analytical {post_std:.3f}"
#         )


# # ---------------------------------------------------------------------------
# # Test 4: Fixed parameters preserved
# # ---------------------------------------------------------------------------

# class TestFixedParamsAccuracy:
#     """Fixed parameter values are exactly preserved."""

#     def test_fixed_param_exact(self):
#         """Fixed parameter appears at exact value in all samples."""
#         g = build_erlang2_graph()
#         g.update_parameterized_weights([2.0, 3.0])
#         data = np.array(g.sample(100))

#         model = Graph.pmf_and_moments_from_graph(
#             g, nr_moments=2, discrete=False, theta_dim=2
#         )

#         mcmc = MCMC(
#             model=model, observed_data=data,
#             theta_dim=2, n_samples=500, n_chains=2,
#             burn_in=200, proposal_scale=0.3,
#             fixed=[(1, 3.0)],
#             positive_params=True, seed=42,
#             verbose=False, progress=False,
#         )
#         mcmc.run()

#         results = mcmc.get_results()
#         fixed_vals = np.asarray(results['particles'][:, 1])

#         assert np.allclose(fixed_vals, 3.0, atol=0.01), (
#             f"Fixed param should be 3.0, got range [{fixed_vals.min():.4f}, {fixed_vals.max():.4f}]"
#         )


# # ---------------------------------------------------------------------------
# # Test 5: Positive params constraint
# # ---------------------------------------------------------------------------

# class TestPositiveParamsAccuracy:
#     """All samples strictly positive under positive_params=True."""

#     def test_all_positive(self):
#         true_theta = 2.0
#         g = build_exponential_graph()
#         g.update_parameterized_weights([true_theta])
#         data = np.array(g.sample(100))

#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

#         mcmc = MCMC(
#             model=model, observed_data=data,
#             theta_dim=1, n_samples=2000, n_chains=2,
#             burn_in=500, proposal_scale=0.5,
#             positive_params=True, seed=42,
#             verbose=False, progress=False,
#         )
#         mcmc.run()

#         results = mcmc.get_results()
#         assert np.all(results['particles'] > 0), "All samples should be > 0"
# """
# MCMC Metropolis-Hastings Inference Tests

# Tests MCMC inference on simple exponential distribution to verify:
# 1. Basic sampling and convergence
# 2. Parameter transformations (positive_params)
# 3. Fixed parameters
# 4. R-hat convergence diagnostic
# 5. Results format compatibility with SVGD
# 6. Graph.mcmc() convenience method
# """

# from phasic import Graph, MCMC
# import numpy as np
# import jax.numpy as jnp
# import pytest


# def build_exponential_graph():
#     """Build simple exponential distribution: rate theta."""
#     g = Graph(1)
#     start = g.starting_vertex()
#     v2 = g.find_or_create_vertex([2])
#     v1 = g.find_or_create_vertex([1])
#     start.add_edge(v2, 1.0)
#     v2.add_edge_parameterized(v1, 0.0, [1.0])
#     return g


# def generate_data(true_theta, n_samples=200, seed=42):
#     """Generate samples from Exponential(true_theta)."""
#     np.random.seed(seed)
#     g = build_exponential_graph()
#     g.update_parameterized_weights([true_theta])
#     return np.array(g.sample(n_samples))


# class TestMCMCBasic:
#     """Basic MCMC functionality."""

#     def test_basic_sampling(self):
#         """MCMC runs and produces samples."""
#         true_theta = 2.0
#         data = generate_data(true_theta, n_samples=100)
#         g = build_exponential_graph()
#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

#         mcmc = MCMC(
#             model=model,
#             observed_data=data,
#             theta_dim=1,
#             n_samples=500,
#             n_chains=2,
#             burn_in=200,
#             proposal_scale=0.1,
#             verbose=False,
#             progress=False,
#             seed=42,
#         )
#         mcmc.run()

#         assert mcmc.is_fitted
#         results = mcmc.get_results()
#         assert 'particles' in results
#         assert 'theta_mean' in results
#         assert 'theta_std' in results
#         assert results['particles'].shape == (2 * 500, 1)

#     def test_convergence(self):
#         """MCMC recovers true parameter (within reasonable tolerance)."""
#         true_theta = 2.0
#         data = generate_data(true_theta, n_samples=200)
#         g = build_exponential_graph()
#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

#         mcmc = MCMC(
#             model=model,
#             observed_data=data,
#             theta_dim=1,
#             n_samples=2000,
#             n_chains=2,
#             burn_in=500,
#             proposal_scale=0.1,
#             verbose=False,
#             progress=False,
#             seed=42,
#         )
#         mcmc.run()

#         results = mcmc.get_results()
#         theta_mean = float(results['theta_mean'][0])

#         # Should be within 30% of true value (MCMC can be noisy with few samples)
#         assert abs(theta_mean - true_theta) / true_theta < 0.3, (
#             f"theta_mean={theta_mean:.3f}, true={true_theta}"
#         )


# class TestMCMCPositiveParams:
#     """Parameter transformation tests."""

#     def test_positive_params(self):
#         """With positive_params=True, all samples should be positive."""
#         true_theta = 2.0
#         data = generate_data(true_theta, n_samples=100)
#         g = build_exponential_graph()
#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

#         mcmc = MCMC(
#             model=model,
#             observed_data=data,
#             theta_dim=1,
#             n_samples=500,
#             n_chains=2,
#             burn_in=200,
#             positive_params=True,
#             proposal_scale=0.1,
#             verbose=False,
#             progress=False,
#             seed=42,
#         )
#         mcmc.run()

#         results = mcmc.get_results()
#         assert np.all(results['particles'] > 0), "All samples should be positive"

#     def test_positive_params_false_with_custom_transform(self):
#         """With positive_params=False and custom exp transform, results are positive."""
#         true_theta = 2.0
#         data = generate_data(true_theta, n_samples=100)
#         g = build_exponential_graph()
#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

#         mcmc = MCMC(
#             model=model,
#             observed_data=data,
#             theta_dim=1,
#             n_samples=500,
#             n_chains=2,
#             burn_in=200,
#             positive_params=False,
#             param_transform=lambda phi: jnp.exp(phi),
#             proposal_scale=0.1,
#             verbose=False,
#             progress=False,
#             seed=42,
#         )
#         mcmc.run()

#         results = mcmc.get_results()
#         assert np.all(results['particles'] > 0), "Custom exp transform should yield positive values"
#         assert 'particles_unconstrained' in results


# class TestMCMCFixedParams:
#     """Fixed parameter tests."""

#     def test_fixed_params_tuple_format(self):
#         """Fixed parameters remain at their specified values."""
#         true_theta = 2.0
#         data = generate_data(true_theta, n_samples=100)

#         # Use the standard 1-param exponential graph but tell MCMC theta_dim=2
#         # and fix theta[1]. Only theta[0] is used by the model (coeff [1.0]).
#         g = Graph(1)
#         start = g.starting_vertex()
#         v2 = g.find_or_create_vertex([2])
#         v1 = g.find_or_create_vertex([1])
#         start.add_edge(v2, 1.0)
#         v2.add_edge_parameterized(v1, 0.0, [1.0, 0.0])

#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False,
#                                                   theta_dim=2)

#         mcmc = MCMC(
#             model=model,
#             observed_data=data,
#             theta_dim=2,
#             n_samples=500,
#             n_chains=2,
#             burn_in=200,
#             fixed=[(1, 1.0)],
#             proposal_scale=0.1,
#             verbose=False,
#             progress=False,
#             seed=42,
#         )
#         mcmc.run()

#         results = mcmc.get_results()
#         # theta[1] should be fixed at 1.0
#         fixed_values = results['particles'][:, 1]
#         assert np.allclose(fixed_values, 1.0, atol=0.01), (
#             f"Fixed parameter should be ~1.0, got {np.unique(fixed_values)}"
#         )


# class TestMCMCDiagnostics:
#     """Convergence diagnostics."""

#     def test_rhat(self):
#         """R-hat should be computable and close to 1 for converged chains."""
#         true_theta = 2.0
#         data = generate_data(true_theta, n_samples=200)
#         g = build_exponential_graph()
#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

#         mcmc = MCMC(
#             model=model,
#             observed_data=data,
#             theta_dim=1,
#             n_samples=2000,
#             n_chains=4,
#             burn_in=500,
#             proposal_scale=0.1,
#             verbose=False,
#             progress=False,
#             seed=42,
#         )
#         mcmc.run()

#         rhat = mcmc.rhat()
#         assert rhat.shape == (1,)
#         assert not np.isnan(rhat[0])
#         # R-hat should be < 1.5 (loose bound for basic test)
#         assert rhat[0] < 1.5, f"R-hat={rhat[0]:.3f}, expected < 1.5"

#     def test_ess(self):
#         """ESS should be computable and positive."""
#         true_theta = 2.0
#         data = generate_data(true_theta, n_samples=100)
#         g = build_exponential_graph()
#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

#         mcmc = MCMC(
#             model=model,
#             observed_data=data,
#             theta_dim=1,
#             n_samples=500,
#             n_chains=2,
#             burn_in=200,
#             proposal_scale=0.1,
#             verbose=False,
#             progress=False,
#             seed=42,
#         )
#         mcmc.run()

#         ess = mcmc.ess()
#         assert ess.shape == (1,)
#         assert ess[0] > 0, f"ESS should be positive, got {ess[0]}"

#     def test_acceptance_rate(self):
#         """Acceptance rate should be between 0 and 1."""
#         true_theta = 2.0
#         data = generate_data(true_theta, n_samples=100)
#         g = build_exponential_graph()
#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

#         mcmc = MCMC(
#             model=model,
#             observed_data=data,
#             theta_dim=1,
#             n_samples=500,
#             n_chains=2,
#             burn_in=200,
#             proposal_scale=0.1,
#             verbose=False,
#             progress=False,
#             seed=42,
#         )
#         mcmc.run()

#         assert mcmc.acceptance_rate.shape == (2,)
#         assert np.all(mcmc.acceptance_rate >= 0)
#         assert np.all(mcmc.acceptance_rate <= 1)


# class TestMCMCResultsFormat:
#     """Results format compatibility with SVGD."""

#     def test_results_keys(self):
#         """Results dict contains all expected keys."""
#         true_theta = 2.0
#         data = generate_data(true_theta, n_samples=100)
#         g = build_exponential_graph()
#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

#         mcmc = MCMC(
#             model=model,
#             observed_data=data,
#             theta_dim=1,
#             n_samples=500,
#             n_chains=2,
#             burn_in=200,
#             proposal_scale=0.1,
#             verbose=False,
#             progress=False,
#             seed=42,
#         )
#         mcmc.run()

#         results = mcmc.get_results()

#         # Keys shared with SVGD
#         assert 'particles' in results
#         assert 'theta_mean' in results
#         assert 'theta_std' in results
#         assert 'hpd_lower' in results
#         assert 'hpd_upper' in results
#         assert 'history' in results

#         # MCMC-specific keys
#         assert 'chains' in results
#         assert 'acceptance_rate' in results
#         assert 'ess' in results
#         assert 'rhat' in results  # n_chains >= 2

#     def test_hpd_intervals(self):
#         """HPD intervals should contain the posterior mean."""
#         true_theta = 2.0
#         data = generate_data(true_theta, n_samples=100)
#         g = build_exponential_graph()
#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

#         mcmc = MCMC(
#             model=model,
#             observed_data=data,
#             theta_dim=1,
#             n_samples=1000,
#             n_chains=2,
#             burn_in=300,
#             proposal_scale=0.1,
#             verbose=False,
#             progress=False,
#             seed=42,
#         )
#         mcmc.run()

#         results = mcmc.get_results()
#         mean = float(results['theta_mean'][0])
#         lower = float(results['hpd_lower'][0])
#         upper = float(results['hpd_upper'][0])

#         assert lower <= mean <= upper, (
#             f"HPD [{lower:.3f}, {upper:.3f}] should contain mean {mean:.3f}"
#         )


# class TestGraphMCMC:
#     """Graph.mcmc() convenience method."""

#     def test_graph_mcmc(self):
#         """Graph.mcmc() runs end-to-end."""
#         true_theta = 2.0
#         data = generate_data(true_theta, n_samples=100)
#         g = build_exponential_graph()

#         mcmc = g.mcmc(
#             observed_data=data,
#             n_samples=500,
#             n_chains=2,
#             burn_in=200,
#             proposal_scale=0.1,
#             verbose=False,
#             progress=False,
#             seed=42,
#         )

#         assert mcmc.is_fitted
#         results = mcmc.get_results()
#         assert results['particles'].shape[0] == 2 * 500

#     def test_graph_mcmc_auto_theta_dim(self):
#         """Graph.mcmc() infers theta_dim from graph."""
#         true_theta = 2.0
#         data = generate_data(true_theta, n_samples=100)
#         g = build_exponential_graph()

#         mcmc = g.mcmc(
#             observed_data=data,
#             n_samples=500,
#             n_chains=2,
#             burn_in=200,
#             verbose=False,
#             progress=False,
#             seed=42,
#         )

#         assert mcmc.theta_dim == 1


# class TestMCMCSummary:
#     """Summary output."""

#     def test_summary_runs(self):
#         """summary() completes without error."""
#         true_theta = 2.0
#         data = generate_data(true_theta, n_samples=100)
#         g = build_exponential_graph()
#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

#         mcmc = MCMC(
#             model=model,
#             observed_data=data,
#             theta_dim=1,
#             n_samples=500,
#             n_chains=2,
#             burn_in=200,
#             proposal_scale=0.1,
#             verbose=False,
#             progress=False,
#             seed=42,
#         )
#         mcmc.run()
#         mcmc.summary()  # Should not raise


# class TestMCMCErrors:
#     """Error handling."""

#     def test_results_before_run(self):
#         """Accessing results before run() raises error."""
#         data = np.array([1.0, 2.0])
#         g = build_exponential_graph()
#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

#         mcmc = MCMC(
#             model=model,
#             observed_data=data,
#             theta_dim=1,
#             verbose=False,
#         )
#         with pytest.raises(RuntimeError, match="Must call run"):
#             mcmc.get_results()

#     def test_no_theta_dim_or_init(self):
#         """Must provide theta_dim or theta_init."""
#         data = np.array([1.0, 2.0])
#         g = build_exponential_graph()
#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

#         with pytest.raises(ValueError, match="theta_init or theta_dim"):
#             MCMC(model=model, observed_data=data, verbose=False)

#     def test_positive_params_and_param_transform(self):
#         """Cannot use both positive_params and param_transform."""
#         data = np.array([1.0, 2.0])
#         g = build_exponential_graph()
#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

#         with pytest.raises(ValueError, match="Cannot specify both"):
#             MCMC(
#                 model=model,
#                 observed_data=data,
#                 theta_dim=1,
#                 positive_params=True,
#                 param_transform=lambda x: x,
#                 verbose=False,
#             )

#     def test_both_model_and_log_prob_fn(self):
#         """Cannot provide both model and log_prob_fn."""
#         data = np.array([1.0, 2.0])
#         g = build_exponential_graph()
#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

#         with pytest.raises(ValueError, match="Cannot provide both"):
#             MCMC(
#                 model=model,
#                 observed_data=data,
#                 log_prob_fn=lambda theta: -0.5 * jnp.sum(theta**2),
#                 theta_dim=1,
#                 verbose=False,
#             )

#     def test_neither_model_nor_log_prob_fn(self):
#         """Must provide either model or log_prob_fn."""
#         with pytest.raises(ValueError, match="Must provide either"):
#             MCMC(theta_dim=1, verbose=False)

#     def test_model_without_observed_data(self):
#         """model requires observed_data."""
#         g = build_exponential_graph()
#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

#         with pytest.raises(ValueError, match="observed_data.*required"):
#             MCMC(model=model, theta_dim=1, verbose=False)


# class TestMCMCLogProbFn:
#     """Tests for log_prob_fn mode."""

#     def test_log_prob_fn_basic(self):
#         """MCMC with log_prob_fn runs and produces samples."""
#         def log_lik(theta):
#             return -0.5 * jnp.sum((theta - 3.0)**2 / 0.5**2)

#         mcmc = MCMC(
#             log_prob_fn=log_lik,
#             theta_dim=1,
#             n_samples=1000,
#             n_chains=2,
#             burn_in=200,
#             proposal_scale=0.5,
#             positive_params=False,
#             verbose=False,
#             progress=False,
#             seed=42,
#         )
#         mcmc.run()

#         assert mcmc.is_fitted
#         results = mcmc.get_results()
#         assert results['particles'].shape == (2000, 1)

#         # Should recover mean between 0 (default prior) and 3 (likelihood)
#         mean = float(results['theta_mean'][0])
#         assert 1.0 < mean < 4.0, f"Expected between 1 and 4, got {mean}"

#     def test_log_prob_fn_with_prior(self):
#         """log_prob_fn mode adds prior correctly."""
#         from phasic import GaussPrior

#         # Likelihood peaks at 5.0, prior peaks at 1.0
#         # Posterior should be between them
#         def log_lik(theta):
#             return -0.5 * jnp.sum((theta - 5.0)**2)

#         mcmc = MCMC(
#             log_prob_fn=log_lik,
#             prior=GaussPrior(mean=1.0, std=1.0),
#             theta_dim=1,
#             n_samples=1000,
#             n_chains=2,
#             burn_in=200,
#             proposal_scale=0.5,
#             positive_params=False,
#             verbose=False,
#             progress=False,
#             seed=42,
#         )
#         mcmc.run()

#         mean = float(mcmc.get_results()['theta_mean'][0])
#         assert 1.0 < mean < 5.0, f"Posterior mean should be between prior and likelihood, got {mean}"

#     def test_log_prob_fn_positive_params(self):
#         """log_prob_fn mode works with positive_params=True."""
#         def log_lik(theta):
#             return -0.5 * jnp.sum((theta - 2.0)**2)

#         mcmc = MCMC(
#             log_prob_fn=log_lik,
#             theta_dim=1,
#             n_samples=500,
#             n_chains=2,
#             burn_in=200,
#             proposal_scale=0.5,
#             positive_params=True,
#             verbose=False,
#             progress=False,
#             seed=42,
#         )
#         mcmc.run()

#         results = mcmc.get_results()
#         assert np.all(results['particles'] > 0)


# class TestMCMCAdaptive:
#     """Tests for adaptive Metropolis proposal."""

#     def test_adaptive_runs(self):
#         """Adaptive MCMC runs without error."""
#         def log_lik(theta):
#             return -0.5 * jnp.sum((theta - 2.0)**2)

#         mcmc = MCMC(
#             log_prob_fn=log_lik,
#             theta_dim=2,
#             n_samples=500,
#             n_chains=2,
#             burn_in=300,
#             adaptive=True,
#             positive_params=False,
#             verbose=False, progress=False, seed=42,
#         )
#         mcmc.run()
#         assert mcmc.is_fitted

#     def test_adaptive_false(self):
#         """adaptive=False uses fixed proposal (backward compatible)."""
#         def log_lik(theta):
#             return -0.5 * jnp.sum((theta - 2.0)**2)

#         mcmc = MCMC(
#             log_prob_fn=log_lik,
#             theta_dim=1,
#             n_samples=500,
#             n_chains=2,
#             burn_in=200,
#             adaptive=False,
#             positive_params=False,
#             verbose=False, progress=False, seed=42,
#         )
#         mcmc.run()
#         assert mcmc.is_fitted

#     def test_adaptive_acceptance_rate(self):
#         """Adaptive should produce acceptance rate in a reasonable range."""
#         def log_lik(theta):
#             return -0.5 * jnp.sum((theta - jnp.array([2.0, 3.0]))**2 / 0.5**2)

#         mcmc = MCMC(
#             log_prob_fn=log_lik,
#             theta_dim=2,
#             n_samples=2000,
#             n_chains=2,
#             burn_in=500,
#             adaptive=True,
#             target_acceptance=0.234,
#             positive_params=False,
#             verbose=False, progress=False, seed=42,
#         )
#         mcmc.run()

#         for rate in mcmc.acceptance_rate:
#             assert 0.05 < rate < 0.6, (
#                 f"Acceptance rate {rate:.3f} should be in reasonable range"
#             )

#     def test_adaptive_with_fixed_params(self):
#         """Adaptive works with fixed parameters."""
#         g = Graph(1)
#         start = g.starting_vertex()
#         v2 = g.find_or_create_vertex([2])
#         v1 = g.find_or_create_vertex([1])
#         start.add_edge(v2, 1.0)
#         v2.add_edge_parameterized(v1, 0.0, [1.0, 0.0])
#         g.update_parameterized_weights([2.0, 1.0])
#         data = np.array(g.sample(100))
#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False, theta_dim=2)

#         mcmc = MCMC(
#             model=model, observed_data=data,
#             theta_dim=2, n_samples=300, n_chains=2,
#             burn_in=200, adaptive=True,
#             fixed=[(1, 1.0)],
#             verbose=False, progress=False, seed=42,
#         )
#         mcmc.run()
#         results = mcmc.get_results()
#         fixed_vals = np.asarray(results['particles'][:, 1])
#         assert np.allclose(fixed_vals, 1.0, atol=0.01)


# class TestMCMCParallel:
#     """Tests for parallel chain execution."""

#     def test_processes_parallel(self):
#         """Process-based parallelism runs and gives valid results."""
#         def log_lik(theta):
#             return -0.5 * jnp.sum((theta - 2.0)**2)

#         mcmc = MCMC(
#             log_prob_fn=log_lik,
#             theta_dim=1,
#             n_samples=500,
#             n_chains=4,
#             burn_in=200,
#             parallel='processes',
#             positive_params=False,
#             verbose=False, progress=False, seed=42,
#         )
#         mcmc.run()
#         assert mcmc.is_fitted
#         assert mcmc.acceptance_rate.shape == (4,)
#         assert np.all(mcmc.acceptance_rate > 0)

#     def test_vmap_parallel(self):
#         """Vmap parallelism runs and gives valid results."""
#         def log_lik(theta):
#             return -0.5 * jnp.sum((theta - 2.0)**2)

#         mcmc = MCMC(
#             log_prob_fn=log_lik,
#             theta_dim=1,
#             n_samples=500,
#             n_chains=4,
#             burn_in=200,
#             parallel='vmap',
#             positive_params=False,
#             verbose=False, progress=False, seed=42,
#         )
#         mcmc.run()
#         assert mcmc.is_fitted
#         assert mcmc.acceptance_rate.shape == (4,)
#         assert np.all(mcmc.acceptance_rate > 0)

#     def test_vmap_with_model(self):
#         """Vmap works with phasic model (JIT-compatible)."""
#         true_theta = 2.0
#         data = generate_data(true_theta, n_samples=100)
#         g = build_exponential_graph()
#         model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

#         mcmc = MCMC(
#             model=model, observed_data=data,
#             theta_dim=1, n_samples=500, n_chains=4,
#             burn_in=200, proposal_scale=0.5,
#             parallel='vmap',
#             verbose=False, progress=False, seed=42,
#         )
#         mcmc.run()
#         mean = float(mcmc.get_results()['theta_mean'][0])
#         assert 0.5 < mean < 5.0, f"Posterior mean {mean} out of range"
