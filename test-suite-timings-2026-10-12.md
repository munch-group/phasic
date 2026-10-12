# Test-suite timings, 2026-10-12

Measured on branch `fix/constant-parameterized-consistency` (commit
`4fbb1075`), macOS, 16 GB RAM. Each file ran in its own pytest process, one
at a time, under a memory-capped runner (6 GB per file, 25 min per file;
follow-ups as noted). Per-test times from `pytest --durations=0` in a
second run of the slowest files. Times vary between runs: one file took
77 s in the suite and 25 s alone, so treat anything under about a minute
as approximate. Each file also pays ~3-5 s of pytest + phasic/JAX import.

## Full suite

2165 passed, 0 failed, 84 skipped, 26 xfailed, over ~165 files (three
files collect no tests by design). Total wall time roughly 100 minutes.

## Slowest files

| File | Wall time | Peak memory |
|---|---|---|
| test_notebook_model_recovery.py | 1556 s (25.9 min; needs > the 25-min cap) | 1.5 GB |
| inference/test_model_selection.py | 991 s (16.5 min) | 2.6-3.1 GB |
| inference/test_svgd_exposure.py | 394 s | 0.6 GB |
| inference/test_mcmc_accuracy.py | 206 s | 0.7 GB |
| inference/test_svgd_correctness.py | 137 s | 0.6 GB |
| inference/test_nan_observations_correctness.py | 133 s | 0.9 GB |
| inference/test_lrt_at.py | 125 s | 1.2 GB |
| test_svgd_divergence_robustness.py | ~115 s (needs 7.5 GB, see below) | 7.5 GB |
| inference/test_exact_grad_callback_mode.py | 77 s (25 s on re-run) | 1.6 GB |
| test_slurm_wp6_distributed.py | 71 s | 0.9 GB |
| inference/test_exact_grad_formula_mode.py | 68-92 s | 1.5-2.3 GB |
| test_method_of_moments.py | 36 s | 2.2-2.5 GB |

## Slowest single tests (>= ~10 s)

| Test | Time |
|---|---|
| **test_notebook_model_recovery.py** | |
| test_mixed_scale_island_moves_all_parameters_off_their_priors | 544 s |
| test_two_island_svgd_recovers_both_rates_hundreds | 353 s |
| test_coalescent_svgd_recovers_rate_thousands | 235 s |
| test_two_island_svgd_recovers_both_rates_tens | 230 s |
| test_mixed_scale_two_locus_recovers_both_parameters | 63 s |
| test_coalescent_svgd_recovers_rate_hundreds | 48 s |
| test_coalescent_with_rewards_svgd_recovers_rate | 27 s |
| test_posterior_interval_covers_truth_from_off_truth_prior | 21 s |
| **inference/test_model_selection.py** | |
| test_lrt_basic_correct_df_and_pvalue | 177 s |
| test_log_likelihood_independent_of_regularization | 100 s |
| test_log_likelihood_exponential_mle | 99 s |
| test_aic_formula | 75 s |
| test_bic_formula | 75 s |
| test_compare_aic_basic | 61 s |
| test_compare_bic_basic | 61 s |
| test_repr_aic_bic_lrt | 52 s |
| **inference/test_svgd_exposure.py** | |
| test_exposure_shifts_posterior_inverse_to_alpha | 342 s |
| test_graph_svgd_forwards_exposure_to_svgd | 17 s |
| **inference/test_exact_grad_formula_mode.py** | |
| test_formula_svgd_frontdoor_default_runs_exact | 87 s |
| **test_svgd_divergence_robustness.py** | |
| test_grad_at_diverged_theta_is_finite_not_nan | 79 s, 7.5 GB peak |
| test_huge_theta_returns_finite_penalty_not_crash | 31 s |
| **inference/test_svgd_correctness.py** | |
| test_basic_convergence | 78 s |
| test_cache_isolation | 27 s |
| test_log_transformation | 10 s |
| test_positive_constraint | 10 s |
| **inference/test_mcmc_accuracy.py** | |
| TestErlang2Posterior::test_two_param_recovery | 69 s |
| TestExponentialPosterior::test_posterior_mean | 62 s |
| TestExponentialPosterior::test_convergence_diagnostics | 61 s |
| TestGaussianLogProbFn::test_gaussian_posterior | 20 s |
| **inference/test_lrt_at.py** | |
| test_lrt_at_rejects_inconsistent_pair | 68 s |
| setup of test_lrt_at_agrees_with_lrt_on_shared_model | 42 s |
| setup of test_lrt_at_tied_vs_free_epoch | 10 s |
| **inference/test_nan_observations_correctness.py** | |
| test_nan_correctness_univariate | 59 s |
| test_nan_correctness_multivariate | 25 s |
| test_nan_vs_filtered_equivalence | 21 s |
| **inference/test_exact_grad_callback_mode.py** | |
| test_callback_svgd_frontdoor_default_runs_exact | 19 s |
| **test_slurm_wp6_distributed.py** | |
| six test_precompute_distributed_* tests | 8-10 s each |

## Observations (from timings only; test code not read)

- The recovery tests and test_model_selection.py are ~42 of ~100 minutes.
- test_model_selection.py: each test appears to run its own SVGD fit, and
  AIC/BIC and their compare variants come in equal-duration pairs; a
  module-scoped fixture sharing fitted models would likely cut most of it.
- One test_svgd_exposure.py test alone takes 5.7 minutes.
- The recovery tests are slow by design (full SVGD fits at three sizes);
  the full suite runs the slow tier because nothing filters the `slow`
  marker. Running `-m "not slow"` routinely would skip ~23 minutes.
- The SLURM tests take almost exactly 10 s each, suggesting a fixed wait
  or timeout rather than computation.
- test_grad_at_diverged_theta_is_finite_not_nan needs 7.5 GB, a lot on a
  16 GB machine.
