# MCMC Metropolis-Hastings — Remaining Steps

## Implemented (Steps 1, 4, 6)

- `src/phasic/mcmc.py`: MCMC class with single-chain MH, multi-chain support, R-hat, ESS, summary()
- `src/phasic/__init__.py`: `Graph.mcmc()` convenience method, MCMC export
- `tests/pytest/test_mcmc.py`: 16 tests (all passing)

## Step 2: Adaptive Metropolis Proposal Tuning

Add Haario et al. (2001) adaptive proposal covariance to the MCMC class:

- During burn-in, accumulate chain history and compute empirical covariance
- After a warm-up period (e.g., `2 * theta_dim` iterations), switch from diagonal `proposal_scale * I` to `s_d * Cov(chain) + s_d * epsilon * I` where `s_d = 2.4^2 / theta_dim`
- Add `adaptive: bool = True` and `target_acceptance: float = 0.234` constructor parameters
- Monitor acceptance rate and optionally scale the proposal to approach the target rate

## Step 3: Plotting Methods

Add visualization methods to the MCMC class, matching SVGD's style (iridis colormap, matplotlib):

- `plot_trace(param_names=None, true_theta=None)`: Per-parameter trace plots showing all chains overlaid. One subplot per parameter.
- `plot_posterior(true_theta=None, param_names=None)`: Posterior density plots from MCMC samples using KDE. Should visually match `SVGD.plot_posterior()`.
- `plot_autocorrelation(max_lag=50, param_names=None)`: ACF plots per parameter to diagnose mixing.

## Step 5: Additional Features (Lower Priority)

- Thinning of stored history vs inference samples (memory optimization for long chains)
- Chain parallelization via `jax.vmap` across chains (currently sequential Python loop)
- Warm restart: ability to continue sampling from where a previous run left off
