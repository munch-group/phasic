# MCMC Metropolis-Hastings — Remaining Steps

## Implemented (Steps 1, 2, 4, 5-partial, 6)

- `src/phasic/mcmc.py`: MCMC class with:
  - Single-chain and multi-chain MH
  - Adaptive Metropolis proposal (Haario et al., 2001): empirical covariance + Robbins-Monro scale adaptation
  - Parallel chain execution via ThreadPoolExecutor (for non-JIT log_prob_fn)
  - `log_prob_fn` mode for custom likelihoods
  - R-hat, ESS, summary()
- `src/phasic/__init__.py`: `Graph.mcmc()` convenience method, MCMC export
- `tests/pytest/test_mcmc.py`: 27 tests (all passing)

## Step 3: Plotting Methods

Add visualization methods to the MCMC class, matching SVGD's style (iridis colormap, matplotlib):

- `plot_trace(param_names=None, true_theta=None)`: Per-parameter trace plots showing all chains overlaid. One subplot per parameter.
- `plot_posterior(true_theta=None, param_names=None)`: Posterior density plots from MCMC samples using KDE. Should visually match `SVGD.plot_posterior()`.
- `plot_autocorrelation(max_lag=50, param_names=None)`: ACF plots per parameter to diagnose mixing.

## Step 5: Additional Features (Lower Priority)

- ~~Chain parallelization via threads (currently sequential Python loop)~~ ✅ Done
- ~~Adaptive Metropolis proposal tuning~~ ✅ Done
- Chain parallelization via `jax.vmap` across chains (for JIT-compatible log_prob_fn)
- Warm restart: ability to continue sampling from where a previous run left off
- Thinning of stored history vs inference samples (memory optimization for long chains)
