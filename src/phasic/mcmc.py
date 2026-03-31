"""
MCMC Metropolis-Hastings inference for phase-type distributions.

This module provides Metropolis-Hastings MCMC sampling as an alternative
to SVGD for Bayesian parameter inference. It reuses the same model
construction, prior classes, and parameter transformation infrastructure
as SVGD.
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable
from time import time

import numpy as np
import jax
import jax.numpy as jnp

from .logging_config import get_logger
from .svgd import (
    Prior,
    GaussPrior,
    HalfCauchyPrior,
    DataPrior,
    SparseObservations,
    is_sparse_observations,
    _compute_hpd,
    _inverse_softplus,
)
from .config import get_config

logger = get_logger(__name__)

try:
    from vscodenb import pqdm as tqdm
    from vscodenb import prange as trange
except ImportError:
    from tqdm import trange, tqdm
    from functools import partial
    trange = partial(trange, leave=False)
    tqdm = partial(tqdm, leave=False)


class MCMC:
    """
    Metropolis-Hastings MCMC for Bayesian parameter inference.

    Samples from the posterior distribution p(theta | data) using
    random-walk Metropolis-Hastings. Supports multiple independent chains,
    parameter transformations, and fixed parameters.

    Parameters
    ----------
    model : callable
        JAX-compatible parameterized model with signature:
        model(theta, data) -> values or model(theta, data) -> (pmf, moments)
    observed_data : np.ndarray or SparseObservations
        Observed data points.
    prior : callable, list of Prior, DataPrior, or None
        Log prior function. Same semantics as SVGD:
        - Single callable: prior(theta) -> scalar
        - List of Prior objects: one per dimension, None for fixed params
        - None: standard normal prior
    n_samples : int, default=10000
        Number of posterior samples to draw per chain (after burn-in).
    n_chains : int, default=4
        Number of independent chains.
    burn_in : int, default=1000
        Number of initial samples to discard per chain.
    thin : int, default=1
        Keep every thin-th sample after burn-in.
    proposal_scale : float or array, optional
        Initial proposal standard deviation. Scalar or per-dimension.
        If None, defaults to 0.1.
    theta_init : np.ndarray, optional
        Initial parameter values. Shape (n_chains, theta_dim) or (theta_dim,).
        If 1D, the same init is used for all chains with small perturbations.
    theta_dim : int, optional
        Dimension of parameter vector. Required if theta_init is None.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, default=True
        Print progress information.
    progress : bool, default=False
        Display progress bar during sampling.
    jit : bool or None, default=None
        JIT-compile the log-probability function. If None, uses config default.
    positive_params : bool, default=True
        If True, constrains parameters to positive domain using softplus.
    param_transform : callable, optional
        Custom parameter transformation. Overrides positive_params.
    rewards : np.ndarray, optional
        Reward vector/matrix for multivariate phase-type distributions.
    fixed : list of tuples or array, optional
        Fixed parameters. Same format as SVGD:
        - [(index, value), ...] or binary mask [0, 1, 0, ...]

    Attributes
    ----------
    samples : np.ndarray
        All posterior samples concatenated across chains, shape
        (n_chains * n_samples, theta_dim) in constrained space.
    chains : np.ndarray
        Per-chain samples, shape (n_chains, n_samples, theta_dim)
        in constrained space.
    acceptance_rate : np.ndarray
        Per-chain acceptance rates.
    is_fitted : bool
        Whether run() has been called.

    Examples
    --------
    >>> from phasic import Graph, MCMC
    >>> g = build_exponential_graph()
    >>> model = Graph.pmf_from_graph(g, discrete=False)
    >>> mcmc = MCMC(model, observed_data, theta_dim=1)
    >>> mcmc.run()
    >>> print(mcmc.get_results()['theta_mean'])
    """

    def __init__(
        self,
        model: Callable,
        observed_data: jnp.ndarray | SparseObservations,
        prior: Prior | list[Prior | None] | DataPrior | None = None,
        n_samples: int = 10_000,
        n_chains: int = 4,
        burn_in: int = 1000,
        thin: int = 1,
        proposal_scale: float | jnp.ndarray | None = None,
        theta_init: jnp.ndarray | None = None,
        theta_dim: int | None = None,
        seed: int | None = None,
        verbose: bool = True,
        progress: bool = False,
        jit: bool | None = None,
        positive_params: bool = True,
        param_transform: Callable | None = None,
        rewards: jnp.ndarray | None = None,
        fixed: list | None = None,
    ) -> None:

        if seed is None:
            seed = np.random.randint(1, 10000)

        # Get configuration
        config = get_config()
        if jit is None:
            jit = config.jit

        self.model = model
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.burn_in = burn_in
        self.thin = thin
        self.seed = seed
        self.verbose = verbose
        self.progress = progress
        self.jit_enabled = jit
        self.rewards = rewards
        self.fixed = fixed

        # Handle observations
        if is_sparse_observations(observed_data):
            self.observed_data = observed_data
            self._sparse_format = True
        else:
            self.observed_data = jnp.array(observed_data)
            self._sparse_format = False

        # Prior handling (same as SVGD)
        self.prior = prior
        self.prior_list = list(prior) if isinstance(prior, (list, tuple, DataPrior)) else None

        # Parameter transformation (same as SVGD)
        if positive_params and param_transform is not None:
            raise ValueError(
                "Cannot specify both positive_params=True and param_transform. "
                "Use positive_params=True for automatic softplus transformation, "
                "or provide a custom param_transform function."
            )

        self.positive_params = positive_params
        if positive_params:
            self.param_transform = lambda phi: jnp.maximum(jax.nn.softplus(phi), 1e-9)
        elif param_transform is not None:
            if not callable(param_transform):
                raise ValueError("param_transform must be a callable function")
            self.param_transform = param_transform
        else:
            self.param_transform = None

        # Wrap priors with transformation (same as SVGD)
        if self.param_transform is not None:
            if self.prior_list is not None:
                for prior_i in self.prior_list:
                    if hasattr(prior_i, '_transform'):
                        prior_i._transform = self.param_transform
            elif self.prior is not None and hasattr(self.prior, '_transform'):
                self.prior._transform = self.param_transform

        # Determine theta_dim
        if theta_init is not None:
            theta_init = jnp.array(theta_init)
            if theta_init.ndim == 1:
                self.theta_dim = theta_init.shape[0]
            elif theta_init.ndim == 2:
                self.theta_dim = theta_init.shape[1]
            else:
                raise ValueError(
                    f"theta_init must be 1D (theta_dim,) or 2D (n_chains, theta_dim), "
                    f"got shape {theta_init.shape}"
                )
        elif theta_dim is not None:
            self.theta_dim = theta_dim
        else:
            raise ValueError(
                "Either theta_init or theta_dim must be provided."
            )

        # Validate prior list
        if self.prior_list is not None and len(self.prior_list) != self.theta_dim:
            raise ValueError(
                f"prior list length ({len(self.prior_list)}) must match theta_dim ({self.theta_dim})"
            )

        # Fixed parameters (same logic as SVGD)
        self.fixed_mask = None
        self.fixed_values = None
        if fixed is not None:
            if isinstance(fixed, list) and len(fixed) > 0 and isinstance(fixed[0], tuple):
                self.fixed_mask = jnp.zeros(self.theta_dim)
                self.fixed_values = jnp.zeros(self.theta_dim)

                for idx, value in fixed:
                    if idx < 0 or idx >= self.theta_dim:
                        raise ValueError(
                            f"Invalid parameter index {idx}, must be in [0, {self.theta_dim})"
                        )
                    self.fixed_mask = self.fixed_mask.at[idx].set(1)
                    if positive_params and value > 0:
                        phi_value = float(jnp.log(jnp.exp(value) - 1))
                        self.fixed_values = self.fixed_values.at[idx].set(phi_value)
                    else:
                        self.fixed_values = self.fixed_values.at[idx].set(value)

                n_fixed = len(fixed)
                n_learnable = self.theta_dim - n_fixed
                if n_learnable == 0:
                    raise ValueError("All parameters are fixed! At least one must be learnable.")
                if verbose:
                    print(f"Fixed parameters: {n_fixed}/{self.theta_dim}")
                    for idx, value in fixed:
                        print(f"  theta[{idx}] fixed at {value}")
            else:
                fixed_array = jnp.array(fixed)
                if len(fixed_array) != self.theta_dim:
                    raise ValueError(
                        f"fixed must have length {self.theta_dim}, got {len(fixed_array)}"
                    )
                self.fixed_mask = fixed_array.astype(float)
                if positive_params:
                    self.fixed_values = _inverse_softplus(jnp.ones(self.theta_dim))
                else:
                    self.fixed_values = jnp.ones(self.theta_dim)

        # Initialize chain starting positions
        key = jax.random.PRNGKey(seed)
        if theta_init is not None:
            if theta_init.ndim == 1:
                # Same init for all chains with perturbation
                keys = jax.random.split(key, n_chains)
                self.theta_init = jnp.array([
                    theta_init + 0.01 * jax.random.normal(keys[c], (self.theta_dim,))
                    for c in range(n_chains)
                ])
            else:
                if theta_init.shape[0] != n_chains:
                    raise ValueError(
                        f"theta_init has {theta_init.shape[0]} rows but n_chains={n_chains}"
                    )
                self.theta_init = theta_init
        else:
            # Initialize from prior or default
            keys = jax.random.split(key, n_chains + 1)
            key = keys[0]
            if self.prior_list is not None:
                inits = []
                for c in range(n_chains):
                    chain_key = keys[c + 1]
                    samples = []
                    for i, prior_i in enumerate(self.prior_list):
                        chain_key, subkey = jax.random.split(chain_key)
                        if prior_i is None:
                            samples.append(jax.random.normal(subkey, (1,)))
                        else:
                            samples.append(prior_i.sample(subkey, (1,)).ravel()[:1])
                    inits.append(jnp.concatenate(samples))
                self.theta_init = jnp.stack(inits)
            elif self.prior is not None and hasattr(self.prior, 'sample'):
                self.theta_init = jnp.stack([
                    self.prior.sample(keys[c + 1], (1, self.theta_dim)).ravel()
                    for c in range(n_chains)
                ])
            elif self.param_transform is not None:
                self.theta_init = jnp.stack([
                    jax.random.normal(keys[c + 1], (self.theta_dim,)) + 1.0
                    for c in range(n_chains)
                ])
            else:
                self.theta_init = jnp.stack([
                    jax.random.normal(keys[c + 1], (self.theta_dim,))
                    for c in range(n_chains)
                ])

        # Apply fixed values to init
        if self.fixed_mask is not None:
            for c in range(n_chains):
                self.theta_init = self.theta_init.at[c].set(
                    jnp.where(self.fixed_mask, self.fixed_values, self.theta_init[c])
                )

        # Proposal scale
        if proposal_scale is None:
            self.proposal_scale = 0.1 * jnp.ones(self.theta_dim)
        elif isinstance(proposal_scale, (int, float)):
            self.proposal_scale = float(proposal_scale) * jnp.ones(self.theta_dim)
        else:
            self.proposal_scale = jnp.array(proposal_scale)

        # Results
        self._chains_phi = None  # Chains in unconstrained space
        self.acceptance_rate = None
        self.is_fitted = False

    def _log_prob(self, phi: jnp.ndarray) -> float:
        """Log probability: log p(data|theta) + log p(theta).

        Parameters
        ----------
        phi : array
            Parameter vector in unconstrained space.

        Returns
        -------
        float
            Log probability.
        """
        if self.param_transform is not None:
            theta = self.param_transform(phi)
        else:
            theta = phi

        # Model evaluation
        result = self.model(theta, self.observed_data)

        if isinstance(result, tuple):
            model_values = result[0]
        else:
            model_values = result

        # Log-likelihood
        if self._sparse_format:
            log_lik = jnp.sum(jnp.log(model_values + 1e-10))
        else:
            obs_mask = ~jnp.isnan(self.observed_data) if self.observed_data.ndim > 1 else jnp.ones(self.observed_data.shape, dtype=bool)
            if self.observed_data.ndim > 1:
                log_lik = jnp.sum(jnp.where(obs_mask, jnp.log(model_values + 1e-10), 0.0))
            else:
                log_lik = jnp.sum(jnp.log(model_values + 1e-10))

        # Log-prior
        if self.prior_list is not None:
            log_pri = sum(
                self.prior_list[i](phi[i:i+1])
                for i in range(len(self.prior_list))
                if self.prior_list[i] is not None
            )
        elif self.prior is not None:
            log_pri = self.prior(phi)
        else:
            log_pri = -0.5 * jnp.sum(phi**2)

        return log_lik + log_pri

    def run(self) -> MCMC:
        """Run MCMC sampling.

        Returns
        -------
        MCMC
            Returns self for method chaining.
        """
        total_iter = self.burn_in + self.n_samples * self.thin
        n_chains = self.n_chains

        # Optionally JIT compile log_prob
        if self.rewards is not None:
            # Wrap model to pass rewards
            original_model = self.model
            def model_with_rewards(theta, data):
                return original_model(theta, data, rewards=self.rewards)
            self.model = model_with_rewards

        log_prob_fn = self._log_prob
        if self.jit_enabled:
            log_prob_fn = jax.jit(self._log_prob)
            # Trigger compilation
            if self.verbose:
                print("JIT compiling log-probability function...", end=" ", flush=True)
            t0 = time()
            _ = log_prob_fn(self.theta_init[0])
            if self.verbose:
                print(f"done ({time() - t0:.1f}s)")

        if self.verbose:
            print(f"\nStarting MCMC ({n_chains} chains, {self.n_samples} samples, "
                  f"burn-in={self.burn_in}, thin={self.thin})")

        # Allocate storage
        chains_phi = np.empty((n_chains, self.n_samples, self.theta_dim))
        acceptance_counts = np.zeros(n_chains)

        key = jax.random.PRNGKey(self.seed + 1000)

        for c in range(n_chains):
            key, chain_key = jax.random.split(key)
            phi_current = self.theta_init[c]
            lp_current = float(log_prob_fn(phi_current))
            sample_idx = 0

            iterator = range(total_iter)
            if self.progress:
                iterator = trange(total_iter, desc=f"Chain {c+1}/{n_chains}")

            for i in iterator:
                chain_key, prop_key, accept_key = jax.random.split(chain_key, 3)

                # Propose
                noise = jax.random.normal(prop_key, (self.theta_dim,))
                phi_proposed = phi_current + self.proposal_scale * noise

                # Enforce fixed params
                if self.fixed_mask is not None:
                    phi_proposed = jnp.where(self.fixed_mask, self.fixed_values, phi_proposed)

                # Evaluate
                lp_proposed = float(log_prob_fn(phi_proposed))

                # Accept/reject
                log_alpha = lp_proposed - lp_current
                log_u = float(jnp.log(jax.random.uniform(accept_key)))

                if log_u < log_alpha:
                    phi_current = phi_proposed
                    lp_current = lp_proposed
                    if i >= self.burn_in:
                        acceptance_counts[c] += 1

                # Store (after burn-in, with thinning)
                if i >= self.burn_in and (i - self.burn_in) % self.thin == 0:
                    chains_phi[c, sample_idx] = np.asarray(phi_current)
                    sample_idx += 1

            if self.verbose:
                post_burnin_total = self.n_samples * self.thin
                rate = acceptance_counts[c] / post_burnin_total
                print(f"  Chain {c+1}: acceptance rate = {rate:.3f}")

        self._chains_phi = chains_phi
        total_post_burnin = self.n_samples * self.thin
        self.acceptance_rate = acceptance_counts / total_post_burnin
        self.is_fitted = True

        if self.verbose:
            results = self.get_results()
            print(f"\nMCMC complete!")
            print(f"Posterior mean: {results['theta_mean']}")
            print(f"Posterior std:  {results['theta_std']}")
            if n_chains >= 2:
                rhat = results['rhat']
                print(f"R-hat:          {rhat}")

        return self

    def get_results(self) -> dict:
        """Get inference results.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'particles': All samples concatenated, shape (n_chains * n_samples, theta_dim)
            - 'theta_mean': Posterior mean
            - 'theta_std': Posterior standard deviation
            - 'hpd_lower', 'hpd_upper': HPD intervals per dimension
            - 'chains': Per-chain samples, shape (n_chains, n_samples, theta_dim)
            - 'acceptance_rate': Per-chain acceptance rates
            - 'rhat': Gelman-Rubin R-hat per dimension (if n_chains >= 2)
            - 'ess': Effective sample size per dimension
            - 'history': Same as 'chains' (for SVGD compatibility)
        """
        if not self.is_fitted:
            raise RuntimeError("Must call run() before accessing results")

        # Transform chains to constrained space
        if self.param_transform is not None:
            chains = np.array([
                [np.asarray(self.param_transform(jnp.array(self._chains_phi[c, s])))
                 for s in range(self.n_samples)]
                for c in range(self.n_chains)
            ])
        else:
            chains = self._chains_phi.copy()

        # Concatenate all chains
        all_samples = chains.reshape(-1, self.theta_dim)

        theta_mean = np.mean(all_samples, axis=0)
        theta_std = np.std(all_samples, axis=0)

        # HPD intervals
        hpd_lower = np.empty(self.theta_dim)
        hpd_upper = np.empty(self.theta_dim)
        for i in range(self.theta_dim):
            hpd_lower[i], hpd_upper[i] = _compute_hpd(all_samples[:, i])

        results = {
            'particles': jnp.array(all_samples),
            'theta_mean': jnp.array(theta_mean),
            'theta_std': jnp.array(theta_std),
            'hpd_lower': hpd_lower,
            'hpd_upper': hpd_upper,
            'chains': chains,
            'acceptance_rate': self.acceptance_rate,
            'ess': self.ess(),
            'history': chains,  # SVGD compatibility
        }

        if self.n_chains >= 2:
            results['rhat'] = self.rhat()

        if self.param_transform is not None:
            results['particles_unconstrained'] = jnp.array(
                self._chains_phi.reshape(-1, self.theta_dim)
            )

        return results

    def rhat(self) -> np.ndarray:
        """Gelman-Rubin split R-hat convergence diagnostic.

        Returns
        -------
        np.ndarray
            R-hat per dimension. Values close to 1.0 indicate convergence.
            Values > 1.1 suggest the chains have not converged.
        """
        if not self.is_fitted:
            raise RuntimeError("Must call run() before computing R-hat")
        if self.n_chains < 2:
            warnings.warn("R-hat requires at least 2 chains")
            return np.full(self.theta_dim, np.nan)

        # Use unconstrained space for diagnostics
        chains = self._chains_phi  # (n_chains, n_samples, theta_dim)
        m = self.n_chains
        n = self.n_samples

        rhat = np.empty(self.theta_dim)
        for d in range(self.theta_dim):
            chain_means = np.mean(chains[:, :, d], axis=1)  # (m,)
            overall_mean = np.mean(chain_means)

            # Between-chain variance
            B = n * np.var(chain_means, ddof=1)

            # Within-chain variance
            chain_vars = np.var(chains[:, :, d], axis=1, ddof=1)  # (m,)
            W = np.mean(chain_vars)

            # Pooled variance estimate
            var_hat = (n - 1) / n * W + B / n

            rhat[d] = np.sqrt(var_hat / W) if W > 0 else np.nan

        return rhat

    def ess(self) -> np.ndarray:
        """Effective sample size per dimension.

        Uses the initial positive sequence estimator.

        Returns
        -------
        np.ndarray
            ESS per dimension.
        """
        if not self.is_fitted:
            raise RuntimeError("Must call run() before computing ESS")

        chains = self._chains_phi  # (n_chains, n_samples, theta_dim)
        total_samples = self.n_chains * self.n_samples
        all_samples = chains.reshape(-1, self.theta_dim)

        ess = np.empty(self.theta_dim)
        for d in range(self.theta_dim):
            x = all_samples[:, d]
            n = len(x)
            mean = np.mean(x)
            var = np.var(x)
            if var == 0:
                ess[d] = float(n)
                continue

            # Autocorrelation at lag k
            max_lag = min(n // 2, 500)
            autocorr_sum = 0.0
            for k in range(1, max_lag):
                rho_k = np.mean((x[:n-k] - mean) * (x[k:] - mean)) / var
                if rho_k < 0.05:
                    break
                autocorr_sum += rho_k

            ess[d] = n / (1.0 + 2.0 * autocorr_sum)

        return ess

    def summary(self) -> None:
        """Print summary table of inference results."""
        if not self.is_fitted:
            raise RuntimeError("Must call run() before summary")

        results = self.get_results()
        print(f"\nMCMC Summary ({self.n_chains} chains, {self.n_samples} samples/chain)")
        print("-" * 70)
        print(f"{'Param':<8} {'Mean':>10} {'Std':>10} {'HPD 2.5%':>10} {'HPD 97.5%':>10} {'R-hat':>8} {'ESS':>8}")
        print("-" * 70)

        rhat = results.get('rhat', np.full(self.theta_dim, np.nan))
        ess = results['ess']

        for i in range(self.theta_dim):
            fixed_str = " (fixed)" if self.fixed_mask is not None and self.fixed_mask[i] else ""
            print(f"theta[{i}]{fixed_str:<1} {float(results['theta_mean'][i]):>10.4f} "
                  f"{float(results['theta_std'][i]):>10.4f} "
                  f"{float(results['hpd_lower'][i]):>10.4f} "
                  f"{float(results['hpd_upper'][i]):>10.4f} "
                  f"{float(rhat[i]):>8.4f} "
                  f"{float(ess[i]):>8.1f}")
        print("-" * 70)
        print(f"Acceptance rates: {', '.join(f'{r:.3f}' for r in self.acceptance_rate)}")
