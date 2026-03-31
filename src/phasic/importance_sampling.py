"""
Importance sampling for inhomogeneous coalescent inference.

Provides closed-form density functions for coalescence waiting times
under homogeneous and piecewise-constant population size models,
and importance weight computation for combining phase-type proposals
with inhomogeneous targets.

All functions are JAX-compatible (support jit, grad, vmap).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial


def _choose2(k: int | jnp.ndarray) -> jnp.ndarray:
    """Compute C(k, 2) = k*(k-1)/2."""
    return k * (k - 1) / 2.0


def piecewise_constant_integral(
    a: float | jnp.ndarray,
    b: float | jnp.ndarray,
    epoch_boundaries: jnp.ndarray,
    epoch_sizes: jnp.ndarray,
) -> jnp.ndarray:
    """Compute integral_a^b 1/N(s) ds for piecewise-constant N(s).

    Parameters
    ----------
    a : float or array
        Lower integration bound.
    b : float or array
        Upper integration bound (must be >= a).
    epoch_boundaries : array, shape (n_epochs - 1,)
        Times at which population size changes. Must be sorted.
        The first epoch spans [0, epoch_boundaries[0]),
        the last epoch spans [epoch_boundaries[-1], infinity).
    epoch_sizes : array, shape (n_epochs,)
        Population size in each epoch. Must be positive.

    Returns
    -------
    float or array
        Value of the integral.
    """
    # Build full boundary array: [0, b1, b2, ..., inf]
    # Each epoch i spans [boundaries_full[i], boundaries_full[i+1])
    n_epochs = len(epoch_sizes)
    boundaries_full = jnp.concatenate([
        jnp.array([0.0]),
        jnp.asarray(epoch_boundaries),
        jnp.array([jnp.inf]),
    ])

    total = jnp.zeros_like(jnp.asarray(a, dtype=jnp.float64))
    for i in range(n_epochs):
        # Overlap of [a, b] with epoch [boundaries_full[i], boundaries_full[i+1])
        lo = jnp.maximum(a, boundaries_full[i])
        hi = jnp.minimum(b, boundaries_full[i + 1])
        overlap = jnp.maximum(hi - lo, 0.0)
        total = total + overlap / epoch_sizes[i]

    return total


def homogeneous_log_density(
    waiting_times: jnp.ndarray,
    n_lineages: int,
    pop_size: float | jnp.ndarray,
) -> jnp.ndarray:
    """Log-density of inter-coalescence waiting times under constant N.

    For each level k = n, n-1, ..., 2, the waiting time is
    Exp(C(k,2) / N) distributed.

    Parameters
    ----------
    waiting_times : array, shape (n_lineages - 1,)
        Waiting times w_n, w_{n-1}, ..., w_2 (in order of decreasing
        lineage count). w_n is the waiting time with n lineages.
    n_lineages : int
        Initial number of lineages.
    pop_size : float or array
        Constant population size.

    Returns
    -------
    scalar
        Log-density.
    """
    n_events = n_lineages - 1
    log_density = 0.0

    for i in range(n_events):
        k = n_lineages - i  # Number of lineages: n, n-1, ..., 2
        rate = _choose2(k) / pop_size
        log_density = log_density + jnp.log(rate) - rate * waiting_times[i]

    return log_density


def inhomogeneous_log_density(
    waiting_times: jnp.ndarray,
    n_lineages: int,
    epoch_boundaries: jnp.ndarray,
    epoch_sizes: jnp.ndarray,
) -> jnp.ndarray:
    """Log-density of inter-coalescence waiting times under piecewise-constant N(t).

    Parameters
    ----------
    waiting_times : array, shape (n_lineages - 1,)
        Waiting times w_n, w_{n-1}, ..., w_2 (in order of decreasing
        lineage count).
    n_lineages : int
        Initial number of lineages.
    epoch_boundaries : array, shape (n_epochs - 1,)
        Times at which population size changes. Must be sorted.
    epoch_sizes : array, shape (n_epochs,)
        Population size in each epoch. Must be positive.

    Returns
    -------
    scalar
        Log-density.
    """
    n_events = n_lineages - 1
    log_density = 0.0
    cumulative_time = 0.0

    for i in range(n_events):
        k = n_lineages - i  # Number of lineages: n, n-1, ..., 2
        ck2 = _choose2(k)
        w_k = waiting_times[i]
        t_start = cumulative_time
        t_end = cumulative_time + w_k

        # Rate at coalescence time (instantaneous rate at t_end)
        # Determine which epoch t_end falls in
        n_size = _pop_size_at(t_end, epoch_boundaries, epoch_sizes)
        rate_at_event = ck2 / n_size

        # Integrated hazard over [t_start, t_end]
        integrated_hazard = ck2 * piecewise_constant_integral(
            t_start, t_end, epoch_boundaries, epoch_sizes
        )

        log_density = log_density + jnp.log(rate_at_event) - integrated_hazard
        cumulative_time = t_end

    return log_density


def _pop_size_at(
    t: float | jnp.ndarray,
    epoch_boundaries: jnp.ndarray,
    epoch_sizes: jnp.ndarray,
) -> jnp.ndarray:
    """Population size at time t for piecewise-constant model.

    Parameters
    ----------
    t : float or array
        Time point(s).
    epoch_boundaries : array, shape (n_epochs - 1,)
        Epoch boundary times.
    epoch_sizes : array, shape (n_epochs,)
        Population sizes.

    Returns
    -------
    float or array
        N(t).
    """
    # Find which epoch t falls in
    # epoch_boundaries = [b1, b2, ...], epochs = [N_0, N_1, N_2, ...]
    # t < b1 → epoch 0, b1 <= t < b2 → epoch 1, etc.
    idx = jnp.searchsorted(epoch_boundaries, t, side='right')
    return epoch_sizes[idx]


def importance_log_weight(
    waiting_times: jnp.ndarray,
    n_lineages: int,
    epoch_boundaries: jnp.ndarray,
    epoch_sizes: jnp.ndarray,
    proposal_pop_size: float | jnp.ndarray,
) -> jnp.ndarray:
    """Log importance weight: log(target density) - log(proposal density).

    Parameters
    ----------
    waiting_times : array, shape (n_lineages - 1,)
        Waiting times from the proposal.
    n_lineages : int
        Initial number of lineages.
    epoch_boundaries : array, shape (n_epochs - 1,)
        Epoch boundary times for the target.
    epoch_sizes : array, shape (n_epochs,)
        Target population sizes.
    proposal_pop_size : float
        Constant population size used in the homogeneous proposal.

    Returns
    -------
    scalar
        Log importance weight.
    """
    log_target = inhomogeneous_log_density(
        waiting_times, n_lineages, epoch_boundaries, epoch_sizes
    )
    log_proposal = homogeneous_log_density(
        waiting_times, n_lineages, proposal_pop_size
    )
    return log_target - log_proposal


def make_demographic_log_likelihood(
    observed_waiting_times: jnp.ndarray,
    n_lineages: int,
    epoch_boundaries: jnp.ndarray,
) -> callable:
    """Create a log-likelihood function over epoch sizes for use with MCMC.

    Parameters
    ----------
    observed_waiting_times : array, shape (n_lineages - 1,)
        Observed inter-coalescence waiting times.
    n_lineages : int
        Initial number of lineages.
    epoch_boundaries : array, shape (n_epochs - 1,)
        Fixed epoch boundary times.

    Returns
    -------
    callable
        JAX-compatible function: log_lik(epoch_sizes) -> scalar.
        Suitable for passing to phasic.MCMC as the model, or directly
        as a log-probability function.
    """
    observed_waiting_times = jnp.asarray(observed_waiting_times)
    epoch_boundaries = jnp.asarray(epoch_boundaries)

    def log_likelihood(epoch_sizes: jnp.ndarray) -> jnp.ndarray:
        return inhomogeneous_log_density(
            observed_waiting_times, n_lineages, epoch_boundaries, epoch_sizes
        )

    return log_likelihood
