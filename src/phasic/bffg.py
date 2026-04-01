"""
Backward Filtering Forward Guiding (BFFG) utilities.

General infrastructure for importance-weighted inference with phase-type
distributions. Given a path sampled from a proposal graph, these functions
compute reward-weighted sojourn times, importance weights from exit rate
ratios, and importance-weighted likelihood estimates.

These utilities are model-agnostic — they work with any phase-type graph.
Model-specific code (e.g., coalescent rate formulas, Poisson mutation
likelihoods) should be defined by the user.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def path_to_rewards(graph, path: dict, rewards: np.ndarray | None = None) -> np.ndarray:
    """Compute reward-weighted sojourn times from a sampled path.

    This is the per-path decomposition of what ``graph.sample(rewards=...)``
    computes. For each step in the path, the sojourn time is multiplied by
    the reward vector at that vertex, and contributions are summed.

    Parameters
    ----------
    graph : Graph
        The graph from which the path was sampled.
    path : dict
        Path dict from ``graph.sample_path()``, with keys
        ``'vertex_indices'`` and ``'entry_times'``.
    rewards : array, optional
        Reward weights per vertex. Can be:
        - 1D array, shape ``(n_vertices,)``: single reward per vertex.
          Returns a scalar.
        - 2D array, shape ``(n_features, n_vertices)``: one reward vector
          per feature. Returns array of shape ``(n_features,)``.
        - None: uses ``graph.states()`` transposed as rewards, giving
          one feature per state dimension. Returns array of shape
          ``(n_state_dims,)``.

    Returns
    -------
    np.ndarray or float
        Total reward-weighted sojourn time. Shape depends on ``rewards``.
    """
    indices = path['vertex_indices']
    times = path['entry_times']
    sojourn_times = np.diff(times)

    if rewards is None:
        rewards = graph.states().T  # (n_state_dims, n_vertices)

    rewards = np.asarray(rewards)

    if rewards.ndim == 1:
        # Single reward vector
        total = 0.0
        for step in range(len(sojourn_times)):
            v_idx = int(indices[step])
            total += sojourn_times[step] * rewards[v_idx]
        return float(total)
    else:
        # 2D: (n_features, n_vertices)
        n_features = rewards.shape[0]
        totals = np.zeros(n_features)
        for step in range(len(sojourn_times)):
            v_idx = int(indices[step])
            totals += sojourn_times[step] * rewards[:, v_idx]
        return totals


def path_exit_rates(graph, path: dict) -> tuple[np.ndarray, np.ndarray]:
    """Compute the total exit rate and sojourn time at each step of a path.

    The exit rate at a vertex is the sum of all outgoing edge weights.

    Parameters
    ----------
    graph : Graph
        The graph from which the path was sampled.
    path : dict
        Path dict from ``graph.sample_path()``.

    Returns
    -------
    exit_rates : np.ndarray, shape ``(n_steps,)``
        Total exit rate at each non-absorbing vertex in the path
        (excluding the starting vertex and absorbing vertex).
    sojourn_times : np.ndarray, shape ``(n_steps,)``
        Sojourn time at each corresponding vertex.
    """
    indices = path['vertex_indices']
    times = path['entry_times']
    vertices = graph.vertices()

    all_sojourn = np.diff(times)
    # Skip starting vertex (index 0, sojourn=0) and absorbing vertex (last, no edges)
    # The path is: start, v1, v2, ..., vK, absorb
    # sojourn_times[0] = time at start (0), sojourn_times[1] = time at v1, etc.
    # We want the rates and sojourns for v1, ..., vK (the transient states)

    rates = []
    sojourns = []
    for step in range(1, len(all_sojourn)):
        v_idx = int(indices[step])
        v = vertices[v_idx]
        if v.edges_length() == 0:
            break  # absorbing vertex
        rate = sum(float(e.weight()) for e in v.edges())
        rates.append(rate)
        sojourns.append(all_sojourn[step])

    return np.array(rates), np.array(sojourns)


def path_exit_rates_by_param(graph, path: dict) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-parameter exit rate contributions along a sampled path.

    For parameterized graphs, the exit rate at each vertex decomposes as:

    .. math::

        r_v = \\sum_p \\theta_p \\cdot \\sum_{\\text{edges from } v} c_{e,p}

    where ``c_{e,p}`` is the coefficient of parameter ``p`` on edge ``e``.
    This function returns the per-parameter rate contributions
    (``sum of coefficients * theta_p``) at each transient vertex.

    Parameters
    ----------
    graph : Graph
        The parameterized graph from which the path was sampled.
    path : dict
        Path dict from ``graph.sample_path()`` or
        ``graph.sample_path_conditioned()``.

    Returns
    -------
    rate_components : np.ndarray, shape ``(n_steps, n_params)``
        Per-parameter exit rate contribution at each transient vertex.
        ``rate_components[step, p]`` is ``theta_p * sum(coeffs_p)`` over
        all outgoing edges at that step's vertex.
    sojourn_times : np.ndarray, shape ``(n_steps,)``
        Sojourn time at each corresponding vertex.
    """
    indices = path['vertex_indices']
    times = path['entry_times']
    vertices = graph.vertices()
    n_params = graph.param_length()

    all_sojourn = np.diff(times)

    components = []
    sojourns = []
    for step in range(1, len(all_sojourn)):
        v_idx = int(indices[step])
        v = vertices[v_idx]
        if v.edges_length() == 0:
            break

        # Sum coefficients per parameter across all outgoing edges
        param_rates = np.zeros(n_params)
        for e in v.parameterized_edges():
            coeffs = e.edge_state(n_params)
            for p in range(len(coeffs)):
                param_rates[p] += coeffs[p]

        components.append(param_rates)
        sojourns.append(all_sojourn[step])

    return np.array(components), np.array(sojourns)


def importance_log_weight_from_rates(
    exit_rates_proposal: np.ndarray,
    exit_rates_target: np.ndarray,
    sojourn_times: np.ndarray,
) -> float:
    """Log importance weight from exit rate ratios along a path.

    For a continuous-time Markov chain path with transient vertices
    ``v_1, ..., v_K``, exit rates ``r_k``, and sojourn times ``s_k``,
    the density contribution from each vertex is
    ``r_k * exp(-r_k * s_k)``. The importance weight is the ratio of
    target to proposal densities:

    .. math::

        \\log w = \\sum_k \\left[
            \\log r^{\\text{target}}_k - \\log r^{\\text{proposal}}_k
            - (r^{\\text{target}}_k - r^{\\text{proposal}}_k) \\cdot s_k
        \\right]

    Parameters
    ----------
    exit_rates_proposal : array, shape ``(n_steps,)``
        Exit rates under the proposal at each transient vertex.
    exit_rates_target : array, shape ``(n_steps,)``
        Exit rates under the target at each transient vertex.
    sojourn_times : array, shape ``(n_steps,)``
        Sojourn time at each transient vertex.

    Returns
    -------
    float
        Log importance weight.
    """
    log_rate_ratio = np.log(exit_rates_target) - np.log(exit_rates_proposal)
    rate_diff = exit_rates_target - exit_rates_proposal
    return float(np.sum(log_rate_ratio - rate_diff * sojourn_times))


def importance_weighted_log_likelihood(
    log_likelihoods: jnp.ndarray,
    log_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Estimate log-likelihood from importance-weighted samples.

    Uses the log-sum-exp trick for numerical stability:

    .. math::

        \\log \\hat{P}(\\text{data} \\mid \\text{target})
        = -\\log M + \\text{logsumexp}(\\log L_m + \\log w_m)

    where ``M`` is the number of samples, ``L_m`` is the likelihood
    under path ``m``, and ``w_m`` is the importance weight.

    This is the pseudo-marginal likelihood estimator used in
    particle MCMC.

    Parameters
    ----------
    log_likelihoods : array, shape ``(M,)``
        Log-likelihood of observations under each sampled path.
    log_weights : array, shape ``(M,)``
        Log importance weights for each path.

    Returns
    -------
    scalar
        Estimated log-likelihood under the target model.
    """
    M = len(log_likelihoods)
    return logsumexp(log_likelihoods + log_weights) - jnp.log(M)
