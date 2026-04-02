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


def bffg_log_prob(jg_disc, jg_continuous, theta_proposal, theta_target_fn,
                  observed_data, n_paths=5, zero_mut_idx=None,
                  theta_proposal_fn=None):
    """Create a BFFG log-probability function for inhomogeneous models.

    Constructs a ``log_prob_fn(theta) -> scalar`` suitable for
    ``MCMC(log_prob_fn=...)``. Internally computes:

    .. math::

        \\log P_{\\text{target}}(s) = \\log P_{\\text{model}}(s \\mid \\theta)
        + \\log \\hat{E}[w]

    where :math:`P_{\\text{model}}(s \\mid \\theta)` is the proposal
    model evaluated at the MCMC parameters (using the discrete graph),
    and :math:`\\hat{E}[w]` corrects from the smooth proposal to the
    true step-function target via conditioned path sampling.

    Parameters
    ----------
    jg_disc : Graph
        Discrete joint_prob_graph for computing SFS probabilities.
        ``update_weights`` is called with ``theta_proposal_fn(theta_mcmc)``
        at each MCMC evaluation.
    jg_continuous : Graph
        Continuous joint_prob_graph (``is_discrete=False``) for
        conditioned path sampling and importance weight computation.
    theta_proposal : array
        Proposal parameter vector used in ``jg_continuous``.
    theta_target_fn : callable
        Maps ``(theta_mcmc, t) -> theta_at_time``. Given the MCMC
        parameter vector and a time along the path, returns the
        target parameter vector at that time.
    observed_data : array of int
        Terminal vertex indices (one per locus).
    n_paths : int, default=5
        Number of conditioned paths per locus per evaluation.
    zero_mut_idx : int or None
        Terminal vertex index for zero mutations. If provided,
        excluded from the normalization.
    theta_proposal_fn : callable, optional
        Maps ``theta_mcmc -> theta_graph``. Converts the MCMC
        parameter vector to the graph parameter vector for
        evaluating ``jg_disc``. If None, defaults to the fixed
        ``theta_proposal``.

    Returns
    -------
    callable
        ``log_prob_fn(theta) -> float`` for use with
        ``MCMC(log_prob_fn=...)``.
    """
    theta_proposal = np.asarray(theta_proposal)
    observed_data = np.asarray(observed_data)
    n_loci = len(observed_data)

    # Precompute and cache edge coefficients and targets for every vertex.
    # This avoids repeated Python->C++ calls during weight computation.
    vertices = jg_continuous.vertices()
    n_params = jg_continuous.param_length()
    n_verts = jg_continuous.vertices_length()

    _vertex_edge_coeffs = [None] * n_verts   # np.array per vertex
    _vertex_edge_targets = [None] * n_verts  # list of target indices
    _vertex_prop_rates = [None] * n_verts    # edge_coeffs @ theta_proposal

    for vi in range(n_verts):
        v = vertices[vi]
        if v.edges_length() == 0:
            continue
        edges = v.parameterized_edges()
        if not edges:
            continue
        coeffs = np.array([list(e.edge_state(n_params)) for e in edges])
        # Ensure coeffs has n_params columns (pad if needed)
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape(1, -1)
        if coeffs.shape[1] < n_params:
            coeffs = np.pad(coeffs, ((0,0),(0, n_params - coeffs.shape[1])))
        targets = [e.to().index() for e in edges]
        _vertex_edge_coeffs[vi] = coeffs
        _vertex_edge_targets[vi] = targets
        _vertex_prop_rates[vi] = coeffs @ theta_proposal

    def _full_importance_log_weight(path, theta_target_fn_bound):
        """Complete importance weight: exit rate ratio + transition prob ratio."""
        indices = path['vertex_indices']
        times = path['entry_times']
        sojourns = np.diff(times)

        log_w = 0.0
        for step in range(1, len(sojourns)):
            vi = int(indices[step])
            coeffs = _vertex_edge_coeffs[vi]
            if coeffs is None:
                break  # absorbing vertex

            theta_t = theta_target_fn_bound(times[step])
            edge_rates_tgt = coeffs @ np.asarray(theta_t)
            edge_rates_prop = _vertex_prop_rates[vi]

            r_prop = edge_rates_prop.sum()
            r_tgt = edge_rates_tgt.sum()

            # Find which edge was taken
            next_idx = int(indices[step + 1]) if step + 1 < len(indices) else -1
            targets = _vertex_edge_targets[vi]
            taken_edge = None
            for ei, t_idx in enumerate(targets):
                if t_idx == next_idx:
                    taken_edge = ei
                    break

            if taken_edge is None or r_prop <= 0 or r_tgt <= 0:
                continue

            s_k = sojourns[step]
            log_w += np.log(r_tgt) - np.log(r_prop) - (r_tgt - r_prop) * s_k

            p_prop = edge_rates_prop[taken_edge] / r_prop
            p_tgt = edge_rates_tgt[taken_edge] / r_tgt
            if p_prop > 0 and p_tgt > 0:
                log_w += np.log(p_tgt) - np.log(p_prop)

        return log_w

    def log_prob_fn(theta_mcmc):
        """BFFG log-probability for inhomogeneous model."""
        theta_mcmc_np = np.asarray(theta_mcmc)

        # Evaluate model at theta_mcmc (not fixed proposal)
        if theta_proposal_fn is not None:
            theta_graph = theta_proposal_fn(theta_mcmc_np)
        else:
            theta_graph = theta_proposal
        jg_disc.update_weights(list(theta_graph))
        jpt = jg_disc.joint_prob_table()

        # Build ascertained log-probabilities for this theta
        total_prob = jpt['prob'].sum()
        if zero_mut_idx is not None:
            total_prob -= float(jpt.loc[zero_mut_idx, 'prob'])
        log_norm = np.log(total_prob) if total_prob > 0 else -1e10

        total = 0.0
        for locus in range(n_loci):
            target_v = int(observed_data[locus])
            p = float(jpt.loc[target_v, 'prob'])
            if p <= 0:
                total += -1e10
                continue
            log_p_model = np.log(p) - log_norm

            # Importance weight correction
            log_weights = np.empty(n_paths)
            theta_fn_bound = lambda t: theta_target_fn(theta_mcmc_np, t)

            for m in range(n_paths):
                path = jg_continuous.sample_path_conditioned([target_v])
                log_weights[m] = _full_importance_log_weight(path, theta_fn_bound)

            log_ratio = float(importance_weighted_log_likelihood(
                jnp.zeros(n_paths), jnp.array(log_weights)
            ))
            total += log_p_model + log_ratio

        return total

    return log_prob_fn
