"""
Probability-matching estimation for joint probability graph parameters.

For joint probability graphs (created via ``Graph.joint_prob_graph()``),
observations are feature-count tuples and the model outputs a probability
table.  This module finds parameter estimates by matching the empirical
probability distribution to the model probability distribution, providing
sensible prior means for SVGD inference.
"""

import numpy as np
from dataclasses import dataclass
from numpy.typing import ArrayLike
from typing import Optional

from .logging_config import get_logger
from .method_of_moments import _reconstruct_theta

logger = get_logger(__name__)


@dataclass
class ProbMatchResult:
    """Result of probability-matching estimation.

    Attributes
    ----------
    theta : np.ndarray
        Probability-matching estimate, shape ``(theta_dim,)``.
    std : np.ndarray
        Standard errors of the estimator (delta method), shape ``(theta_dim,)``.
    prior : list
        List of ``GaussPrior`` (or ``None`` for fixed params), length ``theta_dim``.
    success : bool
        Whether the optimisation converged.
    residual : float
        Sum of squared residuals at the solution.
    empirical_probs : np.ndarray
        Empirical probability vector for unique observations.
    model_probs : np.ndarray
        Model probability vector at the solution.
    unique_indices : np.ndarray
        Vertex indices corresponding to the unique observations.
    message : str
        Solver status message.
    """
    theta: np.ndarray
    std: np.ndarray
    prior: list
    success: bool
    residual: float
    empirical_probs: np.ndarray
    model_probs: np.ndarray
    unique_indices: np.ndarray
    message: str


def _estimate_multinomial_covariance(empirical_probs, n_obs):
    """Estimate the covariance matrix of the empirical proportions.

    For a multinomial with probabilities p_i and n observations,
    Cov(p_hat_i, p_hat_j) = (delta_ij * p_i - p_i * p_j) / n.

    Parameters
    ----------
    empirical_probs : np.ndarray
        Empirical probabilities, shape ``(K,)``.
    n_obs : int
        Number of observations.

    Returns
    -------
    np.ndarray
        Covariance matrix, shape ``(K, K)``.
    """
    K = len(empirical_probs)
    cov = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            if i == j:
                cov[i, j] = (empirical_probs[i] - empirical_probs[i] ** 2) / n_obs
            else:
                cov[i, j] = -empirical_probs[i] * empirical_probs[j] / n_obs
    return cov


def _initial_guess_grid_probs(probs_fn, target_prob, n_free, fixed, theta_dim):
    """Coordinate-wise grid search matching the most common observation probability.

    For each free parameter, searches a log-spaced grid while holding
    all other free parameters at 1.0, picking the value that brings
    the model probability of the most common observation closest to
    the empirical probability.

    Parameters
    ----------
    probs_fn : callable
        Function theta -> model_probs array.
    target_prob : float
        Empirical probability of the most common observation.
    n_free : int
        Number of free parameters.
    fixed : list
        List of ``(index, value)`` tuples for fixed parameters.
    theta_dim : int
        Total number of parameters.

    Returns
    -------
    np.ndarray
        Initial guess for free parameters, shape ``(n_free,)``.
    """
    grid = np.concatenate([
        np.logspace(-2, 2, 30),
    ])

    theta_free = np.ones(n_free, dtype=np.float64)

    for j in range(n_free):
        best_val = theta_free[j]
        best_err = np.inf
        for c in grid:
            theta_try = theta_free.copy()
            theta_try[j] = c
            theta_full = _reconstruct_theta(theta_try, theta_dim, fixed)
            try:
                model_probs = np.asarray(probs_fn(theta_full))
                # Match the maximum model probability to the target
                model_max = float(np.max(model_probs))
                err = abs(model_max - target_prob)
                if np.isfinite(err) and err < best_err:
                    best_err = err
                    best_val = c
            except Exception:
                continue
        theta_free[j] = best_val

    return theta_free


def probability_matching(
    graph,
    observed_indices: ArrayLike,
    theta_dim: int = None,
    theta_init: np.ndarray = None,
    fixed: list = None,
    std_multiplier: float = 2.0,
    verbose: bool = True,
) -> ProbMatchResult:
    """Find parameter estimates by matching model probabilities to empirical probabilities.

    For joint probability graphs, observations are vertex indices (integers).
    The model produces a probability table via normalized expected sojourn times.
    This function minimises::

        ||model_probs(theta) - empirical_probs||^2
        subject to  theta > 0

    using ``scipy.optimize.least_squares`` (Trust Region Reflective).

    Parameters
    ----------
    graph : Graph
        Joint probability graph (must have ``_joint_prob_base_graph_indexer``).
    observed_indices : ArrayLike
        Vertex indices (integers) for each observation, as produced by
        the ``Graph.probability_matching()`` wrapper.
    theta_dim : int, optional
        Number of model parameters.  Inferred from the graph if not given.
    theta_init : np.ndarray, optional
        Initial guess for *free* parameters (after removing fixed ones).
        If ``None``, a coordinate-wise grid search is used.
    fixed : list, optional
        List of ``(index, value)`` tuples pinning specific parameters.
    std_multiplier : float
        ``prior_std = std_multiplier * asymptotic_std``.
    verbose : bool
        Print progress information.

    Returns
    -------
    ProbMatchResult
        Dataclass with ``theta``, ``std``, ``prior``, etc.
    """
    import jax.numpy as jnp
    from scipy.optimize import least_squares

    from . import Graph
    from .svgd import GaussPrior
    from .ffi_wrappers import compute_sojourn_times_ffi

    # ------------------------------------------------------------------
    # 1. Infer dimensions
    # ------------------------------------------------------------------
    if theta_dim is None:
        theta_dim = graph.param_length()
    if theta_dim == 0:
        raise ValueError(
            "theta_dim could not be inferred. The graph has no parameterized edges, "
            "or specify theta_dim explicitly."
        )

    if fixed is None:
        fixed = []
    fixed_dict = dict(fixed)
    n_free = theta_dim - len(fixed)
    if n_free <= 0:
        raise ValueError("All parameters are fixed; nothing to optimise.")

    # ------------------------------------------------------------------
    # 2. Compute empirical probabilities from observed indices
    # ------------------------------------------------------------------
    observed_indices = np.asarray(observed_indices, dtype=np.int64)
    n_obs = len(observed_indices)

    unique_indices, counts = np.unique(observed_indices, return_counts=True)
    empirical_probs = counts.astype(np.float64) / n_obs
    n_unique = len(unique_indices)

    if verbose:
        print(f"ProbMatch: theta_dim={theta_dim}, n_free={n_free}, "
              f"n_obs={n_obs}, n_unique_obs={n_unique}")

    # ------------------------------------------------------------------
    # 3. Build model probability function
    # ------------------------------------------------------------------
    serialized = graph.serialize(theta_dim=theta_dim)

    # Find ALL terminal vertex indices for normalization
    all_terminal_indices = []
    for vertex in graph.vertices():
        for edge in vertex.edges():
            if len(edge.to().edges()) == 0:
                all_terminal_indices.append(vertex.index())
                break
    all_terminal_indices_arr = jnp.array(sorted(set(all_terminal_indices)), dtype=jnp.int32)
    unique_indices_jnp = jnp.array(unique_indices, dtype=jnp.int32)

    def probs_fn(theta_full):
        """Compute model probabilities for unique observed indices."""
        theta_jnp = jnp.array(theta_full)
        # Sojourn times for observed indices
        sojourn_obs = compute_sojourn_times_ffi(serialized, theta_jnp, unique_indices_jnp)
        # Normalization over all terminal vertices
        sojourn_all = compute_sojourn_times_ffi(serialized, theta_jnp, all_terminal_indices_arr)
        norm = jnp.sum(sojourn_all)
        return np.asarray(sojourn_obs / norm)

    # ------------------------------------------------------------------
    # 4. Initial guess
    # ------------------------------------------------------------------
    if theta_init is not None:
        x0 = np.asarray(theta_init, dtype=np.float64).ravel()
        if len(x0) != n_free:
            raise ValueError(
                f"theta_init has {len(x0)} elements but there are {n_free} free parameters"
            )
    else:
        # Grid search: match the most common observation's probability
        most_common_prob = float(np.max(empirical_probs))
        x0 = _initial_guess_grid_probs(
            probs_fn, most_common_prob, n_free, fixed, theta_dim
        )

    if verbose:
        theta_full_init = _reconstruct_theta(x0, theta_dim, fixed)
        print(f"ProbMatch: initial guess (full theta) = {theta_full_init}")

    # ------------------------------------------------------------------
    # 5. Define residual
    # ------------------------------------------------------------------
    def residual_fn(theta_free):
        theta_full = _reconstruct_theta(theta_free, theta_dim, fixed)
        model_probs = probs_fn(theta_full)
        return model_probs.ravel() - empirical_probs.ravel()

    # ------------------------------------------------------------------
    # 6. Optimise
    # ------------------------------------------------------------------
    result = least_squares(
        residual_fn, x0,
        bounds=(1e-10, np.inf),
        method='trf',
        max_nfev=200 * n_free,
    )

    theta_free_opt = result.x
    theta_full_opt = _reconstruct_theta(theta_free_opt, theta_dim, fixed)

    # Final model probabilities
    model_probs_opt = probs_fn(theta_full_opt)

    residual_sum = float(np.sum(result.fun ** 2))

    if verbose:
        print(f"ProbMatch: {'converged' if result.success else 'FAILED'} — {result.message}")
        print(f"ProbMatch: theta = {theta_full_opt}")
        print(f"ProbMatch: residual = {residual_sum:.6e}")

    # ------------------------------------------------------------------
    # 7. Estimate std via the delta method
    #
    # Cov(theta_hat) = G @ Cov(p_hat) @ G.T
    # where G = pinv(J), J = dp/dtheta at solution
    # Cov(p_hat) = multinomial covariance
    # ------------------------------------------------------------------
    std_full = np.full(theta_dim, np.nan)
    try:
        J = result.jac  # shape (n_unique, n_free)
        multinomial_cov = _estimate_multinomial_covariance(empirical_probs, n_obs)
        G = np.linalg.pinv(J)  # shape (n_free, n_unique)
        cov_theta = G @ multinomial_cov @ G.T
        std_free = np.sqrt(np.maximum(np.diag(cov_theta), 0.0))
    except (np.linalg.LinAlgError, ValueError):
        std_free = 0.5 * np.abs(theta_free_opt)
        if verbose:
            print("ProbMatch: WARNING — delta method failed, using fallback std = 0.5 * |theta|")

    # Map std back to full theta vector
    j = 0
    for i in range(theta_dim):
        if i in fixed_dict:
            std_full[i] = 0.0
        else:
            std_full[i] = std_free[j]
            j += 1

    # Ensure std is never zero for free parameters
    for i in range(theta_dim):
        if i not in fixed_dict and (std_full[i] <= 0 or not np.isfinite(std_full[i])):
            std_full[i] = 0.5 * max(abs(theta_full_opt[i]), 1e-4)

    # ------------------------------------------------------------------
    # 8. Build priors
    # ------------------------------------------------------------------
    priors = []
    for i in range(theta_dim):
        if i in fixed_dict:
            priors.append(None)
        else:
            priors.append(GaussPrior(
                mean=float(theta_full_opt[i]),
                std=float(std_multiplier * std_full[i]),
            ))

    return ProbMatchResult(
        theta=theta_full_opt,
        std=std_full,
        prior=priors,
        success=bool(result.success),
        residual=residual_sum,
        empirical_probs=empirical_probs,
        model_probs=model_probs_opt,
        unique_indices=unique_indices,
        message=str(result.message),
    )
