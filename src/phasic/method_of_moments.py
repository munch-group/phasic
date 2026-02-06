"""
Method of moments estimation for phase-type distribution parameters.

Finds parameter estimates by matching model moments to sample moments,
providing sensible prior means for SVGD inference.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from numpy.typing import ArrayLike

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MoMResult:
    """Result of method-of-moments estimation.

    Attributes
    ----------
    theta : np.ndarray
        MoM estimate, shape ``(theta_dim,)``.
    std : np.ndarray
        Standard errors of the MoM estimator (delta method), shape ``(theta_dim,)``.
    prior : list
        List of ``GaussPrior`` (or ``None`` for fixed params), length ``theta_dim``.
    success : bool
        Whether the optimisation converged.
    residual : float
        Sum of squared residuals at the solution.
    sample_moments : np.ndarray
        Target moments computed from the data.
    model_moments : np.ndarray
        Model moments evaluated at ``theta``.
    message : str
        Solver status message.
    """
    theta: np.ndarray
    std: np.ndarray
    prior: list
    success: bool
    residual: float
    sample_moments: np.ndarray
    model_moments: np.ndarray
    message: str


def _estimate_moment_covariance(observed_data: np.ndarray, nr_moments: int, n_features: int) -> np.ndarray:
    """Estimate the covariance matrix of the sample moments.

    For each raw sample moment m_k = mean(X^k), its sampling variance is
    Var(m_k) = Var(X^k) / n.  The covariance between m_j and m_k is
    Cov(X^j, X^k) / n.

    For multivariate data (n_features > 1), the moments are ordered as
    [feature_0_moment_1, ..., feature_0_moment_K, feature_1_moment_1, ...]
    and the covariance is block-diagonal (features are independent samples).
    """
    n_obs = observed_data.shape[0]
    n_eq = n_features * nr_moments

    if observed_data.ndim == 1:
        # Univariate: build matrix of X^k columns
        powers = np.column_stack([observed_data ** k for k in range(1, nr_moments + 1)])
        # Cov(m_hat) = Cov(X^k) / n
        cov = np.cov(powers, rowvar=False, ddof=1) / n_obs
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        return cov
    else:
        # Multivariate: block-diagonal, one block per feature
        cov = np.zeros((n_eq, n_eq), dtype=np.float64)
        for f in range(n_features):
            col = observed_data[:, f]
            # Use only non-NaN observations for this feature
            valid = ~np.isnan(col)
            col_valid = col[valid]
            n_valid = len(col_valid)
            if n_valid < 2:
                # Not enough data — leave this block as zeros (will trigger fallback)
                continue
            powers = np.column_stack([col_valid ** k for k in range(1, nr_moments + 1)])
            block = np.cov(powers, rowvar=False, ddof=1) / n_valid
            if block.ndim == 0:
                block = block.reshape(1, 1)
            idx = f * nr_moments
            cov[idx:idx + nr_moments, idx:idx + nr_moments] = block
        return cov


def _reconstruct_theta(theta_free: np.ndarray, theta_dim: int, fixed: list[tuple[int, float]]) -> np.ndarray:
    """Insert fixed values back into the full theta vector."""
    if not fixed:
        return theta_free
    theta_full = np.empty(theta_dim, dtype=np.float64)
    fixed_dict = dict(fixed)
    j = 0
    for i in range(theta_dim):
        if i in fixed_dict:
            theta_full[i] = fixed_dict[i]
        else:
            theta_full[i] = theta_free[j]
            j += 1
    return theta_full


def _initial_guess_grid(moments_fn: object, target_first_moment: np.ndarray, n_free: int, fixed: list[tuple[int, float]], theta_dim: int) -> np.ndarray:
    """Coordinate-wise grid search matching the first moment.

    For each free parameter, searches a log-spaced grid while holding
    all other free parameters at 1.0, picking the value that brings
    the model's first moment closest to the data's first moment.
    """
    target_m1 = float(np.ravel(target_first_moment)[0])
    # Search range adapted to data scale
    upper = max(10.0 * abs(target_m1), 10.0) if target_m1 != 0 else 10.0
    grid = np.concatenate([
        np.logspace(-2, np.log10(upper), 20),
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
                model_mom = np.asarray(moments_fn(theta_full))
                model_m1 = float(np.ravel(model_mom)[0])
                err = abs(model_m1 - target_m1)
                if np.isfinite(err) and err < best_err:
                    best_err = err
                    best_val = c
            except Exception:
                continue
        theta_free[j] = best_val

    return theta_free


def method_of_moments(
    graph: object,
    observed_data: ArrayLike,
    nr_moments: int | None = None,
    theta_dim: int | None = None,
    theta_init: np.ndarray | None = None,
    rewards: np.ndarray | None = None,
    fixed: list[tuple[int, float]] | None = None,
    std_multiplier: float = 2.0,
    discrete: bool = False,
    verbose: bool = True,
) -> MoMResult:
    """Find parameter estimates by matching model moments to sample moments.

    Solves the nonlinear least-squares problem::

        minimize ||moments_fn(theta) - sample_moments||^2
        subject to  theta > 0

    using ``scipy.optimize.least_squares`` (Trust Region Reflective).

    Parameters
    ----------
    graph : Graph
        Parameterized phasic graph.
    observed_data : ArrayLike
        Observed data. 1-D for univariate, 2-D ``(n_times, n_features)``
        for multivariate models.
    nr_moments : int, optional
        Number of moments to match per feature dimension.  If ``None``
        (default), automatically chosen to overdetermine the system:
        ``max(2 * n_free_params, 4)`` for univariate data, adjusted
        downward per feature for multivariate data.
    theta_dim : int, optional
        Number of model parameters.  Inferred from the graph if not given.
    theta_init : np.ndarray, optional
        Initial guess for *free* parameters (after removing fixed ones).
        If ``None``, a coordinate-wise grid search is used.
    rewards : np.ndarray, optional
        Reward vectors.  ``None`` for standard moments, 1-D for a single
        reward vector, 2-D ``(n_features, n_vertices)`` for multivariate.
    fixed : list, optional
        List of ``(index, value)`` tuples pinning specific parameters.
    std_multiplier : float
        ``prior_std = std_multiplier * asymptotic_std``.
    discrete : bool
        Whether the model is discrete (PMF) or continuous (PDF).
    verbose : bool
        Print progress information.

    Returns
    -------
    MoMResult
        Dataclass with ``theta``, ``std``, ``prior``, etc.
    """
    import jax.numpy as jnp
    from scipy.optimize import least_squares

    from . import Graph
    from .svgd import compute_sample_moments, GaussPrior

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
    # 2. Determine data shape and number of features
    # ------------------------------------------------------------------
    observed_data = np.asarray(observed_data, dtype=np.float64)
    if observed_data.ndim == 1:
        n_features = 1
    elif observed_data.ndim == 2:
        n_features = observed_data.shape[1]
    else:
        raise ValueError(f"observed_data must be 1-D or 2-D, got shape {observed_data.shape}")

    # ------------------------------------------------------------------
    # 3. Choose nr_moments (auto or user-specified)
    # ------------------------------------------------------------------
    if nr_moments is None:
        # Auto-select: overdetermined by at least 2x, minimum 4 moments
        if n_features > 1:
            nr_moments = max(math.ceil(2 * n_free / n_features), 4)
        else:
            nr_moments = max(2 * n_free, 4)

    n_equations = n_features * nr_moments

    # Guard against user-specified values that are too low
    if n_equations < n_free:
        nr_moments_old = nr_moments
        nr_moments = math.ceil(n_free / max(1, n_features))
        n_equations = n_features * nr_moments
        if verbose:
            print(f"MoM: increased nr_moments from {nr_moments_old} to {nr_moments} "
                  f"(need >= {n_free} equations, have {n_equations})")

    # ------------------------------------------------------------------
    # 4. Build moments function
    # ------------------------------------------------------------------
    if rewards is not None:
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        if rewards_arr.ndim == 2:
            # Multivariate: 2-D rewards (n_features, n_vertices)
            model = Graph.pmf_and_moments_from_graph_multivariate(
                graph, nr_moments=nr_moments, discrete=discrete,
                use_ffi=False, theta_dim=theta_dim,
            )
            dummy_times = jnp.ones(1) if observed_data.ndim == 1 else jnp.ones((1, n_features))
            rewards_jnp = jnp.array(rewards_arr)

            def moments_fn(theta):
                _, moments = model(jnp.array(theta), dummy_times, rewards=rewards_jnp)
                return np.asarray(moments)
        else:
            # 1-D rewards
            model = Graph.pmf_and_moments_from_graph(
                graph, nr_moments=nr_moments, discrete=discrete,
                theta_dim=theta_dim,
            )
            dummy_times = jnp.ones(1)
            rewards_jnp = jnp.array(rewards_arr)

            def moments_fn(theta):
                _, moments = model(jnp.array(theta), dummy_times, rewards=rewards_jnp)
                return np.asarray(moments)
    else:
        # No rewards: use pmf_and_moments_from_graph, extract moments only
        model = Graph.pmf_and_moments_from_graph(
            graph, nr_moments=nr_moments, discrete=discrete,
            theta_dim=theta_dim,
        )
        dummy_times = jnp.ones(1)

        def moments_fn(theta):
            _, moments = model(jnp.array(theta), dummy_times)
            return np.asarray(moments)

    # ------------------------------------------------------------------
    # 5. Compute target (sample) moments
    # ------------------------------------------------------------------
    target_moments = np.asarray(compute_sample_moments(observed_data, nr_moments))

    if verbose:
        print(f"MoM: theta_dim={theta_dim}, n_free={n_free}, "
              f"nr_moments={nr_moments}, n_features={n_features}, "
              f"n_equations={n_equations}")
        print(f"MoM: sample moments =\n{target_moments}")

    # ------------------------------------------------------------------
    # 6. Initial guess
    # ------------------------------------------------------------------
    if theta_init is not None:
        x0 = np.asarray(theta_init, dtype=np.float64).ravel()
        if len(x0) != n_free:
            raise ValueError(
                f"theta_init has {len(x0)} elements but there are {n_free} free parameters"
            )
    else:
        # Determine first moment target for grid search
        if target_moments.ndim == 2:
            first_moment_target = target_moments[:, 0]
        else:
            first_moment_target = target_moments[0:1]

        x0 = _initial_guess_grid(
            moments_fn, first_moment_target, n_free, fixed, theta_dim
        )

    if verbose:
        theta_full_init = _reconstruct_theta(x0, theta_dim, fixed)
        print(f"MoM: initial guess (full theta) = {theta_full_init}")

    # ------------------------------------------------------------------
    # 7. Define residual
    # ------------------------------------------------------------------
    def residual_fn(theta_free):
        theta_full = _reconstruct_theta(theta_free, theta_dim, fixed)
        model_mom = moments_fn(theta_full)
        return model_mom.ravel() - target_moments.ravel()

    # ------------------------------------------------------------------
    # 8. Optimise
    # ------------------------------------------------------------------
    result = least_squares(
        residual_fn, x0,
        bounds=(1e-10, np.inf),
        method='trf',
        max_nfev=200 * n_free,
    )

    theta_free_opt = result.x
    theta_full_opt = _reconstruct_theta(theta_free_opt, theta_dim, fixed)

    # Final model moments
    model_moments_opt = moments_fn(theta_full_opt)

    residual_sum = float(np.sum(result.fun ** 2))

    if verbose:
        print(f"MoM: {'converged' if result.success else 'FAILED'} — {result.message}")
        print(f"MoM: theta = {theta_full_opt}")
        print(f"MoM: residual = {residual_sum:.6e}")
        print(f"MoM: model moments =\n{model_moments_opt}")

    # ------------------------------------------------------------------
    # 9. Estimate std via the delta method
    #
    # The sampling covariance of the MoM estimator is:
    #
    #   Cov(theta_hat) = G @ Cov(m_hat) @ G.T
    #
    # where G = (dm/dtheta)^{-1} (or pseudo-inverse) maps moment
    # perturbations to parameter perturbations, and Cov(m_hat) is
    # the covariance of the sample moments estimated from the data.
    # ------------------------------------------------------------------
    n_obs = observed_data.shape[0]
    std_full = np.full(theta_dim, np.nan)
    try:
        # Jacobian dm/dtheta_free at the solution, shape (n_eq, n_free)
        J = result.jac

        # Estimate Cov(m_hat) from the data
        moment_cov = _estimate_moment_covariance(observed_data, nr_moments, n_features)

        # G = pseudo-inverse of J: maps moment perturbations -> theta perturbations
        # J has shape (n_eq, n_free); we want G of shape (n_free, n_eq)
        G = np.linalg.pinv(J)

        # Delta method: Cov(theta_hat) = G @ Cov(m_hat) @ G.T
        cov_theta = G @ moment_cov @ G.T
        std_free = np.sqrt(np.maximum(np.diag(cov_theta), 0.0))
    except (np.linalg.LinAlgError, ValueError):
        # Fall back to coefficient-of-variation heuristic
        std_free = 0.5 * np.abs(theta_free_opt)
        if verbose:
            print("MoM: WARNING — delta method failed, using fallback std = 0.5 * |theta|")

    # Map std back to full theta vector
    j = 0
    for i in range(theta_dim):
        if i in fixed_dict:
            std_full[i] = 0.0
        else:
            std_full[i] = std_free[j]
            j += 1

    # Ensure std is never zero for free parameters (minimum floor)
    for i in range(theta_dim):
        if i not in fixed_dict and (std_full[i] <= 0 or not np.isfinite(std_full[i])):
            std_full[i] = 0.5 * max(abs(theta_full_opt[i]), 1e-4)

    # ------------------------------------------------------------------
    # 10. Build priors
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

    return MoMResult(
        theta=theta_full_opt,
        std=std_full,
        prior=priors,
        success=bool(result.success),
        residual=residual_sum,
        sample_moments=target_moments,
        model_moments=model_moments_opt,
        message=str(result.message),
    )
