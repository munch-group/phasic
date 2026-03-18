"""
Method of moments estimation for phase-type distribution parameters.

Finds parameter estimates by matching model moments to sample moments,
providing sensible prior means for SVGD inference.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass

from .logging_config import get_logger
from .svgd import SparseObservations

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
    weighted : bool
        Whether optimal weighting was used in the estimation.
    """
    theta: np.ndarray
    std: np.ndarray
    prior: list
    success: bool
    residual: float
    sample_moments: np.ndarray
    model_moments: np.ndarray
    message: str
    weighted: bool = False


def _compute_weight_transform(moment_cov: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Compute Lᵀ such that ||Lᵀ r||² = rᵀ W r with W = Σ⁻¹.

    Returns (L_transpose, method) where method is 'cholesky', 'diagonal', or 'identity'.
    Tries full Cholesky first, falls back to diagonal, then identity.
    """
    # Try full Cholesky: W = Σ⁻¹ = (L L^T)⁻¹, so ||L⁻¹ r||² = rᵀ Σ⁻¹ r
    try:
        cond = np.linalg.cond(moment_cov)
        if cond < 1e10 and np.all(np.diag(moment_cov) > 0):
            L = np.linalg.cholesky(moment_cov)
            Lt = np.linalg.inv(L)
            return Lt, 'cholesky'
    except np.linalg.LinAlgError:
        pass

    # Fallback: diagonal weighting W_ii = 1/Var(m_i)
    diag = np.diag(moment_cov)
    if np.all(diag > 0) and np.all(np.isfinite(diag)):
        Lt = np.diag(1.0 / np.sqrt(diag))
        return Lt, 'diagonal'

    # Final fallback: unweighted
    return None, 'identity'


def _select_nr_moments(
    observed_data: SparseObservations,
    n_features: int,
    n_free: int,
    max_nr_moments: int,
    cond_threshold: float = 1e3,
) -> int:
    """Select nr_moments by pruning from max until Σ is well-conditioned.

    Starts at max_nr_moments and reduces until the moment covariance matrix
    has condition number below cond_threshold, or reaches the minimum
    (enough equations to match free parameters).
    """
    min_nr = max(math.ceil(n_free / max(1, n_features)), 1)

    for nr in range(max_nr_moments, min_nr - 1, -1):
        n_eq = n_features * nr
        if n_eq < n_free:
            continue
        try:
            cov = _estimate_moment_covariance(observed_data, nr, n_features)
            cond = np.linalg.cond(cov)
            if cond < cond_threshold:
                return nr
        except (np.linalg.LinAlgError, ValueError):
            continue

    # If nothing is well-conditioned, use minimum that gives enough equations
    return max(min_nr, 1)


def _estimate_moment_covariance(observed_data: SparseObservations, nr_moments: int, n_features: int) -> np.ndarray:
    """Estimate the covariance matrix of the sample moments.

    For each raw sample moment m_k = mean(X^k), its sampling variance is
    Var(m_k) = Var(X^k) / n.  The covariance between m_j and m_k is
    Cov(X^j, X^k) / n.

    The moments are ordered as
    [feature_0_moment_1, ..., feature_0_moment_K, feature_1_moment_1, ...]
    and the covariance is block-diagonal (features are independent samples).
    """
    n_eq = n_features * nr_moments
    cov = np.zeros((n_eq, n_eq), dtype=np.float64)

    for f in range(n_features):
        col_valid = np.asarray(observed_data.get_feature_values(f))
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
    observed_data: np.ndarray | SparseObservations,
    nr_moments: int | None = None,
    theta_dim: int | None = None,
    theta_init: np.ndarray | None = None,
    rewards: np.ndarray | None = None,
    fixed: list[tuple[int, float]] | None = None,
    std_multiplier: float = 2.0,
    discrete: bool = False,
    verbose: bool = True,
    weighted: bool = True,
) -> MoMResult:
    """Find parameter estimates by matching model moments to sample moments.

    Solves the nonlinear least-squares problem::

        minimize ||moments_fn(theta) - sample_moments||^2
        subject to  theta > 0

    using ``scipy.optimize.least_squares`` (Trust Region Reflective).

    When ``weighted=True`` (default), uses two-step efficient GMM:
    Step 1 runs unweighted LS, then Step 2 re-runs with the optimal
    weighting matrix ``W = Σ⁻¹`` using the Step 1 solution as starting
    point.  This down-weights high-variance moments.

    Parameters
    ----------
    graph : Graph
        Parameterized phasic graph.
    observed_data : array or SparseObservations
        Observed data.  A plain 1-D array is accepted for univariate
        models.  For multivariate models use ``SparseObservations``
        (see ``dense_to_sparse()``).
    nr_moments : int, optional
        Number of moments to match per feature dimension.  If ``None``
        (default), automatically chosen based on ``weighted``:
        when ``weighted=True``, adaptive selection prunes high-order
        moments by condition number; when ``weighted=False``, uses
        the heuristic ``max(2 * n_free_params, 4)``.
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
    weighted : bool
        If ``True`` (default), use two-step efficient GMM with optimal
        weighting matrix ``W = Σ⁻¹``.  If ``False``, use unweighted
        least squares (legacy behavior).

    Returns
    -------
    MoMResult
        Dataclass with ``theta``, ``std``, ``prior``, etc.
    """
    import jax.numpy as jnp
    from scipy.optimize import least_squares

    from . import Graph
    import logging as _logging
    from .svgd import compute_sample_moments, GaussPrior

    # Enable INFO output when verbose=True
    if verbose:
        _prev_level = logger.getEffectiveLevel()
        if _prev_level > _logging.INFO:
            logger.setLevel(_logging.INFO)

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
    if not isinstance(observed_data, SparseObservations):
        # Accept a plain 1-D array for univariate data
        import jax.numpy as jnp
        arr = np.asarray(observed_data, dtype=np.float64)
        if arr.ndim != 1:
            raise TypeError(
                "observed_data must be a 1-D array or SparseObservations. "
                "For multivariate data, use dense_to_sparse()."
            )
        if np.any(np.isnan(arr)):
            raise ValueError(
                "observed_data contains NaN values. "
                "Use SparseObservations for data with missing values."
            )
        observed_data = SparseObservations(
            values=jnp.asarray(arr),
            features=jnp.zeros(len(arr), dtype=jnp.int32),
            n_features=1,
            slices=((0, len(arr)),),
        )

    if isinstance(observed_data, SparseObservations) and observed_data.n_features > 1 and rewards is None:
        raise ValueError(
            "rewards must be provided for multivariate SparseObservations (n_features > 1)."
        )
    n_features = observed_data.n_features

    # ------------------------------------------------------------------
    # 3. Choose nr_moments (auto or user-specified)
    # ------------------------------------------------------------------
    if nr_moments is None:
        if weighted:
            # Adaptive selection: start from old heuristic, prune by condition number
            if n_features > 1:
                max_nr = max(math.ceil(2 * n_free / n_features), 4)
            else:
                max_nr = max(2 * n_free, 4)
            nr_moments = _select_nr_moments(
                observed_data, n_features, n_free, max_nr,
            )
        else:
            # Legacy heuristic: overdetermined by at least 2x, minimum 4 moments
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
            logger.info(f"increased nr_moments from {nr_moments_old} to {nr_moments} "
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
            dummy_times = jnp.ones(1) if n_features == 1 else jnp.ones((1, n_features))
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
        logger.info(f"theta_dim={theta_dim}, n_free={n_free}, "
                    f"nr_moments={nr_moments}, n_features={n_features}, "
                    f"n_equations={n_equations}")
        logger.info(f"sample moments =\n{target_moments}")

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
        logger.info(f"initial guess (full theta) = {theta_full_init}")

    # ------------------------------------------------------------------
    # 7. Define residual and compute weight transform
    # ------------------------------------------------------------------
    def residual_fn(theta_free):
        theta_full = _reconstruct_theta(theta_free, theta_dim, fixed)
        model_mom = moments_fn(theta_full)
        return model_mom.ravel() - target_moments.ravel()

    # Compute moment covariance for weighting and std estimation
    moment_cov = _estimate_moment_covariance(observed_data, nr_moments, n_features)

    # ------------------------------------------------------------------
    # 8. Optimise (two-step GMM if weighted=True)
    # ------------------------------------------------------------------
    # Step 1: unweighted least squares
    result = least_squares(
        residual_fn, x0,
        bounds=(1e-10, np.inf),
        method='trf',
        max_nfev=200 * n_free,
    )

    # Step 2: weighted least squares (if requested and weights available)
    applied_weighting = False
    if weighted:
        Lt, weight_method = _compute_weight_transform(moment_cov)
        if Lt is not None:
            def weighted_residual_fn(theta_free):
                r = residual_fn(theta_free)
                return Lt @ r

            result_weighted = least_squares(
                weighted_residual_fn, result.x,
                bounds=(1e-10, np.inf),
                method='trf',
                max_nfev=200 * n_free,
            )
            result = result_weighted
            applied_weighting = True
            if verbose:
                logger.info(f"weighted GMM step 2 ({weight_method}): "
                            f"{'converged' if result.success else 'FAILED'}")
        elif verbose:
            logger.info("weighted GMM: fallback to unweighted (moment covariance singular)")

    theta_free_opt = result.x
    theta_full_opt = _reconstruct_theta(theta_free_opt, theta_dim, fixed)

    # Final model moments
    model_moments_opt = moments_fn(theta_full_opt)

    residual_sum = float(np.sum(result.fun ** 2))

    if verbose:
        logger.info(f"{'converged' if result.success else 'FAILED'} — {result.message}")
        logger.info(f"theta = {theta_full_opt}")
        logger.info(f"residual = {residual_sum:.6e}")
        logger.info(f"model moments =\n{model_moments_opt}")

    # ------------------------------------------------------------------
    # 9. Estimate std via the delta method
    #
    # Unweighted: Cov(theta_hat) = G @ Sigma @ G.T
    #   where G = pinv(J_u), J_u = unweighted Jacobian
    #
    # Weighted GMM with W = Sigma^{-1}:
    #   Cov(theta_hat) = (J_w^T J_w)^{-1}
    #   where J_w = Lt @ J_u = result.jac (the weighted Jacobian)
    # ------------------------------------------------------------------
    n_obs = len(observed_data.values)
    std_full = np.full(theta_dim, np.nan)
    try:
        if applied_weighting:
            # Weighted: result.jac = Lt @ J_u, so (J_w^T J_w)^{-1} is the
            # efficient GMM sandwich estimator
            J_w = result.jac
            JtJ = J_w.T @ J_w
            cov_theta = np.linalg.pinv(JtJ)
        else:
            # Unweighted: standard delta method
            J = result.jac
            G = np.linalg.pinv(J)
            cov_theta = G @ moment_cov @ G.T

        std_free = np.sqrt(np.maximum(np.diag(cov_theta), 0.0))
    except (np.linalg.LinAlgError, ValueError):
        # Fall back to coefficient-of-variation heuristic
        std_free = 0.5 * np.abs(theta_free_opt)
        if verbose:
            logger.warning("delta method failed, using fallback std = 0.5 * |theta|")

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

    # Restore logger level if we changed it
    if verbose and _prev_level > _logging.INFO:
        logger.setLevel(_prev_level)

    return MoMResult(
        theta=theta_full_opt,
        std=std_full,
        prior=priors,
        success=bool(result.success),
        residual=residual_sum,
        sample_moments=target_moments,
        model_moments=model_moments_opt,
        message=str(result.message),
        weighted=applied_weighting,
    )
