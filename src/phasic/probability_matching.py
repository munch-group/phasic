"""
Probability-matching estimation for joint probability graph parameters.

For joint probability graphs (created via ``Graph.joint_prob_graph()``),
observations are feature-count tuples and the model outputs a probability
table.  This module finds parameter estimates by matching the empirical
probability distribution to the model probability distribution, providing
sensible prior means for SVGD inference.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from numpy.typing import ArrayLike

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
    exposure_groups : int
        Number of exposure groups the fit used (1 without exposure).
    exposure_binned : bool
        True when there were more distinct exposure values than
        ``max_exposure_groups`` and observations were grouped into quantile
        bins of exposure (each bin at its mean exposure).

    With exposure, ``empirical_probs``, ``model_probs`` and
    ``unique_indices`` are stacked group by group.
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
    exposure_groups: int = 1
    exposure_binned: bool = False


def _estimate_multinomial_covariance(empirical_probs: np.ndarray, n_obs: int) -> np.ndarray:
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


def _initial_guess_grid_probs(probs_fn: object, target_prob: float, n_free: int, fixed: list[tuple[int, float]], theta_dim: int) -> np.ndarray:
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
    graph: object,
    observed_indices: ArrayLike,
    theta_dim: int | None = None,
    theta_init: np.ndarray | None = None,
    fixed: list[tuple[int, float]] | None = None,
    std_multiplier: float = 2.0,
    verbose: bool = True,
    lower: float | None = 0.0,
    exposure: ArrayLike | float | None = None,
    exposure_param_index: int | None = None,
    max_exposure_groups: int = 100,
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
    lower : {0.0, None}, default=0.0
        ``lower`` of the returned GaussPrior objects: ``0.0`` (truncated at
        zero) for use with ``positive_params=True``, ``None`` for
        ``positive_params=False``.
    exposure : float or array-like, optional
        Per-observation exposure alpha_i (a scalar applies to every
        observation). Observation i is modelled with
        ``theta[exposure_param_index]`` multiplied by alpha_i, as on the
        epoch (daisy-chain) path. Observations are grouped by exposure value;
        each group's outcome frequencies are matched to the model table at
        its own exposure, with residuals weighted by sqrt(n_g / n).
        Requires ``exposure_param_index``.
    exposure_param_index : int, optional
        The parameter that exposure multiplies.
    max_exposure_groups : int, default=100
        With more distinct exposure values than this, observations are
        grouped into this many quantile bins of exposure, each evaluated at
        the mean exposure of its members (an approximation that only affects
        this estimate, e.g. a default prior's centre and width), and
        ``ProbMatchResult.exposure_binned`` is set.

    Raises
    ------
    ValueError
        On a joint probability graph with no parameter fixed: multiplying
        every rate by the same factor leaves every joint probability
        unchanged (continuous and discrete graphs, with or without
        exposure), so the data cannot determine the overall scale. Also when
        the delta-method standard errors cannot be computed, or are zero or
        non-finite for a free parameter (the data do not determine it).

    Returns
    -------
    ProbMatchResult
        Dataclass with ``theta``, ``std``, ``prior``, etc.
    """
    from . import _require_parameterized
    _require_parameterized(graph, 'probability_matching')
    import jax.numpy as jnp
    from scipy.optimize import least_squares

    from . import Graph
    from .svgd import GaussPrior
    from .ffi_wrappers import compute_sojourn_times_ffi, _ensure_json_string

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
    # Validate before casting: a 2-D array of feature-count tuples would
    # otherwise be flattened by np.unique and its counts read as vertex
    # indices, and floats would be truncated silently.
    _obs_raw = np.asarray(observed_indices)
    if _obs_raw.ndim != 1:
        raise ValueError(
            f"observed_indices must be a 1-D sequence of vertex indices; got "
            f"an array of shape {_obs_raw.shape}. Feature-count tuples go "
            f"through Graph.probability_matching(), which maps them to "
            f"vertex indices."
        )
    if _obs_raw.size == 0:
        raise ValueError("observed_indices is empty.")
    if _obs_raw.dtype == np.bool_ or not np.issubdtype(_obs_raw.dtype, np.integer):
        raise TypeError(
            f"observed_indices must be integer vertex indices; got dtype "
            f"{_obs_raw.dtype}."
        )
    observed_indices = _obs_raw.astype(np.int64)
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
    # Encode the graph to JSON once: compute_sojourn_times_ffi re-encodes a
    # dict on every call (~0.13 s at 45k vertices, 158 calls in a default
    # prior build; ticket T6). A string is passed through unchanged.
    serialized = _ensure_json_string(graph.serialize(theta_dim=theta_dim))

    # Find ALL terminal vertex indices for normalization
    all_terminal_indices = []
    for vertex in graph.vertices():
        for edge in vertex.edges():
            if len(edge.to().edges()) == 0:
                all_terminal_indices.append(vertex.index())
                break
    _terminal_set = set(all_terminal_indices)
    _not_outcome = [int(i) for i in unique_indices if int(i) not in _terminal_set]
    if _not_outcome:
        raise ValueError(
            f"observed_indices contains {len(_not_outcome)} vertex index(es) "
            f"that are not outcome vertices of this joint probability graph "
            f"(first: {_not_outcome[:5]}). Pass the vertex indices that "
            f"Graph.probability_matching() maps observations to."
        )
    all_terminal_indices_arr = jnp.array(sorted(_terminal_set), dtype=jnp.int32)
    unique_indices_jnp = jnp.array(unique_indices, dtype=jnp.int32)

    # ------------------------------------------------------------------
    # 3a. Identifiability: the overall rate scale (structural, exact)
    # ------------------------------------------------------------------
    if graph._joint_prob_base_graph_indexer is not None and not fixed:
        raise ValueError(
            "probability_matching: no parameter is fixed, and on a joint "
            "probability graph multiplying every rate by the same factor "
            "leaves every joint probability unchanged, so the data cannot "
            "determine the overall scale of the rates. Fix one rate with "
            "fixed=[(index, value)], or pass prior= explicitly."
        )

    # ------------------------------------------------------------------
    # 3b. Exposure groups
    # ------------------------------------------------------------------
    exposure_binned = False
    groups = None          # None: no exposure (single group, unscaled)
    if exposure is not None:
        if exposure_param_index is None:
            raise ValueError(
                "probability_matching: exposure requires exposure_param_index."
            )
        k_exp = int(exposure_param_index)
        if not (0 <= k_exp < theta_dim):
            raise ValueError(
                f"probability_matching: exposure_param_index={exposure_param_index} "
                f"out of range [0, {theta_dim})."
            )
        alpha = np.asarray(exposure, dtype=np.float64).ravel()
        if alpha.size == 1:
            alpha = np.full(n_obs, float(alpha[0]))
        if alpha.size != n_obs:
            raise ValueError(
                f"probability_matching: exposure has {alpha.size} values for "
                f"{n_obs} observations."
            )
        if not np.all(np.isfinite(alpha) & (alpha > 0)):
            raise ValueError(
                "probability_matching: exposure values must be finite and > 0."
            )
        group_alpha, group_of = np.unique(alpha, return_inverse=True)
        if group_alpha.size > max_exposure_groups:
            exposure_binned = True
            edges = np.quantile(alpha, np.linspace(0.0, 1.0, max_exposure_groups + 1))
            bin_of = np.clip(np.searchsorted(edges[1:-1], alpha, side='right'),
                             0, max_exposure_groups - 1)
            used, group_of = np.unique(bin_of, return_inverse=True)
            group_alpha = np.array([alpha[group_of == g].mean()
                                    for g in range(used.size)])
        _term_sorted = sorted(_terminal_set)
        _term_pos = {v: i for i, v in enumerate(_term_sorted)}
        groups = []
        for g in range(group_alpha.size):
            obs_g = observed_indices[group_of == g]
            u_g, c_g = np.unique(obs_g, return_counts=True)
            groups.append(dict(
                alpha=float(group_alpha[g]),
                n=int(obs_g.size),
                unique=u_g,
                pos=np.array([_term_pos[int(v)] for v in u_g], dtype=np.int64),
                emp=c_g.astype(np.float64) / obs_g.size,
            ))
        if verbose:
            print(f"ProbMatch: {len(groups)} exposure group(s)"
                  + (" (binned)" if exposure_binned else ""))
        _alphas = np.array([gr['alpha'] for gr in groups])
        _weights = np.sqrt(np.array([gr['n'] for gr in groups]) / n_obs)
        import jax
        _batched_sojourn = jax.vmap(
            lambda t: compute_sojourn_times_ffi(serialized, t, all_terminal_indices_arr))

        def grouped_probs(theta_full):
            th = np.tile(np.asarray(theta_full, dtype=np.float64), (len(groups), 1))
            th[:, k_exp] *= _alphas
            soj = np.asarray(_batched_sojourn(jnp.asarray(th)))
            p_all = soj / soj.sum(axis=1, keepdims=True)
            return [p_all[g, gr['pos']] for g, gr in enumerate(groups)]

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
    elif groups is None:
        # Grid search: match the most common observation's probability
        most_common_prob = float(np.max(empirical_probs))
        x0 = _initial_guess_grid_probs(
            probs_fn, most_common_prob, n_free, fixed, theta_dim
        )
    else:
        # Grid search on the largest exposure group's table.
        _gl = int(np.argmax([gr['n'] for gr in groups]))
        x0 = _initial_guess_grid_probs(
            lambda th: grouped_probs(th)[_gl],
            float(np.max(groups[_gl]['emp'])), n_free, fixed, theta_dim
        )

    if verbose:
        theta_full_init = _reconstruct_theta(x0, theta_dim, fixed)
        print(f"ProbMatch: initial guess (full theta) = {theta_full_init}")

    # ------------------------------------------------------------------
    # 5. Define residual
    # ------------------------------------------------------------------
    def residual_fn(theta_free):
        theta_full = _reconstruct_theta(theta_free, theta_dim, fixed)
        if groups is None:
            model_probs = probs_fn(theta_full)
            return model_probs.ravel() - empirical_probs.ravel()
        return np.concatenate([
            w * (p - gr['emp'])
            for w, p, gr in zip(_weights, grouped_probs(theta_full), groups)])

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
    if groups is None:
        model_probs_opt = probs_fn(theta_full_opt)
    else:
        model_probs_opt = np.concatenate(grouped_probs(theta_full_opt))
        empirical_probs = np.concatenate([gr['emp'] for gr in groups])
        unique_indices = np.concatenate([gr['unique'] for gr in groups])

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
        if groups is None:
            multinomial_cov = _estimate_multinomial_covariance(empirical_probs, n_obs)
        else:
            # Weighted residual r_g = sqrt(w_g) (p_g - e_g), w_g = n_g / n:
            # Cov(r_g) = w_g * Cov(e_g), blocks independent across groups.
            from scipy.linalg import block_diag
            multinomial_cov = block_diag(*[
                (gr['n'] / n_obs) * _estimate_multinomial_covariance(gr['emp'], gr['n'])
                for gr in groups])
        G = np.linalg.pinv(J)  # shape (n_free, n_unique)
        cov_theta = G @ multinomial_cov @ G.T
        std_free = np.sqrt(np.maximum(np.diag(cov_theta), 0.0))
    except (np.linalg.LinAlgError, ValueError) as _dm_exc:
        # No silent fallback (was: std = 0.5 * |theta|).
        raise ValueError(
            "probability_matching: the delta-method standard errors could not "
            f"be computed ({type(_dm_exc).__name__}: {_dm_exc}). Pass prior= "
            "explicitly."
        ) from _dm_exc

    # Map std back to full theta vector
    j = 0
    for i in range(theta_dim):
        if i in fixed_dict:
            std_full[i] = 0.0
        else:
            std_full[i] = std_free[j]
            j += 1

    # A zero or non-finite std means the data do not determine that
    # parameter (no silent fallback; was: std = 0.5 * max(|theta|, 1e-4)).
    _bad = [i for i in range(theta_dim)
            if i not in fixed_dict
            and (not np.isfinite(std_full[i]) or std_full[i] <= 0)]
    if _bad:
        raise ValueError(
            f"probability_matching: the standard error of parameter(s) {_bad} "
            f"is zero or not finite at the estimate {theta_full_opt[_bad]}, so "
            f"the observed outcomes do not determine them. Fix them with "
            f"fixed=, or pass prior= explicitly."
        )

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
                lower=lower,
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
        exposure_groups=1 if groups is None else len(groups),
        exposure_binned=exposure_binned,
    )
