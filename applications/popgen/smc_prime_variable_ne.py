"""
smc_prime_variable_ne.py  —  SMC' transition matrix with piecewise-constant N_e.

Extends smc_prime.py to support K epoch-specific effective population sizes
eta[k] = N_e(t) / N_ref for t in [tau_k, tau_{k+1}).

Key mathematical objects
-------------------------
Lambda(t) = integral_0^t 1/eta(s) ds      cumulative coalescent hazard
A(t)      = integral_0^t exp(2*Lambda(s)) ds
beta(t)   = exp(-2*Lambda(t)) * A(t)      telescoping antiderivative

The generalised SMC' density f(t'|t) is:

  t' < t :  (2/eta(t')*t) * exp(-2*Lambda(t')) * A(t')
           = (1/t) * [1 - d/dt'(beta(t'))]    <- integrates in closed form
  t' >= t:  (1/eta(t')*t) * exp(-Lambda(t')-Lambda(t)) * A(t)
           = (exp(-Lambda(t))*A(t)/t) * (-d/dt' exp(-Lambda(t')))

Both branches integrate to closed-form expressions in terms of precomputed
grid quantities, preserving the O(K) construction of the constant-N case.

Recovering constant N_e
-----------------------
Setting eta = jnp.ones(K) reproduces the constant-N results exactly.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

jax.config.update("jax_enable_x64", True)


# ── epoch grid utilities ───────────────────────────────────────────────────────

def make_epoch_quantities(
    tau: jnp.ndarray,
    eta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Precompute the four grid arrays needed for all subsequent calculations.

    For piecewise-constant eta[k] on [tau_k, tau_{k+1}):

        Lambda_k  = sum_{j<k} (tau_{j+1}-tau_j)/eta_j        cumulative hazard
        calA_k    = sum_{j<k} eta_j/2*(exp(2*Lambda_{j+1})-exp(2*Lambda_j))
        beta_k    = exp(-2*Lambda_k) * calA_k
        expnL_k   = exp(-Lambda_k)

    Uses expm1 throughout for stability when increments are small.

    Args:
        tau: (K+1,) time boundaries.
        eta: (K,)   epoch sizes (N_e / N_ref) for each interval.

    Returns:
        Lambda_grid, calA_grid, beta_grid, expnL_grid  -- each shape (K+1,).
    """
    delta_Lambda = (tau[1:] - tau[:-1]) / eta                              # (K,)
    Lambda_grid  = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(delta_Lambda)])

    # delta_calA[k] = eta[k]/2 * exp(2*Lambda[k]) * expm1(2*delta_Lambda[k])
    # Uses expm1 for small increments; stays exact for large ones.
    delta_calA = (eta / 2.0) * jnp.exp(2.0 * Lambda_grid[:-1]) * jnp.expm1(2.0 * delta_Lambda)
    calA_grid  = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(delta_calA)])

    beta_grid  = jnp.exp(-2.0 * Lambda_grid) * calA_grid
    expnL_grid = jnp.exp(-Lambda_grid)

    return Lambda_grid, calA_grid, beta_grid, expnL_grid


def representative_times_var(tau: jnp.ndarray, eta: jnp.ndarray) -> jnp.ndarray:
    """
    Conditional mean of f_T(t) = exp(-Lambda(t))/eta(t) within [tau_i, tau_{i+1}]:

        t_i = tau_i + eta_i - Delta_i / expm1(Delta_i / eta_i)

    where Delta_i = tau_{i+1} - tau_i.

    Derivation: E[T | T in [a,b]] = integral_a^b t*f_T(t)dt / pi_i.
    With piecewise-constant eta_i, the integral evaluates analytically to the
    above formula.

    Recovers the constant-N formula when eta = ones(K):
        t_i = tau_i + 1 - Delta_i / expm1(Delta_i)

    Args:
        tau: (K+1,) boundaries.
        eta: (K,)   epoch sizes.

    Returns:
        t_rep: (K,) representative times.
    """
    Delta = tau[1:] - tau[:-1]
    return tau[:-1] + eta - Delta / jnp.expm1(Delta / eta)


def stationary_distribution_var(expnL_grid: jnp.ndarray) -> jnp.ndarray:
    """
    Prior mass pi_i = exp(-Lambda_i) - exp(-Lambda_{i+1}).

    Args:
        expnL_grid: (K+1,) array of exp(-Lambda) values at grid points.

    Returns:
        pi: (K,) probability vector.
    """
    return expnL_grid[:-1] - expnL_grid[1:]


# ── midpoint transition matrix ────────────────────────────────────────────────

def Q_midpoint_var(
    tau: jnp.ndarray,
    eta: jnp.ndarray,
    normalize: bool = True,
) -> jnp.ndarray:
    """
    Variable-N_e SMC' recombination matrix Q using representative times.

    Outer-product decomposition (generalises the constant-N case):

        j < i :  Q[i,j] = A_var[j] / t_i
        j > i :  Q[i,j] = (eA_i / t_i) * B[j]
        j = i :  Q[i,i] = pre_i + post_i

    where:
        A_var[j] = (tau_{j+1}-tau_j) - beta_{j+1} + beta_j   <- integral of pre-merge density
        B[j]     = expnL_j - expnL_{j+1}                      <- integral of post-merge density
        eA_i     = exp(-Lambda_i) * calA_i                     <- state-specific prefactor

    At constant N_e=1: A_var -> A, eA_i/t_i -> alpha_i*e^{t_i}, B -> B.  All
    constant-N formulas are recovered exactly.

    Args:
        tau:       (K+1,) time boundaries.
        eta:       (K,)   epoch sizes N_e/N_ref.
        normalize: Row-normalise to absorb tail beyond t_max.

    Returns:
        Q: (K, K) row-stochastic matrix.
    """
    K = tau.shape[0] - 1
    Lambda_grid, calA_grid, beta_grid, expnL_grid = make_epoch_quantities(tau, eta)

    # ── precomputed interval scalars O(K) ──
    A_var = (tau[1:] - tau[:-1]) - beta_grid[1:] + beta_grid[:-1]   # (K,)
    B     = expnL_grid[:-1] - expnL_grid[1:]                         # (K,)

    # ── epoch quantities at representative times ──
    t_rep    = representative_times_var(tau, eta)            # (K,)
    dL       = (t_rep - tau[:-1]) / eta                      # (t_rep-tau_i)/eta_i
    Lambda_t = Lambda_grid[:-1] + dL
    # calA at t_rep using expm1: calA_i + eta_i/2 * exp(2*Lambda_i) * expm1(2*dL)
    calA_t   = calA_grid[:-1] + (eta / 2.0) * jnp.exp(2.0 * Lambda_grid[:-1]) * jnp.expm1(2.0 * dL)
    expnL_t  = jnp.exp(-Lambda_t)                           # exp(-Lambda(t_rep))
    beta_t   = jnp.exp(-2.0 * Lambda_t) * calA_t            # beta at t_rep
    eA_t     = expnL_t * calA_t                             # exp(-Lambda)*calA at t_rep

    # ── Q components ──
    Q_below = A_var[None, :] / t_rep[:, None]                         # (K, K)
    Q_above = (eA_t / t_rep)[:, None] * B[None, :]                   # (K, K)

    # Diagonal: split at t_rep
    #   pre  = [(t_rep - tau_i) - (beta(t_rep) - beta(tau_i))] / t_rep
    #   post = (eA(t_rep) / t_rep) * [exp(-Lambda(t_rep)) - exp(-Lambda(tau_{i+1}))]
    pre      = ((t_rep - tau[:-1]) - (beta_t - beta_grid[:-1])) / t_rep
    post     = (eA_t / t_rep) * (expnL_t - expnL_grid[1:])
    Q_diag_v = pre + post

    I, J = jnp.meshgrid(jnp.arange(K), jnp.arange(K), indexing='ij')
    Q = jnp.where(I > J, Q_below,
        jnp.where(I < J, Q_above,
                         jnp.diag(Q_diag_v)))

    return Q / Q.sum(axis=1, keepdims=True) if normalize else Q


# ── marginalised transition matrix ────────────────────────────────────────────

def Q_marginalized_var(
    tau: jnp.ndarray,
    eta: jnp.ndarray,
    n_quad: int = 32,
    normalize: bool = True,
) -> jnp.ndarray:
    """
    Variable-N_e SMC' Q, marginalised over f_T within each source interval via
    Gauss-Legendre quadrature:

        Q[i,j] = (1/pi_i) * integral_{tau_i}^{tau_{i+1}}
                             f_T(t) * I(t; j)  dt

    where f_T(t) = exp(-Lambda(t)) / eta_i for t in [tau_i, tau_{i+1}) and
    I(t; j) = integral_{tau_j}^{tau_{j+1}} f(t'|t) dt' (closed-form inner integral).

    The inner integral uses the same split-at-t trick as the constant-N case,
    but now involves the precomputed beta and expnL arrays:

        below = [(ts - lo) - (beta(ts) - beta(lo))] / t
        above = eA(t) * [expnL(ts) - expnL(hi)] / t

    where ts = clip(t, lo, hi), eA(t) = exp(-Lambda(t)) * calA(t).

    For each GL node t within source interval i:
        - Lambda(t) and calA(t) are computed analytically from the epoch structure
        - beta(t) and expnL(t) follow directly
        - beta and expnL at target boundaries are read from precomputed arrays

    Fully JAX-traceable and jit-compatible.

    Args:
        tau:       (K+1,) time boundaries.
        eta:       (K,)   epoch sizes N_e/N_ref.
        n_quad:    Gauss-Legendre quadrature order (32 sufficient for log grids).
        normalize: Row-normalise to absorb tail beyond t_max.

    Returns:
        Q: (K, K) row-stochastic matrix.
    """
    K = tau.shape[0] - 1
    Lambda_grid, calA_grid, beta_grid, expnL_grid = make_epoch_quantities(tau, eta)
    pi = expnL_grid[:-1] - expnL_grid[1:]

    xi, wi = np.polynomial.legendre.leggauss(n_quad)
    xi = jnp.array(xi)
    wi = jnp.array(wi)

    def row(i):
        a, b   = tau[i], tau[i + 1]
        eta_i  = eta[i]
        Lam_lo = Lambda_grid[i]
        cA_lo  = calA_grid[i]
        e2L_lo = jnp.exp(2.0 * Lam_lo)            # exp(2*Lambda_i) — used in calA formula

        nodes   = 0.5 * (b - a) * xi + 0.5 * (a + b)
        weights = 0.5 * (b - a) * wi

        def at_node(t):
            # ── epoch quantities at GL node t (inside interval i) ──
            dL_t     = (t - a) / eta_i
            Lambda_t = Lam_lo + dL_t
            calA_t   = cA_lo + (eta_i / 2.0) * e2L_lo * jnp.expm1(2.0 * dL_t)
            expnL_t  = jnp.exp(-Lambda_t)
            beta_t   = jnp.exp(-2.0 * Lambda_t) * calA_t
            eA_t     = expnL_t * calA_t

            def inner_j(j):
                lo_j, hi_j = tau[j], tau[j + 1]
                ts = jnp.clip(t, lo_j, hi_j)

                b_lo = beta_grid[j];      e_lo = expnL_grid[j]
                b_hi = beta_grid[j + 1];  e_hi = expnL_grid[j + 1]

                # beta and expnL at the split point ts
                # ts == hi_j  when t >= hi_j  (below-diagonal)
                # ts == lo_j  when t <= lo_j  (above-diagonal)
                # ts == t     otherwise        (diagonal)
                beta_ts = jnp.where(t >= hi_j, b_hi,
                          jnp.where(t <= lo_j, b_lo, beta_t))
                expnL_ts = jnp.where(t >= hi_j, e_hi,
                           jnp.where(t <= lo_j, e_lo, expnL_t))

                below = ((ts - lo_j) - (beta_ts - b_lo)) / t
                above = eA_t * (expnL_ts - e_hi) / t
                return below + above

            target_vals = vmap(inner_j)(jnp.arange(K))     # (K,)
            density = expnL_t / eta_i                       # f_T(t) = exp(-Lambda)/eta
            return target_vals * density

        contributions = vmap(at_node)(nodes)                # (n_quad, K)
        return weights @ contributions                      # (K,)

    rows = vmap(row)(jnp.arange(K))                        # (K, K)
    Q    = rows / pi[:, None]
    return Q / Q.sum(axis=1, keepdims=True) if normalize else Q


# ── full transition matrix ────────────────────────────────────────────────────

def smc_prime_transition_var(
    tau: jnp.ndarray,
    eta: jnp.ndarray,
    rho: float,
    *,
    marginalized: bool = False,
    n_quad: int = 32,
    normalize: bool = True,
) -> jnp.ndarray:
    """
    Full SMC' HMM transition matrix for piecewise-constant N_e:

        P[i,j] = exp(-rho) * delta_{ij}  +  (1 - exp(-rho)) * Q[i,j]

    Args:
        tau:          (K+1,) time boundaries. tau[0] = 0.
        eta:          (K,)   epoch sizes N_e/N_ref. eta = ones(K) recovers constant N.
        rho:          Per-site scaled recombination rate (2 * N_ref * r).
        marginalized: True -> GL quadrature; False -> midpoint.
        n_quad:       GL quadrature order (marginalized only).
        normalize:    Row-normalise Q to absorb the tail beyond t_max.

    Returns:
        P: (K, K) row-stochastic transition matrix.
    """
    Q     = (Q_marginalized_var(tau, eta, n_quad, normalize)
             if marginalized else Q_midpoint_var(tau, eta, normalize))
    K     = Q.shape[0]
    p_rec = -jnp.expm1(-rho)
    return (1.0 - p_rec) * jnp.eye(K) + p_rec * Q
