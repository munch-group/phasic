"""
smc_prime.py — SMC' HMM transition matrix for two sequences.

Derivation
----------
For old TMRCA t, recombination height s ~ Uniform(0, t).
The detached lineage re-coalesces at rate 2 in [s, t) (both original lineages
available) and rate 1 in [t, ∞) (single merged ancestor), giving the density:

    f(t' | t) = (1 − e^{−2t'}) / t              for t' < t   [pre-merge]
                (1 − e^{−2t}) / (2t) e^{−(t'−t)} for t' ≥ t  [post-merge]

This is the convolution:

    T' = E₂ + U   (if E₂ + U < t),  E₂ ~ Exp(2),  U ~ Uniform(0, t)
    T' = t + E₁   (otherwise),       E₁ ~ Exp(1)

so the density is cheaply evaluated/integrated — no special functions needed.

The outer-product decomposition of Q[i,j] = ∫_{τ_j}^{τ_{j+1}} f(t'|t̃_i) dt':

    j < i :  Q[i,j] = A[j] / t̃_i                  (rank-1, below-diagonal)
    j > i :  Q[i,j] = α_i · e^{t̃_i} · B[j]         (rank-1, above-diagonal)
    j = i :  Q[i,i] = pre_i + post_i               (diagonal)

where
    A[j] = ∫_{τ_j}^{τ_{j+1}} (1 − e^{−2t'}) dt'
    B[j] = e^{−τ_j} − e^{−τ_{j+1}}
    α_i  = (1 − e^{−2t̃_i}) / (2 t̃_i)

Full transition matrix:  P = e^{−ρ} I + (1 − e^{−ρ}) Q

Two Q implementations:
  • midpoint    — closed-form, O(K); representative times via Exp(1) cond. mean.
  • marginalized — Gauss-Legendre quadrature over source intervals, O(K² n_q).
                   Eliminates midpoint approximation error.

Both functions return row-normalized Q by default (equivalent to treating the
last interval as [τ_{K-1}, ∞), absorbing the t_max truncation tail).

All times are in coalescent units (N_e generations for haploid).
Enable 64-bit precision (done automatically below) for accurate results.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

jax.config.update("jax_enable_x64", True)


# ── time grid ─────────────────────────────────────────────────────────────────

def make_log_time_grid(K: int, t_max: float = 15.0) -> jnp.ndarray:
    """
    Log-spaced time boundaries tau_0 ... tau_K, shape (K+1,).

    tau_0 = 0, tau_K = t_max. At t_max = 15 the Exp(1) tail mass is e^{-15}
    approx 3e-7, negligible for most scenarios.

    Strategy: K+1 log-spaced points on [t_min, t_max]; strip the endpoints to
    get K-1 strictly interior boundaries; bookend with 0 and t_max. This
    avoids duplicating t_max at tau[K-1] = tau[K] (zero-width last interval).

    Args:
        K:      Number of discrete TMRCA states.
        t_max:  Upper time boundary in coalescent units.

    Returns:
        tau: (K+1,) monotonically increasing boundary array.
    """
    inner = jnp.exp(jnp.linspace(jnp.log(1e-3), jnp.log(t_max), K + 1))[1:-1]
    return jnp.concatenate([jnp.array([0.0]), inner, jnp.array([t_max])])


def representative_times(tau: jnp.ndarray) -> jnp.ndarray:
    """
    Conditional mean of Exp(1) within each interval [tau_i, tau_{i+1}]:

        t_i = E[T | T in [tau_i, tau_{i+1}]]
            = ((tau_i + 1) e^{-tau_i} - (tau_{i+1} + 1) e^{-tau_{i+1}})
              / (e^{-tau_i} - e^{-tau_{i+1}})

    Args:
        tau: (K+1,) boundaries.

    Returns:
        t: (K,) representative times, one per interval.
    """
    a, b = tau[:-1], tau[1:]
    num = (a + 1.0) * jnp.exp(-a) - (b + 1.0) * jnp.exp(-b)
    den = jnp.exp(-a) - jnp.exp(-b)
    return num / den


def stationary_distribution(tau: jnp.ndarray) -> jnp.ndarray:
    """
    Prior P(T in [tau_i, tau_{i+1}]) = e^{-tau_i} - e^{-tau_{i+1}} under Exp(1).

    Args:
        tau: (K+1,) boundaries.

    Returns:
        pi: (K,) probability vector. Sums to 1 - e^{-tau_K} approx 1.
    """
    return jnp.exp(-tau[:-1]) - jnp.exp(-tau[1:])


# ── core integral ─────────────────────────────────────────────────────────────

def _q_inner(t: jax.Array, lo: jax.Array, hi: jax.Array) -> jax.Array:
    """
    Integral_{lo}^{hi} f_{SMC'}(t' | t) dt'  for scalar t, lo, hi.

    The density has a kink at t' = t. We handle this by clipping the split
    point into [lo, hi]: below = integral_{lo}^{clip(t,lo,hi)}, above = rest.

    Uses expm1 for stability near t = 0:
        1 - e^{-2x} = -expm1(-2x)
        alpha = (1 - e^{-2t}) / (2t) = -expm1(-2t) / (2t)

    Args:
        t:   Source TMRCA (scalar JAX array).
        lo:  Lower boundary of target interval.
        hi:  Upper boundary of target interval.

    Returns:
        Scalar probability mass in [lo, hi] under f(. | t).
    """
    ts = jnp.clip(t, lo, hi)

    # pre-merge: integral_{lo}^{ts} (1 - e^{-2t'}) / t dt'
    #          = [(t' + e^{-2t'}/2) / t]_{lo}^{ts}
    below = ((ts - lo) + 0.5 * (jnp.expm1(-2.0 * ts) - jnp.expm1(-2.0 * lo))) / t

    # post-merge: alpha * integral_{ts}^{hi} e^{-(t'-t)} dt'
    #           = alpha * [e^{-(ts-t)} - e^{-(hi-t)}]
    alpha = -jnp.expm1(-2.0 * t) / (2.0 * t)
    above = alpha * (jnp.exp(-(ts - t)) - jnp.exp(-(hi - t)))

    return below + above


# ── midpoint transition matrix ────────────────────────────────────────────────

def Q_midpoint(tau: jnp.ndarray, normalize: bool = True) -> jnp.ndarray:
    """
    SMC' recombination transition matrix Q using representative times t_i.

    Exploits the outer-product decomposition:

        j < i :  Q[i,j] = A[j] / t_i            (below-diagonal, rank-1)
        j > i :  Q[i,j] = alpha_i * e^{t_i} * B[j]  (above-diagonal, rank-1)
        j = i :  Q[i,i] = pre_i + post_i         (diagonal)

    Precomputing A, B, alpha is O(K); matrix assembly is O(K^2).

    Args:
        tau:       (K+1,) time boundaries.
        normalize: Row-normalize to sum to 1. Absorbs the small probability mass
                   beyond t_max (the post-merge exponential tail), equivalent to
                   treating the last interval as [tau_{K-1}, infinity).

    Returns:
        Q: (K, K) row-stochastic matrix.
    """
    t  = representative_times(tau)   # (K,)
    K  = t.shape[0]

    # A[j] = integral_{tau_j}^{tau_{j+1}} (1 - e^{-2t'}) dt'
    A = (tau[1:] - tau[:-1]) + 0.5 * (jnp.expm1(-2.0*tau[1:]) - jnp.expm1(-2.0*tau[:-1]))
    # B[j] = e^{-tau_j} - e^{-tau_{j+1}}  (Laplace weight for above-diagonal)
    B     = jnp.exp(-tau[:-1]) - jnp.exp(-tau[1:])
    alpha = -jnp.expm1(-2.0 * t) / (2.0 * t)   # (K,)

    Q_below = A[None, :] / t[:, None]                       # (K, K)
    Q_above = (alpha * jnp.exp(t))[:, None] * B[None, :]   # (K, K)

    pre      = ((t - tau[:-1]) + 0.5 * (jnp.expm1(-2.0*t) - jnp.expm1(-2.0*tau[:-1]))) / t
    post     = alpha * (1.0 - jnp.exp(-(tau[1:] - t)))
    Q_diag_v = pre + post

    I, J = jnp.meshgrid(jnp.arange(K), jnp.arange(K), indexing='ij')
    Q = jnp.where(I > J, Q_below,
        jnp.where(I < J, Q_above,
                         jnp.diag(Q_diag_v)))

    return Q / Q.sum(axis=1, keepdims=True) if normalize else Q


# ── marginalized transition matrix ────────────────────────────────────────────

def Q_marginalized(tau: jnp.ndarray, n_quad: int = 32,
                   normalize: bool = True) -> jnp.ndarray:
    """
    SMC' recombination transition matrix Q, marginalized over the Exp(1)
    coalescent density within each source interval via Gauss-Legendre quadrature:

        Q[i,j] = (1/p_i) integral_{tau_i}^{tau_{i+1}} e^{-t}
                          [integral_{tau_j}^{tau_{j+1}} f(t'|t) dt'] dt

    where p_i = e^{-tau_i} - e^{-tau_{i+1}}.

    This eliminates the midpoint approximation error. n_quad=32 gives
    double-precision accuracy for typical log-spaced grids. Fully JAX-traceable
    and jit-compatible (numpy is only used once to generate the fixed GL nodes).

    Args:
        tau:       (K+1,) time boundaries.
        n_quad:    Number of GL quadrature points per source interval.
        normalize: Row-normalize to sum to 1; see Q_midpoint docs.

    Returns:
        Q: (K, K) row-stochastic matrix.
    """
    K = tau.shape[0] - 1

    # GL nodes/weights on [-1, 1] — computed once in NumPy, then frozen as JAX arrays
    xi, wi = np.polynomial.legendre.leggauss(n_quad)
    xi = jnp.array(xi)   # (n_quad,)
    wi = jnp.array(wi)   # (n_quad,)

    p = jnp.exp(-tau[:-1]) - jnp.exp(-tau[1:])   # (K,) interval masses

    def row(a: jax.Array, b: jax.Array) -> jax.Array:
        """One row of Q by GL quadrature over source interval [a, b]."""
        nodes   = 0.5 * (b - a) * xi + 0.5 * (a + b)   # (n_quad,)
        weights = 0.5 * (b - a) * wi                    # (n_quad,)

        # inner[q, j] = integral_{tau_j}^{tau_{j+1}} f(t' | nodes[q]) dt'
        inner = vmap(           # over quadrature nodes  -> (n_quad, K)
            lambda t: vmap(    # over target intervals   -> (K,)
                lambda j: _q_inner(t, tau[j], tau[j + 1])
            )(jnp.arange(K))
        )(nodes)

        density = jnp.exp(-nodes)              # Exp(1) coalescent density
        return (weights * density) @ inner     # (K,)

    rows = vmap(row)(tau[:-1], tau[1:])        # (K, K)
    Q    = rows / p[:, None]

    return Q / Q.sum(axis=1, keepdims=True) if normalize else Q


# ── full HMM transition matrix ────────────────────────────────────────────────

def smc_prime_transition(
    tau: jnp.ndarray,
    rho: float,
    *,
    marginalized: bool = False,
    n_quad: int = 32,
    normalize: bool = True,
) -> jnp.ndarray:
    """
    Full SMC' HMM transition matrix:

        P[i,j] = e^{-rho} delta_{ij}  +  (1 - e^{-rho}) Q[i,j]

    The diagonal term is the no-recombination probability; Q[i,j] is the new
    TMRCA distribution given a recombination event occurred.

    Args:
        tau:          (K+1,) time boundaries. tau[0] = 0.
        rho:          Per-site scaled recombination rate. For a haploid model
                      with per-generation rate r: rho = 2 N_e r (per site).
        marginalized: If True, use GL quadrature over source intervals.
                      If False, use representative-time midpoint (faster).
        n_quad:       GL quadrature order (only used when marginalized=True).
        normalize:    Row-normalize Q; see Q_midpoint docs.

    Returns:
        P: (K, K) row-stochastic transition matrix.
    """
    Q     = (Q_marginalized(tau, n_quad, normalize)
             if marginalized else Q_midpoint(tau, normalize))
    K     = Q.shape[0]
    p_rec = -jnp.expm1(-rho)   # 1 - e^{-rho}, numerically stable
    return (1.0 - p_rec) * jnp.eye(K) + p_rec * Q


# ── diagnostics ───────────────────────────────────────────────────────────────

def check_row_stochastic(M: jnp.ndarray, name: str = "M", tol: float = 1e-5) -> None:
    """Assert rows sum to 1 within tol and all entries are non-negative."""
    rs      = M.sum(axis=1)
    max_err = float(jnp.abs(rs - 1.0).max())
    min_val = float(M.min())
    ok      = max_err < tol and min_val >= -tol
    print(f"{'OK' if ok else 'FAIL'} {name}: "
          f"row sums in [{float(rs.min()):.8f}, {float(rs.max()):.8f}], "
          f"min entry = {min_val:.2e}, "
          f"max |row_sum - 1| = {max_err:.2e}")


# ── example / smoke test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import time

    K   = 32
    rho = 4e-4
    tau = make_log_time_grid(K, t_max=15.0)
    t   = representative_times(tau)
    pi  = stationary_distribution(tau)

    print(f"Grid:  K = {K},  tau in [{float(tau[0]):.4f}, {float(tau[-1]):.1f}]")
    print(f"Prior mass covered: {float(pi.sum()):.8f}")
    print()

    P_mid = smc_prime_transition(tau, rho, marginalized=False)
    P_mar = smc_prime_transition(tau, rho, marginalized=True, n_quad=32)

    check_row_stochastic(P_mid, "P_midpoint    (normalized)")
    check_row_stochastic(P_mar, "P_marginalized (GL-32, normalized)")
    print()

    diff     = jnp.abs(P_mid - P_mar)
    off_diag = diff.at[jnp.diag_indices(K)].set(0.0)
    print(f"max |P_mid - P_mar| all entries:       {float(diff.max()):.2e}")
    print(f"max |P_mid - P_mar| off-diagonal only: {float(off_diag.max()):.2e}")
    print()

    # detailed balance: pi_i P_ij = pi_j P_ji
    pi_P   = pi[:, None] * P_mar
    max_db = float(jnp.abs(pi_P - pi_P.T).max())
    print(f"Detailed balance violation (marginalized): {max_db:.2e}")
    print()

    # JIT benchmark
    jitted = jax.jit(
        lambda tau, r: smc_prime_transition(tau, r, marginalized=True, n_quad=32))
    _ = jitted(tau, rho).block_until_ready()  # compile
    t0 = time.time()
    for _ in range(50):
        jitted(tau, rho).block_until_ready()
    print(f"JIT throughput: {50/(time.time()-t0):.0f} matrix/s  (K={K}, n_quad=32)")
    print()

    # figure 1: heatmap
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, P, title in zip(axes,
                             [P_mid, P_mar],
                             ["Midpoint", "Marginalized (GL-32)"]):
        im = ax.imshow(P, origin="upper", norm=LogNorm(vmin=1e-6, vmax=1.0),
                       cmap="viridis", aspect="auto")
        plt.colorbar(im, ax=ax, label="P[i -> j]")
        ax.set_title(f"SMC' transition P  ({title})", fontsize=11)
        ax.set_xlabel("Target state j")
        ax.set_ylabel("Source state i")
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/smc_prime_heatmap.png", dpi=150)

    # figure 2: row slices
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    srcs = [4, 15, 28]
    for ax, P, title in zip(axes,
                             [P_mid, P_mar],
                             ["Midpoint", "Marginalized (GL-32)"]):
        for src in srcs:
            ax.plot(t, P[src], marker="o", ms=3,
                    label=f"src {src}  (t = {float(t[src]):.2f})")
        ax.set_yscale("log"); ax.set_ylim(1e-7, 1.0)
        ax.set_xlabel("Representative TMRCA  t_j")
        ax.set_ylabel("P[i -> j]")
        ax.set_title(f"Row slices  ({title})", fontsize=11)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/smc_prime_rows.png", dpi=150)
    print("Plots saved to outputs/.")
