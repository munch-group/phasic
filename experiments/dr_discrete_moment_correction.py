"""B3 discrete/was_dph de-risk D0.2 (build-free, no phasic C touched).

Python/JAX port of GraphBuilder::continuous_to_discrete_moments
(src/cpp/parameterized/graph_builder.cpp:694), which maps a CONTINUOUS raw
power-moment vector m (m[j-1] = E[T^j]) to the DISCRETE raw moment vector
(E[N], E[N^2], ..., E[N^K]) via a fixed (theta-independent) linear
combination of factorial/binomial/Stirling-2 coefficients.

Verifies:
  1. The map is exactly LINEAR (no constant term) -- required for
     "d(discrete m)/dtheta = C . d(continuous m)/dtheta" to hold.
  2. The known K=2 closed identity out = [m0, m1 - m0] (graph_builder.cpp's
     own comment: "order 2 reduces to E[N^2] = m[1]-m[0]").
  3. The planned contraction recipe -- apply the SAME transform to each
     COLUMN (theta-index) of a continuous-moments Jacobian independently --
     reproduces the true chain-rule Jacobian of the composed map, verified
     against jax.jacobian on a random nonlinear m(theta), for K = 1..6.

Run: pixi run python experiments/dr_discrete_moment_correction.py
"""
import math

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _factorial(n):
    return float(math.factorial(n))


def _binom(n, k):
    if k < 0 or k > n:
        return 0.0
    return float(math.comb(n, k))


_stirling2_cache = {}


def _stirling2(n, k):
    if k == 0:
        return 1.0 if n == 0 else 0.0
    if k > n:
        return 0.0
    if k == n or k == 1:
        return 1.0
    key = (n, k)
    if key in _stirling2_cache:
        return _stirling2_cache[key]
    v = k * _stirling2(n - 1, k) + _stirling2(n - 1, k - 1)
    _stirling2_cache[key] = v
    return v


def continuous_to_discrete_moments(m):
    """m: array-like (jnp or np) length K. Returns array length K (same
    backend as m, via jnp so this stays differentiable under jax.jacobian)."""
    K = len(m)
    u = [0.0] * (K + 1)
    for j in range(1, K + 1):
        u[j] = m[j - 1] / _factorial(j)
    F = [0.0] * (K + 1)
    for r in range(1, K + 1):
        s = 0.0
        for i in range(0, r):
            sign = 1.0 if i % 2 == 0 else -1.0
            s = s + _binom(r - 1, i) * sign * u[r - i]
        F[r] = _factorial(r) * s
    out = []
    for kk in range(1, K + 1):
        s = 0.0
        for r in range(1, kk + 1):
            s = s + _stirling2(kk, r) * F[r]
        out.append(s)
    return jnp.stack(out) if isinstance(m, jnp.ndarray) else np.array(out)


def check_linearity(seed, K):
    rng = np.random.default_rng(seed)
    m1 = rng.uniform(0.5, 5.0, size=K)
    m2 = rng.uniform(0.5, 5.0, size=K)
    a, b = rng.uniform(-2, 2, size=2)
    lhs = continuous_to_discrete_moments(a * m1 + b * m2)
    rhs = a * continuous_to_discrete_moments(m1) + b * continuous_to_discrete_moments(m2)
    err = float(np.max(np.abs(lhs - rhs)))
    ok = err < 1e-9
    print(f"  linearity seed={seed} K={K}: err={err:.3e} {'OK' if ok else 'FAIL'}")
    return ok


def check_k2_identity(seed):
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.5, 5.0, size=2)
    out = continuous_to_discrete_moments(m)
    expected = np.array([m[0], m[1] - m[0]])
    err = float(np.max(np.abs(out - expected)))
    ok = err < 1e-9
    print(f"  K=2 identity seed={seed}: out={out}, expected={expected}, err={err:.3e} {'OK' if ok else 'FAIL'}")
    return ok


def check_column_jacobian(seed, K, P):
    rng = np.random.default_rng(seed)
    A = jnp.asarray(rng.uniform(-1, 1, size=(K, P)))
    B = jnp.asarray(rng.uniform(-1, 1, size=(K, P)))
    b0 = jnp.asarray(rng.uniform(0.5, 5.0, size=K))  # keep m away from 0/negative-ish scale
    theta = jnp.asarray(rng.uniform(0.5, 1.5, size=P))

    def m_of_theta(th):
        return b0 + A @ th + 0.3 * jnp.sin(B @ th)

    def discrete_of_theta(th):
        return continuous_to_discrete_moments(m_of_theta(th))

    # True chain-rule Jacobian of the composed (nonlinear m, then linear c2d) map.
    J_true = np.asarray(jax.jacobian(discrete_of_theta)(theta))

    # Planned recipe: continuous Jacobian, then apply c2d to each COLUMN.
    J_cont = np.asarray(jax.jacobian(m_of_theta)(theta))  # (K, P)
    J_via_columns = np.zeros((K, P))
    for j in range(P):
        J_via_columns[:, j] = np.asarray(continuous_to_discrete_moments(jnp.asarray(J_cont[:, j])))

    err = float(np.max(np.abs(J_true - J_via_columns)))
    ok = err < 1e-8
    print(f"  column-jacobian seed={seed} K={K} P={P}: err={err:.3e} {'OK' if ok else 'FAIL'}")
    return ok


def main():
    all_ok = True

    print("Linearity (no constant term):")
    for seed in range(10):
        K = 1 + seed % 6
        all_ok &= check_linearity(seed, K)

    print("\nKnown K=2 closed identity (E[N^2] = m1 - m0):")
    for seed in range(5):
        all_ok &= check_k2_identity(seed)

    print("\nPer-column Jacobian application == true chain rule (jax.jacobian ground truth):")
    for seed in range(15):
        K = 1 + seed % 6
        P = 1 + (seed * 3) % 5
        all_ok &= check_column_jacobian(seed, K, P)

    print("\n" + ("ALL PASS" if all_ok else "FAILURES PRESENT"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
