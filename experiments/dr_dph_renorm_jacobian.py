"""B3 discrete/was_dph de-risk D0.1 (build-free, no phasic C touched).

Verifies the closed-form renorm edge->theta Jacobian used by the planned
``ptd_moments_grad_theta_dph`` contraction step against ``jax.jacobian`` of
the literal renormalization p_e(theta) = w_e / sum(w), w_e = c_e . theta, on
synthetic per-vertex scenarios (random coefficients, random theta, several
edge counts / param counts / scales, including a mixed-scale regime where FD
would be untrustworthy but exact autodiff is not).

Formula under test (b3-batch3-mpfr-and-discrete-derisk.md, Part 1):
    dp_e/dtheta_j = (c_e^j - p_e * sum_{e'} c_{e'}^j) / S_v
    S_v = sum_{e'} (c_{e'} . theta)

Run: pixi run python experiments/dr_dph_renorm_jacobian.py
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def p_of_theta(C, theta):
    """C: (m, P) coefficients, theta: (P,). Returns p: (m,)."""
    w = C @ theta
    S = jnp.sum(w)
    return w / S


def closed_form_jacobian(C, theta):
    """Analytic dp_e/dtheta_j via the quotient rule (Part 1 formula)."""
    C = np.asarray(C, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    w = C @ theta                      # (m,)
    S = float(np.sum(w))
    p = w / S                          # (m,)
    sigma_c = np.sum(C, axis=0)        # (P,) = sum_e' c_e'^j
    # dp_e/dtheta_j = (C[e,j] - p[e]*sigma_c[j]) / S
    J = (C - np.outer(p, sigma_c)) / S
    return J


def violation(a, b, rtol=1e-8, atol=1e-8):
    """numpy.allclose-style combined abs+rel check, returned as a signed
    "how far past the tolerance" scalar (<=0 means it passes). A pure
    relative metric explodes when the true value is exactly/near zero
    (e.g. P=1: a single scalar theta cancels top and bottom of p_e, so
    dp_e/dtheta is identically 0 -- both sides then carry only ~1e-16
    float noise, but dividing noise by noise inflates the ratio)."""
    return float(np.max(np.abs(a - b) - atol - rtol * np.abs(b)))


def check(C, theta, tag):
    C = np.asarray(C, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    J_closed = closed_form_jacobian(C, theta)
    J_jax = np.asarray(jax.jacobian(p_of_theta, argnums=1)(jnp.asarray(C), jnp.asarray(theta)))
    v = violation(J_closed, J_jax)
    ok = v <= 0.0
    print(f"  {tag:55s} violation={v:+.3e} {'OK' if ok else 'FAIL'}")
    return ok


def main():
    all_ok = True

    print("Uniform-scale random cases:")
    for seed in range(20):
        m = 2 + seed % 4          # 2..5 edges
        P = 1 + seed % 4          # 1..4 params
        rng = np.random.default_rng(seed)
        C = rng.uniform(0.1, 2.0, size=(m, P))
        theta = rng.uniform(0.5, 1.5, size=P)
        all_ok &= check(C, theta, f"seed={seed} m={m} P={P} uniform")

    print("\nMixed-scale theta (per-param, factor 1e-6..1e6):")
    for seed in range(20, 30):
        rng = np.random.default_rng(seed)
        C = rng.uniform(0.1, 2.0, size=(3, 3))
        theta = rng.uniform(0.5, 1.5, size=3) * np.array([1.0, 1e-6, 1e6])
        all_ok &= check(C, theta, f"seed={seed} mixed-scale theta")

    print("\nSingle out-edge (p_e == 1 identically -> Jacobian == 0 exactly):")
    for seed in range(5):
        P = 1 + seed % 3
        rng = np.random.default_rng(seed)
        C = rng.uniform(0.1, 2.0, size=(1, P))
        theta = rng.uniform(0.5, 1.5, size=P)
        all_ok &= check(C, theta, f"seed={seed} m=1 P={P} single-edge")

    print("\nSingle theta param (P=1: scalar theta cancels -> Jacobian == 0 exactly):")
    for seed in range(5):
        m = 2 + seed % 4
        rng = np.random.default_rng(seed)
        C = rng.uniform(0.1, 2.0, size=(m, 1))
        theta = rng.uniform(0.5, 1.5, size=1)
        all_ok &= check(C, theta, f"seed={seed} m={m} P=1 single-param")

    print("\nDegenerate: one edge dominates (p_e -> 1) [near-deterministic renorm]:")
    for seed in range(5):
        rng = np.random.default_rng(seed + 100)
        C = rng.uniform(0.1, 2.0, size=(3, 2))
        C[0] *= 1e4  # first edge dominates S_v
        theta = rng.uniform(0.5, 1.5, size=2)
        all_ok &= check(C, theta, f"seed={seed} dominant-edge")

    print("\n" + ("ALL PASS" if all_ok else "FAILURES PRESENT"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
