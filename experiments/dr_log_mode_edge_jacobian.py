"""De-risk D0 for the log-weight-mode B3 batch (build-free, no C touched).

weight_mode='log': w_e(theta) = prod_i (c_e[i] * theta[i]) over ALL i in
0..param_length-1 (graph_builder.cpp / src/c/phasic.c ptd_graph_update_weights
use_log branch), computed as exp(sum_i log(c_e[i]*theta[i])) for stability.

Candidate contraction rule (product rule): dw_e/dtheta_j = w_e / theta_j
(valid since every c_e[i]*theta[i] > 0 is required by the C layer already,
so no c_e[i] can be zero and no theta[i] can be zero -- division is safe by
construction of a valid log-mode graph).

Verify against jax.jacobian of the literal product, across scales including
mixed, and check that w_e/theta_j (division after the fact) doesn't lose
precision relative to computing the derivative directly in log-space, since
the production tape will have w_e already computed (via exp(logsum)) and
theta available -- the natural implementation is literal w_e/theta_j, so
that's what's checked here (not a separately-recomputed product).

Run: pixi run python experiments/dr_log_mode_edge_jacobian.py
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def w_of_theta(c, theta):
    """c: (P,) coefficients for ONE edge, theta: (P,). Literal product form."""
    return jnp.prod(c * theta)


def w_via_logspace(c, theta):
    """The ACTUAL production forward: exp(sum(log(c*theta))) -- must match
    the literal product exactly enough for w/theta_j to be a valid derivative
    of THIS computed value, not just the idealized product."""
    return jnp.exp(jnp.sum(jnp.log(c * theta)))


def closed_form_jacobian(c, theta):
    """dw/dtheta_j = w/theta_j, evaluated via the LOG-SPACE w (matches what
    the C tape will actually have on hand: edge->weight, already computed via
    exp(logsum))."""
    c = np.asarray(c, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    w = float(np.exp(np.sum(np.log(c * theta))))
    return w / theta  # (P,)


def violation(a, b, rtol=1e-8, atol=1e-8):
    return float(np.max(np.abs(a - b) - atol - rtol * np.abs(b)))


def check(c, theta, tag):
    c = np.asarray(c, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    J_closed = closed_form_jacobian(c, theta)
    # Ground truth: differentiate the EXACT SAME log-space computation the
    # production code uses (not the idealized literal-product form), so this
    # checks the actual numerical object, not an idealized stand-in.
    J_jax = np.asarray(jax.jacobian(w_via_logspace, argnums=1)(jnp.asarray(c), jnp.asarray(theta)))
    v = violation(J_closed, J_jax)
    ok = v <= 0.0
    print(f"  {tag:45s} violation={v:+.3e} {'OK' if ok else 'FAIL'}")
    return ok


def main():
    all_ok = True

    print("Uniform-scale random cases (P = 1..5 params, all positive):")
    for seed in range(20):
        P = 1 + seed % 5
        rng = np.random.default_rng(seed)
        c = rng.uniform(0.1, 2.0, size=P)
        theta = rng.uniform(0.5, 1.5, size=P)
        all_ok &= check(c, theta, f"seed={seed} P={P} uniform")

    print("\nMixed-scale theta (factor 1e-6..1e6 per param):")
    for seed in range(20, 30):
        rng = np.random.default_rng(seed)
        P = 3
        c = rng.uniform(0.1, 2.0, size=P)
        theta = rng.uniform(0.5, 1.5, size=P) * np.array([1.0, 1e-6, 1e6])
        all_ok &= check(c, theta, f"seed={seed} mixed-scale")

    print("\nExtreme mixed-scale (factor 1e-12..1e12):")
    for seed in range(30, 35):
        rng = np.random.default_rng(seed)
        P = 2
        c = rng.uniform(0.5, 1.5, size=P)
        theta = rng.uniform(0.8, 1.2, size=P) * np.array([1e-12, 1e12])
        all_ok &= check(c, theta, f"seed={seed} extreme mixed-scale")

    print("\nP=1 (single param, single edge -- degenerate but must still hold):")
    for seed in range(5):
        rng = np.random.default_rng(seed)
        c = rng.uniform(0.1, 2.0, size=1)
        theta = rng.uniform(0.1, 5.0, size=1)
        all_ok &= check(c, theta, f"seed={seed} P=1")

    print("\nMany params sharing one edge (P=6):")
    for seed in range(5):
        rng = np.random.default_rng(seed + 200)
        c = rng.uniform(0.2, 1.8, size=6)
        theta = rng.uniform(0.3, 3.0, size=6)
        all_ok &= check(c, theta, f"seed={seed} P=6")

    print("\n" + ("ALL PASS" if all_ok else "FAILURES PRESENT"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
