"""Deferred-4 Phase 0: oracle CALIBRATION (plan §2.1's trust bar).

The exact-rational oracle is trusted only after agreeing on
well-conditioned fixtures with (1) shipped primal values, (2) shipped
exact Jacobians, (3) jax.jacobian of an independent float64 dense
reference -- each to ~1e-13. Run BEFORE any disputed sweep point.
"""
import sys

import numpy as np

sys.path.insert(0, "experiments")
from dr_d4_exact_oracle import continuous_moments_and_jac, to_float  # noqa: E402

import phasic  # noqa: E402
from phasic import Graph, set_log_level  # noqa: E402

set_log_level("WARNING")
FAILS = []


def check(label, got, want, tol=1e-13):
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    denom = max(float(np.max(np.abs(want))), 1e-300)
    rel = float(np.max(np.abs(got - want))) / denom
    ok = rel < tol
    print(f"  {label}: rel={rel:.2e} {'PASS' if ok else 'FAIL'}")
    if not ok:
        FAILS.append(label)
    return rel


# --------------------------------------------------------------------------- fixtures
def chain2():
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [1.0, 0.5])
    return g


def branchy():
    g = Graph(1)
    s = g.starting_vertex()
    v = [g.find_or_create_vertex([i + 1]) for i in range(4)]
    va = g.find_or_create_vertex([9])
    s.add_edge(v[0], 1.0)
    v[0].add_edge(v[1], [1.0, 0.0])
    v[0].add_edge(v[2], [2.0, 0.5])
    v[1].add_edge(v[3], [1.5, 0.25])
    v[2].add_edge(v[3], [0.5, 1.0])
    v[3].add_edge(va, [1.0, 1.0])
    return g


def cyclic():
    # the DR-A class: a 2-cycle (self-referencing pair) + exit
    g = Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([3])
    b = g.find_or_create_vertex([2])
    absb = g.find_or_create_vertex([1])
    s.add_edge(a, 1.0)
    a.add_edge(b, [1.0, 0.0])
    b.add_edge(a, [0.0, 0.5])   # the cycle back-edge
    b.add_edge(absb, [1.0, 0.25])
    return g


def log2():
    g = Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge(v2, [1.0, 0.5])
    v2.add_edge(v1, [2.0, 1.0])
    g.weight_mode = 'log'
    return g


K = 3

print("== (1)+(2) linear fixtures: oracle vs shipped primal + shipped exact J ==")
for name, mk, theta in (("chain2", chain2, [2.0, 1.0]),
                        ("branchy", branchy, [1.3, 0.7]),
                        ("cyclic", cyclic, [1.1, 0.9])):
    g = mk()
    g.update_weights(list(theta))
    m_ship = np.asarray(g.moments(K))
    J_ship = np.asarray(g._moments_grad_theta(K)).reshape(K, len(theta))
    m_or, J_or = continuous_moments_and_jac(g, theta, K, mode='linear')
    check(f"{name} primal", to_float(m_or), m_ship)
    check(f"{name} Jacobian", to_float(J_or), J_ship)

print("== (1)+(2) log fixture ==")
g = log2()
theta = [1.0, 2.0]
g.update_weights(theta, log=True)
m_ship = np.asarray(g.moments(K))
J_ship = np.asarray(g._moments_grad_theta_log(K, theta)).reshape(K, 2)
m_or, J_or = continuous_moments_and_jac(g, theta, K, mode='log')
check("log primal", to_float(m_or), m_ship)
check("log Jacobian", to_float(J_or), J_ship)

print("== rewards slice (Batch-A follow-on): oracle vs shipped ==")
g = branchy()
theta = [1.3, 0.7]
g.update_weights(theta)
n = g.vertices_length()
rw_full = [1.0, 0.5, 2.0, 1.0, 0.25, 1.0][:n]
while len(rw_full) < n:
    rw_full.append(1.0)
m_ship = np.asarray(g.moments(K, list(rw_full)))
g.update_weights(theta)
J_ship = np.asarray(g._moments_grad_theta(K, rewards=list(rw_full))
                    ).reshape(K, 2)
# map per-vertex rewards -> per-transient (serialize order: transient =
# vertices with out-edges among 1..n-1; rewards indexed by vertex id)
from dr_d4_exact_oracle import build_structure  # noqa: E402
_, transient, _ = build_structure(g, 2)
rw_transient = [rw_full[v] for v in transient]
m_or, J_or = continuous_moments_and_jac(g, theta, K, mode='linear',
                                        rewards=rw_transient)
check("rewards primal", to_float(m_or), m_ship)
check("rewards Jacobian", to_float(J_or), J_ship)

print("== (3) independent float64 dense reference via jax.jacobian ==")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def jax_dense_moments(theta_arr, g, K):
    """Independent float64 dense reference (jnp linear algebra)."""
    ser = g.serialize(theta_dim=len(theta_arr))
    npar = np.asarray(ser['param_edges'], dtype=float)
    starts = np.asarray(ser['start_edges'], dtype=float)
    outv = sorted(set(int(r[0]) for r in npar))
    tix = {v: i for i, v in enumerate(outv)}
    nt = len(outv)
    T = jnp.zeros((nt, nt))
    for r in npar:
        frm, to = int(r[0]), int(r[1])
        w = jnp.dot(jnp.asarray(r[2:2 + len(theta_arr)]), theta_arr)
        if to in tix:
            T = T.at[tix[frm], tix[to]].add(w)
        T = T.at[tix[frm], tix[frm]].add(-w)
    alpha = np.zeros(nt)
    tot = 0.0
    for r in starts:
        tot += r[1]
        if int(r[0]) in tix:
            alpha[tix[int(r[0])]] += r[1]
    alpha = jnp.asarray(alpha / tot)
    N = jnp.linalg.inv(-T)
    ones = jnp.ones(nt)
    out = []
    Mk = ones
    fact = 1.0
    for k in range(1, K + 1):
        fact *= k
        Mk = N @ Mk
        out.append(fact * (alpha @ Mk))
    return jnp.stack(out)


g = branchy()
theta = jnp.asarray([1.3, 0.7])
J_jax = np.asarray(jax.jacobian(lambda th: jax_dense_moments(th, g, K))(theta))
m_or, J_or = continuous_moments_and_jac(g, [1.3, 0.7], K, mode='linear')
check("jax-dense Jacobian vs oracle", to_float(J_or), J_jax, tol=1e-10)

print()
print("ORACLE CALIBRATED" if not FAILS else f"CALIBRATION FAILURES: {FAILS}")
sys.exit(0 if not FAILS else 1)
