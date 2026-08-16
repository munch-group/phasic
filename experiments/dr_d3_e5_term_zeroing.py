"""Deferred-3 E5 -- term-zeroing verification of the route-(ii) dossier.

Dossier: b3-d3-e5-derivation-dossier.md (predictions written BEFORE
this run, section 5). An EXPLICIT forward-mode tangent implementation
of the dossier's section-3 recursion (independent of jax.grad -- this
is the C-implementation prototype), gated by:
  Z-parity : explicit tangent == jax.grad(intact mixture), all fixtures
  Z1       : drop the p*(dP~) propagation term -> must break parity
  Z2       : drop the p*(da) harvest term      -> must break parity
(Z3, the pi-index mis-alignment, is already measured in
dr_d3_e123_pdf_routes.py: 2.07e-2.)

Fixtures + conventions copied from dr_d3_e123_pdf_routes.py (that
module executes at import, so no import -- self-contained copy).
"""
import sys

import numpy as np

import phasic
from phasic import Graph, set_log_level

set_log_level("WARNING")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

FAILS = []


def check(label, ok, detail=""):
    print(f"  {label}: {'PASS' if ok else 'FAIL'} {detail}")
    if not ok:
        FAILS.append(label)


def expo():
    g = Graph(1)
    s = g.starting_vertex()
    v = g.find_or_create_vertex([2])
    a = g.find_or_create_vertex([1])
    s.add_edge(v, 1.0)
    v.add_edge(a, [1.0])
    return g, 1


def erlang3():
    g = Graph(1)
    s = g.starting_vertex()
    v1 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v3 = g.find_or_create_vertex([4])
    a = g.find_or_create_vertex([1])
    s.add_edge(v1, 1.0)
    v1.add_edge(v2, [1.0])
    v2.add_edge(v3, [1.0])
    v3.add_edge(a, [1.0])
    return g, 1


def cyclic4():
    g = Graph(1)
    s = g.starting_vertex()
    b = g.find_or_create_vertex([3])
    c = g.find_or_create_vertex([2])
    a = g.find_or_create_vertex([1])
    s.add_edge(b, 1.0)
    b.add_edge(c, [1.0, 0.0])
    c.add_edge(b, [0.0, 0.4])
    c.add_edge(a, [1.0, 0.3])
    b.add_aux_vertex_constant(0.15)
    return g, 2


FIXTURES = {'expo': expo, 'erlang3': erlang3, 'cyclic4': cyclic4}


def structure(g, P):
    ser = g.serialize(theta_dim=P)
    pe = np.asarray(ser['param_edges'], dtype=float)
    ce = np.asarray(ser.get('constant_edges', []), dtype=float)
    se = np.asarray(ser['start_edges'], dtype=float)
    n = int(ser['n_vertices'])
    edges = []
    if pe.size:
        for r in pe:
            edges.append((int(r[0]), int(r[1]), r[2:2 + P], None))
    if ce.size:
        for r in ce:
            edges.append((int(r[0]), int(r[1]), None, float(r[2])))
    out_v = set(e[0] for e in edges)
    alpha = np.zeros(n)
    tot = 0.0
    for r in se:
        tot += r[1]
        alpha[int(r[0])] += r[1]
    alpha /= tot
    return n, edges, alpha, out_v


def K_of(lam, t, K_sigma=6.0):
    mean_k = lam * t
    return int(mean_k + K_sigma * np.sqrt(max(mean_k, 1.0)) + 10)


def mixture_pdf_jax(theta, t, n, edges, alpha, out_v, lam):
    """The intact route-(ii) mixture (correct alignment), for the
    jax.grad oracle -- same as dr_d3_e123_pdf_routes.poisson_mixture_pdf."""
    R = jnp.zeros((n, n))
    for (frm, to, coeffs, cw) in edges:
        w = jnp.dot(jnp.asarray(coeffs), theta) if coeffs is not None else cw
        R = R.at[frm, to].add(w)
    absorbing = jnp.asarray([1.0 if i not in out_v else 0.0
                             for i in range(n)])
    K = K_of(lam, t)
    p = jnp.asarray(alpha)

    def step(p, _):
        moved = (p[:, None] * R / lam).sum(axis=0)
        outflow = (R / lam).sum(axis=1) * p
        p_new = p + moved - outflow
        absorbed = (p_new * absorbing).sum()
        p_new = p_new * (1.0 - absorbing)
        return p_new, absorbed

    _, pi = jax.lax.scan(step, p, None, length=K + 2)
    ks = jnp.arange(K + 1)
    log_pois = (ks * jnp.log(lam * t) - lam * t
                - jax.scipy.special.gammaln(ks + 1.0))
    return lam * jnp.sum(jnp.exp(log_pois) * pi[:K + 1])


def explicit_pdf_and_grad(theta, t, n, edges, alpha, out_v, lam,
                          drop=None):
    """Dossier section-3 recursion, explicit numpy forward-mode.

    drop='Z1' zeroes the p*(dP~) contribution to the PROPAGATED
    (transient) tangent; drop='Z2' zeroes the p*(da) contribution to
    the HARVEST tangent. T0 (d alpha) is structurally zero on these
    fixtures (constant start edges) -- dossier section 3.
    """
    theta = np.asarray(theta, float)
    P = theta.size
    R = np.zeros((n, n))
    dR = np.zeros((P, n, n))
    for (frm, to, coeffs, cw) in edges:
        if coeffs is not None:
            R[frm, to] += float(np.dot(coeffs, theta))
            for r in range(P):
                dR[r, frm, to] += coeffs[r]
        else:
            R[frm, to] += cw
    absorbing = np.asarray([1.0 if i not in out_v else 0.0
                            for i in range(n)])
    transient = 1.0 - absorbing
    K = K_of(lam, t)
    p = alpha.copy()
    dp = np.zeros((P, n))                      # T0 = 0 here
    pi = np.zeros(K + 2)
    dpi = np.zeros((P, K + 2))
    for j in range(K + 2):
        # primal step (reference semantics)
        moved = p @ (R / lam)
        outflow = (R / lam).sum(axis=1) * p
        p_full = p + moved - outflow
        pi[j] = (p_full * absorbing).sum()
        # tangent step, split by term family
        prop_from_dp = dp + dp @ (R / lam) - (R / lam).sum(axis=1) * dp
        prop_from_p = np.stack([
            p @ (dR[r] / lam) - (dR[r] / lam).sum(axis=1) * p
            for r in range(P)])
        harvest = prop_from_dp + (0.0 if drop == 'Z2' else prop_from_p)
        dpi[:, j] = (harvest * absorbing).sum(axis=1)
        dp_full = prop_from_dp + (0.0 if drop == 'Z1' else prop_from_p)
        p = p_full * transient
        dp = dp_full * transient
    ks = np.arange(K + 1)
    from scipy.special import gammaln
    pois = np.exp(ks * np.log(lam * t) - lam * t - gammaln(ks + 1.0))
    f = lam * float(np.sum(pois * pi[:K + 1]))
    df = lam * (dpi[:, :K + 1] @ pois)
    return f, df


T_EVAL = 0.7
print("== E5 term-zeroing (dossier section 5) ==")
for name, mk in FIXTURES.items():
    g, P = mk()
    theta = np.asarray([1.3, 0.6][:P])
    n, edges, alpha, out_v = structure(g, P)
    lam = 64.0
    # oracle: jax value + gradient of the intact mixture
    f_jax = float(mixture_pdf_jax(jnp.asarray(theta), T_EVAL, n, edges,
                                  alpha, out_v, lam))
    g_jax = np.asarray(jax.grad(mixture_pdf_jax)(jnp.asarray(theta), T_EVAL,
                                                 n, edges, alpha, out_v, lam))
    gn = max(np.max(np.abs(g_jax)), 1e-300)
    f_ex, g_ex = explicit_pdf_and_grad(theta, T_EVAL, n, edges, alpha,
                                       out_v, lam)
    rel_f = abs(f_ex - f_jax) / max(abs(f_jax), 1e-300)
    rel_g = np.max(np.abs(g_ex - g_jax)) / gn
    check(f"Z-parity {name}: explicit == jax.grad",
          rel_f < 1e-9 and rel_g < 1e-9,
          f"(value rel {rel_f:.2e}, grad rel {rel_g:.2e})")
    for z in ('Z1', 'Z2'):
        _, g_drop = explicit_pdf_and_grad(theta, T_EVAL, n, edges, alpha,
                                          out_v, lam, drop=z)
        rel = np.max(np.abs(g_drop - g_jax)) / gn
        check(f"{z} {name}: dropped term BREAKS parity (predicted)",
              rel > 1e-3, f"(grad rel {rel:.2e})")

print()
print("E5 ZEROING COMPLETE" + ("" if not FAILS else f"; FAILURES: {FAILS}"))
sys.exit(0 if not FAILS else 1)
