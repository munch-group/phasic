"""Deferred-3 de-risks E1+E2+E3: the pdf-gradient route decision.

Plan: deferred-3-pdf-gradient-revival-plan.md §2/§4. Production stepping
semantics grounded from source (NOT the plan's shorthand):
- context creation (phasic.c:12787-12907): probability_at[start]=1; one
  INSTANTANEOUS init step at priv3=1 (full-weight redistribution from
  the start vertex -- the special first step); jumps reset to 0; the
  continuous wrapper then sets priv3 = 1/g for all subsequent steps
  (phasic.c:13064-13081) and seeds _pdf[0] from a cdf probe
  (cdf2-cdf1 over one step of a throwaway context).
- each step (phasic.c:12925-12965): absorbing vertices zeroed at step
  start; per-edge p[to] += p[from]*w/g, p[from] -= p[from]*w/g; pmf =
  mass absorbed this step; continuous pdf = pmf * g; time = jumps/g.
- pdf(t, g) returns _pdf[int(g*t)] (phasiccpp.h:1519-1533); auto
  granularity (g=0) = 2 * max(512, max vertex out-rate)
  (phasic.c:12982-13006).

Routes (plan §2):
  (i)  differentiate the Euler recursion at production lambda semantics
       -- gradient bias vs lambda is the question;
  (ii) Poisson-mixture exactness at PINNED lambda:
       f(t) = lambda * sum_k Poisson(k; lambda*t) * pi_{k+1}
       (mass absorbed at DPH step k+1 -- the off-by-one the closed-form
       gate is built to CATCH), gradient exact because dlambda/dtheta=0.

Scope: linear weight mode, parameterized + constant edges,
continuous-only (plan review F6). Small graphs; memory-trivial.
"""
import sys

import numpy as np

import phasic
from phasic import Graph, set_log_level

set_log_level("WARNING")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

FAILS = []


def check(label, rel, tol):
    ok = rel < tol
    print(f"  {label}: rel={rel:.2e} {'PASS' if ok else 'FAIL'} (tol {tol:g})")
    if not ok:
        FAILS.append(label)


# --------------------------------------------------------------------------- fixtures
def expo():
    g = Graph(1)
    s = g.starting_vertex()
    v = g.find_or_create_vertex([2])
    a = g.find_or_create_vertex([1])
    s.add_edge(v, 1.0)
    v.add_edge(a, [1.0])          # rate = theta0
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


def hypo2():
    g = Graph(1)
    s = g.starting_vertex()
    v1 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    a = g.find_or_create_vertex([1])
    s.add_edge(v1, 1.0)
    v1.add_edge(v2, [1.0, 0.0])   # rate theta0
    v2.add_edge(a, [0.0, 1.0])    # rate theta1
    return g, 2


def cyclic4():
    g = Graph(1)
    s = g.starting_vertex()
    b = g.find_or_create_vertex([3])
    c = g.find_or_create_vertex([2])
    a = g.find_or_create_vertex([1])
    s.add_edge(b, 1.0)
    b.add_edge(c, [1.0, 0.0])
    c.add_edge(b, [0.0, 0.4])     # cycle back
    c.add_edge(a, [1.0, 0.3])
    # a genuine constant edge in a parameterized graph (plan scope):
    # scalar add_edge is refused (mode lock) -- the constant-edge route
    # is add_aux_vertex_constant, which serializes into constant_edges
    b.add_aux_vertex_constant(0.15)
    return g, 2


FIXTURES = {'expo': expo, 'erlang3': erlang3, 'hypo2': hypo2,
            'cyclic4': cyclic4}


# --------------------------------------------------------------------------- structure extraction
def structure(g, P):
    ser = g.serialize(theta_dim=P)
    pe = np.asarray(ser['param_edges'], dtype=float)
    ce = np.asarray(ser.get('constant_edges', []), dtype=float)
    se = np.asarray(ser['start_edges'], dtype=float)
    n = int(ser['n_vertices'])
    edges = []       # (frm, to, coeffs or None, const_w)
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


def rates_matrix(theta, n, edges):
    """Dense per-vertex rate matrix R[i,j] (JAX), absorbing rows zero."""
    R = jnp.zeros((n, n))
    for (frm, to, coeffs, cw) in edges:
        w = jnp.dot(jnp.asarray(coeffs), theta) if coeffs is not None else cw
        R = R.at[frm, to].add(w)
    return R


# --------------------------------------------------------------------------- the JAX Euler reference
def jax_pdf_series(theta, n, edges, alpha, out_v, g_gran, n_steps):
    """Replicates the production series: pdf_k for k=0..n_steps.

    Init: mass alpha over the START edges' targets (the instantaneous
    full-weight first step from the start vertex = alpha itself, since
    start-edge weights sum to 1 and the start vertex is emptied).
    Then per step: absorbing mass harvested (pdf = absorbed * g), and
    p' = p + p @ (R/g) restricted appropriately.
    """
    R = rates_matrix(theta, n, edges)
    absorbing = jnp.asarray([1.0 if i not in out_v else 0.0
                             for i in range(n)])
    p = jnp.asarray(alpha)

    def step(p, _):
        moved = (p[:, None] * R / g_gran).sum(axis=0)   # inflow per vertex
        outflow = (R / g_gran).sum(axis=1) * p          # outflow per vertex
        p_new = p + moved - outflow
        absorbed = (p_new * absorbing).sum()
        p_new = p_new * (1.0 - absorbing)
        return p_new, absorbed * g_gran

    # _pdf[0]: the creation-time probe = absorbed mass of the FIRST step
    _, pdf_all = jax.lax.scan(step, p, None, length=n_steps + 1)
    return pdf_all  # pdf_all[k] corresponds to _pdf[k]


def jax_pdf_at(theta, t, n, edges, alpha, out_v, g_gran):
    # production index convention (grounded): _pdf[0] is the creation
    # probe of STEP 1, and _pdf[k] for k>=1 is the pdf of step k -- so
    # _pdf[0] duplicates _pdf[1], and production _pdf[idx] = my
    # series[idx-1] for idx >= 1 (series[k] = absorbed at scan step
    # k+1). Verified by the parity gate (an off-by-one shows up as a
    # 1/g-class error, ~1e-3 at g=1024 -- the first run's failure).
    idx = int(g_gran * t)
    series = jax_pdf_series(theta, n, edges, alpha, out_v, g_gran, idx + 1)
    return series[max(idx - 1, 0)]


# --------------------------------------------------------------------------- route (ii): Poisson mixture
def poisson_mixture_pdf(theta, t, n, edges, alpha, out_v, lam, K_sigma=6.0,
                        off_by_one_wrong=False):
    """f(t) = lam * sum_k Poisson(k; lam t) * pi_{k+1}, pi_j = mass
    absorbed at DPH step j of the UNIFORMIZED chain at rate lam.
    off_by_one_wrong=True uses pi_k instead (the alignment error the
    gate must catch)."""
    R = rates_matrix(theta, n, edges)
    absorbing = jnp.asarray([1.0 if i not in out_v else 0.0
                             for i in range(n)])
    mean_k = lam * t
    K = int(mean_k + K_sigma * np.sqrt(max(mean_k, 1.0)) + 10)
    p = jnp.asarray(alpha)

    def step(p, _):
        moved = (p[:, None] * R / lam).sum(axis=0)
        outflow = (R / lam).sum(axis=1) * p
        p_new = p + moved - outflow
        absorbed = (p_new * absorbing).sum()
        p_new = p_new * (1.0 - absorbing)
        return p_new, absorbed

    _, pi = jax.lax.scan(step, p, None, length=K + 2)  # pi[j-1] = absorbed at step j
    ks = jnp.arange(K + 1)
    log_pois = ks * jnp.log(lam * t) - lam * t - jax.scipy.special.gammaln(ks + 1.0)
    pois = jnp.exp(log_pois)
    if off_by_one_wrong:
        pi_used = pi[:K + 1]        # pi_{k} -- WRONG alignment
    else:
        pi_used = pi[1:K + 2] if False else jnp.concatenate(
            [pi[0:K + 1]])  # see note below
    # CORRECT alignment: mass absorbed at step k+1 pairs with Poisson(k).
    # pi[j] from the scan is "absorbed at step j+1" (scan step 1 -> index
    # 0), so pi[k] IS pi_{k+1}: the correct sum uses pi[0:K+1] directly,
    # and the WRONG variant shifts by one.
    if off_by_one_wrong:
        pi_used = jnp.concatenate([jnp.zeros(1), pi[:K]])
    else:
        pi_used = pi[:K + 1]
    return lam * jnp.sum(pois * pi_used)


# --------------------------------------------------------------------------- run
print("== E1/E2: production parity + closed forms + route-(i) convergence ==")
T_EVAL = 0.7
for name, mk in FIXTURES.items():
    g, P = mk()
    theta = [1.3, 0.6][:P]
    n, edges, alpha, out_v = structure(g, P)
    g.update_weights(list(theta))
    # production pdf at auto granularity: derive g_gran the same way
    max_rate = 512.0
    R_np = np.zeros((n, n))
    for (frm, to, coeffs, cw) in edges:
        w = float(np.dot(coeffs, theta)) if coeffs is not None else cw
        R_np[frm, to] += w
    for i in range(n):
        max_rate = max(max_rate, R_np[i].sum())
    g_gran = int(max_rate * 2.0)
    prod = g.pdf(T_EVAL, g_gran)
    ref = float(jax_pdf_at(jnp.asarray(theta), T_EVAL, n, edges, alpha,
                           out_v, g_gran))
    rel = abs(prod - ref) / max(abs(prod), 1e-300)
    check(f"E2 parity {name} (g={g_gran})", rel, 1e-10)

# closed form: Exponential pdf + gradient
g, P = expo()
n, edges, alpha, out_v = structure(g, P)
th0 = 1.3


def route_i_pdf(th, lam):
    return jax_pdf_at(jnp.asarray([th]), T_EVAL, n, edges, alpha, out_v, lam)


closed_pdf = th0 * np.exp(-th0 * T_EVAL)
closed_grad = (1 - th0 * T_EVAL) * np.exp(-th0 * T_EVAL)
print("== route (i): gradient bias vs lambda (expo closed form) ==")
prev_bias = None
for lam in (1024, 4096, 16384):
    v = float(route_i_pdf(th0, lam))
    gr = float(jax.grad(route_i_pdf)(th0, lam))
    vb = abs(v - closed_pdf) / closed_pdf
    gb = abs(gr - closed_grad) / abs(closed_grad)
    order = "" if prev_bias is None else f" (ratio {prev_bias/gb:.1f}x per 4x lambda)"
    print(f"  lambda={lam}: value bias {vb:.2e}, grad bias {gb:.2e}{order}")
    prev_bias = gb
fd_err_ref = 5e-8  # FD's typical central-diff error class at benign theta
print(f"  -> route (i) grad bias at practical lambda vs FD-class error "
      f"({fd_err_ref:.0e}): {'BELOW' if prev_bias < fd_err_ref else 'ABOVE'}")

print("== route (ii): Poisson mixture at pinned lambda ==")
LAM = 64.0  # >= max rate (513 not needed: max out-rate here is th0=1.3)


def route_ii_pdf(th, t):
    return poisson_mixture_pdf(jnp.asarray([th]), t, n, edges, alpha,
                               out_v, LAM)


v = float(route_ii_pdf(th0, T_EVAL))
rel = abs(v - closed_pdf) / closed_pdf
check("E3 expo value (correct alignment)", rel, 1e-9)
v_wrong = float(poisson_mixture_pdf(jnp.asarray([th0]), T_EVAL, n, edges,
                                    alpha, out_v, LAM,
                                    off_by_one_wrong=True))
rel_wrong = abs(v_wrong - closed_pdf) / closed_pdf
print(f"  off-by-one variant rel={rel_wrong:.2e} "
      f"{'-- GATE CATCHES IT' if rel_wrong > 1e-3 else '-- GATE TOO WEAK (FAIL)'}")
if rel_wrong <= 1e-3:
    FAILS.append("off-by-one gate")
gr = float(jax.grad(route_ii_pdf)(th0, T_EVAL))
rel_g = abs(gr - closed_grad) / abs(closed_grad)
check("E3 expo gradient (dlambda/dtheta=0 => exact)", rel_g, 1e-9)

# erlang-3 closed form via route (ii)
g3, P3 = erlang3()
n3, e3_, a3, o3 = structure(g3, P3)
lam3 = 64.0


def route_ii_erl(th, t):
    return poisson_mixture_pdf(jnp.asarray([th]), t, n3, e3_, a3, o3, lam3)


th_e = 1.1
t_e = 1.4
closed_e = th_e**3 * t_e**2 * np.exp(-th_e * t_e) / 2.0
d_closed = (3 * th_e**2 * t_e**2 - th_e**3 * t_e**3) * np.exp(-th_e * t_e) / 2.0
check("E3 erlang3 value", abs(float(route_ii_erl(th_e, t_e)) - closed_e) / closed_e, 1e-8)
check("E3 erlang3 gradient",
      abs(float(jax.grad(route_ii_erl)(th_e, t_e)) - d_closed) / abs(d_closed), 1e-8)

# cyclic: route (ii) vs route (i) at very high lambda (self-consistency)
gc, Pc = cyclic4()
nc, ec, ac, oc = structure(gc, Pc)
thc = jnp.asarray([1.3, 0.6])
v_ii = float(poisson_mixture_pdf(thc, T_EVAL, nc, ec, ac, oc, 64.0))
v_i = float(jax_pdf_at(thc, T_EVAL, nc, ec, ac, oc, 16384))
check("E3 cyclic: route-(ii) vs high-lambda route-(i)",
      abs(v_ii - v_i) / max(abs(v_i), 1e-300), 1e-3)
g_ii = np.asarray(jax.grad(
    lambda th: poisson_mixture_pdf(th, T_EVAL, nc, ec, ac, oc, 64.0))(thc))
g_fd = np.zeros(2)
for j in range(2):
    e_ = np.zeros(2); e_[j] = 1e-6
    g_fd[j] = (float(poisson_mixture_pdf(thc + e_, T_EVAL, nc, ec, ac, oc, 64.0))
               - float(poisson_mixture_pdf(thc - e_, T_EVAL, nc, ec, ac, oc, 64.0))) / 2e-6
check("E3 cyclic gradient vs FD-of-mixture",
      float(np.max(np.abs(g_ii - g_fd)) / np.max(np.abs(g_fd))), 1e-6)

# lambda < max rate: must be loud/detectable
print("== route (ii) failure mode: lambda < max out-rate ==")
bad = poisson_mixture_pdf(jnp.asarray([10.0]), T_EVAL, n, edges, alpha,
                          out_v, 2.0)  # rate 10 > lam 2 -> negative probs
print(f"  lam=2 < rate=10: mixture value = {float(bad):.4g} "
      f"(negative intermediate probabilities -> value corrupt; DETECTABLE "
      f"by a p<0 check in the stepper, which the implementation must add)")

print()
print("ALL D3 GATES PASS" if not FAILS else f"FAILURES: {FAILS}")
sys.exit(0 if not FAILS else 1)
