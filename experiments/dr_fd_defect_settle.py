"""Settle the mixed-scale FD defect: extent, threshold, severity.

The claim on record (CLAUDE.md, memory) is that every SVGD-facing
gradient uses a central difference with an ABSOLUTE step eps=1e-7
applied identically to every parameter, and therefore fails when
parameters differ greatly in scale ("4-9% error at theta=[1,1e-8]").
That claim was produced by inspection plus a pinned test, never
adjudicated against an independent oracle across a range, and never
connected to whether an inference RESULT is wrong.

A defect in finite differences cannot be adjudicated with finite
differences, so this uses a CLOSED FORM.

Model: 2-phase hypoexponential, rates (lam1, lam2) = theta. Its density
is analytic,
    f(t) = lam1*lam2/(lam2-lam1) * (exp(-lam1 t) - exp(-lam2 t)),
so the exact gradient of the log-likelihood sum_i log f(t_i) w.r.t.
theta is available in closed form (autodiffed here from the closed
form itself, which is independent of anything phasic computes).

Sweep: lam1 fixed at 1.0, lam2 swept over ten decades. At each point:
  - the SHIPPED gradient, exactly as SVGD obtains it, i.e. jax.grad of
    the loss built on Graph.pmf_and_moments_from_graph;
  - the CLOSED-FORM gradient;
  - relative error per parameter slot.
Also reported: the shipped FORWARD value's own error, to separate a
gradient defect from a primal defect.
"""
import sys

import numpy as np

import phasic
from phasic import Graph, set_log_level

set_log_level("ERROR")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

TIMES = np.array([0.25, 0.8, 1.5, 3.0])


def hypo2_graph():
    g = Graph(1)
    s = g.starting_vertex()
    v1 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    a = g.find_or_create_vertex([1])
    s.add_edge(v1, 1.0)
    v1.add_edge(v2, [1.0, 0.0])     # rate lam1 = theta0
    v2.add_edge(a, [0.0, 1.0])      # rate lam2 = theta1
    return g


def closed_form_loglik(theta, times):
    """log-likelihood of the 2-phase hypoexponential, pure JAX."""
    l1, l2 = theta[0], theta[1]
    # numerically safe form; the sweep never puts l1 == l2
    f = l1 * l2 / (l2 - l1) * (jnp.exp(-l1 * times) - jnp.exp(-l2 * times))
    return jnp.sum(jnp.log(f))


closed_grad = jax.grad(closed_form_loglik)

model = Graph.pmf_and_moments_from_graph(hypo2_graph(), nr_moments=2)


def shipped_loglik(theta):
    """Exactly the shape SVGD uses: sum of log PDF at observed times.
    (SVGD adds a 1e-10 offset inside the log; included so this measures
    the real objective, not an idealised one.)"""
    pmf, _mom = model(theta, jnp.asarray(TIMES))
    return jnp.sum(jnp.log(pmf + 1e-10))


shipped_grad = jax.grad(shipped_loglik)

print("Settling the mixed-scale FD defect against a CLOSED FORM")
print("model: 2-phase hypoexponential, theta=(lam1, lam2), lam1 fixed 1.0")
print("times:", TIMES.tolist())
print()
print(f"{'lam2':>9} {'ratio':>8} | {'value rel err':>14} | "
      f"{'grad rel err d/dlam1':>21} {'d/dlam2':>12} | verdict")
print("-" * 92)

rows = []
for e in range(0, 11):
    lam2 = 10.0 ** (-e)
    if abs(lam2 - 1.0) < 1e-12:
        lam2 = 0.5           # avoid the removable singularity lam1==lam2
    th = jnp.asarray([1.0, lam2])

    v_ship = float(shipped_loglik(th))
    v_true = float(closed_form_loglik(th, jnp.asarray(TIMES)))
    v_rel = abs(v_ship - v_true) / max(abs(v_true), 1e-300)

    g_ship = np.asarray(shipped_grad(th))
    g_true = np.asarray(closed_grad(th, jnp.asarray(TIMES)))
    denom = np.maximum(np.abs(g_true), 1e-300)
    g_rel = np.abs(g_ship - g_true) / denom

    worst = float(np.max(g_rel))
    verdict = ("OK" if worst < 1e-4 else
               "DEGRADED" if worst < 1e-1 else
               "WRONG")
    print(f"{lam2:9.1e} {1.0/lam2:8.0e} | {v_rel:14.2e} | "
          f"{g_rel[0]:21.2e} {g_rel[1]:12.2e} | {verdict}")
    rows.append((lam2, v_rel, g_rel[0], g_rel[1], worst))

print()
bad = [r for r in rows if r[4] >= 1e-1]
deg = [r for r in rows if 1e-4 <= r[4] < 1e-1]
if bad:
    print(f"WRONG (>=10% error) first at lam2 = {bad[0][0]:.1e} "
          f"(scale ratio {1.0/bad[0][0]:.0e})")
if deg:
    print(f"DEGRADED (>=0.01%) first at lam2 = {deg[0][0]:.1e} "
          f"(scale ratio {1.0/deg[0][0]:.0e})")
if not bad and not deg:
    print("No degradation anywhere in the swept range.")
print()
print("Interpretation note: the shipped backward perturbs by an ABSOLUTE")
print("eps=1e-7, so once a parameter falls below ~1e-6 the probe is a")
print("large relative perturbation of it, and below ~1e-7 the probe")
print("crosses zero. The value column isolates whether the FORWARD is")
print("also affected (it should not be).")
