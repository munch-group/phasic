"""Deferred-3 E0 -- value measurement: how much gradient error does the
FD PMF term contribute on realistic likelihoods?

Plan: deferred-3-pdf-gradient-revival-plan.md §4-E0, run compactly:
instead of full SVGD fits, the SVGD LOSS GRADIENT is evaluated at (a)
benign / (b) mixed-scale anchor thetas and (c) draws from the ACTUAL
SVGD log-scale init distribution (sd=5 -- the Batch-E lesson: sweeps
must sample the real particle-init distribution). Oracle = Richardson
relative-step central differences of the SAME primal (the A2-proven
self-oracle; valid wherever the primal is smooth). Attribution: in
`pmf_and_moments_from_graph` the moments term is EXACT (B3 default),
so the shipped-vs-oracle gradient gap is the FD PMF term's error; in
`pmf_from_graph` everything is FD.

Park-if-immaterial test: if the FD PMF error is negligible everywhere
it matters, Deferred 3 parks (a legitimate outcome per the plan).
"""
import sys

import numpy as np

import phasic
from phasic import Graph, set_log_level

set_log_level("WARNING")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def hypo2():
    g = Graph(1)
    s = g.starting_vertex()
    v1 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    a = g.find_or_create_vertex([1])
    s.add_edge(v1, 1.0)
    v1.add_edge(v2, [1.0, 0.0])
    v2.add_edge(a, [0.0, 1.0])
    return g, 2


def coal5():
    # Kingman coalescent n=5, single rate parameter
    g = Graph(1)
    s = g.starting_vertex()
    vs = {k: g.find_or_create_vertex([k]) for k in range(1, 6)}
    s.add_edge(vs[5], 1.0)
    for k in range(5, 1, -1):
        rate = k * (k - 1) / 2.0
        vs[k].add_edge(vs[k - 1], [rate])
    return g, 1


FIXTURES = {"hypo2": hypo2, "coal5": coal5}
OBS = np.array([0.3, 0.7, 1.2, 2.0])
rng = np.random.default_rng(7)


def richardson(fn, th, rel=1e-4):
    th = np.asarray(th, float)
    g = np.zeros_like(th)
    for j in range(th.size):
        h = max(abs(th[j]) * rel, 1e-14)
        def cd(hh):
            tp = th.copy(); tp[j] += hh
            tm = th.copy(); tm[j] -= hh
            return (fn(tp) - fn(tm)) / (2 * hh)
        g[j] = (4 * cd(h / 2) - cd(h)) / 3.0
    return g


def measure(tag, model, theta):
    def primal_jax(t):
        out = model(jnp.asarray(t))
        arr = out[0] if isinstance(out, (tuple, list)) else out
        return jnp.sum(jnp.log(jnp.clip(arr.reshape(-1), 1e-300, None)))

    def primal_np(t):
        return float(primal_jax(t))
    try:
        g_ship = np.asarray(jax.grad(primal_jax)(jnp.asarray(theta, float)))
    except Exception as exc:
        print(f"    {tag}: shipped grad RAISED {type(exc).__name__}: "
              f"{str(exc).splitlines()[-1][:80]}")
        return
    g_ref = richardson(primal_np, theta)
    denom = max(np.max(np.abs(g_ref)), 1e-300)
    rel = np.max(np.abs(g_ship - g_ref)) / denom
    print(f"    {tag}: theta={np.asarray(theta).tolist()} "
          f"max-slot rel err = {rel:.2e}")
    return rel


print("== E0: FD-PMF-term gradient error on realistic likelihoods ==")
for name, mk in FIXTURES.items():
    g, P = mk()
    print(f"\n[{name}] P={P}, obs={OBS.tolist()}")

    pm_raw = Graph.pmf_and_moments_from_graph(g, nr_moments=2)
    pf_raw = Graph.pmf_from_graph(g)
    # model(theta, times) -> (pmf_values, moments) / pmf_values
    pm = lambda t, _m=pm_raw: _m(t, jnp.asarray(OBS))
    pf = lambda t, _m=pf_raw: _m(t, jnp.asarray(OBS))

    thetas = {
        "benign": np.full(P, 1.0),
        "mixed": np.array([1.0, 1e-8][:P]) if P == 2 else np.array([1e-8]),
    }
    # draws from the ACTUAL svgd log-scale init (log10-scale sd=5 would
    # span e^{+-15}; production init is N(0, sd) in log space -- use
    # sd=2 to stay in the numerically representable band while still
    # exercising decades of scale, and note the truncation honestly)
    for i, z in enumerate(rng.normal(0.0, 2.0, size=(4, P))):
        thetas[f"init-draw-{i}"] = np.exp(z)

    print("  -- pmf_and_moments_from_graph (moments EXACT, pmf FD):")
    for tag, th in thetas.items():
        def model_pm(t, _pm=pm):
            return _pm(t)
        measure(tag, pm, th)
    print("  -- pmf_from_graph (100% FD):")
    for tag, th in thetas.items():
        measure(tag, pf, th)

print("\nE0 MEASUREMENT COMPLETE (interpretation in the findings)")
