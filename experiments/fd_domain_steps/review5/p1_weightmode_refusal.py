"""Do the Euler-pdf and joint/moments FD sites REFUSE sub-zero probes under
non-linear weight modes (formula / log), contrary to plan §4.4's
per-site list (which classifies the Euler pdf forward as accepting)?"""
import os, traceback
import numpy as np
import jax, jax.numpy as jnp
import phasic
from phasic import Graph

def chain(mode=None, formula=None):
    g = phasic.Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2]); v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge_parameterized(v1, 0.0, [1.0, 1.0])
    if mode: g.weight_mode = mode
    if formula: g.weight_formula = formula
    return g

times = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float64)
def run(label, g, theta, builder='pmf_from_graph'):
    try:
        if builder == 'pmf_from_graph':
            f = Graph.pmf_from_graph(g)
            L = lambda th: jnp.sum(jnp.log(f(th, times)))
        else:
            f = Graph.pmf_and_moments_from_graph(g, nr_moments=1)
            L = lambda th: jnp.sum(jnp.log(f(th, times)[0]))
        gr = jax.grad(L)(jnp.array(theta))
        print(f"{label:55s} theta={theta} grad={np.asarray(gr)}")
    except Exception as e:
        print(f"{label:55s} theta={theta} RAISES {type(e).__name__}: {str(e)[:160]}")

for b in ['pmf_from_graph', 'pmf_and_moments_from_graph']:
    for th in ([1.0, 5e-8], [1.0, 1e-9]):
        run(f"{b} linear", chain(), th, b)
        run(f"{b} formula c0*t0+c1*t1", chain(formula="c0*t0 + c1*t1"), th, b)
        run(f"{b} formula c0*t0+c1*sqrt(t1)", chain(formula="c0*t0 + c1*sqrt(t1)"), th, b)
        run(f"{b} log", chain(mode='log'), th, b)
