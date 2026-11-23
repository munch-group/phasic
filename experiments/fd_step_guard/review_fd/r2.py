"""Claim check: 'Only the epoch FFI checks for negative rates; every other path
silently evaluates the model at the negative rate.'  Formula weight mode on
pmf_from_graph (S2, +-1e-7 probes) and pmf_and_moments (C3, clamped probes)."""
import sys, os, logging, traceback
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
logging.disable(logging.WARNING)
from test_notebook_model_recovery import two_island_graph
from phasic import Graph
print('rule', os.environ.get('PHASIC_FD_RULE', 'new(or master)'))
T = jnp.array([0.3, 0.8, 1.5, 2.5])
for mode in ('linear', 'formula'):
    for name in ('pmf_from_graph', 'pmf_and_moments'):
        g = two_island_graph(4)
        if mode == 'formula':
            g.weight_formula = 'c0*t0 + c1*t1'
        if name == 'pmf_from_graph':
            m = Graph.pmf_from_graph(g, theta_dim=2)
            f = lambda th: jnp.sum(jnp.log(m(th, T)))
        else:
            m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2)
            f = lambda th: jnp.sum(jnp.log(m(th, T)[0]))
        for v in (1e-8, 1e-6):
            th = jnp.array([0.5, v])   # migration small
            try:
                gr = np.asarray(jax.grad(f)(th))
                print(f'{mode:8s} {name:16s} mig={v:g}: grad={gr}', flush=True)
            except Exception as e:
                print(f'{mode:8s} {name:16s} mig={v:g}: {type(e).__name__}: {str(e).splitlines()[0][:160]}', flush=True)
