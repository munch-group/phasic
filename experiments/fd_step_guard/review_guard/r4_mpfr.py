"""Is moments (elimination) cost rate-independent? Mixed-scale theta can auto-activate MPFR."""
import sys, time, logging
import numpy as np, jax.numpy as jnp
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import two_island_graph
from phasic import Graph
g = two_island_graph(10)
m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2)
T = jnp.array([0.5, 1.0])
for th in ([1.0, 0.5], [1.0, 0.5], [1e-4, 1e4], [1e-7, 1e7], [1e-9, 1e9]):
    # keep max rate low-ish for the pdf: only moments cost matters; use direct moments
    g.update_weights(th)
    t0 = time.time()
    for _ in range(5): mom = g.moments(2)
    dt = (time.time()-t0)/5
    print(f'theta={th}: moments {dt*1e3:.2f} ms  {np.asarray(mom)}', flush=True)
