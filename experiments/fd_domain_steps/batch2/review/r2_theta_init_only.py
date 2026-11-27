"""Graph.svgd with theta_init and no theta_dim: where does it raise?"""
import sys, traceback
sys.path.insert(0, "/Users/kmt/phasic-fdsteps/tests/pytest")
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from phasic import GaussPrior, BetaPrior
import _fd_golden_cells as C

DATA = np.asarray([0.3, 0.5, 0.8, 1.1, 1.7, 2.4])
gp = lambda: GaussPrior(1.0, 0.5, lower=0.0)
g = C.beside()
for prior in ([gp(), gp()], [BetaPrior(2, 2), gp()], None):
    try:
        s = g.svgd(DATA, prior=prior, theta_init=np.full((4, 2), 0.7), n_iterations=1, n_particles=4,
                   progress=False, learning_rate=1e-6)
        print("OK", getattr(s.model, "_param_domains", "?"), flush=True)
    except Exception:
        print("RAISE for prior", prior)
        traceback.print_exc(limit=-6)
