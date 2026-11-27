"""Graph.svgd declarations on formula / callback models with a theta dimension
that differs from the coefficient length (decoupled)."""
import sys, traceback
sys.path.insert(0, "/Users/kmt/phasic-fdsteps/tests/pytest")
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from phasic import GaussPrior
import _fd_golden_cells as C

DATA = np.asarray([0.3, 0.5, 0.8, 1.1, 1.7, 2.4])
FIT = dict(n_particles=4, n_iterations=2, progress=False, seed=1, learning_rate=1e-6)
gp = lambda: GaussPrior(1.0, 0.5, lower=0.0)


def run(name, fn):
    try:
        print(f"[{name}] OK: {fn()}", flush=True)
    except Exception as e:
        print(f"[{name}] RAISE {type(e).__name__}: {str(e)[:400]}", flush=True)
        traceback.print_exc(limit=-4)


def f1():
    g = C.beside()
    s = g.svgd(DATA, weight_formula="c0*t0 + c1*t0", prior=gp(), **FIT)
    return s.theta_dim, s.model._param_domains
run("formula_decoupled_1theta", f1)


def f2():
    g = C.beside()
    s = g.svgd(DATA, weight_formula="c0*t0 + c1*t1 + c1*t2", prior=gp(), **FIT)
    return s.theta_dim, s.model._param_domains
run("formula_decoupled_3theta", f2)


def f3():
    g = C.beside()
    s = g.svgd(DATA, callback=lambda t, c: c[0] * t[0] + c[1] * t[0], theta_dim=1,
               prior=gp(), **FIT)
    return s.theta_dim, s.model._param_domains
run("callback_theta_dim_1", f3)


def f4():
    g = C.beside()
    s = g.svgd(DATA, weight_formula="c0*t0 + c1*t1 + c1*t2", prior=gp(),
               fixed=[(2, 0.5)], **FIT)
    return s.theta_dim, s.model._param_domains
run("formula_decoupled_3theta_fixed", f4)
