"""Review probe r2: cost of the Plan A band machinery on a refusing model (the
epoch model, production path) under jit(vmap(grad)) -- SVGD's usage -- versus
an emulated master (slot_rule None -> TODAY, which reproduces master's traced
program for undeclared slots). jax.debug.callback's batching rule UNROLLS one
callback per batch element (jax 0.7.2 debugging.py:112)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../tests/pytest"))
import phasic
import jax, jax.numpy as jnp
import numpy as np
from phasic import _fd_steps
import _fd_golden_cells as C

_orig = _fd_steps.slot_rule


def master_rule(domain, refuses):
    if domain is None:
        return _fd_steps.TODAY
    return _orig(domain, refuses)


def run(variant, P, reps=5):
    _fd_steps.slot_rule = master_rule if variant == "master" else _orig
    f = C.build("dc_noexp")
    L = lambda t: jnp.sum(jnp.log(f(t)[0]))
    G = jax.jit(jax.vmap(jax.grad(L)))
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.uniform(0.3, 1.5, size=(P, 4)))
    t0 = time.perf_counter(); out = G(X).block_until_ready(); tc = time.perf_counter() - t0
    hlo = G.lower(X).as_text()
    ncb = hlo.count("callback")
    ts = []
    for _ in range(reps):
        X2 = X * (1 + 1e-3 * rng.standard_normal(X.shape))
        t0 = time.perf_counter(); G(X2).block_until_ready(); ts.append(time.perf_counter() - t0)
    _fd_steps.slot_rule = _orig
    return tc, float(np.median(ts)), ncb, np.asarray(out)


for P in (8, 64, 256):
    rm = run("master", P)
    rb = run("branch", P)
    same = rm[3].tobytes() == rb[3].tobytes()
    print(f"P={P:4d} master: first={rm[0]:.2f}s steady={rm[1]*1e3:.1f}ms hlo'callback'={rm[2]} | "
          f"branch: first={rb[0]:.2f}s steady={rb[1]*1e3:.1f}ms hlo'callback'={rb[2]} | bit-identical={same}",
          flush=True)
