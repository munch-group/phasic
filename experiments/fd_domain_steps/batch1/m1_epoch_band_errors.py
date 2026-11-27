"""Measure the epoch-model band-slot gradient error after Batch 1 (to set test
tolerances from evidence): fallback gradient vs references at several steps
(all probes inside the domain) and a Richardson log-coordinate reference."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tests", "pytest"))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import phasic  # noqa
import _fd_golden_cells as C


def rich(L, th, i, s=1e-2):
    def d(h):
        return float(L(th.at[i].multiply(np.exp(h))) - L(th.at[i].multiply(np.exp(-h)))) / (2 * h) / float(th[i])
    a, b, c = d(s), d(s / 2), d(s / 4)
    r1, r2 = (4 * b - a) / 3, (4 * c - b) / 3
    return (16 * r2 - r1) / 15, abs(r2 - r1) / abs((16 * r2 - r1) / 15)


for cell in ("dc_noexp", "dc_exp", "dc_efg", "dcjp"):
    f = C.build(cell)
    L = lambda t: jnp.sum(jnp.log(jax.tree_util.tree_leaves(f(t))[0]))
    for theta in ([1.0, 1e-8, 1.0, 1e-8], [1.0, 1e-9, 1.0, 1e-9], [1e-9, 1e-4, 1e-4, 1e-4]):
        th = jnp.asarray(theta)
        g = np.asarray(jax.grad(L)(th))
        for i, t in enumerate(theta):
            if not (0 < t <= 1e-7):
                continue
            r, sp = rich(L, th, i)
            c2 = float((L(th.at[i].add(t / 2)) - L(th.at[i].add(-t / 2))) / t)
            print(f"{cell:9s} theta={theta} slot {i}: grad {g[i]: .10e} rich {r: .10e} (spread {sp:.1e}) "
                  f"rel-vs-rich {abs(g[i]-r)/abs(r):.2e}  central(t/2) rel {abs(c2-r)/abs(r):.2e}", flush=True)
