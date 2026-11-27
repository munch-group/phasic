"""Review probe r8: magnitudes behind the test cells that survive the zero/half
alternative-quotient mutations (r7): is the asserted value informative?"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../tests/pytest"))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import phasic
from phasic import Graph
import _fd_golden_cells as C
import test_fd_domain_steps as T

for b, mode, disc, rw in (("pam", "log", False, False), ("pam", "log", False, True),
                          ("rvp", "log", False, False), ("rvp", "formula", False, False),
                          ("rvp", "linear", False, False)):
    f, L = T._beside_model(b, mode, disc, rewards=rw)
    th = jnp.asarray([0.7, 5e-8])
    g = np.asarray(jax.grad(L)(th)); ref = T._central(L, th, 1, 1e-9)
    print(f"{b}-{mode}-{'rw' if rw else ''}: g[1]={g[1]:.4g} ref={ref:.4g}  value={float(L(th)):.4g}  (assert: rel 1e-3 OR abs 1e-6)")

for cell in ("dc_noexp", "dcjp"):
    f = C.build(cell)
    L = lambda t: jnp.sum(jnp.log(jax.tree_util.tree_leaves(f(t))[0]))
    for theta in T._EPOCH_CASES[:3]:
        th = jnp.asarray(theta)
        g = np.asarray(jax.grad(L)(th))
        for i, t in enumerate(theta):
            if 0 < t <= 1e-7:
                ref = T._richardson_log(L, th, i)
                tol = 1e-6 * (1 + abs(t * ref))
                print(f"{cell} {theta} slot{i}: g={g[i]:.6g} ref={ref:.6g} |t*g - t*ref|={abs(t*g[i]-t*ref):.3g} "
                      f"tol={tol:.3g}; zero gradient would pass: {abs(t*ref) <= tol}; |g| allowed up to ~{tol/t:.3g}")
