import numpy as np, jax, jax.numpy as jnp
from phasic import Graph, set_log_level
set_log_level("ERROR")
exec(open('p1_modes.py').read().split("times = ")[0].split('set_log_level("ERROR")')[1])
times = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float64)
g = g2()
g.weight_callback = lambda t, c: c[0] * t[0] * jnp.sqrt(c[1] * t[1])
for name, mk in (("pam", lambda: Graph.pmf_and_moments_from_graph(g, nr_moments=1, theta_dim=2)),
                 ("pfg", lambda: Graph.pmf_from_graph(g, theta_dim=2))):
    try:
        m = mk()
    except Exception as e:
        print(name, "BUILD RAISES", str(e)[:200]); continue
    def fw(th):
        out = m(th, times)
        return out[0] if isinstance(out, tuple) else out
    for th in ([1.0, 1.0], [1.0, 5e-8], [1.0, -5e-8]):
        try:
            print(name, "fwd", th, np.asarray(fw(jnp.array(th, dtype=jnp.float64))), flush=True)
            print(name, "grad", th, np.asarray(jax.grad(lambda x: jnp.sum(jnp.log(fw(x))))(jnp.array(th, dtype=jnp.float64))), flush=True)
        except Exception as e:
            print(name, th, "RAISES", str(e).replace(chr(10), ' ')[:200], flush=True)
