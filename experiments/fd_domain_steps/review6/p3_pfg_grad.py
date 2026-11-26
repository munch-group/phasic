import numpy as np, jax, jax.numpy as jnp
from phasic import Graph, set_log_level
set_log_level("ERROR")
exec(open('p1_modes.py').read().split("times = ")[0].split('set_log_level("ERROR")')[1])
import traceback
for mode in ('linear', 'log', 'formula_sqrt'):
    g = make(mode)
    m = Graph.pmf_from_graph(g, theta_dim=2)
    t = jnp.array([0.5, 1.0, 2.0])
    for th in ([1.0, 1.0], [1.0, 5e-8]):
        for name, f in (("grad", jax.grad(lambda x: jnp.sum(jnp.log(m(x, t))))),
                        ("jit(grad)", jax.jit(jax.grad(lambda x: jnp.sum(jnp.log(m(x, t))))))):
            try:
                print(mode, th, name, np.asarray(f(jnp.array(th))))
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                print(mode, th, name, "RAISES", str(e).replace("\n", " ")[:230])
        try:
            print(mode, th, "fwd at minus probe", np.asarray(m(jnp.array([th[0], th[1]-1e-7]), t)))
        except Exception as e:
            print(mode, th, "fwd at minus probe RAISES", str(e).replace("\n", " ")[:200])
