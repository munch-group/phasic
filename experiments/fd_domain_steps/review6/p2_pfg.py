import sys; sys.path.insert(0, '.')
import numpy as np, jax, jax.numpy as jnp
from phasic import Graph, set_log_level
set_log_level("ERROR")
exec(open('p1_modes.py').read().split("times = ")[0].split('set_log_level("ERROR")')[1])
g = make('linear')
m = Graph.pmf_from_graph(g, theta_dim=2)
try:
    print(m(jnp.array([1.0, 1.0]), jnp.array([0.5, 1.0, 2.0])))
except Exception as e:
    print(str(e)[:600])
try:
    print(m(jnp.array([1.0, 1.0]), jnp.array([1, 2, 3])))
except Exception as e:
    print(str(e)[:600])
