"""P5: discrete model, positive-declared parameter near its implicit upper
bound (row sum 1). Full log rule s=1e-4 probes theta*e^s; today probes
theta+1e-7. What does the forward do when the probe crosses row sum 1?"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp
from phasic import Graph, set_log_level
jax.config.update("jax_enable_x64", True)
set_log_level("ERROR")
g = Graph(1)
s = g.starting_vertex()
a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([3])
s.add_edge(a, 1.0)
a.add_edge(b, [1.0])      # leave a with prob theta per step
b.add_edge(c, [0.5])      # leave b with prob theta/2
m = Graph.pmf_from_graph(g, discrete=True, theta_dim=1)
T = jnp.array([1.0, 2.0, 3.0, 5.0])
f = lambda th: np.asarray(m(jnp.asarray(th), T), np.float64)
for th0 in [0.5, 0.9, 0.99995, 0.9999999, 1.0]:
    th = np.array([th0])
    try: v = f(th); vp = f(th * np.exp(1e-4)); vt = f(th + 1e-7)
    except Exception as e: print(th0, 'ERR', type(e).__name__, str(e)[:120]); continue
    print(f"theta={th0}: pmf={v}  pmf(theta*e^1e-4)={vp}  pmf(theta+1e-7)={vt}", flush=True)
