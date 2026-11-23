import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug2/experiments/bug2_fixed_values/review2')
import numpy as np, jax, jax.numpy as jnp
import proto
from phasic import Graph
m = Graph.pmf_and_moments_from_graph(proto.graph3(), nr_moments=2, theta_dim=2)
t = jnp.array([0.2, 0.5, 1.0])
for th in ([1.9, 0.0], [1.9, 1e-12], [1.9, 0.3], [1.9, 5.0], [1.9, -1e-7]):
    try:
        p, mo = m(jnp.array(th), t); print(th, np.asarray(p), np.asarray(mo))
    except Exception as e: print(th, 'RAISED', type(e).__name__, str(e)[:200])
g = proto.graph3(); print('vertices', g.vertices_length())
print('grad at [1.9, 0.0]:')
try: print(np.asarray(jax.grad(lambda th: jnp.sum(jnp.log(m(th, t)[0])))(jnp.array([1.9, 0.0]))))
except Exception as e: print('RAISED', type(e).__name__, str(e)[:300])
print('grad at [1.9, 1e-12]:')
try: print(np.asarray(jax.grad(lambda th: jnp.sum(jnp.log(m(th, t)[0])))(jnp.array([1.9, 1e-12]))))
except Exception as e: print('RAISED', type(e).__name__, str(e)[:300])
