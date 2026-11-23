import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug2/experiments/bug2_fixed_values/review2')
import numpy as np, jax, jax.numpy as jnp
import proto
from phasic import Graph
m = Graph.pmf_and_moments_from_graph(proto.graph3(), nr_moments=2, theta_dim=2)
t = jnp.array([0.2, 0.5, 1.0])
for th in ([1.9, 0.0], [1.9, -0.0]):
    try:
        p, mo = m(jnp.array(th), t); print(repr(th), np.asarray(p), np.asarray(mo))
        print('   grad', np.asarray(jax.grad(lambda x: jnp.sum(jnp.log(m(x, t)[0])))(jnp.array(th))))
    except Exception as e: print(repr(th), 'RAISED', type(e).__name__, str(e)[:200])
# two(): the plan's b1 graph, where theta1 = 0 leaves vertex b with no exit
def two():
    g = Graph(1); s = g.starting_vertex()
    a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0, 0.0]); b.add_edge(c, [0.0, 1.0]); return g
m2 = Graph.pmf_and_moments_from_graph(two(), nr_moments=2, theta_dim=2)
for th in ([1.9, 0.0], [1.9, 1e-12]):
    try:
        p, mo = m2(jnp.array(th), t); print('two()', th, np.asarray(p), np.asarray(mo))
    except Exception as e: print('two()', th, 'RAISED', type(e).__name__, str(e)[:200])
