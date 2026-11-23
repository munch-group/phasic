"""Review 1: once a fixed value of exactly 0 reaches the model (it never does today), what do the
shipped model builders do at theta = [1.5, 0.0]?  (competing edge a->c with rate theta1 = 0)"""
import warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings('ignore')
from phasic import Graph
def g2():
    g = Graph(1); s = g.starting_vertex()
    a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0, 0.0]); a.add_edge(c, [0.0, 1.0]); b.add_edge(c, [2.0, 0.0]); return g
t = jnp.array([0.5, 1.0, 1.5])
def try_(label, build, **kw):
    try:
        m = build()
        f = lambda th: jnp.sum(jnp.log(m(th, t)[0] if isinstance(m(th, t), tuple) else m(th, t)))
        for th in ([1.5, 0.3], [1.5, 0.0]):
            v = f(jnp.array(th)); gr = jax.grad(f)(jnp.array(th))
            print(f'  {label} theta={th}: loglik={float(v):.6f} grad={np.asarray(gr)}')
    except Exception as e:
        print(f'  {label}: {type(e).__name__}: {str(e)[:200]}')
for emg in (True, False):
    try_(f'pmf_and_moments_from_graph exact_moment_grad={emg}, no fixed_mask',
         lambda: Graph.pmf_and_moments_from_graph(g2(), nr_moments=2, theta_dim=2, exact_moment_grad=emg))
    try_(f'pmf_and_moments_from_graph exact_moment_grad={emg}, fixed_mask=[0,1]',
         lambda: Graph.pmf_and_moments_from_graph(g2(), nr_moments=2, theta_dim=2, exact_moment_grad=emg, fixed_mask=jnp.array([0, 1])))
try_('pmf_from_graph', lambda: Graph.pmf_from_graph(g2(), theta_dim=2))
