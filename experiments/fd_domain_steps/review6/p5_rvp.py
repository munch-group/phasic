import numpy as np, jax, jax.numpy as jnp
from phasic import Graph, set_log_level
set_log_level("ERROR")
exec(open('p1_modes.py').read().split("times = ")[0].split('set_log_level("ERROR")')[1])
# branching graph: a -> b (rate t0*.. ) or a -> c (rate t1..), reward on b only
def br(mode):
    g = Graph(1)
    s = g.starting_vertex(); a = g.find_or_create_vertex([3]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([1]); d = g.find_or_create_vertex([4])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0, 1.0]); a.add_edge(c, [2.0, 1.0]); b.add_edge(d, [1.0, 1.0])
    if mode == 'log': g.weight_mode = 'log'
    if mode == 'formula_sqrt': g.weight_formula = "c0*t0*sqrt(c1*t1)"
    return g
for mode in ('linear', 'log', 'formula_sqrt'):
    g = br(mode)
    r = np.zeros(g.vertices_length()); r[2] = 1.0
    f = jax.grad(lambda th: g.reward_visit_probability(r, th))
    for th in ([1.0, 1.0], [1.0, 5e-8]):
        try:
            print(mode, th, np.asarray(f(jnp.array(th, dtype=jnp.float64))), flush=True)
        except Exception as e:
            print(mode, th, "RAISES", str(e).replace(chr(10), ' ')[:160], flush=True)
