import numpy as np, jax, jax.numpy as jnp
from phasic import Graph, set_log_level
set_log_level("ERROR")
exec(open('p1_modes.py').read().split("times = ")[0].split('set_log_level("ERROR")')[1])

def run(label, f, th):
    try:
        print(f"{label:50s} {th} -> {np.asarray(f(jnp.array(th)))}", flush=True)
    except Exception as e:
        print(f"{label:50s} {th} -> RAISES {str(e).replace(chr(10),' ')[:150]}", flush=True)

jumps = jnp.array([4.0, 6.0, 9.0], dtype=jnp.float64)
# (a) discretize()'d (was_dph) linear graph: slot 2 is the discretization rate
gd = make('linear').discretize(1.0)
print("was_dph graph params:", gd.param_length() if hasattr(gd, 'param_length') else '?')
for builder in ('pfg', 'pam'):
    if builder == 'pfg':
        m = Graph.pmf_from_graph(gd, discrete=True, theta_dim=3)
        f = jax.grad(lambda x: jnp.sum(jnp.log(m(x, jumps))))
        fw = lambda x: m(x, jumps)
    else:
        m = Graph.pmf_and_moments_from_graph(gd, nr_moments=1, discrete=True, theta_dim=3)
        f = jax.grad(lambda x: jnp.sum(jnp.log(m(x, jumps)[0])))
        fw = lambda x: m(x, jumps)[0]
    for th in ([1.0, 1.0, 1.0], [1.0, 5e-8, 1.0], [1.0, 1.0, 5e-8], [1.0, 1.0, 0.0]):
        run(f"was_dph {builder} grad", f, th)
    run(f"was_dph {builder} fwd at neg slot1", fw, [1.0, -5e-8, 1.0])
    run(f"was_dph {builder} fwd at neg slot2", fw, [1.0, 1.0, -5e-8])

# (b) native DPH: a -> b with prob t0, b -> c with prob t1 (remainder stays)
def native():
    g = Graph(1)
    s = g.starting_vertex(); a = g.find_or_create_vertex([3]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([1])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0, 0.0]); b.add_edge(c, [0.0, 1.0])
    return g
gn = native()
mn = Graph.pmf_and_moments_from_graph(gn, nr_moments=1, discrete=True, theta_dim=2)
mp = Graph.pmf_from_graph(gn, discrete=True, theta_dim=2)
fn = jax.grad(lambda x: jnp.sum(jnp.log(mn(x, jumps)[0])))
fp = jax.grad(lambda x: jnp.sum(jnp.log(mp(x, jumps))))
for th in ([0.5, 0.5], [0.5, 5e-8], [0.5, 1 - 5e-8], [0.5, 1 - 1e-12], [0.5, 1.0]):
    run("native DPH pam grad", fn, th)
    run("native DPH pfg grad", fp, th)
