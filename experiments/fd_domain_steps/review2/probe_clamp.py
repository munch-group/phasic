"""Probe: today's min_theta clamp at pmf_and_moments_from_graph model_bwd
on a signed (offset) parameter. rate = t0 + t1; d logpdf/dt0 must equal
d logpdf/dt1. Also: bit-identity of 2*eps vs actual probe difference."""
import numpy as np
import jax, jax.numpy as jnp
from phasic import Graph, set_log_level
set_log_level("ERROR")

g = Graph(1)
s = g.starting_vertex()
v = g.find_or_create_vertex([2])
a = g.find_or_create_vertex([1])
s.add_edge(v, 1.0)
v.add_edge(a, [1.0, 1.0])
g.update_weights([6.0, -5.0])

model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, exact_moment_grad=False)
times = jnp.array([0.5, 1.0, 2.0])

def ll(th):
    pmf, mom = model(th, times)
    return jnp.sum(jnp.log(pmf))

for th in ([6.0, -5.0], [0.5, 0.5], [1.0 - 1e-9, 1e-9]):
    th = jnp.array(th)
    gr = jax.grad(ll)(th)
    r = float(th[0] + th[1])
    exact = float(np.sum(1.0 / r - np.asarray(times)))
    print("theta", np.asarray(th), "grad", np.asarray(gr), "exact", exact)

# divisor bit identity
for t in (1.0, 3.7, 1e-3, 123.456):
    print(t, repr((t + 1e-7) - (t - 1e-7)), repr(2 * 1e-7),
          ((t + 1e-7) - (t - 1e-7)) == 2e-7)

# which backward ran? evaluate forward at the clamp probe points
def f(th):
    pmf, _ = model(jnp.array(th), times)
    return float(jnp.sum(jnp.log(pmf)))
print("ll(6,-5)", f([6.0, -5.0]), "ll(6,1e-9)", f([6.0, 1e-9]),
      "ll(6,-5+1e-7)", f([6.0, -5.0 + 1e-7]), "ll(6,-5-1e-7)", f([6.0, -5.0 - 1e-7]))
print("symmetric FD t1:", (f([6.0, -5.0 + 1e-7]) - f([6.0, -5.0 - 1e-7])) / 2e-7)
print("clamp secant t1:", (f([6.0, -5.0 + 1e-7]) - f([6.0, 1e-9])) / ((-5.0 + 1e-7) - 1e-9))
