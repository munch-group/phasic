# Two-stage model E[T] = 1/t0 + 1/t1 (linear continuous, no rewards => NON-refusing).
# Compare d m1/d t1 at small t1: exact, today's clamp FD (master backward),
# plan v2.3 'unconstrained' rule on a non-refusing model (plain +-1e-7 / 2eps),
# and the plan's own capped-positive fallback (theta*e^{+-s}, s=1e-4).
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
from phasic import Graph
g = Graph(1); s = g.starting_vertex()
a = g.find_or_create_vertex([3]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([1])
s.add_edge(a, 1.0); a.add_edge(b, [1.0, 0.0]); b.add_edge(c, [0.0, 1.0])
m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2, exact_moment_grad=False)
t = jnp.array([1.0])
f = lambda th: float(m(jnp.array(th), t)[1][0])
for t1 in (9e-8, 5e-8, 1e-8):
    exact = -1/t1**2
    today = float(jax.grad(lambda p: m(p, t)[1][0])(jnp.array([1.0, t1]))[1])
    eps = 1e-7
    plain = (f([1, t1+eps]) - f([1, t1-eps]))/(2*eps)
    sst = 1e-4; up, dn = t1*np.exp(sst), t1*np.exp(-sst)
    capped = (f([1, up]) - f([1, dn]))/(up-dn)
    print(f"t1={t1:g} exact={exact:.4e} today_clamp={today:.4e} plan_unconstrained_plain={plain:.4e} plan_capped_fallback={capped:.4e}")
