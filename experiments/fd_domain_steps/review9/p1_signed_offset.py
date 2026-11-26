# Signed offset slot: rate a->b = t0 + t1 (t1 a signed offset, forward smooth through t1=0),
# rate b->c = 2*t0. Linear continuous, no rewards => NON-refusing. Under v2.4 an
# 'unconstrained' t1 in [0,1e-7] gets the capped LOG step (theta*e^{+-1e-4}); outside,
# plain +-1e-7 / 2eps. Compare the d/dt1 of the forward's own outputs (pdf at t=1, E[T])
# against a smooth wide-step reference of the same forward.
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
from phasic import Graph
g = Graph(1); s = g.starting_vertex()
a = g.find_or_create_vertex([3]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([1])
s.add_edge(a, 1.0); a.add_edge(b, [1.0, 1.0]); b.add_edge(c, [2.0, 0.0])
m = Graph.pmf_and_moments_from_graph(g, nr_moments=1, theta_dim=2, exact_moment_grad=False)
T = jnp.array([1.0])
def f(th, k):
    out = m(jnp.array(th), T)
    return float(out[0][0]) if k == 'pdf' else float(out[1][0])
t0 = 1.0
for k in ('pdf', 'm1'):
    print("output", k)
    for t1 in (1e-7, 5e-8, 1e-9, 1e-11, 1e-13, 1e-15, 0.0):
        H = 1e-4
        ref = (f([t0, t1+H], k) - f([t0, t1-H], k))/(2*H)
        exact_m1 = -1/(t0+t1)**2
        eps = 1e-7
        plain = (f([t0, t1+eps], k) - f([t0, t1-eps], k))/(2*eps)
        if t1 > 0:
            up, dn = t1*np.exp(1e-4), t1*np.exp(-1e-4)
            fb = (f([t0, up], k) - f([t0, dn], k))/(up-dn)
        else:
            fb = (f([t0, eps], k) - f([t0, 0.0], k))/eps  # one-sided §4.5
        today = float(jax.grad(lambda p: m(p, T)[0 if k=='pdf' else 1][0])(jnp.array([t0, t1]))[1])
        print(f"  t1={t1:7.0e} ref={ref:+.10e} {'exact='+format(exact_m1,'+.10e') if k=='m1' else ''}"
              f"\n     plain_rel_err={abs(plain-ref)/abs(ref):.2e} v2.4_band_rel_err={abs(fb-ref)/abs(ref):.2e} (fb={fb:+.6e}) today_rel_err={abs(today-ref)/abs(ref):.2e}")
