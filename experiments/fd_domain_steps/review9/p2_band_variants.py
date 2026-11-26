# Compare band rules for an 'unconstrained' slot in (0,1e-7]:
#  A = v2.4 capped log step u = min(log1p(1e-7/th), 1e-4)  (= 1e-4 in band)
#  B = candidate u = min(log1p(1e-7/th), max(1e-4, 1e-10/th))  (absolute half-step >= ~1e-10)
# on (i) the pole model (rate = t1 alone; m1 = 1/t0+1/t1, m2 has 1/t1^2 terms) and
# (ii) the signed-offset model (rate = t0 + t1). Reference: exact for the pole m1;
# otherwise a wide (1e-4) central step on the smooth offset forward, and for the pole m2
# a tiny relative central step 1e-6*th (pole forward smooth on that scale).
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
from phasic import Graph
def build(offset):
    g = Graph(1); s = g.starting_vertex()
    a = g.find_or_create_vertex([3]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([1])
    s.add_edge(a, 1.0)
    if offset: a.add_edge(b, [1.0, 1.0]); b.add_edge(c, [2.0, 0.0])
    else:      a.add_edge(b, [1.0, 0.0]); b.add_edge(c, [0.0, 1.0])
    return Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2, exact_moment_grad=False)
T = jnp.array([1.0])
def F(m, th, k):
    o = m(jnp.array(th), T)
    return float(o[0][0]) if k == 'pdf' else float(o[1][k-1])
def logq(m, th1, k, u):
    up, dn = th1*np.exp(u), th1*np.exp(-u)
    return (F(m, [1.0, up], k) - F(m, [1.0, dn], k))/(up - dn)
for name, offset, outs in (('pole', False, (1, 2)), ('offset', True, ('pdf', 1, 2))):
    m = build(offset)
    print("model", name)
    for k in outs:
        for t1 in (9e-8, 5e-8, 1e-9, 1e-11, 1e-13):
            if offset:
                H = 1e-4; ref = (F(m, [1, t1+H], k) - F(m, [1, t1-H], k))/(2*H)
            elif k == 1:
                ref = -1/t1**2
            else:
                h = 1e-6*t1; ref = (F(m, [1, t1+h], k) - F(m, [1, t1-h], k))/(2*h)
            uA = min(np.log1p(1e-7/t1), 1e-4)
            uB = min(np.log1p(1e-7/t1), max(1e-4, 1e-10/t1))
            A = logq(m, t1, k, uA); B = logq(m, t1, k, uB)
            print(f"  out={k} t1={t1:.0e} ref={ref:+.5e}  A(v2.4)={A:+.5e} relerr={abs(A-ref)/abs(ref):.1e}  B(uB={uB:.2g})={B:+.5e} relerr={abs(B-ref)/abs(ref):.1e}")
