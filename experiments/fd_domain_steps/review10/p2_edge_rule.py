# Prototype of v2.5's edge-distance rule h_k = min(1e-7, s*d_k), d_k = min_e |w_e|/|c_ek|,
# on non-refusing linear continuous pmf_and_moments models (no rewards). Reference for
# d E[T]/dtheta: exact_moment_grad=True (analytic reverse mode). Also plain +-1e-7.
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
from phasic import Graph
S = 1e-4; EPS = 1e-7
T = jnp.array([1.0])
def build(spec, P):
    # spec: list of (from_state, to_state, coeffs); state 0 = absorbing; start -> state 9
    g = Graph(1); st = g.starting_vertex(); V = {}
    def v(s):
        if s not in V: V[s] = g.find_or_create_vertex([s])
        return V[s]
    st.add_edge(v(9), 1.0)
    for a, b, c in spec: v(a).add_edge(v(b), list(map(float, c)))
    ser = g.serialize(); pe = np.asarray(ser['param_edges'], float)
    C = pe[:, 2:2+P]
    fd = Graph.pmf_and_moments_from_graph(g, nr_moments=1, theta_dim=P, exact_moment_grad=False)
    ex = Graph.pmf_and_moments_from_graph(g, nr_moments=1, theta_dim=P, exact_moment_grad=True)
    return C, fd, ex
def m1(model, th): return float(model(jnp.array(th), T)[1][0])
def edge_rule(C, fd, th, k):
    w = C @ th; nz = C[:, k] != 0
    d = np.min(np.abs(w[nz]) / np.abs(C[nz, k]))
    h = min(EPS, S * d)
    tp = np.array(th); tm = np.array(th); tp[k] += h; tm[k] -= h
    den = (tp[k] - tm[k]) if h < EPS else 2 * EPS
    return (m1(fd, tp) - m1(fd, tm)) / den, h, d
def vertex_rule(C, fd, th, k, src):
    # alternative: distance of each vertex's TOTAL exit rate from 0
    w = C @ th; best = np.inf
    for v in set(src):
        idx = np.array(src) == v; R = w[idx].sum(); dc = C[idx, k].sum()
        if dc != 0: best = min(best, abs(R) / abs(dc))
    h = min(EPS, S * best)
    tp = np.array(th); tm = np.array(th); tp[k] += h; tm[k] -= h
    den = (tp[k] - tm[k]) if h < EPS else 2 * EPS
    return (m1(fd, tp) - m1(fd, tm)) / den
def plain(fd, th, k):
    tp = np.array(th); tm = np.array(th); tp[k] += EPS; tm[k] -= EPS
    return (m1(fd, tp) - m1(fd, tm)) / (2 * EPS)
cases = {
 'A pole (a->b = t1)':            ([(9, 1, [1, 0]), (1, 0, [0, 1])], 2, lambda x: [1.0, x], 1),
 'B offset one edge (t0+t1)':     ([(9, 1, [1, 1]), (1, 0, [2, 0])], 2, lambda x: [1.0, x], 1),
 'C small rate, not sole exit':   ([(9, 1, [1, 0]), (9, 2, [0, 1]), (1, 0, [2, 0]), (2, 0, [3, 0])], 2, lambda x: [1.0, x], 1),
 'D offset as parallel edges':    ([(9, 1, [1, 0]), (9, 1, [0, 1]), (1, 0, [2, 0])], 2, lambda x: [1.0, x], 1),
}
for name, (spec, P, thf, k) in cases.items():
    C, fd, ex = build(spec, P); src = [s[0] for s in spec]
    print("==", name)
    for x in (5e-8, 1e-9, 1e-11, 1e-13):
        th = np.array(thf(x)); ref = float(jax.grad(lambda p: ex(p, T)[1][0])(jnp.array(th))[k])
        er, h, d = edge_rule(C, fd, th, k); vr = vertex_rule(C, fd, th, k, src); pl = plain(fd, th, k)
        r = lambda g: abs(g - ref) / abs(ref)
        print(f"  t={x:.0e} ref={ref:+.6e} edge(h={h:.1e}) relerr={r(er):.2e}  vertex relerr={r(vr):.2e}  plain relerr={r(pl):.2e}")
print("== E edge-sum on one edge: a->b = t0+t1, t0=1.5e-7, t1=-1e-7; b->abs = t2=1")
C, fd, ex = build([(9, 1, [1, 1, 0]), (1, 0, [0, 0, 1])], 3); th = np.array([1.5e-7, -1e-7, 1.0])
ref = float(jax.grad(lambda p: ex(p, T)[1][0])(jnp.array(th))[1])
er, h, d = edge_rule(C, fd, th, 1); print(f"  ref={ref:+.6e} edge={er:+.6e} (h={h:.1e}) plain={plain(fd, th, 1):+.6e}")
print("== F vertex-sum pole across two edges: a->b = t0 (1), a->c = t1 (-1+5e-8); b,c->abs = t2")
spec = [(9, 1, [1, 0, 0]), (9, 2, [0, 1, 0]), (1, 0, [0, 0, 1]), (2, 0, [0, 0, 1])]
C, fd, ex = build(spec, 3); th = np.array([1.0, -1 + 5e-8, 1.0])
ref = float(jax.grad(lambda p: ex(p, T)[1][0])(jnp.array(th))[1])
er, h, d = edge_rule(C, fd, th, 1); vr = vertex_rule(C, fd, th, 1, [s[0] for s in spec])
print(f"  ref={ref:+.6e} edge={er:+.6e} (h={h:.1e}, d={d:.2e}) vertex={vr:+.6e} plain={plain(fd, th, 1):+.6e}")
