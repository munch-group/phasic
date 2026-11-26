# G: absorption-rate pole inside a cycle: a->b = t0 (1), b->a = t0, b->abs = t1 (near 0).
# Vertex totals stay ~1 (vertex rule blind); edge b->abs = t1 is the pole (edge rule sees it).
import sys; sys.argv = ['x']
exec(open('/Users/kmt/phasic/experiments/fd_domain_steps/review10/p2_edge_rule.py').read().split("cases = {")[0])
spec = [(9, 1, [1, 0]), (1, 9, [1, 0]), (1, 0, [0, 1])]
C, fd, ex = build(spec, 2); src = [s[0] for s in spec]
for x in (5e-8, 1e-9):
    th = np.array([1.0, x]); ref = float(jax.grad(lambda p: ex(p, T)[1][0])(jnp.array(th))[1])
    er, h, d = edge_rule(C, fd, th, 1); vr = vertex_rule(C, fd, th, 1, src); pl = plain(fd, th, 1)
    print(f"G t1={x:.0e} ref={ref:+.6e} edge={er:+.6e} vertex={vr:+.6e} plain={pl:+.6e}")
