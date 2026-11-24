"""(a) tied epoch fit under the wrapper: every exported surface slave == master?
(b) real graph through Graph.svgd with a rate fixed at 0 / 1e-12 (FD minus probe negative)"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug2/experiments/bug2_fixed_values/review2')
sys.path.insert(0, '/Users/kmt/phasic-bug2/tests/pytest')
import numpy as np, jax, jax.numpy as jnp
import proto
from phasic import Graph, GaussPrior, BetaPrior, SVGD
import test_svgd_tied_export as T
def surfaces(s, tag):
    s2m = s.model._tying_info['slave_to_master']
    r = s.get_results()
    for sl, ma in s2m.items():
        P = np.asarray(s.particles); Pc = np.asarray(r['particles']); H = np.asarray(r['history'])
        tm = np.asarray(r['theta_mean']); am = np.asarray(s.theta_mean)
        print(f'   {tag} slave {sl} <- master {ma}: phi-eq={np.array_equal(P[:,sl],P[:,ma])} theta-eq={np.array_equal(Pc[:,sl],Pc[:,ma])} '
              f'hist-eq={np.array_equal(H[...,sl],H[...,ma])} mean {tm[sl]!r} vs {tm[ma]!r}; attr {am[sl]!r} vs {am[ma]!r}')
        m1, _ = s.map_estimate_from_particles(); print(f'      map_from_particles slave={m1[sl]!r} master={m1[ma]!r}')
        try:
            m2, _ = s.map_estimate_with_optimization(n_steps=2); print(f'      map_with_optimization slave={m2[sl]!r} master={m2[ma]!r}')
        except Exception as e: print(f'      map_with_optimization RAISED {type(e).__name__}: {str(e)[:100]}')
        try: print(f'      log_likelihood={s.log_likelihood()!r} refine={s.log_likelihood(refine=True)!r}')
        except Exception as e: print(f'      log_likelihood RAISED {type(e).__name__}: {str(e)[:160]}')
    fx = [i for i in range(s.theta_dim) if s.fixed_mask[i] and i not in s2m]
    print(f'      user-fixed slots {fx}: mean={[float(np.asarray(r["theta_mean"])[i]) for i in fx]} (given 1e-4)')
print('## (a) tied epoch fit')
for patched in (False, True):
    for name, fn in [('homogeneous', lambda: T._fit_tied()), ('mixed Beta master', lambda: T._fit_tied(prior=[
            BetaPrior(ci=[0.2, 0.85]), GaussPrior(ci=[1e-6, 1e-2], lower=0.0), GaussPrior(ci=[1e-6, 1e-2], lower=0.0), GaussPrior(ci=[1e-6, 1e-2], lower=0.0)]))]:
        proto.clear()
        if patched: proto.install('plan')
        try:
            s = fn(); surfaces(s, f'patched={patched} {name}')
        except Exception as e:
            import traceback; print(f'   patched={patched} {name}: RAISED {type(e).__name__}: {str(e)[:300]}')
        finally: proto.uninstall()
print('## (b) Graph.svgd on a real graph, rate fixed at 0 / tiny (linear weights, FD backward)')
d = np.array([0.3, 0.5, 1.0, 1.5, 2.0, 0.8, 1.2, 0.6])
for v in (0.0, 1e-12, 1e-8, 0.3):
    for patched in (True,):
        proto.clear(); proto.install('plan')
        try:
            g = proto.graph3()
            s = g.svgd(d, theta_dim=2, fixed=[(1, v)], prior=[GaussPrior(ci=[0.5, 5], lower=0.0), None],
                       n_particles=6, n_iterations=4, progress=False, seed=1)
            r = s.get_results(); P = np.asarray(r['particles'])
            print(f'   fixed={v:g}: free finite={bool(np.all(np.isfinite(P[:,0])))} free mean={float(P[:,0].mean()):.4f} fixed mean={float(np.asarray(r["theta_mean"])[1])!r}')
        except Exception as e:
            print(f'   fixed={v:g}: RAISED {type(e).__name__}: {str(e)[:300]}')
        finally: proto.uninstall()
print('## (c) confirm the stale 0.3 in t4(a) is the compiled-gradient cache: clear between fits')
base = Graph.pmf_and_moments_from_graph(proto.graph3(), nr_moments=2, theta_dim=2)
for clear_between in (False, True):
    proto.clear(); spy, seen = proto.spy_model(base); marks = []
    for v in (0.3, 2.0):
        proto.install('plan')
        try:
            s = SVGD(model=spy, observed_data=d, theta_dim=2, fixed=[(1, v)], n_particles=6, n_iterations=3,
                     progress=False, seed=1, prior=[GaussPrior(ci=[0.5, 5], lower=0.0), None], preconditioner='none')
            n0 = len(seen); s.optimize(); marks.append(proto.slot_values(seen[n0:], 1))
        finally: proto.uninstall()
        if clear_between: SVGD._compiled_cache.clear()
    print(f'   clear_between={clear_between}: during optimize() the model saw slot1 = fit1 {marks[0]}, fit2 {marks[1]}')
