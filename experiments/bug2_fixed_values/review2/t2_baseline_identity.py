"""(a) is map_estimate_with_optimization broken on the unmodified build?
(b) plan test 7: free-slot particles bit-identical before/after for fixed=2.0?
(c) theta_mean of n identical fixed values: bitwise equal to v?"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug2/experiments/bug2_fixed_values/review2')
import numpy as np, jax, jax.numpy as jnp
import proto
from phasic import Graph, GaussPrior, BetaPrior, SVGD
d = np.array([0.3, 0.5, 1.0, 1.5, 2.0, 0.8, 1.2, 0.6])
def fit(patched, v=2.0, n_particles=6, **kw):
    if patched: proto.install('plan')
    proto.clear()
    base = Graph.pmf_and_moments_from_graph(proto.graph3(), nr_moments=2, theta_dim=2)
    spy, seen = proto.spy_model(base)
    kw.setdefault('prior', [GaussPrior(ci=[0.5, 5], lower=0.0), None])
    try:
        s = SVGD(model=spy, observed_data=d, theta_dim=2, fixed=[(1, v)], n_particles=n_particles,
                 n_iterations=5, progress=False, seed=1, **kw)
        s.optimize(return_history=True)
        return s
    finally:
        proto.uninstall()
print('## (a) unmodified build: map_estimate_with_optimization')
s0 = fit(False)
try: print('   ', s0.map_estimate_with_optimization(n_steps=3))
except Exception as e: print('    RAISED on master:', type(e).__name__, str(e)[:120])
print('## (b) free-slot bit identity, fixed=2.0')
for label, kw in [('prior list, jacobian', {}), ('prior list, none', dict(preconditioner='none')),
                  ('single prior', dict(prior=GaussPrior(ci=[0.5, 5], lower=0.0))),
                  ('fisher', dict(preconditioner='fisher')),
                  ('regularized', dict(regularization=1.0, nr_moments=2))]:
    a = np.asarray(fit(False, **kw).particles)[:, 0]; b = np.asarray(fit(True, **kw).particles)[:, 0]
    print(f'    {label}: identical={np.array_equal(a, b)} maxdiff={np.max(np.abs(a-b)):.3e}')
print('## (c) reported mean/std of a fixed slot (patched), by n_particles')
for v in (1e-12, 0.3, 0.1, 1e-9, 800.0, 1e6):
    for n in (6, 7, 10, 40):
        s = fit(True, v=v, n_particles=n)
        r = s.get_results()
        m = float(np.asarray(r['theta_mean'])[1]); sd = float(np.asarray(r['theta_std'])[1])
        am = float(np.asarray(s.theta_mean)[1]); asd = float(np.asarray(s.theta_std)[1])
        if m != v or sd != 0 or am != v or asd != 0:
            print(f'    v={v:g} n={n}: get_results mean={m!r} std={sd!r}; attr mean={am!r} std={asd!r}')
print('    (only mismatches printed)')
