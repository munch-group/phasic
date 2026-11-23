"""Plan test 7 on the Graph.svgd path with an identified model: does the placeholder leak into free slots?"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug2/experiments/bug2_fixed_values/review2')
import numpy as np, jax, jax.numpy as jnp
import proto
from phasic import Graph, GaussPrior, SVGD
rng = np.random.default_rng(3)
d = rng.exponential(0.5, 60)
def run(v, patched, n_it=30, placeholder=0.0, **kw):
    proto.clear(); proto.PLACEHOLDER = placeholder
    if patched: proto.install('plan')
    try:
        g = proto.graph3()
        s = g.svgd(d, theta_dim=2, fixed=[(1, v)], prior=[GaussPrior(ci=[0.5, 5], lower=0.0), None],
                   n_particles=8, n_iterations=n_it, learning_rate=0.05, progress=False, seed=1, return_history=True, **kw)
        return np.asarray(s.get_results()['particles'])[:, 0], np.asarray(s.history)
    finally: proto.uninstall(); proto.PLACEHOLDER = 0.0
sp = lambda x: float(np.maximum(np.logaddexp(0, np.log(np.expm1(x))), 1e-9))
for v in (2.0, 5.0, 0.3):
    seen_by_master = sp(v)
    a, ha = run(v, False); b, hb = run(v, True); c, hc = run(seen_by_master, True); b2, _ = run(v, True, placeholder=0.7)
    print(f'v={v}: master delivers {seen_by_master!r}')
    print(f'   master vs patched(v)               : identical={np.array_equal(a,b)} maxdiff={np.max(np.abs(a-b)):.3e}')
    print(f'   master vs patched(value master saw): identical={np.array_equal(a,c)} maxdiff={np.max(np.abs(a-c)):.3e}')
    print(f'   patched placeholder 0.0 vs 0.7     : identical={np.array_equal(b,b2)} maxdiff={np.max(np.abs(b-b2)):.3e}')
    k = min(len(ha), len(hc))
    dv = [float(np.max(np.abs(ha[i][:, 0] - hc[i][:, 0]))) for i in range(k)]
    print(f'   first history step where master and patched(value master saw) differ in the free column: {next((i for i, x in enumerate(dv) if x > 0), None)}; diffs first 6: {["%.1e" % x for x in dv[:6]]}')
for kw in (dict(preconditioner='none'),):
    a, _ = run(5.0, False, **kw); c, _ = run(sp(5.0), True, **kw)
    print(f'v=5 {kw}: master vs patched(value master saw) identical={np.array_equal(a,c)} maxdiff={np.max(np.abs(a-c)):.3e}')
