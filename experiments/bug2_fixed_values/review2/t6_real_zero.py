"""Graph.svgd on a real graph with a rate fixed at 0 / tiny: does the fit respond to the fixed value,
and does a fixed 0 survive the FD backward's minus probe?"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug2/experiments/bug2_fixed_values/review2')
import numpy as np, jax, jax.numpy as jnp
import proto
from phasic import Graph, GaussPrior, SVGD
rng = np.random.default_rng(3)
d = rng.exponential(0.5, 60)
def run(v, patched, weight=None, **kw):
    proto.clear()
    if patched: proto.install('plan')
    try:
        g = proto.graph3()
        s = g.svgd(d, theta_dim=2, fixed=[(1, v)], prior=[GaussPrior(ci=[0.5, 5], lower=0.0), None],
                   n_particles=8, n_iterations=30, learning_rate=0.05, progress=False, seed=1, **kw)
        r = s.get_results(); P = np.asarray(r['particles'])
        return f'free finite={bool(np.all(np.isfinite(P[:,0])))} free mean={float(P[:,0].mean()):.10f} fixed={float(P[0,1])!r} ll={s.log_likelihood():.6f}'
    except Exception as e:
        return f'RAISED {type(e).__name__}: {str(e)[:300]}'
    finally: proto.uninstall()
for v in (0.0, 1e-12, 1e-8, 0.3, 5.0):
    print(f'  v={v:g} patched : {run(v, True)}')
    print(f'  v={v:g} master  : {run(v, False)}')
print('## preconditioner none')
for v in (0.0, 0.3):
    print(f'  v={v:g} patched : {run(v, True, preconditioner="none")}')
