"""v2 checks on the UNMODIFIED build: (a) map_estimate_with_optimization under a transform; (c) plan-v2 test 10 (same-build equivalence)."""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(__file__))
from r2_lib import *
print('## (a) map_estimate_with_optimization, default positive_params')
s, _ = fit(2.0, False, n_iter=2)
try: print('  returned', s.map_estimate_with_optimization(n_steps=3))
except Exception as e: print('  RAISED', type(e).__name__, str(e)[:160])
s2 = SVGD(model=full_model(), observed_data=data, theta_dim=2, fixed=[(1, 2.0)], positive_params=False,
          prior=[GaussPrior(ci=[0.5, 5]), None], n_particles=8, n_iterations=2, progress=False, seed=1); s2.optimize()
try: print('  no transform: returned', s2.map_estimate_with_optimization(n_steps=3))
except Exception as e: print('  no transform: RAISED', type(e).__name__, str(e)[:160])
print('## (c) test 10: fit A fixed=(1,v) on the real model vs fit B on a model hard-coding v (B fixed at 7.0)')
def hard(v):
    m = full_model()
    def model(theta, times, rewards=None):
        return m(jnp.stack([theta[0], jnp.asarray(v, dtype=theta.dtype)]), times)
    for a in ('_precondition_output',):
        if hasattr(m, a): setattr(model, a, getattr(m, a))
    return model
for v in (2.0, 0.3, 5.0):
    try:
        _, pa = fit(v, False)
        _, pb = fit(7.0, False, model=hard(v))
        d = np.abs(pa[:, 0] - pb[:, 0])
        print(f'  v={v:g}: free slots array-equal={bool(np.all(pa[:, 0] == pb[:, 0]))}  max abs diff={d.max():.3e}')
    except Exception as e:
        print(f'  v={v:g}: RAISED {type(e).__name__}: {str(e)[:300]}')
