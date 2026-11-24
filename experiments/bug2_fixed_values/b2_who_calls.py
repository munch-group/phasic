import sys, warnings, traceback
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings('ignore')
from phasic import GaussPrior, SVGD
calls = []
def spy(theta, times, rewards=None):
    st = [f'{f.name}:{f.lineno}' for f in traceback.extract_stack() if 'phasic/svgd.py' in f.filename]
    def rec(t, st=st): calls.append((float(np.atleast_2d(np.asarray(t))[0, 1]), ' <- '.join(st[-4:])))
    jax.debug.callback(rec, theta)
    return jnp.ones_like(times), jnp.zeros(2)
d = np.array([0.5, 1.0, 1.5, 2.0])
s = SVGD(model=spy, observed_data=d, theta_dim=2, fixed=[(1, 2.0)], positive_params=False,
         prior=[GaussPrior(ci=[0.5, 5]), None], n_particles=4, n_iterations=1, progress=False, seed=1)
print('after __init__:', sorted(set(calls))); calls.clear()
s.optimize()
for v, st in sorted(set(calls)): print(f'  theta[1]={v:.6g}  via {st}')
