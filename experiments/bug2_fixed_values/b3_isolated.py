import sys, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings('ignore')
from phasic import GaussPrior, SVGD
d = np.array([0.5, 1.0, 1.5, 2.0])
def run(label, pre=None, **kw):
    seen = []
    def spy(theta, times, rewards=None):
        jax.debug.callback(lambda t: seen.append(float(np.atleast_2d(np.asarray(t))[0, 1])), theta)
        return jnp.ones_like(times), jnp.zeros(2)
    if pre: pre()
    s = SVGD(model=spy, observed_data=d, theta_dim=2, n_particles=4, n_iterations=1, progress=False, seed=1, **kw)
    s.optimize(); print(f'  {label}: model saw {sorted(set(seen))}')
print('## isolated: positive_params=False, fixed=(1, 2.0)')
run('alone', fixed=[(1, 2.0)], positive_params=False, prior=[GaussPrior(ci=[0.5, 5]), None])
print('## same call AFTER a positive_params=True fit that used the same prior-free config')
P = [GaussPrior(ci=[0.5, 5]), None]
def pre():
    SVGD(model=lambda t, x, rewards=None: (jnp.ones_like(x), jnp.zeros(2)), observed_data=d, theta_dim=2,
         prior=[GaussPrior(ci=[0.5, 5], lower=0.0), None], fixed=[(1, 0.3)], n_particles=4, n_iterations=1, progress=False, seed=1).optimize()
run('after another fit (fresh spy, fresh priors)', pre=pre, fixed=[(1, 2.0)], positive_params=False, prior=[GaussPrior(ci=[0.5, 5]), None])
