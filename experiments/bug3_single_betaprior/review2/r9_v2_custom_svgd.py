"""v2 decision (b), SVGD rows redone with a FRESH model object per fit (r8 reused one model:
bug-1 cache hazard).  v2 wiring emulated: the prior's _transform is set to the custom function
BEFORE optimize, and SVGD._compiled_cache is cleared."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp
from phasic import BetaPrior, SVGD
d = np.array([0.5, 1.0, 1.5, 2.0])
def fresh():
    def model(theta, times, rewards=None):
        return jnp.exp(-times * jnp.sum(theta)) * jnp.sum(theta), jnp.zeros(2) + jnp.sum(theta)
    return model
for tname, T in [('exp', jnp.exp), ('jax.nn.sigmoid', jax.nn.sigmoid)]:
    for v2 in (False, True):
        for form, mk in [('single', lambda: BetaPrior(alpha=2.0, beta=5.0)), ('list', lambda: [BetaPrior(alpha=2.0, beta=5.0)])]:
            pr = mk(); p0 = pr if form == 'single' else pr[0]
            try:
                SVGD._compiled_cache.clear()
            except Exception as e:
                print('cache clear failed', e)
            init = np.linspace(-1.5, 0.8, 8).reshape(8, 1)      # theta = exp(init) spans 0.22 .. 2.2
            try:
                s = SVGD(model=fresh(), observed_data=d, theta_dim=1, prior=pr, n_particles=8, n_iterations=20, progress=False, seed=1,
                         learning_rate=0.05, verbose=False, positive_params=False, param_transform=T, preconditioner='none', theta_init=init)
                if v2: p0._transform = T
                lp = np.asarray([float(p0(jnp.asarray(init[i]))) for i in range(8)])
                s.optimize(); P = np.asarray(s.get_results()['particles'])[:, 0]
                print(f'{tname:15s} {form:6s} {"v2 (prior gets custom)" if v2 else "master (prior keeps sigmoid)"}: log-prior nan at init {int(np.isnan(lp).sum())}/8; '
                      f'final particles nan {int(np.isnan(P).sum())}/8; range [{np.nanmin(P):.3g}, {np.nanmax(P):.3g}]; >1: {int(np.sum(P > 1))}')
            except Exception as e:
                print(f'{tname} {form} v2={v2}: {type(e).__name__}: {str(e)[:200]}')
