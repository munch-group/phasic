"""Review 1: fixed parameter that is also the exposure-scaled one (SVGD-level wrapper)."""
import warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings('ignore')
from phasic import GaussPrior, SVGD
d = np.array([0.5, 1.0, 1.5, 2.0]); alpha = np.array([1.0, 10.0, 100.0, 1000.0])
def run(v, emulate=False):
    seen = []
    def spy(theta, times, rewards=None):
        jax.debug.callback(lambda t: seen.append(float(np.asarray(t).reshape(-1)[1])), theta)
        return jnp.ones_like(times), jnp.zeros(2)
    s = SVGD(model=spy, observed_data=d, theta_dim=2, fixed=[(1, v)], exposure=alpha, exposure_param_index=1,
             prior=[GaussPrior(ci=[0.5, 5], lower=0.0), None], n_particles=4, n_iterations=1, progress=False, seed=1, preconditioner='none')  # naive wrapper crashes the default preconditioner (see r3_precond.log)
    if emulate:   # plan: model-facing transform overwrites the fixed slot with the exact value
        base = s.param_transform
        s.param_transform = lambda phi: jnp.where(jnp.array([False, True]), jnp.array([0.0, v]), base(phi))
    seen.clear(); s.optimize()
    got = sorted(set(seen)); want = sorted(float(a * v) for a in alpha)
    print(f'  v={v:g} {"EMULATED" if emulate else "today   "}: model saw {got}\n      want alpha*v = {want}  exact: {got == want}')
for v in (1e-10, 1e-8, 0.0, 2.0):
    run(v); run(v, emulate=True)
