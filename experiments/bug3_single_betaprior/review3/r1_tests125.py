"""Review 3: minimal versions of plan tests 1, 2, 5 on the UNMODIFIED build.
Fresh model object per fit."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp
import phasic
from phasic import GaussPrior, BetaPrior, SVGD
from phasic.mcmc import MCMC
from phasic.svgd import _sigmoid
d = np.array([0.5, 1.0, 1.5, 2.0])
def make_spy():
    seen = []
    def spy(theta, times, rewards=None):
        jax.debug.callback(lambda t: seen.append(np.atleast_2d(np.asarray(t)).copy()), theta)
        return jnp.ones_like(times), jnp.zeros(2)
    return spy, seen
def beta(): return BetaPrior(ci=(0.05, 0.95))
def fit(prior, dim, **kw):
    spy, seen = make_spy()
    s = SVGD(model=spy, observed_data=d, theta_dim=dim, prior=prior, n_particles=8, n_iterations=3,
             progress=False, seed=1, learning_rate=0.5, verbose=False, **kw)
    init = np.asarray(s.theta_init).copy()
    s.optimize()
    return s, np.concatenate(seen), init

print('## TEST 1: single BetaPrior, model input in [1e-9,1-1e-9]; param_transform == prior._transform')
for dim in (1, 2):
    pr = beta(); s, seen, init = fit(pr, dim)
    in_range = bool(np.all((seen >= 1e-9) & (seen <= 1 - 1e-9)))
    grid = jnp.linspace(-8, 8, 9)
    agree = all(np.allclose(np.asarray(s.param_transform(jnp.full((dim,), g))), np.asarray(pr._transform(jnp.full((dim,), g)))) for g in grid)
    print(f'  dim {dim}: model-input-in-range={in_range} (max {seen.max():.4g}); transform==prior transform: {agree}  -> test {"PASSES" if in_range and agree else "FAILS"} today')

print('## TEST 2: single vs list form: initial particles and final particles array-equal?')
for dim, mk in ((1, lambda: [beta()]), (2, lambda: [beta(), beta()])):
    s1, _, i1 = fit(beta(), dim); s2, _, i2 = fit(mk(), dim)
    print(f'  dim {dim}: theta_init equal: {np.array_equal(i1, i2)}; max|diff| init {np.abs(i1 - i2).max():.4g}')
    print(f'     single init[:3] {i1[:3].ravel().round(4).tolist()}  list init[:3] {i2[:3].ravel().round(4).tolist()}')
    print(f'     final particles equal: {np.array_equal(np.asarray(s1.particles), np.asarray(s2.particles))}')
# raw prior.sample comparison independent of SVGD: same key, (n,dim) vs per-slot split
key = jax.random.PRNGKey(1)
p = beta(); p._transform = _sigmoid
a = np.asarray(p.sample(key, (8, 1)))
k2, sub = jax.random.split(key); b = np.asarray(p.sample(sub, (8, 1)))
print(f'  prior.sample(key,(8,1)) vs prior.sample(split(key)[1],(8,1)) equal: {np.array_equal(a, b)}')

print('## TEST 5: BetaPrior + positive_params=False raises?')
def tryit(label, f):
    try: f(); print(f'  {label}: NO RAISE  -> test FAILS today')
    except ValueError as e: print(f'  {label}: ValueError: {str(e)[:110]}')
    except Exception as e: print(f'  {label}: {type(e).__name__}: {str(e)[:110]}')
for label, mk in (('single', beta), ('list', lambda: [beta()])):
    for tl, tkw in (('no transform', {}), ('custom exp', dict(param_transform=jnp.exp)), ('custom sigmoid', dict(param_transform=_sigmoid))):
        tryit(f'SVGD {label}, {tl}', lambda: fit(mk(), 1, positive_params=False, **tkw))
        tryit(f'MCMC {label}, {tl}', lambda: MCMC(log_prob_fn=lambda t: 0.0 * jnp.sum(t), prior=mk(), theta_dim=1, n_samples=10,
              n_chains=2, burn_in=2, positive_params=False, verbose=False, progress=False, seed=0, **tkw))
