"""Bug 1 reproduction and scoping (run on unmodified code).
(a) same model object, same data shape, different DATA
(b) same model object and data, different PRIOR
(c) FreeEpochModel: full fit then nested fit (different fixed) vs nested alone
(d) does pickling the compiled gradient ever succeed (disk cache)?
(e) what the cache holds on to (memory)
(f) compile cost the cache saves
"""
import sys, random, time, pickle, gc, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import coalescent_graph, coalescent_indexed
from test_epoch_normalization import normalised, p_n2_two_epoch
import phasic
from phasic import Graph, GaussPrior, SVGD
print('phasic from', phasic.__file__)
KW = dict(n_particles=16, n_iterations=40, learning_rate=0.01, progress=False, seed=1)
def data(theta, n=200, seed=3):
    random.seed(seed); np.random.seed(seed)
    g = coalescent_graph(4); g.update_weights([theta]); return np.asarray(g.sample(n))
def fit(model, d, prior):
    s = SVGD(model=model, observed_data=d, theta_dim=1, prior=prior, **KW); s.optimize()
    return float(np.asarray(s.get_results()['theta_mean'])[0])
g = coalescent_graph(4)
model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=1)
P = GaussPrior(ci=[0.5, 20], lower=0.0)
dA, dB = data(8.0), data(1.0)
print('## (a) same model object, different data')
SVGD._compiled_cache.clear(); a = fit(model, dA, P); b_after_a = fit(model, dB, P)
SVGD._compiled_cache.clear(); b_alone = fit(model, dB, P)
print(f'  A: {a:.4f}   B after A: {b_after_a:.4f}   B alone: {b_alone:.4f}   cache entries {len(SVGD._compiled_cache)}')
print('## (b) same model and data, different prior')
P2 = GaussPrior(ci=[0.01, 0.2], lower=0.0)
SVGD._compiled_cache.clear(); a = fit(model, dA, P); b_after_a = fit(model, dA, P2)
SVGD._compiled_cache.clear(); b_alone = fit(model, dA, P2)
print(f'  prior1: {a:.4f}   prior2 after prior1: {b_after_a:.4f}   prior2 alone: {b_alone:.4f}')
print('## (c) FreeEpochModel: full then nested (fixed differs)')
L = 3; EP = [0.0, 0.4]
gi, idx = coalescent_indexed(2)
J = gi.joint_prob_graph(idx, discrete=False, reward_limit=L)
p = normalised([p_n2_two_epoch(k, 2.0, 1.0, 6.0, 1.0, EP[1]) for k in range(L + 1)])
rows = [[k, 0] for k in range(L + 1) for _ in range(int(round(p[k] * 5000)))]
pri = [GaussPrior(ci=[0.5, 20], lower=0.0), None] * 2
def epoch_fits(order):
    SVGD._compiled_cache.clear()
    em = J.epoch_model(rows, epoch_starts=EP)
    out = {}
    for name in order:
        if name == 'full':     # mu fixed in both epochs (flat slots 1, 3)
            s = em.fit(fixed=[(1, 1.0), (3, 1.0)], prior=pri, **KW)
        else:                  # nested: epoch-2 coalescence (flat slot 2) also fixed, at 0.5
            s = em.fit(fixed=[(1, 1.0), (2, 0.5), (3, 1.0)], prior=pri, **KW)
        out[name] = np.asarray(s.get_results()['theta_mean']).round(4)
    return out
try:
    r1 = epoch_fits(['full', 'nested']); r2 = epoch_fits(['nested'])
    print(f"  full: {r1['full']}\n  nested after full: {r1['nested']}\n  nested alone:      {r2['nested']}")
except Exception as e:
    print('  ERROR', type(e).__name__, str(e)[:300])
print('## (d) pickling the compiled gradient')
SVGD._compiled_cache.clear(); fit(model, dA, P)
cg = list(SVGD._compiled_cache.values())[0]['grad']
try:
    n = len(pickle.dumps({'grad': cg})); print(f'  pickle SUCCEEDED ({n} bytes)')
except Exception as e:
    print(f'  pickle fails: {type(e).__name__}: {str(e)[:120]}')
print('## (e) what one cache entry keeps alive')
import weakref
SVGD._compiled_cache.clear()
s = SVGD(model=model, observed_data=dA, theta_dim=1, prior=P, **KW); s.optimize()
w = weakref.ref(s); del s; gc.collect()
print(f'  SVGD instance alive after del + gc with cache entry: {w() is not None}')
SVGD._compiled_cache.clear(); gc.collect()
print(f'  ... after clearing the cache: {w() is not None}')
print('## (f) compile cost saved by a cache hit (pdf model, 200 obs)')
SVGD._compiled_cache.clear()
t = time.time(); fit(model, dA, P); t1 = time.time() - t
t = time.time(); fit(model, dA, P); t2 = time.time() - t
print(f'  first fit {t1:.2f}s, second fit on the same model (cache hit) {t2:.2f}s')
