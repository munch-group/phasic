# NOTE (after the fix): this script documents the UNFIXED code and uses
# SVGD._compiled_cache / removed helpers, which no longer exist; it does not
# run on builds that include the fix.
"""Review 3: is 'second fit alone' bit-reproducible on unmodified code, and
would each proposed test fail on master?  (cache cleared manually = the fix)"""
import sys, random, gc, weakref, warnings
import numpy as np, jax
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug1/tests/pytest')
from test_notebook_model_recovery import coalescent_graph
import phasic
from phasic import Graph, GaussPrior, SVGD
print('phasic from', phasic.__file__)
KW = dict(n_particles=12, n_iterations=25, learning_rate=0.01, progress=False, seed=1)
def data(theta, n=120, seed=3):
    random.seed(seed); np.random.seed(seed)
    g = coalescent_graph(4); g.update_weights([theta]); return np.asarray(g.sample(n))
def mk():
    return Graph.pmf_and_moments_from_graph(coalescent_graph(4), nr_moments=2, theta_dim=1)
def fit(model, d, prior, opt_kw=None, **kw):
    k = dict(KW); k.update(kw)
    s = SVGD(model=model, observed_data=d, theta_dim=1, prior=prior, **k); s.optimize(**(opt_kw or {}))
    return np.asarray(s.particles).copy()
def cmp(a, b):
    return 'BITEQ' if np.array_equal(a, b) else f'maxabs {np.max(np.abs(a-b)):.3e}'
P = GaussPrior(ci=[0.5, 20], lower=0.0); P2 = GaussPrior(ci=[0.01, 0.2], lower=0.0)
Pu = GaussPrior(ci=[0.5, 20])
dA, dB = data(8.0), data(1.0)

def scenario(name, first, second):
    """first/second: callables model -> particles.  Reports:
       alone twice (same model, cache cleared), alone on a FRESH model object,
       after-first with cache cleared (= fixed behaviour), after-first as on master."""
    C = SVGD._compiled_cache
    m = mk(); C.clear(); a1 = second(m); C.clear(); a2 = second(m)
    m2 = mk(); C.clear(); a3 = second(m2)
    m3 = mk(); C.clear(); first(m3); C.clear(); f = second(m3)
    m4 = mk(); C.clear(); first(m4); bug = second(m4)
    print(f'{name:34s} alone-rerun {cmp(a1,a2):16s} fresh-model {cmp(a1,a3):16s} '
          f'after-first(cleared) {cmp(a1,f):16s} after-first(MASTER) {cmp(a1,bug)}', flush=True)

scenario('1 data', lambda m: fit(m, dA, P), lambda m: fit(m, dB, P))
scenario('2 prior', lambda m: fit(m, dA, P), lambda m: fit(m, dA, P2))
scenario('3 positive_params', lambda m: fit(m, dA, P), lambda m: fit(m, dA, Pu, positive_params=False))
scenario("3b softplus transform vs pos_params", lambda m: fit(m, dA, P), lambda m: fit(m, dA, Pu, positive_params=False, param_transform=lambda x: jax.nn.softplus(x)))
scenario('4 regularization data', lambda m: fit(m, dA, P, regularization=1.0), lambda m: fit(m, dB, P, regularization=1.0))
scenario('8 return_history=False data', lambda m: fit(m, dA, P, opt_kw=dict(return_history=False)), lambda m: fit(m, dB, P, opt_kw=dict(return_history=False)))
scenario('9 hist->nohist same data (ctl)', lambda m: fit(m, dB, P), lambda m: fit(m, dB, P, opt_kw=dict(return_history=False)))
scenario('10 fixed-only differs? n/a 1d; seed', lambda m: fit(m, dA, P, seed=5), lambda m: fit(m, dA, P, seed=1))
scenario('11 jit=False data (ctl)', lambda m: fit(m, dA, P, jit=False), lambda m: fit(m, dB, P, jit=False))
scenario('12 n_particles differs', lambda m: fit(m, dA, P, n_particles=20), lambda m: fit(m, dA, P))
for pm in ('vmap', 'none'):
    try:
        scenario(f'13 parallel={pm} data', lambda m: fit(m, dA, P, parallel=pm), lambda m: fit(m, dB, P, parallel=pm))
    except Exception as e:
        print('13', pm, 'ERROR', type(e).__name__, str(e)[:200])
# same INSTANCE, mutate and re-optimize
m = mk(); SVGD._compiled_cache.clear()
s = SVGD(model=m, observed_data=dA, theta_dim=1, prior=P, **KW); s.optimize(); p1 = np.asarray(s.particles).copy()
s.optimize(); p2 = np.asarray(s.particles).copy()
print('same instance optimize() twice:', cmp(p1, p2))
# Graph.svgd entry point twice on the same graph
g = coalescent_graph(4)
def gs(d):
    return np.asarray(g.svgd(d, prior=P, theta_dim=1, **KW).particles).copy()
try:
    SVGD._compiled_cache.clear(); x1 = gs(dB); SVGD._compiled_cache.clear(); gs(dA); x2 = gs(dB)
    print('Graph.svgd B alone vs B after A (cache NOT cleared):', cmp(x1, x2), ' entries', len(SVGD._compiled_cache))
except Exception as e:
    print('Graph.svgd ERROR', type(e).__name__, str(e)[:300])
# weakref (test 6), with cache cleared = fixed behaviour, several paths
def wr(label, **kw):
    SVGD._compiled_cache.clear()
    k = dict(KW); k.update(kw); ok = k.pop('opt_kw', {})
    s = SVGD(model=mk(), observed_data=dA, theta_dim=1, prior=P, **k); s.optimize(**ok)
    w = weakref.ref(s); del s; gc.collect(); held = w() is not None
    SVGD._compiled_cache.clear(); gc.collect()
    print(f'weakref {label:28s} alive with cache: {held}   alive after clear: {w() is not None}', flush=True)
wr('default'); wr('regularization', regularization=1.0); wr('no history', opt_kw=dict(return_history=False)); wr('jit=False', jit=False)
