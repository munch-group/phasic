"""Review 3 re-review of plan v2: feasibility of test 5 (mutate observed_data, optimize again)
and test 11 (no pickle.load under redirected HOME), on unmodified code with the cache
cleared before each fit (= fixed behaviour)."""
import os, sys, random, pickle, traceback, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug1/tests/pytest')
os.environ['HOME'] = '/Users/kmt/phasic-bug1/experiments/bug1_compiled_cache/review3/fakehome'
from test_notebook_model_recovery import coalescent_graph, coalescent_indexed
from test_epoch_normalization import normalised, p_n2_two_epoch
from phasic import Graph, GaussPrior, SVGD
calls = []
_orig = pickle.load
def rec(*a, **k):
    calls.append(''.join(traceback.format_stack(limit=4)[:-1]).count('_precompile_unified') and 'svgd-cache' or traceback.format_stack(limit=3)[0].strip().split('\n')[0])
    return _orig(*a, **k)
pickle.load = rec
KW = dict(n_particles=12, n_iterations=25, learning_rate=0.01, progress=False, seed=1)
def data(theta, n=120, seed=3):
    random.seed(seed); np.random.seed(seed)
    g = coalescent_graph(4); g.update_weights([theta]); return np.asarray(g.sample(n))
def mk(): return Graph.pmf_and_moments_from_graph(coalescent_graph(4), nr_moments=2, theta_dim=1)
def P(): return GaussPrior(ci=[0.5, 20], lower=0.0)
def cmp(a, b): return 'BITEQ' if np.array_equal(a, b) else f'maxabs {np.max(np.abs(a-b)):.3e}'
dA, dB = data(8.0), data(1.0)
C = SVGD._compiled_cache
for label, kw, okw in [('default', {}, {}), ('no history', {}, dict(return_history=False)), ('regularization', dict(regularization=1.0), {})]:
    C.clear(); r = SVGD(model=mk(), observed_data=dB, theta_dim=1, prior=P(), **KW, **kw); r.optimize(**okw); ref = np.asarray(r.particles).copy()
    for how in ('assign numpy', 'assign same type as stored'):
        C.clear(); s = SVGD(model=mk(), observed_data=dA, theta_dim=1, prior=P(), **KW, **kw); s.optimize(**okw)
        first = np.asarray(s.particles).copy()
        s.observed_data = dB if how == 'assign numpy' else type(s.observed_data)(dB) if not hasattr(s.observed_data, 'at') else __import__('jax').numpy.asarray(dB)
        C.clear()
        try:
            s.optimize(**okw); print(f'test5 [{label}; {how}] mutated-refit (cache cleared) vs fresh instance: {cmp(ref, np.asarray(s.particles))}; first vs ref {cmp(ref, first)}', flush=True)
        except Exception as e:
            print(f'test5 [{label}; {how}] ERROR {type(e).__name__}: {str(e)[:150]}', flush=True)
print('pickle.load calls during pdf-path fits:', sorted(set(calls)), len(calls)); calls.clear()
L = 3; EP = [0.0, 0.4]
gi, idx = coalescent_indexed(2)
J = gi.joint_prob_graph(idx, discrete=False, reward_limit=L)
p = normalised([p_n2_two_epoch(k, 2.0, 1.0, 6.0, 1.0, EP[1]) for k in range(L + 1)])
rows = [[k, 0] for k in range(L + 1) for _ in range(int(round(p[k] * 500)))]
for i in range(2):
    C.clear(); em = J.epoch_model(rows, epoch_starts=EP); em.fit(fixed=[(1, 1.0), (3, 1.0)], **KW)
    C.clear(); g = coalescent_graph(4); g.svgd(dA, prior=P(), theta_dim=1, **KW)
print('pickle.load calls during epoch (default prior) + Graph.svgd fits:', sorted(set(calls)), len(calls))
import glob; print('files under fake home:', sorted(os.path.relpath(x, os.environ['HOME']) for x in glob.glob(os.environ['HOME'] + '/**/*', recursive=True))[:12])
