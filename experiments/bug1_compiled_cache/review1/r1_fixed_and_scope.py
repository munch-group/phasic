"""Adversarial review 1 of bug1-compiled-cache-plan.md (unmodified code).
HOME is redirected into this directory so nothing touches the real ~/.phasic_cache.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
FAKE_HOME = os.path.join(HERE, 'fakehome'); os.makedirs(FAKE_HOME, exist_ok=True)
os.environ['HOME'] = FAKE_HOME
import random, warnings, gc, glob
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug1/tests/pytest')
from test_notebook_model_recovery import coalescent_graph, coalescent_indexed
from test_epoch_normalization import normalised, p_n2_two_epoch
import phasic
from phasic import Graph, GaussPrior, SVGD
print('phasic from', phasic.__file__)
import pathlib; print('Path.home() =', pathlib.Path.home())
KW = dict(n_particles=16, n_iterations=40, learning_rate=0.01, progress=False, seed=1)
def P(s): return np.asarray(s.particles)
def LP(s):
    h = getattr(s, 'history_log_prob', None)
    return None if h is None else np.asarray(h)
def cmp(tag, a, b):
    d = float(np.max(np.abs(P(a) - P(b))))
    print(f'  {tag}: max|particle diff| = {d:.3e}  mean after={np.asarray(a.get_results()["theta_mean"]).round(4)} alone={np.asarray(b.get_results()["theta_mean"]).round(4)}')
    la, lb = LP(a), LP(b)
    if la is not None and lb is not None:
        dd = la - lb
        print(f'     history_log_prob diff: min {np.nanmin(dd):.6g} max {np.nanmax(dd):.6g}')

L = 3; EP = [0.0, 0.4]
gi, idx = coalescent_indexed(2)
J = gi.joint_prob_graph(idx, discrete=False, reward_limit=L)
p = normalised([p_n2_two_epoch(k, 2.0, 1.0, 6.0, 1.0, EP[1]) for k in range(L + 1)])
rows = [[k, 0] for k in range(L + 1) for _ in range(int(round(p[k] * 5000)))]
G = lambda: GaussPrior(ci=[0.5, 20], lower=0.0)
pri = [G(), None, G(), None]
FULL = [(1, 1.0), (3, 1.0)]; NEST = [(1, 1.0), (2, 0.5), (3, 1.0)]

def seq(fits, prior=pri):
    """Fresh FreeEpochModel + cleared cache, then the given list of fixed= in order."""
    SVGD._compiled_cache.clear()
    em = J.epoch_model(rows, epoch_starts=EP)
    return [em.fit(fixed=f, prior=prior, **KW) for f in fits]

print('## E0 determinism: nested alone twice (fresh model each)')
n1, = seq([NEST]); n2, = seq([NEST]); cmp('nested vs nested', n1, n2)
print('## E1 plan row (c): full -> nested (explicit prior)')
f, n = seq([FULL, NEST]); cmp('nested after full', n, n1)
print('## E2 REVERSE order: nested -> full (explicit prior, same list)')
f_alone, = seq([FULL]); n, f = seq([NEST, FULL]); cmp('full after nested', f, f_alone)
print('## E3 different fixed VALUES, same slots, explicit prior')
B = [(1, 3.0), (3, 3.0)]
b_alone, = seq([B]); a, b = seq([FULL, B]); cmp('mu=3 after mu=1', b, b_alone)
print('## E4 different fixed VALUES, same slots, prior=None (FreeEpochModel default prior)')
try:
    b_alone, = seq([B], prior=None); a, b = seq([FULL, B], prior=None); cmp('mu=3 after mu=1 (default prior)', b, b_alone)
    for nm, s in (('A(mu=1)', a), ('B after A', b), ('B alone', b_alone)):
        print('    prior_list', nm, [None if q is None else (round(float(q.mu), 4), round(float(q.sigma), 4)) for q in s.prior_list])
except Exception as e:
    import traceback; traceback.print_exc()
print('## E5 full -> nested with prior=None (default prior)')
try:
    n_alone, = seq([NEST], prior=None); f, n = seq([FULL, NEST], prior=None); cmp('nested after full (default prior)', n, n_alone)
except Exception as e:
    import traceback; traceback.print_exc()

print('## E6 Graph.svgd twice on one graph, different data: does it ever hit?')
def data(theta, n=200, seed=3):
    random.seed(seed); np.random.seed(seed)
    g = coalescent_graph(4); g.update_weights([theta]); return np.asarray(g.sample(n))
g = coalescent_graph(4)
dA, dB = data(8.0), data(1.0)
PP = GaussPrior(ci=[0.5, 20], lower=0.0)
SVGD._compiled_cache.clear()
sa = g.svgd(dA, prior=PP, **KW); k1 = len(SVGD._compiled_cache)
sb = g.svgd(dB, prior=PP, **KW); k2 = len(SVGD._compiled_cache)
SVGD._compiled_cache.clear(); sb2 = g.svgd(dB, prior=PP, **KW)
print(f'  cache entries after 1st/2nd call: {k1}/{k2}; same model object: {sa.model is sb.model}')
cmp('Graph.svgd B after A', sb, sb2)

print('## E7 positive_params differs on one model object (plan test 3)')
model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=1)
def fit(pp, prior): 
    s = SVGD(model=model, observed_data=dA, theta_dim=1, prior=prior, positive_params=pp, **KW); s.optimize(); return s
SVGD._compiled_cache.clear(); s1 = fit(True, GaussPrior(ci=[0.5, 20], lower=0.0)); s2 = fit(False, GaussPrior(ci=[0.5, 20]))
SVGD._compiled_cache.clear(); s2a = fit(False, GaussPrior(ci=[0.5, 20])); cmp('pp=False after pp=True', s2, s2a)

print('## E8 one instance, optimize() twice after replacing observed_data (stale within instance?)')
SVGD._compiled_cache.clear()
s = SVGD(model=model, observed_data=dA, theta_dim=1, prior=PP, **KW); s.optimize(); m1 = float(s.get_results()['theta_mean'][0])
s.observed_data = __import__('jax').numpy.asarray(dB); s.optimize(); m2 = float(s.get_results()['theta_mean'][0])
SVGD._compiled_cache.clear()
s3 = SVGD(model=model, observed_data=dB, theta_dim=1, prior=PP, **KW); s3.optimize()
print(f'  fit(dA) {m1:.4f}; same instance after observed_data=dB {m2:.4f}; fresh fit(dB) {float(s3.get_results()["theta_mean"][0]):.4f}')

print('## E9 files left by the "dead" disk path under the redirected HOME')
for fpath in sorted(glob.glob(os.path.join(FAKE_HOME, '.phasic_cache', 'compiled_svgd_*.pkl'))):
    print('  ', os.path.basename(fpath), os.path.getsize(fpath), 'bytes')
print('  count:', len(glob.glob(os.path.join(FAKE_HOME, '.phasic_cache', 'compiled_svgd_*.pkl'))))
