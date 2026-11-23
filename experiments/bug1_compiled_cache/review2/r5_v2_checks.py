# NOTE (after the fix): this script documents the UNFIXED code and uses
# SVGD._compiled_cache / removed helpers, which no longer exist; it does not
# run on builds that include the fix.
"""Review 2 of plan v2: feasibility of planned tests 10, 11, 14 and the
'precompile=False never touches the cache' statement (unmodified build)."""
import sys, os, gc, weakref, warnings, random, pickle, traceback, tempfile, pathlib
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import coalescent_graph, coalescent_indexed
from phasic import Graph, GaussPrior, SVGD
KW = dict(n_particles=8, n_iterations=5, learning_rate=0.01, progress=False, seed=1)
P = lambda: GaussPrior(ci=[0.5, 20], lower=0.0)
def data(th, seed=3):
    random.seed(seed); np.random.seed(seed)
    g = coalescent_graph(4); g.update_weights([th]); return np.asarray(g.sample(100))

# --- (c) HOME redirect + pickle.load recorder (planned test 11), done first so caches are cold
home = pathlib.Path(__file__).parent / 'r5_home'
import shutil; shutil.rmtree(home, ignore_errors=True); home.mkdir()
os.environ['HOME'] = str(home)
calls = []
_orig = pickle.load
def rec(f, *a, **k):
    st = traceback.extract_stack(limit=3)[0]
    calls.append((getattr(f, 'name', '?'), f'{os.path.basename(st.filename)}:{st.lineno}'))
    return _orig(f, *a, **k)
pickle.load = rec
print('## (c) redirected HOME, pickle.load recorded')
g = coalescent_graph(4)
for k in range(2):
    s = g.svgd(observed_data=data(8.0), prior=P(), **KW)
m = Graph.pmf_and_moments_from_graph(coalescent_graph(4), nr_moments=2, theta_dim=1)
for k in range(2):
    SVGD(model=m, observed_data=data(8.0), theta_dim=1, prior=P(), **KW).optimize()
gi, idx = coalescent_indexed(4)
J = gi.joint_prob_graph(idx, discrete=False, reward_limit=3); J.update_weights([2.0, 0.5])
tab = J.joint_prob_table(); rows = tab[tab.columns[:-1]].to_numpy().tolist()[:40]
em = J.epoch_model(rows, epoch_starts=[0.0, 0.3])
mkpri = lambda: [GaussPrior(ci=[0.5, 20], lower=0.0), None] * 2
FX = [(1, 0.5), (3, 0.5)]
for k in range(2):
    em.fit(fixed=FX, prior=mkpri(), **KW)
# a second, freshly built epoch model (the planned tests use fresh-model references)
gi2, idx2 = coalescent_indexed(4)
J2 = gi2.joint_prob_graph(idx2, discrete=False, reward_limit=3); J2.update_weights([2.0, 0.5])
J2.epoch_model(rows, epoch_starts=[0.0, 0.3]).fit(fixed=FX, prior=mkpri(), **KW)
print('  pickle.load calls:', len(calls))
for c in calls[:20]: print('    ', c)
files = sorted(str(p.relative_to(home)) for p in home.rglob('*') if p.is_file())
print('  files under redirected HOME:', len(files))
for f in files[:30]: print('    ', f, os.path.getsize(home / f))
pickle.load = _orig

# --- (a) planned test 14: same instance, optimize() twice
print('## (a) same instance, optimize() twice')
for label, extra in (('default', {}), ('no history', None)):
    s = SVGD(model=m, observed_data=data(8.0), theta_dim=1, prior=P(), **KW)
    if extra is None:
        s.optimize(return_history=False); a = np.asarray(s.particles).copy()
        s.optimize(return_history=False); b = np.asarray(s.particles)
    else:
        s.optimize(); a = np.asarray(s.particles).copy(); s.optimize(); b = np.asarray(s.particles)
    print(f'  {label}: bit-equal {np.array_equal(a, b)}  max abs diff {np.max(np.abs(a-b)):.3g}')
s = em.fit(fixed=FX, prior=mkpri(), **KW); a = np.asarray(s.particles).copy(); s.optimize(); b = np.asarray(s.particles)
print(f'  epoch: bit-equal {np.array_equal(a, b)}  max abs diff {np.max(np.abs(a-b)):.3g}')

# --- (b) planned test 10: weakref dies once the class cache is out of the way
print('## (b) retention with the class cache cleared (what the fix gives)')
def dies(mk):
    s = mk(); r = weakref.ref(s); del s
    SVGD._compiled_cache.clear(); gc.collect(); gc.collect()
    return r() is None
print('  direct SVGD, default      :', dies(lambda: SVGD(model=m, observed_data=data(8.0), theta_dim=1, prior=P(), **KW).optimize()))
def nh():
    s = SVGD(model=m, observed_data=data(8.0), theta_dim=1, prior=P(), **KW); s.optimize(return_history=False); return s
print('  direct SVGD, no history   :', dies(nh))
print('  FreeEpochModel.fit        :', dies(lambda: em.fit(fixed=FX, prior=mkpri(), **KW)))
print('  Graph.svgd (pdf)          :', dies(lambda: g.svgd(observed_data=data(8.0), prior=P(), **KW)))
print('  Graph.svgd (epoch)        :', dies(lambda: J.svgd(rows, epoch_starts=[0.0, 0.3], fixed=[(1, 0.5)], prior=[GaussPrior(ci=[0.5, 20], lower=0.0), None], **KW)))
print('  direct SVGD, regularized  :', dies(lambda: SVGD(model=m, observed_data=data(8.0), theta_dim=1, prior=P(), regularization=1.0, **KW).optimize()))

# --- (d) does precompile=False avoid the cache?
print('## (d) precompile / jit and the cache')
for kw in (dict(precompile=False), dict(jit=False), dict(jit=False, precompile=False)):
    SVGD._compiled_cache.clear()
    s = SVGD(model=m, observed_data=data(8.0), theta_dim=1, prior=P(), **dict(KW, **kw)); s.optimize()
    print(f'  {kw}: jit_enabled={s.jit_enabled}  cache entries after fit {len(SVGD._compiled_cache)}')
