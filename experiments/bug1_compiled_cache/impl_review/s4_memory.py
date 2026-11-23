# NOTE (after the fix): this script documents the UNFIXED code and uses
# SVGD._compiled_cache / removed helpers, which no longer exist; it does not
# run on builds that include the fix.
"""Impl review check 4: RSS and time over 20 small fits on ONE model object."""
import os, sys, tempfile, warnings, random, gc, time, weakref
HERE = os.path.dirname(os.path.abspath(__file__)); os.environ['HOME'] = tempfile.mkdtemp(prefix='fakehome_', dir=HERE)
warnings.filterwarnings('ignore'); sys.path.insert(0, '/Users/kmt/phasic-bug1/tests/pytest')
import numpy as np, psutil, phasic
from phasic import Graph, GaussPrior, SVGD
from test_epoch_normalization import normalised, p_n2_two_epoch
from test_notebook_model_recovery import coalescent_graph, coalescent_indexed
print('build:', phasic.__file__, 'has cache:', hasattr(SVGD, '_compiled_cache'))
proc = psutil.Process(); rss = lambda: proc.memory_info().rss / 2**20
wide = lambda: GaussPrior(ci=[0.5, 20], lower=0.0)
KW = dict(n_particles=20, n_iterations=30, learning_rate=0.01, progress=False, seed=1)
random.seed(3); np.random.seed(3)
g = coalescent_graph(6); g.update_weights([4.0]); base = np.asarray(g.sample(20000))
m = Graph.pmf_and_moments_from_graph(coalescent_graph(6), nr_moments=2, theta_dim=1)
def loop(name, one, n=20):
    refs, r, t = [], [], []
    for i in range(n):
        t0 = time.time(); s = one(i); t.append(time.time() - t0)
        refs.append(weakref.ref(s)); del s; gc.collect(); r.append(rss())
    alive = sum(x() is not None for x in refs)
    print(f'{name}: RSS MB fit1 {r[0]:.0f} fit5 {r[4]:.0f} fit10 {r[9]:.0f} fit20 {r[-1]:.0f} | growth fit5->fit20 {r[-1]-r[4]:+.0f} MB ({(r[-1]-r[4])/15:+.2f}/fit) | '
          f'SVGD instances alive {alive}/{n} | s/fit first {t[0]:.2f} median(2..) {np.median(t[1:]):.2f}', flush=True)
    print('   rss:', ' '.join(f'{x:.0f}' for x in r), flush=True)
# same shape, different data each time (20000 obs each, kept with history)
def pdf(i):
    s = SVGD(model=m, observed_data=base * (1 + 0.01 * i), theta_dim=1, prior=wide(), **KW); s.optimize(); return s
loop('pdf path, same model, 20k obs', pdf)
L = 3; EP = [0.0, 0.4]
gg, idx = coalescent_indexed(4)
jc = gg.joint_prob_graph(idx, discrete=False, reward_limit=L)
print('joint graph vertices', jc.vertices_length())
g2, i2 = coalescent_indexed(2); j2 = g2.joint_prob_graph(i2, discrete=False, reward_limit=L)
p = normalised([p_n2_two_epoch(k, 2.0, 1.0, 6.0, 1.0, EP[1]) for k in range(L + 1)])
rows = [[k, 0] for k in range(L + 1) for _ in range(int(round(p[k] * 400)))]
em = j2.epoch_model(rows, epoch_starts=EP)
def ep(i):
    return em.fit(fixed=[(1, 1.0), (3, 1.0)], prior=[GaussPrior(ci=[0.5, 20 + i], lower=0.0), None, wide(), None], **KW)
loop('epoch path, same FreeEpochModel', ep)
if hasattr(SVGD, '_compiled_cache'): print('cache entries', len(SVGD._compiled_cache))
print('compiled_svgd files in fake HOME:', sum('compiled_svgd' in f for dp, _, fs in os.walk(os.environ['HOME']) for f in fs))
import shutil; shutil.rmtree(os.environ['HOME'], ignore_errors=True)
