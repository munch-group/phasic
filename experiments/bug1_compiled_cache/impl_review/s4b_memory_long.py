# NOTE (after the fix): this script documents the UNFIXED code and uses
# SVGD._compiled_cache / removed helpers, which no longer exist; it does not
# run on builds that include the fix.
"""Impl review check 4b: (i) 80 quick same-model refits (recompile each time after the fix) - RSS slope;
(ii) 25 Graph.svgd calls on one graph (the pattern that leaks on master: one cache entry per call)."""
import os, sys, tempfile, warnings, random, gc, time, weakref
HERE = os.path.dirname(os.path.abspath(__file__)); os.environ['HOME'] = tempfile.mkdtemp(prefix='fakehome_', dir=HERE)
warnings.filterwarnings('ignore'); sys.path.insert(0, '/Users/kmt/phasic-bug1/tests/pytest')
import numpy as np, psutil, phasic
from phasic import Graph, GaussPrior, SVGD
from test_notebook_model_recovery import coalescent_graph
print('build:', phasic.__file__, 'has cache:', hasattr(SVGD, '_compiled_cache'))
proc = psutil.Process(); rss = lambda: proc.memory_info().rss / 2**20
wide = lambda: GaussPrior(ci=[0.5, 20], lower=0.0)
KW = dict(n_particles=5, n_iterations=3, learning_rate=0.01, progress=False, seed=1)
random.seed(3); np.random.seed(3)
g = coalescent_graph(6); g.update_weights([4.0]); base = np.asarray(g.sample(20000))
m = Graph.pmf_and_moments_from_graph(coalescent_graph(6), nr_moments=2, theta_dim=1)
def loop(name, one, n, every):
    refs, r = [], []; t0 = time.time()
    for i in range(n):
        s = one(i); refs.append(weakref.ref(s)); del s; gc.collect(); r.append(rss())
    k = n // 4
    slope = np.polyfit(np.arange(k, n), r[k:], 1)[0]
    print(f'{name}: alive {sum(x() is not None for x in refs)}/{n}  RSS slope over last 3/4: {slope:+.2f} MB/fit  total {time.time()-t0:.0f}s', flush=True)
    print('   rss every %d:' % every, ' '.join(f'{x:.0f}' for x in r[::every]), f'| last {r[-1]:.0f}', flush=True)
def pdf(i):
    s = SVGD(model=m, observed_data=base * (1 + 0.001 * i), theta_dim=1, prior=wide(), **KW); s.optimize(); return s
loop('(i) same model, 80 refits', pdf, 80, 8)
G = coalescent_graph(6)
loop('(ii) Graph.svgd x25 on one graph', lambda i: G.svgd(base * (1 + 0.001 * i), prior=wide(), **KW), 25, 3)
if hasattr(SVGD, '_compiled_cache'): print('cache entries', len(SVGD._compiled_cache))
import shutil; shutil.rmtree(os.environ['HOME'], ignore_errors=True)
