# NOTE (after the fix): this script documents the UNFIXED code and uses
# SVGD._compiled_cache / removed helpers, which no longer exist; it does not
# run on builds that include the fix.
"""Review 3: epoch (FreeEpochModel) path — bit determinism, test 5, prior override, weakref."""
import sys, gc, weakref, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug1/tests/pytest')
from test_notebook_model_recovery import coalescent_indexed
from test_epoch_normalization import normalised, p_n2_two_epoch
from phasic import GaussPrior, SVGD
KW = dict(n_particles=12, n_iterations=25, learning_rate=0.01, progress=False, seed=1)
L = 3; EP = [0.0, 0.4]
gi, idx = coalescent_indexed(2)
J = gi.joint_prob_graph(idx, discrete=False, reward_limit=L)
p = normalised([p_n2_two_epoch(k, 2.0, 1.0, 6.0, 1.0, EP[1]) for k in range(L + 1)])
rows = [[k, 0] for k in range(L + 1) for _ in range(int(round(p[k] * 5000)))]
pri = [GaussPrior(ci=[0.5, 20], lower=0.0), None] * 2
pri2 = [GaussPrior(ci=[0.01, 0.2], lower=0.0), None] * 2
FULL = [(1, 1.0), (3, 1.0)]; NEST = [(1, 1.0), (2, 0.5), (3, 1.0)]
def cmp(a, b): return 'BITEQ' if np.array_equal(a, b) else f'maxabs {np.max(np.abs(a-b)):.3e}'
def P(s): return np.asarray(s.particles).copy()
C = SVGD._compiled_cache
em = J.epoch_model(rows, epoch_starts=EP)
C.clear(); n1 = P(em.fit(fixed=NEST, prior=pri, **KW)); C.clear(); n2 = P(em.fit(fixed=NEST, prior=pri, **KW))
em2 = J.epoch_model(rows, epoch_starts=EP); C.clear(); n3 = P(em2.fit(fixed=NEST, prior=pri, **KW))
em3 = J.epoch_model(rows, epoch_starts=EP); C.clear(); em3.fit(fixed=FULL, prior=pri, **KW); n4 = P(em3.fit(fixed=NEST, prior=pri, **KW))
print('nested alone rerun', cmp(n1, n2), '| fresh epoch_model', cmp(n1, n3), '| after full, MASTER (test 5)', cmp(n1, n4), flush=True)
em4 = J.epoch_model(rows, epoch_starts=EP); C.clear(); a = P(em4.fit(fixed=FULL, prior=pri2, **KW))
em5 = J.epoch_model(rows, epoch_starts=EP); C.clear(); em5.fit(fixed=FULL, prior=pri, **KW); b = P(em5.fit(fixed=FULL, prior=pri2, **KW))
print('em.fit(prior=pri2) alone vs after em.fit(prior=pri) on same FreeEpochModel, MASTER:', cmp(a, b),
      ' means', np.mean(a, 0).round(3), np.mean(b, 0).round(3), flush=True)
C.clear(); s = em.fit(fixed=FULL, prior=pri, **KW); w = weakref.ref(s); del s; gc.collect(); h = w() is not None
C.clear(); gc.collect(); print('weakref epoch: alive with cache', h, ' alive after clear', w() is not None)
