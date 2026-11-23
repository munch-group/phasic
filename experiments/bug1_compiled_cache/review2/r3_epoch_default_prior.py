"""Review 2: is the shipped LRT reuse (FreeEpochModel full -> nested on one
object) really unaffected on master?  With prior=None the default prior is
rebuilt from the merged fixed list, so full and nested have DIFFERENT priors
on the shared free slots, and the cached gradient carries the full fit's."""
import sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import coalescent_indexed
from phasic import SVGD, GaussPrior
KW = dict(n_particles=16, n_iterations=60, learning_rate=0.01, progress=False, seed=1)
gi, idx = coalescent_indexed(4)
J = gi.joint_prob_graph(idx, discrete=False, reward_limit=3)
J.update_weights([2.0, 0.5])
tab = J.joint_prob_table(); p = tab['prob'].to_numpy(); p = p / p.sum()
rng = np.random.default_rng(3)
rows = tab.loc[rng.choice(tab.index.values, 400, p=p), tab.columns[:-1]].to_numpy().tolist()
def show(tag, s):
    pl = [None if q is None else (round(float(q.mu), 4), round(float(q.sigma), 4)) for q in s.prior_list]
    print(f'  {tag:22s} theta_mean {np.round(np.asarray(s.theta_mean), 4)}  priors {pl}')
    return np.asarray(s.particles)
for label, nested_fixed in (('nested fixes slot 0 at 1.0', [(0, 1.0)]),
                            ('nested fixes slot 2 at 4.0', [(2, 4.0)])):
    print('##', label)
    try:
        em = J.epoch_model(rows, epoch_starts=[0.0, 0.3], fixed=[(1, 0.5)])
        SVGD._compiled_cache.clear()
        full = em.fit(**KW); show('full', full)
        a = show('nested after full', em.fit(fixed=nested_fixed, **KW))
        SVGD._compiled_cache.clear()
        b = show('nested alone', em.fit(fixed=nested_fixed, **KW))
        print('  particles identical:', bool(np.array_equal(a, b)), ' max abs diff', float(np.max(np.abs(a - b))))
    except Exception as e:
        print('  FAILED', type(e).__name__, str(e)[:400])
