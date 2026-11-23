# NOTE (after the fix): this script documents the UNFIXED code and uses
# SVGD._compiled_cache / removed helpers, which no longer exist; it does not
# run on builds that include the fix.
"""Review 3: LRT pattern with the DEFAULT epoch prior (prior=None): full then nested vs nested alone."""
import sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug1/tests/pytest')
from test_notebook_model_recovery import coalescent_indexed
from test_epoch_normalization import normalised, p_n2_two_epoch
from phasic import SVGD
KW = dict(n_particles=12, n_iterations=25, learning_rate=0.01, progress=False, seed=1)
L = 3; EP = [0.0, 0.4]
gi, idx = coalescent_indexed(2)
J = gi.joint_prob_graph(idx, discrete=False, reward_limit=L)
p = normalised([p_n2_two_epoch(k, 2.0, 1.0, 6.0, 1.0, EP[1]) for k in range(L + 1)])
rows = [[k, 0] for k in range(L + 1) for _ in range(int(round(p[k] * 5000)))]
FULL = [(1, 1.0), (3, 1.0)]; NEST = [(1, 1.0), (2, 0.5), (3, 1.0)]
def cmp(a, b): return 'BITEQ' if np.array_equal(a, b) else f'maxabs {np.max(np.abs(a-b)):.3e}'
C = SVGD._compiled_cache
def run(order):
    C.clear(); em = J.epoch_model(rows, epoch_starts=EP); out = {}
    for nm in order:
        s = em.fit(fixed=FULL if nm == 'full' else NEST, **KW)
        out[nm] = (np.asarray(s.particles).copy(), [None if q is None else {k: v for k, v in vars(q).items() if isinstance(v, (int, float))} for q in s.prior_list])
    return out
try:
    r1 = run(['full', 'nested']); r2 = run(['nested']); r3 = run(['nested'])
    print('priors full  :', r1['full'][1]); print('priors nested:', r2['nested'][1])
    print('nested alone rerun', cmp(r2['nested'][0], r3['nested'][0]), '| nested after full, MASTER', cmp(r2['nested'][0], r1['nested'][0]))
    print('means', np.mean(r2['nested'][0], 0).round(4), np.mean(r1['nested'][0], 0).round(4))
except Exception as e:
    import traceback; traceback.print_exc()
