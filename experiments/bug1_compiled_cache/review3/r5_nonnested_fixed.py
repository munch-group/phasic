# NOTE (after the fix): this script documents the UNFIXED code and uses
# SVGD._compiled_cache / removed helpers, which no longer exist; it does not
# run on builds that include the fix.
"""Review 3: FreeEpochModel.fit with NON-nested fixed sets on one model object.
First fit fixes slot 1 (so slot 1 has no prior in the compiled function); second
fit fixes slot 3 and leaves slot 1 FREE.  Does the second fit lose slot 1's prior?"""
import sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug1/tests/pytest')
from test_notebook_model_recovery import coalescent_indexed
from test_epoch_normalization import normalised, p_n2_two_epoch
from phasic import GaussPrior, SVGD
KW = dict(n_particles=12, n_iterations=60, learning_rate=0.02, progress=False, seed=1)
L = 3; EP = [0.0, 0.4]
gi, idx = coalescent_indexed(2)
J = gi.joint_prob_graph(idx, discrete=False, reward_limit=L)
p = normalised([p_n2_two_epoch(k, 2.0, 1.0, 6.0, 1.0, EP[1]) for k in range(L + 1)])
rows = [[k, 0] for k in range(L + 1) for _ in range(int(round(p[k] * 200)))]
pri = GaussPrior(ci=[0.9, 1.1], lower=0.0)          # tight shared prior on every free slot
def cmp(a, b): return 'BITEQ' if np.array_equal(a, b) else f'maxabs {np.max(np.abs(a-b)):.3e}'
C = SVGD._compiled_cache
def th(s): return np.asarray(s.get_results()['theta_mean']).round(4)
em = J.epoch_model(rows, epoch_starts=EP, prior=pri)
C.clear(); a = em.fit(fixed=[(3, 1.0)], **KW)
em2 = J.epoch_model(rows, epoch_starts=EP, prior=pri)
C.clear(); em2.fit(fixed=[(1, 1.0)], **KW); b = em2.fit(fixed=[(3, 1.0)], **KW)
print('fit(fixed=[(3,1)]) alone       theta_mean', th(a))
print('same after fit(fixed=[(1,1)])  theta_mean', th(b), ' particles', cmp(np.asarray(a.particles), np.asarray(b.particles)))
