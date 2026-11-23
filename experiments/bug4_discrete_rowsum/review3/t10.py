"""Is the table read from a FRESH (un-normalised, rows up to 3.03) discrete joint graph equal to the one after update_weights?"""
import sys, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug4/tests/pytest')
from test_notebook_model_recovery import coalescent_indexed
for n, kw in ((3, dict(reward_limit=2, mutation_rate=1e-2)), (4, dict(reward_limit=2)), (5, dict(reward_limit=3))):
    g, idx = coalescent_indexed(n); J = g.joint_prob_graph(idx, **kw)
    a = J.joint_prob_table()['prob'].to_numpy().copy()
    es = np.asarray(J.expected_sojourn_time())
    J.update_weights([1.0, 1.0]); b = J.joint_prob_table()['prob'].to_numpy()
    print(f'n={n} {kw}: fresh-vs-updated max abs diff {np.abs(a-b).max():.3e}, table sum {a.sum():.6f}; fresh expected_sojourn_time finite: {np.isfinite(es).all()}')
