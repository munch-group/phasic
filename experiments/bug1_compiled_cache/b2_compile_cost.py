# NOTE (after the fix): this script documents the UNFIXED code and uses
# SVGD._compiled_cache, which no longer exists; it does not run on builds
# that include the fix.
"""What does a cache hit save?  Time SVGD.optimize() with 1 iteration (so
compile dominates) on the same model object: cold, then cache hit."""
import sys, time, warnings, random
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import coalescent_graph, coalescent_indexed, two_island_graph
from phasic import Graph, GaussPrior, SVGD
KW = dict(n_particles=8, n_iterations=1, learning_rate=0.01, progress=False, seed=1)
def timed(mk):
    SVGD._compiled_cache.clear()
    t = time.time(); mk().optimize(); a = time.time() - t
    t = time.time(); mk().optimize(); b = time.time() - t
    return a, b
random.seed(1); np.random.seed(1)
g = two_island_graph(8); g.update_weights([1.0, 0.5]); d = np.asarray(g.sample(300))
m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2)
a, b = timed(lambda: SVGD(model=m, observed_data=d, theta_dim=2, prior=GaussPrior(ci=[0.1, 5], lower=0.0), **KW))
print(f'pdf path, two-island n=8 ({g.vertices_length()} v), 300 obs: cold {a:.2f}s, cache hit {b:.2f}s, saved {a-b:.2f}s')
for n in (4, 6):
    gi, idx = coalescent_indexed(n)
    J = gi.joint_prob_graph(idx, discrete=False, reward_limit=3)
    J.update_weights([2.0, 0.5])
    tab = J.joint_prob_table(); rows = tab[tab.columns[:-1]].to_numpy().tolist()[:40]
    em = J.epoch_model(rows, epoch_starts=[0.0, 0.3])
    pri = [GaussPrior(ci=[0.5, 20], lower=0.0), None] * 2
    t = time.time(); em.fit(fixed=[(1, 0.5), (3, 0.5)], prior=pri, **KW); a = time.time() - t
    t = time.time(); em.fit(fixed=[(1, 0.5), (3, 0.5)], prior=pri, **KW); b = time.time() - t
    print(f'epoch path, joint n={n} ({J.vertices_length()} v), 40 obs: cold {a:.2f}s, cache hit {b:.2f}s, saved {a-b:.2f}s')
