"""Review 2: Graph.svgd() builds a fresh model per call, so the class cache
never hits there and only retains.  RSS after k fits, with the cache left
alone (master behaviour) vs cleared after every fit (what the fix gives)."""
import sys, gc, os, warnings, resource, subprocess
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import coalescent_indexed
from phasic import SVGD, GaussPrior
mode = sys.argv[1]
KW = dict(n_particles=8, n_iterations=1, learning_rate=0.01, progress=False, seed=1)
def rss():
    return int(subprocess.run(['ps', '-o', 'rss=', '-p', str(os.getpid())], capture_output=True, text=True).stdout) / 1024
gi, idx = coalescent_indexed(6)
J = gi.joint_prob_graph(idx, discrete=False, reward_limit=3)
J.update_weights([2.0, 0.5])
tab = J.joint_prob_table(); rows = tab[tab.columns[:-1]].to_numpy().tolist()[:40]
pri = [GaussPrior(ci=[0.5, 20], lower=0.0), None]
out = []
for k in range(10):
    s = J.svgd(rows, epoch_starts=[0.0, 0.3], fixed=[(1, 0.5)], prior=pri, **KW)
    del s
    if mode == 'clear':
        SVGD._compiled_cache.clear()
    gc.collect()
    out.append((len(SVGD._compiled_cache), round(rss())))
print(mode, J.vertices_length(), 'v  (cache entries, RSS MB) per fit:', out)
