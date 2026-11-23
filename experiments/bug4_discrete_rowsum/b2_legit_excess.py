"""How far above 1 do row sums of LEGITIMATE discrete graphs get, from
rounding alone?  (discretize()-built graphs and joint_prob_graph(discrete=True)
at a range of parameters.)"""
import sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import coalescent_graph, two_island_graph, coalescent_indexed
def rows(g):
    out = []
    for i in range(1, g.vertices_length()):
        es = g.vertex_at(i).edges()
        if es: out.append(sum(e.weight() for e in es))
    return np.array(out)
def rep(label, g):
    r = rows(g)
    print(f'  {label}: {len(r)} rows, max row sum - 1 = {r.max() - 1:+.3e}, rows > 1: {(r > 1).sum()}, rows within 1e-12 of 1: {(np.abs(r - 1) < 1e-12).sum()}')
print('## discretize()')
for name, g0, th in (('coalescent n=10', coalescent_graph(10), [7.0]), ('coalescent n=22', coalescent_graph(22), [0.37]),
                     ('two-island n=8', two_island_graph(8), [0.7, 0.3]), ('two-island n=8', two_island_graph(8), [123.4, 0.017])):
    g0.update_weights(th)
    for step in (None, 0.001):
        try:
            d = g0.discretize() if step is None else g0.discretize(step)
            rep(f'{name} theta={th} discretize({"" if step is None else step})', d[0] if isinstance(d, tuple) else d)
        except Exception as e:
            print(f'  {name} discretize({step}): {type(e).__name__}: {str(e)[:100]}')
print('## joint_prob_graph(discrete=True)')
for n, th in ((4, [1e-4, 1e-2]), (5, [2.0, 0.05]), (5, [0.3, 1.2e-4]), (6, [5.0, 0.01])):
    g, idx = coalescent_indexed(n)
    try:
        J = g.joint_prob_graph(idx, reward_limit=3)
        J.update_weights(th); rep(f'n={n} theta={th}', J)
    except Exception as e:
        print(f'  n={n} theta={th}: {type(e).__name__}: {str(e)[:120]}')
