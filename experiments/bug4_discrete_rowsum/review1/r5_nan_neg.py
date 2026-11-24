"""Review 1, part 5: rows the sum-only check cannot see: NaN weight, negative weight with sum <= 1."""
import warnings, random, numpy as np
warnings.filterwarnings('ignore')
from phasic import Graph
def t(label, f):
    try: print(f'    {label}: {np.asarray(f(), dtype=float).tolist()}')
    except BaseException as e: print(f'    {label}: {type(e).__name__}: {str(e).splitlines()[0][:100]}')
def mk(th):
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0, 0.0]); a.add_edge(c, [0.0, 1.0]); g.is_discrete = True; g.update_weights(th); return g
for th in ([-0.5, 1.2], [float('nan'), 0.2], [1.3, -0.4]):
    print(f'# native DPH, two out-edges with weights {th} (row sum {sum(th)})')
    t('pdf(1..3)', lambda: mk(th).pdf(np.array([1, 2, 3])))
    t('stop_probability(1)', lambda: mk(th).stop_probability(1))
    t('expectation', lambda: mk(th).expectation())
    t('variance', lambda: mk(th).variance())
    t('sample(3)', lambda: (random.seed(1), mk(th).sample(3))[1])
