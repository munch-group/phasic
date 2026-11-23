import sys
import numpy as np
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import two_island_graph
g = two_island_graph(6)
T = [0.002, 0.005, 0.01]
def F(th, gran=0):
    g.update_weights(list(th)); return float(np.sum(np.log([g.pdf(t, gran) for t in T])))
def maxrate(th):
    g.update_weights(list(th)); return max(sum(e.weight() for e in g.vertex_at(i).edges()) for i in range(g.vertices_length()))
rng = np.random.default_rng(0)
for scale in (30.0, 200.0, 2000.0):
    errs = []
    for _ in range(200):
        th = np.array([scale*rng.uniform(0.9, 1.1), 0.5*scale*rng.uniform(0.9, 1.1)])
        G = int(2*maxrate(th))
        fd = lambda h, gran=0: (F([th[0]+h, th[1]], gran) - F([th[0]-h, th[1]], gran))/(2*h)
        ref = (4*fd(th[0]*1e-4, G) - fd(th[0]*2e-4, G))/3
        errs.append(abs(fd(1e-7) - ref)/abs(ref))
    e = np.array(errs)
    print(f'scale={scale:g} (maxrate~{maxrate([scale, scale/2]):.3g}): abs 1e-7 relerr median {np.median(e):.1e}, '
          f'95th pct {np.percentile(e,95):.1e}, max {e.max():.1e}, fraction >1%: {np.mean(e>1e-2):.3f}, >100%: {np.mean(e>1):.3f}')
