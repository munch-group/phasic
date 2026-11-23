import sys, time, resource
import numpy as np
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import two_island_graph
g = two_island_graph(10)
ne = sum(len(g.vertex_at(i).edges()) for i in range(g.vertices_length()))
for s in (1, 1e2, 1e3, 1e4, 3e4):
    g.update_weights([1.0*s, 0.5*s])
    R = max(sum(e.weight() for e in g.vertex_at(i).edges()) for i in range(g.vertices_length()))
    t = time.time(); p = g.pdf(2.5); dt = time.time()-t
    S = 2.5*max(1000, int(2*R))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
    print(f'scale={s:g} maxrate={R:.3g} steps={S:.3g} edges={ne} edge-steps={S*ne:.3g} time={dt:.2f}s ns/edge-step={dt/(S*ne)*1e9:.2f} maxRSS={rss:.0f}MB', flush=True)
