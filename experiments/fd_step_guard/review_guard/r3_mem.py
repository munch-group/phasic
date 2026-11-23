"""Memory of Graph.pdf vs steps on a tiny graph (few edges): is it 16 B/step?"""
import time, resource, sys
from phasic import Graph
g = Graph(1)
a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([3])
g.starting_vertex().add_edge(a, 1.0)
R = float(sys.argv[1])
a.add_edge(b, R); b.add_edge(c, 1.0)   # c absorbing; b slow so mass stays
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
print('baseline maxRSS MB', round(rss()))
for t in (1.0, 5.0, 12.0):
    t0 = time.time(); v = g.pdf(t); dt = time.time()-t0
    print(f'rate={R:g} t={t} steps={2*R*t:.3g} time={dt:.2f}s maxRSS={rss():.0f}MB  expected vectors={2*R*t*16/1e6:.0f}MB', flush=True)
