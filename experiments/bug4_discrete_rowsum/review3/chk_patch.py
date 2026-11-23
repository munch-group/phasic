import phasic, numpy as np
from phasic import Graph
print(phasic.__file__)
def geom(p):
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, p); g.is_discrete = True; return g
for ex in (0, 2.3e-16, 5e-10, 2e-9, 1e-5):
    try: print(ex, np.asarray(geom(1+ex).pdf(np.array([1,2]))))
    except Exception as e: print(ex, 'RAISES', str(e)[:70])
g=Graph(1); s=g.starting_vertex(); a=g.find_or_create_vertex([1]); b=g.find_or_create_vertex([2]); s.add_edge(a,1.0); a.add_edge(b,3.0)
print('continuous', np.asarray(g.pdf(np.array([0.5,1.0]))))
