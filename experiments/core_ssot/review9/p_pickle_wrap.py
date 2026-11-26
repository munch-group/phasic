import pickle, numpy as np
import phasic
from phasic import Graph
g = Graph(1)
s = g.starting_vertex(); a = g.find_or_create_vertex([1]); s.add_edge(a, 1.0)
b = g.find_or_create_vertex([0]); a.add_edge(b, 2.0)
try:
    pickle.dumps(g); print("pickle: OK")
except Exception as e:
    print("pickle:", type(e).__name__, str(e)[:120])
g.is_discrete = True
w = Graph(g)
print("source is_discrete", g.is_discrete, "wrap is_discrete", w.is_discrete)
w.is_discrete = False
print("after wrap assignment, source", g.is_discrete)
