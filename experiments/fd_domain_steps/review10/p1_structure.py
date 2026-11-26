# What does serialize() expose as C? parallel constant+param edges, discretize, start edges.
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np
from phasic import Graph
np.set_printoptions(linewidth=150)
def show(tag, g):
    s = g.serialize()
    print("==", tag, "weight_mode", s.get('weight_mode'), "param_length", s.get('param_length'))
    for k in ('edges','param_edges','start_edges','start_param_edges'):
        print(" ", k, np.asarray(s[k]).tolist())
# 1. parallel constant + param edge a->b (rate = 1 + t1 written as two edges)
g = Graph(1); st = g.starting_vertex()
a = g.find_or_create_vertex([2]); b = g.find_or_create_vertex([1])
try:
    st.add_edge(a, 1.0); a.add_edge(b, 1.0); a.add_edge(b, [0.0, 1.0])
except Exception as e: print("mixing refused:", str(e)[:80])
# 2. two param edges same target
g2 = Graph(1); st2 = g2.starting_vertex()
a2 = g2.find_or_create_vertex([2]); b2 = g2.find_or_create_vertex([1])
st2.add_edge(a2, 1.0); a2.add_edge(b2, [1.0, 0.0]); a2.add_edge(b2, [0.0, 1.0])
print("vertex a2 edges:", [(e.to().state().tolist(), e.weight()) for e in a2.edges()])
show("two param same target", g2)
# 3. parameterized starting edges
g3 = Graph(1); st3 = g3.starting_vertex()
a3 = g3.find_or_create_vertex([2]); b3 = g3.find_or_create_vertex([1])
st3.add_edge(a3, [1.0, 0.0]); st3.add_edge(b3, [0.0, 1.0]); a3.add_edge(b3, [1.0, 1.0])
show("param start edges", g3)
# 4. discretized
g4 = Graph(1); st4 = g4.starting_vertex()
a4 = g4.find_or_create_vertex([2]); b4 = g4.find_or_create_vertex([1])
st4.add_edge(a4, 1.0); a4.add_edge(b4, [1.0, 0.0]); b4.add_edge(g4.find_or_create_vertex([0]), [0.0, 1.0])
try:
    d = g4.discretize(0.1)
    show("discretized", d)
    print("was_dph", getattr(d, 'was_dph', None), d.is_discrete if hasattr(d,'is_discrete') else None)
except Exception as e:
    print("discretize err", type(e).__name__, e)
