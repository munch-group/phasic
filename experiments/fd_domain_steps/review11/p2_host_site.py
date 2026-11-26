# Joint-index host fallback (__init__.py:9551-9562) rule vs the plan's
# 'unconstrained' plain +-1e-7 rule, on expected_sojourn_time of a
# linear continuous graph whose vertex b has sole exit rate t1.
import numpy as np
from phasic import Graph
g = Graph(1)
s = g.starting_vertex()
a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2]); z = g.find_or_create_vertex([3])
s.add_edge(a, 1.0)
a.add_edge_parameterized(b, 0.0, [1.0, 0.0])
b.add_edge_parameterized(z, 0.0, [0.0, 1.0])
idx = [b.index()]
def soj(th):
    g.update_weights(list(th)); return np.asarray(g.expected_sojourn_time(idx))[0]
for t1 in [5e-8, 1e-8, 1e-9]:
    th = np.array([1.0, t1]); exact = -1.0/t1**2
    h = max(abs(t1)*1e-6, 1e-10)
    today = (soj(th+[0,h]) - soj(th-[0,h]))/(2*h)
    try:
        plain = (soj(th+[0,1e-7]) - soj(th-[0,1e-7]))/(2e-7)
    except Exception as e:
        plain = f"raised {type(e).__name__}: {e}"
    print(f"t1={t1:.0e} exact={exact:.4e} today_rel={today:.4e} plain={plain}")
