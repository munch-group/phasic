import jax; jax.config.update("jax_enable_x64", True)
from phasic import Graph
g = Graph(1)
s = g.starting_vertex()
a = g.find_or_create_vertex([3]); b = g.find_or_create_vertex([2])
s.add_edge(a, 1.0); a.add_edge(b, [1.0, 0.0])
d = g.serialize()
print(sorted(d.keys()))
for k in d:
    v = d[k]
    if not hasattr(v, '__len__') or isinstance(v,str) or (hasattr(v,'__len__') and len(v)<5): print(k, v)
gd = g.discretize(0.1) if hasattr(g,'discretize') else None
try:
    dd = gd.serialize(); print("discretized keys diff", set(dd)-set(d), {k:dd[k] for k in dd if k in ('discrete','is_discrete','was_dph','weight_mode')})
except Exception as e: print("discretize", type(e).__name__, str(e)[:200])
