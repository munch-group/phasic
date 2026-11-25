# Test 15 hand-writes a FORMULA-mode entry: are all tape values plain JSON
# (a plain-JSON + ndarray-only encoder must accept them)?
from phasic import Graph
g = Graph(1)
s = g.starting_vertex(); a = g.find_or_create_vertex([2]); b = g.find_or_create_vertex([1])
s.add_edge(a, 1.0); a.add_edge(b, [2.0, 0.5])
g.weight_formula = "c0*t0 + c1*1.5"
d = g.serialize()
t = d["weight_formula_tape"]
def types(v):
    if isinstance(v, dict): return {k: types(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return sorted({type(x).__name__ for x in v})
    return type(v).__name__
print(types(t))
