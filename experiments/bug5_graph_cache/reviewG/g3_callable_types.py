"""reviewG g3: callables that LOOK like Python functions (duck typing) but are not
types.FunctionType: do they expose __code__/__closure__/__defaults__, and would a
duck-typed fingerprint give two different objects the same code fingerprint?"""
import types, functools, inspect
import numpy as np

def show(label, f):
    co = getattr(f, "__code__", None)
    print(f"{label:38s} type={type(f).__name__:26s} FunctionType={type(f) is types.FunctionType!s:5s} "
          f"isfunction={inspect.isfunction(f)!s:5s} has__code__={co is not None!s:5s} "
          f"co_code={getattr(co,'co_code',None)!r:.40} co_names={getattr(co,'co_names',None)} "
          f"closure={getattr(f,'__closure__','<none>')} ")
    return co

# cython functions (cyfunction) from libraries in this env
cands = []
try:
    import scipy.sparse.csgraph._shortest_path as sp
    cands += [("csgraph.dijkstra", sp.dijkstra), ("csgraph.floyd_warshall", sp.floyd_warshall)]
except Exception as e:
    print("csgraph import failed", e)
try:
    import scipy.special._comb as cm
    cands += [(n, getattr(cm, n)) for n in dir(cm) if not n.startswith("__")][:2]
except Exception as e:
    print("_comb import failed", e)
try:
    import pandas._libs.lib as pl
    cands += [("pandas lib.is_scalar", pl.is_scalar), ("pandas lib.is_list_like", pl.is_list_like)]
except Exception as e:
    print("pandas lib import failed", e)
cos = []
for label, f in cands:
    cos.append((label, show(label, f)))
# Two different cyfunctions with an identical code-fingerprint view?
def view(co):
    if co is None: return None
    return tuple(getattr(co, a, None) for a in ("co_code", "co_consts", "co_names", "co_varnames",
              "co_freevars", "co_cellvars", "co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_flags"))
vs = {}
for label, co in cos:
    v = view(co)
    if v is not None:
        vs.setdefault(v, []).append(label)
for v, labels in vs.items():
    if len(labels) > 1:
        print("SAME code-fingerprint view (minus co_name):", labels)
print("co_name per cyfunction:", [(l, getattr(c, 'co_name', None)) for l, c in cos if c is not None])
