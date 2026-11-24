"""reviewK k5: attributes the v3.6 builtin / immutable-type rules read."""
import sys, math, itertools, numpy as np, scipy.special as sp
def resolve(o, use_name=False):
    m = getattr(o, "__module__", None); q = getattr(o, "__name__" if use_name else "__qualname__", None)
    if m is None or q is None: return f"REFUSED(module={m!r}, name={q!r})"
    obj = sys.modules.get(m)
    for part in q.split("."):
        obj = getattr(obj, part, None)
    return "ok" if obj is o else f"REFUSED(resolves to {obj!r})"
for o, un in ((np.exp, True), (np.add, True), (np.sum, False), (np.linalg.solve, False),
              (math.sqrt, False), (len, False), (sp.gamma, True), (np.frompyfunc(abs,1,1), True)):
    print(f"{getattr(o,'__name__',o)!s:25s} type={type(o).__name__:28s} {resolve(o, un)}")
for t in (int, np.int64, np.float64, np.ndarray, itertools.combinations_with_replacement):
    print(f"type {t.__name__:35s} IMMUTABLE={bool(t.__flags__ & (1<<8))} {resolve(t)}")
