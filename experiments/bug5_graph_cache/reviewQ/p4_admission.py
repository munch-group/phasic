"""Review Q: spot-check admission predicates of plan s2.2.1/s2.2.2 on this env."""
import sys, types, math, itertools, collections, functools
import numpy as np

def resolves(x, attr="__qualname__"):
    m = getattr(x, "__module__", None); q = getattr(x, attr, None) or getattr(x, "__name__", None)
    if m is None or q is None or m not in sys.modules:
        return False
    o = sys.modules[m]
    for part in q.split("."):
        o = getattr(o, part, None)
    return o is x

print("dtype class modules:", {n: type(np.dtype(n)).__module__ for n in ["f8", "M8[D]", "U3", "?", "c16"]})
for t in [itertools.combinations_with_replacement, collections.OrderedDict, collections.defaultdict,
          np.ndarray, np.float64, range, frozenset, functools.partial, types.SimpleNamespace]:
    print(f"type {t.__module__}.{t.__qualname__}: immutable={bool(t.__flags__ & (1 << 8))} resolves={resolves(t)}")
for f in [np.array, np.zeros, math.comb, len, np.exp, np.sum, np.add.reduce, np.random.rand]:
    s = getattr(f, "__self__", None)
    print(f"{f!r:60.60} type={type(f).__name__} self={type(s).__name__} resolves={resolves(f) or resolves(f, '__name__')}")
