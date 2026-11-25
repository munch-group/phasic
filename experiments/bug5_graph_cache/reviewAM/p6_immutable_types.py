# Check the immutable-compiled-type rule on types users capture, and the
# builtin-function rule on common numpy/math callables (Python version given).
import sys, types, itertools, collections, functools, operator, math
import numpy as np
print(sys.version.split()[0], "numpy", np.__version__)
def imm(t):
    ok = type(t) is type and bool(t.__flags__ & (1 << 8))
    mod = getattr(t, "__module__", None); qn = getattr(t, "__qualname__", None)
    res = None
    try:
        o = sys.modules[mod]
        for part in qn.split("."):
            o = getattr(o, part)
        res = o is t
    except Exception as e:
        res = f"ERR {type(e).__name__}"
    return ok, mod, qn, res
for t in [itertools.combinations_with_replacement, itertools.product, collections.defaultdict,
          collections.OrderedDict, collections.deque, int, float, range, np.float64, np.ndarray, np.int64,
          functools.partial, collections.Counter]:
    print(f"{t!r:60s} imm/mod/qn/resolves:", imm(t))
def bi(f):
    s = getattr(f, "__self__", None)
    mod = getattr(f, "__module__", None); qn = getattr(f, "__qualname__", None) or getattr(f, "__name__", None)
    try:
        o = sys.modules[mod]
        for part in qn.split("."):
            o = getattr(o, part)
        r = o is f
    except Exception as e:
        r = f"ERR {type(e).__name__}"
    return type(f).__name__, type(s).__name__, mod, qn, r
for f in [math.sqrt, len, np.exp, np.sum, np.linalg.solve, np.array, np.zeros, np.where, np.add.reduce, operator.add, np.random.rand]:
    print(f"{getattr(f,'__name__',f)!s:12s}", bi(f))
