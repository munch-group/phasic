"""reviewY y5: check the attribute facts §2.2.1 relies on for the builtin /
ufunc / dispatcher arm and the immutable-compiled-type arm."""
import sys, types, math, itertools, functools, operator, collections
import numpy as np

def info(label, x):
    g = lambda n: getattr(x, n, "<absent>")
    mod = g("__module__"); qn = g("__qualname__"); nm = g("__name__")
    self_ = getattr(x, "__self__", "<absent>")
    key = qn if qn != "<absent>" else nm
    resolved = "<n/a>"
    if isinstance(mod, str) and isinstance(key, str):
        obj = sys.modules.get(mod)
        for part in key.split("."):
            obj = getattr(obj, part, None) if obj is not None else None
        resolved = obj is x
    flags = getattr(type(x), "__flags__", 0)
    print(f"{label:38s} type={type(x).__name__:28s} mod={mod!s:14.14s} qn={qn!s:22.22s} "
          f"self={type(self_).__name__:10s} resolves_to_same={resolved}")

info("math.sqrt", math.sqrt)
info("len", len)
info("np.exp (ufunc)", np.exp)
info("np.add (ufunc)", np.add)
info("np.sum (dispatcher)", np.sum)
info("np.linalg.solve", np.linalg.solve)
info("np.frompyfunc ufunc", np.frompyfunc(lambda a: a, 1, 1))
info("{}.get", {}.get)
info("operator.itemgetter(0)", operator.itemgetter(0))
try:
    import scipy.special as sp
    info("scipy.special.gamma", sp.gamma)
    info("scipy.special.comb", sp.comb)
except ImportError:
    print("no scipy")

IMM = 1 << 8
for label, t in [("itertools.combinations_with_replacement", itertools.combinations_with_replacement),
                 ("collections.OrderedDict", collections.OrderedDict),
                 ("collections.Counter", collections.Counter),
                 ("np.float64", np.float64), ("np.ndarray", np.ndarray),
                 ("dict", dict), ("functools.partial", functools.partial)]:
    print(f"{label:38s} immutable_flag={bool(t.__flags__ & IMM)} type(t)={type(t).__name__}")

class User: pass
print(f"{'user class':38s} immutable_flag={bool(User.__flags__ & IMM)}")
