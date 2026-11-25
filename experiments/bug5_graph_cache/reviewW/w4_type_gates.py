"""reviewW w4: check plan's type-gate assumptions against installed numpy/scipy (no phasic graph built)."""
import sys, types, functools, itertools, math
import numpy as np
print("python", sys.version.split()[0], "numpy", np.__version__)

def resolves(obj, use_name=False):
    mod = getattr(obj, "__module__", None)
    qn = getattr(obj, "__name__" if use_name else "__qualname__", None)
    if mod is None or qn is None:
        return f"missing attr (module={mod!r}, name={qn!r})"
    m = sys.modules.get(mod)
    cur = m
    try:
        for part in qn.split("."):
            cur = getattr(cur, part)
    except AttributeError:
        return f"no {mod}.{qn}"
    return cur is obj

print("type(np.sum):", type(np.sum), "np.linalg.solve same type:", type(np.linalg.solve) is type(np.sum),
      "resolves:", resolves(np.linalg.solve))
print("np.exp ufunc exact:", type(np.exp) is np.ufunc, "resolves by __name__:", resolves(np.exp, True))
print("np.abs name:", np.abs.__name__, resolves(np.abs, True))
fpy = np.frompyfunc(lambda x: x, 1, 1)
print("frompyfunc ufunc:", type(fpy) is np.ufunc, "module:", getattr(fpy, "__module__", None), "name:", fpy.__name__, resolves(fpy, True))
print("math.sqrt builtin exact:", type(math.sqrt) is types.BuiltinFunctionType, "self:", math.sqrt.__self__, resolves(math.sqrt))
print("{}.get exact builtin:", type({}.get) is types.BuiltinFunctionType, "self type:", type({}.get.__self__).__name__)
cwr = itertools.combinations_with_replacement
print("cwr immutable flag:", bool(cwr.__flags__ & (1 << 8)), resolves(cwr))
class U: pass
print("user class immutable flag:", bool(U.__flags__ & (1 << 8)))
lc = functools.lru_cache(maxsize=None)(lambda x: x)
print("lru_cache type:", type(lc).__name__, "is builtin fn:", type(lc) is types.BuiltinFunctionType)
for name, x in [("f8", np.float64(1)), ("M8", np.datetime64("2020-01-01")), ("U", np.str_("a")), ("void", np.void(b"ab"))]:
    print(f"scalar {name}: exact={type(x) is x.dtype.type} dtype-module={type(x.dtype).__module__} str={x.dtype.str}")
try:
    from numpy.dtypes import StringDType
    a = np.array(["a", "bb"], dtype=StringDType())
    print("StringDType module:", type(a.dtype).__module__, "hasobject:", a.dtype.hasobject, "str:", a.dtype.str)
except Exception as e:
    print("StringDType:", type(e).__name__, e)
try:
    import ml_dtypes
    for t in ("float8_e4m3fn", "bfloat16", "int4"):
        d = np.dtype(getattr(ml_dtypes, t))
        print(f"ml_dtypes {t}: dtype-module={type(d).__module__} str={d.str}")
except ImportError:
    print("ml_dtypes not installed")
try:
    import scipy.special as sp
    print("scipy.special.gamma type:", type(sp.gamma).__name__, "exact ufunc:", type(sp.gamma) is np.ufunc,
          "module:", getattr(sp.gamma, "__module__", "<absent>"), "resolves:", resolves(sp.gamma, True))
except ImportError:
    print("scipy not installed")
# co_varnames computed per access?
def f(a, b):
    c = a
    return c
co = f.__code__
print("co_varnames same object across accesses:", co.co_varnames is co.co_varnames,
      "co_code same object:", co.co_code is co.co_code, "co_names same:", co.co_names is co.co_names)
