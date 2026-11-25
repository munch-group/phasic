"""Review AH p4: check names / attributes the plan's admission rules rely on."""
import sys, math, itertools, types, functools
import numpy as np
def res(mod, qn):
    o = sys.modules.get(mod)
    for part in (qn or "").split("."):
        o = getattr(o, part, None)
    return o
for nm, f in [("np.exp", np.exp), ("np.add", np.add), ("np.sum", np.sum), ("np.linalg.solve", np.linalg.solve),
              ("math.sqrt", math.sqrt), ("len", len), ("np.random.default_rng", np.random.default_rng)]:
    mod = getattr(f, "__module__", None); qn = getattr(f, "__qualname__", None); n = getattr(f, "__name__", None)
    key = qn if qn is not None else n
    print(f"{nm:22s} type={type(f).__name__:28s} module={mod!r} qualname={qn!r} name={n!r} "
          f"has__self__={hasattr(f,'__self__')} self={type(getattr(f,'__self__',None)).__name__} "
          f"resolves(qualname)={res(mod, qn) is f} resolves(name)={res(mod, n) is f}")
t = itertools.combinations_with_replacement
print("cwr immutable flag", bool(t.__flags__ & (1 << 8)), type(t) is type, res(t.__module__, t.__qualname__) is t)
try:
    import scipy.special as sp
    print("scipy.special.gamma", type(sp.gamma).__name__, getattr(sp.gamma, "__module__", "MISSING"), getattr(sp.gamma, "__qualname__", "MISSING"), getattr(sp.gamma, "__name__", "MISSING"))
except Exception as e:
    print("scipy", e)
allow = frozenset(t for t in np.sctypeDict.values() if t.__module__ == "numpy") - {np.void, np.object_, np.longdouble, np.clongdouble}
print("allow-set size", len(allow), sorted(t.__name__ for t in allow))
print("type(np.dtype('int64'))", type(np.dtype("int64")), "metaclass of np.float64:", type(np.float64))
try:
    from numpy.dtypes import StringDType
    a = np.array(["a"], dtype=StringDType())
    print("StringDType .type", a.dtype.type, a.dtype.type in allow)
except Exception as e:
    print("StringDType", e)
print("np.str_ arr dtype.type in allow:", np.array(["ab"]).dtype.type in allow)
print("np.intc / np.int32 distinct", np.intc is np.int32, "np.longlong is np.int64", np.longlong is np.int64, "np.long is np.int64", getattr(np,"long",None) is np.int64)
print("ufunc __qualname__ np.exp:", getattr(np.exp, "__qualname__", "MISSING"))
