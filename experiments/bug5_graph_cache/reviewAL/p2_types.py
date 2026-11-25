"""reviewAL p2: check runtime facts the plan's type rules rely on (no Graph built)."""
import sys, types, math, itertools, functools, importlib
import numpy as np
print("python", sys.version.split()[0], "numpy", np.__version__)

# rule 2 allow-set exactly as the plan builds it
allow = frozenset(t for t in np.sctypeDict.values() if getattr(t, "__module__", None) == "numpy") \
    - {np.void, np.object_, np.longdouble, np.clongdouble}
print("allow-set:", sorted(t.__name__ for t in allow))
for name, arr in [("int default", np.array([1])), ("float", np.array([1.0])), ("bool", np.array([True])),
                  ("U", np.array(["a"])), ("S", np.array([b"a"])), ("M8", np.array(["2020-01-01"], "M8[D]")),
                  ("m8", np.array([1], "m8[s]")), ("intc", np.array([1], np.intc)), ("longlong", np.array([1], np.longlong)),
                  ("uintp", np.array([1], np.uintp)), ("half", np.array([1], np.float16)), ("c64", np.array([1], np.complex64)),
                  ("ulonglong", np.array([1], np.ulonglong)), ("byte", np.array([1], np.byte))]:
    t = arr.dtype.type
    print(f"  array {name:10s} dtype.type={t.__name__:12s} in allow-set: {any(t is a for a in allow)}")
for s in [np.float64(1), np.int64(1), np.str_("ab"), np.bytes_(b"a"), np.datetime64("2020"), np.bool_(True)]:
    print(f"  scalar {type(s).__name__:10s} in allow-set: {any(type(s) is a for a in allow)}  dtype.str={s.dtype.str}")

IMM = 1 << 8
for obj in [itertools.combinations_with_replacement, itertools.product, int, np.float64, functools.partial,
            np.ndarray, range, type(None)]:
    print(f"  type {obj.__module__}.{obj.__qualname__}: type(x) is type={type(obj) is type} immutable={bool(obj.__flags__ & IMM)}"
          f" resolves={getattr(sys.modules.get(obj.__module__), obj.__qualname__.split('.')[0], None) is obj}")

def resolve(mod, qn):
    o = sys.modules.get(mod)
    for part in (qn or "").split("."):
        o = getattr(o, part, None)
    return o

for f in [math.sqrt, len, np.exp, np.sum, np.linalg.solve, np.array, np.zeros, math.comb, np.where, np.add.reduce,
          {}.get, np.random.rand, np.isclose, np.log1p, np.searchsorted]:
    t = type(f)
    gate = (t is types.BuiltinFunctionType) or (t is np.ufunc) or (t is type(np.sum))
    mod = getattr(f, "__module__", None)
    qn = getattr(f, "__qualname__", None) if t is not np.ufunc else getattr(f, "__name__", None)
    slf = getattr(f, "__self__", None)
    slf_ok = slf is None or type(slf) is types.ModuleType
    res = resolve(mod, qn) is f if mod else False
    print(f"  callable {repr(f)[:45]:45s} type={t.__name__:28s} gate={gate} mod={mod} qn={qn} self_ok={slf_ok} resolves={res}")
