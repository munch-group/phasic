"""Review I probe 1: numpy facts the v3.4 value rules depend on (no Graph built)."""
import sys, numpy as np, itertools, functools, math, types
print("python", sys.version.split()[0], "numpy", np.__version__)

# StringDType (numpy >= 2.0): does rule 3 refuse it? what does tobytes() hold?
try:
    T = np.dtypes.StringDType()
    a = np.array(["x" * 40 + "1"], dtype=T)
    b = np.array(["x" * 40 + "2"], dtype=T)
    s = np.array(["ab"], dtype=T)
    print("StringDType str", repr(T.str), "hasobject", T.hasobject, "fields", T.fields,
          "type(a) is ndarray", type(a) is np.ndarray)
    print("short 'ab' bytes:", s.tobytes())
    print("long bytes a:", a.tobytes().hex())
    print("long bytes b:", b.tobytes().hex())
    print("long content in bytes?", b"xxxx" in a.tobytes())
    # copy of same content: are bytes equal?
    a2 = np.array(["x" * 40 + "1"], dtype=T)
    print("same content, equal bytes?", a.tobytes() == a2.tobytes())
    # scalar from StringDType array: what type?
    print("a[0] type", type(a[0]))
except Exception as e:
    print("StringDType probe failed:", type(e).__name__, e)

# ufunc module attrs
for f in (np.exp, np.add, np.sum, np.linalg.solve, math.sqrt, len):
    print(type(f).__name__, getattr(f, "__module__", None), getattr(f, "__qualname__", None),
          getattr(f, "__name__", None), type(getattr(f, "__self__", None)).__name__)

IMM = 1 << 8
for t in (itertools.combinations_with_replacement, functools.partial, np.float64,
          np.ndarray, dict, np.random.Generator, types.SimpleNamespace):
    print(t, bool(t.__flags__ & IMM), t.__module__, t.__qualname__)

# datetime / longdouble padding (miss-only issues)
x = np.longdouble(1.5); y = np.longdouble(1.5)
print("longdouble itemsize", x.itemsize, "equal bytes", x.tobytes() == y.tobytes())
