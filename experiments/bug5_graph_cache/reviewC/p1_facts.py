"""Review C p1: language/library facts the v2.7 key design relies on.
No phasic Graph is built; run with HOME redirected anyway."""
import sys, itertools, functools, math, types
import numpy as np

print("python", sys.version.split()[0], "numpy", np.__version__)

# (a) immutable-type flag on itertools types
for t in (itertools.combinations_with_replacement, itertools.product,
          functools.partial, np.ndarray, np.float64, int):
    print("immutable", t.__qualname__, bool(t.__flags__ & (1 << 8)), t.__module__)

# (b) scipy.special
try:
    import scipy.special as sp
    print("scipy.gamma module", sp.gamma.__module__, type(sp.gamma).__name__)
except Exception as e:
    print("scipy", e)

# (c) co_code stability under specialization
def f(x):
    s = 0
    for i in range(x):
        s += i * 2.0
    return s
c0 = f.__code__.co_code
for _ in range(2000):
    f(50)
print("co_code stable after 2000 calls:", c0 == f.__code__.co_code)

# (d) whitespace / line-split edits
srcs = {
 "a": "def g(s, k=2):\n    r = s + k * 3\n    return [r, (s,\n  k)]\n",
 "b": "def g(s, k=2):\n\n    # comment\n    r = (s +\n         k * 3)\n    return [r, (s, k)]\n",
 "c": "def g(s, k=2):\n    r = s + k * 3\n\n\n    return [r, (s, k)]\n",
 "d": "def g(s, k=2):\n    if s:\n        pass\n    r = s + k * 3\n    return [r, (s, k)]\n",
}
codes = {}
for k, s in srcs.items():
    ns = {}
    exec(compile(s, "<%s>" % k, "exec"), ns)
    codes[k] = ns["g"].__code__
for k in "bcd":
    co, ca = codes[k], codes["a"]
    print("edit", k, "co_code equal:", co.co_code == ca.co_code,
          "exctable equal:", getattr(co, "co_exceptiontable", b"") == getattr(ca, "co_exceptiontable", b""),
          "consts equal:", co.co_consts == ca.co_consts)

# (e) builtins: __self__ and resolution
def resolves(o, attr="__qualname__"):
    mod = getattr(o, "__module__", None)
    name = getattr(o, attr, None) or getattr(o, "__name__", None)
    m = sys.modules.get(mod) if mod else None
    if m is None:
        return False
    obj = m
    for part in name.split("."):
        obj = getattr(obj, part, None)
    return obj is o
for name, o in [("len", len), ("math.sqrt", math.sqrt), ("np.exp", np.exp),
                ("np.linalg.solve", np.linalg.solve), ("np.random.random", np.random.random),
                ("np.add.reduce", np.add.reduce), ("np.sum", np.sum),
                ("np.random.default_rng().random", np.random.default_rng(1).random),
                ("str.join", str.join), ("'x'.join", "x".join), ("dict.fromkeys", dict.fromkeys),
                ("int.from_bytes", int.from_bytes), ("np.ufunc.reduce", np.ufunc.reduce),
                ("math.prod", math.prod), ("operator.mul", __import__("operator").mul)]:
    s = getattr(o, "__self__", "<none>")
    sname = type(s).__name__ if s != "<none>" else "<none>"
    print("builtin", name, type(o).__name__, "mod", getattr(o, "__module__", None),
          "qual", getattr(o, "__qualname__", None), "self", sname, "resolves", resolves(o))

# (f) StringDType arrays
try:
    from numpy.dtypes import StringDType
    a = np.array(["x" * 40, "y" * 40], dtype=StringDType())
    b = np.array(["z" * 40, "w" * 40], dtype=StringDType())
    print("StringDType str", a.dtype.str, "hasobject", a.dtype.hasobject, "fields", a.dtype.fields)
    print("StringDType tobytes equal for different content:", a.tobytes() == b.tobytes())
    a2 = np.array(["x" * 40, "y" * 40], dtype=StringDType())
    print("StringDType tobytes equal for same content:", a.tobytes() == a2.tobytes())
except Exception as e:
    print("StringDType", type(e).__name__, e)

# (g) str subclass keys / enum
import enum
class E(enum.IntEnum):
    A = 1
print("IntEnum exact int?", type(E.A) is int)

# (h) closure of super() method has __class__ cell
class K:
    def to_dict(self): return {}
    def cb(self, s):
        return super().__repr__()
print("super cell freevars", K.cb.__code__.co_freevars)
