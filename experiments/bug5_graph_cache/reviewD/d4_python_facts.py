"""Review D probe 4: Python/numpy/scipy facts the plan's fingerprint rules rely on.
No phasic graph builds, no files."""
import sys, math, itertools, functools, random, dis, types
import numpy as np
print(sys.version)
def res(o):
    m = getattr(o, "__module__", None); q = getattr(o, "__qualname__", None) or getattr(o, "__name__", None)
    mod = sys.modules.get(m) if isinstance(m, str) else None
    obj = mod
    try:
        for part in (q or "").split("."):
            obj = getattr(obj, part)
    except Exception:
        obj = None
    return m, q, obj is o
for name, o in [("len", len), ("math.sqrt", math.sqrt), ("np.exp", np.exp), ("np.linalg.solve", np.linalg.solve),
                ("np.array", np.array), ("random.random", random.random), ("{}.get", {}.get),
                ("frompyfunc", np.frompyfunc(lambda x: x, 1, 1)), ("np.random.rand", np.random.rand),
                ("operator.mul", __import__("operator").mul), ("np.sum", np.sum), ("np.where", np.where),
                ("np.isclose", np.isclose)]:
    s = getattr(o, "__self__", "<absent>")
    print(f"{name:16s} type={type(o).__name__:28s} __self__={type(s).__name__ if s!='<absent>' else s:12s} resolve={res(o)}")
try:
    import scipy.special as sp
    for n in ("gamma", "binom", "comb"):
        o = getattr(sp, n)
        print("scipy.special." + n, type(o).__name__, "has __module__:", hasattr(o, "__module__"),
              "has __qualname__:", hasattr(o, "__qualname__"), "__self__:", type(getattr(o, "__self__", None)).__name__)
except ImportError as e:
    print("scipy missing", e)
IMM = 1 << 8
class U: pass
for name, t in [("combinations_with_replacement", itertools.combinations_with_replacement), ("int", int),
                ("np.ndarray", np.ndarray), ("np.float64", np.float64), ("user class", U),
                ("functools.partial", functools.partial), ("dict", dict)]:
    print(f"{name:30s} immutable={bool(t.__flags__ & IMM)} resolve={res(t)}")
def f(state):
    return state[..., 0]
print("f consts", f.__code__.co_consts)
def g(M):
    return M[1:3, 0]
print("g consts", g.__code__.co_consts)
# comment line / spacing effect on co_code
src1 = "def h(a, b):\n    x = a + b\n    return x\n"
src2 = "def h(a, b):\n    # comment\n\n    x = a   +   b\n    return x\n"
src3 = "def h(a, b):\n    x = (a +\n         b)\n    return x\n"
cc = []
for s in (src1, src2, src3):
    ns = {}; exec(s, ns); cc.append(ns["h"].__code__.co_code)
print("co_code equal src1/src2 (comment+spacing):", cc[0] == cc[1], " src1/src3 (split line):", cc[0] == cc[2])
