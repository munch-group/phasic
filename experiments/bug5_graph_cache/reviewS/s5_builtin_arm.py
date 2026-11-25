"""reviewS s5: does the builtin/ufunc/dispatcher arm of plan 2.2.1 admit what it claims?"""
import sys, types, math, numpy as np, functools, itertools
def resolve(mod, qual):
    m = sys.modules.get(mod)
    if m is None: return None
    o = m
    for part in qual.split("."):
        o = getattr(o, part, None)
        if o is None: return None
    return o
def arm(x):
    t = type(x)
    if not (t is types.BuiltinFunctionType or t is np.ufunc or t is type(np.sum)):
        return "not this arm"
    s = getattr(x, "__self__", None)
    if not (s is None or isinstance(s, types.ModuleType)):
        return "refused: __self__"
    mod = getattr(x, "__module__", None)
    qual = getattr(x, "__qualname__", None) if t is not np.ufunc else getattr(x, "__name__", None)
    if mod is None or qual is None:
        return f"refused: module={mod!r} qual={qual!r}"
    r = resolve(mod, qual)
    return f"ADMIT {mod}.{qual}" if r is x else f"refused: resolves to {r!r}"
import scipy.special as sp
for name, x in [("math.sqrt", math.sqrt), ("len", len), ("np.exp", np.exp), ("np.add", np.add),
                ("np.sum", np.sum), ("np.linalg.solve", np.linalg.solve), ("np.minimum", np.minimum),
                ("np.isclose", np.isclose), ("np.zeros", np.zeros), ("np.array", np.array),
                ("np.concatenate", np.concatenate), ("np.random.default_rng", np.random.default_rng),
                ("sp.gamma", sp.gamma), ("sp.comb", sp.comb), ("{}.get", {}.get),
                ("frompyfunc", np.frompyfunc(lambda a: a, 1, 1)), ("np.abs", np.abs), ("abs", abs),
                ("np.round", np.round), ("np.log1p", np.log1p), ("np.prod", np.prod)]:
    print(f"{name:24s} {type(x).__name__:28s} {arm(x)}")
