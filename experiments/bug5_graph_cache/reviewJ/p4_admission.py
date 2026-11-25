"""reviewJ p4: facts behind the plan's admission rules (no phasic Graph built).
Checks which objects ordinary callbacks capture fall under which rule."""
import sys, types, functools, itertools, math, collections
import numpy as np

IMM = 1 << 8
def resolves(o, attr="__qualname__"):
    m = getattr(o, "__module__", None); q = getattr(o, attr, None)
    if m is None or q is None or m not in sys.modules:
        return f"no ({m!r},{q!r})"
    cur = sys.modules[m]
    try:
        for part in q.split("."):
            cur = getattr(cur, part)
    except AttributeError:
        return f"no-attr ({m}.{q})"
    return "same" if cur is o else f"different ({m}.{q})"

objs = {
    "itertools.combinations_with_replacement": itertools.combinations_with_replacement,
    "itertools.product": itertools.product,
    "collections.defaultdict": collections.defaultdict,
    "int": int, "dict": dict, "np.ndarray": np.ndarray,
    "np.random.Generator": np.random.Generator,
    "collections.Counter": collections.Counter,
}
for name, o in objs.items():
    print(f"[type] {name}: immutable={bool(o.__flags__ & IMM)} resolves={resolves(o)}")

fns = {"math.sqrt": math.sqrt, "math.comb": math.comb, "len": len, "np.array": np.array,
       "np.exp": np.exp, "np.sum": np.sum, "np.linalg.solve": np.linalg.solve,
       "np.random.default_rng": np.random.default_rng, "np.random.rand": np.random.rand,
       "np.zeros": np.zeros, "np.where": np.where, "np.add.reduce": np.add.reduce}
for name, o in fns.items():
    t = type(o)
    gate = (t is types.BuiltinFunctionType) or (t is np.ufunc) or (t is type(np.sum))
    s = getattr(o, "__self__", None)
    selfkind = "None" if s is None else ("module" if isinstance(s, types.ModuleType) else type(s).__name__)
    attr = "__name__" if t is np.ufunc else "__qualname__"
    print(f"[fn] {name}: type={t.__name__} gate={gate} self={selfkind} resolves={resolves(o, attr)}")

try:
    import scipy.special as sp
    for name in ["gamma", "binom", "comb", "factorial"]:
        o = getattr(sp, name)
        print(f"[scipy] {name}: type={type(o).__name__} module={getattr(o,'__module__',None)!r} qualname={getattr(o,'__qualname__',None)!r}")
except Exception as e:
    print("scipy", e)

# constant types in co_consts of a typical callback (3.13)
def cb(state, rate=1.0, idx=(..., 0)):
    """doc"""
    out = []
    if state[..., 0] in {1, 2}:
        out.append(([state[0] - 1], [rate * 0.5, -0.0]))
    xs = [i for i in range(3)]
    return out
def walk(co, acc):
    for c in co.co_consts:
        acc.add(type(c).__name__)
        if isinstance(c, types.CodeType):
            walk(c, acc)
        if isinstance(c, tuple):
            for e in c: acc.add("tuple-elt:" + type(e).__name__)
    return acc
print("[consts]", sorted(walk(cb.__code__, set())), sys.version.split()[0], np.__version__)
