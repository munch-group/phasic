"""reviewH p4: facts the builtin / immutable-type / module arms of §2.2.1 rely on."""
import sys, types, functools, itertools, math, operator
import numpy as np

IMM = 1 << 8


def resolve(mod, qual):
    m = sys.modules.get(mod) if isinstance(mod, str) else None
    if m is None:
        return None
    obj = m
    for part in qual.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def info(label, x):
    t = type(x)
    arm = ("builtin" if t is types.BuiltinFunctionType else
           "ufunc" if t is np.ufunc else
           "dispatcher" if t is type(np.sum) else
           "class" if isinstance(x, type) else "other")
    mod = getattr(x, "__module__", None)
    qual = getattr(x, "__qualname__", None)
    name = getattr(x, "__name__", None)
    slf = getattr(x, "__self__", "<absent>")
    flags = getattr(x, "__flags__", None) if isinstance(x, type) else None
    imm = None if flags is None else bool(flags & IMM)
    key = qual if arm != "ufunc" else name
    res = resolve(mod, key) if (mod and key) else None
    print(f"{label:42s} type={t.__name__:28s} arm={arm:10s} module={mod!r:22s} qual={qual!r:30s} "
          f"self={'module' if isinstance(slf, types.ModuleType) else type(slf).__name__ if slf != '<absent>' else 'absent':10s} "
          f"immutable={imm} resolves_same={res is x}")


for label, x in [
    ("len", len), ("math.sqrt", math.sqrt), ("operator.add", operator.add),
    ("np.exp", np.exp), ("np.add", np.add), ("np.sum", np.sum),
    ("np.linalg.solve", np.linalg.solve), ("np.where", np.where),
    ("np.array", np.array), ("np.zeros", np.zeros),
    ("np.arange", np.arange), ("np.random.rand", np.random.rand),
    ("frompyfunc", np.frompyfunc(lambda a: a, 1, 1)),
    ("itertools.combinations_with_replacement", itertools.combinations_with_replacement),
    ("itertools.product", itertools.product), ("itertools.combinations", itertools.combinations),
    ("dict", dict), ("np.int64", np.int64), ("np.ndarray", np.ndarray),
    ("functools.partial (type)", functools.partial),
    ("functools.reduce", functools.reduce),
    ("operator.itemgetter(1)", operator.itemgetter(1)),
    ("math.comb", math.comb), ("min", min), ("sorted", sorted), ("range (type)", range),
]:
    info(label, x)

try:
    import scipy.special as sp
    for label, x in [("scipy.special.gamma", sp.gamma), ("scipy.special.comb", sp.comb),
                     ("scipy.special.binom", sp.binom)]:
        info(label, x)
except Exception as e:
    print("scipy unavailable", e)

# Cython cyfunction check
try:
    import pandas._libs.lib as L
    f = L.is_scalar if hasattr(L, "is_scalar") else None
    print("pandas is_scalar type:", type(f).__name__, "is FunctionType:", type(f) is types.FunctionType,
          "has __code__:", hasattr(f, "__code__"))
except Exception as e:
    print("pandas lib", e)
