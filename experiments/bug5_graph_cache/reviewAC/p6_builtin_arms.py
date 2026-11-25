"""reviewAC p6: attributes the builtin / immutable-type arms read."""
import sys, types, functools, itertools, collections, math, operator
import numpy as np

def resolve(mod, qual):
    obj = sys.modules.get(mod)
    for part in (qual or "").split("."):
        obj = getattr(obj, part, None)
    return obj

for name, x in [("cwr", itertools.combinations_with_replacement), ("np.float64", np.float64),
                ("np.ndarray", np.ndarray), ("defaultdict", collections.defaultdict),
                ("functools.partial", functools.partial), ("int", int), ("range", range),
                ("Counter", collections.Counter)]:
    imm = bool(x.__flags__ & (1 << 8))
    print(f"type {name:18} type(x) is type={type(x) is type} IMMUTABLE={imm} module={x.__module__} "
          f"qual={x.__qualname__} resolves={resolve(x.__module__, x.__qualname__) is x}")

for name, f in [("len", len), ("math.comb", math.comb), ("np.exp", np.exp), ("np.sum", np.sum),
                ("np.linalg.solve", np.linalg.solve), ("operator.add", operator.add),
                ("np.random.rand", np.random.rand), ("{}.get", {}.get)]:
    mod = getattr(f, "__module__", None); qual = getattr(f, "__qualname__", None)
    nm = getattr(f, "__name__", None); slf = getattr(f, "__self__", None)
    exact = (type(f) is types.BuiltinFunctionType) or (type(f) is np.ufunc) or (type(f) is type(np.sum))
    key = qual if type(f) is not np.ufunc else nm
    print(f"callable {name:16} exact_arm={exact} type={type(f).__name__} module={mod} qual={qual} "
          f"__self__type={type(slf).__name__} resolves={resolve(mod, key) is f if mod else None}")
