"""Review AE p4: evaluate the plan's §2.2.1 name-keyed admission rules
(builtin / ufunc / dispatcher arm; immutable compiled type arm) on common
objects. No phasic import needed beyond numpy/scipy."""
import sys, types, math, operator, functools, itertools, collections, decimal
import numpy as np

DISP = type(np.sum)

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

def builtin_arm(x):
    t = type(x)
    if not (t is types.BuiltinFunctionType or t is np.ufunc or t is DISP):
        return "not-this-arm"
    s = getattr(x, "__self__", None)
    if not (s is None or type(s) is types.ModuleType):
        return "REFUSED(__self__=%s)" % type(s).__name__
    mod = getattr(x, "__module__", None)
    qual = getattr(x, "__name__", None) if t is np.ufunc else getattr(x, "__qualname__", None)
    if mod is None or qual is None:
        return "REFUSED(no module/qualname)"
    return "admitted %s.%s" % (mod, qual) if resolve(mod, qual) is x else "REFUSED(no resolve %s.%s)" % (mod, qual)

def type_arm(x):
    if type(x) is not type:
        return "REFUSED(metaclass %s)" % type(type(x)).__name__
    if not (x.__flags__ & (1 << 8)):
        return "REFUSED(mutable)"
    return "admitted" if resolve(x.__module__, x.__qualname__) is x else "REFUSED(no resolve %s.%s)" % (x.__module__, x.__qualname__)

for name, f in [("len", len), ("max", max), ("math.comb", math.comb), ("np.exp", np.exp), ("np.abs", np.abs),
                ("np.sum", np.sum), ("np.linalg.solve", np.linalg.solve), ("np.array", np.array),
                ("np.zeros", np.zeros), ("np.arange", np.arange), ("np.where", np.where),
                ("np.random.binomial", np.random.binomial), ("operator.mul", operator.mul),
                ("functools.reduce", functools.reduce), ("np.add.reduce", np.add.reduce),
                ("np.frompyfunc", np.frompyfunc(lambda a: a, 1, 1))]:
    print(f"{name:22s} type={type(f).__name__:28s} -> {builtin_arm(f)}")

for name, c in [("functools.partial", functools.partial), ("itertools.combinations_with_replacement", itertools.combinations_with_replacement),
                ("itertools.product", itertools.product), ("collections.defaultdict", collections.defaultdict),
                ("collections.OrderedDict", collections.OrderedDict), ("collections.deque", collections.deque),
                ("collections.Counter", collections.Counter), ("decimal.Decimal", decimal.Decimal),
                ("dict", dict), ("float", float), ("np.float64", np.float64), ("np.ndarray", np.ndarray),
                ("np.int64", np.int64), ("np.longlong", np.longlong), ("np.bool_", np.bool_)]:
    print(f"{name:40s} -> {type_arm(c)}")
try:
    import scipy.special as sp
    print("scipy.special.gamma ->", builtin_arm(sp.gamma), "| comb type:", type(sp.comb).__name__)
except Exception as e:
    print("scipy", e)
