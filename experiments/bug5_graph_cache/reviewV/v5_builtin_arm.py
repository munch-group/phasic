"""reviewV v5: apply the plan's §2.2.1 builtin / ufunc / dispatcher and
immutable-type name rules (as written) to common callables, to see what a
natural implementation admits or refuses."""
import sys, types, math, itertools, functools, collections, operator
import numpy as np
import scipy.special as sps
from math import comb

DISP = type(np.sum)

def resolve(mod, qn):
    m = sys.modules.get(mod) if isinstance(mod, str) else None
    if m is None:
        return None
    obj = m
    for part in qn.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj

def builtin_rule(x):
    if not (type(x) is types.BuiltinFunctionType or type(x) is np.ufunc or type(x) is DISP):
        return None
    slf = getattr(x, "__self__", None)
    if not (slf is None or isinstance(slf, types.ModuleType)):
        return "REFUSED (__self__ not None/module: %s)" % type(slf).__name__
    mod = getattr(x, "__module__", None)
    qn = getattr(x, "__qualname__", None) if type(x) is not np.ufunc else getattr(x, "__name__", None)
    if mod is None or qn is None:
        return f"REFUSED (module={mod!r} qualname={qn!r})"
    if resolve(mod, qn) is x:
        return f"ADMITTED by name {mod}.{qn}"
    return f"REFUSED (does not resolve: {mod}.{qn})"

def type_rule(x):
    if not isinstance(x, type):
        return None
    if not (x.__flags__ & (1 << 8)):
        return "REFUSED (mutable type)"
    mod, qn = getattr(x, "__module__", None), getattr(x, "__qualname__", None)
    return f"ADMITTED type {mod}.{qn}" if resolve(mod, qn) is x else f"REFUSED (no resolve {mod}.{qn})"

cands = {
    "math.sqrt": math.sqrt, "math.comb": math.comb, "len": len, "np.exp": np.exp,
    "np.sum": np.sum, "np.array": np.array, "np.zeros": np.zeros, "np.where": np.where,
    "np.concatenate": np.concatenate, "np.dot": np.dot, "np.linalg.solve": np.linalg.solve,
    "np.add.reduce": np.add.reduce, "np.random.binomial": np.random.binomial,
    "np.isclose": np.isclose, "np.arange": np.arange, "np.unique": np.unique,
    "sps.gamma": sps.gamma, "sps.binom": sps.binom, "sps.comb (py)": sps.comb,
    "operator.add": operator.add, "operator.itemgetter(0)": operator.itemgetter(0),
    "itertools.product": itertools.product, "itertools.cwr": itertools.combinations_with_replacement,
    "collections.Counter": collections.Counter, "collections.defaultdict": collections.defaultdict,
    "np.ndarray": np.ndarray, "np.float64": np.float64, "int": int, "range": range,
    "np.ndarray.sum (method_descriptor)": np.ndarray.sum, "str.join": str.join,
    "np.frompyfunc": np.frompyfunc(lambda a: a, 1, 1),
}
for k, x in cands.items():
    r = builtin_rule(x) or type_rule(x)
    if r is None:
        r = "python function (code-keyed)" if type(x) is types.FunctionType else f"REFUSED (other type {type(x).__name__})"
    print(f"{k:38s} {type(x).__name__:28s} {r}")
