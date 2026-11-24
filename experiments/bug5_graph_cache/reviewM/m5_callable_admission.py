"""reviewM m5: apply v3.8 section 2.2.1's admission rules for compiled callables
(builtin / ufunc / array-function dispatcher by exact type + name round trip;
immutable compiled types by IMMUTABLETYPE flag + name round trip) to callables
that ordinary callbacks capture, to look for false refusals.  A literal
re-implementation of the rule text; no phasic import."""
import sys, types, functools, math, itertools, operator, builtins
import numpy as np

DISPATCHER = type(np.sum)

def resolve(mod, qual):
    obj = sys.modules.get(mod)
    if obj is None:
        return None
    for part in qual.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj

def admit(x):
    t = type(x)
    if t is types.FunctionType:
        return "python function (code)"
    if t is types.MethodType:
        return "bound method"
    if t is functools.partial:
        return "partial"
    if t is types.BuiltinFunctionType or t is np.ufunc or t is DISPATCHER:
        self_ = getattr(x, "__self__", None)
        if not (self_ is None or isinstance(self_, types.ModuleType)):
            return f"REFUSED (builtin __self__ is {type(self_).__name__})"
        mod = getattr(x, "__module__", None)
        qual = getattr(x, "__name__", None) if t is np.ufunc else getattr(x, "__qualname__", None)
        if mod is None or qual is None:
            return f"REFUSED (missing __module__/__qualname__: {mod!r}, {qual!r})"
        r = resolve(mod, qual)
        return "name" if r is x else f"REFUSED (resolve {mod}.{qual} -> {type(r).__name__} is not x)"
    if isinstance(x, type):
        if not (x.__flags__ & (1 << 8)):
            return "REFUSED (mutable type)"
        mod, qual = getattr(x, "__module__", None), getattr(x, "__qualname__", None)
        r = resolve(mod, qual) if mod and qual else None
        return "type name" if r is x else f"REFUSED (type resolve {mod}.{qual} -> {r!r})"
    return f"REFUSED (type {t.__module__}.{t.__qualname__})"

cands = {
    "np.array": np.array, "np.zeros": np.zeros, "np.arange": np.arange, "np.sum": np.sum,
    "np.exp": np.exp, "np.log": np.log, "np.concatenate": np.concatenate, "np.where": np.where,
    "np.dot": np.dot, "np.isclose": np.isclose, "np.prod": np.prod, "np.cumsum": np.cumsum,
    "np.asarray": np.asarray, "np.empty": np.empty, "np.unique": np.unique,
    "np.float64": np.float64, "np.int32": np.int32, "np.ndarray": np.ndarray,
    "np.add.reduce": np.add.reduce, "np.random.default_rng": np.random.default_rng,
    "np.random.rand": np.random.rand, "np.linalg.solve": np.linalg.solve,
    "np.searchsorted": np.searchsorted, "np.count_nonzero": np.count_nonzero,
    "np.any": np.any, "np.all": np.all, "np.maximum": np.maximum, "np.sort": np.sort,
    "np.argsort": np.argsort, "np.nonzero": np.nonzero, "np.copy": np.copy,
    "np.ravel_multi_index": np.ravel_multi_index, "np.unravel_index": np.unravel_index,
    "np.binomial(rng method)": np.random.default_rng(0).binomial,
    "math.comb": math.comb, "math.factorial": math.factorial, "math.exp": math.exp,
    "len": len, "sum": sum, "max": max, "abs": abs, "round": round, "sorted": sorted,
    "range": range, "int": int, "float": float, "tuple": tuple, "list": list, "dict": dict,
    "zip": zip, "enumerate": enumerate, "map": map, "filter": filter,
    "itertools.product": itertools.product, "itertools.combinations": itertools.combinations,
    "itertools.permutations": itertools.permutations,
    "itertools.combinations_with_replacement": itertools.combinations_with_replacement,
    "operator.add": operator.add, "operator.mul": operator.mul,
    "operator.itemgetter(0)": operator.itemgetter(0), "str.join": str.join,
    "functools.reduce": functools.reduce, "collections.Counter": __import__("collections").Counter,
    "dict.get (unbound)": dict.get, "{}.get": {}.get, "list.append(unbound)": list.append,
}
try:
    import scipy.special as sps, scipy.stats as st
    cands.update({"scipy.special.comb": sps.comb, "scipy.special.binom": sps.binom,
                  "scipy.special.gammaln": sps.gammaln, "scipy.special.factorial": sps.factorial,
                  "scipy.stats.poisson.pmf": st.poisson.pmf})
except ImportError:
    pass
w = max(map(len, cands))
for k, v in cands.items():
    print(f"{k:<{w}}  {admit(v)}")
# function __module__ for exec'd / FunctionType-built functions
ns = {}
exec("def f(state):\n    return []\n", ns)
print("exec({}) function __module__:", repr(ns["f"].__module__))
g = types.FunctionType(ns["f"].__code__, {})
print("FunctionType(code, {}) __module__:", repr(g.__module__))
