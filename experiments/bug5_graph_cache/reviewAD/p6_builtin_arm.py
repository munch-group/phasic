# reviewAD p6: the builtin/ufunc/dispatcher arm as specified in plan v5.5 s2.2.1, applied to
# compiled callables a callback might capture in a cell. Admitted iff exact type gate passes,
# __self__ is None or exactly a module, and __module__ + __qualname__ (__name__ for ufuncs)
# resolve through sys.modules to the very same object.
import sys, types, math, operator, itertools, functools
import numpy as np
DISP = type(np.sum)
def resolve(modname, qual):
    m = sys.modules.get(modname)
    if m is None: return None
    obj = m
    for part in qual.split("."):
        obj = getattr(obj, part, None)
        if obj is None: return None
    return obj
def arm(x):
    t = type(x)
    if not (t is types.BuiltinFunctionType or t is np.ufunc or t is DISP):
        return "not this arm (%s)" % t.__name__
    s = getattr(x, "__self__", None)
    if not (s is None or type(s) is types.ModuleType):
        return "REFUSED (__self__ %s)" % type(s).__name__
    mod = getattr(x, "__module__", None)
    q = getattr(x, "__name__", None) if t is np.ufunc else getattr(x, "__qualname__", None)
    if type(mod) is not str or type(q) is not str:
        return "REFUSED (module=%r, name=%r)" % (mod, q)
    return "admitted %s.%s" % (mod, q) if resolve(mod, q) is x else "REFUSED (resolves elsewhere: %s.%s)" % (mod, q)
cands = {"len": len, "max": max, "math.comb": math.comb, "math.exp": math.exp, "operator.mul": operator.mul,
         "np.exp": np.exp, "np.add": np.add, "np.sum": np.sum, "np.linalg.solve": np.linalg.solve,
         "np.array": np.array, "np.zeros": np.zeros, "np.where": np.where, "np.dot": np.dot,
         "np.concatenate": np.concatenate, "np.isclose": np.isclose, "np.prod": np.prod,
         "np.random.random": np.random.random, "np.searchsorted": np.searchsorted, "np.arange": np.arange,
         "np.empty": np.empty}
try:
    import scipy.special as sp
    cands.update({"scipy.special.gamma": sp.gamma, "scipy.special.comb": sp.comb, "scipy.special.binom": sp.binom})
except Exception: pass
for k, v in cands.items():
    print(f"{k:22s} {arm(v)}")
print("np.__version__", np.__version__)
