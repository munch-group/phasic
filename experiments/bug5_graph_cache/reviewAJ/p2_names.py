"""reviewAJ p2: do the names and runtime checks the plan relies on exist / hold?"""
import os, sys, types, functools, itertools, importlib.machinery, math
assert "scratchpad" in os.environ["HOME"]
import numpy as np
print("python", sys.version.split()[0], "numpy", np.__version__)
print("has sctypeDict:", hasattr(np, "sctypeDict"))
excl = {np.void, np.object_, np.longdouble, np.clongdouble}
allow = frozenset(t for t in np.sctypeDict.values() if getattr(t, "__module__", None) == "numpy") - excl
print("allow-set size", len(allow), sorted(t.__name__ for t in allow))
print("np.bool in allow", np.bool_ in allow, "float64", np.float64 in allow, "str_", np.str_ in allow,
      "datetime64", np.datetime64 in allow, "intc", np.intc in allow, "longlong", np.longlong in allow,
      "ulonglong", np.ulonglong in allow, "half", np.float16 in allow)
# every array dtype a user would plausibly pass: is its .type in the allow-set?
for dt in ["i1","i2","i4","i8","u1","u2","u4","u8","f2","f4","f8","c8","c16","?","M8[s]","m8[s]","U3","S3",
           np.intc, np.uintc, np.longlong, np.ulonglong, np.int_, np.uint, np.intp, np.uintp]:
    d = np.dtype(dt)
    print("  dtype", repr(dt), d.str, d.type.__name__, "allowed" if any(d.type is t for t in allow) else "REFUSED")
# metaclass check satisfiable for every admitted class?
bad = [t for t in allow if type(t) is not type]
print("allow-set classes with non-type metaclass:", bad)
for t in (np.ndarray, np.ufunc, type(np.sum), functools.partial, types.FunctionType, types.MethodType):
    print("  metaclass", t.__name__, type(t) is type)
print("type(np.sum)", type(np.sum), "np.linalg.solve", type(np.linalg.solve))
for f in (np.exp, np.sum, np.linalg.solve, math.sqrt, len):
    m = getattr(f, "__module__", None); q = getattr(f, "__qualname__", None) or getattr(f, "__name__", None)
    obj = sys.modules.get(m) if m else None
    for part in (q or "").split("."):
        obj = getattr(obj, part, None) if obj is not None else None
    print("  resolve", getattr(f, "__name__", f), m, q, "self:", type(getattr(f, "__self__", None)).__name__, "same:", obj is f)
IMM = 1 << 8
for t in (itertools.combinations_with_replacement, int, np.float64, np.ndarray, functools.partial):
    print("  immutable flag", t.__qualname__, bool(t.__flags__ & IMM), t.__module__)
try:
    import scipy.special as sp
    print("scipy.special.gamma __module__", getattr(sp.gamma, "__module__", "<absent>"), "__qualname__",
          getattr(sp.gamma, "__qualname__", "<absent>"), "type is ufunc", type(sp.gamma) is np.ufunc)
except ImportError:
    print("no scipy")
print("EXTENSION_SUFFIXES", importlib.machinery.EXTENSION_SUFFIXES)
from phasic.phasic_pybind import Graph as _Graph
print("_Graph(1).get_dyn_ordering()", _Graph(1).get_dyn_ordering())
import phasic
print("phasic_pybind __file__", phasic.phasic_pybind.__file__)
print("pkg dir", os.path.dirname(phasic.__file__))
# pybind submodules spec / file
for n, m in list(sys.modules.items()):
    if n.startswith("phasic.") and m is not None:
        spec = getattr(m, "__spec__", None)
        origin = getattr(spec, "origin", None) if spec is not None else None
        src = origin if origin is not None else getattr(m, "__file__", None)
        if src is None or not str(src).startswith(os.path.dirname(phasic.__file__)):
            print("  module outside pkg / no source:", n, src)
