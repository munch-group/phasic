"""reviewAF p1: check names / runtime conditions the plan relies on (no cache files written)."""
import os, sys, types, functools, itertools, math, importlib.machinery
assert os.environ["HOME"].startswith("/private/tmp/"), os.environ["HOME"]
import numpy as np
import phasic
from phasic import _Graph
pkg = os.path.dirname(phasic.__file__)
print("numpy", np.__version__, "python", sys.version.split()[0])
print("pkg", pkg)
g = _Graph(1)
print("_Graph(1).get_dyn_ordering:", g.get_dyn_ordering(), hasattr(g, "set_dyn_ordering"))
print("Graph has dyn_ordering prop:", isinstance(getattr(phasic.Graph, "dyn_ordering", None), property))
print("phasic_pybind.__file__", phasic.phasic_pybind.__file__)
print("EXTENSION_SUFFIXES", importlib.machinery.EXTENSION_SUFFIXES)
# loaded phasic.* modules
bad = []
for name, m in list(sys.modules.items()):
    if m is None or not (name == "phasic" or name.startswith("phasic.")):
        continue
    spec = getattr(m, "__spec__", None)
    origin = spec.origin if (spec is not None and spec.origin is not None) else getattr(m, "__file__", None)
    inside = origin is not None and os.path.realpath(origin).startswith(os.path.realpath(pkg) + os.sep)
    if not inside:
        bad.append((name, type(m).__name__, spec, origin))
print("phasic.* modules not inside pkg:", bad)
print("n phasic.* modules", sum(1 for n in sys.modules if n.startswith("phasic.")))
# package files
allf = []
for root, dirs, files in os.walk(pkg):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        allf.append(os.path.relpath(os.path.join(root, f), pkg))
print("non-py files:", sorted(f for f in allf if not f.endswith(".py")))
print("subdirs with .py:", sorted({os.path.dirname(f) for f in allf if f.endswith(".py") and os.path.dirname(f)}))
# metaclasses
from phasic.state_indexing import StateIndexer, Property, PropertySet, StateSpace, Slot
for t in (StateIndexer, Property, PropertySet, StateSpace, Slot, np.float64, np.ndarray, np.ufunc, type(np.sum), functools.partial, types.FunctionType):
    print("metaclass", t.__name__, type(t) is type)
# allow set
allow = frozenset(t for t in set(np.sctypeDict.values()) if t.__module__ == "numpy") - {np.void, np.object_, np.longdouble, np.clongdouble}
print("allow-set:", sorted(t.__name__ for t in allow))
# builtin arms
for f in (np.exp, np.sum, np.linalg.solve, math.sqrt, len, np.cumprod, math.comb):
    mod = getattr(f, "__module__", None); qn = getattr(f, "__qualname__", None); nm = getattr(f, "__name__", None)
    slf = getattr(f, "__self__", None)
    res = None
    try:
        o = sys.modules[mod]
        for part in (qn or nm).split("."):
            o = getattr(o, part)
        res = o is f
    except Exception as e:
        res = repr(e)
    print("builtin", nm, type(f).__name__, mod, qn, type(slf).__name__, "resolves:", res)
print("cwr immutable:", bool(itertools.combinations_with_replacement.__flags__ & (1 << 8)))
for t in (int, np.int64, np.ndarray):
    print("immutable", t.__name__, bool(t.__flags__ & (1 << 8)))
