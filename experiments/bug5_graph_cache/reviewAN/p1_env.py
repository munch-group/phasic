# reviewAN p1: environment facts the plan relies on (no Graph built, no cache touched)
import os, sys
assert os.environ["HOME"].startswith("/private/tmp/"), os.environ["HOME"]
import numpy as np
import phasic, importlib.machinery, types, pathlib
print("python", sys.version.split()[0], "numpy", np.__version__)
allow = {t for t in np.sctypeDict.values() if getattr(t, "__module__", None) == "numpy"}
print("allow-set size", len(allow))
print("longdouble is float64:", np.longdouble is np.float64, " clongdouble is complex128:", np.clongdouble is np.complex128)
for name in ["float64","longdouble","int64","longlong","intc","bool","str_","bytes_","datetime64","timedelta64","void","object_"]:
    t = getattr(np, name); print(" ", name, t.__name__, t in allow)
pkg = pathlib.Path(phasic.__file__).parent
bad = []
for n, m in list(sys.modules.items()):
    if m is None or not (n == "phasic" or n.startswith("phasic.")): continue
    spec = getattr(m, "__spec__", None)
    src = spec.origin if (spec is not None and spec.origin is not None) else getattr(m, "__file__", None)
    inside = src is not None and pathlib.Path(src).resolve().is_relative_to(pkg.resolve())
    if not inside: bad.append((n, src, type(m).__name__))
print("phasic.* modules not under package dir:", bad)
print("EXTENSION_SUFFIXES", importlib.machinery.EXTENSION_SUFFIXES)
# itertools immutable flag and resolution
import itertools
c = itertools.combinations_with_replacement
print("cwr immutable:", bool(c.__flags__ & (1<<8)), c.__module__, c.__qualname__)
print("np.sum type", type(np.sum), "has __self__", hasattr(np.sum, "__self__"))
# base pybind get_dyn_ordering exists
print("_Graph has get_dyn_ordering:", hasattr(phasic._Graph, "get_dyn_ordering"), hasattr(phasic._Graph, "dyn_ordering"))
