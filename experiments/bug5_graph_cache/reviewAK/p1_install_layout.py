# reviewAK p1: installed package layout, loaded phasic.* modules, pybind API names
import os, sys, importlib.machinery
import phasic
from phasic import phasic_pybind
pkg = os.path.dirname(phasic.__file__)
print("pkg", pkg)
print("pybind file", getattr(phasic_pybind, "__file__", None))
exts = tuple(importlib.machinery.EXTENSION_SUFFIXES)
print("EXT", exts)
allf = []
for root, dirs, files in os.walk(pkg):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        allf.append(os.path.relpath(os.path.join(root, f), pkg))
py = [f for f in allf if f.endswith(".py")]
so = [f for f in allf if f.endswith(exts)]
other = sorted(set(allf) - set(py) - set(so))
print("n py", len(py), "subdirs with py", sorted({os.path.dirname(f) for f in py}))
print("so", so)
print("n other files", len(other), "sample", other[:15])
hashed = {os.path.realpath(os.path.join(pkg, f)) for f in py + so}
bad = []
for name, m in list(sys.modules.items()):
    if m is None or not (name == "phasic" or name.startswith("phasic.")):
        continue
    spec = getattr(m, "__spec__", None)
    src = spec.origin if (spec is not None and spec.origin is not None) else getattr(m, "__file__", None)
    if src is None:
        if not name.startswith("phasic.phasic_pybind."):
            bad.append((name, "no source", type(m).__name__))
        continue
    if os.path.realpath(src) not in hashed:
        bad.append((name, src))
print("loaded phasic.* not covered:", bad)
from phasic.phasic_pybind import Graph as _G
print("has get_dyn_ordering", hasattr(_G, "get_dyn_ordering"), "set", hasattr(_G, "set_dyn_ordering"))
try:
    print("_Graph(1).get_dyn_ordering() ->", _G(1).get_dyn_ordering())
except Exception as e:
    print("_Graph(1) failed:", type(e).__name__, e)
import numpy as np
print("numpy", np.__version__)
s = {t for t in np.sctypeDict.values() if getattr(t, "__module__", None) == "numpy"}
print("sctype classes", sorted(t.__name__ for t in s))
print("not-numpy-module", sorted({repr(t) for t in np.sctypeDict.values() if getattr(t, "__module__", None) != "numpy"}))
