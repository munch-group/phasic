"""Review AE p2: (a) who imports phasic.phasic_pybind first, and is
OMP_NUM_THREADS set at that moment (env has it removed by the runner);
(b) do stats (size, mtime_ns, ctime_ns, ino, dev) of the package's .py/.so
files change across `import phasic` + a small build (plan's S0 == S1
requirement); (c) numpy scalar allow-set as the plan specifies it."""
import os, sys, importlib.machinery, importlib.abc, traceback

os.environ.pop("OMP_NUM_THREADS", None)
import importlib.util
spec = importlib.util.find_spec("phasic")  # only to locate the dir for (b), before import
pkg = os.path.dirname(spec.origin)
sufs = tuple(importlib.machinery.EXTENSION_SUFFIXES)

def snap():
    out = {}
    for root, dirs, files in os.walk(pkg):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".py") or f.endswith(sufs):
                p = os.path.join(root, f)
                st = os.stat(p)
                out[os.path.relpath(p, pkg)] = (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, st.st_dev)
    return out

S0 = snap()
first = {}

class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "phasic.phasic_pybind" and not first:
            first["omp"] = os.environ.get("OMP_NUM_THREADS")
            stk = [f"{fs.filename.split('site-packages/')[-1]}:{fs.lineno}"
                   for fs in traceback.extract_stack()
                   if "site-packages/phasic" in fs.filename]
            first["stack"] = stk
        return None

sys.meta_path.insert(0, Hook())
import phasic
print("(a) first phasic_pybind import: OMP_NUM_THREADS =", first.get("omp"))
for s in first.get("stack", []):
    print("      ", s)

S1 = snap()
diff = [k for k in S0 if S0[k] != S1.get(k)] + [k for k in S1 if k not in S0]
print("(b) files:", len(S0), "changed across import:", diff)

g = phasic.Graph(lambda s: [] if s[0] <= 1 else [[[s[0] - 1], s[0] * (s[0] - 1) / 2]], ipv=[5])
print("    built", g.vertices_length(), "vertices; E[T] =", g.expectation())
S2 = snap()
diff2 = [k for k in S0 if S0[k] != S2.get(k)]
print("    changed after build:", diff2)

import numpy as np
allow = frozenset(t for t in np.sctypeDict.values() if getattr(t, "__module__", None) == "numpy") - {
    np.void, np.object_, np.longdouble, np.clongdouble}
print("(c) numpy", np.__version__, "allow-set:", sorted(t.__name__ for t in allow))
missing = [t.__name__ for t in (np.float64, np.float32, np.float16, np.int8, np.int16, np.int32, np.int64,
           np.uint8, np.uint64, np.bool_, np.complex64, np.complex128, np.str_, np.bytes_,
           np.datetime64, np.timedelta64, np.intc, np.longlong, np.ulonglong) if t not in allow]
print("    common types missing:", missing)
