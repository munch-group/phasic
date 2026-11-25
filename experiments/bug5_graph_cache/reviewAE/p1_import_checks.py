"""Review AE p1: simulate the plan's end-of-import checks (§2.2.3) on the
real installed package, at end of import AND after typical lazy imports.
Read-only; no cache files written."""
import os, sys, importlib.machinery, time

t0 = time.perf_counter()
import phasic
t1 = time.perf_counter()

pkg = os.path.dirname(os.path.abspath(phasic.__file__))
sufs = tuple(importlib.machinery.EXTENSION_SUFFIXES)
print("EXTENSION_SUFFIXES", sufs)
hashed = set()
for root, dirs, files in os.walk(pkg):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        if f.endswith(".py") or f.endswith(sufs):
            hashed.add(os.path.realpath(os.path.join(root, f)))
print("hashed files", len(hashed))
print("non-.py hashed:", sorted(h for h in hashed if not h.endswith(".py")))
so = phasic.phasic_pybind.__file__
print("phasic_pybind.__file__ in hashed:", os.path.realpath(so) in hashed)

def check(label):
    bad = []
    n = 0
    for name, m in list(sys.modules.items()):
        if m is None:
            continue
        if not (name == "phasic" or name.startswith("phasic.")):
            continue
        n += 1
        spec = getattr(m, "__spec__", None)
        origin = getattr(spec, "origin", None) if spec is not None else None
        src = origin if origin is not None else getattr(m, "__file__", None)
        if src is None:
            if name.startswith("phasic.phasic_pybind."):
                continue
            bad.append((name, "no source", type(m).__name__))
            continue
        if os.path.realpath(src) not in hashed:
            bad.append((name, src, type(m).__name__))
    print(f"[{label}] phasic.* modules: {n}; failing: {len(bad)}")
    for b in bad:
        print("   FAIL", b)

check("end of import")
# typical lazy usage: JAX-dependent symbols, svgd, etc.
for attr in ["SVGD", "svgd", "ffi_wrappers", "trace_elimination"]:
    try:
        getattr(phasic, attr)
    except Exception as e:
        print("getattr", attr, type(e).__name__, e)
import phasic.svgd, phasic.ffi_wrappers  # noqa
check("after lazy imports")
print("import time %.2fs" % (t1 - t0))
# also: pybind submodules
for name in sorted(sys.modules):
    if name.startswith("phasic.phasic_pybind"):
        m = sys.modules[name]
        print("  ", name, getattr(m, "__spec__", "NA"), getattr(m, "__file__", "NA"))
