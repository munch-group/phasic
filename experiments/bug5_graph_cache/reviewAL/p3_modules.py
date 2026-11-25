"""reviewAL p3: the plan's end-of-import coverage check, simulated on the real install.
Lists every loaded phasic.* module and whether its source file (spec.origin else __file__)
is among the files the plan hashes (*.py recursive excl. __pycache__, + EXTENSION_SUFFIXES).
Also lists phasic.* modules loaded LATER by ordinary use (svgd import, a Graph build)."""
import os, sys, importlib.machinery
assert os.environ["HOME"].startswith("/private/tmp/")
import phasic

pkg = os.path.dirname(phasic.__file__)
hashed = set()
for root, dirs, files in os.walk(pkg):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        if f.endswith(".py") or any(f.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES):
            hashed.add(os.path.join(root, f))
print("hashed files:", len(hashed))

def check(label):
    bad = []
    for name, m in list(sys.modules.items()):
        if m is None or not (name == "phasic" or name.startswith("phasic.")):
            continue
        spec = getattr(m, "__spec__", None)
        src = spec.origin if (spec is not None and spec.origin is not None) else getattr(m, "__file__", None)
        if src is None:
            ok = name.startswith("phasic.phasic_pybind.")
        else:
            ok = src in hashed
        if not ok:
            bad.append((name, src, type(m).__name__))
    n = sum(1 for k in sys.modules if k == "phasic" or k.startswith("phasic."))
    print(f"[{label}] phasic.* modules loaded: {n}; NOT covered: {bad}")

check("end of import")
import phasic.svgd  # noqa
from phasic import Graph
def cb(state, **kw):
    return [] if state[0] <= 1 else [[[state[0] - 1], 1.0]]
g = Graph(cb, ipv=[3], graph_cache=False)
g.expectation()
check("after svgd import + build")
print("top-level non-phasic.* names containing 'phasic':",
      sorted(k for k in sys.modules if "phasic" in k and not (k == "phasic" or k.startswith("phasic."))))
