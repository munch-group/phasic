# reviewAD p1: which phasic.* modules are in sys.modules after import, and their spec origins
import sys, os
import phasic
pkg = os.path.dirname(phasic.__file__)
for name in sorted(m for m in sys.modules if m == "phasic" or m.startswith("phasic.")):
    mod = sys.modules[name]
    spec = getattr(mod, "__spec__", "MISSING")
    origin = getattr(spec, "origin", None) if spec not in (None, "MISSING") else None
    inside = bool(origin) and os.path.dirname(os.path.abspath(origin)).startswith(pkg)
    print(f"{name:45s} type={type(mod).__name__:12s} spec={'None' if spec is None else ('MISSING' if spec=='MISSING' else 'ok')} origin_in_pkg={inside} origin={origin}")
# after typical use: lazy imports
from phasic import Graph
import phasic.svgd, phasic.state_indexing
print("--- non-module or None-spec entries after more imports:")
for name in sorted(m for m in sys.modules if m.startswith("phasic.")):
    mod = sys.modules[name]
    if mod is None or getattr(mod, "__spec__", None) is None:
        print("PROBLEM", name, mod)
print("n phasic modules", sum(1 for m in sys.modules if m.startswith("phasic")))
