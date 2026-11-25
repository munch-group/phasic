"""Review AH p5: prototype the end-of-import coverage check (plan 2.2.3) on the
real install: every loaded phasic.* module's source must be a hashed file."""
import os, sys, importlib.machinery as m
assert "scratchpad" in os.environ.get("HOME", "")
import phasic
pkg = os.path.dirname(phasic.__file__)
hashed = set()
for root, dirs, files in os.walk(pkg):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        if f.endswith(".py") or any(f.endswith(s) for s in m.EXTENSION_SUFFIXES):
            hashed.add(os.path.join(root, f))
print("hashed files:", len(hashed), "ext:", [os.path.basename(h) for h in hashed if not h.endswith(".py")])
print("pybind __file__ in hashed:", phasic.phasic_pybind.__file__ in hashed)
bad = []
for name, mod in list(sys.modules.items()):
    if mod is None or not (name == "phasic" or name.startswith("phasic.")):
        continue
    spec = getattr(mod, "__spec__", None)
    src = spec.origin if spec is not None and spec.origin is not None else getattr(mod, "__file__", None)
    if src is None:
        if not name.startswith("phasic.phasic_pybind."):
            bad.append((name, "no source"))
        continue
    if src not in hashed:
        bad.append((name, src))
print("loaded phasic modules:", sum(1 for n in sys.modules if n == "phasic" or n.startswith("phasic.")))
print("coverage failures:", bad)
# after lazily importing heavier submodules too
import phasic.svgd, phasic.trace_elimination  # noqa
print("after lazy imports, failures:", [(n, getattr(getattr(mo,'__spec__',None),'origin',None)) for n, mo in sys.modules.items()
      if mo is not None and n.startswith("phasic.") and ((getattr(mo,'__spec__',None) and mo.__spec__.origin) or getattr(mo,'__file__',None)) not in hashed and not n.startswith("phasic.phasic_pybind.")])
