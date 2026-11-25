"""Review AG p5: the end-of-import coverage rule of §2.2.3 evaluated on the
real install, at end of `import phasic` AND after typical later imports
(svgd / jax paths) -- does any phasic.* module escape the rule?"""
import importlib.machinery, os, sys
import phasic

pkg = os.path.dirname(phasic.__file__)
files = set()
for root, dirs, fs in os.walk(pkg):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in fs:
        if f.endswith(".py") or any(f.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES):
            files.add(os.path.join(root, f))


def audit(label):
    bad = []
    n = 0
    for name, mod in list(sys.modules.items()):
        if mod is None or not (name == "phasic" or name.startswith("phasic.")):
            continue
        n += 1
        spec = getattr(mod, "__spec__", None)
        src = spec.origin if (spec is not None and spec.origin is not None) else getattr(mod, "__file__", None)
        if src is None:
            if not name.startswith("phasic.phasic_pybind."):
                bad.append((name, "no source"))
            continue
        if src not in files:
            bad.append((name, src))
    print(f"{label}: {n} phasic modules, violations: {bad}")


audit("end of import phasic")
import phasic.svgd, phasic.ffi_wrappers  # noqa
try:
    phasic.Graph  # noqa
    import jax  # noqa
except Exception as e:
    print("jax import:", e)
audit("after svgd/ffi_wrappers/jax imports")
print("extension __file__ in files:", phasic.phasic_pybind.__file__ in files)
