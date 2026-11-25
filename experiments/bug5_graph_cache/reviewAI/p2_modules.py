"""Review AI probe 2: which phasic.* modules are loaded after import, from where;
and whether common operations load further phasic.* modules (all must be
inside the package dir)."""
import os, sys, types, importlib.machinery
import phasic

pkg = os.path.dirname(os.path.realpath(phasic.__file__))
print("pkg dir:", pkg)
def origin(m):
    sp = getattr(m, "__spec__", None)
    if sp is not None and getattr(sp, "origin", None) is not None:
        return sp.origin, "spec"
    return getattr(m, "__file__", None), "file"
bad = []
for name, m in sorted(list(sys.modules.items())):
    if not (name == "phasic" or name.startswith("phasic.")) or m is None:
        continue
    o, how = origin(m)
    inside = o is not None and os.path.realpath(o).startswith(pkg + os.sep)
    kind = type(m).__name__
    if not inside or kind != "module":
        bad.append((name, kind, o, how))
print("loaded phasic modules:", sum(1 for n in sys.modules if n == "phasic" or n.startswith("phasic.")))
print("outside/odd:", bad)
# names whose origin is not a .py or extension
exts = tuple(importlib.machinery.EXTENSION_SUFFIXES)
odd_suffix = []
for name, m in sorted(list(sys.modules.items())):
    if m is None or not (name == "phasic" or name.startswith("phasic.")):
        continue
    o, _ = origin(m)
    if o is not None and not (o.endswith(".py") or o.endswith(exts)):
        odd_suffix.append((name, o))
print("non-.py/.so origins:", odd_suffix)
