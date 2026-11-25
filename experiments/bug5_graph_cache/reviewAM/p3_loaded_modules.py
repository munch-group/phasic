# End-of-import view of loaded phasic.* modules: origin/file vs package dir.
import sys, os
import phasic
pkg = os.path.dirname(phasic.__file__)
for name, m in sorted(list(sys.modules.items())):
    if m is None or not (name == "phasic" or name.startswith("phasic.")):
        continue
    spec = getattr(m, "__spec__", None)
    origin = getattr(spec, "origin", None) if spec is not None else None
    f = origin or getattr(m, "__file__", None)
    inside = f is not None and os.path.dirname(os.path.abspath(f)).startswith(pkg)
    print(f"{name:45s} spec={'Y' if spec else 'N'} origin={'Y' if origin else 'N'} inside={inside} {os.path.basename(f) if f else None}")
