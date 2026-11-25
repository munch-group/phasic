"""reviewG g1: who imports phasic.phasic_pybind first, and is OMP_NUM_THREADS set then?
Usage: python g1_import_order.py [pkgdir]   (pkgdir prepended to sys.path, e.g. a patched copy)
Run with OMP_NUM_THREADS removed from env.
"""
import os, sys, traceback, importlib.abc
if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])
os.environ.pop("OMP_NUM_THREADS", None)
rec = {}
class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "phasic.phasic_pybind" and "omp" not in rec:
            rec["omp"] = os.environ.get("OMP_NUM_THREADS")
            st = [f"{os.path.basename(f.filename)}:{f.lineno}" for f in traceback.extract_stack()
                  if "phasic" in f.filename and "importlib" not in f.filename]
            rec["stack"] = st
        return None
sys.meta_path.insert(0, Hook())
import phasic
print("phasic from:", os.path.dirname(phasic.__file__))
print("OMP_NUM_THREADS at first phasic_pybind import:", rec.get("omp"))
print("stack:", " <- ".join(reversed(rec.get("stack", []))))
