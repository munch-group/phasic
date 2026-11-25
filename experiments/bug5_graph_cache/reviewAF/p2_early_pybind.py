"""reviewAF p2: list every attempt to import phasic.phasic_pybind while OMP_NUM_THREADS is unset.

The hook refuses the import (ImportError) while OMP_NUM_THREADS is unset, so
every early importer shows up (not just the first). Run with OMP_NUM_THREADS removed.
"""
import os, sys, traceback
assert "OMP_NUM_THREADS" not in os.environ
attempts = []

class Hook:
    def find_spec(self, name, path=None, target=None):
        if name == "phasic.phasic_pybind" and "OMP_NUM_THREADS" not in os.environ:
            st = [f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in traceback.extract_stack()[:-1]
                  if "importlib" not in f.filename]
            attempts.append(st[-6:])
            raise ImportError("blocked early phasic_pybind import (probe)")
        return None

sys.meta_path.insert(0, Hook())
try:
    import phasic  # noqa
    print("import ok")
except Exception as e:
    print("import failed:", type(e).__name__, e)
print("early attempts:", len(attempts))
for a in attempts:
    print("---")
    for line in a:
        print("   ", line)
