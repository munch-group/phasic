"""Review AH p2: who first imports phasic.phasic_pybind, and is OMP_NUM_THREADS
set then? Run with OMP_NUM_THREADS unset and HOME redirected. Also patches
logging.getLogger-free path: simulate the plan's lazy logger in graph_cache by
pre-seeding nothing; we only report the importer stack on master."""
import os, sys, traceback
assert "scratchpad" in os.environ.get("HOME", "")
assert "OMP_NUM_THREADS" not in os.environ
hits = []
class Hook:
    def find_spec(self, name, path=None, target=None):
        if name == "phasic.phasic_pybind" and not hits:
            st = [f"{f.filename.split('site-packages/')[-1]}:{f.lineno}" for f in traceback.extract_stack()[:-1]
                  if "phasic" in f.filename and "importlib" not in f.filename]
            hits.append((os.environ.get("OMP_NUM_THREADS"), st))
        return None
sys.meta_path.insert(0, Hook())
import phasic
omp, st = hits[0]
print("OMP_NUM_THREADS at first phasic_pybind import:", omp)
print("importer chain:")
for s in st:
    print("  ", s)
