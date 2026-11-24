"""Review D probe 5: when is phasic.phasic_pybind first imported relative to
OMP_NUM_THREADS being set? Run with OMP_NUM_THREADS removed from the env.
argv[1] = label. The package imported is whatever PYTHONPATH points at
(installed master, or a scratch copy with graph_cache's logger made lazy)."""
import os, sys, importlib.abc
assert "OMP_NUM_THREADS" not in os.environ
seen = {}
class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "phasic.phasic_pybind" and "omp" not in seen:
            seen["omp"] = os.environ.get("OMP_NUM_THREADS")
            import traceback
            seen["who"] = [f"{f.filename.split('/')[-1]}:{f.lineno}" for f in traceback.extract_stack()[:-1]
                           if "phasic" in f.filename][-3:]
        return None
sys.meta_path.insert(0, Hook())
import phasic
print(sys.argv[1], "phasic from", os.path.dirname(phasic.__file__).split("/")[-3:],
      "| OMP_NUM_THREADS at first phasic_pybind import:", seen.get("omp"), "| importer:", seen.get("who"))
