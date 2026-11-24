"""reviewE e1: who imports phasic.phasic_pybind first, and is OMP_NUM_THREADS set then?
Mode 'master': observe only. Mode 'simfix': refuse the import when it comes from
logging_config.setup_logging while graph_cache is being imported (simulates the plan's
lazy logger in graph_cache) and report the next importer."""
import sys, os, traceback, importlib.abc
mode = sys.argv[1]
assert "OMP_NUM_THREADS" not in os.environ
rec = []
class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "phasic.phasic_pybind" or rec:
            return None
        st = traceback.extract_stack()[:-1]
        mods = [f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in st if "phasic" in f.filename]
        if mode == "simfix" and any("graph_cache.py" in m for m in mods) and "OMP_NUM_THREADS" not in os.environ:
            print("REFUSED (simulated lazy logger):", mods[-3:])
            raise ImportError("simulated")
        rec.append(("OMP set" if "OMP_NUM_THREADS" in os.environ else "OMP NOT set", mods))
        return None
sys.meta_path.insert(0, Hook())
import phasic
print(mode, rec[0][0])
for m in rec[0][1]: print("   ", m)
