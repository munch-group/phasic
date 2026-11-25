import os, sys, importlib.abc
assert "OMP_NUM_THREADS" not in os.environ, "env must not set OMP_NUM_THREADS"
seen = {}
class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in seen and name in ("phasic.phasic_pybind", "phasic.callback_hash", "phasic.graph_cache", "phasic.cache", "phasic.trace_serialization", "phasic.logging_config"):
            seen[name] = ("OMP_NUM_THREADS" in os.environ, sorted(k for k in sys.modules if k.startswith("phasic.")))
        return None
sys.meta_path.insert(0, Hook())
import phasic
for k, (omp, mods) in seen.items():
    print("RESULT", k, "omp_set_at_import=", omp, "phasic_mods_loaded_before=", len(mods))
print("RESULT pybind_before_omp:", seen.get("phasic.phasic_pybind", (None,))[0] is False)
