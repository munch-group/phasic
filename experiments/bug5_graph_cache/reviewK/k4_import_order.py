"""reviewK k4: who first imports phasic.phasic_pybind during `import phasic`
(OMP_NUM_THREADS removed from env)? Prints the importing frames, so we can
see whether graph_cache's logger is the ONLY early trigger (v3.6 §2.2.3)."""
import os, sys, traceback
os.environ.pop("OMP_NUM_THREADS", None)
seen = {}


class Hook:
    def find_spec(self, name, path=None, target=None):
        if name == "phasic.phasic_pybind" and "first" not in seen:
            seen["first"] = True
            seen["omp_set"] = "OMP_NUM_THREADS" in os.environ
            stack = [f"{os.path.basename(f.filename)}:{f.lineno} {f.name}"
                     for f in traceback.extract_stack()
                     if "importlib" not in f.filename and "<frozen" not in f.filename]
            seen["stack"] = stack[-8:]
        return None


sys.meta_path.insert(0, Hook())
import phasic  # noqa
print("OMP set when phasic_pybind first imported:", seen.get("omp_set"))
for s in seen.get("stack", []):
    print("   ", s)
