"""Review I probe 2: who first imports phasic.phasic_pybind during `import phasic`,
and is OMP_NUM_THREADS set at that moment? Run with OMP_NUM_THREADS removed."""
import os, sys, traceback
os.environ.pop("OMP_NUM_THREADS", None)

class Hook:
    done = False
    def find_spec(self, name, path=None, target=None):
        if name == "phasic.phasic_pybind" and not Hook.done:
            Hook.done = True
            print("first phasic_pybind import; OMP_NUM_THREADS set:", "OMP_NUM_THREADS" in os.environ)
            st = [f for f in traceback.extract_stack() if "phasic" in f.filename and "importlib" not in f.filename]
            for f in st:
                print("  ", f.filename.split("site-packages/")[-1], f.lineno, f.line)
        return None

sys.meta_path.insert(0, Hook())
import phasic
print("imported phasic", phasic.__file__.split("site-packages/")[-1])
