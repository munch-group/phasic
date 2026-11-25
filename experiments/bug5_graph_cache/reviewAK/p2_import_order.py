# reviewAK p2: on master, who first imports phasic.phasic_pybind, and is OMP_NUM_THREADS set then?
import os, sys, traceback
os.environ.pop("OMP_NUM_THREADS", None)
class Hook:
    done = False
    def find_spec(self, name, path=None, target=None):
        if name == "phasic.phasic_pybind" and not Hook.done:
            Hook.done = True
            print("OMP set at first pybind import:", "OMP_NUM_THREADS" in os.environ)
            st = [f"{os.path.basename(f.filename)}:{f.lineno}" for f in traceback.extract_stack()
                  if "phasic" in f.filename and "importlib" not in f.filename]
            print("stack:", st)
        return None
sys.meta_path.insert(0, Hook())
import phasic
