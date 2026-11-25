# Which import first loads phasic.phasic_pybind, and is OMP_NUM_THREADS set then?
import os, sys, traceback
os.environ.pop("OMP_NUM_THREADS", None)
class Hook:
    done = False
    def find_spec(self, name, path=None, target=None):
        if name == "phasic.phasic_pybind" and not Hook.done:
            Hook.done = True
            print("OMP set at phasic_pybind import:", "OMP_NUM_THREADS" in os.environ)
            st = [f"{f.filename.split('site-packages/')[-1]}:{f.lineno}" for f in traceback.extract_stack()[:-1]]
            print("stack:", " <- ".join(reversed(st[-8:])))
        return None
sys.meta_path.insert(0, Hook())
import phasic
print("logging configured:", phasic.logging_config._logging_configured)
import logging
lg = logging.getLogger("phasic")
print("phasic logger handlers:", lg.handlers, "propagate", lg.propagate)
