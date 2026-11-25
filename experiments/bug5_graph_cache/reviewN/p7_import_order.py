"""Who imports phasic.phasic_pybind first, and is OMP_NUM_THREADS set then?
Run with OMP_NUM_THREADS removed from the environment. Reports every phasic
module frame on the stack at that moment, so we can see whether anything
OTHER than graph_cache's module-level get_logger triggers it early."""
import os, sys, traceback
os.environ.pop("OMP_NUM_THREADS", None)

class Hook:
    done = False
    def find_spec(self, name, path=None, target=None):
        if name == "phasic.phasic_pybind" and not Hook.done:
            Hook.done = True
            print("phasic_pybind first imported; OMP_NUM_THREADS set:",
                  "OMP_NUM_THREADS" in os.environ)
            for fr in traceback.extract_stack()[:-1]:
                if "phasic" in fr.filename and "site-packages" in fr.filename:
                    print("   ", fr.filename.split("site-packages/")[1], fr.lineno, fr.name, "|", (fr.line or "")[:70])
        return None

sys.meta_path.insert(0, Hook())
import phasic  # noqa
