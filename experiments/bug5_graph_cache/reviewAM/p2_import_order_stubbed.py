# Simulate the planned lazy logger: stub phasic.graph_cache (no get_logger at
# module level) and see whether anything else loads phasic_pybind before
# OMP_NUM_THREADS is set, and whether phasic's log handler is still configured.
import os, sys, types, traceback
os.environ.pop("OMP_NUM_THREADS", None)
stub = types.ModuleType("phasic.graph_cache")
class GraphCache: pass
stub.GraphCache = GraphCache
stub.get_graph_cache_stats = lambda: {}
stub.print_graph_cache_info = lambda: None
sys.modules["phasic.graph_cache"] = stub
class Hook:
    done = False
    def find_spec(self, name, path=None, target=None):
        if name == "phasic.phasic_pybind" and not Hook.done:
            Hook.done = True
            print("OMP set at phasic_pybind import:", "OMP_NUM_THREADS" in os.environ)
            st = [f"{f.filename.split('site-packages/')[-1]}:{f.lineno}" for f in traceback.extract_stack()[:-1]]
            print("stack:", " <- ".join(reversed(st[-6:])))
        return None
sys.meta_path.insert(0, Hook())
import phasic
print("logging configured after import:", phasic.logging_config._logging_configured)
import logging
print("phasic logger handlers:", logging.getLogger("phasic").handlers)
