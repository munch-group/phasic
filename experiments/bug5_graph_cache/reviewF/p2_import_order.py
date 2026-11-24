"""With graph_cache stubbed out (simulating the plan's lazy logger), who
imports phasic.phasic_pybind first, and is OMP_NUM_THREADS set then?"""
import os, sys, types, traceback
first = {}
class Hook:
    def find_spec(self, name, path=None, target=None):
        if name == "phasic.phasic_pybind" and not first:
            first["omp_set"] = "OMP_NUM_THREADS" in os.environ
            first["stack"] = [f"{f.filename.split('site-packages/')[-1]}:{f.lineno}"
                              for f in traceback.extract_stack() if "phasic" in f.filename][:12]
        return None
sys.meta_path.insert(0, Hook())
stub = types.ModuleType("phasic.graph_cache")
stub.GraphCache = object
stub.get_graph_cache_stats = lambda: None
stub.print_graph_cache_info = lambda: None
sys.modules["phasic.graph_cache"] = stub
import phasic
print("OMP set at first phasic_pybind import (graph_cache stubbed):", first.get("omp_set"))
for s in first.get("stack", []):
    print("    ", s)
