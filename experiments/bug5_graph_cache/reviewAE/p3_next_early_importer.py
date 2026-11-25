"""Review AE p3: with phasic.graph_cache replaced by a stub that does NOT
touch the logger (simulating the plan's lazy-logger fix), who imports
phasic.phasic_pybind first, and is OMP_NUM_THREADS set by then?
(Plan test 14 asserts it is.)"""
import os, sys, types, importlib.abc, importlib.machinery, traceback

os.environ.pop("OMP_NUM_THREADS", None)
first = {}

class StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None
    def exec_module(self, m):
        class GraphCache: pass
        m.GraphCache = GraphCache
        m.get_graph_cache_stats = lambda: {}
        m.print_graph_cache_info = lambda: None

class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "phasic.graph_cache":
            return importlib.machinery.ModuleSpec(name, StubLoader())
        if name == "phasic.phasic_pybind" and not first:
            first["omp"] = os.environ.get("OMP_NUM_THREADS")
            first["stack"] = [f"{fs.filename.split('site-packages/')[-1]}:{fs.lineno}"
                              for fs in traceback.extract_stack()
                              if "site-packages/phasic" in fs.filename]
        return None

sys.meta_path.insert(0, Hook())
import phasic  # noqa
print("first phasic_pybind import with graph_cache stubbed: OMP_NUM_THREADS =", first.get("omp"))
for s in first.get("stack", []):
    print("    ", s)
