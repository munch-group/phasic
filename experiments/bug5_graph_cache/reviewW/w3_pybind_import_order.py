"""reviewW w3: who imports phasic.phasic_pybind first, and is OMP_NUM_THREADS set then?

Also: with phasic.graph_cache's module-level get_logger neutralised (simulating
the plan's lazy stdlib logger), is phasic_pybind still imported before
OMP_NUM_THREADS is set?  Run with OMP_NUM_THREADS removed from the env.
Usage: python w3.py [lazy]
"""
import os, sys, traceback, importlib.abc, importlib.machinery
assert "/scratchpad/" in os.environ["HOME"]
os.environ.pop("OMP_NUM_THREADS", None)
LAZY = len(sys.argv) > 1 and sys.argv[1] == "lazy"
rec = {}

class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "phasic.phasic_pybind" and "first" not in rec:
            rec["first"] = ("OMP_NUM_THREADS" in os.environ,
                            [f"{f.filename.split('site-packages/')[-1]}:{f.lineno}" for f in traceback.extract_stack()[:-1] if "phasic" in f.filename])
        if LAZY and name == "phasic.logging_config" and "patched" not in rec:
            rec["patched"] = True
        return None

sys.meta_path.insert(0, Hook())

if LAZY:
    # Simulate the plan: graph_cache must not call phasic's get_logger at module
    # level. We patch logging_config.get_logger to a stdlib logger only while
    # phasic.graph_cache's module body runs, by pre-importing logging_config
    # standalone is impossible without the package, so instead we intercept the
    # source of graph_cache via a loader wrapper.
    import importlib.util
    class GCLoader(importlib.abc.Loader):
        def __init__(self, orig): self.orig = orig
        def create_module(self, spec): return None
        def exec_module(self, module):
            src = self.orig.get_source(module.__name__)
            src = src.replace("logger = get_logger(__name__)", "import logging as _l; logger = _l.getLogger('phasic.graph_cache')")
            exec(compile(src, self.orig.path, "exec"), module.__dict__)
    class GCFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "phasic.graph_cache":
                spec = importlib.machinery.PathFinder.find_spec(name, path)
                spec.loader = GCLoader(spec.loader)
                return spec
            return None
    sys.meta_path.insert(0, GCFinder())

import phasic
print("lazy-simulation:", LAZY)
print("OMP set when phasic_pybind first imported:", rec["first"][0])
print("import chain (phasic frames):")
for l in rec["first"][1]:
    print("   ", l)
