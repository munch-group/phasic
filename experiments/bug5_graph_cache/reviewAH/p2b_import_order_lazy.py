"""Review AH p2b: simulate the plan's lazy logger in graph_cache (source patched
in memory at import, no files written) and report who first imports
phasic.phasic_pybind and whether OMP_NUM_THREADS is set then."""
import os, sys, traceback, importlib.machinery as m
assert "scratchpad" in os.environ.get("HOME", "")
assert "OMP_NUM_THREADS" not in os.environ
hits = []
class PatchedLoader(m.SourceFileLoader):
    def get_code(self, fullname):
        data = open(self.path, "rb").read()
        assert b"logger = get_logger(__name__)" in data
        data = data.replace(b"logger = get_logger(__name__)", b"import logging as _lg; logger = _lg.getLogger('phasic.graph_cache')")
        return compile(data, self.path, "exec", dont_inherit=True)
class Hook:
    def find_spec(self, name, path=None, target=None):
        if name == "phasic.graph_cache":
            spec = m.PathFinder.find_spec(name, path)
            spec.loader = PatchedLoader(name, spec.origin)
            spec.cached = None
            return spec
        if name == "phasic.phasic_pybind" and not hits:
            st = [f"{f.filename.split('site-packages/')[-1]}:{f.lineno}" for f in traceback.extract_stack()[:-1]
                  if "phasic" in f.filename and "importlib" not in f.filename]
            hits.append((os.environ.get("OMP_NUM_THREADS"), st))
        return None
sys.dont_write_bytecode = True
sys.meta_path.insert(0, Hook())
import phasic
omp, st = hits[0]
print("graph_cache logger:", phasic.graph_cache.logger)
print("OMP_NUM_THREADS at first phasic_pybind import:", omp)
for s in st:
    print("  ", s)
