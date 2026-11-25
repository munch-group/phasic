"""reviewH p3: who imports phasic.phasic_pybind first during `import phasic`,
and is OMP_NUM_THREADS set then?  Mode 'master': record the first import.
Mode 'simfix': while phasic.graph_cache's module body runs, an import of
phasic.phasic_pybind raises ImportError (logging_config swallows it), which
simulates graph_cache using a stdlib logger; record the next importer.
Run with OMP_NUM_THREADS removed from the environment."""
import os, sys, traceback, importlib.abc
mode = sys.argv[1]
assert "OMP_NUM_THREADS" not in os.environ
events = []


class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "phasic.phasic_pybind":
            return None
        stack = traceback.extract_stack()
        in_gc_body = any(f.filename.endswith("graph_cache.py") and f.name == "<module>" for f in stack)
        if mode == "simfix" and in_gc_body:
            events.append(("BLOCKED (graph_cache body)", None, None))
            raise ImportError("simulated: graph_cache uses a stdlib logger")
        if not any(e[0] == "first" for e in events):
            frames = [f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in stack[:-1] if "importlib" not in f.filename][-8:]
            events.append(("first", "OMP_NUM_THREADS" in os.environ, frames))
        return None


sys.meta_path.insert(0, Hook())
import phasic  # noqa
for kind, omp, frames in events:
    print(kind, "OMP set:", omp)
    if frames:
        for f in frames:
            print("    ", f)
