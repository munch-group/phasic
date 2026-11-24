"""reviewL l2: on master, which import first loads phasic.phasic_pybind, and
is OMP_NUM_THREADS set at that point? Run with OMP_NUM_THREADS removed."""
import os, sys, traceback
assert "scratchpad" in os.environ["HOME"]
os.environ.pop("OMP_NUM_THREADS", None)


class Hook:
    done = False

    def find_spec(self, name, path=None, target=None):
        if name == "phasic.phasic_pybind" and not Hook.done:
            Hook.done = True
            print("phasic_pybind first import; OMP_NUM_THREADS set:",
                  "OMP_NUM_THREADS" in os.environ)
            stack = [f for f in traceback.extract_stack()
                     if "phasic" in f.filename and "importlib" not in f.filename]
            for f in stack:
                print("   ", f.filename.split("site-packages/")[-1], f.lineno, f.line)
        return None


sys.meta_path.insert(0, Hook())
if os.environ.get("STUB_GRAPH_CACHE"):
    # simulate the fixed graph_cache (no module-level get_logger) by a stub
    import types
    stub = types.ModuleType("phasic.graph_cache")
    stub.GraphCache = stub.get_graph_cache_stats = stub.print_graph_cache_info = None
    sys.modules["phasic.graph_cache"] = stub
    print("graph_cache stubbed")
import phasic  # noqa
print("import ok")
