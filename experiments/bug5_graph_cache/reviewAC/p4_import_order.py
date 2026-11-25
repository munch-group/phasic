"""reviewAC p4: with OMP_NUM_THREADS removed, record (on master) which import
chain first loads phasic.phasic_pybind, and whether OMP_NUM_THREADS was set.
Then repeat with graph_cache's module-level get_logger neutralised (by
pre-seeding phasic.logging_config._logging_configured is not possible before
import, so instead we stub get_logger lookups via a meta_path finder that
patches logging_config right after it is executed)."""
import os, sys, traceback, importlib.abc, importlib.machinery
os.environ.pop("OMP_NUM_THREADS", None)
MODE = sys.argv[1] if len(sys.argv) > 1 else "plain"
seen = {}

class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "phasic.phasic_pybind" and "first" not in seen:
            seen["first"] = ("OMP set" if "OMP_NUM_THREADS" in os.environ else "OMP NOT set")
            stack = [f"{f.filename.split('site-packages/')[-1]}:{f.lineno}" for f in traceback.extract_stack()
                     if "site-packages/phasic" in f.filename]
            seen["stack"] = stack
        if MODE == "lazylog" and name == "phasic.logging_config" and "patched" not in seen:
            seen["patched"] = True
            # load it normally, then make setup_logging a no-op until __init__ line 107 passed
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            orig_exec = spec.loader.exec_module
            def exec_module(mod, orig_exec=orig_exec):
                orig_exec(mod)
                real = mod.setup_logging
                def guarded(*a, **k):
                    if "OMP_NUM_THREADS" not in os.environ:
                        return None   # emulate a lazy stdlib logger before OMP is set
                    return real(*a, **k)
                mod.setup_logging = guarded
            spec.loader.exec_module = exec_module
            return spec
        return None

sys.meta_path.insert(0, Hook())
import phasic
print("mode", MODE, "->", seen.get("first"))
print("stack at first phasic_pybind import:")
for s in seen.get("stack", []):
    print("   ", s)
