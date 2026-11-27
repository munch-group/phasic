"""Who looks up phasic.phasic_pybind before OMP_NUM_THREADS is set?
(test_graph_cache_key::test_14_import_order regressed in Batch 1.)"""
import importlib.abc
import os
import sys
import traceback

os.environ.pop("OMP_NUM_THREADS", None)


class Hook(importlib.abc.MetaPathFinder):
    done = False

    def find_spec(self, name, path, target=None):
        if name == "phasic.phasic_pybind" and not Hook.done:
            Hook.done = True
            print("OMP set:", "OMP_NUM_THREADS" in os.environ)
            traceback.print_stack(limit=12)
        return None


sys.meta_path.insert(0, Hook())
import phasic  # noqa: E402,F401
