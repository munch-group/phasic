"""Probe: importlib.reload(phasic) re-executes phasic/__init__.py (so an
end-of-import _record_build_stats() call would run again and re-hash the
files on disk) while the already-imported submodules and the compiled
extension are NOT re-executed / reloaded (old code keeps running)."""
import importlib, sys
import phasic
gc_mod = sys.modules["phasic.graph_cache"]
ch_mod = sys.modules["phasic.callback_hash"]
ext = sys.modules["phasic.phasic_pybind"]
# mark the package namespace to detect re-execution of __init__
phasic._probe_marker = "before"
Graph_before = phasic.Graph
importlib.reload(phasic)
print("reload(phasic) succeeded")
print("__init__ re-executed (Graph class object replaced):", phasic.Graph is not Graph_before)
print("graph_cache submodule same object (not re-executed):", sys.modules["phasic.graph_cache"] is gc_mod)
print("callback_hash submodule same object (not re-executed):", sys.modules["phasic.callback_hash"] is ch_mod)
print("phasic_pybind extension same object (cannot be reloaded):", sys.modules["phasic.phasic_pybind"] is ext)
