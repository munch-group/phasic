"""Does an attribute stored on phasic.phasic_pybind survive (a) importlib.reload
of phasic and (b) purging phasic* from sys.modules and re-importing?
Plan v4.2 s2.2.3 anchors the build-id record on that module object."""
import sys, importlib
import phasic
ext = phasic.phasic_pybind
ext._reviewQ_record = "recorded-at-first-import"
print("pybind11 module type:", type(ext), "has PyInit multi-phase slots? n/a")
importlib.reload(phasic)
print("(a) after reload(phasic): same ext object:", phasic.phasic_pybind is ext,
      "record present:", hasattr(phasic.phasic_pybind, "_reviewQ_record"))
for name in [n for n in list(sys.modules) if n == "phasic" or n.startswith("phasic.")]:
    del sys.modules[name]
import phasic as phasic2
ext2 = phasic2.phasic_pybind
print("(b) after purge + re-import: same ext object:", ext2 is ext,
      "record present:", hasattr(ext2, "_reviewQ_record"))
print("    same Graph class object:", ext2.Graph is ext.Graph if hasattr(ext, "Graph") else "n/a")
print("    sys has no record (control):", not hasattr(sys, "_reviewQ_record"))
