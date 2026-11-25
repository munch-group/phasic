"""reviewY y4: plan §2.2.3 anchors the build record on the phasic_pybind
extension module object ("the one object importlib.reload can never
replace"). Check what a sys.modules purge + re-import (the usual
'reload everything' notebook idiom) does to that object and to an
attribute stored on it, and whether the native library is re-loaded."""
import sys
import phasic
ext = sys.modules["phasic.phasic_pybind"]
ext._reviewY_record = "recorded-at-first-import"
id1 = id(ext)
graph_type_1 = ext.Graph

for name in [m for m in list(sys.modules) if m == "phasic" or m.startswith("phasic.")]:
    del sys.modules[name]

import phasic as phasic2
ext2 = sys.modules["phasic.phasic_pybind"]
print("same extension module object after purge+reimport:", ext2 is ext)
print("record attribute survives:", hasattr(ext2, "_reviewY_record"))
print("same native Graph type object (native code NOT reloaded):", ext2.Graph is graph_type_1)
