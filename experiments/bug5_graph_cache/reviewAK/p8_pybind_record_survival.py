# reviewAK p8: does an attribute stored on phasic.phasic_pybind survive (a) importlib.reload(phasic)
# and (b) purging phasic* from sys.modules and importing again? (plan §2.2.3 anchors the build record there)
import sys, importlib
import phasic
pb = phasic.phasic_pybind
pb._reviewAK_record = "R"
importlib.reload(phasic)
print("after reload: same module object", phasic.phasic_pybind is pb, "record", getattr(phasic.phasic_pybind, "_reviewAK_record", None))
for k in [k for k in sys.modules if k == "phasic" or k.startswith("phasic.")]:
    del sys.modules[k]
import phasic as phasic2
pb2 = phasic2.phasic_pybind
print("after purge+reimport: same module object", pb2 is pb, "record", getattr(pb2, "_reviewAK_record", None))
print("same Graph class", pb2.Graph is pb.Graph)
