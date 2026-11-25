"""reviewAL p5: does an attribute set on phasic.phasic_pybind survive
(a) importlib.reload(phasic.phasic_pybind) and (b) purging every phasic* entry from
sys.modules and importing phasic again in the same process?  (The plan anchors the
build-id record on that extension module as 'the one object reload can never replace'.)"""
import os, sys, importlib
assert os.environ["HOME"].startswith("/private/tmp/")
import phasic
ext = phasic.phasic_pybind
ext._reviewAL_record = "R"
r = importlib.reload(ext)
print("(a) reload: same object:", r is ext, " attr survives:", getattr(r, "_reviewAL_record", None))
for k in [k for k in sys.modules if k == "phasic" or k.startswith("phasic.")]:
    del sys.modules[k]
try:
    import phasic as phasic2
    ext2 = phasic2.phasic_pybind
    print("(b) purge+reimport: import OK; same ext object:", ext2 is ext,
          " attr survives:", getattr(ext2, "_reviewAL_record", None),
          " Graph class same:", phasic2.Graph is phasic.Graph)
except Exception as e:
    print("(b) purge+reimport raised", type(e).__name__, str(e)[:200])
