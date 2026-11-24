"""Review C p6: who imports phasic.phasic_pybind first during `import phasic`
(OMP_NUM_THREADS removed from the env), and which image loads libomp."""
import os, sys, traceback
os.environ.pop("OMP_NUM_THREADS", None)
class Hook:
    seen = False
    def find_spec(self, name, path, target=None):
        if name == "phasic.phasic_pybind" and not Hook.seen:
            Hook.seen = True
            print("OMP at pybind import:", os.environ.get("OMP_NUM_THREADS"))
            st = [f"{os.path.basename(f.filename)}:{f.lineno} {f.line}" for f in traceback.extract_stack()
                  if "importlib" not in f.filename]
            print("\n".join(st[-8:]))
        return None
sys.meta_path.insert(0, Hook())
import numpy, pandas
import ctypes
def libomp_loaded():
    d = ctypes.CDLL(None)
    d._dyld_image_count.restype = ctypes.c_uint32
    d._dyld_get_image_name.restype = ctypes.c_char_p
    d._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    return any(os.path.basename(d._dyld_get_image_name(i)) == b"libomp.dylib" for i in range(d._dyld_image_count()))
print("libomp loaded after numpy+pandas:", libomp_loaded())
import phasic
print("final OMP:", os.environ.get("OMP_NUM_THREADS"))
