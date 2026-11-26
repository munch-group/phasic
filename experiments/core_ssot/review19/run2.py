import sys, os, ctypes, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phasic
_pb = glob.glob(os.path.join(os.path.dirname(phasic.__file__), "phasic_pybind*.so"))[0]
ctypes.CDLL(_pb, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
import mod_nocpp
g = phasic.Graph(mod_nocpp.chain(4))
print("vertices", g.vertices_length(), "expectation", g.expectation())
