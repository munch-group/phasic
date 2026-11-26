import sys, os, ctypes, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phasic
_pb = glob.glob(os.path.join(os.path.dirname(phasic.__file__), "phasic_pybind*.so"))[0]
ctypes.CDLL(_pb, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
g = phasic.Graph(1)
v = g.find_or_create_vertex([1]); g.starting_vertex().add_edge(v, 1.0)
for name in sys.argv[1:]:
    try:
        mod = __import__(name); print(name, "IMPORTED; nv =", mod.nv(g))
    except Exception as e:
        print(name, "EXC", type(e).__name__, str(e)[:200])
