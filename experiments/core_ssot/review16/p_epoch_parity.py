"""Review 16 probe: does C++ add_epoch equal Python add_epoch today (the 0d
equality test's premise)? And does Python add_epoch on a C++-built epoch
graph (with the wrapper's attributes set) equal it on the Python-built one?"""
import os, sys, glob, ctypes
import numpy as np
import phasic
from phasic import Graph
sys.path.insert(0, "/Users/kmt/phasic/tests/pytest")
if hasattr(os, "RTLD_GLOBAL"):
    _pb = glob.glob(os.path.join(os.path.dirname(phasic.__file__), "phasic_pybind*.so"))
    ctypes.CDLL(_pb[0], mode=os.RTLD_NOW | os.RTLD_GLOBAL)
import cppimport
cpp = cppimport.imp_from_filepath("/Users/kmt/phasic/tests/pytest/_cpp_add_epoch_fixture.cpp", "_cpp_add_epoch_fixture")
import _discretize_pins as pins

def struct(g):
    out = []
    for v in g.vertices():
        it = iter(v.parameterized_edges())
        es = []
        for e in v.edges():
            k = e.coefficients_length()
            c = tuple(next(it).edge_state(k)) if k else ()
            es.append((e.to().index(), float(e.weight()), c))
        out.append((tuple(v.state()), es))
    return out

N = 4
cg = Graph(cpp.epoch_graph(N, False, 1.0))
pg = pins.coal_param(N); pg.update_weights([1.0]); pg = pg.add_epoch(1.0)
sc, sp = struct(cg), struct(pg)
print("V", len(sc), len(sp), "param_length", cg.param_length(), pg.param_length())
print("epoch1 structure equal:", sc == sp)
if sc != sp:
    for i,(a,b) in enumerate(zip(sc, sp)):
        if a != b:
            print(" first diff at vertex", i, "\n  cpp", a, "\n  py ", b); break
# Python add_epoch on the C++ one with wrapper attributes set
cg._epoch_state_index = N; cg._n_epochs = 1; cg._base_param_length = 1
cb = pins._coalescent_cb(N, True)
try:
    cg.update_weights([1.0]*cg.param_length()); pg.update_weights([1.0]*pg.param_length())
    c2 = cg.add_epoch(2.0, callback=cb)
    p2 = pg.add_epoch(2.0)
    s2c, s2p = struct(c2), struct(p2)
    print("epoch2 V", len(s2c), len(s2p), "equal:", s2c == s2p)
    if s2c != s2p:
        for i,(a,b) in enumerate(zip(s2c, s2p)):
            if a != b:
                print(" first diff at vertex", i, "\n  cpp", a, "\n  py ", b); break
except Exception as ex:
    print("epoch2 error:", type(ex).__name__, ex)
