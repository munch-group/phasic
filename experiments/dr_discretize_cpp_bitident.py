"""De-risk (discretize-cpp-ssot-plan.md, phase -1 (c)): is the existing C++
discretize bit-identical to Python discretize on CONSTANT graphs (the only
graphs C++ supports today)? Compares vertex states, is_aux, per-vertex edge
lists (target, weight) in order, and rewards, exactly (no tolerance)."""
import os, sys, glob, ctypes
import numpy as np
import phasic
from phasic import Graph
_pb = glob.glob(os.path.join(os.path.dirname(phasic.__file__), "phasic_pybind*.so"))
ctypes.CDLL(_pb[0], mode=os.RTLD_NOW | os.RTLD_GLOBAL)
import cppimport
sys.path.insert(0, 'experiments')
fx = cppimport.imp_from_filepath('experiments/_dr_cpp_discretize_fixture.cpp', '_dr_cpp_discretize_fixture')
from dr_discretize_cpp_sizes import coal


def snapshot(g):
    return ([tuple(int(x) for x in v.state()) for v in g.vertices()],
            [bool(v.is_aux()) for v in g.vertices()],
            [[(e.to().index(), e.weight()) for e in v.edges()] for v in g.vertices()])


for n in [int(a) for a in sys.argv[1:]]:
    g = coal(n, param=False)
    assert not g.parameterized()
    py_fixed = g.discretize(0.1)
    cg, crew = fx.cpp_discretize_fixed(g, 0.1)
    cg = Graph(cg)
    ok_fixed = snapshot(py_fixed) == snapshot(cg) and list(py_fixed.rewards) == list(crew)
    py_cb = g.discretize(lambda s: 0.3 * float(s.sum()))
    cg2, crew2 = fx.cpp_discretize_cb(g, 0.3)
    cg2 = Graph(cg2)
    ok_cb = snapshot(py_cb) == snapshot(cg2) and list(py_cb.rewards) == list(crew2)
    print(f"coal n={n} V={g.vertices_length()}: fixed-rate bit-identical={ok_fixed}, "
          f"callback bit-identical={ok_cb}; was_dph py={py_fixed.get_was_dph()} cpp={cg.get_was_dph()}",
          flush=True)
