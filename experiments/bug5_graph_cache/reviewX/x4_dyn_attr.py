"""reviewX x4: the plan's hit-path expression `_Graph(1).dyn_ordering`
(§2.1) -- does the pybind base class expose that attribute?"""
import os
from phasic import Graph
from phasic.phasic_pybind import Graph as _Graph
g = _Graph(1)
print("PHASIC_DYN_ORDERING set:", "PHASIC_DYN_ORDERING" in os.environ)
print("hasattr(_Graph(1), 'dyn_ordering'):", hasattr(g, "dyn_ordering"))
try:
    print("_Graph(1).dyn_ordering =", g.dyn_ordering)
except AttributeError as e:
    print("AttributeError:", e)
print("_Graph(1).get_dyn_ordering() =", g.get_dyn_ordering())
print("Graph(1).dyn_ordering =", Graph(1).dyn_ordering)
