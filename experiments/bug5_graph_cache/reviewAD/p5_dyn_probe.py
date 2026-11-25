# reviewAD p5: does _Graph(1).get_dyn_ordering() reflect PHASIC_DYN_ORDERING like a fresh callback build?
import os, sys
from phasic import Graph
from phasic.phasic_pybind import Graph as _Graph
def cb(state):
    n = state[0]
    return [] if n <= 1 else [[[n - 1], float(n)]]
print("env PHASIC_DYN_ORDERING =", os.environ.get("PHASIC_DYN_ORDERING"))
print("_Graph(1).get_dyn_ordering():", _Graph(1).get_dyn_ordering())
print("fresh callback build dyn_ordering:", Graph(cb, ipv=[4], graph_cache=False).dyn_ordering)
