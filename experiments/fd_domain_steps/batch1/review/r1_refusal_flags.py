"""Review probe r1: does build_info see weight_mode / is_discrete / was_dph for
every graph kind and builder? (refusal flag per §4.4)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../tests/pytest"))
import phasic
from phasic import Graph, _fd_steps
from phasic import ffi_wrappers as FW
import _fd_golden_cells as C
import numpy as np

for mode in ("linear", "log", "formula", "callback"):
    g = C.beside(mode)
    s = g.serialize(theta_dim=2)
    print(mode, "serialize keys:", {k: s.get(k) for k in ("weight_mode", "is_discrete", "was_dph")})
    for disc in (False, True):
        try:
            m = Graph.pmf_from_graph(C.beside(mode), discrete=disc, theta_dim=2)
            print("  pfg disc=%s" % disc, m._fd_build_info)
        except Exception as e:
            print("  pfg disc=%s ERR %s" % (disc, str(e)[:100]))
        try:
            m = Graph.pmf_and_moments_from_graph(C.beside(mode), nr_moments=2, discrete=disc, theta_dim=2)
            print("  pam disc=%s" % disc, m._fd_build_info)
        except Exception as e:
            print("  pam disc=%s ERR %s" % (disc, str(e)[:100]))
    try:
        print("  rvp", FW._rvp_fd_spec(s, None)[1])
    except Exception as e:
        print("  rvp ERR", e)

gd = C.beside()
try:
    gd.is_discrete = True
except Exception as e:
    print("cannot set is_discrete:", e)
sd = gd.serialize(theta_dim=2)
print("graph.is_discrete=True serialize:", {k: sd.get(k) for k in ("weight_mode", "is_discrete", "was_dph")})
print("  rvp on it:", FW._rvp_fd_spec(sd, None)[1])
print("  pfg(discrete=False) on is_discrete graph:", Graph.pmf_from_graph(gd, theta_dim=2)._fd_build_info)

gw = C.beside().discretize(1.0)
sw = gw.serialize()
print("discretized serialize:", {k: sw.get(k) for k in ("weight_mode", "is_discrete", "was_dph")})
print("  pfg discrete=True:", Graph.pmf_from_graph(gw, discrete=True, theta_dim=3)._fd_build_info)
print("  pfg discrete=False:", Graph.pmf_from_graph(C.beside().discretize(1.0), theta_dim=3)._fd_build_info)
print("  rvp:", FW._rvp_fd_spec(sw, None)[1])

jpg, oi = C.joint_graphs()
print("joint (continuous) jix:", Graph.pmf_from_graph_joint_index(jpg, theta_dim=2)._fd_build_info)
