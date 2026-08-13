"""Batch 0 M4 gate: the linear moments-Jacobian binding declines on was_dph
graphs (renormalized probability weights would make the linear contraction
silently wrong) and stays applicable on continuous graphs. Exits nonzero on
failure."""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np
import phasic
from phasic import Graph


def two_stage():
    g = Graph(1); s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3]); v2 = g.find_or_create_vertex([2]); v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge(v2, [1.0, 0.0]); v2.add_edge(v1, [0.0, 1.0])
    return g


ok = True
g = two_stage(); g.update_weights([1.0, 2.0])
J = np.asarray(g._moments_grad_theta(2))
cont_ok = J.size > 0
ok &= cont_ok
print(f"continuous: [{'OK' if cont_ok else 'FAIL'}] applicable")

# Native DPH (is_discrete=True, was_dph=False): the guard tests the was_dph
# LATCH, not discreteness -- native DPH must stay applicable (an over-broad
# is_discrete guard would wrongly decline here; G4 review finding 2).
gn = Graph(1)
sn = gn.starting_vertex()
vs = [gn.find_or_create_vertex([3 - i]) for i in range(3)]
sn.add_edge(vs[0], 1.0)
for i in range(2):
    coeff = [0.0] * 2
    coeff[i] = 1.0
    vs[i].add_edge(vs[i + 1], coeff)
gn.is_discrete = True
gn.update_weights([0.3, 0.4])
J = np.asarray(gn._moments_grad_theta(2))
native_ok = J.size > 0
ok &= native_ok
print(f"native DPH (is_discrete, not was_dph): [{'OK' if native_ok else 'FAIL'}] "
      f"{'applicable' if native_ok else 'wrongly declined'}")

gd = two_stage(); gd.update_weights([1.0, 2.0])
gd = gd.discretize(0.5)
J = np.asarray(gd._moments_grad_theta(2))
declined = J.size == 0
ok &= declined
print(f"was_dph (discretize): [{'OK' if declined else 'FAIL'}] "
      f"{'declined' if declined else 'DID NOT decline'}")
print("ALL PASS" if ok else "FAILURES")
sys.exit(0 if ok else 1)
