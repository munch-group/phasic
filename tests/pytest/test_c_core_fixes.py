"""C-core fixes of 2026-10-14 (discretize-cpp-derisk-findings.md):
- ptd_graph_pdf_with_gradient resolves edge targets by index instead of a
  per-edge scan (O(steps*E*V) before); outputs pinned bit-exactly from the
  build before the change. These pins are REGRESSION pins, not correctness
  checks: for one-coefficient edges the function counts theta in the
  uniformisation rate but not in the transitions (recorded in
  discretize-cpp-derisk-findings.md), so the coal4 values are probably
  wrong. The scan fallback for stale indices is the old code verbatim and is
  not exercised here;
- ptd_normalize_graph leaves a vertex with zero total weight unchanged
  (it produced 0/0 = NaN weights);
- joint_stop_prob_graph refuses a discrete joint graph (its output was
  flagged is_discrete without was_dph).
"""
import glob
import math
import os

import numpy as np
import pytest

import phasic
from phasic import Graph
import _discretize_pins as pins

if hasattr(os, "RTLD_GLOBAL"):
    import ctypes as _ctypes
    _pb = glob.glob(os.path.join(os.path.dirname(phasic.__file__), "phasic_pybind*.so"))
    if _pb:
        _ctypes.CDLL(_pb[0], mode=os.RTLD_NOW | os.RTLD_GLOBAL)

cppimport = pytest.importorskip("cppimport")
_FIXTURE = os.path.join(os.path.dirname(__file__), "_cpp_c_core_fixture.cpp")

# value then gradient, at t = 0.3, 1.0, 2.5, as float.hex (build before the fix)
PDF_GRAD_BEFORE = {
 "coal4": [
  [
   "0x1.2d69d9bedf154p-5",
   "0x1.20a45ed90d8b9p-5"
  ],
  [
   "0x1.17494fd9aa972p-2",
   "-0x1.761f0655aa424p-6"
  ],
  [
   "0x1.113012d6958e3p-2",
   "-0x1.bfd00410b8941p-2"
  ]
 ],
 "two_param_chain": [
  [
   "0x1.5ee215dc0dfcfp-5",
   "0x1.7f2d982d7fe25p-4",
   "0x1.4dd368d33a9c4p-6"
  ],
  [
   "0x1.c0130d27f7e94p-3",
   "0x1.2590aac8eb6d2p-3",
   "0x1.3c2d5acabeb42p-4"
  ],
  [
   "0x1.0cf76698a5430p-2",
   "-0x1.4839d7c10182ap-1",
   "0x1.25384d6e70990p-6"
  ]
 ]
}
THETAS = {"coal4": [1.3], "two_param_chain": [0.7, 1.9]}
BUILDERS = {"coal4": pins.coal_param, "two_param_chain": pins.two_param_chain}


@pytest.fixture(scope="module")
def cpp():
    return cppimport.imp_from_filepath(_FIXTURE, "_cpp_c_core_fixture")


@pytest.mark.parametrize("name", sorted(PDF_GRAD_BEFORE))
def test_pdf_with_gradient_bit_identical(cpp, name):
    g = BUILDERS[name]()
    theta = THETAS[name]
    g.update_weights(theta)
    for t, want in zip((0.3, 1.0, 2.5), PDF_GRAD_BEFORE[name]):
        value, grad = cpp.pdf_with_gradient(g, t, 0, theta)
        assert [float(x).hex() for x in [value] + list(grad)] == want


def test_normalize_leaves_zero_total_vertex_unchanged():
    g = Graph(1)
    s = g.starting_vertex()
    a, b, c = (g.find_or_create_vertex([k]) for k in (3, 2, 1))
    s.add_edge(a, 1.0)
    a.add_edge(b, 2.0)
    a.add_edge(c, 6.0)
    b.add_edge(c, 0.0)          # b's only edge has weight 0
    g.normalize()
    assert [e.weight() for e in g.find_vertex([2]).edges()] == [0.0]
    assert [e.weight() for e in g.find_vertex([3]).edges()] == [0.25, 0.75]
    assert not any(math.isnan(e.weight()) for v in g.vertices() for e in v.edges())


def test_joint_stop_prob_graph_refuses_a_discrete_joint_graph():
    g = pins.coal_indexed(3)
    discrete = g.joint_prob_graph(g._indexer, reward_limit=1, mutation_rate=1.0)
    assert discrete.is_discrete
    with pytest.raises(ValueError, match="requires a continuous joint-probability graph"):
        discrete.joint_stop_prob_graph()
    cont = g.joint_prob_graph(g._indexer, reward_limit=1, mutation_rate=1.0, discrete=False)
    jsp = cont.joint_stop_prob_graph()
    assert jsp.is_discrete is False and not jsp.get_was_dph()


def test_joint_sojourn_graph_refuses_a_discrete_joint_graph():
    g = pins.coal_indexed(3)
    discrete = g.joint_prob_graph(g._indexer, reward_limit=1, mutation_rate=1.0)
    with pytest.raises(ValueError, match="requires a continuous joint-probability graph"):
        discrete.joint_sojourn_graph()
    cont = g.joint_prob_graph(g._indexer, reward_limit=1, mutation_rate=1.0, discrete=False)
    assert cont.joint_sojourn_graph().is_discrete is False
