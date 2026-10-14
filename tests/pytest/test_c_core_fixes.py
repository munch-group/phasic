"""C-core fixes of 2026-10-14 (discretize-cpp-derisk-findings.md):
- ptd_normalize_graph leaves a vertex with zero total weight unchanged
  (it produced 0/0 = NaN weights);
- joint_stop_prob_graph and joint_sojourn_graph refuse a discrete joint graph
  (joint_stop's output was flagged is_discrete without was_dph; joint_sojourn
  read per-step probabilities as rates).
(The PDF-gradient fix of the same day became moot: the unused C function
ptd_graph_pdf_with_gradient was removed.)
"""
import math

import pytest

from phasic import Graph
import _discretize_pins as pins


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
