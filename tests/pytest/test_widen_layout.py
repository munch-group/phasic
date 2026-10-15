"""The shared C++ widening primitive behind Graph._rebuild_with_wider_layout
(phase 1 of discretize-cpp-ssot-plan.md): mechanics only, no policy checks
(primitive contract P1-P4)."""
import numpy as np
import pytest

from phasic import Graph


def _constant_chain():
    g = Graph(1)
    s = g.starting_vertex()
    a, b, c = (g.find_or_create_vertex([k]) for k in (3, 2, 1))
    s.add_edge(a, 1.0)
    a.add_edge(b, 2.0)
    b.add_edge(c, 0.5)
    return g


def _param_chain():
    g = Graph(1)
    s = g.starting_vertex()
    a, b, c = (g.find_or_create_vertex([k]) for k in (3, 2, 1))
    s.add_edge(a, 1.0)
    a.add_edge(b, [2.0])
    b.add_edge(c, [0.5])
    return g


def _out(g):
    return [[(e.to().index(), e.weight(), e.coefficients_length()) for e in v.edges()]
            for v in g.vertices()]


def test_widens_constant_graph_without_policy_checks():
    """P4: the primitive itself widens a constant graph (operations refuse)."""
    g = _constant_chain()
    w = g._rebuild_with_wider_layout(extra_state_dims=2)
    assert w.state_length() == 3
    assert [list(v.state())[:1] for v in w.vertices()] == [list(v.state()) for v in g.vertices()]
    assert all(list(v.state())[1:] == [0, 0] for v in w.vertices())
    assert _out(w) == _out(g)
    np.testing.assert_allclose(w.expectation(), g.expectation(), rtol=1e-15)


def test_appends_zero_slots_and_resets_weights_to_theta_one():
    g = _param_chain()
    g.update_weights([3.0])
    w = g._rebuild_with_wider_layout(extra_coeff_slots=2)
    assert w.param_length() == 3
    coeffs = [list(e.edge_state(3)) for v in w.vertices() for e in v.parameterized_edges()
              if v.index() != w.starting_vertex().index()]
    assert coeffs == [[2.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    # documented: weights are reset to their value at theta = 1
    assert [e.weight() for e in w.vertex_at(1).edges()] == [2.0]


def test_refuses_synthetic_graph():
    g = _param_chain()
    g._set_synthetic(True)
    with pytest.raises(ValueError, match="synthetic SCC graph"):
        g._rebuild_with_wider_layout(extra_coeff_slots=1)


def test_widen_layout_binding_is_private():
    assert hasattr(_param_chain(), "_widen_layout")
    assert not hasattr(_param_chain(), "widen_layout")
