"""Graph.clone() is an exact structural copy (the C ptd_clone_graph, whose
O(V*E) target scan was replaced by an index lookup, 2026-10-13)."""
import numpy as np
import pytest

from phasic import Graph
import _discretize_pins as pins


def _structure(g):
    def edges(v):
        it = iter(v.parameterized_edges())
        out = []
        for e in v.edges():
            n = e.coefficients_length()
            out.append((e.to().index(), e.weight(), n,
                        tuple(next(it).edge_state(n)) if n else ()))
        return out
    return ([list(v.state()) for v in g.vertices()],
            [v.is_aux() for v in g.vertices()],
            [edges(v) for v in g.vertices()],
            g.param_length(), g.parameterized(), g.get_was_dph())


def _with_constant_edge():
    g = pins.coal_param()
    v = g.vertex_at(2)
    v.add_edge_constant(g.vertex_at(g.vertices_length() - 1), 0.25)
    return g


GRAPHS = {
    "param": pins.coal_param,
    "constant": pins.coal_constant,
    "discretized_param": lambda: pins.coal_param().discretize(0.1),
    "discretized_constant": lambda: pins.coal_constant().discretize(0.1),
    "constant_edge": _with_constant_edge,
    "epoch": pins.epoch_graph,
    "two_param_chain": pins.two_param_chain,
    "joint": lambda: (lambda g: g.joint_prob_graph(g._indexer, reward_limit=1,
                                                   mutation_rate=1.0))(pins.coal_indexed()),
}


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_clone_is_exact_structural_copy(name):
    g = GRAPHS[name]()
    if g.parameterized():
        g.update_weights(np.linspace(0.5, 1.5, g.param_length()))
    assert _structure(g.clone()) == _structure(g)


def test_clone_lookup_still_works():
    g = pins.coal_param()
    c = g.clone()
    for v in g.vertices():
        if v.index() == g.starting_vertex().index():
            continue
        assert c.find_vertex(list(v.state())).index() == v.index()
