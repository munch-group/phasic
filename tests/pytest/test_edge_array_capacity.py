"""Edge-array capacity (Part 3a of constant-parameterized-consistency-plan.md).
"""
import numpy as np
import pytest

from phasic import Graph


# Edge-array capacity. ptd_graph_add_edge grows a vertex's edge array only
# when edges_length is 0 or a power of two, so it relies on the capacity
# being the next power of two. Appenders that realloc'ed to exactly len+1
# (add_edge_constant, add_aux_vertex[_constant], the clone's constant and
# starting-vertex loops, GraphBuilder constant edges, synthetic graphs)
# broke that, and the next normal add_edge wrote past the array end. An
# overflow need not crash without a guard allocator, so these tests check
# structure and values; the plan also runs them once under one.

def _fan(n_before=5, n_after=2, *, constant_middle=True, parameterized=True):
    """Vertex a with n_before rate edges, a coefficient-less constant edge,
    then n_after more rate edges, each to its own absorbing target."""
    g = Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([100])
    s.add_edge(a, 1.0)
    rate = (lambda r: [r]) if parameterized else (lambda r: r)
    targets = [g.find_or_create_vertex([k + 1]) for k in range(n_before + n_after + 1)]
    for k in range(n_before):
        a.add_edge(targets[k], rate(1.0 + k))
    if constant_middle:
        a.add_edge_constant(targets[n_before], 0.5)
    for k in range(n_before + 1, n_before + n_after + 1):
        a.add_edge(targets[k], rate(1.0 + k))
    if not constant_middle:
        a.add_edge_constant(targets[n_before], 0.5)
    return g, a


def _out(v):
    return sorted((int(e.to().state()[0]), round(e.weight(), 12)) for e in v.edges())


@pytest.mark.parametrize("parameterized", [True, False])
@pytest.mark.parametrize("n_before", [1, 2, 3, 4, 5, 6, 7, 8])
def test_add_edge_constant_then_add_edge(parameterized, n_before):
    g, a = _fan(n_before, 2, constant_middle=True, parameterized=parameterized)
    twin, ta = _fan(n_before, 2, constant_middle=False, parameterized=parameterized)
    assert a.edges_length() == n_before + 3
    assert _out(a) == _out(ta)
    np.testing.assert_allclose(g.expectation(), twin.expectation(), rtol=1e-14)


@pytest.mark.parametrize("n_before", [1, 2, 3, 4, 5, 6, 7])
def test_add_aux_vertex_constant_then_add_edge(n_before):
    g, a = _fan(n_before, 0, constant_middle=False)
    a.add_aux_vertex_constant(0.25)
    extra = g.find_or_create_vertex([999])
    a.add_edge(extra, [2.0])
    a.add_edge(g.find_or_create_vertex([998]), [3.0])
    assert a.edges_length() == n_before + 1 + 1 + 2
    weights = sorted(round(e.weight(), 12) for e in a.edges())
    assert weights == sorted([1.0 + k for k in range(n_before)] + [0.5, 0.25, 2.0, 3.0])


@pytest.mark.parametrize("n_start", [1, 2, 3, 4, 5, 6, 7, 9])
def test_clone_with_many_start_edges_then_start_add_edge(n_start):
    g = Graph(1)
    s = g.starting_vertex()
    vs = [g.find_or_create_vertex([k + 1]) for k in range(n_start + 2)]
    for k in range(n_start):
        s.add_edge(vs[k], 1.0)
    vs[0].add_edge(vs[-1], [1.0])
    c = g.clone()
    cs = c.starting_vertex()
    cs.add_edge(c.find_vertex([n_start + 1]), 1.0)
    cs.add_edge(c.find_vertex([n_start + 2]), 1.0)
    assert cs.edges_length() == n_start + 2
    assert sorted(int(e.to().state()[0]) for e in cs.edges()) == list(range(1, n_start + 3))


@pytest.mark.parametrize("n_before", [2, 3, 5, 6])
def test_laplace_transform_of_constant_middle_pattern(n_before):
    g, _ = _fan(n_before, 2, constant_middle=True, parameterized=False)
    twin, _ = _fan(n_before, 2, constant_middle=False, parameterized=False)
    lg, lt = g.laplace_transform(0.5), twin.laplace_transform(0.5)
    assert lg.vertices_length() == lt.vertices_length()
    np.testing.assert_allclose(np.asarray(lg.moments(2)), np.asarray(lt.moments(2)),
                               rtol=1e-12)


@pytest.mark.parametrize("n_before", [2, 3, 5, 6])
def test_discretize_clone_branch_of_constant_middle_pattern(n_before):
    """discretize() of a constant graph clones, then add_aux_vertex adds an
    edge to every transient vertex, including a after its constant edge."""
    g, _ = _fan(n_before, 2, constant_middle=True, parameterized=False)
    twin, _ = _fan(n_before, 2, constant_middle=False, parameterized=False)
    d, dt = g.discretize(0.1), twin.discretize(0.1)
    a = d.find_vertex([100])
    assert a.edges_length() == n_before + 3 + 1
    # Same edges in a different order: equal up to summation round-off.
    for n in (1, 2, 3, 5):
        np.testing.assert_allclose(d.pdf(n), dt.pdf(n), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(d.expectation_discrete(d.rewards),
                               dt.expectation_discrete(dt.rewards), rtol=1e-12)


def test_is_synthetic_getter():
    """Additive read-only accessor next to _set_synthetic (needed by Part 3,
    which refuses to widen synthetic graphs)."""
    g, _ = _fan(2, 1)
    assert g._is_synthetic() is False
    g._set_synthetic(True)
    assert g._is_synthetic() is True
    assert g.clone()._is_synthetic() is True
