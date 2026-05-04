"""Regression tests for ptd_err global-buffer leakage across calls.

A failed C call sets the global ptd_err buffer. Earlier, the C++ wrapper
threw without clearing it, so a later post-call ptd_err[0] check (e.g. in
the pybind add_edge dispatcher) would re-throw the stale message even
when the new C call had succeeded — causing order-dependent test failures.
"""

import pytest

from phasic import Graph


def test_add_edge_after_caught_negative_weight():
    """A caught failed add_edge must not poison the next add_edge call."""
    g_bad = Graph(1)
    v1 = g_bad.find_or_create_vertex([1])
    v2 = g_bad.find_or_create_vertex([2])
    with pytest.raises(RuntimeError):
        v1.add_edge(v2, -1.0)

    # Fresh graph; the prior failure must not leak into this call.
    g = Graph(1)
    start = g.starting_vertex()
    v = g.find_or_create_vertex([1])
    start.add_edge(v, 1e-9)


def test_add_edge_after_caught_self_loop():
    """Self-loop attempts also set ptd_err; the next call must be clean."""
    g_bad = Graph(1)
    v = g_bad.find_or_create_vertex([1])
    with pytest.raises(RuntimeError):
        v.add_edge(v, 1.0)

    g = Graph(1)
    start = g.starting_vertex()
    target = g.find_or_create_vertex([1])
    start.add_edge(target, 0.5)


def test_add_edge_parameterized_after_caught_failure():
    """Same contract for the parameterized add_edge dispatch path."""
    g_bad = Graph(1)
    v1 = g_bad.find_or_create_vertex([1])
    v2 = g_bad.find_or_create_vertex([2])
    with pytest.raises(RuntimeError):
        v1.add_edge(v2, [-1.0])

    g = Graph(1)
    start = g.starting_vertex()
    v = g.find_or_create_vertex([1])
    start.add_edge(v, [1.0])
