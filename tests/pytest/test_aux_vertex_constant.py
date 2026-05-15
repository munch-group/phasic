"""Unit tests for the new ``Vertex.add_aux_vertex_constant(weight)``.

The method creates an aux vertex and bidirectional coefficient-less
constant-weight edges. Both edges are skipped by
``ptd_graph_update_weights``, so the trapping rate is decoupled from
theta. This is the building block used by ``joint_stop_prob_graph()``
to install trapping loops on parameterised JSP graphs without coupling
the trap rate to the user's parameter scaling (e.g. ``exposure``).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from phasic import Graph


def _build_param_graph_with_aux():
    """Build a tiny parameterised graph with a constant-weight aux loop.

    Topology:
        v_start -(IPV 1.0)-> v1 -(param [1,0])-> v2 -(param [0,1])-> v_abs
        v1 <-(constant 1.0)-> v_aux
    """
    g = Graph(1, parameterized=True)
    v_start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    v_abs = g.find_or_create_vertex([0])

    v_start.add_edge(v1, 1.0)                  # IPV scalar
    v1.add_edge(v2, [1.0, 0.0])                # param edge
    v2.add_edge(v_abs, [0.0, 1.0])             # param edge

    aux = v1.add_aux_vertex_constant(1.0)
    return g, v1, v2, aux


def test_aux_state_and_edge_count():
    g, v1, v2, aux = _build_param_graph_with_aux()

    # Aux vertex has all-zero state (matching add_aux_vertex convention).
    assert tuple(aux.state()) == (0,)

    # v1 has TWO outgoing edges: the original param edge to v2, and the
    # new constant edge to aux.
    assert v1.edges_length() == 2

    # The aux vertex has exactly ONE outgoing edge (back to v1).
    assert aux.edges_length() == 1
    assert aux.edges()[0].to().index() == v1.index()


def test_constant_edges_are_not_parameterised():
    g, v1, v2, aux = _build_param_graph_with_aux()

    # v1's parameterised_edges() returns ONLY the dynamic param edge to v2;
    # the new constant edge to aux is excluded.
    v1_param = v1.parameterized_edges()
    assert len(v1_param) == 1
    assert v1_param[0].to().index() == v2.index()

    # The aux vertex's edge back to v1 is coefficient-less, so
    # parameterized_edges() is empty.
    assert len(aux.parameterized_edges()) == 0


def test_update_weights_does_not_touch_constant_edges():
    g, v1, v2, aux = _build_param_graph_with_aux()

    g.update_weights(np.array([7.0, 11.0]))

    # The dynamic param edge updates as expected: 1.0*7 + 0.0*11 = 7.
    v1_to_v2 = next(e for e in v1.edges() if e.to().index() == v2.index())
    assert v1_to_v2.weight() == 7.0

    # Both new constant edges keep weight exactly 1.0 regardless of theta.
    v1_to_aux = next(e for e in v1.edges() if e.to().index() == aux.index())
    assert v1_to_aux.weight() == 1.0
    assert aux.edges()[0].weight() == 1.0


def test_update_weights_runs_with_only_constant_aux_loop():
    """update_weights must not fail when a vertex has only constant
    edges (which can happen if all dynamic edges have been redirected
    elsewhere). Sanity check the no-op semantics of
    coefficients_length == 0."""
    g, v1, v2, aux = _build_param_graph_with_aux()

    # Run multiple update_weights calls; constant edges stay constant.
    for theta in (np.array([1.0, 1.0]), np.array([2.0, 3.0]), np.array([0.1, 100.0])):
        g.update_weights(theta)
        assert next(e for e in v1.edges() if e.to().index() == aux.index()).weight() == 1.0
        assert aux.edges()[0].weight() == 1.0


def test_serialize_emits_constant_edges_field():
    """Round-trip through the C++ GraphBuilder is exercised end-to-end
    by ``test_exposure_daisy_chain.py::test_uniform_alpha_equivalence``
    (which calls ``daisy_chain_joint_probs`` -> FFI -> GraphBuilder).
    Here we just confirm the Python serialiser correctly tags
    coefficient-less edges in the dedicated ``constant_edges`` field
    and excludes them from the regular ``edges`` field.
    """
    g, v1, v2, aux = _build_param_graph_with_aux()
    structure = g.serialize(theta_dim=2)

    constant_edges = structure['constant_edges']
    edges = structure['edges']

    # The two new aux-loop edges (v1<->aux) must appear in
    # constant_edges, not in edges.
    assert constant_edges.shape[0] == 2

    aux_idx = aux.index()
    v1_idx = v1.index()
    pairs = {(int(row[0]), int(row[1])) for row in constant_edges}
    assert (v1_idx, aux_idx) in pairs
    assert (aux_idx, v1_idx) in pairs

    # The regular ``edges`` field must NOT contain the aux-loop edges.
    edge_pairs = {(int(row[0]), int(row[1])) for row in edges}
    assert (v1_idx, aux_idx) not in edge_pairs
    assert (aux_idx, v1_idx) not in edge_pairs

    # All weights in constant_edges are 1.0.
    assert all(row[2] == 1.0 for row in constant_edges)


def test_rejects_non_positive_weight():
    g = Graph(1, parameterized=True)
    v_start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v_abs = g.find_or_create_vertex([0])
    v_start.add_edge(v1, 1.0)
    v1.add_edge(v_abs, [1.0, 0.0])

    with pytest.raises(Exception, match=r"strictly positive"):
        v1.add_aux_vertex_constant(0.0)
    with pytest.raises(Exception, match=r"strictly positive"):
        v1.add_aux_vertex_constant(-1.0)


def test_works_on_non_parameterised_graph_too():
    """The method should work whether or not the graph is parameterised
    (the test above covers parameterised; here we cover the constant-
    edge-mode graph). This is symmetry with add_aux_vertex(rate)."""
    g = Graph(1)
    v_start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v_abs = g.find_or_create_vertex([0])
    v_start.add_edge(v1, 1.0)
    v1.add_edge(v_abs, 2.5)

    aux = v1.add_aux_vertex_constant(1.0)
    assert aux.edges_length() == 1
    assert aux.edges()[0].weight() == 1.0
    v1_to_aux = next(e for e in v1.edges() if e.to().index() == aux.index())
    assert v1_to_aux.weight() == 1.0
