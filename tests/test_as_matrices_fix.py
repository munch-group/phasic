#!/usr/bin/env python
"""
Test for graph.as_matrices() method to ensure no segmentation fault.

This test was added after fixing a bug where as_matrices() caused a
segmentation fault due to dereferencing an uninitialized pointer in
the C++ binding code.
"""

import numpy as np
from phasic import Graph, MatrixRepresentation


def test_as_matrices_simple():
    """Test as_matrices() on a simple graph."""
    g = Graph(1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])

    start.add_edge(v1, 1.0)
    v1.add_edge(v2, 2.0)
    g.normalize()

    # This should not crash
    matrices = g.as_matrices()

    # Check that it returns a NamedTuple
    assert isinstance(matrices, MatrixRepresentation)
    assert isinstance(matrices, tuple)

    # Check that we can access attributes
    assert hasattr(matrices, 'states')
    assert hasattr(matrices, 'sim')
    assert hasattr(matrices, 'ipv')
    assert hasattr(matrices, 'indices')

    # Check that all values are numpy arrays
    assert isinstance(matrices.states, np.ndarray)
    assert isinstance(matrices.sim, np.ndarray)
    assert isinstance(matrices.ipv, np.ndarray)
    assert isinstance(matrices.indices, np.ndarray)

    # Check shapes
    assert matrices.states.shape == (1, 1)
    assert matrices.sim.shape == (1, 1)
    assert len(matrices.ipv) == 1
    assert len(matrices.indices) == 1


def test_as_matrices_complex():
    """Test as_matrices() on a more complex graph (Erlang-3)."""
    g = Graph(1)
    start = g.starting_vertex()

    # Create chain of states
    states = []
    for i in range(1, 5):  # States 1, 2, 3, 4
        states.append(g.find_or_create_vertex([i]))

    # Connect them in sequence
    start.add_edge(states[0], 1.0)
    for i in range(3):
        states[i].add_edge(states[i+1], 2.0)

    g.normalize()

    # This should not crash
    matrices = g.as_matrices()

    # Check that it returns a NamedTuple with numpy arrays
    assert isinstance(matrices, MatrixRepresentation)
    assert isinstance(matrices.states, np.ndarray)
    assert isinstance(matrices.sim, np.ndarray)
    assert isinstance(matrices.ipv, np.ndarray)
    assert isinstance(matrices.indices, np.ndarray)

    # Check shapes
    assert matrices.states.shape == (3, 1)
    assert matrices.sim.shape == (3, 3)
    assert len(matrices.ipv) == 3
    assert len(matrices.indices) == 3

    # Check initial probability vector
    assert matrices.ipv[0] == 1.0
    assert matrices.ipv[1] == 0.0
    assert matrices.ipv[2] == 0.0

    # Check sub-intensity matrix diagonal is negative
    for i in range(3):
        assert matrices.sim[i, i] < 0


def test_as_matrices_multidimensional():
    """Test as_matrices() with multidimensional states."""
    g = Graph(2)  # 2-dimensional states
    start = g.starting_vertex()

    v1 = g.find_or_create_vertex([1, 0])
    v2 = g.find_or_create_vertex([0, 1])
    v3 = g.find_or_create_vertex([1, 1])

    start.add_edge(v1, 0.5)
    start.add_edge(v2, 0.5)
    v1.add_edge(v3, 1.0)
    v2.add_edge(v3, 1.0)

    g.normalize()

    # This should not crash
    matrices = g.as_matrices()

    # Check that it returns a NamedTuple with numpy arrays
    assert isinstance(matrices, MatrixRepresentation)
    assert isinstance(matrices.states, np.ndarray)
    assert isinstance(matrices.sim, np.ndarray)
    assert isinstance(matrices.ipv, np.ndarray)
    assert isinstance(matrices.indices, np.ndarray)

    # Check shapes
    assert matrices.states.shape[1] == 2  # 2-dimensional states
    assert matrices.sim.shape[0] == matrices.sim.shape[1]  # Square matrix
    assert len(matrices.ipv) == matrices.sim.shape[0]

    # Test tuple unpacking (order is: ipv, sim, states, indices)
    ipv, sim, states, indices = matrices
    assert np.array_equal(ipv, matrices.ipv)
    assert np.array_equal(sim, matrices.sim)
    assert np.array_equal(states, matrices.states)
    assert np.array_equal(indices, matrices.indices)


if __name__ == "__main__":
    test_as_matrices_simple()
    print("✅ test_as_matrices_simple passed")

    test_as_matrices_complex()
    print("✅ test_as_matrices_complex passed")

    test_as_matrices_multidimensional()
    print("✅ test_as_matrices_multidimensional passed")

    print("\nAll tests passed!")