#!/usr/bin/env python
"""
Comprehensive test suite for Graph.from_matrices() method.
Tests construction of graphs from matrix representations and round-trip conversions.
"""

import numpy as np
from phasic import Graph, MatrixRepresentation


def test_from_matrices_basic():
    """Test basic from_matrices functionality."""
    # Create simple 2-state phase-type
    ipv = np.array([0.6, 0.4])
    sim = np.array([
        [-2.0, 1.0],
        [0.0, -3.0]
    ])

    # Create graph from matrices
    g = Graph.from_matrices(ipv, sim)

    assert g is not None
    assert g.vertices_length() > 0

    # Test PDF computation
    pdf = g.pdf(1.0, 100)
    assert 0 <= pdf <= 1
    print(f"✅ Basic from_matrices test passed (PDF at t=1.0: {pdf:.6f})")


def test_from_matrices_with_states():
    """Test from_matrices with custom state vectors."""
    ipv = np.array([0.5, 0.5])
    sim = np.array([
        [-1.0, 0.0],
        [0.0, -2.0]
    ])

    # Test with custom states that don't conflict
    states = np.array([[10], [20]], dtype=np.int32)

    g = Graph.from_matrices(ipv, sim, states)
    assert g is not None
    assert g.vertices_length() > 0

    print("✅ from_matrices with custom states test passed")


def test_from_matrices_multidimensional():
    """Test from_matrices with multidimensional states."""
    ipv = np.array([1.0, 0.0, 0.0])
    sim = np.array([
        [-2.0, 1.0, 0.0],
        [0.0, -3.0, 2.0],
        [0.0, 0.0, -1.0]
    ])

    # 2D states
    states = np.array([
        [0, 0],
        [1, 0],
        [1, 1]
    ], dtype=np.int32)

    g = Graph.from_matrices(ipv, sim, states)
    assert g is not None
    assert g.state_length() == 2  # Should have 2D states

    print("✅ from_matrices with multidimensional states test passed")


def test_round_trip_simple():
    """Test round-trip: create graph -> as_matrices -> from_matrices."""
    # Create original graph
    g_orig = Graph(1)
    start = g_orig.starting_vertex()
    v1 = g_orig.find_or_create_vertex([10])  # Use non-conflicting states
    v2 = g_orig.find_or_create_vertex([20])

    start.add_edge(v1, 0.7)
    start.add_edge(v2, 0.3)
    v1.add_edge(v2, 1.5)

    g_orig.normalize()

    # Convert to matrices
    matrices = g_orig.as_matrices()

    # Check that we got a NamedTuple (if our wrapper is used)
    if isinstance(matrices, dict):
        # Fallback for C++ Graph
        ipv = matrices['ipv']
        sim = matrices['sim']
        states = matrices['states']
    else:
        # NamedTuple from Python wrapper
        assert isinstance(matrices, MatrixRepresentation)
        ipv = matrices.ipv
        sim = matrices.sim
        states = matrices.states

    # Reconstruct from matrices
    g_recon = Graph.from_matrices(ipv, sim, states)

    # Compare PDFs at several points
    times = np.linspace(0.1, 3.0, 10)
    for t in times:
        pdf_orig = g_orig.pdf(t, 100)
        pdf_recon = g_recon.pdf(t, 100)
        assert abs(pdf_orig - pdf_recon) < 1e-6, f"PDFs differ at t={t}"

    print("✅ Round-trip test passed")


def test_from_matrices_validation():
    """Test input validation for from_matrices."""
    # Test dimension mismatch
    ipv = np.array([0.5, 0.5])
    sim_wrong = np.array([[-1.0]])  # Wrong size

    try:
        g = Graph.from_matrices(ipv, sim_wrong)
        assert False, "Should have raised error for dimension mismatch"
    except RuntimeError as e:
        assert "square" in str(e) or "dimension" in str(e)

    # Test non-square SIM
    sim_nonsquare = np.array([[-1.0, 0.0]])
    try:
        g = Graph.from_matrices(ipv, sim_nonsquare)
        assert False, "Should have raised error for non-square matrix"
    except RuntimeError as e:
        assert "square" in str(e)

    print("✅ Input validation test passed")


def test_from_matrices_edge_cases():
    """Test edge cases for from_matrices."""
    # Single state
    ipv_single = np.array([1.0])
    sim_single = np.array([[-5.0]])

    g = Graph.from_matrices(ipv_single, sim_single)
    assert g is not None
    pdf = g.pdf(0.5, 100)
    # Just check it's a valid PDF value, exact calculation depends on implementation
    assert 0 <= pdf <= 5.0  # Max PDF for exponential with rate 5

    # Zero initial probability for some states
    ipv_sparse = np.array([0.0, 1.0, 0.0])
    sim_sparse = np.array([
        [-1.0, 0.5, 0.0],
        [0.0, -2.0, 1.0],
        [0.0, 0.0, -0.5]
    ])

    g = Graph.from_matrices(ipv_sparse, sim_sparse)
    assert g is not None

    print("✅ Edge cases test passed")


def test_from_matrices_performance():
    """Test from_matrices with larger matrices."""
    n = 10

    # Create a chain of states
    sim = np.zeros((n, n))
    for i in range(n):
        sim[i, i] = -(i + 1)  # Increasing exit rates
        if i < n - 1:
            sim[i, i + 1] = i + 1  # Transition to next state

    ipv = np.zeros(n)
    ipv[0] = 1.0  # Start in first state

    g = Graph.from_matrices(ipv, sim)
    assert g is not None
    assert g.vertices_length() >= n

    # Test PDF computation works
    pdf = g.pdf(1.0, 100)
    assert 0 <= pdf <= 1

    print(f"✅ Performance test passed (n={n} states)")


if __name__ == "__main__":
    print("Testing Graph.from_matrices() functionality")
    print("=" * 60)

    test_from_matrices_basic()
    test_from_matrices_with_states()
    test_from_matrices_multidimensional()
    test_round_trip_simple()
    test_from_matrices_validation()
    test_from_matrices_edge_cases()
    test_from_matrices_performance()

    print("\n" + "=" * 60)
    print("✅ All from_matrices tests passed!")