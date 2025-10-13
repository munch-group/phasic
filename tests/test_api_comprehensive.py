#!/usr/bin/env python
"""
Comprehensive test suite for the ptdalgorithms Python API.
Tests all core functionality of the Graph, Vertex, and Edge classes.
Standalone version that doesn't require pytest.
"""

import numpy as np
import sys
import traceback
import ptdalgorithms as ptd
from ptdalgorithms import Graph, Vertex, Edge, MatrixRepresentation


def run_test(test_func, test_name):
    """Run a single test function and report results."""
    try:
        test_func()
        print(f"✅ {test_name}")
        return True
    except AssertionError as e:
        print(f"❌ {test_name}: {e}")
        return False
    except Exception as e:
        print(f"❌ {test_name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# Graph Construction Tests
# ============================================================================

def test_construct_with_state_length():
    """Test basic graph construction with state_length."""
    g = Graph(state_length=1)
    assert g is not None
    assert g.vertices_length() == 1  # Only starting vertex
    assert g.state_length() == 1


def test_construct_with_multidimensional_state():
    """Test graph with multidimensional state vectors."""
    g = Graph(state_length=3)
    assert g.state_length() == 3
    v = g.find_or_create_vertex([1, 2, 3])
    assert list(v.state()) == [1, 2, 3]


def test_construct_with_callback():
    """Test graph construction with callback function."""
    def callback(state):
        if len(state) > 0 and state[0] < 5:
            return [([state[0] + 1], 1.0)]
        return []

    g = Graph(callback=callback)
    assert g.vertices_length() > 1  # Should generate vertices


def test_construct_parameterized():
    """Test parameterized graph construction."""
    def callback(state):
        if len(state) > 0 and state[0] < 3:
            return [([state[0] + 1], 0.0, [1.0, 0.0])]
        return []

    g = Graph(callback=callback, parameterized=True)
    assert g.vertices_length() > 1

    # Check serialization detects parameterized edges
    serialized = g.serialize()
    assert serialized.get('param_length', 0) > 0


def test_starting_vertex():
    """Test that starting vertex is always present."""
    g = Graph(state_length=1)
    start = g.starting_vertex()
    assert start is not None


# ============================================================================
# Vertex Operations Tests
# ============================================================================

def test_find_or_create_vertex():
    """Test vertex creation."""
    g = Graph(state_length=2)
    v1 = g.find_or_create_vertex([1, 2])
    assert v1 is not None
    assert list(v1.state()) == [1, 2]

    # Finding existing vertex should return same vertex
    v2 = g.find_or_create_vertex([1, 2])
    assert v1 == v2


def test_create_vertex():
    """Test create_vertex (always creates new)."""
    g = Graph(state_length=1)
    v1 = g.create_vertex([5])
    v2 = g.create_vertex([5])
    assert v1.index() != v2.index()  # Different indices


def test_find_vertex():
    """Test finding existing vertices."""
    g = Graph(state_length=1)
    v = g.find_or_create_vertex([10])

    found = g.find_vertex([10])
    assert found is not None
    assert found == v

    # find_vertex raises RuntimeError if not found
    try:
        not_found = g.find_vertex([99])
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass  # Expected


def test_vertex_at():
    """Test accessing vertices by index."""
    g = Graph(state_length=1)
    v = g.vertex_at(0)  # Starting vertex
    assert v is not None


def test_vertex_exists():
    """Test checking vertex existence."""
    g = Graph(state_length=1)
    g.find_or_create_vertex([5])

    assert g.vertex_exists([5])
    assert not g.vertex_exists([99])


def test_vertex_rate():
    """Test vertex rate computation."""
    g = Graph(state_length=1)
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])

    v1.add_edge(v2, 2.5)
    rate = v1.rate()
    assert abs(rate - 2.5) < 0.001


# ============================================================================
# Edge Operations Tests
# ============================================================================

def test_add_edge():
    """Test basic edge addition."""
    g = Graph(state_length=1)
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])

    v1.add_edge(v2, 1.5)

    edges = v1.edges()
    assert len(edges) > 0
    assert abs(edges[0].weight() - 1.5) < 0.001
    assert edges[0].to() == v2


def test_add_edge_alias():
    """Test ae() alias for add_edge."""
    g = Graph(state_length=1)
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])

    v1.ae(v2, 2.0)

    edges = v1.edges()
    assert len(edges) > 0
    assert abs(edges[0].weight() - 2.0) < 0.001


def test_edge_weight():
    """Test edge weight access."""
    g = Graph(state_length=1)
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])

    v1.add_edge(v2, 3.14)
    edge = v1.edges()[0]
    assert abs(edge.weight() - 3.14) < 0.001


def test_edge_update_weight():
    """Test updating edge weight."""
    g = Graph(state_length=1)
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])

    v1.add_edge(v2, 1.0)
    edge = v1.edges()[0]
    edge.update_weight(2.0)

    assert abs(edge.weight() - 2.0) < 0.001


# ============================================================================
# Matrix Operations Tests
# ============================================================================

def test_as_matrices_basic():
    """Test converting graph to matrices."""
    g = Graph(state_length=1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])

    start.add_edge(v1, 0.7)
    start.add_edge(v2, 0.3)
    v1.add_edge(v2, 1.0)

    g.normalize()

    matrices = g.as_matrices()
    assert isinstance(matrices, MatrixRepresentation)
    assert matrices.ipv is not None
    assert matrices.sim is not None
    assert matrices.states is not None

    # Check dimensions
    n = len(matrices.ipv)
    assert matrices.sim.shape == (n, n)
    assert len(matrices.states) == n


def test_from_matrices_basic():
    """Test creating graph from matrices."""
    ipv = np.array([0.6, 0.4])
    sim = np.array([[-2.0, 1.0], [0.0, -3.0]])

    g = Graph.from_matrices(ipv, sim)
    assert g is not None
    assert g.vertices_length() > 0


def test_round_trip_matrices():
    """Test as_matrices -> from_matrices round trip."""
    # Create original graph
    g_orig = Graph(state_length=1)
    start = g_orig.starting_vertex()
    v1 = g_orig.find_or_create_vertex([10])
    v2 = g_orig.find_or_create_vertex([20])

    start.add_edge(v1, 0.7)
    start.add_edge(v2, 0.3)
    v1.add_edge(v2, 1.5)
    g_orig.normalize()

    # Convert to matrices and back
    matrices = g_orig.as_matrices()

    # Normalize IPV to sum to 1
    ipv_normalized = matrices.ipv / matrices.ipv.sum()

    g_recon = Graph.from_matrices(ipv_normalized, matrices.sim, matrices.states)

    # Compare PDFs
    times = [0.5, 1.0, 2.0]
    for t in times:
        pdf_orig = g_orig.pdf(t, 100)
        pdf_recon = g_recon.pdf(t, 100)
        assert abs(pdf_orig - pdf_recon) < 1e-4


# ============================================================================
# Distribution Computations Tests
# ============================================================================

def test_pdf_continuous():
    """Test PDF computation."""
    g = Graph(state_length=1)
    start = g.starting_vertex()
    v = g.find_or_create_vertex([1])
    start.add_edge(v, 1.0)
    g.normalize()

    pdf = g.pdf(1.0, 100)
    assert pdf >= 0
    assert pdf < float('inf')


def test_cdf_continuous():
    """Test CDF computation."""
    g = Graph(state_length=1)
    start = g.starting_vertex()
    v = g.find_or_create_vertex([1])
    start.add_edge(v, 1.0)
    g.normalize()

    cdf = g.cdf(1.0, 100)
    assert 0 <= cdf <= 1


def test_pmf_discrete():
    """Test discrete PMF computation."""
    g = Graph(state_length=1)
    start = g.starting_vertex()
    v = g.find_or_create_vertex([1])
    start.add_edge(v, 1.0)
    g.normalize()

    pmf = g.pmf_discrete(5)
    assert pmf >= 0
    assert pmf <= 1


# ============================================================================
# Moments Tests
# ============================================================================

def test_expectation():
    """Test expectation computation."""
    g = Graph(state_length=1)
    start = g.starting_vertex()
    v = g.find_or_create_vertex([1])
    start.add_edge(v, 2.0)
    g.normalize()

    exp = g.expectation()
    assert exp > 0
    # For exponential with rate 2, expectation should be 0.5
    assert abs(exp - 0.5) < 0.1


def test_variance():
    """Test variance computation."""
    g = Graph(state_length=1)
    start = g.starting_vertex()
    v = g.find_or_create_vertex([1])
    start.add_edge(v, 2.0)
    g.normalize()

    var = g.variance()
    assert var >= 0


def test_moments():
    """Test general moment computation."""
    g = Graph(state_length=1)
    start = g.starting_vertex()
    v = g.find_or_create_vertex([1])
    start.add_edge(v, 1.0)
    g.normalize()

    # Test different moments (returns list/array)
    m1_list = g.moments(1)  # First moment (expectation)
    m2_list = g.moments(2)  # Second moment

    # Extract first element
    m1 = m1_list[0] if hasattr(m1_list, '__len__') else m1_list
    m2 = m2_list[0] if hasattr(m2_list, '__len__') else m2_list

    assert m1 > 0
    assert m2 > 0
    assert m2 >= m1  # Second moment >= first moment


# ============================================================================
# Sampling Tests
# ============================================================================

def test_sample_continuous():
    """Test continuous sampling."""
    g = Graph(state_length=1)
    start = g.starting_vertex()
    v = g.find_or_create_vertex([1])
    start.add_edge(v, 1.0)
    g.normalize()

    sample_result = g.sample()
    # sample() may return list or array
    sample = sample_result[0] if hasattr(sample_result, '__len__') else sample_result
    assert sample >= 0


def test_sample_discrete():
    """Test discrete sampling."""
    g = Graph(state_length=1)
    start = g.starting_vertex()
    v = g.find_or_create_vertex([1])
    start.add_edge(v, 1.0)
    g.normalize()

    sample_result = g.sample_discrete()
    # sample_discrete() may return list or array
    sample = sample_result[0] if hasattr(sample_result, '__len__') else sample_result
    assert sample >= 0
    assert isinstance(sample, (int, np.integer, float))


# ============================================================================
# Discretization Tests
# ============================================================================

def test_discretize_basic():
    """Test basic discretization."""
    g = Graph(state_length=1)
    start = g.starting_vertex()
    v = g.find_or_create_vertex([1])
    start.add_edge(v, 1.0)
    g.normalize()

    g_discrete, rewards = g.discretize(reward_rate=0.1)

    assert g_discrete is not None
    assert g_discrete.vertices_length() >= g.vertices_length()
    assert rewards.shape[1] == g_discrete.vertices_length()


# ============================================================================
# Graph Operations Tests
# ============================================================================

def test_normalize():
    """Test graph normalization."""
    g = Graph(state_length=1)
    start = g.starting_vertex()
    v = g.find_or_create_vertex([1])
    start.add_edge(v, 2.0)

    scaling_result = g.normalize()
    # normalize() may return list or array
    scaling = scaling_result[0] if hasattr(scaling_result, '__len__') else scaling_result
    assert scaling > 0


def test_copy():
    """Test graph copying."""
    g1 = Graph(state_length=1)
    start = g1.starting_vertex()
    v = g1.find_or_create_vertex([1])
    start.add_edge(v, 1.0)
    g1.normalize()

    g2 = g1.copy()

    # Should be different objects
    assert g2 is not g1

    # But same structure
    assert g2.vertices_length() == g1.vertices_length()
    assert g2.state_length() == g1.state_length()


def test_is_acyclic():
    """Test acyclic check."""
    g = Graph(state_length=1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    start.add_edge(v1, 1.0)
    v1.add_edge(v2, 1.0)

    # This should be acyclic
    assert g.is_acyclic()


# ============================================================================
# Serialization Tests
# ============================================================================

def test_serialize_basic():
    """Test basic serialization."""
    g = Graph(state_length=2)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1, 0])
    v2 = g.find_or_create_vertex([0, 1])
    start.add_edge(v1, 0.5)
    start.add_edge(v2, 0.5)
    v1.add_edge(v2, 1.0)
    g.normalize()

    serialized = g.serialize()

    # Check required fields
    assert 'states' in serialized
    assert 'edges' in serialized
    assert 'start_edges' in serialized
    assert 'state_dim' in serialized
    assert 'n_vertices' in serialized

    # Check dimensions
    assert serialized['state_dim'] == 2
    assert len(serialized['states']) > 0


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all tests."""
    tests = [
        # Graph Construction
        (test_construct_with_state_length, "Graph construction with state_length"),
        (test_construct_with_multidimensional_state, "Graph with multidimensional state"),
        (test_construct_with_callback, "Graph construction with callback"),
        (test_construct_parameterized, "Parameterized graph construction"),
        (test_starting_vertex, "Starting vertex exists"),

        # Vertex Operations
        (test_find_or_create_vertex, "Find or create vertex"),
        (test_create_vertex, "Create vertex"),
        (test_find_vertex, "Find vertex"),
        (test_vertex_at, "Vertex at index"),
        (test_vertex_exists, "Vertex exists check"),
        (test_vertex_rate, "Vertex rate computation"),

        # Edge Operations
        (test_add_edge, "Add edge"),
        (test_add_edge_alias, "Add edge alias (ae)"),
        (test_edge_weight, "Edge weight access"),
        (test_edge_update_weight, "Edge weight update"),

        # Matrix Operations
        (test_as_matrices_basic, "Convert to matrices"),
        (test_from_matrices_basic, "Create from matrices"),
        (test_round_trip_matrices, "Round-trip matrix conversion"),

        # Distribution Computations
        (test_pdf_continuous, "PDF computation"),
        (test_cdf_continuous, "CDF computation"),
        (test_pmf_discrete, "Discrete PMF computation"),

        # Moments
        (test_expectation, "Expectation computation"),
        (test_variance, "Variance computation"),
        (test_moments, "General moments computation"),

        # Sampling
        (test_sample_continuous, "Continuous sampling"),
        (test_sample_discrete, "Discrete sampling"),

        # Discretization
        (test_discretize_basic, "Basic discretization"),

        # Graph Operations
        (test_normalize, "Graph normalization"),
        (test_copy, "Graph copying"),
        (test_is_acyclic, "Acyclic check"),

        # Serialization
        (test_serialize_basic, "Basic serialization"),
    ]

    print("=" * 80)
    print("Running Comprehensive API Tests")
    print("=" * 80)
    print()

    passed = 0
    failed = 0

    for test_func, test_name in tests:
        if run_test(test_func, test_name):
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
