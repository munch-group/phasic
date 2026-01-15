"""
Test parameterized IPV (Initial Probability Vector) edge updates.

This test verifies that:
1. Constant IPV edges (created with scalar) remain unchanged after update_weights()
2. Parameterized IPV edges (created with array) ARE updated by update_weights()
3. Mixed graphs work correctly (constant IPV + parameterized graph edges)
4. Trace recording/instantiation works with parameterized IPV
5. Serialization/deserialization preserves parameterized IPV
"""

import numpy as np
import pytest
from phasic import Graph


def test_constant_ipv_remains_unchanged():
    """Constant IPV edges (created with scalar) should NOT be updated."""
    g = Graph(1)  # state_length=1

    # Create vertices
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    v_absorbing = g.find_or_create_vertex([0])

    # Constant IPV edges (scalar syntax)
    g.starting_vertex().add_edge(v1, 0.4)
    g.starting_vertex().add_edge(v2, 0.6)

    # Parameterized graph edges (array syntax - locks graph to parameterized mode)
    v1.add_edge(v_absorbing, [1.0, 0.0])
    v2.add_edge(v_absorbing, [0.0, 1.0])

    # Get initial IPV weights
    ipv_edges = list(g.starting_vertex().edges())
    initial_weight_0 = ipv_edges[0].weight()
    initial_weight_1 = ipv_edges[1].weight()

    # Update parameters
    g.update_weights([2.0, 3.0])

    # IPV weights should be UNCHANGED (constant edges)
    assert abs(ipv_edges[0].weight() - initial_weight_0) < 1e-10, \
        f"Constant IPV edge 0 changed: {initial_weight_0} -> {ipv_edges[0].weight()}"
    assert abs(ipv_edges[1].weight() - initial_weight_1) < 1e-10, \
        f"Constant IPV edge 1 changed: {initial_weight_1} -> {ipv_edges[1].weight()}"

    # But graph edges SHOULD be updated
    graph_edges_v1 = list(v1.edges())
    graph_edges_v2 = list(v2.edges())
    assert abs(graph_edges_v1[0].weight() - 2.0) < 1e-10, \
        f"v1 edge weight wrong: expected 2.0, got {graph_edges_v1[0].weight()}"
    assert abs(graph_edges_v2[0].weight() - 3.0) < 1e-10, \
        f"v2 edge weight wrong: expected 3.0, got {graph_edges_v2[0].weight()}"


def test_parameterized_ipv_updates():
    """Parameterized IPV edges (created with array) SHOULD be updated."""
    g = Graph(1)  # state_length=1

    # Create vertices
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    v_absorbing = g.find_or_create_vertex([0])

    # Parameterized graph edges first (locks mode)
    v1.add_edge(v_absorbing, [1.0, 0.0])
    v2.add_edge(v_absorbing, [0.0, 1.0])

    # Parameterized IPV edges (array syntax)
    # These should be updated by update_weights()
    g.starting_vertex().add_edge(v1, [0.4, 0.0])  # weight = 0.4 * theta[0]
    g.starting_vertex().add_edge(v2, [0.6, 0.0])  # weight = 0.6 * theta[0]

    # Update parameters
    theta = np.array([2.0, 3.0])
    g.update_weights(theta)

    # IPV edges should NOW be updated (parameterized edges)
    ipv_edges = list(g.starting_vertex().edges())

    # Expected weights: [0.4, 0.0] · [2.0, 3.0] = 0.8
    #                   [0.6, 0.0] · [2.0, 3.0] = 1.2
    expected_weight_0 = 0.4 * 2.0
    expected_weight_1 = 0.6 * 2.0

    assert abs(ipv_edges[0].weight() - expected_weight_0) < 1e-10, \
        f"Parameterized IPV edge 0 wrong: expected {expected_weight_0}, got {ipv_edges[0].weight()}"
    assert abs(ipv_edges[1].weight() - expected_weight_1) < 1e-10, \
        f"Parameterized IPV edge 1 wrong: expected {expected_weight_1}, got {ipv_edges[1].weight()}"


def test_mixed_ipv_coefficients():
    """Test IPV edges with multiple non-zero coefficients."""
    g = Graph(1)  # state_length=1

    # Create vertices
    v1 = g.find_or_create_vertex([1])
    v_absorbing = g.find_or_create_vertex([0])

    # Graph edge (locks mode)
    v1.add_edge(v_absorbing, [1.0, 0.0, 0.0])

    # Parameterized IPV with multiple coefficients
    g.starting_vertex().add_edge(v1, [0.2, 0.3, 0.5])  # weight = 0.2*θ₀ + 0.3*θ₁ + 0.5*θ₂

    # Update parameters
    theta = np.array([1.0, 2.0, 3.0])
    g.update_weights(theta)

    # Expected weight: 0.2*1.0 + 0.3*2.0 + 0.5*3.0 = 0.2 + 0.6 + 1.5 = 2.3
    expected_weight = 0.2 * 1.0 + 0.3 * 2.0 + 0.5 * 3.0

    ipv_edges = list(g.starting_vertex().edges())
    assert abs(ipv_edges[0].weight() - expected_weight) < 1e-10, \
        f"Multi-coefficient IPV edge wrong: expected {expected_weight}, got {ipv_edges[0].weight()}"


def test_trace_with_parameterized_ipv():
    """Test trace recording and instantiation with parameterized IPV."""
    from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace

    # Build parameterized graph with parameterized IPV
    g = Graph(1)  # state_length=1

    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    v_absorbing = g.find_or_create_vertex([0])

    # Parameterized graph edges
    v1.add_edge(v_absorbing, [1.0, 0.0])
    v2.add_edge(v_absorbing, [0.0, 1.0])

    # Parameterized IPV
    g.starting_vertex().add_edge(v1, [0.3, 0.1])
    g.starting_vertex().add_edge(v2, [0.5, 0.2])

    # Record trace
    trace = record_elimination_trace(g, param_length=2)

    # Instantiate with concrete parameters
    theta = np.array([2.0, 5.0])
    concrete_g = instantiate_from_trace(trace, theta)

    # Check IPV edge weights in instantiated graph
    # Expected: [0.3, 0.1] · [2.0, 5.0] = 0.6 + 0.5 = 1.1
    #           [0.5, 0.2] · [2.0, 5.0] = 1.0 + 1.0 = 2.0
    ipv_edges = list(concrete_g.starting_vertex().edges())

    expected_weight_0 = 0.3 * 2.0 + 0.1 * 5.0
    expected_weight_1 = 0.5 * 2.0 + 0.2 * 5.0

    assert abs(ipv_edges[0].weight() - expected_weight_0) < 1e-10, \
        f"Trace IPV edge 0 wrong: expected {expected_weight_0}, got {ipv_edges[0].weight()}"
    assert abs(ipv_edges[1].weight() - expected_weight_1) < 1e-10, \
        f"Trace IPV edge 1 wrong: expected {expected_weight_1}, got {ipv_edges[1].weight()}"


def test_serialization_with_parameterized_ipv():
    """Test that serialization preserves parameterized IPV edges."""
    # Build graph with parameterized IPV
    g = Graph(1)  # state_length=1

    v1 = g.find_or_create_vertex([1])
    v_absorbing = g.find_or_create_vertex([0])

    # Graph edge
    v1.add_edge(v_absorbing, [1.0, 0.0])

    # Parameterized IPV
    g.starting_vertex().add_edge(v1, [0.7, 0.3])

    # Update weights
    theta1 = np.array([2.0, 3.0])
    g.update_weights(theta1)

    # Serialize
    serialized = g.serialize()

    # Check that start_param_edges is populated
    assert 'start_param_edges' in serialized, "start_param_edges missing from serialization"
    start_param_edges = serialized['start_param_edges']
    assert start_param_edges.shape[0] > 0, "start_param_edges should not be empty"

    # Deserialize and rebuild
    g2 = Graph.from_serialized(serialized)

    # Update with different parameters
    theta2 = np.array([5.0, 7.0])
    g2.update_weights(theta2)

    # Check IPV edge weights
    # Expected: [0.7, 0.3] · [5.0, 7.0] = 3.5 + 2.1 = 5.6
    ipv_edges = list(g2.starting_vertex().edges())
    expected_weight = 0.7 * 5.0 + 0.3 * 7.0

    assert abs(ipv_edges[0].weight() - expected_weight) < 1e-10, \
        f"Deserialized IPV edge wrong: expected {expected_weight}, got {ipv_edges[0].weight()}"


def test_backward_compatibility_constant_ipv():
    """Ensure constant IPV behavior is preserved (backward compatibility)."""
    # This is the traditional use case: constant IPV, parameterized graph
    g = Graph(1)  # state_length=1

    v1 = g.find_or_create_vertex([1])
    v_absorbing = g.find_or_create_vertex([0])

    # Constant IPV (scalar syntax) - traditional behavior
    g.starting_vertex().add_edge(v1, 1.0)

    # Parameterized graph edge
    v1.add_edge(v_absorbing, [2.0])

    # Update weights multiple times
    for theta_val in [1.0, 2.0, 3.0, 5.0]:
        g.update_weights([theta_val])

        # IPV should ALWAYS be 1.0 (constant)
        ipv_edges = list(g.starting_vertex().edges())
        assert abs(ipv_edges[0].weight() - 1.0) < 1e-10, \
            f"Constant IPV changed with theta={theta_val}: {ipv_edges[0].weight()}"

        # Graph edge should update
        graph_edges = list(v1.edges())
        expected_graph_weight = 2.0 * theta_val
        assert abs(graph_edges[0].weight() - expected_graph_weight) < 1e-10, \
            f"Graph edge wrong with theta={theta_val}: expected {expected_graph_weight}, got {graph_edges[0].weight()}"


def test_coefficient_length_validation():
    """Ensure parameterized IPV edges must match graph param_length."""
    g = Graph(1)  # state_length=1

    v1 = g.find_or_create_vertex([1])
    v_absorbing = g.find_or_create_vertex([0])

    # Lock graph to param_length=2
    v1.add_edge(v_absorbing, [1.0, 0.0])

    # Try to add IPV edge with wrong coefficient length
    with pytest.raises(Exception) as exc_info:
        g.starting_vertex().add_edge(v1, [0.5, 0.3, 0.2])  # 3 coefficients, should fail

    # Should get error about coefficient length mismatch
    assert "mismatch" in str(exc_info.value).lower() or "length" in str(exc_info.value).lower(), \
        f"Expected coefficient length error, got: {exc_info.value}"


if __name__ == "__main__":
    # Run tests
    test_constant_ipv_remains_unchanged()
    print("✓ test_constant_ipv_remains_unchanged")

    test_parameterized_ipv_updates()
    print("✓ test_parameterized_ipv_updates")

    test_mixed_ipv_coefficients()
    print("✓ test_mixed_ipv_coefficients")

    test_trace_with_parameterized_ipv()
    print("✓ test_trace_with_parameterized_ipv")

    test_serialization_with_parameterized_ipv()
    print("✓ test_serialization_with_parameterized_ipv")

    test_backward_compatibility_constant_ipv()
    print("✓ test_backward_compatibility_constant_ipv")

    test_coefficient_length_validation()
    print("✓ test_coefficient_length_validation")

    print("\nAll tests passed! ✓")
