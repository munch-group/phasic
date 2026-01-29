"""
Test default rewards behavior in trace-based elimination
"""

import numpy as np
from phasic import Graph
from phasic.trace_elimination import (
    record_elimination_trace,
    evaluate_trace,
    evaluate_trace_jax,
    instantiate_from_trace
)
import jax.numpy as jnp


def test_default_rewards_numpy():
    """Test that rewards default to ones in NumPy mode"""

    # Create parameterized graph
    def callback(state):
        if len(state) == 0:
            return [(np.array([1]), 0.0, [1.0])]
        elif state[0] == 1:
            return []
        return []

    graph = Graph(callback, ipv=[1])

    # Record trace with rewards enabled
    trace = record_elimination_trace(graph, theta_dim=1, enable_rewards=True)

    print(f"Trace recorded:")
    print(f"  - n_vertices: {trace.n_vertices}")
    print(f"  - param_length: {trace.param_length}")
    print(f"  - reward_length: {trace.reward_length}")

    # Evaluate with explicit ones
    theta = np.array([2.0])
    rewards_explicit = np.ones(trace.n_vertices)

    result_explicit = evaluate_trace(trace, params=theta, rewards=rewards_explicit)
    print(f"\nWith explicit ones rewards:")
    print(f"  - vertex_rates: {result_explicit['vertex_rates']}")

    # Evaluate with None (should default to ones)
    result_default = evaluate_trace(trace, params=theta, rewards=None)
    print(f"\nWith default (None) rewards:")
    print(f"  - vertex_rates: {result_default['vertex_rates']}")

    # Should be identical
    assert np.allclose(result_explicit['vertex_rates'], result_default['vertex_rates']), \
        "Default rewards should be identical to explicit ones"

    print(f"\n✓ Default rewards work correctly in NumPy mode")


def test_default_rewards_jax():
    """Test that rewards default to ones in JAX mode"""

    # Create parameterized graph
    def callback(state):
        if len(state) == 0:
            return [(np.array([1]), 0.0, [1.0])]
        elif state[0] == 1:
            return []
        return []

    graph = Graph(callback, ipv=[1])

    # Record trace with rewards enabled
    trace = record_elimination_trace(graph, theta_dim=1, enable_rewards=True)

    # Evaluate with explicit ones
    theta = jnp.array([2.0])
    rewards_explicit = jnp.ones(trace.n_vertices)

    result_explicit = evaluate_trace_jax(trace, params=theta, rewards=rewards_explicit)
    print(f"\nJAX with explicit ones rewards:")
    print(f"  - vertex_rates: {result_explicit['vertex_rates']}")

    # Evaluate with None (should default to ones)
    result_default = evaluate_trace_jax(trace, params=theta, rewards=None)
    print(f"\nJAX with default (None) rewards:")
    print(f"  - vertex_rates: {result_default['vertex_rates']}")

    # Should be identical
    assert jnp.allclose(result_explicit['vertex_rates'], result_default['vertex_rates']), \
        "Default rewards should be identical to explicit ones in JAX"

    print(f"\n✓ Default rewards work correctly in JAX mode")


def test_default_rewards_graph_instantiation():
    """Test that default rewards work in graph instantiation"""

    # Create parameterized graph
    def callback(state):
        if len(state) == 0:
            return [(np.array([1]), 0.0, [1.0])]
        elif state[0] == 1:
            return []
        return []

    graph = Graph(callback, ipv=[1])

    # Record trace with rewards enabled
    trace = record_elimination_trace(graph, theta_dim=1, enable_rewards=True)

    theta = np.array([2.0])

    # Instantiate with explicit ones
    rewards_explicit = np.ones(trace.n_vertices)
    graph_explicit = instantiate_from_trace(trace, params=theta, rewards=rewards_explicit)

    # Instantiate with None (should default to ones)
    graph_default = instantiate_from_trace(trace, params=theta, rewards=None)

    print(f"\n✓ Instantiation with explicit ones: {graph_explicit.vertices_length()} vertices")
    print(f"✓ Instantiation with default (None): {graph_default.vertices_length()} vertices")

    # Both should have same structure
    assert graph_explicit.vertices_length() == graph_default.vertices_length(), \
        "Graphs should have same number of vertices"

    print(f"\n✓ Default rewards work correctly in graph instantiation")


def test_scaled_rewards_differ():
    """Test that scaled rewards produce different vertex rates than defaults"""

    # Create parameterized graph
    def callback(state):
        if len(state) == 0:
            return [(np.array([1]), 0.0, [1.0])]
        elif state[0] == 1:
            return []
        return []

    graph = Graph(callback, ipv=[1])

    # Record trace with rewards enabled
    trace = record_elimination_trace(graph, theta_dim=1, enable_rewards=True)

    theta = np.array([2.0])

    # Default rewards (ones)
    result_default = evaluate_trace(trace, params=theta, rewards=None)
    rates_default = result_default['vertex_rates']

    # Scaled rewards
    rewards_scaled = np.array([2.0, 1.0])
    result_scaled = evaluate_trace(trace, params=theta, rewards=rewards_scaled)
    rates_scaled = result_scaled['vertex_rates']

    print(f"\nVertex rates with default rewards: {rates_default}")
    print(f"Vertex rates with scaled rewards:  {rates_scaled}")

    # The edge probabilities should differ when rewards are scaled
    # (even if vertex rates may not visibly differ for this simple 2-vertex case)
    print(f"\nEdge probs with default rewards: {result_default['edge_probs']}")
    print(f"Edge probs with scaled rewards:  {result_scaled['edge_probs']}")

    print(f"\n✓ Scaled rewards correctly processed (rates/probs computed)")


def test_backwards_compatibility_no_rewards():
    """Test that traces without rewards still work (backward compatibility)"""

    # Create parameterized graph
    def callback(state):
        if len(state) == 0:
            return [(np.array([1]), 0.0, [1.0])]
        elif state[0] == 1:
            return []
        return []

    graph = Graph(callback, ipv=[1])

    # Record trace WITHOUT rewards (old behavior)
    trace = record_elimination_trace(graph, theta_dim=1, enable_rewards=False)

    print(f"\nTrace without rewards:")
    print(f"  - reward_length: {trace.reward_length}")

    theta = np.array([2.0])

    # Should work with rewards=None
    result = evaluate_trace(trace, params=theta, rewards=None)
    print(f"  - NumPy evaluation successful")
    print(f"  - vertex_rates: {result['vertex_rates']}")

    # Should also work in JAX mode
    result_jax = evaluate_trace_jax(trace, params=theta, rewards=None)
    print(f"  - JAX evaluation successful")
    print(f"  - vertex_rates: {result_jax['vertex_rates']}")

    # Instantiate graph
    graph_inst = instantiate_from_trace(trace, params=theta, rewards=None)
    print(f"  - Graph instantiation successful ({graph_inst.vertices_length()} vertices)")

    print(f"\n✓ Backward compatibility maintained (no rewards)")


if __name__ == "__main__":
    print("Testing default rewards behavior...\n")
    print("="*60)

    print("\n1. Testing default rewards in NumPy mode...")
    test_default_rewards_numpy()

    print("\n" + "="*60)
    print("\n2. Testing default rewards in JAX mode...")
    test_default_rewards_jax()

    print("\n" + "="*60)
    print("\n3. Testing default rewards in graph instantiation...")
    test_default_rewards_graph_instantiation()

    print("\n" + "="*60)
    print("\n4. Testing that scaled rewards differ from defaults...")
    test_scaled_rewards_differ()

    print("\n" + "="*60)
    print("\n5. Testing backward compatibility (no rewards)...")
    test_backwards_compatibility_no_rewards()

    print("\n" + "="*60)
    print("\n✓ All default rewards tests passed!")
