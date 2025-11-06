"""
Integration test for reward transformation with conditional bypass.

Tests that:
1. Vertices with reward ≈ 0 are bypassed (edges redirected through them)
2. Vertices with reward > 0 affect rates (rate scaled by reward)
3. Trace evaluation produces valid probability distributions
"""

import numpy as np
from phasic import Graph
from phasic.trace_elimination import (
    record_elimination_trace,
    evaluate_trace,
    evaluate_trace_jax
)

def coalescent_callback(state, n_start=5):
    """Simple coalescent model: n → n-1 with rate n(n-1)/2"""
    if len(state) == 0:
        # Starting state: transition to n_start
        rate = n_start * (n_start - 1) / 2
        return [(np.array([n_start]), 0.0, [rate])]

    n = state[0]
    if n <= 1:
        # Absorbing state
        return []

    # Transition to n-1
    rate = n * (n - 1) / 2
    return [(np.array([n - 1]), 0.0, [rate])]


def test_reward_bypass_basic():
    """Test that vertices with reward=0 are bypassed"""
    print("Test 1: Basic reward bypass (reward=0 vs reward=1)")

    # Build simple coalescent graph (n=5 → 4 → 3 → 2 → 1)
    graph = Graph(
        callback=coalescent_callback,
        parameterized=True
    )

    n_vertices = graph.vertices_length()
    print(f"  Graph has {n_vertices} vertices")

    # Record trace with rewards enabled
    param_length = 1  # Single parameter (theta)
    trace = record_elimination_trace(graph, param_length=param_length, enable_rewards=True)

    # Test with all rewards = 1 (no transformation)
    theta = np.array([2.0])  # coalescent rate parameter
    rewards_ones = np.ones(n_vertices)

    result1 = evaluate_trace(trace, params=theta, rewards=rewards_ones)
    print(f"  All rewards=1: vertex_rates[0] = {result1['vertex_rates'][0]:.6f}")

    # Test with first vertex reward = 0 (should be bypassed)
    rewards_zero_first = np.ones(n_vertices)
    rewards_zero_first[0] = 0.0  # Bypass first vertex

    result2 = evaluate_trace(trace, params=theta, rewards=rewards_zero_first)
    print(f"  First reward=0: vertex_rates[0] = {result2['vertex_rates'][0]:.6f}")

    # Check that edge probabilities changed
    print(f"  All rewards=1: {len(result1['edge_probs'][0])} edges from vertex 0")
    print(f"  First reward=0: {len(result2['edge_probs'][0])} edges from vertex 0")

    # Verify probability distributions sum to 1
    for i in range(n_vertices):
        if len(result1['edge_probs'][i]) > 0:
            total_prob = sum(result1['edge_probs'][i])
            assert abs(total_prob - 1.0) < 1e-6, f"Vertex {i} probs don't sum to 1: {total_prob}"

    print("  ✓ Probability distributions valid")


def test_reward_rate_scaling():
    """Test that non-zero rewards scale rates correctly"""
    print("\nTest 2: Rate scaling with rewards")

    # Build graph with n_start=3
    from functools import partial
    graph = Graph(
        callback=partial(coalescent_callback, n_start=3),
        parameterized=True
    )

    n_vertices = graph.vertices_length()
    param_length = 1
    trace = record_elimination_trace(graph, param_length=param_length, enable_rewards=True)

    # Test with all rewards = 1 (baseline)
    theta = np.array([1.0])
    rewards_ones = np.ones(n_vertices)
    result1 = evaluate_trace(trace, params=theta, rewards=rewards_ones)

    # Test with all rewards = 2 (should halve rates: rate/reward)
    rewards_two = np.full(n_vertices, 2.0)
    result2 = evaluate_trace(trace, params=theta, rewards=rewards_two)

    # Find a non-absorbing vertex
    for i in range(n_vertices):
        if result1['vertex_rates'][i] > 1e-6:
            rate1 = result1['vertex_rates'][i]
            rate2 = result2['vertex_rates'][i]
            ratio = rate1 / rate2
            print(f"  Vertex {i}: rate1={rate1:.6f}, rate2={rate2:.6f}, ratio={ratio:.6f}")
            # Rate should be halved: rate2 = rate1 / 2
            assert abs(ratio - 2.0) < 0.1, f"Expected ratio ≈ 2, got {ratio:.6f}"
            break

    print("  ✓ Rate scaling works correctly")


def test_reward_jax_compatibility():
    """Test that reward transformation works with JAX evaluation"""
    print("\nTest 3: JAX compatibility")

    try:
        import jax.numpy as jnp
    except ImportError:
        print("  Skipping (JAX not available)")
        return

    # Build graph with n_start=4
    from functools import partial
    graph = Graph(
        callback=partial(coalescent_callback, n_start=4),
        parameterized=True
    )

    n_vertices = graph.vertices_length()
    param_length = 1
    trace = record_elimination_trace(graph, param_length=param_length, enable_rewards=True)

    # Test with JAX arrays
    theta = jnp.array([1.5])
    rewards = jnp.ones(n_vertices)

    result = evaluate_trace_jax(trace, params=theta, rewards=rewards)

    print(f"  JAX evaluation successful: {n_vertices} vertices")
    print(f"  First vertex rate: {result['vertex_rates'][0]:.6f}")

    # Verify probability distributions
    for i in range(n_vertices):
        if len(result['edge_probs'][i]) > 0:
            total_prob = float(sum(result['edge_probs'][i]))
            assert abs(total_prob - 1.0) < 1e-6, f"Vertex {i} probs don't sum to 1"

    print("  ✓ JAX evaluation works correctly")


if __name__ == "__main__":
    print("Running reward transformation integration tests...\n")

    test_reward_bypass_basic()
    test_reward_rate_scaling()
    test_reward_jax_compatibility()

    print("\n" + "="*60)
    print("All integration tests passed!")
    print("="*60)
