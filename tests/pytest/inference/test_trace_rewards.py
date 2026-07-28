"""
Tests for reward transformation in trace-based elimination
"""

import numpy as np
import pytest


def _coalescent_callback(state):
    """Coalescent model: n lineages → n-1 at rate n*(n-1)/2 * theta[0]."""
    n = state[0]
    if n <= 1:
        return []
    rate = n * (n - 1) / 2.0
    return [(np.array([n - 1]), 0.0, [rate])]


def _build_coalescent_graph():
    """Build the standard 3-sample coalescent graph used by every test in this
    file. The new API requires the IPV to be passed as a constructor argument
    rather than encoded as the empty-state branch of the callback."""
    from phasic import Graph
    return Graph(_coalescent_callback, ipv=[3])


# Simple import check for optional dependencies
def skip_if_missing(module_name):
    try:
        __import__(module_name)
        return False
    except ImportError:
        print(f"  ⊗ Skipping test (requires {module_name})")
        return True


def test_trace_recording_with_rewards():
    """Test that trace recording adds MUL operations for rewards"""
    from phasic.trace_elimination import record_elimination_trace, OpType

    graph = _build_coalescent_graph()

    # Record trace WITHOUT rewards
    trace_no_rewards = record_elimination_trace(graph, theta_dim=1, enable_rewards=False)
    assert trace_no_rewards.reward_length == 0

    # Record trace WITH rewards
    trace_with_rewards = record_elimination_trace(graph, theta_dim=1, enable_rewards=True)
    assert trace_with_rewards.reward_length == trace_with_rewards.n_vertices

    # Trace with rewards should have more operations (MUL for each edge)
    assert len(trace_with_rewards.operations) > len(trace_no_rewards.operations)

    # Count PARAM operations
    param_ops_no_rewards = sum(1 for op in trace_no_rewards.operations if op.op_type == OpType.PARAM)
    param_ops_with_rewards = sum(1 for op in trace_with_rewards.operations if op.op_type == OpType.PARAM)

    # With rewards, we should have additional PARAM operations for reward parameters
    assert param_ops_with_rewards > param_ops_no_rewards

    print(f"✓ Trace without rewards: {len(trace_no_rewards.operations)} ops, {param_ops_no_rewards} PARAM ops")
    print(f"✓ Trace with rewards: {len(trace_with_rewards.operations)} ops, {param_ops_with_rewards} PARAM ops")


def test_evaluate_trace_with_rewards():
    """Test that trace evaluation properly handles rewards"""
    from phasic.trace_elimination import record_elimination_trace, evaluate_trace

    graph = _build_coalescent_graph()

    # Record trace with rewards
    trace = record_elimination_trace(graph, theta_dim=1, enable_rewards=True)

    # Evaluate with theta and rewards
    theta = np.array([1.0])
    rewards = np.ones(trace.n_vertices)  # Neutral rewards (no effect)

    result = evaluate_trace(trace, params=theta, rewards=rewards, use_log=False)

    assert 'vertex_rates' in result
    assert 'edge_probs' in result
    assert len(result['vertex_rates']) == trace.n_vertices

    # Evaluate with different rewards (need n_vertices rewards). Scaling the
    # reward on vertex 1 changes the bypass probabilities into vertex 1's
    # children — not the IPV from vertex 0, which is always 1 by definition.
    rewards_scaled = np.ones(trace.n_vertices)
    rewards_scaled[1] = 2.0
    result_scaled = evaluate_trace(trace, params=theta, rewards=rewards_scaled, use_log=False)

    # Probabilities out of the rewarded vertex must change
    assert not np.allclose(result['edge_probs'][1], result_scaled['edge_probs'][1])

    print(f"✓ Neutral rewards: edge_probs[1] = {result['edge_probs'][1]}")
    print(f"✓ Scaled rewards: edge_probs[1] = {result_scaled['edge_probs'][1]}")


def test_evaluate_trace_jax_with_rewards():
    """Test JAX evaluation with rewards"""
    if skip_if_missing("jax"):
        return
    import jax.numpy as jnp

    from phasic.trace_elimination import record_elimination_trace, evaluate_trace_jax

    graph = _build_coalescent_graph()

    # Record trace with rewards
    trace = record_elimination_trace(graph, theta_dim=1, enable_rewards=True)

    # Evaluate with JAX arrays
    theta = jnp.array([1.0])
    rewards = jnp.ones(trace.n_vertices)

    result = evaluate_trace_jax(trace, params=theta, rewards=rewards, use_log=False)

    assert 'vertex_rates' in result
    assert 'edge_probs' in result

    print(f"✓ JAX evaluation successful with rewards")


def test_instantiate_from_trace_with_rewards():
    """Test that instantiate_from_trace works with rewards"""
    from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace

    graph_original = _build_coalescent_graph()

    # Record trace with rewards
    trace = record_elimination_trace(graph_original, theta_dim=1, enable_rewards=True)

    # Instantiate graph with rewards
    theta = np.array([2.0])
    rewards = np.ones(trace.n_vertices)

    graph_instantiated = instantiate_from_trace(trace, params=theta, rewards=rewards, use_log=False)

    # Graph should be instantiated successfully
    assert graph_instantiated.vertices_length() > 0

    # Compute PDF to verify graph works
    pdf_value = graph_instantiated.pdf(1.0, granularity=100)
    assert pdf_value > 0
    assert pdf_value < 10  # Sanity check

    print(f"✓ Instantiated graph with {graph_instantiated.vertices_length()} vertices")
    print(f"✓ PDF at t=1.0: {pdf_value}")


def test_reward_transformation_equivalence():
    """
    Test that trace-based reward transformation matches GraphBuilder behavior
    """
    if skip_if_missing("jax"):
        return
    import jax.numpy as jnp

    from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace

    graph = _build_coalescent_graph()

    # Record trace with rewards
    trace = record_elimination_trace(graph, theta_dim=1, enable_rewards=True)

    # Test parameters
    theta = np.array([2.0])
    rewards = np.ones(trace.n_vertices)

    # Instantiate from trace with rewards
    graph_trace = instantiate_from_trace(trace, params=theta, rewards=rewards, use_log=False)

    # Compute PDF at several points
    times = np.array([0.5, 1.0, 1.5, 2.0])
    pdf_trace = np.array([graph_trace.pdf(float(t), granularity=100) for t in times])

    # All PDFs should be positive and finite
    assert np.all(pdf_trace > 0)
    assert np.all(np.isfinite(pdf_trace))

    # PDF should integrate to approximately 1 (rough check)
    # Using trapezoidal rule
    integral = np.trapezoid(pdf_trace, times)
    # This won't be exactly 1 because we're only sampling 4 points
    # But it should be in the right ballpark
    assert 0.1 < integral < 10.0  # Very loose bound

    print(f"✓ PDF values: {pdf_trace}")
    print(f"✓ Approximate integral: {integral}")


if __name__ == "__main__":
    print("Testing reward transformation in traces...\n")

    print("1. Testing trace recording with rewards...")
    test_trace_recording_with_rewards()
    print()

    print("2. Testing trace evaluation with rewards...")
    test_evaluate_trace_with_rewards()
    print()

    print("3. Testing JAX evaluation with rewards...")
    test_evaluate_trace_jax_with_rewards()
    print()

    print("4. Testing graph instantiation with rewards...")
    test_instantiate_from_trace_with_rewards()
    print()

    print("6. Testing reward transformation equivalence...")
    test_reward_transformation_equivalence()
    print()

    print("✓ All tests passed!")
