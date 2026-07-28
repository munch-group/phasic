"""
Simple tests for reward transformation in trace-based elimination.

Uses the canonical Exp(theta) graph: start → [2] (transient) → [1] (absorbing).
The transient vertex carries the parameterized rate edge — putting the rate
edge on the IPV would encode an instant-absorption distribution (PDF = 0)
instead of an exponential.
"""

import numpy as np


def _build_exp_graph():
    """Build start → [2] (transient) → [1] (absorbing) with rate = theta[0]."""
    from phasic import Graph
    g = Graph(1)
    v_start = g.starting_vertex()
    v_trans = g.find_or_create_vertex([2])
    v_abs = g.find_or_create_vertex([1])
    v_start.add_edge(v_trans, 1.0)
    v_trans.add_edge_parameterized(v_abs, 0.0, [1.0])
    return g


def _transient_index(trace):
    """Return the trace index of the transient state [2]."""
    return next(i for i in range(trace.n_vertices)
                if list(trace.states[i]) == [2])


def test_simple_reward_transformation():
    """Test reward transformation with a simple exponential model"""
    from phasic.trace_elimination import record_elimination_trace, evaluate_trace, instantiate_from_trace

    graph = _build_exp_graph()

    # Record trace WITH rewards
    trace = record_elimination_trace(graph, theta_dim=1, enable_rewards=True)

    print(f"✓ Recorded trace: {trace.n_vertices} vertices, {len(trace.operations)} operations")
    print(f"  - param_length: {trace.param_length}")
    print(f"  - reward_length: {trace.reward_length}")

    # Evaluate with theta and neutral rewards
    theta = np.array([2.0])  # Rate = 2.0
    rewards_neutral = np.ones(trace.n_vertices)

    result_neutral = evaluate_trace(trace, params=theta, rewards=rewards_neutral, use_log=False)
    print(f"✓ Evaluated with neutral rewards")
    print(f"  - vertex_rates: {result_neutral['vertex_rates']}")

    # Evaluate with reward scaled on the transient vertex (the only one
    # that contributes sojourn time).
    rewards_scaled = np.ones(trace.n_vertices)
    rewards_scaled[_transient_index(trace)] = 2.0

    result_scaled = evaluate_trace(trace, params=theta, rewards=rewards_scaled, use_log=False)
    print(f"✓ Evaluated with scaled rewards (2x on transient vertex)")
    print(f"  - vertex_rates: {result_scaled['vertex_rates']}")

    # Instantiate graphs
    graph_neutral = instantiate_from_trace(trace, params=theta, rewards=rewards_neutral, use_log=False)
    graph_scaled = instantiate_from_trace(trace, params=theta, rewards=rewards_scaled, use_log=False)

    # Compute PDF at t=1.0
    pdf_neutral = graph_neutral.pdf(1.0, granularity=200)
    pdf_scaled = graph_scaled.pdf(1.0, granularity=200)

    print(f"✓ PDF at t=1.0:")
    print(f"  - Neutral rewards: {pdf_neutral}")
    print(f"  - Scaled rewards:  {pdf_scaled}")

    # With scaled rewards, the PDF should be different
    # (reward transformation scales the exit rate from the transient vertex)
    assert pdf_neutral > 0, "Neutral PDF should be positive"
    assert pdf_scaled > 0, "Scaled PDF should be positive"
    assert not np.isclose(pdf_neutral, pdf_scaled), "PDFs should differ with different rewards"

    print(f"✓ Reward transformation changes PDF as expected")


def test_reward_transformation_theory():
    """
    Test that reward transformation matches theoretical expectation

    For an exponential distribution with rate λ:
    - PDF(t) = λ * exp(-λt)

    With reward r on the transient vertex:
    - The effective rate becomes λ * r
    - PDF(t) = λr * exp(-λrt)
    """
    from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace

    graph = _build_exp_graph()
    trace = record_elimination_trace(graph, theta_dim=1, enable_rewards=True)

    # Test parameters
    theta = np.array([2.0])  # λ = 2.0
    reward = 3.0  # r = 3.0

    # Expected: λr = 2.0 * 3.0 = 6.0
    # At t=1.0: PDF = 6.0 * exp(-6.0 * 1.0) = 6.0 * exp(-6.0) ≈ 0.0149

    rewards = np.ones(trace.n_vertices)
    rewards[_transient_index(trace)] = reward

    graph_transformed = instantiate_from_trace(trace, params=theta, rewards=rewards, use_log=False)
    # Effective rate is 6.0 — uniformization needs higher granularity than
    # default to stay under 5% at rate*t = 6.
    pdf_actual = graph_transformed.pdf(1.0, granularity=2000)

    # Theoretical value
    lambda_r = theta[0] * reward
    pdf_expected = lambda_r * np.exp(-lambda_r * 1.0)

    print(f"✓ Theoretical test:")
    print(f"  - λ = {theta[0]}, r = {reward}")
    print(f"  - λr = {lambda_r}")
    print(f"  - PDF(t=1.0) expected: {pdf_expected}")
    print(f"  - PDF(t=1.0) actual:   {pdf_actual}")
    print(f"  - Relative error: {abs(pdf_actual - pdf_expected) / pdf_expected * 100:.2f}%")

    # Allow 5% error due to discretization
    assert abs(pdf_actual - pdf_expected) / pdf_expected < 0.05, \
        f"PDF should match theoretical value within 5%"

    print(f"✓ Reward transformation matches theory")


if __name__ == "__main__":
    print("Testing reward transformation (simple tests)...\n")

    print("1. Testing simple reward transformation...")
    test_simple_reward_transformation()
    print()

    print("2. Testing reward transformation theory...")
    test_reward_transformation_theory()
    print()

    print("✓ All simple tests passed!")
