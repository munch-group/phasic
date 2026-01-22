"""
Simple tests for reward transformation in trace-based elimination
Using a simple 2-state model for easier testing
"""

import numpy as np

def test_simple_reward_transformation():
    """Test reward transformation with a simple 2-state model"""
    from phasic import Graph
    from phasic.trace_elimination import record_elimination_trace, evaluate_trace, instantiate_from_trace

    # Create simple 2-state model: Start → Absorbing
    # Rate = theta[0]
    def callback(state):
        if len(state) == 0:
            # From starting vertex to state 1
            return [(np.array([1]), 0.0, [1.0])]
        elif state[0] == 1:
            # State 1 is absorbing
            return []
        return []

    graph = Graph(callback, parameterized=True)

    # Record trace WITH rewards
    trace = record_elimination_trace(graph, theta_dim=1, enable_rewards=True)

    print(f"✓ Recorded trace: {trace.n_vertices} vertices, {len(trace.operations)} operations")
    print(f"  - param_length: {trace.param_length}")
    print(f"  - reward_length: {trace.reward_length}")

    # Evaluate with theta and neutral rewards
    theta = np.array([2.0])  # Rate = 2.0
    rewards_neutral = np.ones(trace.n_vertices)

    result_neutral = evaluate_trace(trace, params=theta, rewards=rewards_neutral)
    print(f"✓ Evaluated with neutral rewards")
    print(f"  - vertex_rates: {result_neutral['vertex_rates']}")

    # Evaluate with scaled rewards
    rewards_scaled = np.ones(trace.n_vertices)
    rewards_scaled[0] = 2.0  # Scale the starting vertex

    result_scaled = evaluate_trace(trace, params=theta, rewards=rewards_scaled)
    print(f"✓ Evaluated with scaled rewards (2x on vertex 0)")
    print(f"  - vertex_rates: {result_scaled['vertex_rates']}")

    # Instantiate graphs
    graph_neutral = instantiate_from_trace(trace, params=theta, rewards=rewards_neutral)
    graph_scaled = instantiate_from_trace(trace, params=theta, rewards=rewards_scaled)

    # Compute PDF at t=1.0
    pdf_neutral = graph_neutral.pdf(1.0, granularity=100)
    pdf_scaled = graph_scaled.pdf(1.0, granularity=100)

    print(f"✓ PDF at t=1.0:")
    print(f"  - Neutral rewards: {pdf_neutral}")
    print(f"  - Scaled rewards:  {pdf_scaled}")

    # With scaled rewards, the PDF should be different
    # (reward transformation scales the exit rate from vertex 0)
    assert pdf_neutral > 0, "Neutral PDF should be positive"
    assert pdf_scaled > 0, "Scaled PDF should be positive"
    assert not np.isclose(pdf_neutral, pdf_scaled), "PDFs should differ with different rewards"

    print(f"✓ Reward transformation changes PDF as expected")


def test_reward_transformation_theory():
    """
    Test that reward transformation matches theoretical expectation

    For an exponential distribution with rate λ:
    - PDF(t) = λ * exp(-λt)

    With reward transformation (reward r on the starting vertex):
    - The exit rate becomes λ * r
    - PDF(t) = λr * exp(-λrt)
    """
    from phasic import Graph
    from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace

    # Create exponential model
    def callback(state):
        if len(state) == 0:
            return [(np.array([1]), 0.0, [1.0])]  # Rate = theta[0]
        elif state[0] == 1:
            return []  # Absorbing
        return []

    graph = Graph(callback, parameterized=True)
    trace = record_elimination_trace(graph, theta_dim=1, enable_rewards=True)

    # Test parameters
    theta = np.array([2.0])  # λ = 2.0
    reward = 3.0  # r = 3.0

    # Expected: λr = 2.0 * 3.0 = 6.0
    # At t=1.0: PDF = 6.0 * exp(-6.0 * 1.0) = 6.0 * exp(-6.0) ≈ 0.0149

    rewards = np.ones(trace.n_vertices)
    rewards[0] = reward

    graph_transformed = instantiate_from_trace(trace, params=theta, rewards=rewards)
    pdf_actual = graph_transformed.pdf(1.0, granularity=200)

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
