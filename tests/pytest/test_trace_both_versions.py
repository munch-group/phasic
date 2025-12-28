"""
Test both trace versions: simple (without rewards) and full (with rewards)
"""

import numpy as np

def test_simple_version_no_rewards():
    """Test that simple version works correctly without rewards"""
    from phasic import Graph
    from phasic.trace_elimination import record_elimination_trace_simple, instantiate_from_trace

    # Create simple 2-state exponential model manually
    # State [0] (starting) -> State [1] (absorbing) with rate theta[0]
    graph = Graph(state_length=1)
    v_start = graph.starting_vertex()  # State [0]
    v_abs = graph.find_or_create_vertex([1])  # State [1] - absorbing
    # Add parameterized edge: weight = 0.0 + 1.0 * theta[0]
    v_start.add_edge_parameterized(v_abs, 0.0, [1.0])  # weight, edge_state

    # Record with simple version
    trace = record_elimination_trace_simple(graph, param_length=1)

    assert trace.param_length == 1
    assert trace.reward_length == 0
    assert trace.n_vertices == 2

    print(f"✓ Simple version: {trace.n_vertices} vertices, {len(trace.operations)} operations")
    print(f"  - param_length: {trace.param_length}")
    print(f"  - reward_length: {trace.reward_length}")

    # Instantiate graph
    theta = np.array([2.0])
    graph_inst = instantiate_from_trace(trace, params=theta)

    # Compute PDF
    pdf = graph_inst.pdf(1.0, granularity=200)

    # Theoretical: λ exp(-λt) = 2.0 * exp(-2.0)
    pdf_expected = 2.0 * np.exp(-2.0 * 1.0)

    print(f"  - PDF(1.0) = {pdf}")
    print(f"  - Expected = {pdf_expected}")
    print(f"  - Error = {abs(pdf - pdf_expected) / pdf_expected * 100:.2f}%")

    assert abs(pdf - pdf_expected) / pdf_expected < 0.05, "PDF should match theory"
    print(f"✓ Simple version produces correct results")


def test_full_version_no_rewards():
    """Test that full version works correctly with rewards disabled"""
    from phasic import Graph
    from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace

    # Create simple 2-state exponential model manually
    graph = Graph(state_length=1)
    v_start = graph.starting_vertex()
    v_abs = graph.find_or_create_vertex([1])
    v_start.add_edge_parameterized(v_abs, 0.0, [1.0])  # weight, edge_state

    # Record with full version but rewards disabled
    trace = record_elimination_trace(graph, param_length=1, enable_rewards=False)

    assert trace.param_length == 1
    assert trace.reward_length == 0
    assert trace.n_vertices == 2

    print(f"✓ Full version (no rewards): {trace.n_vertices} vertices, {len(trace.operations)} operations")

    # Instantiate graph
    theta = np.array([2.0])
    graph_inst = instantiate_from_trace(trace, params=theta)

    # Compute PDF
    pdf = graph_inst.pdf(1.0, granularity=200)
    pdf_expected = 2.0 * np.exp(-2.0 * 1.0)

    print(f"  - PDF(1.0) = {pdf}")
    print(f"  - Expected = {pdf_expected}")
    print(f"  - Error = {abs(pdf - pdf_expected) / pdf_expected * 100:.2f}%")

    assert abs(pdf - pdf_expected) / pdf_expected < 0.05
    print(f"✓ Full version (no rewards) produces correct results")


def test_full_version_with_rewards():
    """Test that full version works correctly with rewards enabled"""
    from phasic import Graph
    from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace

    # Create simple 2-state exponential model manually
    graph = Graph(state_length=1)
    v_start = graph.starting_vertex()
    v_abs = graph.find_or_create_vertex([1])
    v_start.add_edge_parameterized(v_abs, 0.0, [1.0])  # weight, edge_state

    # Record with rewards enabled
    trace = record_elimination_trace(graph, param_length=1, enable_rewards=True)

    assert trace.param_length == 1
    assert trace.reward_length == 2  # n_vertices
    assert trace.n_vertices == 2

    print(f"✓ Full version (with rewards): {trace.n_vertices} vertices, {len(trace.operations)} operations")
    print(f"  - param_length: {trace.param_length}")
    print(f"  - reward_length: {trace.reward_length}")

    # Instantiate with neutral rewards
    theta = np.array([2.0])
    rewards = np.ones(trace.n_vertices)

    graph_neutral = instantiate_from_trace(trace, params=theta, rewards=rewards)
    pdf_neutral = graph_neutral.pdf(1.0, granularity=200)

    # Theoretical: λ exp(-λt) = 2.0 * exp(-2.0)
    pdf_expected = 2.0 * np.exp(-2.0 * 1.0)

    print(f"  - PDF(1.0) neutral = {pdf_neutral}")
    print(f"  - Expected = {pdf_expected}")

    assert abs(pdf_neutral - pdf_expected) / pdf_expected < 0.05

    # Instantiate with reward scaling
    rewards_scaled = np.ones(trace.n_vertices)
    rewards_scaled[0] = 3.0  # Scale starting vertex by 3

    graph_scaled = instantiate_from_trace(trace, params=theta, rewards=rewards_scaled)
    pdf_scaled = graph_scaled.pdf(1.0, granularity=200)

    # With reward on starting vertex: λ' = λ * r = 2.0 * 3.0 = 6.0
    # PDF = 6.0 * exp(-6.0)
    pdf_expected_scaled = 6.0 * np.exp(-6.0 * 1.0)

    print(f"  - PDF(1.0) scaled = {pdf_scaled}")
    print(f"  - Expected = {pdf_expected_scaled}")
    print(f"  - Error = {abs(pdf_scaled - pdf_expected_scaled) / pdf_expected_scaled * 100:.2f}%")

    assert abs(pdf_scaled - pdf_expected_scaled) / pdf_expected_scaled < 0.05

    print(f"✓ Full version with rewards produces correct results")


def test_operation_count_comparison():
    """Compare operation counts between versions"""
    from phasic import Graph
    from phasic.trace_elimination import record_elimination_trace_simple, record_elimination_trace

    def callback(state):
        if len(state) == 0:
            return [(np.array([1]), 0.0, [1.0])]
        elif state[0] == 1:
            return []
        return []

    graph = Graph(callback=callback, parameterized=True)

    # Record both versions
    trace_simple = record_elimination_trace_simple(graph, param_length=1)
    trace_no_rewards = record_elimination_trace(graph, param_length=1, enable_rewards=False)
    trace_with_rewards = record_elimination_trace(graph, param_length=1, enable_rewards=True)

    print(f"✓ Operation count comparison:")
    print(f"  - Simple version:      {len(trace_simple.operations)} operations")
    print(f"  - Full (no rewards):   {len(trace_no_rewards.operations)} operations")
    print(f"  - Full (with rewards): {len(trace_with_rewards.operations)} operations")

    # Simple and full-no-rewards should have same operation count
    assert len(trace_simple.operations) == len(trace_no_rewards.operations)

    # With rewards should have more operations (MUL + PARAM for each edge)
    assert len(trace_with_rewards.operations) > len(trace_simple.operations)

    print(f"✓ Operation counts are as expected")


def test_backward_compatibility():
    """Test that default behavior is backward compatible"""
    from phasic import Graph
    from phasic.trace_elimination import record_elimination_trace

    def callback(state):
        if len(state) == 0:
            return [(np.array([1]), 0.0, [1.0])]
        elif state[0] == 1:
            return []
        return []

    graph = Graph(callback=callback, parameterized=True)

    # Default call (should have no rewards)
    trace = record_elimination_trace(graph, param_length=1)

    assert trace.reward_length == 0, "Default should have no rewards"

    print(f"✓ Backward compatibility: default has reward_length=0")


if __name__ == "__main__":
    print("Testing both trace versions...\n")

    print("1. Testing simple version (no rewards)...")
    test_simple_version_no_rewards()
    print()

    print("2. Testing full version with rewards disabled...")
    test_full_version_no_rewards()
    print()

    print("3. Testing full version with rewards enabled...")
    test_full_version_with_rewards()
    print()

    print("4. Testing operation count comparison...")
    test_operation_count_comparison()
    print()

    print("5. Testing backward compatibility...")
    test_backward_compatibility()
    print()

    print("✓ All tests passed!")
    print("\nConclusion:")
    print("- Simple version works correctly (lightweight, no reward overhead)")
    print("- Full version works correctly both with and without rewards")
    print("- Reward transformation produces theoretically correct results")
    print("- Both versions are available and backward compatible")
