"""
Simple comparison test: verify simple and full versions produce identical results
"""

import numpy as np

def test_simple_vs_full_equivalence():
    """Test that simple and full (no rewards) versions produce identical traces"""
    from phasic import Graph
    from phasic.trace_elimination import record_elimination_trace_simple, record_elimination_trace

    # Create simple parameterized graph
    graph = Graph(1)
    v0 = graph.starting_vertex()
    v1 = graph.find_or_create_vertex([1])
    v0.add_edge_parameterized(v1, 0.0, [1.0])

    # Record with both versions
    trace_simple = record_elimination_trace_simple(graph, theta_dim=1)
    trace_full = record_elimination_trace(graph, theta_dim=1, enable_rewards=False)

    # Compare traces
    print(f"Simple version:")
    print(f"  - n_vertices: {trace_simple.n_vertices}")
    print(f"  - param_length: {trace_simple.param_length}")
    print(f"  - reward_length: {trace_simple.reward_length}")
    print(f"  - n_operations: {len(trace_simple.operations)}")

    print(f"\nFull version (no rewards):")
    print(f"  - n_vertices: {trace_full.n_vertices}")
    print(f"  - param_length: {trace_full.param_length}")
    print(f"  - reward_length: {trace_full.reward_length}")
    print(f"  - n_operations: {len(trace_full.operations)}")

    # Should be identical
    assert trace_simple.n_vertices == trace_full.n_vertices
    assert trace_simple.param_length == trace_full.param_length
    assert trace_simple.reward_length == trace_full.reward_length
    assert len(trace_simple.operations) == len(trace_full.operations)

    print(f"\n✓ Simple and full (no rewards) produce identical traces")


def test_full_with_rewards_has_more_operations():
    """Test that full version with rewards has more operations"""
    from phasic import Graph
    from phasic.trace_elimination import record_elimination_trace

    graph = Graph(1)
    v0 = graph.starting_vertex()
    v1 = graph.find_or_create_vertex([1])
    v0.add_edge_parameterized(v1, 0.0, [1.0])

    trace_no_rewards = record_elimination_trace(graph, theta_dim=1, enable_rewards=False)
    trace_with_rewards = record_elimination_trace(graph, theta_dim=1, enable_rewards=True)

    print(f"Without rewards: {len(trace_no_rewards.operations)} operations")
    print(f"With rewards: {len(trace_with_rewards.operations)} operations")

    assert trace_with_rewards.reward_length == 2  # n_vertices
    assert len(trace_with_rewards.operations) > len(trace_no_rewards.operations)

    print(f"✓ With rewards version has more operations (reward MUL added)")


def test_backward_compatibility():
    """Test default behavior is no rewards"""
    from phasic import Graph
    from phasic.trace_elimination import record_elimination_trace

    graph = Graph(1)
    v0 = graph.starting_vertex()
    v1 = graph.find_or_create_vertex([1])
    v0.add_edge_parameterized(v1, 0.0, [1.0])

    # Default should have no rewards
    trace = record_elimination_trace(graph, theta_dim=1)

    assert trace.reward_length == 0
    print(f"✓ Default behavior: reward_length = 0 (backward compatible)")


if __name__ == "__main__":
    print("Testing trace version equivalence...\n")

    print("1. Simple vs Full (no rewards) equivalence...")
    test_simple_vs_full_equivalence()
    print()

    print("2. Full version with rewards has more operations...")
    test_full_with_rewards_has_more_operations()
    print()

    print("3. Backward compatibility...")
    test_backward_compatibility()
    print()

    print("✓ All comparison tests passed!")
    print("\nConclusion:")
    print("- Simple and full (no rewards) versions produce identical traces")
    print("- Full version with rewards adds MUL operations as expected")
    print("- Default behavior is backward compatible (no rewards)")
