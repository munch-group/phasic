"""
Compare manually constructed graph vs trace-instantiated graph
"""

import numpy as np
from phasic import Graph
from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace


def test_manual_graph_works():
    """Verify that a manually constructed graph works"""

    print("="*60)
    print("Test 1: Manually constructed exponential graph")
    print("="*60)

    # Create graph with concrete weight
    graph = Graph(1)
    v0 = graph.starting_vertex()
    v1 = graph.find_or_create_vertex([1])
    v0.add_edge(v1, 10.0)  # Concrete weight, not parameterized

    print(f"Manual graph: {graph.vertices_length()} vertices")
    print(f"Starting vertex: {v0.state()}")

    # Normalize
    graph.normalize()

    # Sample
    samples = np.array(graph.sample(10))
    print(f"\nSamples: {samples[:5]}")
    print(f"  Mean: {samples.mean():.6f}")
    print(f"  Expected: {1/10:.6f}")

    # PDF
    pdf = graph.pdf(0.1, granularity=200)
    print(f"\nPDF(0.1) = {pdf:.6f}")
    print(f"Expected = {10.0 * np.exp(-10.0 * 0.1):.6f}")

    return samples.mean()


def test_trace_graph():
    """Test trace-instantiated graph"""

    print("\n" + "="*60)
    print("Test 2: Trace-instantiated graph")
    print("="*60)

    # Create parameterized graph
    graph = Graph(1)
    v0 = graph.starting_vertex()
    v1 = graph.find_or_create_vertex([1])
    v0.add_edge_parameterized(v1, 0.0, [1.0])

    print(f"Original graph: {graph.vertices_length()} vertices")

    # Record trace
    trace = record_elimination_trace(graph, theta_dim=1, enable_rewards=False)
    print(f"Trace recorded: {len(trace.operations)} operations")

    # Instantiate with theta=10.0
    graph_inst = instantiate_from_trace(trace, params=np.array([10.0]), use_log=False)
    print(f"Instantiated graph: {graph_inst.vertices_length()} vertices")

    # Check edge structure
    v0_inst = graph_inst.starting_vertex()
    print(f"Starting vertex: {v0_inst.state()}")
    print(f"Edges: {len(v0_inst.edges())}")
    for i, edge in enumerate(v0_inst.edges()):
        print(f"  Edge {i}: to={edge.to().state()}, weight={edge.weight()}")

    # Normalize
    graph_inst.normalize()
    print(f"\nAfter normalize:")
    for i, edge in enumerate(v0_inst.edges()):
        print(f"  Edge {i}: to={edge.to().state()}, weight={edge.weight()}")

    # Sample
    samples = np.array(graph_inst.sample(10))
    print(f"\nSamples: {samples[:5]}")
    print(f"  Mean: {samples.mean():.6f}")
    print(f"  Expected: {1/10:.6f}")

    # PDF
    pdf = graph_inst.pdf(0.1, granularity=200)
    print(f"\nPDF(0.1) = {pdf:.6f}")
    print(f"Expected = {10.0 * np.exp(-10.0 * 0.1):.6f}")

    return samples.mean()


def test_simple_concrete_trace():
    """Test trace with concrete (non-parameterized) edges"""

    print("\n" + "="*60)
    print("Test 3: Trace from concrete edges")
    print("="*60)

    # Create graph with concrete edge
    graph = Graph(1)
    v0 = graph.starting_vertex()
    v1 = graph.find_or_create_vertex([1])
    v0.add_edge(v1, 10.0)  # Concrete weight

    print(f"Original graph: {graph.vertices_length()} vertices")

    # Record trace (no parameters needed)
    trace = record_elimination_trace(graph, theta_dim=0, enable_rewards=False)
    print(f"Trace recorded: {len(trace.operations)} operations")
    print(f"Param length: {trace.param_length}")

    # Instantiate (no params needed)
    graph_inst = instantiate_from_trace(trace, params=None, use_log=False)
    print(f"Instantiated graph: {graph_inst.vertices_length()} vertices")

    # Normalize
    graph_inst.normalize()

    # Sample
    samples = np.array(graph_inst.sample(10))
    print(f"\nSamples: {samples[:5]}")
    print(f"  Mean: {samples.mean():.6f}")
    print(f"  Expected: {1/10:.6f}")

    # PDF
    pdf = graph_inst.pdf(0.1, granularity=200)
    print(f"\nPDF(0.1) = {pdf:.6f}")
    print(f"Expected = {10.0 * np.exp(-10.0 * 0.1):.6f}")

    return samples.mean()


if __name__ == "__main__":
    print("Comparing manual vs trace-instantiated graphs...\n")

    mean_manual = test_manual_graph_works()
    mean_trace = test_trace_graph()
    mean_concrete = test_simple_concrete_trace()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Manual graph sample mean:    {mean_manual:.6f} (expected: 0.100000)")
    print(f"Trace graph sample mean:     {mean_trace:.6f} (expected: 0.100000)")
    print(f"Concrete trace sample mean:  {mean_concrete:.6f} (expected: 0.100000)")
