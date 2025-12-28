"""
Debug why graph instantiation from trace produces zero samples
"""

import numpy as np
from phasic import Graph
from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace, evaluate_trace


def test_manual_graph():
    """Test that manually constructed graph works"""

    print("="*60)
    print("Test 1: Manually constructed graph")
    print("="*60)

    # Create graph manually
    graph = Graph(state_length=1)
    v0 = graph.starting_vertex()
    v1 = graph.find_or_create_vertex([1])
    v0.add_edge_parameterized(v1, 0.0, [1.0])  # weight = 0.0 + 1.0 * theta[0]

    print(f"Manual graph: {graph.vertices_length()} vertices")

    # Record trace
    trace = record_elimination_trace(graph, param_length=1, enable_rewards=False)
    print(f"Trace: {len(trace.operations)} operations")

    # Evaluate trace
    result = evaluate_trace(trace, params=np.array([10.0]))
    print(f"Vertex rates: {result['vertex_rates']}")
    print(f"Edge probs: {result['edge_probs']}")

    # Instantiate
    graph_inst = instantiate_from_trace(trace, params=np.array([10.0]))
    print(f"Instantiated graph: {graph_inst.vertices_length()} vertices")

    # Check if we need to normalize
    print(f"\nBefore normalize:")
    print(f"  Starting vertex: {graph_inst.starting_vertex()}")

    # Sample
    samples = np.array(graph_inst.sample(10))
    print(f"\nSamples (before normalize): {samples[:5]}")
    print(f"  Mean: {samples.mean():.6f}")
    print(f"  Expected: {1/10:.6f}")

    # Try normalizing
    graph_inst.normalize()
    samples_norm = np.array(graph_inst.sample(10))
    print(f"\nSamples (after normalize): {samples_norm[:5]}")
    print(f"  Mean: {samples_norm.mean():.6f}")

    # Try PDF
    pdf = graph_inst.pdf(0.1, granularity=200)
    print(f"\nPDF(0.1) = {pdf}")
    print(f"Expected PDF(0.1) = {10.0 * np.exp(-10.0 * 0.1):.6f}")


def test_callback_graph():
    """Test that callback-constructed graph works"""

    print("\n" + "="*60)
    print("Test 2: Callback-constructed graph")
    print("="*60)

    def callback(state):
        if len(state) == 0:
            # Starting vertex -> absorbing
            return [(np.array([1]), 0.0, [1.0])]
        elif state[0] == 1:
            # Absorbing vertex
            return []
        return []

    graph = Graph(callback=callback, parameterized=True)
    print(f"Callback graph: {graph.vertices_length()} vertices")

    # Record trace
    trace = record_elimination_trace(graph, param_length=1, enable_rewards=False)
    print(f"Trace: {len(trace.operations)} operations")

    # Evaluate trace
    result = evaluate_trace(trace, params=np.array([10.0]))
    print(f"Vertex rates: {result['vertex_rates']}")
    print(f"Edge probs: {result['edge_probs']}")
    print(f"Vertex targets: {result['vertex_targets']}")

    # Instantiate
    graph_inst = instantiate_from_trace(trace, params=np.array([10.0]))
    print(f"Instantiated graph: {graph_inst.vertices_length()} vertices")

    # Sample without normalize
    samples = np.array(graph_inst.sample(10))
    print(f"\nSamples (before normalize): {samples[:5]}")
    print(f"  Mean: {samples.mean():.6f}")

    # Try normalizing
    graph_inst.normalize()
    samples_norm = np.array(graph_inst.sample(10))
    print(f"\nSamples (after normalize): {samples_norm[:5]}")
    print(f"  Mean: {samples_norm.mean():.6f}")
    print(f"  Expected: {1/10:.6f}")

    # Try PDF
    pdf = graph_inst.pdf(0.1, granularity=200)
    print(f"\nPDF(0.1) = {pdf}")
    print(f"Expected PDF(0.1) = {10.0 * np.exp(-10.0 * 0.1):.6f}")


if __name__ == "__main__":
    print("Debugging graph instantiation from trace...\n")

    test_manual_graph()
    test_callback_graph()

    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    print("If samples are 0.0, the graph needs normalize() after instantiation")
