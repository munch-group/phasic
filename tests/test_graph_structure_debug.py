"""
Debug: Check graph structure after instantiation from trace
"""

import numpy as np
from phasic import Graph
from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace, evaluate_trace


def inspect_graph(graph, label="Graph"):
    """Print detailed graph structure"""
    print(f"\n{label}:")
    print(f"  Total vertices: {graph.vertices_length()}")

    for i in range(graph.vertices_length()):
        vertices = [graph.starting_vertex()] + [
            graph.find_or_create_vertex([j]) for j in range(1, graph.vertices_length())
        ]
        if i < len(vertices):
            v = vertices[i]
            print(f"\n  Vertex {i}: state={v.state()}")
            print(f"    Regular edges: {len(v.edges())}")
            for j, edge in enumerate(v.edges()):
                print(f"      Edge {j}: to={edge.to().state()}, weight={edge.weight()}")
            print(f"    Parameterized edges: {len(v.parameterized_edges())}")
            for j, edge in enumerate(v.parameterized_edges()):
                print(f"      P-Edge {j}: to={edge.to().state()}, weight={edge.weight()}")


def test_instantiation():
    """Test graph instantiation from trace"""

    print("="*60)
    print("Testing graph instantiation")
    print("="*60)

    # Create original graph
    graph = Graph(state_length=1)
    v0 = graph.starting_vertex()
    v1 = graph.find_or_create_vertex([1])
    v0.add_edge_parameterized(v1, 0.0, [1.0])

    inspect_graph(graph, "Original graph (before trace)")

    # Record trace
    trace = record_elimination_trace(graph, param_length=1, enable_rewards=False)
    print(f"\n\nTrace recorded: {len(trace.operations)} operations")

    # Evaluate trace
    result = evaluate_trace(trace, params=np.array([10.0]))
    print(f"\nTrace evaluation:")
    print(f"  vertex_rates: {result['vertex_rates']}")
    print(f"  edge_probs: {result['edge_probs']}")
    print(f"  vertex_targets: {result['vertex_targets']}")

    # Instantiate graph
    graph_inst = instantiate_from_trace(trace, params=np.array([10.0]))

    inspect_graph(graph_inst, "Instantiated graph (after trace)")

    # Check if we can get matrices
    try:
        alpha, S = graph_inst.as_matrices()
        print(f"\nGraph matrices:")
        print(f"  alpha (initial distribution): {alpha}")
        print(f"  S (sub-intensity matrix):")
        print(f"{S}")
    except Exception as e:
        print(f"\nCould not get matrices: {e}")

    # Try to sample
    print(f"\n\nSampling test:")
    samples = graph_inst.sample(5)
    print(f"  Samples: {samples}")

    # Try normalize and sample again
    print(f"\nAfter normalize():")
    graph_inst.normalize()

    inspect_graph(graph_inst, "After normalize()")

    samples_norm = graph_inst.sample(5)
    print(f"  Samples: {samples_norm}")

    # Try PDF
    pdf = graph_inst.pdf(0.1, granularity=200)
    print(f"  PDF(0.1) = {pdf}")


if __name__ == "__main__":
    test_instantiation()
