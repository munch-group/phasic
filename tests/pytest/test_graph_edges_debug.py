"""
Debug: Check what edges are in the graph before trace recording
"""

import numpy as np
from phasic import Graph


def test_manual_graph_edges():
    """Check edges in manually constructed graph"""

    print("="*60)
    print("Manual graph construction")
    print("="*60)

    graph = Graph(1)
    v0 = graph.starting_vertex()
    v1 = graph.find_or_create_vertex([1])

    print(f"Before adding edge:")
    print(f"  v0 edges: {len(v0.edges())}")
    print(f"  v0 parameterized_edges: {len(v0.parameterized_edges())}")

    v0.add_edge_parameterized(v1, 0.0, [1.0])

    print(f"\nAfter adding parameterized edge:")
    print(f"  v0 edges: {len(v0.edges())}")
    print(f"  v0 parameterized_edges: {len(v0.parameterized_edges())}")

    # Check regular edges
    for i, edge in enumerate(v0.edges()):
        print(f"  Regular edge {i}:")
        print(f"    to: {edge.to().state()}")
        print(f"    weight: {edge.weight()}")

    # Check parameterized edges
    for i, edge in enumerate(v0.parameterized_edges()):
        print(f"  Parameterized edge {i}:")
        print(f"    to: {edge.to().state()}")
        print(f"    weight: {edge.weight()}")
        print(f"    edge_state: {edge.edge_state(1)}")


def test_callback_graph_edges():
    """Check edges in callback-constructed graph"""

    print("\n" + "="*60)
    print("Callback graph construction")
    print("="*60)

    def callback(state):
        if len(state) == 0:
            print(f"  Callback called with state={state}")
            edge_spec = (np.array([1]), 0.0, [1.0])
            print(f"  Returning: {edge_spec}")
            return [edge_spec]
        elif state[0] == 1:
            print(f"  Callback called with state={state} (absorbing)")
            return []
        return []

    graph = Graph(callback, parameterized=True)

    print(f"\nGraph constructed:")
    print(f"  Vertices: {graph.vertices_length()}")

    # Get vertices
    v0 = graph.starting_vertex()

    print(f"\nStarting vertex:")
    print(f"  State: {v0.state()}")
    print(f"  Regular edges: {len(v0.edges())}")
    print(f"  Parameterized edges: {len(v0.parameterized_edges())}")

    # Check regular edges
    for i, edge in enumerate(v0.edges()):
        print(f"  Regular edge {i}:")
        print(f"    to: {edge.to().state()}")
        print(f"    weight: {edge.weight()}")

    # Check parameterized edges
    for i, edge in enumerate(v0.parameterized_edges()):
        print(f"  Parameterized edge {i}:")
        print(f"    to: {edge.to().state()}")
        print(f"    weight: {edge.weight()}")
        print(f"    edge_state: {edge.edge_state(1)}")


if __name__ == "__main__":
    print("Debugging graph edges before trace recording...\n")

    test_manual_graph_edges()
    test_callback_graph_edges()
