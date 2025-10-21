#!/usr/bin/env python3
"""
Quick test to verify O(n) performance of iterative expression tree operations
"""

import time
from phasic import Graph

def construct_parameterized_rabbit_model(nr_rabbits):
    """Rabbit model: 2D state [left_island, right_island]"""
    graph = Graph(state_length=2)
    initial_state = [nr_rabbits, 0]
    graph.starting_vertex().add_edge(graph.find_or_create_vertex(initial_state), 1)

    index = 1
    while index < graph.vertices_length():
        vertex = graph.vertex_at(index)
        state = vertex.state()

        if state[0] > 0:
            # Jump left to right
            child_state = [state[0] - 1, state[1] + 1]
            vertex.add_edge_parameterized(
                graph.find_or_create_vertex(child_state), 0, [state[0], 0, 0]
            )
            # Left island flooding
            child_state = [0, state[1]]
            vertex.add_edge_parameterized(
                graph.find_or_create_vertex(child_state), 0, [0, 1, 0]
            )

        if state[1] > 0:
            # Jump right to left
            child_state = [state[0] + 1, state[1] - 1]
            vertex.add_edge_parameterized(
                graph.find_or_create_vertex(child_state), 0, [state[1], 0, 0]
            )
            # Right island flooding
            child_state = [state[0], 0]
            vertex.add_edge_parameterized(
                graph.find_or_create_vertex(child_state), 0, [0, 0, 1]
            )

        index += 1

    return graph

print("Testing O(n) performance with different model sizes")
print("=" * 70)

test_sizes = [3, 5, 7]

for n_rabbits in test_sizes:
    print(f"\nTesting {n_rabbits} rabbits...")

    # Construct model
    start = time.time()
    g = construct_parameterized_rabbit_model(n_rabbits)
    construct_time = time.time() - start
    n_vertices = g.vertices_length()

    print(f"  Constructed: {n_vertices} vertices in {construct_time:.3f}s")

    # Set initial parameters
    g.update_parameterized_weights([1.0, 2.0, 4.0])

    # Perform symbolic elimination
    start = time.time()
    dag = g.eliminate_to_dag()
    elim_time = time.time() - start

    print(f"  Elimination: {elim_time:.3f}s")

    # Test instantiation (exercises ptd_expr_evaluate_iterative)
    start = time.time()
    for _ in range(10):
        instantiated = dag.instantiate([0.5, 0.1, 0.1])
    inst_time = (time.time() - start) / 10

    print(f"  Instantiation: {inst_time:.4f}s (avg of 10)")
    print(f"  Total per instance: {elim_time + inst_time:.4f}s")

print("\n" + "=" * 70)
print("Test complete!")
print("\nIf O(n) is achieved, times should scale linearly with vertices.")
