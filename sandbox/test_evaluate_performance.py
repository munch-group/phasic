#!/usr/bin/env python3
"""
Test to measure evaluation performance of symbolic expressions
"""
import time
from ptdalgorithms import Graph

def test_model(n_rabbits):
    """Test with n rabbits"""
    expected_vertices = ((n_rabbits + 1) * (n_rabbits + 2)) // 2
    print(f"Building {n_rabbits}-rabbit model ({expected_vertices} vertices)...")
    g = Graph(state_length=2)
    initial_state = [n_rabbits, 0]
    g.starting_vertex().add_edge(g.find_or_create_vertex(initial_state), 1)

    index = 1
    while index < g.vertices_length():
        vertex = g.vertex_at(index)
        state = vertex.state()

        if state[0] > 0:
            child_state = [state[0] - 1, state[1] + 1]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [state[0], 0, 0]
            )
            child_state = [0, state[1]]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [0, 1, 0]
            )

        if state[1] > 0:
            child_state = [state[0] + 1, state[1] - 1]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [state[1], 0, 0]
            )
            child_state = [state[0], 0]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [0, 0, 1]
            )

        index += 1

    print(f"Model built: {g.vertices_length()} vertices")

    # Set initial parameters
    print("Setting initial parameters...")
    g.update_parameterized_weights([1.0, 2.0, 4.0])

    # Perform symbolic elimination
    print("Starting symbolic elimination...")
    start = time.time()
    dag = g.eliminate_to_dag()
    elim_time = time.time() - start
    print(f"Elimination complete in {elim_time:.2f}s")
    print(f"DAG vertices: {dag.vertices_length}")

    # Test a SINGLE instantiation with timing
    print("\nTesting single instantiation...")
    start = time.time()
    instantiated = dag.instantiate([0.5, 0.1, 0.1])
    inst_time = time.time() - start
    print(f"Single instantiation: {inst_time:.4f}s")
    print(f"Result has {instantiated.vertices_length()} vertices")

    # If that worked, try 10 instantiations
    if inst_time < 1.0:
        print("\nTesting 10 instantiations...")
        start = time.time()
        for i in range(10):
            instantiated = dag.instantiate([0.5, 0.1, 0.1])
            if i % 5 == 0:
                print(f"  Completed {i}/10...")
        total_time = time.time() - start
        print(f"10 instantiations: {total_time:.4f}s ({total_time/10:.4f}s each)")
    else:
        print(f"Single instantiation too slow ({inst_time:.4f}s), skipping batch test")

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    test_model(n)
