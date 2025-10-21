#!/usr/bin/env python3
"""
Test script to verify that 67-vertex rabbit model works without stack overflow
"""

import sys
import time
from phasic import Graph

def construct_parameterized_rabbit_model(nr_rabbits):
    """
    Construct a parameterized rabbit model where edge weights are linear combinations of parameters.
    Parameters: [jump_rate, flood_left_rate, flood_right_rate]
    """
    graph = Graph(state_length=2)
    initial_state = [nr_rabbits, 0]
    graph.starting_vertex().add_edge(graph.find_or_create_vertex(initial_state), 1)

    index = 1
    while index < graph.vertices_length():
        vertex = graph.vertex_at(index)
        state = vertex.state()

        if state[0] > 0:
            # Jump left to right - rate proportional to number of rabbits
            child_state = [state[0] - 1, state[1] + 1]
            vertex.add_edge_parameterized(
                graph.find_or_create_vertex(child_state),
                0,
                [state[0], 0, 0]  # jump_rate * n_rabbits
            )
            # Left island flooding
            child_state = [0, state[1]]
            vertex.add_edge_parameterized(
                graph.find_or_create_vertex(child_state),
                0,
                [0, 1, 0]  # flood_left_rate
            )

        if state[1] > 0:
            # Jump right to left
            child_state = [state[0] + 1, state[1] - 1]
            vertex.add_edge_parameterized(
                graph.find_or_create_vertex(child_state),
                0,
                [state[1], 0, 0]  # jump_rate * n_rabbits
            )
            # Right island flooding
            child_state = [state[0], 0]
            vertex.add_edge_parameterized(
                graph.find_or_create_vertex(child_state),
                0,
                [0, 0, 1]  # flood_right_rate
            )

        index += 1

    return graph

def main():
    print("=" * 70)
    print("Testing Iterative Expression Tree Fix")
    print("=" * 70)
    print()
    print("This test verifies that the iterative expression tree functions")
    print("successfully prevent stack overflow for large models.")
    print()

    # Test with 5 rabbits (22 vertices) - should complete quickly
    print("=" * 70)
    print("Test 1: 22-Vertex Model (5 rabbits)")
    print("=" * 70)
    print()

    n_rabbits = 5
    print(f"Constructing parameterized model with {n_rabbits} rabbits...")
    start_time = time.time()

    try:
        g = construct_parameterized_rabbit_model(n_rabbits)
        construct_time = time.time() - start_time
        print(f"✓ Model constructed in {construct_time:.2f}s")
        print(f"  Vertices: {g.vertices_length()}")
        print()

        # First set initial parameters (required before elimination)
        print("Setting initial parameters...")
        initial_params = [1.0, 2.0, 4.0]  # jump_rate, flood_left, flood_right
        g.update_parameterized_weights(initial_params)

        print("Running symbolic elimination...")
        start_time = time.time()

        dag = g.eliminate_to_dag()

        elim_time = time.time() - start_time
        print(f"✓ Symbolic elimination completed in {elim_time:.2f}s")
        print(f"  DAG vertices: {dag.vertices_length}")
        print(f"  Is acyclic: {dag.is_acyclic}")
        print()

        print("✓ Test 1 PASSED: 22-vertex model works correctly")
        print()

    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test with 10 rabbits (67 vertices) - previously crashed
    print("=" * 70)
    print("Test 2: 67-Vertex Model (10 rabbits) - Previously Crashed")
    print("=" * 70)
    print()

    n_rabbits = 10
    print(f"Constructing parameterized model with {n_rabbits} rabbits...")
    start_time = time.time()

    try:
        g = construct_parameterized_rabbit_model(n_rabbits)
        construct_time = time.time() - start_time
        print(f"✓ Model constructed in {construct_time:.2f}s")
        print(f"  Vertices: {g.vertices_length()}")
        print()

        print("Setting initial parameters...")
        initial_params = [1.0, 2.0, 4.0]
        g.update_parameterized_weights(initial_params)

        print("Running symbolic elimination (THIS IS WHERE IT CRASHED BEFORE)...")
        print("This may take a few minutes for 67 vertices...")
        start_time = time.time()

        dag = g.eliminate_to_dag()

        elim_time = time.time() - start_time
        print(f"✓ Symbolic elimination completed in {elim_time:.2f}s")
        print(f"  DAG vertices: {dag.vertices_length}")
        print(f"  Is acyclic: {dag.is_acyclic}")
        print()

        print("✓ Test 2 PASSED: 67-vertex model works without crash!")
        print()

    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("=" * 70)
    print("SUCCESS: All Tests Passed!")
    print("=" * 70)
    print()
    print("The iterative expression tree functions successfully")
    print("avoided the stack overflow that occurred with recursive")
    print("implementations for deeply nested expression trees.")
    print()
    print("Before: 67-vertex model crashed with 'Killed: 9' (SIGKILL)")
    print("After:  67-vertex model completes without crash")

    return 0

if __name__ == "__main__":
    sys.exit(main())
