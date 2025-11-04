"""Test exact usage pattern from notebook"""

import phasic
import numpy as np
import os

def construct_rabbit_graph(nr_rabbits, flood_left, flood_right):
    state_vector_length = 2
    graph = phasic.Graph(state_vector_length)
    initial_state = [nr_rabbits, 0]
    vertex = graph.find_or_create_vertex(initial_state)
    graph.starting_vertex().add_edge(vertex, 1)

    index = 1
    while index < graph.vertices_length():
        vertex = graph.vertex_at(index)
        state = vertex.state()

        if state[0] > 0:
            child_state = [state[0] - 1, state[1] + 1]
            vertex.add_edge(graph.find_or_create_vertex(child_state), 1.0)
            child_state = [0, state[1]]
            vertex.add_edge(graph.find_or_create_vertex(child_state), flood_left)
        if state[1] > 0:
            child_state = [state[0] + 1, state[1] - 1]
            vertex.add_edge(graph.find_or_create_vertex(child_state), 1)
            child_state = [state[0], 0]
            vertex.add_edge(graph.find_or_create_vertex(child_state), flood_right)

        index += 1
    return graph

print("Building rabbit graph...")
graph = construct_rabbit_graph(2, 2, 4)
print(f"  Graph: {graph.vertices_length()} vertices")

# Do some operations on the graph like in the notebook
print("\nGetting states...")
states = graph.states()
print(f"  States shape: {states.shape}")
print(f"  States column 1: {states[:, 1]}")

print("\nComputing covariance...")
cov = graph.covariance(states[:,0], states[:,1])
print(f"  Covariance: {cov}")

# Now clone - this is where the notebook crashes according to user
print("\n" + "="*60)
print("CLONING GRAPH (this is where notebook crashes)")
print("="*60)

try:
    mdph_carrot_graph = graph.clone()
    print(f"✅ Clone succeeded: {mdph_carrot_graph.vertices_length()} vertices")

    # Continue with notebook operations
    print("\nAccessing clone...")
    vlength = mdph_carrot_graph.vertices_length()
    carrot_vertices_left = np.repeat(False, vlength*2)
    carrot_vertices_right = np.repeat(False, vlength*2)

    print(f"  Created arrays: left={len(carrot_vertices_left)}, right={len(carrot_vertices_right)}")

    # Access vertices like in notebook
    for i in range(vlength):
        vertex = mdph_carrot_graph.vertex_at(i)
        rabbits = vertex.state()
        print(f"  Vertex {i}: {rabbits}")

    print("\n" + "="*60)
    print("✅ ALL OPERATIONS SUCCESSFUL!")
    print("="*60)
    os._exit(0)

except Exception as e:
    print(f"\n❌ CRASH: {e}")
    import traceback
    traceback.print_exc()
    os._exit(1)
