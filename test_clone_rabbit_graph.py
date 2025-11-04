"""Test clone() with rabbit graph from tutorial"""

import phasic
import os

def construct_rabbit_graph(nr_rabbits, flood_left, flood_right):
    # we represent the vector as two integers, the number of
    # rabbits on the left and right island
    state_vector_length = 2
    graph = phasic.Graph(state_vector_length)

    # the initial state is the only starting state, with probability 1
    initial_state = [nr_rabbits, 0]
    vertex = graph.find_or_create_vertex(initial_state)
    graph.starting_vertex().add_edge(vertex, 1)

    index = 1
    # iterate over all unvisited vertices
    while index < graph.vertices_length():
        vertex = graph.vertex_at(index)
        state = vertex.state()

        if state[0] > 0:
            # rabbit jump left to right
            child_state = [state[0] - 1, state[1] + 1]
            vertex.add_edge(
                graph.find_or_create_vertex(child_state),
                1.0
            )
            # left island flooding
            child_state = [0, state[1]]
            vertex.add_edge(
                graph.find_or_create_vertex(child_state),
                flood_left
            )
        if state[1] > 0:
            child_state = [state[0] + 1, state[1] - 1]
            vertex.add_edge(
                graph.find_or_create_vertex(child_state),
                1
            )
            # right island flooding
            child_state = [state[0], 0]
            vertex.add_edge(
                graph.find_or_create_vertex(child_state),
                flood_right
            )

        index += 1
    return graph


print("Creating rabbit graph...")
graph = construct_rabbit_graph(2, 2, 4)
print(f"Graph has {graph.vertices_length()} vertices")

print("\nCloning graph...")
try:
    mdph_carrot_graph = graph.clone()
    print(f"Clone successful! Clone has {mdph_carrot_graph.vertices_length()} vertices")

    # Verify independence
    vlength = mdph_carrot_graph.vertices_length()
    print(f"Accessing vertices in clone...")
    for i in range(min(5, vlength)):
        vertex = mdph_carrot_graph.vertex_at(i)
        rabbits = vertex.state()
        print(f"  Vertex {i}: state={rabbits}")

    print("\n✅ SUCCESS: Rabbit graph clones correctly!")
    os._exit(0)

except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    os._exit(1)
