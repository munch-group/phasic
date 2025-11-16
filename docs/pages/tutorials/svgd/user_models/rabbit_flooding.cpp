/**
 * Rabbit Flooding Model
 *
 * Models rabbits hopping between two islands with flooding risk.
 *
 * State: [left_rabbits, right_rabbits]
 *
 * Parameters:
 * - theta[0] = starting number of rabbits (integer)
 * - theta[1] = flooding rate on left island
 * - theta[2] = flooding rate on right island
 *
 * Transitions:
 * - Rabbits hop between islands at rate 1
 * - Flooding eliminates all rabbits on an island at specified rate
 */

#include "../../include/user_model.h"
#include <vector>
#include <cstring>  // for memcpy

phasic::Graph build_model(const double* theta, int n_params) {
    // Extract parameters
    int starting_rabbits = (n_params > 0) ? static_cast<int>(theta[0]) : 3;
    double flooding_left = (n_params > 1) ? theta[1] : 0.5;
    double flooding_right = (n_params > 2) ? theta[2] : 0.5;

    // State size: [left_rabbits, right_rabbits]
    size_t state_size = 2;

    // Create the graph structure
    phasic::Graph g(state_size);

    // Get starting vertex
    auto start = g.starting_vertex();

    // Initial state: all rabbits on the left island
    std::vector<int> initial_state = {starting_rabbits, 0};

    // Add the starting edge
    auto initial_vertex = g.find_or_create_vertex(initial_state);
    start.add_edge(initial_vertex, 1.0);

    // Buffer for manipulating child states
    std::vector<int> child_state(state_size);

    // Visit all vertices once
    // We need to iterate while new vertices are being added
    size_t current_index = 1;
    while (current_index < g.vertices_length()) {
        auto vertex = g.vertex_at(current_index);
        auto state = vertex.state();

        // Transitions from left island
        if (state[0] > 0) {
            // Rabbit jumps from left to right
            child_state[0] = state[0] - 1;
            child_state[1] = state[1] + 1;
            auto child_vertex = g.find_or_create_vertex(child_state);
            vertex.add_edge(child_vertex, 1.0);

            // Flooding on left island (all rabbits die)
            child_state[0] = 0;
            child_state[1] = state[1];
            auto flood_vertex = g.find_or_create_vertex(child_state);
            vertex.add_edge(flood_vertex, flooding_left);
        }

        // Transitions from right island
        if (state[1] > 0) {
            // Rabbit jumps from right to left
            child_state[0] = state[0] + 1;
            child_state[1] = state[1] - 1;
            auto child_vertex = g.find_or_create_vertex(child_state);
            vertex.add_edge(child_vertex, 1.0);

            // Flooding on right island (all rabbits die)
            child_state[0] = state[0];
            child_state[1] = 0;
            auto flood_vertex = g.find_or_create_vertex(child_state);
            vertex.add_edge(flood_vertex, flooding_right);
        }

        current_index++;
    }

    return g;
}