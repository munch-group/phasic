/**
 * Erlang Distribution Model
 *
 * Erlang distribution with k stages (shape parameter)
 * Each stage has the same rate parameter
 *
 * Parameters:
 * - theta[0] = rate parameter for each stage
 * - theta[1] = number of stages (optional, defaults to 3)
 */

#include "../../include/user_model.h"
#include <vector>
#include <cmath>

phasic::Graph build_model(const double* theta, int n_params) {
    // Create graph with state vectors of length 1
    phasic::Graph g(1);

    // Get starting vertex
    auto start = g.starting_vertex();

    double rate = theta[0];
    int n_stages = (n_params > 1) ? static_cast<int>(theta[1]) : 3;

    // Create vertices for each stage
    std::vector<phasic::Vertex> vertices;
    for (int i = 0; i <= n_stages; i++) {
        std::vector<int> state = {i};
        vertices.push_back(g.find_or_create_vertex(state));
    }

    // Set initial distribution: start at stage 0
    start.add_edge(vertices[0], 1.0);

    // Add transitions through the stages
    for (int i = 0; i < n_stages; i++) {
        vertices[i].add_edge(vertices[i + 1], rate);
    }

    // The last state (n_stages) is absorbing

    return g;
}