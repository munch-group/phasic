/**
 * Simple exponential distribution model
 *
 * This model represents a single exponential transition
 * Parameter: theta[0] = rate
 */

#include "../../include/user_model.h"
#include <vector>

phasic::Graph build_model(const double* theta, int n_params) {
    // Create graph with state vectors of length 1
    phasic::Graph g(1);

    // Get the rate parameter
    double rate = theta[0];

    // Get starting vertex
    auto start = g.starting_vertex();

    // Create two states: 0 (initial) and 1 (absorbing)
    std::vector<int> state0 = {0};
    std::vector<int> state1 = {1};
    auto v0 = g.find_or_create_vertex(state0);
    auto v1 = g.find_or_create_vertex(state1);

    // Set initial distribution: start at state 0
    start.add_edge(v0, 1.0);

    // Add transition from state 0 to state 1 with given rate
    v0.add_edge(v1, rate);

    return g;
}