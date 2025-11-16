/**
 * Birth-Death Process Model
 *
 * Parameters:
 * - theta[0] = birth rate per individual
 * - theta[1] = death rate per individual
 *
 * States represent population size (0 to max_population)
 * Absorption occurs at state 0 (extinction)
 */

#include "../../include/user_model.h"
#include <vector>

phasic::Graph build_model(const double* theta, int n_params) {
    const int max_population = 20;

    // Create graph with state vectors of length 1
    phasic::Graph g(1);

    // Get starting vertex
    auto start = g.starting_vertex();

    double birth_rate = theta[0];
    double death_rate = (n_params > 1) ? theta[1] : 0.0;

    // Create all states (population sizes 0 to max_population)
    std::vector<phasic::Vertex> vertices;
    for (int i = 0; i <= max_population; i++) {
        std::vector<int> state = {i};
        vertices.push_back(g.find_or_create_vertex(state));
    }

    // Set initial distribution: start at population 1
    start.add_edge(vertices[1], 1.0);

    // Add birth transitions: i -> i+1
    for (int i = 1; i < max_population; i++) {
        double rate = birth_rate * i;  // Birth rate proportional to population
        vertices[i].add_edge(vertices[i + 1], rate);
    }

    // Add death transitions: i -> i-1
    for (int i = 1; i <= max_population; i++) {
        double rate = death_rate * i;  // Death rate proportional to population
        vertices[i].add_edge(vertices[i - 1], rate);
    }

    // Note: State 0 is absorbing (extinction), no need to add explicit absorption

    return g;
}