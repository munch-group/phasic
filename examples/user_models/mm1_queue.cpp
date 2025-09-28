/**
 * M/M/1 Queue Model
 *
 * Single-server queue with exponential arrivals and service times
 *
 * Parameters:
 * - theta[0] = arrival rate (lambda)
 * - theta[1] = service rate (mu)
 *
 * States represent queue length (0 to max_queue_size)
 * Absorption occurs when queue reaches maximum size (overflow)
 */

#include "../../include/user_model.h"

ptdalgorithms::Graph build_model(const double* theta, int n_params) {
    const int max_queue_size = 30;

    // Create graph with state vectors of length 1
    ptdalgorithms::Graph g(1);

    // Get starting vertex
    auto start = g.starting_vertex();

    double arrival_rate = theta[0];
    double service_rate = (n_params > 1) ? theta[1] : 1.0;

    // Create vertices for each queue size
    std::vector<ptdalgorithms::Vertex> vertices;
    for (int i = 0; i <= max_queue_size; i++) {
        vertices.push_back(g.find_or_create_vertex({i}));
    }

    // Set initial distribution: start with empty queue
    start.add_edge(vertices[0], 1.0);

    // Add arrival transitions: queue_size -> queue_size + 1
    for (int i = 0; i < max_queue_size; i++) {
        vertices[i].add_edge(vertices[i + 1], arrival_rate);
    }

    // Add service transitions: queue_size -> queue_size - 1
    for (int i = 1; i <= max_queue_size; i++) {
        vertices[i].add_edge(vertices[i - 1], service_rate);
    }

    // The max_queue_size state is absorbing (overflow)

    return g;
}