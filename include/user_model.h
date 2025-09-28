/**
 * Simple header for user-defined phase-type models
 *
 * Users should include this header and implement the build_model function
 * that returns a ptdalgorithms::Graph instance.
 */

#ifndef USER_MODEL_H
#define USER_MODEL_H

#include "../api/cpp/ptdalgorithmscpp.h"

/**
 * User-implemented function to build a phase-type model.
 *
 * @param theta Array of model parameters
 * @param n_params Number of parameters
 * @return A configured Graph instance representing the phase-type model
 *
 * Example implementation:
 *
 * ptdalgorithms::Graph build_model(const double* theta, int n_params) {
 *     ptdalgorithms::Graph g(1);  // Create graph with state_length=1
 *
 *     double rate = theta[0];
 *
 *     // Create vertices
 *     auto v0 = g.find_or_create_vertex({0});
 *     auto v1 = g.find_or_create_vertex({1});
 *
 *     // Add transitions
 *     v0.add_edge(v1, rate);
 *
 *     return g;
 * }
 */
ptdalgorithms::Graph build_model(const double* theta, int n_params);

#endif // USER_MODEL_H