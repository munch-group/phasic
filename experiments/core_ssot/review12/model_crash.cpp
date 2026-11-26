#include "user_model.h"
phasic::Graph build_model(const double* theta, int n_params) {
    phasic::Graph g(1);
    auto s = g.starting_vertex();
    auto prev = g.find_or_create_vertex({1});
    s.add_edge(prev, 1.0);
    auto v = g.find_or_create_vertex({2});
    prev.add_edge(v, theta[0]);
    prev = v;   // Vertex copy-assignment
    return g;
}
