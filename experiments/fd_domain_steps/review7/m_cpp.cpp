#include "user_model.h"
phasic::Graph build_model(const double* theta, int n_params) {
    phasic::Graph g(1);
    auto s = g.starting_vertex();
    auto a = g.find_or_create_vertex({1});
    auto b = g.find_or_create_vertex({2});
    auto c = g.find_or_create_vertex({3});
    s.add_edge(a, 1.0);
    a.add_edge(b, theta[0]);
    a.add_edge(c, theta[1]);
    b.add_edge(c, 1.0);
    return g;
}
