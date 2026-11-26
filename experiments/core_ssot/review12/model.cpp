#include "user_model.h"
phasic::Graph build_model(const double* theta, int n_params) {
    int N = (int)theta[1];
    phasic::Graph g(1);
    auto s = g.starting_vertex();
    auto a = g.find_or_create_vertex({1});
    s.add_edge(a, 1.0);
    for (int i = 1; i < N; i++) {
        auto u = g.find_or_create_vertex({i});
        auto v = g.find_or_create_vertex({i+1});
        u.add_edge(v, theta[0] * i);
    }
    return g;
}
