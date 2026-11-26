#include "phasiccpp.h"
#include <vector>
#include "model_crash.cpp"
extern "C" {
    void* build_graph_ffi(const double* theta, int n_params) {
        phasic::Graph* g = new phasic::Graph(build_model(theta, n_params));
        g->normalize();
        return static_cast<void*>(g);
    }
    void free_graph_ffi(void* graph) { delete static_cast<phasic::Graph*>(graph); }
}
