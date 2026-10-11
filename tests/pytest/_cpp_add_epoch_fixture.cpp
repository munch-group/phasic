// cppimport
//
// Fixture for tests/pytest/test_cpp_add_epoch_lossless.py. Exercises the
// native C++ phasic::Graph::add_epoch (its layout-widening helper) against
// the installed phasic C++ API, the same way _cpp_from_callback_fixture.cpp
// does. NOT collected by pytest; the test module compiles it via cppimport.
#include <pybind11/pybind11.h>
#include <phasiccpp.h>

#include <vector>
#include <utility>

namespace py = pybind11;

// Parameterized coalescent (rate = c0 * theta0), param_length = 1.
static phasic::Graph coalescent_param(int nr_samples) {
    std::vector<int> initial_state(nr_samples, 0);
    initial_state[0] = nr_samples;
    std::vector<std::pair<std::vector<int>, double>> ipv = {{initial_state, 1.0}};

    auto cb = [nr_samples](const std::vector<int> &state) -> std::vector<phasic::Transition> {
        std::vector<phasic::Transition> out;
        for (int i = 0; i < nr_samples; ++i) {
            for (int j = i; j < nr_samples && i + j + 1 < nr_samples; ++j) {
                bool same = (i == j);
                std::vector<int> s = state;
                if (same && s[i] < 2) continue;
                if (!same && (s[i] < 1 || s[j] < 1)) continue;
                s[i]--; s[j]--; s[i + j + 1]++;
                double coeff = same ? (s[i] + 1) * (s[i] + 2) / 2.0
                                    : (s[i] + 1) * (s[j] + 1);
                out.emplace_back(s, 0.0, std::vector<double>{coeff});
            }
        }
        return out;
    };
    return phasic::Graph::from_callback(nr_samples, ipv, cb, /*param_length=*/1);
}

// The base graph, optionally with one coefficient-less constant edge added
// from the first transient vertex (index 1) to the last vertex.
phasic::Graph base_graph(int nr_samples, bool with_constant_edge) {
    phasic::Graph g = coalescent_param(nr_samples);
    g.update_weights(std::vector<double>{1.0});
    if (with_constant_edge) {
        std::vector<phasic::Vertex> vs = g.vertices();
        vs[1].add_edge_constant(vs.back(), 0.75);
    }
    return g;
}

// add_epoch(time) on base_graph(...).
phasic::Graph epoch_graph(int nr_samples, bool with_constant_edge, double time) {
    phasic::Graph g = base_graph(nr_samples, with_constant_edge);
    return g.add_epoch(time);
}

// Base graph with an aux vertex (all-zero state) on each of the first two
// transient vertices: two vertices share the zero state.
phasic::Graph base_with_aux(int nr_samples) {
    phasic::Graph g = coalescent_param(nr_samples);
    std::vector<phasic::Vertex> vs = g.vertices();
    vs[1].add_aux_vertex(std::vector<double>{0.5}).set_aux(true);
    vs[2].add_aux_vertex(std::vector<double>{0.5}).set_aux(true);
    g.update_weights(std::vector<double>{1.0});
    return g;
}

phasic::Graph epoch_with_aux(int nr_samples, double time) {
    phasic::Graph g = base_with_aux(nr_samples);
    return g.add_epoch(time);
}

// Constant coalescent (scalar rates): add_epoch must refuse it.
phasic::Graph constant_epoch(int nr_samples, double time) {
    std::vector<int> initial_state(nr_samples, 0);
    initial_state[0] = nr_samples;
    std::vector<std::pair<std::vector<int>, double>> ipv = {{initial_state, 1.0}};
    auto cb = [nr_samples](const std::vector<int> &state) -> std::vector<phasic::Transition> {
        std::vector<phasic::Transition> out;
        for (int i = 0; i < nr_samples; ++i) {
            for (int j = i; j < nr_samples && i + j + 1 < nr_samples; ++j) {
                bool same = (i == j);
                std::vector<int> s = state;
                if (same && s[i] < 2) continue;
                if (!same && (s[i] < 1 || s[j] < 1)) continue;
                s[i]--; s[j]--; s[i + j + 1]++;
                out.emplace_back(s, same ? (s[i] + 1) * (s[i] + 2) / 2.0
                                         : (s[i] + 1) * (s[j] + 1));
            }
        }
        return out;
    };
    phasic::Graph g = phasic::Graph::from_callback(nr_samples, ipv, cb);
    return g.add_epoch(time);
}

PYBIND11_MODULE(_cpp_add_epoch_fixture, m) {
    m.def("base_graph", &base_graph);
    m.def("epoch_graph", &epoch_graph);
    m.def("base_with_aux", &base_with_aux);
    m.def("epoch_with_aux", &epoch_with_aux);
    m.def("constant_epoch", &constant_epoch);
}

/*
<%
import os, sys, phasic
phasic_dir = os.path.dirname(phasic.__file__)
cfg["include_dirs"] += [
    os.path.join(phasic_dir, "include", "c"),
    os.path.join(phasic_dir, "include", "cpp"),
]
# Compile the installed C++ wrapper (which carries Graph::from_callback) into
# this module. ptd_* C symbols resolve at load time from the already-loaded
# phasic_pybind extension. See docs/pages/tutorial/cpp_state_spaces.cpp for the
# rationale behind this linkage recipe.
cfg["sources"] += [os.path.join(phasic_dir, "include", "cpp", "phasiccpp.cpp")]
cfg["extra_compile_args"] += ["-std=c++17"]
if sys.platform == "darwin":
    cfg["extra_link_args"] += [
        "-Wl,-undefined,dynamic_lookup",
        "-Wl,-w",
    ]
elif sys.platform == "win32":
    raise RuntimeError(
        "Windows cppimport linkage is not supported for this test. "
        "Build phasic from source or use the Python API directly."
    )
setup_pybind11(cfg)
%>
*/
