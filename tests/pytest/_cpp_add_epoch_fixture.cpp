// cppimport
//
// Fixture for tests/pytest/test_cpp_add_epoch_lossless.py. Exercises the
// native C++ phasic::Graph::add_epoch (its layout-widening helper) against
// the installed phasic C++ API, the same way _cpp_from_callback_fixture.cpp
// does. NOT collected by pytest; the test module compiles it via cppimport.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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

// add_epoch refusals (operation-level guards; the widening primitive does
// not check these). Each builds the base graph, breaks one precondition and
// calls add_epoch.
phasic::Graph ragged_epoch(int nr_samples) {
    phasic::Graph g = coalescent_param(nr_samples);
    std::vector<phasic::Vertex> vs = g.vertices();
    vs[1].add_edge_parameterized(vs.back(), 0.0, std::vector<double>{1.0, 2.0});
    // no update_weights: it already refuses the overlong vector; add_epoch
    // must refuse it before using any weights.
    return g.add_epoch(1.0);
}

phasic::Graph was_dph_epoch(int nr_samples) {
    phasic::Graph g = base_graph(nr_samples, false);
    g.c_graph()->was_dph = true;
    return g.add_epoch(1.0);
}

phasic::Graph formula_epoch(int nr_samples) {
    phasic::Graph g = base_graph(nr_samples, false);
    g.weight_formula("t0*c0");
    return g.add_epoch(1.0);
}

// theta layout names after the C++ copy paths not reachable from Python.
static py::list layout_names(phasic::Graph &g) {
    py::list out;
    for (const phasic::ParameterSlot &p : g.theta_layout()) out.append(p.name + ":" + p.kind);
    return out;
}

py::tuple layout_after_copies(int nr_samples) {
    phasic::Graph g = coalescent_param(nr_samples);
    g.set_parameter_name(0, "coalescent");
    phasic::Graph *p = g.clone_p();
    py::list via_clone_p = layout_names(*p);
    delete p;
    phasic::Graph assigned = coalescent_param(nr_samples);
    assigned = g;                     // copy-assign
    phasic::Graph moved = coalescent_param(nr_samples);
    moved = g.clone();                // rvalue: falls back to copy-assign
    phasic::Graph widened = g.widen_layout(0, 1);
    return py::make_tuple(layout_names(g), via_clone_p, layout_names(assigned),
                          layout_names(moved), layout_names(widened));
}

// C++-only discretize of a parameterized graph (phase 3): fixed rate and a
// per-state callback written in C++.
py::tuple cpp_discretize_param(int nr_samples, double rate) {
    phasic::Graph g = coalescent_param(nr_samples);
    phasic::DiscretizeResult r = g.discretize(rate);
    return py::make_tuple(r.graph, r.rewards);
}

py::tuple cpp_discretize_param_callback(int nr_samples, double mu) {
    phasic::Graph g = coalescent_param(nr_samples);
    phasic::DiscretizeResult r = g.discretize(
        [mu](const std::vector<int> &s) { double t = 0; for (int x : s) t += x; return mu * t; });
    return py::make_tuple(r.graph, r.rewards);
}

phasic::Graph discretize_then_epoch(int nr_samples) {
    phasic::Graph g = coalescent_param(nr_samples);
    phasic::Graph d = g.discretize(0.1).graph;
    return d.add_epoch(1.0);
}

PYBIND11_MODULE(_cpp_add_epoch_fixture, m) {
    m.def("discretize_then_epoch", &discretize_then_epoch);
    m.def("cpp_discretize_param", &cpp_discretize_param);
    m.def("cpp_discretize_param_callback", &cpp_discretize_param_callback);
    m.def("layout_after_copies", &layout_after_copies);
    m.def("ragged_epoch", &ragged_epoch);
    m.def("was_dph_epoch", &was_dph_epoch);
    m.def("formula_epoch", &formula_epoch);
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
# Rebuild whenever the installed C++ API changes (cppimport's checksum
# otherwise covers only this file, so a stale .so could test old code).
cfg["dependencies"] = [
    os.path.join(phasic_dir, "include", "cpp", "phasiccpp.cpp"),
    os.path.join(phasic_dir, "include", "cpp", "phasiccpp.h"),
]
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
