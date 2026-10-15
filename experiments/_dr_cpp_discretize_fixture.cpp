// cppimport
// De-risk fixture (discretize-cpp-ssot-plan.md phase -1 (c)): call the
// existing C++ phasic::Graph::discretize on a graph handed in from Python.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <phasiccpp.h>
#include <vector>
namespace py = pybind11;

py::tuple cpp_discretize_fixed(phasic::Graph &g, double rate) {
    phasic::DiscretizeResult r = g.discretize(
        [rate](const std::vector<int> &) { return rate; }, false);
    return py::make_tuple(r.graph, r.rewards);
}

py::tuple cpp_discretize_cb(phasic::Graph &g, double mu) {
    phasic::DiscretizeResult r = g.discretize(
        [mu](const std::vector<int> &s) { double t = 0; for (int x : s) t += x; return mu * t; }, false);
    return py::make_tuple(r.graph, r.rewards);
}

PYBIND11_MODULE(_dr_cpp_discretize_fixture, m) {
    m.def("cpp_discretize_fixed", &cpp_discretize_fixed);
    m.def("cpp_discretize_cb", &cpp_discretize_cb);
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
