// cppimport
//
// Fixture for tests/pytest/test_c_core_fixes.py: calls C-core functions that
// have no Python binding (ptd_graph_pdf_with_gradient). Compiled like
// _cpp_add_epoch_fixture.cpp. NOT collected by pytest.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <phasiccpp.h>

#include <vector>
#include <stdexcept>

namespace py = pybind11;

py::tuple pdf_with_gradient(phasic::Graph &g, double time, size_t granularity,
                            std::vector<double> params) {
    double value = 0.0;
    std::vector<double> grad(params.size(), 0.0);
    int rc = ptd_graph_pdf_with_gradient(g.c_graph(), time, granularity,
                                         params.data(), params.size(), &value, grad.data());
    if (rc != 0) throw std::runtime_error("ptd_graph_pdf_with_gradient failed");
    return py::make_tuple(value, grad);
}

PYBIND11_MODULE(_cpp_c_core_fixture, m) {
    m.def("pdf_with_gradient", &pdf_with_gradient);
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
