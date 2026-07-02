// cppimport
//
// Fixture for tests/pytest/test_cpp_from_callback.py. Exercises the native
// C++ callback-based graph constructor phasic::Graph::from_callback against the
// installed phasic C++ API (compiled the same way as
// docs/pages/tutorial/cpp_state_spaces.cpp). NOT collected by pytest (it is a
// .cpp, not a test_*.py); the test module compiles it via cppimport.
#include <pybind11/pybind11.h>
#include <phasiccpp.h>

#include <vector>
#include <utility>

namespace py = pybind11;

// Coalescent expressed through the new Graph::from_callback API (constant edges).
phasic::Graph coalescent_from_callback(int nr_samples) {
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
                double weight = same ? (s[i] + 1) * (s[i] + 2) / 2.0
                                     : (s[i] + 1) * (s[j] + 1);
                out.emplace_back(s, weight);
            }
        }
        return out;
    };

    return phasic::Graph::from_callback(nr_samples, ipv, cb);
}

// Same coalescent, but every interior edge is parameterized: rate = c0 * theta0
// with c0 the combinatorial coalescence coefficient. param_length = 1.
phasic::Graph coalescent_param_from_callback(int nr_samples) {
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

// Deliberately invalid: empty IPV. Must throw (surfaced to Python).
phasic::Graph build_empty_ipv() {
    std::vector<std::pair<std::vector<int>, double>> ipv; // empty
    auto cb = [](const std::vector<int> &) -> std::vector<phasic::Transition> {
        return {};
    };
    return phasic::Graph::from_callback(2, ipv, cb);
}

PYBIND11_MODULE(_cpp_from_callback_fixture, m) {
    m.def("coalescent_from_callback", &coalescent_from_callback);
    m.def("coalescent_param_from_callback", &coalescent_param_from_callback);
    m.def("build_empty_ipv", &build_empty_ipv);
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
