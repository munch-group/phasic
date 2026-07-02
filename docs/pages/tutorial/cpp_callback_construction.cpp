// cppimport
#include <pybind11/pybind11.h>
#include <phasiccpp.h>

#include <vector>
#include <utility>

namespace py = pybind11;

using namespace pybind11::literals; // to bring in the `_a` literal

/* ----------------- Don't change the code above! ----------------- */
/* -----------------------------------------------------------------*/


// The same coalescent as cpp_state_spaces.cpp, but built with the callback-
// based constructor Graph::from_callback -- the native C++ mirror of the
// Python `Graph(callback, ipv=..., theta_dim=...)` UI.
//
// Instead of writing the breadth-first state-space exploration loop by hand,
// you supply:
//   * the state length,
//   * the initial probability vector (ipv) as (state, probability) pairs, and
//   * a callback returning the out-transitions of any given (non-empty) state.
// from_callback runs the exploration for you, deduplicating repeated states.
// A transition with an empty coefficient vector is a constant edge; a
// transition carrying coefficients is a parameterized edge.
phasic::Graph coalescent(int nr_samples) {
  std::vector<int> initial_state(nr_samples, 0);
  initial_state[0] = nr_samples;

  std::vector<std::pair<std::vector<int>, double>> ipv = {{initial_state, 1.0}};

  auto callback =
      [nr_samples](const std::vector<int> &state) -> std::vector<phasic::Transition> {
    std::vector<phasic::Transition> transitions;
    for (int i = 0; i < nr_samples; ++i) {
      for (int j = i; j < nr_samples && i + j + 1 < nr_samples; ++j) {
        bool same = (i == j);
        std::vector<int> child = state;
        if (same && child[i] < 2) {
          continue;
        }
        if (!same && (child[i] < 1 || child[j] < 1)) {
          continue;
        }
        child[i]--;
        child[j]--;
        child[i + j + 1]++;
        double weight = same ? (child[i] + 1) * (child[i] + 2) / 2.0
                             : (child[i] + 1) * (child[j] + 1);
        transitions.emplace_back(child, weight);
      }
    }
    return transitions;
  };

  return phasic::Graph::from_callback(nr_samples, ipv, callback);
}

/* You can define as many functions as you like */

PYBIND11_MODULE(cpp_callback_construction, m) { // NB: module name must match file base name

        // NB: must match names of functions defined above
        m.def("coalescent", &coalescent);

}


/* -----------------------------------------------------------------*/
/* --------------- Don't change the content below! ---------------- */


/*
<%
import os, sys, phasic
phasic_dir = os.path.dirname(phasic.__file__)
cfg["include_dirs"] += [
    os.path.join(phasic_dir, "include", "c"),
    os.path.join(phasic_dir, "include", "cpp"),
]
# Compile the C++ wrapper implementation alongside this module. The C++
# methods (Graph::from_callback, Graph::find_or_create_vertex, etc.) are
# defined out-of-line in phasiccpp.cpp; the phasic_pybind extension does not
# re-export C++ symbols, so we build our own copy here. The methods themselves
# just call into the ptd_* C API, which IS exported from phasic_pybind.so.
cfg["sources"] += [os.path.join(phasic_dir, "include", "cpp", "phasiccpp.cpp")]
cfg["extra_compile_args"] += ["-std=c++17"]

# Resolve ptd_* symbols at runtime from whatever's already loaded in the
# Python process (phasic_pybind.so exports the ptd_* C symbols).
if sys.platform == "darwin":
    cfg["extra_link_args"] += [
        "-Wl,-undefined,dynamic_lookup",
        "-Wl,-w",
    ]
elif sys.platform == "win32":
    raise RuntimeError(
        "Windows cppimport linkage is not supported in this tutorial. "
        "Build phasic from source or use the Python API directly."
    )
setup_pybind11(cfg)
%>
*/
