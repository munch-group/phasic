// cppimport
#include <pybind11/pybind11.h>
#include <phasiccpp.h>

#include <vector>
#include <utility>

namespace py = pybind11;

using namespace pybind11::literals; // to bring in the `_a` literal

/* ----------------- Don't change the code above! ----------------- */
/* -----------------------------------------------------------------*/


// C++ port of the Python coalescent built with the callback API:
//
//     def coalescent(state):
//         transitions = []
//         for i in range(state.size):
//             for j in range(i, state.size):
//                 same = int(i == j)
//                 if same and state[i] < 2:
//                     continue
//                 if not same and (state[i] < 1 or state[j] < 1):
//                     continue
//                 new = state.copy()
//                 new[i] -= 1
//                 new[j] -= 1
//                 new[i+j+1] += 1
//                 transitions.append((new, state[i]*(state[j]-same)/(1+same)))
//         return transitions
//
//     nr_samples = 4
//     ipv = [([nr_samples]+[0]*(nr_samples-1), 1)]
//     graph = Graph(coalescent, ipv=ipv)
//
// Graph::from_callback is the native C++ mirror of Graph(callback, ipv=...):
// it runs the breadth-first state-space exploration in C++ (no Python, no GIL)
// and deduplicates repeated states. state[k] is the number of blocks of size
// (k+1); coalescing a size-(i+1) and a size-(j+1) block yields a size-(i+j+2)
// block at index i+j+1. The coalescent invariant sum_k (k+1)*state[k] ==
// nr_samples guarantees i+j+1 stays in range whenever the transition is kept.
phasic::Graph coalescent(int nr_samples) {
  std::vector<int> initial_state(nr_samples, 0);
  initial_state[0] = nr_samples;                                  // [nr_samples, 0, 0, ...]
  std::vector<std::pair<std::vector<int>, double>> ipv = {{initial_state, 1.0}};

  auto callback =
      [](const std::vector<int> &state) -> std::vector<phasic::Transition> {
    std::vector<phasic::Transition> transitions;
    int size = static_cast<int>(state.size());
    for (int i = 0; i < size; ++i) {
      for (int j = i; j < size; ++j) {
        int same = (i == j) ? 1 : 0;
        if (same && state[i] < 2) {
          continue;
        }
        if (!same && (state[i] < 1 || state[j] < 1)) {
          continue;
        }
        std::vector<int> next = state;                           // state.copy()
        next[i] -= 1;
        next[j] -= 1;
        next[i + j + 1] += 1;
        double rate = static_cast<double>(state[i]) * (state[j] - same) / (1 + same);
        transitions.emplace_back(next, rate);                    // constant edge
      }
    }
    return transitions;
  };

  return phasic::Graph::from_callback(nr_samples, ipv, callback);
}

// The C++ equivalent of Python's graph.expectation(): the mean time to
// absorption, E[T]. expected_waiting_time()[v] is E[time to absorption from
// vertex v]; the starting vertex is index 0, so [0] is the distribution mean.
double expectation(int nr_samples) {
  phasic::Graph graph = coalescent(nr_samples);
  return graph.expected_waiting_time()[0];
}

/* You can define as many functions as you like */

PYBIND11_MODULE(cpp_callback_construction, m) { // NB: module name must match file base name

        // NB: must match names of functions defined above
        m.def("coalescent", &coalescent);
        m.def("expectation", &expectation);

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
