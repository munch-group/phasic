// cppimport
#include <pybind11/pybind11.h>
#include <phasiccpp.h>

#include <vector>

namespace py = pybind11;

using namespace pybind11::literals; // to bring in the `_a` literal

/* ----------------- Don't change the code above! ----------------- */
/* -----------------------------------------------------------------*/


phasic::Graph coalescent(int nr_samples) {
  phasic::Graph graph(nr_samples);
  phasic::Vertex start = graph.starting_vertex();
  std::vector<int> initial_state(graph.state_length(), 0);
  initial_state[0] = nr_samples;
  phasic::Vertex initial = graph.find_or_create_vertex(initial_state);
  start.add_edge(initial, 1.0);
  for (size_t k = 1; k < graph.vertices_length(); k++) {
    phasic::Vertex vertex = graph.vertex_at(k);
    for (int i = 0; i < nr_samples; ++i) {
      for (int j = i; j < nr_samples && i + j + 1 < nr_samples; ++j) {
        bool same = (i == j);
        std::vector<int> state = vertex.state();
        if (same && state[i] < 2) {
          continue;
        }
        if (!same && (state[i] < 1 || state[j] < 1)) {
          continue;
        }
        state[i]--;
        state[j]--;
        state[i + j + 1]++;
        double weight = same ? (state[i] + 1) * (state[i] + 2) / 2.0
                              : (state[i] + 1) * (state[j] + 1);
        phasic::Vertex child = graph.find_or_create_vertex(state);
        vertex.add_edge(child, weight);
      }
    }
  }
  return graph;
}

/* You can define as many functions as you like */

// phasic::Graph some_other_model(int state_length) {
//   phasic::Graph graph(state_length);
//   phasic::Vertex start = graph.starting_vertex();
//   std::vector<int> initial_state(graph.state_length(), 0);

//   // whatever your initial state/states
//   initial_state[0] = state_length; 
//   phasic::Vertex initial = graph.find_or_create_vertex(initial_state);
//   start.add_edge(initial, 1.0);

//   for (size_t k = 1; k < graph.vertices_length(); k++) {
//     phasic::Vertex vertex = graph.vertex_at(k);

//     // your code here....

//   }
//   return graph;
// }

PYBIND11_MODULE(cpp_state_spaces, m) { // NB: module name must match file base name

        // NB: must match names of functions defined above
        m.def("coalescent", &coalescent); 
        // m.def("some_other_model", &some_other_model); 

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
# methods (Graph::starting_vertex, Graph::find_or_create_vertex, etc.)
# are defined out-of-line in phasiccpp.cpp; the phasic_pybind extension
# does not re-export C++ symbols, so we build our own copy here. The
# methods themselves just call into the ptd_* C API, which IS exported
# from phasic_pybind.so.
cfg["sources"] += [os.path.join(phasic_dir, "include", "cpp", "phasiccpp.cpp")]

# Resolve ptd_* symbols at runtime from whatever's already loaded in the
# Python process. The phasic Python package loads phasic_pybind.so into
# the interpreter, which exports the ptd_* C symbols. Dynamic-lookup
# defers resolution to module-load time. IMPORTANT: this only works if
# phasic_pybind's symbols are in the GLOBAL dynamic-linker scope, but
# CPython loads extensions with RTLD_LOCAL. So before importing this module
# you must re-open phasic_pybind with RTLD_GLOBAL, e.g.:
#     import os, glob, ctypes, phasic
#     _pb = glob.glob(os.path.join(os.path.dirname(phasic.__file__),
#                                  "phasic_pybind*.so"))[0]
#     ctypes.CDLL(_pb, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
# On macOS we additionally pass -undefined,dynamic_lookup below; on Windows
# we'd need to link against an import library.
if sys.platform == "darwin":
    # -undefined,dynamic_lookup defers ptd_* symbol resolution to load time.
    # -w silences the harmless "duplicate -rpath" linker warning that
    # pixi/conda's Python triggers because its LDFLAGS lists the env's lib
    # rpath twice. The warning is upstream packaging, not our build.
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
