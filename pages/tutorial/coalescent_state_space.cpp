// cppimport
#include <pybind11/pybind11.h>
#include <phasic.h>

namespace py = pybind11;

using namespace pybind11::literals; // to bring in the `_a` literal

/* Basic C libraries */
#include "stdint.h"
#include "stdlib.h"

/* ----------------- Don't change the code above! ----------------- */


phasic::Graph build(int nr_samples) {

  struct ptd_graph *graph = ptd_graph_create(nr_samples);
  struct ptd_avl_tree *avl_tree = ptd_avl_tree_create(nr_samples);
  int *initial_state = (int *) calloc(graph->state_length, sizeof(int));
  initial_state[0] = nr_samples;
  ptd_graph_add_edge(
    graph->starting_vertex,
    ptd_find_or_create_vertex(graph, avl_tree, initial_state),
    1
  );
  free(initial_state);
  
  int *state = (int *) calloc(graph->state_length, sizeof(int));
  for (size_t k = 1; k < graph->vertices_length; k++) {

    struct ptd_vertex *vertex = graph->vertices[k];
    memcpy(state, vertex->state, graph->state_length * sizeof(int));
    
    for (int i=1; i < nr_samples; ++i) {
      for (int j=i; j < nr_samples; ++j) {
        int same = i == j;
        if (same && state[i] < 2) {
          continue;
        }
        if (same > 0 && (state[i] < 1 or state[j] < 1)) {
          continue ;
        }
        state[i]--;
        state[j]--;
        state[i+j+1]++;
      
        struct ptd_vertex *new_vertex = ptd_find_or_create_vertex(graph, avl_tree, state);
        ptd_graph_add_edge(vertex, child, weight);
      }
    }  
  }
  free(state);

  return graph;
}

/* NB: Change module name below to match the name of this file (without the suffix) */

PYBIND11_MODULE(rabbit_state_space, m) {

        m.def("build", &build);
}

/*
<%
setup_pybind11(cfg)
%>
*/