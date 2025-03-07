// cppimport
#include <pybind11/pybind11.h>
#include <ptdalgorithms.h>

namespace py = pybind11;

using namespace pybind11::literals; // to bring in the `_a` literal

/* Basic C libraries */
#include "stdint.h"
#include "stdlib.h"

/* ----------------- Don't change the code above! ----------------- */


ptdalgorithms::Graph build(int starting_rabbits, float flooding_left, float flooding_right) {

    // using namespace ptdalgorithms;

    /* Same state size (left_rabbits, right_rabbits) */
    size_t state_size = 2;

    /* Create the graph structure */
    struct ptd_graph *graph = ptd_graph_create(state_size);

    /* We must crease the lookup tree as well! */
    struct ptd_avl_tree *avl_tree = ptd_avl_tree_create(state_size);

    /* Allocate the initial state, which means create an "array" of two values */
    int *initial_state = (int*)calloc(graph->state_length, sizeof(*initial_state));

    /* A "buffer" that allows us to manipulate the new child.
     * We need to 'free' this manually.
     */
    int *child_state = (int*)calloc(graph->state_length, sizeof(*initial_state));
    initial_state[0] = starting_rabbits;

    /* Add the starting edge, just like in the R api */
    ptd_graph_add_edge(
            graph->starting_vertex,
            ptd_find_or_create_vertex(graph, avl_tree, initial_state),
            1
    );

    /* Visit all vertices once */
    for (size_t k = 1; k < graph->vertices_length; k++) {
        struct ptd_vertex *vertex = graph->vertices[k];
        int *state = vertex->state;

        if (state[0] > 0) {
            // Rabbit jump left to right

            /* We use memcpy to copy the state into the child buffer!
             * As C or most other languages will *not* clone by default
             */
            memcpy(child_state, vertex->state, graph->state_length * sizeof(int));
            child_state[0] -= 1;
            child_state[1] += 1;

            /* ptd_find_or_create_vertex will *clone* the child_state
             * so no worries there!
             */
            ptd_graph_add_edge(
                    vertex,
                    ptd_find_or_create_vertex(graph, avl_tree, child_state),
                    1
            );

            memcpy(child_state, vertex->state, graph->state_length * sizeof(int));
            child_state[0] = 0;
            ptd_graph_add_edge(
                    vertex,
                    ptd_find_or_create_vertex(graph, avl_tree, child_state),
                    flooding_left
            );
        }

        if (state[1] > 0) {
            // Rabbit jump right to left
            memcpy(child_state, vertex->state, graph->state_length * sizeof(int));
            child_state[1] -= 1;
            child_state[0] += 1;
            ptd_graph_add_edge(
                    vertex,
                    ptd_find_or_create_vertex(graph, avl_tree, child_state),
                    1
            );

            memcpy(child_state, vertex->state, graph->state_length * sizeof(int));
            child_state[1] = 0;
            ptd_graph_add_edge(
                    vertex,
                    ptd_find_or_create_vertex(graph, avl_tree, child_state),
                    flooding_right
            );
        }
    }

    /* Manually free the allocated buffer memory */
    free(child_state);

    /*
     * This is the first time we use the C++ api to
     * bind the allocated graph to a C++ instance.
     * Only few methods should be used...
     * It has memory management by destructure, so no need
     * to manually free when done using.
     */
    ptdalgorithms::Graph *result = new ptdalgorithms::Graph(graph, avl_tree);

    /*
     * Use Rcpp to make this C++ instance visible to R. This is the
     * exact same type returned by `create_graph` in the R api,
     * which is why we need to make sure the cloned code is identical
     * to the currently installed R library (ptdalgorithms).
     */
    return *result;
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