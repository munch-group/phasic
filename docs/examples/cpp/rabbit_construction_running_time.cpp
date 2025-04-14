#include <string.h>
#include "./../../../PtDAlgorithms/api/c/ptdalgorithms.h"
#include "./../../../PtDAlgorithms/api/cpp/ptdalgorithmscpp.h"
#include "./../../../PtDAlgorithms/src/c/ptdalgorithms.c"


// for n in 1000 2000 3000 4000 5000 ; do ( echo ; echo $n ; time ./a.out $n ) >>file.txt 2>&1 ; done


int main(int argc, char **argv) {
    size_t state_size = 2;
    int starting_rabbits;

    if (argc == 1) {
        starting_rabbits = 2;
    } else {
        starting_rabbits = atoi(argv[1]);
    }

    struct ptd_graph *graph = ptd_graph_create(state_size);
    struct ptd_avl_tree *avl_tree = ptd_avl_tree_create(state_size);
    int *initial_state = (int*)calloc(graph->state_length, sizeof(*initial_state));
    int *child_state = (int*)calloc(graph->state_length, sizeof(*initial_state));
    initial_state[0] = starting_rabbits;
    ptd_graph_add_edge(
            graph->starting_vertex,
            ptd_vertex_create_state(graph, initial_state),
            1
    );

    for (size_t k = 1; k < graph->vertices_length; k++) {
        struct ptd_vertex *vertex = graph->vertices[k];
        int *state = vertex->state;

        if (state[0] > 0) {
            // Rabbit jump left to right
            memcpy(child_state, vertex->state, graph->state_length * sizeof(int));
            child_state[0] -= 1;
            child_state[1] += 1;
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
                    2
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
                    4
            );
        }
    }

    free(child_state);

    size_t edges = 0;

    for (size_t i = 0; i < graph->vertices_length; ++i) {
        edges += graph->vertices[i]->edges_length;
    }

    fprintf(stderr, "Finished. Vertices: %zu edges %zu\n", graph->vertices_length, edges);

    ptd_avl_tree_destroy(avl_tree);
    ptd_graph_destroy(graph);

    return 0;
}
