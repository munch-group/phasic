/*
 * Test edge mode locking at C level
 */

#include <stdio.h>
#include <stdlib.h>
#include "api/c/phasic.h"

int main() {
    // Create graph
    struct ptd_graph *graph = ptd_graph_create(1);
    printf("Created graph\n");

    // Get starting vertex
    struct ptd_vertex *v0 = graph->starting_vertex;
    printf("Starting vertex: %p\n", v0);

    // Create vertices
    int state1[] = {1};
    int state2[] = {2};
    int state3[] = {0};

    struct ptd_vertex *v1 = ptd_vertex_create_state(graph, state1);
    struct ptd_vertex *v2 = ptd_vertex_create_state(graph, state2);
    struct ptd_vertex *v3 = ptd_vertex_create_state(graph, state3);

    printf("Created vertices v1=%p, v2=%p, v3=%p\n", v1, v2, v3);

    // Add IPV edge (doesn't lock mode)
    double coeff1 = 1.0;
    struct ptd_edge *e1 = ptd_graph_add_edge(v0, v1, &coeff1, 1);
    if (e1 == NULL) {
        printf("ERROR adding IPV edge: %s\n", ptd_err);
        return 1;
    }
    printf("Added IPV edge v0->v1: %p\n", e1);

    // Add constant edge (locks to CONSTANT mode)
    double coeff2 = 3.0;
    struct ptd_edge *e2 = ptd_graph_add_edge(v1, v2, &coeff2, 1);
    if (e2 == NULL) {
        printf("ERROR adding constant edge: %s\n", ptd_err);
        return 1;
    }
    printf("Added constant edge v1->v2: %p\n", e2);
    printf("Graph edge_mode after constant edge: %d\n", graph->edge_mode);

    // Try to add parameterized edge (should fail)
    double coeffs3[] = {2.0, 0.5};  // Two coefficients = parameterized
    printf("\nAttempting to add parameterized edge v2->v3 with 2 coefficients...\n");
    struct ptd_edge *e3 = ptd_graph_add_edge(v2, v3, coeffs3, 2);
    if (e3 == NULL) {
        printf("✓ Correctly rejected: %s\n", ptd_err);
    } else {
        printf("✗ ERROR: Mixing was allowed! Edge: %p\n", e3);
        return 1;
    }

    printf("\nAll tests passed!\n");
    return 0;
}
