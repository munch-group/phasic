/**
 * @file scc_starting_vertex_isolation.c
 * @brief Isolate the starting vertex into its own SCC at index 0
 *
 * This ensures that:
 * 1. The starting vertex is always in SCC 0
 * 2. SCC 0 contains ONLY the starting vertex
 * 3. All other SCCs are shifted accordingly
 *
 * This simplifies trace stitching by eliminating special cases.
 */

/**
 * Isolate the starting vertex into its own SCC at position 0.
 *
 * If the starting vertex is already alone in its SCC, ensure it's at index 0.
 * If not, extract it from its current SCC and create a new SCC for it.
 *
 * @param scc_graph The SCC graph to modify in-place
 * @return 0 on success, non-zero on error
 */
static int ptd_isolate_starting_vertex_scc(struct ptd_scc_graph *scc_graph) {
    if (scc_graph == NULL || scc_graph->graph == NULL) {
        return -1;
    }

    struct ptd_vertex *starting_vertex = scc_graph->graph->starting_vertex;
    struct ptd_scc_vertex *starting_scc = scc_graph->starting_vertex;

    // Case 1: Starting vertex is already alone in its SCC
    if (starting_scc->internal_vertices_length == 1) {
        // Check if it's already at index 0
        if (starting_scc->index == 0) {
            return 0;  // Nothing to do
        }

        // Move it to index 0
        struct ptd_scc_vertex *temp = scc_graph->vertices[0];
        scc_graph->vertices[0] = starting_scc;
        scc_graph->vertices[starting_scc->index] = temp;

        // Update indices
        temp->index = starting_scc->index;
        starting_scc->index = 0;

        return 0;
    }

    // Case 2: Starting vertex shares its SCC with other vertices
    // We need to extract it

    // Create a new SCC for just the starting vertex
    struct ptd_scc_vertex *new_scc = single_vertex_as_scc(starting_vertex);
    if (new_scc == NULL) {
        return -1;
    }

    // Remove starting vertex from its current SCC
    size_t old_scc_size = starting_scc->internal_vertices_length;
    struct ptd_vertex **new_internal_vertices = (struct ptd_vertex **) malloc(
        (old_scc_size - 1) * sizeof(struct ptd_vertex *)
    );
    if (new_internal_vertices == NULL) {
        free(new_scc);
        return -1;
    }

    size_t write_idx = 0;
    for (size_t i = 0; i < old_scc_size; i++) {
        if (starting_scc->internal_vertices[i] != starting_vertex) {
            new_internal_vertices[write_idx++] = starting_scc->internal_vertices[i];
        }
    }

    free(starting_scc->internal_vertices);
    starting_scc->internal_vertices = new_internal_vertices;
    starting_scc->internal_vertices_length = old_scc_size - 1;

    // Reallocate SCC array to make room for the new SCC at position 0
    size_t new_length = scc_graph->vertices_length + 1;
    struct ptd_scc_vertex **new_vertices = (struct ptd_scc_vertex **) malloc(
        new_length * sizeof(struct ptd_scc_vertex *)
    );
    if (new_vertices == NULL) {
        free(new_scc);
        return -1;
    }

    // Insert new SCC at position 0
    new_vertices[0] = new_scc;
    new_scc->index = 0;

    // Copy existing SCCs, shifting indices
    for (size_t i = 0; i < scc_graph->vertices_length; i++) {
        new_vertices[i + 1] = scc_graph->vertices[i];
        new_vertices[i + 1]->index = i + 1;
    }

    free(scc_graph->vertices);
    scc_graph->vertices = new_vertices;
    scc_graph->vertices_length = new_length;
    scc_graph->starting_vertex = new_scc;

    // Now we need to update edges:
    // 1. Edges from the new starting SCC to other SCCs (based on starting vertex's edges)
    // 2. Edges from other SCCs that pointed to the old starting SCC might need updating

    // Build a mapping: vertex -> SCC
    struct ptd_scc_vertex **sccs_for_vertices = (struct ptd_scc_vertex **) calloc(
        scc_graph->graph->vertices_length,
        sizeof(*sccs_for_vertices)
    );
    if (sccs_for_vertices == NULL) {
        return -1;
    }

    for (size_t i = 0; i < scc_graph->vertices_length; i++) {
        struct ptd_scc_vertex *scc = scc_graph->vertices[i];
        for (size_t j = 0; j < scc->internal_vertices_length; j++) {
            struct ptd_vertex *vertex = scc->internal_vertices[j];
            sccs_for_vertices[vertex->index] = scc;
        }
    }

    // Set up edges for the new starting SCC
    // Find all unique target SCCs from starting vertex's edges
    struct ptd_avl_tree *external_sccs = ptd_avl_tree_create(1);
    if (external_sccs == NULL) {
        free(sccs_for_vertices);
        return -1;
    }

    for (size_t i = 0; i < starting_vertex->edges_length; i++) {
        struct ptd_vertex *target = starting_vertex->edges[i]->to;
        struct ptd_scc_vertex *target_scc = sccs_for_vertices[target->index];

        if (target_scc != new_scc) {
            ptd_avl_tree_find_or_insert(external_sccs, (int *) &(target_scc->index), target_scc);
        }
    }

    // Convert tree to array
    struct ptd_vector *external_sccs_vector = vector_create();
    if (external_sccs_vector == NULL) {
        ptd_avl_tree_destroy(external_sccs);
        free(sccs_for_vertices);
        return -1;
    }

    struct ptd_stack *tree_stack = stack_create();
    if (tree_stack == NULL) {
        vector_destroy(external_sccs_vector);
        ptd_avl_tree_destroy(external_sccs);
        free(sccs_for_vertices);
        return -1;
    }

    if (external_sccs->root != NULL) {
        stack_push(tree_stack, external_sccs->root);
    }

    while (!stack_empty(tree_stack)) {
        struct ptd_avl_node *node = (struct ptd_avl_node *) stack_pop(tree_stack);
        vector_add(external_sccs_vector, node->entry);

        if (node->left != NULL) {
            stack_push(tree_stack, node->left);
        }
        if (node->right != NULL) {
            stack_push(tree_stack, node->right);
        }
    }

    new_scc->edges_length = vector_length(external_sccs_vector);
    if (new_scc->edges_length > 0) {
        new_scc->edges = (struct ptd_scc_edge **) calloc(
            new_scc->edges_length,
            sizeof(struct ptd_scc_edge *)
        );

        for (size_t i = 0; i < new_scc->edges_length; i++) {
            new_scc->edges[i] = (struct ptd_scc_edge *) malloc(sizeof(struct ptd_scc_edge));
            new_scc->edges[i]->to = (struct ptd_scc_vertex *) vector_get(external_sccs_vector, i);
        }
    }

    vector_destroy(external_sccs_vector);
    stack_destroy(tree_stack);
    ptd_avl_tree_destroy(external_sccs);
    free(sccs_for_vertices);

    return 0;
}
