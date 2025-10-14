/**
 * Symbolic Graph Elimination Implementation
 *
 * This file contains the core symbolic graph elimination algorithm that transforms
 * a parameterized graph into an acyclic DAG with symbolic expression trees at edges.
 *
 * This eliminates the need to recompute O(n³) graph elimination for each parameter
 * vector - instead we do it once symbolically and then evaluate O(n) expressions.
 */

#include "ptdalgorithms.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// ============================================================================
// Internal Data Structures for Elimination Algorithm
// ============================================================================

/**
 * Internal representation of symbolic edge during elimination
 * Uses linked list for efficient insertion/deletion
 */
struct ptd_edge_symbolic_ll {
    struct ptd_vertex *to;                      // Target vertex
    struct ptd_expression *prob_expr;           // Probability expression
    struct ptd_edge_symbolic_ll *next;
    struct ptd_edge_symbolic_ll *prev;
    struct ptd_parent_link_ll *parent_link;     // Back-pointer to parent list
};

/**
 * Parent link tracking (which parents point to this vertex)
 */
struct ptd_parent_link_ll {
    struct ptd_vertex *parent_vertex;
    struct ptd_edge_symbolic_ll *edge;          // The edge from parent to this vertex
    struct ptd_parent_link_ll *next;
    struct ptd_parent_link_ll *prev;
};

/**
 * Internal vertex representation during elimination
 */
struct ptd_vertex_symbolic_ll {
    struct ptd_vertex *original;                // Link to original vertex
    struct ptd_expression *rate_expr;           // Rate expression (1 / sum(weights))

    // Edge list with dummy sentinels for easy insertion
    struct ptd_edge_symbolic_ll *first_edge;    // Dummy head
    struct ptd_edge_symbolic_ll *last_edge;     // Dummy tail

    // Parent list
    struct ptd_parent_link_ll *parents;         // List of parents
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Determine parameter length from graph
 *
 * Strategy:
 * 1. If graph->param_length is already set, return it
 * 2. Otherwise, scan parameterized edges and detect length by checking for
 *    garbage values in uninitialized memory beyond the allocated array
 */
static size_t determine_param_length(struct ptd_graph *graph) {
    // If already set, return it
    if (graph->param_length > 0) {
        return graph->param_length;
    }

    // Scan all vertices for parameterized edges
    for (size_t i = 0; i < graph->vertices_length; i++) {
        struct ptd_vertex *v = graph->vertices[i];

        for (size_t j = 0; j < v->edges_length; j++) {
            struct ptd_edge *e = v->edges[j];

            if (!e->parameterized) {
                continue;
            }

            struct ptd_edge_parameterized *ep = (struct ptd_edge_parameterized *)e;
            if (ep->state == NULL) {
                continue;
            }

            // Try increasing lengths to detect the actual array size
            // Strategy: Find the last non-zero coefficient
            // (calloc zeros memory, so beyond the real data we just see zeros)
            size_t detected_len = 0;
            for (size_t try_len = 1; try_len <= 20; try_len++) {
                double val = ep->state[try_len - 1];

                // Check for garbage values indicating we've gone beyond allocated memory
                if (isnan(val) || isinf(val) || fabs(val) > 1e100 ||
                    (val != 0.0 && fabs(val) < 1e-300)) {
                    // Hit garbage
                    break;
                }

                // If non-zero, this is probably real data
                if (val != 0.0) {
                    detected_len = try_len;
                }
                // If zero, it might be trailing zeros or uninitialized memory
                // Keep the last detected_len (last non-zero position)
            }

            // Add 1 more to include trailing explicit zeros (common pattern: [1.0, 0.0])
            if (detected_len > 0 && detected_len < 20) {
                detected_len++;
            }

            // Return the first detected length
            if (detected_len > 0) {
                DEBUG_PRINT("INFO: Auto-detected param_length=%zu from edge state\n", detected_len);
                return detected_len;
            }
        }
    }

    // No parameterized edges found
    DEBUG_PRINT("WARNING: No parameterized edges found, param_length=0\n");
    return 0;
}

/**
 * Extract edge weight as symbolic expression
 *
 * For parameterized edges, creates DOT expression from edge state.
 * For regular edges, creates CONST expression.
 */
static struct ptd_expression *edge_weight_to_expr(struct ptd_edge *edge, size_t param_length) {
    if (edge->parameterized) {
        struct ptd_edge_parameterized *ep = (struct ptd_edge_parameterized *)edge;

        // The edge state is stored in ep->state
        // Need to check if state exists and extract non-zero terms
        if (ep->state == NULL || param_length == 0) {
            return ptd_expr_const(ep->weight);
        }

        // Count non-zero terms
        size_t n_nonzero = 0;
        for (size_t i = 0; i < param_length; i++) {
            if (ep->state[i] != 0.0) {
                n_nonzero++;
            }
        }

        if (n_nonzero == 0) {
            return ptd_expr_const(ep->weight);
        }

        // Build DOT expression
        size_t *indices = (size_t *) malloc(n_nonzero * sizeof(size_t));
        double *coeffs = (double *) malloc(n_nonzero * sizeof(double));
        size_t idx = 0;

        for (size_t i = 0; i < param_length; i++) {
            if (ep->state[i] != 0.0) {
                indices[idx] = i;
                coeffs[idx] = ep->state[i];
                idx++;
            }
        }

        struct ptd_expression *expr;
        if (n_nonzero == 1 && coeffs[0] == 1.0) {
            // Simple parameter reference
            expr = ptd_expr_param(indices[0]);
        } else {
            // Dot product
            expr = ptd_expr_dot(indices, coeffs, n_nonzero);
        }

        free(indices);
        free(coeffs);
        return expr;
    } else {
        return ptd_expr_const(edge->weight);
    }
}

/**
 * Sum an array of expressions
 */
static struct ptd_expression *sum_expressions(struct ptd_expression **exprs, size_t n) {
    if (n == 0) {
        return ptd_expr_const(0.0);
    }
    if (n == 1) {
        return ptd_expr_copy_iterative(exprs[0]);
    }

    struct ptd_expression *sum = ptd_expr_copy_iterative(exprs[0]);
    for (size_t i = 1; i < n; i++) {
        sum = ptd_expr_add(sum, ptd_expr_copy_iterative(exprs[i]));
    }
    return sum;
}

/**
 * Find edge from vertex to target, returns NULL if not found
 */
static struct ptd_edge_symbolic_ll *find_edge_ll(
    struct ptd_vertex_symbolic_ll *from,
    struct ptd_vertex *to
) {
    struct ptd_edge_symbolic_ll *edge = from->first_edge->next;
    while (edge != from->last_edge) {
        if (edge->to == to) {
            return edge;
        }
        edge = edge->next;
    }
    return NULL;
}

/**
 * Insert edge in sorted position (by target vertex address)
 */
static void insert_edge_sorted_ll(
    struct ptd_vertex_symbolic_ll *vertex,
    struct ptd_edge_symbolic_ll *new_edge
) {
    // Find insertion point
    struct ptd_edge_symbolic_ll *curr = vertex->first_edge->next;
    while (curr != vertex->last_edge && curr->to < new_edge->to) {
        curr = curr->next;
    }

    // Insert before curr
    new_edge->next = curr;
    new_edge->prev = curr->prev;
    curr->prev->next = new_edge;
    curr->prev = new_edge;
}

/**
 * Remove edge from list
 */
static void remove_edge_ll(struct ptd_edge_symbolic_ll *edge) {
    edge->prev->next = edge->next;
    edge->next->prev = edge->prev;
}

/**
 * Add parent link
 */
static void add_parent_link_ll(
    struct ptd_vertex_symbolic_ll **vertices,
    size_t child_idx,
    struct ptd_vertex *parent_vertex,
    struct ptd_edge_symbolic_ll *edge
) {
    struct ptd_parent_link_ll *link =
        (struct ptd_parent_link_ll *) malloc(sizeof(*link));
    link->parent_vertex = parent_vertex;
    link->edge = edge;
    link->next = vertices[child_idx]->parents;
    link->prev = NULL;

    if (vertices[child_idx]->parents != NULL) {
        vertices[child_idx]->parents->prev = link;
    }

    vertices[child_idx]->parents = link;
    edge->parent_link = link;
}

/**
 * Remove parent link
 */
static void remove_parent_link_ll(
    struct ptd_vertex_symbolic_ll *child,
    struct ptd_parent_link_ll *link
) {
    if (link->prev != NULL) {
        link->prev->next = link->next;
    } else {
        child->parents = link->next;
    }

    if (link->next != NULL) {
        link->next->prev = link->prev;
    }
}

// ============================================================================
// Main Symbolic Elimination Algorithm
// ============================================================================

/**
 * Symbolic Graph Elimination
 *
 * Performs graph elimination symbolically, building expression trees instead
 * of computing numeric values. The result is an acyclic DAG where edge weights
 * are expressions that can be evaluated with any parameter vector.
 *
 * Time Complexity: O(n³) for elimination (same as numeric)
 * Space Complexity: O(n² * expr_depth) for expression trees
 *
 * The resulting graph enables O(n) instantiation for each parameter vector.
 */
struct ptd_graph_symbolic *ptd_graph_symbolic_elimination(
    struct ptd_graph *graph
) {
    if (graph == NULL) {
        return NULL;
    }

    if (!graph->parameterized) {
        DEBUG_PRINT("WARNING: Symbolic elimination on non-parameterized graph\n");
    }

    // Determine parameter length
    size_t param_length = determine_param_length(graph);

    DEBUG_PRINT("INFO: Starting symbolic elimination (param_length=%zu, vertices=%zu)\n",
                param_length, graph->vertices_length);

    // ========================================================================
    // PHASE 1: Reordering (Topological Sort)
    // ========================================================================

    struct ptd_scc_graph *scc = ptd_find_strongly_connected_components(graph);
    struct ptd_scc_vertex **topo_sorted = ptd_scc_graph_topological_sort(scc);

    // Reorder vertices: non-absorbing first, then absorbing
    struct ptd_vertex **vertices =
        (struct ptd_vertex **) calloc(graph->vertices_length, sizeof(*vertices));
    size_t *original_indices =
        (size_t *) calloc(graph->vertices_length, sizeof(*original_indices));

    size_t idx = 0;

    // First, add non-absorbing vertices
    for (size_t sii = 0; sii < scc->vertices_length; ++sii) {
        for (size_t j = 0; j < topo_sorted[sii]->internal_vertices_length; ++j) {
            if (topo_sorted[sii]->internal_vertices[j]->edges_length == 0) {
                continue;
            }
            original_indices[idx] = topo_sorted[sii]->internal_vertices[j]->index;
            topo_sorted[sii]->internal_vertices[j]->index = idx;
            vertices[idx] = topo_sorted[sii]->internal_vertices[j];
            idx++;
        }
    }

    // Then, add absorbing vertices
    for (size_t sii = 0; sii < scc->vertices_length; ++sii) {
        for (size_t j = 0; j < topo_sorted[sii]->internal_vertices_length; ++j) {
            if (topo_sorted[sii]->internal_vertices[j]->edges_length != 0) {
                continue;
            }
            original_indices[idx] = topo_sorted[sii]->internal_vertices[j]->index;
            topo_sorted[sii]->internal_vertices[j]->index = idx;
            vertices[idx] = topo_sorted[sii]->internal_vertices[j];
            idx++;
        }
    }

    // ========================================================================
    // PHASE 2: Create Symbolic Vertices and Compute Rates
    // ========================================================================

    struct ptd_vertex_symbolic_ll **sym_vertices =
        (struct ptd_vertex_symbolic_ll **) calloc(
            graph->vertices_length, sizeof(*sym_vertices)
        );

    for (size_t i = 0; i < graph->vertices_length; i++) {
        struct ptd_vertex *v = vertices[i];
        struct ptd_vertex_symbolic_ll *sv =
            (struct ptd_vertex_symbolic_ll *) calloc(1, sizeof(*sv));

        sv->original = v;
        sv->parents = NULL;

        // Create dummy sentinels for edge list
        sv->first_edge = (struct ptd_edge_symbolic_ll *) calloc(1, sizeof(*sv->first_edge));
        sv->last_edge = (struct ptd_edge_symbolic_ll *) calloc(1, sizeof(*sv->last_edge));
        sv->first_edge->next = sv->last_edge;
        sv->first_edge->prev = NULL;
        sv->first_edge->to = (struct ptd_vertex *) 1;  // Sentinel marker
        sv->last_edge->prev = sv->first_edge;
        sv->last_edge->next = NULL;
        sv->last_edge->to = (struct ptd_vertex *) ((size_t)-1);  // Sentinel marker

        // Compute rate expression: rate = 1 / sum(edge_weights)
        if (v->edges_length > 0) {
            struct ptd_expression **edge_exprs =
                (struct ptd_expression **) malloc(v->edges_length * sizeof(*edge_exprs));

            for (size_t j = 0; j < v->edges_length; j++) {
                edge_exprs[j] = edge_weight_to_expr(v->edges[j], param_length);
            }

            struct ptd_expression *sum = sum_expressions(edge_exprs, v->edges_length);
            sv->rate_expr = ptd_expr_inv(sum);

            // Clean up temporary expressions
            for (size_t j = 0; j < v->edges_length; j++) {
                ptd_expr_destroy_iterative(edge_exprs[j]);
            }
            free(edge_exprs);
        } else {
            // Absorbing state - rate is undefined, use 0
            sv->rate_expr = ptd_expr_const(0.0);
        }

        sym_vertices[i] = sv;
    }

    // ========================================================================
    // PHASE 3: Convert Edge Weights to Probabilities
    // ========================================================================

    for (size_t i = 0; i < graph->vertices_length; i++) {
        struct ptd_vertex *v = vertices[i];
        struct ptd_vertex_symbolic_ll *sv = sym_vertices[i];

        for (size_t j = 0; j < v->edges_length; j++) {
            struct ptd_edge *e = v->edges[j];

            // Get weight expression
            struct ptd_expression *weight_expr = edge_weight_to_expr(e, param_length);

            // prob = weight * rate
            struct ptd_expression *prob_expr =
                ptd_expr_mul(weight_expr, ptd_expr_copy_iterative(sv->rate_expr));

            // Create symbolic edge
            struct ptd_edge_symbolic_ll *se =
                (struct ptd_edge_symbolic_ll *) calloc(1, sizeof(*se));
            se->to = e->to;
            se->prob_expr = prob_expr;
            se->parent_link = NULL;

            // Insert in sorted order
            insert_edge_sorted_ll(sv, se);

            // Track parent link
            add_parent_link_ll(sym_vertices, e->to->index, v, se);
        }
    }

    DEBUG_PRINT("INFO: Initial setup complete, starting elimination loop\n");

    // ========================================================================
    // PHASE 4: Elimination Loop
    // ========================================================================

    for (size_t i = 0; i < graph->vertices_length; i++) {
        struct ptd_vertex_symbolic_ll *me = sym_vertices[i];
        struct ptd_vertex *me_orig = vertices[i];

        // Copy children to array for stable iteration
        size_t n_children = 0;
        struct ptd_edge_symbolic_ll *edge = me->first_edge->next;
        while (edge != me->last_edge) {
            n_children++;
            edge = edge->next;
        }

        if (n_children == 0) {
            // Absorbing state, nothing to eliminate
            continue;
        }

        struct ptd_edge_symbolic_ll **children =
            (struct ptd_edge_symbolic_ll **) malloc(n_children * sizeof(*children));
        edge = me->first_edge->next;
        for (size_t c = 0; c < n_children; c++) {
            children[c] = edge;
            edge = edge->next;
        }

        // For each parent of me
        struct ptd_parent_link_ll *parent_link = me->parents;
        while (parent_link != NULL) {
            struct ptd_vertex *parent_orig = parent_link->parent_vertex;

            // Skip if parent already processed
            if (parent_orig->index < i) {
                parent_link = parent_link->next;
                continue;
            }

            struct ptd_vertex_symbolic_ll *parent = sym_vertices[parent_orig->index];
            struct ptd_expression *parent_to_me_expr = parent_link->edge->prob_expr;

            // For each of my children
            for (size_t c = 0; c < n_children; c++) {
                struct ptd_edge_symbolic_ll *me_to_child = children[c];
                struct ptd_vertex *child_orig = me_to_child->to;

                // CASE A: Self-loop (child == parent)
                if (child_orig == parent_orig) {
                    // scale = 1 / (1 - parent_to_me * me_to_parent)
                    struct ptd_expression *loop_prob =
                        ptd_expr_mul(
                            ptd_expr_copy_iterative(parent_to_me_expr),
                            ptd_expr_copy_iterative(me_to_child->prob_expr)
                        );
                    struct ptd_expression *one_minus_prob =
                        ptd_expr_sub(ptd_expr_const(1.0), loop_prob);
                    struct ptd_expression *scale = ptd_expr_inv(one_minus_prob);

                    // Find parent's self-loop edge (if exists)
                    struct ptd_edge_symbolic_ll *self_loop = find_edge_ll(parent, parent_orig);
                    if (self_loop != NULL) {
                        // Multiply existing self-loop by scale
                        self_loop->prob_expr =
                            ptd_expr_mul(self_loop->prob_expr, scale);
                    }
                    // If no self-loop exists, the scale factor affects all edges
                    // (implicitly incorporated in normalization)

                    continue;
                }

                // Skip edge back to me
                if (child_orig == me_orig) {
                    continue;
                }

                // Find if parent already has edge to child
                struct ptd_edge_symbolic_ll *parent_to_child = find_edge_ll(parent, child_orig);

                if (parent_to_child != NULL) {
                    // CASE B: Matching edge - add bypass probability
                    // new_prob = old_prob + (parent_to_me * me_to_child)
                    struct ptd_expression *bypass =
                        ptd_expr_mul(
                            ptd_expr_copy_iterative(parent_to_me_expr),
                            ptd_expr_copy_iterative(me_to_child->prob_expr)
                        );
                    parent_to_child->prob_expr =
                        ptd_expr_add(parent_to_child->prob_expr, bypass);
                } else {
                    // CASE C: New edge
                    // new_prob = parent_to_me * me_to_child
                    struct ptd_expression *new_prob =
                        ptd_expr_mul(
                            ptd_expr_copy_iterative(parent_to_me_expr),
                            ptd_expr_copy_iterative(me_to_child->prob_expr)
                        );

                    struct ptd_edge_symbolic_ll *new_edge =
                        (struct ptd_edge_symbolic_ll *) calloc(1, sizeof(*new_edge));
                    new_edge->to = child_orig;
                    new_edge->prob_expr = new_prob;

                    insert_edge_sorted_ll(parent, new_edge);
                    add_parent_link_ll(sym_vertices, child_orig->index, parent_orig, new_edge);
                }
            }

            // Remove edge from parent to me
            remove_edge_ll(parent_link->edge);
            remove_parent_link_ll(me, parent_link);

            // NORMALIZATION: Renormalize parent's edges
            // total = sum(all edge probs)
            size_t parent_n_edges = 0;
            edge = parent->first_edge->next;
            while (edge != parent->last_edge) {
                parent_n_edges++;
                edge = edge->next;
            }

            if (parent_n_edges > 0) {
                struct ptd_expression **parent_edge_exprs =
                    (struct ptd_expression **) malloc(parent_n_edges * sizeof(*parent_edge_exprs));
                edge = parent->first_edge->next;
                for (size_t e = 0; e < parent_n_edges; e++) {
                    parent_edge_exprs[e] = edge->prob_expr;
                    edge = edge->next;
                }

                struct ptd_expression *total = sum_expressions(parent_edge_exprs, parent_n_edges);

                // Normalize: prob = prob / total
                edge = parent->first_edge->next;
                while (edge != parent->last_edge) {
                    edge->prob_expr = ptd_expr_div(edge->prob_expr, ptd_expr_copy_iterative(total));
                    edge = edge->next;
                }

                ptd_expr_destroy_iterative(total);
                free(parent_edge_exprs);
            }

            parent_link = parent_link->next;
        }

        free(children);
    }

    DEBUG_PRINT("INFO: Elimination loop complete, building result structure\n");

    // ========================================================================
    // PHASE 5: Build Result Structure
    // ========================================================================

    struct ptd_graph_symbolic *result =
        (struct ptd_graph_symbolic *) calloc(1, sizeof(*result));
    result->vertices_length = graph->vertices_length;
    result->state_length = graph->state_length;
    result->param_length = param_length;
    result->is_acyclic = true;
    result->is_discrete = graph->was_dph;
    result->original_graph = graph;

    result->vertices = (struct ptd_vertex_symbolic **)
        calloc(graph->vertices_length, sizeof(*result->vertices));

    // Convert internal representation to public API
    for (size_t i = 0; i < graph->vertices_length; i++) {
        struct ptd_vertex_symbolic_ll *sv = sym_vertices[i];
        struct ptd_vertex_symbolic *public_sv =
            (struct ptd_vertex_symbolic *) calloc(1, sizeof(*public_sv));

        public_sv->index = i;
        public_sv->original_vertex = sv->original;

        // Copy state
        public_sv->state = (int *) malloc(graph->state_length * sizeof(int));
        memcpy(public_sv->state, sv->original->state, graph->state_length * sizeof(int));

        // Copy rate expression (1/rate scaling factor)
        public_sv->rate_expr = ptd_expr_copy_iterative(sv->rate_expr);

        // Count edges
        size_t n_edges = 0;
        struct ptd_edge_symbolic_ll *edge = sv->first_edge->next;
        while (edge != sv->last_edge) {
            n_edges++;
            edge = edge->next;
        }

        public_sv->edges_length = n_edges;
        public_sv->edges = (struct ptd_edge_symbolic **)
            calloc(n_edges, sizeof(*public_sv->edges));

        // Convert edges
        edge = sv->first_edge->next;
        for (size_t j = 0; j < n_edges; j++) {
            struct ptd_edge_symbolic *public_edge =
                (struct ptd_edge_symbolic *) calloc(1, sizeof(*public_edge));
            public_edge->to_index = edge->to->index;  // Store index, not pointer!
            public_edge->weight_expr = ptd_expr_copy_iterative(edge->prob_expr);
            public_edge->next = NULL;

            public_sv->edges[j] = public_edge;
            edge = edge->next;
        }

        result->vertices[i] = public_sv;
    }

    // Set starting vertex to point to the first vertex (which is the original starting vertex)
    // The starting vertex is always at index 0 in the vertices array
    if (graph->vertices_length > 0) {
        result->starting_vertex = result->vertices[0];
    } else {
        result->starting_vertex = NULL;
    }

    // Cleanup internal structures
    for (size_t i = 0; i < graph->vertices_length; i++) {
        struct ptd_vertex_symbolic_ll *sv = sym_vertices[i];

        // Free edges
        struct ptd_edge_symbolic_ll *edge = sv->first_edge->next;
        while (edge != sv->last_edge) {
            struct ptd_edge_symbolic_ll *next = edge->next;
            ptd_expr_destroy_iterative(edge->prob_expr);
            free(edge);
            edge = next;
        }
        free(sv->first_edge);
        free(sv->last_edge);

        // Free parents
        struct ptd_parent_link_ll *plink = sv->parents;
        while (plink != NULL) {
            struct ptd_parent_link_ll *next = plink->next;
            free(plink);
            plink = next;
        }

        ptd_expr_destroy_iterative(sv->rate_expr);
        free(sv);
    }

    free(sym_vertices);
    free(vertices);
    free(original_indices);
    free(topo_sorted);
    ptd_scc_graph_destroy(scc);

    DEBUG_PRINT("INFO: Symbolic elimination complete!\n");

    return result;
}

// ============================================================================
// Graph Instantiation and Cleanup
// ============================================================================

/**
 * Destroy symbolic graph
 */
void ptd_graph_symbolic_destroy(struct ptd_graph_symbolic *symbolic) {
    if (symbolic == NULL) {
        return;
    }

    if (symbolic->vertices != NULL) {
        for (size_t i = 0; i < symbolic->vertices_length; i++) {
            if (symbolic->vertices[i] != NULL) {
                struct ptd_vertex_symbolic *sv = symbolic->vertices[i];

                if (sv->edges != NULL) {
                    for (size_t j = 0; j < sv->edges_length; j++) {
                        if (sv->edges[j] != NULL) {
                            ptd_expr_destroy_iterative(sv->edges[j]->weight_expr);
                            free(sv->edges[j]);
                        }
                    }
                    free(sv->edges);
                }

                // Free rate expression
                if (sv->rate_expr != NULL) {
                    ptd_expr_destroy_iterative(sv->rate_expr);
                }

                free(sv->state);
                free(sv);
            }
        }
        free(symbolic->vertices);
    }

    // starting_vertex now just points to vertices[0], so we don't free it separately
    // (it's already freed in the loop above)

    free(symbolic);
}

/**
 * Instantiate symbolic graph with concrete parameters
 *
 * This is O(n) instead of O(n³)!
 */
struct ptd_graph *ptd_graph_symbolic_instantiate(
    const struct ptd_graph_symbolic *symbolic,
    const double *params,
    size_t n_params
) {
    if (symbolic == NULL || params == NULL) {
        return NULL;
    }

    // Create new graph
    struct ptd_graph *graph = ptd_graph_create(symbolic->state_length);
    if (graph == NULL) {
        return NULL;
    }

    // ptd_graph_create() already created a starting vertex at index 0 with state [0,0,...]
    // We need to replace it with the correct state from symbolic->vertices[0]
    if (symbolic->vertices_length > 0 && graph->vertices_length > 0) {
        // Update the starting vertex's state
        memcpy(graph->vertices[0]->state, symbolic->vertices[0]->state,
               symbolic->state_length * sizeof(int));
    }

    // Create remaining vertices (starting from index 1)
    for (size_t i = 1; i < symbolic->vertices_length; i++) {
        struct ptd_vertex *v = ptd_vertex_create_state(
            graph,
            symbolic->vertices[i]->state
        );
        (void)v;  // Vertices created in order
    }

    // Add edges with evaluated weights
    for (size_t i = 0; i < symbolic->vertices_length; i++) {
        struct ptd_vertex_symbolic *sv = symbolic->vertices[i];
        struct ptd_vertex *v = graph->vertices[i];

        // Evaluate rate expression (1/rate) for this vertex
        double inv_rate = ptd_expr_evaluate_iterative(sv->rate_expr, params, n_params);

        for (size_t j = 0; j < sv->edges_length; j++) {
            struct ptd_edge_symbolic *se = sv->edges[j];

            // Evaluate probability expression
            double prob = ptd_expr_evaluate_iterative(se->weight_expr, params, n_params);

            // Convert probability back to rate: weight = prob * rate = prob / (1/rate)
            double weight = prob / inv_rate;

            // Get target vertex from new graph using index
            struct ptd_vertex *to_vertex = graph->vertices[se->to_index];

            // Add edge
            ptd_graph_add_edge(v, to_vertex, weight);
        }
    }

    // Graph is already acyclic!
    return graph;
}

/**
 * Batch instantiation
 */
void ptd_graph_symbolic_instantiate_batch(
    const struct ptd_graph_symbolic *symbolic,
    const double *params_batch,
    size_t batch_size,
    size_t n_params,
    struct ptd_graph **graphs_out
) {
    if (symbolic == NULL || params_batch == NULL || graphs_out == NULL) {
        return;
    }

    for (size_t i = 0; i < batch_size; i++) {
        const double *params_i = params_batch + i * n_params;
        graphs_out[i] = ptd_graph_symbolic_instantiate(symbolic, params_i, n_params);
    }
}
