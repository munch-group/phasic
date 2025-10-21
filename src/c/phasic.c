/*
 * MIT License
 *
 * Copyright (c) 2021 Tobias Røikjer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "phasic.h"
#include "../../api/c/phasic_hash.h"

volatile char ptd_err[4096] = {'\0'};

/*
 * Utility data structures
 */

struct ptd_ll {
    void *value;
    struct ptd_ll *next;
};

struct ptd_vector {
    size_t entries;
    void **arr;
};

static struct ptd_vector *vector_create();

static int vector_add(struct ptd_vector *vector, void *entry);

static void *vector_get(struct ptd_vector *vector, size_t index);

static size_t vector_length(struct ptd_vector *vector);

static void vector_destroy(struct ptd_vector *vector);

/* Forward declarations for trace cache functions */
static struct ptd_elimination_trace *load_trace_from_cache(const char *hash_hex);
static bool save_trace_to_cache(const char *hash_hex, const struct ptd_elimination_trace *trace);

struct ptd_queue {
    struct ptd_ll *ll;
    struct ptd_ll *tail;
};

static struct ptd_queue *queue_create();

static void queue_destroy(struct ptd_queue *queue);

static int queue_enqueue(struct ptd_queue *queue, void *entry);

static void *queue_dequeue(struct ptd_queue *queue);

static int queue_empty(struct ptd_queue *queue);

struct ptd_stack {
    struct ptd_ll *ll;
};

static struct ptd_stack *stack_create();

static void stack_destroy(struct ptd_stack *stack);

static int stack_push(struct ptd_stack *stack, void *entry);

static void *stack_pop(struct ptd_stack *stack);

static int stack_empty(struct ptd_stack *stack);

struct ll_of_a {
    struct ll_of_a *next;
    double *mem;
    size_t current_mem_index;
    double *current_mem_position;
};

/*
 * AVL tree
 */

static void _ptd_avl_tree_destroy(struct ptd_avl_node *avl_vertex);

struct ptd_avl_tree *ptd_avl_tree_create(size_t key_length) {
    struct ptd_avl_tree *avl_tree = (struct ptd_avl_tree *) malloc(sizeof(struct ptd_avl_tree));

    if (avl_tree == NULL) {
        return NULL;
    }

    avl_tree->root = NULL;
    avl_tree->key_length = key_length;

    return avl_tree;
}

void ptd_avl_tree_destroy(struct ptd_avl_tree *avl_tree) {
    _ptd_avl_tree_destroy((struct ptd_avl_node *) avl_tree->root);
    avl_tree->root = NULL;
    free(avl_tree);
}

/* Example:
*     A            A            A
*   B   (left)    B  (right)   D
* C       ->    D      ->    C   B
*   D         C
* In this case:
*  C: child
*  B: parent
*  D: child_right
*/
struct ptd_avl_node *rotate_left_right(struct ptd_avl_node *parent, struct ptd_avl_node *child) {
    struct ptd_avl_node *child_right_left, *child_right_right;
    struct ptd_avl_node *child_right = child->right;
    child_right_left = child_right->left;
    child->right = child_right_left;

    if (child_right_left != NULL) {
        child_right_left->parent = child;
    }

    child_right->left = child;
    child->parent = child_right;
    child_right_right = child_right->right;
    parent->left = child_right_right;

    if (child_right_right != NULL) {
        child_right_right->parent = parent;
    }

    child_right->right = parent;
    parent->parent = child_right;

    if (child_right->balance > 0) {
        parent->balance = -1;
        child->balance = 0;
    } else if (child_right->balance == 0) {
        parent->balance = 0;
        child->balance = 0;
    } else {
        parent->balance = 0;
        child->balance = +1;
    }

    child_right->balance = 0;

    return child_right;
}

/* Example:
*  A          A            A
*   B  (right)  B   (left)   D
*     C   ->      D    -> B    C
*   D               C
* In this case:
*  C: child
*  B: parent
*  D: child_left
*/
struct ptd_avl_node *rotate_right_left(struct ptd_avl_node *parent, struct ptd_avl_node *child) {
    struct ptd_avl_node *child_left_right, *child_left_left;
    struct ptd_avl_node *child_left = child->left;

    child_left_right = child_left->right;

    child->left = child_left_right;

    if (child_left_right != NULL) {
        child_left_right->parent = child;
    }

    child_left->right = child;

    child->parent = child_left;
    child_left_left = child_left->left;
    parent->right = child_left_left;

    if (child_left_left != NULL) {
        child_left_left->parent = parent;
    }

    child_left->left = parent;
    parent->parent = child_left;

    if (child_left->balance > 0) {
        parent->balance = -1;
        child->balance = 0;
    } else if (child_left->balance == 0) {
        parent->balance = 0;
        child->balance = 0;
    } else {
        parent->balance = 0;
        child->balance = 1;
    }

    child_left->balance = 0;

    return child_left;
}

/*
* Example:
*  A              B
*    B   (left) A   C
*      C   ->
*/
struct ptd_avl_node *rotate_left(struct ptd_avl_node *parent, struct ptd_avl_node *child) {
    struct ptd_avl_node *child_left;

    child_left = child->left;
    parent->right = child_left;

    if (child_left != NULL) {
        child_left->parent = parent;
    }

    child->left = parent;
    parent->parent = child;

    if (child->balance == 0) {
        parent->balance = 1;
        child->balance = -1;
    } else {
        parent->balance = 0;
        child->balance = 0;
    }

    return child;
}

/*
* Example:
*      A            B
*    B    (right) C   A
*  C        ->
*/
struct ptd_avl_node *rotate_right(struct ptd_avl_node *parent, struct ptd_avl_node *child) {
    struct ptd_avl_node *child_right;

    child_right = child->right;
    parent->left = child_right;

    if (child_right != NULL) {
        child_right->parent = parent;
    }

    child->right = parent;
    parent->parent = child;

    if (child->balance == 0) {
        parent->balance = +1;
        child->balance = -1;
    } else {
        parent->balance = 0;
        child->balance = 0;
    }

    return child;
}

struct ptd_avl_node *ptd_avl_node_create(const int *key, const void *entry, struct ptd_avl_node *parent) {
    struct ptd_avl_node *vertex;

    if ((vertex = (struct ptd_avl_node *) malloc(sizeof(*vertex))) == NULL) {
        return NULL;
    }

    vertex->key = (int *) key;
    vertex->entry = (void *) entry;
    vertex->left = NULL;
    vertex->right = NULL;
    vertex->parent = parent;
    vertex->balance = 0;

    return vertex;
}

static void ptd_avl_node_destroy(struct ptd_avl_node *vertex) {
    if (vertex == NULL) {
        return;
    }

    ptd_avl_node_destroy(vertex->left);
    ptd_avl_node_destroy(vertex->right);

    free(vertex);
}

static void avl_free(struct ptd_avl_node *vertex) {
    if (vertex == NULL) {
        return;
    }

    avl_free(vertex->left);
    avl_free(vertex->right);
    free(vertex);
}

const struct ptd_avl_node *
avl_vec_find(const struct ptd_avl_node *rootptr, const char *key, const size_t vec_length) {
    if (rootptr == NULL) {
        return NULL;
    }

    const struct ptd_avl_node *vertex = rootptr;

    while (true) {
        int res = memcmp(key, vertex->key, vec_length);

        if (res < 0) {
            if (vertex->left == NULL) {
                return NULL;
            } else {
                vertex = vertex->left;
            }
        } else if (res > 0) {
            if (vertex->right == NULL) {
                return NULL;
            } else {
                vertex = vertex->right;
            }
        } else {
            return vertex;
        }
    }
}

int find_or_insert_vec(struct ptd_avl_node **out, struct ptd_avl_node *rootptr, int *key, void *entry,
                       const size_t vec_length) {
    if ((*out = ptd_avl_node_create(key, entry, NULL)) == NULL) {
        return -1;
    }

    if (rootptr == NULL) {
        return 1;
    }

    struct ptd_avl_node *vertex = rootptr;

    while (true) {
        int res = memcmp(key, vertex->key, vec_length);

        if (res < 0) {
            if (vertex->left == NULL) {
                vertex->left = *out;
                break;
            } else {
                vertex = vertex->left;
            }
        } else if (res > 0) {
            if (vertex->right == NULL) {
                vertex->right = *out;
                break;
            } else {
                vertex = vertex->right;
            }
        } else {
            free(*out);
            *out = vertex;
            return 0;
        }
    }

    (*out)->parent = vertex;

    return 0;
}

int avl_rebalance_tree(struct ptd_avl_node **root, struct ptd_avl_node *child) {
    struct ptd_avl_node *pivot, *rotated_parent;

    for (struct ptd_avl_node *parent = child->parent; parent != NULL; parent = child->parent) {
        if (child == parent->right) {
            if (parent->balance > 0) {
                pivot = parent->parent;

                if (child->balance < 0) {
                    rotated_parent = rotate_right_left(parent, child);
                } else {
                    rotated_parent = rotate_left(parent, child);
                }
            } else {
                if (parent->balance < 0) {
                    parent->balance = 0;

                    return 0;
                }

                parent->balance = 1;
                child = parent;

                continue;
            }
        } else {
            if (parent->balance < 0) {
                pivot = parent->parent;

                if (child->balance > 0) {
                    rotated_parent = rotate_left_right(parent, child);
                } else {
                    rotated_parent = rotate_right(parent, child);
                }
            } else {
                if (parent->balance > 0) {
                    parent->balance = 0;

                    return 0;
                }

                parent->balance = -1;
                child = parent;
                continue;
            }
        }

        rotated_parent->parent = pivot;

        if (pivot != NULL) {
            if (parent == pivot->left) {
                pivot->left = rotated_parent;
            } else {
                pivot->right = rotated_parent;
            }

            return 0;
        } else {
            *root = rotated_parent;
        }
    }

    return 0;
}


static size_t avl_vec_get_size(struct ptd_avl_node *vertex) {
    if (vertex == NULL) {
        return 0;
    }

    return 1 + avl_vec_get_size(vertex->left) + avl_vec_get_size(vertex->right);
}


static void _ptd_avl_tree_destroy(struct ptd_avl_node *avl_vertex) {
    if (avl_vertex == NULL) {
        return;
    }

    _ptd_avl_tree_destroy(avl_vertex->left);
    _ptd_avl_tree_destroy(avl_vertex->right);

    avl_vertex->left = NULL;
    avl_vertex->right = NULL;
    avl_vertex->entry = NULL;
    free(avl_vertex);
}

#define _ptd_max(a, b) a >= b ? a : b
#define _ptd_min(a, b) a <= b ? a : b

size_t ptd_avl_tree_max_depth(void *avl_vec_vertex) {
    if ((struct ptd_avl_node *) avl_vec_vertex == NULL) {
        return 0;
    }

    return _ptd_max(
                   ptd_avl_tree_max_depth((void *) ((struct ptd_avl_node *) avl_vec_vertex)->left) + 1,
                   ptd_avl_tree_max_depth((void *) ((struct ptd_avl_node *) avl_vec_vertex)->left) + 1
           );
}


struct ptd_avl_node *ptd_avl_tree_find_or_insert(struct ptd_avl_tree *avl_tree, const int *key, const void *entry) {
    struct ptd_avl_node *new_node = ptd_avl_node_create(key, entry, NULL);

    if (new_node == NULL) {
        return NULL;
    }

    if (avl_tree->root == NULL) {
        avl_tree->root = new_node;

        return new_node;
    }

    struct ptd_avl_node *vertex = avl_tree->root;

    while (true) {
        int res = memcmp(key, vertex->key, sizeof(int) * avl_tree->key_length);

        if (res < 0) {
            if (vertex->left == NULL) {
                vertex->left = new_node;
                break;
            } else {
                vertex = vertex->left;
            }
        } else if (res > 0) {
            if (vertex->right == NULL) {
                vertex->right = new_node;
                break;
            } else {
                vertex = vertex->right;
            }
        } else {
            free(new_node);
            return vertex;
        }
    }

    new_node->parent = vertex;

    avl_rebalance_tree(&avl_tree->root, new_node);

    return new_node;
}

struct ptd_avl_node *ptd_avl_tree_find(const struct ptd_avl_tree *avl_tree, const int *key) {
    struct ptd_avl_node *vertex = avl_tree->root;

    while (true) {
        if (vertex == NULL) {
            return NULL;
        }

        int res = memcmp(key, vertex->key, sizeof(int) * avl_tree->key_length);

        if (res < 0) {
            vertex = vertex->left;
        } else if (res > 0) {
            vertex = vertex->right;
        } else {
            return vertex;
        }
    }
}

int ptd_precompute_reward_compute_graph(struct ptd_graph *graph) {
    if (graph->was_dph) {
        graph->was_dph = false;

        if (graph->reward_compute_graph != NULL) {
            free(graph->reward_compute_graph->commands);
            free(graph->reward_compute_graph);
        }

        if (graph->parameterized_reward_compute_graph != NULL) {
            ptd_parameterized_reward_compute_graph_destroy(
                    graph->parameterized_reward_compute_graph
            );
        }

        graph->reward_compute_graph = NULL;
        graph->parameterized_reward_compute_graph = NULL;
    }

    if (graph->reward_compute_graph == NULL) {
        if (graph->parameterized) {
            // Use trace-based path if trace exists and parameters are available
            if (graph->elimination_trace != NULL && graph->current_params != NULL) {
                // DEBUG_PRINT("INFO: using trace-based path for reward compute graph...\n");

                // Evaluate trace with current parameters
                struct ptd_trace_result *trace_result = ptd_evaluate_trace(
                    graph->elimination_trace,
                    graph->current_params,
                    graph->param_length
                );

                if (trace_result == NULL) {
                    DEBUG_PRINT("WARNING: trace evaluation failed, falling back to traditional path\n");
                    goto traditional_path;
                }

                // Build reward_compute from trace result
                graph->reward_compute_graph = ptd_build_reward_compute_from_trace(
                    trace_result,
                    graph
                );

                ptd_trace_result_destroy(trace_result);

                if (graph->reward_compute_graph == NULL) {
                    DEBUG_PRINT("WARNING: reward_compute building failed, falling back to traditional path\n");
                    goto traditional_path;
                }

                // DEBUG_PRINT("INFO: trace-based reward compute graph built successfully\n");
            } else {
                // Traditional path
traditional_path:
                if (graph->parameterized_reward_compute_graph == NULL) {
                    DEBUG_PRINT("INFO: building parameterized compute graph...\n");
                    graph->parameterized_reward_compute_graph =
                            ptd_graph_ex_absorbation_time_comp_graph_parameterized(graph);
                }

                if (graph->reward_compute_graph != NULL) {
                    free(graph->reward_compute_graph->commands);
                    free(graph->reward_compute_graph);
                }

                DEBUG_PRINT("INFO: building reward compute graph from parameterized compute graph...\n");
                graph->reward_compute_graph =
                        ptd_graph_build_ex_absorbation_time_comp_graph_parameterized(
                                graph->parameterized_reward_compute_graph
                        );
            }
        } else {
            // DEBUG_PRINT("INFO: building reward compute graph...\n");
            graph->reward_compute_graph = ptd_graph_ex_absorbation_time_comp_graph(graph);

            if (graph->reward_compute_graph == NULL) {
                return -1;
            }
        }
    }

    return 0;
}


static struct ptd_stack *scc_stack2 = NULL;
static struct ptd_vector *scc_components2 = NULL;
static size_t scc_index2 = 0;
static size_t *scc_indices2 = NULL;
static size_t *low_links2 = NULL;
static bool *scc_on_stack2 = NULL;
static bool *visited = NULL;

static int strongconnect2(struct ptd_vertex *vertex) {
    scc_indices2[vertex->index] = scc_index2;
    low_links2[vertex->index] = scc_index2;
    visited[vertex->index] = true;
    scc_index2++;
    stack_push(scc_stack2, vertex);
    scc_on_stack2[vertex->index] = true;

    for (size_t i = 0; i < vertex->edges_length; ++i) {
        struct ptd_edge *edge = vertex->edges[i];

        if (!visited[edge->to->index]) {
            int res = strongconnect2(edge->to);

            if (res != 0) {
                return res;
            }

            low_links2[vertex->index] = _ptd_min(
                                                low_links2[vertex->index],
                                                low_links2[edge->to->index]
                                        );
        } else if (scc_on_stack2[edge->to->index]) {
            low_links2[vertex->index] = _ptd_min(
                                                low_links2[vertex->index],
                                                scc_indices2[edge->to->index]
                                        );
        }
    }

    if (low_links2[vertex->index] == scc_indices2[vertex->index]) {
        struct ptd_vertex *w;
        struct ptd_vector *list = vector_create();

        do {
            if (stack_empty(scc_stack2)) {
                DIE_ERROR(1, "Stack is empty.\n");
            }
            w = (struct ptd_vertex *) stack_pop(scc_stack2);
            scc_on_stack2[w->index] = false;

            vector_add(list, w);
        } while (w != vertex);

        struct ptd_scc_vertex *scc = (struct ptd_scc_vertex *) malloc(sizeof(*scc));

        if (scc == NULL) {
            return -1;
        }

        scc->internal_vertices_length = vector_length(list);
        scc->internal_vertices = (struct ptd_vertex **) calloc(
                scc->internal_vertices_length,
                sizeof(*(scc->internal_vertices))
        );

        for (size_t i = 0; i < scc->internal_vertices_length; ++i) {
            scc->internal_vertices[i] = (struct ptd_vertex *) vector_get(list, i);
        }

        vector_add(scc_components2, scc);
        vector_destroy(list);
    }

    return 0;
}

static int scc_edge_cmp(const void *a, const void *b) {
    struct ptd_scc_edge *ea = *((struct ptd_scc_edge **) a);
    struct ptd_scc_edge *eb = *((struct ptd_scc_edge **) b);

    if (ea->to->index < eb->to->index) {
        return -1;
    } else if (ea->to->index > eb->to->index) {
        return 1;
    } else {
        return 0;
    }
}

static struct ptd_scc_vertex *single_vertex_as_scc(struct ptd_vertex *vertex) {
    struct ptd_scc_vertex* r = (struct ptd_scc_vertex*) malloc(sizeof(*r));

    r->index = vertex->index;
    r->internal_vertices_length = 1;
    r->internal_vertices = (struct ptd_vertex**) malloc(sizeof(struct ptd_vertex*));
    r->internal_vertices[0] = vertex;
    r->edges_length = vertex->edges_length;
    if (vertex->edges_length != 0) {
        r->edges = (struct ptd_scc_edge**) calloc(
                r->edges_length,
                sizeof(struct ptd_scc_edge*)
        );
    } else {
        r->edges = NULL;
    }

    return r;
}

static struct ptd_scc_graph *ptd_find_strongly_connected_components_acyclic(struct ptd_graph *graph) {
    struct ptd_scc_graph *scc_graph = (struct ptd_scc_graph *) malloc(
            sizeof(*scc_graph)
    );

    scc_graph->graph = graph;

    scc_graph->vertices = (struct ptd_scc_vertex **) calloc(
            graph->vertices_length,
            sizeof(struct ptd_scc_vertex *)
    );

    scc_graph->vertices_length = graph->vertices_length;

    for (size_t i = 0; i < graph->vertices_length; i++) {
        scc_graph->vertices[i] = single_vertex_as_scc(graph->vertices[i]);
    }

    for (size_t i = 0; i < graph->vertices_length; i++) {
        for (size_t j = 0; j < graph->vertices[i]->edges_length; j++) {
            size_t to_index = graph->vertices[i]->edges[j]->to->index;
            scc_graph->vertices[i]->edges[j] = (struct ptd_scc_edge*) malloc(sizeof(struct ptd_scc_edge));

            scc_graph->vertices[i]->edges[j]->to = scc_graph->vertices[to_index];
        }
    }

    return scc_graph;
}

struct ptd_scc_graph *ptd_find_strongly_connected_components(struct ptd_graph *graph) {
    if (ptd_graph_is_acyclic(graph)) {
        return ptd_find_strongly_connected_components_acyclic(graph);
    }

    struct ptd_scc_graph *scc_graph = (struct ptd_scc_graph *) malloc(
            sizeof(*scc_graph)
    );

    scc_graph->graph = graph;

    scc_stack2 = NULL;
    scc_components2 = NULL;
    scc_index2 = 0;
    scc_indices2 = NULL;
    low_links2 = NULL;
    scc_on_stack2 = NULL;
    visited = NULL;

    scc_stack2 = stack_create();

    scc_index2 = 0;
    scc_indices2 = (size_t *) calloc(graph->vertices_length * 10, sizeof(size_t));
    low_links2 = (size_t *) calloc(graph->vertices_length * 10, sizeof(size_t));
    scc_on_stack2 = (bool *) calloc(graph->vertices_length * 10, sizeof(bool));
    visited = (bool *) calloc(graph->vertices_length * 10, sizeof(bool));
    scc_components2 = vector_create();

    for (size_t i = 0; i < graph->vertices_length; ++i) {
        struct ptd_vertex *vertex = graph->vertices[i];

        if (!visited[i]) {
            if (strongconnect2(vertex) != 0) {
                return NULL;
            }
        }
    }

    size_t non_empty_components = 0;

    for (size_t i = 0; i < vector_length(scc_components2); ++i) {
        struct ptd_scc_vertex *c =
                (struct ptd_scc_vertex *) vector_get(scc_components2, i);

        if (c->internal_vertices_length != 0) {
            non_empty_components++;
        }
    }

    scc_graph->vertices_length = non_empty_components;
    scc_graph->vertices = (struct ptd_scc_vertex **) calloc(
            scc_graph->vertices_length,
            sizeof(*(scc_graph->vertices))
    );

    size_t index = 0;

    for (size_t i = 0; i < scc_graph->vertices_length; ++i) {
        struct ptd_scc_vertex *scc =
                (struct ptd_scc_vertex *) vector_get(scc_components2, i);

        if (scc->internal_vertices_length != 0) {
            scc_graph->vertices[index] =
                    (struct ptd_scc_vertex *) vector_get(scc_components2, i);
            scc_graph->vertices[index]->index = index;
            index++;
        } else {
            free(scc->internal_vertices);
            free(scc);
        }
    }

    struct ptd_scc_vertex **sccs_for_vertices = (struct ptd_scc_vertex **) calloc(
            graph->vertices_length,
            sizeof(*sccs_for_vertices)
    );

    for (size_t i = 0; i < scc_graph->vertices_length; ++i) {
        struct ptd_scc_vertex *scc = scc_graph->vertices[i];
        scc->index = i;

        for (size_t j = 0; j < scc->internal_vertices_length; ++j) {
            struct ptd_vertex *vertex = scc->internal_vertices[j];

            sccs_for_vertices[vertex->index] = scc;
        }
    }

    scc_graph->starting_vertex = sccs_for_vertices[graph->starting_vertex->index];

    for (size_t i = 0; i < scc_graph->vertices_length; ++i) {
        struct ptd_scc_vertex *scc = scc_graph->vertices[i];
        struct ptd_avl_tree *external_sccs = ptd_avl_tree_create(1);

        for (size_t j = 0; j < scc->internal_vertices_length; ++j) {
            struct ptd_vertex *vertex = scc->internal_vertices[j];

            for (size_t k = 0; k < vertex->edges_length; ++k) {
                struct ptd_vertex *child = vertex->edges[k]->to;
                struct ptd_scc_vertex *child_scc = sccs_for_vertices[child->index];

                if (child_scc != scc) {
                    ptd_avl_tree_find_or_insert(external_sccs, (int *) &(child_scc->index), child_scc);
                }
            }
        }

        struct ptd_vector *external_sccs_vector = vector_create();
        struct ptd_stack *tree_stack;
        tree_stack = stack_create();

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

        scc->edges_length = vector_length(external_sccs_vector);
        scc->edges = (struct ptd_scc_edge **) calloc(
                scc->edges_length,
                sizeof(*(scc->edges))
        );

        size_t set_index;

        set_index = 0;

        for (size_t l = 0; l < vector_length(external_sccs_vector); ++l) {
            scc->edges[set_index] = (struct ptd_scc_edge *) malloc(sizeof(*(scc->edges[set_index])));
            scc->edges[set_index]->to = (struct ptd_scc_vertex *) vector_get(external_sccs_vector, l);
            set_index++;
        }

        qsort(scc->edges, scc->edges_length, sizeof(*(scc->edges)), scc_edge_cmp);

        vector_destroy(external_sccs_vector);
        stack_destroy(tree_stack);
        ptd_avl_tree_destroy(external_sccs);
    }

    free(scc_indices2);
    free(low_links2);
    free(scc_on_stack2);
    free(visited);
    vector_destroy(scc_components2);
    stack_destroy(scc_stack2);

    free(sccs_for_vertices);


    scc_stack2 = NULL;
    scc_components2 = NULL;
    scc_index2 = 0;
    scc_indices2 = NULL;
    low_links2 = NULL;
    scc_on_stack2 = NULL;
    visited = NULL;

    return scc_graph;
}

void ptd_scc_graph_destroy(struct ptd_scc_graph *scc_graph) {
    if (scc_graph == NULL) {
        return;
    }

    for (size_t i = 0; i < scc_graph->vertices_length; ++i) {
        struct ptd_scc_vertex *scc = scc_graph->vertices[i];

        for (size_t j = 0; j < scc->edges_length; ++j) {
            free(scc->edges[j]);
        }


        free(scc->edges);
        free(scc->internal_vertices);
        free(scc);
    }

    free(scc_graph->vertices);
    free(scc_graph);
}

double *ptd_normalize_graph(struct ptd_graph *graph) {
    double *res = (double *) calloc(graph->vertices_length, sizeof(*res));

    for (size_t i = 0; i < graph->vertices_length; ++i) {
        struct ptd_vertex *vertex = graph->vertices[i];
        double rate = 0;

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            rate += vertex->edges[j]->weight;
        }

        if (rate == 0) {
            res[i] = 1.0;
        } else {
            res[i] = 1.0 / rate;
        }

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            vertex->edges[j]->weight /= rate;
        }
    }

    return res;
}

double *ptd_dph_normalize_graph(struct ptd_graph *graph) {
    size_t old_length = graph->vertices_length;
    double *res = (double *) calloc(old_length * 2, sizeof(*res));

    for (size_t i = 0; i < old_length; ++i) {
        res[i] = 1;

        struct ptd_vertex *vertex = graph->vertices[i];
        double rate = 0;

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            rate += vertex->edges[j]->weight;
        }

        if (rate == 0 || graph->starting_vertex == vertex) {
            continue;
        }

        if (1 - rate > 0.0000001) {
            struct ptd_vertex *auxiliary_vertex = ptd_vertex_create(graph);
            ptd_graph_add_edge(vertex, auxiliary_vertex, 1 - rate);
            ptd_graph_add_edge(auxiliary_vertex, vertex, 1);
            res[auxiliary_vertex->index] = 0;
        }
    }

    return res;
}

struct ptd_phase_type_distribution *ptd_graph_as_phase_type_distribution(struct ptd_graph *graph) {
    struct ptd_phase_type_distribution *res = (struct ptd_phase_type_distribution *) malloc(sizeof(*res));

    if (res == NULL) {
        return NULL;
    }

    res->length = 0;

    size_t size = graph->vertices_length;

    res->memory_allocated = size;
    res->vertices = (struct ptd_vertex **) calloc(size, sizeof(struct ptd_vertex *));

    if (res->vertices == NULL) {
        free(res);
        return NULL;
    }

    res->initial_probability_vector = (double *) calloc(size, sizeof(double));

    if (res->initial_probability_vector == NULL) {
        free(res->vertices);
        free(res);
        return NULL;
    }

    res->sub_intensity_matrix = (double **) calloc(size, sizeof(double *));

    if (res->sub_intensity_matrix == NULL) {
        free(res->initial_probability_vector);
        free(res->vertices);
        free(res);
        return NULL;
    }

    for (size_t i = 0; i < size; ++i) {
        res->sub_intensity_matrix[i] = (double *) calloc(size, sizeof(double));

        if ((res->sub_intensity_matrix)[i] == NULL) {
            for (size_t j = 0; j < i; ++j) {
                free(res->sub_intensity_matrix[j]);
            }

            free(res->sub_intensity_matrix);
            free(res->initial_probability_vector);
            free(res->vertices);
            free(res);
            return NULL;
        }
    }

    struct ptd_scc_graph *scc = ptd_find_strongly_connected_components(graph);
    struct ptd_vertex **vertices =
            (struct ptd_vertex **) calloc(graph->vertices_length, sizeof(*vertices));
    struct ptd_scc_vertex **v = ptd_scc_graph_topological_sort(scc);
    size_t idx = 0;

    for (size_t i = 0; i < scc->vertices_length; ++i) {
        for (size_t j = 0; j < v[i]->internal_vertices_length; ++j) {
            vertices[idx] = v[i]->internal_vertices[j];
            vertices[idx]->index = idx;
            idx++;
        }
    }

    size_t *indices = (size_t *) calloc(size, sizeof(*indices));
    size_t index = 0;

    for (size_t k = 0; k < graph->vertices_length; ++k) {
        struct ptd_vertex *vertex = vertices[k];

        if (graph->starting_vertex != vertex && vertex->edges_length != 0) {
            indices[vertex->index] = index;
            res->vertices[index] = vertex;
            index++;
        }
    }

    res->length = index;

    for (size_t k = 0; k < graph->vertices_length; ++k) {
        struct ptd_vertex *vertex = vertices[k];

        if (vertex->edges_length == 0) {
            continue;
        }

        if (vertex == graph->starting_vertex) {
            double rate = 0;

            for (size_t i = 0; i < vertex->edges_length; ++i) {
                struct ptd_edge *edge = vertex->edges[i];

                rate += edge->weight;
            }

            for (size_t i = 0; i < vertex->edges_length; ++i) {
                struct ptd_edge *edge = vertex->edges[i];

                if (edge->to->edges_length != 0) {
                    res->initial_probability_vector[indices[edge->to->index]] = edge->weight / rate;
                }
            }

            continue;
        }

        for (size_t i = 0; i < vertex->edges_length; ++i) {
            struct ptd_edge *edge = vertex->edges[i];

            if (edge->to->edges_length != 0) {
                res->sub_intensity_matrix[indices[vertex->index]][indices[edge->to->index]] += edge->weight;
            }

            res->sub_intensity_matrix[indices[vertex->index]][indices[vertex->index]] -= edge->weight;
        }
    }

    for (size_t i = 0; i < graph->vertices_length; ++i) {
        graph->vertices[i]->index = i;
    }

    free(v);
    ptd_scc_graph_destroy(scc);
    free(indices);
    free(vertices);

    return res;
}

void ptd_phase_type_distribution_destroy(struct ptd_phase_type_distribution *ptd) {
    for (size_t i = 0; i < ptd->memory_allocated; ++i) {
        free(ptd->sub_intensity_matrix[i]);

        ptd->sub_intensity_matrix[i] = NULL;
    }

    free(ptd->vertices);
    free(ptd->sub_intensity_matrix);
    free(ptd->initial_probability_vector);

    ptd->vertices = NULL;
    ptd->sub_intensity_matrix = NULL;
    ptd->initial_probability_vector = NULL;

    ptd->memory_allocated = 0;
    ptd->length = 0;

    free(ptd);
}

int ptd_vertex_to_s(struct ptd_vertex *vertex, char *buffer, size_t buffer_length) {
    memset(buffer, '\0', buffer_length);

    char *build = (char *) calloc(buffer_length, sizeof(char));

    for (size_t i = 0; i < vertex->graph->state_length; ++i) {
        if (i == 0) {
            snprintf(build, buffer_length, "%s%i", buffer, vertex->state[i]);
        } else {
            snprintf(build, buffer_length, "%s %i", buffer, vertex->state[i]);
        }

        strncpy(buffer, build, buffer_length);
    }

    free(build);

    return 0;
}

void ptd_directed_graph_destroy(struct ptd_directed_graph *graph) {
    for (size_t i = 0; i < graph->vertices_length; ++i) {
        ptd_directed_vertex_destroy(graph->vertices[i]);
    }

    free(graph->vertices);
    graph->vertices = NULL;
    free(graph);
}

int ptd_directed_vertex_add(struct ptd_directed_graph *graph, struct ptd_directed_vertex *vertex) {
    bool is_power_of_2 = (graph->vertices_length & (graph->vertices_length - 1)) == 0;

    if (is_power_of_2) {
        size_t new_length = graph->vertices_length == 0 ? 1 : graph->vertices_length * 2;

        if ((graph->vertices = (struct ptd_directed_vertex **) realloc(
                graph->vertices, new_length *
                                 sizeof(struct ptd_directed_vertex *))
            ) == NULL) {
            return -1;
        }
    }

    vertex->graph = graph;

    graph->vertices[graph->vertices_length] = vertex;
    vertex->index = graph->vertices_length;
    graph->vertices_length++;

    return 0;
}

int ptd_directed_graph_add_edge(struct ptd_directed_vertex *vertex, struct ptd_directed_edge *edge) {
    bool is_power_of_2 = (vertex->edges_length & (vertex->edges_length - 1)) == 0;

    if (is_power_of_2) {
        size_t new_length = vertex->edges_length == 0 ? 1 : vertex->edges_length * 2;

        if ((vertex->edges = (struct ptd_directed_edge **) realloc(
                vertex->edges,
                new_length * sizeof(struct ptd_directed_edge *))
            ) == NULL) {
            return -1;
        }
    }

    vertex->edges[vertex->edges_length] = edge;
    vertex->edges_length++;

    return 0;
}

void ptd_directed_vertex_destroy(struct ptd_directed_vertex *vertex) {
    for (size_t i = 0; i < vertex->edges_length; ++i) {
        free(vertex->edges[i]);
    }

    free(vertex->edges);
    vertex->edges = NULL;
    free(vertex);
}

struct ptd_graph *ptd_graph_create(size_t state_length) {
    struct ptd_graph *graph = (struct ptd_graph *) malloc(sizeof(*graph));
    graph->vertices_length = 0;
    graph->state_length = state_length;
    graph->param_length = 0;  // Will be set when first parameterized edge is added
    graph->vertices = NULL;
    graph->reward_compute_graph = NULL;
    graph->parameterized_reward_compute_graph = NULL;
    graph->starting_vertex = ptd_vertex_create(graph);
    graph->parameterized = false;
    graph->was_dph = false;
    graph->elimination_trace = NULL;
    graph->current_params = NULL;

    return graph;
}

void ptd_parameterized_reward_compute_graph_destroy(
        struct ptd_desc_reward_compute_parameterized *compute_graph
) {
    struct ll_of_a *mem = (struct ll_of_a *) compute_graph->mem;

    while (mem != NULL) {
        struct ll_of_a *memp = mem;
        mem = mem->next;
        free(memp->mem);
        free(memp);
    }

    free(compute_graph->memr);
    free(compute_graph->commands);
    free(compute_graph);
}

void ptd_graph_destroy(struct ptd_graph *graph) {
    for (size_t i = 0; i < graph->vertices_length; ++i) {
        ptd_vertex_destroy(graph->vertices[i]);
    }

    free(graph->vertices);

    if (graph->reward_compute_graph != NULL) {
        free(graph->reward_compute_graph->commands);
        free(graph->reward_compute_graph);
    }

    if (graph->parameterized_reward_compute_graph != NULL) {
        ptd_parameterized_reward_compute_graph_destroy(
                graph->parameterized_reward_compute_graph
        );
    }

    if (graph->elimination_trace != NULL) {
        ptd_elimination_trace_destroy(graph->elimination_trace);
    }

    if (graph->current_params != NULL) {
        free(graph->current_params);
    }

    graph->reward_compute_graph = NULL;
    graph->parameterized_reward_compute_graph = NULL;
    graph->elimination_trace = NULL;
    graph->current_params = NULL;
    memset(graph, 0, sizeof(*graph));
    free(graph);
}

struct ptd_vertex *ptd_vertex_create(struct ptd_graph *graph) {
    int *state = (int *) calloc(graph->state_length, sizeof(*state));

    return ptd_vertex_create_state(graph, state);
}

struct ptd_vertex *ptd_vertex_create_state(struct ptd_graph *graph, int *state) {
    struct ptd_vertex *vertex = (struct ptd_vertex *) malloc(sizeof(*vertex));
    vertex->graph = graph;
    vertex->edges_length = 0;
    vertex->state = state;
    vertex->edges = NULL;
    ptd_directed_vertex_add(
            (struct ptd_directed_graph *) graph,
            (struct ptd_directed_vertex *) vertex
    );

    return vertex;
}

double ptd_vertex_rate(struct ptd_vertex *vertex) {
    double rate = 0;

    for (size_t i = 0; i < vertex->edges_length; ++i) {
        rate += vertex->edges[i]->weight;
    }

    return rate;
}

void ptd_vertex_destroy(struct ptd_vertex *vertex) {
    for (size_t i = 0; i < vertex->edges_length; ++i) {
        if (vertex->edges[i]->parameterized) {
            if (((struct ptd_edge_parameterized *) vertex->edges[i])->should_free_state) {
                free(((struct ptd_edge_parameterized *) vertex->edges[i])->state);
            }
        }

        free(vertex->edges[i]);
    }

    free(vertex->edges);
    free(vertex->state);
    memset(vertex, 0, sizeof(*vertex));
    free(vertex);
}


static inline int edge_cmp(const void *a, const void *b) {
    if ((*((struct ptd_edge **) a))->to < (*((struct ptd_edge **) b))->to) {
        return -1;
    } else {
        return 1;
    }
}

int ptd_validate_graph(const struct ptd_graph *graph) {
    struct ptd_edge **edges_buffer = (struct ptd_edge **) calloc(graph->vertices_length, sizeof(*edges_buffer));

    for (size_t i = 0; i < graph->vertices_length; ++i) {
        struct ptd_vertex *vertex = graph->vertices[i];

        if (vertex->edges_length >= graph->vertices_length) {
            // Definitely have a problem...
            edges_buffer = (struct ptd_edge **) realloc(edges_buffer, vertex->edges_length * sizeof(*edges_buffer));
        }

        memcpy(edges_buffer, vertex->edges, vertex->edges_length * sizeof(*edges_buffer));
        qsort(edges_buffer, vertex->edges_length, sizeof(*edges_buffer), edge_cmp);

        for (size_t j = 1; j < vertex->edges_length; ++j) {
            if (vertex->edges[j]->to == vertex->edges[j - 1]->to) {
                struct ptd_vertex *from = vertex;
                struct ptd_vertex *to = vertex->edges[j]->to;
                size_t debug_index_from = from->index;
                size_t debug_index_to = to->index;

                if (PTD_DEBUG_1_INDEX) {
                    debug_index_from++;
                    debug_index_to++;
                }

                char state[1024] = {'\0'};
                char state_to[1024] = {'\0'};
                char starting_vertex[] = " (starting vertex)";

                if (from != from->graph->starting_vertex) {
                    starting_vertex[0] = '\0';
                }

                ptd_vertex_to_s(from, state, 1023);
                ptd_vertex_to_s(to, state_to, 1023);

                snprintf(
                        (char *) ptd_err,
                        sizeof(ptd_err),
                        "Multiple edges to the same vertex!. From vertex with index %i%s (state %s)."
                        " To vertex with index %i (state %s)\n",
                        (int) debug_index_from, starting_vertex, state,
                        (int) debug_index_to, state_to
                );

                free(edges_buffer);
                return 1;
            }
        }
    }

    free(edges_buffer);

    return 0;
}

struct ptd_edge *ptd_graph_add_edge(
        struct ptd_vertex *from,
        struct ptd_vertex *to,
        double weight
) {
    if (weight < 0) {
        size_t debug_index = from->index;

        if (PTD_DEBUG_1_INDEX) {
            debug_index++;
        }

        char state[1024] = {'\0'};
        char starting_vertex[] = " (starting vertex)";

        if (from != from->graph->starting_vertex) {
            starting_vertex[0] = '\0';
        }

        ptd_vertex_to_s(from, state, 1023);

        snprintf(
                (char *) ptd_err,
                sizeof(ptd_err),
                "Tried to add edge with non-positive weight '%f'. Vertex index %i%s (state %s). Weight must be strictly larger than 0.\n",
                weight, (int) debug_index, starting_vertex, state
        );

        return NULL;
    }

    if (from == to) {
        size_t debug_index = from->index;

        if (PTD_DEBUG_1_INDEX) {
            debug_index++;
        }

        char state[1024] = {'\0'};
        char starting_vertex[] = " (starting vertex)";

        if (from != from->graph->starting_vertex) {
            starting_vertex[0] = '\0';
        }

        ptd_vertex_to_s(from, state, 1023);

        snprintf(
                (char *) ptd_err,
                sizeof(ptd_err),
                "Tried to add edge to itself. Vertex index %i%s (state %s). Self-loops are not allowed, discrete self-loops are set as the missing out-going weight.\n",
                (int) debug_index, starting_vertex, state
        );

        return NULL;
    }

    /*for (size_t i = 0; i < from->edges_length; ++i) {
        if (from->edges[i]->to == to) {
            size_t debug_index = from->index;

            if (PTD_DEBUG_1_INDEX) {
                debug_index++;
            }
            size_t debug_index_to = to->index;

            if (PTD_DEBUG_1_INDEX) {
                debug_index_to++;
            }

            char state[1024] = {'\0'};
            char starting_vertex[] = " (starting vertex)";

            if (from != from->graph->starting_vertex) {
                starting_vertex[0] = '\0';
            }

            ptd_vertex_to_s(from, state, 1023);

            char state_to[1024] = {'\0'};

            ptd_vertex_to_s(to, state_to, 1023);

            snprintf(
                    (char *) ptd_err,
                    sizeof(ptd_err),
                    "Tried to add to a vertex with an already existing edge. Vertex index %i%s (state %s), to %i (state %s).\n",
                    (int) debug_index, starting_vertex, state, (int) debug_index_to, state_to
            );

            return NULL;
        }
    }*/

    struct ptd_edge *edge = (struct ptd_edge *) malloc(sizeof(*edge));

    edge->to = to;
    edge->weight = weight;
    edge->parameterized = false;

    ptd_directed_graph_add_edge(
            (struct ptd_directed_vertex *) from,
            (struct ptd_directed_edge *) edge
    );

    if (from->graph->reward_compute_graph != NULL) {
        free(from->graph->reward_compute_graph->commands);
        free(from->graph->reward_compute_graph);
    }

    if (from->graph->parameterized_reward_compute_graph != NULL) {
        ptd_parameterized_reward_compute_graph_destroy(
                from->graph->parameterized_reward_compute_graph
        );
    }

    from->graph->reward_compute_graph = NULL;
    from->graph->parameterized_reward_compute_graph = NULL;

    return edge;
}

struct ptd_edge_parameterized *ptd_graph_add_edge_parameterized(
        struct ptd_vertex *from,
        struct ptd_vertex *to,
        double weight,
        double *edge_state,
        size_t edge_state_length
) {
    from->graph->parameterized = true;

    struct ptd_edge_parameterized *edge = (struct ptd_edge_parameterized *) malloc(sizeof(*edge));

    edge->to = to;
    edge->weight = weight;
    edge->base_weight = weight;  // Store original base weight for gradient computation
    edge->parameterized = true;
    edge->state = edge_state;
    edge->state_length = edge_state_length;  // Store the actual allocated length
    edge->should_free_state = true;

    ptd_directed_graph_add_edge(
            (struct ptd_directed_vertex *) from,
            (struct ptd_directed_edge *) edge
    );

    if (from->graph->reward_compute_graph != NULL) {
        free(from->graph->reward_compute_graph->commands);
        free(from->graph->reward_compute_graph);
    }

    if (from->graph->parameterized_reward_compute_graph != NULL) {
        ptd_parameterized_reward_compute_graph_destroy(
                from->graph->parameterized_reward_compute_graph
        );
    }

    from->graph->reward_compute_graph = NULL;
    from->graph->parameterized_reward_compute_graph = NULL;

    return edge;
}

void ptd_notify_change(
        struct ptd_graph *graph
) {
    if (graph->reward_compute_graph != NULL) {
        free(graph->reward_compute_graph->commands);
        free(graph->reward_compute_graph);
        graph->reward_compute_graph = NULL;
    }
}

void ptd_edge_update_weight(
        struct ptd_edge *edge,
        double weight
) {
    edge->weight = weight;

    if (edge->to->graph->reward_compute_graph != NULL) {
        free(edge->to->graph->reward_compute_graph->commands);
        edge->to->graph->reward_compute_graph = NULL;
    }
}

void ptd_edge_update_to(
    struct ptd_edge *edge,
    struct ptd_vertex *vertex
) {

if (edge->to->graph->reward_compute_graph != NULL) {
    free(edge->to->graph->reward_compute_graph->commands);
    edge->to->graph->reward_compute_graph = NULL;
}

edge->to = vertex;

}

void ptd_edge_update_weight_parameterized(
        struct ptd_edge *edge,
        double *scalars,
        size_t scalars_length
) {
    double weight = 0;

    for (size_t i = 0; i < scalars_length; ++i) {
        weight += scalars[i] * ((struct ptd_edge_parameterized *) edge)->state[i];
    }

    edge->weight = weight;

    // Invalidate both regular and parameterized compute graphs
    if (edge->to->graph->reward_compute_graph != NULL) {
        free(edge->to->graph->reward_compute_graph->commands);
        free(edge->to->graph->reward_compute_graph);
        edge->to->graph->reward_compute_graph = NULL;
    }

    if (edge->to->graph->parameterized_reward_compute_graph != NULL) {
        ptd_parameterized_reward_compute_graph_destroy(
                edge->to->graph->parameterized_reward_compute_graph
        );
        edge->to->graph->parameterized_reward_compute_graph = NULL;
    }
}

void ptd_graph_update_weight_parameterized(
        struct ptd_graph *graph,
        double *scalars,
        size_t scalars_length
) {
    // Store parameter length on first call
    if (graph->param_length == 0 && scalars_length > 0) {
        graph->param_length = scalars_length;
    }

    // Store current parameters for trace evaluation
    if (graph->current_params == NULL && scalars_length > 0) {
        graph->current_params = (double *) malloc(scalars_length * sizeof(double));
    }
    if (graph->current_params != NULL && scalars_length > 0) {
        memcpy(graph->current_params, scalars, scalars_length * sizeof(double));
    }

    // Record trace on first call (if parameterized graph and not yet recorded)
    if (graph->parameterized && graph->elimination_trace == NULL) {
        // Compute graph hash for cache lookup
        struct ptd_hash_result *hash = ptd_graph_content_hash(graph);

        if (hash != NULL) {
            // Try to load from cache
            graph->elimination_trace = load_trace_from_cache(hash->hash_hex);

            if (graph->elimination_trace != NULL) {
                DEBUG_PRINT("INFO: loaded elimination trace from cache (%s)\n", hash->hash_hex);
            }
        }

        // Cache miss or hash failed - record trace
        if (graph->elimination_trace == NULL) {
            DEBUG_PRINT("INFO: recording elimination trace for parameterized graph...\n");
            graph->elimination_trace = ptd_record_elimination_trace(graph);

            if (graph->elimination_trace == NULL) {
                DEBUG_PRINT("WARNING: trace recording failed, falling back to traditional path\n");
            } else if (hash != NULL) {
                // Save newly recorded trace to cache
                save_trace_to_cache(hash->hash_hex, graph->elimination_trace);
            }
        }

        if (hash != NULL) {
            ptd_hash_destroy(hash);
        }
    }

    for (size_t i = 0; i < graph->vertices_length; ++i) {
        for (size_t j = 0; j < graph->vertices[i]->edges_length; ++j) {
            if (graph->vertices[i]->edges[j]->parameterized) {
                ptd_edge_update_weight_parameterized(
                        graph->vertices[i]->edges[j], scalars, scalars_length
                );
            }
        }
    }

    // Invalidate cached compute graphs after updating all edge weights
    if (graph->reward_compute_graph != NULL) {
        free(graph->reward_compute_graph->commands);
        free(graph->reward_compute_graph);
    }

    if (graph->parameterized_reward_compute_graph != NULL) {
        ptd_parameterized_reward_compute_graph_destroy(
                graph->parameterized_reward_compute_graph
        );
    }

    graph->reward_compute_graph = NULL;
    graph->parameterized_reward_compute_graph = NULL;
}


struct ptd_directed_vertex **ptd_directed_graph_topological_sort(struct ptd_directed_graph *graph) {
    struct ptd_directed_vertex **res = (struct ptd_directed_vertex **) calloc(
            graph->vertices_length, sizeof(*res)
    );

    bool *visited = (bool *) calloc(graph->vertices_length, sizeof(*visited));
    size_t *nparents = (size_t *) calloc(graph->vertices_length, sizeof(*nparents));

    for (size_t i = 0; i < graph->vertices_length; ++i) {
        struct ptd_directed_vertex *vertex = graph->vertices[i];

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            struct ptd_directed_vertex *child = vertex->edges[j]->to;

            nparents[child->index]++;
        }
    }

    bool has_pushed_all_others = false;
    struct ptd_queue *q = queue_create();
    queue_enqueue(q, graph->vertices[0]);
    size_t topo_index = 0;

    while (!queue_empty(q)) {
        struct ptd_directed_vertex *vertex = (struct ptd_directed_vertex *) queue_dequeue(q);

        res[topo_index] = vertex;
        visited[vertex->index] = true;
        topo_index++;

        for (size_t i = 0; i < vertex->edges_length; ++i) {
            struct ptd_directed_vertex *child = vertex->edges[i]->to;

            nparents[child->index]--;

            if (nparents[child->index] == 0 && !visited[child->index]) {
                visited[child->index] = true;
                queue_enqueue(q, child);
            }
        }

        if (queue_empty(q) && !has_pushed_all_others) {
            for (size_t i = 0; i < graph->vertices_length; ++i) {
                struct ptd_directed_vertex *independent_vertex = graph->vertices[i];

                if (nparents[independent_vertex->index] == 0 && !visited[independent_vertex->index]) {
                    queue_enqueue(q, independent_vertex);
                }
            }

            has_pushed_all_others = true;
        }
    }

    for (size_t i = 0; i < graph->vertices_length; ++i) {
        if (nparents[i] != 0) {
            free(nparents);
            free(visited);
            free(res);
            queue_destroy(q);

            return NULL;
        }
    }

    free(nparents);
    free(visited);
    queue_destroy(q);

    return res;
}

bool ptd_graph_is_acyclic(struct ptd_graph *graph) {
    struct ptd_vertex **sorted = ptd_graph_topological_sort(graph);

    bool is_acyclic = (sorted != NULL);

    free(sorted);

    return is_acyclic;
}

struct ptd_vertex **ptd_graph_topological_sort(struct ptd_graph *graph) {
    return (struct ptd_vertex **) ptd_directed_graph_topological_sort((struct ptd_directed_graph *) graph);
}

struct ptd_scc_vertex **ptd_scc_graph_topological_sort(struct ptd_scc_graph *graph) {
    return (struct ptd_scc_vertex **) ptd_directed_graph_topological_sort((struct ptd_directed_graph *) graph);
}

struct ll_c {
    struct ll_c *next;
    struct ll_c *prev;
    double weight;
    struct ptd_vertex *c;
    struct ll_p *ll_p;
};

struct ll_p {
    struct ll_p *next;
    struct ll_p *prev;
    struct ptd_vertex *p;
    struct ll_c *ll_c;
};


struct ll_c2 {
    struct ll_c2 *next;
    struct ll_c2 *prev;
    double *weight;
    struct ptd_vertex *c;
    struct ll_p2 *ll_p;
};

struct ll_p2 {
    struct ll_p2 *next;
    struct ll_p2 *prev;
    struct ptd_vertex *p;
    struct ll_c2 *ll_c;
};

#define REWARD_EPSILON 0.000001

struct arr_c {
    double prob;
    struct ptd_vertex *to;
    size_t arr_p_index;
};

struct arr_p {
    struct ptd_vertex *p;
    size_t arr_c_index;
};

static inline int arr_c_cmp(const void *a, const void *b) {
    if ((*((struct arr_c *) a)).to < (*((struct arr_c *) b)).to) {
        return -1;
    } else {
        return 1;
    }
}

// struct ptd_clone_res _ptd_graph_expectation_dag(struct ptd_graph *graph, double *rewards) {

//     struct ptd_clone_res ret;
//     ret.graph = NULL;

//     if (ptd_precompute_reward_compute_graph(graph)) {
//         printf("Error in precomputing reward compute graph\n");
//         return ret;
//     }


//     double *dag_vertex_props = (double *) calloc(graph->vertices_length, sizeof(*dag_vertex_props));

//     if (rewards != NULL) {
//         // TODO: fix this if reward is nan...
//         memcpy(dag_vertex_props, rewards, sizeof(*dag_vertex_props) * graph->vertices_length);
//     } else {
//         for (size_t j = 0; j < graph->vertices_length; ++j) {
//             dag_vertex_props[j] = 1;
//         }
//     }

//     // we want only the acyclic graph so we we subtract graph->vertices_length to skip 
//     // the commands computing the expected waiting time
//     for (size_t j = 0; j < graph->reward_compute_graph->length - graph->vertices_length; ++j) {
//         struct ptd_reward_increase command = graph->reward_compute_graph->commands[j];
//         dag_vertex_props[command.from] += dag_vertex_props[command.to] * command.multiplier;
//         //TODO: if inf, give error stating that there is an infinite loop
//     }

//     // construct the acyclic graph
//     struct ptd_graph *dag = ptd_graph_create(graph->state_length);
//     struct ptd_avl_tree *dag_avl_tree = ptd_avl_tree_create(graph->state_length);
 
//     for (size_t j = 0; j < graph->starting_vertex->edges_length; ++j) {
//         ptd_graph_add_edge(dag->starting_vertex, 
//                             ptd_find_or_create_vertex(dag, dag_avl_tree, graph->starting_vertex->edges[j]->to->state), 
//                             graph->starting_vertex->edges[j]->weight);
//     }

//     // for (size_t j = 2; j < graph->vertices_length; ++j) {
//     //     struct ptd_reward_increase command = graph->reward_compute_graph->commands[graph->reward_compute_graph->length - j];
//     for (size_t j = 2; j < graph->vertices_length; ++j) {
//         struct ptd_reward_increase command = graph->reward_compute_graph->commands[graph->reward_compute_graph->length - j];

//         int idx = command.from;
//         int child_idx = command.to;
//         double child_prob = command.multiplier;

//         struct ptd_vertex *vertex = ptd_find_or_create_vertex(dag, dag_avl_tree, graph->vertices[idx]->state);
//         struct ptd_vertex *child_vertex = ptd_find_or_create_vertex(dag, dag_avl_tree, graph->vertices[child_idx]->state); 

//         // TODO: parametrization is meaningful here as DAG would need to be recomputed if rewards change
//         // maybe alert user that this is not supported

//         // if (e->parameterized) {
//         //     ptd_graph_add_edge_parameterized(
//         //             vertex,
//         //             child_vertex,
//         //             child_prob / dag_vertex_props[idx],
//         //             ((struct ptd_edge_parameterized *) e)->state
//         //     )->should_free_state = false;
//         // } else {
//             ptd_graph_add_edge(vertex, child_vertex, child_prob / dag_vertex_props[idx]);
//         // }

//     }

//     // TODO: make version for discrete graphs

//     free(dag_vertex_props);

//     ret.graph = dag;
//     ret.avl_tree = dag_avl_tree;
//     return ret;
// }

// struct ptd_clone_res ptd_graph_expectation_dag(struct ptd_graph *graph, double *rewards) {
//     if (ptd_validate_graph(graph)) {
//         struct ptd_clone_res res;
//         res.graph = NULL;
//         return res;
//     }

//     struct ptd_clone_res res = _ptd_graph_expectation_dag(graph, rewards);
//     return res;
// }

struct ptd_graph *_ptd_graph_reward_transform(struct ptd_graph *graph, double *__rewards, size_t **new_indices_r) {
    double *rewards = (double *) calloc(graph->vertices_length, sizeof(*rewards));

    struct ptd_vertex *dummy__ptd_min = (struct ptd_vertex *) 1, *dummy__ptd_max = 0;

    struct ptd_vertex **vertices = (struct ptd_vertex **) calloc(graph->vertices_length, sizeof(*vertices));
    size_t *original_indices = (size_t *) calloc(graph->vertices_length, sizeof(*original_indices));

    size_t vertices_length = graph->vertices_length;

    struct ptd_scc_graph *scc = ptd_find_strongly_connected_components(graph);
    struct ptd_scc_vertex **v = ptd_scc_graph_topological_sort(scc);

    size_t idx = 0;

    for (size_t sii = 0; sii < scc->vertices_length; ++sii) {
        for (size_t j = 0; j < v[sii]->internal_vertices_length; ++j) {
            if (v[sii]->internal_vertices[j]->edges_length == 0) {
                continue;
            }

            original_indices[idx] = v[sii]->internal_vertices[j]->index;
            v[sii]->internal_vertices[j]->index = idx;
            vertices[idx] = v[sii]->internal_vertices[j];
            idx++;
        }
    }

    for (size_t sii = 0; sii < scc->vertices_length; ++sii) {
        for (size_t j = 0; j < v[sii]->internal_vertices_length; ++j) {
            if (v[sii]->internal_vertices[j]->edges_length != 0) {
                continue;
            }

            original_indices[idx] = v[sii]->internal_vertices[j]->index;
            v[sii]->internal_vertices[j]->index = idx;
            vertices[idx] = v[sii]->internal_vertices[j];
            idx++;
        }
    }

    for (size_t i = 0; i < vertices_length; ++i) {
        if (__rewards[original_indices[i]] <= REWARD_EPSILON) {
            rewards[i] = 0;
        } else {
            rewards[i] = __rewards[original_indices[i]];
        }

        if (graph->starting_vertex == vertices[i] || vertices[i]->edges_length == 0) {
            rewards[i] = 1;
        }
    }

    struct arr_p **vertex_parents;
    size_t *vertex_parents_length;
    struct arr_c **vertex_edges;
    size_t *vertex_edges_length;
    double *old_rates = (double *) calloc(vertices_length, sizeof(*old_rates));

    for (size_t i = 0; i < vertices_length; ++i) {
        struct ptd_vertex *vertex = vertices[i];

        if (vertex >= dummy__ptd_max) {
            dummy__ptd_max = vertex + 1;
        }

        if (vertex <= dummy__ptd_min) {
            dummy__ptd_min = vertex - 1;
        }

        double rate = 0;

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            rate += vertex->edges[j]->weight;
        }

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            vertex->edges[j]->weight /= rate;
        }

        if (rewards[i] != 0) {
            rewards[i] /= rate;
        }

        old_rates[i] = rate;
    }

    vertex_parents = (struct arr_p **) calloc(vertices_length, sizeof(*vertex_parents));
    vertex_parents_length = (size_t *) calloc(vertices_length, sizeof(*vertex_parents_length));
    size_t *vertex_parents_alloc_length = (size_t *) calloc(vertices_length, sizeof(*vertex_parents_alloc_length));
    vertex_edges = (struct arr_c **) calloc(vertices_length, sizeof(*vertex_edges));
    vertex_edges_length = (size_t *) calloc(vertices_length, sizeof(*vertex_edges_length));
    size_t *vertex_edges_alloc_length = (size_t *) calloc(vertices_length, sizeof(*vertex_edges_alloc_length));

    for (size_t i = 0; i < vertices_length; ++i) {
        vertex_edges_alloc_length[i] = 64;
        struct ptd_vertex *vertex = vertices[i];

        while (vertex->edges_length + 2 >= vertex_edges_alloc_length[i]) {
            vertex_edges_alloc_length[i] *= 2;
        }

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            vertex_parents_length[vertex->edges[j]->to->index]++;
        }

        vertex_edges[i] = (struct arr_c *) calloc(vertex_edges_alloc_length[i], sizeof(*(vertex_edges[i])));
        vertex_edges_length[i] = vertex->edges_length + 2;
    }

    for (size_t i = 0; i < vertices_length; ++i) {
        vertex_parents_alloc_length[i] = 64;

        while (vertex_parents_length[i] >= vertex_parents_alloc_length[i]) {
            vertex_parents_alloc_length[i] *= 2;
        }

        vertex_parents[i] = (struct arr_p *) calloc(vertex_parents_alloc_length[i], sizeof(*(vertex_parents[i])));
        vertex_parents_length[i] = 0;
    }

    for (size_t i = 0; i < vertices_length; ++i) {
        struct ptd_vertex *vertex = vertices[i];

        vertex_edges[i][0].to = dummy__ptd_min;
        vertex_edges[i][0].prob = 0;
        vertex_edges[i][0].arr_p_index = (unsigned int) ((int) -1);

        double rate = 0;

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            rate += vertex->edges[j]->weight;
        }

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            vertex_edges[i][j + 1].to = vertex->edges[j]->to;
            vertex_edges[i][j + 1].prob = vertex->edges[j]->weight / rate;
        }

        vertex_edges[i][vertex->edges_length + 1].prob = 0;
        vertex_edges[i][vertex->edges_length + 1].to = dummy__ptd_max;
        vertex_edges[i][vertex->edges_length + 1].arr_p_index = (unsigned int) ((int) -1);

        qsort(vertex_edges[i], vertex_edges_length[i], sizeof(*(vertex_edges[i])), arr_c_cmp);
    }


    for (size_t i = 0; i < vertices_length; ++i) {
        struct ptd_vertex *vertex = vertices[i];

        for (size_t j = 1; j < vertex_edges_length[i] - 1; ++j) {
            struct arr_c *child = &(vertex_edges[i][j]);
            size_t k = child->to->index;
            child->arr_p_index = vertex_parents_length[k];
            vertex_parents[k][vertex_parents_length[k]].p = vertex;
            vertex_parents[k][vertex_parents_length[k]].arr_c_index = j;
            vertex_parents_length[k]++;
        }
    }

    struct arr_c *old_edges_buffer =
            (struct arr_c *) calloc(vertices_length + 2, sizeof(*old_edges_buffer));

    for (size_t i = 0; i < vertices_length; ++i) {
        if (rewards[i] != 0) {
            continue;
        }

        struct ptd_vertex *me = vertices[i];
        struct arr_c *my_children = vertex_edges[i];
        size_t my_parents_length = vertex_parents_length[i];
        size_t my_edges_length = vertex_edges_length[i];


        for (size_t p = 0; p < my_parents_length; ++p) {
            struct arr_p me_to_parent = vertex_parents[i][p];
            struct ptd_vertex *parent_vertex = me_to_parent.p;

            size_t parent_vertex_index = parent_vertex->index;
            struct arr_c parent_to_me = vertex_edges[parent_vertex_index][me_to_parent.arr_c_index];

            size_t parent_edges_length = vertex_edges_length[parent_vertex_index];

            bool should_resize = false;
            size_t new_parent_edges_alloc_length = my_edges_length + parent_edges_length;

            while (new_parent_edges_alloc_length >= vertex_edges_alloc_length[parent_vertex_index]) {
                vertex_edges_alloc_length[parent_vertex_index] *= 2;
                should_resize = true;
            }

            if (should_resize) {
                vertex_edges[parent_vertex_index] = (struct arr_c *) realloc(
                        vertex_edges[parent_vertex_index],
                        vertex_edges_alloc_length[parent_vertex_index] * sizeof(*(vertex_edges[parent_vertex_index]))
                );
            }

            vertex_edges_length[parent_vertex_index] = 0;

            double parent_weight_to_me = parent_to_me.prob;
            double new_parent_total_prob = 0;

            memcpy(
                    old_edges_buffer, vertex_edges[parent_vertex_index],
                    sizeof(struct arr_c) * parent_edges_length
            );

            struct arr_c *new_parent_children = vertex_edges[parent_vertex_index];

            size_t child_index = 0;
            size_t parent_child_index = 0;

            while (child_index < my_edges_length || parent_child_index < parent_edges_length) {
                struct arr_c me_to_child = my_children[child_index];
                struct ptd_vertex *me_to_child_v = me_to_child.to;
                struct arr_c parent_to_child = old_edges_buffer[parent_child_index];
                struct ptd_vertex *parent_to_child_v = parent_to_child.to;
                double me_to_child_p = me_to_child.prob;

                if (me_to_child_v == parent_vertex) {
                    double prob = parent_weight_to_me * me_to_child_p;
                    rewards[parent_vertex_index] *= 1 / (1 - prob);

                    child_index++;
                    continue;
                }

                if (parent_to_child_v == me) {
                    parent_child_index++;
                    continue;
                }

                if (me_to_child_v == parent_to_child_v) {
                    new_parent_children[vertex_edges_length[parent_vertex_index]].to = parent_to_child_v;
                    new_parent_children[vertex_edges_length[parent_vertex_index]].prob =
                            parent_to_child.prob + me_to_child_p * parent_weight_to_me;

                    new_parent_children[vertex_edges_length[parent_vertex_index]].arr_p_index = parent_to_child.arr_p_index;

                    if (parent_to_child_v != dummy__ptd_min && parent_to_child_v != dummy__ptd_max) {
                        size_t current_parent_index = parent_to_child.arr_p_index;
                        vertex_parents[parent_to_child_v->index][current_parent_index].arr_c_index = vertex_edges_length[parent_vertex_index];

                    }

                    new_parent_total_prob += new_parent_children[vertex_edges_length[parent_vertex_index]].prob;
                    vertex_edges_length[parent_vertex_index]++;

                    child_index++;
                    parent_child_index++;
                } else if (me_to_child_v < parent_to_child_v) {
                    size_t child_parents_length = vertex_parents_length[me_to_child_v->index];

                    if (child_parents_length >= vertex_parents_alloc_length[me_to_child_v->index]) {
                        vertex_parents_alloc_length[me_to_child_v->index] *= 2;
                        vertex_parents[me_to_child_v->index] = (struct arr_p *) realloc(
                                vertex_parents[me_to_child_v->index],
                                vertex_parents_alloc_length[me_to_child_v->index] *
                                sizeof(*(vertex_parents[me_to_child_v->index]))
                        );
                    }

                    vertex_parents[me_to_child_v->index][child_parents_length].arr_c_index = vertex_edges_length[parent_vertex_index];
                    vertex_parents[me_to_child_v->index][child_parents_length].p = parent_vertex;

                    new_parent_children[vertex_edges_length[parent_vertex_index]].to = me_to_child_v;
                    new_parent_children[vertex_edges_length[parent_vertex_index]].prob =
                            me_to_child_p * parent_weight_to_me;
                    new_parent_children[vertex_edges_length[parent_vertex_index]].arr_p_index = child_parents_length;
                    new_parent_total_prob += me_to_child_p * parent_weight_to_me;

                    vertex_edges_length[parent_vertex_index]++;
                    vertex_parents_length[me_to_child_v->index]++;

                    child_index++;
                } else {
                    new_parent_children[vertex_edges_length[parent_vertex_index]] = parent_to_child;
                    vertex_parents[parent_to_child_v->index][parent_to_child.arr_p_index].arr_c_index = vertex_edges_length[parent_vertex_index];
                    new_parent_total_prob += parent_to_child.prob;
                    vertex_edges_length[parent_vertex_index]++;

                    parent_child_index++;
                }
            }


            // Make sure parent has rate of 1
            for (size_t j = 0; j < vertex_edges_length[parent_vertex_index]; ++j) {
                new_parent_children[j].prob /= new_parent_total_prob;
            }

            vertex_edges_length[parent_vertex_index] = vertex_edges_length[parent_vertex_index];
        }

        for (size_t j = 1; j < my_edges_length - 1; ++j) {
            struct arr_c me_to_child = my_children[j];
            struct ptd_vertex *me_to_child_v = me_to_child.to;
            size_t index_to_remove = me_to_child.arr_p_index;
            size_t index_to_move = vertex_parents_length[me_to_child_v->index] - 1;
            vertex_parents[me_to_child_v->index][index_to_remove] =
                    vertex_parents[me_to_child_v->index][index_to_move];
            vertex_parents_length[me_to_child_v->index]--;
            struct arr_p child_to_move_parent = vertex_parents[me_to_child_v->index][index_to_remove];
            vertex_edges[child_to_move_parent.p->index][child_to_move_parent.arr_c_index].arr_p_index = index_to_remove;
        }
    }

    struct ptd_graph *new_graph = ptd_graph_create(graph->state_length);
    size_t *new_indicesGtoN = (size_t *) calloc(vertices_length, sizeof(*new_indicesGtoN));
    size_t *new_indicesNtoG = (size_t *) calloc(vertices_length, sizeof(*new_indicesNtoG));
    size_t *new_indicesNtoO = (size_t *) calloc(vertices_length, sizeof(*new_indicesNtoO));
    new_indicesGtoN[graph->starting_vertex->index] = 0;
    new_indicesNtoG[0] = graph->starting_vertex->index;
    new_indicesNtoO[0] = 0;
    size_t new_idx = 1;
    memcpy(graph->starting_vertex->state, new_graph->starting_vertex->state, graph->state_length * sizeof(int));

    for (size_t i = 0; i < vertices_length; ++i) {
        if (vertices[i] == graph->starting_vertex) {
            continue;
        }

        if (rewards[i] == 0) {
            continue;
        }

        struct ptd_vertex *vertex = ptd_vertex_create(new_graph);
        memcpy(vertex->state, vertices[i]->state, graph->state_length * sizeof(int));
        new_indicesGtoN[i] = new_idx;
        new_indicesNtoG[new_idx] = i;
        new_indicesNtoO[new_idx] = original_indices[i];
        new_idx++;
    }

    for (size_t i = 0; i < vertices_length; ++i) {
        if (rewards[i] == 0) {
            continue;
        }

        for (size_t j = 1; j < vertex_edges_length[i] - 1; ++j) {
            ptd_graph_add_edge(
                    new_graph->vertices[new_indicesGtoN[i]],
                    new_graph->vertices[new_indicesGtoN[vertex_edges[i][j].to->index]],
                    vertex_edges[i][j].prob / rewards[i]
            );
        }
    }

    *(new_indices_r) = new_indicesNtoO;

    free(new_indicesGtoN);
    free(new_indicesNtoG);

    for (size_t i = 0; i < vertices_length; ++i) {
        struct ptd_vertex *vertex = vertices[i];

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            vertex->edges[j]->weight *= old_rates[i];
        }
    }

    for (size_t i = 0; i < vertices_length; ++i) {
        graph->vertices[i]->index = i;
    }

    for (size_t i = 0; i < vertices_length; ++i) {
        free(vertex_edges[i]);
        free(vertex_parents[i]);
    }

    free(old_rates);
    free(vertex_parents_length);
    free(vertex_parents_alloc_length);
    free(vertex_parents);
    free(vertex_edges);
    free(vertex_edges_length);
    free(vertex_edges_alloc_length);
    free(original_indices);
    free(vertices);
    free(old_edges_buffer);
    free(v);
    ptd_scc_graph_destroy(scc);
    free(rewards);


    return new_graph;
}

struct ptd_graph *ptd_graph_reward_transform(struct ptd_graph *graph, double *rewards) {
    if (ptd_validate_graph(graph)) {
        return NULL;
    }

    size_t *new_indices;
    struct ptd_graph *res = _ptd_graph_reward_transform(graph, rewards, &new_indices);

    free(new_indices);

    return res;
}

struct ptd_graph *ptd_graph_dph_reward_transform(struct ptd_graph *_graph, int *rewards) {
    if (ptd_validate_graph(_graph)) {
        return NULL;
    }

    for (size_t i = 0; i < _graph->vertices_length; ++i) {
        if (rewards[i] <= REWARD_EPSILON) {
            continue;
        }

        struct ptd_vertex *vertex = _graph->vertices[i];

        if (vertex->edges_length == 0) {
            continue;
        }
        if (vertex == _graph->starting_vertex) {
            continue;
        }

        double rate = 0;

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            rate += vertex->edges[j]->weight;
        }

        if (rate > 1.0001) {
            size_t debug_index = vertex->index;

            if (PTD_DEBUG_1_INDEX) {
                debug_index++;
            }

            char state[1024] = {'\0'};
            char starting_vertex[] = " (starting vertex)";

            if (vertex != _graph->starting_vertex) {
                starting_vertex[0] = '\0';
            }

            ptd_vertex_to_s(vertex, state, 1023);

            snprintf(
                    (char *) ptd_err,
                    sizeof(ptd_err),
                    "Expected vertex with index %i%s (state %s) to have outgoing rate <= 1. Is '%f'. Are you sure this is a discrete phase-type distribution?\n",
                    (int) debug_index, starting_vertex, state, (float) rate
            );

            return NULL;
        }
    }

    double *zero_rewards = (double *) calloc(_graph->vertices_length, sizeof(*zero_rewards));

    for (size_t i = 0; i < _graph->vertices_length; ++i) {
        if (rewards[i] == 0) {
            zero_rewards[i] = 0;
        } else {
            zero_rewards[i] = 1;
        }
    }

    zero_rewards[0] = 1;

    size_t *new_graph_indices;
    struct ptd_graph *graph = _ptd_graph_reward_transform(_graph, zero_rewards, &new_graph_indices);

    struct ptd_vertex **vertices = (struct ptd_vertex **) calloc(
            graph->vertices_length, sizeof(*vertices)
    );

    for (size_t i = 0; i < graph->vertices_length; ++i) {
        vertices[i] = graph->vertices[i];
    }

    free(zero_rewards);

    int *non_zero_rewards = (int *) calloc(
            graph->vertices_length, sizeof(*non_zero_rewards)
    );

    for (size_t i = 1; i < graph->vertices_length; ++i) {
        size_t old_index = new_graph_indices[i];

        non_zero_rewards[i] = rewards[old_index];
    }

    free(vertices);

    size_t old_length = graph->vertices_length;

    for (size_t i = 0; i < old_length; ++i) {
        struct ptd_vertex *vertex = graph->vertices[i];

        if (vertex->edges_length == 0) {
            continue;
        }

        if (non_zero_rewards[i] == 1) {
            continue;
        }

        if (vertex == graph->starting_vertex) {
            continue;
        }

        double rate = 0;

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            rate += vertex->edges[j]->weight;
        }

        if (rate > 1.0001) {
            size_t debug_index = vertex->index;

            if (PTD_DEBUG_1_INDEX) {
                debug_index++;
            }

            char state[1024] = {'\0'};
            char starting_vertex[] = " (starting vertex)";

            if (vertex != graph->starting_vertex) {
                starting_vertex[0] = '\0';
            }

            ptd_vertex_to_s(vertex, state, 1023);

            snprintf(
                    (char *) ptd_err,
                    sizeof(ptd_err),
                    "Expected vertex with index %i%s (state %s) to have outgoing rate <= 1. Is '%f'. Are you sure this is a discrete phase-type distribution?\n",
                    (int) debug_index, starting_vertex, state, (float) rate
            );

            free(non_zero_rewards);

            return NULL;
        }

        struct ptd_vertex **auxiliary_vertices = (struct ptd_vertex **) calloc(
                (size_t) non_zero_rewards[i],
                sizeof(*auxiliary_vertices)
        );

        auxiliary_vertices[0] = vertex;

        for (int k = 1; k < non_zero_rewards[i]; ++k) {
            auxiliary_vertices[k] = ptd_vertex_create(graph);
        }

        size_t edges_length = vertex->edges_length;

        for (size_t j = 0; j < edges_length; ++j) {
            ptd_graph_add_edge(
                    auxiliary_vertices[non_zero_rewards[i] - 1],
                    vertex->edges[j]->to,
                    vertex->edges[j]->weight
            );

            free(vertex->edges[j]);
        }

        vertex->edges_length = 0;

        for (int k = 0; k < non_zero_rewards[i] - 1; ++k) {
            ptd_graph_add_edge(
                    auxiliary_vertices[k],
                    auxiliary_vertices[k + 1],
                    1
            );
        }

        if (1 - rate > REWARD_EPSILON) {
            ptd_graph_add_edge(
                    auxiliary_vertices[non_zero_rewards[i] - 1],
                    vertex,
                    1 - rate
            );
        }

        free(auxiliary_vertices);
    }

    free(new_graph_indices);
    free(non_zero_rewards);

    return graph;
}


static struct ptd_reward_increase *add_command(
        struct ptd_reward_increase *cmd,
        size_t from,
        size_t to,
        double weight,
        size_t index
) {
    bool is_power_of_2 = (index & (index - 1)) == 0;

    if (is_power_of_2) {
        size_t new_length = index == 0 ? 1 : index * 2;

        cmd = (struct ptd_reward_increase *) realloc(
                cmd, new_length *
                     sizeof(*cmd)
        );
    }

    if (from != to) {
//        fprintf(stderr, "ADD COMMAND %zu += %zu * %f\n", from, to, weight);
        cmd[index].from = from;
        cmd[index].to = to;
        cmd[index].multiplier = weight;
    } else {
        //      fprintf(stderr, "ADD COMMAND %zu *= %f\n", from, weight);
        cmd[index].from = from;
        cmd[index].to = to;
        cmd[index].multiplier = weight - 1;
    }

    return cmd;
}

enum command_types {
    PP = 3,
    P = 1,
    INV = 2,
    ZERO = 6,
    DIVIDE = 5,
    ONE_MINUS = 4,
    NEW_ADD = 0
};

static struct ptd_comp_graph_parameterized *add_command_param_pp(
        struct ptd_comp_graph_parameterized *cmd,
        double *from,
        double *to,
        double *weight,
        size_t index
) {
    bool is_power_of_2 = (index & (index - 1)) == 0;

    if (is_power_of_2) {
        size_t new_length = index == 0 ? 1 : index * 2;

        cmd = (struct ptd_comp_graph_parameterized *) realloc(
                cmd, new_length *
                     sizeof(*cmd)
        );
    }

    cmd[index].type = PP;

    if (from != to) {
        cmd[index].fromT = from;
        cmd[index].toT = to;
        cmd[index].multiplierptr = weight;
    } else {
        cmd[index].fromT = from;
        cmd[index].toT = to;
        cmd[index].multiplierptr = weight - 1;
    }

    return cmd;
}


static struct ptd_comp_graph_parameterized *add_command_param_p(
        struct ptd_comp_graph_parameterized *cmd,
        double *from,
        double *to,
        double weight,
        size_t index
) {
    bool is_power_of_2 = (index & (index - 1)) == 0;

    if (is_power_of_2) {
        size_t new_length = index == 0 ? 1 : index * 2;

        cmd = (struct ptd_comp_graph_parameterized *) realloc(
                cmd, new_length *
                     sizeof(*cmd)
        );
    }

    cmd[index].type = P;

    if (from != to) {
        cmd[index].fromT = from;
        cmd[index].toT = to;
        cmd[index].multiplier = weight;
    } else {
        cmd[index].fromT = from;
        cmd[index].toT = to;
        cmd[index].multiplier = weight - 1;
    }

    return cmd;
}

static struct ptd_comp_graph_parameterized *add_command_param_inverse(
        struct ptd_comp_graph_parameterized *cmd,
        double *from,
        size_t index
) {
    bool is_power_of_2 = (index & (index - 1)) == 0;

    if (is_power_of_2) {
        size_t new_length = index == 0 ? 1 : index * 2;

        cmd = (struct ptd_comp_graph_parameterized *) realloc(
                cmd, new_length *
                     sizeof(*cmd)
        );
    }

    cmd[index].type = INV;
    cmd[index].fromT = from;

    return cmd;
}

static struct ptd_comp_graph_parameterized *add_command_param_zero(
        struct ptd_comp_graph_parameterized *cmd,
        double *from,
        size_t index
) {
    bool is_power_of_2 = (index & (index - 1)) == 0;

    if (is_power_of_2) {
        size_t new_length = index == 0 ? 1 : index * 2;

        cmd = (struct ptd_comp_graph_parameterized *) realloc(
                cmd, new_length *
                     sizeof(*cmd)
        );
    }

    cmd[index].type = ZERO;
    cmd[index].fromT = from;

    return cmd;
}


static struct ptd_comp_graph_parameterized *add_command_param_p_divide(
        struct ptd_comp_graph_parameterized *cmd,
        double *from,
        double *to,
        size_t index
) {
    bool is_power_of_2 = (index & (index - 1)) == 0;

    if (is_power_of_2) {
        size_t new_length = index == 0 ? 1 : index * 2;

        cmd = (struct ptd_comp_graph_parameterized *) realloc(
                cmd, new_length *
                     sizeof(*cmd)
        );
    }

    cmd[index].type = DIVIDE;
    cmd[index].fromT = from;
    cmd[index].toT = to;

    return cmd;
}


static struct ptd_comp_graph_parameterized *add_command_param_one__ptd_minus(
        struct ptd_comp_graph_parameterized *cmd,
        double *from,
        size_t index
) {
    bool is_power_of_2 = (index & (index - 1)) == 0;

    if (is_power_of_2) {
        size_t new_length = index == 0 ? 1 : index * 2;

        cmd = (struct ptd_comp_graph_parameterized *) realloc(
                cmd, new_length *
                     sizeof(*cmd)
        );
    }

    cmd[index].type = ONE_MINUS;
    cmd[index].fromT = from;

    return cmd;
}


static struct ptd_comp_graph_parameterized *add_command_param(
        struct ptd_comp_graph_parameterized *cmd,
        size_t from,
        size_t to,
        double *weight,
        size_t index
) {
    bool is_power_of_2 = (index & (index - 1)) == 0;

    if (is_power_of_2) {
        size_t new_length = index == 0 ? 1 : index * 2;

        cmd = (struct ptd_comp_graph_parameterized *) realloc(
                cmd, new_length *
                     sizeof(*cmd)
        );
    }

    cmd[index].type = NEW_ADD;

    cmd[index].from = from;
    cmd[index].to = to;
    cmd[index].multiplierptr = weight;

    return cmd;
}

struct ptd_desc_reward_compute *ptd_graph_ex_absorbation_time_comp_graph(struct ptd_graph *graph) {
    if (ptd_validate_graph(graph)) {
        return NULL;
    }

    struct ptd_vertex *dummy__ptd_min = (struct ptd_vertex *) 1, *dummy__ptd_max = 0;

    struct ptd_vertex **vertices = (struct ptd_vertex **) calloc(graph->vertices_length, sizeof(*vertices));
    size_t *original_indices = (size_t *) calloc(graph->vertices_length, sizeof(*original_indices));

    struct ptd_reward_increase *commands = NULL;
    size_t command_index = 0;
    size_t vertices_length = graph->vertices_length;

    struct ptd_scc_graph *scc = ptd_find_strongly_connected_components(graph);
    struct ptd_scc_vertex **v = ptd_scc_graph_topological_sort(scc);

    size_t idx = 0;

    for (size_t sii = 0; sii < scc->vertices_length; ++sii) {
        for (size_t j = 0; j < v[sii]->internal_vertices_length; ++j) {
            if (v[sii]->internal_vertices[j]->edges_length == 0) {
                continue;
            }

            original_indices[idx] = v[sii]->internal_vertices[j]->index;
            v[sii]->internal_vertices[j]->index = idx;
            vertices[idx] = v[sii]->internal_vertices[j];
            idx++;
        }
    }

    for (size_t sii = 0; sii < scc->vertices_length; ++sii) {
        for (size_t j = 0; j < v[sii]->internal_vertices_length; ++j) {
            if (v[sii]->internal_vertices[j]->edges_length != 0) {
                continue;
            }

            original_indices[idx] = v[sii]->internal_vertices[j]->index;
            v[sii]->internal_vertices[j]->index = idx;
            vertices[idx] = v[sii]->internal_vertices[j];
            idx++;
        }
    }

    struct arr_p **vertex_parents;
    size_t *vertex_parents_length;
    struct arr_c **vertex_edges;
    size_t *vertex_edges_length;

    for (size_t i = 0; i < vertices_length; ++i) {
        struct ptd_vertex *vertex = vertices[i];

        if (vertex >= dummy__ptd_max) {
            dummy__ptd_max = vertex + 1;
        }

        if (vertex <= dummy__ptd_min) {
            dummy__ptd_min = vertex - 1;
        }

        double rate = 0;

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            rate += vertex->edges[j]->weight;
        }

        // Add the "real" rate as our first reward

        if (graph->starting_vertex == vertex || vertex->edges_length == 0) {
            commands = add_command(
                    commands,
                    original_indices[i],
                    original_indices[i],
                    0,
                    command_index++
            );
        } else {
            commands = add_command(
                    commands,
                    original_indices[i],
                    original_indices[i],
                    1 / rate,
                    command_index++
            );
        }
    }

    vertex_parents = (struct arr_p **) calloc(vertices_length, sizeof(*vertex_parents));
    vertex_parents_length = (size_t *) calloc(vertices_length, sizeof(*vertex_parents_length));
    size_t *vertex_parents_alloc_length = (size_t *) calloc(vertices_length, sizeof(*vertex_parents_alloc_length));
    vertex_edges = (struct arr_c **) calloc(vertices_length, sizeof(*vertex_edges));
    vertex_edges_length = (size_t *) calloc(vertices_length, sizeof(*vertex_edges_length));
    size_t *vertex_edges_alloc_length = (size_t *) calloc(vertices_length, sizeof(*vertex_edges_alloc_length));

    for (size_t i = 0; i < vertices_length; ++i) {
        vertex_edges_alloc_length[i] = 64;
        struct ptd_vertex *vertex = vertices[i];

        while (vertex->edges_length + 2 >= vertex_edges_alloc_length[i]) {
            vertex_edges_alloc_length[i] *= 2;
        }

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            vertex_parents_length[vertex->edges[j]->to->index]++;
        }

        vertex_edges[i] = (struct arr_c *) calloc(vertex_edges_alloc_length[i], sizeof(*(vertex_edges[i])));
        vertex_edges_length[i] = vertex->edges_length + 2;
    }

    for (size_t i = 0; i < vertices_length; ++i) {
        vertex_parents_alloc_length[i] = 64;

        while (vertex_parents_length[i] >= vertex_parents_alloc_length[i]) {
            vertex_parents_alloc_length[i] *= 2;
        }

        vertex_parents[i] = (struct arr_p *) calloc(vertex_parents_alloc_length[i], sizeof(*(vertex_parents[i])));
        vertex_parents_length[i] = 0;
    }

    for (size_t i = 0; i < vertices_length; ++i) {
        struct ptd_vertex *vertex = vertices[i];

        vertex_edges[i][0].to = dummy__ptd_min;
        vertex_edges[i][0].prob = 0;
        vertex_edges[i][0].arr_p_index = (unsigned int) ((int) -1);

        double rate = 0;

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            rate += vertex->edges[j]->weight;
        }

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            vertex_edges[i][j + 1].to = vertex->edges[j]->to;
            vertex_edges[i][j + 1].prob = vertex->edges[j]->weight / rate;
        }

        vertex_edges[i][vertex->edges_length + 1].prob = 0;
        vertex_edges[i][vertex->edges_length + 1].to = dummy__ptd_max;
        vertex_edges[i][vertex->edges_length + 1].arr_p_index = (unsigned int) ((int) -1);

        qsort(vertex_edges[i], vertex_edges_length[i], sizeof(*(vertex_edges[i])), arr_c_cmp);
    }


    for (size_t i = 0; i < vertices_length; ++i) {
        struct ptd_vertex *vertex = vertices[i];

        for (size_t j = 1; j < vertex_edges_length[i] - 1; ++j) {
            struct arr_c *child = &(vertex_edges[i][j]);
            size_t k = child->to->index;
            child->arr_p_index = vertex_parents_length[k];
            vertex_parents[k][vertex_parents_length[k]].p = vertex;
            vertex_parents[k][vertex_parents_length[k]].arr_c_index = j;
            vertex_parents_length[k]++;
        }
    }

    struct arr_c *old_edges_buffer =
            (struct arr_c *) calloc(vertices_length + 2, sizeof(*old_edges_buffer));

    for (size_t i = 0; i < vertices_length; ++i) {
        struct ptd_vertex *me = vertices[i];
        struct arr_c *my_children = vertex_edges[i];
        size_t my_parents_length = vertex_parents_length[i];
        size_t my_edges_length = vertex_edges_length[i];


        for (size_t p = 0; p < my_parents_length; ++p) {
            struct arr_p me_to_parent = vertex_parents[i][p];
            struct ptd_vertex *parent_vertex = me_to_parent.p;

            size_t parent_vertex_index = parent_vertex->index;
            struct arr_c parent_to_me = vertex_edges[parent_vertex_index][me_to_parent.arr_c_index];

            size_t parent_edges_length = vertex_edges_length[parent_vertex_index];

            if (parent_vertex_index < i) {
                continue;
            }

            bool should_resize = false;
            size_t new_parent_edges_alloc_length = my_edges_length + parent_edges_length;

            while (new_parent_edges_alloc_length >= vertex_edges_alloc_length[parent_vertex_index]) {
                vertex_edges_alloc_length[parent_vertex_index] *= 2;
                should_resize = true;
            }

            if (should_resize) {
                vertex_edges[parent_vertex_index] = (struct arr_c *) realloc(
                        vertex_edges[parent_vertex_index],
                        vertex_edges_alloc_length[parent_vertex_index] * sizeof(*(vertex_edges[parent_vertex_index]))
                );
            }

            vertex_edges_length[parent_vertex_index] = 0;

            double parent_weight_to_me = parent_to_me.prob;
            double new_parent_total_prob = 0;

            if (memcpy(
                    old_edges_buffer, vertex_edges[parent_vertex_index],
                    sizeof(struct arr_c) * parent_edges_length
            ) != old_edges_buffer) {
                return NULL;
            }

            struct arr_c *new_parent_children = vertex_edges[parent_vertex_index];

            commands = add_command(
                    commands,
                    original_indices[parent_vertex_index],
                    original_indices[i],
                    parent_weight_to_me,
                    command_index++
            );

            size_t child_index = 0;
            size_t parent_child_index = 0;

            while (child_index < my_edges_length || parent_child_index < parent_edges_length) {
                struct arr_c me_to_child = my_children[child_index];
                struct ptd_vertex *me_to_child_v = me_to_child.to;
                struct arr_c parent_to_child = old_edges_buffer[parent_child_index];
                struct ptd_vertex *parent_to_child_v = parent_to_child.to;
                double me_to_child_p = me_to_child.prob;

                if (me_to_child_v == parent_vertex) {
                    double prob = parent_weight_to_me * me_to_child_p;
                    commands = add_command(
                            commands,
                            original_indices[parent_vertex->index],
                            original_indices[parent_vertex->index],
                            1 / (1 - prob),
                            command_index++
                    );

                    child_index++;
                    continue;
                }

                if (parent_to_child_v == me) {
                    parent_child_index++;
                    continue;
                }

                if (me_to_child_v == parent_to_child_v) {
                    new_parent_children[vertex_edges_length[parent_vertex_index]].to = parent_to_child_v;
                    new_parent_children[vertex_edges_length[parent_vertex_index]].prob =
                            parent_to_child.prob + me_to_child_p * parent_weight_to_me;
                    new_parent_children[vertex_edges_length[parent_vertex_index]].arr_p_index = parent_to_child.arr_p_index;
                    if (parent_to_child_v != dummy__ptd_min && parent_to_child_v != dummy__ptd_max) {
                        size_t current_parent_index = parent_to_child.arr_p_index;
                        vertex_parents[parent_to_child_v->index][current_parent_index].arr_c_index = vertex_edges_length[parent_vertex_index];

                    }
                    new_parent_total_prob += new_parent_children[vertex_edges_length[parent_vertex_index]].prob;
                    vertex_edges_length[parent_vertex_index]++;

                    child_index++;
                    parent_child_index++;
                } else if (me_to_child_v < parent_to_child_v) {
                    size_t child_parents_length = vertex_parents_length[me_to_child_v->index];

                    if (child_parents_length >= vertex_parents_alloc_length[me_to_child_v->index]) {
                        vertex_parents_alloc_length[me_to_child_v->index] *= 2;
                        vertex_parents[me_to_child_v->index] = (struct arr_p *) realloc(
                                vertex_parents[me_to_child_v->index],
                                vertex_parents_alloc_length[me_to_child_v->index] *
                                sizeof(*(vertex_parents[me_to_child_v->index]))
                        );
                    }

                    vertex_parents[me_to_child_v->index][child_parents_length].arr_c_index = vertex_edges_length[parent_vertex_index];
                    vertex_parents[me_to_child_v->index][child_parents_length].p = parent_vertex;

                    new_parent_children[vertex_edges_length[parent_vertex_index]].to = me_to_child_v;
                    new_parent_children[vertex_edges_length[parent_vertex_index]].prob =
                            me_to_child_p * parent_weight_to_me;
                    new_parent_children[vertex_edges_length[parent_vertex_index]].arr_p_index = child_parents_length;
                    new_parent_total_prob += me_to_child_p * parent_weight_to_me;

                    vertex_edges_length[parent_vertex_index]++;
                    vertex_parents_length[me_to_child_v->index]++;

                    child_index++;
                } else {
                    new_parent_children[vertex_edges_length[parent_vertex_index]] = parent_to_child;
                    vertex_parents[parent_to_child_v->index][parent_to_child.arr_p_index].arr_c_index = vertex_edges_length[parent_vertex_index];
                    new_parent_total_prob += parent_to_child.prob;
                    vertex_edges_length[parent_vertex_index]++;

                    parent_child_index++;
                }
            }


            // Make sure parent has rate of 1
            for (size_t j = 0; j < vertex_edges_length[parent_vertex_index]; ++j) {
                new_parent_children[j].prob /= new_parent_total_prob;
            }

            //free(vertex_edges[parent->p->index]);
            //vertex_edges[parent->p->index] = new_parent_children;
            vertex_edges_length[parent_vertex_index] = vertex_edges_length[parent_vertex_index];
        }
    }

    for (size_t ii = 0; ii < vertices_length; ++ii) {
        size_t i = vertices_length - ii - 1;
        struct ptd_vertex *vertex = vertices[i];


        for (size_t j = 1; j < vertex_edges_length[i] - 1; ++j) {
            struct arr_c child = vertex_edges[i][j];
            commands = add_command(
                    commands,
                    original_indices[vertex->index],
                    original_indices[child.to->index],
                    child.prob,
                    command_index++
            );
        }
    }

    for (size_t i = 0; i < vertices_length; ++i) {
        graph->vertices[i]->index = i;
    }

    for (size_t i = 0; i < vertices_length; ++i) {
        free(vertex_edges[i]);
        free(vertex_parents[i]);
    }

    free(vertex_parents_length);
    free(vertex_parents_alloc_length);
    free(vertex_parents);
    free(vertex_edges);
    free(vertex_edges_length);
    free(vertex_edges_alloc_length);
    free(original_indices);
    free(vertices);
    free(old_edges_buffer);
    free(v);
    ptd_scc_graph_destroy(scc);

    commands = add_command(
            commands,
            0,
            0,
            NAN,
            command_index
    );

    struct ptd_desc_reward_compute *res = (struct ptd_desc_reward_compute *) malloc(sizeof(*res));
    res->length = command_index;
    res->commands = commands;

    return res;
}


struct ll_c2_a {
    struct ll_c2_a *next;
    struct ll_c2 *mem;
};

static struct ll_c2_a **ll_c2_alloced;
static size_t ll_c2_alloced__ptd_max = 1024;
static size_t *ll_c2_alloced_index;

static void ll_c2_alloc_init(size_t length) {
    ll_c2_alloced_index = (size_t *) calloc(length, sizeof(*ll_c2_alloced_index));
    ll_c2_alloced = (struct ll_c2_a **) calloc(length, sizeof(*ll_c2_alloced));

    for (size_t i = 0; i < length; ++i) {
        ll_c2_alloced[i] = (struct ll_c2_a *) malloc(sizeof(*(ll_c2_alloced[i])));
        ll_c2_alloced[i]->next = NULL;
        ll_c2_alloced[i]->mem = (struct ll_c2 *) calloc(ll_c2_alloced__ptd_max, sizeof(struct ll_c2));
        ll_c2_alloced_index[i] = 0;
    }
}

static void ll_c2_alloc_init_free(size_t length) {
    free(ll_c2_alloced_index);
    free(ll_c2_alloced);
}

static struct ll_c2 *ll_c2_alloc(size_t index) {
    if (ll_c2_alloced_index[index] >= ll_c2_alloced__ptd_max) {
        struct ll_c2_a *old = ll_c2_alloced[index];
        ll_c2_alloced[index] = (struct ll_c2_a *) malloc(sizeof(*(ll_c2_alloced[index])));
        ll_c2_alloced[index]->next = old;
        ll_c2_alloced[index]->mem = (struct ll_c2 *) calloc(ll_c2_alloced__ptd_max, sizeof(struct ll_c2));
        ll_c2_alloced_index[index] = 0;
    }

    return &(ll_c2_alloced[index]->mem[ll_c2_alloced_index[index]++]);
}

static void ll_c2_free(size_t index) {
    struct ll_c2_a *old = ll_c2_alloced[index];

    while (old != NULL) {
        free(old->mem);
        struct ll_c2_a *next = old->next;
        free(old);
        old = next;
    }
}

struct ll_p2_a {
    struct ll_p2_a *next;
    struct ll_p2 *mem;
};

static struct ll_p2_a **ll_p2_alloced;
static size_t ll_p2_alloced__ptd_max = 1024;
static size_t *ll_p2_alloced_index;

static void ll_p2_alloc_init(size_t length) {
    ll_p2_alloced_index = (size_t *) calloc(length, sizeof(*ll_p2_alloced_index));
    ll_p2_alloced = (struct ll_p2_a **) calloc(length, sizeof(*ll_p2_alloced));

    for (size_t i = 0; i < length; ++i) {
        ll_p2_alloced[i] = (struct ll_p2_a *) malloc(sizeof(*(ll_p2_alloced[i])));
        ll_p2_alloced[i]->next = NULL;
        ll_p2_alloced[i]->mem = (struct ll_p2 *) calloc(ll_p2_alloced__ptd_max, sizeof(struct ll_p2));
        ll_p2_alloced_index[i] = 0;
    }
}

static void ll_p2_alloc_init_free(size_t length) {
    free(ll_p2_alloced);
    free(ll_p2_alloced_index);
}

static struct ll_p2 *ll_p2_alloc(size_t index) {
    if (ll_p2_alloced_index[index] >= ll_p2_alloced__ptd_max) {
        struct ll_p2_a *old = ll_p2_alloced[index];
        ll_p2_alloced[index] = (struct ll_p2_a *) malloc(sizeof(*(ll_p2_alloced[index])));
        ll_p2_alloced[index]->next = old;
        ll_p2_alloced[index]->mem = (struct ll_p2 *) calloc(ll_p2_alloced__ptd_max, sizeof(struct ll_p2));
        ll_p2_alloced_index[index] = 0;
    }

    return &(ll_p2_alloced[index]->mem[ll_p2_alloced_index[index]++]);
}

static void ll_p2_free(size_t index) {
    struct ll_p2_a *old = ll_p2_alloced[index];

    while (old != NULL) {
        free(old->mem);
        struct ll_p2_a *next = old->next;
        free(old);
        old = next;
    }
}

static int t = 0;

static struct ll_of_a *add_mem(struct ll_of_a *current_mem_ll, double what) {
    struct ll_of_a *n;

    if (current_mem_ll == NULL || current_mem_ll->current_mem_index >= 32768) {
        n = (struct ll_of_a *) malloc(sizeof(*n));
        n->next = current_mem_ll;
        n->mem = (double *) calloc(32768, sizeof(double));
        n->current_mem_index = 0;
        n->current_mem_position = n->mem;
        t++;
    } else {
        n = current_mem_ll;
    }

    n->mem[n->current_mem_index] = what;
    n->current_mem_position = &(n->mem[n->current_mem_index]);
    n->current_mem_index++;

    return n;
}

struct ptd_desc_reward_compute_parameterized *ptd_graph_ex_absorbation_time_comp_graph_parameterized(
        struct ptd_graph *graph
) {
    struct ptd_vertex *dummy__ptd_min = 0, *dummy__ptd_max = 0;

    struct ll_of_a *current_mem_ll = NULL;
    current_mem_ll = add_mem(current_mem_ll, 0);
    double *SIMPLE_ZERO = current_mem_ll->current_mem_position;

    struct ll_c2 **edges;
    struct ll_p2 **parents;

    struct ptd_vertex **vertices = (struct ptd_vertex **) calloc(graph->vertices_length, sizeof(*vertices));
    size_t *original_indices = (size_t *) calloc(graph->vertices_length, sizeof(*original_indices));
    edges = (struct ll_c2 **) calloc(graph->vertices_length, sizeof(*edges));
    parents = (struct ll_p2 **) calloc(graph->vertices_length, sizeof(*parents));
    ll_c2_alloc_init(1);
    ll_p2_alloc_init(1);
    struct ptd_comp_graph_parameterized *commands = NULL;
    size_t command_index = 0;
    size_t vertices_length = graph->vertices_length;


    struct ptd_scc_graph *scc = ptd_find_strongly_connected_components(graph);
    struct ptd_scc_vertex **v = ptd_scc_graph_topological_sort(scc);
    size_t idx = 0;

    for (size_t sii = 0; sii < scc->vertices_length; ++sii) {
        for (size_t j = 0; j < v[sii]->internal_vertices_length; ++j) {
            if (v[sii]->internal_vertices[j]->edges_length == 0) {
                continue;
            }

            original_indices[idx] = v[sii]->internal_vertices[j]->index;
            v[sii]->internal_vertices[j]->index = idx;
            vertices[idx] = v[sii]->internal_vertices[j];
            idx++;
        }
    }

    for (size_t sii = 0; sii < scc->vertices_length; ++sii) {
        for (size_t j = 0; j < v[sii]->internal_vertices_length; ++j) {
            if (v[sii]->internal_vertices[j]->edges_length != 0) {
                continue;
            }

            original_indices[idx] = v[sii]->internal_vertices[j]->index;
            v[sii]->internal_vertices[j]->index = idx;
            vertices[idx] = v[sii]->internal_vertices[j];
            idx++;
        }
    }

    double **rates = (double **) calloc(graph->vertices_length, sizeof(*rates));

    for (size_t i = 0; i < vertices_length; ++i) {
        struct ptd_vertex *vertex = vertices[i];

        if (vertex >= dummy__ptd_max) {
            dummy__ptd_max = vertex + 1;
        }

        if (vertex <= dummy__ptd_min) {
            dummy__ptd_min = vertex - 1;
        }

        current_mem_ll = add_mem(current_mem_ll, 0);
        rates[i] = current_mem_ll->current_mem_position;
        commands = add_command_param_zero(
                commands,
                rates[i],
                command_index++
        );

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            commands = add_command_param_p(
                    commands,
                    rates[i],
                    &(vertex->edges[j]->weight),
                    1,
                    command_index++
            );
        }

        commands = add_command_param_inverse(
                commands,
                rates[i],
                command_index++
        );

        // Add the "real" rate as our first reward

        if (graph->starting_vertex == vertex || vertex->edges_length == 0) {
            commands = add_command_param(
                    commands,
                    original_indices[i],
                    original_indices[i],
                    SIMPLE_ZERO,
                    command_index++
            );
        } else {
            commands = add_command_param(
                    commands,
                    original_indices[i],
                    original_indices[i],
                    rates[i],
                    command_index++
            );
        }
    }

    for (size_t i = 0; i < vertices_length; ++i) {
        struct ptd_vertex *vertex = vertices[i];

        struct ll_c2 *dummy_first = ll_c2_alloc(0);
        dummy_first->next = NULL;
        dummy_first->prev = NULL;
        dummy_first->weight = 0;
        dummy_first->c = dummy__ptd_min;
        dummy_first->ll_p = NULL;
        edges[i] = dummy_first;

        struct ll_c2 *last = dummy_first;

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            struct ll_p2 *n = ll_p2_alloc(0);

            n->next = parents[vertex->edges[j]->to->index];
            n->p = vertex;
            n->prev = NULL;

            if (parents[vertex->edges[j]->to->index] != NULL) {
                parents[vertex->edges[j]->to->index]->prev = n;
            }

            parents[vertex->edges[j]->to->index] = n;

            struct ll_c2 *nc = ll_c2_alloc(0);
            nc->next = NULL;

            nc->prev = last;
            last->next = nc;

            current_mem_ll = add_mem(current_mem_ll, 0);

            commands = add_command_param_zero(
                    commands,
                    current_mem_ll->current_mem_position,
                    command_index++
            );

            commands = add_command_param_pp(
                    commands,
                    current_mem_ll->current_mem_position,
                    &(vertex->edges[j]->weight),
                    rates[i],
                    command_index++
            );

            nc->weight = current_mem_ll->current_mem_position;

            nc->c = vertex->edges[j]->to;
            nc->ll_p = n;
            n->ll_c = nc;
            last = nc;
        }

        struct ll_c2 *dummy_last = ll_c2_alloc(0);
        dummy_last->next = NULL;
        dummy_last->prev = last;
        dummy_last->weight = 0;
        dummy_last->c = dummy__ptd_max;
        dummy_last->ll_p = NULL;
        last->next = dummy_last;
    }

    int ri = 0;

    for (size_t i = 0; i < vertices_length; ++i) {
        struct ptd_vertex *vertex = vertices[i];

        ri++;

        struct ll_p2 *parent = parents[i];

        struct ll_c2 *c = edges[i];
        size_t n_edges = 0;

        while (c != NULL) {
            n_edges += 1;
            c = c->next;
        }

        struct ll_c2 *children_arr = (struct ll_c2 *) calloc(n_edges, sizeof(*children_arr));
        c = edges[i];
        size_t l = 0;

        while (c != NULL) {
            children_arr[l] = *c;
            l++;
            c = c->next;
        }

        while (parent != NULL) {
            if (parent->p->index < i) {
                parent = parent->next;
                continue;
            }

            l = 0;
            struct ll_c2 *parent_child = edges[parent->p->index];
            double *parent_weight_to_me = parent->ll_c->weight;

            commands = add_command_param(
                    commands,
                    original_indices[parent->p->index],
                    original_indices[i],
                    parent_weight_to_me,
                    command_index++
            );

            while (children_arr[l].c != dummy__ptd_max) {
                double *prob = children_arr[l].weight;
                struct ptd_vertex *child_vertex = children_arr[l].c;
                struct ptd_vertex *parent_vertex = parent->p;
                struct ptd_vertex *parent_child_vertex = parent_child->c;

                if (child_vertex == parent_vertex) {
                    current_mem_ll = add_mem(current_mem_ll, 0);
                    double *p = current_mem_ll->current_mem_position;

                    commands = add_command_param_zero(
                            commands,
                            p,
                            command_index++
                    );

                    commands = add_command_param_pp(
                            commands,
                            p,
                            parent_weight_to_me,
                            prob,
                            command_index++
                    );

                    commands = add_command_param_one__ptd_minus(
                            commands,
                            p,
                            command_index++
                    );

                    commands = add_command_param_inverse(
                            commands,
                            p,
                            command_index++
                    );

                    commands = add_command_param(
                            commands,
                            original_indices[parent_vertex->index],
                            original_indices[parent_vertex->index],
                            p,
                            command_index++
                    );

                    l++;
                    continue;
                }

                if (parent_child_vertex == vertex) {
                    parent_child = parent_child->next;
                    continue;
                }

                if (child_vertex == parent_child_vertex) {
                    if (child_vertex != dummy__ptd_min) {
                        current_mem_ll = add_mem(current_mem_ll, 0);
                        double *p = current_mem_ll->current_mem_position;

                        commands = add_command_param_zero(
                                commands,
                                p,
                                command_index++
                        );

                        commands = add_command_param_pp(
                                commands,
                                p,
                                parent_weight_to_me,
                                prob,
                                command_index++
                        );

                        commands = add_command_param_p(
                                commands,
                                parent_child->weight,
                                p,
                                1,
                                command_index++
                        );
                    }

                    l++;
                    parent_child = parent_child->next;
                } else if (child_vertex < parent_child_vertex) {
                    current_mem_ll = add_mem(current_mem_ll, 0);
                    double *p = current_mem_ll->current_mem_position;
                    commands = add_command_param_zero(
                            commands,
                            p,
                            command_index++
                    );

                    commands = add_command_param_pp(
                            commands,
                            p,
                            parent_weight_to_me,
                            prob,
                            command_index++
                    );

                    struct ll_c2 *to = ll_c2_alloc(0);
                    to->c = child_vertex;

                    current_mem_ll = add_mem(current_mem_ll, 0);
                    commands = add_command_param_zero(
                            commands,
                            current_mem_ll->current_mem_position,
                            command_index++
                    );
                    to->weight = current_mem_ll->current_mem_position;

                    commands = add_command_param_p(
                            commands,
                            to->weight,
                            p,
                            1,
                            command_index++
                    );
                    to->next = parent_child;
                    to->prev = parent_child->prev;


                    struct ll_p2 *ll_p = ll_p2_alloc(0);
                    ll_p->next = parents[child_vertex->index];
                    parents[child_vertex->index]->prev = ll_p;
                    parents[child_vertex->index] = ll_p;
                    ll_p->prev = NULL;
                    ll_p->p = parent_vertex;

                    ll_p->ll_c = to;
                    to->ll_p = ll_p;

                    to->next = parent_child;
                    to->prev = parent_child->prev;
                    parent_child->prev->next = to;
                    parent_child->prev = to;

                    l++;
                } else {
                    parent_child = parent_child->next;
                }
            }

            struct ll_c2 *edge_to_me = parent->ll_c;
            edge_to_me->prev->next = edge_to_me->next;
            edge_to_me->next->prev = edge_to_me->prev;

            // Make sure parent has rate of 1
            current_mem_ll = add_mem(current_mem_ll, 0);
            double *rate = current_mem_ll->current_mem_position;
            commands = add_command_param_zero(
                    commands,
                    rate,
                    command_index++
            );

            parent_child = edges[parent->p->index]->next;

            while (parent_child->c != dummy__ptd_max) {
                commands = add_command_param_p(
                        commands,
                        rate,
                        parent_child->weight,
                        1,
                        command_index++
                );

                parent_child = parent_child->next;
            }

            parent_child = edges[parent->p->index]->next;

            while (parent_child->c != dummy__ptd_max) {
                commands = add_command_param_p_divide(
                        commands,
                        parent_child->weight,
                        rate,
                        command_index++
                );

                parent_child = parent_child->next;
            }

            parent_child = edges[parent->p->index]->next;

            while (parent_child->c != dummy__ptd_max) {
                parent_child = parent_child->next;
            }

            parent = parent->next;
        }

        struct ll_c2 *child = edges[i]->next;

        while (child->c != dummy__ptd_max) {
            if (child->ll_p->prev != NULL) {
                if (child->ll_p->next != NULL) {
                    child->ll_p->next->prev = child->ll_p->prev;
                    child->ll_p->prev->next = child->ll_p->next;
                } else {
                    child->ll_p->prev->next = NULL;
                }
            } else {
                if (child->ll_p->next != NULL) {
                    child->ll_p->next->prev = NULL;
                }

                parents[child->c->index] = child->ll_p->next;
            }

            child = child->next;
        }

        free(children_arr);
    }

    for (size_t ii = 0; ii < vertices_length; ++ii) {
        size_t i = vertices_length - ii - 1;
        struct ptd_vertex *vertex = vertices[i];

        struct ll_c2 *child = edges[vertex->index]->next;

        while (child->c != dummy__ptd_max) {
            commands = add_command_param(
                    commands,
                    original_indices[vertex->index],
                    original_indices[child->c->index],
                    child->weight,
                    command_index++
            );
            child = child->next;
        }
    }

    for (size_t i = 0; i < vertices_length; ++i) {
        graph->vertices[i]->index = i;
    }


    free(original_indices);
    free(vertices);
    free(edges);
    free(parents);
    free(v);
    ptd_scc_graph_destroy(scc);
    ll_c2_free(0);
    ll_p2_free(0);
    ll_c2_alloc_init_free(1);
    ll_p2_alloc_init_free(1);

    commands = add_command_param(
            commands,
            0,
            0,
            NULL,
            command_index
    );

    struct ptd_desc_reward_compute_parameterized *res = (struct ptd_desc_reward_compute_parameterized *) malloc(
            sizeof(*res)
    );
    res->length = command_index;
    res->commands = commands;
    res->mem = current_mem_ll;
    res->memr = rates;

    return res;
}

struct ptd_desc_reward_compute *ptd_graph_build_ex_absorbation_time_comp_graph_parameterized(
        struct ptd_desc_reward_compute_parameterized *compute
) {
    struct ptd_reward_increase *commands = NULL;
    size_t command_index = 0;
    enum command_types {
        PP = 3,
        P = 1,
        INV = 2,
        ZERO = 6,
        DIVIDE = 5,
        ONE_MINUS = 4,
        NEW_ADD = 0
    };
    for (size_t i = 0; i < compute->length; ++i) {
        struct ptd_comp_graph_parameterized command = compute->commands[i];

        switch (command.type) {
            case NEW_ADD:
                commands = add_command(
                        commands,
                        command.from,
                        command.to,
                        *command.multiplierptr,
                        command_index++
                );
                break;
            case P:
                *(command.fromT) = *(command.fromT) + *command.toT * command.multiplier;
                break;
            case PP:
                *(command.fromT) = *(command.fromT) + *command.toT * *command.multiplierptr;
                break;
            case INV:
                *(command.fromT) = 1 / *(command.fromT);
                break;
            case ONE_MINUS:
                *(command.fromT) = 1 - *command.fromT;
                break;
            case DIVIDE:
                *(command.fromT) /= *command.toT;
                break;
            case ZERO:
                *command.fromT = 0;
                break;
            default:
                DIE_ERROR(1, "Unknown command\n");
        }
    }

    struct ptd_desc_reward_compute *res = (struct ptd_desc_reward_compute *) malloc(sizeof(*res));
    res->length = command_index;
    res->commands = commands;

    return res;
}


double *ptd_expected_waiting_time(struct ptd_graph *graph, double *rewards) {
    if (ptd_precompute_reward_compute_graph(graph)) {
        return NULL;
    }

    double *result = (double *) calloc(graph->vertices_length, sizeof(*result));

    if (rewards != NULL) {
        // TODO: fix this if reward is nan...
        memcpy(result, rewards, sizeof(*result) * graph->vertices_length);
    } else {
        for (size_t j = 0; j < graph->vertices_length; ++j) {
            result[j] = 1;
        }
    }

    for (size_t j = 0; j < graph->reward_compute_graph->length; ++j) {
        struct ptd_reward_increase command = graph->reward_compute_graph->commands[j];

        result[command.from] += result[command.to] * command.multiplier;

        //TODO: if inf, give error stating that there is an infinite loop
    }

    return result;
}

double *ptd_expected_residence_time(struct ptd_graph *graph, double *rewards) {
    if (ptd_precompute_reward_compute_graph(graph)) {
        return NULL;
    }

    double *result = (double *) calloc(graph->vertices_length, sizeof(*result));

    if (rewards != NULL) {
        // TODO: fix this if reward is nan...
        memcpy(result, rewards, sizeof(*result) * graph->vertices_length);
    } else {
        for (size_t j = 0; j < graph->vertices_length; ++j) {
            result[j] = 1;
        }
    }

    // we want only the acyclic graph so we we subtract graph->vertices_length to skip 
    // the commands computing the expected waiting time
    for (size_t j = 0; j < graph->reward_compute_graph->length - graph->vertices_length; ++j) {
        struct ptd_reward_increase command = graph->reward_compute_graph->commands[j];
        result[command.from] += result[command.to] * command.multiplier;
        //TODO: if inf, give error stating that there is an infinite loop
    }

    // make a copy of the result at this point
    double *dag_vertex_props = (double *) calloc(graph->vertices_length, sizeof(*dag_vertex_props));
    memcpy(dag_vertex_props, result, sizeof(*result) * graph->vertices_length);

    // continue computing the expected waiting time
    for (size_t j = graph->reward_compute_graph->length - graph->vertices_length; j < graph->reward_compute_graph->length; ++j) {
        struct ptd_reward_increase command = graph->reward_compute_graph->commands[j];
        result[command.from] += result[command.to] * command.multiplier;
        //TODO: if inf, give error stating that there is an infinite loop
    }

    // compute the expected residence time
    double *res_times = (double *) calloc(graph->vertices_length, sizeof(*res_times));
    for (size_t j = 0; j < graph->vertices_length; ++j) {
        res_times[j] = 0;
    }
    res_times[0] = result[0]; // expected waiting time
    double *scalars = (double *) calloc(graph->vertices_length, sizeof(*scalars));
    for (size_t j = 0; j < graph->vertices_length; ++j) {
        scalars[j] = 0 ;
    } 
    scalars[0] = 1;
    struct ptd_vertex *start_vertex = graph->starting_vertex;
    double pushed = 0;
    int prev_idx = -1;
    int prev_child_idx = -1;
    // for (size_t j = graph->reward_compute_graph->length - graph->vertices_length; j <  graph->reward_compute_graph->length; ++j) {
    //     struct ptd_reward_increase command = graph->reward_compute_graph->commands[j];
    // for (size_t j = 1; j <  graph->vertices_length; ++j) {
    //     struct ptd_reward_increase command = graph->reward_compute_graph->commands[graph->reward_compute_graph->length - j];
    for (size_t j = 0; j <  graph->vertices_length; ++j) {
        struct ptd_reward_increase command = graph->reward_compute_graph->commands[graph->reward_compute_graph->length - j - 1];

        int idx = command.from;
        int child_idx = command.to;
        double child_prob = command.multiplier;
        double wt = 1 / dag_vertex_props[idx] * scalars[idx];

        // fprintf(stderr, "%d\n", graph->vertices[idx]->index);
        // fprintf(stderr, "%d -> %d, %f, %f\n", idx, child_idx, child_prob, wt);

        // char message[1024];
        // sprintf(message, "%zu -> %d, %f, %f\n", idx, child_idx, child_prob, wt);
        // DEBUG_PRINT(message);
        // DEBUG_PRINT("HELLO\n");
        
        

        if (wt < 0) {
            snprintf(
                (char *) ptd_err, 
                sizeof(ptd_err),
                "%d -> %d, %f, %f\n",
                idx, child_idx, (float) child_prob, (float) wt
            );
            return NULL;
        }


        // snprintf(
        //         (char *) ptd_err,
        //         sizeof(ptd_err),
        //         "Multiple edges to the same vertex!. From vertex with index %i%s (state %s)."
        //         " To vertex with index %i (state %s)\n",
        //         (int) debug_index_from, starting_vertex, state,
        //         (int) debug_index_to, state_to
        // );

        if (idx == start_vertex->index) {
            wt = 0;
        }
        if (prev_child_idx != child_idx) {
            // fprintf(stderr, "removing total push from vertex %zu: %f\n", child_idx, pushed);
            res_times[prev_idx] -= pushed;
            pushed = 0;
        }
        if (dag_vertex_props[child_idx] > 0) { // don't push to absorbing
            double push = (res_times[idx] - wt) * child_prob;
            // fprintf(stderr, "pushing %f to %zu\n", push, child_idx);
            res_times[child_idx] += push;
            scalars[child_idx] += scalars[idx] * child_prob;
            pushed += push;
        }
        prev_idx = idx;
        prev_child_idx = child_idx;
        //TODO: if inf, give error stating that there is an infinite loop
    }

    free(result);
    free(scalars);
    free(dag_vertex_props);

    return res_times;
}

/////////////////////////////////////////

// the commands are in reverse toplogogical order so 
// command.from is the parent index
// command.to is the child index
// command.multiplier is the edge weight
// dag_vertex_props[command.from] is the parent vertex reward
// dag_vertex_props[command.to] is the child vertex reward


// I can make the parent rewards 1 and make the edge weights the child rewards: command.multiplier / result[command.from]


// residence_times <- function(graph) {
//     res <- rep(0, vertices_length(graph))
//     res[1] <- expectation(graph)
//     sca <- rep(0, vertices_length(graph))
//     sca[1] <- 1
//     start_idx <- starting_vertex(graph)$index
//     for (vertex in vertices(graph)) {
//         idx <- vertex$index
//         pushed <- 0
//         for (edge in edges(vertex)) {
//             child_idx <- edge$child$index
//             child_prob <- edge$weight / vertex$rate
//             wt <- 1/vertex$rate * sca[idx]            
//             # if (vertex$index == 1) {
//             if (idx == start_idx) {
//                 wt <- 0
//             } 
//             if (length(edges(edge$child)) > 0) { # don't push to absorbing
//                 push <- (res[idx] - wt) * child_prob
//                 res[child_idx] <- res[child_idx] + push
//                 sca[child_idx] <- sca[child_idx] + sca[idx] * child_prob
//                 pushed <- pushed + push
//             }
//         }
//         res[idx] <- res[idx] - pushed
//     }
//     return(res)
// }

/////////////////////////////////////////

long double ptd_random_sample(struct ptd_graph *graph, double *rewards) {
    long double outcome = 0;

    struct ptd_vertex *vertex = graph->starting_vertex;

    while (vertex->edges_length != 0) {
        long double draw_wait = (long double) rand() / (long double) RAND_MAX;

        double rate = 0;

        for (size_t i = 0; i < vertex->edges_length; ++i) {
            long double edge_weight = vertex->edges[i]->weight;
            rate += edge_weight;
        }

        long double waiting_time = -logl(draw_wait + 0.0000001) / rate;

        if (rewards != NULL) {
            waiting_time *= rewards[vertex->index];
        }

        if (vertex == graph->starting_vertex) {
            waiting_time = 0;
        }

        outcome += waiting_time;

        long double draw_direction = (long double) rand() / (long double) RAND_MAX;
        long double weight_sum = 0;
        size_t edge_index = 0;

        for (size_t i = 0; i < vertex->edges_length; ++i) {
            long double edge_weight = vertex->edges[i]->weight;
            weight_sum += edge_weight;

            if (weight_sum / rate >= draw_direction) {
                edge_index = i;
                break;
            }
        }

        vertex = vertex->edges[edge_index]->to;
    }

    return outcome;
}

long double *ptd_mph_random_sample(struct ptd_graph *graph, double *rewards, size_t vertex_rewards_length) {
    long double *outcome = (long double *) calloc(vertex_rewards_length, sizeof(*outcome));

    for (size_t j = 0; j < vertex_rewards_length; ++j) {
        outcome[j] = 0;
    }

    struct ptd_vertex *vertex = graph->starting_vertex;

    while (vertex->edges_length != 0) {
        long double draw_wait = (long double) rand() / (long double) RAND_MAX;

        double rate = 0;

        for (size_t i = 0; i < vertex->edges_length; ++i) {
            long double edge_weight = vertex->edges[i]->weight;
            rate += edge_weight;
        }

        long double waiting_time = -logl(draw_wait + 0.0000001) / rate;

        if (vertex != graph->starting_vertex) {
            for (size_t i = 0; i < vertex_rewards_length; ++i) {
                outcome[i] += waiting_time * rewards[vertex->index * vertex_rewards_length + i];
            }
        }

        long double draw_direction = (long double) rand() / (long double) RAND_MAX;
        long double weight_sum = 0;
        size_t edge_index = 0;

        for (size_t i = 0; i < vertex->edges_length; ++i) {
            long double edge_weight = vertex->edges[i]->weight;
            weight_sum += edge_weight;

            if (weight_sum / rate >= draw_direction) {
                edge_index = i;
                break;
            }
        }

        vertex = vertex->edges[edge_index]->to;
    }

    return outcome;
}


long double ptd_dph_random_sample(struct ptd_graph *graph, double *rewards) {
    long double jumps = 0;

    struct ptd_vertex *vertex = graph->starting_vertex;

    while (vertex->edges_length != 0) {
        long double draw_direction = (long double) rand() / (long double) RAND_MAX;
        long double weight_sum = 0;
        int edge_index = -1;

        double rate = 0;

        for (size_t i = 0; i < vertex->edges_length; ++i) {
            long double edge_weight = vertex->edges[i]->weight;
            rate += edge_weight;
        }

        if (rate > 1.0001) {
            size_t debug_index = vertex->index;

            if (PTD_DEBUG_1_INDEX) {
                debug_index++;
            }

            char state[1024] = {'\0'};
            char starting_vertex[] = " (starting vertex)";

            if (vertex != graph->starting_vertex) {
                starting_vertex[0] = '\0';
            }

            ptd_vertex_to_s(vertex, state, 1023);

            snprintf(
                    (char *) ptd_err,
                    sizeof(ptd_err),
                    "Expected vertex with index %i%s (state %s) to have outgoing rate <= 1. Is '%f'. Are you sure this is a discrete phase-type distribution?\n",
                    (int) debug_index, starting_vertex, state, (float) rate
            );

            return NAN;
        }

        for (int i = 0; i < (int) vertex->edges_length; ++i) {
            long double edge_weight = vertex->edges[i]->weight;
            weight_sum += edge_weight;

            if (weight_sum >= draw_direction) {
                edge_index = i;
                break;
            }
        }

        if (vertex != graph->starting_vertex) {
            if (rewards == NULL) {
                jumps += 1;
            } else {
                jumps += rewards[vertex->index];
            }
        }

        if (edge_index != -1) {
            vertex = vertex->edges[edge_index]->to;
        }
    }

    return jumps;
}

long double *ptd_mdph_random_sample(struct ptd_graph *graph, double *rewards, size_t vertex_rewards_length) {
    long double *jumps = (long double *) calloc(vertex_rewards_length, sizeof(*jumps));

    for (size_t j = 0; j < vertex_rewards_length; ++j) {
        jumps[j] = 0;
    }

    struct ptd_vertex *vertex = graph->starting_vertex;

    while (vertex->edges_length != 0) {
        long double draw_direction = (long double) rand() / (long double) RAND_MAX;
        long double weight_sum = 0;
        int edge_index = -1;
        double rate = 0;

        for (size_t i = 0; i < vertex->edges_length; ++i) {
            long double edge_weight = vertex->edges[i]->weight;
            rate += edge_weight;
        }

        if (rate > 1.0001) {
            size_t debug_index = vertex->index;

            if (PTD_DEBUG_1_INDEX) {
                debug_index++;
            }

            char state[1024] = {'\0'};
            char starting_vertex[] = " (starting vertex)";

            if (vertex != graph->starting_vertex) {
                starting_vertex[0] = '\0';
            }

            ptd_vertex_to_s(vertex, state, 1023);

            snprintf(
                    (char *) ptd_err,
                    sizeof(ptd_err),
                    "Expected vertex with index %i%s (state %s) to have outgoing rate <= 1. Is '%f'. Are you sure this is a discrete phase-type distribution?\n",
                    (int) debug_index, starting_vertex, state, (float) rate
            );

            free(jumps);

            return NULL;
        }

        for (int i = 0; i < (int) vertex->edges_length; ++i) {
            long double edge_weight = vertex->edges[i]->weight;
            weight_sum += edge_weight;

            if (weight_sum >= draw_direction) {
                edge_index = i;
                break;
            }
        }


        if (vertex != graph->starting_vertex) {
            for (size_t i = 0; i < vertex_rewards_length; ++i) {
                jumps[i] += rewards[vertex->index * vertex_rewards_length + i];
            }
        }


        if (edge_index != -1) {
            vertex = vertex->edges[edge_index]->to;
        }
    }

    return jumps;
}

struct ptd_vertex *ptd_random_sample_stop_vertex(struct ptd_graph *graph, double time) {
    double time_spent = 0;

    struct ptd_vertex *vertex = graph->starting_vertex;

    while (vertex->edges_length != 0) {
        long double draw_wait = (long double) rand() / (long double) RAND_MAX;

        double rate = 0;

        for (size_t i = 0; i < vertex->edges_length; ++i) {
            long double edge_weight = vertex->edges[i]->weight;
            rate += edge_weight;
        }

        long double waiting_time = -logl(draw_wait + 0.0000001) / rate;

        if (vertex == graph->starting_vertex) {
            waiting_time = 0;
        }

        time_spent += waiting_time;

        if (time_spent >= time && vertex != graph->starting_vertex) {
            return vertex;
        }

        long double draw_direction = (long double) rand() / (long double) RAND_MAX;
        long double weight_sum = 0;
        size_t edge_index = 0;

        for (size_t i = 0; i < vertex->edges_length; ++i) {
            long double edge_weight = vertex->edges[i]->weight;
            weight_sum += edge_weight;

            if (weight_sum / rate >= draw_direction) {
                edge_index = i;
                break;
            }
        }

        vertex = vertex->edges[edge_index]->to;
    }

    return vertex;
}

struct ptd_vertex *ptd_dph_random_sample_stop_vertex(struct ptd_graph *graph, int jumps) {
    int jumps_taken = -1;

    struct ptd_vertex *vertex = graph->starting_vertex;

    while (vertex->edges_length != 0 && jumps < jumps_taken) {
        long double draw_direction = (long double) rand() / (long double) RAND_MAX;
        long double weight_sum = 0;
        int edge_index = -1;

        double rate = 0;

        for (size_t i = 0; i < vertex->edges_length; ++i) {
            long double edge_weight = vertex->edges[i]->weight;
            rate += edge_weight;
        }

        if (rate > 1.0001) {
            size_t debug_index = vertex->index;

            if (PTD_DEBUG_1_INDEX) {
                debug_index++;
            }

            char state[1024] = {'\0'};
            char starting_vertex[] = " (starting vertex)";

            if (vertex != graph->starting_vertex) {
                starting_vertex[0] = '\0';
            }

            ptd_vertex_to_s(vertex, state, 1023);

            snprintf(
                    (char *) ptd_err,
                    sizeof(ptd_err),
                    "Expected vertex with index %i%s (state %s) to have outgoing rate <= 1. Is '%f'. Are you sure this is a discrete phase-type distribution?\n",
                    (int) debug_index, starting_vertex, state, (float) rate
            );

            return NULL;
        }

        for (int i = 0; i < (int) vertex->edges_length; ++i) {
            long double edge_weight = vertex->edges[i]->weight;
            weight_sum += edge_weight;

            if (weight_sum >= draw_direction) {
                edge_index = i;
                break;
            }
        }

        if (vertex != graph->starting_vertex) {
            jumps += 1;
        }

        if (edge_index != -1) {
            vertex = vertex->edges[edge_index]->to;
        }
    }

    return vertex;
}


struct ptd_vertex *
ptd_find_or_create_vertex(struct ptd_graph *graph, struct ptd_avl_tree *avl_tree, const int *child_state) {
    struct ptd_vertex *child;
    struct ptd_avl_node *avl_node = ptd_avl_tree_find(avl_tree, child_state);

    if (avl_node == NULL) {
        child = ptd_vertex_create(graph);
        memcpy(child->state, child_state, graph->state_length * sizeof(int));

        ptd_avl_tree_find_or_insert(avl_tree, child->state, child);
    } else {
        child = (struct ptd_vertex *) avl_node->entry;
    }

    return child;
}

struct dph_prob_increment {
    size_t from;
    size_t to;
    double *weight;
};

struct ptd_dph_probability_distribution_context *_ptd_dph_probability_distribution_context_create(
        struct ptd_graph *graph,
        bool dont_worry
) {
    if (!dont_worry) {
        for (size_t i = 0; i < graph->vertices_length; ++i) {
            double rate = 0;

            struct ptd_vertex *vertex = graph->vertices[i];

            for (size_t j = 0; j < vertex->edges_length; ++j) {
                rate += vertex->edges[j]->weight;
            }

            if (rate > 1.0001) {
                size_t debug_index = vertex->index;

                if (PTD_DEBUG_1_INDEX) {
                    debug_index++;
                }

                char state[1024] = {'\0'};
                char starting_vertex[] = " (starting vertex)";

                if (vertex != graph->starting_vertex) {
                    starting_vertex[0] = '\0';
                }

                ptd_vertex_to_s(vertex, state, 1023);

                snprintf(
                        (char *) ptd_err,
                        sizeof(ptd_err),
                        "Expected vertex with index %i%s (state %s) to have outgoing rate <= 1. Is '%f'. Are you sure this is a discrete phase-type distribution?\n",
                        (int) debug_index, starting_vertex, state, (float) rate
                );

                return NULL;
            }
        }
    }

    struct ptd_dph_probability_distribution_context *res =
            (struct ptd_dph_probability_distribution_context *) malloc(sizeof(*res));

    res->graph = graph;
    res->probability_at = (long double *) calloc(
            graph->vertices_length,
            sizeof(*(res->probability_at))
    );
    res->accumulated_visits = (long double *) calloc(
            graph->vertices_length,
            sizeof(*(res->accumulated_visits))
    );

    size_t number_of_edges = 0;

    for (size_t i = 0; i < graph->vertices_length; ++i) {
        struct ptd_vertex *vertex = graph->vertices[i];
        res->accumulated_visits[i] = 0;
        number_of_edges += vertex->edges_length;
    }

    res->priv2 = number_of_edges;
    res->priv3 = 1;

    res->priv = calloc(number_of_edges, sizeof(struct dph_prob_increment));
    struct dph_prob_increment *inc_list = (struct dph_prob_increment *) res->priv;
    size_t inc_index = 0;

    for (size_t i = 0; i < graph->vertices_length; ++i) {
        struct ptd_vertex *vertex = graph->vertices[i];

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            inc_list[inc_index].from = i;
            inc_list[inc_index].to = vertex->edges[j]->to->index;
            inc_list[inc_index].weight = &(vertex->edges[j]->weight);

            inc_index++;
        }
    }

    for (size_t i = 0; i < graph->vertices_length; ++i) {
        res->probability_at[i] = 0;
    }

    res->probability_at[0] = 1;

    res->cdf = 0;
    res->pmf = 0;
    res->jumps = 0;

    ptd_dph_probability_distribution_step(res);

    res->jumps = 0;

    return res;
}

struct ptd_dph_probability_distribution_context *ptd_dph_probability_distribution_context_create(
        struct ptd_graph *graph
) {
    return _ptd_dph_probability_distribution_context_create(graph, false);
}

void ptd_dph_probability_distribution_context_destroy(struct ptd_dph_probability_distribution_context *context) {
    if (context == NULL) {
        return;
    }

    free(context->accumulated_visits);
    free(context->probability_at);
    free(context->priv);
    free(context);
}

int ptd_dph_probability_distribution_step(
        struct ptd_dph_probability_distribution_context *context
) {
    context->jumps++;
    context->pmf = 0;

    long double *old_probability_at = (long double *) calloc(
            context->graph->vertices_length, sizeof(*old_probability_at)
    );

    memcpy(
            old_probability_at,
            context->probability_at,
            sizeof(*old_probability_at) * context->graph->vertices_length
    );

    for (size_t i = 0; i < context->graph->vertices_length; ++i) {
        old_probability_at[i] = context->probability_at[i];
        struct ptd_vertex *vertex = context->graph->vertices[i];

        if (vertex->edges_length == 0) {
            context->probability_at[i] = 0;
        }
    }

    for (size_t i = 0; i < context->priv2; ++i) {
        struct dph_prob_increment inc = ((struct dph_prob_increment *) (context->priv))[i];
        long double add = old_probability_at[inc.from] * (*inc.weight) * context->priv3;
        context->probability_at[inc.to] += add;
        context->probability_at[inc.from] -= add;
    }

    for (size_t i = 0; i < context->graph->vertices_length; ++i) {
        struct ptd_vertex *vertex = context->graph->vertices[i];

        if (vertex->edges_length == 0) {
            context->pmf += context->probability_at[i];
            context->probability_at[i] = 0;
        } else {
            context->accumulated_visits[i] += context->probability_at[i];
        }
    }

    context->accumulated_visits[0] = 0;
    context->probability_at[0] = 0;

    context->cdf += context->pmf;

    free(old_probability_at);

    return 0;
}

struct ptd_probability_distribution_context *ptd_probability_distribution_context_create(
        struct ptd_graph *graph,
        int granularity
) {
    double max_rate = 512;

    for (size_t i = 0; i < graph->vertices_length; ++i) {
        double rate = 0;

        struct ptd_vertex *vertex = graph->vertices[i];

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            rate += vertex->edges[j]->weight;
        }

        if (rate > max_rate) {
            max_rate = rate;
        }
    }

    if (granularity == 0) {
        granularity = max_rate * 2;
    }

    for (size_t i = 0; i < graph->vertices_length; ++i) {
        double rate = 0;

        struct ptd_vertex *vertex = graph->vertices[i];

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            rate += vertex->edges[j]->weight;
        }

        if (rate / granularity > 1.0001) {
            size_t debug_index = vertex->index;

            if (PTD_DEBUG_1_INDEX) {
                debug_index++;
            }

            char state[1024] = {'\0'};
            char starting_vertex[] = " (starting vertex)";

            if (vertex != graph->starting_vertex) {
                starting_vertex[0] = '\0';
            }

            ptd_vertex_to_s(vertex, state, 1023);

            snprintf(
                    (char *) ptd_err,
                    sizeof(ptd_err),
                    "Expected vertex with index %i%s (state %s) to have outgoing rate divided by granularity <= 1. Rate is '%f' ('%f'). Increase the granularity\n",
                    (int) debug_index, starting_vertex, state, (float) rate, (float) (rate / granularity)
            );

            return NULL;
        }
    }

    struct ptd_probability_distribution_context *res = (struct ptd_probability_distribution_context *)
            malloc(sizeof(*res));

    struct ptd_dph_probability_distribution_context *dph_res =
            _ptd_dph_probability_distribution_context_create(graph, true);
    dph_res->priv3 = (double) 1.0 / granularity;

    long double cdf1 = dph_res->cdf * granularity;

    ptd_dph_probability_distribution_step(dph_res);

    long double cdf2 = dph_res->cdf * granularity;

    ptd_dph_probability_distribution_context_destroy(dph_res);

    dph_res = _ptd_dph_probability_distribution_context_create(graph, true);
    dph_res->priv3 = (double) 1.0 / granularity;

    res->cdf = dph_res->cdf;
    res->pdf = (double) ((cdf2 - cdf1));
    res->graph = dph_res->graph;
    res->probability_at = dph_res->probability_at;
    res->accumulated_visits = dph_res->accumulated_visits;

    res->time = 0;
    res->priv = (void *) dph_res;
    res->granularity = granularity;

    return res;
}

void ptd_probability_distribution_context_destroy(struct ptd_probability_distribution_context *context) {
    if (context == NULL) {
        return;
    }

    ptd_dph_probability_distribution_context_destroy(
            (struct ptd_dph_probability_distribution_context *) context->priv
    );

    free(context);
}

int ptd_probability_distribution_step(
        struct ptd_probability_distribution_context *context
) {
    struct ptd_dph_probability_distribution_context *dph_context =
            (struct ptd_dph_probability_distribution_context *) context->priv;

    ptd_dph_probability_distribution_step(dph_context);

    context->time = ((long double) dph_context->jumps) / context->granularity;
    context->cdf = dph_context->cdf;
    context->pdf = dph_context->pmf * context->granularity;

    return 0;
}

/**
 * Helper: Allocate 2D array
 */
static double **alloc_2d(size_t rows, size_t cols) {
    double **arr = (double **)malloc(rows * sizeof(double*));
    if (arr == NULL) return NULL;

    for (size_t i = 0; i < rows; i++) {
        arr[i] = (double *)calloc(cols, sizeof(double));
        if (arr[i] == NULL) {
            for (size_t j = 0; j < i; j++) free(arr[j]);
            free(arr);
            return NULL;
        }
    }
    return arr;
}

/**
 * Helper: Free 2D array
 */
static void free_2d(double **arr, size_t rows) {
    if (arr == NULL) return;
    for (size_t i = 0; i < rows; i++) {
        free(arr[i]);
    }
    free(arr);
}

/**
 * Helper: Compute PMF with gradient tracking
 * Returns PMF(time) and ∇PMF(time) using uniformization
 * PMF = Σ_k Poisson(k; λt) * P(absorption at step k)
 */
static int compute_pmf_with_gradient(
    struct ptd_graph *graph,
    double time,
    double lambda,
    size_t granularity,
    const double *params,
    size_t n_params,
    double *pmf_value,
    double *pmf_gradient
) {
    if (graph == NULL || params == NULL || pmf_value == NULL || pmf_gradient == NULL) {
        return -1;
    }

    size_t max_jumps = (size_t)(granularity * time * lambda) + 100;

    // Initialize probability and gradient arrays
    double *prob = (double *)calloc(graph->vertices_length, sizeof(double));
    double **prob_grad = alloc_2d(graph->vertices_length, n_params);

    if (prob == NULL || prob_grad == NULL) {
        free(prob);
        free_2d(prob_grad, graph->vertices_length);
        return -1;
    }

    // Starting vertex has probability 1, gradient 0
    prob[0] = 1.0;

    // Initialize output accumulators
    *pmf_value = 0.0;
    for (size_t i = 0; i < n_params; i++) {
        pmf_gradient[i] = 0.0;
    }

    // Precompute Poisson probabilities
    double *poisson_cache = (double *)malloc(max_jumps * sizeof(double));
    if (poisson_cache == NULL) {
        free(prob);
        free_2d(prob_grad, graph->vertices_length);
        return -1;
    }

    double lambda_t = lambda * time;
    for (size_t k = 0; k < max_jumps; k++) {
        poisson_cache[k] = exp(-lambda_t + k * log(lambda_t) - lgamma(k + 1));
    }

    // DP iteration
    for (size_t k = 0; k < max_jumps; k++) {
        double *next_prob = (double *)calloc(graph->vertices_length, sizeof(double));
        double **next_prob_grad = alloc_2d(graph->vertices_length, n_params);

        if (next_prob == NULL || next_prob_grad == NULL) {
            free(next_prob);
            free_2d(next_prob_grad, graph->vertices_length);
            free(prob);
            free_2d(prob_grad, graph->vertices_length);
            free(poisson_cache);
            return -1;
        }

        // Forward step
        for (size_t v = 0; v < graph->vertices_length; v++) {
            struct ptd_vertex *vertex = graph->vertices[v];

            // Compute exit rate and gradient for self-loop
            double exit_rate = 0.0;
            double *exit_rate_grad = (double *)calloc(n_params, sizeof(double));
            if (exit_rate_grad == NULL) {
                free(next_prob);
                free_2d(next_prob_grad, graph->vertices_length);
                free(prob);
                free_2d(prob_grad, graph->vertices_length);
                free(poisson_cache);
                return -1;
            }

            for (size_t e = 0; e < vertex->edges_length; e++) {
                struct ptd_edge *edge = vertex->edges[e];
                if (edge->parameterized) {
                    struct ptd_edge_parameterized *ep = (struct ptd_edge_parameterized *)edge;
                    double w = ep->base_weight;  // Use base weight for gradient computation
                    if (ep->state != NULL) {
                        for (size_t i = 0; i < n_params; i++) {
                            w += ep->state[i] * params[i];
                            exit_rate_grad[i] += ep->state[i];
                        }
                    }
                    exit_rate += w;
                } else {
                    exit_rate += edge->weight;
                }
            }

            // Process outgoing edges
            for (size_t e = 0; e < vertex->edges_length; e++) {
                struct ptd_edge *edge = vertex->edges[e];

                size_t to_idx = 0;
                for (size_t i = 0; i < graph->vertices_length; i++) {
                    if (graph->vertices[i] == edge->to) {
                        to_idx = i;
                        break;
                    }
                }

                double weight;
                double *weight_grad = (double *)calloc(n_params, sizeof(double));
                if (weight_grad == NULL) {
                    free(exit_rate_grad);
                    free(next_prob);
                    free_2d(next_prob_grad, graph->vertices_length);
                    free(prob);
                    free_2d(prob_grad, graph->vertices_length);
                    free(poisson_cache);
                    return -1;
                }

                if (edge->parameterized) {
                    struct ptd_edge_parameterized *ep = (struct ptd_edge_parameterized *)edge;
                    weight = ep->base_weight;  // Use base weight for gradient computation
                    if (ep->state != NULL) {
                        for (size_t i = 0; i < n_params; i++) {
                            weight += ep->state[i] * params[i];
                            weight_grad[i] = ep->state[i];
                        }
                    }
                } else {
                    weight = edge->weight;
                }

                next_prob[to_idx] += prob[v] * weight / lambda;

                for (size_t i = 0; i < n_params; i++) {
                    next_prob_grad[to_idx][i] +=
                        prob_grad[v][i] * weight / lambda +
                        prob[v] * weight_grad[i] / lambda;
                }

                free(weight_grad);
            }

            // Self-loop
            double self_prob = (lambda - exit_rate) / lambda;
            next_prob[v] += prob[v] * self_prob;

            for (size_t i = 0; i < n_params; i++) {
                next_prob_grad[v][i] +=
                    prob_grad[v][i] * self_prob +
                    prob[v] * (-exit_rate_grad[i]) / lambda;
            }

            free(exit_rate_grad);
        }

        // Swap buffers
        free(prob);
        free_2d(prob_grad, graph->vertices_length);
        prob = next_prob;
        prob_grad = next_prob_grad;

        // Accumulate PMF contributions from absorbing states
        for (size_t i = 0; i < graph->vertices_length; i++) {
            struct ptd_vertex *v = graph->vertices[i];
            if (v->edges_length == 0 && i > 0) {
                double poisson_k = poisson_cache[k];
                *pmf_value += poisson_k * prob[i];

                for (size_t p = 0; p < n_params; p++) {
                    pmf_gradient[p] += poisson_k * prob_grad[i][p];
                }

                // CRITICAL: Zero out absorbed probability (pattern from line 4559)
                prob[i] = 0;
                for (size_t p = 0; p < n_params; p++) {
                    prob_grad[i][p] = 0;
                }
            }
        }

        if (k > 10 && poisson_cache[k] < 1e-12) {
            break;
        }
    }

    free(prob);
    free_2d(prob_grad, graph->vertices_length);
    free(poisson_cache);

    return 0;
}

/**
 * Forward algorithm with gradient tracking
 * Uses uniformization to compute PDF = PMF * granularity
 */
int ptd_graph_pdf_with_gradient(
    struct ptd_graph *graph,
    double time,
    size_t granularity,
    const double *params,
    size_t n_params,
    double *pdf_value,
    double *pdf_gradient
) {
    if (graph == NULL || params == NULL || pdf_value == NULL || pdf_gradient == NULL) {
        return -1;
    }

    // 1. Compute uniformization rate (max exit rate across all vertices)
    double lambda = 0.0;
    for (size_t i = 0; i < graph->vertices_length; i++) {
        struct ptd_vertex *v = graph->vertices[i];
        double exit_rate = 0.0;

        for (size_t j = 0; j < v->edges_length; j++) {
            struct ptd_edge *e = v->edges[j];

            if (e->parameterized) {
                struct ptd_edge_parameterized *ep = (struct ptd_edge_parameterized *)e;
                double weight = ep->weight;
                if (ep->state != NULL) {
                    for (size_t k = 0; k < n_params; k++) {
                        weight += ep->state[k] * params[k];
                    }
                }
                exit_rate += weight;
            } else {
                exit_rate += e->weight;
            }
        }

        if (exit_rate > lambda) {
            lambda = exit_rate;
        }
    }

    if (lambda <= 0.0) {
        *pdf_value = 0.0;
        for (size_t i = 0; i < n_params; i++) {
            pdf_gradient[i] = 0.0;
        }
        return 0;
    }

    // 2. Determine granularity (auto-select if not specified)
    if (granularity == 0) {
        granularity = (size_t)(lambda * 2.0);
        if (granularity < 100) granularity = 100;
    }

    // 3. Compute PMF and its gradient
    double pmf;
    double *pmf_grad = (double *)malloc(n_params * sizeof(double));
    if (pmf_grad == NULL) {
        return -1;
    }

    int status = compute_pmf_with_gradient(graph, time, lambda, granularity,
                                          params, n_params, &pmf, pmf_grad);

    if (status != 0) {
        free(pmf_grad);
        return -1;
    }

    // 4. Convert PMF to PDF: PDF = PMF * lambda
    //    In uniformization: dt = 1/lambda, so PDF = PMF / dt = PMF * lambda
    *pdf_value = pmf * lambda;
    for (size_t i = 0; i < n_params; i++) {
        pdf_gradient[i] = pmf_grad[i] * lambda;
    }

    free(pmf_grad);
    return 0;
}

/**
 * Compute PDF for parameterized graph using current parameters
 */
int ptd_graph_pdf_parameterized(
    struct ptd_graph *graph,
    double time,
    size_t granularity,
    double *pdf_value,
    double *pdf_gradient
) {
    // Validate inputs
    if (graph == NULL || pdf_value == NULL) {
        sprintf((char*)ptd_err, "ptd_graph_pdf_parameterized: graph or pdf_value is NULL");
        return -1;
    }

    // Check if graph is parameterized
    if (!graph->parameterized) {
        sprintf((char*)ptd_err, "ptd_graph_pdf_parameterized: graph is not parameterized");
        return -1;
    }

    // Check if parameters have been set
    if (graph->current_params == NULL) {
        sprintf((char*)ptd_err, "ptd_graph_pdf_parameterized: parameters not set. "
                "Call ptd_graph_update_weight_parameterized() first");
        return -1;
    }

    if (graph->param_length == 0) {
        sprintf((char*)ptd_err, "ptd_graph_pdf_parameterized: param_length is 0");
        return -1;
    }

    // If gradients requested, use gradient-aware function
    if (pdf_gradient != NULL) {
        return ptd_graph_pdf_with_gradient(
            graph,
            time,
            granularity,
            graph->current_params,
            graph->param_length,
            pdf_value,
            pdf_gradient
        );
    }

    // Otherwise, fall back to gradient computation anyway
    // (There's no separate PDF-only function for parameterized graphs at C level)
    // The Python/C++ layers handle this through reward_compute_graph
    // For now, just compute with gradients and ignore them internally
    double *temp_gradient = (double*)malloc(graph->param_length * sizeof(double));
    if (temp_gradient == NULL) {
        sprintf((char*)ptd_err, "ptd_graph_pdf_parameterized: failed to allocate temp gradient");
        return -1;
    }

    int result = ptd_graph_pdf_with_gradient(
        graph,
        time,
        granularity,
        graph->current_params,
        graph->param_length,
        pdf_value,
        temp_gradient
    );

    free(temp_gradient);
    return result;
}

double ptd_defect(struct ptd_graph *graph) {
    double rate = 0;

    for (size_t i = 0; i < graph->starting_vertex->edges_length; ++i) {
        struct ptd_edge *edge = graph->starting_vertex->edges[i];

        rate += edge->weight;
    }

    double defect = 0;

    for (size_t i = 0; i < graph->starting_vertex->edges_length; ++i) {
        struct ptd_edge *edge = graph->starting_vertex->edges[i];

        if (edge->to->edges_length == 0) {
            defect += edge->weight / rate;
        }
    }

    return defect;
}

struct ptd_clone_res ptd_clone_graph(struct ptd_graph *graph, struct ptd_avl_tree *avl_tree) {
    struct ptd_graph *res = ptd_graph_create(graph->state_length);

    for (size_t i = 1; i < graph->vertices_length; ++i) {
        ptd_vertex_create(res);
    }

    for (size_t i = 0; i < graph->vertices_length; ++i) {
        struct ptd_vertex *v = graph->vertices[i];
        struct ptd_vertex *v2 = res->vertices[i];

        if (v->state != NULL) {
            memcpy(v2->state, v->state, sizeof(int) * res->state_length);
        }
    }

    for (size_t i = 0; i < graph->vertices_length; ++i) {
        struct ptd_vertex *v = graph->vertices[i];
        struct ptd_vertex *v2 = res->vertices[i];

        for (size_t j = 0; j < v->edges_length; ++j) {
            struct ptd_edge *e = v->edges[j];

            if (e->parameterized) {
                struct ptd_edge_parameterized *param_e = (struct ptd_edge_parameterized *) e;
                ptd_graph_add_edge_parameterized(
                        v2,
                        res->vertices[e->to->index],
                        e->weight,
                        param_e->state,
                        param_e->state_length
                )->should_free_state = false;
            } else {
                ptd_graph_add_edge(v2, res->vertices[e->to->index], e->weight);
            }
        }
    }

    struct ptd_avl_tree *new_tree = ptd_avl_tree_create(avl_tree->key_length);

    struct ptd_stack *stack = stack_create();

    stack_push(stack, avl_tree->root);

    while (!stack_empty(stack)) {
        struct ptd_avl_node *v = (struct ptd_avl_node *) stack_pop(stack);

        if (v == NULL) {
            continue;
        }

        ptd_avl_tree_find_or_insert(
                new_tree,
                v->key,
                res->vertices[((struct ptd_vertex *) v->entry)->index]
        );

        stack_push(stack, v->left);
        stack_push(stack, v->right);
    }

    stack_destroy(stack);

    struct ptd_clone_res ret;
    ret.graph = res;
    ret.avl_tree = new_tree;

    return ret;
}

/*
 * Utilities
 */


static struct ptd_vector *vector_create() {
    struct ptd_vector *vector = (struct ptd_vector *) malloc(sizeof(*vector));

    vector->entries = 0;
    vector->arr = NULL;

    return vector;
}

static int vector_add(struct ptd_vector *vector, void *entry) {
    bool is_power_of_2 = (vector->entries & (vector->entries - 1)) == 0;

    if (is_power_of_2) {
        size_t new_length = vector->entries == 0 ? 1 : vector->entries * 2;

        if ((vector->arr = (void **) realloc(
                vector->arr,
                new_length * sizeof(void *))
            ) == NULL) {
            return -1;
        }
    }

    vector->arr[vector->entries] = entry;
    vector->entries++;

    return 0;
}

static void *vector_get(struct ptd_vector *vector, size_t index) {
    return vector->arr[index];
}

static size_t vector_length(struct ptd_vector *vector) {
    return vector->entries;
}

static void vector_destroy(struct ptd_vector *vector) {
    free(vector->arr);
    free(vector);
}

static struct ptd_queue *queue_create() {
    struct ptd_queue *queue = (struct ptd_queue *) malloc(sizeof(struct ptd_queue));

    queue->ll = NULL;
    queue->tail = NULL;

    return queue;
}

static void queue_destroy(struct ptd_queue *queue) {
    free(queue->ll);
    free(queue);
}

static int queue_enqueue(struct ptd_queue *queue, void *entry) {
    struct ptd_ll *new_ll = (struct ptd_ll *) malloc(sizeof(*new_ll));
    new_ll->next = NULL;
    new_ll->value = entry;

    if (queue->tail != NULL) {
        queue->tail->next = new_ll;
    } else {
        queue->ll = new_ll;
    }

    queue->tail = new_ll;

    return 0;
}

static void *queue_dequeue(struct ptd_queue *queue) {
    void *result = queue->ll->value;
    struct ptd_ll *value = queue->ll;
    queue->ll = queue->ll->next;

    if (queue->tail == value) {
        queue->tail = NULL;
    }

    free(value);

    return result;
}

static int queue_empty(struct ptd_queue *queue) {
    return (queue->tail == NULL);
}

static struct ptd_stack *stack_create() {
    struct ptd_stack *stack = (struct ptd_stack *) malloc(sizeof(struct ptd_stack));
    stack->ll = NULL;

    return stack;
}

static void stack_destroy(struct ptd_stack *stack) {
    free(stack->ll);
    free(stack);
}

static int stack_push(struct ptd_stack *stack, void *entry) {
    struct ptd_ll *new_ll = (struct ptd_ll *) malloc(sizeof(*new_ll));
    new_ll->next = stack->ll;
    new_ll->value = entry;

    stack->ll = new_ll;

    return 0;
}

static void *stack_pop(struct ptd_stack *stack) {
    void *result = stack->ll->value;
    struct ptd_ll *ll = stack->ll;

    stack->ll = stack->ll->next;
    free(ll);

    return result;
}

static int stack_empty(struct ptd_stack *stack) {
    return (stack->ll == NULL);
}

// ============================================================================
// Symbolic Expression System Implementation
// ============================================================================

/**
 * Create a constant expression node
 */
struct ptd_expression *ptd_expr_const(double value) {
    struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate memory for constant expression");
    }
    expr->type = PTD_EXPR_CONST;
    expr->const_value = value;
    return expr;
}

/**
 * Create a parameter reference expression node
 */
struct ptd_expression *ptd_expr_param(size_t param_idx) {
    struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate memory for parameter expression");
    }
    expr->type = PTD_EXPR_PARAM;
    expr->param_index = param_idx;
    return expr;
}

/**
 * Create a dot product expression node (optimized for linear combinations)
 */
struct ptd_expression *ptd_expr_dot(const size_t *indices, const double *coeffs, size_t n) {
    struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate memory for dot expression");
    }
    expr->type = PTD_EXPR_DOT;
    expr->n_terms = n;

    // Allocate and copy indices
    expr->param_indices = (size_t *) malloc(n * sizeof(size_t));
    if (expr->param_indices == NULL) {
        free(expr);
        DIE_ERROR(1, "Failed to allocate memory for dot expression indices");
    }
    memcpy(expr->param_indices, indices, n * sizeof(size_t));

    // Allocate and copy coefficients
    expr->coefficients = (double *) malloc(n * sizeof(double));
    if (expr->coefficients == NULL) {
        free(expr->param_indices);
        free(expr);
        DIE_ERROR(1, "Failed to allocate memory for dot expression coefficients");
    }
    memcpy(expr->coefficients, coeffs, n * sizeof(double));

    return expr;
}

/**
 * Create an addition expression node
 */
struct ptd_expression *ptd_expr_add(struct ptd_expression *left, struct ptd_expression *right) {
    // Simplification: 0 + x = x
    if (left->type == PTD_EXPR_CONST && left->const_value == 0.0) {
        ptd_expr_destroy_iterative(left);
        return right;
    }
    if (right->type == PTD_EXPR_CONST && right->const_value == 0.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }

    // Constant folding: c1 + c2 = c3
    if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
        double result = left->const_value + right->const_value;
        ptd_expr_destroy_iterative(left);
        ptd_expr_destroy_iterative(right);
        return ptd_expr_const(result);
    }

    // Original allocation
    struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate memory for addition expression");
    }
    expr->type = PTD_EXPR_ADD;
    expr->left = left;
    expr->right = right;
    return expr;
}

/**
 * Create a multiplication expression node
 */
struct ptd_expression *ptd_expr_mul(struct ptd_expression *left, struct ptd_expression *right) {
    // Simplification: 0 * x = 0
    if (left->type == PTD_EXPR_CONST && left->const_value == 0.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }
    if (right->type == PTD_EXPR_CONST && right->const_value == 0.0) {
        ptd_expr_destroy_iterative(left);
        return right;
    }

    // Simplification: 1 * x = x
    if (left->type == PTD_EXPR_CONST && left->const_value == 1.0) {
        ptd_expr_destroy_iterative(left);
        return right;
    }
    if (right->type == PTD_EXPR_CONST && right->const_value == 1.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }

    // Constant folding: c1 * c2 = c3
    if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
        double result = left->const_value * right->const_value;
        ptd_expr_destroy_iterative(left);
        ptd_expr_destroy_iterative(right);
        return ptd_expr_const(result);
    }

    // Original allocation
    struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate memory for multiplication expression");
    }
    expr->type = PTD_EXPR_MUL;
    expr->left = left;
    expr->right = right;
    return expr;
}

/**
 * Create a division expression node
 */
struct ptd_expression *ptd_expr_div(struct ptd_expression *left, struct ptd_expression *right) {
    // Simplification: 0 / x = 0 (x != 0)
    if (left->type == PTD_EXPR_CONST && left->const_value == 0.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }

    // Simplification: x / 1 = x
    if (right->type == PTD_EXPR_CONST && right->const_value == 1.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }

    // Constant folding: c1 / c2 = c3
    if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
        if (right->const_value == 0.0) {
            DIE_ERROR(1, "Division by zero in constant folding");
        }
        double result = left->const_value / right->const_value;
        ptd_expr_destroy_iterative(left);
        ptd_expr_destroy_iterative(right);
        return ptd_expr_const(result);
    }

    // Original allocation
    struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate memory for division expression");
    }
    expr->type = PTD_EXPR_DIV;
    expr->left = left;
    expr->right = right;
    return expr;
}

/**
 * Create an inversion expression node (1/x)
 */
struct ptd_expression *ptd_expr_inv(struct ptd_expression *child) {
    struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate memory for inversion expression");
    }
    expr->type = PTD_EXPR_INV;
    expr->left = child;  // Use left for unary operations
    return expr;
}

/**
 * Create a subtraction expression node
 */
struct ptd_expression *ptd_expr_sub(struct ptd_expression *left, struct ptd_expression *right) {
    // Simplification: x - 0 = x
    if (right->type == PTD_EXPR_CONST && right->const_value == 0.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }

    // Constant folding: c1 - c2 = c3
    if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
        double result = left->const_value - right->const_value;
        ptd_expr_destroy_iterative(left);
        ptd_expr_destroy_iterative(right);
        return ptd_expr_const(result);
    }

    // Original allocation
    struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate memory for subtraction expression");
    }
    expr->type = PTD_EXPR_SUB;
    expr->left = left;
    expr->right = right;
    return expr;
}

/**
 * Stack entry for iterative expression copying
 */
struct ptd_expr_copy_stack_entry {
    const struct ptd_expression *src;      // Source node to copy
    struct ptd_expression **dst_location;  // Where to store the copy pointer
    bool processed;                        // Children already copied?
};

/**
 * Deep copy an expression tree (iterative version to avoid stack overflow)
 */
struct ptd_expression *ptd_expr_copy_iterative(const struct ptd_expression *expr) {
    if (expr == NULL) {
        return NULL;
    }

    // Explicit stack for iterative traversal
    size_t stack_capacity = 256;
    size_t stack_size = 0;
    struct ptd_expr_copy_stack_entry *stack = (struct ptd_expr_copy_stack_entry *)
        malloc(stack_capacity * sizeof(struct ptd_expr_copy_stack_entry));

    if (stack == NULL) {
        DIE_ERROR(1, "Failed to allocate copy stack");
    }

    struct ptd_expression *root = NULL;

    // Push root onto stack
    stack[stack_size++] = (struct ptd_expr_copy_stack_entry){
        .src = expr,
        .dst_location = &root,
        .processed = false
    };

    while (stack_size > 0) {
        struct ptd_expr_copy_stack_entry *entry = &stack[stack_size - 1];

        if (entry->processed) {
            // This node and its children are done
            stack_size--;
            continue;
        }

        // Allocate copy for this node
        struct ptd_expression *copy = (struct ptd_expression *)
            calloc(1, sizeof(struct ptd_expression));
        if (copy == NULL) {
            free(stack);
            DIE_ERROR(1, "Failed to allocate memory for expression copy");
        }

        copy->type = entry->src->type;
        *(entry->dst_location) = copy;

        // Mark as processed before pushing children
        entry->processed = true;

        // Handle node type and push children if needed
        switch (entry->src->type) {
            case PTD_EXPR_CONST:
                copy->const_value = entry->src->const_value;
                break;

            case PTD_EXPR_PARAM:
                copy->param_index = entry->src->param_index;
                break;

            case PTD_EXPR_DOT:
                copy->n_terms = entry->src->n_terms;
                copy->param_indices = (size_t *) malloc(entry->src->n_terms * sizeof(size_t));
                copy->coefficients = (double *) malloc(entry->src->n_terms * sizeof(double));
                if (copy->param_indices == NULL || copy->coefficients == NULL) {
                    free(copy->param_indices);
                    free(copy->coefficients);
                    free(copy);
                    free(stack);
                    DIE_ERROR(1, "Failed to allocate memory for dot expression copy");
                }
                memcpy(copy->param_indices, entry->src->param_indices,
                      entry->src->n_terms * sizeof(size_t));
                memcpy(copy->coefficients, entry->src->coefficients,
                      entry->src->n_terms * sizeof(double));
                break;

            case PTD_EXPR_INV:
                if (entry->src->left != NULL) {
                    // Grow stack if needed
                    if (stack_size >= stack_capacity) {
                        stack_capacity *= 2;
                        struct ptd_expr_copy_stack_entry *new_stack =
                            (struct ptd_expr_copy_stack_entry *)
                            realloc(stack, stack_capacity * sizeof(struct ptd_expr_copy_stack_entry));
                        if (new_stack == NULL) {
                            free(stack);
                            DIE_ERROR(1, "Failed to grow copy stack");
                        }
                        stack = new_stack;
                        entry = &stack[stack_size - 1];  // Re-point after realloc
                    }

                    // Push left child
                    stack[stack_size++] = (struct ptd_expr_copy_stack_entry){
                        .src = entry->src->left,
                        .dst_location = &copy->left,
                        .processed = false
                    };
                }
                break;

            case PTD_EXPR_ADD:
            case PTD_EXPR_MUL:
            case PTD_EXPR_DIV:
            case PTD_EXPR_SUB:
                // Grow stack if needed for 2 children
                while (stack_size + 2 > stack_capacity) {
                    stack_capacity *= 2;
                    struct ptd_expr_copy_stack_entry *new_stack =
                        (struct ptd_expr_copy_stack_entry *)
                        realloc(stack, stack_capacity * sizeof(struct ptd_expr_copy_stack_entry));
                    if (new_stack == NULL) {
                        free(stack);
                        DIE_ERROR(1, "Failed to grow copy stack");
                    }
                    stack = new_stack;
                    entry = &stack[stack_size - 1];  // Re-point after realloc
                }

                // Push children (right first, then left for proper ordering)
                if (entry->src->right != NULL) {
                    stack[stack_size++] = (struct ptd_expr_copy_stack_entry){
                        .src = entry->src->right,
                        .dst_location = &copy->right,
                        .processed = false
                    };
                }
                if (entry->src->left != NULL) {
                    stack[stack_size++] = (struct ptd_expr_copy_stack_entry){
                        .src = entry->src->left,
                        .dst_location = &copy->left,
                        .processed = false
                    };
                }
                break;

            default:
                free(copy);
                free(stack);
                DIE_ERROR(1, "Unknown expression type in ptd_expr_copy_iterative");
        }
    }

    free(stack);
    return root;
}

/**
 * Deep copy an expression tree (recursive version - kept for compatibility)
 * WARNING: May cause stack overflow for deeply nested expressions (>1000 levels)
 * Use ptd_expr_copy_iterative() for deep trees
 */
struct ptd_expression *ptd_expr_copy(const struct ptd_expression *expr) {
    if (expr == NULL) {
        return NULL;
    }

    struct ptd_expression *copy = (struct ptd_expression *) calloc(1, sizeof(*copy));
    if (copy == NULL) {
        DIE_ERROR(1, "Failed to allocate memory for expression copy");
    }

    copy->type = expr->type;

    switch (expr->type) {
        case PTD_EXPR_CONST:
            copy->const_value = expr->const_value;
            break;

        case PTD_EXPR_PARAM:
            copy->param_index = expr->param_index;
            break;

        case PTD_EXPR_DOT:
            copy->n_terms = expr->n_terms;
            copy->param_indices = (size_t *) malloc(expr->n_terms * sizeof(size_t));
            copy->coefficients = (double *) malloc(expr->n_terms * sizeof(double));
            if (copy->param_indices == NULL || copy->coefficients == NULL) {
                free(copy->param_indices);
                free(copy->coefficients);
                free(copy);
                DIE_ERROR(1, "Failed to allocate memory for dot expression copy");
            }
            memcpy(copy->param_indices, expr->param_indices, expr->n_terms * sizeof(size_t));
            memcpy(copy->coefficients, expr->coefficients, expr->n_terms * sizeof(double));
            break;

        case PTD_EXPR_INV:
            copy->left = ptd_expr_copy(expr->left);
            break;

        case PTD_EXPR_ADD:
        case PTD_EXPR_MUL:
        case PTD_EXPR_DIV:
        case PTD_EXPR_SUB:
            copy->left = ptd_expr_copy(expr->left);
            copy->right = ptd_expr_copy(expr->right);
            break;

        default:
            free(copy);
            DIE_ERROR(1, "Unknown expression type in ptd_expr_copy");
    }

    return copy;
}

/**
 * Stack entry for iterative expression destruction
 */
struct ptd_expr_destroy_stack_entry {
    struct ptd_expression *expr;
    bool children_pushed;
};

/**
 * Destroy an expression tree and free all memory (iterative version, O(n))
 */
void ptd_expr_destroy_iterative(struct ptd_expression *expr) {
    if (expr == NULL) {
        return;
    }

    // Stack for post-order destruction
    size_t stack_capacity = 256;
    size_t stack_size = 0;
    struct ptd_expr_destroy_stack_entry *stack =
        (struct ptd_expr_destroy_stack_entry *)
        malloc(stack_capacity * sizeof(struct ptd_expr_destroy_stack_entry));

    if (stack == NULL) {
        DIE_ERROR(1, "Failed to allocate destruction stack");
    }

    // Push root
    stack[stack_size++] = (struct ptd_expr_destroy_stack_entry){
        .expr = expr,
        .children_pushed = false
    };

    while (stack_size > 0) {
        struct ptd_expr_destroy_stack_entry *entry = &stack[stack_size - 1];

        if (!entry->children_pushed) {
            // First visit: push children
            entry->children_pushed = true;
            struct ptd_expression *node = entry->expr;

            // Grow stack if needed (max 2 children)
            if (stack_size + 2 > stack_capacity) {
                stack_capacity *= 2;
                struct ptd_expr_destroy_stack_entry *new_stack =
                    (struct ptd_expr_destroy_stack_entry *)
                    realloc(stack, stack_capacity * sizeof(struct ptd_expr_destroy_stack_entry));
                if (new_stack == NULL) {
                    free(stack);
                    DIE_ERROR(1, "Failed to grow destruction stack");
                }
                stack = new_stack;
                entry = &stack[stack_size - 1];  // Re-point after realloc
            }

            // Push children (right first for left-to-right processing)
            switch (node->type) {
                case PTD_EXPR_INV:
                    if (node->left != NULL) {
                        stack[stack_size++] = (struct ptd_expr_destroy_stack_entry){
                            .expr = node->left,
                            .children_pushed = false
                        };
                    }
                    break;

                case PTD_EXPR_ADD:
                case PTD_EXPR_MUL:
                case PTD_EXPR_DIV:
                case PTD_EXPR_SUB:
                    if (node->right != NULL) {
                        stack[stack_size++] = (struct ptd_expr_destroy_stack_entry){
                            .expr = node->right,
                            .children_pushed = false
                        };
                    }
                    if (node->left != NULL) {
                        stack[stack_size++] = (struct ptd_expr_destroy_stack_entry){
                            .expr = node->left,
                            .children_pushed = false
                        };
                    }
                    break;

                default:
                    // Leaf nodes (CONST, PARAM, DOT) - no children
                    break;
            }
        } else {
            // Second visit: children are done, destroy this node
            struct ptd_expression *node = entry->expr;
            stack_size--;

            // Free node-specific data
            if (node->type == PTD_EXPR_DOT) {
                free(node->param_indices);
                free(node->coefficients);
            }

            free(node);
        }
    }

    free(stack);
}

/**
 * Destroy an expression tree and free all memory (recursive version - kept for compatibility)
 * WARNING: May cause stack overflow for deeply nested expressions (>1000 levels)
 * Use ptd_expr_destroy_iterative() for deep trees
 */
void ptd_expr_destroy(struct ptd_expression *expr) {
    if (expr == NULL) {
        return;
    }

    // Recursively destroy children
    switch (expr->type) {
        case PTD_EXPR_INV:
            ptd_expr_destroy(expr->left);
            break;

        case PTD_EXPR_ADD:
        case PTD_EXPR_MUL:
        case PTD_EXPR_DIV:
        case PTD_EXPR_SUB:
            ptd_expr_destroy(expr->left);
            ptd_expr_destroy(expr->right);
            break;

        case PTD_EXPR_DOT:
            free(expr->param_indices);
            free(expr->coefficients);
            break;

        case PTD_EXPR_CONST:
        case PTD_EXPR_PARAM:
            // No children or allocated arrays
            break;

        default:
            // Unknown type, but still free the node
            break;
    }

    free(expr);
}

// =============================================================================
// Expression Hashing and Equality (for CSE - Common Subexpression Elimination)
// =============================================================================

/**
 * Compute structural hash of expression tree
 *
 * Uses FNV-1a-like hash with type and value mixing.
 * For commutative operations (ADD, MUL), sorts child hashes for consistency.
 */
uint64_t ptd_expr_hash(const struct ptd_expression *expr) {
    if (expr == NULL) return 0;

    uint64_t hash = 14695981039346656037ULL;  // FNV offset basis
    const uint64_t prime = 1099511628211ULL;  // FNV prime

    // Mix in type
    hash ^= (uint64_t)expr->type;
    hash *= prime;

    switch (expr->type) {
        case PTD_EXPR_CONST: {
            // Hash double value by reinterpreting bits
            uint64_t bits;
            memcpy(&bits, &expr->const_value, sizeof(uint64_t));
            hash ^= bits;
            hash *= prime;
            break;
        }

        case PTD_EXPR_PARAM:
            hash ^= expr->param_index;
            hash *= prime;
            break;

        case PTD_EXPR_DOT:
            hash ^= expr->n_terms;
            hash *= prime;
            for (size_t i = 0; i < expr->n_terms; i++) {
                hash ^= expr->param_indices[i];
                hash *= prime;

                uint64_t coeff_bits;
                memcpy(&coeff_bits, &expr->coefficients[i], sizeof(uint64_t));
                hash ^= coeff_bits;
                hash *= prime;
            }
            break;

        case PTD_EXPR_INV:
            hash ^= ptd_expr_hash(expr->left);
            hash *= prime;
            break;

        case PTD_EXPR_ADD:
        case PTD_EXPR_MUL:
        case PTD_EXPR_DIV:
        case PTD_EXPR_SUB: {
            uint64_t left_hash = ptd_expr_hash(expr->left);
            uint64_t right_hash = ptd_expr_hash(expr->right);

            // Commutative operations: sort hashes for consistency
            if (expr->type == PTD_EXPR_ADD || expr->type == PTD_EXPR_MUL) {
                if (left_hash > right_hash) {
                    uint64_t tmp = left_hash;
                    left_hash = right_hash;
                    right_hash = tmp;
                }
            }

            hash ^= left_hash;
            hash *= prime;
            hash ^= right_hash;
            hash *= prime;
            break;
        }
    }

    return hash;
}

/**
 * Check structural equality of two expressions
 *
 * Performs deep comparison, handling commutativity of ADD and MUL.
 */
bool ptd_expr_equal(const struct ptd_expression *a, const struct ptd_expression *b) {
    if (a == b) return true;
    if (a == NULL || b == NULL) return false;
    if (a->type != b->type) return false;

    switch (a->type) {
        case PTD_EXPR_CONST:
            return a->const_value == b->const_value;

        case PTD_EXPR_PARAM:
            return a->param_index == b->param_index;

        case PTD_EXPR_DOT:
            if (a->n_terms != b->n_terms) return false;
            for (size_t i = 0; i < a->n_terms; i++) {
                if (a->param_indices[i] != b->param_indices[i]) return false;
                if (a->coefficients[i] != b->coefficients[i]) return false;
            }
            return true;

        case PTD_EXPR_INV:
            return ptd_expr_equal(a->left, b->left);

        case PTD_EXPR_ADD:
        case PTD_EXPR_MUL:
            // Commutative: check both orderings
            return (ptd_expr_equal(a->left, b->left) && ptd_expr_equal(a->right, b->right)) ||
                   (ptd_expr_equal(a->left, b->right) && ptd_expr_equal(a->right, b->left));

        case PTD_EXPR_DIV:
        case PTD_EXPR_SUB:
            // Non-commutative: order matters
            return ptd_expr_equal(a->left, b->left) && ptd_expr_equal(a->right, b->right);
    }

    return false;
}

// =============================================================================
// Expression Intern Table (for CSE)
// =============================================================================

/**
 * Expression intern table entry (linked list for collision handling)
 */
struct ptd_expr_intern_entry {
    struct ptd_expression *expr;
    uint64_t hash;
    struct ptd_expr_intern_entry *next;
};

/**
 * Expression intern table for CSE
 *
 * Hash table mapping expression structure → canonical instance.
 * Multiple references to identical expressions share single instance.
 */
struct ptd_expr_intern_table {
    struct ptd_expr_intern_entry **buckets;
    size_t capacity;
    size_t size;
    size_t collisions;  // Statistics
};

/**
 * Create intern table with specified capacity
 */
struct ptd_expr_intern_table *ptd_expr_intern_table_create(size_t capacity) {
    struct ptd_expr_intern_table *table =
        (struct ptd_expr_intern_table *)malloc(sizeof(struct ptd_expr_intern_table));

    if (table == NULL) {
        DIE_ERROR(1, "Failed to allocate intern table");
    }

    table->capacity = capacity;
    table->size = 0;
    table->collisions = 0;
    table->buckets = (struct ptd_expr_intern_entry **)
        calloc(capacity, sizeof(struct ptd_expr_intern_entry *));

    if (table->buckets == NULL) {
        free(table);
        DIE_ERROR(1, "Failed to allocate intern table buckets");
    }

    return table;
}

/**
 * Intern an expression (returns existing if found, otherwise adds to table)
 *
 * IMPORTANT: If existing expression found, destroys input and returns existing.
 * Caller must not use input pointer after calling this function.
 */
struct ptd_expression *ptd_expr_intern(
    struct ptd_expr_intern_table *table,
    struct ptd_expression *expr
) {
    if (expr == NULL || table == NULL) return expr;

    uint64_t hash = ptd_expr_hash(expr);
    size_t bucket = hash % table->capacity;

    // Search for existing expression
    struct ptd_expr_intern_entry *entry = table->buckets[bucket];
    bool first = true;
    while (entry != NULL) {
        if (entry->hash == hash && ptd_expr_equal(entry->expr, expr)) {
            // Found existing - destroy input and return existing
            ptd_expr_destroy_iterative(expr);
            return entry->expr;
        }
        if (!first) table->collisions++;
        first = false;
        entry = entry->next;
    }

    // Not found - add to table
    struct ptd_expr_intern_entry *new_entry =
        (struct ptd_expr_intern_entry *)malloc(sizeof(struct ptd_expr_intern_entry));

    if (new_entry == NULL) {
        DIE_ERROR(1, "Failed to allocate intern table entry");
    }

    new_entry->expr = expr;
    new_entry->hash = hash;
    new_entry->next = table->buckets[bucket];
    table->buckets[bucket] = new_entry;
    table->size++;

    return expr;
}

/**
 * Destroy intern table
 *
 * TEMPORARY: Not destroying expressions to debug crash issue.
 * TODO: Properly implement expression lifecycle management for interned expressions.
 */
void ptd_expr_intern_table_destroy(struct ptd_expr_intern_table *table) {
    if (table == NULL) return;

    // Free table structure only (expressions leak for now - debugging)
    for (size_t i = 0; i < table->capacity; i++) {
        struct ptd_expr_intern_entry *entry = table->buckets[i];
        while (entry != NULL) {
            struct ptd_expr_intern_entry *next = entry->next;
            // TEMPORARY: Don't destroy expressions - causes crash
            // if (entry->expr != NULL) {
            //     ptd_expr_destroy_iterative(entry->expr);
            // }
            free(entry);
            entry = next;
        }
    }
    free(table->buckets);
    free(table);
}

/**
 * Print intern table statistics (for debugging/profiling)
 */
void ptd_expr_intern_table_stats(const struct ptd_expr_intern_table *table) {
    if (table == NULL) return;

    printf("Expression Intern Table Statistics:\n");
    printf("  Capacity: %zu\n", table->capacity);
    printf("  Size: %zu entries\n", table->size);
    printf("  Load factor: %.2f%%\n", 100.0 * table->size / table->capacity);
    printf("  Total collisions: %zu\n", table->collisions);

    // Compute chain length distribution
    size_t max_chain = 0;
    size_t empty_buckets = 0;
    size_t chain_lengths[10] = {0};  // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9+

    for (size_t i = 0; i < table->capacity; i++) {
        size_t chain_len = 0;
        struct ptd_expr_intern_entry *e = table->buckets[i];
        while (e) {
            chain_len++;
            e = e->next;
        }

        if (chain_len == 0) {
            empty_buckets++;
        } else {
            size_t idx = chain_len < 9 ? chain_len : 9;
            chain_lengths[idx]++;
        }

        if (chain_len > max_chain) max_chain = chain_len;
    }

    printf("  Empty buckets: %zu (%.1f%%)\n", empty_buckets,
           100.0 * empty_buckets / table->capacity);
    printf("  Max chain length: %zu\n", max_chain);
    printf("  Chain length distribution:\n");
    for (size_t i = 1; i < 10; i++) {
        if (chain_lengths[i] > 0) {
            printf("    Length %zu: %zu buckets\n",
                   i < 9 ? i : 9, chain_lengths[i]);
        }
    }
}

// =============================================================================
// Interned Expression Constructors (for CSE)
// =============================================================================

/**
 * Create addition expression with interning
 */
struct ptd_expression *ptd_expr_add_interned(
    struct ptd_expr_intern_table *table,
    struct ptd_expression *left,
    struct ptd_expression *right
) {
    // Apply simplifications first (from Phase 1)
    if (left->type == PTD_EXPR_CONST && left->const_value == 0.0) {
        ptd_expr_destroy_iterative(left);
        return right;
    }
    if (right->type == PTD_EXPR_CONST && right->const_value == 0.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }
    if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
        double result = left->const_value + right->const_value;
        ptd_expr_destroy_iterative(left);
        ptd_expr_destroy_iterative(right);
        return ptd_expr_const(result);
    }

    // Create expression
    struct ptd_expression *expr = (struct ptd_expression *)calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate addition expression");
    }
    expr->type = PTD_EXPR_ADD;
    expr->left = left;
    expr->right = right;

    // Intern if table provided
    if (table != NULL) {
        return ptd_expr_intern(table, expr);
    }
    return expr;
}

/**
 * Create multiplication expression with interning
 */
struct ptd_expression *ptd_expr_mul_interned(
    struct ptd_expr_intern_table *table,
    struct ptd_expression *left,
    struct ptd_expression *right
) {
    // Apply simplifications first
    if (left->type == PTD_EXPR_CONST && left->const_value == 0.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }
    if (right->type == PTD_EXPR_CONST && right->const_value == 0.0) {
        ptd_expr_destroy_iterative(left);
        return right;
    }
    if (left->type == PTD_EXPR_CONST && left->const_value == 1.0) {
        ptd_expr_destroy_iterative(left);
        return right;
    }
    if (right->type == PTD_EXPR_CONST && right->const_value == 1.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }
    if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
        double result = left->const_value * right->const_value;
        ptd_expr_destroy_iterative(left);
        ptd_expr_destroy_iterative(right);
        return ptd_expr_const(result);
    }

    // Create expression
    struct ptd_expression *expr = (struct ptd_expression *)calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate multiplication expression");
    }
    expr->type = PTD_EXPR_MUL;
    expr->left = left;
    expr->right = right;

    // Intern if table provided
    if (table != NULL) {
        return ptd_expr_intern(table, expr);
    }
    return expr;
}

/**
 * Create division expression with interning
 */
struct ptd_expression *ptd_expr_div_interned(
    struct ptd_expr_intern_table *table,
    struct ptd_expression *left,
    struct ptd_expression *right
) {
    // Apply simplifications first
    if (left->type == PTD_EXPR_CONST && left->const_value == 0.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }
    if (right->type == PTD_EXPR_CONST && right->const_value == 1.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }
    if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
        if (right->const_value == 0.0) {
            DIE_ERROR(1, "Division by zero in constant folding");
        }
        double result = left->const_value / right->const_value;
        ptd_expr_destroy_iterative(left);
        ptd_expr_destroy_iterative(right);
        return ptd_expr_const(result);
    }

    // Create expression
    struct ptd_expression *expr = (struct ptd_expression *)calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate division expression");
    }
    expr->type = PTD_EXPR_DIV;
    expr->left = left;
    expr->right = right;

    // Intern if table provided
    if (table != NULL) {
        return ptd_expr_intern(table, expr);
    }
    return expr;
}

/**
 * Create subtraction expression with interning
 */
struct ptd_expression *ptd_expr_sub_interned(
    struct ptd_expr_intern_table *table,
    struct ptd_expression *left,
    struct ptd_expression *right
) {
    // Apply simplifications first
    if (right->type == PTD_EXPR_CONST && right->const_value == 0.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }
    if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
        double result = left->const_value - right->const_value;
        ptd_expr_destroy_iterative(left);
        ptd_expr_destroy_iterative(right);
        return ptd_expr_const(result);
    }

    // Create expression
    struct ptd_expression *expr = (struct ptd_expression *)calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate subtraction expression");
    }
    expr->type = PTD_EXPR_SUB;
    expr->left = left;
    expr->right = right;

    // Intern if table provided
    if (table != NULL) {
        return ptd_expr_intern(table, expr);
    }
    return expr;
}

/**
 * Create inversion expression with interning
 */
struct ptd_expression *ptd_expr_inv_interned(
    struct ptd_expr_intern_table *table,
    struct ptd_expression *child
) {
    // Simplification: inv(const) = const(1/c)
    if (child->type == PTD_EXPR_CONST) {
        if (child->const_value == 0.0) {
            DIE_ERROR(1, "Division by zero in constant inversion");
        }
        double result = 1.0 / child->const_value;
        ptd_expr_destroy_iterative(child);
        return ptd_expr_const(result);
    }

    // Create expression
    struct ptd_expression *expr = (struct ptd_expression *)calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate inversion expression");
    }
    expr->type = PTD_EXPR_INV;
    expr->left = child;

    // Intern if table provided
    if (table != NULL) {
        return ptd_expr_intern(table, expr);
    }
    return expr;
}

/**
 * Stack entry for iterative expression evaluation
 */
struct ptd_expr_eval_stack_entry {
    const struct ptd_expression *expr;
    bool children_pushed;
    double result;
};

/**
 * Simple hash table for expression results (pointer -> double)
 */
struct ptd_expr_result_entry {
    const struct ptd_expression *expr;
    double result;
    struct ptd_expr_result_entry *next;
};

struct ptd_expr_result_map {
    struct ptd_expr_result_entry **buckets;
    size_t capacity;
};

static struct ptd_expr_result_map *ptd_expr_result_map_create(size_t capacity) {
    struct ptd_expr_result_map *map = (struct ptd_expr_result_map *)malloc(sizeof(struct ptd_expr_result_map));
    if (map == NULL) return NULL;

    map->capacity = capacity;
    map->buckets = (struct ptd_expr_result_entry **)calloc(capacity, sizeof(struct ptd_expr_result_entry *));
    if (map->buckets == NULL) {
        free(map);
        return NULL;
    }
    return map;
}

static void ptd_expr_result_map_put(struct ptd_expr_result_map *map, const struct ptd_expression *expr, double result) {
    size_t bucket = ((size_t)expr / sizeof(void*)) % map->capacity;
    struct ptd_expr_result_entry *entry = (struct ptd_expr_result_entry *)malloc(sizeof(struct ptd_expr_result_entry));
    entry->expr = expr;
    entry->result = result;
    entry->next = map->buckets[bucket];
    map->buckets[bucket] = entry;
}

static double ptd_expr_result_map_get(struct ptd_expr_result_map *map, const struct ptd_expression *expr) {
    size_t bucket = ((size_t)expr / sizeof(void*)) % map->capacity;
    struct ptd_expr_result_entry *entry = map->buckets[bucket];
    while (entry != NULL) {
        if (entry->expr == expr) {
            return entry->result;
        }
        entry = entry->next;
    }
    return 0.0;  // Should not happen
}

static void ptd_expr_result_map_destroy(struct ptd_expr_result_map *map) {
    for (size_t i = 0; i < map->capacity; i++) {
        struct ptd_expr_result_entry *entry = map->buckets[i];
        while (entry != NULL) {
            struct ptd_expr_result_entry *next = entry->next;
            free(entry);
            entry = next;
        }
    }
    free(map->buckets);
    free(map);
}

/**
 * Evaluate an expression with given parameters (iterative version, O(n))
 * Uses post-order traversal with result hash map
 */
double ptd_expr_evaluate_iterative(
    const struct ptd_expression *expr,
    const double *params,
    size_t n_params
) {
    if (expr == NULL) {
        return 0.0;
    }

    // Create result map
    struct ptd_expr_result_map *results = ptd_expr_result_map_create(256);
    if (results == NULL) {
        DIE_ERROR(1, "Failed to allocate result map");
    }

    // Stack for post-order traversal
    size_t stack_capacity = 256;
    size_t stack_size = 0;
    struct ptd_expr_eval_stack_entry *stack = (struct ptd_expr_eval_stack_entry *)
        malloc(stack_capacity * sizeof(struct ptd_expr_eval_stack_entry));

    if (stack == NULL) {
        ptd_expr_result_map_destroy(results);
        DIE_ERROR(1, "Failed to allocate evaluation stack");
    }

    // Push root
    stack[stack_size++] = (struct ptd_expr_eval_stack_entry){
        .expr = expr,
        .children_pushed = false,
        .result = 0.0
    };

    while (stack_size > 0) {
        struct ptd_expr_eval_stack_entry *entry = &stack[stack_size - 1];
        const struct ptd_expression *e = entry->expr;

        if (!entry->children_pushed) {
            // First visit: push children for operators, compute leaves
            entry->children_pushed = true;

            switch (e->type) {
                case PTD_EXPR_CONST: {
                    double result = e->const_value;
                    ptd_expr_result_map_put(results, e, result);
                    stack_size--;  // Pop ourselves
                    break;
                }

                case PTD_EXPR_PARAM: {
                    if (e->param_index >= n_params) {
                        free(stack);
                        ptd_expr_result_map_destroy(results);
                        DIE_ERROR(1, "Parameter index out of bounds in expression evaluation");
                    }
                    double result = params[e->param_index];
                    ptd_expr_result_map_put(results, e, result);
                    stack_size--;  // Pop ourselves
                    break;
                }

                case PTD_EXPR_DOT: {
                    double result = 0.0;
                    for (size_t i = 0; i < e->n_terms; i++) {
                        if (e->param_indices[i] >= n_params) {
                            free(stack);
                            ptd_expr_result_map_destroy(results);
                            DIE_ERROR(1, "Parameter index out of bounds in dot expression evaluation");
                        }
                        result += e->coefficients[i] * params[e->param_indices[i]];
                    }
                    ptd_expr_result_map_put(results, e, result);
                    stack_size--;  // Pop ourselves
                    break;
                }

                case PTD_EXPR_INV:
                case PTD_EXPR_ADD:
                case PTD_EXPR_MUL:
                case PTD_EXPR_DIV:
                case PTD_EXPR_SUB:
                    // Grow stack if needed
                    if (stack_size + 2 > stack_capacity) {
                        stack_capacity *= 2;
                        struct ptd_expr_eval_stack_entry *new_stack =
                            (struct ptd_expr_eval_stack_entry *)
                            realloc(stack, stack_capacity * sizeof(struct ptd_expr_eval_stack_entry));
                        if (new_stack == NULL) {
                            free(stack);
                            ptd_expr_result_map_destroy(results);
                            DIE_ERROR(1, "Failed to grow evaluation stack");
                        }
                        stack = new_stack;
                        entry = &stack[stack_size - 1];
                    }

                    // Push children (right first, then left)
                    if (e->right != NULL) {
                        stack[stack_size++] = (struct ptd_expr_eval_stack_entry){
                            .expr = e->right,
                            .children_pushed = false,
                            .result = 0.0
                        };
                    }
                    if (e->left != NULL) {
                        stack[stack_size++] = (struct ptd_expr_eval_stack_entry){
                            .expr = e->left,
                            .children_pushed = false,
                            .result = 0.0
                        };
                    }
                    break;

                default:
                    free(stack);
                    ptd_expr_result_map_destroy(results);
                    DIE_ERROR(1, "Unknown expression type in evaluation");
            }
        } else {
            // Second visit: children processed, compute result from children's results
            switch (e->type) {
                case PTD_EXPR_CONST:
                case PTD_EXPR_PARAM:
                case PTD_EXPR_DOT:
                    // Already handled in first visit
                    break;

                case PTD_EXPR_INV: {
                    double child_val = ptd_expr_result_map_get(results, e->left);
                    if (child_val == 0.0) {
                        free(stack);
                        ptd_expr_result_map_destroy(results);
                        DIE_ERROR(1, "Division by zero in inversion expression evaluation");
                    }
                    double result = 1.0 / child_val;
                    ptd_expr_result_map_put(results, e, result);
                    stack_size--;  // Pop ourselves
                    break;
                }

                case PTD_EXPR_ADD:
                case PTD_EXPR_MUL:
                case PTD_EXPR_DIV:
                case PTD_EXPR_SUB: {
                    double left_val = ptd_expr_result_map_get(results, e->left);
                    double right_val = ptd_expr_result_map_get(results, e->right);

                    double result;
                    switch (e->type) {
                        case PTD_EXPR_ADD:
                            result = left_val + right_val;
                            break;
                        case PTD_EXPR_MUL:
                            result = left_val * right_val;
                            break;
                        case PTD_EXPR_DIV:
                            if (right_val == 0.0) {
                                free(stack);
                                ptd_expr_result_map_destroy(results);
                                DIE_ERROR(1, "Division by zero in expression evaluation");
                            }
                            result = left_val / right_val;
                            break;
                        case PTD_EXPR_SUB:
                            result = left_val - right_val;
                            break;
                        default:
                            result = 0.0;
                            break;
                    }

                    ptd_expr_result_map_put(results, e, result);
                    stack_size--;  // Pop ourselves
                    break;
                }

                default:
                    free(stack);
                    ptd_expr_result_map_destroy(results);
                    DIE_ERROR(1, "Unknown expression type in evaluation");
            }
        }
    }

    // Get result for root
    double final_result = ptd_expr_result_map_get(results, expr);

    free(stack);
    ptd_expr_result_map_destroy(results);
    return final_result;
}

/**
 * Evaluate an expression with given parameters (recursive version - kept for compatibility)
 * WARNING: May cause stack overflow for deeply nested expressions (>1000 levels)
 * Use ptd_expr_evaluate_iterative() for deep trees
 */
double ptd_expr_evaluate(
    const struct ptd_expression *expr,
    const double *params,
    size_t n_params
) {
    if (expr == NULL) {
        return 0.0;
    }

    switch (expr->type) {
        case PTD_EXPR_CONST:
            return expr->const_value;

        case PTD_EXPR_PARAM:
            if (expr->param_index >= n_params) {
                DIE_ERROR(1, "Parameter index out of bounds in expression evaluation");
            }
            return params[expr->param_index];

        case PTD_EXPR_DOT: {
            double result = 0.0;
            for (size_t i = 0; i < expr->n_terms; i++) {
                if (expr->param_indices[i] >= n_params) {
                    DIE_ERROR(1, "Parameter index out of bounds in dot expression evaluation");
                }
                result += expr->coefficients[i] * params[expr->param_indices[i]];
            }
            return result;
        }

        case PTD_EXPR_ADD: {
            double left_val = ptd_expr_evaluate(expr->left, params, n_params);
            double right_val = ptd_expr_evaluate(expr->right, params, n_params);
            return left_val + right_val;
        }

        case PTD_EXPR_MUL: {
            double left_val = ptd_expr_evaluate(expr->left, params, n_params);
            double right_val = ptd_expr_evaluate(expr->right, params, n_params);
            return left_val * right_val;
        }

        case PTD_EXPR_DIV: {
            double left_val = ptd_expr_evaluate(expr->left, params, n_params);
            double right_val = ptd_expr_evaluate(expr->right, params, n_params);
            if (right_val == 0.0) {
                DIE_ERROR(1, "Division by zero in expression evaluation");
            }
            return left_val / right_val;
        }

        case PTD_EXPR_INV: {
            double child_val = ptd_expr_evaluate(expr->left, params, n_params);
            if (child_val == 0.0) {
                DIE_ERROR(1, "Division by zero in inversion expression evaluation");
            }
            return 1.0 / child_val;
        }

        case PTD_EXPR_SUB: {
            double left_val = ptd_expr_evaluate(expr->left, params, n_params);
            double right_val = ptd_expr_evaluate(expr->right, params, n_params);
            return left_val - right_val;
        }

        default:
            DIE_ERROR(1, "Unknown expression type in evaluation");
            return 0.0;
    }
}

/**
 * Evaluate an expression for multiple parameter sets (batch evaluation)
 */
void ptd_expr_evaluate_batch(
    const struct ptd_expression *expr,
    const double *params_batch,      // shape: (batch_size, n_params)
    size_t batch_size,
    size_t n_params,
    double *output                   // shape: (batch_size,)
) {
    if (expr == NULL || params_batch == NULL || output == NULL) {
        return;
    }

    // Evaluate for each parameter set
    for (size_t i = 0; i < batch_size; i++) {
        const double *params_i = params_batch + i * n_params;
        output[i] = ptd_expr_evaluate(expr, params_i, n_params);
    }
}

// ============================================================================
// Trace-Based Elimination Implementation
// ============================================================================

/**
 * Helper: Ensure trace operations array has sufficient capacity
 */
static int ensure_trace_capacity(
    struct ptd_elimination_trace *trace,
    size_t required_capacity
) {
    if (trace->operations_length >= required_capacity) {
        return 0; // Already has capacity
    }

    // Find current capacity (stored separately, but we'll compute it)
    size_t current_capacity = trace->operations_length;
    if (current_capacity == 0) {
        current_capacity = 1000; // Initial capacity
    }

    // Double capacity until we have enough
    size_t new_capacity = current_capacity;
    while (new_capacity < required_capacity) {
        new_capacity *= 2;
    }

    // Realloc
    struct ptd_trace_operation *new_ops = (struct ptd_trace_operation *)realloc(
        trace->operations,
        new_capacity * sizeof(struct ptd_trace_operation)
    );
    if (new_ops == NULL) {
        return -1; // Allocation failed
    }

    trace->operations = new_ops;
    return 0;
}

/**
 * Helper: Add CONST operation to trace
 */
static size_t add_const_to_trace(
    struct ptd_elimination_trace *trace,
    double value
) {
    // Ensure capacity (allow for growth)
    if (ensure_trace_capacity(trace, trace->operations_length + 1) != 0) {
        return (size_t)-1;
    }

    size_t idx = trace->operations_length++;
    struct ptd_trace_operation *op = &trace->operations[idx];

    op->op_type = PTD_OP_CONST;
    op->const_value = value;
    op->param_idx = 0;
    op->coefficients = NULL;
    op->coefficients_length = 0;
    op->operands = NULL;
    op->operands_length = 0;

    return idx;
}

/**
 * Helper: Add DOT operation to trace
 * DOT product: Σ(coefficients[i] * θ[i])
 */
static size_t add_dot_to_trace(
    struct ptd_elimination_trace *trace,
    const double *coefficients,
    size_t coefficients_length
) {
    if (ensure_trace_capacity(trace, trace->operations_length + 1) != 0) {
        return (size_t)-1;
    }

    size_t idx = trace->operations_length++;
    struct ptd_trace_operation *op = &trace->operations[idx];

    op->op_type = PTD_OP_DOT;

    // Copy coefficients
    op->coefficients = (double *)malloc(coefficients_length * sizeof(double));
    if (op->coefficients == NULL) {
        trace->operations_length--; // Roll back
        return (size_t)-1;
    }
    memcpy(op->coefficients, coefficients, coefficients_length * sizeof(double));
    op->coefficients_length = coefficients_length;

    op->const_value = 0.0;
    op->param_idx = 0;
    op->operands = NULL;
    op->operands_length = 0;

    return idx;
}

/**
 * Helper: Add ADD operation to trace
 * ADD: operands[0] + operands[1]
 */
static size_t add_add_to_trace(
    struct ptd_elimination_trace *trace,
    size_t left_idx,
    size_t right_idx
) {
    if (ensure_trace_capacity(trace, trace->operations_length + 1) != 0) {
        return (size_t)-1;
    }

    size_t idx = trace->operations_length++;
    struct ptd_trace_operation *op = &trace->operations[idx];

    op->op_type = PTD_OP_ADD;

    op->operands = (size_t *)malloc(2 * sizeof(size_t));
    if (op->operands == NULL) {
        trace->operations_length--;
        return (size_t)-1;
    }
    op->operands[0] = left_idx;
    op->operands[1] = right_idx;
    op->operands_length = 2;

    op->const_value = 0.0;
    op->param_idx = 0;
    op->coefficients = NULL;
    op->coefficients_length = 0;

    return idx;
}

/**
 * Helper: Add MUL operation to trace
 * MUL: operands[0] * operands[1]
 */
static size_t add_mul_to_trace(
    struct ptd_elimination_trace *trace,
    size_t left_idx,
    size_t right_idx
) {
    if (ensure_trace_capacity(trace, trace->operations_length + 1) != 0) {
        return (size_t)-1;
    }

    size_t idx = trace->operations_length++;
    struct ptd_trace_operation *op = &trace->operations[idx];

    op->op_type = PTD_OP_MUL;

    op->operands = (size_t *)malloc(2 * sizeof(size_t));
    if (op->operands == NULL) {
        trace->operations_length--;
        return (size_t)-1;
    }
    op->operands[0] = left_idx;
    op->operands[1] = right_idx;
    op->operands_length = 2;

    op->const_value = 0.0;
    op->param_idx = 0;
    op->coefficients = NULL;
    op->coefficients_length = 0;

    return idx;
}

/**
 * Helper: Add DIV operation to trace
 * DIV: operands[0] / operands[1]
 */
static size_t add_div_to_trace(
    struct ptd_elimination_trace *trace,
    size_t left_idx,
    size_t right_idx
) {
    if (ensure_trace_capacity(trace, trace->operations_length + 1) != 0) {
        return (size_t)-1;
    }

    size_t idx = trace->operations_length++;
    struct ptd_trace_operation *op = &trace->operations[idx];

    op->op_type = PTD_OP_DIV;

    op->operands = (size_t *)malloc(2 * sizeof(size_t));
    if (op->operands == NULL) {
        trace->operations_length--;
        return (size_t)-1;
    }
    op->operands[0] = left_idx;
    op->operands[1] = right_idx;
    op->operands_length = 2;

    op->const_value = 0.0;
    op->param_idx = 0;
    op->coefficients = NULL;
    op->coefficients_length = 0;

    return idx;
}

/**
 * Helper: Add INV operation to trace
 * INV: 1 / operands[0]
 */
static size_t add_inv_to_trace(
    struct ptd_elimination_trace *trace,
    size_t operand_idx
) {
    if (ensure_trace_capacity(trace, trace->operations_length + 1) != 0) {
        return (size_t)-1;
    }

    size_t idx = trace->operations_length++;
    struct ptd_trace_operation *op = &trace->operations[idx];

    op->op_type = PTD_OP_INV;

    op->operands = (size_t *)malloc(1 * sizeof(size_t));
    if (op->operands == NULL) {
        trace->operations_length--;
        return (size_t)-1;
    }
    op->operands[0] = operand_idx;
    op->operands_length = 1;

    op->const_value = 0.0;
    op->param_idx = 0;
    op->coefficients = NULL;
    op->coefficients_length = 0;

    return idx;
}

/**
 * Helper: Add SUM operation to trace
 * SUM: sum(operands[0], operands[1], ..., operands[n-1])
 */
static size_t add_sum_to_trace(
    struct ptd_elimination_trace *trace,
    const size_t *operand_indices,
    size_t n_operands
) {
    if (ensure_trace_capacity(trace, trace->operations_length + 1) != 0) {
        return (size_t)-1;
    }

    size_t idx = trace->operations_length++;
    struct ptd_trace_operation *op = &trace->operations[idx];

    op->op_type = PTD_OP_SUM;

    op->operands = (size_t *)malloc(n_operands * sizeof(size_t));
    if (op->operands == NULL) {
        trace->operations_length--;
        return (size_t)-1;
    }
    memcpy(op->operands, operand_indices, n_operands * sizeof(size_t));
    op->operands_length = n_operands;

    op->const_value = 0.0;
    op->param_idx = 0;
    op->coefficients = NULL;
    op->coefficients_length = 0;

    return idx;
}

/**
 * Record elimination trace from parameterized graph
 *
 * Performs graph elimination while recording all arithmetic operations
 * in a linear sequence. Currently implements Phase 1 (vertex rates).
 */
struct ptd_elimination_trace *ptd_record_elimination_trace(
    struct ptd_graph *graph
) {
    if (!graph->parameterized) {
        sprintf((char*)ptd_err, "ptd_record_elimination_trace: graph is not parameterized");
        return NULL;
    }

    // Check cache first (if hash computation succeeds)
    struct ptd_hash_result *hash = ptd_graph_content_hash(graph);
    struct ptd_elimination_trace *cached_trace = NULL;

    if (hash != NULL) {
        cached_trace = load_trace_from_cache(hash->hash_hex);
        if (cached_trace != NULL) {
            DEBUG_PRINT("INFO: loaded elimination trace from cache (%s)\n", hash->hash_hex);
            ptd_hash_destroy(hash);
            return cached_trace;
        }
    }

    // Cache miss or hash failed - record trace normally
    DEBUG_PRINT("INFO: cache miss, recording elimination trace...\n");

    // Allocate trace structure
    struct ptd_elimination_trace *trace = (struct ptd_elimination_trace *)malloc(sizeof(*trace));
    if (trace == NULL) {
        sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate trace");
        return NULL;
    }

    // Initialize metadata
    trace->n_vertices = graph->vertices_length;
    trace->state_length = graph->state_length;
    trace->param_length = graph->param_length;
    trace->is_discrete = graph->was_dph;

    // Find starting vertex index
    trace->starting_vertex_idx = 0;
    if (graph->starting_vertex != NULL) {
        trace->starting_vertex_idx = graph->starting_vertex->index;
    }

    // Allocate operations array (initial capacity)
    size_t operations_capacity = 1000;
    trace->operations = (struct ptd_trace_operation *)malloc(operations_capacity * sizeof(struct ptd_trace_operation));
    if (trace->operations == NULL) {
        sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate operations");
        free(trace);
        return NULL;
    }
    trace->operations_length = 0;

    // Allocate vertex mappings
    trace->vertex_rates = (size_t *)malloc(trace->n_vertices * sizeof(size_t));
    trace->edge_probs = (size_t **)malloc(trace->n_vertices * sizeof(size_t*));
    trace->edge_probs_lengths = (size_t *)calloc(trace->n_vertices, sizeof(size_t));
    trace->vertex_targets = (size_t **)malloc(trace->n_vertices * sizeof(size_t*));
    trace->vertex_targets_lengths = (size_t *)calloc(trace->n_vertices, sizeof(size_t));

    if (trace->vertex_rates == NULL || trace->edge_probs == NULL ||
        trace->edge_probs_lengths == NULL || trace->vertex_targets == NULL ||
        trace->vertex_targets_lengths == NULL) {
        sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate vertex mappings");
        ptd_elimination_trace_destroy(trace);
        return NULL;
    }

    // Initialize edge arrays to NULL
    for (size_t i = 0; i < trace->n_vertices; i++) {
        trace->edge_probs[i] = NULL;
        trace->vertex_targets[i] = NULL;
    }

    // Copy vertex states
    trace->states = (int **)malloc(trace->n_vertices * sizeof(int*));
    if (trace->states == NULL) {
        sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate states");
        ptd_elimination_trace_destroy(trace);
        return NULL;
    }

    for (size_t i = 0; i < trace->n_vertices; i++) {
        trace->states[i] = (int *)malloc(trace->state_length * sizeof(int));
        if (trace->states[i] == NULL) {
            sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate state for vertex %zu", i);
            // Free previously allocated states
            for (size_t j = 0; j < i; j++) {
                free(trace->states[j]);
            }
            free(trace->states);
            ptd_elimination_trace_destroy(trace);
            return NULL;
        }

        if (graph->vertices[i]->state != NULL) {
            memcpy(trace->states[i], graph->vertices[i]->state,
                   trace->state_length * sizeof(int));
        } else {
            // Zero initialize if no state
            memset(trace->states[i], 0, trace->state_length * sizeof(int));
        }
    }

    // PHASE 1: Compute vertex rates
    for (size_t i = 0; i < graph->vertices_length; i++) {
        struct ptd_vertex *v = graph->vertices[i];

        if (v->edges_length == 0) {
            // Absorbing state: rate = 0
            trace->vertex_rates[i] = add_const_to_trace(trace, 0.0);
        } else {
            // rate = 1 / sum(edge_weights)
            size_t *weight_indices = (size_t *)malloc(v->edges_length * sizeof(size_t));
            if (weight_indices == NULL) {
                sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate weight_indices");
                ptd_elimination_trace_destroy(trace);
                return NULL;
            }

            for (size_t j = 0; j < v->edges_length; j++) {
                struct ptd_edge *edge = v->edges[j];

                if (edge->parameterized) {
                    struct ptd_edge_parameterized *param_edge =
                        (struct ptd_edge_parameterized*)edge;

                    // Extract coefficients from param_edge->state
                    double *coeffs = param_edge->state;
                    size_t coeffs_len = graph->param_length;

                    // Check if all coefficients are zero
                    bool all_zero = true;
                    for (size_t k = 0; k < coeffs_len; k++) {
                        if (fabs(coeffs[k]) > 1e-15) {
                            all_zero = false;
                            break;
                        }
                    }

                    if (all_zero) {
                        // No parameterization, just use base weight
                        weight_indices[j] = add_const_to_trace(trace, param_edge->weight);
                    } else {
                        // DOT product: c₁*θ₁ + c₂*θ₂ + ...
                        size_t dot_idx = add_dot_to_trace(trace, coeffs, coeffs_len);

                        // Add base weight if non-zero
                        if (fabs(param_edge->weight) > 1e-15) {
                            size_t base_idx = add_const_to_trace(trace, param_edge->weight);
                            weight_indices[j] = add_add_to_trace(trace, base_idx, dot_idx);
                        } else {
                            weight_indices[j] = dot_idx;
                        }
                    }
                } else {
                    // Regular edge
                    weight_indices[j] = add_const_to_trace(trace, edge->weight);
                }
            }

            // Sum all weights
            size_t sum_idx = add_sum_to_trace(trace, weight_indices, v->edges_length);

            // Rate = 1 / sum
            trace->vertex_rates[i] = add_inv_to_trace(trace, sum_idx);

            free(weight_indices);
        }
    }

    // PHASE 2: Convert edges to probabilities
    // Allocate dynamic edge arrays (will grow during elimination)
    size_t *edge_capacities = (size_t *)malloc(trace->n_vertices * sizeof(size_t));
    if (edge_capacities == NULL) {
        sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate edge_capacities");
        ptd_elimination_trace_destroy(trace);
        return NULL;
    }

    for (size_t i = 0; i < trace->n_vertices; i++) {
        struct ptd_vertex *v = graph->vertices[i];
        size_t n_edges = v->edges_length;

        trace->edge_probs_lengths[i] = n_edges;
        trace->vertex_targets_lengths[i] = n_edges;
        edge_capacities[i] = n_edges > 0 ? n_edges : 1;

        if (n_edges > 0) {
            trace->edge_probs[i] = (size_t *)malloc(edge_capacities[i] * sizeof(size_t));
            trace->vertex_targets[i] = (size_t *)malloc(edge_capacities[i] * sizeof(size_t));

            if (trace->edge_probs[i] == NULL || trace->vertex_targets[i] == NULL) {
                sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate edge arrays");
                free(edge_capacities);
                ptd_elimination_trace_destroy(trace);
                return NULL;
            }

            // Convert edge weights to probabilities
            for (size_t j = 0; j < n_edges; j++) {
                struct ptd_edge *edge = v->edges[j];

                // Get edge weight index (recompute like in Phase 1)
                size_t weight_idx;
                if (edge->parameterized) {
                    struct ptd_edge_parameterized *param_edge =
                        (struct ptd_edge_parameterized*)edge;

                    double *coeffs = param_edge->state;
                    size_t coeffs_len = graph->param_length;

                    bool all_zero = true;
                    for (size_t k = 0; k < coeffs_len; k++) {
                        if (fabs(coeffs[k]) > 1e-15) {
                            all_zero = false;
                            break;
                        }
                    }

                    if (all_zero) {
                        weight_idx = add_const_to_trace(trace, param_edge->weight);
                    } else {
                        size_t dot_idx = add_dot_to_trace(trace, coeffs, coeffs_len);
                        if (fabs(param_edge->weight) > 1e-15) {
                            size_t base_idx = add_const_to_trace(trace, param_edge->weight);
                            weight_idx = add_add_to_trace(trace, base_idx, dot_idx);
                        } else {
                            weight_idx = dot_idx;
                        }
                    }
                } else {
                    weight_idx = add_const_to_trace(trace, edge->weight);
                }

                // prob = weight * rate
                size_t prob_idx = add_mul_to_trace(trace, weight_idx, trace->vertex_rates[i]);
                trace->edge_probs[i][j] = prob_idx;

                // Store target vertex index
                trace->vertex_targets[i][j] = edge->to->index;
            }
        }
    }

    // PHASE 3: Elimination loop
    // Build parent-child relationships
    size_t **parents = (size_t **)malloc(trace->n_vertices * sizeof(size_t*));
    size_t *parents_lengths = (size_t *)calloc(trace->n_vertices, sizeof(size_t));
    size_t *parents_capacities = (size_t *)malloc(trace->n_vertices * sizeof(size_t));

    if (parents == NULL || parents_lengths == NULL || parents_capacities == NULL) {
        sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate parent arrays");
        free(edge_capacities);
        free(parents);
        free(parents_lengths);
        free(parents_capacities);
        ptd_elimination_trace_destroy(trace);
        return NULL;
    }

    for (size_t i = 0; i < trace->n_vertices; i++) {
        parents_capacities[i] = 4;  // Initial capacity
        parents[i] = (size_t *)malloc(parents_capacities[i] * sizeof(size_t));
        if (parents[i] == NULL) {
            sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate parent list");
            for (size_t k = 0; k < i; k++) {
                free(parents[k]);
            }
            free(edge_capacities);
            free(parents);
            free(parents_lengths);
            free(parents_capacities);
            ptd_elimination_trace_destroy(trace);
            return NULL;
        }
    }

    // Build parent lists
    for (size_t i = 0; i < trace->n_vertices; i++) {
        for (size_t j = 0; j < trace->vertex_targets_lengths[i]; j++) {
            size_t to_idx = trace->vertex_targets[i][j];

            // Add i to parents of to_idx
            if (parents_lengths[to_idx] >= parents_capacities[to_idx]) {
                parents_capacities[to_idx] *= 2;
                size_t *new_parents = (size_t *)realloc(parents[to_idx],
                    parents_capacities[to_idx] * sizeof(size_t));
                if (new_parents == NULL) {
                    sprintf((char*)ptd_err, "ptd_record_elimination_trace: realloc parent failed");
                    for (size_t k = 0; k < trace->n_vertices; k++) {
                        free(parents[k]);
                    }
                    free(edge_capacities);
                    free(parents);
                    free(parents_lengths);
                    free(parents_capacities);
                    ptd_elimination_trace_destroy(trace);
                    return NULL;
                }
                parents[to_idx] = new_parents;
            }
            parents[to_idx][parents_lengths[to_idx]++] = i;
        }
    }

    // Elimination loop
    for (size_t i = 0; i < trace->n_vertices; i++) {
        size_t n_children = trace->vertex_targets_lengths[i];

        if (n_children == 0) {
            // Absorbing state, nothing to eliminate
            continue;
        }

        // For each parent of vertex i
        for (size_t p = 0; p < parents_lengths[i]; p++) {
            size_t parent_idx = parents[i][p];

            // Skip if parent already processed
            if (parent_idx < i) {
                continue;
            }

            // Find edge from parent to i
            size_t parent_to_i_edge_idx = (size_t)-1;
            for (size_t e = 0; e < trace->vertex_targets_lengths[parent_idx]; e++) {
                if (trace->vertex_targets[parent_idx][e] == i &&
                    trace->edge_probs[parent_idx][e] != (size_t)-1) {
                    parent_to_i_edge_idx = e;
                    break;
                }
            }

            if (parent_to_i_edge_idx == (size_t)-1) {
                // Parent no longer has edge to i
                continue;
            }

            size_t parent_to_i_prob = trace->edge_probs[parent_idx][parent_to_i_edge_idx];

            // For each child of i
            for (size_t c = 0; c < n_children; c++) {
                size_t child_idx = trace->vertex_targets[i][c];
                size_t i_to_child_prob = trace->edge_probs[i][c];

                // Skip self-loops (TODO: implement properly later)
                if (child_idx == parent_idx || child_idx == i) {
                    continue;
                }

                // Bypass probability: parent_to_i * i_to_child
                size_t bypass_prob = add_mul_to_trace(trace, parent_to_i_prob, i_to_child_prob);

                // Check if parent already has edge to child
                size_t parent_to_child_edge_idx = (size_t)-1;
                for (size_t e = 0; e < trace->vertex_targets_lengths[parent_idx]; e++) {
                    if (trace->vertex_targets[parent_idx][e] == child_idx &&
                        trace->edge_probs[parent_idx][e] != (size_t)-1) {
                        parent_to_child_edge_idx = e;
                        break;
                    }
                }

                if (parent_to_child_edge_idx != (size_t)-1) {
                    // Update existing edge
                    size_t old_prob = trace->edge_probs[parent_idx][parent_to_child_edge_idx];
                    size_t new_prob = add_add_to_trace(trace, old_prob, bypass_prob);
                    trace->edge_probs[parent_idx][parent_to_child_edge_idx] = new_prob;
                } else {
                    // Create new edge
                    size_t new_idx = trace->vertex_targets_lengths[parent_idx];

                    // Ensure capacity
                    if (new_idx >= edge_capacities[parent_idx]) {
                        edge_capacities[parent_idx] *= 2;
                        size_t *new_probs = (size_t *)realloc(trace->edge_probs[parent_idx],
                            edge_capacities[parent_idx] * sizeof(size_t));
                        size_t *new_targets = (size_t *)realloc(trace->vertex_targets[parent_idx],
                            edge_capacities[parent_idx] * sizeof(size_t));

                        if (new_probs == NULL || new_targets == NULL) {
                            sprintf((char*)ptd_err, "ptd_record_elimination_trace: realloc edge failed");
                            for (size_t k = 0; k < trace->n_vertices; k++) {
                                free(parents[k]);
                            }
                            free(edge_capacities);
                            free(parents);
                            free(parents_lengths);
                            free(parents_capacities);
                            ptd_elimination_trace_destroy(trace);
                            return NULL;
                        }

                        trace->edge_probs[parent_idx] = new_probs;
                        trace->vertex_targets[parent_idx] = new_targets;
                    }

                    trace->edge_probs[parent_idx][new_idx] = bypass_prob;
                    trace->vertex_targets[parent_idx][new_idx] = child_idx;
                    trace->vertex_targets_lengths[parent_idx]++;
                    trace->edge_probs_lengths[parent_idx]++;
                }
            }

            // Mark edge from parent to i as removed
            trace->edge_probs[parent_idx][parent_to_i_edge_idx] = (size_t)-1;

            // Renormalize parent's edges
            // Count valid (non-removed) edges
            size_t valid_count = 0;
            for (size_t e = 0; e < trace->edge_probs_lengths[parent_idx]; e++) {
                if (trace->edge_probs[parent_idx][e] != (size_t)-1) {
                    valid_count++;
                }
            }

            if (valid_count > 0) {
                // Compute sum of valid edges
                size_t *valid_probs = (size_t *)malloc(valid_count * sizeof(size_t));
                if (valid_probs == NULL) {
                    sprintf((char*)ptd_err, "ptd_record_elimination_trace: malloc valid_probs failed");
                    for (size_t k = 0; k < trace->n_vertices; k++) {
                        free(parents[k]);
                    }
                    free(edge_capacities);
                    free(parents);
                    free(parents_lengths);
                    free(parents_capacities);
                    ptd_elimination_trace_destroy(trace);
                    return NULL;
                }

                size_t valid_idx = 0;
                for (size_t e = 0; e < trace->edge_probs_lengths[parent_idx]; e++) {
                    if (trace->edge_probs[parent_idx][e] != (size_t)-1) {
                        valid_probs[valid_idx++] = trace->edge_probs[parent_idx][e];
                    }
                }

                size_t total_idx = add_sum_to_trace(trace, valid_probs, valid_count);
                free(valid_probs);

                // Normalize each valid edge: prob = prob / total
                for (size_t e = 0; e < trace->edge_probs_lengths[parent_idx]; e++) {
                    if (trace->edge_probs[parent_idx][e] != (size_t)-1) {
                        size_t old_prob = trace->edge_probs[parent_idx][e];
                        size_t new_prob = add_div_to_trace(trace, old_prob, total_idx);
                        trace->edge_probs[parent_idx][e] = new_prob;
                    }
                }
            }
        }
    }

    // PHASE 4: Clean up removed edges
    for (size_t i = 0; i < trace->n_vertices; i++) {
        // Count valid edges
        size_t valid_count = 0;
        for (size_t j = 0; j < trace->edge_probs_lengths[i]; j++) {
            if (trace->edge_probs[i][j] != (size_t)-1) {
                valid_count++;
            }
        }

        // Compact arrays
        if (valid_count < trace->edge_probs_lengths[i]) {
            size_t *new_probs = (size_t *)malloc(valid_count * sizeof(size_t));
            size_t *new_targets = (size_t *)malloc(valid_count * sizeof(size_t));

            if ((valid_count > 0 && (new_probs == NULL || new_targets == NULL))) {
                sprintf((char*)ptd_err, "ptd_record_elimination_trace: cleanup malloc failed");
                for (size_t k = 0; k < trace->n_vertices; k++) {
                    free(parents[k]);
                }
                free(edge_capacities);
                free(parents);
                free(parents_lengths);
                free(parents_capacities);
                free(new_probs);
                free(new_targets);
                ptd_elimination_trace_destroy(trace);
                return NULL;
            }

            size_t write_idx = 0;
            for (size_t j = 0; j < trace->edge_probs_lengths[i]; j++) {
                if (trace->edge_probs[i][j] != (size_t)-1) {
                    new_probs[write_idx] = trace->edge_probs[i][j];
                    new_targets[write_idx] = trace->vertex_targets[i][j];
                    write_idx++;
                }
            }

            free(trace->edge_probs[i]);
            free(trace->vertex_targets[i]);

            trace->edge_probs[i] = new_probs;
            trace->vertex_targets[i] = new_targets;
            trace->edge_probs_lengths[i] = valid_count;
            trace->vertex_targets_lengths[i] = valid_count;
        }
    }

    // Cleanup temporary arrays
    for (size_t i = 0; i < trace->n_vertices; i++) {
        free(parents[i]);
    }
    free(edge_capacities);
    free(parents);
    free(parents_lengths);
    free(parents_capacities);

    // Save newly recorded trace to cache
    if (hash != NULL) {
        save_trace_to_cache(hash->hash_hex, trace);
        ptd_hash_destroy(hash);
    }

    return trace;
}

/**
 * Destroy elimination trace and free all memory
 */
void ptd_elimination_trace_destroy(struct ptd_elimination_trace *trace) {
    if (trace == NULL) {
        return;
    }

    // Free operations
    if (trace->operations != NULL) {
        for (size_t i = 0; i < trace->operations_length; i++) {
            struct ptd_trace_operation *op = &trace->operations[i];
            if (op->coefficients != NULL) {
                free(op->coefficients);
            }
            if (op->operands != NULL) {
                free(op->operands);
            }
        }
        free(trace->operations);
    }

    // Free vertex mappings
    if (trace->vertex_rates != NULL) {
        free(trace->vertex_rates);
    }

    if (trace->edge_probs != NULL) {
        for (size_t i = 0; i < trace->n_vertices; i++) {
            if (trace->edge_probs[i] != NULL) {
                free(trace->edge_probs[i]);
            }
        }
        free(trace->edge_probs);
    }

    if (trace->edge_probs_lengths != NULL) {
        free(trace->edge_probs_lengths);
    }

    if (trace->vertex_targets != NULL) {
        for (size_t i = 0; i < trace->n_vertices; i++) {
            if (trace->vertex_targets[i] != NULL) {
                free(trace->vertex_targets[i]);
            }
        }
        free(trace->vertex_targets);
    }

    if (trace->vertex_targets_lengths != NULL) {
        free(trace->vertex_targets_lengths);
    }

    // Free states
    if (trace->states != NULL) {
        for (size_t i = 0; i < trace->n_vertices; i++) {
            if (trace->states[i] != NULL) {
                free(trace->states[i]);
            }
        }
        free(trace->states);
    }

    free(trace);
}

/**
 * Evaluate elimination trace with concrete parameter values
 *
 * Executes the recorded operation sequence with given parameters
 * to produce vertex rates and edge probabilities.
 */
struct ptd_trace_result *ptd_evaluate_trace(
    const struct ptd_elimination_trace *trace,
    const double *params,
    size_t params_length
) {
    // Validate parameters
    if (trace == NULL) {
        sprintf((char*)ptd_err, "ptd_evaluate_trace: trace is NULL");
        return NULL;
    }

    if (trace->param_length > 0) {
        if (params == NULL) {
            sprintf((char*)ptd_err, "ptd_evaluate_trace: params is NULL but trace has %zu parameters",
                    trace->param_length);
            return NULL;
        }

        if (params_length != trace->param_length) {
            sprintf((char*)ptd_err, "ptd_evaluate_trace: expected %zu parameters, got %zu",
                    trace->param_length, params_length);
            return NULL;
        }
    }

    // Allocate value array for all operations
    double *values = (double *)calloc(trace->operations_length, sizeof(double));
    if (values == NULL) {
        sprintf((char*)ptd_err, "ptd_evaluate_trace: failed to allocate values array");
        return NULL;
    }

    // Execute operations in order
    for (size_t i = 0; i < trace->operations_length; i++) {
        const struct ptd_trace_operation *op = &trace->operations[i];

        switch (op->op_type) {
            case PTD_OP_CONST:
                values[i] = op->const_value;
                break;

            case PTD_OP_PARAM:
                if (op->param_idx < params_length) {
                    values[i] = params[op->param_idx];
                }
                break;

            case PTD_OP_DOT:
                // Dot product: Σ(cᵢ * θᵢ)
                values[i] = 0.0;
                for (size_t j = 0; j < op->coefficients_length && j < params_length; j++) {
                    values[i] += op->coefficients[j] * params[j];
                }
                break;

            case PTD_OP_ADD:
                if (op->operands_length >= 2) {
                    values[i] = values[op->operands[0]] + values[op->operands[1]];
                }
                break;

            case PTD_OP_MUL:
                if (op->operands_length >= 2) {
                    values[i] = values[op->operands[0]] * values[op->operands[1]];
                }
                break;

            case PTD_OP_DIV:
                if (op->operands_length >= 2) {
                    double denominator = values[op->operands[1]];
                    if (fabs(denominator) > 1e-15) {
                        values[i] = values[op->operands[0]] / denominator;
                    } else {
                        values[i] = 0.0;  // Handle division by zero
                    }
                }
                break;

            case PTD_OP_INV:
                if (op->operands_length >= 1) {
                    double val = values[op->operands[0]];
                    if (fabs(val) > 1e-15) {
                        values[i] = 1.0 / val;
                    } else {
                        values[i] = 0.0;  // Handle inverse of zero
                    }
                }
                break;

            case PTD_OP_SUM:
                values[i] = 0.0;
                for (size_t j = 0; j < op->operands_length; j++) {
                    values[i] += values[op->operands[j]];
                }
                break;

            default:
                // Unknown operation type
                values[i] = 0.0;
                break;
        }
    }

    // Allocate result structure
    struct ptd_trace_result *result = (struct ptd_trace_result *)malloc(sizeof(*result));
    if (result == NULL) {
        sprintf((char*)ptd_err, "ptd_evaluate_trace: failed to allocate result");
        free(values);
        return NULL;
    }

    result->n_vertices = trace->n_vertices;

    // Extract vertex rates
    result->vertex_rates = (double *)malloc(trace->n_vertices * sizeof(double));
    if (result->vertex_rates == NULL) {
        sprintf((char*)ptd_err, "ptd_evaluate_trace: failed to allocate vertex_rates");
        free(values);
        free(result);
        return NULL;
    }

    for (size_t i = 0; i < trace->n_vertices; i++) {
        result->vertex_rates[i] = values[trace->vertex_rates[i]];
    }

    // Extract edge probabilities
    result->edge_probs = (double **)malloc(trace->n_vertices * sizeof(double*));
    result->edge_probs_lengths = (size_t *)malloc(trace->n_vertices * sizeof(size_t));
    result->vertex_targets = (size_t **)malloc(trace->n_vertices * sizeof(size_t*));
    result->vertex_targets_lengths = (size_t *)malloc(trace->n_vertices * sizeof(size_t));

    if (result->edge_probs == NULL || result->edge_probs_lengths == NULL ||
        result->vertex_targets == NULL || result->vertex_targets_lengths == NULL) {
        sprintf((char*)ptd_err, "ptd_evaluate_trace: failed to allocate edge arrays");
        free(values);
        ptd_trace_result_destroy(result);
        return NULL;
    }

    // Initialize to NULL
    for (size_t i = 0; i < trace->n_vertices; i++) {
        result->edge_probs[i] = NULL;
        result->vertex_targets[i] = NULL;
    }

    for (size_t i = 0; i < trace->n_vertices; i++) {
        size_t n_edges = trace->edge_probs_lengths[i];
        result->edge_probs_lengths[i] = n_edges;
        result->vertex_targets_lengths[i] = n_edges;

        if (n_edges > 0) {
            result->edge_probs[i] = (double *)malloc(n_edges * sizeof(double));
            result->vertex_targets[i] = (size_t *)malloc(n_edges * sizeof(size_t));

            if (result->edge_probs[i] == NULL || result->vertex_targets[i] == NULL) {
                sprintf((char*)ptd_err, "ptd_evaluate_trace: failed to allocate edge arrays for vertex %zu", i);
                free(values);
                ptd_trace_result_destroy(result);
                return NULL;
            }

            for (size_t j = 0; j < n_edges; j++) {
                result->edge_probs[i][j] = values[trace->edge_probs[i][j]];
                result->vertex_targets[i][j] = trace->vertex_targets[i][j];
            }
        } else {
            result->edge_probs[i] = NULL;
            result->vertex_targets[i] = NULL;
        }
    }

    free(values);
    return result;
}

/**
 * Destroy trace evaluation result and free all memory
 */
void ptd_trace_result_destroy(struct ptd_trace_result *result) {
    if (result == NULL) {
        return;
    }

    if (result->vertex_rates != NULL) {
        free(result->vertex_rates);
    }

    if (result->edge_probs != NULL) {
        for (size_t i = 0; i < result->n_vertices; i++) {
            if (result->edge_probs[i] != NULL) {
                free(result->edge_probs[i]);
            }
        }
        free(result->edge_probs);
    }

    if (result->edge_probs_lengths != NULL) {
        free(result->edge_probs_lengths);
    }

    if (result->vertex_targets != NULL) {
        for (size_t i = 0; i < result->n_vertices; i++) {
            if (result->vertex_targets[i] != NULL) {
                free(result->vertex_targets[i]);
            }
        }
        free(result->vertex_targets);
    }

    if (result->vertex_targets_lengths != NULL) {
        free(result->vertex_targets_lengths);
    }

    free(result);
}

/**
 * Instantiate a complete graph from trace evaluation result
 *
 * Creates a new graph with all vertices and edges from the evaluated trace.
 * This mirrors the Python instantiate_from_trace() function.
 */
struct ptd_graph *ptd_instantiate_from_trace(
    const struct ptd_trace_result *result,
    const struct ptd_elimination_trace *trace
) {
    // Validate inputs
    if (result == NULL || trace == NULL) {
        sprintf((char*)ptd_err, "ptd_instantiate_from_trace: NULL input");
        return NULL;
    }

    if (result->n_vertices != trace->n_vertices) {
        sprintf((char*)ptd_err, "ptd_instantiate_from_trace: vertex count mismatch");
        return NULL;
    }

    // Create new graph
    struct ptd_graph *graph = ptd_graph_create(trace->state_length);
    if (graph == NULL) {
        return NULL;
    }

    // Create AVL tree for vertex lookup
    struct ptd_avl_tree *avl_tree = ptd_avl_tree_create(graph->state_length);
    if (avl_tree == NULL) {
        sprintf((char*)ptd_err, "ptd_instantiate_from_trace: failed to create AVL tree");
        ptd_graph_destroy(graph);
        return NULL;
    }

    // Build state-to-vertex mapping
    struct ptd_vertex **vertices = (struct ptd_vertex **)malloc(trace->n_vertices * sizeof(struct ptd_vertex *));
    if (vertices == NULL) {
        sprintf((char*)ptd_err, "ptd_instantiate_from_trace: failed to allocate vertex array");
        ptd_avl_tree_destroy(avl_tree);
        ptd_graph_destroy(graph);
        return NULL;
    }

    // Get starting vertex
    struct ptd_vertex *start_vertex = graph->starting_vertex;

    // Check if starting vertex matches trace->states[starting_vertex_idx]
    bool start_matches = true;
    for (size_t j = 0; j < trace->state_length; j++) {
        if (start_vertex->state[j] != trace->states[trace->starting_vertex_idx][j]) {
            start_matches = false;
            break;
        }
    }

    // Create all vertices
    for (size_t i = 0; i < trace->n_vertices; i++) {
        // Check if this is the starting vertex
        if (i == trace->starting_vertex_idx && start_matches) {
            vertices[i] = start_vertex;
            // Add to AVL tree
            ptd_avl_tree_find_or_insert(avl_tree, start_vertex->state, start_vertex);
        } else {
            // Find or create vertex with this state
            vertices[i] = ptd_find_or_create_vertex(graph, avl_tree, trace->states[i]);
            if (vertices[i] == NULL) {
                sprintf((char*)ptd_err, "ptd_instantiate_from_trace: failed to create vertex %zu", i);
                free(vertices);
                ptd_avl_tree_destroy(avl_tree);
                ptd_graph_destroy(graph);
                return NULL;
            }
        }
    }

    // Add edges
    for (size_t i = 0; i < trace->n_vertices; i++) {
        double inv_rate = result->vertex_rates[i];

        // Skip if absorbing (rate = 0 means inv_rate would be 0 or invalid)
        if (inv_rate <= 0.0 || result->vertex_targets_lengths[i] == 0) {
            continue;
        }

        struct ptd_vertex *from_vertex = vertices[i];

        for (size_t j = 0; j < result->vertex_targets_lengths[i]; j++) {
            double prob = result->edge_probs[i][j];
            size_t to_idx = result->vertex_targets[i][j];

            // Convert probability back to weight: weight = prob / inv_rate
            // Since rate = 1 / sum(weights), we have inv_rate = sum(weights)
            // And prob = weight / sum(weights) = weight * rate
            // So weight = prob / rate = prob * (1 / inv_rate) = prob / inv_rate
            double weight = prob / inv_rate;

            struct ptd_vertex *to_vertex = vertices[to_idx];

            // Add edge
            struct ptd_edge *edge = ptd_graph_add_edge(from_vertex, to_vertex, weight);
            if (edge == NULL) {
                sprintf((char*)ptd_err, "ptd_instantiate_from_trace: failed to add edge from %zu to %zu", i, to_idx);
                free(vertices);
                ptd_avl_tree_destroy(avl_tree);
                ptd_graph_destroy(graph);
                return NULL;
            }
        }
    }

    // Cleanup
    free(vertices);
    ptd_avl_tree_destroy(avl_tree);

    return graph;
}

/**
 * Build reward computation graph from trace evaluation result
 *
 * Converts the evaluated trace result (vertex rates, edge probabilities, targets)
 * into a reward_compute structure that can be used for PDF/PMF computation.
 *
 * This is the trace-based equivalent of ptd_graph_ex_absorbation_time_comp_graph().
 *
 * @param result Evaluation result from ptd_evaluate_trace()
 * @param graph Original graph structure (for metadata)
 * @return Reward computation structure, or NULL on error
 */
struct ptd_desc_reward_compute *ptd_build_reward_compute_from_trace(
    const struct ptd_trace_result *result,
    struct ptd_graph *graph
) {
    if (result == NULL) {
        snprintf((char*)ptd_err, sizeof(ptd_err), "Trace result is NULL");
        return NULL;
    }

    if (graph == NULL) {
        snprintf((char*)ptd_err, sizeof(ptd_err), "Graph is NULL");
        return NULL;
    }

    size_t n_vertices = result->n_vertices;
    struct ptd_reward_increase *commands = NULL;
    size_t command_index = 0;

    // Phase 1: Add vertex rate commands
    // For each vertex, add self-command with its rate
    // Command format: from[i] *= (rate - 1) when from == to
    // This is represented as: add_command(i, i, rate, ...)

    for (size_t i = 0; i < n_vertices; i++) {
        double rate = result->vertex_rates[i];

        // Starting vertex or absorbing state gets rate 0
        if (i == 0 || result->edge_probs_lengths[i] == 0) {
            commands = add_command(commands, i, i, 0.0, command_index++);
        } else {
            commands = add_command(commands, i, i, rate, command_index++);
        }
    }

    // Phase 2: Add edge probability commands
    // Traverse vertices in reverse order (topological order for DAG)
    // For each edge, add command: from[i] += to[j] * probability

    for (size_t ii = 0; ii < n_vertices; ii++) {
        size_t i = n_vertices - ii - 1;  // Reverse order

        size_t n_edges = result->edge_probs_lengths[i];

        for (size_t j = 0; j < n_edges; j++) {
            double prob = result->edge_probs[i][j];
            size_t target = result->vertex_targets[i][j];

            // Add command: vertex[i] += vertex[target] * prob
            commands = add_command(commands, i, target, prob, command_index++);
        }
    }

    // Phase 3: Add terminating command with NAN
    commands = add_command(commands, 0, 0, NAN, command_index);

    // Create and return result structure
    struct ptd_desc_reward_compute *res =
        (struct ptd_desc_reward_compute *) malloc(sizeof(*res));

    if (res == NULL) {
        snprintf((char*)ptd_err, sizeof(ptd_err),
                "Failed to allocate reward_compute structure");
        free(commands);
        return NULL;
    }

    res->length = command_index;
    res->commands = commands;

    return res;
}

/* ==================================================================
 * Trace Caching - Internal Functions
 * ==================================================================
 * These functions implement automatic caching of elimination traces
 * to avoid O(n³) re-recording for graphs with identical structure.
 */

/**
 * Get path to cache directory, creating it if needed
 * Returns newly allocated string that must be freed by caller
 */
static char *get_cache_dir(void) {
    char *home = getenv("HOME");
    if (home == NULL) {
        return NULL;
    }

    // Allocate space for ~/.phasic_cache/traces
    size_t len = strlen(home) + 40;
    char *cache_dir = (char *)malloc(len);
    if (cache_dir == NULL) {
        return NULL;
    }

    snprintf(cache_dir, len, "%s/.phasic_cache", home);
    mkdir(cache_dir, 0755);  // Create if doesn't exist

    snprintf(cache_dir, len, "%s/.phasic_cache/traces", home);
    mkdir(cache_dir, 0755);  // Create traces subdirectory

    return cache_dir;
}

/**
 * Get full path to cached trace file for given hash
 * Returns newly allocated string that must be freed by caller
 */
static char *get_cache_path(const char *hash_hex) {
    char *cache_dir = get_cache_dir();
    if (cache_dir == NULL) {
        return NULL;
    }

    size_t len = strlen(cache_dir) + strlen(hash_hex) + 10;
    char *path = (char *)malloc(len);
    if (path == NULL) {
        free(cache_dir);
        return NULL;
    }

    snprintf(path, len, "%s/%s.json", cache_dir, hash_hex);
    free(cache_dir);

    return path;
}

/**
 * Serialize trace operation to JSON string
 * Appends to the provided string buffer
 */
static void operation_to_json(const struct ptd_trace_operation *op,
                              char **buffer, size_t *buffer_len, size_t *buffer_cap) {
    // Ensure buffer has space
    while (*buffer_len + 512 > *buffer_cap) {
        *buffer_cap *= 2;
        *buffer = (char *)realloc(*buffer, *buffer_cap);
    }

    *buffer_len += snprintf(*buffer + *buffer_len, *buffer_cap - *buffer_len,
                           "{\"op_type\":%d,\"const_value\":%.17g,\"param_idx\":%zu,",
                           op->op_type, op->const_value, op->param_idx);

    // Coefficients array
    *buffer_len += snprintf(*buffer + *buffer_len, *buffer_cap - *buffer_len,
                           "\"coefficients\":[");
    for (size_t i = 0; i < op->coefficients_length; i++) {
        if (i > 0) {
            *buffer_len += snprintf(*buffer + *buffer_len, *buffer_cap - *buffer_len, ",");
        }
        *buffer_len += snprintf(*buffer + *buffer_len, *buffer_cap - *buffer_len,
                               "%.17g", op->coefficients[i]);
    }
    *buffer_len += snprintf(*buffer + *buffer_len, *buffer_cap - *buffer_len, "],");

    // Operands array
    *buffer_len += snprintf(*buffer + *buffer_len, *buffer_cap - *buffer_len,
                           "\"operands\":[");
    for (size_t i = 0; i < op->operands_length; i++) {
        if (i > 0) {
            *buffer_len += snprintf(*buffer + *buffer_len, *buffer_cap - *buffer_len, ",");
        }
        *buffer_len += snprintf(*buffer + *buffer_len, *buffer_cap - *buffer_len,
                               "%zu", op->operands[i]);
    }
    *buffer_len += snprintf(*buffer + *buffer_len, *buffer_cap - *buffer_len, "]}");
}

/**
 * Serialize elimination trace to JSON string (internal use only)
 * Returns newly allocated JSON string, or NULL on error
 * Caller must free the returned string
 */
static char *trace_to_json_internal(const struct ptd_elimination_trace *trace) {
    if (trace == NULL) {
        return NULL;
    }

    // Start with reasonable buffer size
    size_t buffer_cap = 8192;
    size_t buffer_len = 0;
    char *buffer = (char *)malloc(buffer_cap);
    if (buffer == NULL) {
        return NULL;
    }

    // Start JSON object
    buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
                          "{\"n_vertices\":%zu,\"param_length\":%zu,\"state_length\":%zu,",
                          trace->n_vertices, trace->param_length, trace->state_length);

    // Operations array
    buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
                          "\"operations\":[");
    for (size_t i = 0; i < trace->operations_length; i++) {
        if (i > 0) {
            buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
        }
        operation_to_json(&trace->operations[i], &buffer, &buffer_len, &buffer_cap);
    }
    buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "],");

    // Vertex rates array
    buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
                          "\"vertex_rates\":[");
    for (size_t i = 0; i < trace->n_vertices; i++) {
        if (i > 0) {
            buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
        }
        buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
                              "%zu", trace->vertex_rates[i]);
    }
    buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "],");

    // Edge probs arrays (2D)
    buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
                          "\"edge_probs\":[");
    for (size_t i = 0; i < trace->n_vertices; i++) {
        if (i > 0) {
            buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
        }
        buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "[");
        for (size_t j = 0; j < trace->edge_probs_lengths[i]; j++) {
            if (j > 0) {
                buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
            }
            buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
                                  "%zu", trace->edge_probs[i][j]);
        }
        buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "]");
    }
    buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "],");

    // Edge probs lengths
    buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
                          "\"edge_probs_lengths\":[");
    for (size_t i = 0; i < trace->n_vertices; i++) {
        if (i > 0) {
            buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
        }
        buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
                              "%zu", trace->edge_probs_lengths[i]);
    }
    buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "],");

    // Vertex targets arrays (2D)
    buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
                          "\"vertex_targets\":[");
    for (size_t i = 0; i < trace->n_vertices; i++) {
        if (i > 0) {
            buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
        }
        buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "[");
        for (size_t j = 0; j < trace->vertex_targets_lengths[i]; j++) {
            if (j > 0) {
                buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
            }
            buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
                                  "%zu", trace->vertex_targets[i][j]);
        }
        buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "]");
    }
    buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "],");

    // Vertex targets lengths
    buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
                          "\"vertex_targets_lengths\":[");
    for (size_t i = 0; i < trace->n_vertices; i++) {
        if (i > 0) {
            buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
        }
        buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
                              "%zu", trace->vertex_targets_lengths[i]);
    }
    buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "],");

    // States arrays (2D)
    buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
                          "\"states\":[");
    for (size_t i = 0; i < trace->n_vertices; i++) {
        if (i > 0) {
            buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
        }
        buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "[");
        for (size_t j = 0; j < trace->state_length; j++) {
            if (j > 0) {
                buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
            }
            buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
                                  "%d", trace->states[i][j]);
        }
        buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "]");
    }
    buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "],");

    // Starting vertex index
    buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
                          "\"starting_vertex_idx\":%zu,", trace->starting_vertex_idx);

    // Is discrete
    buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
                          "\"is_discrete\":%s", trace->is_discrete ? "true" : "false");

    // Close JSON object
    buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "}");

    return buffer;
}

/**
 * Simple JSON parser helpers for trace deserialization
 */

/* Skip whitespace in JSON string */
static const char *skip_whitespace(const char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') {
        s++;
    }
    return s;
}

/* Find the closing bracket/brace, accounting for nesting */
static const char *find_closing(const char *s, char open, char close) {
    int depth = 1;
    s++; // Skip opening bracket
    while (*s && depth > 0) {
        if (*s == open) depth++;
        else if (*s == close) depth--;
        if (depth > 0) s++;
    }
    return s;
}

/* Parse a size_t value from JSON */
static size_t parse_size_t(const char *s) {
    return (size_t)strtoull(s, NULL, 10);
}

/* Parse an int value from JSON */
static int parse_int(const char *s) {
    return (int)strtol(s, NULL, 10);
}

/* Parse a double value from JSON */
static double parse_double(const char *s) {
    return strtod(s, NULL);
}

/* Parse a bool value from JSON */
static bool parse_bool(const char *s) {
    s = skip_whitespace(s);
    return (strncmp(s, "true", 4) == 0);
}

/* Find a JSON field by name and return pointer to its value */
static const char *find_field(const char *json, const char *field_name) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\":", field_name);
    const char *field = strstr(json, search);
    if (field == NULL) {
        return NULL;
    }
    field += strlen(search);
    return skip_whitespace(field);
}

/* Parse array of size_t values */
static size_t *parse_size_t_array(const char *json, size_t *out_length) {
    json = skip_whitespace(json);
    if (*json != '[') {
        return NULL;
    }

    // Count elements
    size_t count = 0;
    const char *p = json + 1;
    while (*p && *p != ']') {
        if (*p >= '0' && *p <= '9') {
            count++;
            while (*p && *p != ',' && *p != ']') p++;
        }
        if (*p == ',') p++;
        p = skip_whitespace(p);
    }

    if (count == 0) {
        *out_length = 0;
        return NULL;
    }

    // Allocate and parse
    size_t *arr = (size_t *)malloc(count * sizeof(size_t));
    if (arr == NULL) {
        return NULL;
    }

    p = json + 1;
    for (size_t i = 0; i < count; i++) {
        p = skip_whitespace(p);
        arr[i] = parse_size_t(p);
        while (*p && *p != ',' && *p != ']') p++;
        if (*p == ',') p++;
    }

    *out_length = count;
    return arr;
}

/* Parse array of double values */
static double *parse_double_array(const char *json, size_t *out_length) {
    json = skip_whitespace(json);
    if (*json != '[') {
        return NULL;
    }

    // Count elements
    size_t count = 0;
    const char *p = json + 1;
    while (*p && *p != ']') {
        p = skip_whitespace(p);
        if (*p == '-' || (*p >= '0' && *p <= '9')) {
            count++;
            while (*p && *p != ',' && *p != ']') p++;
        }
        if (*p == ',') p++;
    }

    if (count == 0) {
        *out_length = 0;
        return NULL;
    }

    // Allocate and parse
    double *arr = (double *)malloc(count * sizeof(double));
    if (arr == NULL) {
        return NULL;
    }

    p = json + 1;
    for (size_t i = 0; i < count; i++) {
        p = skip_whitespace(p);
        arr[i] = parse_double(p);
        while (*p && *p != ',' && *p != ']') p++;
        if (*p == ',') p++;
    }

    *out_length = count;
    return arr;
}

/* Parse array of int values */
static int *parse_int_array(const char *json, size_t *out_length) {
    json = skip_whitespace(json);
    if (*json != '[') {
        return NULL;
    }

    // Count elements
    size_t count = 0;
    const char *p = json + 1;
    while (*p && *p != ']') {
        p = skip_whitespace(p);
        if (*p == '-' || (*p >= '0' && *p <= '9')) {
            count++;
            while (*p && *p != ',' && *p != ']') p++;
        }
        if (*p == ',') p++;
    }

    if (count == 0) {
        *out_length = 0;
        return NULL;
    }

    // Allocate and parse
    int *arr = (int *)malloc(count * sizeof(int));
    if (arr == NULL) {
        return NULL;
    }

    p = json + 1;
    for (size_t i = 0; i < count; i++) {
        p = skip_whitespace(p);
        arr[i] = parse_int(p);
        while (*p && *p != ',' && *p != ']') p++;
        if (*p == ',') p++;
    }

    *out_length = count;
    return arr;
}

/* Parse a trace operation from JSON object */
static int parse_operation(const char *json, struct ptd_trace_operation *op) {
    const char *field;

    // op_type
    field = find_field(json, "op_type");
    if (field == NULL) return -1;
    op->op_type = (enum ptd_trace_op_type)parse_int(field);

    // const_value
    field = find_field(json, "const_value");
    if (field == NULL) return -1;
    op->const_value = parse_double(field);

    // param_idx
    field = find_field(json, "param_idx");
    if (field == NULL) return -1;
    op->param_idx = parse_size_t(field);

    // coefficients
    field = find_field(json, "coefficients");
    if (field == NULL) return -1;
    op->coefficients = parse_double_array(field, &op->coefficients_length);

    // operands
    field = find_field(json, "operands");
    if (field == NULL) return -1;
    op->operands = parse_size_t_array(field, &op->operands_length);

    return 0;
}

/**
 * Load elimination trace from cache file (internal use only)
 * Returns trace if found in cache, NULL otherwise
 */
static struct ptd_elimination_trace *load_trace_from_cache(const char *hash_hex) {
    char *path = get_cache_path(hash_hex);
    if (path == NULL) {
        return NULL;
    }

    // Check if file exists
    if (access(path, F_OK) != 0) {
        free(path);
        return NULL;
    }

    // Read file
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        DEBUG_PRINT("WARNING: failed to open cache file for reading: %s\n", path);
        free(path);
        return NULL;
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0) {
        fclose(f);
        free(path);
        return NULL;
    }

    // Read entire file into memory
    char *json = (char *)malloc(file_size + 1);
    if (json == NULL) {
        fclose(f);
        free(path);
        return NULL;
    }

    size_t bytes_read = fread(json, 1, file_size, f);
    fclose(f);

    if (bytes_read != (size_t)file_size) {
        DEBUG_PRINT("WARNING: failed to read complete cache file: %s\n", path);
        free(json);
        free(path);
        return NULL;
    }
    json[file_size] = '\0';

    // Allocate trace structure
    struct ptd_elimination_trace *trace = (struct ptd_elimination_trace *)calloc(1, sizeof(*trace));
    if (trace == NULL) {
        free(json);
        free(path);
        return NULL;
    }

    // Declare all variables at the beginning to avoid goto issues
    const char *field;
    const char *p;
    size_t op_count;
    size_t vr_len;
    size_t epl_len;
    size_t vtl_len;
    size_t len;
    int depth;

    // Parse metadata fields
    field = find_field(json, "n_vertices");
    if (field == NULL) goto error;
    trace->n_vertices = parse_size_t(field);

    field = find_field(json, "param_length");
    if (field == NULL) goto error;
    trace->param_length = parse_size_t(field);

    field = find_field(json, "state_length");
    if (field == NULL) goto error;
    trace->state_length = parse_size_t(field);

    field = find_field(json, "starting_vertex_idx");
    if (field == NULL) goto error;
    trace->starting_vertex_idx = parse_size_t(field);

    field = find_field(json, "is_discrete");
    if (field == NULL) goto error;
    trace->is_discrete = parse_bool(field);

    // Parse operations array
    field = find_field(json, "operations");
    if (field == NULL) goto error;

    field = skip_whitespace(field);
    if (*field != '[') goto error;

    // Count operations
    op_count = 0;
    p = field + 1;
    depth = 0;
    while (*p) {
        if (*p == '{') {
            if (depth == 0) op_count++;
            depth++;
        } else if (*p == '}') {
            depth--;
        } else if (*p == ']' && depth == 0) {
            break;
        }
        p++;
    }

    trace->operations_length = op_count;
    trace->operations = (struct ptd_trace_operation *)calloc(op_count, sizeof(struct ptd_trace_operation));
    if (trace->operations == NULL) goto error;

    // Parse each operation
    p = field + 1;
    for (size_t i = 0; i < op_count; i++) {
        p = skip_whitespace(p);
        if (*p != '{') goto error;

        const char *op_end = find_closing(p, '{', '}');
        if (op_end == NULL) goto error;

        if (parse_operation(p, &trace->operations[i]) != 0) {
            goto error;
        }

        p = op_end + 1;
        if (*p == ',') p++;
    }

    // Parse vertex_rates
    field = find_field(json, "vertex_rates");
    if (field == NULL) goto error;
    trace->vertex_rates = parse_size_t_array(field, &vr_len);
    if (vr_len != trace->n_vertices) goto error;

    // Parse edge_probs_lengths
    field = find_field(json, "edge_probs_lengths");
    if (field == NULL) goto error;
    trace->edge_probs_lengths = parse_size_t_array(field, &epl_len);
    if (epl_len != trace->n_vertices) goto error;

    // Parse edge_probs (2D array)
    field = find_field(json, "edge_probs");
    if (field == NULL) goto error;
    field = skip_whitespace(field);
    if (*field != '[') goto error;

    trace->edge_probs = (size_t **)malloc(trace->n_vertices * sizeof(size_t*));
    if (trace->edge_probs == NULL) goto error;

    p = field + 1;
    for (size_t i = 0; i < trace->n_vertices; i++) {
        p = skip_whitespace(p);
        if (*p != '[') goto error;

        trace->edge_probs[i] = parse_size_t_array(p, &len);
        if (len != trace->edge_probs_lengths[i]) goto error;

        p = find_closing(p, '[', ']');
        if (*p == ']') p++;
        if (*p == ',') p++;
    }

    // Parse vertex_targets_lengths
    field = find_field(json, "vertex_targets_lengths");
    if (field == NULL) goto error;
    trace->vertex_targets_lengths = parse_size_t_array(field, &vtl_len);
    if (vtl_len != trace->n_vertices) goto error;

    // Parse vertex_targets (2D array)
    field = find_field(json, "vertex_targets");
    if (field == NULL) goto error;
    field = skip_whitespace(field);
    if (*field != '[') goto error;

    trace->vertex_targets = (size_t **)malloc(trace->n_vertices * sizeof(size_t*));
    if (trace->vertex_targets == NULL) goto error;

    p = field + 1;
    for (size_t i = 0; i < trace->n_vertices; i++) {
        p = skip_whitespace(p);
        if (*p != '[') goto error;

        trace->vertex_targets[i] = parse_size_t_array(p, &len);
        if (len != trace->vertex_targets_lengths[i]) goto error;

        p = find_closing(p, '[', ']');
        if (*p == ']') p++;
        if (*p == ',') p++;
    }

    // Parse states (2D array of ints)
    field = find_field(json, "states");
    if (field == NULL) goto error;
    field = skip_whitespace(field);
    if (*field != '[') goto error;

    trace->states = (int **)malloc(trace->n_vertices * sizeof(int*));
    if (trace->states == NULL) goto error;

    p = field + 1;
    for (size_t i = 0; i < trace->n_vertices; i++) {
        p = skip_whitespace(p);
        if (*p != '[') goto error;

        trace->states[i] = parse_int_array(p, &len);
        if (len != trace->state_length) goto error;

        p = find_closing(p, '[', ']');
        if (*p == ']') p++;
        if (*p == ',') p++;
    }

    free(json);
    free(path);

    DEBUG_PRINT("INFO: loaded elimination trace from cache (%s): %zu operations, %zu vertices\n",
                hash_hex, trace->operations_length, trace->n_vertices);

    return trace;

error:
    DEBUG_PRINT("WARNING: failed to deserialize trace from cache: %s\n", path);
    if (trace != NULL) {
        ptd_elimination_trace_destroy(trace);
    }
    free(json);
    free(path);
    return NULL;
}

/**
 * Save elimination trace to cache file (internal use only)
 * Returns true on success, false on error
 */
static bool save_trace_to_cache(const char *hash_hex,
                                const struct ptd_elimination_trace *trace) {
    if (hash_hex == NULL || trace == NULL) {
        return false;
    }

    char *path = get_cache_path(hash_hex);
    if (path == NULL) {
        return false;
    }

    // Serialize trace to JSON
    char *json = trace_to_json_internal(trace);
    if (json == NULL) {
        free(path);
        return false;
    }

    // Write to file
    FILE *f = fopen(path, "w");
    if (f == NULL) {
        DEBUG_PRINT("WARNING: failed to open cache file for writing: %s\n", path);
        free(path);
        free(json);
        return false;
    }

    size_t json_len = strlen(json);
    size_t written = fwrite(json, 1, json_len, f);
    fclose(f);

    bool success = (written == json_len);

    if (success) {
        DEBUG_PRINT("INFO: saved trace to cache (%zu bytes): %s\n", json_len, path);
    } else {
        DEBUG_PRINT("WARNING: failed to write complete trace to cache\n");
    }

    free(path);
    free(json);

    return success;
}
