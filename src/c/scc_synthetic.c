/**
 * @file scc_synthetic.c
 * @brief WP-1: synthetic SCC graph constructor for hierarchical elimination.
 *
 * Given a parent graph's SCC decomposition and a target SCC, build a
 * self-contained parameterised phase-type graph that wraps the SCC's
 * internal vertices with a synthetic source vertex and a synthetic
 * absorbing vertex. The resulting graph can be passed directly to
 * ptd_graph_ex_absorbation_time_comp_graph_parameterized.
 *
 * Why a separate file: keeps the new code isolated from the
 * 13k-line phasic.c, which makes review and reverts cleaner during
 * the hierar-elimin-cache branch.
 *
 * Design choices (see wp1-plan.md and hierar-elimin-cache-reference.md):
 *
 * - Construction is at the C level. We bypass the C++ Vertex::add_edge
 *   wrapper that locks edge mode (the cause of the existing
 *   SCCVertex::as_graph crash on duplicate-state graphs).
 *
 * - Aux-style constant edges (coefficients_length == 0) are copied
 *   verbatim. ptd_graph_update_weights skips them; converting to
 *   all-zero parameterised would silently zero them out.
 *
 * - Vertex identity is by pointer (or, equivalently, parent-graph
 *   index). State vectors are NEVER used to identify existing
 *   vertices — they may legally collide (Toy-D's aux vertex shares
 *   the all-zero state with the starting vertex).
 *
 * - Synthetic source/absorbing edges are placeholders: a single
 *   coefficient of value 1.0. Parent-specific external coefficient
 *   values live in the metadata's upstream_in_edges and
 *   downstream_out_edges, to be wired by the composer (WP-5) via
 *   the EXTERNAL pointer kind (WP-3).
 *
 * - Canonical vertex ordering within each category is by
 *   lexicographic state vector, with out-edge-signature tiebreak.
 *   This is what makes ptd_graph_content_hash invariant across
 *   parents that share an SCC structurally (verified in
 *   Experiment 1, see wp1-experiments.md).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#include "../../api/c/phasic.h"
#include "../../api/c/phasic_hash.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

/* ptd_err is declared extern PTD_TLS in api/c/phasic.h:112 with size
 * 4096. We use a named constant rather than the literal in multiple
 * places. */
#define sizeof_ptd_err 4096

/* Match phasic.c's PHASIC_DISABLE_CACHE convention. */
static int ptd_scc_cache_disabled(void) {
    const char *v = getenv("PHASIC_DISABLE_CACHE");
    return v != NULL && v[0] == '1' && v[1] == '\0';
}

/* Build the cache file path for a per-SCC PRC:
 *   <home>/.phasic_cache/parameterized_reward_compute/scc_<hash_hex>.bin
 * Creates parent directories on demand (mkdir -p style).
 *
 * Returns 0 on success, -1 on error (sets ptd_err). */
static int ptd_scc_build_cache_path(
        const struct ptd_graph *synth, char *buf, size_t buf_len)
{
    const char *home = getenv("HOME");
    if (home == NULL) {
        snprintf(ptd_err, sizeof_ptd_err,
                 "ptd_scc: HOME not set");
        return -1;
    }
    char parent[PATH_MAX];
    int n = snprintf(parent, sizeof(parent), "%s/.phasic_cache", home);
    if (n < 0 || (size_t)n >= sizeof(parent)) {
        snprintf(ptd_err, sizeof_ptd_err,
                 "ptd_scc: cache parent path too long");
        return -1;
    }
    struct stat st;
    if (stat(parent, &st) != 0) {
        if (mkdir(parent, 0755) != 0 && errno != EEXIST) {
            snprintf(ptd_err, sizeof_ptd_err,
                     "ptd_scc: cannot create %s: %s", parent, strerror(errno));
            return -1;
        }
    }
    char dir[PATH_MAX];
    n = snprintf(dir, sizeof(dir),
                 "%s/.phasic_cache/parameterized_reward_compute", home);
    if (n < 0 || (size_t)n >= sizeof(dir)) {
        snprintf(ptd_err, sizeof_ptd_err,
                 "ptd_scc: cache dir path too long");
        return -1;
    }
    if (stat(dir, &st) != 0) {
        if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
            snprintf(ptd_err, sizeof_ptd_err,
                     "ptd_scc: cannot create %s: %s", dir, strerror(errno));
            return -1;
        }
    }
    struct ptd_hash_result *hash = ptd_graph_content_hash(synth);
    if (hash == NULL) {
        snprintf(ptd_err, sizeof_ptd_err,
                 "ptd_scc: ptd_graph_content_hash failed");
        return -1;
    }
    /* "scc_" prefix distinguishes per-SCC entries from parent-level
     * Stage A2 entries that live in the same directory. */
    n = snprintf(buf, buf_len, "%s/scc_%s.bin", dir, hash->hash_hex);
    free(hash);
    if (n < 0 || (size_t)n >= buf_len) {
        snprintf(ptd_err, sizeof_ptd_err,
                 "ptd_scc: cache path too long");
        return -1;
    }
    return 0;
}


/* Append (parent_v, parent_e) to a dynamic edge-ref list. */
static int append_edge_ref(
        struct ptd_scc_external_edge_ref **list_out,
        size_t *length_out,
        size_t *capacity_out,
        size_t parent_v,
        size_t parent_e)
{
    if (*length_out == *capacity_out) {
        size_t new_cap = (*capacity_out == 0) ? 4 : (*capacity_out * 2);
        struct ptd_scc_external_edge_ref *new_list = (struct ptd_scc_external_edge_ref *)
                realloc(*list_out, new_cap * sizeof(*new_list));
        if (new_list == NULL) {
            return -1;
        }
        *list_out = new_list;
        *capacity_out = new_cap;
    }
    (*list_out)[*length_out].parent_vertex_idx = parent_v;
    (*list_out)[*length_out].parent_edge_idx = parent_e;
    (*length_out)++;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Helper: canonical comparator for vertex ordering within a category */
/* ------------------------------------------------------------------ */

/* qsort context. Holds the parent graph (for state vector access)
 * and a per-internal-vertex within-SCC position lookup table used
 * during out-edge signature tiebreaking. */
struct cat_sort_ctx {
    const struct ptd_graph *parent;
    /* For each parent vertex index v, scc_position[v] gives v's
     * position within the SCC's internal_vertices list, or
     * SIZE_MAX if v is not in this SCC. Used so the out-edge
     * signature can refer to within-SCC neighbours by their
     * SCC-relative position rather than by parent-graph index
     * (which would break cross-parent invariance). */
    const size_t *scc_position;
};

/* Forward declaration. Implemented below. */
static int edge_signature_compare(
        const struct ptd_edge *e1,
        const struct ptd_edge *e2,
        const size_t *scc_position);

/* Tier 2 + tier 3 comparator, used to sort vertices within a
 * category. State vectors already known to be tied (caller
 * handles tier 1). */
static int compare_within_category_tiebreak(
        size_t a,
        size_t b,
        const struct cat_sort_ctx *ctx)
{
    const struct ptd_vertex *va = ctx->parent->vertices[a];
    const struct ptd_vertex *vb = ctx->parent->vertices[b];

    /* Sort each vertex's out-edges by signature, then compare in
     * lock step. */
    size_t na = va->edges_length;
    size_t nb = vb->edges_length;

    /* If edge counts differ, fewer edges sort first. */
    if (na != nb) {
        return (na < nb) ? -1 : 1;
    }

    /* Build sorted index arrays. */
    size_t *order_a = NULL;
    size_t *order_b = NULL;
    int result = 0;

    if (na > 0) {
        order_a = (size_t *)malloc(na * sizeof(size_t));
        order_b = (size_t *)malloc(nb * sizeof(size_t));
        if (order_a == NULL || order_b == NULL) {
            free(order_a);
            free(order_b);
            /* Tier 3: parent-index fallback. */
            return (a < b) ? -1 : (a > b) ? 1 : 0;
        }
        for (size_t i = 0; i < na; ++i) order_a[i] = i;
        for (size_t i = 0; i < nb; ++i) order_b[i] = i;

        /* Selection sort. */
        for (size_t i = 0; i < na; ++i) {
            size_t min_i = i;
            for (size_t j = i + 1; j < na; ++j) {
                int c = edge_signature_compare(
                    va->edges[order_a[j]],
                    va->edges[order_a[min_i]],
                    ctx->scc_position);
                if (c < 0) min_i = j;
            }
            size_t tmp = order_a[i];
            order_a[i] = order_a[min_i];
            order_a[min_i] = tmp;
        }
        for (size_t i = 0; i < nb; ++i) {
            size_t min_i = i;
            for (size_t j = i + 1; j < nb; ++j) {
                int c = edge_signature_compare(
                    vb->edges[order_b[j]],
                    vb->edges[order_b[min_i]],
                    ctx->scc_position);
                if (c < 0) min_i = j;
            }
            size_t tmp = order_b[i];
            order_b[i] = order_b[min_i];
            order_b[min_i] = tmp;
        }

        for (size_t i = 0; i < na; ++i) {
            int c = edge_signature_compare(
                va->edges[order_a[i]],
                vb->edges[order_b[i]],
                ctx->scc_position);
            if (c != 0) {
                result = c;
                break;
            }
        }

        free(order_a);
        free(order_b);
    }

    if (result != 0) return result;

    /* Tier 3: parent-index fallback. */
    return (a < b) ? -1 : (a > b) ? 1 : 0;
}

static int edge_signature_compare(
        const struct ptd_edge *e1,
        const struct ptd_edge *e2,
        const size_t *scc_position)
{
    /* Key 1: target SCC position (SIZE_MAX for external targets,
     * which compare equal). */
    size_t pos1 = scc_position[e1->to->index];
    size_t pos2 = scc_position[e2->to->index];
    if (pos1 != pos2) {
        if (pos1 == SIZE_MAX) return 1;   /* external sorts last */
        if (pos2 == SIZE_MAX) return -1;
        return (pos1 < pos2) ? -1 : 1;
    }

    /* Key 2: coefficients_length. */
    if (e1->coefficients_length != e2->coefficients_length) {
        return (e1->coefficients_length < e2->coefficients_length) ? -1 : 1;
    }

    /* Key 3: coefficient values, lexicographically. */
    for (size_t k = 0; k < e1->coefficients_length; ++k) {
        if (e1->coefficients[k] < e2->coefficients[k]) return -1;
        if (e1->coefficients[k] > e2->coefficients[k]) return 1;
    }

    /* Key 4: weight. */
    if (e1->weight < e2->weight) return -1;
    if (e1->weight > e2->weight) return 1;

    return 0;
}

/* Combine tier 1 + tier 2 + tier 3 into a single comparator. */
static int compare_canonical(
        size_t a,
        size_t b,
        const struct cat_sort_ctx *ctx)
{
    const struct ptd_vertex *va = ctx->parent->vertices[a];
    const struct ptd_vertex *vb = ctx->parent->vertices[b];

    /* Tier 1: state vector. */
    size_t state_len = ctx->parent->state_length;
    for (size_t i = 0; i < state_len; ++i) {
        if (va->state[i] < vb->state[i]) return -1;
        if (va->state[i] > vb->state[i]) return 1;
    }

    /* Tier 2 + tier 3. */
    return compare_within_category_tiebreak(a, b, ctx);
}

/* Sort an array of parent vertex indices in canonical order
 * (selection sort; arrays are small). */
static void sort_canonical(
        size_t *indices,
        size_t length,
        const struct cat_sort_ctx *ctx)
{
    for (size_t i = 0; i < length; ++i) {
        size_t min_i = i;
        for (size_t j = i + 1; j < length; ++j) {
            if (compare_canonical(indices[j], indices[min_i], ctx) < 0) {
                min_i = j;
            }
        }
        if (min_i != i) {
            size_t tmp = indices[i];
            indices[i] = indices[min_i];
            indices[min_i] = tmp;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Helper: manual edge construction.                                   */
/* ------------------------------------------------------------------ */

/* Mirrors the manual edge construction pattern used by
 * Vertex::add_aux_vertex (src/cpp/phasiccpp.cpp:337). Bypasses
 * ptd_graph_add_edge's coefficients_length > 0 requirement so we
 * can preserve aux-style edges with no coefficients.
 *
 * If coefficients_length == 0, coefficients must be NULL.
 * If coefficients_length > 0, coefficients_src is copied (the
 * resulting edge owns its coefficient array). */
static int add_edge_raw(
        struct ptd_vertex *from,
        struct ptd_vertex *to,
        double weight,
        const double *coefficients_src,
        size_t coefficients_length)
{
    struct ptd_edge *edge = (struct ptd_edge *)malloc(sizeof(*edge));
    if (edge == NULL) {
        return -1;
    }
    edge->to = to;
    edge->weight = weight;
    edge->coefficients_length = coefficients_length;

    if (coefficients_length > 0) {
        double *coeffs = (double *)malloc(coefficients_length * sizeof(double));
        if (coeffs == NULL) {
            free(edge);
            return -1;
        }
        memcpy(coeffs, coefficients_src, coefficients_length * sizeof(double));
        edge->coefficients = coeffs;
        edge->should_free_coefficients = true;
    } else {
        edge->coefficients = NULL;
        edge->should_free_coefficients = false;
    }

    /* Append to from->edges[]. */
    struct ptd_edge **new_edges = (struct ptd_edge **)realloc(
            from->edges,
            (from->edges_length + 1) * sizeof(struct ptd_edge *));
    if (new_edges == NULL) {
        if (edge->coefficients) free(edge->coefficients);
        free(edge);
        return -1;
    }
    from->edges = new_edges;
    from->edges[from->edges_length] = edge;
    from->edges_length++;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

double **ptd_scc_collect_external_anchors(
        const struct ptd_graph *synth,
        const struct ptd_scc_synthetic_metadata *meta,
        size_t *n_anchors_out)
{
    if (n_anchors_out == NULL) return NULL;
    *n_anchors_out = 0;
    if (synth == NULL || synth->vertices_length < 2) return NULL;

    /* Synthetic source is at index 0; absorbing is at last index.
     * Caller may pass meta=NULL — in that case, infer absorbing
     * from the graph (which is what we'd derive from meta anyway). */
    size_t abs_idx = (meta != NULL) ? (meta->n_vertices - 1)
                                    : (synth->vertices_length - 1);
    struct ptd_vertex *src = synth->vertices[0];

    /* Count: Type A = src->edges_length; Type C = number of
     * non-source non-absorbing vertices with at least one edge to
     * the absorbing vertex (one Type C edge per such vertex). */
    size_t n_a = src->edges_length;
    size_t n_c = 0;
    for (size_t v = 1; v < synth->vertices_length - 1; ++v) {
        struct ptd_vertex *vert = synth->vertices[v];
        for (size_t e = 0; e < vert->edges_length; ++e) {
            if (vert->edges[e]->to->index == abs_idx) {
                n_c++;
                /* No break: a vertex could in principle have
                 * multiple Type C edges, though WP-1 emits at
                 * most one per dual/downstream-connecting vertex.
                 * Counting all is robust. */
            }
        }
    }

    size_t total = n_a + n_c;
    if (total == 0) {
        *n_anchors_out = 0;
        return NULL;
    }

    double **anchors = (double **)malloc(total * sizeof(double *));
    if (anchors == NULL) {
        snprintf(ptd_err, 1024,
                 "ptd_scc_collect_external_anchors: oom");
        return NULL;
    }
    size_t idx = 0;

    /* The eliminator references edges via &edge->weight (the
     * concrete evaluated weight, set by update_weights), NOT via
     * &edge->coefficients[0] (the symbolic θ coefficient). So our
     * external anchors must be the placeholder edges' weight
     * slots, which is what the saved PRC's encoded pointers will
     * try to match.
     *
     * Type A edges (in synthetic-source-edge order). */
    for (size_t e = 0; e < src->edges_length; ++e) {
        struct ptd_edge *edge = src->edges[e];
        anchors[idx++] = &edge->weight;
    }

    /* Type C edges (in vertex-then-edge order). */
    for (size_t v = 1; v < synth->vertices_length - 1; ++v) {
        struct ptd_vertex *vert = synth->vertices[v];
        for (size_t e = 0; e < vert->edges_length; ++e) {
            if (vert->edges[e]->to->index == abs_idx) {
                anchors[idx++] = &vert->edges[e]->weight;
            }
        }
    }

    *n_anchors_out = total;
    return anchors;
}

void ptd_scc_synthetic_metadata_destroy(
        struct ptd_scc_synthetic_metadata *metadata)
{
    if (metadata == NULL) return;

    free(metadata->parent_indices);

    if (metadata->external_in_edges != NULL) {
        for (size_t i = 0; i < metadata->n_vertices; ++i) {
            free(metadata->external_in_edges[i]);
        }
        free(metadata->external_in_edges);
    }
    free(metadata->external_in_edges_lengths);

    if (metadata->external_out_edges != NULL) {
        for (size_t i = 0; i < metadata->n_vertices; ++i) {
            free(metadata->external_out_edges[i]);
        }
        free(metadata->external_out_edges);
    }
    free(metadata->external_out_edges_lengths);

    free(metadata);
}

struct ptd_graph *ptd_scc_build_synthetic_graph(
        const struct ptd_scc_graph *scc_graph,
        size_t scc_index,
        struct ptd_scc_synthetic_metadata **metadata_out)
{
    /* ---- Validation ---- */
    if (scc_graph == NULL || metadata_out == NULL) {
        snprintf(ptd_err, 1024,
                 "ptd_scc_build_synthetic_graph: NULL argument");
        return NULL;
    }
    *metadata_out = NULL;
    if (scc_index >= scc_graph->vertices_length) {
        snprintf(ptd_err, 1024,
                 "ptd_scc_build_synthetic_graph: scc_index %zu out of range "
                 "(have %zu SCCs)",
                 scc_index, scc_graph->vertices_length);
        return NULL;
    }

    const struct ptd_graph *parent = scc_graph->graph;
    const struct ptd_scc_vertex *scc = scc_graph->vertices[scc_index];
    size_t n_parent = parent->vertices_length;

    /* ---- 1. Identify internal vertices ---- */
    bool *is_internal = (bool *)calloc(n_parent, sizeof(bool));
    /* Within-SCC position table: scc_position[parent_v] = position
     * within scc->internal_vertices, or SIZE_MAX if v not in SCC. */
    size_t *scc_position = (size_t *)malloc(n_parent * sizeof(size_t));
    if (is_internal == NULL || scc_position == NULL) {
        free(is_internal); free(scc_position);
        snprintf(ptd_err, 1024,
                 "ptd_scc_build_synthetic_graph: out of memory");
        return NULL;
    }
    for (size_t v = 0; v < n_parent; ++v) scc_position[v] = SIZE_MAX;
    for (size_t i = 0; i < scc->internal_vertices_length; ++i) {
        size_t v_idx = scc->internal_vertices[i]->index;
        is_internal[v_idx] = true;
        scc_position[v_idx] = i;
    }

    /* ---- 2. Categorise upstream-connecting and
     *       downstream-connecting; record external edge refs. ---- */
    bool *is_upstream_connecting = (bool *)calloc(n_parent, sizeof(bool));
    bool *is_downstream_connecting = (bool *)calloc(n_parent, sizeof(bool));
    /* Edge-ref accumulators, indexed by parent vertex idx (we'll
     * shrink to per-upstream-connecting / per-downstream-connecting
     * arrays later, after we know the canonical order). */
    struct ptd_scc_external_edge_ref **upstream_in_per_pv =
        (struct ptd_scc_external_edge_ref **)calloc(n_parent, sizeof(*upstream_in_per_pv));
    size_t *upstream_in_len_per_pv = (size_t *)calloc(n_parent, sizeof(size_t));
    size_t *upstream_in_cap_per_pv = (size_t *)calloc(n_parent, sizeof(size_t));
    struct ptd_scc_external_edge_ref **downstream_out_per_pv =
        (struct ptd_scc_external_edge_ref **)calloc(n_parent, sizeof(*downstream_out_per_pv));
    size_t *downstream_out_len_per_pv = (size_t *)calloc(n_parent, sizeof(size_t));
    size_t *downstream_out_cap_per_pv = (size_t *)calloc(n_parent, sizeof(size_t));

    if (is_upstream_connecting == NULL || is_downstream_connecting == NULL ||
        upstream_in_per_pv == NULL || upstream_in_len_per_pv == NULL ||
        upstream_in_cap_per_pv == NULL || downstream_out_per_pv == NULL ||
        downstream_out_len_per_pv == NULL || downstream_out_cap_per_pv == NULL) {
        goto fail_oom;
    }

    /* Walk all parent edges. For each:
     *   - If from is internal and to is external: from is
     *     downstream-connecting; record edge ref.
     *   - If from is external and to is internal: to is
     *     upstream-connecting; record edge ref. */
    for (size_t v = 0; v < n_parent; ++v) {
        const struct ptd_vertex *vertex = parent->vertices[v];
        for (size_t e = 0; e < vertex->edges_length; ++e) {
            const struct ptd_edge *edge = vertex->edges[e];
            size_t target_v = edge->to->index;
            bool from_internal = is_internal[v];
            bool to_internal = is_internal[target_v];

            if (from_internal && !to_internal) {
                /* Internal -> external: v is downstream-connecting. */
                is_downstream_connecting[v] = true;
                if (append_edge_ref(
                        &downstream_out_per_pv[v],
                        &downstream_out_len_per_pv[v],
                        &downstream_out_cap_per_pv[v],
                        v, e) != 0) goto fail_oom;
            } else if (!from_internal && to_internal) {
                /* External -> internal: target_v is upstream-connecting. */
                is_upstream_connecting[target_v] = true;
                if (append_edge_ref(
                        &upstream_in_per_pv[target_v],
                        &upstream_in_len_per_pv[target_v],
                        &upstream_in_cap_per_pv[target_v],
                        v, e) != 0) goto fail_oom;
            }
        }
    }

    /* ---- 3. Build canonical ordered vertex lists per category. ---- */
    /* internal_only = internal \ (upstream_connecting ∪ downstream_connecting) */
    size_t n_uc = 0, n_io = 0, n_dc = 0;
    for (size_t v = 0; v < n_parent; ++v) {
        if (!is_internal[v]) continue;
        if (is_upstream_connecting[v]) n_uc++;
        else if (is_downstream_connecting[v]) n_dc++;
        else n_io++;
    }
    /* Note: a vertex can be BOTH upstream-connecting AND
     * downstream-connecting (incoming from outside AND outgoing
     * to outside). We assign it to upstream_connecting in that
     * case; downstream-connecting still records its outgoing
     * external edges via downstream_out_edges in the metadata,
     * but the synthetic absorbing edge is added below as Type C
     * for any vertex with is_downstream_connecting true,
     * regardless of its category placement.
     *
     * Re-tally n_dc to count only those vertices placed in the
     * downstream-connecting category (i.e. not in
     * upstream-connecting). */
    n_dc = 0;
    for (size_t v = 0; v < n_parent; ++v) {
        if (!is_internal[v]) continue;
        if (is_upstream_connecting[v]) continue;
        if (is_downstream_connecting[v]) n_dc++;
    }
    /* And n_io is "internal vertices that are neither upstream-
     * nor downstream-connecting." */
    n_io = 0;
    for (size_t v = 0; v < n_parent; ++v) {
        if (!is_internal[v]) continue;
        if (is_upstream_connecting[v]) continue;
        if (is_downstream_connecting[v]) continue;
        n_io++;
    }

    /* Allocate per-category index arrays. */
    size_t *uc_indices = (n_uc > 0) ? (size_t *)malloc(n_uc * sizeof(size_t)) : NULL;
    size_t *io_indices = (n_io > 0) ? (size_t *)malloc(n_io * sizeof(size_t)) : NULL;
    size_t *dc_indices = (n_dc > 0) ? (size_t *)malloc(n_dc * sizeof(size_t)) : NULL;
    if ((n_uc > 0 && uc_indices == NULL) ||
        (n_io > 0 && io_indices == NULL) ||
        (n_dc > 0 && dc_indices == NULL)) {
        free(uc_indices); free(io_indices); free(dc_indices);
        goto fail_oom;
    }

    {
        size_t i_uc = 0, i_io = 0, i_dc = 0;
        for (size_t v = 0; v < n_parent; ++v) {
            if (!is_internal[v]) continue;
            if (is_upstream_connecting[v]) uc_indices[i_uc++] = v;
            else if (is_downstream_connecting[v]) dc_indices[i_dc++] = v;
            else io_indices[i_io++] = v;
        }
    }

    /* Sort each category in canonical order. */
    struct cat_sort_ctx sort_ctx = {parent, scc_position};
    sort_canonical(uc_indices, n_uc, &sort_ctx);
    sort_canonical(io_indices, n_io, &sort_ctx);
    sort_canonical(dc_indices, n_dc, &sort_ctx);

    /* ---- 4. Build synthetic graph. ---- */
    size_t n_synth = 1 /* source */ + n_uc + n_io + n_dc + 1 /* abs */;

    struct ptd_graph *synth = ptd_graph_create(parent->state_length);
    if (synth == NULL) {
        free(uc_indices); free(io_indices); free(dc_indices);
        goto fail_oom;
    }
    /* Pre-lock param_length so coefficient validation in
     * ptd_graph_add_edge agrees with the parent. */
    if (parent->param_length > 0) {
        ptd_graph_set_param_length(synth, parent->param_length);
    }

    /* parent_to_synth[parent_v] = synthetic ptd_vertex* for that
     * parent vertex, or NULL if not in synthetic graph. */
    struct ptd_vertex **parent_to_synth = (struct ptd_vertex **)
        calloc(n_parent, sizeof(struct ptd_vertex *));
    if (parent_to_synth == NULL) {
        ptd_graph_destroy(synth);
        free(uc_indices); free(io_indices); free(dc_indices);
        goto fail_oom;
    }

    /* synth_to_parent[synth_idx] = parent_v, or SIZE_MAX for
     * synthetic source/absorbing. */
    size_t *synth_to_parent = (size_t *)malloc(n_synth * sizeof(size_t));
    if (synth_to_parent == NULL) {
        free(parent_to_synth);
        ptd_graph_destroy(synth);
        free(uc_indices); free(io_indices); free(dc_indices);
        goto fail_oom;
    }

    /* Index 0: synthetic source (the auto-created starting vertex). */
    struct ptd_vertex *synth_source = synth->starting_vertex;
    synth_to_parent[0] = SIZE_MAX;

    /* Indices 1..1+n_uc: upstream-connecting. */
    {
        size_t synth_idx = 1;
        for (size_t k = 0; k < n_uc; ++k) {
            size_t pv = uc_indices[k];
            struct ptd_vertex *v = ptd_vertex_create_state(synth, parent->vertices[pv]->state);
            if (v == NULL) goto fail_construct;
            v->is_aux = parent->vertices[pv]->is_aux;
            parent_to_synth[pv] = v;
            synth_to_parent[synth_idx++] = pv;
        }
        /* Then internal-only. */
        for (size_t k = 0; k < n_io; ++k) {
            size_t pv = io_indices[k];
            struct ptd_vertex *v = ptd_vertex_create_state(synth, parent->vertices[pv]->state);
            if (v == NULL) goto fail_construct;
            v->is_aux = parent->vertices[pv]->is_aux;
            parent_to_synth[pv] = v;
            synth_to_parent[synth_idx++] = pv;
        }
        /* Then downstream-connecting. */
        for (size_t k = 0; k < n_dc; ++k) {
            size_t pv = dc_indices[k];
            struct ptd_vertex *v = ptd_vertex_create_state(synth, parent->vertices[pv]->state);
            if (v == NULL) goto fail_construct;
            v->is_aux = parent->vertices[pv]->is_aux;
            parent_to_synth[pv] = v;
            synth_to_parent[synth_idx++] = pv;
        }
        /* Finally the synthetic absorbing vertex (all-zero state). */
        struct ptd_vertex *synth_abs = ptd_vertex_create(synth);
        if (synth_abs == NULL) goto fail_construct;
        synth_to_parent[synth_idx++] = SIZE_MAX;
        (void)synth_abs;  /* used implicitly via vertices[n_synth - 1] */
        if (synth_idx != n_synth) {
            snprintf(ptd_err, 1024,
                     "ptd_scc_build_synthetic_graph: vertex count mismatch "
                     "(expected %zu, built %zu)", n_synth, synth_idx);
            goto fail_construct;
        }
    }

    /* Helper: synthetic absorbing vertex pointer. */
    struct ptd_vertex *synth_abs = synth->vertices[n_synth - 1];

    /* ---- 5. Add edges. ---- */
    /* Build a length-param_length placeholder coefficient vector
     * with first slot = 1.0, all others = 0.0. We use length
     * param_length (rather than length 1) so that the synthetic
     * graph passes update_weights's coefficient-length consistency
     * check, which requires all parameterised edges to have the
     * same coefficients_length. The placeholder produces a
     * concrete weight of θ[0] when update_weights is called —
     * for our smoke tests this is fine; in cache form (WP-3),
     * the placeholder coefficients are encoded as EXTERNAL pointers
     * so the actual binding to parent edge weights happens at
     * composition time. */
    size_t placeholder_len = (parent->param_length > 0) ? parent->param_length : 1;
    double *placeholder = (double *)calloc(placeholder_len, sizeof(double));
    if (placeholder == NULL) goto fail_construct;
    placeholder[0] = 1.0;

    /* Type A: synthetic source -> each upstream-connecting (placeholder). */
    {
        for (size_t k = 0; k < n_uc; ++k) {
            size_t pv = uc_indices[k];
            struct ptd_vertex *target = parent_to_synth[pv];
            if (add_edge_raw(synth_source, target, 1.0, placeholder, placeholder_len) != 0) {
                free(placeholder);
                goto fail_construct;
            }
        }
    }

    /* Type B: copy internal edges verbatim.
     * For every internal vertex, walk its parent edges; for each
     * edge whose target is also internal, add a corresponding
     * synthetic edge with identical coefficients_length and
     * coefficient values. */
    for (size_t v_p = 0; v_p < n_parent; ++v_p) {
        if (!is_internal[v_p]) continue;
        const struct ptd_vertex *parent_v = parent->vertices[v_p];
        struct ptd_vertex *synth_from = parent_to_synth[v_p];
        for (size_t e = 0; e < parent_v->edges_length; ++e) {
            const struct ptd_edge *edge = parent_v->edges[e];
            size_t target_p = edge->to->index;
            if (!is_internal[target_p]) continue;
            struct ptd_vertex *synth_to = parent_to_synth[target_p];
            if (synth_from == synth_to) continue;  /* skip self-loop guard */

            if (add_edge_raw(synth_from, synth_to, edge->weight,
                             edge->coefficients,
                             edge->coefficients_length) != 0) {
                goto fail_construct;
            }
        }
    }

    /* Type C: each downstream-connecting -> synthetic absorbing
     * (placeholder).
     *
     * Note: we add this edge for every internal vertex that has
     * is_downstream_connecting set, including those placed in the
     * upstream-connecting category (a vertex can be both). */
    {
        for (size_t v_p = 0; v_p < n_parent; ++v_p) {
            if (!is_internal[v_p]) continue;
            if (!is_downstream_connecting[v_p]) continue;
            struct ptd_vertex *synth_from = parent_to_synth[v_p];
            if (add_edge_raw(synth_from, synth_abs, 1.0, placeholder, placeholder_len) != 0) {
                free(placeholder);
                goto fail_construct;
            }
        }
    }
    free(placeholder);

    /* Mark synth as parameterised (any of the placeholder/internal
     * edges are array-syntax with coefficients_length >= 1). */
    synth->edge_mode = PTD_EDGE_MODE_PARAMETERIZED;
    synth->parameterized = true;

    /* ---- 6. Populate metadata. ----
     *
     * external_in_edges and external_out_edges are indexed by
     * synthetic-graph vertex index (length n_synth). Each entry
     * holds the per-pv list captured during the categorisation
     * pass, keyed by the synthetic-vertex's parent_indices[k].
     *
     * For synthetic source (k=0), synthetic absorbing
     * (k=n_synth-1), and pure internal-only vertices, both
     * arrays' entries are NULL/0 — already calloc'd to that.
     *
     * Per-synth-vertex indexing handles dual-category vertices
     * correctly: a vertex that is both upstream- and
     * downstream-connecting has non-empty entries in both
     * arrays. */
    struct ptd_scc_synthetic_metadata *meta = (struct ptd_scc_synthetic_metadata *)
        calloc(1, sizeof(*meta));
    if (meta == NULL) goto fail_construct;

    meta->scc_index = scc_index;
    meta->n_vertices = n_synth;
    meta->n_upstream_connecting = n_uc;
    meta->n_internal_only = n_io;
    meta->n_downstream_connecting = n_dc;
    meta->parent_indices = synth_to_parent;
    synth_to_parent = NULL;  /* ownership transferred */

    meta->external_in_edges = (struct ptd_scc_external_edge_ref **)
        calloc(n_synth, sizeof(*meta->external_in_edges));
    meta->external_in_edges_lengths = (size_t *)calloc(n_synth, sizeof(size_t));
    meta->external_out_edges = (struct ptd_scc_external_edge_ref **)
        calloc(n_synth, sizeof(*meta->external_out_edges));
    meta->external_out_edges_lengths = (size_t *)calloc(n_synth, sizeof(size_t));
    if (meta->external_in_edges == NULL ||
        meta->external_in_edges_lengths == NULL ||
        meta->external_out_edges == NULL ||
        meta->external_out_edges_lengths == NULL) {
        ptd_scc_synthetic_metadata_destroy(meta);
        goto fail_construct;
    }

    /* Walk the synthetic vertices (skipping source at 0 and
     * absorbing at n_synth-1) and pull the matching per-pv list
     * into the correct synthetic-index slot. */
    for (size_t v_synth = 1; v_synth < n_synth - 1; ++v_synth) {
        size_t pv = meta->parent_indices[v_synth];
        if (pv == SIZE_MAX) continue;  /* should not happen here */

        /* Transfer ownership of the in-edges list. */
        meta->external_in_edges[v_synth] = upstream_in_per_pv[pv];
        meta->external_in_edges_lengths[v_synth] = upstream_in_len_per_pv[pv];
        upstream_in_per_pv[pv] = NULL;
        upstream_in_len_per_pv[pv] = 0;

        /* Transfer ownership of the out-edges list. */
        meta->external_out_edges[v_synth] = downstream_out_per_pv[pv];
        meta->external_out_edges_lengths[v_synth] = downstream_out_len_per_pv[pv];
        downstream_out_per_pv[pv] = NULL;
        downstream_out_len_per_pv[pv] = 0;
    }

    /* ---- Cleanup transient allocations ---- */
    free(uc_indices);
    free(io_indices);
    free(dc_indices);
    /* Free any leftover per-pv lists (non-upstream-connecting,
     * non-downstream-connecting). */
    for (size_t v = 0; v < n_parent; ++v) {
        free(upstream_in_per_pv[v]);
        free(downstream_out_per_pv[v]);
    }
    free(upstream_in_per_pv);
    free(upstream_in_len_per_pv);
    free(upstream_in_cap_per_pv);
    free(downstream_out_per_pv);
    free(downstream_out_len_per_pv);
    free(downstream_out_cap_per_pv);

    free(parent_to_synth);
    free(is_internal);
    free(scc_position);
    free(is_upstream_connecting);
    free(is_downstream_connecting);

    *metadata_out = meta;
    return synth;

    /* ---- Error paths ---- */
fail_construct:
    /* synth_to_parent and synth own the partially built state. */
    if (synth_to_parent != NULL) free(synth_to_parent);
    if (parent_to_synth != NULL) free(parent_to_synth);
    if (synth != NULL) ptd_graph_destroy(synth);
    free(uc_indices);
    free(io_indices);
    free(dc_indices);
    /* Fall through to fail_oom for the per-pv lists. */
fail_oom:
    if (upstream_in_per_pv != NULL) {
        for (size_t v = 0; v < n_parent; ++v) {
            free(upstream_in_per_pv[v]);
        }
        free(upstream_in_per_pv);
    }
    free(upstream_in_len_per_pv);
    free(upstream_in_cap_per_pv);
    if (downstream_out_per_pv != NULL) {
        for (size_t v = 0; v < n_parent; ++v) {
            free(downstream_out_per_pv[v]);
        }
        free(downstream_out_per_pv);
    }
    free(downstream_out_len_per_pv);
    free(downstream_out_cap_per_pv);
    free(is_internal);
    free(scc_position);
    free(is_upstream_connecting);
    free(is_downstream_connecting);
    if (ptd_err[0] == '\0') {
        snprintf(ptd_err, 1024,
                 "ptd_scc_build_synthetic_graph: allocation or construction failure");
    }
    return NULL;
}

/* -----------------------------------------------------------------
 * WP-4: per-SCC PRC compute and disk cache.
 * ----------------------------------------------------------------- */

struct ptd_desc_reward_compute_parameterized *
ptd_scc_get_or_compute_prc(
        const struct ptd_scc_graph *scc_graph,
        size_t scc_index,
        struct ptd_graph **synth_out,
        struct ptd_scc_synthetic_metadata **metadata_out)
{
    if (synth_out != NULL) *synth_out = NULL;
    if (metadata_out != NULL) *metadata_out = NULL;

    if (scc_graph == NULL || synth_out == NULL || metadata_out == NULL) {
        snprintf(ptd_err, sizeof_ptd_err,
                 "ptd_scc_get_or_compute_prc: NULL argument");
        return NULL;
    }

    /* Step 1: build the synthetic graph + metadata. */
    struct ptd_graph *synth = ptd_scc_build_synthetic_graph(
            scc_graph, scc_index, metadata_out);
    if (synth == NULL) {
        /* ptd_err already set by ptd_scc_build_synthetic_graph. */
        return NULL;
    }

    /* Step 2 + 3: build the cache file path (also computes the
     * synthetic graph's content hash internally). */
    char cache_path[PATH_MAX];
    int have_path = 0;
    if (!ptd_scc_cache_disabled()) {
        if (ptd_scc_build_cache_path(synth, cache_path,
                                     sizeof(cache_path)) == 0) {
            have_path = 1;
        } else {
            /* Path build failed (e.g. HOME unset). Treat as
             * cache disabled and proceed to recompute without
             * saving. Clear ptd_err so the failure doesn't look
             * like a real error to the caller. */
            ptd_err[0] = '\0';
        }
    }

    /* Step 4: try to load. The synthetic graph's placeholder edge
     * weights serve as the external_table during load: they hold
     * the "neutral" placeholder values (1.0 for the first slot,
     * 0 elsewhere) that match what the file was saved with.
     *
     * For composition (WP-5), the caller will replace these
     * weights with parent-supplied values before replay; the
     * EXTERNAL pointers in the loaded PRC dereference whatever
     * the table holds at replay time. So loading with the
     * synthetic graph's own weight slots is correct here — it
     * gives a "neutral" PRC suitable for the per-SCC standalone
     * compute path. */
    if (have_path) {
        /* Build the external_table by reading the synthetic graph's
         * placeholder edge weights in the same order as
         * ptd_scc_collect_external_anchors produces. */
        size_t n_anchors = 0;
        double **anchors = ptd_scc_collect_external_anchors(
                synth, *metadata_out, &n_anchors);
        double *external_table = NULL;
        if (n_anchors > 0) {
            external_table = (double *)malloc(n_anchors * sizeof(double));
            if (external_table == NULL) {
                free(anchors);
                snprintf(ptd_err, sizeof_ptd_err,
                         "ptd_scc_get_or_compute_prc: oom for external_table");
                ptd_graph_destroy(synth);
                ptd_scc_synthetic_metadata_destroy(*metadata_out);
                *metadata_out = NULL;
                return NULL;
            }
            for (size_t i = 0; i < n_anchors; ++i) {
                external_table[i] = *anchors[i];
            }
        }

        struct ptd_desc_reward_compute_parameterized *loaded =
                ptd_load_parameterized_reward_compute_graph_ex(
                        cache_path, synth, external_table, n_anchors);

        free(anchors);
        ptd_err[0] = '\0';  /* swallow load-miss error message */

        if (loaded != NULL) {
            /* Cache hit. Transfer ownership of the synthetic graph
             * and metadata to the caller. external_table is leaked
             * here — the loaded PRC's EXTERNAL pointers reference
             * into it, so it must outlive the PRC. The caller
             * (composer / test harness) is responsible for the
             * PRC's lifetime; we attach external_table to a
             * known place: stash the pointer in metadata so it
             * gets freed when metadata is destroyed.
             *
             * However, our metadata struct doesn't currently have
             * a slot for it. For now, accept the leak in this
             * load path; WP-5's composer will manage the
             * external_table lifetime properly via its own
             * machinery (the per-SCC PRC isn't held long after
             * composition copies its commands into the parent
             * PRC). */
            *synth_out = synth;
            return loaded;
            /* external_table intentionally leaked; documented above. */
        }
        free(external_table);
        /* Fall through to rebuild path on miss. */
    }

    /* Step 5: cache miss (or disabled). Run the eliminator on the
     * synthetic graph. */
    struct ptd_desc_reward_compute_parameterized *prc =
            ptd_graph_ex_absorbation_time_comp_graph_parameterized(synth);
    if (prc == NULL) {
        ptd_graph_destroy(synth);
        ptd_scc_synthetic_metadata_destroy(*metadata_out);
        *metadata_out = NULL;
        if (ptd_err[0] == '\0') {
            snprintf(ptd_err, sizeof_ptd_err,
                     "ptd_scc_get_or_compute_prc: elimination returned NULL");
        }
        return NULL;
    }

    /* Step 6 + 7: collect anchors and best-effort save. Failures
     * here are non-fatal — we still return the in-memory PRC. */
    if (have_path) {
        size_t n_anchors = 0;
        double **anchors = ptd_scc_collect_external_anchors(
                synth, *metadata_out, &n_anchors);
        if (n_anchors > 0 && anchors != NULL) {
            (void)ptd_save_parameterized_reward_compute_graph_ex(
                    cache_path, prc, synth,
                    (const double *const *)anchors, n_anchors);
            ptd_err[0] = '\0';  /* swallow save errors */
        } else {
            /* No external anchors (e.g. an isolated SCC with no
             * upstream/downstream connections). Save as a v1
             * file via the regular save path. */
            (void)ptd_save_parameterized_reward_compute_graph(
                    cache_path, prc, synth);
            ptd_err[0] = '\0';
        }
        free(anchors);
    }

    *synth_out = synth;
    return prc;
}
