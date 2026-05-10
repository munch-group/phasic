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
    char parent[PATH_MAX];
    if (ptd_cache_root_dir(parent, sizeof(parent)) != 0) {
        return -1;  /* ptd_err set by helper */
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
    int n = snprintf(dir, sizeof(dir),
                     "%s/parameterized_reward_compute", parent);
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

    /* Per the per-channel layout:
     *   - Type A:    synth_source -> upstream-connecting (n_a edges)
     *   - Type C:    downstream-connecting -> per-channel-absorbing
     *   - Phantom:   per-channel-absorbing -> phantom_absorbing
     *
     * All three carry composer-supplied values at compose time.
     * We collect them in order: Type A, Type C (matching
     * meta->channels), Phantom (matching meta->channels). */
    struct ptd_vertex *src = synth->vertices[0];
    size_t n_uc = (meta != NULL) ? meta->n_upstream_connecting : 0;
    size_t n_channels = (meta != NULL) ? meta->n_channels : 0;
    size_t total = n_uc + 2 * n_channels;

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

    /* Type A edges (synth source -> upstream-connecting), ONLY
     * when parent_start is NOT in this SCC. When it is, the
     * synth source IS the parent start, with Type C outgoing
     * edges that are recorded in the channels[] iteration below.
     *
     * Detection: in the parent_start_in_scc case, all of synth
     * source's out-edges target per-channel absorbings (vertex
     * indices >= channel_first_idx). In the not-in-scc case, all
     * target upstream-connecting vertices (indices in [1, 1+n_uc)).
     *
     * We use n_uc (from meta) directly — it's 0 when there are
     * no upstream-connecting vertices. */
    if (n_uc > 0) {
        for (size_t e = 0; e < n_uc && e < src->edges_length; ++e) {
            anchors[idx++] = &src->edges[e]->weight;
        }
    }

    if (meta != NULL && meta->channels != NULL) {
        /* Type C edges. */
        for (size_t k = 0; k < meta->n_channels; ++k) {
            const struct ptd_scc_channel_info *ch = &meta->channels[k];
            struct ptd_vertex *d_j = synth->vertices[ch->d_j_synth_idx];
            anchors[idx++] = &d_j->edges[ch->type_c_edge_idx]->weight;
        }
        /* Phantom edges. */
        for (size_t k = 0; k < meta->n_channels; ++k) {
            const struct ptd_scc_channel_info *ch = &meta->channels[k];
            struct ptd_vertex *s_abs = synth->vertices[ch->s_abs_synth_idx];
            anchors[idx++] = &s_abs->edges[ch->phantom_edge_idx]->weight;
        }
    }

    *n_anchors_out = idx;
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

    free(metadata->channels);

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

    /* Count external out-channels: one per parent-graph external
     * edge from any internal vertex (whether in dc_indices or
     * dual-category in uc_indices). One per-channel absorbing
     * vertex per channel. */
    size_t n_channels = 0;
    for (size_t v = 0; v < n_parent; ++v) {
        if (!is_internal[v]) continue;
        n_channels += downstream_out_len_per_pv[v];
    }

    /* ---- 4. Build synthetic graph. ----
     *
     * Layout:
     *   index 0:                              synthetic source
     *   indices [1, 1+n_uc):                  upstream-connecting
     *   indices [1+n_uc, 1+n_uc+n_io):        internal-only
     *   indices [1+n_uc+n_io,                 downstream-connecting
     *            1+n_uc+n_io+n_dc):
     *   indices [1+n_uc+n_io+n_dc,            per-channel absorbing
     *            1+n_uc+n_io+n_dc+n_channels):
     *   index n_synth - 1:                    phantom absorbing
     */
    /* If the parent's start is in this SCC, the synth's auto-created
     * starting_vertex (already at index 0) doubles as the synth-copy
     * of the parent's start — so we don't create a separate vertex
     * for it. Total vertex count drops by 1 in that case. */
    bool parent_start_in_scc_pre = is_internal[parent->starting_vertex->index];
    size_t parent_start_skip = parent_start_in_scc_pre ? 1 : 0;

    size_t n_synth = 1 /* source */ + n_uc + n_io + n_dc - parent_start_skip
                   + n_channels + 1 /* phantom */;
    size_t channel_first_idx = 1 + n_uc + n_io + n_dc - parent_start_skip;
    size_t phantom_idx = n_synth - 1;

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

    /* Detect whether the parent's starting vertex is internal to
     * this SCC. If so, the synthetic graph's auto-created
     * starting vertex (at index 0) acts as the synth-copy of the
     * parent's starting vertex. The parent's start has no IPV in
     * the parent (it IS the IPV), so no Type A edge exists for
     * it; its outgoing edges become Type C edges directly.
     *
     * Treating it as the synth's starting_vertex makes the
     * eliminator give it the special "1/rate=0" treatment that
     * matches the parent's monolithic computation. */
    size_t parent_start_idx = parent->starting_vertex->index;
    bool parent_start_in_scc = parent_start_in_scc_pre;

    /* Index 0: synthetic source. If parent's start is in this
     * SCC, this index doubles as the synth-copy of parent's start.
     * Otherwise it's an unused entry-point (no out-edges; gets
     * treated as absorbing by the eliminator). */
    struct ptd_vertex *synth_source = synth->starting_vertex;
    synth_to_parent[0] = parent_start_in_scc ? parent_start_idx : SIZE_MAX;
    if (parent_start_in_scc) {
        parent_to_synth[parent_start_idx] = synth_source;
        /* synth_source already has all-zero state (matches parent
         * start). Copy is_aux from the parent. */
        synth_source->is_aux = parent->vertices[parent_start_idx]->is_aux;
    }

    /* Indices 1..1+n_uc: upstream-connecting. */
    {
        size_t synth_idx = 1;
        for (size_t k = 0; k < n_uc; ++k) {
            size_t pv = uc_indices[k];
            if (parent_start_in_scc && pv == parent_start_idx) {
                /* Skip — already mapped to synth_source. */
                continue;
            }
            struct ptd_vertex *v = ptd_vertex_create_state(synth, parent->vertices[pv]->state);
            if (v == NULL) goto fail_construct;
            v->is_aux = parent->vertices[pv]->is_aux;
            parent_to_synth[pv] = v;
            synth_to_parent[synth_idx++] = pv;
        }
        /* Then internal-only. */
        for (size_t k = 0; k < n_io; ++k) {
            size_t pv = io_indices[k];
            if (parent_start_in_scc && pv == parent_start_idx) continue;
            struct ptd_vertex *v = ptd_vertex_create_state(synth, parent->vertices[pv]->state);
            if (v == NULL) goto fail_construct;
            v->is_aux = parent->vertices[pv]->is_aux;
            parent_to_synth[pv] = v;
            synth_to_parent[synth_idx++] = pv;
        }
        /* Then downstream-connecting. */
        for (size_t k = 0; k < n_dc; ++k) {
            size_t pv = dc_indices[k];
            if (parent_start_in_scc && pv == parent_start_idx) continue;
            struct ptd_vertex *v = ptd_vertex_create_state(synth, parent->vertices[pv]->state);
            if (v == NULL) goto fail_construct;
            v->is_aux = parent->vertices[pv]->is_aux;
            parent_to_synth[pv] = v;
            synth_to_parent[synth_idx++] = pv;
        }
        /* Then per-channel absorbing vertices (one per external
         * out-channel). All have all-zero state. */
        for (size_t k = 0; k < n_channels; ++k) {
            struct ptd_vertex *v = ptd_vertex_create(synth);
            if (v == NULL) goto fail_construct;
            synth_to_parent[synth_idx++] = SIZE_MAX;
        }
        /* Finally the phantom absorbing vertex (the only true
         * absorbing in the graph). All-zero state. */
        struct ptd_vertex *phantom = ptd_vertex_create(synth);
        if (phantom == NULL) goto fail_construct;
        synth_to_parent[synth_idx++] = SIZE_MAX;
        (void)phantom;  /* used implicitly via vertices[phantom_idx] */
        if (synth_idx != n_synth) {
            snprintf(ptd_err, 1024,
                     "ptd_scc_build_synthetic_graph: vertex count mismatch "
                     "(expected %zu, built %zu)", n_synth, synth_idx);
            goto fail_construct;
        }
    }

    /* Helper: phantom absorbing vertex pointer. */
    struct ptd_vertex *phantom_vertex = synth->vertices[phantom_idx];

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

    /* Type A: synthetic source -> each upstream-connecting
     * (placeholder). Skip the parent-start vertex if it's in
     * this SCC, since synth_source IS the parent-start surrogate
     * and giving it a self-edge would be a self-loop. */
    {
        for (size_t k = 0; k < n_uc; ++k) {
            size_t pv = uc_indices[k];
            if (parent_start_in_scc && pv == parent_start_idx) continue;
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

    /* Type C: per-channel edges. Each external out-edge from a
     * downstream-connecting vertex gets its own synthetic
     * absorbing vertex (already allocated above) and a
     * placeholder Type C edge. The composer overwrites the
     * weight slot at compose time with the parent's external
     * edge weight.
     *
     * Plus phantom edges: each per-channel absorbing has one
     * outgoing edge to the phantom, with placeholder weight.
     * The composer overwrites this weight slot with
     * 1/result[w_in_parent] so the per-channel absorbing's
     * elimination result equals the downstream parent value.
     *
     * We also populate the channels[] metadata array as we go,
     * recording per-channel vertex and edge indices so the
     * composer can find them. */
    struct ptd_scc_channel_info *channels_info = NULL;
    if (n_channels > 0) {
        channels_info = (struct ptd_scc_channel_info *)calloc(
                n_channels, sizeof(*channels_info));
        if (channels_info == NULL) {
            free(placeholder);
            goto fail_construct;
        }
    }

    /* Walk synthetic vertices in canonical order so the channel
     * iteration order is deterministic. For each downstream-
     * connecting vertex (whether placed in uc or dc category, or
     * the parent-start vertex at synth index 0 when parent_start_in_scc),
     * walk its parent's external_out_edges list in order. */
    size_t channel_cursor = 0;
    /* Start from index 0 when parent_start is in this SCC (because
     * the parent_start sits at synth index 0 in that case);
     * otherwise start from 1. */
    size_t v_walk_start = parent_start_in_scc ? 0 : 1;
    for (size_t v_synth_idx = v_walk_start; v_synth_idx < channel_first_idx; ++v_synth_idx) {
        size_t pv = synth_to_parent[v_synth_idx]; /* parent vertex idx */
        if (pv == SIZE_MAX) continue;
        if (!is_downstream_connecting[pv]) continue;

        struct ptd_vertex *synth_from = parent_to_synth[pv];
        size_t n_out_edges = downstream_out_len_per_pv[pv];
        for (size_t e = 0; e < n_out_edges; ++e) {
            struct ptd_scc_external_edge_ref *ref = &downstream_out_per_pv[pv][e];

            /* Per-channel absorbing vertex index in synthetic graph. */
            size_t s_abs_idx = channel_first_idx + channel_cursor;
            if (s_abs_idx >= phantom_idx) {
                snprintf(ptd_err, 1024,
                         "ptd_scc_build_synthetic_graph: channel index "
                         "%zu out of range", s_abs_idx);
                free(channels_info);
                free(placeholder);
                goto fail_construct;
            }
            struct ptd_vertex *s_abs = synth->vertices[s_abs_idx];

            /* Type C edge: d_j_synth -> s_abs_for_channel.
             * Record the edge index BEFORE adding (it's the
             * current edges_length, since add_edge_raw appends). */
            size_t type_c_edge_idx = synth_from->edges_length;
            if (add_edge_raw(synth_from, s_abs, 1.0,
                             placeholder, placeholder_len) != 0) {
                free(channels_info);
                free(placeholder);
                goto fail_construct;
            }

            /* Phantom edge: s_abs_for_channel -> phantom.
             * Edge index is 0 since this is its only outgoing. */
            if (add_edge_raw(s_abs, phantom_vertex, 1.0,
                             placeholder, placeholder_len) != 0) {
                free(channels_info);
                free(placeholder);
                goto fail_construct;
            }

            /* Populate channel info. */
            channels_info[channel_cursor].parent_vertex_idx = ref->parent_vertex_idx;
            channels_info[channel_cursor].parent_edge_idx = ref->parent_edge_idx;
            channels_info[channel_cursor].d_j_synth_idx = v_synth_idx;
            channels_info[channel_cursor].s_abs_synth_idx = s_abs_idx;
            channels_info[channel_cursor].type_c_edge_idx = type_c_edge_idx;
            channels_info[channel_cursor].phantom_edge_idx = 0;
            channel_cursor++;
        }
    }

    free(placeholder);

    if (channel_cursor != n_channels) {
        snprintf(ptd_err, 1024,
                 "ptd_scc_build_synthetic_graph: channel count mismatch "
                 "(expected %zu, wired %zu)", n_channels, channel_cursor);
        free(channels_info);
        goto fail_construct;
    }

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
    meta->n_channels = n_channels;
    meta->channels = channels_info;
    channels_info = NULL;  /* ownership transferred to meta */
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

    /* Walk the synthetic vertices and pull the matching per-pv
     * list into the correct synthetic-index slot. We start from
     * index 0 to also cover the parent-start case (where index 0
     * holds the parent's start as a real internal vertex). */
    for (size_t v_synth = 0; v_synth < n_synth; ++v_synth) {
        size_t pv = meta->parent_indices[v_synth];
        if (pv == SIZE_MAX) continue;

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

/* SLURM-WP-4: synth-only cache-or-compute. Operates on an
 * already-built synthetic graph (from ptd_scc_build_synthetic_graph
 * or from JSON deserialisation in a worker). Builds the cache
 * path, tries to load, and on miss runs the eliminator and saves.
 *
 * Used both by ptd_scc_get_or_compute_prc (which wraps the
 * synth-build step) and by the worker CLI in distributed mode.
 *
 * Returns the PRC on success (caller owns and must destroy via
 * ptd_parameterized_reward_compute_graph_destroy unless installed
 * on a graph). NULL on failure (sets ptd_err). The synth itself
 * is NOT freed by this function. */
struct ptd_desc_reward_compute_parameterized *
ptd_synth_get_or_compute_prc(struct ptd_graph *synth)
{
    if (synth == NULL) {
        snprintf(ptd_err, sizeof_ptd_err,
                 "ptd_synth_get_or_compute_prc: NULL synth");
        return NULL;
    }

    /* Build the cache file path (also computes the synthetic
     * graph's content hash internally). */
    char cache_path[PATH_MAX];
    int have_path = 0;
    if (!ptd_scc_cache_disabled()) {
        if (ptd_scc_build_cache_path(synth, cache_path,
                                     sizeof(cache_path)) == 0) {
            have_path = 1;
        } else {
            ptd_err[0] = '\0';
        }
    }

    /* Try to load. */
    if (have_path) {
        struct ptd_desc_reward_compute_parameterized *loaded =
                ptd_load_parameterized_reward_compute_graph(
                        cache_path, synth);
        ptd_err[0] = '\0';  /* swallow load-miss error message */

        if (loaded != NULL) {
            ptd_scc_compose_stats_record_hit();  /* WP-8 */
            return loaded;
        }
        ptd_scc_compose_stats_record_miss();  /* WP-8 */
    }

    /* Cache miss (or disabled). Run the eliminator. */
    struct ptd_desc_reward_compute_parameterized *prc =
            ptd_graph_ex_absorbation_time_comp_graph_parameterized(synth);
    if (prc == NULL) {
        if (ptd_err[0] == '\0') {
            snprintf(ptd_err, sizeof_ptd_err,
                     "ptd_synth_get_or_compute_prc: elimination returned NULL");
        }
        return NULL;
    }

    /* Best-effort save. */
    if (have_path) {
        (void)ptd_save_parameterized_reward_compute_graph(
                cache_path, prc, synth);
        ptd_err[0] = '\0';
    }
    return prc;
}

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

    /* Build the synthetic graph + metadata. */
    struct ptd_graph *synth = ptd_scc_build_synthetic_graph(
            scc_graph, scc_index, metadata_out);
    if (synth == NULL) {
        return NULL;
    }

    /* Delegate cache-or-compute to the synth-only helper. */
    struct ptd_desc_reward_compute_parameterized *prc =
            ptd_synth_get_or_compute_prc(synth);
    if (prc == NULL) {
        ptd_graph_destroy(synth);
        ptd_scc_synthetic_metadata_destroy(*metadata_out);
        *metadata_out = NULL;
        return NULL;
    }

    *synth_out = synth;
    return prc;
}
