/* Thread-local re-entrancy guard definition. The declaration
 * (with PTD_SCC_TLS macro) lives in api/c/phasic.h so other
 * translation units see it as TLS too. */
#include "../../api/c/phasic.h"
PTD_SCC_TLS int ptd_scc_compose_in_progress = 0;

/**
 * @file scc_compose.c
 * @brief WP-5: SCC composition — assemble parent-level result vector
 *        from per-SCC PRCs.
 *
 * Walks SCCs in reverse-topological order (sink-first). For each
 * SCC: builds (or retrieves cached) the synthetic graph + PRC,
 * sets per-channel placeholder edge weights to parent values
 * and downstream-result-derived values, runs the elimination,
 * copies per-internal-vertex results into the parent-wide
 * result vector.
 *
 * The math (verified empirically in /tmp/wp5_optionB_math.py and
 * documented in wp5-investigation.md):
 *   - Type C edge d_j -> s_abs_for_(d_j,w) gets weight =
 *     parent's external edge weight at current theta.
 *   - Phantom edge s_abs_for_(d_j,w) -> phantom gets weight =
 *     1/parent_result[w], so by the phase-type identity
 *     result[v] = 1/rate(v) + ..., we get
 *     result[s_abs_for_(d_j,w)] = parent_result[w].
 *   - The per-SCC elimination then computes result[v] for every
 *     internal vertex of the SCC, accounting for cross-SCC
 *     dataflow via the injected per-channel values.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <time.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "../../api/c/phasic.h"

#define sizeof_ptd_err 4096

/* WP-8: process-wide telemetry counters. Atomic when OpenMP
 * is available so parallel composer threads can bump them
 * safely. The counters reflect activity since process start
 * (or last ptd_scc_compose_stats_reset call). */
static uint64_t g_cache_hits = 0;
static uint64_t g_cache_misses = 0;
static uint64_t g_compose_calls = 0;
static uint64_t g_total_compose_ns = 0;
static uint64_t g_cache_bypassed = 0;

static void atomic_add_u64(uint64_t *dst, uint64_t delta)
{
#ifdef PHASIC_HAVE_OPENMP
    #pragma omp atomic
    *dst += delta;
#else
    *dst += delta;
#endif
}

void ptd_scc_compose_stats_record_hit(void) { atomic_add_u64(&g_cache_hits, 1); }
void ptd_scc_compose_stats_record_miss(void) { atomic_add_u64(&g_cache_misses, 1); }
void ptd_scc_compose_stats_record_bypass(void) { atomic_add_u64(&g_cache_bypassed, 1); }

void ptd_scc_compose_stats_get(struct ptd_scc_compose_stats *out)
{
    if (out == NULL) return;
    /* Reads of aligned 8-byte values are atomic on the platforms
     * we support; we don't bother with explicit barriers. */
    out->cache_hits = g_cache_hits;
    out->cache_misses = g_cache_misses;
    out->compose_calls = g_compose_calls;
    out->total_compose_ns = g_total_compose_ns;
    out->cache_bypassed = g_cache_bypassed;
}

void ptd_scc_compose_stats_reset(void)
{
    g_cache_hits = 0;
    g_cache_misses = 0;
    g_compose_calls = 0;
    g_total_compose_ns = 0;
    g_cache_bypassed = 0;
}

static uint64_t monotonic_ns(void)
{
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&counter);
    return (uint64_t)((counter.QuadPart * 1000000000ull) / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

/* Read PHASIC_MAX_PARALLEL_SCCS: cap on simultaneous per-SCC
 * computes within a level. Defaults to 0 (no cap — use OpenMP's
 * default thread count). When set, the cap is enforced as the
 * per-level limit regardless of OMP_NUM_THREADS, so users can
 * keep OpenMP threads available for inner work (e.g. each SCC's
 * own eliminator) while limiting the SCC-level fan-out for
 * memory-pressure reasons. */
static int ptd_scc_max_parallel(void)
{
    const char *v = getenv("PHASIC_MAX_PARALLEL_SCCS");
    if (v == NULL) return 0;
    char *end = NULL;
    long parsed = strtol(v, &end, 10);
    if (end == v || parsed < 1) return 0;
    if (parsed > 1024) return 1024;  /* sanity cap */
    return (int)parsed;
}

/* Per-SCC processor: build synth + PRC, override per-channel
 * edge weights, run elimination, copy results into parent_result.
 *
 * Returns 0 on success, -1 on failure. Sets err_msg (caller-
 * provided buffer) on failure. parent_result is mutated; the
 * indices it writes are disjoint from those written by other
 * SCCs at the same level (so OpenMP-safe).
 *
 * Returns separate from setting ptd_err so the caller (parallel
 * loop) can route errors to a single thread without conflicting
 * thread-local storage. */
static int ptd_compose_scc_one(
        struct ptd_graph *parent,
        const struct ptd_scc_graph *scc_graph,
        size_t i,
        const double *theta,
        size_t theta_len,
        double *parent_result,
        char *err_msg,
        size_t err_msg_size)
{
    /* The re-entrancy guard ptd_scc_compose_in_progress is
     * __thread (PTD_SCC_TLS). Under the parallel-for above,
     * each OpenMP worker thread has its OWN TLS slot
     * initialised to 0 — so the master's bump in
     * ptd_compose_scc_prcs_inner does not propagate. Without
     * this local bump, the worker's inner call to
     * ptd_expected_waiting_time on the synth graph would
     * recurse back into the hierarchical path, multiplying
     * cache hit/miss counters and breaking compose telemetry.
     *
     * Bump on entry; decrement at the single goto-out exit. */
    ptd_scc_compose_in_progress++;

    int rc = 0;
    struct ptd_graph *synth = NULL;
    struct ptd_scc_synthetic_metadata *meta = NULL;
    struct ptd_desc_reward_compute_parameterized *prc =
            ptd_scc_get_or_compute_prc(scc_graph, i, &synth, &meta);
    if (prc == NULL || synth == NULL || meta == NULL) {
        snprintf(err_msg, err_msg_size,
                 "ptd_compose_scc_one: per-SCC compute failed for SCC %zu", i);
        rc = -1;
        goto out;
    }

    /* Step 1: update synth's edge weights from theta. */
    {
        double *theta_copy = (double *)malloc(theta_len * sizeof(double));
        if (theta_copy == NULL) {
            snprintf(err_msg, err_msg_size,
                     "ptd_compose_scc_one: oom for theta_copy (SCC %zu)", i);
            rc = -1;
            goto out;
        }
        memcpy(theta_copy, theta, theta_len * sizeof(double));
        ptd_graph_update_weights(synth, theta_copy, theta_len, false);
        free(theta_copy);
    }

    /* Step 2: override per-channel Type C and phantom edge weights. */
    for (size_t k = 0; k < meta->n_channels; ++k) {
        const struct ptd_scc_channel_info *ch = &meta->channels[k];

        struct ptd_vertex *parent_dj = parent->vertices[ch->parent_vertex_idx];
        if (ch->parent_edge_idx >= parent_dj->edges_length) {
            snprintf(err_msg, err_msg_size,
                     "ptd_compose_scc_one: parent edge idx %zu out of range "
                     "for vertex %zu (SCC %zu)",
                     ch->parent_edge_idx, ch->parent_vertex_idx, i);
            rc = -1;
            goto out;
        }
        double parent_external_weight = parent_dj->edges[ch->parent_edge_idx]->weight;
        size_t parent_target_idx = parent_dj->edges[ch->parent_edge_idx]->to->index;

        struct ptd_vertex *synth_dj = synth->vertices[ch->d_j_synth_idx];
        ptd_edge_update_weight(synth_dj->edges[ch->type_c_edge_idx],
                               parent_external_weight);

        struct ptd_vertex *synth_sabs = synth->vertices[ch->s_abs_synth_idx];
        double phantom_weight;
        if (parent_result[parent_target_idx] > 0.0) {
            phantom_weight = 1.0 / parent_result[parent_target_idx];
        } else if (parent_result[parent_target_idx] < 0.0) {
            snprintf(err_msg, err_msg_size,
                     "ptd_compose_scc_one: negative parent_result[%zu]=%g (SCC %zu)",
                     parent_target_idx, parent_result[parent_target_idx], i);
            rc = -1;
            goto out;
        } else {
            phantom_weight = 1e300;
        }
        ptd_edge_update_weight(synth_sabs->edges[ch->phantom_edge_idx],
                               phantom_weight);
    }

    /* Step 3: run elimination. */
    if (synth->reward_compute_graph != NULL) {
        free(synth->reward_compute_graph->commands);
        free(synth->reward_compute_graph);
        synth->reward_compute_graph = NULL;
    }
    if (synth->parameterized_reward_compute_graph != NULL) {
        ptd_parameterized_reward_compute_graph_destroy(
                synth->parameterized_reward_compute_graph);
    }
    synth->parameterized_reward_compute_graph = prc;
    prc = NULL;  /* transferred to synth */

    double *synth_result = ptd_expected_waiting_time(synth, NULL);
    if (synth_result == NULL) {
        snprintf(err_msg, err_msg_size,
                 "ptd_compose_scc_one: per-SCC elimination failed for SCC %zu", i);
        rc = -1;
        goto out;
    }

    /* Step 4: copy results to parent_result. */
    for (size_t v_synth = 0; v_synth < meta->n_vertices; ++v_synth) {
        size_t parent_idx = meta->parent_indices[v_synth];
        if (parent_idx == SIZE_MAX) continue;
        parent_result[parent_idx] = synth_result[v_synth];
    }
    free(synth_result);

out:
    if (synth) ptd_graph_destroy(synth);
    if (meta) ptd_scc_synthetic_metadata_destroy(meta);
    if (prc) ptd_parameterized_reward_compute_graph_destroy(prc);
    ptd_scc_compose_in_progress--;
    return rc;
}

/* Inner body. Returns parent_result on success, NULL on
 * failure. The outer wrapper handles call/timing telemetry
 * so all early-return paths automatically contribute to
 * total_compose_ns. */
static double *ptd_compose_scc_prcs_inner(
        struct ptd_graph *parent,
        const struct ptd_scc_graph *scc_graph,
        const double *theta,
        size_t theta_len)
{
    /* Set re-entrancy guard so inner ptd_expected_waiting_time
     * calls on synthetic graphs don't recurse into composition. */
    ptd_scc_compose_in_progress++;
    if (theta == NULL || theta_len != parent->param_length) {
        snprintf(ptd_err, sizeof_ptd_err,
                 "ptd_compose_scc_prcs: theta length %zu != param_length %zu",
                 theta_len, parent->param_length);
        ptd_scc_compose_in_progress--;
        return NULL;
    }

    /* Allocate parent-wide result vector. Initialize to 0
     * (absorbing vertices stay 0 by convention). */
    double *parent_result = (double *)calloc(
            parent->vertices_length, sizeof(double));
    if (parent_result == NULL) {
        snprintf(ptd_err, sizeof_ptd_err,
                 "ptd_compose_scc_prcs: oom for parent_result");
        ptd_scc_compose_in_progress--;
        return NULL;
    }

    /* Update parent's edge weights from theta so we can read them
     * for Type C bindings. The caller may have already done this;
     * doing it again is idempotent and safe. */
    {
        /* update_weights mutates a copy of theta internally. */
        double *theta_copy = (double *)malloc(theta_len * sizeof(double));
        if (theta_copy == NULL) {
            snprintf(ptd_err, sizeof_ptd_err,
                     "ptd_compose_scc_prcs: oom for theta_copy");
            free(parent_result);
            ptd_scc_compose_in_progress--;
            return NULL;
        }
        memcpy(theta_copy, theta, theta_len * sizeof(double));
        ptd_graph_update_weights(parent, theta_copy, theta_len, false);
        free(theta_copy);
        if (ptd_err[0] != '\0') {
            free(parent_result);
            ptd_scc_compose_in_progress--;
            return NULL;
        }
    }

    /* Build a sink-first traversal of the condensation. The
     * stored order (after ptd_isolate_starting_vertex_scc) puts
     * the starting SCC at position 0 but is NOT a valid
     * topological order otherwise (verified empirically: stored
     * positions can have an SCC at position k with edges to
     * SCCs at positions < k). We need to iterate so each SCC's
     * downstream are processed before it: i.e. reverse-
     * topological / sink-first.
     *
     * Run Kahn's algorithm on the condensation to get
     * forward-topological order, then iterate it in reverse. */
    size_t n_sccs = scc_graph->vertices_length;
    /* in_degree[k] = number of incoming SCC edges to SCC k. */
    size_t *in_degree = (size_t *)calloc(n_sccs, sizeof(size_t));
    /* topo_order[k] gives the k-th SCC in forward-topological order. */
    size_t *topo_order = (size_t *)malloc(n_sccs * sizeof(size_t));
    /* queue of SCCs ready to emit (in-degree 0). */
    size_t *queue = (size_t *)malloc(n_sccs * sizeof(size_t));
    if (in_degree == NULL || topo_order == NULL || queue == NULL) {
        free(in_degree); free(topo_order); free(queue);
        free(parent_result);
        snprintf(ptd_err, sizeof_ptd_err,
                 "ptd_compose_scc_prcs: oom for topo ordering");
        ptd_scc_compose_in_progress--;
        return NULL;
    }
    /* Count incoming edges. */
    for (size_t k = 0; k < n_sccs; ++k) {
        struct ptd_scc_vertex *scc = scc_graph->vertices[k];
        for (size_t e = 0; e < scc->edges_length; ++e) {
            in_degree[scc->edges[e]->to->index]++;
        }
    }
    /* Kahn's algorithm. */
    size_t q_head = 0, q_tail = 0;
    for (size_t k = 0; k < n_sccs; ++k) {
        if (in_degree[k] == 0) queue[q_tail++] = k;
    }
    size_t topo_idx = 0;
    while (q_head < q_tail) {
        size_t k = queue[q_head++];
        topo_order[topo_idx++] = k;
        struct ptd_scc_vertex *scc = scc_graph->vertices[k];
        for (size_t e = 0; e < scc->edges_length; ++e) {
            size_t target = scc->edges[e]->to->index;
            if (--in_degree[target] == 0) {
                queue[q_tail++] = target;
            }
        }
    }
    free(in_degree);
    free(queue);
    if (topo_idx != n_sccs) {
        free(topo_order);
        free(parent_result);
        snprintf(ptd_err, sizeof_ptd_err,
                 "ptd_compose_scc_prcs: condensation has cycles? "
                 "topo_idx=%zu n_sccs=%zu", topo_idx, n_sccs);
        ptd_scc_compose_in_progress--;
        return NULL;
    }

    /* WP-6: compute level numbers (longest path to a sink) for
     * each SCC. SCCs at the same level are independent — they
     * have no inter-SCC edges between them in the condensation —
     * and can be processed in parallel.
     *
     * level[k] = 0 if SCC k has no outgoing edges (it's a sink),
     *          = 1 + max(level[target]) over outgoing edges.
     *
     * We compute by walking topo_order in REVERSE (sinks first)
     * so that level[target] is already known when we reach k. */
    size_t *level_of = (size_t *)calloc(n_sccs, sizeof(size_t));
    if (level_of == NULL) {
        free(topo_order);
        free(parent_result);
        snprintf(ptd_err, sizeof_ptd_err,
                 "ptd_compose_scc_prcs: oom for level array");
        ptd_scc_compose_in_progress--;
        return NULL;
    }
    size_t max_level = 0;
    for (size_t ii = 0; ii < n_sccs; ++ii) {
        size_t k = topo_order[n_sccs - 1 - ii];  /* sink-first */
        struct ptd_scc_vertex *scc = scc_graph->vertices[k];
        size_t lvl = 0;
        for (size_t e = 0; e < scc->edges_length; ++e) {
            size_t t = scc->edges[e]->to->index;
            if (level_of[t] + 1 > lvl) {
                lvl = level_of[t] + 1;
            }
        }
        level_of[k] = lvl;
        if (lvl > max_level) max_level = lvl;
    }

    /* Group SCC indices by level. */
    size_t *level_counts = (size_t *)calloc(max_level + 1, sizeof(size_t));
    if (level_counts == NULL) {
        free(level_of); free(topo_order); free(parent_result);
        snprintf(ptd_err, sizeof_ptd_err,
                 "ptd_compose_scc_prcs: oom for level counts");
        ptd_scc_compose_in_progress--;
        return NULL;
    }
    for (size_t k = 0; k < n_sccs; ++k) {
        level_counts[level_of[k]]++;
    }
    size_t *level_offsets = (size_t *)malloc((max_level + 2) * sizeof(size_t));
    size_t *level_indices = (size_t *)malloc(n_sccs * sizeof(size_t));
    if (level_offsets == NULL || level_indices == NULL) {
        free(level_offsets); free(level_indices);
        free(level_counts); free(level_of); free(topo_order); free(parent_result);
        snprintf(ptd_err, sizeof_ptd_err,
                 "ptd_compose_scc_prcs: oom for level layout");
        ptd_scc_compose_in_progress--;
        return NULL;
    }
    level_offsets[0] = 0;
    for (size_t l = 0; l <= max_level; ++l) {
        level_offsets[l + 1] = level_offsets[l] + level_counts[l];
    }
    /* Place each SCC into its level slot. */
    {
        size_t *cursor = (size_t *)calloc(max_level + 1, sizeof(size_t));
        if (cursor == NULL) {
            free(level_offsets); free(level_indices); free(level_counts);
            free(level_of); free(topo_order); free(parent_result);
            snprintf(ptd_err, sizeof_ptd_err,
                     "ptd_compose_scc_prcs: oom for cursor");
            ptd_scc_compose_in_progress--;
            return NULL;
        }
        for (size_t k = 0; k < n_sccs; ++k) {
            size_t l = level_of[k];
            level_indices[level_offsets[l] + cursor[l]++] = k;
        }
        free(cursor);
    }
    free(level_counts);

    /* Process levels in order 0, 1, ..., max_level (sink-first).
     * Within each level, SCCs are independent and can be
     * processed in parallel. */
    for (size_t l = 0; l <= max_level; ++l) {
        size_t l_start = level_offsets[l];
        size_t l_end = level_offsets[l + 1];
        size_t l_size = l_end - l_start;

        /* Per-thread error storage. We let parallel iterations
         * each write into their own slot of err_msgs and aggregate
         * after the loop. Errors stop the level early via
         * compose_error flag. */
        char *err_msgs = (char *)calloc(l_size * sizeof_ptd_err, 1);
        int *iter_status = (int *)calloc(l_size, sizeof(int));
        if (err_msgs == NULL || iter_status == NULL) {
            free(err_msgs); free(iter_status);
            free(level_offsets); free(level_indices); free(level_of);
            free(topo_order); free(parent_result);
            snprintf(ptd_err, sizeof_ptd_err,
                     "ptd_compose_scc_prcs: oom for per-level err buffers");
            ptd_scc_compose_in_progress--;
            return NULL;
        }

#ifdef PHASIC_HAVE_OPENMP
        /* Determine the per-level thread count. PHASIC_MAX_PARALLEL_SCCS
         * caps fan-out within this level; if unset, OpenMP picks
         * its default (usually OMP_NUM_THREADS). We never exceed
         * the level's actual SCC count.
         *
         * The cap applies only to the SCC-level parallel-for, not
         * to any nested OpenMP regions inside individual SCC
         * eliminations — those still see the full OMP_NUM_THREADS.
         * This lets users say "limit SCC-level fan-out to 4 but
         * keep all 32 threads available for whatever the inner
         * eliminator does." */
        int max_par = ptd_scc_max_parallel();
        int omp_threads;
        if (max_par > 0 && (size_t)max_par < l_size) {
            omp_threads = max_par;
        } else {
            omp_threads = (int)l_size;
            if (omp_threads < 1) omp_threads = 1;
        }
        #pragma omp parallel for schedule(dynamic) if(l_size > 1) \
                num_threads(omp_threads)
#endif
        /* MSVC's OpenMP only supports OpenMP 2.0, which requires a
         * signed integer loop variable. Use ptrdiff_t (signed) here
         * and cast back inside the body. */
        for (ptrdiff_t li_s = 0; li_s < (ptrdiff_t)l_size; ++li_s) {
            size_t li = (size_t)li_s;
            size_t i = level_indices[l_start + li];
            char *err_msg = err_msgs + li * sizeof_ptd_err;
            iter_status[li] = ptd_compose_scc_one(
                    parent, scc_graph, i, theta, theta_len,
                    parent_result, err_msg, sizeof_ptd_err);
        }

        /* Aggregate errors. */
        for (size_t li = 0; li < l_size; ++li) {
            if (iter_status[li] != 0) {
                snprintf(ptd_err, sizeof_ptd_err, "%s",
                         err_msgs + li * sizeof_ptd_err);
                free(err_msgs); free(iter_status);
                free(level_offsets); free(level_indices); free(level_of);
                free(topo_order); free(parent_result);
                ptd_scc_compose_in_progress--;
                return NULL;
            }
        }
        free(err_msgs); free(iter_status);
    }


    free(level_offsets); free(level_indices); free(level_of);
    free(topo_order);
    ptd_scc_compose_in_progress--;
    return parent_result;
}

/* Public entry point: handles arg validation, call counting,
 * and timing — then delegates to the inner body. We split it
 * this way so every early-return path (including OOM and
 * argument errors) still contributes to total_compose_ns and
 * compose_calls. */
double *ptd_compose_scc_prcs(
        struct ptd_graph *parent,
        const struct ptd_scc_graph *scc_graph,
        const double *theta,
        size_t theta_len)
{
    if (parent == NULL || scc_graph == NULL) {
        snprintf(ptd_err, sizeof_ptd_err,
                 "ptd_compose_scc_prcs: NULL argument");
        return NULL;
    }

    /* WP-8: count + time the compose call. */
    uint64_t start_ns = monotonic_ns();
    atomic_add_u64(&g_compose_calls, 1);

    double *result = ptd_compose_scc_prcs_inner(
            parent, scc_graph, theta, theta_len);

    uint64_t elapsed = monotonic_ns() - start_ns;
    atomic_add_u64(&g_total_compose_ns, elapsed);

    return result;
}
