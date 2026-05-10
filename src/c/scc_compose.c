/* Thread-local re-entrancy guard. While the composer is running,
 * inner calls to ptd_expected_waiting_time on synthetic graphs
 * must NOT take the hierarchical-elimination path themselves.
 * The integration in ptd_expected_waiting_time checks this flag
 * before consulting PHASIC_HIERAR_ELIMINATION. */
#if defined(__APPLE__) || defined(__linux__)
#define SCC_COMPOSE_TLS __thread
#else
#define SCC_COMPOSE_TLS
#endif
SCC_COMPOSE_TLS int ptd_scc_compose_in_progress = 0;

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
#include <math.h>

#include "../../api/c/phasic.h"

#define sizeof_ptd_err 4096

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
    /* Iterate in reverse-topological (sink-first) order: process
     * topo_order[n_sccs-1], topo_order[n_sccs-2], ..., topo_order[0]. */

    for (size_t ii = 0; ii < n_sccs; ++ii) {
        size_t i = topo_order[n_sccs - 1 - ii];
        /* Build synthetic graph + metadata + PRC for this SCC.
         * Uses the disk cache via ptd_scc_get_or_compute_prc. */
        struct ptd_graph *synth = NULL;
        struct ptd_scc_synthetic_metadata *meta = NULL;
        struct ptd_desc_reward_compute_parameterized *prc =
                ptd_scc_get_or_compute_prc(scc_graph, i, &synth, &meta);

        if (prc == NULL || synth == NULL || meta == NULL) {
            free(topo_order);
free(parent_result);
            if (ptd_err[0] == '\0') {
                snprintf(ptd_err, sizeof_ptd_err,
                         "ptd_compose_scc_prcs: per-SCC compute failed for SCC %zu", i);
            }
            ptd_scc_compose_in_progress--;
            return NULL;
        }

        /* Step 1: update synth's edge weights from theta. This
         * sets internal edges via coefficients, and Type A / Type C /
         * phantom edges via their placeholder coefficients. We
         * override Type C and phantom below.
         *
         * synth was just freshly built (or just loaded from cache);
         * either way, calling update_weights now is safe and
         * triggers any re-derivation we need. */
        {
            double *theta_copy = (double *)malloc(theta_len * sizeof(double));
            if (theta_copy == NULL) {
                ptd_graph_destroy(synth);
                ptd_scc_synthetic_metadata_destroy(meta);
                ptd_parameterized_reward_compute_graph_destroy(prc);
                free(topo_order);
free(parent_result);
                snprintf(ptd_err, sizeof_ptd_err,
                         "ptd_compose_scc_prcs: oom for synth theta_copy");
                ptd_scc_compose_in_progress--;
                return NULL;
            }
            memcpy(theta_copy, theta, theta_len * sizeof(double));
            ptd_graph_update_weights(synth, theta_copy, theta_len, false);
            free(theta_copy);
            if (ptd_err[0] != '\0') {
                ptd_graph_destroy(synth);
                ptd_scc_synthetic_metadata_destroy(meta);
                ptd_parameterized_reward_compute_graph_destroy(prc);
                free(topo_order);
free(parent_result);
                ptd_scc_compose_in_progress--;
                return NULL;
            }
        }

        /* Step 2: override per-channel Type C and phantom edge
         * weights to inject parent values + downstream results. */
        for (size_t k = 0; k < meta->n_channels; ++k) {
            const struct ptd_scc_channel_info *ch = &meta->channels[k];

            /* Type C edge: d_j_synth -> s_abs_for_channel.
             * Weight = parent's external edge weight at current θ. */
            struct ptd_vertex *parent_dj = parent->vertices[ch->parent_vertex_idx];
            if (ch->parent_edge_idx >= parent_dj->edges_length) {
                ptd_graph_destroy(synth);
                ptd_scc_synthetic_metadata_destroy(meta);
                ptd_parameterized_reward_compute_graph_destroy(prc);
                free(topo_order);
free(parent_result);
                snprintf(ptd_err, sizeof_ptd_err,
                         "ptd_compose_scc_prcs: parent edge idx %zu out of range for vertex %zu",
                         ch->parent_edge_idx, ch->parent_vertex_idx);
                ptd_scc_compose_in_progress--;
                return NULL;
            }
            double parent_external_weight = parent_dj->edges[ch->parent_edge_idx]->weight;
            size_t parent_target_idx = parent_dj->edges[ch->parent_edge_idx]->to->index;

            struct ptd_vertex *synth_dj = synth->vertices[ch->d_j_synth_idx];
            ptd_edge_update_weight(synth_dj->edges[ch->type_c_edge_idx],
                                   parent_external_weight);

            /* Phantom edge: s_abs_for_channel -> phantom.
             * Weight = 1/parent_result[parent_target] so that
             * result[s_abs_for_channel] = parent_result[parent_target]
             * via the phase-type identity result[v] = 1/rate(v)
             * for an absorbing-with-one-child vertex.
             *
             * Special case: if parent_result[parent_target] == 0
             * (e.g. the downstream is a true absorbing vertex like
             * Ω), the channel contributes 0 — set phantom weight
             * to a very large value so result[s_abs] ~ 0. We use
             * +infinity sentinel; the phase-type math handles this
             * via the (1/inf) = 0 limit. Or more robustly, set
             * to 1.0 and rely on result[s_abs] = 1/1 but multiply
             * the contribution by 0 elsewhere — that's complicated.
             * Simpler: leave phantom weight set by update_weights
             * (= 1.0 from placeholder coefficient at current θ), but
             * note result[downstream]=0 means the contribution to
             * result[s_abs] is just 1.0 from the 1/rate term.
             *
             * Actually, the simplest fix: set phantom weight to a
             * large value, getting result[s_abs] ≈ 0 when downstream
             * result is 0. */
            struct ptd_vertex *synth_sabs = synth->vertices[ch->s_abs_synth_idx];
            double phantom_weight;
            if (parent_result[parent_target_idx] > 0.0) {
                phantom_weight = 1.0 / parent_result[parent_target_idx];
            } else if (parent_result[parent_target_idx] < 0.0) {
                /* Negative result shouldn't occur in well-formed
                 * phase-type computations; bail. */
                ptd_graph_destroy(synth);
                ptd_scc_synthetic_metadata_destroy(meta);
                ptd_parameterized_reward_compute_graph_destroy(prc);
                free(topo_order);
free(parent_result);
                snprintf(ptd_err, sizeof_ptd_err,
                         "ptd_compose_scc_prcs: negative parent_result[%zu]=%g",
                         parent_target_idx, parent_result[parent_target_idx]);
                ptd_scc_compose_in_progress--;
                return NULL;
            } else {
                /* Downstream result is 0 (true absorbing).
                 * Set phantom weight to a large value so
                 * result[s_abs] = 1/large ≈ 0. */
                phantom_weight = 1e300;
            }
            ptd_edge_update_weight(synth_sabs->edges[ch->phantom_edge_idx],
                                   phantom_weight);
        }

        /* Step 3: run the elimination. We invalidate
         * reward_compute_graph manually and re-trigger build so
         * the replay sees our overridden weights. */
        if (synth->reward_compute_graph != NULL) {
            free(synth->reward_compute_graph->commands);
            free(synth->reward_compute_graph);
            synth->reward_compute_graph = NULL;
        }
        /* The PRC we got from get_or_compute is owned by us;
         * install it on synth so ptd_expected_waiting_time can
         * use it. (It builds reward_compute_graph from
         * parameterized_reward_compute_graph if the latter is
         * non-NULL.) */
        if (synth->parameterized_reward_compute_graph != NULL) {
            ptd_parameterized_reward_compute_graph_destroy(
                    synth->parameterized_reward_compute_graph);
        }
        synth->parameterized_reward_compute_graph = prc;
        prc = NULL;  /* ownership transferred to synth */

        double *synth_result = ptd_expected_waiting_time(synth, NULL);
        if (synth_result == NULL) {
            ptd_graph_destroy(synth);  /* destroys prc via parameterized_reward_compute_graph */
            ptd_scc_synthetic_metadata_destroy(meta);
            free(topo_order);
free(parent_result);
            if (ptd_err[0] == '\0') {
                snprintf(ptd_err, sizeof_ptd_err,
                         "ptd_compose_scc_prcs: per-SCC elimination failed for SCC %zu", i);
            }
            ptd_scc_compose_in_progress--;
            return NULL;
        }

        /* Step 4: copy per-internal-vertex results into
         * parent_result. Iterate ALL synthetic indices (not just
         * 1..n-1) because when the parent's start is in this SCC,
         * synth index 0 maps to the parent's start vertex. */
        for (size_t v_synth = 0; v_synth < meta->n_vertices; ++v_synth) {
            size_t parent_idx = meta->parent_indices[v_synth];
            if (parent_idx == SIZE_MAX) continue;
            parent_result[parent_idx] = synth_result[v_synth];
        }

        free(synth_result);
        ptd_graph_destroy(synth);  /* also destroys prc via parameterized_reward_compute_graph */
        ptd_scc_synthetic_metadata_destroy(meta);
    }

    free(topo_order);
    ptd_scc_compose_in_progress--;
    return parent_result;
}
