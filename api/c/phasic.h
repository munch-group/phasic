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

#ifndef PTDALGORITHMS_PTD_H
#define PTDALGORITHMS_PTD_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

/* Portable mutex shim for ptd_graph::compute_graph_lock. MSVC (Windows)
 * has no pthread.h, so fall back to a CRITICAL_SECTION. POSIX platforms
 * use pthread_mutex_t directly. The PTD_MUTEX_* macros below give
 * phasic.c a single uniform call site. */
#ifdef _WIN32
#  include <windows.h>
   typedef CRITICAL_SECTION ptd_mutex_t;
#  define PTD_MUTEX_INIT(m)    (InitializeCriticalSection(m), 0)
#  define PTD_MUTEX_LOCK(m)    EnterCriticalSection(m)
#  define PTD_MUTEX_UNLOCK(m)  LeaveCriticalSection(m)
#  define PTD_MUTEX_DESTROY(m) DeleteCriticalSection(m)
#else
#  include <pthread.h>
   typedef pthread_mutex_t ptd_mutex_t;
#  define PTD_MUTEX_INIT(m)    pthread_mutex_init((m), NULL)
#  define PTD_MUTEX_LOCK(m)    pthread_mutex_lock(m)
#  define PTD_MUTEX_UNLOCK(m)  pthread_mutex_unlock(m)
#  define PTD_MUTEX_DESTROY(m) pthread_mutex_destroy(m)
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct ptd_avl_node {
    struct ptd_avl_node *left;
    struct ptd_avl_node *right;
    struct ptd_avl_node *parent;
    signed short balance;
    int *key;
    void *entry;
};

struct ptd_avl_tree {
    struct ptd_avl_node *root;
    size_t key_length;
};

struct ptd_graph;
struct ptd_edge;
struct ptd_vertex;

struct ptd_scc_graph;
struct ptd_scc_edge;
struct ptd_scc_vertex;

/* Thread-local-storage macro. Required for thread safety under concurrent
 * JAX pmap / OpenMP execution: ``ptd_err`` (below) and several scratch
 * allocators in src/c/phasic.c must give each OS thread its own slot to
 * avoid races.
 *
 * We deliberately prefer the GCC/Clang ``__thread`` extension over C11
 * ``_Thread_local`` and C++11 ``thread_local``. Reason: ``ptd_err`` is
 * declared in this header (visible in both C and C++ TUs through
 * extern "C") and defined in src/c/phasic.c. On macOS arm64, mixing
 * a C-side ``_Thread_local`` definition with a C++-side ``thread_local``
 * extern declaration produces an unresolved
 * ``thread-local wrapper routine`` symbol because the two language
 * standards lower to different TLS ABIs. Using ``__thread`` for both
 * sides gives a single consistent ABI (raw TLS, no wrapper functions).
 * Both GCC and Clang support ``__thread`` for C and C++. On MSVC,
 * which defines neither ``__GNUC__`` nor ``__STDC_VERSION__ >= C11``
 * by default, we use ``__declspec(thread)`` which provides the same
 * raw-TLS ABI for both C and C++ translation units. */
#ifndef PTD_TLS
#  if defined(__GNUC__) || defined(__clang__)
#    define PTD_TLS __thread
#  elif defined(_MSC_VER)
#    define PTD_TLS __declspec(thread)
#  elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && \
        !defined(__STDC_NO_THREADS__) && !defined(__cplusplus)
#    define PTD_TLS _Thread_local
#  elif defined(__cplusplus) && __cplusplus >= 201103L
#    define PTD_TLS thread_local
#  else
#    error "Compiler does not support thread-local storage; required for phasic"
#  endif
#endif

extern PTD_TLS char ptd_err[4096];

#ifndef PTD_DEBUG_1_INDEX
#define PTD_DEBUG_1_INDEX 0
#endif

struct ptd_avl_tree *ptd_avl_tree_create(size_t key_length);

void ptd_avl_tree_destroy(struct ptd_avl_tree *avl_tree);

struct ptd_avl_node *ptd_avl_tree_find_or_insert(struct ptd_avl_tree *avl_tree, const int *key, const void *entry);

struct ptd_avl_node *ptd_avl_tree_find(const struct ptd_avl_tree *avl_tree, const int *key);

size_t ptd_avl_tree_max_depth(void *avl_vec_vertex);

struct ptd_directed_graph;
struct ptd_directed_edge;
struct ptd_directed_vertex;

struct ptd_directed_graph {
    size_t vertices_length;
    struct ptd_directed_vertex **vertices;
    struct ptd_directed_vertex *starting_vertex;
};

struct ptd_directed_edge {
    struct ptd_directed_vertex *to;
};

struct ptd_directed_vertex {
    size_t edges_length;
    struct ptd_directed_edge **edges;
    struct ptd_directed_graph *graph;
    size_t index;
};

int ptd_directed_graph_add_edge(struct ptd_directed_vertex *vertex, struct ptd_directed_edge *edge);

void ptd_directed_graph_destroy(struct ptd_directed_graph *graph);

int ptd_directed_vertex_add(struct ptd_directed_graph *graph, struct ptd_directed_vertex *vertex);

void ptd_directed_vertex_destroy(struct ptd_directed_vertex *vertex);

/**
 * Edge mode for graph: determines whether edges are constant or parameterized.
 * Mode is locked after first non-IPV edge is added to prevent mixing.
 */
enum ptd_edge_mode {
    PTD_EDGE_MODE_UNLOCKED = 0,      // No non-IPV edges added yet
    PTD_EDGE_MODE_CONSTANT = 1,       // All non-IPV edges are constant (scalar syntax)
    PTD_EDGE_MODE_PARAMETERIZED = 2   // All non-IPV edges are parameterized (array syntax)
};

/* Opaque per-edge weight formula tape (weight_mode='formula'). A flat
 * stack-machine bytecode that computes weight = f(theta, edge coefficients)
 * for an arbitrary closed-form formula, evaluated in C with no Python. See
 * ptd_weight_tape_* below and src/phasic/weight_formula.py for the compiler. */
struct ptd_weight_tape;

struct ptd_graph {
    size_t vertices_length;
    struct ptd_vertex **vertices;
    struct ptd_vertex *starting_vertex;
    size_t state_length;
    size_t param_length;  // Length of parameter/edge state vectors (set by first add_edge)
    bool parameterized;   // true if param_length > 1
    bool param_length_locked;  // true after first edge added
    enum ptd_edge_mode edge_mode;  // Locked after first non-IPV edge added
    struct ptd_desc_reward_compute *reward_compute_graph;
    struct ptd_desc_reward_compute_parameterized *parameterized_reward_compute_graph;
    /* Zero-copy offset form (dual-form cache load path; defined in phasic.c).
     * When set (cache HIT), the offset executor runs against it and the raw
     * parameterized_reward_compute_graph stays NULL. See zero-copy-cache-plan.md. */
    struct ptd_desc_reward_compute_parameterized_off *parameterized_reward_compute_graph_off;
    /* DEPRECATED (MPFR-A-WP-3): the MPFR-precision compute
     * graph. Used only when PHASIC_USE_MPFR_LEGACY=1 selects
     * the old re-elimination path. The default (and recommended)
     * path is now the MPFR-A consumer that operates on the
     * regular reward_compute_graph at high precision. This
     * field will be removed once the legacy opt-in is no
     * longer needed. */
    struct ptd_desc_reward_compute_mpfr *reward_compute_graph_mpfr;  // DEPRECATED: see MPFR-A
    bool was_dph;
    /* Latches when the graph becomes discrete (was_dph false→true) so the
     * compute-graph wipe in ptd_precompute_reward_compute_graph runs once
     * and then stays off. Originally that wipe was gated on was_dph and
     * was_dph itself was reset to false after the first call; making
     * was_dph permanent (to keep auto-normalisation in update_weights)
     * accidentally turned the wipe into a per-call O(n^3) rebuild. */
    bool dph_compute_invalidated;

    /* Elimination ordering strategy */
    bool use_dyn_ordering;  // If true, use dynamic minimum-degree ordering within SCCs

    /* Trace-based elimination (NULL until first parameter update) */
    struct ptd_elimination_trace *elimination_trace;
    double *current_params;  // Current parameter values (NULL until first update)

    /* Edge-weight version counter. Incremented on every C-level mutation
     * (ptd_graph_update_weights, ptd_edge_update_weight, ptd_edge_update_to).
     * Forward-state context structs stamp their value at creation; cache
     * guards in api/cpp/phasiccpp.h compare against this to detect stale
     * contexts that need rebuilding. Starts at 0; never decreases. */
    uint64_t weight_version;

    /* Per-graph mutex protecting the lazy build inside
     * ptd_precompute_reward_compute_graph. Two threads sharing the same
     * ptd_graph * can both observe reward_compute_graph as NULL and try
     * to build it, racing on the result pointer. The mutex serialises
     * the lazy build. The flag tracks whether ptd_graph_destroy must
     * call PTD_MUTEX_DESTROY (false if PTD_MUTEX_INIT failed during
     * ptd_graph_create). Type is platform-portable (pthread on POSIX,
     * CRITICAL_SECTION on Windows) — see PTD_MUTEX_* macros above. */
    ptd_mutex_t compute_graph_lock;
    bool compute_graph_lock_initialized;

    /* Optional per-edge weight formula tape (weight_mode='formula').
     * NULL unless a formula was compiled and installed via
     * ptd_graph_set_weight_tape. When non-NULL, ptd_graph_update_weights
     * fills each non-IPV edge weight by running the tape over
     * (theta, edge->coefficients) instead of the linear/log inner product.
     * Owned by the graph; freed by ptd_graph_destroy. Theta-independent
     * structure, so it does not affect the symbolic elimination cache. */
    struct ptd_weight_tape *weight_tape;

    /* Per-edge specialized residual tapes (weight_mode='formula'). Each base
     * edge's theta-INDEPENDENT subexpressions are folded once (using that
     * edge's coefficients) and dead select() arms pruned, yielding a small
     * theta-only residual tape. Built lazily on the first tape-mode
     * ptd_graph_update_weights and reused for every later theta (the SVGD inner
     * loop); freed/invalidated when weight_tape changes. Length =
     * wf_residuals_length, one per non-IPV coefficiented edge in update-weights
     * iteration order. NULL until built; wf_residuals_for_tape records which
     * weight_tape they correspond to. */
    struct ptd_weight_tape **wf_residuals;
    size_t wf_residuals_length;
    struct ptd_weight_tape *wf_residuals_for_tape;
};

struct ptd_edge {
    struct ptd_vertex *to;
    double weight;              // Current evaluated weight
    double *coefficients;       // ALWAYS non-NULL, length = graph->param_length
    size_t coefficients_length; // Always = graph->param_length
    bool should_free_coefficients;
};


struct ptd_vertex {
    size_t edges_length;
    struct ptd_edge **edges;
    struct ptd_graph *graph;
    size_t index;
    int *state;
    bool is_aux;  // Marks auxiliary vertices created by add_aux_vertex()
};

struct ptd_graph *ptd_graph_create(size_t state_length);

void ptd_graph_destroy(struct ptd_graph *graph);

struct ptd_vertex *ptd_vertex_create(struct ptd_graph *graph);

struct ptd_vertex *ptd_vertex_create_state(
        struct ptd_graph *graph,
        int *state
);

struct ptd_vertex *ptd_find_or_create_vertex(
        struct ptd_graph *graph,
        struct ptd_avl_tree *avl_tree,
        const int *child_state
);

double ptd_vertex_rate(struct ptd_vertex *vertex);

void ptd_vertex_destroy(struct ptd_vertex *vertex);

struct ptd_edge *ptd_graph_add_edge(
        struct ptd_vertex *from,
        struct ptd_vertex *to,
        double *coefficients,
        size_t coefficients_length
);

void ptd_edge_update_weight(
        struct ptd_edge *edge,
        double weight
);

void ptd_edge_update_to(
    struct ptd_edge *edge,
    struct ptd_vertex *vertex
);

void ptd_notify_change(
        struct ptd_graph *graph
);

void ptd_graph_update_weights(
        struct ptd_graph *graph,
        double *params,
        size_t params_length,
        bool use_log
);

/* ---- Per-edge weight formula tape (weight_mode='formula') ----------------
 *
 * A formula string (compiled in Python by src/phasic/weight_formula.py into
 * a flat bytecode tape) is evaluated per edge in C to fill edge->weight from
 * (theta, edge coefficients). This keeps non-inner-product weights on the
 * fast C/FFI path that parameterises the recorded elimination trace, instead
 * of the slow Python weight-callback. Opcodes mirror weight_formula.OPCODES.
 */

/* Allocate a tape, copying ops/consts. Returns NULL on allocation failure. */
struct ptd_weight_tape *ptd_weight_tape_create(
        const int *ops, size_t ops_length,
        const double *consts, size_t consts_length,
        size_t stack_depth, size_t n_theta, size_t n_coeff
);

void ptd_weight_tape_destroy(struct ptd_weight_tape *tape);

/* Install a tape on the graph (takes ownership; frees any prior tape).
 * Pass NULL to clear. */
void ptd_graph_set_weight_tape(
        struct ptd_graph *graph, struct ptd_weight_tape *tape
);

/* Evaluate a tape for one edge into *out_weight. Returns 0 on success, or
 * non-zero (with ptd_err set) on a structural error (bad opcode, stack
 * under/overflow, or a theta/coeff index out of range). Does NOT check the
 * result's finiteness/sign — the caller does, with edge context. */
int ptd_weight_tape_eval(
        const struct ptd_weight_tape *tape,
        const double *theta, size_t theta_len,
        const double *coeff, size_t coeff_len,
        double *out_weight
);

/* Same evaluation over raw arrays (no tape struct), so the C++ GraphBuilder
 * can evaluate build-time/IPV weights through the single shared VM. */
int ptd_weight_tape_eval_arrays(
        const int *ops, size_t ops_length,
        const double *consts, size_t consts_length,
        size_t stack_depth,
        const double *theta, size_t theta_len,
        const double *coeff, size_t coeff_len,
        double *out_weight
);

/**
 * Set the initial probability vector after the graph has been constructed.
 *
 * Walks only the starting-vertex edges and writes ipv[k] to the k-th edge's
 * weight (in construction order). Does NOT touch any non-starting-vertex
 * edges. Bumps weight_version so the C++ wrapper's forward-state caches
 * (ph_context_markov etc.) detect the change. The symbolic
 * parameterized_reward_compute_graph survives this call (Stage A0
 * invariant) — the symbolic structure depends only on topology +
 * coefficients, neither of which IPV edge-weight updates mutate.
 *
 * @param graph        Graph whose IPV is being set.
 * @param ipv          Array of new IPV edge weights.
 * @param ipv_length   Must equal the number of starting-vertex edges.
 *
 * Sets ptd_err and returns without mutating any edges if:
 *   - graph or ipv is NULL, ipv_length == 0,
 *   - ipv_length does not match starting_vertex->edges_length,
 *   - any ipv[k] is NaN or Inf.
 */
void ptd_graph_update_ipv(
        struct ptd_graph *graph,
        double *ipv,
        size_t ipv_length
);

/**
 * Set the parameter length for a graph before adding edges
 *
 * This function allows explicitly setting the number of model parameters (θ)
 * before adding edges. Useful when edges may have more coefficients than
 * the number of model parameters.
 *
 * @param graph Graph to configure
 * @param param_length Number of model parameters
 *
 * @note Must be called before adding any non-IPV edges
 * @note Once set, all edges must have coefficients_length >= param_length
 */
void ptd_graph_set_param_length(
        struct ptd_graph *graph,
        size_t param_length
);

double *ptd_normalize_graph(struct ptd_graph *graph);

double *ptd_dph_normalize_graph(struct ptd_graph *graph);

double *ptd_expected_waiting_time(struct ptd_graph *graph, double *rewards);

/**
 * Compute expected sojourn time for all states in a single pass
 *
 * Computes the expected time spent in each state before absorption,
 * starting from the initial state. This is equivalent to calling
 * ptd_expected_waiting_time() with unit reward vectors for each state,
 * but much faster (single pass through elimination trace).
 *
 * @param graph Phase-type graph (must be built and have weights set)
 * @return Array of length vertices_length where result[i] = expected time
 *         spent in state i before absorption. Returns NULL on error.
 *         Caller must free() the returned array.
 *
 * Performance: O(n² × m) where n = vertices, m = trace length.
 * Typical speedup vs n calls to expectation(): 10-100x for n > 50.
 */
double *ptd_expected_sojourn_time(struct ptd_graph *graph);

/**
 * Compute expected sojourn times for a subset of vertices (memory-efficient).
 *
 * Instead of allocating an n×n matrix to compute sojourn times for all vertices,
 * this function allocates an n×k matrix where k is the number of target vertices.
 *
 * @param graph The graph to compute sojourn times for
 * @param indices Array of k vertex indices to compute sojourn times for
 * @param k Number of vertices in the indices array
 * @return Array of k sojourn times (must be freed by caller), or NULL on error
 */
double *ptd_expected_sojourn_time_subset(struct ptd_graph *graph, const size_t *indices, size_t k);

// double *ptd_expected_residence_time(struct ptd_graph *graph, double *rewards);

bool ptd_graph_is_acyclic(struct ptd_graph *graph);

struct ptd_vertex **ptd_graph_topological_sort(struct ptd_graph *graph);

struct ptd_graph *ptd_graph_reward_transform(struct ptd_graph *graph, double *rewards);

// struct ptd_clone_res ptd_graph_expectation_dag(struct ptd_graph *graph, double *rewards);

struct ptd_graph *ptd_graph_dph_reward_transform(struct ptd_graph *graph, int *rewards);

long double ptd_random_sample(struct ptd_graph *graph, double *rewards);

long double *ptd_mph_random_sample(struct ptd_graph *graph, double *rewards, size_t vertex_rewards_length);

long double ptd_dph_random_sample(struct ptd_graph *graph, double *rewards);

long double *ptd_mdph_random_sample(struct ptd_graph *graph, double *rewards, size_t vertex_rewards_length);

struct ptd_vertex *ptd_random_sample_stop_vertex(struct ptd_graph *graph, double time);

struct ptd_vertex *ptd_dph_random_sample_stop_vertex(struct ptd_graph *graph, int jumps);

struct ptd_sample_path {
    size_t *vertex_indices;   /* Array of vertex indices visited */
    double *entry_times;      /* Cumulative time when each vertex was entered */
    size_t length;            /* Number of vertices in the path */
};

struct ptd_sample_path *ptd_random_sample_path(struct ptd_graph *graph);

struct ptd_sample_path *ptd_random_sample_path_conditioned(
    struct ptd_graph *graph,
    double *backward_probs
);

double *ptd_backward_probabilities(
    struct ptd_graph *graph,
    size_t *target_vertices,
    size_t n_targets
);

void ptd_sample_path_destroy(struct ptd_sample_path *path);

size_t ptd_random_sample_path_conditioned_fixed(
    struct ptd_graph *graph,
    double *backward_probs,
    size_t max_length,
    unsigned int seed,
    int *out_vertex_indices,
    double *out_entry_times
);

double ptd_defect(struct ptd_graph *graph);

int ptd_validate_graph(const struct ptd_graph *graph);

struct ptd_clone_res {
    struct ptd_avl_tree *avl_tree;
    struct ptd_graph *graph;
};

struct ptd_clone_res ptd_clone_graph(struct ptd_graph *graph, struct ptd_avl_tree *avl_tree);

struct ptd_desc_reward_compute {
    size_t length;
    struct ptd_reward_increase *commands;
};

// MPFR version: for high-precision graph elimination
struct ptd_desc_reward_compute_mpfr {
    size_t length;
    struct ptd_reward_increase_mpfr *commands;
};

struct ptd_reward_increase {
    size_t from;
    size_t to;
    double multiplier;
};

// MPFR version: stores multiplier as string for arbitrary precision
struct ptd_reward_increase_mpfr {
    size_t from;
    size_t to;
    char *multiplier_str;  // String representation from mpfr_get_str()
};

struct ptd_comp_graph_parameterized {
    size_t from;
    size_t to;
    double multiplier;
    double *multiplierptr;
    int type;
    double *fromT;
    double *toT;
};

struct ptd_desc_reward_compute_parameterized {
    size_t length;
    struct ptd_comp_graph_parameterized *commands;
    void *mem;
    void *memr;
};

struct ptd_desc_reward_compute *ptd_graph_ex_absorbation_time_comp_graph(struct ptd_graph *graph);

struct ptd_desc_reward_compute_parameterized *
ptd_graph_ex_absorbation_time_comp_graph_parameterized(struct ptd_graph *graph);

struct ptd_desc_reward_compute *
ptd_graph_build_ex_absorbation_time_comp_graph_parameterized(struct ptd_desc_reward_compute_parameterized *compute);

void ptd_parameterized_reward_compute_graph_destroy(
        struct ptd_desc_reward_compute_parameterized *compute_graph
);

/**
 * Resolve the on-disk cache root directory.
 *
 * Honours the ``PHASIC_CACHE_DIR`` environment variable: if set,
 * its value is used verbatim as the cache root. Otherwise the
 * default ``$HOME/.phasic_cache`` is used.
 *
 * The resolved path is written into ``buf``. The directory is
 * NOT created — callers that need to write into it should ensure
 * the directory exists themselves.
 *
 * On SLURM-style clusters, set ``PHASIC_CACHE_DIR`` to a path on
 * a shared filesystem (e.g. ``$WORK/.phasic_cache``) so workers
 * and the orchestrator share cache entries. The shared filesystem
 * must support POSIX ``rename(2)`` semantics for the atomic
 * write-then-rename used by ``ptd_save_parameterized_reward_compute_graph``
 * to be reliable. NFSv3+, Lustre, and ext4 are fine; some GPFS
 * configurations are not.
 *
 * @param buf Output buffer (should be at least ``PATH_MAX`` bytes).
 * @param buf_len Length of ``buf``.
 * @return 0 on success, -1 on error (sets ``ptd_err`` —
 *         typically because the resolved path is too long).
 */
int ptd_cache_root_dir(char *buf, size_t buf_len);

/**
 * Serialise a parameterised reward compute graph to disk.
 *
 * Walks the commands, encodes each pointer (fromT, toT, multiplierptr)
 * as either a mem-buffer offset or an (vertex_idx, edge_idx, byte_offset)
 * triple, flattens the linked-list mem chain into a single contiguous
 * buffer, and writes the result to ``path`` via atomic write-then-rename
 * so concurrent readers never see a partial file.
 *
 * @param path Destination cache file (parent directory must exist).
 * @param compute Symbolic compute graph to save.
 * @param graph The graph that ``compute`` was built from. Used at save
 *              time to look up edge pointers for the encoding step;
 *              the saved file embeds (vertex_idx, edge_idx) which the
 *              loader resolves against the live graph passed to
 *              ptd_load_parameterized_reward_compute_graph.
 * @return 0 on success, -1 on error (sets ptd_err).
 */
int ptd_save_parameterized_reward_compute_graph(
        const char *path,
        const struct ptd_desc_reward_compute_parameterized *compute,
        const struct ptd_graph *graph);

/**
 * Load a parameterised reward compute graph from disk.
 *
 * Reads the header, validates magic/version, allocates a single
 * ``mem`` block (wrapped in one ll_of_a node so the existing destroy
 * function works unchanged), reads the commands, and re-points all
 * fromT/toT/multiplierptr fields against the loaded mem and the
 * supplied graph's edge weights.
 *
 * @param path Cache file path.
 * @param graph The graph to bind edge-pointer fields to. Must have
 *              the same structure as the graph used at save time.
 * @return Newly allocated compute graph (caller owns; destroy via
 *         ptd_parameterized_reward_compute_graph_destroy), or NULL
 *         on error (cache miss / corrupt file / version mismatch /
 *         OOM). On NULL the caller should fall back to rebuilding
 *         the cache from scratch and ptd_err is set to a
 *         human-readable description of the miss reason.
 */
struct ptd_desc_reward_compute_parameterized *
ptd_load_parameterized_reward_compute_graph(
        const char *path,
        const struct ptd_graph *graph);

/* rev-3 zero-copy cache (see zero-copy-cache-plan.md). ptd_save_pcg_rev3 converts
 * the raw PRC to the offset form and writes the rev-3 layout (magic PTDPRMC3).
 * ptd_load_pcg_rev3 mmaps it (or falls back to read+copy) and returns the offset
 * descriptor, with inputs[] bound against `graph`'s live edge weights; NULL on a
 * miss/stale file. Defined in phasic.c; declared here so the per-SCC cache in
 * scc_synthetic.c uses the same format. ptd_desc_reward_compute_parameterized_off
 * is opaque to callers (only handled by pointer). */
int ptd_save_pcg_rev3(
        const char *path,
        const struct ptd_desc_reward_compute_parameterized *raw,
        const struct ptd_graph *graph);
struct ptd_desc_reward_compute_parameterized_off *
ptd_load_pcg_rev3(const char *path, const struct ptd_graph *graph);

/**
 * v2 save: write a parameterised reward compute graph with
 * EXTERNAL pointer support.
 *
 * Like ptd_save_parameterized_reward_compute_graph, but takes an
 * additional list of external anchors. Any pointer in ``compute``
 * that matches an entry in ``external_anchors`` is encoded as
 * PTD_PCG_PTR_EXTERNAL with the matching index. All other
 * pointers are encoded as MEM/EDGE/NULL as in v1.
 *
 * The on-disk file is written as format revision 2 if
 * ``n_external > 0``, or as revision 1 (v1-equivalent) otherwise.
 *
 * @param path Destination cache file.
 * @param compute Symbolic compute graph to save.
 * @param graph The graph from which compute was built (typically
 *              the synthetic graph). Used to resolve EDGE pointers.
 * @param external_anchors Array of double* pointers; pointers in
 *              compute matching one of these get encoded as
 *              EXTERNAL with the matching array index. May be
 *              NULL if n_external == 0.
 * @param n_external Length of external_anchors.
 * @return 0 on success, -1 on error (sets ptd_err).
 */
int ptd_save_parameterized_reward_compute_graph_ex(
        const char *path,
        const struct ptd_desc_reward_compute_parameterized *compute,
        const struct ptd_graph *graph,
        const double *const *external_anchors,
        size_t n_external);

/**
 * v2 load: read a parameterised reward compute graph with
 * EXTERNAL pointer support.
 *
 * Accepts both rev-1 and rev-2 files. Rev-2 files containing
 * EXTERNAL pointers require external_table to be non-NULL with
 * sufficient n_external; otherwise the load fails.
 *
 * @param path Cache file path.
 * @param graph The graph to bind EDGE pointers against (typically
 *              a freshly-rebuilt synthetic graph at load time).
 * @param external_table Array of doubles. EXTERNAL pointers in
 *              the file resolve to &external_table[index]. May be
 *              NULL if n_external == 0 and the file contains no
 *              EXTERNAL pointers. Caller owns; must outlive the
 *              returned compute graph.
 * @param n_external Length of external_table.
 * @return Newly allocated compute graph (caller owns; destroy via
 *         ptd_parameterized_reward_compute_graph_destroy), or
 *         NULL on cache miss / corrupt file / version mismatch /
 *         missing external_table for a rev-2 file.
 */
struct ptd_desc_reward_compute_parameterized *
ptd_load_parameterized_reward_compute_graph_ex(
        const char *path,
        const struct ptd_graph *graph,
        const double *external_table,
        size_t n_external);


// ============================================================================
// Trace-Based Elimination for Efficient Parameter Updates
// ============================================================================

/**
 * Operation types for trace-based elimination
 *
 * These operations form a linear sequence that can be efficiently
 * replayed with different parameter values.
 */
enum ptd_trace_op_type {
    PTD_OP_CONST = 0,   /* Constant value */
    PTD_OP_PARAM = 1,   /* Parameter reference θ[i] */
    PTD_OP_DOT = 2,     /* Dot product: Σ(cᵢ * θᵢ) */
    PTD_OP_ADD = 3,     /* Addition: a + b */
    PTD_OP_MUL = 4,     /* Multiplication: a * b */
    PTD_OP_DIV = 5,     /* Division: a / b */
    PTD_OP_INV = 6,     /* Inverse: 1 / a */
    PTD_OP_SUM = 7      /* Sum: sum([a, b, c, ...]) */
};

/**
 * Single operation in elimination trace
 */
struct ptd_trace_operation {
    enum ptd_trace_op_type op_type;

    /* For CONST */
    double const_value;

    /* For PARAM */
    size_t param_idx;

    /* For DOT (optimized linear combination) */
    double *coefficients;           /* Coefficient array */
    size_t coefficients_length;     /* Length of coefficient array */

    /* For binary/unary operations */
    size_t *operands;               /* Indices of operand operations */
    size_t operands_length;         /* Number of operands */
};

/**
 * Complete elimination trace
 *
 * This structure records all operations needed to eliminate a graph,
 * enabling fast replay with different parameter values.
 */
struct ptd_elimination_trace {
    /* Operation sequence */
    struct ptd_trace_operation *operations;
    size_t operations_length;

    /* Vertex rate mappings (vertex_idx → operation_idx) */
    size_t *vertex_rates;

    /* Edge probability mappings (vertex_idx → [operation_idx]) */
    size_t **edge_probs;
    size_t *edge_probs_lengths;

    /* Target vertex mappings (vertex_idx → [target_vertex_idx]) */
    size_t **vertex_targets;
    size_t *vertex_targets_lengths;

    /* Vertex states (copied from graph) */
    int **states;
    size_t state_length;

    /* Metadata */
    size_t starting_vertex_idx;
    size_t n_vertices;
    size_t param_length;
    bool is_discrete;
};

/**
 * Result of trace evaluation
 *
 * Contains evaluated vertex rates and edge probabilities for
 * specific parameter values.
 */
struct ptd_trace_result {
    double *vertex_rates;              /* Array[n_vertices] */
    double **edge_probs;               /* Array[n_vertices][n_edges] */
    size_t *edge_probs_lengths;        /* Array[n_vertices] */
    size_t **vertex_targets;           /* Array[n_vertices][n_edges] */
    size_t *vertex_targets_lengths;    /* Array[n_vertices] */
    size_t n_vertices;
};

// Trace-based elimination functions

/**
 * Record elimination trace from parameterized graph
 *
 * Performs graph elimination while recording all arithmetic operations
 * in a linear sequence. The trace can be efficiently replayed with
 * different parameter values.
 *
 * @param graph Parameterized graph
 * @return Elimination trace, or NULL on error
 *
 * Time complexity: O(n³) one-time cost
 * Space complexity: O(n²) for trace storage
 */
struct ptd_elimination_trace *ptd_record_elimination_trace(
    struct ptd_graph *graph
);

/**
 * Evaluate elimination trace with concrete parameter values
 *
 * Executes the recorded operation sequence with given parameters
 * to produce vertex rates and edge probabilities.
 *
 * @param trace Elimination trace
 * @param params Parameter array
 * @param params_length Length of parameter array
 * @return Trace evaluation result, or NULL on error
 *
 * Time complexity: O(n) where n = number of operations
 */
struct ptd_trace_result *ptd_evaluate_trace(
    const struct ptd_elimination_trace *trace,
    const double *params,
    size_t params_length
);

/**
 * Build reward compute graph from trace evaluation result
 *
 * Converts trace evaluation results into the internal reward_compute_graph
 * structure used by pdf/moment computations.
 *
 * @param result Trace evaluation result
 * @param graph Graph structure (for vertex references)
 * @return Reward compute graph, or NULL on error
 */
struct ptd_desc_reward_compute *ptd_build_reward_compute_from_trace(
    const struct ptd_trace_result *result,
    struct ptd_graph *graph
);


/**
 * Destroy elimination trace and free all memory
 */
void ptd_elimination_trace_destroy(struct ptd_elimination_trace *trace);

/**
 * Destroy trace evaluation result and free all memory
 */
void ptd_trace_result_destroy(struct ptd_trace_result *result);

/**
 * Load elimination trace from disk cache
 *
 * Traces are stored in ~/.phasic_cache/traces/ as JSON files.
 * The cache can be disabled by setting PHASIC_DISABLE_CACHE=1.
 *
 * @param hash_hex Hexadecimal hash string identifying the trace (64 chars)
 * @return Loaded trace (caller must call ptd_elimination_trace_destroy), or NULL if not found
 *
 * Time complexity: O(n) where n = trace size (file I/O + JSON parsing)
 */
struct ptd_elimination_trace *ptd_load_trace_from_cache(const char *hash_hex);

/**
 * Save elimination trace to disk cache
 *
 * Traces are stored in ~/.phasic_cache/traces/ as JSON files.
 * The cache can be disabled by setting PHASIC_DISABLE_CACHE=1.
 *
 * @param hash_hex Hexadecimal hash string identifying the trace (64 chars)
 * @param trace The trace to save
 * @return true on success, false on error or if cache is disabled
 *
 * Time complexity: O(n) where n = trace size (JSON serialization + file I/O)
 */
bool ptd_save_trace_to_cache(const char *hash_hex, const struct ptd_elimination_trace *trace);


struct ptd_scc_graph {
    size_t vertices_length;
    struct ptd_scc_vertex **vertices;
    struct ptd_scc_vertex *starting_vertex;
    struct ptd_graph *graph;
};

struct ptd_scc_edge {
    struct ptd_scc_vertex *to;
};

struct ptd_scc_vertex {
    size_t edges_length;
    struct ptd_scc_edge **edges;
    struct ptd_scc_graph *graph;
    size_t index;
    size_t internal_vertices_length;
    struct ptd_vertex **internal_vertices;
};

int ptd_precompute_reward_compute_graph(struct ptd_graph *graph);

struct ptd_scc_graph *ptd_find_strongly_connected_components(struct ptd_graph *graph);

struct ptd_scc_vertex **ptd_scc_graph_topological_sort(struct ptd_scc_graph *graph);

void ptd_scc_graph_destroy(struct ptd_scc_graph *scc_graph);

/**
 * Reference to a parent-graph edge that crosses the boundary of an SCC.
 *
 * Used by ptd_scc_synthetic_metadata to record where the parent's
 * external edges live so the composer (WP-5) can wire them to the
 * synthetic graph's placeholder edges.
 */
struct ptd_scc_external_edge_ref {
    size_t parent_vertex_idx;  /* original-graph vertex index */
    size_t parent_edge_idx;    /* index into that vertex's edges[] */
};

/**
 * Per-channel info for one external out-edge from a downstream-
 * connecting vertex of an SCC.
 *
 * One channel = one parent-graph edge from a downstream-connecting
 * vertex `d_j` of the SCC to a parent vertex `w` outside the SCC.
 * Each channel gets its own synthetic absorbing vertex (the
 * "per-channel s_abs"), with two synthetic edges:
 *
 *   - Type C: ``d_j_synth → s_abs_for_channel`` with the parent's
 *     external edge weight as its (live) weight. The parent edge
 *     weight is set on the synthetic graph's edge slot before each
 *     compose run.
 *
 *   - Phantom: ``s_abs_for_channel → phantom_absorbing`` with a
 *     weight that the composer sets to ``1 / result[w_in_parent]``
 *     so that the per-SCC elimination produces
 *     ``result[s_abs_for_channel] = result[w_in_parent]``. This
 *     injects the downstream value into this SCC's elimination
 *     via the standard phase-type identity
 *     ``result[v] = 1/rate(v) + Σ prob·result[child]``.
 */
struct ptd_scc_channel_info {
    /* Parent-graph edge that this channel represents. */
    size_t parent_vertex_idx;     /* the d_j vertex in the parent */
    size_t parent_edge_idx;       /* index into d_j's edges[] */

    /* Synthetic-graph vertex indices for this channel. */
    size_t d_j_synth_idx;         /* the downstream-connecting vertex */
    size_t s_abs_synth_idx;       /* the per-channel absorbing vertex */

    /* Synthetic-graph edge indices (into the from-vertex's
     * edges[]) for the Type C edge and the phantom edge.
     * The composer reads/writes weights at these slots. */
    size_t type_c_edge_idx;       /* d_j_synth → s_abs_for_channel */
    size_t phantom_edge_idx;      /* s_abs_for_channel → phantom (always 0; only one out-edge) */
};

/**
 * Vertex categorisation and parent-mapping for a synthetic SCC graph.
 *
 * The synthetic graph contains, in canonical layout:
 *   index 0:                            synthetic source vertex
 *   indices [1, 1+n_upstream_connecting):
 *                                       upstream-connecting vertices
 *   indices [1+n_upstream_connecting,
 *            1+n_upstream_connecting+n_internal_only):
 *                                       pure-internal vertices
 *   indices [1+n_upstream_connecting+n_internal_only,
 *            1+n_uc+n_io+n_dc):         downstream-connecting vertices
 *   indices [1+n_uc+n_io+n_dc,
 *            1+n_uc+n_io+n_dc+n_channels):
 *                                       per-channel absorbing vertices
 *                                       (one per external out-edge)
 *   index n_vertices - 1:               phantom absorbing vertex
 *                                       (only true absorbing in graph)
 *
 * `parent_indices[k]` is the original-graph vertex index of the
 * synthetic-graph vertex at synthetic-graph index k, or SIZE_MAX
 * for the synthetic source, per-channel absorbing vertices, and
 * the phantom absorbing vertex (none have parent counterparts).
 *
 * The per-channel absorbing layout — one synthetic vertex per
 * external out-edge — is what makes downstream-result injection
 * work without breaking cross-parent reuse. Two parents with
 * identical SCC internals AND identical external fan-out shape
 * (same `(d_j, channel-position)` count per d_j) produce
 * identical synthetic graphs.
 */
struct ptd_scc_synthetic_metadata {
    /* The source SCC's index in the parent's SCC decomposition. */
    size_t scc_index;

    /* Total number of vertices in the synthetic graph. Equals
     * 1 + n_upstream_connecting + n_internal_only +
     * n_downstream_connecting + n_channels + 1
     * (source + internals + per-channel absorbings + phantom). */
    size_t n_vertices;

    /* Per-category counts. */
    size_t n_upstream_connecting;
    size_t n_internal_only;
    size_t n_downstream_connecting;

    /* Number of external out-channels (sum over all
     * downstream-connecting vertices of their external out-edge
     * counts). One per-channel absorbing vertex per channel. */
    size_t n_channels;

    /* Mapping from synthetic-graph vertex index to parent-graph
     * vertex index. Length == n_vertices. SIZE_MAX for synthetic
     * source, per-channel absorbing vertices, and phantom. */
    size_t *parent_indices;

    /* Per-synthetic-vertex external edge references.
     *
     * Both arrays are indexed by synthetic-graph vertex index
     * (length n_vertices).
     *
     * external_in_edges[v_synth] lists every parent-graph edge
     * whose target is the parent-vertex corresponding to
     * v_synth and whose source is OUTSIDE the SCC. Non-empty
     * exactly for upstream-connecting vertices.
     *
     * external_out_edges[v_synth] lists every parent-graph edge
     * whose source is the parent-vertex corresponding to v_synth
     * and whose target is OUTSIDE the SCC. Non-empty exactly for
     * downstream-connecting vertices.
     */
    struct ptd_scc_external_edge_ref **external_in_edges;
    size_t *external_in_edges_lengths;  /* length: n_vertices */
    struct ptd_scc_external_edge_ref **external_out_edges;
    size_t *external_out_edges_lengths;  /* length: n_vertices */

    /* Per-channel info — one entry per external out-channel.
     * Length == n_channels. The composer iterates this array,
     * setting Type C edge weights from parent values and
     * phantom edge weights from downstream results.
     *
     * Channels are ordered in the same order they were
     * encountered while walking downstream-connecting vertices
     * (in canonical synthetic order) and their external_out_edges
     * lists. */
    struct ptd_scc_channel_info *channels;
};

/**
 * Build a self-contained synthetic graph for one SCC of a parent.
 *
 * Output topology (per ``ptd_scc_synthetic_metadata``):
 *   - Synthetic source vertex.
 *   - All internal vertices of the SCC (state vectors copied,
 *     internal edges copied verbatim including aux-style
 *     coefficients_length==0 edges).
 *   - One synthetic absorbing vertex per external out-channel
 *     (i.e. per parent-graph edge ``d_j -> w`` where ``d_j`` is
 *     in the SCC and ``w`` is outside).
 *   - One phantom absorbing vertex (the only true absorbing
 *     vertex in the graph; result == 0 by structural fact).
 *
 * Edge structure:
 *   - Synthetic source -> upstream-connecting (Type A): placeholder
 *     coefficient (length param_length, first slot 1.0).
 *   - Internal -> internal: copied verbatim from parent.
 *   - Downstream-connecting -> per-channel absorbing (Type C):
 *     placeholder coefficient (length param_length, first slot
 *     1.0). The Type C edge weight slot is what the composer
 *     overwrites with the parent's actual external edge weight
 *     before each compose run.
 *   - Per-channel absorbing -> phantom (one outgoing edge each):
 *     placeholder coefficient. The composer overwrites this
 *     edge's weight slot with ``1/result[w_in_parent]`` so
 *     ``result[s_abs_for_channel] = result[w_in_parent]`` (the
 *     phase-type identity ``result[v] = 1/rate(v)`` for an
 *     absorbing-with-one-child vertex).
 *
 * Cross-parent invariance: synthetic-graph topology depends only
 * on the SCC's intrinsic content (internal structure + per-d_j
 * external out-edge count). Two parents matching on those
 * properties produce identical synthetic graphs and identical
 * content hashes. The parent-specific values (Type C weights,
 * phantom weights) flow through edge weight slots at compose
 * time, not through the cached PRC structure.
 *
 * Vertex ordering is canonical (lexicographic state vector,
 * out-edge-signature tiebreak for duplicates). The returned
 * graph can be passed to
 * ptd_graph_ex_absorbation_time_comp_graph_parameterized without
 * modification.
 *
 * @param scc_graph     SCC decomposition of the parent graph.
 * @param scc_index     Index of the SCC to wrap. Must be < scc_graph->vertices_length.
 * @param metadata_out  Receives a newly allocated metadata struct.
 *                      Caller owns; destroy via
 *                      ptd_scc_synthetic_metadata_destroy. NULL on error.
 * @return Newly allocated synthetic graph (caller owns; destroy
 *         via ptd_graph_destroy), or NULL on error (sets ptd_err).
 */
struct ptd_graph *ptd_scc_build_synthetic_graph(
        const struct ptd_scc_graph *scc_graph,
        size_t scc_index,
        struct ptd_scc_synthetic_metadata **metadata_out);

/**
 * Free a metadata struct produced by ptd_scc_build_synthetic_graph.
 * Safe to call with NULL.
 */
void ptd_scc_synthetic_metadata_destroy(
        struct ptd_scc_synthetic_metadata *metadata);

/**
 * Collect external-anchor pointers for a synthetic graph.
 *
 * Walks the synthetic graph and produces a flat array of
 * (double *) pointers: one per Type A edge (synthetic source ->
 * upstream-connecting) and one per Type C edge (downstream-
 * connecting -> synthetic absorbing). Each pointer is the address
 * of the placeholder edge's coefficients[0] slot.
 *
 * The returned array is suitable for passing as ``external_anchors``
 * to ptd_save_parameterized_reward_compute_graph_ex.
 *
 * Anchor ordering (deterministic):
 *   1. Type A edges first, in synthetic-source-edge order.
 *   2. Type C edges next, in the order encountered while walking
 *      vertex synthetic-indices 1..n-2 and collecting edges that
 *      target the synthetic absorbing vertex (last index).
 *
 * @param synth Synthetic graph produced by
 *              ptd_scc_build_synthetic_graph.
 * @param meta  Metadata for the synthetic graph (used to identify
 *              the synthetic absorbing vertex's index). May be
 *              NULL; if NULL, the absorbing vertex is taken to
 *              be at index synth->vertices_length - 1.
 * @param n_anchors_out Receives the number of anchors written.
 * @return Newly allocated array of double* (caller must free()),
 *         or NULL if the graph is empty / on OOM (with ptd_err
 *         set on OOM).
 */
double **ptd_scc_collect_external_anchors(
        const struct ptd_graph *synth,
        const struct ptd_scc_synthetic_metadata *meta,
        size_t *n_anchors_out);

/**
 * Get-or-compute the per-SCC PRC, with disk caching.
 *
 * For the SCC at ``scc_index`` of ``scc_graph``, returns the
 * parameterised reward compute graph for the SCC's synthetic
 * graph (built via ptd_scc_build_synthetic_graph). The result is
 * cached on disk under
 * ``~/.phasic_cache/parameterized_reward_compute/scc_<hash>.bin``,
 * where <hash> is ptd_graph_content_hash of the synthetic graph.
 *
 * On cold cache, builds the synthetic graph, eliminates it, saves
 * the PRC. On warm cache, loads from disk. Either way, returns
 * the synthetic graph (caller owns; destroy via
 * ptd_graph_destroy) and the metadata (caller owns; destroy via
 * ptd_scc_synthetic_metadata_destroy) so the composer can
 * resolve the PRC's pointer references.
 *
 * Set PHASIC_DISABLE_CACHE=1 to skip both load and save.
 *
 * @param scc_graph     SCC decomposition of the parent graph.
 * @param scc_index     Index of the SCC. Must be < scc_graph->vertices_length.
 * @param synth_out     Receives the synthetic graph (caller owns).
 *                      The returned PRC's pointers reference this
 *                      graph's edge weights, so synth_out must
 *                      outlive the PRC.
 * @param metadata_out  Receives the synthetic-graph metadata
 *                      (caller owns).
 * @return Newly allocated PRC (caller destroys via
 *         ptd_parameterized_reward_compute_graph_destroy), or
 *         NULL on error. On NULL, *synth_out and *metadata_out
 *         are also NULL.
 */
struct ptd_desc_reward_compute_parameterized *
ptd_scc_get_or_compute_prc(
        const struct ptd_scc_graph *scc_graph,
        size_t scc_index,
        struct ptd_graph **synth_out,
        struct ptd_scc_synthetic_metadata **metadata_out);

/**
 * WP-5: SCC composition. Compute the parent-level expected
 * waiting time vector by composing per-SCC PRCs.
 *
 * Walks SCCs in reverse-topological order (sink-first). For
 * each SCC: builds (or retrieves cached) the synthetic graph
 * + PRC, sets per-channel placeholder edge weights based on
 * the parent's external edge weights and downstream-SCC results
 * already computed, runs the per-SCC elimination, copies the
 * results into the parent-wide result vector.
 *
 * The result is numerically equivalent to running the
 * monolithic eliminator on the parent graph at the same theta.
 *
 * The parent graph's edges are updated to reflect ``theta``
 * (via ``ptd_graph_update_weights``) before composition, so
 * the parent's ``edge->weight`` slots can be read for Type C
 * bindings.
 *
 * @param parent     Parent graph. Must be parameterised.
 *                   ``parent->edges`` are mutated to reflect
 *                   theta after the call.
 * @param scc_graph  Parent's SCC decomposition.
 * @param theta      Parameter vector. Length must equal
 *                   ``parent->param_length``.
 * @param theta_len  Length of theta.
 * @return Newly allocated result vector of length
 *         ``parent->vertices_length`` (caller frees), or NULL
 *         on error (sets ptd_err).
 */
double *ptd_compose_scc_prcs(
        struct ptd_graph *parent,
        const struct ptd_scc_graph *scc_graph,
        const double *theta,
        size_t theta_len);

/* SLURM-WP-4: synth-only cache-or-compute.
 *
 * Operates on an already-built synthetic SCC graph (from
 * ptd_scc_build_synthetic_graph, or from JSON deserialisation
 * in a distributed worker). Builds the cache path from the
 * synth's content hash, tries to load, and on miss runs the
 * parameterised eliminator and saves the result.
 *
 * Used by ptd_scc_get_or_compute_prc (after the synth-build
 * step) and intended for direct use by the standalone worker
 * CLI in SLURM distributed mode (SLURM-WP-3).
 *
 * @param synth Synthetic SCC graph. Must be non-NULL.
 * @return PRC on success (caller owns; destroy via
 *         ptd_parameterized_reward_compute_graph_destroy unless
 *         installed on a graph), NULL on error (sets ptd_err).
 *         The synth itself is NOT freed by this function.
 */
struct ptd_desc_reward_compute_parameterized *
ptd_synth_get_or_compute_prc(struct ptd_graph *synth);

/* Thread-local re-entrancy guard for the composer. While
 * ptd_compose_scc_prcs is running on the current thread, inner
 * ptd_expected_waiting_time calls on synthetic SCC subgraphs
 * must NOT take the hierarchical path themselves. Defined in
 * scc_compose.c. */
#if defined(__APPLE__) || defined(__linux__)
#define PTD_SCC_TLS __thread
#else
#define PTD_SCC_TLS
#endif
extern PTD_SCC_TLS int ptd_scc_compose_in_progress;

/* WP-8: telemetry for the SCC composer.
 *
 * The composer maintains process-wide counters of cache hits,
 * cache misses, total compose calls, and cumulative wall time
 * spent inside ptd_compose_scc_prcs. Counters are atomic and
 * safe to read concurrently with composer activity.
 *
 * Cache hits are counted when ptd_scc_get_or_compute_prc loads
 * a per-SCC PRC from disk; misses are counted when it falls
 * through to recompute. Counters reset on
 * ptd_scc_compose_stats_reset().
 */
struct ptd_scc_compose_stats {
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t compose_calls;
    uint64_t total_compose_ns;
    /* SCCs whose synth was below PHASIC_MIN_SCC_SIZE_TO_CACHE
     * and so bypassed the cache entirely (no load attempt, no
     * save). These are NOT counted as hits or misses. */
    uint64_t cache_bypassed;
};

void ptd_scc_compose_stats_get(struct ptd_scc_compose_stats *out);
void ptd_scc_compose_stats_reset(void);

/* Internal helpers used by ptd_scc_get_or_compute_prc and
 * ptd_compose_scc_prcs to bump counters. Exposed so they can
 * be linked across translation units. Not part of the stable
 * public API. */
void ptd_scc_compose_stats_record_hit(void);
void ptd_scc_compose_stats_record_miss(void);
void ptd_scc_compose_stats_record_bypass(void);

struct ptd_phase_type_distribution {
    size_t length;
    double *initial_probability_vector;
    double **sub_intensity_matrix;
    struct ptd_vertex **vertices;
    size_t memory_allocated;
};

struct ptd_phase_type_distribution *ptd_graph_as_phase_type_distribution(struct ptd_graph *graph);

void ptd_phase_type_distribution_destroy(struct ptd_phase_type_distribution *ptd);

int ptd_vertex_to_s(struct ptd_vertex *vertex, char *buffer, size_t buffer_length);

struct ptd_probability_distribution_context {
    double pdf;
    double cdf;
    long double *probability_at;
    long double *accumulated_visits;
    struct ptd_graph *graph;
    void *priv;
    long double time;
    int64_t granularity;
    /* graph->weight_version snapshot taken when this context was created.
     * Used by C++ wrapper cache guards (api/cpp/phasiccpp.h) to detect
     * stale forward state after edge weights change. */
    uint64_t weight_version_at_creation;
};

struct ptd_probability_distribution_context *ptd_probability_distribution_context_create(
        struct ptd_graph *graph,
        int64_t granularity
);

void ptd_probability_distribution_context_destroy(
        struct ptd_probability_distribution_context *context
);

int ptd_probability_distribution_step(
        struct ptd_probability_distribution_context *context
);

struct ptd_dph_probability_distribution_context {
    double pmf;
    double cdf;
    long double *probability_at;
    long double *accumulated_visits;
    struct ptd_graph *graph;
    void *priv;
    size_t priv2;
    double priv3;
    int jumps;
    /* graph->weight_version snapshot at creation; see same field on
     * ptd_probability_distribution_context for the rationale. */
    uint64_t weight_version_at_creation;
};

struct ptd_dph_probability_distribution_context *ptd_dph_probability_distribution_context_create(
        struct ptd_graph *graph
);

void ptd_dph_probability_distribution_context_destroy(
        struct ptd_dph_probability_distribution_context *context
);

int ptd_dph_probability_distribution_step(
        struct ptd_dph_probability_distribution_context *context
);

/**
 * Compute PDF and gradient w.r.t. parameters using forward algorithm
 *
 * This extends the standard forward algorithm (Algorithm 4) to track
 * probability gradients through the DP recursion. Gradients are computed
 * via chain rule through graph traversal - no matrix operations.
 *
 * @param graph Parameterized graph with symbolic edge expressions
 * @param time Time point to evaluate PDF at
 * @param granularity Discretization granularity (0 = auto-select)
 * @param params Parameter vector θ
 * @param n_params Length of params array
 * @param pdf_value Output: PDF(time|θ)
 * @param pdf_gradient Output: ∇PDF(time|θ), shape (n_params,)
 *        Must be pre-allocated with size n_params
 *
 * @return 0 on success, non-zero on error
 *
 * @note This uses the same graph-based approach as pdf(), just with
 *       gradient tracking. No matrix exponentiation.
 *
 * @note For multiple time points, call this function in a loop.
 *       Each call is independent.
 *
 * @note Complexity: O(k·m·p) where k=max_jumps, m=edges, p=n_params
 *       This is p× slower than forward-only, but still graph-based.
 */
int ptd_graph_pdf_with_gradient(
    struct ptd_graph *graph,
    double time,
    size_t granularity,
    const double *params,
    size_t n_params,
    double *pdf_value,
    double *pdf_gradient
);


#ifndef PTD_INTEGRATE_EXCEPTIONS
#define DIE_ERROR(error_code, error, ...) do {     \
char error_formatted[1024];                        \
char error_formatted_line[1024];                   \
                                                   \
snprintf(error_formatted,                          \
         sizeof(error_formatted),                  \
         error, ##__VA_ARGS__);                    \
snprintf(error_formatted_line,                     \
         sizeof(error_formatted_line),             \
         "%s @ %s (%d)", error_formatted,          \
         __FILE__, __LINE__);                      \
                                                   \
fprintf(stderr, "%s\n", error_formatted_line);     \
exit(error_code);   \
} while(0)
#else
#include <stdexcept>
#define DIE_ERROR(error_code, error, ...) do {     \
char error_formatted[1024];                        \
char error_formatted_line[1024];                   \
                                                   \
snprintf(error_formatted,                          \
         sizeof(error_formatted),                  \
         error, ##__VA_ARGS__);                    \
snprintf(error_formatted_line,                     \
         sizeof(error_formatted_line),             \
         "%s @ %s (%d)", error_formatted,          \
         __FILE__, __LINE__);                      \
                                                   \
fprintf(stderr, "%s\n", error_formatted_line);     \
throw std::runtime_error(error_formatted_line); \
} while(0)
#endif

// DEBUG_PRINT is deprecated - use PTD_LOG_DEBUG from phasic_log.h instead
// This macro is kept for backward compatibility but routes through the logging system
#include "phasic_log.h"
#define DEBUG_PRINT(message, ...) ptd_log(PTD_LOG_LEVEL_DEBUG, message, ##__VA_ARGS__)

/**
 * Creates a Laplace-transformed graph.
 *
 * Returns a new graph where each transient state has an additional edge
 * to the absorbing state with weight theta (or theta added to existing
 * absorbing edge weight). This transformation allows computing the Laplace
 * transform L(theta) = E[exp(-theta * T)] via expectation() on the result.
 *
 * @param graph The phase-type graph
 * @param avl_tree The AVL tree for vertex lookup
 * @param theta The Laplace transform parameter
 * @return A clone result containing the transformed graph and its AVL tree
 */
struct ptd_clone_res ptd_graph_laplace_transform(struct ptd_graph *graph, struct ptd_avl_tree *avl_tree, double theta);

#ifdef __cplusplus
}
#endif

#endif //PTDALGORITHMS_PTD_H
