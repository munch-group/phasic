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
#include <limits.h>

// Platform-specific includes
#ifndef _WIN32
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <unistd.h>
#else
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <io.h>      // _commit, _unlink, _getpid, _fileno
    #include <process.h> // _getpid
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h> // MoveFileExA, MOVEFILE_REPLACE_EXISTING
    // Windows doesn't have rand_r; provide a simple replacement
    static int rand_r(unsigned int *seedp) {
        *seedp = *seedp * 1103515245 + 12345;
        return (int)((*seedp / 65536) % 32768);
    }
    // POSIX names map to Microsoft's underscore-prefixed equivalents.
    // _commit() is the closest analogue to fsync(): it flushes the
    // file descriptor's buffers to disk via FlushFileBuffers().
    #define fsync(fd)  _commit(fd)
    #define fileno(fp) _fileno(fp)
    #define unlink(p)  _unlink(p)
    #define getpid()   _getpid()
#endif

/* Cross-platform monotonic clock in nanoseconds. POSIX clock_gettime /
 * CLOCK_MONOTONIC are unavailable under MSVC, so Windows uses the
 * high-resolution performance counter. Mirrors monotonic_ns() in
 * scc_compose.c. */
static uint64_t monotonic_ns(void) {
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

/* Atomic replace of `dst` with `src`. POSIX rename() already replaces
 * an existing destination atomically on the same filesystem; the MSVCRT
 * rename() does not (it fails with EEXIST). MoveFileExA with
 * MOVEFILE_REPLACE_EXISTING gives the POSIX semantics on NTFS.
 * Returns 0 on success, non-zero on failure (errno set on POSIX;
 * GetLastError() on Windows, mapped to a non-zero return). */
static int ptd_atomic_rename(const char *src, const char *dst) {
#ifdef _WIN32
    if (MoveFileExA(src, dst, MOVEFILE_REPLACE_EXISTING) == 0) {
        return -1;
    }
    return 0;
#else
    return rename(src, dst);
#endif
}

#include "phasic.h"
#include "../../api/c/phasic_hash.h"
#include "phasic_log.h"

#ifdef HAVE_MPFR
#include <mpfr.h>
#include <gmp.h>
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

/* Per-thread storage: each OS thread gets its own zero-initialised slot.
 * The matching extern declaration and the PTD_TLS macro itself live in
 * api/c/phasic.h. ``volatile`` was previously used but served no purpose
 * (no signal handlers or hardware accessors read it); TLS gives correct
 * isolation under concurrent execution. */
PTD_TLS char ptd_err[4096];

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

// ============================================================================
// Kahan Summation for Numerical Stability
// ============================================================================
// Compensated summation algorithm that reduces rounding errors from O(nε) to O(ε)
// where n = number of operations, ε = machine precision
//
// Critical for large graphs where thousands of operations accumulate

/**
 * Kahan summation state
 */
struct kahan_sum {
    double sum;            // Running sum
    double compensation;   // Running compensation for lost low-order bits
};

/**
 * Initialize Kahan summation
 */
static inline void kahan_init(struct kahan_sum *k) {
    k->sum = 0.0;
    k->compensation = 0.0;
}

/**
 * Add value to Kahan sum with compensation
 *
 * Algorithm:
 * y = value - compensation   # Compensate for previous lost bits
 * t = sum + y               # Add to sum (may lose low-order bits)
 * compensation = (t - sum) - y  # Recover lost bits for next iteration
 * sum = t
 */
static inline void kahan_add(struct kahan_sum *k, double value) {
    double y = value - k->compensation;
    double t = k->sum + y;
    k->compensation = (t - k->sum) - y;
    k->sum = t;
}

/**
 * Get final Kahan sum result
 */
static inline double kahan_result(const struct kahan_sum *k) {
    return k->sum;
}

// ============================================================================
// Utility Data Structure Implementations
// ============================================================================
// Full implementations of vector, queue, and stack utilities needed for
// SCC/topological sort and other graph algorithms.

// Vector implementation
static struct ptd_vector *vector_create() {
    struct ptd_vector *vec = (struct ptd_vector *)calloc(1, sizeof(struct ptd_vector));
    if (vec == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate vector");
        return NULL;
    }
    vec->entries = 0;
    vec->arr = NULL;
    return vec;
}

static int vector_add(struct ptd_vector *vector, void *entry) {
    if (vector == NULL) return -1;

    // Check if we need to resize (use power of 2 sizing)
    bool is_power_of_2 = (vector->entries & (vector->entries - 1)) == 0;

    if (is_power_of_2) {
        size_t new_size = (vector->entries == 0) ? 1 : vector->entries * 2;
        void **new_arr = (void **)realloc(vector->arr, new_size * sizeof(void *));
        if (new_arr == NULL) {
            sprintf((char*)ptd_err, "Failed to grow vector");
            return -1;
        }
        vector->arr = new_arr;
    }

    vector->arr[vector->entries++] = entry;
    return 0;
}

static void *vector_get(struct ptd_vector *vector, size_t index) {
    if (vector == NULL || index >= vector->entries) {
        return NULL;
    }
    return vector->arr[index];
}

static size_t vector_length(struct ptd_vector *vector) {
    if (vector == NULL) return 0;
    return vector->entries;
}

static void vector_destroy(struct ptd_vector *vector) {
    if (vector == NULL) return;
    free(vector->arr);
    free(vector);
}

// Queue implementation (using linked list)
static struct ptd_queue *queue_create() {
    struct ptd_queue *queue = (struct ptd_queue *)calloc(1, sizeof(struct ptd_queue));
    if (queue == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate queue");
        return NULL;
    }
    return queue;
}

static void queue_destroy(struct ptd_queue *queue) {
    if (queue == NULL) return;

    struct ptd_ll *current = queue->ll;
    while (current != NULL) {
        struct ptd_ll *next = current->next;
        free(current);
        current = next;
    }
    free(queue);
}

static int queue_enqueue(struct ptd_queue *queue, void *entry) {
    if (queue == NULL) return -1;

    struct ptd_ll *node = (struct ptd_ll *)malloc(sizeof(struct ptd_ll));
    if (node == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate queue node");
        return -1;
    }

    node->value = entry;
    node->next = NULL;

    if (queue->tail != NULL) {
        queue->tail->next = node;
    } else {
        queue->ll = node;
    }

    queue->tail = node;

    return 0;
}

static void *queue_dequeue(struct ptd_queue *queue) {
    if (queue == NULL || queue->ll == NULL) {
        return NULL;
    }

    struct ptd_ll *node = queue->ll;
    void *value = node->value;

    queue->ll = node->next;

    if (queue->tail == node) {
        queue->tail = NULL;
    }

    free(node);

    return value;
}

static int queue_empty(struct ptd_queue *queue) {
    if (queue == NULL) return 1;
    return (queue->tail == NULL) ? 1 : 0;
}

// Stack implementation (using linked list)
static struct ptd_stack *stack_create() {
    struct ptd_stack *stack = (struct ptd_stack *)calloc(1, sizeof(struct ptd_stack));
    if (stack == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate stack");
        return NULL;
    }
    return stack;
}

static void stack_destroy(struct ptd_stack *stack) {
    if (stack == NULL) return;

    struct ptd_ll *current = stack->ll;
    while (current != NULL) {
        struct ptd_ll *next = current->next;
        free(current);
        current = next;
    }
    free(stack);
}

static int stack_push(struct ptd_stack *stack, void *entry) {
    if (stack == NULL) return -1;

    struct ptd_ll *node = (struct ptd_ll *)malloc(sizeof(struct ptd_ll));
    if (node == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate stack node");
        return -1;
    }

    node->value = entry;
    node->next = stack->ll;
    stack->ll = node;

    return 0;
}

static void *stack_pop(struct ptd_stack *stack) {
    if (stack == NULL || stack->ll == NULL) {
        return NULL;
    }

    struct ptd_ll *node = stack->ll;
    void *value = node->value;

    stack->ll = node->next;
    free(node);

    return value;
}

static int stack_empty(struct ptd_stack *stack) {
    if (stack == NULL) return 1;
    return (stack->ll == NULL) ? 1 : 0;
}

// ============================================================================
// Phase 2: Cache I/O Functions
// ============================================================================

/**
 * Get path to cache directory (~/.phasic_cache/traces/)
 * Creates directory if it doesn't exist
 *
 * @param buffer Output buffer for path (should be at least PATH_MAX bytes)
 * @return 0 on success, -1 on error
 */
static int get_cache_dir(char *buffer, size_t buffer_size) {
    char parent_dir[PATH_MAX];
    if (ptd_cache_root_dir(parent_dir, sizeof(parent_dir)) != 0) {
        return -1;  /* ptd_err set by helper */
    }
    // Build path: <root>/traces
    int ret = snprintf(buffer, buffer_size, "%s/traces", parent_dir);
    if (ret < 0 || (size_t)ret >= buffer_size) {
        sprintf((char*)ptd_err, "Cache directory path too long");
        return -1;
    }

    // Create parent directory
    struct stat st = {0};
    if (stat(parent_dir, &st) == -1) {
        if (mkdir(parent_dir, 0755) == -1) {
            sprintf((char*)ptd_err, "Failed to create cache directory: %s", parent_dir);
            return -1;
        }
    }

    // Create traces subdirectory
    if (stat(buffer, &st) == -1) {
        if (mkdir(buffer, 0755) == -1) {
            sprintf((char*)ptd_err, "Failed to create traces directory: %s", buffer);
            return -1;
        }
    }

    return 0;
}

/**
 * Serialize elimination trace to JSON string
 *
 * @param trace Trace to serialize
 * @return JSON string (caller must free), or NULL on error
 */
static char *trace_to_json(const struct ptd_elimination_trace *trace) {
    if (trace == NULL) {
        sprintf((char*)ptd_err, "trace is NULL");
        return NULL;
    }

    // Start with large buffer (will grow as needed)
    size_t capacity = 8192;
    size_t length = 0;
    char *json = (char *)malloc(capacity);
    if (json == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate JSON buffer");
        return NULL;
    }

    // Helper macro to append to buffer with automatic growth
    #define APPEND(fmt, ...) do { \
        while (length + 1024 > capacity) { \
            capacity *= 2; \
            char *new_json = (char *)realloc(json, capacity); \
            if (new_json == NULL) { \
                free(json); \
                sprintf((char*)ptd_err, "Failed to grow JSON buffer"); \
                return NULL; \
            } \
            json = new_json; \
        } \
        int written = snprintf(json + length, capacity - length, fmt, ##__VA_ARGS__); \
        if (written < 0) { \
            free(json); \
            sprintf((char*)ptd_err, "snprintf failed"); \
            return NULL; \
        } \
        length += written; \
    } while(0)

    // Start JSON object
    APPEND("{");
    APPEND("\"n_vertices\":%zu,", trace->n_vertices);
    APPEND("\"param_length\":%zu,", trace->param_length);
    APPEND("\"state_length\":%zu,", trace->state_length);
    APPEND("\"starting_vertex_idx\":%zu,", trace->starting_vertex_idx);
    APPEND("\"is_discrete\":%s,", trace->is_discrete ? "true" : "false");
    APPEND("\"operations_length\":%zu,", trace->operations_length);

    // Operations array
    APPEND("\"operations\":[");
    for (size_t i = 0; i < trace->operations_length; i++) {
        if (i > 0) APPEND(",");

        struct ptd_trace_operation *op = &trace->operations[i];
        APPEND("{");
        APPEND("\"op_type\":%d,", op->op_type);
        APPEND("\"const_value\":%.17g,", op->const_value);
        APPEND("\"param_idx\":%zu,", op->param_idx);

        // Coefficients array
        APPEND("\"coefficients\":[");
        for (size_t j = 0; j < op->coefficients_length; j++) {
            if (j > 0) APPEND(",");
            APPEND("%.17g", op->coefficients[j]);
        }
        APPEND("],");

        // Operands array
        APPEND("\"operands\":[");
        for (size_t j = 0; j < op->operands_length; j++) {
            if (j > 0) APPEND(",");
            APPEND("%zu", op->operands[j]);
        }
        APPEND("]");

        APPEND("}");
    }
    APPEND("],");

    // Vertex rates
    APPEND("\"vertex_rates\":[");
    for (size_t i = 0; i < trace->n_vertices; i++) {
        if (i > 0) APPEND(",");
        APPEND("%zu", trace->vertex_rates[i]);
    }
    APPEND("],");

    // Edge probs lengths
    APPEND("\"edge_probs_lengths\":[");
    for (size_t i = 0; i < trace->n_vertices; i++) {
        if (i > 0) APPEND(",");
        APPEND("%zu", trace->edge_probs_lengths[i]);
    }
    APPEND("],");

    // Edge probs (2D array)
    APPEND("\"edge_probs\":[");
    for (size_t i = 0; i < trace->n_vertices; i++) {
        if (i > 0) APPEND(",");
        APPEND("[");
        for (size_t j = 0; j < trace->edge_probs_lengths[i]; j++) {
            if (j > 0) APPEND(",");
            APPEND("%zu", trace->edge_probs[i][j]);
        }
        APPEND("]");
    }
    APPEND("],");

    // Vertex targets lengths
    APPEND("\"vertex_targets_lengths\":[");
    for (size_t i = 0; i < trace->n_vertices; i++) {
        if (i > 0) APPEND(",");
        APPEND("%zu", trace->vertex_targets_lengths[i]);
    }
    APPEND("],");

    // Vertex targets (2D array)
    APPEND("\"vertex_targets\":[");
    for (size_t i = 0; i < trace->n_vertices; i++) {
        if (i > 0) APPEND(",");
        APPEND("[");
        for (size_t j = 0; j < trace->vertex_targets_lengths[i]; j++) {
            if (j > 0) APPEND(",");
            APPEND("%zu", trace->vertex_targets[i][j]);
        }
        APPEND("]");
    }
    APPEND("],");

    // States (2D array)
    APPEND("\"states\":[");
    for (size_t i = 0; i < trace->n_vertices; i++) {
        if (i > 0) APPEND(",");
        APPEND("[");
        for (size_t j = 0; j < trace->state_length; j++) {
            if (j > 0) APPEND(",");
            APPEND("%d", trace->states[i][j]);
        }
        APPEND("]");
    }
    APPEND("]");

    APPEND("}");

    #undef APPEND
    return json;
}

/**
 * Helper: Skip whitespace in JSON string
 */
static const char *skip_whitespace(const char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') s++;
    return s;
}

/**
 * Helper: Find JSON field by name
 * Returns pointer to value (after ":"), or NULL if not found
 */
static const char *find_field(const char *json, const char *field_name) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\":", field_name);
    const char *field = strstr(json, search);
    if (field == NULL) return NULL;
    field += strlen(search);
    return skip_whitespace(field);
}

/**
 * Helper: Parse size_t from JSON
 */
static size_t parse_size_t(const char *s) {
    return (size_t)strtoull(s, NULL, 10);
}

/**
 * Helper: Parse double from JSON
 */
static double parse_double(const char *s) {
    return strtod(s, NULL);
}

/**
 * Helper: Parse int from JSON
 */
static int parse_int(const char *s) {
    return (int)strtol(s, NULL, 10);
}

/**
 * Helper: Parse bool from JSON
 */
static bool parse_bool(const char *s) {
    s = skip_whitespace(s);
    return (strncmp(s, "true", 4) == 0);
}

/**
 * Helper: Parse array of size_t from JSON
 * Format: [1,2,3]
 *
 * @param json Pointer to start of array (should be '[')
 * @param out_length Output: number of elements parsed
 * @return Array of size_t (caller must free), or NULL if empty array
 */
static size_t *parse_size_t_array(const char *json, size_t *out_length) {
    json = skip_whitespace(json);
    if (*json != '[') {
        sprintf((char*)ptd_err, "Expected '[' at start of array");
        return NULL;
    }

    // Count elements
    size_t count = 0;
    const char *p = json + 1;
    p = skip_whitespace(p);
    if (*p == ']') {
        *out_length = 0;
        return NULL;  // Empty array
    }

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
        sprintf((char*)ptd_err, "Failed to allocate array");
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

/**
 * Helper: Parse array of doubles from JSON
 * Format: [1.5,2.3,3.7]
 */
static double *parse_double_array(const char *json, size_t *out_length) {
    json = skip_whitespace(json);
    if (*json != '[') {
        sprintf((char*)ptd_err, "Expected '[' at start of array");
        return NULL;
    }

    // Count elements
    size_t count = 0;
    const char *p = json + 1;
    p = skip_whitespace(p);
    if (*p == ']') {
        *out_length = 0;
        return NULL;  // Empty array
    }

    while (*p && *p != ']') {
        // Look for numbers (including negative and decimal)
        if (*p == '-' || *p == '+' || (*p >= '0' && *p <= '9')) {
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
    double *arr = (double *)malloc(count * sizeof(double));
    if (arr == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate double array");
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

/**
 * Helper: Parse array of ints from JSON
 * Format: [1,2,3]
 */
static int *parse_int_array(const char *json, size_t *out_length) {
    json = skip_whitespace(json);
    if (*json != '[') {
        sprintf((char*)ptd_err, "Expected '[' at start of array");
        return NULL;
    }

    // Count elements
    size_t count = 0;
    const char *p = json + 1;
    p = skip_whitespace(p);
    if (*p == ']') {
        *out_length = 0;
        return NULL;  // Empty array
    }

    while (*p && *p != ']') {
        if (*p == '-' || (*p >= '0' && *p <= '9')) {
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
    int *arr = (int *)malloc(count * sizeof(int));
    if (arr == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate int array");
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

/**
 * Deserialize JSON string to elimination trace
 *
 * @param json JSON string
 * @return Trace structure (caller must call ptd_elimination_trace_destroy), or NULL on error
 */
static struct ptd_elimination_trace *json_to_trace(const char *json) {
    if (json == NULL) {
        sprintf((char*)ptd_err, "json is NULL");
        return NULL;
    }

    struct ptd_elimination_trace *trace = (struct ptd_elimination_trace *)calloc(1, sizeof(*trace));
    if (trace == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate trace");
        return NULL;
    }

    // Parse metadata fields
    const char *field;
    const char *op_start;  // Move declaration here to avoid crossing initialization

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

    field = find_field(json, "operations_length");
    if (field == NULL) goto error;
    trace->operations_length = parse_size_t(field);

    // Parse operations array
    field = find_field(json, "operations");
    if (field == NULL) goto error;

    trace->operations = (struct ptd_trace_operation *)calloc(trace->operations_length, sizeof(struct ptd_trace_operation));
    if (trace->operations == NULL) goto error;

    // Parse each operation (simplified - assumes well-formed JSON)
    op_start = field;  // Use already declared variable
    if (*op_start != '[') goto error;
    op_start++;

    for (size_t i = 0; i < trace->operations_length; i++) {
        // Find operation object
        op_start = skip_whitespace(op_start);
        if (*op_start != '{') goto error;

        // Parse op_type
        const char *op_type_field = strstr(op_start, "\"op_type\":");
        if (op_type_field == NULL) goto error;
        op_type_field += strlen("\"op_type\":");
        trace->operations[i].op_type = (enum ptd_trace_op_type)parse_int(op_type_field);

        // Parse const_value
        const char *const_val_field = strstr(op_start, "\"const_value\":");
        if (const_val_field == NULL) goto error;
        const_val_field += strlen("\"const_value\":");
        trace->operations[i].const_value = parse_double(const_val_field);

        // Parse param_idx
        const char *param_idx_field = strstr(op_start, "\"param_idx\":");
        if (param_idx_field == NULL) goto error;
        param_idx_field += strlen("\"param_idx\":");
        trace->operations[i].param_idx = parse_size_t(param_idx_field);

        // Parse coefficients array
        const char *coeffs_field = strstr(op_start, "\"coefficients\":");
        if (coeffs_field == NULL) goto error;
        coeffs_field += strlen("\"coefficients\":");
        trace->operations[i].coefficients = parse_double_array(coeffs_field,
                                                               &trace->operations[i].coefficients_length);

        // Parse operands array
        const char *operands_field = strstr(op_start, "\"operands\":");
        if (operands_field == NULL) goto error;
        operands_field += strlen("\"operands\":");
        trace->operations[i].operands = parse_size_t_array(operands_field,
                                                            &trace->operations[i].operands_length);

        // Move to next operation
        op_start = strchr(op_start, '}');
        if (op_start == NULL) goto error;
        op_start++;
        op_start = skip_whitespace(op_start);
        if (*op_start == ',') op_start++;
    }

    // Parse vertex_rates
    field = find_field(json, "vertex_rates");
    if (field == NULL) goto error;
    size_t vr_len;
    trace->vertex_rates = parse_size_t_array(field, &vr_len);

    // Parse edge_probs_lengths
    field = find_field(json, "edge_probs_lengths");
    if (field == NULL) goto error;
    size_t epl_len;
    trace->edge_probs_lengths = parse_size_t_array(field, &epl_len);

    // Parse vertex_targets_lengths
    field = find_field(json, "vertex_targets_lengths");
    if (field == NULL) goto error;
    size_t vtl_len;
    trace->vertex_targets_lengths = parse_size_t_array(field, &vtl_len);

    // Allocate 2D arrays
    trace->edge_probs = (size_t **)calloc(trace->n_vertices, sizeof(size_t *));
    trace->vertex_targets = (size_t **)calloc(trace->n_vertices, sizeof(size_t *));
    trace->states = (int **)calloc(trace->n_vertices, sizeof(int *));

    // Parse edge_probs (2D array)
    field = find_field(json, "edge_probs");
    if (field == NULL) goto error;
    field = skip_whitespace(field);
    if (*field != '[') goto error;
    field++;

    for (size_t i = 0; i < trace->n_vertices; i++) {
        field = skip_whitespace(field);
        size_t len;
        trace->edge_probs[i] = parse_size_t_array(field, &len);
        // Skip to next sub-array
        while (*field && *field != ']') field++;
        if (*field == ']') field++;
        field = skip_whitespace(field);
        if (*field == ',') field++;
    }

    // Parse vertex_targets (2D array)
    field = find_field(json, "vertex_targets");
    if (field == NULL) goto error;
    field = skip_whitespace(field);
    if (*field != '[') goto error;
    field++;

    for (size_t i = 0; i < trace->n_vertices; i++) {
        field = skip_whitespace(field);
        size_t len;
        trace->vertex_targets[i] = parse_size_t_array(field, &len);
        // Skip to next sub-array
        while (*field && *field != ']') field++;
        if (*field == ']') field++;
        field = skip_whitespace(field);
        if (*field == ',') field++;
    }

    // Parse states (2D array of ints)
    field = find_field(json, "states");
    if (field == NULL) goto error;
    field = skip_whitespace(field);
    if (*field != '[') goto error;
    field++;

    for (size_t i = 0; i < trace->n_vertices; i++) {
        field = skip_whitespace(field);
        size_t len;
        trace->states[i] = parse_int_array(field, &len);
        // Skip to next sub-array
        while (*field && *field != ']') field++;
        if (*field == ']') field++;
        field = skip_whitespace(field);
        if (*field == ',') field++;
    }

    return trace;

error:
    sprintf((char*)ptd_err, "Failed to parse JSON trace");
    ptd_elimination_trace_destroy(trace);
    return NULL;
}

/**
 * Load elimination trace from cache
 *
 * @param hash_hex Hash of graph structure (hex string)
 * @return Trace if found in cache, NULL otherwise
 */
struct ptd_elimination_trace *ptd_load_trace_from_cache(const char *hash_hex) {
    if (hash_hex == NULL) return NULL;

    // Cache is off by default (see ptd_pcg_cache_disabled at line ~3231).
    // Only the positive opt-in PHASIC_REWARD_COMPUTE_CACHE="1" enables.
    {
        const char *enable = getenv("PHASIC_REWARD_COMPUTE_CACHE");
        if (enable == NULL || strcmp(enable, "1") != 0) {
            return NULL;  // Cache disabled (default policy)
        }
    }

    // Get cache directory
    char cache_dir[PATH_MAX];
    if (get_cache_dir(cache_dir, sizeof(cache_dir)) != 0) {
        PTD_LOG_WARNING("Cache directory unavailable");
        return NULL;  // Cache directory unavailable
    }

    // Build cache file path
    char cache_file[PATH_MAX];
    snprintf(cache_file, sizeof(cache_file), "%s/%s.json", cache_dir, hash_hex);

    // Check if file exists
    FILE *f = fopen(cache_file, "r");
    if (f == NULL) {
        return NULL;  // Cache miss
    }

    // Read file into buffer
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0 || file_size > 100*1024*1024) {  // Max 100MB
        fclose(f);
        return NULL;
    }

    char *json = (char *)malloc(file_size + 1);
    if (json == NULL) {
        fclose(f);
        return NULL;
    }

    size_t read = fread(json, 1, file_size, f);
    fclose(f);

    if ((long)read != file_size) {
        free(json);
        return NULL;
    }

    json[file_size] = '\0';

    // Deserialize
    struct ptd_elimination_trace *trace = json_to_trace(json);
    free(json);

    return trace;
}

/**
 * Save elimination trace to cache
 *
 * @param hash_hex Hash of graph structure (hex string)
 * @param trace Trace to save
 * @return true on success, false on error
 */
bool ptd_save_trace_to_cache(const char *hash_hex, const struct ptd_elimination_trace *trace) {
    if (hash_hex == NULL || trace == NULL) return false;

    // Cache is off by default; only the positive opt-in
    // PHASIC_REWARD_COMPUTE_CACHE="1" enables writes.
    {
        const char *enable = getenv("PHASIC_REWARD_COMPUTE_CACHE");
        if (enable == NULL || strcmp(enable, "1") != 0) {
            return false;
        }
    }

    // Get cache directory
    char cache_dir[PATH_MAX];
    if (get_cache_dir(cache_dir, sizeof(cache_dir)) != 0) {
        PTD_LOG_WARNING("Cache directory unavailable, cannot save trace");
        return false;  // Cache unavailable
    }

    // Build cache file path
    char cache_file[PATH_MAX];
    snprintf(cache_file, sizeof(cache_file), "%s/%s.json", cache_dir, hash_hex);

    // Serialize to JSON
    char *json = trace_to_json(trace);
    if (json == NULL) {
        return false;
    }

    // Write to file
    FILE *f = fopen(cache_file, "w");
    if (f == NULL) {
        free(json);
        return false;
    }

    size_t len = strlen(json);
    size_t written = fwrite(json, 1, len, f);
    fclose(f);
    free(json);

    return (written == len);
}

// Stub for ptd_clone_graph (missing implementation)
/**
 * Deep clone a graph with all vertices, edges, and coefficient arrays
 * Supports unified edge interface (all edges have coefficient arrays)
 */
struct ptd_clone_res ptd_clone_graph(struct ptd_graph *graph, struct ptd_avl_tree *avl_tree) {
    struct ptd_clone_res res;
    res.graph = NULL;
    res.avl_tree = NULL;

    if (graph == NULL) {
        snprintf((char*)ptd_err, sizeof(ptd_err), "Cannot clone NULL graph");
        return res;
    }

    // Create new graph with same state_length
    struct ptd_graph *new_graph = ptd_graph_create(graph->state_length);
    if (new_graph == NULL) {
        snprintf((char*)ptd_err, sizeof(ptd_err), "Failed to allocate new graph");
        return res;
    }

    // Copy metadata
    new_graph->param_length = graph->param_length;
    new_graph->parameterized = graph->parameterized;
    new_graph->param_length_locked = graph->param_length_locked;
    new_graph->edge_mode = graph->edge_mode;
    new_graph->was_dph = graph->was_dph;
    new_graph->dph_compute_invalidated = graph->dph_compute_invalidated;

    // Create mapping from old vertices to new vertices
    struct ptd_vertex **vertex_map = (struct ptd_vertex **)calloc(
        graph->vertices_length, sizeof(struct ptd_vertex *)
    );
    if (vertex_map == NULL) {
        ptd_graph_destroy(new_graph);
        snprintf((char*)ptd_err, sizeof(ptd_err), "Failed to allocate vertex map");
        return res;
    }

    // Clone all vertices (just structure, not edges yet)
    // Note: Starting vertex may be in vertices array, so we must check for it
    // to avoid creating a duplicate
    for (size_t i = 0; i < graph->vertices_length; i++) {
        struct ptd_vertex *old_v = graph->vertices[i];

        // Check if this is the starting vertex by comparing pointer
        if (old_v == graph->starting_vertex) {
            // This is the starting vertex - map to new starting vertex (no duplicate)
            // But we need to copy its state array!
            int *new_state = (int *)malloc(graph->state_length * sizeof(int));
            if (new_state == NULL) {
                free(vertex_map);
                ptd_graph_destroy(new_graph);
                snprintf((char*)ptd_err, sizeof(ptd_err), "Failed to allocate state for starting vertex");
                return res;
            }
            memcpy(new_state, old_v->state, graph->state_length * sizeof(int));
            free(new_graph->starting_vertex->state);  // Free the default state
            new_graph->starting_vertex->state = new_state;
            vertex_map[i] = new_graph->starting_vertex;
        } else {
            // Create vertex with COPIED state (not shared pointer)
            int *new_state = (int *)malloc(graph->state_length * sizeof(int));
            if (new_state == NULL) {
                free(vertex_map);
                ptd_graph_destroy(new_graph);
                snprintf((char*)ptd_err, sizeof(ptd_err), "Failed to allocate state for vertex %zu", i);
                return res;
            }
            memcpy(new_state, old_v->state, graph->state_length * sizeof(int));

            struct ptd_vertex *new_v = ptd_vertex_create_state(new_graph, new_state);
            if (new_v == NULL) {
                free(new_state);
                free(vertex_map);
                ptd_graph_destroy(new_graph);
                snprintf((char*)ptd_err, sizeof(ptd_err), "Failed to clone vertex %zu", i);
                return res;
            }

            // Free the temporary state since ptd_vertex_create_state() makes a copy
            free(new_state);

            new_v->is_aux = old_v->is_aux;

            vertex_map[i] = new_v;
        }
    }

    // Clone starting vertex edges
    struct ptd_vertex *old_start = graph->starting_vertex;
    struct ptd_vertex *new_start = new_graph->starting_vertex;

    for (size_t i = 0; i < old_start->edges_length; i++) {
        struct ptd_edge *old_edge = old_start->edges[i];

        // Find target vertex index in old graph
        size_t target_idx = (size_t)-1;
        for (size_t j = 0; j < graph->vertices_length; j++) {
            if (graph->vertices[j] == old_edge->to) {
                target_idx = j;
                break;
            }
        }

        if (target_idx == (size_t)-1) {
            free(vertex_map);
            ptd_graph_destroy(new_graph);
            snprintf((char*)ptd_err, sizeof(ptd_err), "Starting vertex edge target not found");
            return res;
        }

        struct ptd_vertex *new_target = vertex_map[target_idx];

        // Clone starting vertex edge directly (bypass validation for IPV edges)
        // IPV edges may have different coefficient lengths than regular edges
        struct ptd_edge *new_edge = (struct ptd_edge *)malloc(sizeof(*new_edge));
        if (new_edge == NULL) {
            free(vertex_map);
            ptd_graph_destroy(new_graph);
            snprintf((char*)ptd_err, sizeof(ptd_err), "Failed to allocate edge");
            return res;
        }

        new_edge->to = new_target;
        new_edge->weight = old_edge->weight;
        new_edge->coefficients_length = old_edge->coefficients_length;

        if (old_edge->coefficients != NULL && old_edge->coefficients_length > 0) {
            // Clone coefficients array
            new_edge->coefficients = (double *)malloc(old_edge->coefficients_length * sizeof(double));
            if (new_edge->coefficients == NULL) {
                free(new_edge);
                free(vertex_map);
                ptd_graph_destroy(new_graph);
                snprintf((char*)ptd_err, sizeof(ptd_err), "Failed to allocate edge coefficients");
                return res;
            }
            memcpy(new_edge->coefficients, old_edge->coefficients, old_edge->coefficients_length * sizeof(double));
            new_edge->should_free_coefficients = true;
        } else {
            // No coefficients (shouldn't happen with unified interface, but handle it)
            new_edge->coefficients = NULL;
            new_edge->should_free_coefficients = false;
        }

        // Add edge to starting vertex's edge list
        struct ptd_edge **new_edges = (struct ptd_edge **)realloc(
            new_start->edges,
            (new_start->edges_length + 1) * sizeof(struct ptd_edge *)
        );
        if (new_edges == NULL) {
            if (new_edge->should_free_coefficients) free(new_edge->coefficients);
            free(new_edge);
            free(vertex_map);
            ptd_graph_destroy(new_graph);
            snprintf((char*)ptd_err, sizeof(ptd_err), "Failed to resize edges array");
            return res;
        }
        new_start->edges = new_edges;
        new_start->edges[new_start->edges_length] = new_edge;
        new_start->edges_length++;
    }

    // Clone all edges (skip starting vertex - already cloned above)
    for (size_t i = 0; i < graph->vertices_length; i++) {
        struct ptd_vertex *old_v = graph->vertices[i];
        struct ptd_vertex *new_v = vertex_map[i];

        // Skip starting vertex - its edges were already cloned
        if (old_v == graph->starting_vertex) {
            continue;
        }

        for (size_t j = 0; j < old_v->edges_length; j++) {
            struct ptd_edge *old_edge = old_v->edges[j];

            // Find target vertex index
            size_t target_idx = (size_t)-1;
            for (size_t k = 0; k < graph->vertices_length; k++) {
                if (graph->vertices[k] == old_edge->to) {
                    target_idx = k;
                    break;
                }
            }

            if (target_idx == (size_t)-1) {
                free(vertex_map);
                ptd_graph_destroy(new_graph);
                snprintf((char*)ptd_err, sizeof(ptd_err), "Edge target not found at vertex %zu", i);
                return res;
            }

            struct ptd_vertex *new_target = vertex_map[target_idx];

            /* Constant edges (coefficients_length == 0, coefficients == NULL)
             * are hand-rolled by helpers like phasic::Vertex::add_aux_vertex
             * and bypass ptd_graph_add_edge's validation. The clone path
             * must mirror that: ptd_graph_add_edge rejects NULL/zero-length
             * coefficients up front, so we hand-roll the constant edge
             * directly. Symmetric to the starting-vertex IPV-edge clone
             * loop above. */
            if (old_edge->coefficients == NULL || old_edge->coefficients_length == 0) {
                struct ptd_edge *new_edge = (struct ptd_edge *)malloc(sizeof(*new_edge));
                if (new_edge == NULL) {
                    free(vertex_map);
                    ptd_graph_destroy(new_graph);
                    snprintf((char*)ptd_err, sizeof(ptd_err), "Failed to allocate constant edge at vertex %zu", i);
                    return res;
                }
                new_edge->to = new_target;
                new_edge->weight = old_edge->weight;
                new_edge->coefficients_length = 0;
                new_edge->coefficients = NULL;
                new_edge->should_free_coefficients = false;

                struct ptd_edge **new_edges = (struct ptd_edge **)realloc(
                    new_v->edges,
                    (new_v->edges_length + 1) * sizeof(struct ptd_edge *)
                );
                if (new_edges == NULL) {
                    free(new_edge);
                    free(vertex_map);
                    ptd_graph_destroy(new_graph);
                    snprintf((char*)ptd_err, sizeof(ptd_err), "Failed to resize edges array at vertex %zu", i);
                    return res;
                }
                new_v->edges = new_edges;
                new_v->edges[new_v->edges_length] = new_edge;
                new_v->edges_length++;
                continue;
            }

            // Add edge with cloned coefficients array
            struct ptd_edge *new_edge = ptd_graph_add_edge(
                new_v, new_target,
                old_edge->coefficients,
                old_edge->coefficients_length
            );

            if (new_edge == NULL) {
                free(vertex_map);
                ptd_graph_destroy(new_graph);
                snprintf((char*)ptd_err, sizeof(ptd_err), "Failed to clone edge at vertex %zu", i);
                return res;
            }

            /* ptd_graph_add_edge initialises edge->weight as
             * sum(coefficients * 1) (default theta=1). The source edge
             * may have been updated since (via update_weights), so its
             * current weight differs from the default. Copy the
             * source's current weight so the clone faithfully
             * reproduces the source's runtime state — required by
             * callers like ptd_graph_reward_transform that read
             * edge->weight for SCC normalisation. */
            new_edge->weight = old_edge->weight;
        }
    }

    free(vertex_map);

    // Create new AVL tree for new graph
    struct ptd_avl_tree *new_avl = ptd_avl_tree_create(graph->state_length);
    if (new_avl == NULL) {
        ptd_graph_destroy(new_graph);
        snprintf((char*)ptd_err, sizeof(ptd_err), "Failed to create AVL tree for cloned graph");
        return res;
    }

    // Rebuild AVL tree with new vertices
    for (size_t i = 0; i < new_graph->vertices_length; i++) {
        struct ptd_vertex *v = new_graph->vertices[i];
        ptd_avl_tree_find_or_insert(new_avl, v->state, v);
    }

    res.graph = new_graph;
    res.avl_tree = new_avl;
    return res;
}

// Stub for ptd_defect (commented out in original code)
double ptd_defect(struct ptd_graph *graph) {
    // Simple stub implementation
    return 0.0;
}

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

// Forward declarations for dynamic ordering variants
struct ptd_desc_reward_compute *ptd_graph_ex_absorbation_time_comp_graph_dyn(struct ptd_graph *graph);
struct ptd_desc_reward_compute_parameterized *ptd_graph_ex_absorbation_time_comp_graph_parameterized_dyn(struct ptd_graph *graph);

// Forward declarations for Stage A2 cache helpers (defined further down).
static int ptd_pcg_cache_disabled(void);
static int ptd_pcg_build_cache_path(
        const struct ptd_graph *graph, char *buf, size_t buf_len);

/* --- B1 (zero-copy plan): offset/index form of the parameterized PRC.
 * Dual-form: the raw-pointer build+executor stay untouched (default path);
 * this offset form + executor run only on loaded PRCs (and an env-gated
 * self-check). Structs use fixed-width + natural alignment so B3 can cast
 * them directly from an mmap. EDGE and EXTERNAL operands both collapse to a
 * single `inputs[]` indirection bound once. See zero-copy-cache-plan.md. */
enum ptd_pcg_op_kind { PTD_PCG_OP_NULL = 0, PTD_PCG_OP_MEM = 1, PTD_PCG_OP_INPUT = 2 };
struct ptd_pcg_operand {
    int64_t  mem_offset;   /* PTD_PCG_OP_MEM: doubles offset into mem_base */
    uint32_t input_idx;    /* PTD_PCG_OP_INPUT: index into inputs[] */
    uint8_t  kind;         /* enum ptd_pcg_op_kind */
    uint8_t  pad[3];
};
struct ptd_pcg_command_off {
    int32_t  type;
    uint32_t pad;
    uint64_t from;
    uint64_t to;
    double   multiplier;
    struct ptd_pcg_operand fromT;        /* always MEM (write target) */
    struct ptd_pcg_operand toT;
    struct ptd_pcg_operand multiplierptr;
};
/* One inputs[] binding spec: which live double an input slot resolves to.
 * Carried in the descriptor so the rev-3 save can serialize it and the load
 * can re-bind inputs[] against the current graph. */
struct ptd_pcg_input_spec { uint8_t kind; uint32_t v; uint32_t e; int64_t byte; };
struct ptd_desc_reward_compute_parameterized_off {
    size_t length;
    struct ptd_pcg_command_off *commands;
    double *mem_base;      /* flat scratch+const doubles (writable; COW under mmap) */
    size_t mem_doubles;
    int mem_is_mmap;       /* 0 = heap (free), 1 = mmap (munmap) — set by B3 loader */
    double **inputs;       /* inputs[k] -> live &edge->weight (+byte) or external coeff */
    size_t n_inputs;
    struct ptd_pcg_input_spec *input_specs;  /* n_inputs entries; for save/re-bind (heap-load only) */
    void *mmap_base;       /* B3: mmap'd file base (mem_is_mmap=1); else NULL */
    size_t mmap_len;       /* B3: mapping length for munmap */
};
static struct ptd_desc_reward_compute_parameterized_off *ptd_pcg_convert_to_offset(
        const struct ptd_desc_reward_compute_parameterized *raw,
        const struct ptd_graph *graph,
        const double *const *external_anchors, size_t n_external);
static struct ptd_desc_reward_compute *
ptd_graph_build_ex_absorbation_time_comp_graph_parameterized_off(
        const struct ptd_desc_reward_compute_parameterized_off *off);
static void ptd_pcg_desc_off_destroy(
        struct ptd_desc_reward_compute_parameterized_off *off);
/* ptd_save_pcg_rev3 / ptd_load_pcg_rev3 are declared in phasic.h (non-static)
 * so scc_synthetic.c shares the rev-3 format. */

int ptd_precompute_reward_compute_graph(struct ptd_graph *graph) {
    /* Take the per-graph mutex unconditionally. A double-checked-lock
     * fast path on graph->reward_compute_graph would need acquire/release
     * memory ordering on the read/store to be safe on weakly-ordered
     * architectures (arm64) — phasic doesn't otherwise use C atomics, so
     * we keep the simpler always-lock variant. An uncontended pthread
     * mutex on macOS / Linux is ~10–20ns; the surrounding PDF/moments
     * computation is several orders of magnitude more expensive, so the
     * lock is in the noise.
     *
     * The flag check guards against the unlikely case where
     * pthread_mutex_init failed during graph creation; in that case we
     * fall through to the lock-free path which is correct under
     * single-threaded use. */
    if (graph->compute_graph_lock_initialized) {
        PTD_MUTEX_LOCK(&graph->compute_graph_lock);
    }

    if (graph->dph_compute_invalidated) {
        // One-shot wipe: drops any compute graphs that were built against
        // the pre-discretization topology (set_was_dph(true) flips this
        // flag, this branch clears it). Originally the wipe was gated on
        // was_dph and was_dph was reset to false after the first call;
        // making was_dph permanent (so update_weights keeps normalising)
        // would have turned the wipe into a per-call O(n^3) rebuild,
        // which dominated SVGD wall time on discrete joint-prob graphs.

        if (graph->reward_compute_graph != NULL) {
            free(graph->reward_compute_graph->commands);
            free(graph->reward_compute_graph);
        }

        if (graph->parameterized_reward_compute_graph != NULL) {
            ptd_parameterized_reward_compute_graph_destroy(
                    graph->parameterized_reward_compute_graph
            );
        }
        if (graph->parameterized_reward_compute_graph_off != NULL) {
            ptd_pcg_desc_off_destroy(graph->parameterized_reward_compute_graph_off);
        }

        graph->reward_compute_graph = NULL;
        graph->parameterized_reward_compute_graph = NULL;
        graph->parameterized_reward_compute_graph_off = NULL;
        graph->dph_compute_invalidated = false;
    }

    if (graph->reward_compute_graph == NULL) {
        if (graph->parameterized) {
            if (graph->parameterized_reward_compute_graph == NULL
                    && graph->parameterized_reward_compute_graph_off == NULL) {
                /* Stage A2: try the on-disk symbolic-elimination cache
                 * first. The cache is theta-independent (Stage A0
                 * showed multiplierptr is dereferenced at replay
                 * time), so it's keyed only on graph topology +
                 * coefficients via ptd_graph_content_hash. On a hit,
                 * we save an O(n^3) Gaussian elimination per fresh
                 * process. On a miss, the elimination runs as before
                 * and we populate the cache. Off by default; opt in
                 * via phasic.configure(reward_compute_cache=True),
                 * backed by PHASIC_REWARD_COMPUTE_CACHE=1. */
                int cache_used = 0;
                if (!ptd_pcg_cache_disabled()) {
                    char cache_path[PATH_MAX];
                    if (ptd_pcg_build_cache_path(graph, cache_path,
                                                 sizeof(cache_path)) == 0) {
                        struct ptd_desc_reward_compute_parameterized_off *loaded =
                                ptd_load_pcg_rev3(cache_path, graph);
                        /* NULL = cache miss (file absent / not rev-3 / stale
                         * for this graph). Fall through to rebuild; clear
                         * ptd_err so the miss isn't seen as a real error. */
                        ptd_err[0] = '\0';
                        if (loaded != NULL) {
                            graph->parameterized_reward_compute_graph_off = loaded;
                            cache_used = 1;
                        }
                    } else {
                        /* build_cache_path failed (e.g. HOME unset).
                         * Treat as cache disabled, rebuild, and don't
                         * try to save. */
                        ptd_err[0] = '\0';
                    }
                }

                if (!cache_used) {
                    if (graph->use_dyn_ordering) {
                        graph->parameterized_reward_compute_graph =
                                ptd_graph_ex_absorbation_time_comp_graph_parameterized_dyn(graph);
                    } else {
                        graph->parameterized_reward_compute_graph =
                                ptd_graph_ex_absorbation_time_comp_graph_parameterized(graph);
                    }
                    /* Best-effort save. Failure here is non-fatal —
                     * we'll just rebuild again next process. */
                    if (!ptd_pcg_cache_disabled()
                            && graph->parameterized_reward_compute_graph != NULL) {
                        char cache_path[PATH_MAX];
                        if (ptd_pcg_build_cache_path(graph, cache_path,
                                                     sizeof(cache_path)) == 0) {
                            if (ptd_save_pcg_rev3(
                                    cache_path,
                                    graph->parameterized_reward_compute_graph,
                                    graph) != 0) {
                                /* Non-fatal: the elimination is recomputed
                                 * next process. Warn (rather than swallow
                                 * silently) so a full disk / unwritable
                                 * cache dir is diagnosable. */
                                PTD_LOG_WARNING(
                                    "reward-compute cache save failed "
                                    "(non-fatal, will recompute next run): %s",
                                    ptd_err);
                            }
                            ptd_err[0] = '\0';  /* swallow into return path */
                        } else {
                            PTD_LOG_WARNING(
                                "reward-compute cache save skipped: "
                                "could not build cache path");
                            ptd_err[0] = '\0';
                        }
                    }
                }
            }

            if (graph->reward_compute_graph != NULL) {
                free(graph->reward_compute_graph->commands);
                free(graph->reward_compute_graph);
            }

            /* B1 self-check (env-gated): convert the PRISTINE PRC to the offset
             * form BEFORE the raw executor mutates the mem chain (the raw
             * executor writes *fromT into compute->mem; flatten must capture the
             * mem state the executor STARTS from, not the post-execution state.
             * This is the same point at which the rev-2 save runs, line ~1999). */
            struct ptd_desc_reward_compute_parameterized_off *_off = NULL;
            double _cs = 0.0;
            int _sc = (getenv("PHASIC_PCG_SELFCHECK") != NULL
                       && graph->parameterized_reward_compute_graph != NULL);
            if (_sc) {
                uint64_t _c0 = monotonic_ns();
                _off = ptd_pcg_convert_to_offset(
                        graph->parameterized_reward_compute_graph, graph, NULL, 0);
                uint64_t _c1 = monotonic_ns();
                _cs = (double)(_c1 - _c0) / 1e9;
            }

            /* Dual-form executor fork: cache HIT (offset form loaded) runs the
             * offset executor; otherwise the freshly-built raw PRC. */
            if (graph->parameterized_reward_compute_graph_off != NULL) {
                graph->reward_compute_graph =
                        ptd_graph_build_ex_absorbation_time_comp_graph_parameterized_off(
                                graph->parameterized_reward_compute_graph_off);
            } else {
                graph->reward_compute_graph =
                        ptd_graph_build_ex_absorbation_time_comp_graph_parameterized(
                                graph->parameterized_reward_compute_graph);
            }
            if (_sc) {
                size_t _n = graph->parameterized_reward_compute_graph->length;
                if (_off == NULL) {
                    PTD_LOG_ERROR("PCG_SELFCHECK: convert_to_offset returned NULL "
                                  "(n_cmds=%zu)", _n);
                }
#ifdef PHASIC_B3_VALIDATORS
                else if (getenv("PHASIC_DBG_STASH_OFF") != NULL) {
                    /* B3 validator: stash the CLEAN pre-execution _off and SKIP
                     * the _oo self-comparison. Building _oo runs the _off
                     * executor, which writes *fromT into _off->mem_base in place
                     * -- that would dirty the stashed mem_base (the exact
                     * post-vs-pre execution pitfall documented at :2060), so
                     * ptd_debug_fwdmode_grad would replay from post-exec mem. */
                    if (graph->_dbg_off_clean != NULL)
                        ptd_pcg_desc_off_destroy(
                            (struct ptd_desc_reward_compute_parameterized_off *)
                                graph->_dbg_off_clean);
                    graph->_dbg_off_clean = _off;
                }
#endif
                else {
                    struct ptd_desc_reward_compute *_ro = graph->reward_compute_graph;
                    struct ptd_desc_reward_compute *_oo =
                        ptd_graph_build_ex_absorbation_time_comp_graph_parameterized_off(_off);
                    int _id = (_ro != NULL && _oo != NULL
                               && _ro->length == _oo->length);
                    if (_id) {
                        for (size_t _i = 0; _i < _ro->length; _i++) {
                            if (_ro->commands[_i].from != _oo->commands[_i].from
                                || _ro->commands[_i].to != _oo->commands[_i].to
                                || _ro->commands[_i].multiplier
                                       != _oo->commands[_i].multiplier) {
                                _id = 0; break;
                            }
                        }
                    }
                    PTD_LOG_WARNING("PCG_SELFCHECK n_cmds=%zu n_inputs=%zu "
                                    "convert_s=%.4f offset_vs_raw=%s",
                                    _n, _off->n_inputs, _cs,
                                    _id ? "IDENTICAL" : "DIFFERS");
                    if (_oo != NULL) {
                        if (_oo->commands) free(_oo->commands);
                        free(_oo);
                    }
                    ptd_pcg_desc_off_destroy(_off);
                }
            }
        } else {
            if (graph->use_dyn_ordering) {
                graph->reward_compute_graph = ptd_graph_ex_absorbation_time_comp_graph_dyn(graph);
            } else {
                graph->reward_compute_graph = ptd_graph_ex_absorbation_time_comp_graph(graph);
            }

            if (graph->reward_compute_graph == NULL) {
                if (graph->compute_graph_lock_initialized) {
                    PTD_MUTEX_UNLOCK(&graph->compute_graph_lock);
                }
                return -1;
            }
        }
    }

    if (graph->compute_graph_lock_initialized) {
        PTD_MUTEX_UNLOCK(&graph->compute_graph_lock);
    }

    return 0;
}


// SCC state structure for re-entrant Tarjan's algorithm
struct scc_state {
    struct ptd_stack *scc_stack;
    struct ptd_vector *scc_components;
    size_t scc_index;
    size_t *scc_indices;
    size_t *low_links;
    bool *scc_on_stack;
    bool *visited;
};

static int strongconnect2(struct ptd_vertex *vertex, struct scc_state *state) {
    state->scc_indices[vertex->index] = state->scc_index;
    state->low_links[vertex->index] = state->scc_index;
    state->visited[vertex->index] = true;
    state->scc_index++;
    stack_push(state->scc_stack, vertex);
    state->scc_on_stack[vertex->index] = true;

    for (size_t i = 0; i < vertex->edges_length; ++i) {
        struct ptd_edge *edge = vertex->edges[i];

        if (!state->visited[edge->to->index]) {
            int res = strongconnect2(edge->to, state);

            if (res != 0) {
                return res;
            }

            state->low_links[vertex->index] = _ptd_min(
                                                state->low_links[vertex->index],
                                                state->low_links[edge->to->index]
                                        );
        } else if (state->scc_on_stack[edge->to->index]) {
            state->low_links[vertex->index] = _ptd_min(
                                                state->low_links[vertex->index],
                                                state->scc_indices[edge->to->index]
                                        );
        }
    }

    if (state->low_links[vertex->index] == state->scc_indices[vertex->index]) {
        struct ptd_vertex *w;
        struct ptd_vector *list = vector_create();

        do {
            if (stack_empty(state->scc_stack)) {
                DIE_ERROR(1, "Stack is empty.\n");
            }
            w = (struct ptd_vertex *) stack_pop(state->scc_stack);
            state->scc_on_stack[w->index] = false;

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

        vector_add(state->scc_components, scc);
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

/**
 * Isolate the starting vertex into its own SCC at position 0.
 *
 * This ensures that:
 * 1. The starting vertex is always in SCC 0
 * 2. SCC 0 contains ONLY the starting vertex
 * 3. All other SCCs are shifted accordingly
 *
 * This simplifies trace stitching by eliminating special cases.
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

    // Allocate local SCC state (re-entrant safe)
    struct scc_state state;
    state.scc_stack = stack_create();
    state.scc_index = 0;
    state.scc_indices = (size_t *) calloc(graph->vertices_length * 10, sizeof(size_t));
    state.low_links = (size_t *) calloc(graph->vertices_length * 10, sizeof(size_t));
    state.scc_on_stack = (bool *) calloc(graph->vertices_length * 10, sizeof(bool));
    state.visited = (bool *) calloc(graph->vertices_length * 10, sizeof(bool));
    state.scc_components = vector_create();

    for (size_t i = 0; i < graph->vertices_length; ++i) {
        struct ptd_vertex *vertex = graph->vertices[i];

        if (!state.visited[i]) {
            if (strongconnect2(vertex, &state) != 0) {
                // Cleanup on error
                stack_destroy(state.scc_stack);
                vector_destroy(state.scc_components);
                free(state.scc_indices);
                free(state.low_links);
                free(state.scc_on_stack);
                free(state.visited);
                free(scc_graph);
                return NULL;
            }
        }
    }

    size_t non_empty_components = 0;

    for (size_t i = 0; i < vector_length(state.scc_components); ++i) {
        struct ptd_scc_vertex *c =
                (struct ptd_scc_vertex *) vector_get(state.scc_components, i);

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
                (struct ptd_scc_vertex *) vector_get(state.scc_components, i);

        if (scc->internal_vertices_length != 0) {
            scc_graph->vertices[index] =
                    (struct ptd_scc_vertex *) vector_get(state.scc_components, i);
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

    // Cleanup local SCC state
    free(state.scc_indices);
    free(state.low_links);
    free(state.scc_on_stack);
    free(state.visited);
    vector_destroy(state.scc_components);
    stack_destroy(state.scc_stack);

    free(sccs_for_vertices);

    // Isolate the starting vertex into its own SCC at position 0
    // This simplifies trace stitching
    if (ptd_isolate_starting_vertex_scc(scc_graph) != 0) {
        // If isolation fails, still return the graph (non-critical optimization)
        // Trace stitching will handle the general case
    }

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
            double weight1 = 1 - rate;
            double weight2 = 1.0;
            ptd_graph_add_edge(vertex, auxiliary_vertex, &weight1, 1);
            ptd_graph_add_edge(auxiliary_vertex, vertex, &weight2, 1);
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
    graph->param_length = 0;           // Set by first add_edge() call
    graph->parameterized = false;      // Set by first add_edge() call
    graph->param_length_locked = false; // Set by first add_edge() call
    graph->edge_mode = PTD_EDGE_MODE_UNLOCKED;  // Locked by first non-IPV edge
    graph->vertices = NULL;
    graph->reward_compute_graph = NULL;
    graph->parameterized_reward_compute_graph = NULL;
    graph->parameterized_reward_compute_graph_off = NULL;
#ifdef PHASIC_B3_VALIDATORS
    graph->_dbg_off_clean = NULL;  /* B3 validator only; see phasic.h */
#endif
    graph->reward_compute_graph_mpfr = NULL;
    graph->starting_vertex = ptd_vertex_create(graph);
    graph->was_dph = false;
    graph->dph_compute_invalidated = false;
    graph->use_dyn_ordering = (getenv("PHASIC_DYN_ORDERING") != NULL);
    graph->elimination_trace = NULL;
    graph->current_params = NULL;
    graph->weight_version = 0;
    graph->weight_tape = NULL;
    graph->wf_residuals = NULL;
    graph->wf_residuals_length = 0;
    graph->wf_residuals_for_tape = NULL;

    /* Initialise the per-graph compute-graph lock. The init can fail
     * (out of memory), in which case we leave the flag false; later
     * destroys are conditional on the flag and lazy builds still
     * succeed because the unprotected double-NULL-check path is
     * a correctness fallback (lossy under contention but not unsafe
     * for single-threaded use). */
    if (PTD_MUTEX_INIT(&graph->compute_graph_lock) == 0) {
        graph->compute_graph_lock_initialized = true;
    } else {
        graph->compute_graph_lock_initialized = false;
    }

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

// ===========================================================================
// Stage A2: disk-persistent symbolic compute graph cache
// ===========================================================================
//
// On-disk format for ptd_desc_reward_compute_parameterized.
//
// Each command's three pointer fields (fromT, toT, multiplierptr) get
// re-encoded as a (kind, offset, vertex_idx, edge_idx) tuple at save
// time and re-resolved at load time against the live graph.
//
// Two pointer kinds occur in practice (verified by reading the
// recorder in src/c/phasic.c):
//   - "mem":  the pointer is &mem_buffer[N] for some integer N. We
//             save N (in doubles).
//   - "edge": the pointer is &graph->vertices[v]->edges[e]->weight,
//             possibly with a small byte offset. The recorder uses
//             offset 0 (most common) or -sizeof(double) (the
//             ``weight - 1`` self-loop trick at line 4444). We save
//             (v, e, byte_offset) and reconstruct at load.

#include <fcntl.h>
#include <errno.h>
#ifndef _WIN32
#include <sys/mman.h>   /* B3: zero-copy mmap load (POSIX) */
#endif

#define PTD_PCG_MAGIC "PTDPRMC1"
#define PTD_PCG_VERSION 1u
/* Format revision history:
 *   1 — original Stage A2 format. Pointers are NULL/MEM/EDGE.
 *   2 — adds PTD_PCG_PTR_EXTERNAL for per-SCC PRCs that resolve
 *       placeholder edges against a parent-supplied external
 *       table at composition time. v2 readers handle v1 files
 *       (strict superset); v1 readers refuse v2 files. */
#define PTD_PCG_FORMAT_REVISION 2u

enum ptd_pcg_ptr_kind {
    PTD_PCG_PTR_NULL = 0,
    PTD_PCG_PTR_MEM = 1,       // payload: doubles offset into flat mem buffer
    PTD_PCG_PTR_EDGE = 2,      // payload: (vertex_idx, edge_idx, byte_offset)
    PTD_PCG_PTR_EXTERNAL = 3,  // payload: vertex_idx == external_table index (rev 2+)
};

#pragma pack(push, 1)
struct ptd_pcg_disk_header {
    char     magic[8];                  // "PTDPRMC1"
    uint32_t version;
    uint32_t format_revision;
    uint64_t graph_hash_truncated;      // first 8 bytes of SHA-256
    uint64_t commands_length;
    uint64_t mem_total_doubles;
    uint64_t memr_length;
    uint64_t reserved;                  // future use; write 0
};

struct ptd_pcg_disk_ptr {
    uint8_t  kind;
    uint8_t  pad[7];
    int64_t  doubles_offset;            // for MEM kind
    uint32_t vertex_idx;                // for EDGE kind
    uint32_t edge_idx;                  // for EDGE kind
    int64_t  byte_offset_from_edge_weight; // for EDGE kind (typically 0 or -8)
};

struct ptd_pcg_disk_command {
    int32_t  type;
    uint32_t pad;
    uint64_t from;
    uint64_t to;
    double   multiplier;
    struct ptd_pcg_disk_ptr fromT;
    struct ptd_pcg_disk_ptr toT;
    struct ptd_pcg_disk_ptr multiplierptr;
};
#pragma pack(pop)

// Encode a (double*) pointer either as a mem offset or as an
// edge-weight reference. mem_chain is the live ll_of_a chain we
// search through ptd_pcg_chain_offset_of (cannot pointer-compare
// against a flat copy because mem allocations are scattered in the
// heap). edge_anchors is a sorted array of (anchor_ptr, vertex_idx,
// edge_idx) triples used as a binary-search table to find the edge
// a pointer falls within (after accounting for the small offsets
// the recorder uses).
struct ptd_pcg_edge_anchor {
    const double *anchor; // &edge->weight
    uint32_t vertex_idx;
    uint32_t edge_idx;
};

static int ptd_pcg_anchor_cmp(const void *a, const void *b) {
    const struct ptd_pcg_edge_anchor *aa = (const struct ptd_pcg_edge_anchor *)a;
    const struct ptd_pcg_edge_anchor *bb = (const struct ptd_pcg_edge_anchor *)b;
    if (aa->anchor < bb->anchor) return -1;
    if (aa->anchor > bb->anchor) return 1;
    return 0;
}

// Forward declarations for helpers defined further down.
static int64_t ptd_pcg_chain_offset_of(
        const struct ll_of_a *head, const double *ptr);

// Internal encoder that supports both v1 (NULL/MEM/EDGE) and v2
// (additionally EXTERNAL) pointer kinds. external_anchors is a
// caller-provided array of double* pointers; any encode-target
// matching one of these is encoded as PTD_PCG_PTR_EXTERNAL with
// the matching index. Pass external_anchors=NULL, n_external=0
// for v1 behaviour.
static void ptd_pcg_encode_ptr_impl(
        const double *ptr,
        const struct ll_of_a *mem_chain,
        const struct ptd_pcg_edge_anchor *anchors,
        size_t n_anchors,
        const double *const *external_anchors,
        size_t n_external,
        struct ptd_pcg_disk_ptr *out)
{
    memset(out, 0, sizeof(*out));
    if (ptr == NULL) {
        out->kind = PTD_PCG_PTR_NULL;
        return;
    }
    // External-anchor fast path: cheap O(n_external) scan, since
    // n_external is bounded by the number of synthetic placeholder
    // edges (small in practice). Checked first so a placeholder
    // coefficient sitting in a malloc'd block doesn't accidentally
    // match a different vertex's edge anchor by sheer pointer
    // proximity.
    if (external_anchors != NULL && n_external > 0) {
        for (size_t i = 0; i < n_external; ++i) {
            if (ptr == external_anchors[i]) {
                out->kind = PTD_PCG_PTR_EXTERNAL;
                out->vertex_idx = (uint32_t)i;
                return;
            }
        }
    }
    // Mem-pointer fast path: walk the linked-list chain looking for
    // a node whose mem buffer contains ptr. The `mem` allocations
    // are separate calloc()s scattered in the heap, so we cannot
    // pointer-compare against a flat copy — we have to scan the
    // chain. ptd_pcg_chain_offset_of returns the doubles offset
    // into the (oldest-first) flattened layout, or -1 on miss.
    int64_t mem_off = ptd_pcg_chain_offset_of(mem_chain, ptr);
    if (mem_off >= 0) {
        out->kind = PTD_PCG_PTR_MEM;
        out->doubles_offset = mem_off;
        return;
    }
    // Edge-pointer search: find the anchor with the smallest non-
    // negative byte distance to ptr, then verify the offset is
    // small (<= one double — the recorder's only known offsets are
    // 0 and -sizeof(double)).
    if (anchors != NULL && n_anchors > 0) {
        // Binary search for the largest anchor with anchor <= ptr.
        size_t lo = 0, hi = n_anchors;
        while (lo < hi) {
            size_t mid = (lo + hi) / 2;
            if (anchors[mid].anchor <= ptr) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // Candidates: anchor at index (lo - 1), or one of its
        // neighbours, since the -1 trick puts ptr just before the
        // anchor.
        for (int delta = -1; delta <= 1; delta++) {
            ptrdiff_t idx = (ptrdiff_t)lo - 1 + delta;
            if (idx < 0 || (size_t)idx >= n_anchors) continue;
            const double *anchor = anchors[idx].anchor;
            ptrdiff_t byte_off = (const char *)ptr - (const char *)anchor;
            if (byte_off >= -((ptrdiff_t)sizeof(double))
                    && byte_off <= (ptrdiff_t)sizeof(double)) {
                out->kind = PTD_PCG_PTR_EDGE;
                out->vertex_idx = anchors[idx].vertex_idx;
                out->edge_idx = anchors[idx].edge_idx;
                out->byte_offset_from_edge_weight = (int64_t)byte_off;
                return;
            }
        }
    }
    // Pointer doesn't match anything we know about. Caller signals
    // failure to the user — saving with an unencodable pointer would
    // produce a load-time crash later.
    out->kind = PTD_PCG_PTR_NULL;  // sentinel; caller checks
    out->doubles_offset = -1;       // marker for "encoding failed"
}

// v1 encoder: thin wrapper preserving the original signature.
static void ptd_pcg_encode_ptr(
        const double *ptr,
        const struct ll_of_a *mem_chain,
        const struct ptd_pcg_edge_anchor *anchors,
        size_t n_anchors,
        struct ptd_pcg_disk_ptr *out)
{
    ptd_pcg_encode_ptr_impl(ptr, mem_chain, anchors, n_anchors,
                            NULL, 0, out);
}

// Internal decoder supporting both v1 (NULL/MEM/EDGE) and v2
// (additionally EXTERNAL) pointer kinds. external_table is a
// caller-provided array of doubles; EXTERNAL pointers resolve to
// &external_table[vertex_idx]. Pass external_table=NULL,
// n_external=0 for v1 behaviour (encountering an EXTERNAL pointer
// then yields NULL, indicating corruption / version mismatch).
static double *ptd_pcg_decode_ptr_impl(
        const struct ptd_pcg_disk_ptr *enc,
        double *mem_base,
        const struct ptd_graph *graph,
        const double *external_table,
        size_t n_external)
{
    if (enc->kind == PTD_PCG_PTR_NULL) {
        return NULL;
    }
    if (enc->kind == PTD_PCG_PTR_MEM) {
        return mem_base + enc->doubles_offset;
    }
    if (enc->kind == PTD_PCG_PTR_EDGE) {
        if (enc->vertex_idx >= graph->vertices_length) {
            return NULL;  // corrupt
        }
        struct ptd_vertex *v = graph->vertices[enc->vertex_idx];
        if (enc->edge_idx >= v->edges_length) {
            return NULL;  // corrupt
        }
        char *base = (char *)&v->edges[enc->edge_idx]->weight;
        return (double *)(base + enc->byte_offset_from_edge_weight);
    }
    if (enc->kind == PTD_PCG_PTR_EXTERNAL) {
        if (external_table == NULL || enc->vertex_idx >= n_external) {
            return NULL;  // corrupt or v1 loader on v2 file
        }
        // The cast drops const because the replay loop expects
        // double* (it never writes through these pointers when
        // they're EXTERNAL — those pointers appear only as
        // multiplierptr, which the replay reads but doesn't write).
        return (double *)&external_table[enc->vertex_idx];
    }
    return NULL;
}

// v1 decoder: thin wrapper preserving the original signature.
static double *ptd_pcg_decode_ptr(
        const struct ptd_pcg_disk_ptr *enc,
        double *mem_base,
        const struct ptd_graph *graph)
{
    return ptd_pcg_decode_ptr_impl(enc, mem_base, graph, NULL, 0);
}

// Build a sorted anchor table from a graph: one entry per
// (vertex, edge) pair, sorted by &edge->weight ascending. Used by
// the save path to encode edge-weight pointers.
static struct ptd_pcg_edge_anchor *ptd_pcg_build_anchors(
        const struct ptd_graph *graph,
        size_t *out_n_anchors)
{
    size_t total = 0;
    for (size_t i = 0; i < graph->vertices_length; i++) {
        total += graph->vertices[i]->edges_length;
    }
    if (total == 0) {
        *out_n_anchors = 0;
        return NULL;
    }
    struct ptd_pcg_edge_anchor *arr =
            (struct ptd_pcg_edge_anchor *)malloc(total * sizeof(*arr));
    if (arr == NULL) {
        *out_n_anchors = 0;
        return NULL;
    }
    size_t idx = 0;
    for (size_t i = 0; i < graph->vertices_length; i++) {
        struct ptd_vertex *v = graph->vertices[i];
        for (size_t j = 0; j < v->edges_length; j++) {
            arr[idx].anchor = &v->edges[j]->weight;
            arr[idx].vertex_idx = (uint32_t)i;
            arr[idx].edge_idx = (uint32_t)j;
            idx++;
        }
    }
    qsort(arr, total, sizeof(*arr), ptd_pcg_anchor_cmp);
    *out_n_anchors = total;
    return arr;
}

// Walk the linked-list mem chain and produce a flat doubles buffer
// containing every double the chain references in chain order. The
// chain's head is the most-recently-added node; we reverse on the
// fly so the flat buffer's indexing is stable across save/load.
//
// Each node's payload is `node->mem[0..node->current_mem_index)`.
// The head node may be partially filled; older nodes are full
// (32768 doubles) — see add_mem at line ~5915.
static double *ptd_pcg_flatten_mem(
        const struct ll_of_a *head,
        size_t *out_total_doubles)
{
    // Count chain nodes and total doubles.
    size_t chain_len = 0;
    size_t total = 0;
    for (const struct ll_of_a *p = head; p != NULL; p = p->next) {
        chain_len++;
        total += p->current_mem_index;
    }
    *out_total_doubles = total;
    if (total == 0) {
        return NULL;
    }
    // Collect node pointers so we can walk in reverse (oldest first).
    const struct ll_of_a **nodes =
            (const struct ll_of_a **)malloc(chain_len * sizeof(*nodes));
    if (nodes == NULL) {
        return NULL;
    }
    size_t k = chain_len;
    for (const struct ll_of_a *p = head; p != NULL; p = p->next) {
        nodes[--k] = p;
    }
    double *flat = (double *)malloc(total * sizeof(double));
    if (flat == NULL) {
        free(nodes);
        return NULL;
    }
    size_t off = 0;
    for (size_t i = 0; i < chain_len; i++) {
        size_t n = nodes[i]->current_mem_index;
        if (n > 0) {
            memcpy(flat + off, nodes[i]->mem, n * sizeof(double));
        }
        off += n;
    }
    free(nodes);
    return flat;
}

// Compute the offset (in doubles) of a pointer within the
// linked-list mem chain, traversing the chain in oldest-first order.
// Returns -1 if the pointer doesn't fall inside any node.
static int64_t ptd_pcg_chain_offset_of(
        const struct ll_of_a *head,
        const double *ptr)
{
    if (ptr == NULL) {
        return -1;
    }
    // Same chain-reverse logic as ptd_pcg_flatten_mem.
    size_t chain_len = 0;
    for (const struct ll_of_a *p = head; p != NULL; p = p->next) {
        chain_len++;
    }
    if (chain_len == 0) {
        return -1;
    }
    const struct ll_of_a **nodes =
            (const struct ll_of_a **)malloc(chain_len * sizeof(*nodes));
    if (nodes == NULL) {
        return -1;
    }
    size_t k = chain_len;
    for (const struct ll_of_a *p = head; p != NULL; p = p->next) {
        nodes[--k] = p;
    }
    int64_t off = 0;
    int64_t result = -1;
    int64_t onepast = -1;   // fallback: ptr exactly one past a block's last slot
    for (size_t i = 0; i < chain_len; i++) {
        const double *base = nodes[i]->mem;
        size_t n = nodes[i]->current_mem_index;
        // Prefer a STRICT-interior match (base <= ptr < base+n): unambiguous,
        // and immune to the adjacency hazard below. Only if no block strictly
        // contains ptr do we accept a one-past match (ptr == base+n) — writers
        // hold end pointers eagerly. The old code accepted one-past inline,
        // which let a pointer at the start of an adjacent (later) block falsely
        // match the previous block's one-past slot — a wrong-but-in-bounds
        // offset, intermittent with malloc layout. (See zero-copy-cache-plan.md.)
        if (ptr >= base && ptr < base + n) {
            result = off + (int64_t)(ptr - base);
            break;
        }
        if (ptr == base + n && onepast < 0) {
            onepast = off + (int64_t)(ptr - base);
        }
        off += (int64_t)n;
    }
    free(nodes);
    return (result >= 0) ? result : onepast;
}

/* B1: convert a freshly-built raw-pointer parameterized PRC to the offset form.
 * Reuses ptd_pcg_flatten_mem / ptd_pcg_build_anchors / ptd_pcg_encode_ptr_impl
 * (the same classifier the save path uses). MEM operands -> doubles offset into
 * the flattened mem; EDGE/EXTERNAL operands -> a deduped inputs[] index bound to
 * the live address. Returns NULL on OOM or an unencodable pointer (no silent
 * fallback). */
static struct ptd_desc_reward_compute_parameterized_off *ptd_pcg_convert_to_offset(
        const struct ptd_desc_reward_compute_parameterized *raw,
        const struct ptd_graph *graph,
        const double *const *external_anchors, size_t n_external)
{
    struct ptd_desc_reward_compute_parameterized_off *off =
        (struct ptd_desc_reward_compute_parameterized_off *)calloc(1, sizeof(*off));
    if (off == NULL) return NULL;
    off->length = raw->length;
    off->mem_base = ptd_pcg_flatten_mem((const struct ll_of_a *)raw->mem,
                                        &off->mem_doubles);
    size_t n_anchors = 0;
    struct ptd_pcg_edge_anchor *anchors = ptd_pcg_build_anchors(graph, &n_anchors);
    off->commands = (struct ptd_pcg_command_off *)calloc(
            raw->length ? raw->length : 1, sizeof(*off->commands));
    if (off->commands == NULL) { free(anchors); ptd_pcg_desc_off_destroy(off); return NULL; }
    /* Dedup table of distinct EDGE/EXTERNAL references -> input_idx. */
    struct ptd_pcg_input_spec *spec = NULL;
    size_t n_spec = 0, cap_spec = 0;
    int ok = 1;
    /* Open-addressing dedup table: hashed (kind,v,e,byte) -> (spec index + 1);
     * 0 = empty. Sized for a low load factor. */
    size_t htcap = 16;
    while (htcap < (n_anchors + n_external) * 4 + 64) htcap <<= 1;
    size_t *ht = (size_t *)calloc(htcap, sizeof(size_t));
    if (ht == NULL) { free(anchors); ptd_pcg_desc_off_destroy(off); return NULL; }
    for (size_t i = 0; i < raw->length && ok; i++) {
        const struct ptd_comp_graph_parameterized *c = &raw->commands[i];
        off->commands[i].type = (int32_t)c->type;
        off->commands[i].from = (uint64_t)c->from;
        off->commands[i].to = (uint64_t)c->to;
        off->commands[i].multiplier = c->multiplier;
        const double *ptrs[3] = { c->fromT, c->toT, c->multiplierptr };
        struct ptd_pcg_operand *outs[3] = {
            &off->commands[i].fromT, &off->commands[i].toT, &off->commands[i].multiplierptr };
        for (int k = 0; k < 3 && ok; k++) {
            outs[k]->kind = PTD_PCG_OP_NULL;
            if (ptrs[k] == NULL) continue;
            struct ptd_pcg_disk_ptr dp;
            ptd_pcg_encode_ptr_impl(ptrs[k], (const struct ll_of_a *)raw->mem,
                                    anchors, n_anchors, external_anchors, n_external, &dp);
            if (dp.kind == PTD_PCG_PTR_MEM) {
                outs[k]->kind = PTD_PCG_OP_MEM;
                outs[k]->mem_offset = dp.doubles_offset;
            } else if (dp.kind == PTD_PCG_PTR_EDGE || dp.kind == PTD_PCG_PTR_EXTERNAL) {
                uint8_t kk = dp.kind;
                uint32_t vv = dp.vertex_idx, ee = dp.edge_idx;
                int64_t bb = (dp.kind == PTD_PCG_PTR_EDGE)
                             ? dp.byte_offset_from_edge_weight : 0;
                uint64_t _hh = 1469598103934665603ULL;
                _hh = (_hh ^ kk) * 1099511628211ULL;
                _hh = (_hh ^ vv) * 1099511628211ULL;
                _hh = (_hh ^ ee) * 1099511628211ULL;
                _hh = (_hh ^ (uint64_t)bb) * 1099511628211ULL;
                size_t _slot = (size_t)(_hh & (htcap - 1));
                size_t found = (size_t)-1;
                while (ht[_slot] != 0) {
                    size_t si = ht[_slot] - 1;
                    if (spec[si].kind == kk && spec[si].v == vv
                            && spec[si].e == ee && spec[si].byte == bb) { found = si; break; }
                    _slot = (_slot + 1) & (htcap - 1);
                }
                if (found == (size_t)-1) {
                    if (n_spec == cap_spec) {
                        size_t nc = cap_spec ? cap_spec * 2 : 16;
                        struct ptd_pcg_input_spec *np =
                            (struct ptd_pcg_input_spec *)realloc(spec, nc * sizeof(*spec));
                        if (np == NULL) { ok = 0; break; }
                        spec = np; cap_spec = nc;
                    }
                    spec[n_spec].kind = kk; spec[n_spec].v = vv;
                    spec[n_spec].e = ee; spec[n_spec].byte = bb;
                    found = n_spec++;
                    ht[_slot] = found + 1;
                }
                outs[k]->kind = PTD_PCG_OP_INPUT;
                outs[k]->input_idx = (uint32_t)found;
            } else {
                /* encoder returned NULL kind = unencodable pointer: fail. */
                ok = 0;
            }
        }
    }
    free(anchors);
    free(ht);
    if (!ok) { free(spec); ptd_pcg_desc_off_destroy(off); return NULL; }
    off->n_inputs = n_spec;
    if (n_spec > 0) {
        off->inputs = (double **)malloc(n_spec * sizeof(double *));
        if (off->inputs == NULL) { free(spec); ptd_pcg_desc_off_destroy(off); return NULL; }
        for (size_t s = 0; s < n_spec; s++) {
            if (spec[s].kind == PTD_PCG_PTR_EDGE) {
                struct ptd_vertex *v = graph->vertices[spec[s].v];
                char *base = (char *)&v->edges[spec[s].e]->weight;
                off->inputs[s] = (double *)(base + spec[s].byte);
            } else { /* EXTERNAL: the raw ptr == external_anchors[v] (live coeff) */
                off->inputs[s] = (double *)external_anchors[spec[s].v];
            }
        }
    }
    off->input_specs = spec;   /* descriptor owns spec (freed in destroy) */
    return off;
}

/* ===== rev-3 zero-copy cache format =====================================
 * File = [header | commands_off[] | mem doubles | input-specs[]]. The commands
 * are the offset/index POD form, so load is fixup-free (B2: read+copy; B3: mmap).
 * Distinct magic "PTDPRMC3" so a stale rev-1/2 file at the same path is a miss. */
#define PTD_PCG3_MAGIC "PTDPRMC3"
struct ptd_pcg3_header {
    char     magic[8];
    uint64_t n_commands;
    uint64_t mem_doubles;
    uint64_t n_inputs;
    uint64_t reserved;
};
struct ptd_pcg3_dinput {     /* fixed on-disk form of ptd_pcg_input_spec */
    int64_t  byte;
    uint32_t v;
    uint32_t e;
    uint8_t  kind;
    uint8_t  pad[7];
};

int ptd_save_pcg_rev3(const char *path,
        const struct ptd_desc_reward_compute_parameterized *raw,
        const struct ptd_graph *graph) {
    struct ptd_desc_reward_compute_parameterized_off *off =
        ptd_pcg_convert_to_offset(raw, graph, NULL, 0);
    if (off == NULL) {
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_save_pcg_rev3: convert_to_offset failed");
        return -1;
    }
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, (int)getpid());
    FILE *fp = fopen(tmp, "wb");
    if (fp == NULL) {
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_save_pcg_rev3: fopen %s failed", tmp);
        ptd_pcg_desc_off_destroy(off);
        return -1;
    }
    struct ptd_pcg3_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, PTD_PCG3_MAGIC, 8);
    h.n_commands = off->length;
    h.mem_doubles = off->mem_doubles;
    h.n_inputs = off->n_inputs;
    int werr = (fwrite(&h, sizeof(h), 1, fp) != 1);
    if (!werr && off->length > 0)
        werr = (fwrite(off->commands, sizeof(*off->commands), off->length, fp)
                != off->length);
    if (!werr && off->mem_doubles > 0)
        werr = (fwrite(off->mem_base, sizeof(double), off->mem_doubles, fp)
                != off->mem_doubles);
    for (size_t i = 0; i < off->n_inputs && !werr; i++) {
        struct ptd_pcg3_dinput di;
        memset(&di, 0, sizeof(di));
        di.byte = off->input_specs[i].byte;
        di.v = off->input_specs[i].v;
        di.e = off->input_specs[i].e;
        di.kind = off->input_specs[i].kind;
        werr = (fwrite(&di, sizeof(di), 1, fp) != 1);
    }
    ptd_pcg_desc_off_destroy(off);
    if (werr || fclose(fp) != 0) {
        if (!werr) werr = 1; else fclose(fp);
        remove(tmp);
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_save_pcg_rev3: write/close failed for %s", tmp);
        return -1;
    }
    if (rename(tmp, path) != 0) {
        remove(tmp);
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_save_pcg_rev3: rename %s -> %s failed", tmp, path);
        return -1;
    }
    return 0;
}

/* Overflow-safe size_t multiply / add. Return 1 on success (result in *out),
 * 0 if the operation would overflow. Used to validate untrusted header counts
 * and section sizes from a cache file before they drive malloc / mmap bounds. */
static int ptd_size_mul_ok(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}
static int ptd_size_add_ok(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a) return 0;
    *out = a + b;
    return 1;
}

/* Validate every command in a loaded rev-3 descriptor against untrusted input.
 * A cache file can be corrupt, from another user on a shared filesystem, or
 * pulled from the community registry, and the executor (ptd_pcg_resolve /
 * ptd_graph_build_ex_absorbation_time_comp_graph_parameterized_off) resolves
 * operands as mem_base+mem_offset and inputs[input_idx] and WRITES through the
 * resolved pointer with no bounds check. Reject (treat as cache miss) any file
 * whose command type is unknown (would DIE_ERROR-abort the process), whose
 * write target is not a writable MEM slot, or whose MEM offset / INPUT index
 * is out of range. Returns 0 if safe, -1 otherwise. */
static int ptd_pcg3_off_validate(
        const struct ptd_desc_reward_compute_parameterized_off *off) {
    const size_t md = off->mem_doubles;
    const size_t ni = off->n_inputs;
    if (off->length > 0 && off->commands == NULL) return -1;
    const int dbg = (getenv("PHASIC_PCG_LOADLOG") != NULL);
    for (size_t i = 0; i < off->length; ++i) {
        const struct ptd_pcg_command_off *c = &off->commands[i];
        /* Known command type only (see the executor's command_types enum:
         * NEW_ADD=0, P=1, INV=2, PP=3, ONE_MINUS=4, DIVIDE=5, ZERO=6). An
         * unknown type would DIE_ERROR-abort the whole process. */
        if (c->type < 0 || c->type > 6) {
            if (dbg) PTD_LOG_WARNING("pcg validate: cmd %zu bad type %d", i, (int)c->type);
            return -1;
        }
        /* Every operand ptd_pcg_resolve() turns into a pointer must resolve
         * IN-BOUNDS: a MEM offset within the mem scratch buffer, an INPUT
         * index within inputs[]. This is what closes the arbitrary-write
         * (mem_base + attacker_offset) and OOB-read (inputs[huge]) the
         * executor would otherwise perform. (A NULL operand resolves to NULL;
         * if a malformed file dereferences it that is a benign crash, not a
         * memory-corruption primitive, so NULL is permitted for the operands
         * a given command type does not use.) */
        const struct ptd_pcg_operand *ops[3] =
            { &c->fromT, &c->toT, &c->multiplierptr };
        for (int k = 0; k < 3; ++k) {
            const struct ptd_pcg_operand *op = ops[k];
            if (op->kind == PTD_PCG_OP_MEM) {
                if (op->mem_offset < 0 || (size_t)op->mem_offset >= md) {
                    if (dbg) PTD_LOG_WARNING("pcg validate: cmd %zu op %d MEM offset %lld "
                                             "out of range md=%zu", i, k,
                                             (long long)op->mem_offset, md);
                    return -1;
                }
            } else if (op->kind == PTD_PCG_OP_INPUT) {
                if ((size_t)op->input_idx >= ni) {
                    if (dbg) PTD_LOG_WARNING("pcg validate: cmd %zu op %d INPUT idx %u "
                                             ">= ni=%zu", i, k, op->input_idx, ni);
                    return -1;
                }
            } else if (op->kind != PTD_PCG_OP_NULL) {
                if (dbg) PTD_LOG_WARNING("pcg validate: cmd %zu op %d bad kind %u",
                                         i, k, (unsigned)op->kind);
                return -1;   /* unknown operand kind */
            }
        }
    }
    return 0;
}

/* Load a rev-3 file into the offset descriptor by READ+COPY (the mmap-free
 * fallback). Binds inputs[] against the supplied graph; an out-of-range (v,e)
 * means the file is stale for this graph -> return NULL (cache miss). */
static struct ptd_desc_reward_compute_parameterized_off *ptd_load_pcg_rev3_copy(
        const char *path, const struct ptd_graph *graph) {
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) return NULL;
    struct ptd_pcg3_header h;
    if (fread(&h, sizeof(h), 1, fp) != 1
            || memcmp(h.magic, PTD_PCG3_MAGIC, 8) != 0) {
        fclose(fp);
        return NULL;   /* absent / not a rev-3 file -> miss */
    }
    struct ptd_desc_reward_compute_parameterized_off *off =
        (struct ptd_desc_reward_compute_parameterized_off *)calloc(1, sizeof(*off));
    if (off == NULL) { fclose(fp); return NULL; }
    off->length = (size_t)h.n_commands;
    off->mem_doubles = (size_t)h.mem_doubles;
    off->n_inputs = (size_t)h.n_inputs;
    int err = 0;
    size_t nbytes = 0;
    if (h.n_commands > 0) {
        /* Overflow-safe: an attacker-chosen n_commands must not wrap the
         * malloc size and let the subsequent fread overflow the buffer. */
        if (!ptd_size_mul_ok((size_t)h.n_commands, sizeof(*off->commands), &nbytes)) {
            err = 1;
        } else {
            off->commands = (struct ptd_pcg_command_off *)malloc(nbytes);
            err = (off->commands == NULL
                   || fread(off->commands, sizeof(*off->commands),
                            (size_t)h.n_commands, fp) != h.n_commands);
        }
    }
    if (!err && h.mem_doubles > 0) {
        if (!ptd_size_mul_ok((size_t)h.mem_doubles, sizeof(double), &nbytes)) {
            err = 1;
        } else {
            off->mem_base = (double *)malloc(nbytes);
            err = (off->mem_base == NULL
                   || fread(off->mem_base, sizeof(double),
                            (size_t)h.mem_doubles, fp) != h.mem_doubles);
        }
    }
    if (!err && h.n_inputs > 0
            && (!ptd_size_mul_ok((size_t)h.n_inputs, sizeof(*off->input_specs), &nbytes))) {
        err = 1;
    }
    if (!err && h.n_inputs > 0) {
        off->input_specs = (struct ptd_pcg_input_spec *)
            malloc((size_t)h.n_inputs * sizeof(*off->input_specs));
        off->inputs = (double **)malloc((size_t)h.n_inputs * sizeof(double *));
        if (off->input_specs == NULL || off->inputs == NULL) err = 1;
        for (size_t i = 0; i < h.n_inputs && !err; i++) {
            struct ptd_pcg3_dinput di;
            if (fread(&di, sizeof(di), 1, fp) != 1) { err = 1; break; }
            off->input_specs[i].kind = di.kind;
            off->input_specs[i].v = di.v;
            off->input_specs[i].e = di.e;
            off->input_specs[i].byte = di.byte;
            if (di.kind == PTD_PCG_PTR_EDGE) {
                if (di.v >= graph->vertices_length) { err = 1; break; }
                struct ptd_vertex *vx = graph->vertices[di.v];
                if (di.e >= vx->edges_length) { err = 1; break; }
                char *base = (char *)&vx->edges[di.e]->weight;
                off->inputs[i] = (double *)(base + di.byte);
            } else {
                /* EXTERNAL not expected for the monolith cache (B5). */
                err = 1; break;
            }
        }
    }
    fclose(fp);
    if (!err && ptd_pcg3_off_validate(off) != 0) err = 1;
    if (err) {
        ptd_pcg_desc_off_destroy(off);
        ptd_err[0] = '\0';   /* treat as a cache miss, not an error */
        return NULL;
    }
    if (getenv("PHASIC_PCG_LOADLOG") != NULL) {
        PTD_LOG_WARNING("rev3 LOAD(copy) hit: n_cmds=%zu n_inputs=%zu",
                        off->length, off->n_inputs);
    }
    return off;
}

#ifndef _WIN32
/* B3: zero-copy load — mmap the rev-3 file (MAP_PRIVATE so the mem section is
 * copy-on-write writable for the executor's *fromT stores; the file is never
 * modified). commands/mem/input-specs are pointed directly into the mapping
 * (no fread, no per-command fixup); only inputs[] is bound (heap, O(n_inputs)).
 * Returns NULL on any failure -> dispatcher falls back to the copy loader. */
static struct ptd_desc_reward_compute_parameterized_off *ptd_load_pcg_rev3_mmap(
        const char *path, const struct ptd_graph *graph) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0
            || (size_t)st.st_size < sizeof(struct ptd_pcg3_header)) {
        close(fd); return NULL;
    }
    size_t fsize = (size_t)st.st_size;
    void *base = mmap(NULL, fsize, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) return NULL;
    const struct ptd_pcg3_header *h = (const struct ptd_pcg3_header *)base;
    if (memcmp(h->magic, PTD_PCG3_MAGIC, 8) != 0) { munmap(base, fsize); return NULL; }
    size_t n_cmds = (size_t)h->n_commands;
    size_t mem_d = (size_t)h->mem_doubles;
    size_t n_in = (size_t)h->n_inputs;
    /* Section offsets (sizes are 8-byte multiples; mmap base is page-aligned, so
     * the casts below are naturally aligned). Verify the file holds them all. */
    size_t off_cmds = sizeof(struct ptd_pcg3_header);
    /* Overflow-safe section arithmetic: attacker-chosen counts must not wrap
     * `need` small enough to pass the `need > fsize` check while the section
     * pointers below run past the end of the mapping. */
    size_t tmp = 0, off_mem = 0, off_specs = 0, need = 0;
    if (!ptd_size_mul_ok(n_cmds, sizeof(struct ptd_pcg_command_off), &tmp)
            || !ptd_size_add_ok(off_cmds, tmp, &off_mem)
            || !ptd_size_mul_ok(mem_d, sizeof(double), &tmp)
            || !ptd_size_add_ok(off_mem, tmp, &off_specs)
            || !ptd_size_mul_ok(n_in, sizeof(struct ptd_pcg3_dinput), &tmp)
            || !ptd_size_add_ok(off_specs, tmp, &need)) {
        munmap(base, fsize); return NULL;
    }
    if (need > fsize) { munmap(base, fsize); return NULL; }
    struct ptd_desc_reward_compute_parameterized_off *o =
        (struct ptd_desc_reward_compute_parameterized_off *)calloc(1, sizeof(*o));
    if (o == NULL) { munmap(base, fsize); return NULL; }
    o->mem_is_mmap = 1;
    o->mmap_base = base;
    o->mmap_len = fsize;
    o->length = n_cmds;
    o->commands = (struct ptd_pcg_command_off *)((char *)base + off_cmds);
    o->mem_doubles = mem_d;
    o->mem_base = (double *)((char *)base + off_mem);
    o->n_inputs = n_in;
    o->input_specs = NULL;   /* a loaded descriptor is never re-saved */
    if (n_in > 0) {
        o->inputs = (double **)malloc(n_in * sizeof(double *));
        if (o->inputs == NULL) { ptd_pcg_desc_off_destroy(o); return NULL; }
        const struct ptd_pcg3_dinput *di =
            (const struct ptd_pcg3_dinput *)((char *)base + off_specs);
        for (size_t i = 0; i < n_in; i++) {
            if (di[i].kind != PTD_PCG_PTR_EDGE) { ptd_pcg_desc_off_destroy(o); return NULL; }
            if (di[i].v >= graph->vertices_length) { ptd_pcg_desc_off_destroy(o); return NULL; }
            struct ptd_vertex *vx = graph->vertices[di[i].v];
            if (di[i].e >= vx->edges_length) { ptd_pcg_desc_off_destroy(o); return NULL; }
            char *eb = (char *)&vx->edges[di[i].e]->weight;
            o->inputs[i] = (double *)(eb + di[i].byte);
        }
    }
    /* Validate command operands against the (now bounds-checked) section
     * sizes before the executor resolves and writes through them. */
    if (ptd_pcg3_off_validate(o) != 0) { ptd_pcg_desc_off_destroy(o); return NULL; }
    if (getenv("PHASIC_PCG_LOADLOG") != NULL) {
        PTD_LOG_WARNING("rev3 LOAD(mmap) hit: n_cmds=%zu n_inputs=%zu", n_cmds, n_in);
    }
    return o;
}
#endif

/* B3 dispatcher: try the zero-copy mmap load; on any failure fall back to the
 * read+copy loader (identical descriptor, identical results). Fallback is
 * explicit (env PHASIC_PCG_DISABLE_MMAP forces it; Windows always uses copy). */
struct ptd_desc_reward_compute_parameterized_off *ptd_load_pcg_rev3(
        const char *path, const struct ptd_graph *graph) {
#ifndef _WIN32
    if (getenv("PHASIC_PCG_DISABLE_MMAP") == NULL) {
        struct ptd_desc_reward_compute_parameterized_off *o =
            ptd_load_pcg_rev3_mmap(path, graph);
        if (o != NULL) return o;
        ptd_err[0] = '\0';   /* mmap miss/failure -> fall back, not an error */
    }
#endif
    return ptd_load_pcg_rev3_copy(path, graph);
}

// Return non-zero if the reward-compute disk cache is disabled.
// The Python field `phasic.configure(reward_compute_cache=...)`
// defaults to False (caching OFF). The positive env var
// PHASIC_REWARD_COMPUTE_CACHE is set to "1" by configure() when
// the user opts in; absence means "default policy" = disabled.
static int ptd_pcg_cache_disabled(void) {
    const char *v = getenv("PHASIC_REWARD_COMPUTE_CACHE");
    if (v == NULL) return 1;
    return !(v[0] == '1' && v[1] == '\0');
}

// Build the path to the per-graph cache file:
//   <home>/.phasic_cache/parameterized_reward_compute/<hash_hex>.bin
// Creates parent directories as needed (mkdir -p style).
//
// Returns 0 on success, -1 on error (sets ptd_err). The bin file
// itself is NOT created.
/* SLURM-WP-1: resolve the cache root directory.
 *
 * Honours PHASIC_CACHE_DIR env var; falls back to $HOME/.phasic_cache.
 * Public-API helper declared in api/c/phasic.h. */
int ptd_cache_root_dir(char *buf, size_t buf_len)
{
    const char *override = getenv("PHASIC_CACHE_DIR");
    if (override != NULL && override[0] != '\0') {
        int n = snprintf(buf, buf_len, "%s", override);
        if (n < 0 || (size_t)n >= buf_len) {
            snprintf((char *)ptd_err, sizeof(ptd_err),
                     "ptd_cache_root_dir: PHASIC_CACHE_DIR path too long");
            return -1;
        }
        return 0;
    }
    const char *home = getenv("HOME");
    if (home == NULL) {
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_cache_root_dir: HOME not set and "
                 "PHASIC_CACHE_DIR not set");
        return -1;
    }
    int n = snprintf(buf, buf_len, "%s/.phasic_cache", home);
    if (n < 0 || (size_t)n >= buf_len) {
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_cache_root_dir: $HOME/.phasic_cache too long");
        return -1;
    }
    return 0;
}

static int ptd_pcg_build_cache_path(
        const struct ptd_graph *graph, char *buf, size_t buf_len)
{
    char root[PATH_MAX];
    if (ptd_cache_root_dir(root, sizeof(root)) != 0) {
        return -1;  /* ptd_err set by helper */
    }
    char dir[PATH_MAX];
    int n = snprintf(dir, sizeof(dir),
                     "%s/parameterized_reward_compute", root);
    if (n < 0 || (size_t)n >= sizeof(dir)) {
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_pcg: cache dir path too long");
        return -1;
    }
    // Best-effort mkdir of each parent. Ignore EEXIST.
    struct stat st;
    if (stat(root, &st) != 0) {
        if (mkdir(root, 0755) != 0 && errno != EEXIST) {
            snprintf((char *)ptd_err, sizeof(ptd_err),
                     "ptd_pcg: cannot create %s: %s", root, strerror(errno));
            return -1;
        }
    }
    if (stat(dir, &st) != 0) {
        if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
            snprintf((char *)ptd_err, sizeof(ptd_err),
                     "ptd_pcg: cannot create %s: %s", dir, strerror(errno));
            return -1;
        }
    }
    struct ptd_hash_result *hash = ptd_graph_content_hash(graph);
    if (hash == NULL) {
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_pcg: ptd_graph_content_hash failed");
        return -1;
    }
    n = snprintf(buf, buf_len, "%s/%s.bin", dir, hash->hash_hex);
    free(hash);
    if (n < 0 || (size_t)n >= buf_len) {
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_pcg: cache path too long");
        return -1;
    }
    return 0;
}

// Encode a single command into its on-disk form. Factored out of the
// save path so the writer can stream commands in fixed-size chunks
// (O(chunk) memory) instead of materialising the whole length-sized
// array (O(commands_length * 128B) — multi-GB for large parameterized
// models). `out` is fully zeroed here, so unused fields and the struct
// padding are 0, matching the previous calloc'd-array layout byte-for-byte.
// Returns 0 on success, -1 on an unencodable pointer (ptd_err set).
static int ptd_pcg_encode_one_disk_command(
        const struct ptd_comp_graph_parameterized *cmd,
        size_t cmd_index,
        const struct ll_of_a *mem,
        const struct ptd_pcg_edge_anchor *anchors,
        size_t n_anchors,
        const double *const *external_anchors,
        size_t n_external,
        struct ptd_pcg_disk_command *out)
{
    /* Per-type which-fields-are-live table. Mirrors the replay loop in
     * ptd_graph_build_ex_absorbation_time_comp_graph_parameterized. The
     * recorder doesn't initialise unused fields, so encoding them would
     * dereference uninitialised memory ("neither mem nor edge" failure). */
    enum { NEW_ADD_T = 0, P_T = 1, INV_T = 2, PP_T = 3, ONE_MINUS_T = 4,
           DIVIDE_T = 5, ZERO_T = 6 };
    memset(out, 0, sizeof(*out));
    out->type = (int32_t)cmd->type;
    out->from = (uint64_t)cmd->from;
    out->to = (uint64_t)cmd->to;
    out->multiplier = cmd->multiplier;

    bool live_fromT = false, live_toT = false, live_multptr = false;
    switch (cmd->type) {
        case NEW_ADD_T:    live_multptr = true; break;
        case P_T:          live_fromT = true; live_toT = true; break;
        case PP_T:         live_fromT = true; live_toT = true; live_multptr = true; break;
        case INV_T:        live_fromT = true; break;
        case ONE_MINUS_T:  live_fromT = true; break;
        case DIVIDE_T:     live_fromT = true; live_toT = true; break;
        case ZERO_T:       live_fromT = true; break;
        default:
            snprintf((char *)ptd_err, sizeof(ptd_err),
                     "ptd_save: command %zu has unknown type %d", cmd_index, cmd->type);
            return -1;
    }

    if (live_fromT) {
        ptd_pcg_encode_ptr_impl(cmd->fromT, mem, anchors, n_anchors,
                                external_anchors, n_external, &out->fromT);
    } else {
        out->fromT.kind = PTD_PCG_PTR_NULL;
    }
    if (live_toT) {
        ptd_pcg_encode_ptr_impl(cmd->toT, mem, anchors, n_anchors,
                                external_anchors, n_external, &out->toT);
    } else {
        out->toT.kind = PTD_PCG_PTR_NULL;
    }
    if (live_multptr) {
        ptd_pcg_encode_ptr_impl(cmd->multiplierptr, mem, anchors, n_anchors,
                                external_anchors, n_external, &out->multiplierptr);
    } else {
        out->multiplierptr.kind = PTD_PCG_PTR_NULL;
    }

    // Verify each LIVE pointer was successfully encoded. A live pointer
    // that was non-NULL at record time but ended up PTD_PCG_PTR_NULL is
    // unencodable — abort rather than silently produce a corrupt file.
    if ((live_fromT && cmd->fromT != NULL && out->fromT.kind == PTD_PCG_PTR_NULL) ||
            (live_toT && cmd->toT != NULL && out->toT.kind == PTD_PCG_PTR_NULL) ||
            (live_multptr && cmd->multiplierptr != NULL
                    && out->multiplierptr.kind == PTD_PCG_PTR_NULL)) {
        const char *which = "?";
        const void *bad = NULL;
        if (live_fromT && cmd->fromT != NULL && out->fromT.kind == PTD_PCG_PTR_NULL) {
            which = "fromT"; bad = (const void *)cmd->fromT;
        } else if (live_toT && cmd->toT != NULL && out->toT.kind == PTD_PCG_PTR_NULL) {
            which = "toT"; bad = (const void *)cmd->toT;
        } else if (live_multptr && cmd->multiplierptr != NULL && out->multiplierptr.kind == PTD_PCG_PTR_NULL) {
            which = "multiplierptr"; bad = (const void *)cmd->multiplierptr;
        }
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_save: command %zu (type=%d) field=%s pointer=%p "
                 "is neither in the mem buffer nor near a known edge weight; "
                 "cache save aborted to avoid a corrupt file.",
                 cmd_index, cmd->type, which, bad);
        return -1;
    }
    return 0;
}

// Shared implementation for the v1 and v2 save entry points. Pass
// external_anchors=NULL, n_external=0 to write a v1 file (no
// EXTERNAL pointers; format_revision is forced to 1 in that case
// for max compatibility with v1 readers). Pass non-NULL/non-zero
// to enable EXTERNAL pointer encoding; format_revision is then
// written as 2.
static int ptd_save_parameterized_reward_compute_graph_impl(
        const char *path,
        const struct ptd_desc_reward_compute_parameterized *compute,
        const struct ptd_graph *graph,
        const double *const *external_anchors,
        size_t n_external)
{
    if (path == NULL || compute == NULL || graph == NULL) {
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_save_parameterized_reward_compute_graph: NULL argument");
        return -1;
    }

    size_t n_anchors = 0;
    struct ptd_pcg_edge_anchor *anchors = ptd_pcg_build_anchors(graph, &n_anchors);
    // n_anchors == 0 is fine; the compute graph might only reference mem.

    size_t mem_total = 0;
    double *flat_mem = ptd_pcg_flatten_mem(
            (const struct ll_of_a *)compute->mem, &mem_total);
    // flat_mem may be NULL when mem_total == 0; that's not an error.

    struct ptd_hash_result *hash = ptd_graph_content_hash(graph);
    uint64_t hash_truncated = 0;
    if (hash != NULL) {
        for (int i = 0; i < 8; i++) {
            hash_truncated = (hash_truncated << 8) | (uint64_t)hash->hash_full[i];
        }
        free(hash);
    }

    struct ptd_pcg_disk_header header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, PTD_PCG_MAGIC, 8);
    header.version = PTD_PCG_VERSION;
    // Write rev 1 when no EXTERNAL anchors are in play, so v1 readers
    // accept the file. Write rev 2 only when EXTERNAL pointers might
    // appear in the encoded commands.
    header.format_revision = (n_external > 0) ? 2u : 1u;
    header.graph_hash_truncated = hash_truncated;
    header.commands_length = (uint64_t)compute->length;
    header.mem_total_doubles = (uint64_t)mem_total;
    header.memr_length = (uint64_t)graph->vertices_length;

    // Commands are encoded and written in fixed-size chunks at write
    // time (below) via ptd_pcg_encode_one_disk_command, instead of one
    // length-sized array, so the transient memory is O(chunk) not
    // O(commands_length * sizeof(disk_command)) — which is multiple GB
    // for large parameterized models. The on-disk byte layout is
    // unchanged (identical per-command encoding, identical order). The
    // atomic tmp+rename below already guards against a partial file if
    // encoding fails mid-stream, so deferring encoding past fopen() is safe.
    struct ptd_pcg_disk_command *encoded_cmds = NULL;  // chunk buffer (alloc'd below)

    // Encode memr (rates) as doubles offsets into flat_mem.
    int64_t *memr_offsets = NULL;
    if (graph->vertices_length > 0) {
        memr_offsets = (int64_t *)malloc(graph->vertices_length * sizeof(int64_t));
        if (memr_offsets == NULL) {
            snprintf((char *)ptd_err, sizeof(ptd_err),
                     "ptd_save: failed to allocate memr_offsets");
            free(encoded_cmds);
            free(flat_mem);
            free(anchors);
            return -1;
        }
        double **memr = (double **)compute->memr;
        for (size_t i = 0; i < graph->vertices_length; i++) {
            if (memr == NULL || memr[i] == NULL) {
                memr_offsets[i] = -1;
                continue;
            }
            int64_t off = ptd_pcg_chain_offset_of(
                    (const struct ll_of_a *)compute->mem, memr[i]);
            if (off < 0) {
                snprintf((char *)ptd_err, sizeof(ptd_err),
                         "ptd_save: memr[%zu] does not point into the mem buffer; "
                         "cache save aborted.", i);
                free(memr_offsets);
                free(encoded_cmds);
                free(flat_mem);
                free(anchors);
                return -1;
            }
            memr_offsets[i] = off;
        }
    }

    // Atomic write: write to <path>.tmp.<pid>, fsync, then rename.
    char tmp_path[PATH_MAX];
    int written = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%d",
                           path, (int)getpid());
    if (written < 0 || (size_t)written >= sizeof(tmp_path)) {
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_save: cache temp path too long");
        free(memr_offsets);
        free(encoded_cmds);
        free(flat_mem);
        free(anchors);
        return -1;
    }

    FILE *fp = fopen(tmp_path, "wb");
    if (fp == NULL) {
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_save: cannot open temp file %s: %s",
                 tmp_path, strerror(errno));
        free(memr_offsets);
        free(encoded_cmds);
        free(flat_mem);
        free(anchors);
        return -1;
    }
#define WRITE_OR_FAIL(buf, sz)                                                 \
    do {                                                                       \
        if (fwrite((buf), 1, (sz), fp) != (sz)) {                              \
            snprintf((char *)ptd_err, sizeof(ptd_err),                         \
                     "ptd_save: short write to %s: %s",                        \
                     tmp_path, strerror(errno));                               \
            fclose(fp);                                                        \
            unlink(tmp_path);                                                  \
            free(memr_offsets);                                                \
            free(encoded_cmds);                                                \
            free(flat_mem);                                                    \
            free(anchors);                                                     \
            return -1;                                                         \
        }                                                                      \
    } while (0)

    WRITE_OR_FAIL(&header, sizeof(header));
    // Stream-encode the commands in fixed-size chunks (one reused buffer)
    // rather than holding the whole length-sized array in memory. Output
    // bytes are identical to the previous one-shot write.
    if (compute->length > 0) {
        const size_t PTD_PCG_WRITE_CHUNK = 8192;
        size_t buf_n = (compute->length < PTD_PCG_WRITE_CHUNK)
                     ? compute->length : PTD_PCG_WRITE_CHUNK;
        encoded_cmds = (struct ptd_pcg_disk_command *)
                calloc(buf_n, sizeof(*encoded_cmds));
        if (encoded_cmds == NULL) {
            snprintf((char *)ptd_err, sizeof(ptd_err),
                     "ptd_save: failed to allocate command write chunk");
            fclose(fp);
            unlink(tmp_path);
            free(memr_offsets);
            free(flat_mem);
            free(anchors);
            return -1;
        }
        for (size_t base = 0; base < compute->length; base += buf_n) {
            size_t this_n = (compute->length - base < buf_n)
                          ? (compute->length - base) : buf_n;
            for (size_t j = 0; j < this_n; j++) {
                if (ptd_pcg_encode_one_disk_command(
                        &compute->commands[base + j], base + j,
                        (const struct ll_of_a *)compute->mem,
                        anchors, n_anchors, external_anchors, n_external,
                        &encoded_cmds[j]) != 0) {
                    // ptd_err already set by the helper.
                    fclose(fp);
                    unlink(tmp_path);
                    free(encoded_cmds);
                    free(memr_offsets);
                    free(flat_mem);
                    free(anchors);
                    return -1;
                }
            }
            WRITE_OR_FAIL(encoded_cmds, this_n * sizeof(*encoded_cmds));
        }
    }
    if (mem_total > 0) {
        WRITE_OR_FAIL(flat_mem, mem_total * sizeof(double));
    }
    if (graph->vertices_length > 0) {
        WRITE_OR_FAIL(memr_offsets, graph->vertices_length * sizeof(int64_t));
    }
#undef WRITE_OR_FAIL

    // Flush + atomic rename.
    fflush(fp);
    int fd = fileno(fp);
    if (fd >= 0) {
        // fsync is best-effort; failures here are non-fatal because
        // the rename is the durability boundary.
        (void)fsync(fd);
    }
    if (fclose(fp) != 0) {
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_save: fclose failed for %s: %s",
                 tmp_path, strerror(errno));
        unlink(tmp_path);
        free(memr_offsets);
        free(encoded_cmds);
        free(flat_mem);
        free(anchors);
        return -1;
    }
    if (ptd_atomic_rename(tmp_path, path) != 0) {
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_save: rename %s -> %s failed: %s",
                 tmp_path, path, strerror(errno));
        unlink(tmp_path);
        free(memr_offsets);
        free(encoded_cmds);
        free(flat_mem);
        free(anchors);
        return -1;
    }

    free(memr_offsets);
    free(encoded_cmds);
    free(flat_mem);
    free(anchors);
    return 0;
}

// v1 entry point: writes a format-revision-1 file with no
// EXTERNAL pointers. Backward-compatible signature; existing
// callers (Stage A2 cache write in ptd_precompute_reward_compute_graph)
// keep working unchanged.
int ptd_save_parameterized_reward_compute_graph(
        const char *path,
        const struct ptd_desc_reward_compute_parameterized *compute,
        const struct ptd_graph *graph)
{
    return ptd_save_parameterized_reward_compute_graph_impl(
            path, compute, graph, NULL, 0);
}

// v2 entry point: writes a format-revision-2 file with EXTERNAL
// pointer support. Pointers in compute that match an entry in
// external_anchors are encoded as PTD_PCG_PTR_EXTERNAL with the
// matching index. Pass n_external > 0 to actually use this path;
// passing 0 yields v1-equivalent behaviour and writes a rev-1 file.
int ptd_save_parameterized_reward_compute_graph_ex(
        const char *path,
        const struct ptd_desc_reward_compute_parameterized *compute,
        const struct ptd_graph *graph,
        const double *const *external_anchors,
        size_t n_external)
{
    return ptd_save_parameterized_reward_compute_graph_impl(
            path, compute, graph, external_anchors, n_external);
}

// Shared implementation for the v1 and v2 load entry points. The
// loader accepts both rev-1 and rev-2 files; the choice of which
// version to write happens at save time (based on whether
// EXTERNAL anchors were passed). external_table may be NULL with
// n_external == 0; if a rev-2 file contains EXTERNAL pointers and
// no table is supplied, the load fails with a clear error.
static struct ptd_desc_reward_compute_parameterized *
ptd_load_parameterized_reward_compute_graph_impl(
        const char *path,
        const struct ptd_graph *graph,
        const double *external_table,
        size_t n_external)
{
    if (path == NULL || graph == NULL) {
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_load: NULL argument");
        return NULL;
    }

    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        // Cache miss — file doesn't exist (the common case). Set a
        // mild error so callers that care can introspect, but this
        // is expected, not a failure.
        if (errno == ENOENT) {
            ptd_err[0] = '\0';
        } else {
            snprintf((char *)ptd_err, sizeof(ptd_err),
                     "ptd_load: fopen %s failed: %s", path, strerror(errno));
        }
        return NULL;
    }

    struct ptd_pcg_disk_header header;
    if (fread(&header, 1, sizeof(header), fp) != sizeof(header)) {
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_load: short header read in %s", path);
        fclose(fp);
        return NULL;
    }
    // Accept both rev 1 and rev 2 files. The decoder handles
    // EXTERNAL pointers only if external_table is non-NULL; rev 1
    // files won't contain any.
    if (memcmp(header.magic, PTD_PCG_MAGIC, 8) != 0
            || header.version != PTD_PCG_VERSION
            || header.format_revision < 1u
            || header.format_revision > PTD_PCG_FORMAT_REVISION) {
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_load: %s has wrong magic/version "
                 "(got %.8s v%u r%u; expected %s v%u r1..%u). "
                 "Treat as cache miss and rebuild.",
                 path, header.magic, header.version, header.format_revision,
                 PTD_PCG_MAGIC, PTD_PCG_VERSION, PTD_PCG_FORMAT_REVISION);
        fclose(fp);
        return NULL;
    }
    if (header.memr_length != (uint64_t)graph->vertices_length) {
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_load: %s memr_length %llu != graph->vertices_length %zu",
                 path, (unsigned long long)header.memr_length,
                 graph->vertices_length);
        fclose(fp);
        return NULL;
    }

    // Read commands.
    struct ptd_pcg_disk_command *encoded_cmds = NULL;
    if (header.commands_length > 0) {
        encoded_cmds = (struct ptd_pcg_disk_command *)
                malloc(header.commands_length * sizeof(*encoded_cmds));
        if (encoded_cmds == NULL) {
            snprintf((char *)ptd_err, sizeof(ptd_err),
                     "ptd_load: oom for encoded commands");
            fclose(fp);
            return NULL;
        }
        size_t n = header.commands_length * sizeof(*encoded_cmds);
        if (fread(encoded_cmds, 1, n, fp) != n) {
            snprintf((char *)ptd_err, sizeof(ptd_err),
                     "ptd_load: short commands read in %s", path);
            free(encoded_cmds);
            fclose(fp);
            return NULL;
        }
    }

    // Read flat mem.
    double *flat_mem = NULL;
    if (header.mem_total_doubles > 0) {
        flat_mem = (double *)malloc(header.mem_total_doubles * sizeof(double));
        if (flat_mem == NULL) {
            snprintf((char *)ptd_err, sizeof(ptd_err),
                     "ptd_load: oom for flat mem");
            free(encoded_cmds);
            fclose(fp);
            return NULL;
        }
        size_t n = header.mem_total_doubles * sizeof(double);
        if (fread(flat_mem, 1, n, fp) != n) {
            snprintf((char *)ptd_err, sizeof(ptd_err),
                     "ptd_load: short mem read in %s", path);
            free(flat_mem);
            free(encoded_cmds);
            fclose(fp);
            return NULL;
        }
    }

    // Read memr offsets.
    int64_t *memr_offsets = NULL;
    if (graph->vertices_length > 0) {
        memr_offsets = (int64_t *)malloc(graph->vertices_length * sizeof(int64_t));
        if (memr_offsets == NULL) {
            snprintf((char *)ptd_err, sizeof(ptd_err),
                     "ptd_load: oom for memr_offsets");
            free(flat_mem);
            free(encoded_cmds);
            fclose(fp);
            return NULL;
        }
        size_t n = graph->vertices_length * sizeof(int64_t);
        if (fread(memr_offsets, 1, n, fp) != n) {
            snprintf((char *)ptd_err, sizeof(ptd_err),
                     "ptd_load: short memr read in %s", path);
            free(memr_offsets);
            free(flat_mem);
            free(encoded_cmds);
            fclose(fp);
            return NULL;
        }
    }
    fclose(fp);

    // Wrap flat_mem in a single-node ll_of_a chain so the existing
    // destroy function works unchanged. The node owns the flat_mem
    // allocation; if we OOM building the wrapper, clean up.
    struct ll_of_a *mem_node = (struct ll_of_a *)malloc(sizeof(*mem_node));
    if (mem_node == NULL) {
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_load: oom for mem node");
        free(memr_offsets);
        free(flat_mem);
        free(encoded_cmds);
        return NULL;
    }
    mem_node->next = NULL;
    mem_node->mem = flat_mem;  // may be NULL when mem_total_doubles == 0
    mem_node->current_mem_index = (size_t)header.mem_total_doubles;
    mem_node->current_mem_position =
            (flat_mem != NULL) ? flat_mem + header.mem_total_doubles : NULL;

    // Decode commands: re-resolve pointers against flat_mem and graph.
    struct ptd_comp_graph_parameterized *commands = NULL;
    if (header.commands_length > 0) {
        commands = (struct ptd_comp_graph_parameterized *)
                malloc(header.commands_length * sizeof(*commands));
        if (commands == NULL) {
            snprintf((char *)ptd_err, sizeof(ptd_err),
                     "ptd_load: oom for commands");
            free(mem_node);
            free(memr_offsets);
            free(flat_mem);
            free(encoded_cmds);
            return NULL;
        }
        for (size_t i = 0; i < header.commands_length; i++) {
            commands[i].type = encoded_cmds[i].type;
            commands[i].from = (size_t)encoded_cmds[i].from;
            commands[i].to = (size_t)encoded_cmds[i].to;
            commands[i].multiplier = encoded_cmds[i].multiplier;
            commands[i].fromT = ptd_pcg_decode_ptr_impl(
                    &encoded_cmds[i].fromT, flat_mem, graph,
                    external_table, n_external);
            commands[i].toT = ptd_pcg_decode_ptr_impl(
                    &encoded_cmds[i].toT, flat_mem, graph,
                    external_table, n_external);
            commands[i].multiplierptr = ptd_pcg_decode_ptr_impl(
                    &encoded_cmds[i].multiplierptr, flat_mem, graph,
                    external_table, n_external);
            // Detect EXTERNAL pointers in a rev-2 file when no
            // external_table was supplied. v1 callers loading v2
            // files would otherwise silently get NULL pointers in
            // commands they need.
            if (header.format_revision >= 2u && external_table == NULL) {
                if (encoded_cmds[i].fromT.kind == PTD_PCG_PTR_EXTERNAL ||
                    encoded_cmds[i].toT.kind == PTD_PCG_PTR_EXTERNAL ||
                    encoded_cmds[i].multiplierptr.kind == PTD_PCG_PTR_EXTERNAL) {
                    snprintf((char *)ptd_err, sizeof(ptd_err),
                             "ptd_load: %s is rev 2 with EXTERNAL "
                             "pointers but no external_table was "
                             "supplied. Use ptd_load_parameterized_"
                             "reward_compute_graph_ex().", path);
                    free(commands);
                    free(encoded_cmds);
                    free(memr_offsets);
                    // mem_node owns flat_mem; free both via the node.
                    free(flat_mem);
                    free(mem_node);
                    return NULL;
                }
            }
            // Resolved-pointer liveness check: every command type
            // dereferences a specific subset of fromT / toT /
            // multiplierptr (see ptd_graph_build_ex_absorbation_time_
            // comp_graph_parameterized at the consume site). If the
            // encoder said the pointer is live (non-NULL kind) but the
            // decoder couldn't resolve it (out-of-range vertex/edge
            // index, mismatched topology), treat the whole file as a
            // cache miss instead of letting the consumer segfault on
            // *NULL. Two graphs can pass the topology-hash and
            // memr_length checks above and still disagree on edge
            // ordering — this is the catch.
            //
            // Liveness per command type (mirrors the consume switch):
            //   NEW_ADD: multiplierptr
            //   P:       fromT, toT
            //   PP:      fromT, toT, multiplierptr
            //   INV:     fromT
            //   ONE_MINUS: fromT
            //   DIVIDE:  fromT, toT
            //   ZERO:    fromT
            int t = commands[i].type;
            int need_from = (t == 1 /*P*/ || t == 3 /*PP*/ || t == 2 /*INV*/
                             || t == 4 /*ONE_MINUS*/ || t == 5 /*DIVIDE*/
                             || t == 6 /*ZERO*/);
            int need_to = (t == 1 /*P*/ || t == 3 /*PP*/ || t == 5 /*DIVIDE*/);
            int need_mptr = (t == 0 /*NEW_ADD*/ || t == 3 /*PP*/);
            const char *bad_field = NULL;
            if (need_from && commands[i].fromT == NULL
                    && encoded_cmds[i].fromT.kind != PTD_PCG_PTR_NULL) {
                bad_field = "fromT";
            } else if (need_to && commands[i].toT == NULL
                    && encoded_cmds[i].toT.kind != PTD_PCG_PTR_NULL) {
                bad_field = "toT";
            } else if (need_mptr && commands[i].multiplierptr == NULL
                    && encoded_cmds[i].multiplierptr.kind != PTD_PCG_PTR_NULL) {
                bad_field = "multiplierptr";
            }
            if (bad_field != NULL) {
                snprintf((char *)ptd_err, sizeof(ptd_err),
                         "ptd_load: %s command %zu (type=%d) %s "
                         "pointer failed to resolve against current "
                         "graph topology (likely stale cache from "
                         "a different graph build). Treating as "
                         "cache miss.",
                         path, i, t, bad_field);
                free(commands);
                free(encoded_cmds);
                free(memr_offsets);
                free(flat_mem);
                free(mem_node);
                return NULL;
            }
        }
    }
    free(encoded_cmds);

    // Decode memr.
    double **memr = NULL;
    if (graph->vertices_length > 0) {
        memr = (double **)malloc(graph->vertices_length * sizeof(double *));
        if (memr == NULL) {
            snprintf((char *)ptd_err, sizeof(ptd_err),
                     "ptd_load: oom for memr");
            free(commands);
            free(mem_node);
            free(memr_offsets);
            free(flat_mem);
            return NULL;
        }
        for (size_t i = 0; i < graph->vertices_length; i++) {
            memr[i] = (memr_offsets[i] >= 0) ? (flat_mem + memr_offsets[i]) : NULL;
        }
    }
    free(memr_offsets);

    struct ptd_desc_reward_compute_parameterized *res =
            (struct ptd_desc_reward_compute_parameterized *)malloc(sizeof(*res));
    if (res == NULL) {
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "ptd_load: oom for result struct");
        free(memr);
        free(commands);
        free(mem_node);
        free(flat_mem);
        return NULL;
    }
    res->length = (size_t)header.commands_length;
    res->commands = commands;
    res->mem = mem_node;
    res->memr = memr;
    return res;
}

// v1 entry point: backward-compatible signature. Loads a rev-1
// file. If passed a rev-2 file, this function succeeds only if
// the file contains no EXTERNAL pointers (e.g. an SCC with no
// external boundary); otherwise it returns NULL with ptd_err
// describing the mismatch. Existing callers (Stage A2 cache
// load in ptd_precompute_reward_compute_graph) keep working
// unchanged for rev-1 files.
struct ptd_desc_reward_compute_parameterized *
ptd_load_parameterized_reward_compute_graph(
        const char *path,
        const struct ptd_graph *graph)
{
    return ptd_load_parameterized_reward_compute_graph_impl(
            path, graph, NULL, 0);
}

// v2 entry point: loads either rev-1 or rev-2 files. EXTERNAL
// pointers in the file are resolved to &external_table[index].
// Caller owns external_table; it must outlive the returned
// compute graph.
struct ptd_desc_reward_compute_parameterized *
ptd_load_parameterized_reward_compute_graph_ex(
        const char *path,
        const struct ptd_graph *graph,
        const double *external_table,
        size_t n_external)
{
    return ptd_load_parameterized_reward_compute_graph_impl(
            path, graph, external_table, n_external);
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
    if (graph->parameterized_reward_compute_graph_off != NULL) {
        ptd_pcg_desc_off_destroy(graph->parameterized_reward_compute_graph_off);
    }
#ifdef PHASIC_B3_VALIDATORS
    if (graph->_dbg_off_clean != NULL) {  /* B3 validator stash; normally NULL */
        ptd_pcg_desc_off_destroy(
            (struct ptd_desc_reward_compute_parameterized_off *)graph->_dbg_off_clean);
    }
#endif

#ifdef HAVE_MPFR
    if (graph->reward_compute_graph_mpfr != NULL) {
        for (size_t i = 0; i < graph->reward_compute_graph_mpfr->length; i++) {
            if (graph->reward_compute_graph_mpfr->commands[i].multiplier_str != NULL) {
                free(graph->reward_compute_graph_mpfr->commands[i].multiplier_str);
            }
        }
        free(graph->reward_compute_graph_mpfr->commands);
        free(graph->reward_compute_graph_mpfr);
    }
#endif

    if (graph->elimination_trace != NULL) {
        ptd_elimination_trace_destroy(graph->elimination_trace);
    }

    if (graph->current_params != NULL) {
        free(graph->current_params);
    }

    if (graph->weight_tape != NULL) {
        ptd_weight_tape_destroy(graph->weight_tape);
        graph->weight_tape = NULL;
    }
    if (graph->wf_residuals != NULL) {
        for (size_t ri = 0; ri < graph->wf_residuals_length; ri++) {
            ptd_weight_tape_destroy(graph->wf_residuals[ri]);
        }
        free(graph->wf_residuals);
        graph->wf_residuals = NULL;
        graph->wf_residuals_length = 0;
        graph->wf_residuals_for_tape = NULL;
    }

    graph->reward_compute_graph = NULL;
    graph->parameterized_reward_compute_graph = NULL;
#ifdef HAVE_MPFR
    graph->reward_compute_graph_mpfr = NULL;
#endif
    graph->elimination_trace = NULL;
    graph->current_params = NULL;
    /* Destroy the mutex BEFORE the memset zeros the struct — destroying
     * a mutex that's been overwritten with zeros is undefined. */
    if (graph->compute_graph_lock_initialized) {
        PTD_MUTEX_DESTROY(&graph->compute_graph_lock);
        graph->compute_graph_lock_initialized = false;
    }
    memset(graph, 0, sizeof(*graph));
    free(graph);
}

struct ptd_vertex *ptd_vertex_create(struct ptd_graph *graph) {
    int *state = (int *) calloc(graph->state_length, sizeof(*state));

    struct ptd_vertex *vertex = ptd_vertex_create_state(graph, state);

    // Free the temporary state since ptd_vertex_create_state() makes a copy
    free(state);

    return vertex;
}

struct ptd_vertex *ptd_vertex_create_state(struct ptd_graph *graph, int *state) {
    struct ptd_vertex *vertex = (struct ptd_vertex *) malloc(sizeof(*vertex));
    vertex->graph = graph;
    vertex->edges_length = 0;

    // ALWAYS copy the state to avoid shared ownership issues
    // This prevents bugs where Python passes a pointer it doesn't own
    int *state_copy = (int *)malloc(graph->state_length * sizeof(int));
    if (state_copy == NULL) {
        free(vertex);
        return NULL;
    }
    memcpy(state_copy, state, graph->state_length * sizeof(int));
    vertex->state = state_copy;

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
        struct ptd_edge *edge = vertex->edges[i];
        if (edge->should_free_coefficients && edge->coefficients != NULL) {
            free(edge->coefficients);
        }
        free(edge);
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
        double *coefficients,
        size_t coefficients_length
) {
    if (coefficients == NULL || coefficients_length == 0) {
        snprintf((char*)ptd_err, sizeof(ptd_err),
            "ptd_graph_add_edge: coefficients cannot be NULL or empty");
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

    // VALIDATION: Check consistency with existing edges
    if (from->graph->param_length_locked) {
        // Allow edges with MORE coefficients than param_length
        // Only the first param_length coefficients will be used
        if (from->index != from->graph->starting_vertex->index && coefficients_length < from->graph->param_length) {
            snprintf((char*)ptd_err, sizeof(ptd_err),
                "Edge has too few coefficients: graph expects at least %zu, got %zu. "
                "Edges must have coefficients_length >= param_length.",
                (size_t)from->graph->param_length,
                (size_t)coefficients_length);
            return NULL;
        }
    } else if (from->index != from->graph->starting_vertex->index) {
        // First edge: set graph mode (if not explicitly set via set_param_length)
        from->graph->param_length = coefficients_length;
        from->graph->param_length_locked = true;
    }

    // NOTE: Edge mode locking is now handled in C++ layer (phasiccpp.cpp)
    // The C++ add_edge() and add_edge_parameterized() methods set graph->edge_mode
    // before calling this function.

    // Set parameterized flag based on edge_mode
    if (from->graph->edge_mode == PTD_EDGE_MODE_PARAMETERIZED) {
        from->graph->parameterized = true;
    } else if (from->graph->edge_mode == PTD_EDGE_MODE_CONSTANT) {
        from->graph->parameterized = false;
    }

    // Create edge
    struct ptd_edge *edge = (struct ptd_edge *)malloc(sizeof(*edge));
    if (edge == NULL) {
        snprintf((char*)ptd_err, sizeof(ptd_err), "Failed to allocate edge");
        return NULL;
    }

    edge->to = to;
    edge->coefficients_length = coefficients_length;
    edge->coefficients = (double *)malloc(coefficients_length * sizeof(double));
    if (edge->coefficients == NULL) {
        free(edge);
        snprintf((char*)ptd_err, sizeof(ptd_err), "Failed to allocate edge coefficients");
        return NULL;
    }

    memcpy(edge->coefficients, coefficients, coefficients_length * sizeof(double));
    edge->should_free_coefficients = true;

    // Compute initial weight with default params (theta=[1,1,...])
    // Only use first min(coefficients_length, param_length) coefficients
    // BUT: if param_length not yet set (IPV edges), use all coefficients
    edge->weight = 0.0;
    size_t n_coeffs_for_init;
    if (from->graph->param_length_locked) {
        // param_length is set, use min(coefficients_length, param_length)
        n_coeffs_for_init = coefficients_length < from->graph->param_length ?
                            coefficients_length : from->graph->param_length;
    } else {
        // param_length not set yet (IPV edges), use all coefficients
        n_coeffs_for_init = coefficients_length;
    }
    for (size_t i = 0; i < n_coeffs_for_init; i++) {
        edge->weight += coefficients[i] * 1.0;
    }

    // Validate weight is positive
    if (edge->weight < 0) {
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
                "Edge weight evaluates to non-positive value '%f'. Vertex index %i%s (state %s). Weight must be strictly larger than 0.\n",
                edge->weight, (int) debug_index, starting_vertex, state
        );

        free(edge->coefficients);
        free(edge);
        return NULL;
    }

    // Add to vertex using directed graph API
    ptd_directed_graph_add_edge(
            (struct ptd_directed_vertex *) from,
            (struct ptd_directed_edge *) edge
    );

    // Invalidate cached compute graphs
    if (from->graph->reward_compute_graph != NULL) {
        free(from->graph->reward_compute_graph->commands);
        free(from->graph->reward_compute_graph);
    }

    if (from->graph->parameterized_reward_compute_graph != NULL) {
        ptd_parameterized_reward_compute_graph_destroy(
                from->graph->parameterized_reward_compute_graph
        );
    }
    if (from->graph->parameterized_reward_compute_graph_off != NULL) {
        ptd_pcg_desc_off_destroy(from->graph->parameterized_reward_compute_graph_off);
    }

    from->graph->reward_compute_graph = NULL;
    from->graph->parameterized_reward_compute_graph = NULL;
    from->graph->parameterized_reward_compute_graph_off = NULL;

#ifdef HAVE_MPFR
    if (from->graph->reward_compute_graph_mpfr != NULL) {
        for (size_t i = 0; i < from->graph->reward_compute_graph_mpfr->length; i++) {
            free(from->graph->reward_compute_graph_mpfr->commands[i].multiplier_str);
        }
        free(from->graph->reward_compute_graph_mpfr->commands);
        free(from->graph->reward_compute_graph_mpfr);
        from->graph->reward_compute_graph_mpfr = NULL;
    }
#endif

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
    // Update weight
    edge->weight = weight;

    // UNIFIED INTERFACE: Also update coefficients for constant edges
    // For constant edges (coefficients_length == 1), maintain invariant: coefficients[0] == weight
    // For parameterized edges (coefficients_length > 1), this function shouldn't be used -
    // use ptd_graph_update_weights() instead
    if (edge->coefficients_length == 1) {
        edge->coefficients[0] = weight;
    }

    edge->to->graph->weight_version++;

    if (edge->to->graph->reward_compute_graph != NULL) {
        free(edge->to->graph->reward_compute_graph->commands);
        edge->to->graph->reward_compute_graph = NULL;
    }
}

void ptd_edge_update_to(
    struct ptd_edge *edge,
    struct ptd_vertex *vertex
) {

edge->to->graph->weight_version++;

if (edge->to->graph->reward_compute_graph != NULL) {
    free(edge->to->graph->reward_compute_graph->commands);
    edge->to->graph->reward_compute_graph = NULL;
}

edge->to = vertex;

}

/**
 * Set the parameter length for a graph before adding edges
 */
void ptd_graph_set_param_length(
        struct ptd_graph *graph,
        size_t param_length
) {
    if (graph == NULL) {
        snprintf((char*)ptd_err, sizeof(ptd_err), "Graph is NULL");
        return;
    }

    if (graph->param_length_locked) {
        snprintf((char*)ptd_err, sizeof(ptd_err),
            "Cannot set param_length: graph already has edges added (param_length=%zu). "
            "Call ptd_graph_set_param_length() before adding any non-IPV edges.",
            (size_t)graph->param_length);
        return;
    }

    if (param_length == 0) {
        snprintf((char*)ptd_err, sizeof(ptd_err),
            "param_length must be > 0");
        return;
    }

    graph->param_length = param_length;
    graph->param_length_locked = true;
    graph->parameterized = (param_length > 1);
}


/* ===========================================================================
 * Per-edge weight formula tape (weight_mode='formula')
 * ===========================================================================
 *
 * A flat stack-machine that computes weight = f(theta, edge coefficients) for
 * an arbitrary closed-form formula compiled in Python (weight_formula.py).
 * Opcodes MUST stay in lock-step with weight_formula.OPCODES. The tape is
 * read-only at eval time, so it is safe under the FFI's OpenMP parallelism
 * (each thread evaluates with its own theta/coeff and its own on-stack VM).
 */
enum ptd_wf_opcode {
    PTD_WF_PUSH_THETA = 0, PTD_WF_PUSH_COEFF = 1, PTD_WF_PUSH_CONST = 2,
    PTD_WF_ADD = 3, PTD_WF_SUB = 4, PTD_WF_MUL = 5, PTD_WF_DIV = 6,
    PTD_WF_POW = 7, PTD_WF_NEG = 8,
    PTD_WF_EXP = 9, PTD_WF_LOG = 10, PTD_WF_SQRT = 11, PTD_WF_LOGISTIC = 12,
    PTD_WF_EQ = 13, PTD_WF_NE = 14, PTD_WF_LT = 15, PTD_WF_GT = 16,
    PTD_WF_LE = 17, PTD_WF_GE = 18,
    PTD_WF_AND = 19, PTD_WF_OR = 20, PTD_WF_NOT = 21, PTD_WF_SELECT = 22
};

struct ptd_weight_tape {
    int *ops;
    size_t ops_length;
    double *consts;
    size_t consts_length;
    size_t stack_depth;
    size_t n_theta;
    size_t n_coeff;
};

#define PTD_WF_MAX_STACK 256

/* Numerically stable logistic; mirrors weight_formula._logistic. */
static double ptd_wf_logistic(double a) {
    if (a >= 0.0) {
        double z = exp(-a);
        return 1.0 / (1.0 + z);
    }
    double z = exp(a);
    return z / (1.0 + z);
}

int ptd_weight_tape_eval_arrays(
        const int *ops, size_t ops_length,
        const double *consts, size_t consts_length,
        size_t stack_depth,
        const double *theta, size_t theta_len,
        const double *coeff, size_t coeff_len,
        double *out_weight
) {
    double stack[PTD_WF_MAX_STACK];
    size_t sp = 0;
    size_t i = 0;

    if (stack_depth > PTD_WF_MAX_STACK) {
        snprintf((char *) ptd_err, sizeof(ptd_err),
                 "weight_formula tape stack depth %zu exceeds limit %d",
                 stack_depth, PTD_WF_MAX_STACK);
        return 1;
    }

    /* Helper guards. NEED(k): require k operands; PUSH1(v): push with overflow
     * guard. Both set ptd_err and 'goto fail' on violation. */
#define WF_NEED(k) do { if (sp < (size_t)(k)) { \
        snprintf((char *) ptd_err, sizeof(ptd_err), \
            "weight_formula tape stack underflow"); return 1; } } while (0)
#define WF_PUSH(v) do { if (sp >= PTD_WF_MAX_STACK) { \
        snprintf((char *) ptd_err, sizeof(ptd_err), \
            "weight_formula tape stack overflow"); return 1; } \
        stack[sp++] = (v); } while (0)

    while (i < ops_length) {
        int op = ops[i++];
        switch (op) {
            case PTD_WF_PUSH_THETA: {
                int idx = ops[i++];
                if (idx < 0 || (size_t) idx >= theta_len) {
                    snprintf((char *) ptd_err, sizeof(ptd_err),
                        "weight_formula references t%d but only %zu parameters "
                        "are available", idx, theta_len);
                    return 1;
                }
                WF_PUSH(theta[idx]);
                break;
            }
            case PTD_WF_PUSH_COEFF: {
                int idx = ops[i++];
                if (idx < 0 || (size_t) idx >= coeff_len) {
                    snprintf((char *) ptd_err, sizeof(ptd_err),
                        "weight_formula references c%d but the edge has only "
                        "%zu coefficients", idx, coeff_len);
                    return 1;
                }
                WF_PUSH(coeff[idx]);
                break;
            }
            case PTD_WF_PUSH_CONST: {
                int idx = ops[i++];
                if (idx < 0 || (size_t) idx >= consts_length) {
                    snprintf((char *) ptd_err, sizeof(ptd_err),
                        "weight_formula const index %d out of range", idx);
                    return 1;
                }
                WF_PUSH(consts[idx]);
                break;
            }
            case PTD_WF_ADD: { WF_NEED(2); double b = stack[--sp], a = stack[--sp]; WF_PUSH(a + b); break; }
            case PTD_WF_SUB: { WF_NEED(2); double b = stack[--sp], a = stack[--sp]; WF_PUSH(a - b); break; }
            case PTD_WF_MUL: { WF_NEED(2); double b = stack[--sp], a = stack[--sp]; WF_PUSH(a * b); break; }
            case PTD_WF_DIV: { WF_NEED(2); double b = stack[--sp], a = stack[--sp]; WF_PUSH(a / b); break; }
            case PTD_WF_POW: { WF_NEED(2); double b = stack[--sp], a = stack[--sp]; WF_PUSH(pow(a, b)); break; }
            case PTD_WF_NEG: { WF_NEED(1); double a = stack[--sp]; WF_PUSH(-a); break; }
            case PTD_WF_EXP: { WF_NEED(1); double a = stack[--sp]; WF_PUSH(exp(a)); break; }
            case PTD_WF_LOG: { WF_NEED(1); double a = stack[--sp]; WF_PUSH(log(a)); break; }
            case PTD_WF_SQRT: { WF_NEED(1); double a = stack[--sp]; WF_PUSH(sqrt(a)); break; }
            case PTD_WF_LOGISTIC: { WF_NEED(1); double a = stack[--sp]; WF_PUSH(ptd_wf_logistic(a)); break; }
            case PTD_WF_EQ: { WF_NEED(2); double b = stack[--sp], a = stack[--sp]; WF_PUSH(a == b ? 1.0 : 0.0); break; }
            case PTD_WF_NE: { WF_NEED(2); double b = stack[--sp], a = stack[--sp]; WF_PUSH(a != b ? 1.0 : 0.0); break; }
            case PTD_WF_LT: { WF_NEED(2); double b = stack[--sp], a = stack[--sp]; WF_PUSH(a < b ? 1.0 : 0.0); break; }
            case PTD_WF_GT: { WF_NEED(2); double b = stack[--sp], a = stack[--sp]; WF_PUSH(a > b ? 1.0 : 0.0); break; }
            case PTD_WF_LE: { WF_NEED(2); double b = stack[--sp], a = stack[--sp]; WF_PUSH(a <= b ? 1.0 : 0.0); break; }
            case PTD_WF_GE: { WF_NEED(2); double b = stack[--sp], a = stack[--sp]; WF_PUSH(a >= b ? 1.0 : 0.0); break; }
            case PTD_WF_AND: { WF_NEED(2); double b = stack[--sp], a = stack[--sp]; WF_PUSH((a != 0.0 && b != 0.0) ? 1.0 : 0.0); break; }
            case PTD_WF_OR: { WF_NEED(2); double b = stack[--sp], a = stack[--sp]; WF_PUSH((a != 0.0 || b != 0.0) ? 1.0 : 0.0); break; }
            case PTD_WF_NOT: { WF_NEED(1); double a = stack[--sp]; WF_PUSH(a == 0.0 ? 1.0 : 0.0); break; }
            case PTD_WF_SELECT: {
                WF_NEED(3);
                double b = stack[--sp], a = stack[--sp], c = stack[--sp];
                WF_PUSH(c != 0.0 ? a : b);
                break;
            }
            default:
                snprintf((char *) ptd_err, sizeof(ptd_err),
                         "weight_formula tape: unknown opcode %d", op);
                return 1;
        }
    }

#undef WF_NEED
#undef WF_PUSH

    if (sp != 1) {
        snprintf((char *) ptd_err, sizeof(ptd_err),
                 "weight_formula tape left %zu values on the stack (expected 1)",
                 sp);
        return 1;
    }
    *out_weight = stack[0];
    return 0;
}

int ptd_weight_tape_eval(
        const struct ptd_weight_tape *tape,
        const double *theta, size_t theta_len,
        const double *coeff, size_t coeff_len,
        double *out_weight
) {
    return ptd_weight_tape_eval_arrays(
            tape->ops, tape->ops_length, tape->consts, tape->consts_length,
            tape->stack_depth, theta, theta_len, coeff, coeff_len, out_weight);
}

struct ptd_weight_tape *ptd_weight_tape_create(
        const int *ops, size_t ops_length,
        const double *consts, size_t consts_length,
        size_t stack_depth, size_t n_theta, size_t n_coeff
) {
    struct ptd_weight_tape *t =
            (struct ptd_weight_tape *) malloc(sizeof(*t));
    if (t == NULL) {
        return NULL;
    }
    t->ops = NULL;
    t->consts = NULL;
    t->ops_length = ops_length;
    t->consts_length = consts_length;
    t->stack_depth = stack_depth;
    t->n_theta = n_theta;
    t->n_coeff = n_coeff;

    if (ops_length > 0) {
        t->ops = (int *) malloc(ops_length * sizeof(int));
        if (t->ops == NULL) {
            free(t);
            return NULL;
        }
        memcpy(t->ops, ops, ops_length * sizeof(int));
    }
    if (consts_length > 0) {
        t->consts = (double *) malloc(consts_length * sizeof(double));
        if (t->consts == NULL) {
            free(t->ops);
            free(t);
            return NULL;
        }
        memcpy(t->consts, consts, consts_length * sizeof(double));
    }
    return t;
}

void ptd_weight_tape_destroy(struct ptd_weight_tape *tape) {
    if (tape == NULL) {
        return;
    }
    free(tape->ops);
    free(tape->consts);
    free(tape);
}

/* --- weight_formula per-edge partial evaluation (constant folding) -------- *
 * Specialize a tape for ONE edge's coefficients: fold every theta-INDEPENDENT
 * subexpression to a constant and prune dead select() arms, producing a small
 * residual tape that depends on theta only (no PUSH_COEFF / comparison /
 * boolean / select ops remain). Because the folded parts are theta-independent
 * by construction (the formula rules forbid theta-dependent conditions), the
 * residual is bit-identical to the full tape for every theta. Evaluated per-
 * theta in place of the full tape, this removes the dominant cost of complex
 * formulas (e.g. a sum of select() dispatches evaluates only the taken arm,
 * with its coefficient arithmetic already folded). Returns NULL on allocation
 * failure or an unexpected theta-dependent condition; the caller then falls
 * back to evaluating the full tape (same result, just slower). */
struct ptd_wf_ai {        /* abstract-interpretation stack entry */
    int is_const;
    double cval;          /* valid when is_const */
    int *rops;            /* residual ops reproducing the value from theta */
    size_t rlen, rcap;
};

static int ptd_wf_ai_push(struct ptd_wf_ai *e, int op) {
    if (e->rlen >= e->rcap) {
        size_t nc = e->rcap ? e->rcap * 2 : 8;
        int *p = (int *) realloc(e->rops, nc * sizeof(int));
        if (p == NULL) return 1;
        e->rops = p; e->rcap = nc;
    }
    e->rops[e->rlen++] = op;
    return 0;
}

static long ptd_wf_const_ix(double **consts, size_t *len, size_t *cap, double v) {
    for (size_t i = 0; i < *len; i++) if ((*consts)[i] == v) return (long) i;
    if (*len >= *cap) {
        size_t nc = *cap ? *cap * 2 : 8;
        double *p = (double *) realloc(*consts, nc * sizeof(double));
        if (p == NULL) return -1;
        *consts = p; *cap = nc;
    }
    (*consts)[*len] = v;
    return (long) (*len)++;
}

/* Append the ops that reproduce entry e into dst (a constant becomes PUSH_CONST). */
static int ptd_wf_emit(struct ptd_wf_ai *dst, const struct ptd_wf_ai *e,
                       double **consts, size_t *clen, size_t *ccap) {
    if (e->is_const) {
        long ci = ptd_wf_const_ix(consts, clen, ccap, e->cval);
        if (ci < 0) return 1;
        if (ptd_wf_ai_push(dst, PTD_WF_PUSH_CONST)) return 1;
        if (ptd_wf_ai_push(dst, (int) ci)) return 1;
    } else {
        for (size_t i = 0; i < e->rlen; i++)
            if (ptd_wf_ai_push(dst, e->rops[i])) return 1;
    }
    return 0;
}

struct ptd_weight_tape *ptd_weight_tape_specialize(
        const struct ptd_weight_tape *tape,
        const double *coeff, size_t coeff_len) {
    if (tape == NULL) return NULL;
    struct ptd_wf_ai stack[PTD_WF_MAX_STACK];
    size_t sp = 0;
    double *rc = NULL; size_t rclen = 0, rccap = 0;
    const int *ops = tape->ops; size_t n = tape->ops_length, i = 0;

#define AI_FAIL() do { goto fail; } while (0)
#define AI_PUSHC(v) do { if (sp >= PTD_WF_MAX_STACK) AI_FAIL(); \
        stack[sp].is_const = 1; stack[sp].cval = (v); \
        stack[sp].rops = NULL; stack[sp].rlen = stack[sp].rcap = 0; sp++; } while (0)

    while (i < n) {
        int op = ops[i++];
        switch (op) {
        case PTD_WF_PUSH_THETA: {
            int idx = ops[i++];
            if (sp >= PTD_WF_MAX_STACK) AI_FAIL();
            struct ptd_wf_ai *e = &stack[sp++];
            e->is_const = 0; e->cval = 0; e->rops = NULL; e->rlen = e->rcap = 0;
            if (ptd_wf_ai_push(e, PTD_WF_PUSH_THETA) || ptd_wf_ai_push(e, idx)) { sp--; free(e->rops); AI_FAIL(); }
            break;
        }
        case PTD_WF_PUSH_COEFF: {
            int idx = ops[i++];
            if (idx < 0 || (size_t) idx >= coeff_len) AI_FAIL();
            AI_PUSHC(coeff[idx]);
            break;
        }
        case PTD_WF_PUSH_CONST: {
            int idx = ops[i++];
            if (idx < 0 || (size_t) idx >= tape->consts_length) AI_FAIL();
            AI_PUSHC(tape->consts[idx]);
            break;
        }
        case PTD_WF_ADD: case PTD_WF_SUB: case PTD_WF_MUL:
        case PTD_WF_DIV: case PTD_WF_POW: {
            if (sp < 2) AI_FAIL();
            struct ptd_wf_ai b = stack[--sp], a = stack[--sp];
            if (a.is_const && b.is_const) {
                double r = (op == PTD_WF_ADD) ? a.cval + b.cval :
                           (op == PTD_WF_SUB) ? a.cval - b.cval :
                           (op == PTD_WF_MUL) ? a.cval * b.cval :
                           (op == PTD_WF_DIV) ? a.cval / b.cval : pow(a.cval, b.cval);
                AI_PUSHC(r);
            } else {
                struct ptd_wf_ai e; e.is_const = 0; e.cval = 0; e.rops = NULL; e.rlen = e.rcap = 0;
                if (ptd_wf_emit(&e, &a, &rc, &rclen, &rccap) ||
                    ptd_wf_emit(&e, &b, &rc, &rclen, &rccap) ||
                    ptd_wf_ai_push(&e, op)) { free(a.rops); free(b.rops); free(e.rops); AI_FAIL(); }
                free(a.rops); free(b.rops);
                if (sp >= PTD_WF_MAX_STACK) { free(e.rops); AI_FAIL(); }
                stack[sp++] = e;
            }
            break;
        }
        case PTD_WF_NEG: case PTD_WF_EXP: case PTD_WF_LOG:
        case PTD_WF_SQRT: case PTD_WF_LOGISTIC: {
            if (sp < 1) AI_FAIL();
            struct ptd_wf_ai a = stack[--sp];
            if (a.is_const) {
                double r = (op == PTD_WF_NEG) ? -a.cval :
                           (op == PTD_WF_EXP) ? exp(a.cval) :
                           (op == PTD_WF_LOG) ? log(a.cval) :
                           (op == PTD_WF_SQRT) ? sqrt(a.cval) : ptd_wf_logistic(a.cval);
                AI_PUSHC(r);
            } else {
                if (ptd_wf_ai_push(&a, op)) { free(a.rops); AI_FAIL(); }
                if (sp >= PTD_WF_MAX_STACK) { free(a.rops); AI_FAIL(); }
                stack[sp++] = a;   /* reuse a's residual ops */
            }
            break;
        }
        case PTD_WF_EQ: case PTD_WF_NE: case PTD_WF_LT: case PTD_WF_GT:
        case PTD_WF_LE: case PTD_WF_GE: case PTD_WF_AND: case PTD_WF_OR: {
            if (sp < 2) AI_FAIL();
            struct ptd_wf_ai b = stack[--sp], a = stack[--sp];
            if (!a.is_const || !b.is_const) { free(a.rops); free(b.rops); AI_FAIL(); }
            int r = (op == PTD_WF_EQ) ? (a.cval == b.cval) :
                    (op == PTD_WF_NE) ? (a.cval != b.cval) :
                    (op == PTD_WF_LT) ? (a.cval < b.cval) :
                    (op == PTD_WF_GT) ? (a.cval > b.cval) :
                    (op == PTD_WF_LE) ? (a.cval <= b.cval) :
                    (op == PTD_WF_GE) ? (a.cval >= b.cval) :
                    (op == PTD_WF_AND) ? (a.cval != 0.0 && b.cval != 0.0) :
                                         (a.cval != 0.0 || b.cval != 0.0);
            AI_PUSHC(r ? 1.0 : 0.0);
            break;
        }
        case PTD_WF_NOT: {
            if (sp < 1) AI_FAIL();
            struct ptd_wf_ai a = stack[--sp];
            if (!a.is_const) { free(a.rops); AI_FAIL(); }
            AI_PUSHC(a.cval == 0.0 ? 1.0 : 0.0);
            break;
        }
        case PTD_WF_SELECT: {
            if (sp < 3) AI_FAIL();
            struct ptd_wf_ai b = stack[--sp], a = stack[--sp], c = stack[--sp];
            if (!c.is_const) { free(a.rops); free(b.rops); free(c.rops); AI_FAIL(); }
            /* condition is a known constant -> keep one arm, discard the other */
            if (c.cval != 0.0) { free(b.rops); stack[sp++] = a; }
            else               { free(a.rops); stack[sp++] = b; }
            break;
        }
        default: AI_FAIL();
        }
    }
    if (sp != 1) AI_FAIL();

    {
        struct ptd_wf_ai top = stack[0];
        struct ptd_weight_tape *res;
        if (top.is_const) {
            int rops2[2] = { PTD_WF_PUSH_CONST, 0 };
            double cv = top.cval;
            res = ptd_weight_tape_create(rops2, 2, &cv, 1, 1, 0, 0);
        } else {
            /* The residual is a subset of the original tape, so the original
             * stack_depth is a safe upper bound for the VM's bounds check. */
            res = ptd_weight_tape_create(top.rops, top.rlen,
                    rc, rclen, tape->stack_depth, tape->n_theta, 0);
        }
        free(top.rops);
        free(rc);
        return res;   /* may be NULL on allocation failure -> caller falls back */
    }

fail:
    for (size_t s = 0; s < sp; s++) free(stack[s].rops);
    free(rc);
    return NULL;
#undef AI_FAIL
#undef AI_PUSHC
}

static void ptd_graph_free_wf_residuals(struct ptd_graph *graph) {
    if (graph->wf_residuals != NULL) {
        for (size_t i = 0; i < graph->wf_residuals_length; i++) {
            ptd_weight_tape_destroy(graph->wf_residuals[i]);
        }
        free(graph->wf_residuals);
    }
    graph->wf_residuals = NULL;
    graph->wf_residuals_length = 0;
    graph->wf_residuals_for_tape = NULL;
}

/* Build the per-edge residual tapes for the current weight_tape. Iterates edges
 * in the SAME order as ptd_graph_update_weights (skip starting vertex + edges
 * with no coefficients) so residual[k] lines up with the k-th tape-evaluated
 * edge. On any failure leaves wf_residuals == NULL; the caller then evaluates
 * the full tape (correct, just slower). */
static void ptd_graph_build_wf_residuals(struct ptd_graph *graph) {
    ptd_graph_free_wf_residuals(graph);
    if (graph->weight_tape == NULL) return;
    size_t count = 0;
    for (size_t i = 0; i < graph->vertices_length; i++) {
        struct ptd_vertex *v = graph->vertices[i];
        if (v == graph->starting_vertex) continue;
        for (size_t j = 0; j < v->edges_length; j++)
            if (v->edges[j]->coefficients_length > 0) count++;
    }
    graph->wf_residuals_for_tape = graph->weight_tape;
    if (count == 0) return;
    struct ptd_weight_tape **arr =
        (struct ptd_weight_tape **) calloc(count, sizeof(*arr));
    if (arr == NULL) { graph->wf_residuals_for_tape = NULL; return; }
    size_t k = 0; int ok = 1;
    for (size_t i = 0; i < graph->vertices_length && ok; i++) {
        struct ptd_vertex *v = graph->vertices[i];
        if (v == graph->starting_vertex) continue;
        for (size_t j = 0; j < v->edges_length; j++) {
            struct ptd_edge *e = v->edges[j];
            if (e->coefficients_length == 0) continue;
            struct ptd_weight_tape *r = ptd_weight_tape_specialize(
                graph->weight_tape, e->coefficients, e->coefficients_length);
            if (r == NULL) { ok = 0; break; }
            arr[k++] = r;
        }
    }
    if (!ok) {
        for (size_t x = 0; x < k; x++) ptd_weight_tape_destroy(arr[x]);
        free(arr);
        graph->wf_residuals_for_tape = NULL;   /* fall back to full tape */
        return;
    }
    graph->wf_residuals = arr;
    graph->wf_residuals_length = count;
}

void ptd_graph_set_weight_tape(
        struct ptd_graph *graph, struct ptd_weight_tape *tape
) {
    if (graph->weight_tape != NULL) {
        ptd_weight_tape_destroy(graph->weight_tape);
    }
    graph->weight_tape = tape;
    /* Invalidate any cached per-edge residuals; rebuilt lazily for the new tape. */
    ptd_graph_free_wf_residuals(graph);
}


void ptd_graph_update_weights(
        struct ptd_graph *graph,
        double *params,
        size_t params_length,
        bool use_log
) {
    // VALIDATION: Ensure graph has edges
    if (graph->edge_mode == PTD_EDGE_MODE_UNLOCKED) {
        snprintf((char*)ptd_err, sizeof(ptd_err),
            "Cannot call update_weights() on empty graph (no edges added yet). "
            "Add edges using add_edge() before calling update_weights().");
        return;
    }

    // VALIDATION: Ensure graph is parameterized (not constant)
    if (graph->edge_mode == PTD_EDGE_MODE_CONSTANT) {
        snprintf((char*)ptd_err, sizeof(ptd_err),
            "Cannot call update_weights() on constant graph. "
            "Graph has constant edges (created with scalar syntax: add_edge(v, 3.0)). "
            "Use parameterized edges (array syntax: add_edge(v, [3.0])) if you need to update weights.");
        return;
    }

    double *theta;
    size_t theta_len;
    bool need_free = false;

    if (params == NULL || params_length == 0) {
        // Use default theta = [1, 1, ..., 1]
        theta_len = graph->param_length;
        if (theta_len == 0) {
            // No edges yet, nothing to do
            return;
        }
        theta = (double *)malloc(theta_len * sizeof(double));
        if (theta == NULL) {
            snprintf((char*)ptd_err, sizeof(ptd_err), "Failed to allocate default parameters");
            return;
        }
        for (size_t i = 0; i < theta_len; i++) {
            theta[i] = 1.0;
        }
        need_free = true;
    } else {
        // Validate parameter length
        if (params_length != graph->param_length) {
            snprintf((char*)ptd_err, sizeof(ptd_err),
                "Parameter length mismatch: graph expects %zu parameters, got %zu",
                (size_t)graph->param_length,
                (size_t)params_length);
            return;
        }

        // // Validate no NaN or Inf parameters
        // for (size_t i = 0; i < params_length; i++) {
        //     if (isnan(params[i])) {
        //         snprintf((char*)ptd_err, sizeof(ptd_err),
        //             "Invalid parameter value: params[%zu] is NaN. "
        //             "SVGD or optimization may have diverged.",
        //             (unsigned long)i);
        //         PTD_LOG_ERROR("update_weights: params[%zu] is NaN", i);
        //         return;
        //     }
        //     if (isinf(params[i])) {
        //         snprintf((char*)ptd_err, sizeof(ptd_err),
        //             "Invalid parameter value: params[%zu] is Inf. "
        //             "SVGD or optimization may have diverged.",
        //             (unsigned long)i);
        //         PTD_LOG_ERROR("update_weights: params[%zu] is Inf", i);
        //         return;
        //     }
        // }

        theta = params;
        theta_len = params_length;
    }

    // Store current parameters
    if (graph->current_params == NULL && theta_len > 0) {
        graph->current_params = (double *)malloc(theta_len * sizeof(double));
    }
    if (graph->current_params != NULL && theta_len > 0) {
        memcpy(graph->current_params, theta, theta_len * sizeof(double));
    }

    // NOTE: Trace recording removed - it was causing memory explosion on large graphs.
    // The C-side trace (graph->elimination_trace) was never used after being recorded.
    // Python handles trace computation via hierarchical_trace_cache when needed for
    // moments/expectation computation.

    // Formula mode: build/refresh the per-edge specialized residual tapes once
    // for this tape (theta-independent constant folding + dead select-arm
    // pruning), then evaluate the small theta-only residuals per theta below.
    // wf_k indexes residuals in the SAME edge order as the loop builds them.
    if (graph->weight_tape != NULL &&
        graph->wf_residuals_for_tape != graph->weight_tape) {
        ptd_graph_build_wf_residuals(graph);
    }
    size_t wf_k = 0;

    // Update all edge weights using direct computation
    for (size_t i = 0; i < graph->vertices_length; i++) {
        struct ptd_vertex *vertex = graph->vertices[i];

        // Skip starting vertex edges - they should never be rescaled
        // Starting vertex edges represent the initial probability vector (IPV)
        // and must remain constant regardless of parameter values
        if (vertex == graph->starting_vertex) {
            continue;
        }

        for (size_t j = 0; j < vertex->edges_length; j++) {
            struct ptd_edge *edge = vertex->edges[j];

            // Skip edges with no coefficients (pure constant, like aux→parent edges)
            // These edges have hardcoded weights and should never be rescaled
            if (edge->coefficients_length == 0) {
                continue;
            }

            // Formula mode: evaluate the compiled tape over (theta, edge
            // coefficients) entirely in C. This bypasses the strict
            // coefficient-length check below — the tape does its own index
            // bounds-checking, like callback mode tolerates auxiliary
            // coefficients of a different length than theta.
            if (graph->weight_tape != NULL) {
                double w;
                // Prefer this edge's specialized residual tape (theta-only, with
                // coefficient arithmetic folded and dead select() arms pruned);
                // fall back to the full tape if residuals were not built. Both
                // yield identical weights. wf_k advances for every tape edge so
                // it stays aligned with the residual build order.
                const struct ptd_weight_tape *etape = graph->weight_tape;
                const double *ecoeff = edge->coefficients;
                size_t ecoeff_len = edge->coefficients_length;
                if (graph->wf_residuals != NULL &&
                    wf_k < graph->wf_residuals_length) {
                    etape = graph->wf_residuals[wf_k];
                    ecoeff = NULL;
                    ecoeff_len = 0;
                }
                wf_k++;
                if (ptd_weight_tape_eval(etape, theta, theta_len,
                                         ecoeff, ecoeff_len, &w) != 0) {
                    if (need_free) {
                        free(theta);
                    }
                    return;  // ptd_err set by eval
                }
                if (!isfinite(w) || w < 0.0) {
                    snprintf((char *) ptd_err, sizeof(ptd_err),
                        "weight_formula produced a %s edge weight (%g) at "
                        "vertex %zu, edge %zu. Phase-type edge weights must be "
                        "finite and non-negative; check the formula for log/sqrt "
                        "of a non-positive value, division by zero, or pow with "
                        "a negative base.",
                        (isfinite(w) ? "negative" : "non-finite"), w,
                        (size_t) i, (size_t) j);
                    if (need_free) {
                        free(theta);
                    }
                    return;
                }
                edge->weight = w;
                continue;
            }

            // Compute weight based on mode
            // STRICT VALIDATION: In non-callback mode, coefficient length must match theta length exactly
            // To use extra coefficients, call update_weights(theta, callback) with a custom callback
            if (edge->coefficients_length != theta_len) {
                snprintf((char*)ptd_err, sizeof(ptd_err),
                    "Coefficient length mismatch: edge has %zu coefficients but theta has %zu parameters. "
                    "In non-callback mode, they must match exactly. "
                    "To use extra coefficients, call update_weights(theta, callback) with a custom callback function.",
                    (size_t)edge->coefficients_length, (size_t)theta_len);
                if (need_free) {
                    free(theta);
                }
                return;
            }

            if (use_log) {
                // Product in log-space: exp(sum(log(c_k * θ_k)))
                double log_sum = 0.0;
                for (size_t k = 0; k < edge->coefficients_length; k++) {
                    double product = edge->coefficients[k] * theta[k];

                    // Validate positive (log requires positive arguments)
                    if (product <= 0.0) {
                        snprintf((char*)ptd_err, sizeof(ptd_err),
                            "log=True requires all (coefficient * parameter) products to be positive. "
                            "Got %f at coefficient index %zu (coefficient=%f, parameter=%f). "
                            "Check that all coefficients and parameters are positive.",
                            product, k, edge->coefficients[k], theta[k]);
                        if (need_free) {
                            free(theta);
                        }
                        return;
                    }

                    log_sum += log(product);
                }
                edge->weight = exp(log_sum);
            } else {
                // Standard dot product
                edge->weight = 0.0;
                for (size_t k = 0; k < edge->coefficients_length; k++) {
                    edge->weight += edge->coefficients[k] * theta[k];
                }
            }

            // // Validate computed weight
            // if (isnan(edge->weight)) {
            //     snprintf((char*)ptd_err, sizeof(ptd_err),
            //         "Edge weight computation produced NaN at vertex %zu, edge %zu. "
            //         "Check coefficients and parameter values.",
            //         (unsigned long)i, (unsigned long)j);
            //     PTD_LOG_ERROR("update_weights: edge[%zu][%zu] weight is NaN", i, j);
            //     if (need_free) {
            //         free(theta);
            //     }
            //     return;
            // }
            // if (edge->weight < 0.0) {
            //     snprintf((char*)ptd_err, sizeof(ptd_err),
            //         "Edge weight computation produced negative value (%g) at vertex %zu, edge %zu. "
            //         "Check coefficients and parameter values.",
            //         edge->weight, (unsigned long)i, (unsigned long)j);
            //     PTD_LOG_ERROR("update_weights: edge[%zu][%zu] weight is negative: %g", i, j, edge->weight);
            //     if (need_free) {
            //         free(theta);
            //     }
            //     return;
            // }
        }
    }

    // Auto-normalize weights for discrete graphs
    // For discrete graphs (created via discretize()), edge weights represent probabilities
    // and must sum to 1.0 for each vertex. This normalization is applied automatically
    // after computing weights from coefficients, ensuring the discrete phase-type
    // distribution remains valid without requiring a separate normalize() call.
    if (graph->was_dph) {
        for (size_t i = 0; i < graph->vertices_length; i++) {
            struct ptd_vertex *vertex = graph->vertices[i];

            // Compute total outgoing weight
            double total_weight = 0.0;
            for (size_t j = 0; j < vertex->edges_length; j++) {
                total_weight += vertex->edges[j]->weight;
            }

            // Normalize if total weight is positive
            if (total_weight > 0.0) {
                for (size_t j = 0; j < vertex->edges_length; j++) {
                    vertex->edges[j]->weight /= total_weight;
                }
            }
        }
    }

    // Bump weight version so any forward-state context cached against
    // the previous weights is detected as stale by the C++ wrapper guards.
    graph->weight_version++;

    // Invalidate the *concrete* reward_compute_graph. It contains
    // concrete double multipliers evaluated against the previous edge
    // weights, so a theta change makes those values stale.
    if (graph->reward_compute_graph != NULL) {
        free(graph->reward_compute_graph->commands);
        free(graph->reward_compute_graph);
        graph->reward_compute_graph = NULL;
    }

    /* DO NOT destroy parameterized_reward_compute_graph here.
     *
     * The symbolic compute graph stores commands whose `multiplierptr`
     * fields are pointers into the live edge weight slots
     * (&edge->weight, set during graph construction). The replay loop
     * in ptd_graph_build_ex_absorbation_time_comp_graph_parameterized
     * dereferences `*command.multiplierptr` at replay time, so it
     * automatically picks up whatever value `update_weights` just
     * wrote into edge->weight. The symbolic structure depends only on
     * graph topology + coefficients (theta-independent), neither of
     * which `update_weights` mutates.
     *
     * Destroying it here forces ptd_precompute_reward_compute_graph
     * to rebuild the symbolic structure (O(n^3) Gaussian elimination)
     * on every theta update, which is exactly the cost SVGD pays
     * thousands of times during inference. Preserving the cache means
     * the second-and-beyond forward call only pays the cheap
     * O(commands) concrete-build path (line ~1833).
     *
     * The cache IS still invalidated on legitimate structural changes:
     * - ptd_graph_add_edge invalidates it (new edge means new symbolic
     *   structure).
     * - The was_dph branch of ptd_precompute_reward_compute_graph
     *   invalidates and rebuilds it.
     * - ptd_graph_destroy frees it on graph end-of-life. */

#ifdef HAVE_MPFR
    if (graph->reward_compute_graph_mpfr != NULL) {
        for (size_t i = 0; i < graph->reward_compute_graph_mpfr->length; i++) {
            free(graph->reward_compute_graph_mpfr->commands[i].multiplier_str);
        }
        free(graph->reward_compute_graph_mpfr->commands);
        free(graph->reward_compute_graph_mpfr);
        graph->reward_compute_graph_mpfr = NULL;
    }
#endif

    if (need_free) {
        free(theta);
    }
}


void ptd_graph_update_ipv(
        struct ptd_graph *graph,
        double *ipv,
        size_t ipv_length
) {
    if (graph == NULL) {
        snprintf((char*)ptd_err, sizeof(ptd_err), "ptd_graph_update_ipv: graph is NULL");
        return;
    }
    if (ipv == NULL || ipv_length == 0) {
        snprintf((char*)ptd_err, sizeof(ptd_err),
            "ptd_graph_update_ipv: ipv array is NULL or empty");
        return;
    }

    struct ptd_vertex *start = graph->starting_vertex;
    if (start == NULL) {
        snprintf((char*)ptd_err, sizeof(ptd_err),
            "ptd_graph_update_ipv: graph has no starting vertex");
        return;
    }

    if (ipv_length != start->edges_length) {
        snprintf((char*)ptd_err, sizeof(ptd_err),
            "ptd_graph_update_ipv: ipv length %zu does not match number of "
            "starting-vertex edges %zu",
            (size_t)ipv_length, (size_t)start->edges_length);
        return;
    }

    /* Validate values up front so we never leave the graph in a half-updated
     * state if a bad entry appears mid-loop. */
    for (size_t k = 0; k < ipv_length; k++) {
        if (isnan(ipv[k])) {
            snprintf((char*)ptd_err, sizeof(ptd_err),
                "ptd_graph_update_ipv: ipv[%zu] is NaN", (size_t)k);
            return;
        }
        if (isinf(ipv[k])) {
            snprintf((char*)ptd_err, sizeof(ptd_err),
                "ptd_graph_update_ipv: ipv[%zu] is Inf", (size_t)k);
            return;
        }
    }

    /* Direct scalar write — no coefficient/log/callback kernel. IPV is a
     * model property, not derived from a parameter vector. */
    for (size_t k = 0; k < ipv_length; k++) {
        start->edges[k]->weight = ipv[k];
    }

    /* Bump weight version so any cached forward-state context
     * (ph_context_markov etc.) notices the change. */
    graph->weight_version++;

    /* Invalidate the *concrete* reward_compute_graph (it embeds concrete
     * edge-weight values from the previous IPV). The *symbolic*
     * parameterized_reward_compute_graph is preserved by the same Stage A0
     * argument as ptd_graph_update_weights: it depends only on topology +
     * coefficients, neither of which IPV edge-weight writes mutate. */
    if (graph->reward_compute_graph != NULL) {
        free(graph->reward_compute_graph->commands);
        free(graph->reward_compute_graph);
        graph->reward_compute_graph = NULL;
    }

#ifdef HAVE_MPFR
    if (graph->reward_compute_graph_mpfr != NULL) {
        for (size_t i = 0; i < graph->reward_compute_graph_mpfr->length; i++) {
            free(graph->reward_compute_graph_mpfr->commands[i].multiplier_str);
        }
        free(graph->reward_compute_graph_mpfr->commands);
        free(graph->reward_compute_graph_mpfr);
        graph->reward_compute_graph_mpfr = NULL;
    }
#endif
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

    // // Special case: Ensure starting vertex is in parent lists for its children
    // // The starting vertex may not be in the sorted vertices array if it has special properties
    // // We need to explicitly add it as a parent for reward_transform bypass edge creation
    // size_t start_idx = graph->starting_vertex->index;
    // if (start_idx < vertices_length && vertices[start_idx] == graph->starting_vertex) {
    //     // Starting vertex is in the vertices array, parents already added above
    // } else {
    //     // Starting vertex not in vertices array or has different index - add manually
    //     // This handles the case where starting vertex has no incoming edges
    //     for (size_t j = 0; j < graph->starting_vertex->edges_length; ++j) {
    //         struct ptd_vertex *child = graph->starting_vertex->edges[j]->to;
    //         size_t child_idx = child->index;

    //         if (child_idx < vertices_length) {
    //             // Check if starting vertex not already in child's parent list
    //             bool already_parent = false;
    //             for (size_t p = 0; p < vertex_parents_length[child_idx]; ++p) {
    //                 if (vertex_parents[child_idx][p].p == graph->starting_vertex) {
    //                     already_parent = true;
    //                     break;
    //                 }
    //             }

    //             if (!already_parent) {
    //                 // Add starting vertex as parent
    //                 if (vertex_parents_length[child_idx] >= vertex_parents_alloc_length[child_idx]) {
    //                     vertex_parents_alloc_length[child_idx] *= 2;
    //                     vertex_parents[child_idx] = (struct arr_p *) realloc(
    //                         vertex_parents[child_idx],
    //                         vertex_parents_alloc_length[child_idx] * sizeof(*(vertex_parents[child_idx]))
    //                     );
    //                 }

    //                 vertex_parents[child_idx][vertex_parents_length[child_idx]].p = graph->starting_vertex;
    //                 vertex_parents[child_idx][vertex_parents_length[child_idx]].arr_c_index = j + 1; // +1 because of dummy at index 0
    //                 vertex_parents_length[child_idx]++;
    //             }
    //         }
    //     }
    // }

    struct arr_c *old_edges_buffer =
            (struct arr_c *) calloc(vertices_length + 2, sizeof(*old_edges_buffer));

    // // Track which vertices have been bypassed (for final graph construction)
    // bool *bypassed = (bool *) calloc(vertices_length, sizeof(*bypassed));

    // // Process vertices in REVERSE topological order
    // // This ensures parents are still active when we process each vertex
    // for (size_t rev_idx = 0; rev_idx < vertices_length; ++rev_idx) {
    //     size_t i = vertices_length - 1 - rev_idx;

    for (size_t i = 0; i < vertices_length; ++i) {
        if (rewards[i] != 0) {
            continue;
        }

        // Never bypass starting vertex
        if (vertices[i] == graph->starting_vertex) {
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

        // // Mark this vertex as bypassed
        // bypassed[i] = true;
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
            double weight = vertex_edges[i][j].prob / rewards[i];
            ptd_graph_add_edge(
                    new_graph->vertices[new_indicesGtoN[i]],
                    new_graph->vertices[new_indicesGtoN[vertex_edges[i][j].to->index]],
                    &weight,
                    1
            );
        }
    }
    // for (size_t i = 0; i < vertices_length; ++i) {
    //     if (bypassed[i]) {
    //         continue;
    //     }

    //     for (size_t j = 1; j < vertex_edges_length[i] - 1; ++j) {
    //         size_t child_idx = vertex_edges[i][j].to->index;

    //         // Skip edges to bypassed vertices
    //         if (bypassed[child_idx]) {
    //             continue;
    //         }

    //         double rate = vertex_edges[i][j].prob / rewards[i];
    //         size_t new_from_idx = new_indicesGtoN[i];
    //         size_t new_to_idx = new_indicesGtoN[child_idx];

    //         ptd_graph_add_edge(
    //                 new_graph->vertices[new_from_idx],
    //                 new_graph->vertices[new_to_idx],
    //                 rate
    //         );
    //     }
    // }



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
    // free(bypassed);
    free(v);
    ptd_scc_graph_destroy(scc);
    free(rewards);


    return new_graph;
}

struct ptd_graph *ptd_graph_reward_transform(struct ptd_graph *graph, double *rewards) {
    if (ptd_validate_graph(graph)) {
        return NULL;
    }

    /* _ptd_graph_reward_transform mutates the source graph (vertex
     * indices are reordered to SCC-topological order, edge weights are
     * normalised then partially restored). The internal restoration
     * loop is not idempotent, so successive calls on the same graph
     * silently corrupt it. Clone the input first so the caller's graph
     * survives unchanged. ptd_clone_graph preserves vertex order, so
     * ``rewards`` keyed by source-graph indices remains valid against
     * the clone. */
    struct ptd_clone_res clone_res = ptd_clone_graph(graph, NULL);
    if (clone_res.graph == NULL) {
        return NULL;
    }

    size_t *new_indices;
    struct ptd_graph *res = _ptd_graph_reward_transform(
            clone_res.graph, rewards, &new_indices);

    free(new_indices);
    if (clone_res.avl_tree != NULL) {
        ptd_avl_tree_destroy(clone_res.avl_tree);
    }
    ptd_graph_destroy(clone_res.graph);

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

    /* Clone first so _ptd_graph_reward_transform's mutating SCC index
     * reordering and weight normalisation does not corrupt the
     * caller's graph. ptd_clone_graph preserves vertex order, so the
     * ``rewards[old_index]`` lookup below (line ~4143) keyed by
     * indices returned in ``new_graph_indices`` (which are clone-graph
     * vertex indices, identical to source-graph vertex indices)
     * remains correct. */
    struct ptd_clone_res clone_res = ptd_clone_graph(_graph, NULL);
    if (clone_res.graph == NULL) {
        free(zero_rewards);
        return NULL;
    }

    size_t *new_graph_indices;
    struct ptd_graph *graph = _ptd_graph_reward_transform(clone_res.graph, zero_rewards, &new_graph_indices);
    /* The internal function returns a freshly allocated transformed
     * graph; the cloned ``clone_res.graph`` was the (mutated) input
     * and is no longer needed. Free it (and the avl tree if any) so
     * the only graph we leak references to is ``graph`` (which the
     * caller owns). */
    if (clone_res.avl_tree != NULL) {
        ptd_avl_tree_destroy(clone_res.avl_tree);
    }
    ptd_graph_destroy(clone_res.graph);

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
            double weight = vertex->edges[j]->weight;
            ptd_graph_add_edge(
                    auxiliary_vertices[non_zero_rewards[i] - 1],
                    vertex->edges[j]->to,
                    &weight,
                    1
            );

            free(vertex->edges[j]);
        }

        vertex->edges_length = 0;

        for (int k = 0; k < non_zero_rewards[i] - 1; ++k) {
            double weight = 1.0;
            ptd_graph_add_edge(
                    auxiliary_vertices[k],
                    auxiliary_vertices[k + 1],
                    &weight,
                    1
            );
        }

        if (1 - rate > REWARD_EPSILON) {
            double weight = 1 - rate;
            ptd_graph_add_edge(
                    auxiliary_vertices[non_zero_rewards[i] - 1],
                    vertex,
                    &weight,
                    1
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

    memset(&cmd[index], 0, sizeof(cmd[index]));   /* zero unused fields -> deterministic .bin */
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

    memset(&cmd[index], 0, sizeof(cmd[index]));   /* zero unused fields -> deterministic .bin */
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

    memset(&cmd[index], 0, sizeof(cmd[index]));   /* zero unused fields -> deterministic .bin */
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

    memset(&cmd[index], 0, sizeof(cmd[index]));   /* zero unused fields -> deterministic .bin */
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

    memset(&cmd[index], 0, sizeof(cmd[index]));   /* zero unused fields -> deterministic .bin */
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

    memset(&cmd[index], 0, sizeof(cmd[index]));   /* zero unused fields -> deterministic .bin */
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

    memset(&cmd[index], 0, sizeof(cmd[index]));   /* zero unused fields -> deterministic .bin */
    cmd[index].type = NEW_ADD;

    cmd[index].from = from;
    cmd[index].to = to;
    cmd[index].multiplierptr = weight;

    return cmd;
}

#ifdef HAVE_MPFR
/**
 * Helper function to add an MPFR command with string-stored multiplier
 *
 * Converts MPFR value to scientific notation string for maximum precision preservation.
 */
static void add_mpfr_command(
    struct ptd_reward_increase_mpfr **commands,
    size_t *command_index,
    size_t from,
    size_t to,
    mpfr_t multiplier
) {
    // Reallocate if needed (power of 2 growth)
    bool is_power_of_2 = ((*command_index) & ((*command_index) - 1)) == 0;
    if (is_power_of_2) {
        size_t new_length = (*command_index) == 0 ? 1 : (*command_index) * 2;
        *commands = (struct ptd_reward_increase_mpfr *) realloc(
                *commands, new_length * sizeof(**commands)
        );
    }

    (*commands)[*command_index].from = from;
    (*commands)[*command_index].to = to;

    // Apply the same transformation as add_command():
    // When from == to, store (multiplier - 1) instead of multiplier
    // This transforms: result[i] += result[i] * (mult - 1)
    // into: result[i] *= mult
    mpfr_t adjusted_mult;
    mpfr_init2(adjusted_mult, mpfr_get_prec(multiplier));
    if (from == to) {
        mpfr_sub_ui(adjusted_mult, multiplier, 1, MPFR_RNDN);  // adjusted = mult - 1
    } else {
        mpfr_set(adjusted_mult, multiplier, MPFR_RNDN);  // adjusted = mult
    }

    // Convert MPFR to string (base 10, auto precision)
    mp_exp_t exp;
    char *str = mpfr_get_str(NULL, &exp, 10, 0, adjusted_mult, MPFR_RNDN);
    mpfr_clear(adjusted_mult);

    // Format as scientific notation: "X.YYYeZZ"
    if (str != NULL && str[0] != '\0') {
        size_t len = strlen(str);
        size_t final_len = len + 20;
        char *formatted = (char *)malloc(final_len);

        int offset = 0;
        if (str[0] == '-') {
            formatted[0] = '-';
            offset = 1;
        }

        if (len > offset) {
            formatted[offset] = str[offset];
            formatted[offset + 1] = '.';
            strcpy(formatted + offset + 2, str + offset + 1);
            snprintf(formatted + strlen(formatted), 20, "e%ld", (long)(exp - 1));
        } else {
            strcpy(formatted, "0");
        }

        mpfr_free_str(str);
        (*commands)[*command_index].multiplier_str = formatted;
    } else {
        (*commands)[*command_index].multiplier_str = strdup("0");
    }

    (*command_index)++;
}
#endif

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

// Dynamic minimum-degree ordering variant.
// Within each SCC, eliminates the vertex with fewest current edges first.
// Topological ordering across SCCs is preserved.
struct ptd_desc_reward_compute *ptd_graph_ex_absorbation_time_comp_graph_dyn(struct ptd_graph *graph) {
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
    size_t *scc_id = (size_t *) calloc(graph->vertices_length, sizeof(*scc_id));
    size_t n_sccs = scc->vertices_length;

    for (size_t sii = 0; sii < scc->vertices_length; ++sii) {
        for (size_t j = 0; j < v[sii]->internal_vertices_length; ++j) {
            if (v[sii]->internal_vertices[j]->edges_length == 0) {
                continue;
            }

            original_indices[idx] = v[sii]->internal_vertices[j]->index;
            v[sii]->internal_vertices[j]->index = idx;
            vertices[idx] = v[sii]->internal_vertices[j];
            scc_id[idx] = sii;
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
            scc_id[idx] = sii;
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

    // Dynamic ordering state
    bool *eliminated = (bool *) calloc(vertices_length, sizeof(bool));
    size_t *elimination_order = (size_t *) calloc(vertices_length, sizeof(size_t));
    size_t n_eliminated = 0;

    // Process SCCs in topological order; within each SCC use dynamic min-degree
    for (size_t current_scc = 0; current_scc < n_sccs; ++current_scc) {
    while (1) {
        // Find uneliminated vertex in current SCC with minimum current degree
        size_t i = SIZE_MAX;
        size_t min_degree = SIZE_MAX;
        for (size_t k = 0; k < vertices_length; ++k) {
            if (!eliminated[k] && scc_id[k] == current_scc &&
                vertex_edges_length[k] < min_degree) {
                min_degree = vertex_edges_length[k];
                i = k;
            }
        }
        if (i == SIZE_MAX) break;
        eliminated[i] = true;
        elimination_order[n_eliminated++] = i;

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

            if (eliminated[parent_vertex_index]) {
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

            vertex_edges_length[parent_vertex_index] = vertex_edges_length[parent_vertex_index];
        }
    } // end while(1) — dynamic min-degree within current SCC
    } // end for(current_scc) — topological SCC order

    // Back-substitution in reverse elimination order
    for (size_t ii = 0; ii < n_eliminated; ++ii) {
        size_t i = elimination_order[n_eliminated - ii - 1];
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
    free(eliminated);
    free(elimination_order);
    free(scc_id);
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

#ifdef HAVE_MPFR
/* DEPRECATED (MPFR-A-WP-3): builds a separate MPFR-precision
 * compute graph by re-eliminating from scratch. The default
 * MPFR path (MPFR-A) reads the regular reward_compute_graph
 * directly, avoiding this re-elimination cost. This builder
 * is now reachable only via PHASIC_USE_MPFR_LEGACY=1 and will
 * be removed once the legacy opt-in is no longer needed. */
static struct ptd_desc_reward_compute_mpfr *ptd_graph_ex_absorbation_time_comp_graph_mpfr(
    struct ptd_graph *graph,
    size_t precision
) {
    if (ptd_validate_graph(graph)) {
        return NULL;
    }

    // Initialize MPFR variables
    mpfr_t rate, prob, temp, one;
    mpfr_init2(rate, precision);
    mpfr_init2(prob, precision);
    mpfr_init2(temp, precision);
    mpfr_init2(one, precision);
    mpfr_set_d(one, 1.0, MPFR_RNDN);

    struct ptd_vertex *dummy__ptd_min = (struct ptd_vertex *) 1, *dummy__ptd_max = 0;

    struct ptd_vertex **vertices = (struct ptd_vertex **) calloc(graph->vertices_length, sizeof(*vertices));
    size_t *original_indices = (size_t *) calloc(graph->vertices_length, sizeof(*original_indices));

    struct ptd_reward_increase_mpfr *commands = NULL;
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

        mpfr_set_d(rate, 0.0, MPFR_RNDN);
        for (size_t j = 0; j < vertex->edges_length; ++j) {
            mpfr_add_d(rate, rate, vertex->edges[j]->weight, MPFR_RNDN);
        }

        // Add the "real" rate as our first reward

        if (graph->starting_vertex == vertex || vertex->edges_length == 0) {
            mpfr_set_d(temp, 0.0, MPFR_RNDN);
            add_mpfr_command(&commands, &command_index, original_indices[i], original_indices[i], temp);
        } else {
            mpfr_ui_div(temp, 1, rate, MPFR_RNDN);
            add_mpfr_command(&commands, &command_index, original_indices[i], original_indices[i], temp);
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

        mpfr_set_d(rate, 0.0, MPFR_RNDN);
        for (size_t j = 0; j < vertex->edges_length; ++j) {
            mpfr_add_d(rate, rate, vertex->edges[j]->weight, MPFR_RNDN);
        }

        for (size_t j = 0; j < vertex->edges_length; ++j) {
            vertex_edges[i][j + 1].to = vertex->edges[j]->to;
            mpfr_d_div(prob, vertex->edges[j]->weight, rate, MPFR_RNDN);
            vertex_edges[i][j + 1].prob = mpfr_get_d(prob, MPFR_RNDN);
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
                mpfr_clear(rate);
                mpfr_clear(prob);
                mpfr_clear(temp);
                mpfr_clear(one);
                return NULL;
            }

            struct arr_c *new_parent_children = vertex_edges[parent_vertex_index];

            mpfr_set_d(temp, parent_weight_to_me, MPFR_RNDN);
            add_mpfr_command(&commands, &command_index, original_indices[parent_vertex_index],
                            original_indices[i], temp);

            size_t child_index = 0;
            size_t parent_child_index = 0;

            while (child_index < my_edges_length || parent_child_index < parent_edges_length) {
                struct arr_c me_to_child = my_children[child_index];
                struct ptd_vertex *me_to_child_v = me_to_child.to;
                struct arr_c parent_to_child = old_edges_buffer[parent_child_index];
                struct ptd_vertex *parent_to_child_v = parent_to_child.to;
                double me_to_child_p = me_to_child.prob;

                if (me_to_child_v == parent_vertex) {
                    mpfr_set_d(prob, parent_weight_to_me, MPFR_RNDN);
                    mpfr_set_d(temp, me_to_child_p, MPFR_RNDN);
                    mpfr_mul(prob, prob, temp, MPFR_RNDN);
                    mpfr_sub(temp, one, prob, MPFR_RNDN);
                    mpfr_ui_div(prob, 1, temp, MPFR_RNDN);
                    add_mpfr_command(&commands, &command_index,
                                    original_indices[parent_vertex->index],
                                    original_indices[parent_vertex->index], prob);

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
            mpfr_set_d(temp, child.prob, MPFR_RNDN);
            add_mpfr_command(&commands, &command_index, original_indices[vertex->index],
                            original_indices[child.to->index], temp);
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

    mpfr_set_nan(temp);
    add_mpfr_command(&commands, &command_index, 0, 0, temp);

    // Clean up MPFR variables
    mpfr_clear(rate);
    mpfr_clear(prob);
    mpfr_clear(temp);
    mpfr_clear(one);

    struct ptd_desc_reward_compute_mpfr *res = (struct ptd_desc_reward_compute_mpfr *) malloc(sizeof(*res));
    res->length = command_index;
    res->commands = commands;

    return res;
}
#endif  // HAVE_MPFR


struct ll_c2_a {
    struct ll_c2_a *next;
    struct ll_c2 *mem;
};

/* Per-thread scratch allocator for the parameterised reward compute
 * graph. Each call to ptd_graph_ex_absorbation_time_comp_graph_parameterized
 * runs ll_c2_alloc_init / _free around its body, so each thread sees a
 * clean private allocator. ``__ptd_max`` is read-only after init and
 * stays plain static. */
static PTD_TLS struct ll_c2_a **ll_c2_alloced;
static size_t ll_c2_alloced__ptd_max = 1024;
static PTD_TLS size_t *ll_c2_alloced_index;

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

/* Per-thread sibling of ll_c2; same lifetime/usage pattern. */
static PTD_TLS struct ll_p2_a **ll_p2_alloced;
static size_t ll_p2_alloced__ptd_max = 1024;
static PTD_TLS size_t *ll_p2_alloced_index;

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

/* Per-thread allocation counter for the linked-list-of-arrays mem pool
 * used by the parameterised compute graph builder. Read here, written
 * by add_mem; isolating per thread is the minimum to avoid the
 * cross-thread torn-write hazard observed under JAX pmap. */
static PTD_TLS int t = 0;

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

/* Deterministic elimination ordering helpers.
 *
 * The adjacency-list sorted-merge in the parameterized builders below
 * historically ordered by vertex POINTER ADDRESS (`child_vertex <
 * parent_child_vertex`), which is non-reproducible run-to-run (heap layout
 * varies) -> the elimination trace length and FP rounding wobble between
 * runs. We order by the re-assigned vertex ->index instead (deterministic;
 * every vertex gets a unique index in [0, vertices_length) during the
 * SCC-topological re-indexing). dummy__ptd_min/max are fake sentinel
 * pointers (not real vertices), so they are special-cased to -inf/+inf. */
static inline size_t ptd_vtx_order_key(const struct ptd_vertex *v,
                                       const struct ptd_vertex *dmin,
                                       const struct ptd_vertex *dmax) {
    if (v == dmin) return (size_t) 0;
    if (v == dmax) return SIZE_MAX;
    return (size_t) v->index + 1;
}

/* Key for sorting a vertex's out-edges by target ->index; ties (parallel
 * edges to the same target) are broken by original array position so the
 * order is stable/deterministic. */
struct ptd_edge_order_key { uint64_t to_index; uint32_t orig; };
static int ptd_edge_order_cmp(const void *a, const void *b) {
    const struct ptd_edge_order_key *x = (const struct ptd_edge_order_key *) a;
    const struct ptd_edge_order_key *y = (const struct ptd_edge_order_key *) b;
    if (x->to_index < y->to_index) return -1;
    if (x->to_index > y->to_index) return 1;
    if (x->orig < y->orig) return -1;
    if (x->orig > y->orig) return 1;
    return 0;
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

        /* Build this vertex's child list in target-index order so it is
         * sorted by the same key the merge below uses (was: edge-array
         * order, merged by pointer address -> non-deterministic). */
        struct ptd_edge_order_key *edge_order = NULL;
        if (vertex->edges_length > 0) {
            edge_order = (struct ptd_edge_order_key *) malloc(
                    vertex->edges_length * sizeof(*edge_order));
            for (size_t j = 0; j < vertex->edges_length; ++j) {
                edge_order[j].to_index = (uint64_t) vertex->edges[j]->to->index;
                edge_order[j].orig = (uint32_t) j;
            }
            qsort(edge_order, vertex->edges_length, sizeof(*edge_order),
                  ptd_edge_order_cmp);
        }

        for (size_t jj = 0; jj < vertex->edges_length; ++jj) {
            struct ptd_edge *e = vertex->edges[edge_order[jj].orig];
            struct ll_p2 *n = ll_p2_alloc(0);

            n->next = parents[e->to->index];
            n->p = vertex;
            n->prev = NULL;

            if (parents[e->to->index] != NULL) {
                parents[e->to->index]->prev = n;
            }

            parents[e->to->index] = n;

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
                    &(e->weight),
                    rates[i],
                    command_index++
            );

            nc->weight = current_mem_ll->current_mem_position;

            nc->c = e->to;
            nc->ll_p = n;
            n->ll_c = nc;
            last = nc;
        }
        free(edge_order);

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
                } else if (ptd_vtx_order_key(child_vertex, dummy__ptd_min, dummy__ptd_max)
                           < ptd_vtx_order_key(parent_child_vertex, dummy__ptd_min, dummy__ptd_max)) {
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

// Dynamic minimum-degree ordering variant of the parameterized elimination.
// Within each SCC, eliminates the vertex with fewest current edges first.
struct ptd_desc_reward_compute_parameterized *ptd_graph_ex_absorbation_time_comp_graph_parameterized_dyn(
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
    size_t *scc_id = (size_t *) calloc(graph->vertices_length, sizeof(*scc_id));
    size_t n_sccs = scc->vertices_length;

    for (size_t sii = 0; sii < scc->vertices_length; ++sii) {
        for (size_t j = 0; j < v[sii]->internal_vertices_length; ++j) {
            if (v[sii]->internal_vertices[j]->edges_length == 0) {
                continue;
            }

            original_indices[idx] = v[sii]->internal_vertices[j]->index;
            v[sii]->internal_vertices[j]->index = idx;
            vertices[idx] = v[sii]->internal_vertices[j];
            scc_id[idx] = sii;
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
            scc_id[idx] = sii;
            idx++;
        }
    }

    double **rates = (double **) calloc(graph->vertices_length, sizeof(*rates));

    // Track degree for dynamic ordering
    size_t *degree = (size_t *) calloc(vertices_length, sizeof(size_t));

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

        // Initialize degree from edge count (+ 2 for dummy sentinels)
        degree[i] = vertex->edges_length + 2;
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

        /* Build this vertex's child list in target-index order so it is
         * sorted by the same key the merge below uses (was: edge-array
         * order, merged by pointer address -> non-deterministic). */
        struct ptd_edge_order_key *edge_order = NULL;
        if (vertex->edges_length > 0) {
            edge_order = (struct ptd_edge_order_key *) malloc(
                    vertex->edges_length * sizeof(*edge_order));
            for (size_t j = 0; j < vertex->edges_length; ++j) {
                edge_order[j].to_index = (uint64_t) vertex->edges[j]->to->index;
                edge_order[j].orig = (uint32_t) j;
            }
            qsort(edge_order, vertex->edges_length, sizeof(*edge_order),
                  ptd_edge_order_cmp);
        }

        for (size_t jj = 0; jj < vertex->edges_length; ++jj) {
            struct ptd_edge *e = vertex->edges[edge_order[jj].orig];
            struct ll_p2 *n = ll_p2_alloc(0);

            n->next = parents[e->to->index];
            n->p = vertex;
            n->prev = NULL;

            if (parents[e->to->index] != NULL) {
                parents[e->to->index]->prev = n;
            }

            parents[e->to->index] = n;

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
                    &(e->weight),
                    rates[i],
                    command_index++
            );

            nc->weight = current_mem_ll->current_mem_position;

            nc->c = e->to;
            nc->ll_p = n;
            n->ll_c = nc;
            last = nc;
        }
        free(edge_order);

        struct ll_c2 *dummy_last = ll_c2_alloc(0);
        dummy_last->next = NULL;
        dummy_last->prev = last;
        dummy_last->weight = 0;
        dummy_last->c = dummy__ptd_max;
        dummy_last->ll_p = NULL;
        last->next = dummy_last;
    }

    // Dynamic ordering state
    bool *eliminated = (bool *) calloc(vertices_length, sizeof(bool));
    size_t *elimination_order = (size_t *) calloc(vertices_length, sizeof(size_t));
    size_t n_eliminated = 0;

    // Pre-eliminate absorbing vertices and starting vertex
    for (size_t k = 0; k < vertices_length; ++k) {
        if (vertices[k]->edges_length == 0 || vertices[k] == graph->starting_vertex) {
            eliminated[k] = true;
            elimination_order[n_eliminated++] = k;
        }
    }

    // Process SCCs in topological order; within each SCC use dynamic min-degree
    for (size_t current_scc = 0; current_scc < n_sccs; ++current_scc) {
    while (1) {
        // Find uneliminated vertex in current SCC with minimum current degree
        size_t i = SIZE_MAX;
        size_t min_deg = SIZE_MAX;
        for (size_t k = 0; k < vertices_length; ++k) {
            if (!eliminated[k] && scc_id[k] == current_scc &&
                degree[k] < min_deg) {
                min_deg = degree[k];
                i = k;
            }
        }
        if (i == SIZE_MAX) break;
        eliminated[i] = true;
        elimination_order[n_eliminated++] = i;

        struct ptd_vertex *vertex = vertices[i];

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
            if (eliminated[parent->p->index]) {
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
                } else if (ptd_vtx_order_key(child_vertex, dummy__ptd_min, dummy__ptd_max)
                           < ptd_vtx_order_key(parent_child_vertex, dummy__ptd_min, dummy__ptd_max)) {
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

                    // New edge added to parent — update parent's degree
                    degree[parent_vertex->index]++;

                    l++;
                } else {
                    parent_child = parent_child->next;
                }
            }

            struct ll_c2 *edge_to_me = parent->ll_c;
            edge_to_me->prev->next = edge_to_me->next;
            edge_to_me->next->prev = edge_to_me->prev;

            // Edge removed from parent — update parent's degree
            degree[parent->p->index]--;

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
    } // end while(1) — dynamic min-degree within current SCC
    } // end for(current_scc) — topological SCC order

    // Back-substitution in reverse elimination order
    for (size_t ii = 0; ii < n_eliminated; ++ii) {
        size_t i = elimination_order[n_eliminated - ii - 1];
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
    free(eliminated);
    free(elimination_order);
    free(degree);
    free(scc_id);
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

/* B1: offset-form executor — byte-identical arithmetic to the raw
 * parameterized executor below, resolving operands via mem_base+offset /
 * inputs[idx]. fromT is always MEM (a writable scratch slot); toT/multiplierptr
 * are MEM or INPUT (read). */
static inline double *ptd_pcg_resolve(
        const struct ptd_desc_reward_compute_parameterized_off *off,
        const struct ptd_pcg_operand *op) {
    switch (op->kind) {
        case PTD_PCG_OP_MEM:   return off->mem_base + op->mem_offset;
        case PTD_PCG_OP_INPUT: return off->inputs[op->input_idx];
        default:               return NULL;
    }
}
static struct ptd_desc_reward_compute *
ptd_graph_build_ex_absorbation_time_comp_graph_parameterized_off(
        const struct ptd_desc_reward_compute_parameterized_off *off) {
    struct ptd_reward_increase *commands = NULL;
    size_t command_index = 0;
    enum command_types { PP = 3, P = 1, INV = 2, ZERO = 6, DIVIDE = 5, ONE_MINUS = 4, NEW_ADD = 0 };
    for (size_t i = 0; i < off->length; ++i) {
        const struct ptd_pcg_command_off *c = &off->commands[i];
        double *fromT = ptd_pcg_resolve(off, &c->fromT);
        double *toT = ptd_pcg_resolve(off, &c->toT);
        double *mptr = ptd_pcg_resolve(off, &c->multiplierptr);
        switch (c->type) {
            case NEW_ADD:
                commands = add_command(commands, (size_t)c->from, (size_t)c->to,
                                       *mptr, command_index++);
                break;
            case P:         *fromT = *fromT + *toT * c->multiplier; break;
            case PP:        *fromT = *fromT + *toT * *mptr; break;
            case INV:       *fromT = 1 / *fromT; break;
            case ONE_MINUS: *fromT = 1 - *fromT; break;
            case DIVIDE:    *fromT /= *toT; break;
            case ZERO:      *fromT = 0; break;
            default: DIE_ERROR(1, "Unknown command\n");
        }
    }
    struct ptd_desc_reward_compute *res =
        (struct ptd_desc_reward_compute *)malloc(sizeof(*res));
    res->length = command_index;
    res->commands = commands;
    return res;
}
static void ptd_pcg_desc_off_destroy(
        struct ptd_desc_reward_compute_parameterized_off *off) {
    if (off == NULL) return;
    if (off->mem_is_mmap) {
#ifndef _WIN32
        if (off->mmap_base) munmap(off->mmap_base, off->mmap_len);
#endif
    } else {
        if (off->commands) free(off->commands);
        if (off->mem_base) free(off->mem_base);
        if (off->input_specs) free(off->input_specs);
    }
    if (off->inputs) free(off->inputs);   /* inputs[] is always heap (bound at load) */
    free(off);
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
                // // Check for NaN in multiplier before adding command
                // if (isnan(*command.multiplierptr)) {
                //     PTD_LOG_ERROR("build_parameterized: NaN multiplier detected at command %zu "
                //         "(from=%zu, to=%zu). Numerical instability during graph elimination.",
                //         i, command.from, command.to);
                //     if (commands) free(commands);
                //     return NULL;
                // }
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
                // Check for NaN after arithmetic
                // if (isnan(*(command.fromT))) {
                //     sprintf((char*)ptd_err, "build_parameterized: P command %zu produced NaN (toT=%g, multiplier=%g).", i, *command.toT, command.multiplier);
                //     if (commands) free(commands);
                //     return NULL;
                // }
                break;
            case PP:
                *(command.fromT) = *(command.fromT) + *command.toT * *command.multiplierptr;
                // // Check for NaN after arithmetic (e.g., 0 * Inf = NaN)
                // if (isnan(*(command.fromT))) {
                //     sprintf((char*)ptd_err, "build_parameterized: PP command %zu produced NaN (toT=%g, multiplierptr=%g).", i, *command.toT, *command.multiplierptr);
                //     if (commands) free(commands);
                //     return NULL;
                // }
                break;
            case INV:
                // // Check for NaN input
                // if (isnan(*(command.fromT))) {
                //     sprintf((char*)ptd_err, "build_parameterized: INV command %zu has NaN input.", i);
                //     if (commands) free(commands);
                //     return NULL;
                // }
                // // Check for division by zero (self-loop probability >= 1)
                // if (fabs(*(command.fromT)) <= 1e-15) {
                //     sprintf((char*)ptd_err, "build_parameterized: Inverse of near-zero value at command %zu (value=%g). Likely self-loop probability >= 1.",i, *(command.fromT));
                //     if (commands) free(commands);
                //     return NULL;
                // }
                *(command.fromT) = 1 / *(command.fromT);
                break;
            case ONE_MINUS:
                *(command.fromT) = 1 - *command.fromT;
                // // Check for NaN propagation
                // if (isnan(*(command.fromT))) {
                //     sprintf((char*)ptd_err, "build_parameterized: ONE_MINUS command %zu produced NaN.", i);
                //     if (commands) free(commands);
                //     return NULL;
                // }
                break;
            case DIVIDE:
                // // Check for NaN inputs
                // if (isnan(*(command.fromT)) || isnan(*command.toT)) {
                //     sprintf((char*)ptd_err, "build_parameterized: DIVIDE command %zu has NaN input (fromT=%g, toT=%g).", i, *(command.fromT), *command.toT);
                //     if (commands) free(commands);
                //     return NULL;
                // }
                // // Check for division by zero (zero outgoing probability)
                // if (fabs(*command.toT) <= 1e-15) {
                //     sprintf((char*)ptd_err, "build_parameterized: Division by near-zero value at command %zu (divisor=%g). Parent has zero outgoing probability.", i, *command.toT);
                //     if (commands) free(commands);
                //     return NULL;
                // }
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

#ifdef HAVE_MPFR
/**
 * Execute reward computation using MPFR for high-precision arithmetic
 *
 * @param graph Graph with precomputed MPFR command list
 * @param rewards Initial reward vector (or NULL for all 1.0)
 * @param precision MPFR precision in bits
 * @return Result vector (double precision) or NULL on error
 */
/* DEPRECATED (MPFR-A-WP-3): consumes reward_compute_graph_mpfr.
 * Reachable only via PHASIC_USE_MPFR_LEGACY=1. Use
 * ptd_expected_waiting_time_mpfr_from_double_pcg instead, which
 * skips the MPFR re-elimination step. */
static double *ptd_expected_waiting_time_mpfr(
    struct ptd_graph *graph,
    double *rewards,
    size_t precision
) {
    if (graph->reward_compute_graph_mpfr == NULL) {
        PTD_LOG_ERROR("MPFR graph not computed");
        return NULL;
    }

    size_t n = graph->vertices_length;
    struct ptd_desc_reward_compute_mpfr *compute = graph->reward_compute_graph_mpfr;

    // Allocate MPFR result array
    mpfr_t *result = (mpfr_t *)malloc(n * sizeof(mpfr_t));
    if (result == NULL) {
        PTD_LOG_ERROR("Failed to allocate MPFR result array");
        return NULL;
    }

    // Initialize result array
    for (size_t i = 0; i < n; i++) {
        mpfr_init2(result[i], precision);
        if (rewards != NULL) {
            mpfr_set_d(result[i], rewards[i], MPFR_RNDN);
        } else {
            mpfr_set_d(result[i], 1.0, MPFR_RNDN);
        }
    }

    // Temporary variables for computation
    mpfr_t multiplier, product;
    mpfr_init2(multiplier, precision);
    mpfr_init2(product, precision);

    // Execute commands
    for (size_t j = 0; j < compute->length; j++) {
        struct ptd_reward_increase_mpfr cmd = compute->commands[j];

        // Parse multiplier string to MPFR
        if (cmd.multiplier_str == NULL) {
            PTD_LOG_ERROR("NULL multiplier string at command %zu", j);
            goto cleanup_error;
        }

        int parse_result = mpfr_set_str(multiplier, cmd.multiplier_str, 10, MPFR_RNDN);
        if (parse_result != 0) {
            // Check for NaN terminator (various formats from MPFR)
            if (strcmp(cmd.multiplier_str, "nan") == 0 ||
                strcmp(cmd.multiplier_str, "NaN") == 0 ||
                strcmp(cmd.multiplier_str, "@NaN@") == 0 ||
                strstr(cmd.multiplier_str, "NaN") != NULL ||
                strstr(cmd.multiplier_str, ".NaN") != NULL) {
                // This is the terminator - we're done
                break;
            }
            PTD_LOG_ERROR("Failed to parse multiplier '%s' at command %zu",
                         cmd.multiplier_str, j);
            goto cleanup_error;
        }

        // Skip if multiplier is zero (0 × ∞ = 0 limit)
        if (mpfr_zero_p(multiplier)) {
            continue;
        }

        // Check for inf × 0 = 0 limit
        if (mpfr_inf_p(multiplier) && mpfr_zero_p(result[cmd.to])) {
            continue;
        }

        // Compute: result[from] += result[to] * multiplier
        mpfr_mul(product, result[cmd.to], multiplier, MPFR_RNDN);
        mpfr_add(result[cmd.from], result[cmd.from], product, MPFR_RNDN);
    }

    // Convert MPFR results back to double
    double *final_result = (double *)calloc(n, sizeof(double));
    if (final_result == NULL) {
        PTD_LOG_ERROR("Failed to allocate final result array");
        goto cleanup_error;
    }

    for (size_t i = 0; i < n; i++) {
        final_result[i] = mpfr_get_d(result[i], MPFR_RNDN);

        // Check for catastrophic errors
        if (isnan(final_result[i])) {
            PTD_LOG_ERROR("MPFR computation produced NaN at vertex %zu - numerical catastrophe", i);
            sprintf((char*)ptd_err, "MPFR computation produced NaN at vertex %zu - numerical catastrophe", i);
            goto cleanup_error;
        }

        // Infinity is OK (expected for graphs with inescapable cycles)
        if (mpfr_inf_p(result[i])) {
            final_result[i] = INFINITY;
            PTD_LOG_DEBUG("Result[%zu] is infinite (expected for inescapable cycles)", i);
        }
    }

    // Cleanup
    mpfr_clear(multiplier);
    mpfr_clear(product);
    for (size_t i = 0; i < n; i++) {
        mpfr_clear(result[i]);
    }
    free(result);

    PTD_LOG_DEBUG("MPFR computation completed successfully with %zu-bit precision", precision);
    return final_result;

cleanup_error:
    mpfr_clear(multiplier);
    mpfr_clear(product);
    for (size_t i = 0; i < n; i++) {
        mpfr_clear(result[i]);
    }
    free(result);
    return NULL;
}

/* MPFR-A-WP-1: high-precision consumer over the double-precision
 * reward_compute_graph.
 *
 * Reads graph->reward_compute_graph (concrete double multipliers,
 * already θ-bound) and performs the multiply-accumulate sweep at
 * MPFR precision. Avoids the cost of building a separate
 * reward_compute_graph_mpfr — we already have the symbolic
 * elimination encoded in reward_compute_graph; we just want the
 * arithmetic at higher precision.
 *
 * This is the cheap, high-value path: most workloads need
 * higher-precision *consumption* (to avoid catastrophic
 * cancellation in the multiply-accumulate sum) but not
 * higher-precision *elimination*. The cost saved is the entire
 * MPFR re-elimination — typically the dominant cost when the
 * MPFR path triggers.
 *
 * Returns a newly-allocated double-precision result vector
 * (caller frees) on success, NULL on failure (sets ptd_err).
 */
static double *ptd_expected_waiting_time_mpfr_from_double_pcg(
    struct ptd_graph *graph,
    double *rewards,
    size_t precision
) {
    if (graph->reward_compute_graph == NULL) {
        PTD_LOG_ERROR("MPFR-A: reward_compute_graph is NULL");
        snprintf((char *)ptd_err, sizeof(ptd_err),
                 "MPFR-A: reward_compute_graph not built");
        return NULL;
    }

    size_t n = graph->vertices_length;
    struct ptd_desc_reward_compute *compute = graph->reward_compute_graph;

    mpfr_t *result = (mpfr_t *)malloc(n * sizeof(mpfr_t));
    if (result == NULL) {
        PTD_LOG_ERROR("MPFR-A: failed to allocate result array");
        return NULL;
    }
    for (size_t i = 0; i < n; i++) {
        mpfr_init2(result[i], precision);
        if (rewards != NULL) {
            mpfr_set_d(result[i], rewards[i], MPFR_RNDN);
        } else {
            mpfr_set_d(result[i], 1.0, MPFR_RNDN);
        }
    }

    mpfr_t multiplier, product;
    mpfr_init2(multiplier, precision);
    mpfr_init2(product, precision);

    for (size_t j = 0; j < compute->length; j++) {
        struct ptd_reward_increase cmd = compute->commands[j];

        /* Skip the NaN sentinel that the regular consumer treats
         * as a terminator. */
        if (isnan(cmd.multiplier)) {
            break;
        }
        if (cmd.multiplier == 0.0) {
            continue;
        }
        if (isinf(cmd.multiplier) && mpfr_zero_p(result[cmd.to])) {
            continue;
        }

        mpfr_set_d(multiplier, cmd.multiplier, MPFR_RNDN);
        mpfr_mul(product, result[cmd.to], multiplier, MPFR_RNDN);
        mpfr_add(result[cmd.from], result[cmd.from], product, MPFR_RNDN);
    }

    double *final_result = (double *)calloc(n, sizeof(double));
    if (final_result == NULL) {
        PTD_LOG_ERROR("MPFR-A: failed to allocate final result");
        goto cleanup_error;
    }

    for (size_t i = 0; i < n; i++) {
        if (mpfr_inf_p(result[i])) {
            final_result[i] = INFINITY;
        } else {
            final_result[i] = mpfr_get_d(result[i], MPFR_RNDN);
            if (isnan(final_result[i])) {
                PTD_LOG_ERROR("MPFR-A: NaN at vertex %zu", i);
                snprintf((char *)ptd_err, sizeof(ptd_err),
                         "MPFR-A: NaN at vertex %zu", i);
                goto cleanup_error;
            }
        }
    }

    mpfr_clear(multiplier);
    mpfr_clear(product);
    for (size_t i = 0; i < n; i++) {
        mpfr_clear(result[i]);
    }
    free(result);

    PTD_LOG_DEBUG("MPFR-A: completed with %zu-bit precision over double PRC", precision);
    return final_result;

cleanup_error:
    mpfr_clear(multiplier);
    mpfr_clear(product);
    for (size_t i = 0; i < n; i++) {
        mpfr_clear(result[i]);
    }
    free(result);
    return NULL;
}
#endif  // HAVE_MPFR


double *ptd_expected_waiting_time(struct ptd_graph *graph, double *rewards) {
    /* WP-7: hierarchical SCC pipeline opt-in via env var.
     *
     * When PHASIC_HIERAR_ELIMINATION=1 and the graph is
     * parameterised and no reward vector is supplied, route
     * through ptd_compose_scc_prcs instead of the monolithic
     * eliminator. The composer reads the parent's current edge
     * weights (already set by ptd_graph_update_weights), so we
     * pass those weights as theta rather than re-deriving from
     * coefficients. Reward-transformed cases (rewards != NULL)
     * fall through to the monolithic path.
     *
     * Re-entrancy: the composer itself calls
     * ptd_expected_waiting_time recursively on synthetic SCC
     * subgraphs. Those inner calls must NOT take the
     * hierarchical path (they should just monolithically
     * eliminate the synth graph, which is itself the unit of
     * caching). The composer sets ptd_scc_compose_in_progress
     * to suppress recursion. */
    const char *hierar_env = getenv("PHASIC_HIERAR_ELIMINATION");
    bool use_hierarchical = (hierar_env != NULL
                             && hierar_env[0] == '1'
                             && hierar_env[1] == '\0'
                             && !ptd_scc_compose_in_progress);
    if (use_hierarchical && graph->parameterized && rewards == NULL
        && graph->param_length > 0) {
        /* Default-theta unlock: construction initialises edge weights at
         * theta=1 (sum of coefficients * 1.0, see ptd_graph_add_edge) and
         * leaves current_params NULL until the first update_weights(). When no
         * theta has been set explicitly the live weights correspond to
         * theta=ones, so compose at ones; ptd_compose_scc_prcs re-derives
         * weights from coefficients * theta, reproducing the monolithic
         * default-weight result exactly. If theta was set, use it unchanged. */
        const double *compose_theta = graph->current_params;
        double *ones = NULL;
        if (compose_theta == NULL) {
            ones = (double *) malloc(graph->param_length * sizeof(double));
            if (ones != NULL) {
                for (size_t i = 0; i < graph->param_length; i++) {
                    ones[i] = 1.0;
                }
                compose_theta = ones;
            }
        }
        if (compose_theta != NULL) {
            struct ptd_scc_graph *scc_graph =
                    ptd_find_strongly_connected_components(graph);
            if (scc_graph != NULL) {
                double *result = ptd_compose_scc_prcs(
                        graph, scc_graph,
                        compose_theta, graph->param_length);
                ptd_scc_graph_destroy(scc_graph);
                if (result != NULL) {
                    free(ones);
                    return result;
                }
            }
            /* SCC or compose failed: clear error, fall through to monolithic. */
            ptd_err[0] = '\0';
        }
        free(ones);
    }

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

    // Track condition number statistics
    double max_multiplier = 0.0;
    double min_multiplier = INFINITY;
    size_t ill_conditioned_count = 0;

    // Pre-scan elimination commands for condition number
    double prescanned_max = 0.0;
    double prescanned_min = INFINITY;
    for (size_t j = 0; j < graph->reward_compute_graph->length; ++j) {
        struct ptd_reward_increase command = graph->reward_compute_graph->commands[j];
        if (!isinf(command.multiplier) && command.multiplier != 0.0) {
            double abs_mult = fabs(command.multiplier);
            if (abs_mult > prescanned_max) prescanned_max = abs_mult;
            if (abs_mult < prescanned_min) prescanned_min = abs_mult;
        }
    }

    // Get condition threshold from config (default 1e12)
    double condition_threshold = 1e12;
    const char *threshold_env = getenv("PHASIC_CONDITION_THRESHOLD");
    if (threshold_env != NULL) {
        condition_threshold = atof(threshold_env);
    }

    double condition_number = (prescanned_min != INFINITY && prescanned_max > 0.0)
                             ? (prescanned_max / prescanned_min) : 0.0;

#ifdef HAVE_MPFR
    bool force_mpfr = (getenv("PHASIC_FORCE_MPFR") != NULL);

    if (force_mpfr || condition_number > condition_threshold) {
        PTD_LOG_INFO("Auto-activating MPFR for moment computation (condition %.2e > threshold %.2e)",
                     condition_number, condition_threshold);

        // Calculate precision: check env var first, then auto-calculate
        size_t mpfr_precision = 0;
        const char *precision_env = getenv("PHASIC_MPFR_BITS");
        if (precision_env != NULL) {
            mpfr_precision = (size_t)atoi(precision_env);
        } else {
            // Auto-calculate: log2(condition) + 64
            mpfr_precision = (size_t)(log2(condition_number)) + 64;
        }

        // Clamp to reasonable range
        if (mpfr_precision < 128) mpfr_precision = 128;
        if (mpfr_precision > 1024) mpfr_precision = 1024;

        /* MPFR-A: prefer the new consumer that reads the
         * existing double reward_compute_graph and does MPFR
         * arithmetic on top — avoids the cost of building a
         * separate MPFR compute graph. PHASIC_USE_MPFR_LEGACY=1
         * opts back into the old MPFR-builder path for
         * comparison / safety-net during transition. */
        const char *legacy = getenv("PHASIC_USE_MPFR_LEGACY");
        bool use_legacy = (legacy != NULL && legacy[0] == '1' && legacy[1] == '\0');

        double *mpfr_result = NULL;
        if (!use_legacy) {
            PTD_LOG_INFO("MPFR-A: consuming double PRC at %zu-bit precision", mpfr_precision);
            mpfr_result = ptd_expected_waiting_time_mpfr_from_double_pcg(
                    graph, rewards, mpfr_precision);
        } else {
            // Legacy: build separate MPFR PRC and consume that.
            if (graph->reward_compute_graph_mpfr == NULL) {
                PTD_LOG_INFO("Computing MPFR graph with %zu-bit precision (legacy)", mpfr_precision);
                graph->reward_compute_graph_mpfr = ptd_graph_ex_absorbation_time_comp_graph_mpfr(
                    graph, mpfr_precision
                );
            }
            mpfr_result = ptd_expected_waiting_time_mpfr(graph, rewards, mpfr_precision);
        }

        if (mpfr_result != NULL) {
            PTD_LOG_INFO("MPFR computation successful - returning high-precision results");
            return mpfr_result;
        } else {
            PTD_LOG_WARNING("MPFR execution failed - falling back to double precision");
        }
    }
#else
    // Non-MPFR build: warn if poor conditioning detected
    if (condition_number > condition_threshold) {
        PTD_LOG_WARNING("Poor conditioning detected (condition number = %.2e > threshold %.2e). "
                        "Rebuild with MPFR support for accurate high-precision results.",
                        condition_number, condition_threshold);
    }
#endif

    for (size_t j = 0; j < graph->reward_compute_graph->length; ++j) {
        struct ptd_reward_increase command = graph->reward_compute_graph->commands[j];

        // Handle 0 × ∞ = 0 (limit interpretation)
        // Skip when multiplier is zero to avoid NaN from 0.0 × inf
        if (command.multiplier == 0.0) {
            continue;
        }

        // Handle inf × 0 = 0 (limit interpretation)
        // If multiplier is infinite and result[to] is zero, treat as 0
        if (isinf(command.multiplier) && result[command.to] == 0.0) {
            continue;
        }

        // Track conditioning (skip infinite/zero multipliers)
        if (!isinf(command.multiplier) && command.multiplier != 0.0) {
            double abs_mult = fabs(command.multiplier);
            if (abs_mult > max_multiplier) max_multiplier = abs_mult;
            if (abs_mult < min_multiplier) min_multiplier = abs_mult;

            // Track ill-conditioned multipliers for debug logging
            if (abs_mult > 1e10 || abs_mult < 1e-10) {
                ill_conditioned_count++;
                if (ill_conditioned_count == 1) {  // Log first occurrence only
                    PTD_LOG_DEBUG("Ill-conditioned multiplier detected: %.2e at command %zu",
                                 command.multiplier, j);
                }
            }
        }

        result[command.from] += result[command.to] * command.multiplier;

        // Check for catastrophic errors
        if (isnan(result[command.from])) {
            PTD_LOG_ERROR("Computation produced NaN at vertex %zu (command %zu: from=%zu to=%zu multiplier=%.15e result[to]=%.15e) - numerical catastrophe",
                command.from, j, command.from, command.to, command.multiplier, result[command.to]);
            sprintf((char*)ptd_err, "Computation produced NaN at vertex %zu (command %zu) - numerical catastrophe",
                command.from, j);
            free(result);
            return NULL;
        }

        // Infinity is OK (expected for graphs with inescapable cycles)
        if (isinf(result[command.from])) {
            PTD_LOG_DEBUG("Result[%zu] is infinite (expected for inescapable cycles)", command.from);
        }
    }

    // Log conditioning summary at DEBUG level
    if (min_multiplier != INFINITY && max_multiplier > 0.0) {
        double observed_condition = max_multiplier / min_multiplier;
        PTD_LOG_DEBUG("Conditioning summary: max_mult=%.2e, min_mult=%.2e, condition=%.2e (%zu ill-conditioned operations)",
                     max_multiplier, min_multiplier, observed_condition, ill_conditioned_count);
    }

    return result;
}

double *ptd_expected_sojourn_time_subset(struct ptd_graph *graph, const size_t *indices, size_t k) {
    // Precompute elimination trace if not already done
    if (ptd_precompute_reward_compute_graph(graph)) {
        PTD_LOG_ERROR("Failed to precompute reward compute graph");
        return NULL;
    }

    size_t n = graph->vertices_length;
    struct ptd_desc_reward_compute *compute = graph->reward_compute_graph;

    // Expected sojourn (residence) time at `k` target vertices. Both paths below
    // replay the same linear elimination trace, whose command
    //     (from, to, mult):  results[from] += results[to] * mult
    // evaluated forward with a reward seed s (results[v] initialised to s[v])
    // yields, at the starting vertex 0,  results[0] = sum_v sojourn(v) * s[v].
    //
    // (A) ADJOINT (default). sojourn(v) is exactly d results[0] / d s[v] -- the
    //     gradient of that scalar w.r.t. the seed. Reverse-mode differentiation
    //     produces the gradient for ALL v in a SINGLE pass over the trace using
    //     O(n) memory: seed adjoint[0] = 1, walk the commands in REVERSE order
    //     applying the transpose update  adjoint[to] += adjoint[from] * mult,
    //     then sojourn(v) = adjoint[v]. The requested targets are gathered from
    //     that length-n vector. O(n) memory + O(len(trace)) time.
    //
    // (B) FORWARD (legacy, opt-in via PHASIC_SOJOURN_FORWARD=1). One one-hot
    //     reward column per target, replayed forward together as an n*k dense
    //     matrix: O(n*k) memory + O(len(trace)*k) time. Retained purely as a
    //     bit-for-bit correctness reference / escape hatch -- it allocates
    //     n*k doubles and OOMs once the state space is large (the joint-
    //     probability table over a big graph is exactly n ~ k, i.e. n^2),
    //     which is why the adjoint is the default.
    const char *fwd_env = getenv("PHASIC_SOJOURN_FORWARD");
    const bool use_forward = (fwd_env != NULL && fwd_env[0] == '1' && fwd_env[1] == '\0');

    if (!use_forward) {
        // ---- (A) adjoint fast path ----
        double *adjoint = (double *) calloc(n, sizeof(double));
        if (adjoint == NULL) {
            PTD_LOG_ERROR("Failed to allocate adjoint sojourn vector (%zu doubles)", n);
            return NULL;
        }
        adjoint[0] = 1.0;  // starting vertex index = 0

        // Reverse replay. The guards mirror the forward path's limit
        // conventions in the transpose direction:
        //   0 x inf = 0 : a zero multiplier contributes nothing (skip);
        //   inf x 0 = 0 : the transpose multiplies adjoint[from], so skip the
        //                 infinite command when that operand is 0 (the forward
        //                 path guards on results[to], the operand it multiplies,
        //                 for the same reason).
        // Trap / deficit-sink vertices whose forward sojourn is NaN stay NaN
        // here as well, but they are never among the requested targets -- the
        // t-vertices that carry joint probability all have finite sojourn.
        for (size_t ci = compute->length; ci-- > 0; ) {
            struct ptd_reward_increase cmd = compute->commands[ci];

            if (cmd.multiplier == 0.0) {
                continue;
            }
            if (isinf(cmd.multiplier) && adjoint[cmd.from] == 0.0) {
                continue;
            }
            adjoint[cmd.to] += adjoint[cmd.from] * cmd.multiplier;
        }

        double *sojourn_times = (double *) malloc(k * sizeof(double));
        if (sojourn_times == NULL) {
            PTD_LOG_ERROR("Failed to allocate sojourn times array");
            free(adjoint);
            return NULL;
        }
        for (size_t r = 0; r < k; r++) {
            size_t vertex_idx = indices[r];
            if (vertex_idx >= n) {
                PTD_LOG_ERROR("Invalid vertex index %zu (graph has %zu vertices)", vertex_idx, n);
                free(adjoint);
                free(sojourn_times);
                return NULL;
            }
            sojourn_times[r] = adjoint[vertex_idx];
        }

        free(adjoint);
        return sojourn_times;
    }

    // ---- (B) forward reference path (PHASIC_SOJOURN_FORWARD=1) ----
    // Allocate results matrix as a single flat block of n*k doubles, row-major:
    // results_flat[v*k + r] = accumulated reward at vertex v for reward vector r.
    double *results_flat = (double *) calloc(n * k, sizeof(double));
    if (results_flat == NULL) {
        PTD_LOG_ERROR("Failed to allocate results matrix (%zu x %zu doubles)", n, k);
        return NULL;
    }

    // Initialize with one-hot vectors: reward vector r has value 1 at indices[r].
    for (size_t r = 0; r < k; r++) {
        size_t vertex_idx = indices[r];
        if (vertex_idx >= n) {
            PTD_LOG_ERROR("Invalid vertex index %zu (graph has %zu vertices)", vertex_idx, n);
            free(results_flat);
            return NULL;
        }
        results_flat[vertex_idx * k + r] = 1.0;
    }

    // Apply all elimination trace commands to k reward vectors:
    // results[from][r] += results[to][r] * multiplier for all r.
    for (size_t cmd_idx = 0; cmd_idx < compute->length; cmd_idx++) {
        struct ptd_reward_increase cmd = compute->commands[cmd_idx];

        // 0 x inf = 0 (limit interpretation): skip a zero multiplier.
        if (cmd.multiplier == 0.0) {
            continue;
        }

        double *from_row = results_flat + cmd.from * k;
        double *to_row = results_flat + cmd.to * k;
        double multiplier = cmd.multiplier;
        bool mult_is_inf = isinf(multiplier);

        for (size_t r = 0; r < k; r++) {
            // inf x 0 = 0 (limit interpretation).
            if (mult_is_inf && to_row[r] == 0.0) {
                continue;
            }
            from_row[r] += to_row[r] * multiplier;
        }
    }

    // Extract sojourn times at the starting vertex (index 0) for each column.
    double *sojourn_times = (double *) malloc(k * sizeof(double));
    if (sojourn_times == NULL) {
        PTD_LOG_ERROR("Failed to allocate sojourn times array");
        free(results_flat);
        return NULL;
    }

    for (size_t r = 0; r < k; r++) {
        sojourn_times[r] = results_flat[0 * k + r];  // Starting vertex index = 0
    }

    free(results_flat);

    return sojourn_times;
}

#ifdef PHASIC_B3_VALIDATORS
/* ================= B3 Tier-3 DE-RISK (non-shippable validator) =================
 * Replays the REAL parameterized+numeric elimination tape (via the _off form
 * from ptd_pcg_convert_to_offset, mirroring the executor at
 * ptd_graph_build_ex_absorbation_time_comp_graph_parameterized) at the current
 * edge weights, computing E[T] (result[0], reward=1) and, by FORWARD-MODE,
 * dE[T]/d(input edge weight k) for every _off input; plus a self-contained
 * central difference of the same tape forward. Confirms the tape is a complete,
 * correct differentiable trace and the _off operand provenance resolves. */
static double ptd_dbg_run_tape(
        const struct ptd_desc_reward_compute_parameterized_off *off,
        const double *mem0, const double *inp0, size_t n_vertices,
        const double *input_dot, double *ewt_dot_out) {
    /* Mirror ptd_graph_build_ex_absorbation_time_comp_graph_parameterized_off
     * EXACTLY: each operand resolves (via ptd_pcg_resolve) to a slot in EITHER
     * mem_base OR inputs[], and the executor writes *fromT in place -- so
     * elimination mutates input (edge-weight) slots too. We therefore keep
     * LOCAL mutable copies of BOTH mem and the input values, and resolve every
     * operand into whichever, never touching the graph's real edge weights. A
     * parallel forward-mode tangent rides alongside. */
    size_t md = off->mem_doubles, ni = off->n_inputs;
    int tang = (input_dot != NULL);
    double *mem = (double *) malloc(md * sizeof(double));
    double *inv = (double *) malloc((ni ? ni : 1) * sizeof(double));
    memcpy(mem, mem0, md * sizeof(double));
    memcpy(inv, inp0, ni * sizeof(double));
    double *memd = tang ? (double *) calloc(md, sizeof(double)) : NULL;
    double *invd = tang ? (double *) malloc((ni ? ni : 1) * sizeof(double)) : NULL;
    if (tang) for (size_t k = 0; k < ni; ++k) invd[k] = input_dot[k];
    size_t cap = off->length + 1;
    uint64_t *nf = (uint64_t*)malloc(cap*sizeof(uint64_t));
    uint64_t *nt = (uint64_t*)malloc(cap*sizeof(uint64_t));
    double *nm = (double*)malloc(cap*sizeof(double));
    double *nmd = (double*)malloc(cap*sizeof(double));
    size_t nc = 0;
/* Resolve an operand to a pointer into the LOCAL mem/inv copy (mirrors
 * ptd_pcg_resolve); RVAL for values, RTAN for tangents. NULL for OP_NULL. */
#define RVAL(op) ((op).kind==PTD_PCG_OP_MEM ? &mem[(op).mem_offset] : \
                 ((op).kind==PTD_PCG_OP_INPUT ? &inv[(op).input_idx] : (double*)NULL))
#define RTAN(op) ((op).kind==PTD_PCG_OP_MEM ? &memd[(op).mem_offset] : \
                 ((op).kind==PTD_PCG_OP_INPUT ? &invd[(op).input_idx] : (double*)NULL))
    for (size_t i = 0; i < off->length; ++i) {
        struct ptd_pcg_command_off c = off->commands[i];
        double *rf = RVAL(c.fromT), *rt = RVAL(c.toT), *rm = RVAL(c.multiplierptr);
        double *rfd = tang ? RTAN(c.fromT) : NULL;
        double *rtd = tang ? RTAN(c.toT)   : NULL;
        double *rmd = tang ? RTAN(c.multiplierptr) : NULL;
        switch (c.type) {
            case 0: /* NEW_ADD -- mirror add_command(): a DIAGONAL (from==to)
                     * command stores (multiplier - 1) (the identity subtraction
                     * from Gaussian elimination); off-diagonal stores it as-is.
                     * The -1 is a constant, so the tangent is unchanged. */
                nf[nc]=c.from; nt[nc]=c.to;
                nm[nc] = (c.from==c.to) ? (*rm - 1.0) : *rm;
                if (tang) nmd[nc]=*rmd;
                nc++;
                break;
            case 1: { /* P: *fromT += *toT * multiplier(const) */
                if (tang) *rfd += (*rtd)*c.multiplier;
                *rf += (*rt)*c.multiplier; } break;
            case 3: { /* PP: *fromT += *toT * *mptr */
                double tv=*rt, mv=*rm;
                if (tang) *rfd += (*rtd)*mv + tv*(*rmd);
                *rf += tv*mv; } break;
            case 2: { /* INV: *fromT = 1/ *fromT */
                double fv=*rf;
                if (tang) *rfd = -(*rfd)/(fv*fv);
                *rf = 1.0/fv; } break;
            case 4: /* ONE_MINUS: *fromT = 1 - *fromT */
                if (tang) *rfd = -(*rfd);
                *rf = 1.0 - *rf; break;
            case 5: { /* DIVIDE: *fromT /= *toT */
                double fv=*rf, tv=*rt;
                if (tang) *rfd = (*rfd)/tv - fv*(*rtd)/(tv*tv);
                *rf = fv/tv; } break;
            case 6: /* ZERO */ *rf=0.0; if (tang) *rfd=0.0; break;
            default: break;
        }
    }
    double *res = (double*)malloc(n_vertices*sizeof(double));
    double *resd = tang ? (double*)calloc(n_vertices,sizeof(double)) : NULL;
    for (size_t v=0; v<n_vertices; ++v){ res[v]=1.0; if(resd) resd[v]=0.0; }
    for (size_t j=0;j<nc;++j){
        double m = nm[j], md_ = nmd ? nmd[j] : 0.0;
        /* inf*0 (res[to]==0): native skips entirely (0*inf=0 limit). */
        if (isinf(m) && res[nt[j]]==0.0) continue;
        /* Tangent d(res[from]) += d(res[to])*m + res[to]*d(m). This must be
         * applied even when m==0 (a diagonal whose weight is exactly 1, so
         * stored multiplier weight-1==0) -- there m_dot can still be nonzero,
         * and skipping the whole command (as the primal does) would drop that
         * gradient term (the forward-mode bug at mixed diagonal-==-1 points). */
        if (resd) resd[nf[j]] += resd[nt[j]]*m + res[nt[j]]*md_;
        /* Primal: skip m==0 to match native exactly and avoid 0*inf=nan. */
        if (m != 0.0) res[nf[j]] += res[nt[j]]*m;
    }
    double q = res[0];
    if (ewt_dot_out) *ewt_dot_out = resd ? resd[0] : 0.0;
    free(mem); free(inv); free(nf); free(nt); free(nm); free(nmd); free(res);
    if (memd) free(memd);
    if (invd) free(invd);
    if (resd) free(resd);
#undef RVAL
#undef RTAN
    return q;
}

/* Acquire the CLEAN pre-execution _off tape (the exact tape native replays), not
 * a post-hoc convert of the dirtied param-tape mem. Force a fresh param-tape
 * rebuild (dph_compute_invalidated wipes any prior build), enable the
 * pre-executor SELFCHECK convert (:2069) and the stash hook (:2115), then
 * precompute -- the stash lands in graph->_dbg_off_clean with mem_base = the
 * state the executor STARTS from. Ownership is TRANSFERRED to the caller (the
 * graph field is cleared); caller must ptd_pcg_desc_off_destroy() it. */
static struct ptd_desc_reward_compute_parameterized_off *
ptd_dbg_acquire_clean_off(struct ptd_graph *graph) {
    setenv("PHASIC_PCG_SELFCHECK", "1", 1);
    setenv("PHASIC_DBG_STASH_OFF", "1", 1);
    if (graph->_dbg_off_clean != NULL) {
        ptd_pcg_desc_off_destroy(
            (struct ptd_desc_reward_compute_parameterized_off *)graph->_dbg_off_clean);
        graph->_dbg_off_clean = NULL;
    }
    graph->dph_compute_invalidated = true;
    int pc = ptd_precompute_reward_compute_graph(graph);
    unsetenv("PHASIC_DBG_STASH_OFF");
    unsetenv("PHASIC_PCG_SELFCHECK");
    if (pc) return NULL;
    struct ptd_desc_reward_compute_parameterized_off *off =
        (struct ptd_desc_reward_compute_parameterized_off *)graph->_dbg_off_clean;
    graph->_dbg_off_clean = NULL;  /* transfer ownership to caller */
    return off;
}

int ptd_debug_fwdmode_grad(struct ptd_graph *graph,
        double *ewt_out, double **fwd_out, double **cd_out, size_t *ni_out) {
    struct ptd_desc_reward_compute_parameterized_off *off =
        ptd_dbg_acquire_clean_off(graph);
    if (off == NULL) return -1;  /* stash did not fire (not parameterized?) */
    size_t n = graph->vertices_length, ni = off->n_inputs, md = off->mem_doubles;
    double *mem0 = (double*)malloc(md*sizeof(double));
    memcpy(mem0, off->mem_base, md*sizeof(double));
    /* Snapshot initial input VALUES (current edge weights) into a local copy;
     * central difference perturbs this copy, never the graph's edge weights. */
    double *inp0 = (double*)malloc((ni?ni:1)*sizeof(double));
    for (size_t k=0;k<ni;++k) inp0[k] = *off->inputs[k];
    double *fwd = (double*)calloc(ni,sizeof(double));
    double *cd  = (double*)calloc(ni,sizeof(double));
    double *idot= (double*)calloc(ni,sizeof(double));
    *ewt_out = ptd_dbg_run_tape(off, mem0, inp0, n, NULL, NULL);
    for (size_t k=0;k<ni;++k){
        memset(idot,0,ni*sizeof(double)); idot[k]=1.0;
        double qd=0.0; ptd_dbg_run_tape(off, mem0, inp0, n, idot, &qd); fwd[k]=qd;
        double w0=inp0[k];
        double eps = 1e-6*(fabs(w0)>0.0?fabs(w0):1.0);
        inp0[k]=w0+eps; double qp = ptd_dbg_run_tape(off, mem0, inp0, n, NULL, NULL);
        inp0[k]=w0-eps; double qm = ptd_dbg_run_tape(off, mem0, inp0, n, NULL, NULL);
        inp0[k]=w0; cd[k]=(qp-qm)/(2.0*eps);
    }
    free(mem0); free(inp0); free(idot);
    ptd_pcg_desc_off_destroy(off);  /* acquire transferred ownership to us */
    *fwd_out=fwd; *cd_out=cd; *ni_out=ni;
    return 0;
}

/* ===== B3 Batch-1: reverse-mode theta-adjoint over the REAL _off two-tier tape.
 * Computes dQ/d(input edge weight) for Q = E[T] = result[target] with seed all-1
 * (first moment, continuous). Ports the verified reference interpreter
 * experiments/dr_twotier_full_adjoint.py, adapted to the real _off form and the
 * two Batch-0 findings:
 *   - add_command stores (multiplier-1) for DIAGONAL (from==to) numeric commands;
 *     the -1 is constant so the glue bar[mptr]+=dm is unchanged, but the primal /
 *     snapshot / stage-1 transpose use the real m_c = weight-1.
 *   - the mult==0 primal-skip must NOT skip the gradient: dm_c=adj[a]*snap_to is
 *     emitted regardless; only the transpose adj[b]+=adj[a]*m_c is a no-op at 0.
 * Operands resolve to mem_base OR inputs[]; adjoints ride on bmem[]/binp[] and the
 * answer is binp[]. Never touches the graph's real edge weights (local copies). */
static int ptd_dbg_reverse_tape(
        const struct ptd_desc_reward_compute_parameterized_off *off,
        const double *mem0, const double *inp0, size_t n_vertices,
        size_t target, double *q_out, double *grad_out /* size ni */) {
    size_t md = off->mem_doubles, ni = off->n_inputs, L = off->length;
    double *mem = (double*)malloc(md*sizeof(double));       memcpy(mem, mem0, md*sizeof(double));
    double *inv = (double*)malloc((ni?ni:1)*sizeof(double));memcpy(inv, inp0, ni*sizeof(double));
    double *s0  = (double*)malloc((L?L:1)*sizeof(double));  /* operand primal snapshots */
    double *s1  = (double*)malloc((L?L:1)*sizeof(double));
    uint64_t *na=(uint64_t*)malloc((L?L:1)*sizeof(uint64_t));/* numeric command from */
    uint64_t *nb=(uint64_t*)malloc((L?L:1)*sizeof(uint64_t));/* numeric command to   */
    double *nm  = (double*)malloc((L?L:1)*sizeof(double));  /* m_c (diagonal -1 applied) */
    double *nto = (double*)malloc((L?L:1)*sizeof(double));  /* snap result[to] at exec */
    size_t nc = 0;
#define RV(op) ((op).kind==PTD_PCG_OP_MEM ? &mem[(op).mem_offset] : \
               ((op).kind==PTD_PCG_OP_INPUT ? &inv[(op).input_idx] : (double*)NULL))
    /* ---- param tape forward: mutate mem/inv, snapshot operand primals, record
     * numeric commands (m_c with diagonal -1) ---- */
    for (size_t i=0;i<L;++i) {
        struct ptd_pcg_command_off c = off->commands[i];
        double *rf=RV(c.fromT), *rt=RV(c.toT), *rm=RV(c.multiplierptr);
        switch (c.type) {
            case 0: { /* NEW_ADD */
                double mc = (c.from==c.to) ? (*rm - 1.0) : *rm;
                na[nc]=c.from; nb[nc]=c.to; nm[nc]=mc; nc++;
            } break;
            case 1: /* P  */ *rf += (*rt)*c.multiplier; break;
            case 3: /* PP */ s0[i]=*rt; s1[i]=*rm; *rf += (*rt)*(*rm); break;
            case 2: /* INV*/ s0[i]=*rf; *rf = 1.0/(*rf); break;
            case 4: /* OM */ *rf = 1.0 - *rf; break;
            case 5: /* DIV*/ s0[i]=*rf; s1[i]=*rt; *rf = (*rf)/(*rt); break;
            case 6: /* ZERO*/ *rf = 0.0; break;
            default: break;
        }
    }
    /* ---- numeric replay forward: seed all-1, snapshot result[to] per command ---- */
    double *res = (double*)malloc(n_vertices*sizeof(double));
    for (size_t v=0;v<n_vertices;++v) res[v]=1.0;
    for (size_t j=0;j<nc;++j) {
        nto[j] = res[nb[j]];                                /* snapshot BEFORE update */
        double m = nm[j];
        if (isinf(m) && res[nb[j]]==0.0) continue;          /* inf*0 skip (native) */
        if (m != 0.0) res[na[j]] += res[nb[j]]*m;           /* primal skip on 0 (native) */
    }
    if (q_out) *q_out = res[target];
    /* ---- stage 1: reverse the numeric replay -> dm[c] ---- */
    double *adj = (double*)calloc(n_vertices, sizeof(double));
    adj[target] = 1.0;
    double *dm = (double*)malloc((nc?nc:1)*sizeof(double));
    for (long c=(long)nc-1;c>=0;--c) {
        double m = nm[c];
        if (isinf(m) && nto[c]==0.0) { dm[c]=0.0; continue; }/* mirror forward inf*0 skip */
        dm[c] = adj[na[c]] * nto[c];                        /* emit BEFORE transpose */
        if (m != 0.0) adj[nb[c]] += adj[na[c]] * m;         /* transpose; m==0 no-op */
    }
    /* ---- stage 2: reverse the param tape (REPLACE/kill per op) + glue dm ---- */
    double *bmem = (double*)calloc(md, sizeof(double));
    double *binp = (double*)calloc(ni?ni:1, sizeof(double));
#define RB(op) ((op).kind==PTD_PCG_OP_MEM ? &bmem[(op).mem_offset] : \
               ((op).kind==PTD_PCG_OP_INPUT ? &binp[(op).input_idx] : (double*)NULL))
    long numptr = (long)nc - 1;
    for (long i=(long)L-1;i>=0;--i) {
        struct ptd_pcg_command_off c = off->commands[i];
        double *bf=RB(c.fromT), *bt=RB(c.toT), *bm=RB(c.multiplierptr);
        switch (c.type) {
            case 0: /* NEW_ADD glue (in-order at the op) */
                if (bm) *bm += dm[numptr];
                numptr--; break;
            case 1: /* P  */ *bt += (*bf)*c.multiplier; break;                 /* keep bf */
            case 3: { /* PP */ double v=*bf; *bt += v*s1[i]; *bm += v*s0[i]; } break; /* keep bf */
            case 2: /* INV*/ *bf = (*bf)*(-1.0/(s0[i]*s0[i])); break;          /* REPLACE */
            case 4: /* OM */ *bf = -(*bf); break;                             /* REPLACE */
            case 5: { /* DIV*/ double v=*bf; *bt += v*(-s0[i]/(s1[i]*s1[i])); *bf = v/s1[i]; } break; /* REPLACE */
            case 6: /* ZERO*/ *bf = 0.0; break;                               /* kill */
            default: break;
        }
    }
    for (size_t k=0;k<ni;++k) grad_out[k] = binp[k];
    free(mem); free(inv); free(s0); free(s1); free(na); free(nb); free(nm);
    free(nto); free(res); free(adj); free(dm); free(bmem); free(binp);
#undef RV
#undef RB
    return 0;
}

int ptd_debug_reverse_grad(struct ptd_graph *graph,
        double *ewt_out, double **grad_out, size_t *ni_out) {
    struct ptd_desc_reward_compute_parameterized_off *off =
        ptd_dbg_acquire_clean_off(graph);
    if (off == NULL) return -1;
    size_t n = graph->vertices_length, ni = off->n_inputs, md = off->mem_doubles;
    double *mem0 = (double*)malloc(md*sizeof(double));
    memcpy(mem0, off->mem_base, md*sizeof(double));
    double *inp0 = (double*)malloc((ni?ni:1)*sizeof(double));
    for (size_t k=0;k<ni;++k) inp0[k] = *off->inputs[k];
    double *grad = (double*)calloc(ni?ni:1, sizeof(double));
    double q = 0.0;
    int rc = ptd_dbg_reverse_tape(off, mem0, inp0, n, /*target=*/0, &q, grad);
    free(mem0); free(inp0);
    ptd_pcg_desc_off_destroy(off);  /* acquire transferred ownership to us */
    if (rc) { free(grad); return -1; }
    *ewt_out = q; *grad_out = grad; *ni_out = ni;
    return 0;
}
#endif /* PHASIC_B3_VALIDATORS (validators: run_tape/acquire/fwdmode/reverse) */

/* B3 Batch-3 (MPFR safety gate): would the primal (expected_waiting_time) divert
 * to MPFR for this tape? If so, the double-precision tape adjoint would be
 * inconsistent with the MPFR forward, so the exact gradient must decline (the
 * caller falls back to FD, which differentiates the same MPFR forward). Mirrors
 * ptd_expected_waiting_time's gate EXACTLY (nm[] carries the same numeric
 * multipliers as reward_compute_graph, incl. the diagonal -1). Only relevant when
 * MPFR is compiled in; without it the forward stays double and the adjoint
 * matches, so no fallback is needed. */
static int ptd_dbg_tape_needs_mpfr(const double *nm, size_t nc) {
#ifdef HAVE_MPFR
    double pmax = 0.0, pmin = INFINITY;
    for (size_t c = 0; c < nc; ++c) {
        double m = nm[c];
        if (!isinf(m) && m != 0.0) {
            double a = fabs(m);
            if (a > pmax) pmax = a;
            if (a < pmin) pmin = a;
        }
    }
    double cond = (pmin != INFINITY && pmax > 0.0) ? (pmax / pmin) : 0.0;
    double thr = 1e12;
    const char *e = getenv("PHASIC_CONDITION_THRESHOLD");
    if (e != NULL) thr = atof(e);
    if (getenv("PHASIC_FORCE_MPFR") != NULL) return 1;
    return cond > thr;
#else
    (void)nm; (void)nc;
    return 0;
#endif
}

#ifdef PHASIC_B3_VALIDATORS
/* ===== B3 Batch-2: reverse-mode gradient of the FIRST moment. Validator-only:
 * superseded by ptd_moments_grad_theta (K=1 case). Kept as a de-risk oracle.
 * Computes E[T] (=moments[0], result[0], seed all-1) and d(E[T])/dtheta_j for a
 * continuous / weight_mode=linear / monolithic parameterized graph, WITHOUT env
 * vars or global state (thread-safe): builds a LOCAL param-tape recorder (whose
 * mem starts clean because the numeric executor never runs on it), converts to
 * the _off form (clean mem_base), runs the validated reverse tape adjoint for
 * dQ/d(edge weight), then contracts to dtheta via the linear edge Jacobian
 * dw_e/dtheta_j = coefficients[j]. dtheta_out must hold graph->param_length
 * doubles. Returns 0 on success; -1 if not applicable (caller falls back to FD):
 * non-parameterized, external/log/formula inputs, or a non-finite result. */
int ptd_moment0_grad_theta(struct ptd_graph *graph,
        double *ewt_out, double *dtheta_out) {
    if (!graph->parameterized || graph->param_length == 0) return -1;
    size_t P = graph->param_length;
    /* Build a LOCAL param tape (the recorder; does NOT run the numeric executor,
     * so its mem stays in the clean pre-execution state). Match the primal's
     * ordering choice. Not attached to graph->parameterized_reward_compute_graph,
     * so it never interferes with the graph's own cached tape. */
    struct ptd_desc_reward_compute_parameterized *ptape =
        graph->use_dyn_ordering
            ? ptd_graph_ex_absorbation_time_comp_graph_parameterized_dyn(graph)
            : ptd_graph_ex_absorbation_time_comp_graph_parameterized(graph);
    if (ptape == NULL) return -1;
    struct ptd_desc_reward_compute_parameterized_off *off =
        ptd_pcg_convert_to_offset(ptape, graph, NULL, 0);  /* clean mem_base */
    if (off == NULL) { ptd_parameterized_reward_compute_graph_destroy(ptape); return -1; }

    size_t n = graph->vertices_length, ni = off->n_inputs, md = off->mem_doubles;
    double *mem0 = (double*)malloc(md*sizeof(double));
    memcpy(mem0, off->mem_base, md*sizeof(double));
    double *inp0 = (double*)malloc((ni?ni:1)*sizeof(double));
    for (size_t k=0;k<ni;++k) inp0[k] = *off->inputs[k];
    double *ge = (double*)calloc(ni?ni:1, sizeof(double));  /* dQ/d(edge weight) */
    double q = 0.0;
    /* MPFR gate backport (successor pattern, ptd_moments_grad_theta): nm[] are
     * locals of ptd_dbg_reverse_tape and not otherwise exposed, so recover them
     * by replaying stage-0 on SCRATCH copies (mem0/inp0 must stay clean for the
     * reverse-tape call below). Alloc failure declines (FD fallback) rather
     * than proceeding ungated. */
    {
        size_t L = off->length;
        double *smem = (double*)malloc(md*sizeof(double));
        double *sinv = (double*)malloc((ni?ni:1)*sizeof(double));
        double *nm_local = (double*)malloc((L?L:1)*sizeof(double));
        size_t nc_local = 0;
        int gate = 1;
        if (smem != NULL && sinv != NULL && nm_local != NULL) {
            memcpy(smem, mem0, md*sizeof(double));
            memcpy(sinv, inp0, ni*sizeof(double));
#define D2RV(op) ((op).kind==PTD_PCG_OP_MEM ? &smem[(op).mem_offset] : \
                 ((op).kind==PTD_PCG_OP_INPUT ? &sinv[(op).input_idx] : (double*)NULL))
            for (size_t i=0;i<L;++i) {
                struct ptd_pcg_command_off c = off->commands[i];
                double *rf=D2RV(c.fromT), *rt=D2RV(c.toT), *rm=D2RV(c.multiplierptr);
                switch (c.type) {
                    case 0: { /* NEW_ADD: record m_c with the diagonal -1 */
                        double mc = (c.from==c.to) ? (*rm - 1.0) : *rm;
                        nm_local[nc_local++] = mc;
                    } break;
                    case 1: *rf += (*rt)*c.multiplier; break;
                    case 3: *rf += (*rt)*(*rm); break;
                    case 2: *rf = 1.0/(*rf); break;
                    case 4: *rf = 1.0 - *rf; break;
                    case 5: *rf = (*rf)/(*rt); break;
                    case 6: *rf = 0.0; break;
                    default: break;
                }
            }
#undef D2RV
            gate = ptd_dbg_tape_needs_mpfr(nm_local, nc_local);
        }
        free(smem); free(sinv); free(nm_local);
        if (gate) {
            free(mem0); free(inp0); free(ge);
            ptd_pcg_desc_off_destroy(off);
            ptd_parameterized_reward_compute_graph_destroy(ptape);
            return -1;
        }
    }
    int rc = ptd_dbg_reverse_tape(off, mem0, inp0, n, /*target=*/0, &q, ge);

    int ok = (rc == 0);
    for (size_t j=0;j<P;++j) dtheta_out[j] = 0.0;
    /* Contract edge -> theta: dtheta_j = sum_k ge[k] * coeff_j(edge k). Each input
     * k must be an internal EDGE weight (byte 0); EXTERNAL (SCC) / sub-double
     * inputs mean the graph is out of the linear/monolithic scope -> FD fallback. */
    for (size_t k=0; ok && k<ni; ++k) {
        struct ptd_pcg_input_spec sp = off->input_specs[k];
        if (sp.kind != PTD_PCG_PTR_EDGE || sp.byte != 0
                || sp.v >= graph->vertices_length
                || sp.e >= graph->vertices[sp.v]->edges_length) { ok = 0; break; }
        struct ptd_edge *e = graph->vertices[sp.v]->edges[sp.e];
        /* Constant tape-input edges (coefficients_length==0, e.g. aux vertices)
         * have dw/dtheta = 0 and a NULL coefficient array: skip, do not deref
         * (successor pattern). */
        if (e->coefficients_length == 0) continue;
        for (size_t j=0;j<P;++j) dtheta_out[j] += ge[k] * e->coefficients[j];
    }
    if (ok) for (size_t j=0;j<P;++j) if (!isfinite(dtheta_out[j])) { ok = 0; break; }
    if (ok && !isfinite(q)) ok = 0;
    if (ok && ewt_out) *ewt_out = q;

    free(mem0); free(inp0); free(ge);
    ptd_pcg_desc_off_destroy(off);
    ptd_parameterized_reward_compute_graph_destroy(ptape);
    return ok ? 0 : -1;
}
#endif /* PHASIC_B3_VALIDATORS (validator: ptd_moment0_grad_theta) */

/* ===== B3 Batch-3: exact Jacobian d[m_0..m_{K-1}]/dtheta for the standard
 * moment vector (K=nr_moments). The moment recurrence (graph_builder.cpp:512)
 * is a_1=ewt(ones), a_{j+1}=ewt(a_j), m_k=(k+1)!*a_{k+1}[0], and EVERY ewt
 * replays the SAME numeric tape with a new seed. The reverse is a CHAIN: each
 * replay's reverse yields dm[] contributions AND a seed-adjoint that becomes the
 * next-lower replay's output cotangent; stage-2 (param reverse -> edge grads)
 * runs once per output moment on the accumulated dm[]. Verified build-free vs
 * JAX autodiff (experiments/dr_moment_chain_adjoint.py, 230/230). J_out is
 * row-major nr_moments*param_length; continuous / linear / monolithic. Returns 0
 * on success, -1 (FD fallback) otherwise. */
/* ===== Batch 0: shared stage-0/1/2 core for the moments-gradient family.
 * Extracted verbatim from ptd_moments_grad_theta (the comment-richest of the
 * three code-identical copies); the linear/log/dph variants differ ONLY in
 * their contraction step (the switch below) and their wrapper-side pre/post
 * passes. Ownership: WRAPPERS build and destroy ptape/off (and any per-kind
 * ctx); the core allocates and frees only its own locals. Consumers extend
 * this static core's signature when they land (Batch A: rewards; Batch B:
 * PTD_B3_FORMULA + internal pre-outk dw/dtheta stage; Batch C: its
 * pre-contraction exit) — nothing speculative is designed in. */
enum ptd_b3_contract { PTD_B3_LINEAR, PTD_B3_LOG, PTD_B3_DPH };

struct ptd_b3_dph_ctx { const double *Sv; const double *SigmaCv; };

static int ptd_b3_moments_core(
        struct ptd_graph *graph,
        const struct ptd_desc_reward_compute_parameterized_off *off,
        int nr_moments,
        const double *theta, size_t theta_len,   /* NULL/0 for linear */
        enum ptd_b3_contract kind,
        const struct ptd_b3_dph_ctx *dph_ctx,    /* non-NULL iff DPH */
        double *J_out) {
    (void)theta_len;  /* wrapper-validated; kept in the signature for
                         future consumers (Batch A/B extend it) */
    size_t P = graph->param_length, K = (size_t)nr_moments;
    size_t n = graph->vertices_length, ni = off->n_inputs, md = off->mem_doubles, L = off->length;
    /* [seeding block 1/2 -- Deferred-1 marker] `target` selects the tape
     * output vertex. A future per-SCC cotangent-seeded VJP (deferred-1 plan
     * section 4-P2, declined for Batch 0) would generalize ONLY this seeding
     * block (parts 1+2), never the contraction: stage-1 seeding (input side)
     * and any stage-2 exit are orthogonal and compose. */
    size_t target = 0;
    double *mem = (double*)malloc(md*sizeof(double)); memcpy(mem, off->mem_base, md*sizeof(double));
    double *inv = (double*)malloc((ni?ni:1)*sizeof(double));
    for (size_t k=0;k<ni;++k) inv[k]=*off->inputs[k];
    double *s0=(double*)malloc((L?L:1)*sizeof(double));
    double *s1=(double*)malloc((L?L:1)*sizeof(double));
    uint64_t *na=(uint64_t*)malloc((L?L:1)*sizeof(uint64_t));
    uint64_t *nb=(uint64_t*)malloc((L?L:1)*sizeof(uint64_t));
    double *nm=(double*)malloc((L?L:1)*sizeof(double));
    size_t nc=0;
#define RV(op) ((op).kind==PTD_PCG_OP_MEM ? &mem[(op).mem_offset] : \
               ((op).kind==PTD_PCG_OP_INPUT ? &inv[(op).input_idx] : (double*)NULL))
    /* param tape forward: snapshot operand primals, record numeric commands */
    for (size_t i=0;i<L;++i) {
        struct ptd_pcg_command_off c = off->commands[i];
        double *rf=RV(c.fromT), *rt=RV(c.toT), *rm=RV(c.multiplierptr);
        switch (c.type) {
            case 0: { double mcv=(c.from==c.to)?(*rm-1.0):*rm; na[nc]=c.from; nb[nc]=c.to; nm[nc]=mcv; nc++; } break;
            case 1: *rf += (*rt)*c.multiplier; break;
            case 3: s0[i]=*rt; s1[i]=*rm; *rf += (*rt)*(*rm); break;
            case 2: s0[i]=*rf; *rf = 1.0/(*rf); break;
            case 4: *rf = 1.0 - *rf; break;
            case 5: s0[i]=*rf; s1[i]=*rt; *rf = (*rf)/(*rt); break;
            case 6: *rf = 0.0; break;
            default: break;
        }
    }
    /* MPFR safety gate: if the primal would use MPFR for this (ill-conditioned)
     * tape, the double adjoint is inconsistent with the MPFR forward -> decline
     * so the caller falls back to FD. */
    if (ptd_dbg_tape_needs_mpfr(nm, nc)) {
        free(mem); free(inv); free(s0); free(s1); free(na); free(nb); free(nm);
        /* off/ptape are wrapper-owned; the core frees only its locals */
        return -1;
    }
    /* forward moment chain: seeds[j] (j=0..K), snap_to per replay (j=1..K) */
    double *seeds = (double*)malloc((K+1)*n*sizeof(double));
    double *snaptos = (double*)malloc((K?K:1)*(nc?nc:1)*sizeof(double));
    for (size_t v=0; v<n; ++v) seeds[v]=1.0;                 /* a_0 = ones */
    for (size_t j=1;j<=K;++j) {
        double *seed = seeds + (j-1)*n, *out = seeds + j*n, *st = snaptos + (j-1)*nc;
        /* [Batch-A rewards hook 1/2: seed-scale line] */
        for (size_t v=0; v<n; ++v) out[v]=seed[v];
        for (size_t c=0;c<nc;++c) {
            st[c]=out[nb[c]]; double m=nm[c];
            if (isinf(m) && out[nb[c]]==0.0) continue;
            if (m!=0.0) out[na[c]] += out[nb[c]]*m;
        }
    }
    /* per-output-moment reverse chain + stage-2 + edge->theta contraction */
    int ok=1;
    double *dm=(double*)malloc((nc?nc:1)*sizeof(double));
    double *bar_out=(double*)malloc(n*sizeof(double));
    double *adj=(double*)malloc(n*sizeof(double));
    double *bmem=(double*)malloc(md*sizeof(double));
    double *binp=(double*)malloc((ni?ni:1)*sizeof(double));
#define RB(op) ((op).kind==PTD_PCG_OP_MEM ? &bmem[(op).mem_offset] : \
               ((op).kind==PTD_PCG_OP_INPUT ? &binp[(op).input_idx] : (double*)NULL))
    for (size_t outk=0; ok && outk<K; ++outk) {
        for (size_t c=0;c<nc;++c) dm[c]=0.0;
        for (size_t v=0; v<n; ++v) bar_out[v]=0.0;
        for (size_t j=K; j>=1; --j) {
            for (size_t v=0; v<n; ++v) adj[v]=bar_out[v];
            /* [seeding block 2/2 -- Deferred-1 marker] factorial one-hot seed */
            if (outk == j-1) {                                /* m_outk = j! * a_j[target] */
                double fac=1.0; for (size_t t=2;t<=j;++t) fac*=(double)t;
                adj[target] += fac;
            }
            double *st = snaptos + (j-1)*nc;
            for (long c=(long)nc-1;c>=0;--c) {
                double m=nm[c];
                if (isinf(m) && st[c]==0.0) continue;
                dm[c] += adj[na[c]] * st[c];
                if (m!=0.0) adj[nb[c]] += adj[na[c]] * m;
            }
            /* [Batch-A rewards hook 2/2: adjoint-scale line] */
            for (size_t v=0; v<n; ++v) bar_out[v]=adj[v];     /* seed-adjoint -> replay j-1 */
            if (j==1) break;                                  /* size_t underflow guard */
        }
        /* stage-2: reverse the param tape ONCE on the accumulated dm[] */
        for (size_t i=0;i<md;++i) bmem[i]=0.0;
        for (size_t k=0;k<ni;++k) binp[k]=0.0;
        long numptr=(long)nc-1;
        for (long i=(long)L-1;i>=0;--i) {
            struct ptd_pcg_command_off c = off->commands[i];
            double *bf=RB(c.fromT), *bt=RB(c.toT), *bm=RB(c.multiplierptr);
            switch (c.type) {
                case 0: if (bm) *bm += dm[numptr]; numptr--; break;
                case 1: *bt += (*bf)*c.multiplier; break;
                case 3: { double v=*bf; *bt += v*s1[i]; *bm += v*s0[i]; } break;
                case 2: *bf = (*bf)*(-1.0/(s0[i]*s0[i])); break;
                case 4: *bf = -(*bf); break;
                case 5: { double v=*bf; *bt += v*(-s0[i]/(s1[i]*s1[i])); *bf = v/s1[i]; } break;
                case 6: *bf = 0.0; break;
                default: break;
            }
        }
        /* contract edge -> theta into J_out row outk (per-kind step) */
        for (size_t j=0;j<P;++j) J_out[outk*P + j] = 0.0;
        for (size_t k=0; ok && k<ni; ++k) {
            struct ptd_pcg_input_spec sp = off->input_specs[k];
            if (sp.kind != PTD_PCG_PTR_EDGE || sp.byte != 0
                    || sp.v >= graph->vertices_length
                    || sp.e >= graph->vertices[sp.v]->edges_length) { ok=0; break; }
            struct ptd_edge *e = graph->vertices[sp.v]->edges[sp.e];
            switch (kind) {
                case PTD_B3_LINEAR:
                    /* The tape registers EVERY edge weight as a free input, including
                     * coefficient-less constant edges (coefficients_length==0,
                     * coefficients==NULL -- e.g. the aux back-edges from
                     * add_aux_vertex/add_aux_vertex_constant, used by
                     * Graph.discretize() and Graph.joint_stop_prob_graph()). Such an
                     * edge's weight never depends on theta, so its exact gradient
                     * contribution is 0 -- skip it rather than dereference a NULL
                     * coefficients pointer (found via the discrete/was_dph extension,
                     * see B3-DISCRETE-MERGE-REVIEW.md sec 3.1: this was an always-
                     * latent segfault for ANY continuous parameterized graph built
                     * with add_aux_vertex/add_aux_vertex_constant, not just discrete
                     * ones -- just never previously exercised). */
                    if (e->coefficients_length == 0) break;
                    for (size_t j=0;j<P;++j) J_out[outk*P + j] += binp[k] * e->coefficients[j];
                    break;
                case PTD_B3_LOG:
                    /* log-mode product rule: dw_e/dtheta_j = w_e/theta_j for
                     * ALL j (every param multiplies into every log-mode edge
                     * -- not conditioned on coefficients[j], unlike linear).
                     * Same tape-input hygiene as linear/dph: a
                     * coefficient-less constant edge's weight never depends on
                     * theta (contributes 0); a starting-vertex edge's weight is
                     * never recomputed by update_weights regardless of mode (its
                     * true dw/dtheta is 0 too) -- included for consistency with
                     * ptd_moments_grad_theta_dph even though a directly-constructed
                     * parameterized start edge was NOT found to register as a tape
                     * input in practice (verified empirically against the shipped
                     * linear function; see b3-log-weight-mode-plan.md). NOTE: this
                     * guard's unreachability currently rests on
                     * _graph_serialize.py's `if False:` around start_param_edges
                     * (serialize() never emits a parameterized start edge) -- if a
                     * future change revives that branch, re-verify this guard is
                     * still exercised/correct rather than assuming it stays dead.
                     * The e->weight read is LIVE from the edge (not the inv[k]
                     * snapshot) -- kept verbatim from the pre-extraction code. */
                    if (graph->vertices[sp.v] == graph->starting_vertex) break;
                    if (e->coefficients_length == 0) break;
                    for (size_t j=0;j<P;++j) {
                        J_out[outk*P + j] += binp[k] * (e->weight / theta[j]);
                    }
                    break;
                case PTD_B3_DPH:
                    /* was_dph: renorm quotient rule (sibling coupling via
                     * Sv/SigmaCv from the wrapper's pre-pass); else: plain
                     * linear rule (same as ptd_moments_grad_theta).
                     * The constant-edge skip mirrors linear's (coefficients is
                     * NULL for discretize()'s aux back-edges). A starting-vertex
                     * edge is an IPV probability: update_weights() NEVER
                     * recomputes its weight from theta (skipped from both the
                     * coefficient dot-product AND, effectively, the renorm --
                     * its out-edges' weights are theta-independent constants,
                     * so their total is too), regardless of whether it happens
                     * to carry a (possibly widened/padded) non-empty
                     * coefficient array. Its true dp_e/dtheta is therefore
                     * identically 0 -- skip it rather than risk 0 * (a
                     * quotient-rule term that can be +-inf when Sv is
                     * uncomputed/zero for this vertex, which is 0*inf = NaN in
                     * IEEE754, not 0). */
                    if (e->coefficients_length == 0
                            || graph->vertices[sp.v] == graph->starting_vertex) break;
                    if (graph->was_dph) {
                        double p_e = e->weight;
                        double S = dph_ctx->Sv[sp.v];
                        const double *sigma = dph_ctx->SigmaCv + sp.v * P;
                        for (size_t j=0;j<P;++j) {
                            J_out[outk*P + j] += binp[k] * (e->coefficients[j] - p_e * sigma[j]) / S;
                        }
                    } else {
                        for (size_t j=0;j<P;++j) J_out[outk*P + j] += binp[k] * e->coefficients[j];
                    }
                    break;
                default:
                    ok = 0;
                    break;
            }
        }
    }
    if (ok) for (size_t x=0; x<K*P; ++x) if (!isfinite(J_out[x])) { ok=0; break; }

    free(mem); free(inv); free(s0); free(s1); free(na); free(nb); free(nm);
    free(seeds); free(snaptos); free(dm); free(bar_out); free(adj); free(bmem); free(binp);
#undef RV
#undef RB
    return ok ? 0 : -1;
}

/* Public linear-mode entry (see api/c/phasic.h for the full contract):
 * J_out row-major nr_moments x param_length; 0 = success, -1 = not
 * applicable (caller falls back to FD). Thin wrapper over
 * ptd_b3_moments_core. */
int ptd_moments_grad_theta(struct ptd_graph *graph, int nr_moments,
        double *J_out) {
    if (!graph->parameterized || graph->param_length == 0) return -1;
    if (nr_moments < 1) return -1;
    /* Batch 0 M4 (deliberate, reviewed behavior addition): a was_dph graph's
     * edge weights are renormalized per-step probabilities, not c.theta dot
     * products -- the linear contraction would silently compute the WRONG
     * Jacobian. Unreachable via Python routing (_effective_discrete sends
     * was_dph graphs to _dph), but this makes the C surface safe for direct
     * callers too: decline early (before the O(n^3) tape build) -> FD. */
    if (graph->was_dph) return -1;
    struct ptd_desc_reward_compute_parameterized *ptape =
        graph->use_dyn_ordering
            ? ptd_graph_ex_absorbation_time_comp_graph_parameterized_dyn(graph)
            : ptd_graph_ex_absorbation_time_comp_graph_parameterized(graph);
    if (ptape == NULL) return -1;
    struct ptd_desc_reward_compute_parameterized_off *off =
        ptd_pcg_convert_to_offset(ptape, graph, NULL, 0);
    if (off == NULL) { ptd_parameterized_reward_compute_graph_destroy(ptape); return -1; }
    int rc = ptd_b3_moments_core(graph, off, nr_moments, NULL, 0,
                                 PTD_B3_LINEAR, NULL, J_out);
    ptd_pcg_desc_off_destroy(off);
    ptd_parameterized_reward_compute_graph_destroy(ptape);
    return rc;
}

/* ===== B3 log-weight-mode extension: exact Jacobian d[m_0..m_{K-1}]/dtheta
 * for a CONTINUOUS, weight_mode='log' parameterized graph. Reuses
 * ptd_moments_grad_theta's stage-0 (forward moment chain + MPFR gate) /
 * stage-1 (reverse chain) / stage-2 (param-tape reverse) VERBATIM -- proven
 * agnostic to how edge->weight relates to theta (the was_dph batch already
 * established this; the reverse only ever reads the current edge->weight as
 * an opaque free variable). The only new math is the edge->theta
 * contraction: weight_mode='log' computes w_e = exp(sum_i log(c_e[i]*theta[i]))
 * over ALL i in 0..param_length-1 (every parameterized edge multiplies EVERY
 * theta component, unlike linear's sparse dot product), so by the product
 * rule dw_e/dtheta_j = w_e/theta_j for every j. The C layer's update_weights
 * already requires every c_e[i]*theta[i] > 0 strictly (raises otherwise), so
 * in any graph that reaches this function no theta[j] is ever exactly 0 --
 * the division is safe by construction, not something this function needs
 * to additionally guard. Verified build-free vs jax.jacobian of the same
 * log-space computation (experiments/dr_log_mode_edge_jacobian.py, ALL PASS,
 * incl. mixed/extreme-mixed scale) and against native central-difference
 * (experiments/dr_log_mode_moments_jac_gate.py).
 *
 * Declines (-1, FD fallback) for a was_dph graph: log+discretize() is NOT
 * guaranteed to fail at update_weights (confirmed by direct repro --
 * discretize() via a callable rate does not widen the coefficient layout
 * and can pass log's positivity check), so this exclusion is load-bearing,
 * not defensive redundancy. See b3-log-weight-mode-plan.md. NOTE:
 * is_discrete (native DPH, was_dph=False) has NO C-level ptd_graph field --
 * it is a Python-only attribute reaching C++ only via serialize()'s JSON for
 * the GraphBuilder/FFI forward path, never onto the raw C struct -- so it
 * cannot be checked here. The caller (pmf_and_moments_from_graph's Python
 * gate) MUST exclude is_discrete before ever calling this function; was_dph
 * is checked here only as an additional safety net for the subset of
 * is_discrete graphs that DO set it.
 *
 * theta/theta_len must match the values the caller most recently passed to
 * update_weights(theta, log=True). J_out is row-major nr_moments*param_length. */
int ptd_moments_grad_theta_log(struct ptd_graph *graph, int nr_moments,
        const double *theta, size_t theta_len, double *J_out) {
    if (!graph->parameterized || graph->param_length == 0) return -1;
    if (nr_moments < 1) return -1;
    if (graph->was_dph) return -1;
    if (theta_len != graph->param_length) return -1;
    struct ptd_desc_reward_compute_parameterized *ptape =
        graph->use_dyn_ordering
            ? ptd_graph_ex_absorbation_time_comp_graph_parameterized_dyn(graph)
            : ptd_graph_ex_absorbation_time_comp_graph_parameterized(graph);
    if (ptape == NULL) return -1;
    struct ptd_desc_reward_compute_parameterized_off *off =
        ptd_pcg_convert_to_offset(ptape, graph, NULL, 0);
    if (off == NULL) { ptd_parameterized_reward_compute_graph_destroy(ptape); return -1; }
    int rc = ptd_b3_moments_core(graph, off, nr_moments, theta, theta_len,
                                 PTD_B3_LOG, NULL, J_out);
    ptd_pcg_desc_off_destroy(off);
    ptd_parameterized_reward_compute_graph_destroy(ptape);
    return rc;
}
/* =============================================================================*/

/* ===== B3 discrete/was_dph extension: small combinatorial helpers for the
 * continuous->discrete moment correction. Mirror GraphBuilder's file-local
 * d_factorial/d_binomial/d_stirling2 (graph_builder.cpp:672-683) exactly;
 * duplicated here (rather than shared) because they live in different
 * translation units (C vs C++) with no common header for this leaf helper. */
static double ptd_dph_factorial(int n) {
    double r = 1.0;
    for (int i = 2; i <= n; i++) r *= (double)i;
    return r;
}
static double ptd_dph_binomial(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    return ptd_dph_factorial(n) / (ptd_dph_factorial(k) * ptd_dph_factorial(n - k));
}
static double ptd_dph_stirling2(int n, int k) {
    if (k == 0) return (n == 0) ? 1.0 : 0.0;
    if (k > n) return 0.0;
    if (k == n || k == 1) return 1.0;
    return (double)k * ptd_dph_stirling2(n - 1, k) + ptd_dph_stirling2(n - 1, k - 1);
}

/* Applies, IN PLACE, the fixed (theta-independent) linear map from continuous
 * power-moment space to discrete raw-moment space (mirrors
 * GraphBuilder::continuous_to_discrete_moments, graph_builder.cpp:694) to
 * EVERY COLUMN of J (row-major K x P: J[k*P+j] = d(continuous m_k)/dtheta_j
 * on input, d(discrete m_k)/dtheta_j on output). Valid because the map is
 * linear in the K axis, so applying it per column equals the chain rule
 * (verified vs jax.jacobian, experiments/dr_discrete_moment_correction.py). */
static void ptd_dph_correct_discrete_moment_grad(double *J, size_t K, size_t P) {
    if (K == 0 || P == 0) return;
    double *u = (double *) calloc(K + 1, sizeof(double));
    double *F = (double *) calloc(K + 1, sizeof(double));
    double *col = (double *) malloc(K * sizeof(double));
    for (size_t j = 0; j < P; ++j) {
        for (size_t k = 0; k < K; ++k) col[k] = J[k * P + j];
        for (size_t jj = 1; jj <= K; ++jj) {
            u[jj] = col[jj - 1] / ptd_dph_factorial((int) jj);
        }
        for (size_t r = 1; r <= K; ++r) {
            double s = 0.0;
            for (size_t i = 0; i < r; ++i) {
                double sign = (i % 2 == 0) ? 1.0 : -1.0;
                s += ptd_dph_binomial((int) (r - 1), (int) i) * sign * u[r - i];
            }
            F[r] = ptd_dph_factorial((int) r) * s;
        }
        for (size_t kk = 1; kk <= K; ++kk) {
            double s = 0.0;
            for (size_t r = 1; r <= kk; ++r) s += ptd_dph_stirling2((int) kk, (int) r) * F[r];
            J[(kk - 1) * P + j] = s;
        }
    }
    free(u); free(F); free(col);
}

/* ===== B3 discrete/was_dph extension: exact Jacobian d[discrete m]/dtheta.
 * Reuses ptd_moments_grad_theta's forward moment chain + reverse chain +
 * stage-2 param-tape reverse VERBATIM (unchanged: the stage-1 reverse only
 * ever reads the CURRENT edge->weight as an opaque free variable, so it is
 * agnostic to whether that value is a direct linear w_e = c_e.theta or a
 * was_dph-renormalised p_e = w_e/S_v -- verified in
 * b3-batch3-mpfr-and-discrete-derisk.md). The only new math: (1) the final
 * edge->theta contraction branches on graph->was_dph between the plain
 * linear rule (same as ptd_moments_grad_theta) and the renorm quotient rule
 * dp_e/dtheta_j = (c_e^j - p_e*sum_e' c_e'^j) / S_v (sibling coupling,
 * de-risked vs jax.jacobian in experiments/dr_dph_renorm_jacobian.py); (2)
 * the discrete moment correction is applied to every output column
 * afterwards (ptd_dph_correct_discrete_moment_grad), since the primal for a
 * discrete graph is continuous_to_discrete_moments(continuous moments)
 * regardless of was_dph. Declines (-1) on: MPFR-conditioned tapes (same gate
 * as the continuous path); a was_dph vertex whose out-edges mix constant
 * (coefficients_length==0) and parameterized edges -- S_v would need the
 * constant edges' pre-renormalisation weight, which is not recoverable from
 * the current (already divided) edge->weight and does not arise for any
 * graph produced by Graph.discretize() (its only constant edges are lone
 * aux back-edges, never mixed with a parameterized sibling). */
int ptd_moments_grad_theta_dph(struct ptd_graph *graph, int nr_moments,
        const double *theta, size_t theta_len, double *J_out) {
    if (!graph->parameterized || graph->param_length == 0) return -1;
    if (nr_moments < 1) return -1;
    size_t P = graph->param_length, K = (size_t) nr_moments;
    if (theta_len != P) return -1;

    /* Precompute per-vertex S_v = sum_e' (c_e' . theta) and SigmaCv[v][j] =
     * sum_e' c_e'^j, over PARAMETERIZED out-edges only (constant out-edges,
     * e.g. discretize()'s aux back-edges, are never tape inputs and so never
     * need a theta-gradient contribution). Only needed when graph->was_dph;
     * for a native DPH (was_dph=False) edge->weight IS c_e.theta directly, so
     * the plain linear contraction applies, exactly like the continuous
     * path. Declines if any was_dph vertex mixes constant + parameterized
     * out-edges (S_v would need the constant edge's raw weight, which is not
     * recoverable post-renormalisation). */
    double *Sv = NULL, *SigmaCv = NULL;
    if (graph->was_dph) {
        Sv = (double *) calloc(graph->vertices_length, sizeof(double));
        SigmaCv = (double *) calloc(graph->vertices_length * P, sizeof(double));
        int mixed = 0;
        for (size_t v = 0; v < graph->vertices_length && !mixed; ++v) {
            struct ptd_vertex *vertex = graph->vertices[v];
            if (vertex == graph->starting_vertex) continue;
            int has_param = 0, has_const = 0;
            for (size_t e = 0; e < vertex->edges_length; ++e) {
                struct ptd_edge *edge = vertex->edges[e];
                if (edge->coefficients_length == 0) { has_const = 1; continue; }
                has_param = 1;
                double w = 0.0;
                for (size_t j = 0; j < P; ++j) w += edge->coefficients[j] * theta[j];
                Sv[v] += w;
                for (size_t j = 0; j < P; ++j) SigmaCv[v * P + j] += edge->coefficients[j];
            }
            if (has_param && has_const) mixed = 1;
        }
        if (mixed) { free(Sv); free(SigmaCv); return -1; }
    }

    struct ptd_desc_reward_compute_parameterized *ptape =
        graph->use_dyn_ordering
            ? ptd_graph_ex_absorbation_time_comp_graph_parameterized_dyn(graph)
            : ptd_graph_ex_absorbation_time_comp_graph_parameterized(graph);
    if (ptape == NULL) { free(Sv); free(SigmaCv); return -1; }
    struct ptd_desc_reward_compute_parameterized_off *off =
        ptd_pcg_convert_to_offset(ptape, graph, NULL, 0);
    if (off == NULL) {
        free(Sv); free(SigmaCv);
        ptd_parameterized_reward_compute_graph_destroy(ptape);
        return -1;
    }
    struct ptd_b3_dph_ctx ctx = { Sv, SigmaCv };
    int rc = ptd_b3_moments_core(graph, off, nr_moments, theta, theta_len,
                                 PTD_B3_DPH, &ctx, J_out);
    /* POST-pass: the theta-independent continuous->discrete correction, then
     * a SECOND isfinite sweep (the correction itself can produce non-finite
     * output; the core's sweep ran before it). */
    if (rc == 0) {
        ptd_dph_correct_discrete_moment_grad(J_out, K, P);
        for (size_t x = 0; x < K*P; ++x) {
            if (!isfinite(J_out[x])) { rc = -1; break; }
        }
    }
    /* ctx is wrapper-owned: freed unconditionally, success or decline; the
     * core never frees it. */
    free(Sv); free(SigmaCv);
    ptd_pcg_desc_off_destroy(off);
    ptd_parameterized_reward_compute_graph_destroy(ptape);
    return rc;
}

/* ===== B3 joint-index extension: forward-mode theta-adjoint for the SOJOURN
 * vector (ptd_expected_sojourn_time_subset). See b3-joint-index-plan.md.
 *
 * Every other B3 gradient function above differentiates a FEW-output
 * quantity (nr_moments, small) w.r.t. MANY-ish inputs (theta, param_length)
 * via REVERSE-mode: one backward pass gives every theta component at once.
 * Sojourn is the opposite shape: MANY outputs (sojourn(v) for up to n target
 * vertices -- confirmed in the hundreds of thousands for real joint-
 * probability graphs, tests/pytest/test_sojourn_subset_adjoint.py) but FEW
 * inputs (param_length, typically 1-10). Reverse-mode here would cost
 * O(k*nc) (one pass per requested vertex) -- exactly the O(n*k) blowup
 * ptd_expected_sojourn_time_subset's own adjoint exists to avoid for the
 * PRIMAL. So this function uses FORWARD-mode instead: one pass PER THETA
 * COMPONENT (P total), each giving the FULL (n,) sojourn-gradient vector in
 * one O(nc) pass -- O(P*nc) total, independent of k/n beyond the O(nc) cost
 * the primal already pays. This is the first B3 gradient function to use
 * forward-mode in production; the param-tape tangent formulas below are not
 * new (they are ptd_dbg_run_tape's already-shipped-as-a-validator logic,
 * ptd_dbg_run_tape:10362, promoted to an unguarded production path here).
 * De-risked build-free (experiments/dr_sojourn_fwdmode_adjoint.py): primal
 * vs JAX (259/259), forward-mode vs jax.jacobian (243/243), forward-mode vs
 * an independent reverse-mode-per-target-vertex cross-check (79/79), plus a
 * crafted diagonal-weight-exactly-1 case validating the guard asymmetry
 * below (all machine-precision, see the plan for the full D1 review).
 *
 * Stage-0 (param-tape forward: build (na,nb,nm) numeric commands + operand
 * snapshots s0/s1, MPFR gate) is IDENTICAL to ptd_moments_grad_theta's own
 * stage-0 -- reused verbatim, not re-derived.
 *
 * Unlike ptd_moments_grad_theta (which rebuilds the whole O(n^3)
 * parameterized tape from scratch on every call, tolerable there because
 * moment-graphs are typically modest-sized), this function REUSES the
 * graph-level tape cache via ptd_precompute_reward_compute_graph -- the
 * same entrypoint ptd_expected_sojourn_time_subset and
 * ptd_expected_waiting_time already call. Deliberate: this function's
 * target graphs ARE the large joint-probability case (n up to ~7x10^5),
 * where an uncached O(n^3) rebuild per gradient call would regress badly
 * against the current FD path (which reuses a per-thread-cached graph via
 * the FFI, graph_builder_ffi.cpp ComputeSojournTimesFfiImpl). See the plan's
 * "Adversarial review findings" #5.
 *
 * Guards in the sojourn recurrence are DELIBERATELY ASYMMETRIC between
 * primal and tangent: a diagonal (from==to) numeric command stores
 * (multiplier - 1), so a diagonal edge at weight exactly 1.0 stores m==0
 * while its theta-derivative can be nonzero. The primal keeps
 * ptd_expected_sojourn_time_subset's existing guards (skip on m==0 OR
 * isinf-with-zero-operand); the tangent must skip ONLY the isinf case, never
 * m==0, or it would silently drop that gradient term. Production's own
 * reverse-mode moments gradient (this function's stage-2 walk, just above)
 * already uses the identical asymmetry (dm[c] accumulation unconditional,
 * only the adj[]-continuation guarded on m!=0) -- an established idiom here,
 * not a novel invention.
 *
 * Declines (-1, FD fallback) for: non-parameterized graphs; was_dph graphs
 * (discretize()'s renormalization needs a different quotient-rule
 * contraction, deferred -- see the plan's scope section; native DPH,
 * is_discrete=True/was_dph=False, needs NO special handling and is NOT
 * excluded here, confirmed both ComputeSojournTimesFfiImpl and
 * ptd_expected_sojourn_time_subset have zero is_discrete branching); any
 * tape input whose spec is not a plain internal edge weight (SCC/external
 * inputs are out of scope); an MPFR-conditioned tape (mirrors the
 * continuous moments gate).
 *
 * indices/k select which of the n sojourn-gradient rows to gather -- pass
 * the union of every index set the caller needs (see the plan's "Wiring"
 * section for why one call over a union beats two calls over the parts).
 * J_out must hold k*graph->param_length doubles (row-major: row r =
 * d(sojourn(indices[r]))/dtheta). Returns 0 on success; -1 for FD fallback. */
static int ptd_b3_sojourn_grad_core(struct ptd_graph *graph,
        const size_t *indices, size_t k, double *J_out,
        int skip_condition_gate) {
    if (!graph->parameterized || graph->param_length == 0) return -1;
    if (graph->was_dph) return -1;
    size_t P = graph->param_length;
    if (k == 0) return 0;

    if (ptd_precompute_reward_compute_graph(graph)) return -1;

    struct ptd_desc_reward_compute_parameterized_off *off;
    int owns_off = 0;
    if (graph->parameterized_reward_compute_graph_off != NULL) {
        off = graph->parameterized_reward_compute_graph_off;
        /* An mmap-loaded rev-3 descriptor (Stage-A2 on-disk cache,
         * ptd_load_pcg_rev3_mmap) never carries input_specs -- it is set
         * to NULL unconditionally (phasic.c:3816, "a loaded descriptor is
         * never re-saved"), since input_specs exists only for save/re-bind,
         * not for the numeric replay the mmap path is built for. This
         * function's seeding loop below dereferences off->input_specs[kk]
         * for every one of the off->n_inputs tape inputs -- decline instead
         * of a NULL-pointer segfault (found via adversarial review of the
         * D6 plan; the guard is needed regardless of D6, since it is
         * reachable today whenever PHASIC_REWARD_COMPUTE_CACHE=1 is set and
         * the on-disk cache is warm). */
        if (off->n_inputs > 0 && off->input_specs == NULL) return -1;
    } else if (graph->parameterized_reward_compute_graph != NULL) {
        off = ptd_pcg_convert_to_offset(graph->parameterized_reward_compute_graph, graph, NULL, 0);
        if (off == NULL) return -1;
        owns_off = 1;
    } else {
        return -1;
    }

    size_t n = graph->vertices_length, ni = off->n_inputs, md = off->mem_doubles, L = off->length;

    for (size_t r = 0; r < k; ++r) {
        if (indices[r] >= n) { if (owns_off) ptd_pcg_desc_off_destroy(off); return -1; }
    }

    /* Size guard (found via adversarial review of the implemented fix):
     * unlike the FD path (which reads the already-numeric, cached
     * graph->reward_compute_graph), this function allocates ~40 bytes per
     * tape command (s0/s1/na/nb/nm below) PLUS -- when the Stage-A2
     * on-disk offset cache is not populated (its default state) -- a full
     * ptd_pcg_convert_to_offset() copy of the tape, allocated and freed on
     * EVERY call. On the large joint-probability graphs this function
     * targets (n up to ~7e5, see b3-joint-index-plan.md), L can be large
     * enough that this becomes a real, repeated memory spike rather than a
     * one-time cost. Decline (not segfault/OOM) rather than risk it; the
     * threshold is a conservative, round-number safety net (~2GB across
     * the L-sized arrays), not a tuned value -- FD stays correct either
     * way. */
    if (L > 50000000) {
        if (owns_off) ptd_pcg_desc_off_destroy(off);
        return -1;
    }

    /* Validate every tape input is a plain internal edge weight and cache its
     * edge pointer -- done ONCE up front, not per theta component. A
     * coefficients_length==0 edge (e.g. the aux back-edges from
     * add_aux_vertex/add_aux_vertex_constant used by Graph.discretize() and
     * Graph.joint_stop_prob_graph() -- this function's target workload, not
     * a corner case) is left in the array and seeded to idot=0 below WITHOUT
     * dereferencing its NULL coefficients pointer (the same NULL-pointer
     * class already found and fixed in the discrete/was_dph batch, see
     * ptd_moments_grad_theta's contraction step above). */
    int ok = 1;
    struct ptd_edge **edge_for_input =
        (struct ptd_edge **) malloc((ni ? ni : 1) * sizeof(struct ptd_edge *));
    if (edge_for_input == NULL) {
        if (owns_off) ptd_pcg_desc_off_destroy(off);
        return -1;
    }
    for (size_t kk = 0; kk < ni; ++kk) {
        struct ptd_pcg_input_spec sp = off->input_specs[kk];
        if (sp.kind != PTD_PCG_PTR_EDGE || sp.byte != 0
                || sp.v >= graph->vertices_length
                || sp.e >= graph->vertices[sp.v]->edges_length) { ok = 0; break; }
        edge_for_input[kk] = graph->vertices[sp.v]->edges[sp.e];
    }
    if (!ok) {
        free(edge_for_input);
        if (owns_off) ptd_pcg_desc_off_destroy(off);
        return -1;
    }

    double *mem = (double*)malloc((md?md:1)*sizeof(double));
    double *inv = (double*)malloc((ni?ni:1)*sizeof(double));
    double *s0=(double*)malloc((L?L:1)*sizeof(double));
    double *s1=(double*)malloc((L?L:1)*sizeof(double));
    uint64_t *na=(uint64_t*)malloc((L?L:1)*sizeof(uint64_t));
    uint64_t *nb=(uint64_t*)malloc((L?L:1)*sizeof(uint64_t));
    double *nm=(double*)malloc((L?L:1)*sizeof(double));
    if (mem == NULL || inv == NULL || s0 == NULL || s1 == NULL
            || na == NULL || nb == NULL || nm == NULL) {
        free(mem); free(inv); free(s0); free(s1); free(na); free(nb); free(nm);
        free(edge_for_input);
        if (owns_off) ptd_pcg_desc_off_destroy(off);
        return -1;
    }
    memcpy(mem, off->mem_base, md*sizeof(double));
    for (size_t kk=0; kk<ni; ++kk) inv[kk] = *off->inputs[kk];
    size_t nc=0;
#define RV(op) ((op).kind==PTD_PCG_OP_MEM ? &mem[(op).mem_offset] : \
               ((op).kind==PTD_PCG_OP_INPUT ? &inv[(op).input_idx] : (double*)NULL))
    /* Stage-0: param-tape forward -- IDENTICAL to ptd_moments_grad_theta's. */
    for (size_t i=0;i<L;++i) {
        struct ptd_pcg_command_off c = off->commands[i];
        double *rf=RV(c.fromT), *rt=RV(c.toT), *rm=RV(c.multiplierptr);
        switch (c.type) {
            case 0: { double mcv=(c.from==c.to)?(*rm-1.0):*rm; na[nc]=c.from; nb[nc]=c.to; nm[nc]=mcv; nc++; } break;
            case 1: *rf += (*rt)*c.multiplier; break;
            case 3: s0[i]=*rt; s1[i]=*rm; *rf += (*rt)*(*rm); break;
            case 2: s0[i]=*rf; *rf = 1.0/(*rf); break;
            case 4: *rf = 1.0 - *rf; break;
            case 5: s0[i]=*rf; s1[i]=*rt; *rf = (*rf)/(*rt); break;
            case 6: *rf = 0.0; break;
            default: break;
        }
    }

    /* Conditioning gate. NOTE (16b item 2, corrected here in Batch H):
     * unlike ptd_moments_grad_theta's gate -- which protects a genuine
     * primal/gradient MPFR-representation mismatch -- this path's primal
     * (ptd_expected_sojourn_time_subset) has NO MPFR path at all, so the
     * gate is a pure conservatism knob here, not a correctness necessity.
     * Batch H's de-risk measured it declining 100% of realistic
     * coalescent-scale calls (theta ~1e-4, handoff entries down to
     * ~5e-148) while the gated answers matched an fp64 oracle to ~1e-13
     * (experiments/dr_batchH_oracle.py). skip_condition_gate=1 (the
     * _nogate entry below, user-sanctioned 2026-08-13) bypasses ONLY
     * this check; the final isfinite sweep stays the live defense. */
    if (!skip_condition_gate && ptd_dbg_tape_needs_mpfr(nm, nc)) ok = 0;

    double *mem_dot = ok ? (double*)malloc((md?md:1)*sizeof(double)) : NULL;
    double *inv_dot = ok ? (double*)malloc((ni?ni:1)*sizeof(double)) : NULL;
    double *mdot    = ok ? (double*)malloc((nc?nc:1)*sizeof(double)) : NULL;
    double *y       = ok ? (double*)malloc(n*sizeof(double)) : NULL;
    double *y_dot   = ok ? (double*)malloc(n*sizeof(double)) : NULL;
    if (ok && (mem_dot == NULL || inv_dot == NULL || mdot == NULL || y == NULL || y_dot == NULL)) {
        ok = 0;
    }
#define RD(op) ((op).kind==PTD_PCG_OP_MEM ? &mem_dot[(op).mem_offset] : \
               ((op).kind==PTD_PCG_OP_INPUT ? &inv_dot[(op).input_idx] : (double*)NULL))
    for (size_t j=0; ok && j<P; ++j) {
        for (size_t i=0;i<md;++i) mem_dot[i]=0.0;
        for (size_t kk=0; kk<ni; ++kk) {
            struct ptd_edge *e = edge_for_input[kk];
            inv_dot[kk] = (e->coefficients_length == 0) ? 0.0 : e->coefficients[j];
        }
        /* Stage-1: param-tape tangent, seeded via the coefficient column --
         * identical formulas to ptd_dbg_run_tape (P/PP/INV/OM/DIV/ZERO),
         * reusing stage-0's s0/s1 snapshots rather than re-walking mem[]. */
        size_t ncr = 0;
        for (size_t i=0;i<L;++i) {
            struct ptd_pcg_command_off c = off->commands[i];
            double *rfd=RD(c.fromT), *rtd=RD(c.toT), *rmd=RD(c.multiplierptr);
            switch (c.type) {
                case 0: mdot[ncr] = *rmd; ncr++; break;
                case 1: *rfd += (*rtd)*c.multiplier; break;
                case 3: *rfd += (*rtd)*s1[i] + s0[i]*(*rmd); break;
                case 2: *rfd = -(*rfd)/(s0[i]*s0[i]); break;
                case 4: *rfd = -(*rfd); break;
                case 5: *rfd = (*rfd)/s1[i] - s0[i]*(*rtd)/(s1[i]*s1[i]); break;
                case 6: *rfd = 0.0; break;
                default: break;
            }
        }

        /* Sojourn recurrence: seed y[target=0]=1 (mirrors
         * ptd_expected_sojourn_time_subset), walk REVERSED with roles
         * swapped, primal + tangent interleaved (forward-mode needs the
         * LIVE evolving y[], not a pre-recorded snapshot). */
        for (size_t v=0; v<n; ++v) { y[v]=0.0; y_dot[v]=0.0; }
        y[0] = 1.0;
        for (size_t c=nc; c-- > 0; ) {
            size_t a=na[c], b=nb[c]; double m=nm[c], md_=mdot[c];
            /* TANGENT: NOT the primal's guards verbatim. Dropping the m==0
             * skip (required -- see the m==0/mdot!=0 diagonal case in the
             * function comment) would also drop the 0*inf=0 limit
             * convention on EACH term individually if y[a]/y_dot[a] is
             * +-inf at a trap/deficit-sink vertex (real on production
             * joint-prob graphs, see ptd_expected_sojourn_time_subset's own
             * comment) -- so each of the two summands gets its OWN 0*inf=0
             * guard instead of one guard over the whole update. This is a
             * no-op whenever y[a]/y_dot[a] are finite (found via adversarial
             * review of the implemented fix). */
            if (!(isinf(m) && y[a]==0.0)) {
                double t = 0.0;
                if (m != 0.0) t += y_dot[a]*m;
                if (md_ != 0.0) t += y[a]*md_;
                y_dot[b] += t;
            }
            /* PRIMAL: unchanged from ptd_expected_sojourn_time_subset. */
            if (m == 0.0) continue;
            if (isinf(m) && y[a]==0.0) continue;
            y[b] += y[a]*m;
        }
        for (size_t r=0;r<k;++r) J_out[r*P+j] = y_dot[indices[r]];
    }

    if (ok) for (size_t x=0; x<k*P; ++x) if (!isfinite(J_out[x])) { ok=0; break; }

    free(edge_for_input); free(mem); free(inv); free(s0); free(s1); free(na); free(nb); free(nm);
    if (mem_dot) free(mem_dot);
    if (inv_dot) free(inv_dot);
    if (mdot) free(mdot);
    if (y) free(y);
    if (y_dot) free(y_dot);
    if (owns_off) ptd_pcg_desc_off_destroy(off);
#undef RV
#undef RD
    return ok ? 0 : -1;
}

/* Public entry, default semantics (conditioning gate ON) -- behavior
 * identical to the pre-Batch-H function (verified byte-identical by
 * experiments/dr_batchH_i1_gate.py micro-gate (a)). */
int ptd_sojourn_grad_theta_subset(struct ptd_graph *graph,
        const size_t *indices, size_t k, double *J_out) {
    return ptd_b3_sojourn_grad_core(graph, indices, k, J_out, 0);
}

/* Public ADDITIVE entry (Batch H, user decision 2026-08-13): identical
 * to ptd_sojourn_grad_theta_subset EXCEPT the MPFR conditioning gate is
 * skipped (see the 16b-item-2 note at the gate line in the core). Every
 * other decline stays live: was_dph, the L size guard, allocation
 * failures, tape-input scope, and the final per-row isfinite sweep. */
int ptd_sojourn_grad_theta_subset_nogate(struct ptd_graph *graph,
        const size_t *indices, size_t k, double *J_out) {
    return ptd_b3_sojourn_grad_core(graph, indices, k, J_out, 1);
}
/* =============================================================================*/

double *ptd_expected_sojourn_time(struct ptd_graph *graph) {
    // Precompute elimination trace if not already done
    if (ptd_precompute_reward_compute_graph(graph)) {
        PTD_LOG_ERROR("Failed to precompute reward compute graph");
        return NULL;
    }

    size_t n = graph->vertices_length;
    struct ptd_desc_reward_compute *compute = graph->reward_compute_graph;

    // Allocate results matrix: results[vertex_idx][reward_idx]
    // Layout: results[v][r] = accumulated reward at vertex v for reward vector r
    double **results = (double **) malloc(n * sizeof(double *));
    if (results == NULL) {
        PTD_LOG_ERROR("Failed to allocate results matrix");
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        results[i] = (double *) calloc(n, sizeof(double));
        if (results[i] == NULL) {
            PTD_LOG_ERROR("Failed to allocate results row %zu", i);
            // Free previously allocated rows
            for (size_t j = 0; j < i; j++) {
                free(results[j]);
            }
            free(results);
            return NULL;
        }
    }

    // Initialize with identity matrix: reward vector r has value 1 at vertex r
    for (size_t v = 0; v < n; v++) {
        results[v][v] = 1.0;
    }

    // Allocate Kahan compensation arrays for numerical stability
    // Each row needs its own compensation array
    struct kahan_sum **kahan_states = (struct kahan_sum **) malloc(n * sizeof(struct kahan_sum *));
    if (kahan_states == NULL) {
        PTD_LOG_ERROR("Failed to allocate Kahan compensation arrays");
        for (size_t i = 0; i < n; i++) {
            free(results[i]);
        }
        free(results);
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        kahan_states[i] = (struct kahan_sum *) calloc(n, sizeof(struct kahan_sum));
        if (kahan_states[i] == NULL) {
            PTD_LOG_ERROR("Failed to allocate Kahan compensation row %zu", i);
            for (size_t j = 0; j < i; j++) {
                free(kahan_states[j]);
            }
            free(kahan_states);
            for (size_t j = 0; j < n; j++) {
                free(results[j]);
            }
            free(results);
            return NULL;
        }
        // Initialize Kahan states with current results as initial sums
        for (size_t r = 0; r < n; r++) {
            kahan_states[i][r].sum = results[i][r];
            kahan_states[i][r].compensation = 0.0;
        }
    }

    // Apply all elimination trace commands to all reward vectors.
    // Command: results[from][r] += results[to][r] * multiplier for all r,
    // with Kahan summation for numerical stability.
    //
    // The n reward-vector columns are independent: a command only reads/writes
    // column r within itself, so the whole command trace for a given column is
    // self-contained. We parallelise over columns. One `omp parallel` region
    // wraps the command loop; `omp for schedule(static) nowait` hands each
    // thread a fixed contiguous column slice for EVERY command (static => the
    // same slice each time), so a thread runs the entire trace for its columns
    // with no cross-thread sharing and no per-command barrier (the parallel
    // region's closing barrier syncs before extraction). Per-column arithmetic
    // and order are unchanged, so the result is identical for any thread count.
    // Gated on parallel_elimination=True (PHASIC_HIERAR_ELIMINATION) and a size
    // threshold; otherwise the region runs single-threaded (== prior behaviour).
    const char *hierar_env = getenv("PHASIC_HIERAR_ELIMINATION");
    const bool use_par = (hierar_env != NULL && hierar_env[0] == '1'
                          && hierar_env[1] == '\0' && n >= 512);
    (void) use_par;  // referenced only by the OpenMP if() clause
    // MSVC implements OpenMP 2.0, whose canonical loop form requires the
    // omp-for index to be a signed integer declared *outside* the loop with
    // a plain-assignment init (`r = 0`) — a declaration in the init
    // (`int r = 0`) is rejected with C3015. Mirror the proven pattern in
    // scc_compose.c: declare `r` inside the parallel region (so it is
    // private per thread) and cast the size_t column bound to int.
    #pragma omp parallel if(use_par)
    {
    int r;
    for (size_t cmd_idx = 0; cmd_idx < compute->length; cmd_idx++) {
        struct ptd_reward_increase cmd = compute->commands[cmd_idx];

        // Handle 0 × ∞ = 0 (limit interpretation)
        // Skip when multiplier is zero to avoid NaN from 0.0 × inf
        if (cmd.multiplier == 0.0) {
            continue;
        }

        double *from_row = results[cmd.from];
        double *to_row = results[cmd.to];
        struct kahan_sum *from_kahan = kahan_states[cmd.from];
        double multiplier = cmd.multiplier;

        // Check if multiplier is infinite
        bool mult_is_inf = isinf(multiplier);

        // Inner loop: each thread takes a fixed static slice of the columns.
        #pragma omp for schedule(static) nowait
        for (r = 0; r < (int) n; r++) {
            // Handle inf × 0 = 0 (limit interpretation)
            if (mult_is_inf && to_row[r] == 0.0) {
                continue;
            }

            // Kahan compensated addition
            double increment = to_row[r] * multiplier;
            kahan_add(&from_kahan[r], increment);
            from_row[r] = kahan_result(&from_kahan[r]);
        }

        // Debug: check for NaN
        #ifdef DEBUG
        #pragma omp for schedule(static) nowait
        for (r = 0; r < (int) n; r++) {
            if (isnan(from_row[r])) {
                PTD_LOG_WARNING("results[%zu][%zu] became nan at command %zu",
                    cmd.from, (size_t) r, cmd_idx);
            }
        }
        #endif
    }
    }

    // Free Kahan compensation arrays
    for (size_t i = 0; i < n; i++) {
        free(kahan_states[i]);
    }
    free(kahan_states);

    // Extract sojourn times: results[starting_vertex][r] for each reward vector r
    // Starting vertex is at index 0
    double *sojourn_times = (double *) malloc(n * sizeof(double));
    if (sojourn_times == NULL) {
        PTD_LOG_ERROR("Failed to allocate sojourn times array");
        for (size_t i = 0; i < n; i++) {
            free(results[i]);
        }
        free(results);
        return NULL;
    }

    for (size_t r = 0; r < n; r++) {
        sojourn_times[r] = results[0][r];  // Starting vertex index = 0
    }

    // Free intermediate results matrix
    for (size_t i = 0; i < n; i++) {
        free(results[i]);
    }
    free(results);

    // PTD_LOG_DEBUG("Computed sojourn times for %zu states", n);
    return sojourn_times;
}

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

struct ptd_sample_path *ptd_random_sample_path(struct ptd_graph *graph) {
    size_t capacity = 16;
    /* Safety limit: max steps = 100 * number of vertices */
    size_t max_steps = 100 * graph->vertices_length;
    struct ptd_sample_path *path = (struct ptd_sample_path *) malloc(sizeof(*path));
    path->vertex_indices = (size_t *) malloc(capacity * sizeof(size_t));
    path->entry_times = (double *) malloc(capacity * sizeof(double));
    path->length = 0;

    double cumulative_time = 0.0;
    struct ptd_vertex *vertex = graph->starting_vertex;

    /* Record starting vertex */
    path->vertex_indices[path->length] = vertex->index;
    path->entry_times[path->length] = 0.0;
    path->length++;

    while (vertex->edges_length != 0 && path->length < max_steps) {
        /* Sample waiting time */
        long double draw_wait = (long double) rand() / (long double) RAND_MAX;

        double rate = 0;
        for (size_t i = 0; i < vertex->edges_length; ++i) {
            rate += vertex->edges[i]->weight;
        }

        long double waiting_time = -logl(draw_wait + 0.0000001) / rate;

        if (vertex == graph->starting_vertex) {
            waiting_time = 0;
        }

        cumulative_time += (double) waiting_time;

        /* Select next vertex */
        long double draw_direction = (long double) rand() / (long double) RAND_MAX;
        long double weight_sum = 0;
        size_t edge_index = 0;

        for (size_t i = 0; i < vertex->edges_length; ++i) {
            weight_sum += vertex->edges[i]->weight;
            if (weight_sum / rate >= draw_direction) {
                edge_index = i;
                break;
            }
        }

        vertex = vertex->edges[edge_index]->to;

        /* Grow arrays if needed */
        if (path->length >= capacity) {
            capacity *= 2;
            path->vertex_indices = (size_t *) realloc(path->vertex_indices, capacity * sizeof(size_t));
            path->entry_times = (double *) realloc(path->entry_times, capacity * sizeof(double));
        }

        /* Record this vertex */
        path->vertex_indices[path->length] = vertex->index;
        path->entry_times[path->length] = cumulative_time;
        path->length++;
    }

    return path;
}

double *ptd_backward_probabilities(struct ptd_graph *graph,
                                   size_t *target_vertices,
                                   size_t n_targets) {
    size_t n = graph->vertices_length;
    double *h = (double *) calloc(n, sizeof(double));

    /* Mark target vertices */
    for (size_t i = 0; i < n_targets; ++i) {
        if (target_vertices[i] < n) {
            h[target_vertices[i]] = 1.0;
        }
    }

    /* Backward pass: process vertices in reverse order.
     * For each non-terminal vertex v:
     *   h[v] = sum_i (weight_i / total_weight) * h[edge_i.to]
     * where weight_i / total_weight is the transition probability.
     * Target vertices keep h=1 regardless of their edges.
     */
    for (size_t vi = n; vi > 0; --vi) {
        size_t idx = vi - 1;
        struct ptd_vertex *v = graph->vertices[idx];

        if (v->edges_length == 0) {
            continue;  /* Absorbing vertex: keep h as initialized */
        }

        /* Skip target vertices — they keep h=1 even if they have edges */
        if (h[idx] == 1.0) {
            int is_target = 0;
            for (size_t i = 0; i < n_targets; ++i) {
                if (target_vertices[i] == idx) {
                    is_target = 1;
                    break;
                }
            }
            if (is_target) continue;
        }

        /* Compute total exit rate for normalization */
        double total_weight = 0;
        for (size_t i = 0; i < v->edges_length; ++i) {
            total_weight += v->edges[i]->weight;
        }

        if (total_weight <= 0) {
            h[idx] = 0;
            continue;
        }

        double prob = 0;
        for (size_t i = 0; i < v->edges_length; ++i) {
            double trans_prob = v->edges[i]->weight / total_weight;
            prob += trans_prob * h[v->edges[i]->to->index];
        }
        h[idx] = prob;
    }

    return h;
}

struct ptd_sample_path *ptd_random_sample_path_conditioned(
    struct ptd_graph *graph,
    double *backward_probs) {

    size_t capacity = 16;
    struct ptd_sample_path *path = (struct ptd_sample_path *) malloc(sizeof(*path));
    path->vertex_indices = (size_t *) malloc(capacity * sizeof(size_t));
    path->entry_times = (double *) malloc(capacity * sizeof(double));
    path->length = 0;

    double cumulative_time = 0.0;
    struct ptd_vertex *vertex = graph->starting_vertex;

    /* Record starting vertex */
    path->vertex_indices[path->length] = vertex->index;
    path->entry_times[path->length] = 0.0;
    path->length++;

    while (vertex->edges_length != 0 && backward_probs[vertex->index] < 1.0) {
        /* Stop if we've reached a target vertex (h=1) even if it has edges */

        /* Sample waiting time (same as ptd_random_sample_path) */
        long double draw_wait = (long double) rand() / (long double) RAND_MAX;

        double rate = 0;
        for (size_t i = 0; i < vertex->edges_length; ++i) {
            rate += vertex->edges[i]->weight;
        }

        long double waiting_time = -logl(draw_wait + 0.0000001) / rate;

        if (vertex == graph->starting_vertex) {
            waiting_time = 0;
        }

        cumulative_time += (double) waiting_time;

        /* Guided selection: weight by edge_weight * backward_prob[target] */
        double guided_total = 0;
        for (size_t i = 0; i < vertex->edges_length; ++i) {
            guided_total += vertex->edges[i]->weight *
                           backward_probs[vertex->edges[i]->to->index];
        }

        if (guided_total <= 0) {
            break;  /* No reachable target from here — stop */
        }

        long double draw_direction = (long double) rand() / (long double) RAND_MAX;
        long double weight_sum = 0;
        size_t edge_index = 0;

        for (size_t i = 0; i < vertex->edges_length; ++i) {
            double guided_weight = vertex->edges[i]->weight *
                                  backward_probs[vertex->edges[i]->to->index];
            weight_sum += guided_weight;

            if (weight_sum / guided_total >= draw_direction) {
                edge_index = i;
                break;
            }
        }

        vertex = vertex->edges[edge_index]->to;

        /* Grow arrays if needed */
        if (path->length >= capacity) {
            capacity *= 2;
            path->vertex_indices = (size_t *) realloc(path->vertex_indices, capacity * sizeof(size_t));
            path->entry_times = (double *) realloc(path->entry_times, capacity * sizeof(double));
        }

        /* Record this vertex */
        path->vertex_indices[path->length] = vertex->index;
        path->entry_times[path->length] = cumulative_time;
        path->length++;
    }

    return path;
}

void ptd_sample_path_destroy(struct ptd_sample_path *path) {
    if (path != NULL) {
        free(path->vertex_indices);
        free(path->entry_times);
        free(path);
    }
}

size_t ptd_random_sample_path_conditioned_fixed(
    struct ptd_graph *graph,
    double *backward_probs,
    size_t max_length,
    unsigned int seed,
    int *out_vertex_indices,
    double *out_entry_times) {

    /* Thread-safe RNG using rand_r */
    unsigned int rng_state = seed;

    size_t length = 0;
    double cumulative_time = 0.0;
    struct ptd_vertex *vertex = graph->starting_vertex;

    /* Record starting vertex */
    out_vertex_indices[length] = (int) vertex->index;
    out_entry_times[length] = 0.0;
    length++;

    while (vertex->edges_length != 0 && backward_probs[vertex->index] < 1.0) {
        if (length >= max_length) break;

        /* Sample waiting time */
        long double draw_wait = (long double) rand_r(&rng_state) / (long double) RAND_MAX;

        double rate = 0;
        for (size_t i = 0; i < vertex->edges_length; ++i) {
            rate += vertex->edges[i]->weight;
        }

        long double waiting_time = -logl(draw_wait + 0.0000001) / rate;

        if (vertex == graph->starting_vertex) {
            waiting_time = 0;
        }

        cumulative_time += (double) waiting_time;

        /* Guided selection */
        double guided_total = 0;
        for (size_t i = 0; i < vertex->edges_length; ++i) {
            guided_total += vertex->edges[i]->weight *
                           backward_probs[vertex->edges[i]->to->index];
        }

        if (guided_total <= 0) break;

        long double draw_direction = (long double) rand_r(&rng_state) / (long double) RAND_MAX;
        long double weight_sum = 0;
        size_t edge_index = 0;

        for (size_t i = 0; i < vertex->edges_length; ++i) {
            double guided_weight = vertex->edges[i]->weight *
                                  backward_probs[vertex->edges[i]->to->index];
            weight_sum += guided_weight;
            if (weight_sum / guided_total >= draw_direction) {
                edge_index = i;
                break;
            }
        }

        vertex = vertex->edges[edge_index]->to;

        out_vertex_indices[length] = (int) vertex->index;
        out_entry_times[length] = cumulative_time;
        length++;
    }

    /* Pad remaining with sentinels */
    for (size_t i = length; i < max_length; ++i) {
        out_vertex_indices[i] = -1;
        out_entry_times[i] = 0.0;
    }

    return length;
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

    // // Normalize starting edge weights to ensure PDF integrates to 1.0
    // // Phase-type distributions have PH(α, S) where α is the initial probability
    // // vector (must sum to 1.0). Starting edges represent α, so we normalize them.
    // double total_start_weight = 0.0;
    // struct ptd_vertex *start_vertex = graph->starting_vertex;

    // for (size_t i = 0; i < start_vertex->edges_length; ++i) {
    //     total_start_weight += start_vertex->edges[i]->weight;
    // }

    // // Normalize starting edge weights if they don't sum to 1.0
    // if (total_start_weight > 0.0 && fabs(total_start_weight - 1.0) > 1e-10) {
    //     double scale_factor = 1.0 / total_start_weight;
    //     for (size_t i = 0; i < start_vertex->edges_length; ++i) {
    //         start_vertex->edges[i]->weight *= scale_factor;
    //     }
    // }

    // Take initialization step to move from starting vertex (instantaneous transition)
    // Starting vertex edges now sum to 1.0, ensuring correct PDF normalization
    ptd_dph_probability_distribution_step(res);

    res->jumps = 0;
    // res->cdf = 0;  // Reset CDF after initialization
    // res->pmf = 0;  // Reset PMF after initialization
    res->weight_version_at_creation = graph->weight_version;
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
        int64_t granularity
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

    // Auto-select granularity with higher minimum for numerical stability
    if (granularity == 0) {
        // (int64_t)(max_rate * 2) is undefined behavior once max_rate*2 exceeds
        // INT64_MAX (~9.2e18): the cast collapses to a garbage (usually negative)
        // value that is then floored to 1000, so the rate/granularity check below
        // fails with a misleading "Increase the granularity" message. Detect the
        // overflow (and any NaN/Inf max_rate) up front and report the real cause —
        // an outsized rate, typically a diverged model or unscaled rate/time units.
        double desired_granularity = max_rate * 2.0;
        if (!(desired_granularity <= 9.0e18)) {
            snprintf(
                    (char *) ptd_err,
                    sizeof(ptd_err),
                    "Maximum outgoing rate (%.3e) is too large to build a phase-type "
                    "distribution: the auto-selected granularity (~2x the max rate) would "
                    "overflow the representable range. This usually means the model "
                    "diverged or the rate/time units need rescaling.\n",
                    max_rate
            );
            return NULL;
        }
        granularity = (int64_t)(desired_granularity);
        if (granularity < 1000) {
            PTD_LOG_DEBUG("Auto-selected granularity (%lld) increased to minimum (1000) for numerical stability", (long long) granularity);
            granularity = 1000;
        } else {
            PTD_LOG_DEBUG("Auto-selected granularity: %lld (max_rate=%.2f)", (long long) granularity, max_rate);
        }
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
    res->weight_version_at_creation = graph->weight_version;

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
 * Helper: Allocate 2D array for gradient computation
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
 * Compute PMF with gradient tracking using uniformization
 *
 * Returns PMF(time) and ∇PMF(time) with respect to parameters
 * PMF = Σ_k Poisson(k; λt) * P(absorption at step k)
 *
 * @param graph Parameterized phase-type graph
 * @param time Time point to evaluate
 * @param lambda Uniformization rate (max exit rate)
 * @param granularity Discretization steps per unit time
 * @param params Parameter vector (theta)
 * @param n_params Number of parameters
 * @param pmf_value Output: PMF value
 * @param pmf_gradient Output: PMF gradient (n_params,)
 * @return 0 on success, -1 on error
 */
static int compute_pmf_with_gradient(
    struct ptd_graph *graph,
    double time,
    double lambda,
    const double *lambda_grad,  // Gradient of uniformization rate: ∂λ/∂θ
    size_t granularity,
    const double *params,
    size_t n_params,
    double *pmf_value,
    double *pmf_gradient
) {
    if (graph == NULL || params == NULL || lambda_grad == NULL ||
        pmf_value == NULL || pmf_gradient == NULL) {
        return -1;
    }

    // Estimate max_jumps to capture Poisson tail (improved from fixed +100 buffer)
    // Use 6-sigma rule: cover 99.9999% of Poisson mass
    double lambda_t = lambda * time;
    double sigma = sqrt(lambda_t);
    size_t max_jumps = (size_t)(lambda_t + 6.0 * sigma + 100);

    PTD_LOG_DEBUG("PMF gradient computation: lambda=%.2f, time=%.2f, lambda*t=%.2f, max_jumps=%zu",
                 lambda, time, lambda_t, max_jumps);

    // Initialize probability and gradient arrays
    double *prob = (double *)calloc(graph->vertices_length, sizeof(double));
    double **prob_grad = alloc_2d(graph->vertices_length, n_params);

    if (prob == NULL || prob_grad == NULL) {
        free(prob);
        free_2d(prob_grad, graph->vertices_length);
        return -1;
    }

    // Starting vertex has probability 1, gradient 0
    // CRITICAL: Use graph->starting_vertex->index, NOT hardcoded 0!
    size_t starting_idx = graph->starting_vertex->index;
    prob[starting_idx] = 1.0;

    // Initialize output accumulators with Kahan summation for numerical stability
    struct kahan_sum pmf_kahan;
    kahan_init(&pmf_kahan);
    *pmf_value = 0.0;

    struct kahan_sum *grad_kahan = (struct kahan_sum *)malloc(n_params * sizeof(struct kahan_sum));
    if (grad_kahan == NULL) {
        free(prob);
        free_2d(prob_grad, graph->vertices_length);
        return -1;
    }
    for (size_t i = 0; i < n_params; i++) {
        kahan_init(&grad_kahan[i]);
        pmf_gradient[i] = 0.0;
    }

    // Precompute Poisson probabilities
    double *poisson_cache = (double *)malloc(max_jumps * sizeof(double));
    if (poisson_cache == NULL) {
        free(prob);
        free_2d(prob_grad, graph->vertices_length);
        return -1;
    }

    // lambda_t already defined at function scope (line 6635)
    for (size_t k = 0; k < max_jumps; k++) {
        poisson_cache[k] = exp(-lambda_t + k * log(lambda_t) - lgamma(k + 1));
    }

    // DP iteration over discrete time steps
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

        // Forward step through each vertex
        for (size_t v = 0; v < graph->vertices_length; v++) {
            struct ptd_vertex *vertex = graph->vertices[v];

            // Compute exit rate and its gradient
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

            // Sum all outgoing edge weights for exit rate
            for (size_t e = 0; e < vertex->edges_length; e++) {
                struct ptd_edge *edge = vertex->edges[e];

                // Current API: ALL edges have coefficients array
                // Parameterized edges: coefficients_length > 1
                // Constant edges: coefficients_length == 1
                double weight = 0.0;
                if (edge->coefficients_length > 1) {
                    // Parameterized edge: weight = Σ coefficients[i] * params[i]
                    for (size_t i = 0; i < n_params && i < edge->coefficients_length; i++) {
                        weight += edge->coefficients[i] * params[i];
                        exit_rate_grad[i] += edge->coefficients[i];
                    }
                } else {
                    // Constant edge: weight = coefficients[0]
                    weight = edge->coefficients[0];
                }
                exit_rate += weight;
            }

            // Process outgoing edges for probability transitions
            for (size_t e = 0; e < vertex->edges_length; e++) {
                struct ptd_edge *edge = vertex->edges[e];

                // Find target vertex index
                size_t to_idx = 0;
                for (size_t i = 0; i < graph->vertices_length; i++) {
                    if (graph->vertices[i] == edge->to) {
                        to_idx = i;
                        break;
                    }
                }

                // Compute edge weight and its gradient
                double weight = 0.0;
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

                if (edge->coefficients_length > 1) {
                    // Parameterized edge
                    for (size_t i = 0; i < n_params && i < edge->coefficients_length; i++) {
                        weight += edge->coefficients[i] * params[i];
                        weight_grad[i] = edge->coefficients[i];
                    }
                } else {
                    // Constant edge
                    weight = edge->coefficients[0];
                }

                // Update next probability using chain rule
                next_prob[to_idx] += prob[v] * weight / lambda;

                for (size_t i = 0; i < n_params; i++) {
                    next_prob_grad[to_idx][i] +=
                        prob_grad[v][i] * weight / lambda +
                        prob[v] * weight_grad[i] / lambda;
                }

                free(weight_grad);
            }

            // Self-loop probability (staying in same state)
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

        // Accumulate PMF contributions from absorbing states with Kahan summation
        for (size_t i = 0; i < graph->vertices_length; i++) {
            struct ptd_vertex *v = graph->vertices[i];
            if (v->edges_length == 0 && i > 0) {
                double poisson_k = poisson_cache[k];

                // Kahan summation for PMF value
                kahan_add(&pmf_kahan, poisson_k * prob[i]);
                *pmf_value = kahan_result(&pmf_kahan);

                // Gradient has TWO terms (chain rule through Poisson):
                // Term 1: ∂Poisson/∂P · ∂P/∂θ = Poisson(k) · ∂P_k/∂θ
                // Term 2: ∂Poisson/∂λ · ∂λ/∂θ · P = Poisson(k) · (k-λt)/λ · ∂λ/∂θ · P_k
                // (lambda_t already defined at function scope)
                double poisson_grad_factor = poisson_k * ((double)k - lambda_t) / lambda;

                for (size_t p = 0; p < n_params; p++) {
                    // Term 1: Poisson weight times probability gradient
                    kahan_add(&grad_kahan[p], poisson_k * prob_grad[i][p]);

                    // Term 2: Poisson gradient times probability
                    kahan_add(&grad_kahan[p], poisson_grad_factor * lambda_grad[p] * prob[i]);

                    pmf_gradient[p] = kahan_result(&grad_kahan[p]);
                }

                // CRITICAL: Zero out absorbed probability (prevent cumulation)
                prob[i] = 0;
                for (size_t p = 0; p < n_params; p++) {
                    prob_grad[i][p] = 0;
                }
            }
        }

        // Early termination when Poisson probability becomes negligible
        if (k > 10 && poisson_cache[k] < 1e-15) {
            PTD_LOG_DEBUG("Early termination at k=%zu (Poisson weight < 1e-15)", k);
            break;
        }

        // Convergence check: warn if last iteration has significant mass
        if (k == max_jumps - 1 && poisson_cache[k] > 1e-10) {
            PTD_LOG_WARNING("Poisson tail may be truncated: P(k=%zu) = %.2e (consider increasing granularity)",
                           k, poisson_cache[k]);
        }
    }

    free(prob);
    free_2d(prob_grad, graph->vertices_length);
    free(poisson_cache);
    free(grad_kahan);

    return 0;
}

/**
 * Forward algorithm with gradient tracking
 * Uses uniformization to compute PDF = PMF * lambda
 *
 * @param graph Parameterized phase-type graph
 * @param time Time point to evaluate
 * @param granularity Discretization steps (0 = auto)
 * @param params Parameter vector (theta)
 * @param n_params Number of parameters
 * @param pdf_value Output: PDF value
 * @param pdf_gradient Output: PDF gradient (n_params,)
 * @return 0 on success, -1 on error
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
    // Also track which vertex achieves max rate for gradient computation
    double lambda = 0.0;
    size_t max_vertex_idx = 0;

    for (size_t i = 0; i < graph->vertices_length; i++) {
        struct ptd_vertex *v = graph->vertices[i];
        double exit_rate = 0.0;

        for (size_t j = 0; j < v->edges_length; j++) {
            struct ptd_edge *e = v->edges[j];

            // Compute weight from coefficients and parameters
            // In gradient mode, treat all edges as potentially parameterized
            double weight = 0.0;
            if (e->coefficients_length >= n_params && n_params > 0) {
                // Parameterized edge: weight = Σ_k coeff[k] * param[k]
                for (size_t k = 0; k < n_params; k++) {
                    weight += e->coefficients[k] * params[k];
                }
            } else if (e->coefficients_length == 1) {
                // Constant edge or degenerate case
                weight = e->coefficients[0];
            } else {
                // Mismatch: not enough coefficients for parameters
                // Treat as zero weight (edge will be ignored)
                weight = 0.0;
            }
            exit_rate += weight;
        }

        if (exit_rate > lambda) {
            lambda = exit_rate;
            max_vertex_idx = i;  // Track which vertex has max rate
        }
    }

    // Compute ∂λ/∂θ: gradient of uniformization rate
    // λ is determined by the vertex with maximum exit rate
    // ∂λ/∂θ_k = sum of coefficients[k] for all edges from max_vertex
    double *lambda_grad = (double *)calloc(n_params, sizeof(double));
    if (lambda_grad == NULL) {
        return -1;
    }

    struct ptd_vertex *max_v = graph->vertices[max_vertex_idx];
    for (size_t j = 0; j < max_v->edges_length; j++) {
        struct ptd_edge *e = max_v->edges[j];
        if (e->coefficients_length >= n_params && n_params > 0) {
            // Parameterized edge: accumulate coefficients for ∂λ/∂θ
            for (size_t k = 0; k < n_params; k++) {
                lambda_grad[k] += e->coefficients[k];
            }
        }
        // Constant edges (coefficients_length < n_params) contribute 0 to gradient
    }

    if (lambda <= 0.0) {
        *pdf_value = 0.0;
        for (size_t i = 0; i < n_params; i++) {
            pdf_gradient[i] = 0.0;
        }
        free(lambda_grad);
        return 0;
    }

    // 2. Determine granularity (auto-select if not specified)
    // Higher minimum granularity (1000) improves numerical stability
    // Discretization error scales as O(1/granularity²)
    if (granularity == 0) {
        // Same overflow hazard as ptd_probability_distribution_context_create:
        // (size_t)(lambda * 2.0) is undefined once lambda*2 leaves the size_t
        // range, silently yielding a garbage granularity. Reject an outsized rate
        // with a clear message instead (a diverged model / unscaled units).
        double desired_granularity = lambda * 2.0;
        if (!(desired_granularity <= 9.0e18)) {
            snprintf(
                    (char *) ptd_err,
                    sizeof(ptd_err),
                    "Maximum outgoing rate (%.3e) is too large to compute a phase-type "
                    "PDF gradient: the auto-selected granularity (~2x the max rate) would "
                    "overflow. This usually means the model diverged or the rate/time "
                    "units need rescaling.\n",
                    lambda
            );
            free(lambda_grad);
            return -1;
        }
        granularity = (size_t)(desired_granularity);
        if (granularity < 1000) {
            PTD_LOG_DEBUG("Auto-selected granularity (%zu) increased to minimum (1000) for gradient computation", granularity);
            granularity = 1000;
        } else {
            PTD_LOG_DEBUG("Auto-selected granularity for gradient: %zu (lambda=%.2f)", granularity, lambda);
        }
    }

    // 3. Compute PMF and its gradient
    double pmf;
    double *pmf_grad = (double *)malloc(n_params * sizeof(double));
    if (pmf_grad == NULL) {
        free(lambda_grad);
        return -1;
    }

    int status = compute_pmf_with_gradient(graph, time, lambda, lambda_grad, granularity,
                                          params, n_params, &pmf, pmf_grad);

    if (status != 0) {
        free(pmf_grad);
        free(lambda_grad);
        return -1;
    }

    // 4. Convert PMF to PDF with gradient
    //    PDF(t; θ) = PMF(t; θ) · λ(θ)
    //
    // NOTE: Empirically determined that the lambda gradient term should be SUBTRACTED.
    // Mathematical analysis suggests this is because pmf_grad already accounts for
    // λ dependence through the Poisson gradient term, creating a double-counting issue
    // if we naively apply the product rule. The minus sign gives correct results.
    *pdf_value = pmf * lambda;
    for (size_t i = 0; i < n_params; i++) {
        pdf_gradient[i] = pmf_grad[i] * lambda - pmf * lambda_grad[i];
    }

    free(pmf_grad);
    free(lambda_grad);
    return 0;
}

/**
 * Helper: Ensure trace operations array has sufficient capacity
 *
 * Grows the operations array to accommodate at least 'required' operations.
 * Uses exponential growth strategy (doubling) for amortized O(1) append.
 *
 * @param trace Elimination trace to grow
 * @param capacity Pointer to current capacity (will be updated)
 * @param required Minimum required capacity
 * @return 0 on success, -1 on allocation failure
 */
static int ensure_operation_capacity(
    struct ptd_elimination_trace *trace,
    size_t *capacity,
    size_t required
) {
    if (required <= *capacity) {
        return 0; // Already has sufficient capacity
    }

    // Exponential growth: double capacity until we meet requirement
    size_t new_capacity = (*capacity) * 2;
    if (new_capacity < required) {
        new_capacity = required;
    }

    // Reallocate operations array
    struct ptd_trace_operation *new_ops = (struct ptd_trace_operation *)realloc(
        trace->operations,
        new_capacity * sizeof(struct ptd_trace_operation)
    );
    if (new_ops == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate memory for trace operations (required: %zu)", new_capacity);
        return -1;
    }

    trace->operations = new_ops;
    *capacity = new_capacity;
    return 0;
}

/**
 * Helper: Add CONST operation to trace
 *
 * Records a constant value operation.
 *
 * @param trace Elimination trace
 * @param capacity Pointer to current capacity
 * @param value Constant value
 * @return Operation index, or (size_t)-1 on error
 */
static size_t add_const_to_trace(
    struct ptd_elimination_trace *trace,
    size_t *capacity,
    double value
) {
    if (ensure_operation_capacity(trace, capacity, trace->operations_length + 1) != 0) {
        return (size_t)-1;
    }

    size_t idx = trace->operations_length++;
    struct ptd_trace_operation *op = &trace->operations[idx];

    op->op_type = PTD_OP_CONST;
    op->const_value = value;
    op->param_idx = 0;
    op->coefficients = NULL;

    // Debug: check for nan constants
    if (isnan(value)) {
        PTD_LOG_WARNING("add_const_to_trace called with NAN value at op_idx=%zu", idx);
    }
    op->coefficients_length = 0;
    op->operands = NULL;
    op->operands_length = 0;

    return idx;
}

/**
 * Helper: Add DOT operation to trace
 *
 * Records a dot product operation: dot(coefficients, params)
 *
 * @param trace Elimination trace
 * @param capacity Pointer to current capacity
 * @param coefficients Coefficient array
 * @param coefficients_length Length of coefficient array
 * @return Operation index, or (size_t)-1 on error
 */
static size_t add_dot_to_trace(
    struct ptd_elimination_trace *trace,
    size_t *capacity,
    const double *coefficients,
    size_t coefficients_length
) {
    if (ensure_operation_capacity(trace, capacity, trace->operations_length + 1) != 0) {
        return (size_t)-1;
    }

    size_t idx = trace->operations_length++;
    struct ptd_trace_operation *op = &trace->operations[idx];

    op->op_type = PTD_OP_DOT;
    op->const_value = 0.0;
    op->param_idx = 0;

    // Copy coefficients
    op->coefficients = (double *)malloc(coefficients_length * sizeof(double));
    if (op->coefficients == NULL) {
        trace->operations_length--; // Roll back
        sprintf((char*)ptd_err, "Failed to allocate memory for DOT coefficients");
        return (size_t)-1;
    }
    memcpy(op->coefficients, coefficients, coefficients_length * sizeof(double));
    op->coefficients_length = coefficients_length;

    op->operands = NULL;
    op->operands_length = 0;

    return idx;
}

/**
 * Helper: Add binary operation to trace (ADD, MUL, DIV)
 *
 * @param trace Elimination trace
 * @param capacity Pointer to current capacity
 * @param op_type Operation type (PTD_OP_ADD, PTD_OP_MUL, PTD_OP_DIV)
 * @param left_idx Left operand index
 * @param right_idx Right operand index
 * @return Operation index, or (size_t)-1 on error
 */
static size_t add_binary_op_to_trace(
    struct ptd_elimination_trace *trace,
    size_t *capacity,
    enum ptd_trace_op_type op_type,
    size_t left_idx,
    size_t right_idx
) {
    if (ensure_operation_capacity(trace, capacity, trace->operations_length + 1) != 0) {
        return (size_t)-1;
    }

    size_t idx = trace->operations_length++;
    struct ptd_trace_operation *op = &trace->operations[idx];

    op->op_type = op_type;
    op->const_value = 0.0;
    op->param_idx = 0;
    op->coefficients = NULL;
    op->coefficients_length = 0;

    // Allocate operands array (size 2)
    op->operands = (size_t *)malloc(2 * sizeof(size_t));
    if (op->operands == NULL) {
        trace->operations_length--; // Roll back
        sprintf((char*)ptd_err, "Failed to allocate memory for binary op operands");
        return (size_t)-1;
    }
    op->operands[0] = left_idx;
    op->operands[1] = right_idx;
    op->operands_length = 2;

    return idx;
}

/**
 * Helper: Add INV operation to trace (1 / x)
 *
 * @param trace Elimination trace
 * @param capacity Pointer to current capacity
 * @param operand_idx Operand index
 * @return Operation index, or (size_t)-1 on error
 */
static size_t add_inv_to_trace(
    struct ptd_elimination_trace *trace,
    size_t *capacity,
    size_t operand_idx
) {
    if (ensure_operation_capacity(trace, capacity, trace->operations_length + 1) != 0) {
        return (size_t)-1;
    }

    size_t idx = trace->operations_length++;
    struct ptd_trace_operation *op = &trace->operations[idx];

    op->op_type = PTD_OP_INV;
    op->const_value = 0.0;
    op->param_idx = 0;
    op->coefficients = NULL;
    op->coefficients_length = 0;

    // Allocate operands array (size 1)
    op->operands = (size_t *)malloc(1 * sizeof(size_t));
    if (op->operands == NULL) {
        trace->operations_length--; // Roll back
        sprintf((char*)ptd_err, "Failed to allocate memory for INV operand");
        return (size_t)-1;
    }
    op->operands[0] = operand_idx;
    op->operands_length = 1;

    return idx;
}

/**
 * Helper: Add SUM operation to trace
 *
 * Records a sum operation: sum(operands)
 *
 * @param trace Elimination trace
 * @param capacity Pointer to current capacity
 * @param operand_indices Array of operand indices
 * @param n_operands Number of operands
 * @return Operation index, or (size_t)-1 on error
 */
static size_t add_sum_to_trace(
    struct ptd_elimination_trace *trace,
    size_t *capacity,
    const size_t *operand_indices,
    size_t n_operands
) {
    if (n_operands == 0) {
        return add_const_to_trace(trace, capacity, 0.0);
    }
    if (n_operands == 1) {
        return operand_indices[0];
    }

    if (ensure_operation_capacity(trace, capacity, trace->operations_length + 1) != 0) {
        return (size_t)-1;
    }

    size_t idx = trace->operations_length++;
    struct ptd_trace_operation *op = &trace->operations[idx];

    op->op_type = PTD_OP_SUM;
    op->const_value = 0.0;
    op->param_idx = 0;
    op->coefficients = NULL;
    op->coefficients_length = 0;

    // Copy operand indices
    op->operands = (size_t *)malloc(n_operands * sizeof(size_t));
    if (op->operands == NULL) {
        trace->operations_length--; // Roll back
        sprintf((char*)ptd_err, "Failed to allocate memory for SUM operands");
        return (size_t)-1;
    }
    memcpy(op->operands, operand_indices, n_operands * sizeof(size_t));
    op->operands_length = n_operands;

    return idx;
}

// ============================================================================
// Public API Functions
// ============================================================================

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

    // Free vertex_rates
    if (trace->vertex_rates != NULL) {
        free(trace->vertex_rates);
    }

    // Free edge_probs (2D array)
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

    // Free vertex_targets (2D array)
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

    // Free states (2D array)
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
 * Destroy trace evaluation result and free all memory
 */
void ptd_trace_result_destroy(struct ptd_trace_result *result) {
    if (result == NULL) {
        return;
    }

    // Free vertex_rates
    if (result->vertex_rates != NULL) {
        free(result->vertex_rates);
    }

    // Free edge_probs (2D array)
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

    // Free vertex_targets (2D array)
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
 * Evaluate elimination trace with concrete parameter values
 *
 * Executes the recorded operation sequence with given parameters to produce
 * vertex rates and edge probabilities.
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
) {
    PTD_LOG_DEBUG("Evaluating trace with %zu parameters", params_length);

    // Validate parameters
    if (trace == NULL) {
        PTD_LOG_ERROR("Cannot evaluate trace: trace is NULL");
        sprintf((char*)ptd_err, "Trace is NULL");
        return NULL;
    }

    if (trace->param_length > 0 && params == NULL) {
        PTD_LOG_ERROR("Cannot evaluate trace: parameters required but not provided");
        sprintf((char*)ptd_err, "Parameters required for parameterized trace");
        return NULL;
    }

    if (params_length != trace->param_length) {
        PTD_LOG_ERROR("Cannot evaluate trace: expected %zu parameters, got %zu",
                      trace->param_length, params_length);
        sprintf((char*)ptd_err, "Expected %zu parameters, got %zu", trace->param_length, params_length);
        return NULL;
    }

    PTD_LOG_DEBUG("Evaluating %zu operations for %zu vertices",
                  trace->operations_length, trace->n_vertices);

    // Allocate values array for operation results
    size_t n_ops = trace->operations_length;
    double *values = (double *)calloc(n_ops, sizeof(double));
    if (values == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate values array");
        return NULL;
    }

    // Execute operations in order
    for (size_t i = 0; i < n_ops; i++) {
        const struct ptd_trace_operation *op = &trace->operations[i];

        switch (op->op_type) {
            case PTD_OP_CONST:
                values[i] = op->const_value;
                break;

            case PTD_OP_PARAM:
                values[i] = params[op->param_idx];
                break;

            case PTD_OP_DOT:
                // Dot product: sum(coefficients[j] * params[j])
                values[i] = 0.0;
                for (size_t j = 0; j < op->coefficients_length; j++) {
                    values[i] += op->coefficients[j] * params[j];
                }
                break;

            case PTD_OP_ADD:
                values[i] = values[op->operands[0]] + values[op->operands[1]];
                break;

            case PTD_OP_MUL:
                values[i] = values[op->operands[0]] * values[op->operands[1]];
                break;

            case PTD_OP_DIV:
                values[i] = values[op->operands[0]] / values[op->operands[1]];
                break;

            case PTD_OP_INV:
                values[i] = 1.0 / values[op->operands[0]];
                break;

            case PTD_OP_SUM:
                values[i] = 0.0;
                for (size_t j = 0; j < op->operands_length; j++) {
                    values[i] += values[op->operands[j]];
                }
                break;

            default:
                sprintf((char*)ptd_err, "Unknown operation type: %d", op->op_type);
                free(values);
                return NULL;
        }

        // Debug: Check for nan after operation
        if (isnan(values[i])) {
            PTD_LOG_WARNING("Operation %zu produced nan (type=%d)", i, op->op_type);
            if (op->op_type == PTD_OP_DIV) {
                PTD_LOG_DEBUG("  DIV: values[%zu]=%f / values[%zu]=%f",
                    op->operands[0], values[op->operands[0]],
                    op->operands[1], values[op->operands[1]]);
            } else if (op->op_type == PTD_OP_INV) {
                PTD_LOG_DEBUG("  INV: 1.0 / values[%zu]=%f",
                    op->operands[0], values[op->operands[0]]);
            } else if (op->op_type == PTD_OP_CONST) {
                PTD_LOG_DEBUG("  CONST: const_value=%f", op->const_value);
            }
        }

        // Debug: print first few operations
        if (i < 5) {
            PTD_LOG_DEBUG("Op %zu type=%d result=%f", i, op->op_type, values[i]);
        }
    }

    // Allocate result structure
    struct ptd_trace_result *result = (struct ptd_trace_result *)calloc(1, sizeof(*result));
    if (result == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate trace result");
        free(values);
        return NULL;
    }

    result->n_vertices = trace->n_vertices;

    // Extract vertex rates
    result->vertex_rates = (double *)malloc(trace->n_vertices * sizeof(double));
    if (result->vertex_rates == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate vertex rates");
        free(values);
        free(result);
        return NULL;
    }
    for (size_t i = 0; i < trace->n_vertices; i++) {
        size_t op_idx = trace->vertex_rates[i];
        result->vertex_rates[i] = values[op_idx];

        // Check for catastrophic errors
        if (isnan(result->vertex_rates[i])) {
            PTD_LOG_ERROR("Trace evaluation produced NaN for vertex_rates[%zu] (from values[%zu]) - numerical catastrophe", i, op_idx);
            sprintf((char*)ptd_err, "Trace evaluation produced NaN for vertex_rates[%zu] - numerical catastrophe", i);
            free(values);
            free(result->vertex_rates);
            free(result->edge_probs);
            free(result);
            return NULL;
        }
    }

    // Extract edge probabilities
    result->edge_probs_lengths = (size_t *)malloc(trace->n_vertices * sizeof(size_t));
    result->edge_probs = (double **)malloc(trace->n_vertices * sizeof(double *));
    if (result->edge_probs_lengths == NULL || result->edge_probs == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate edge probs arrays");
        free(values);
        ptd_trace_result_destroy(result);
        return NULL;
    }

    for (size_t i = 0; i < trace->n_vertices; i++) {
        size_t n_edges = trace->edge_probs_lengths[i];
        result->edge_probs_lengths[i] = n_edges;

        if (n_edges > 0) {
            result->edge_probs[i] = (double *)malloc(n_edges * sizeof(double));
            if (result->edge_probs[i] == NULL) {
                sprintf((char*)ptd_err, "Failed to allocate edge probs for vertex %zu", i);
                free(values);
                ptd_trace_result_destroy(result);
                return NULL;
            }

            for (size_t j = 0; j < n_edges; j++) {
                size_t op_idx = trace->edge_probs[i][j];
                result->edge_probs[i][j] = values[op_idx];

                // Check for catastrophic errors
                if (isnan(result->edge_probs[i][j])) {
                    PTD_LOG_ERROR("Trace evaluation produced NaN for edge_probs[%zu][%zu] (from values[%zu]) - numerical catastrophe", i, j, op_idx);
                    sprintf((char*)ptd_err, "Trace evaluation produced NaN for edge_probs[%zu][%zu] - numerical catastrophe", i, j);
                    free(values);
                    ptd_trace_result_destroy(result);
                    return NULL;
                }
            }
        } else {
            result->edge_probs[i] = NULL;
        }
    }

    // Copy vertex targets
    result->vertex_targets_lengths = (size_t *)malloc(trace->n_vertices * sizeof(size_t));
    result->vertex_targets = (size_t **)malloc(trace->n_vertices * sizeof(size_t *));
    if (result->vertex_targets_lengths == NULL || result->vertex_targets == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate vertex targets arrays");
        free(values);
        ptd_trace_result_destroy(result);
        return NULL;
    }

    for (size_t i = 0; i < trace->n_vertices; i++) {
        size_t n_targets = trace->vertex_targets_lengths[i];
        result->vertex_targets_lengths[i] = n_targets;

        if (n_targets > 0) {
            result->vertex_targets[i] = (size_t *)malloc(n_targets * sizeof(size_t));
            if (result->vertex_targets[i] == NULL) {
                sprintf((char*)ptd_err, "Failed to allocate vertex targets for vertex %zu", i);
                free(values);
                ptd_trace_result_destroy(result);
                return NULL;
            }

            memcpy(result->vertex_targets[i], trace->vertex_targets[i], n_targets * sizeof(size_t));
        } else {
            result->vertex_targets[i] = NULL;
        }
    }

    free(values);
    return result;
}

/**
 * Build reward compute graph from trace evaluation result
 *
 * Converts trace evaluation results into the internal reward_compute_graph
 * structure used by pdf/moment computations.
 *
 * The command array encodes the eliminated graph as a sequence of operations:
 * - Phase 1: Self-edges for vertex rates (from=to=vertex_idx)
 * - Phase 2: Edges for transition probabilities (in reverse order for DAG)
 * - Phase 3: Terminating command with NAN
 *
 * @param result Trace evaluation result
 * @param graph Graph structure (for vertex references)
 * @return Reward compute graph, or NULL on error
 */
struct ptd_desc_reward_compute *ptd_build_reward_compute_from_trace(
    const struct ptd_trace_result *result,
    struct ptd_graph *graph
) {
    if (result == NULL || graph == NULL) {
        sprintf((char*)ptd_err, "Invalid arguments to ptd_build_reward_compute_from_trace");
        return NULL;
    }

    size_t n_vertices = result->n_vertices;

    // Count total edges for command array size
    size_t total_edges = 0;
    for (size_t i = 0; i < n_vertices; i++) {
        total_edges += result->edge_probs_lengths[i];
    }

    // Allocate command array: vertex_rates + edges + terminator
    size_t n_commands = n_vertices + total_edges + 1;
    struct ptd_reward_increase *commands = (struct ptd_reward_increase *)calloc(
        n_commands, sizeof(struct ptd_reward_increase)
    );
    if (commands == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate reward compute commands");
        return NULL;
    }

    PTD_LOG_DEBUG("Building reward_compute: n_vertices=%zu total_edges=%zu n_commands=%zu",
        n_vertices, total_edges, n_commands);

    size_t cmd_idx = 0;

    // Phase 1: Add vertex rate commands (self-edges)
    for (size_t i = 0; i < n_vertices; i++) {
        commands[cmd_idx].from = i;
        commands[cmd_idx].to = i;
        commands[cmd_idx].multiplier = result->vertex_rates[i];

        // Debug vertex 0
        if (i == 0) {
            PTD_LOG_DEBUG("Building reward_compute cmd %zu: from=0 to=0 multiplier=%f",
                cmd_idx, result->vertex_rates[i]);
        }

        cmd_idx++;
    }

    // Phase 2: Add edge probability commands (reverse order for DAG)
    for (size_t ii = 0; ii < n_vertices; ii++) {
        size_t i = n_vertices - ii - 1;  // Reverse order

        for (size_t j = 0; j < result->edge_probs_lengths[i]; j++) {
            size_t target = result->vertex_targets[i][j];
            double prob = result->edge_probs[i][j];

            commands[cmd_idx].from = i;
            commands[cmd_idx].to = target;
            commands[cmd_idx].multiplier = prob;
            cmd_idx++;
        }
    }

    // Create result structure
    struct ptd_desc_reward_compute *res = (struct ptd_desc_reward_compute *)malloc(
        sizeof(struct ptd_desc_reward_compute)
    );
    if (res == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate reward compute descriptor");
        free(commands);
        return NULL;
    }

    res->length = cmd_idx;  // No terminator needed
    res->commands = commands;

    return res;
}

/**
 * Record elimination trace from parameterized graph
 *
 * Performs Gaussian elimination while recording all arithmetic operations
 * as a linear sequence. The trace can be efficiently replayed with different
 * parameter values without re-performing elimination.
 *
 * Algorithm: Gaussian elimination on graph structure (Algorithm 3 from paper)
 *
 * @param graph Parameterized graph (graph->parameterized must be true)
 * @return Elimination trace, or NULL on error (sets ptd_err)
 *
 * Time complexity: O(n³) where n = number of vertices
 * Space complexity: O(n²) for operation sequence
 *
 * This is a ONE-TIME cost - trace can be cached and reused.
 */

// Helper function to find edge index from parent to target
// Returns (size_t)-1 if not found
static inline size_t find_edge_idx(const struct ptd_elimination_trace *trace,
                                   size_t parent_idx,
                                   size_t target_idx) {
    for (size_t k = 0; k < trace->vertex_targets_lengths[parent_idx]; k++) {
        if (trace->vertex_targets[parent_idx][k] == target_idx) {
            return k;
        }
    }
    return (size_t)-1;
}

struct ptd_elimination_trace *ptd_record_elimination_trace(struct ptd_graph *graph) {
    PTD_LOG_DEBUG("Starting trace recording...");

    // Validate graph is parameterized
    if (graph == NULL) {
        PTD_LOG_ERROR("Cannot record trace: graph is NULL");
        sprintf((char*)ptd_err, "Graph is NULL");
        return NULL;
    }

    PTD_LOG_DEBUG("Recording trace for graph: %zu vertices, param_length=%zu",
                  graph->vertices_length, graph->param_length);

    // Allow trace recording for graphs with coefficient arrays (param_length >= 1)
    // This includes both single-parameter (param_length=1, is_parameterized=False)
    // and multi-parameter (param_length>1, is_parameterized=True) graphs
    if (graph->param_length < 1) {
        PTD_LOG_ERROR("Cannot record trace: graph has no parameters (param_length=0)");
        sprintf((char*)ptd_err, "Graph has no parameters (param_length=0). Trace recording requires parameterized edges.");
        return NULL;
    }

    // Get graph structure
    size_t n_vertices = graph->vertices_length;
    if (n_vertices == 0) {
        PTD_LOG_ERROR("Cannot record trace: graph has no vertices");
        sprintf((char*)ptd_err, "Graph has no vertices");
        return NULL;
    }

    PTD_LOG_DEBUG("Allocating trace structure for %zu vertices", n_vertices);

    // Allocate trace structure
    struct ptd_elimination_trace *trace = (struct ptd_elimination_trace *)calloc(1, sizeof(*trace));
    if (trace == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate trace structure");
        return NULL;
    }

    // Initialize operations array with initial capacity
    size_t operations_capacity = 1024;
    trace->operations = (struct ptd_trace_operation *)malloc(
        operations_capacity * sizeof(struct ptd_trace_operation)
    );
    if (trace->operations == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate operations array");
        free(trace);
        return NULL;
    }
    trace->operations_length = 0;

    // Copy graph metadata
    trace->n_vertices = n_vertices;
    trace->param_length = graph->param_length;
    trace->state_length = graph->state_length;
    trace->starting_vertex_idx = 0;  // Convention: first vertex in list
    trace->is_discrete = false;  // TODO: detect from graph

    // Allocate vertex_rates array
    trace->vertex_rates = (size_t *)malloc(n_vertices * sizeof(size_t));
    if (trace->vertex_rates == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate vertex_rates");
        ptd_elimination_trace_destroy(trace);
        return NULL;
    }

    // Allocate edge_probs arrays (2D ragged array)
    trace->edge_probs_lengths = (size_t *)calloc(n_vertices, sizeof(size_t));
    trace->edge_probs = (size_t **)calloc(n_vertices, sizeof(size_t *));
    if (trace->edge_probs_lengths == NULL || trace->edge_probs == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate edge_probs arrays");
        ptd_elimination_trace_destroy(trace);
        return NULL;
    }

    // Allocate vertex_targets arrays (2D ragged array)
    trace->vertex_targets_lengths = (size_t *)calloc(n_vertices, sizeof(size_t));
    trace->vertex_targets = (size_t **)calloc(n_vertices, sizeof(size_t *));
    if (trace->vertex_targets_lengths == NULL || trace->vertex_targets == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate vertex_targets arrays");
        ptd_elimination_trace_destroy(trace);
        return NULL;
    }

    // Copy vertex states
    trace->states = (int **)malloc(n_vertices * sizeof(int *));
    if (trace->states == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate states array");
        ptd_elimination_trace_destroy(trace);
        return NULL;
    }
    for (size_t i = 0; i < n_vertices; i++) {
        trace->states[i] = (int *)malloc(graph->state_length * sizeof(int));
        if (trace->states[i] == NULL) {
            sprintf((char*)ptd_err, "Failed to allocate state for vertex %zu", i);
            ptd_elimination_trace_destroy(trace);
            return NULL;
        }
        memcpy(trace->states[i], graph->vertices[i]->state,
               graph->state_length * sizeof(int));
    }

    // ========================================================================
    // PHASE 1: Compute vertex rates (rate = 1 / sum(edge_weights))
    // ========================================================================

    for (size_t i = 0; i < n_vertices; i++) {
        struct ptd_vertex *vertex = graph->vertices[i];

        // Count total edges (regular + parameterized)
        size_t total_edges = vertex->edges_length;

        if (total_edges == 0) {
            // Absorbing state: rate = 0
            trace->vertex_rates[i] = add_const_to_trace(trace, &operations_capacity, 0.0);
            if (trace->vertex_rates[i] == (size_t)-1) {
                ptd_elimination_trace_destroy(trace);
                return NULL;
            }
        } else {
            // Collect all edge weight operation indices
            size_t *weight_indices = (size_t *)malloc(total_edges * sizeof(size_t));
            if (weight_indices == NULL) {
                sprintf((char*)ptd_err, "Failed to allocate weight_indices");
                ptd_elimination_trace_destroy(trace);
                return NULL;
            }

            size_t weight_idx = 0;

            // Add all edges (all edges now have coefficient arrays)
            for (size_t j = 0; j < vertex->edges_length; j++) {
                struct ptd_edge *edge = vertex->edges[j];

                // All edges use DOT product with their coefficient arrays
                // (constant edges have single-element arrays)
                size_t op_idx = add_dot_to_trace(
                    trace, &operations_capacity,
                    edge->coefficients,
                    edge->coefficients_length
                );
                if (op_idx == (size_t)-1) {
                    free(weight_indices);
                    ptd_elimination_trace_destroy(trace);
                    return NULL;
                }

                if (i == 0 && j == 0) {
                    PTD_LOG_DEBUG("vertex 0 edge 0: coefficients_length=%zu, op_idx=%zu",
                        edge->coefficients_length, op_idx);
                }
                weight_indices[weight_idx++] = op_idx;
            }

            // Sum all weights
            size_t sum_idx = add_sum_to_trace(trace, &operations_capacity,
                                              weight_indices, weight_idx);
            if (sum_idx == (size_t)-1) {
                free(weight_indices);
                ptd_elimination_trace_destroy(trace);
                return NULL;
            }

            // Debug: check vertex 0
            if (i == 0) {
                PTD_LOG_DEBUG("vertex 0 has %zu edges, sum_idx=%zu", total_edges, sum_idx);
                PTD_LOG_DEBUG("vertex 0 edge 0: coefficients_length=%zu, weight=%f",
                    vertex->edges[0]->coefficients_length, vertex->edges[0]->weight);
                if (weight_idx > 0) {
                    PTD_LOG_DEBUG("vertex 0 weight_indices[0]=%zu", weight_indices[0]);
                }
            }

            // Rate = 1 / sum
            trace->vertex_rates[i] = add_inv_to_trace(trace, &operations_capacity, sum_idx);
            if (trace->vertex_rates[i] == (size_t)-1) {
                free(weight_indices);
                ptd_elimination_trace_destroy(trace);
                return NULL;
            }

            // Debug: check vertex 0
            if (i == 0) {
                // DEBUG_PRINT("DEBUG: vertex 0 rate op_idx=%zu\n", trace->vertex_rates[i]);
            }

            free(weight_indices);
        }
    }

    // ========================================================================
    // PHASE 2: Convert edges to probabilities and store in trace
    // ========================================================================

    // Allocate temporary edge map: (from, to) -> edge_prob_idx
    // For simplicity, use linear search for now (could optimize with hash table)

    for (size_t i = 0; i < n_vertices; i++) {
        struct ptd_vertex *vertex = graph->vertices[i];
        size_t n_edges = vertex->edges_length;

        if (n_edges == 0) {
            trace->edge_probs_lengths[i] = 0;
            trace->edge_probs[i] = NULL;
            trace->vertex_targets_lengths[i] = 0;
            trace->vertex_targets[i] = NULL;
            continue;
        }

        // Allocate arrays for this vertex's edges
        trace->edge_probs[i] = (size_t *)malloc(n_edges * sizeof(size_t));
        trace->vertex_targets[i] = (size_t *)malloc(n_edges * sizeof(size_t));
        if (trace->edge_probs[i] == NULL || trace->vertex_targets[i] == NULL) {
            sprintf((char*)ptd_err, "Failed to allocate edge arrays for vertex %zu", i);
            ptd_elimination_trace_destroy(trace);
            return NULL;
        }
        trace->edge_probs_lengths[i] = n_edges;
        trace->vertex_targets_lengths[i] = n_edges;

        // Process each edge
        for (size_t j = 0; j < n_edges; j++) {
            struct ptd_edge *edge = vertex->edges[j];
            size_t to_idx = edge->to->index;

            trace->vertex_targets[i][j] = to_idx;

            // Get edge weight operation index
            // All edges now use DOT product with their coefficient arrays
            size_t weight_idx = add_dot_to_trace(
                trace, &operations_capacity,
                edge->coefficients,
                edge->coefficients_length
            );

            if (weight_idx == (size_t)-1) {
                ptd_elimination_trace_destroy(trace);
                return NULL;
            }

            // Probability = weight * rate
            size_t prob_idx = add_binary_op_to_trace(
                trace, &operations_capacity,
                PTD_OP_MUL,
                weight_idx,
                trace->vertex_rates[i]
            );
            if (prob_idx == (size_t)-1) {
                ptd_elimination_trace_destroy(trace);
                return NULL;
            }

            trace->edge_probs[i][j] = prob_idx;
        }
    }

    // ========================================================================
    // PHASE 3: Gaussian Elimination
    // ========================================================================
    //
    // Algorithm: For each vertex i in order:
    //   - For each parent of i (where parent_idx >= i):
    //     - For each child of i:
    //       - Add bypass edge: parent -> child with prob = parent_to_i * i_to_child
    //     - Remove edge from parent to i
    //     - Renormalize parent's edges
    //

    // Build parent-child relationships (store parent indices, not pointers)
    size_t **parents_lists = (size_t **)calloc(n_vertices, sizeof(size_t *));
    size_t *parents_counts = (size_t *)calloc(n_vertices, sizeof(size_t));
    size_t *parents_capacities = (size_t *)calloc(n_vertices, sizeof(size_t));

    if (parents_lists == NULL || parents_counts == NULL || parents_capacities == NULL) {
        sprintf((char*)ptd_err, "Failed to allocate parent tracking arrays");
        free(parents_lists);
        free(parents_counts);
        free(parents_capacities);
        ptd_elimination_trace_destroy(trace);
        return NULL;
    }

    // Build parent lists (array of parent indices for each vertex)
    for (size_t i = 0; i < n_vertices; i++) {
        parents_lists[i] = NULL;
        parents_counts[i] = 0;
        parents_capacities[i] = 0;
    }

    for (size_t i = 0; i < n_vertices; i++) {
        for (size_t j = 0; j < trace->vertex_targets_lengths[i]; j++) {
            size_t to_idx = trace->vertex_targets[i][j];

            // Add i to parents of to_idx
            if (parents_counts[to_idx] >= parents_capacities[to_idx]) {
                size_t new_cap = parents_capacities[to_idx] == 0 ? 4 : parents_capacities[to_idx] * 2;
                size_t *new_parents = (size_t *)realloc(
                    parents_lists[to_idx], new_cap * sizeof(size_t)
                );
                if (new_parents == NULL) {
                    sprintf((char*)ptd_err, "Failed to resize parents list for vertex %zu", to_idx);
                    for (size_t k = 0; k < n_vertices; k++) {
                        free(parents_lists[k]);
                    }
                    free(parents_lists);
                    free(parents_counts);
                    free(parents_capacities);
                    ptd_elimination_trace_destroy(trace);
                    return NULL;
                }
                parents_lists[to_idx] = new_parents;
                parents_capacities[to_idx] = new_cap;
            }

            parents_lists[to_idx][parents_counts[to_idx]++] = i;
        }
    }

    // Eliminate vertices in order
    for (size_t i = 0; i < n_vertices; i++) {
        size_t n_children = trace->vertex_targets_lengths[i];

        if (n_children == 0) {
            // Absorbing state, nothing to eliminate
            continue;
        }

        // For each parent of vertex i
        for (size_t parent_list_idx = 0; parent_list_idx < parents_counts[i]; parent_list_idx++) {
            size_t parent_idx = parents_lists[i][parent_list_idx];

            // Skip if parent already processed
            if (parent_idx < i) {
                continue;
            }

            // Find edge from parent to i
            size_t parent_to_i_edge_idx = find_edge_idx(trace, parent_idx, i);
            if (parent_to_i_edge_idx == (size_t)-1) {
                // Parent no longer has edge to i (removed in earlier iteration)
                continue;
            }

            size_t parent_to_i_prob = trace->edge_probs[parent_idx][parent_to_i_edge_idx];

            // For each child of i
            for (size_t child_edge_idx = 0; child_edge_idx < n_children; child_edge_idx++) {
                size_t child_idx = trace->vertex_targets[i][child_edge_idx];
                size_t i_to_child_prob = trace->edge_probs[i][child_edge_idx];

                // CASE A: Self-loop (child == parent)
                if (child_idx == parent_idx) {
                    // Skip self-loops for now (TODO: handle in future)
                    continue;
                }

                // Skip edge back to i
                if (child_idx == i) {
                    continue;
                }

                // Bypass probability: parent_to_i * i_to_child
                size_t bypass_prob = add_binary_op_to_trace(
                    trace, &operations_capacity,
                    PTD_OP_MUL,
                    parent_to_i_prob,
                    i_to_child_prob
                );
                if (bypass_prob == (size_t)-1) {
                    for (size_t k = 0; k < n_vertices; k++) {
                        free(parents_lists[k]);
                    }
                    free(parents_lists);
                    free(parents_counts);
                    free(parents_capacities);
                    ptd_elimination_trace_destroy(trace);
                    return NULL;
                }

                // Check if parent already has edge to child
                size_t parent_to_child_edge_idx = find_edge_idx(trace, parent_idx, child_idx);

                if (parent_to_child_edge_idx != (size_t)-1) {
                    // CASE B: Update existing edge
                    size_t old_prob = trace->edge_probs[parent_idx][parent_to_child_edge_idx];

                    // new_prob = old_prob + bypass_prob
                    size_t new_prob = add_binary_op_to_trace(
                        trace, &operations_capacity,
                        PTD_OP_ADD,
                        old_prob,
                        bypass_prob
                    );
                    if (new_prob == (size_t)-1) {
                        for (size_t k = 0; k < n_vertices; k++) {
                            free(parents_lists[k]);
                        }
                        free(parents_lists);
                        free(parents_counts);
                        free(parents_capacities);
                        ptd_elimination_trace_destroy(trace);
                        return NULL;
                    }

                    trace->edge_probs[parent_idx][parent_to_child_edge_idx] = new_prob;
                } else {
                    // CASE C: Create new edge
                    // Need to resize edge_probs and vertex_targets arrays
                    size_t old_length = trace->edge_probs_lengths[parent_idx];
                    size_t new_length = old_length + 1;

                    size_t *new_edge_probs = (size_t *)realloc(
                        trace->edge_probs[parent_idx],
                        new_length * sizeof(size_t)
                    );
                    size_t *new_vertex_targets = (size_t *)realloc(
                        trace->vertex_targets[parent_idx],
                        new_length * sizeof(size_t)
                    );

                    if (new_edge_probs == NULL || new_vertex_targets == NULL) {
                        sprintf((char*)ptd_err, "Failed to resize edge arrays for vertex %zu", parent_idx);
                        free(new_edge_probs);
                        free(new_vertex_targets);
                        for (size_t k = 0; k < n_vertices; k++) {
                            free(parents_lists[k]);
                        }
                        free(parents_lists);
                        free(parents_counts);
                        free(parents_capacities);
                        ptd_elimination_trace_destroy(trace);
                        return NULL;
                    }

                    trace->edge_probs[parent_idx] = new_edge_probs;
                    trace->vertex_targets[parent_idx] = new_vertex_targets;
                    trace->edge_probs_lengths[parent_idx] = new_length;
                    trace->vertex_targets_lengths[parent_idx] = new_length;

                    trace->edge_probs[parent_idx][old_length] = bypass_prob;
                    trace->vertex_targets[parent_idx][old_length] = child_idx;
                }
            }

            // Remove edge from parent to i (mark as -1)
            trace->edge_probs[parent_idx][parent_to_i_edge_idx] = (size_t)-1;

            // NORMALIZATION: Renormalize parent's edges
            // Collect indices of non-removed edges
            size_t *valid_edge_indices = (size_t *)malloc(
                trace->edge_probs_lengths[parent_idx] * sizeof(size_t)
            );
            if (valid_edge_indices == NULL) {
                sprintf((char*)ptd_err, "Failed to allocate valid_edge_indices");
                for (size_t k = 0; k < n_vertices; k++) {
                    free(parents_lists[k]);
                }
                free(parents_lists);
                free(parents_counts);
                free(parents_capacities);
                ptd_elimination_trace_destroy(trace);
                return NULL;
            }

            size_t n_valid = 0;
            for (size_t k = 0; k < trace->edge_probs_lengths[parent_idx]; k++) {
                if (trace->edge_probs[parent_idx][k] != (size_t)-1) {
                    valid_edge_indices[n_valid++] = k;
                }
            }

            if (n_valid > 0) {
                // Create array of probability operation indices
                size_t *prob_ops = (size_t *)malloc(n_valid * sizeof(size_t));
                if (prob_ops == NULL) {
                    sprintf((char*)ptd_err, "Failed to allocate prob_ops");
                    free(valid_edge_indices);
                    for (size_t k = 0; k < n_vertices; k++) {
                        free(parents_lists[k]);
                    }
                    free(parents_lists);
                    free(parents_counts);
                    free(parents_capacities);
                    ptd_elimination_trace_destroy(trace);
                    return NULL;
                }

                for (size_t k = 0; k < n_valid; k++) {
                    prob_ops[k] = trace->edge_probs[parent_idx][valid_edge_indices[k]];
                }

                // Sum all valid probabilities
                size_t total_idx = add_sum_to_trace(trace, &operations_capacity, prob_ops, n_valid);
                free(prob_ops);

                if (total_idx == (size_t)-1) {
                    free(valid_edge_indices);
                    for (size_t k = 0; k < n_vertices; k++) {
                        free(parents_lists[k]);
                    }
                    free(parents_lists);
                    free(parents_counts);
                    free(parents_capacities);
                    ptd_elimination_trace_destroy(trace);
                    return NULL;
                }

                // Normalize: prob = prob / total
                for (size_t k = 0; k < n_valid; k++) {
                    size_t edge_idx = valid_edge_indices[k];
                    size_t old_prob = trace->edge_probs[parent_idx][edge_idx];

                    size_t new_prob = add_binary_op_to_trace(
                        trace, &operations_capacity,
                        PTD_OP_DIV,
                        old_prob,
                        total_idx
                    );
                    if (new_prob == (size_t)-1) {
                        free(valid_edge_indices);
                        for (size_t k = 0; k < n_vertices; k++) {
                            free(parents_lists[k]);
                        }
                        free(parents_lists);
                        free(parents_counts);
                        free(parents_capacities);
                        ptd_elimination_trace_destroy(trace);
                        return NULL;
                    }

                    trace->edge_probs[parent_idx][edge_idx] = new_prob;
                }
            }

            free(valid_edge_indices);
        }
    }

    // ========================================================================
    // PHASE 4: Clean up removed edges
    // ========================================================================

    for (size_t i = 0; i < n_vertices; i++) {
        // Count valid edges
        size_t n_valid = 0;
        for (size_t j = 0; j < trace->edge_probs_lengths[i]; j++) {
            if (trace->edge_probs[i][j] != (size_t)-1) {
                n_valid++;
            }
        }

        if (n_valid == 0) {
            // No edges left
            free(trace->edge_probs[i]);
            free(trace->vertex_targets[i]);
            trace->edge_probs[i] = NULL;
            trace->vertex_targets[i] = NULL;
            trace->edge_probs_lengths[i] = 0;
            trace->vertex_targets_lengths[i] = 0;
        } else if (n_valid < trace->edge_probs_lengths[i]) {
            // Compact arrays to remove -1 entries
            size_t *new_edge_probs = (size_t *)malloc(n_valid * sizeof(size_t));
            size_t *new_vertex_targets = (size_t *)malloc(n_valid * sizeof(size_t));

            if (new_edge_probs == NULL || new_vertex_targets == NULL) {
                sprintf((char*)ptd_err, "Failed to allocate compacted edge arrays for vertex %zu", i);
                free(new_edge_probs);
                free(new_vertex_targets);
                for (size_t k = 0; k < n_vertices; k++) {
                    free(parents_lists[k]);
                }
                free(parents_lists);
                free(parents_counts);
                free(parents_capacities);
                ptd_elimination_trace_destroy(trace);
                return NULL;
            }

            size_t compact_idx = 0;
            for (size_t j = 0; j < trace->edge_probs_lengths[i]; j++) {
                if (trace->edge_probs[i][j] != (size_t)-1) {
                    new_edge_probs[compact_idx] = trace->edge_probs[i][j];
                    new_vertex_targets[compact_idx] = trace->vertex_targets[i][j];
                    compact_idx++;
                }
            }

            free(trace->edge_probs[i]);
            free(trace->vertex_targets[i]);
            trace->edge_probs[i] = new_edge_probs;
            trace->vertex_targets[i] = new_vertex_targets;
            trace->edge_probs_lengths[i] = n_valid;
            trace->vertex_targets_lengths[i] = n_valid;
        }
    }

    // Cleanup parent tracking structures
    for (size_t i = 0; i < n_vertices; i++) {
        free(parents_lists[i]);
    }
    free(parents_lists);
    free(parents_counts);
    free(parents_capacities);

    PTD_LOG_INFO("Trace recording complete: %zu vertices, %zu operations, param_length=%zu",
                 trace->n_vertices, trace->operations_length, trace->param_length);

    return trace;
}


/* For Laplace transform */

// find (one of) the absorbing children if any of each state
struct ptd_edge** ptd_graph_vertices_absorbing_edge(struct ptd_graph *graph) {

  struct ptd_edge **abs_edges = (struct ptd_edge **) calloc(graph->vertices_length, sizeof(*abs_edges));
    
  for (size_t v = 0; v < graph->vertices_length; ++v) {
      abs_edges[v] = NULL;
      for (size_t e = 0; e < graph->vertices[v]->edges_length; ++e) {
          if (graph->vertices[v]->edges[e]->to->edges_length == 0) {
            abs_edges[v] = graph->vertices[v]->edges[e];
            break;
          }
      }      
  }
  return abs_edges;
}

struct ptd_clone_res ptd_graph_laplace_transform(struct ptd_graph *graph, struct ptd_avl_tree *avl_tree, double theta) {

    // Clone the graph
    struct ptd_clone_res cloned = ptd_clone_graph(graph, avl_tree);
    struct ptd_graph *new_graph = cloned.graph;

    // Get array mapping each vertex to an absorbing edge if it has any
    struct ptd_edge **vertices_absorbing_edge = ptd_graph_vertices_absorbing_edge(new_graph);

    // Find an absorbing state
    struct ptd_vertex *absorbing_vertex = NULL;
    for (size_t v = new_graph->vertices_length; v > 0; --v) {
        if (new_graph->vertices[v-1]->edges_length == 0) {
            absorbing_vertex = new_graph->vertices[v-1];
            break;
        }
    }

    // For each transient state, add theta to existing absorbing edge or create new one
    for (size_t v = 0; v < new_graph->vertices_length; ++v) {
        struct ptd_vertex *vertex = new_graph->vertices[v];
        // Skip starting and absorbing vertices
        if (vertex == new_graph->starting_vertex || vertex->edges_length == 0) {
            continue;
        }
        if (vertices_absorbing_edge[v]) {
            // Add weight to existing edge to absorbing
            for (size_t e = 0; e < vertex->edges_length; ++e) {
                if (vertices_absorbing_edge[v] == vertex->edges[e]) {
                    ptd_edge_update_weight(vertex->edges[e], vertex->edges[e]->weight + theta);
                    break;
                }
            }
        } else {
            // Add new edge to absorbing with weight theta
            ptd_graph_add_edge(vertex, absorbing_vertex, &theta, 1);
        }
    }
    free(vertices_absorbing_edge);

    return cloned;
}
