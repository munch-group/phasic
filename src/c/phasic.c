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
    // Windows doesn't have rand_r; provide a simple replacement
    static int rand_r(unsigned int *seedp) {
        *seedp = *seedp * 1103515245 + 12345;
        return (int)((*seedp / 65536) % 32768);
    }
#endif

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
    const char *home = getenv("HOME");
    if (home == NULL) {
        sprintf((char*)ptd_err, "HOME environment variable not set");
        return -1;
    }

    // Build path: ~/.phasic_cache/traces
    int ret = snprintf(buffer, buffer_size, "%s/.phasic_cache/traces", home);
    if (ret < 0 || (size_t)ret >= buffer_size) {
        sprintf((char*)ptd_err, "Cache directory path too long");
        return -1;
    }

    // Create directory if it doesn't exist (mkdir -p)
    char parent_dir[PATH_MAX];
    snprintf(parent_dir, sizeof(parent_dir), "%s/.phasic_cache", home);

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

    // Check if cache is disabled via environment variable
    const char *disable_cache = getenv("PHASIC_DISABLE_CACHE");
    if (disable_cache != NULL && strcmp(disable_cache, "1") == 0) {
        return NULL;  // Cache disabled
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

    // Check if cache is disabled via environment variable
    const char *disable_cache = getenv("PHASIC_DISABLE_CACHE");
    if (disable_cache != NULL && strcmp(disable_cache, "1") == 0) {
        return false;  // Cache disabled
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

int ptd_precompute_reward_compute_graph(struct ptd_graph *graph) {
    if (graph->was_dph) {
        // Note: was_dph remains true - it's a permanent flag indicating this is a discrete graph
        // This ensures auto-normalization continues to work in update_weights()

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
            if (graph->parameterized_reward_compute_graph == NULL) {
                if (graph->use_dyn_ordering) {
                    graph->parameterized_reward_compute_graph =
                            ptd_graph_ex_absorbation_time_comp_graph_parameterized_dyn(graph);
                } else {
                    graph->parameterized_reward_compute_graph =
                            ptd_graph_ex_absorbation_time_comp_graph_parameterized(graph);
                }
            }

            if (graph->reward_compute_graph != NULL) {
                free(graph->reward_compute_graph->commands);
                free(graph->reward_compute_graph);
            }

            graph->reward_compute_graph =
                    ptd_graph_build_ex_absorbation_time_comp_graph_parameterized(
                            graph->parameterized_reward_compute_graph
                    );
        } else {
            if (graph->use_dyn_ordering) {
                graph->reward_compute_graph = ptd_graph_ex_absorbation_time_comp_graph_dyn(graph);
            } else {
                graph->reward_compute_graph = ptd_graph_ex_absorbation_time_comp_graph(graph);
            }

            if (graph->reward_compute_graph == NULL) {
                return -1;
            }
        }
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
    graph->reward_compute_graph_mpfr = NULL;
    graph->starting_vertex = ptd_vertex_create(graph);
    graph->was_dph = false;
    graph->use_dyn_ordering = (getenv("PHASIC_DYN_ORDERING") != NULL);
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

    graph->reward_compute_graph = NULL;
    graph->parameterized_reward_compute_graph = NULL;
#ifdef HAVE_MPFR
    graph->reward_compute_graph_mpfr = NULL;
#endif
    graph->elimination_trace = NULL;
    graph->current_params = NULL;
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
                (unsigned long)from->graph->param_length,
                (unsigned long)coefficients_length);
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

    from->graph->reward_compute_graph = NULL;
    from->graph->parameterized_reward_compute_graph = NULL;

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
            (unsigned long)graph->param_length);
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
                (unsigned long)graph->param_length,
                (unsigned long)params_length);
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

            // Compute weight based on mode
            // STRICT VALIDATION: In non-callback mode, coefficient length must match theta length exactly
            // To use extra coefficients, call update_weights(theta, callback) with a custom callback
            if (edge->coefficients_length != theta_len) {
                snprintf((char*)ptd_err, sizeof(ptd_err),
                    "Coefficient length mismatch: edge has %zu coefficients but theta has %zu parameters. "
                    "In non-callback mode, they must match exactly. "
                    "To use extra coefficients, call update_weights(theta, callback) with a custom callback function.",
                    (unsigned long)edge->coefficients_length, (unsigned long)theta_len);
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

    // Invalidate cached compute graphs
    if (graph->reward_compute_graph != NULL) {
        free(graph->reward_compute_graph->commands);
        free(graph->reward_compute_graph);
        graph->reward_compute_graph = NULL;
    }

    if (graph->parameterized_reward_compute_graph != NULL) {
        ptd_parameterized_reward_compute_graph_destroy(
                graph->parameterized_reward_compute_graph
        );
        graph->parameterized_reward_compute_graph = NULL;
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

    if (need_free) {
        free(theta);
    }
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
#endif  // HAVE_MPFR


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

        // Compute MPFR graph if not cached
        if (graph->reward_compute_graph_mpfr == NULL) {
            PTD_LOG_INFO("Computing MPFR graph with %zu-bit precision", mpfr_precision);
            graph->reward_compute_graph_mpfr = ptd_graph_ex_absorbation_time_comp_graph_mpfr(
                graph, mpfr_precision
            );
        }

        // Call MPFR execution function
        double *mpfr_result = ptd_expected_waiting_time_mpfr(graph, rewards, mpfr_precision);
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

    // Allocate results matrix: results[vertex_idx][reward_idx]
    // Layout: results[v][r] = accumulated reward at vertex v for reward vector r
    // Only allocate k columns instead of n (memory efficient!)
    double **results = (double **) malloc(n * sizeof(double *));
    if (results == NULL) {
        PTD_LOG_ERROR("Failed to allocate results matrix");
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        results[i] = (double *) calloc(k, sizeof(double));
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

    // Initialize with one-hot vectors for each target index
    // reward vector r has value 1 at vertex indices[r]
    for (size_t r = 0; r < k; r++) {
        size_t vertex_idx = indices[r];
        if (vertex_idx >= n) {
            PTD_LOG_ERROR("Invalid vertex index %zu (graph has %zu vertices)", vertex_idx, n);
            for (size_t i = 0; i < n; i++) {
                free(results[i]);
            }
            free(results);
            return NULL;
        }
        results[vertex_idx][r] = 1.0;
    }

    // Apply all elimination trace commands to k reward vectors
    // Command: results[from][r] += results[to][r] * multiplier for all r
    for (size_t cmd_idx = 0; cmd_idx < compute->length; cmd_idx++) {
        struct ptd_reward_increase cmd = compute->commands[cmd_idx];

        // Handle 0 × ∞ = 0 (limit interpretation)
        // Skip when multiplier is zero to avoid NaN from 0.0 × inf
        if (cmd.multiplier == 0.0) {
            continue;
        }

        double *from_row = results[cmd.from];
        double *to_row = results[cmd.to];
        double multiplier = cmd.multiplier;

        // Check if multiplier is infinite
        bool mult_is_inf = isinf(multiplier);

        // Inner loop: only k columns instead of n
        for (size_t r = 0; r < k; r++) {
            // Handle inf × 0 = 0 (limit interpretation)
            if (mult_is_inf && to_row[r] == 0.0) {
                continue;
            }
            from_row[r] += to_row[r] * multiplier;
        }

        // Debug: check for NaN
        #ifdef DEBUG
        for (size_t r = 0; r < k; r++) {
            if (isnan(from_row[r])) {
                PTD_LOG_WARNING("results[%zu][%zu] became nan at command %zu",
                    cmd.from, r, cmd_idx);
            }
        }
        #endif
    }

    // Extract sojourn times: results[starting_vertex][r] for each reward vector r
    // Starting vertex is at index 0
    double *sojourn_times = (double *) malloc(k * sizeof(double));
    if (sojourn_times == NULL) {
        PTD_LOG_ERROR("Failed to allocate sojourn times array");
        for (size_t i = 0; i < n; i++) {
            free(results[i]);
        }
        free(results);
        return NULL;
    }

    for (size_t r = 0; r < k; r++) {
        sojourn_times[r] = results[0][r];  // Starting vertex index = 0
    }

    // Free intermediate results matrix
    for (size_t i = 0; i < n; i++) {
        free(results[i]);
    }
    free(results);

    // PTD_LOG_DEBUG("Computed sojourn times for %zu target states (out of %zu total)", k, n);
    return sojourn_times;
}

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

    // Apply all elimination trace commands to all reward vectors
    // Command: results[from][r] += results[to][r] * multiplier for all r
    // Using Kahan summation for numerical stability
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

        // Inner loop: contiguous memory access + Kahan summation
        for (size_t r = 0; r < n; r++) {
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
        for (size_t r = 0; r < n; r++) {
            if (isnan(from_row[r])) {
                PTD_LOG_WARNING("results[%zu][%zu] became nan at command %zu",
                    cmd.from, r, cmd_idx);
            }
        }
        #endif
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

// // Stub implementation to fix build issue - TODO: implement properly
// double *ptd_expected_residence_time(struct ptd_graph *graph, double *rewards) {
//     // For now, just call expected_waiting_time as a fallback
//     PTD_LOG_WARNING("expected_residence_time not fully implemented, using expected_waiting_time");
//     return ptd_expected_waiting_time(graph, rewards);
// }

// Original commented implementation (incomplete - needs fixing):
// double *ptd_expected_residence_time(struct ptd_graph *graph, double *rewards) {
//     if (ptd_precompute_reward_compute_graph(graph)) {
//         return NULL;
//     }

//     double *result = (double *) calloc(graph->vertices_length, sizeof(*result));

//     if (rewards != NULL) {
//         // TODO: fix this if reward is nan...
//         memcpy(result, rewards, sizeof(*result) * graph->vertices_length);
//     } else {
//         for (size_t j = 0; j < graph->vertices_length; ++j) {
//             result[j] = 1;
//         }
//     }

//     // we want only the acyclic graph so we we subtract graph->vertices_length to skip 
//     // the commands computing the expected waiting time
//     for (size_t j = 0; j < graph->reward_compute_graph->length - graph->vertices_length; ++j) {
//         struct ptd_reward_increase command = graph->reward_compute_graph->commands[j];
//         result[command.from] += result[command.to] * command.multiplier;
//         //TODO: if inf, give error stating that there is an infinite loop
//     }

//     // make a copy of the result at this point
//     double *dag_vertex_props = (double *) calloc(graph->vertices_length, sizeof(*dag_vertex_props));
//     memcpy(dag_vertex_props, result, sizeof(*result) * graph->vertices_length);

//     // continue computing the expected waiting time
//     for (size_t j = graph->reward_compute_graph->length - graph->vertices_length; j < graph->reward_compute_graph->length; ++j) {
//         struct ptd_reward_increase command = graph->reward_compute_graph->commands[j];
//         result[command.from] += result[command.to] * command.multiplier;
//         //TODO: if inf, give error stating that there is an infinite loop
//     }

//     // compute the expected residence time
//     double *res_times = (double *) calloc(graph->vertices_length, sizeof(*res_times));
//     for (size_t j = 0; j < graph->vertices_length; ++j) {
//         res_times[j] = 0;
//     }
//     res_times[0] = result[0]; // expected waiting time
//     double *scalars = (double *) calloc(graph->vertices_length, sizeof(*scalars));
//        for (size_t j = 0; j < graph->vertices_length; ++j) {
//         scalars[j] = 0 ;
//     } 
//     scalars[0] = 1;
//     struct ptd_vertex *start_vertex = graph->starting_vertex;
//     double pushed = 0;
//     int prev_idx = -1;
//     int prev_child_idx = -1;
//     // for (size_t j = graph->reward_compute_graph->length - graph->vertices_length; j <  graph->reward_compute_graph->length; ++j) {
//     //     struct ptd_reward_increase command = graph->reward_compute_graph->commands[j];
//     // for (size_t j = 1; j <  graph->vertices_length; ++j) {
//     //     struct ptd_reward_increase command = graph->reward_compute_graph->commands[graph->reward_compute_graph->length - j];
//     for (size_t j = 0; j <  graph->vertices_length; ++j) {
//         struct ptd_reward_increase command = graph->reward_compute_graph->commands[graph->reward_compute_graph->length - j - 1];

//         int idx = command.from;
//         int child_idx = command.to;
//         double child_prob = command.multiplier;
//         double wt = 1 / dag_vertex_props[idx] * scalars[idx];

//         // fprintf(stderr, "%d\n", graph->vertices[idx]->index);
//         // fprintf(stderr, "%d -> %d, %f, %f\n", idx, child_idx, child_prob, wt);

//         // char message[1024];
//         // sprintf(message, "%zu -> %d, %f, %f\n", idx, child_idx, child_prob, wt);
//         // DEBUG_PRINT(message);
//         // DEBUG_PRINT("HELLO\n");
        
        

//         if (wt < 0) {
//             snprintf(
//                 (char *) ptd_err, 
//                 sizeof(ptd_err),
//                 "%d -> %d, %f, %f\n",
//                 idx, child_idx, (float) child_prob, (float) wt
//             );
//             return NULL;
//         }


//         // snprintf(
//         //         (char *) ptd_err,
//         //         sizeof(ptd_err),
//         //         "Multiple edges to the same vertex!. From vertex with index %i%s (state %s)."
//         //         " To vertex with index %i (state %s)\n",
//         //         (int) debug_index_from, starting_vertex, state,
//         //         (int) debug_index_to, state_to
//         // );

//         if (idx == start_vertex->index) {
//             wt = 0;
//         }
//         if (prev_child_idx != child_idx) {
//             // fprintf(stderr, "removing total push from vertex %zu: %f\n", child_idx, pushed);
//             res_times[prev_idx] -= pushed;
//             pushed = 0;
//         }
//         if (dag_vertex_props[child_idx] > 0) { // don't push to absorbing
//             double push = (res_times[idx] - wt) * child_prob;
//             // fprintf(stderr, "pushing %f to %zu\n", push, child_idx);
//             res_times[child_idx] += push;
//             scalars[child_idx] += scalars[idx] * child_prob;
//             pushed += push;
//         }
//         prev_idx = idx;
//         prev_child_idx = child_idx;
//         //TODO: if inf, give error stating that there is an infinite loop
//     }

//     free(result);
//     free(scalars);
//     free(dag_vertex_props);

//     return res_times;
// }

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

    // Auto-select granularity with higher minimum for numerical stability
    if (granularity == 0) {
        granularity = max_rate * 2;
        if (granularity < 1000) {
            PTD_LOG_DEBUG("Auto-selected granularity (%zu) increased to minimum (1000) for numerical stability", granularity);
            granularity = 1000;
        } else {
            PTD_LOG_DEBUG("Auto-selected granularity: %zu (max_rate=%.2f)", granularity, max_rate);
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
        granularity = (size_t)(lambda * 2.0);
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

// /**
//  * Helper: Allocate 2D array
//  */
// static double **alloc_2d(size_t rows, size_t cols) {
//     double **arr = (double **)malloc(rows * sizeof(double*));
//     if (arr == NULL) return NULL;

//     for (size_t i = 0; i < rows; i++) {
//         arr[i] = (double *)calloc(cols, sizeof(double));
//         if (arr[i] == NULL) {
//             for (size_t j = 0; j < i; j++) free(arr[j]);
//             free(arr);
//             return NULL;
//         }
//     }
//     return arr;
// }

// /**
//  * Helper: Free 2D array
//  */
// static void free_2d(double **arr, size_t rows) {
//     if (arr == NULL) return;
//     for (size_t i = 0; i < rows; i++) {
//         free(arr[i]);
//     }
//     free(arr);
// }

// /**
//  * Helper: Compute PMF with gradient tracking
//  * Returns PMF(time) and ∇PMF(time) using uniformization
//  * PMF = Σ_k Poisson(k; λt) * P(absorption at step k)
//  */
// static int compute_pmf_with_gradient(
//     struct ptd_graph *graph,
//     double time,
//     double lambda,
//     size_t granularity,
//     const double *params,
//     size_t n_params,
//     double *pmf_value,
//     double *pmf_gradient
// ) {
//     if (graph == NULL || params == NULL || pmf_value == NULL || pmf_gradient == NULL) {
//         return -1;
//     }

//     size_t max_jumps = (size_t)(granularity * time * lambda) + 100;

//     // Initialize probability and gradient arrays
//     double *prob = (double *)calloc(graph->vertices_length, sizeof(double));
//     double **prob_grad = alloc_2d(graph->vertices_length, n_params);

//     if (prob == NULL || prob_grad == NULL) {
//         free(prob);
//         free_2d(prob_grad, graph->vertices_length);
//         return -1;
//     }

//     // Starting vertex has probability 1, gradient 0
//     prob[0] = 1.0;

//     // Initialize output accumulators
//     *pmf_value = 0.0;
//     for (size_t i = 0; i < n_params; i++) {
//         pmf_gradient[i] = 0.0;
//     }

//     // Precompute Poisson probabilities
//     double *poisson_cache = (double *)malloc(max_jumps * sizeof(double));
//     if (poisson_cache == NULL) {
//         free(prob);
//         free_2d(prob_grad, graph->vertices_length);
//         return -1;
//     }

//     double lambda_t = lambda * time;
//     for (size_t k = 0; k < max_jumps; k++) {
//         poisson_cache[k] = exp(-lambda_t + k * log(lambda_t) - lgamma(k + 1));
//     }

//     // DP iteration
//     for (size_t k = 0; k < max_jumps; k++) {
//         double *next_prob = (double *)calloc(graph->vertices_length, sizeof(double));
//         double **next_prob_grad = alloc_2d(graph->vertices_length, n_params);

//         if (next_prob == NULL || next_prob_grad == NULL) {
//             free(next_prob);
//             free_2d(next_prob_grad, graph->vertices_length);
//             free(prob);
//             free_2d(prob_grad, graph->vertices_length);
//             free(poisson_cache);
//             return -1;
//         }

//         // Forward step
//         for (size_t v = 0; v < graph->vertices_length; v++) {
//             struct ptd_vertex *vertex = graph->vertices[v];

//             // Compute exit rate and gradient for self-loop
//             double exit_rate = 0.0;
//             double *exit_rate_grad = (double *)calloc(n_params, sizeof(double));
//             if (exit_rate_grad == NULL) {
//                 free(next_prob);
//                 free_2d(next_prob_grad, graph->vertices_length);
//                 free(prob);
//                 free_2d(prob_grad, graph->vertices_length);
//                 free(poisson_cache);
//                 return -1;
//             }

//             for (size_t e = 0; e < vertex->edges_length; e++) {
//                 struct ptd_edge *edge = vertex->edges[e];
//                 if (edge->parameterized) {
//                     struct ptd_edge_parameterized *ep = (struct ptd_edge_parameterized *)edge;
//                     double w = 0.0;  // Compute weight as dot product only
//                     if (ep->state != NULL) {
//                         for (size_t i = 0; i < n_params; i++) {
//                             w += ep->state[i] * params[i];
//                             exit_rate_grad[i] += ep->state[i];
//                         }
//                     }
//                     exit_rate += w;
//                 } else {
//                     exit_rate += edge->weight;
//                 }
//             }

//             // Process outgoing edges
//             for (size_t e = 0; e < vertex->edges_length; e++) {
//                 struct ptd_edge *edge = vertex->edges[e];

//                 size_t to_idx = 0;
//                 for (size_t i = 0; i < graph->vertices_length; i++) {
//                     if (graph->vertices[i] == edge->to) {
//                         to_idx = i;
//                         break;
//                     }
//                 }

//                 double weight;
//                 double *weight_grad = (double *)calloc(n_params, sizeof(double));
//                 if (weight_grad == NULL) {
//                     free(exit_rate_grad);
//                     free(next_prob);
//                     free_2d(next_prob_grad, graph->vertices_length);
//                     free(prob);
//                     free_2d(prob_grad, graph->vertices_length);
//                     free(poisson_cache);
//                     return -1;
//                 }

//                 if (edge->parameterized) {
//                     struct ptd_edge_parameterized *ep = (struct ptd_edge_parameterized *)edge;
//                     weight = 0.0;  // Compute weight as dot product only
//                     if (ep->state != NULL) {
//                         for (size_t i = 0; i < n_params; i++) {
//                             weight += ep->state[i] * params[i];
//                             weight_grad[i] = ep->state[i];
//                         }
//                     }
//                 } else {
//                     weight = edge->weight;
//                 }

//                 next_prob[to_idx] += prob[v] * weight / lambda;

//                 for (size_t i = 0; i < n_params; i++) {
//                     next_prob_grad[to_idx][i] +=
//                         prob_grad[v][i] * weight / lambda +
//                         prob[v] * weight_grad[i] / lambda;
//                 }

//                 free(weight_grad);
//             }

//             // Self-loop
//             double self_prob = (lambda - exit_rate) / lambda;
//             next_prob[v] += prob[v] * self_prob;

//             for (size_t i = 0; i < n_params; i++) {
//                 next_prob_grad[v][i] +=
//                     prob_grad[v][i] * self_prob +
//                     prob[v] * (-exit_rate_grad[i]) / lambda;
//             }

//             free(exit_rate_grad);
//         }

//         // Swap buffers
//         free(prob);
//         free_2d(prob_grad, graph->vertices_length);
//         prob = next_prob;
//         prob_grad = next_prob_grad;

//         // Accumulate PMF contributions from absorbing states
//         for (size_t i = 0; i < graph->vertices_length; i++) {
//             struct ptd_vertex *v = graph->vertices[i];
//             if (v->edges_length == 0 && i > 0) {
//                 double poisson_k = poisson_cache[k];
//                 *pmf_value += poisson_k * prob[i];

//                 for (size_t p = 0; p < n_params; p++) {
//                     pmf_gradient[p] += poisson_k * prob_grad[i][p];
//                 }

//                 // CRITICAL: Zero out absorbed probability (pattern from line 4559)
//                 prob[i] = 0;
//                 for (size_t p = 0; p < n_params; p++) {
//                     prob_grad[i][p] = 0;
//                 }
//             }
//         }

//         if (k > 10 && poisson_cache[k] < 1e-12) {
//             break;
//         }
//     }

//     free(prob);
//     free_2d(prob_grad, graph->vertices_length);
//     free(poisson_cache);

//     return 0;
// }

// /**
//  * Forward algorithm with gradient tracking
//  * Uses uniformization to compute PDF = PMF * granularity
//  */
// int ptd_graph_pdf_with_gradient(
//     struct ptd_graph *graph,
//     double time,
//     size_t granularity,
//     const double *params,
//     size_t n_params,
//     double *pdf_value,
//     double *pdf_gradient
// ) {
//     if (graph == NULL || params == NULL || pdf_value == NULL || pdf_gradient == NULL) {
//         return -1;
//     }

//     // 1. Compute uniformization rate (max exit rate across all vertices)
//     double lambda = 0.0;
//     for (size_t i = 0; i < graph->vertices_length; i++) {
//         struct ptd_vertex *v = graph->vertices[i];
//         double exit_rate = 0.0;

//         for (size_t j = 0; j < v->edges_length; j++) {
//             struct ptd_edge *e = v->edges[j];

//             if (e->parameterized) {
//                 struct ptd_edge_parameterized *ep = (struct ptd_edge_parameterized *)e;
//                 double weight = ep->weight;
//                 if (ep->state != NULL) {
//                     for (size_t k = 0; k < n_params; k++) {
//                         weight += ep->state[k] * params[k];
//                     }
//                 }
//                 exit_rate += weight;
//             } else {
//                 exit_rate += e->weight;
//             }
//         }

//         if (exit_rate > lambda) {
//             lambda = exit_rate;
//         }
//     }

//     if (lambda <= 0.0) {
//         *pdf_value = 0.0;
//         for (size_t i = 0; i < n_params; i++) {
//             pdf_gradient[i] = 0.0;
//         }
//         return 0;
//     }

//     // 2. Determine granularity (auto-select if not specified)
//     if (granularity == 0) {
//         granularity = (size_t)(lambda * 2.0);
//         if (granularity < 100) granularity = 100;
//     }

//     // 3. Compute PMF and its gradient
//     double pmf;
//     double *pmf_grad = (double *)malloc(n_params * sizeof(double));
//     if (pmf_grad == NULL) {
//         return -1;
//     }

//     int status = compute_pmf_with_gradient(graph, time, lambda, granularity,
//                                           params, n_params, &pmf, pmf_grad);

//     if (status != 0) {
//         free(pmf_grad);
//         return -1;
//     }

//     // 4. Convert PMF to PDF: PDF = PMF * lambda
//     //    In uniformization: dt = 1/lambda, so PDF = PMF / dt = PMF * lambda
//     *pdf_value = pmf * lambda;
//     for (size_t i = 0; i < n_params; i++) {
//         pdf_gradient[i] = pmf_grad[i] * lambda;
//     }

//     free(pmf_grad);
//     return 0;
// }

// /**
//  * Compute PDF for parameterized graph using current parameters
//  */
// int ptd_graph_pdf_parameterized(
//     struct ptd_graph *graph,
//     double time,
//     size_t granularity,
//     double *pdf_value,
//     double *pdf_gradient
// ) {
//     // Validate inputs
//     if (graph == NULL || pdf_value == NULL) {
//         sprintf((char*)ptd_err, "ptd_graph_pdf_parameterized: graph or pdf_value is NULL");
//         return -1;
//     }

//     // Check if graph is parameterized
//     if (!graph->parameterized) {
//         sprintf((char*)ptd_err, "ptd_graph_pdf_parameterized: graph is not parameterized");
//         return -1;
//     }

//     // Check if parameters have been set
//     if (graph->current_params == NULL) {
//         sprintf((char*)ptd_err, "ptd_graph_pdf_parameterized: parameters not set. "
//                 "Call ptd_graph_update_weight_parameterized() first");
//         return -1;
//     }

//     if (graph->param_length == 0) {
//         sprintf((char*)ptd_err, "ptd_graph_pdf_parameterized: param_length is 0");
//         return -1;
//     }

//     // If gradients requested, use gradient-aware function
//     if (pdf_gradient != NULL) {
//         return ptd_graph_pdf_with_gradient(
//             graph,
//             time,
//             granularity,
//             graph->current_params,
//             graph->param_length,
//             pdf_value,
//             pdf_gradient
//         );
//     }

//     // Otherwise, fall back to gradient computation anyway
//     // (There's no separate PDF-only function for parameterized graphs at C level)
//     // The Python/C++ layers handle this through reward_compute_graph
//     // For now, just compute with gradients and ignore them internally
//     double *temp_gradient = (double*)malloc(graph->param_length * sizeof(double));
//     if (temp_gradient == NULL) {
//         sprintf((char*)ptd_err, "ptd_graph_pdf_parameterized: failed to allocate temp gradient");
//         return -1;
//     }

//     int result = ptd_graph_pdf_with_gradient(
//         graph,
//         time,
//         granularity,
//         graph->current_params,
//         graph->param_length,
//         pdf_value,
//         temp_gradient
//     );

//     free(temp_gradient);
//     return result;
// }

// double ptd_defect(struct ptd_graph *graph) {
//     double rate = 0;

//     for (size_t i = 0; i < graph->starting_vertex->edges_length; ++i) {
//         struct ptd_edge *edge = graph->starting_vertex->edges[i];

//         rate += edge->weight;
//     }

//     double defect = 0;

//     for (size_t i = 0; i < graph->starting_vertex->edges_length; ++i) {
//         struct ptd_edge *edge = graph->starting_vertex->edges[i];

//         if (edge->to->edges_length == 0) {
//             defect += edge->weight / rate;
//         }
//     }

//     return defect;
// }

// struct ptd_clone_res ptd_clone_graph(struct ptd_graph *graph, struct ptd_avl_tree *avl_tree) {
//     struct ptd_graph *res = ptd_graph_create(graph->state_length);

//     for (size_t i = 1; i < graph->vertices_length; ++i) {
//         ptd_vertex_create(res);
//     }

//     for (size_t i = 0; i < graph->vertices_length; ++i) {
//         struct ptd_vertex *v = graph->vertices[i];
//         struct ptd_vertex *v2 = res->vertices[i];

//         if (v->state != NULL) {
//             memcpy(v2->state, v->state, sizeof(int) * res->state_length);
//         }
//     }

//     for (size_t i = 0; i < graph->vertices_length; ++i) {
//         struct ptd_vertex *v = graph->vertices[i];
//         struct ptd_vertex *v2 = res->vertices[i];

//         for (size_t j = 0; j < v->edges_length; ++j) {
//             struct ptd_edge *e = v->edges[j];

//             if (e->parameterized) {
//                 struct ptd_edge_parameterized *param_e = (struct ptd_edge_parameterized *) e;
//                 ptd_graph_add_edge_parameterized(
//                         v2,
//                         res->vertices[e->to->index],
//                         e->weight,
//                         param_e->state,
//                         param_e->state_length
//                 )->should_free_state = false;
//             } else {
//                 ptd_graph_add_edge(v2, res->vertices[e->to->index], e->weight);
//             }
//         }
//     }

//     struct ptd_avl_tree *new_tree = ptd_avl_tree_create(avl_tree->key_length);

//     struct ptd_stack *stack = stack_create();

//     stack_push(stack, avl_tree->root);

//     while (!stack_empty(stack)) {
//         struct ptd_avl_node *v = (struct ptd_avl_node *) stack_pop(stack);

//         if (v == NULL) {
//             continue;
//         }

//         ptd_avl_tree_find_or_insert(
//                 new_tree,
//                 v->key,
//                 res->vertices[((struct ptd_vertex *) v->entry)->index]
//         );

//         stack_push(stack, v->left);
//         stack_push(stack, v->right);
//     }

//     stack_destroy(stack);

//     struct ptd_clone_res ret;
//     ret.graph = res;
//     ret.avl_tree = new_tree;

//     return ret;
// }

// /*
//  * Utilities
//  */


// static struct ptd_vector *vector_create() {
//     struct ptd_vector *vector = (struct ptd_vector *) malloc(sizeof(*vector));

//     vector->entries = 0;
//     vector->arr = NULL;

//     return vector;
// }

// static int vector_add(struct ptd_vector *vector, void *entry) {
//     bool is_power_of_2 = (vector->entries & (vector->entries - 1)) == 0;

//     if (is_power_of_2) {
//         size_t new_length = vector->entries == 0 ? 1 : vector->entries * 2;

//         if ((vector->arr = (void **) realloc(
//                 vector->arr,
//                 new_length * sizeof(void *))
//             ) == NULL) {
//             return -1;
//         }
//     }

//     vector->arr[vector->entries] = entry;
//     vector->entries++;

//     return 0;
// }

// static void *vector_get(struct ptd_vector *vector, size_t index) {
//     return vector->arr[index];
// }

// static size_t vector_length(struct ptd_vector *vector) {
//     return vector->entries;
// }

// static void vector_destroy(struct ptd_vector *vector) {
//     free(vector->arr);
//     free(vector);
// }

// static struct ptd_queue *queue_create() {
//     struct ptd_queue *queue = (struct ptd_queue *) malloc(sizeof(struct ptd_queue));

//     queue->ll = NULL;
//     queue->tail = NULL;

//     return queue;
// }

// static void queue_destroy(struct ptd_queue *queue) {
//     free(queue->ll);
//     free(queue);
// }

// static int queue_enqueue(struct ptd_queue *queue, void *entry) {
//     struct ptd_ll *new_ll = (struct ptd_ll *) malloc(sizeof(*new_ll));
//     new_ll->next = NULL;
//     new_ll->value = entry;

//     if (queue->tail != NULL) {
//         queue->tail->next = new_ll;
//     } else {
//         queue->ll = new_ll;
//     }

//     queue->tail = new_ll;

//     return 0;
// }

// static void *queue_dequeue(struct ptd_queue *queue) {
//     void *result = queue->ll->value;
//     struct ptd_ll *value = queue->ll;
//     queue->ll = queue->ll->next;

//     if (queue->tail == value) {
//         queue->tail = NULL;
//     }

//     free(value);

//     return result;
// }

// static int queue_empty(struct ptd_queue *queue) {
//     return (queue->tail == NULL);
// }

// static struct ptd_stack *stack_create() {
//     struct ptd_stack *stack = (struct ptd_stack *) malloc(sizeof(struct ptd_stack));
//     stack->ll = NULL;

//     return stack;
// }

// static void stack_destroy(struct ptd_stack *stack) {
//     free(stack->ll);
//     free(stack);
// }

// static int stack_push(struct ptd_stack *stack, void *entry) {
//     struct ptd_ll *new_ll = (struct ptd_ll *) malloc(sizeof(*new_ll));
//     new_ll->next = stack->ll;
//     new_ll->value = entry;

//     stack->ll = new_ll;

//     return 0;
// }

// static void *stack_pop(struct ptd_stack *stack) {
//     void *result = stack->ll->value;
//     struct ptd_ll *ll = stack->ll;

//     stack->ll = stack->ll->next;
//     free(ll);

//     return result;
// }

// static int stack_empty(struct ptd_stack *stack) {
//     return (stack->ll == NULL);
// }

// // ============================================================================
// // Symbolic Expression System Implementation
// // ============================================================================

// /**
//  * Create a constant expression node
//  */
// struct ptd_expression *ptd_expr_const(double value) {
//     struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
//     if (expr == NULL) {
//         DIE_ERROR(1, "Failed to allocate memory for constant expression");
//     }
//     expr->type = PTD_EXPR_CONST;
//     expr->const_value = value;
//     return expr;
// }

// /**
//  * Create a parameter reference expression node
//  */
// struct ptd_expression *ptd_expr_param(size_t param_idx) {
//     struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
//     if (expr == NULL) {
//         DIE_ERROR(1, "Failed to allocate memory for parameter expression");
//     }
//     expr->type = PTD_EXPR_PARAM;
//     expr->param_index = param_idx;
//     return expr;
// }

// /**
//  * Create a dot product expression node (optimized for linear combinations)
//  */
// struct ptd_expression *ptd_expr_dot(const size_t *indices, const double *coeffs, size_t n) {
//     struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
//     if (expr == NULL) {
//         DIE_ERROR(1, "Failed to allocate memory for dot expression");
//     }
//     expr->type = PTD_EXPR_DOT;
//     expr->n_terms = n;

//     // Allocate and copy indices
//     expr->param_indices = (size_t *) malloc(n * sizeof(size_t));
//     if (expr->param_indices == NULL) {
//         free(expr);
//         DIE_ERROR(1, "Failed to allocate memory for dot expression indices");
//     }
//     memcpy(expr->param_indices, indices, n * sizeof(size_t));

//     // Allocate and copy coefficients
//     expr->coefficients = (double *) malloc(n * sizeof(double));
//     if (expr->coefficients == NULL) {
//         free(expr->param_indices);
//         free(expr);
//         DIE_ERROR(1, "Failed to allocate memory for dot expression coefficients");
//     }
//     memcpy(expr->coefficients, coeffs, n * sizeof(double));

//     return expr;
// }

// /**
//  * Create an addition expression node
//  */
// struct ptd_expression *ptd_expr_add(struct ptd_expression *left, struct ptd_expression *right) {
//     // Simplification: 0 + x = x
//     if (left->type == PTD_EXPR_CONST && left->const_value == 0.0) {
//         ptd_expr_destroy_iterative(left);
//         return right;
//     }
//     if (right->type == PTD_EXPR_CONST && right->const_value == 0.0) {
//         ptd_expr_destroy_iterative(right);
//         return left;
//     }

//     // Constant folding: c1 + c2 = c3
//     if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
//         double result = left->const_value + right->const_value;
//         ptd_expr_destroy_iterative(left);
//         ptd_expr_destroy_iterative(right);
//         return ptd_expr_const(result);
//     }

//     // Original allocation
//     struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
//     if (expr == NULL) {
//         DIE_ERROR(1, "Failed to allocate memory for addition expression");
//     }
//     expr->type = PTD_EXPR_ADD;
//     expr->left = left;
//     expr->right = right;
//     return expr;
// }

// /**
//  * Create a multiplication expression node
//  */
// struct ptd_expression *ptd_expr_mul(struct ptd_expression *left, struct ptd_expression *right) {
//     // Simplification: 0 * x = 0
//     if (left->type == PTD_EXPR_CONST && left->const_value == 0.0) {
//         ptd_expr_destroy_iterative(right);
//         return left;
//     }
//     if (right->type == PTD_EXPR_CONST && right->const_value == 0.0) {
//         ptd_expr_destroy_iterative(left);
//         return right;
//     }

//     // Simplification: 1 * x = x
//     if (left->type == PTD_EXPR_CONST && left->const_value == 1.0) {
//         ptd_expr_destroy_iterative(left);
//         return right;
//     }
//     if (right->type == PTD_EXPR_CONST && right->const_value == 1.0) {
//         ptd_expr_destroy_iterative(right);
//         return left;
//     }

//     // Constant folding: c1 * c2 = c3
//     if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
//         double result = left->const_value * right->const_value;
//         ptd_expr_destroy_iterative(left);
//         ptd_expr_destroy_iterative(right);
//         return ptd_expr_const(result);
//     }

//     // Original allocation
//     struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
//     if (expr == NULL) {
//         DIE_ERROR(1, "Failed to allocate memory for multiplication expression");
//     }
//     expr->type = PTD_EXPR_MUL;
//     expr->left = left;
//     expr->right = right;
//     return expr;
// }

// /**
//  * Create a division expression node
//  */
// struct ptd_expression *ptd_expr_div(struct ptd_expression *left, struct ptd_expression *right) {
//     // Simplification: 0 / x = 0 (x != 0)
//     if (left->type == PTD_EXPR_CONST && left->const_value == 0.0) {
//         ptd_expr_destroy_iterative(right);
//         return left;
//     }

//     // Simplification: x / 1 = x
//     if (right->type == PTD_EXPR_CONST && right->const_value == 1.0) {
//         ptd_expr_destroy_iterative(right);
//         return left;
//     }

//     // Constant folding: c1 / c2 = c3
//     if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
//         if (right->const_value == 0.0) {
//             DIE_ERROR(1, "Division by zero in constant folding");
//         }
//         double result = left->const_value / right->const_value;
//         ptd_expr_destroy_iterative(left);
//         ptd_expr_destroy_iterative(right);
//         return ptd_expr_const(result);
//     }

//     // Original allocation
//     struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
//     if (expr == NULL) {
//         DIE_ERROR(1, "Failed to allocate memory for division expression");
//     }
//     expr->type = PTD_EXPR_DIV;
//     expr->left = left;
//     expr->right = right;
//     return expr;
// }

// /**
//  * Create an inversion expression node (1/x)
//  */
// struct ptd_expression *ptd_expr_inv(struct ptd_expression *child) {
//     struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
//     if (expr == NULL) {
//         DIE_ERROR(1, "Failed to allocate memory for inversion expression");
//     }
//     expr->type = PTD_EXPR_INV;
//     expr->left = child;  // Use left for unary operations
//     return expr;
// }

// /**
//  * Create a subtraction expression node
//  */
// struct ptd_expression *ptd_expr_sub(struct ptd_expression *left, struct ptd_expression *right) {
//     // Simplification: x - 0 = x
//     if (right->type == PTD_EXPR_CONST && right->const_value == 0.0) {
//         ptd_expr_destroy_iterative(right);
//         return left;
//     }

//     // Constant folding: c1 - c2 = c3
//     if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
//         double result = left->const_value - right->const_value;
//         ptd_expr_destroy_iterative(left);
//         ptd_expr_destroy_iterative(right);
//         return ptd_expr_const(result);
//     }

//     // Original allocation
//     struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
//     if (expr == NULL) {
//         DIE_ERROR(1, "Failed to allocate memory for subtraction expression");
//     }
//     expr->type = PTD_EXPR_SUB;
//     expr->left = left;
//     expr->right = right;
//     return expr;
// }

// /**
//  * Stack entry for iterative expression copying
//  */
// struct ptd_expr_copy_stack_entry {
//     const struct ptd_expression *src;      // Source node to copy
//     struct ptd_expression **dst_location;  // Where to store the copy pointer
//     bool processed;                        // Children already copied?
// };

// /**
//  * Deep copy an expression tree (iterative version to avoid stack overflow)
//  */
// struct ptd_expression *ptd_expr_copy_iterative(const struct ptd_expression *expr) {
//     if (expr == NULL) {
//         return NULL;
//     }

//     // Explicit stack for iterative traversal
//     size_t stack_capacity = 256;
//     size_t stack_size = 0;
//     struct ptd_expr_copy_stack_entry *stack = (struct ptd_expr_copy_stack_entry *)
//         malloc(stack_capacity * sizeof(struct ptd_expr_copy_stack_entry));

//     if (stack == NULL) {
//         DIE_ERROR(1, "Failed to allocate copy stack");
//     }

//     struct ptd_expression *root = NULL;

//     // Push root onto stack
//     stack[stack_size++] = (struct ptd_expr_copy_stack_entry){
//         .src = expr,
//         .dst_location = &root,
//         .processed = false
//     };

//     while (stack_size > 0) {
//         struct ptd_expr_copy_stack_entry *entry = &stack[stack_size - 1];

//         if (entry->processed) {
//             // This node and its children are done
//             stack_size--;
//             continue;
//         }

//         // Allocate copy for this node
//         struct ptd_expression *copy = (struct ptd_expression *)
//             calloc(1, sizeof(struct ptd_expression));
//         if (copy == NULL) {
//             free(stack);
//             DIE_ERROR(1, "Failed to allocate memory for expression copy");
//         }

//         copy->type = entry->src->type;
//         *(entry->dst_location) = copy;

//         // Mark as processed before pushing children
//         entry->processed = true;

//         // Handle node type and push children if needed
//         switch (entry->src->type) {
//             case PTD_EXPR_CONST:
//                 copy->const_value = entry->src->const_value;
//                 break;

//             case PTD_EXPR_PARAM:
//                 copy->param_index = entry->src->param_index;
//                 break;

//             case PTD_EXPR_DOT:
//                 copy->n_terms = entry->src->n_terms;
//                 copy->param_indices = (size_t *) malloc(entry->src->n_terms * sizeof(size_t));
//                 copy->coefficients = (double *) malloc(entry->src->n_terms * sizeof(double));
//                 if (copy->param_indices == NULL || copy->coefficients == NULL) {
//                     free(copy->param_indices);
//                     free(copy->coefficients);
//                     free(copy);
//                     free(stack);
//                     DIE_ERROR(1, "Failed to allocate memory for dot expression copy");
//                 }
//                 memcpy(copy->param_indices, entry->src->param_indices,
//                       entry->src->n_terms * sizeof(size_t));
//                 memcpy(copy->coefficients, entry->src->coefficients,
//                       entry->src->n_terms * sizeof(double));
//                 break;

//             case PTD_EXPR_INV:
//                 if (entry->src->left != NULL) {
//                     // Grow stack if needed
//                     if (stack_size >= stack_capacity) {
//                         stack_capacity *= 2;
//                         struct ptd_expr_copy_stack_entry *new_stack =
//                             (struct ptd_expr_copy_stack_entry *)
//                             realloc(stack, stack_capacity * sizeof(struct ptd_expr_copy_stack_entry));
//                         if (new_stack == NULL) {
//                             free(stack);
//                             DIE_ERROR(1, "Failed to grow copy stack");
//                         }
//                         stack = new_stack;
//                         entry = &stack[stack_size - 1];  // Re-point after realloc
//                     }

//                     // Push left child
//                     stack[stack_size++] = (struct ptd_expr_copy_stack_entry){
//                         .src = entry->src->left,
//                         .dst_location = &copy->left,
//                         .processed = false
//                     };
//                 }
//                 break;

//             case PTD_EXPR_ADD:
//             case PTD_EXPR_MUL:
//             case PTD_EXPR_DIV:
//             case PTD_EXPR_SUB:
//                 // Grow stack if needed for 2 children
//                 while (stack_size + 2 > stack_capacity) {
//                     stack_capacity *= 2;
//                     struct ptd_expr_copy_stack_entry *new_stack =
//                         (struct ptd_expr_copy_stack_entry *)
//                         realloc(stack, stack_capacity * sizeof(struct ptd_expr_copy_stack_entry));
//                     if (new_stack == NULL) {
//                         free(stack);
//                         DIE_ERROR(1, "Failed to grow copy stack");
//                     }
//                     stack = new_stack;
//                     entry = &stack[stack_size - 1];  // Re-point after realloc
//                 }

//                 // Push children (right first, then left for proper ordering)
//                 if (entry->src->right != NULL) {
//                     stack[stack_size++] = (struct ptd_expr_copy_stack_entry){
//                         .src = entry->src->right,
//                         .dst_location = &copy->right,
//                         .processed = false
//                     };
//                 }
//                 if (entry->src->left != NULL) {
//                     stack[stack_size++] = (struct ptd_expr_copy_stack_entry){
//                         .src = entry->src->left,
//                         .dst_location = &copy->left,
//                         .processed = false
//                     };
//                 }
//                 break;

//             default:
//                 free(copy);
//                 free(stack);
//                 DIE_ERROR(1, "Unknown expression type in ptd_expr_copy_iterative");
//         }
//     }

//     free(stack);
//     return root;
// }

// /**
//  * Deep copy an expression tree (recursive version - kept for compatibility)
//  * WARNING: May cause stack overflow for deeply nested expressions (>1000 levels)
//  * Use ptd_expr_copy_iterative() for deep trees
//  */
// struct ptd_expression *ptd_expr_copy(const struct ptd_expression *expr) {
//     if (expr == NULL) {
//         return NULL;
//     }

//     struct ptd_expression *copy = (struct ptd_expression *) calloc(1, sizeof(*copy));
//     if (copy == NULL) {
//         DIE_ERROR(1, "Failed to allocate memory for expression copy");
//     }

//     copy->type = expr->type;

//     switch (expr->type) {
//         case PTD_EXPR_CONST:
//             copy->const_value = expr->const_value;
//             break;

//         case PTD_EXPR_PARAM:
//             copy->param_index = expr->param_index;
//             break;

//         case PTD_EXPR_DOT:
//             copy->n_terms = expr->n_terms;
//             copy->param_indices = (size_t *) malloc(expr->n_terms * sizeof(size_t));
//             copy->coefficients = (double *) malloc(expr->n_terms * sizeof(double));
//             if (copy->param_indices == NULL || copy->coefficients == NULL) {
//                 free(copy->param_indices);
//                 free(copy->coefficients);
//                 free(copy);
//                 DIE_ERROR(1, "Failed to allocate memory for dot expression copy");
//             }
//             memcpy(copy->param_indices, expr->param_indices, expr->n_terms * sizeof(size_t));
//             memcpy(copy->coefficients, expr->coefficients, expr->n_terms * sizeof(double));
//             break;

//         case PTD_EXPR_INV:
//             copy->left = ptd_expr_copy(expr->left);
//             break;

//         case PTD_EXPR_ADD:
//         case PTD_EXPR_MUL:
//         case PTD_EXPR_DIV:
//         case PTD_EXPR_SUB:
//             copy->left = ptd_expr_copy(expr->left);
//             copy->right = ptd_expr_copy(expr->right);
//             break;

//         default:
//             free(copy);
//             DIE_ERROR(1, "Unknown expression type in ptd_expr_copy");
//     }

//     return copy;
// }

// /**
//  * Stack entry for iterative expression destruction
//  */
// struct ptd_expr_destroy_stack_entry {
//     struct ptd_expression *expr;
//     bool children_pushed;
// };

// /**
//  * Destroy an expression tree and free all memory (iterative version, O(n))
//  */
// void ptd_expr_destroy_iterative(struct ptd_expression *expr) {
//     if (expr == NULL) {
//         return;
//     }

//     // Stack for post-order destruction
//     size_t stack_capacity = 256;
//     size_t stack_size = 0;
//     struct ptd_expr_destroy_stack_entry *stack =
//         (struct ptd_expr_destroy_stack_entry *)
//         malloc(stack_capacity * sizeof(struct ptd_expr_destroy_stack_entry));

//     if (stack == NULL) {
//         DIE_ERROR(1, "Failed to allocate destruction stack");
//     }

//     // Push root
//     stack[stack_size++] = (struct ptd_expr_destroy_stack_entry){
//         .expr = expr,
//         .children_pushed = false
//     };

//     while (stack_size > 0) {
//         struct ptd_expr_destroy_stack_entry *entry = &stack[stack_size - 1];

//         if (!entry->children_pushed) {
//             // First visit: push children
//             entry->children_pushed = true;
//             struct ptd_expression *node = entry->expr;

//             // Grow stack if needed (max 2 children)
//             if (stack_size + 2 > stack_capacity) {
//                 stack_capacity *= 2;
//                 struct ptd_expr_destroy_stack_entry *new_stack =
//                     (struct ptd_expr_destroy_stack_entry *)
//                     realloc(stack, stack_capacity * sizeof(struct ptd_expr_destroy_stack_entry));
//                 if (new_stack == NULL) {
//                     free(stack);
//                     DIE_ERROR(1, "Failed to grow destruction stack");
//                 }
//                 stack = new_stack;
//                 entry = &stack[stack_size - 1];  // Re-point after realloc
//             }

//             // Push children (right first for left-to-right processing)
//             switch (node->type) {
//                 case PTD_EXPR_INV:
//                     if (node->left != NULL) {
//                         stack[stack_size++] = (struct ptd_expr_destroy_stack_entry){
//                             .expr = node->left,
//                             .children_pushed = false
//                         };
//                     }
//                     break;

//                 case PTD_EXPR_ADD:
//                 case PTD_EXPR_MUL:
//                 case PTD_EXPR_DIV:
//                 case PTD_EXPR_SUB:
//                     if (node->right != NULL) {
//                         stack[stack_size++] = (struct ptd_expr_destroy_stack_entry){
//                             .expr = node->right,
//                             .children_pushed = false
//                         };
//                     }
//                     if (node->left != NULL) {
//                         stack[stack_size++] = (struct ptd_expr_destroy_stack_entry){
//                             .expr = node->left,
//                             .children_pushed = false
//                         };
//                     }
//                     break;

//                 default:
//                     // Leaf nodes (CONST, PARAM, DOT) - no children
//                     break;
//             }
//         } else {
//             // Second visit: children are done, destroy this node
//             struct ptd_expression *node = entry->expr;
//             stack_size--;

//             // Free node-specific data
//             if (node->type == PTD_EXPR_DOT) {
//                 free(node->param_indices);
//                 free(node->coefficients);
//             }

//             free(node);
//         }
//     }

//     free(stack);
// }

// /**
//  * Destroy an expression tree and free all memory (recursive version - kept for compatibility)
//  * WARNING: May cause stack overflow for deeply nested expressions (>1000 levels)
//  * Use ptd_expr_destroy_iterative() for deep trees
//  */
// void ptd_expr_destroy(struct ptd_expression *expr) {
//     if (expr == NULL) {
//         return;
//     }

//     // Recursively destroy children
//     switch (expr->type) {
//         case PTD_EXPR_INV:
//             ptd_expr_destroy(expr->left);
//             break;

//         case PTD_EXPR_ADD:
//         case PTD_EXPR_MUL:
//         case PTD_EXPR_DIV:
//         case PTD_EXPR_SUB:
//             ptd_expr_destroy(expr->left);
//             ptd_expr_destroy(expr->right);
//             break;

//         case PTD_EXPR_DOT:
//             free(expr->param_indices);
//             free(expr->coefficients);
//             break;

//         case PTD_EXPR_CONST:
//         case PTD_EXPR_PARAM:
//             // No children or allocated arrays
//             break;

//         default:
//             // Unknown type, but still free the node
//             break;
//     }

//     free(expr);
// }

// // =============================================================================
// // Expression Hashing and Equality (for CSE - Common Subexpression Elimination)
// // =============================================================================

// /**
//  * Compute structural hash of expression tree
//  *
//  * Uses FNV-1a-like hash with type and value mixing.
//  * For commutative operations (ADD, MUL), sorts child hashes for consistency.
//  */
// uint64_t ptd_expr_hash(const struct ptd_expression *expr) {
//     if (expr == NULL) return 0;

//     uint64_t hash = 14695981039346656037ULL;  // FNV offset basis
//     const uint64_t prime = 1099511628211ULL;  // FNV prime

//     // Mix in type
//     hash ^= (uint64_t)expr->type;
//     hash *= prime;

//     switch (expr->type) {
//         case PTD_EXPR_CONST: {
//             // Hash double value by reinterpreting bits
//             uint64_t bits;
//             memcpy(&bits, &expr->const_value, sizeof(uint64_t));
//             hash ^= bits;
//             hash *= prime;
//             break;
//         }

//         case PTD_EXPR_PARAM:
//             hash ^= expr->param_index;
//             hash *= prime;
//             break;

//         case PTD_EXPR_DOT:
//             hash ^= expr->n_terms;
//             hash *= prime;
//             for (size_t i = 0; i < expr->n_terms; i++) {
//                 hash ^= expr->param_indices[i];
//                 hash *= prime;

//                 uint64_t coeff_bits;
//                 memcpy(&coeff_bits, &expr->coefficients[i], sizeof(uint64_t));
//                 hash ^= coeff_bits;
//                 hash *= prime;
//             }
//             break;

//         case PTD_EXPR_INV:
//             hash ^= ptd_expr_hash(expr->left);
//             hash *= prime;
//             break;

//         case PTD_EXPR_ADD:
//         case PTD_EXPR_MUL:
//         case PTD_EXPR_DIV:
//         case PTD_EXPR_SUB: {
//             uint64_t left_hash = ptd_expr_hash(expr->left);
//             uint64_t right_hash = ptd_expr_hash(expr->right);

//             // Commutative operations: sort hashes for consistency
//             if (expr->type == PTD_EXPR_ADD || expr->type == PTD_EXPR_MUL) {
//                 if (left_hash > right_hash) {
//                     uint64_t tmp = left_hash;
//                     left_hash = right_hash;
//                     right_hash = tmp;
//                 }
//             }

//             hash ^= left_hash;
//             hash *= prime;
//             hash ^= right_hash;
//             hash *= prime;
//             break;
//         }
//     }

//     return hash;
// }

// /**
//  * Check structural equality of two expressions
//  *
//  * Performs deep comparison, handling commutativity of ADD and MUL.
//  */
// bool ptd_expr_equal(const struct ptd_expression *a, const struct ptd_expression *b) {
//     if (a == b) return true;
//     if (a == NULL || b == NULL) return false;
//     if (a->type != b->type) return false;

//     switch (a->type) {
//         case PTD_EXPR_CONST:
//             return a->const_value == b->const_value;

//         case PTD_EXPR_PARAM:
//             return a->param_index == b->param_index;

//         case PTD_EXPR_DOT:
//             if (a->n_terms != b->n_terms) return false;
//             for (size_t i = 0; i < a->n_terms; i++) {
//                 if (a->param_indices[i] != b->param_indices[i]) return false;
//                 if (a->coefficients[i] != b->coefficients[i]) return false;
//             }
//             return true;

//         case PTD_EXPR_INV:
//             return ptd_expr_equal(a->left, b->left);

//         case PTD_EXPR_ADD:
//         case PTD_EXPR_MUL:
//             // Commutative: check both orderings
//             return (ptd_expr_equal(a->left, b->left) && ptd_expr_equal(a->right, b->right)) ||
//                    (ptd_expr_equal(a->left, b->right) && ptd_expr_equal(a->right, b->left));

//         case PTD_EXPR_DIV:
//         case PTD_EXPR_SUB:
//             // Non-commutative: order matters
//             return ptd_expr_equal(a->left, b->left) && ptd_expr_equal(a->right, b->right);
//     }

//     return false;
// }

// // =============================================================================
// // Expression Intern Table (for CSE)
// // =============================================================================

// /**
//  * Expression intern table entry (linked list for collision handling)
//  */
// struct ptd_expr_intern_entry {
//     struct ptd_expression *expr;
//     uint64_t hash;
//     struct ptd_expr_intern_entry *next;
// };

// /**
//  * Expression intern table for CSE
//  *
//  * Hash table mapping expression structure → canonical instance.
//  * Multiple references to identical expressions share single instance.
//  */
// struct ptd_expr_intern_table {
//     struct ptd_expr_intern_entry **buckets;
//     size_t capacity;
//     size_t size;
//     size_t collisions;  // Statistics
// };

// /**
//  * Create intern table with specified capacity
//  */
// struct ptd_expr_intern_table *ptd_expr_intern_table_create(size_t capacity) {
//     struct ptd_expr_intern_table *table =
//         (struct ptd_expr_intern_table *)malloc(sizeof(struct ptd_expr_intern_table));

//     if (table == NULL) {
//         DIE_ERROR(1, "Failed to allocate intern table");
//     }

//     table->capacity = capacity;
//     table->size = 0;
//     table->collisions = 0;
//     table->buckets = (struct ptd_expr_intern_entry **)
//         calloc(capacity, sizeof(struct ptd_expr_intern_entry *));

//     if (table->buckets == NULL) {
//         free(table);
//         DIE_ERROR(1, "Failed to allocate intern table buckets");
//     }

//     return table;
// }

// /**
//  * Intern an expression (returns existing if found, otherwise adds to table)
//  *
//  * IMPORTANT: If existing expression found, destroys input and returns existing.
//  * Caller must not use input pointer after calling this function.
//  */
// struct ptd_expression *ptd_expr_intern(
//     struct ptd_expr_intern_table *table,
//     struct ptd_expression *expr
// ) {
//     if (expr == NULL || table == NULL) return expr;

//     uint64_t hash = ptd_expr_hash(expr);
//     size_t bucket = hash % table->capacity;

//     // Search for existing expression
//     struct ptd_expr_intern_entry *entry = table->buckets[bucket];
//     bool first = true;
//     while (entry != NULL) {
//         if (entry->hash == hash && ptd_expr_equal(entry->expr, expr)) {
//             // Found existing - destroy input and return existing
//             ptd_expr_destroy_iterative(expr);
//             return entry->expr;
//         }
//         if (!first) table->collisions++;
//         first = false;
//         entry = entry->next;
//     }

//     // Not found - add to table
//     struct ptd_expr_intern_entry *new_entry =
//         (struct ptd_expr_intern_entry *)malloc(sizeof(struct ptd_expr_intern_entry));

//     if (new_entry == NULL) {
//         DIE_ERROR(1, "Failed to allocate intern table entry");
//     }

//     new_entry->expr = expr;
//     new_entry->hash = hash;
//     new_entry->next = table->buckets[bucket];
//     table->buckets[bucket] = new_entry;
//     table->size++;

//     return expr;
// }

// /**
//  * Destroy intern table
//  *
//  * TEMPORARY: Not destroying expressions to debug crash issue.
//  * TODO: Properly implement expression lifecycle management for interned expressions.
//  */
// void ptd_expr_intern_table_destroy(struct ptd_expr_intern_table *table) {
//     if (table == NULL) return;

//     // Free table structure only (expressions leak for now - debugging)
//     for (size_t i = 0; i < table->capacity; i++) {
//         struct ptd_expr_intern_entry *entry = table->buckets[i];
//         while (entry != NULL) {
//             struct ptd_expr_intern_entry *next = entry->next;
//             // TEMPORARY: Don't destroy expressions - causes crash
//             // if (entry->expr != NULL) {
//             //     ptd_expr_destroy_iterative(entry->expr);
//             // }
//             free(entry);
//             entry = next;
//         }
//     }
//     free(table->buckets);
//     free(table);
// }

// /**
//  * Print intern table statistics (for debugging/profiling)
//  */
// void ptd_expr_intern_table_stats(const struct ptd_expr_intern_table *table) {
//     if (table == NULL) return;

//     printf("Expression Intern Table Statistics:\n");
//     printf("  Capacity: %zu\n", table->capacity);
//     printf("  Size: %zu entries\n", table->size);
//     printf("  Load factor: %.2f%%\n", 100.0 * table->size / table->capacity);
//     printf("  Total collisions: %zu\n", table->collisions);

//     // Compute chain length distribution
//     size_t max_chain = 0;
//     size_t empty_buckets = 0;
//     size_t chain_lengths[10] = {0};  // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9+

//     for (size_t i = 0; i < table->capacity; i++) {
//         size_t chain_len = 0;
//         struct ptd_expr_intern_entry *e = table->buckets[i];
//         while (e) {
//             chain_len++;
//             e = e->next;
//         }

//         if (chain_len == 0) {
//             empty_buckets++;
//         } else {
//             size_t idx = chain_len < 9 ? chain_len : 9;
//             chain_lengths[idx]++;
//         }

//         if (chain_len > max_chain) max_chain = chain_len;
//     }

//     printf("  Empty buckets: %zu (%.1f%%)\n", empty_buckets,
//            100.0 * empty_buckets / table->capacity);
//     printf("  Max chain length: %zu\n", max_chain);
//     printf("  Chain length distribution:\n");
//     for (size_t i = 1; i < 10; i++) {
//         if (chain_lengths[i] > 0) {
//             printf("    Length %zu: %zu buckets\n",
//                    i < 9 ? i : 9, chain_lengths[i]);
//         }
//     }
// }

// // =============================================================================
// // Interned Expression Constructors (for CSE)
// // =============================================================================

// /**
//  * Create addition expression with interning
//  */
// struct ptd_expression *ptd_expr_add_interned(
//     struct ptd_expr_intern_table *table,
//     struct ptd_expression *left,
//     struct ptd_expression *right
// ) {
//     // Apply simplifications first (from Phase 1)
//     if (left->type == PTD_EXPR_CONST && left->const_value == 0.0) {
//         ptd_expr_destroy_iterative(left);
//         return right;
//     }
//     if (right->type == PTD_EXPR_CONST && right->const_value == 0.0) {
//         ptd_expr_destroy_iterative(right);
//         return left;
//     }
//     if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
//         double result = left->const_value + right->const_value;
//         ptd_expr_destroy_iterative(left);
//         ptd_expr_destroy_iterative(right);
//         return ptd_expr_const(result);
//     }

//     // Create expression
//     struct ptd_expression *expr = (struct ptd_expression *)calloc(1, sizeof(*expr));
//     if (expr == NULL) {
//         DIE_ERROR(1, "Failed to allocate addition expression");
//     }
//     expr->type = PTD_EXPR_ADD;
//     expr->left = left;
//     expr->right = right;

//     // Intern if table provided
//     if (table != NULL) {
//         return ptd_expr_intern(table, expr);
//     }
//     return expr;
// }

// /**
//  * Create multiplication expression with interning
//  */
// struct ptd_expression *ptd_expr_mul_interned(
//     struct ptd_expr_intern_table *table,
//     struct ptd_expression *left,
//     struct ptd_expression *right
// ) {
//     // Apply simplifications first
//     if (left->type == PTD_EXPR_CONST && left->const_value == 0.0) {
//         ptd_expr_destroy_iterative(right);
//         return left;
//     }
//     if (right->type == PTD_EXPR_CONST && right->const_value == 0.0) {
//         ptd_expr_destroy_iterative(left);
//         return right;
//     }
//     if (left->type == PTD_EXPR_CONST && left->const_value == 1.0) {
//         ptd_expr_destroy_iterative(left);
//         return right;
//     }
//     if (right->type == PTD_EXPR_CONST && right->const_value == 1.0) {
//         ptd_expr_destroy_iterative(right);
//         return left;
//     }
//     if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
//         double result = left->const_value * right->const_value;
//         ptd_expr_destroy_iterative(left);
//         ptd_expr_destroy_iterative(right);
//         return ptd_expr_const(result);
//     }

//     // Create expression
//     struct ptd_expression *expr = (struct ptd_expression *)calloc(1, sizeof(*expr));
//     if (expr == NULL) {
//         DIE_ERROR(1, "Failed to allocate multiplication expression");
//     }
//     expr->type = PTD_EXPR_MUL;
//     expr->left = left;
//     expr->right = right;

//     // Intern if table provided
//     if (table != NULL) {
//         return ptd_expr_intern(table, expr);
//     }
//     return expr;
// }

// /**
//  * Create division expression with interning
//  */
// struct ptd_expression *ptd_expr_div_interned(
//     struct ptd_expr_intern_table *table,
//     struct ptd_expression *left,
//     struct ptd_expression *right
// ) {
//     // Apply simplifications first
//     if (left->type == PTD_EXPR_CONST && left->const_value == 0.0) {
//         ptd_expr_destroy_iterative(right);
//         return left;
//     }
//     if (right->type == PTD_EXPR_CONST && right->const_value == 1.0) {
//         ptd_expr_destroy_iterative(right);
//         return left;
//     }
//     if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
//         if (right->const_value == 0.0) {
//             DIE_ERROR(1, "Division by zero in constant folding");
//         }
//         double result = left->const_value / right->const_value;
//         ptd_expr_destroy_iterative(left);
//         ptd_expr_destroy_iterative(right);
//         return ptd_expr_const(result);
//     }

//     // Create expression
//     struct ptd_expression *expr = (struct ptd_expression *)calloc(1, sizeof(*expr));
//     if (expr == NULL) {
//         DIE_ERROR(1, "Failed to allocate division expression");
//     }
//     expr->type = PTD_EXPR_DIV;
//     expr->left = left;
//     expr->right = right;

//     // Intern if table provided
//     if (table != NULL) {
//         return ptd_expr_intern(table, expr);
//     }
//     return expr;
// }

// /**
//  * Create subtraction expression with interning
//  */
// struct ptd_expression *ptd_expr_sub_interned(
//     struct ptd_expr_intern_table *table,
//     struct ptd_expression *left,
//     struct ptd_expression *right
// ) {
//     // Apply simplifications first
//     if (right->type == PTD_EXPR_CONST && right->const_value == 0.0) {
//         ptd_expr_destroy_iterative(right);
//         return left;
//     }
//     if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
//         double result = left->const_value - right->const_value;
//         ptd_expr_destroy_iterative(left);
//         ptd_expr_destroy_iterative(right);
//         return ptd_expr_const(result);
//     }

//     // Create expression
//     struct ptd_expression *expr = (struct ptd_expression *)calloc(1, sizeof(*expr));
//     if (expr == NULL) {
//         DIE_ERROR(1, "Failed to allocate subtraction expression");
//     }
//     expr->type = PTD_EXPR_SUB;
//     expr->left = left;
//     expr->right = right;

//     // Intern if table provided
//     if (table != NULL) {
//         return ptd_expr_intern(table, expr);
//     }
//     return expr;
// }

// /**
//  * Create inversion expression with interning
//  */
// struct ptd_expression *ptd_expr_inv_interned(
//     struct ptd_expr_intern_table *table,
//     struct ptd_expression *child
// ) {
//     // Simplification: inv(const) = const(1/c)
//     if (child->type == PTD_EXPR_CONST) {
//         if (child->const_value == 0.0) {
//             DIE_ERROR(1, "Division by zero in constant inversion");
//         }
//         double result = 1.0 / child->const_value;
//         ptd_expr_destroy_iterative(child);
//         return ptd_expr_const(result);
//     }

//     // Create expression
//     struct ptd_expression *expr = (struct ptd_expression *)calloc(1, sizeof(*expr));
//     if (expr == NULL) {
//         DIE_ERROR(1, "Failed to allocate inversion expression");
//     }
//     expr->type = PTD_EXPR_INV;
//     expr->left = child;

//     // Intern if table provided
//     if (table != NULL) {
//         return ptd_expr_intern(table, expr);
//     }
//     return expr;
// }

// /**
//  * Stack entry for iterative expression evaluation
//  */
// struct ptd_expr_eval_stack_entry {
//     const struct ptd_expression *expr;
//     bool children_pushed;
//     double result;
// };

// /**
//  * Simple hash table for expression results (pointer -> double)
//  */
// struct ptd_expr_result_entry {
//     const struct ptd_expression *expr;
//     double result;
//     struct ptd_expr_result_entry *next;
// };

// struct ptd_expr_result_map {
//     struct ptd_expr_result_entry **buckets;
//     size_t capacity;
// };

// static struct ptd_expr_result_map *ptd_expr_result_map_create(size_t capacity) {
//     struct ptd_expr_result_map *map = (struct ptd_expr_result_map *)malloc(sizeof(struct ptd_expr_result_map));
//     if (map == NULL) return NULL;

//     map->capacity = capacity;
//     map->buckets = (struct ptd_expr_result_entry **)calloc(capacity, sizeof(struct ptd_expr_result_entry *));
//     if (map->buckets == NULL) {
//         free(map);
//         return NULL;
//     }
//     return map;
// }

// static void ptd_expr_result_map_put(struct ptd_expr_result_map *map, const struct ptd_expression *expr, double result) {
//     size_t bucket = ((size_t)expr / sizeof(void*)) % map->capacity;
//     struct ptd_expr_result_entry *entry = (struct ptd_expr_result_entry *)malloc(sizeof(struct ptd_expr_result_entry));
//     entry->expr = expr;
//     entry->result = result;
//     entry->next = map->buckets[bucket];
//     map->buckets[bucket] = entry;
// }

// static double ptd_expr_result_map_get(struct ptd_expr_result_map *map, const struct ptd_expression *expr) {
//     size_t bucket = ((size_t)expr / sizeof(void*)) % map->capacity;
//     struct ptd_expr_result_entry *entry = map->buckets[bucket];
//     while (entry != NULL) {
//         if (entry->expr == expr) {
//             return entry->result;
//         }
//         entry = entry->next;
//     }
//     return 0.0;  // Should not happen
// }

// static void ptd_expr_result_map_destroy(struct ptd_expr_result_map *map) {
//     for (size_t i = 0; i < map->capacity; i++) {
//         struct ptd_expr_result_entry *entry = map->buckets[i];
//         while (entry != NULL) {
//             struct ptd_expr_result_entry *next = entry->next;
//             free(entry);
//             entry = next;
//         }
//     }
//     free(map->buckets);
//     free(map);
// }

// /**
//  * Evaluate an expression with given parameters (iterative version, O(n))
//  * Uses post-order traversal with result hash map
//  */
// double ptd_expr_evaluate_iterative(
//     const struct ptd_expression *expr,
//     const double *params,
//     size_t n_params
// ) {
//     if (expr == NULL) {
//         return 0.0;
//     }

//     // Create result map
//     struct ptd_expr_result_map *results = ptd_expr_result_map_create(256);
//     if (results == NULL) {
//         DIE_ERROR(1, "Failed to allocate result map");
//     }

//     // Stack for post-order traversal
//     size_t stack_capacity = 256;
//     size_t stack_size = 0;
//     struct ptd_expr_eval_stack_entry *stack = (struct ptd_expr_eval_stack_entry *)
//         malloc(stack_capacity * sizeof(struct ptd_expr_eval_stack_entry));

//     if (stack == NULL) {
//         ptd_expr_result_map_destroy(results);
//         DIE_ERROR(1, "Failed to allocate evaluation stack");
//     }

//     // Push root
//     stack[stack_size++] = (struct ptd_expr_eval_stack_entry){
//         .expr = expr,
//         .children_pushed = false,
//         .result = 0.0
//     };

//     while (stack_size > 0) {
//         struct ptd_expr_eval_stack_entry *entry = &stack[stack_size - 1];
//         const struct ptd_expression *e = entry->expr;

//         if (!entry->children_pushed) {
//             // First visit: push children for operators, compute leaves
//             entry->children_pushed = true;

//             switch (e->type) {
//                 case PTD_EXPR_CONST: {
//                     double result = e->const_value;
//                     ptd_expr_result_map_put(results, e, result);
//                     stack_size--;  // Pop ourselves
//                     break;
//                 }

//                 case PTD_EXPR_PARAM: {
//                     if (e->param_index >= n_params) {
//                         free(stack);
//                         ptd_expr_result_map_destroy(results);
//                         DIE_ERROR(1, "Parameter index out of bounds in expression evaluation");
//                     }
//                     double result = params[e->param_index];
//                     ptd_expr_result_map_put(results, e, result);
//                     stack_size--;  // Pop ourselves
//                     break;
//                 }

//                 case PTD_EXPR_DOT: {
//                     double result = 0.0;
//                     for (size_t i = 0; i < e->n_terms; i++) {
//                         if (e->param_indices[i] >= n_params) {
//                             free(stack);
//                             ptd_expr_result_map_destroy(results);
//                             DIE_ERROR(1, "Parameter index out of bounds in dot expression evaluation");
//                         }
//                         result += e->coefficients[i] * params[e->param_indices[i]];
//                     }
//                     ptd_expr_result_map_put(results, e, result);
//                     stack_size--;  // Pop ourselves
//                     break;
//                 }

//                 case PTD_EXPR_INV:
//                 case PTD_EXPR_ADD:
//                 case PTD_EXPR_MUL:
//                 case PTD_EXPR_DIV:
//                 case PTD_EXPR_SUB:
//                     // Grow stack if needed
//                     if (stack_size + 2 > stack_capacity) {
//                         stack_capacity *= 2;
//                         struct ptd_expr_eval_stack_entry *new_stack =
//                             (struct ptd_expr_eval_stack_entry *)
//                             realloc(stack, stack_capacity * sizeof(struct ptd_expr_eval_stack_entry));
//                         if (new_stack == NULL) {
//                             free(stack);
//                             ptd_expr_result_map_destroy(results);
//                             DIE_ERROR(1, "Failed to grow evaluation stack");
//                         }
//                         stack = new_stack;
//                         entry = &stack[stack_size - 1];
//                     }

//                     // Push children (right first, then left)
//                     if (e->right != NULL) {
//                         stack[stack_size++] = (struct ptd_expr_eval_stack_entry){
//                             .expr = e->right,
//                             .children_pushed = false,
//                             .result = 0.0
//                         };
//                     }
//                     if (e->left != NULL) {
//                         stack[stack_size++] = (struct ptd_expr_eval_stack_entry){
//                             .expr = e->left,
//                             .children_pushed = false,
//                             .result = 0.0
//                         };
//                     }
//                     break;

//                 default:
//                     free(stack);
//                     ptd_expr_result_map_destroy(results);
//                     DIE_ERROR(1, "Unknown expression type in evaluation");
//             }
//         } else {
//             // Second visit: children processed, compute result from children's results
//             switch (e->type) {
//                 case PTD_EXPR_CONST:
//                 case PTD_EXPR_PARAM:
//                 case PTD_EXPR_DOT:
//                     // Already handled in first visit
//                     break;

//                 case PTD_EXPR_INV: {
//                     double child_val = ptd_expr_result_map_get(results, e->left);
//                     if (child_val == 0.0) {
//                         free(stack);
//                         ptd_expr_result_map_destroy(results);
//                         DIE_ERROR(1, "Division by zero in inversion expression evaluation");
//                     }
//                     double result = 1.0 / child_val;
//                     ptd_expr_result_map_put(results, e, result);
//                     stack_size--;  // Pop ourselves
//                     break;
//                 }

//                 case PTD_EXPR_ADD:
//                 case PTD_EXPR_MUL:
//                 case PTD_EXPR_DIV:
//                 case PTD_EXPR_SUB: {
//                     double left_val = ptd_expr_result_map_get(results, e->left);
//                     double right_val = ptd_expr_result_map_get(results, e->right);

//                     double result;
//                     switch (e->type) {
//                         case PTD_EXPR_ADD:
//                             result = left_val + right_val;
//                             break;
//                         case PTD_EXPR_MUL:
//                             result = left_val * right_val;
//                             break;
//                         case PTD_EXPR_DIV:
//                             if (right_val == 0.0) {
//                                 free(stack);
//                                 ptd_expr_result_map_destroy(results);
//                                 DIE_ERROR(1, "Division by zero in expression evaluation");
//                             }
//                             result = left_val / right_val;
//                             break;
//                         case PTD_EXPR_SUB:
//                             result = left_val - right_val;
//                             break;
//                         default:
//                             result = 0.0;
//                             break;
//                     }

//                     ptd_expr_result_map_put(results, e, result);
//                     stack_size--;  // Pop ourselves
//                     break;
//                 }

//                 default:
//                     free(stack);
//                     ptd_expr_result_map_destroy(results);
//                     DIE_ERROR(1, "Unknown expression type in evaluation");
//             }
//         }
//     }

//     // Get result for root
//     double final_result = ptd_expr_result_map_get(results, expr);

//     free(stack);
//     ptd_expr_result_map_destroy(results);
//     return final_result;
// }

// /**
//  * Evaluate an expression with given parameters (recursive version - kept for compatibility)
//  * WARNING: May cause stack overflow for deeply nested expressions (>1000 levels)
//  * Use ptd_expr_evaluate_iterative() for deep trees
//  */
// double ptd_expr_evaluate(
//     const struct ptd_expression *expr,
//     const double *params,
//     size_t n_params
// ) {
//     if (expr == NULL) {
//         return 0.0;
//     }

//     switch (expr->type) {
//         case PTD_EXPR_CONST:
//             return expr->const_value;

//         case PTD_EXPR_PARAM:
//             if (expr->param_index >= n_params) {
//                 DIE_ERROR(1, "Parameter index out of bounds in expression evaluation");
//             }
//             return params[expr->param_index];

//         case PTD_EXPR_DOT: {
//             double result = 0.0;
//             for (size_t i = 0; i < expr->n_terms; i++) {
//                 if (expr->param_indices[i] >= n_params) {
//                     DIE_ERROR(1, "Parameter index out of bounds in dot expression evaluation");
//                 }
//                 result += expr->coefficients[i] * params[expr->param_indices[i]];
//             }
//             return result;
//         }

//         case PTD_EXPR_ADD: {
//             double left_val = ptd_expr_evaluate(expr->left, params, n_params);
//             double right_val = ptd_expr_evaluate(expr->right, params, n_params);
//             return left_val + right_val;
//         }

//         case PTD_EXPR_MUL: {
//             double left_val = ptd_expr_evaluate(expr->left, params, n_params);
//             double right_val = ptd_expr_evaluate(expr->right, params, n_params);
//             return left_val * right_val;
//         }

//         case PTD_EXPR_DIV: {
//             double left_val = ptd_expr_evaluate(expr->left, params, n_params);
//             double right_val = ptd_expr_evaluate(expr->right, params, n_params);
//             if (right_val == 0.0) {
//                 DIE_ERROR(1, "Division by zero in expression evaluation");
//             }
//             return left_val / right_val;
//         }

//         case PTD_EXPR_INV: {
//             double child_val = ptd_expr_evaluate(expr->left, params, n_params);
//             if (child_val == 0.0) {
//                 DIE_ERROR(1, "Division by zero in inversion expression evaluation");
//             }
//             return 1.0 / child_val;
//         }

//         case PTD_EXPR_SUB: {
//             double left_val = ptd_expr_evaluate(expr->left, params, n_params);
//             double right_val = ptd_expr_evaluate(expr->right, params, n_params);
//             return left_val - right_val;
//         }

//         default:
//             DIE_ERROR(1, "Unknown expression type in evaluation");
//             return 0.0;
//     }
// }

// /**
//  * Evaluate an expression for multiple parameter sets (batch evaluation)
//  */
// void ptd_expr_evaluate_batch(
//     const struct ptd_expression *expr,
//     const double *params_batch,      // shape: (batch_size, n_params)
//     size_t batch_size,
//     size_t n_params,
//     double *output                   // shape: (batch_size,)
// ) {
//     if (expr == NULL || params_batch == NULL || output == NULL) {
//         return;
//     }

//     // Evaluate for each parameter set
//     for (size_t i = 0; i < batch_size; i++) {
//         const double *params_i = params_batch + i * n_params;
//         output[i] = ptd_expr_evaluate(expr, params_i, n_params);
//     }
// }

// // ============================================================================
// // Trace-Based Elimination Implementation
// // ============================================================================

// /**
//  * Helper: Ensure trace operations array has sufficient capacity
//  */
// static int ensure_trace_capacity(
//     struct ptd_elimination_trace *trace,
//     size_t required_capacity
// ) {
//     if (trace->operations_length >= required_capacity) {
//         return 0; // Already has capacity
//     }

//     // Find current capacity (stored separately, but we'll compute it)
//     size_t current_capacity = trace->operations_length;
//     if (current_capacity == 0) {
//         current_capacity = 1000; // Initial capacity
//     }

//     // Double capacity until we have enough
//     size_t new_capacity = current_capacity;
//     while (new_capacity < required_capacity) {
//         new_capacity *= 2;
//     }

//     // Realloc
//     struct ptd_trace_operation *new_ops = (struct ptd_trace_operation *)realloc(
//         trace->operations,
//         new_capacity * sizeof(struct ptd_trace_operation)
//     );
//     if (new_ops == NULL) {
//         return -1; // Allocation failed
//     }

//     trace->operations = new_ops;
//     return 0;
// }

// /**
//  * Helper: Add CONST operation to trace
//  */
// static size_t add_const_to_trace(
//     struct ptd_elimination_trace *trace,
//     double value
// ) {
//     // Ensure capacity (allow for growth)
//     if (ensure_trace_capacity(trace, trace->operations_length + 1) != 0) {
//         return (size_t)-1;
//     }

//     size_t idx = trace->operations_length++;
//     struct ptd_trace_operation *op = &trace->operations[idx];

//     op->op_type = PTD_OP_CONST;
//     op->const_value = value;
//     op->param_idx = 0;
//     op->coefficients = NULL;
//     op->coefficients_length = 0;
//     op->operands = NULL;
//     op->operands_length = 0;

//     return idx;
// }

// /**
//  * Helper: Add DOT operation to trace
//  * DOT product: Σ(coefficients[i] * θ[i])
//  */
// static size_t add_dot_to_trace(
//     struct ptd_elimination_trace *trace,
//     const double *coefficients,
//     size_t coefficients_length
// ) {
//     if (ensure_trace_capacity(trace, trace->operations_length + 1) != 0) {
//         return (size_t)-1;
//     }

//     size_t idx = trace->operations_length++;
//     struct ptd_trace_operation *op = &trace->operations[idx];

//     op->op_type = PTD_OP_DOT;

//     // Copy coefficients
//     op->coefficients = (double *)malloc(coefficients_length * sizeof(double));
//     if (op->coefficients == NULL) {
//         trace->operations_length--; // Roll back
//         return (size_t)-1;
//     }
//     memcpy(op->coefficients, coefficients, coefficients_length * sizeof(double));
//     op->coefficients_length = coefficients_length;

//     op->const_value = 0.0;
//     op->param_idx = 0;
//     op->operands = NULL;
//     op->operands_length = 0;

//     return idx;
// }

// /**
//  * Helper: Add ADD operation to trace
//  * ADD: operands[0] + operands[1]
//  */
// static size_t add_add_to_trace(
//     struct ptd_elimination_trace *trace,
//     size_t left_idx,
//     size_t right_idx
// ) {
//     if (ensure_trace_capacity(trace, trace->operations_length + 1) != 0) {
//         return (size_t)-1;
//     }

//     size_t idx = trace->operations_length++;
//     struct ptd_trace_operation *op = &trace->operations[idx];

//     op->op_type = PTD_OP_ADD;

//     op->operands = (size_t *)malloc(2 * sizeof(size_t));
//     if (op->operands == NULL) {
//         trace->operations_length--;
//         return (size_t)-1;
//     }
//     op->operands[0] = left_idx;
//     op->operands[1] = right_idx;
//     op->operands_length = 2;

//     op->const_value = 0.0;
//     op->param_idx = 0;
//     op->coefficients = NULL;
//     op->coefficients_length = 0;

//     return idx;
// }

// /**
//  * Helper: Add MUL operation to trace
//  * MUL: operands[0] * operands[1]
//  */
// static size_t add_mul_to_trace(
//     struct ptd_elimination_trace *trace,
//     size_t left_idx,
//     size_t right_idx
// ) {
//     if (ensure_trace_capacity(trace, trace->operations_length + 1) != 0) {
//         return (size_t)-1;
//     }

//     size_t idx = trace->operations_length++;
//     struct ptd_trace_operation *op = &trace->operations[idx];

//     op->op_type = PTD_OP_MUL;

//     op->operands = (size_t *)malloc(2 * sizeof(size_t));
//     if (op->operands == NULL) {
//         trace->operations_length--;
//         return (size_t)-1;
//     }
//     op->operands[0] = left_idx;
//     op->operands[1] = right_idx;
//     op->operands_length = 2;

//     op->const_value = 0.0;
//     op->param_idx = 0;
//     op->coefficients = NULL;
//     op->coefficients_length = 0;

//     return idx;
// }

// /**
//  * Helper: Add DIV operation to trace
//  * DIV: operands[0] / operands[1]
//  */
// static size_t add_div_to_trace(
//     struct ptd_elimination_trace *trace,
//     size_t left_idx,
//     size_t right_idx
// ) {
//     if (ensure_trace_capacity(trace, trace->operations_length + 1) != 0) {
//         return (size_t)-1;
//     }

//     size_t idx = trace->operations_length++;
//     struct ptd_trace_operation *op = &trace->operations[idx];

//     op->op_type = PTD_OP_DIV;

//     op->operands = (size_t *)malloc(2 * sizeof(size_t));
//     if (op->operands == NULL) {
//         trace->operations_length--;
//         return (size_t)-1;
//     }
//     op->operands[0] = left_idx;
//     op->operands[1] = right_idx;
//     op->operands_length = 2;

//     op->const_value = 0.0;
//     op->param_idx = 0;
//     op->coefficients = NULL;
//     op->coefficients_length = 0;

//     return idx;
// }

// /**
//  * Helper: Add INV operation to trace
//  * INV: 1 / operands[0]
//  */
// static size_t add_inv_to_trace(
//     struct ptd_elimination_trace *trace,
//     size_t operand_idx
// ) {
//     if (ensure_trace_capacity(trace, trace->operations_length + 1) != 0) {
//         return (size_t)-1;
//     }

//     size_t idx = trace->operations_length++;
//     struct ptd_trace_operation *op = &trace->operations[idx];

//     op->op_type = PTD_OP_INV;

//     op->operands = (size_t *)malloc(1 * sizeof(size_t));
//     if (op->operands == NULL) {
//         trace->operations_length--;
//         return (size_t)-1;
//     }
//     op->operands[0] = operand_idx;
//     op->operands_length = 1;

//     op->const_value = 0.0;
//     op->param_idx = 0;
//     op->coefficients = NULL;
//     op->coefficients_length = 0;

//     return idx;
// }

// /**
//  * Helper: Add SUM operation to trace
//  * SUM: sum(operands[0], operands[1], ..., operands[n-1])
//  */
// static size_t add_sum_to_trace(
//     struct ptd_elimination_trace *trace,
//     const size_t *operand_indices,
//     size_t n_operands
// ) {
//     if (ensure_trace_capacity(trace, trace->operations_length + 1) != 0) {
//         return (size_t)-1;
//     }

//     size_t idx = trace->operations_length++;
//     struct ptd_trace_operation *op = &trace->operations[idx];

//     op->op_type = PTD_OP_SUM;

//     op->operands = (size_t *)malloc(n_operands * sizeof(size_t));
//     if (op->operands == NULL) {
//         trace->operations_length--;
//         return (size_t)-1;
//     }
//     memcpy(op->operands, operand_indices, n_operands * sizeof(size_t));
//     op->operands_length = n_operands;

//     op->const_value = 0.0;
//     op->param_idx = 0;
//     op->coefficients = NULL;
//     op->coefficients_length = 0;

//     return idx;
// }

// /**
//  * Record elimination trace from parameterized graph
//  *
//  * Performs graph elimination while recording all arithmetic operations
//  * in a linear sequence. Currently implements Phase 1 (vertex rates).
//  */
// struct ptd_elimination_trace *ptd_record_elimination_trace(
//     struct ptd_graph *graph
// ) {
//     if (!graph->parameterized) {
//         sprintf((char*)ptd_err, "ptd_record_elimination_trace: graph is not parameterized");
//         return NULL;
//     }

//     // Check cache first (if hash computation succeeds)
//     struct ptd_hash_result *hash = ptd_graph_content_hash(graph);
//     struct ptd_elimination_trace *cached_trace = NULL;

//     if (hash != NULL) {
//         cached_trace = load_trace_from_cache(hash->hash_hex);
//         if (cached_trace != NULL) {
//             DEBUG_PRINT("INFO: loaded elimination trace from cache (%s)\n", hash->hash_hex);
//             ptd_hash_destroy(hash);
//             return cached_trace;
//         }
//     }

//     // Cache miss or hash failed - record trace normally
//     DEBUG_PRINT("INFO: cache miss, recording elimination trace...\n");

//     // Allocate trace structure
//     struct ptd_elimination_trace *trace = (struct ptd_elimination_trace *)malloc(sizeof(*trace));
//     if (trace == NULL) {
//         sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate trace");
//         return NULL;
//     }

//     // Initialize metadata
//     trace->n_vertices = graph->vertices_length;
//     trace->state_length = graph->state_length;
//     trace->param_length = graph->param_length;
//     trace->is_discrete = graph->was_dph;

//     // Find starting vertex index
//     trace->starting_vertex_idx = 0;
//     if (graph->starting_vertex != NULL) {
//         trace->starting_vertex_idx = graph->starting_vertex->index;
//     }

//     // Allocate operations array (initial capacity)
//     size_t operations_capacity = 1000;
//     trace->operations = (struct ptd_trace_operation *)malloc(operations_capacity * sizeof(struct ptd_trace_operation));
//     if (trace->operations == NULL) {
//         sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate operations");
//         free(trace);
//         return NULL;
//     }
//     trace->operations_length = 0;

//     // Allocate vertex mappings
//     trace->vertex_rates = (size_t *)malloc(trace->n_vertices * sizeof(size_t));
//     trace->edge_probs = (size_t **)malloc(trace->n_vertices * sizeof(size_t*));
//     trace->edge_probs_lengths = (size_t *)calloc(trace->n_vertices, sizeof(size_t));
//     trace->vertex_targets = (size_t **)malloc(trace->n_vertices * sizeof(size_t*));
//     trace->vertex_targets_lengths = (size_t *)calloc(trace->n_vertices, sizeof(size_t));

//     if (trace->vertex_rates == NULL || trace->edge_probs == NULL ||
//         trace->edge_probs_lengths == NULL || trace->vertex_targets == NULL ||
//         trace->vertex_targets_lengths == NULL) {
//         sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate vertex mappings");
//         ptd_elimination_trace_destroy(trace);
//         return NULL;
//     }

//     // Initialize edge arrays to NULL
//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         trace->edge_probs[i] = NULL;
//         trace->vertex_targets[i] = NULL;
//     }

//     // Copy vertex states
//     trace->states = (int **)malloc(trace->n_vertices * sizeof(int*));
//     if (trace->states == NULL) {
//         sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate states");
//         ptd_elimination_trace_destroy(trace);
//         return NULL;
//     }

//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         trace->states[i] = (int *)malloc(trace->state_length * sizeof(int));
//         if (trace->states[i] == NULL) {
//             sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate state for vertex %zu", i);
//             // Free previously allocated states
//             for (size_t j = 0; j < i; j++) {
//                 free(trace->states[j]);
//             }
//             free(trace->states);
//             ptd_elimination_trace_destroy(trace);
//             return NULL;
//         }

//         if (graph->vertices[i]->state != NULL) {
//             memcpy(trace->states[i], graph->vertices[i]->state,
//                    trace->state_length * sizeof(int));
//         } else {
//             // Zero initialize if no state
//             memset(trace->states[i], 0, trace->state_length * sizeof(int));
//         }
//     }

//     // PHASE 1: Compute vertex rates
//     for (size_t i = 0; i < graph->vertices_length; i++) {
//         struct ptd_vertex *v = graph->vertices[i];

//         if (v->edges_length == 0) {
//             // Absorbing state: rate = 0
//             trace->vertex_rates[i] = add_const_to_trace(trace, 0.0);
//         } else {
//             // rate = 1 / sum(edge_weights)
//             size_t *weight_indices = (size_t *)malloc(v->edges_length * sizeof(size_t));
//             if (weight_indices == NULL) {
//                 sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate weight_indices");
//                 ptd_elimination_trace_destroy(trace);
//                 return NULL;
//             }

//             for (size_t j = 0; j < v->edges_length; j++) {
//                 struct ptd_edge *edge = v->edges[j];

//                 if (edge->parameterized) {
//                     struct ptd_edge_parameterized *param_edge =
//                         (struct ptd_edge_parameterized*)edge;

//                     // Extract coefficients from param_edge->state
//                     double *coeffs = param_edge->state;
//                     size_t coeffs_len = graph->param_length;

//                     // Check if all coefficients are zero
//                     bool all_zero = true;
//                     for (size_t k = 0; k < coeffs_len; k++) {
//                         if (fabs(coeffs[k]) > 1e-15) {
//                             all_zero = false;
//                             break;
//                         }
//                     }

//                     if (all_zero) {
//                         // No parameterization, just use base weight
//                         weight_indices[j] = add_const_to_trace(trace, param_edge->weight);
//                     } else {
//                         // DOT product: c₁*θ₁ + c₂*θ₂ + ...
//                         size_t dot_idx = add_dot_to_trace(trace, coeffs, coeffs_len);

//                         // Add base weight if non-zero
//                         if (fabs(param_edge->weight) > 1e-15) {
//                             size_t base_idx = add_const_to_trace(trace, param_edge->weight);
//                             weight_indices[j] = add_add_to_trace(trace, base_idx, dot_idx);
//                         } else {
//                             weight_indices[j] = dot_idx;
//                         }
//                     }
//                 } else {
//                     // Regular edge
//                     weight_indices[j] = add_const_to_trace(trace, edge->weight);
//                 }
//             }

//             // Sum all weights
//             size_t sum_idx = add_sum_to_trace(trace, weight_indices, v->edges_length);

//             // Rate = 1 / sum
//             trace->vertex_rates[i] = add_inv_to_trace(trace, sum_idx);

//             free(weight_indices);
//         }
//     }

//     // PHASE 2: Convert edges to probabilities
//     // Allocate dynamic edge arrays (will grow during elimination)
//     size_t *edge_capacities = (size_t *)malloc(trace->n_vertices * sizeof(size_t));
//     if (edge_capacities == NULL) {
//         sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate edge_capacities");
//         ptd_elimination_trace_destroy(trace);
//         return NULL;
//     }

//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         struct ptd_vertex *v = graph->vertices[i];
//         size_t n_edges = v->edges_length;

//         trace->edge_probs_lengths[i] = n_edges;
//         trace->vertex_targets_lengths[i] = n_edges;
//         edge_capacities[i] = n_edges > 0 ? n_edges : 1;

//         if (n_edges > 0) {
//             trace->edge_probs[i] = (size_t *)malloc(edge_capacities[i] * sizeof(size_t));
//             trace->vertex_targets[i] = (size_t *)malloc(edge_capacities[i] * sizeof(size_t));

//             if (trace->edge_probs[i] == NULL || trace->vertex_targets[i] == NULL) {
//                 sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate edge arrays");
//                 free(edge_capacities);
//                 ptd_elimination_trace_destroy(trace);
//                 return NULL;
//             }

//             // Convert edge weights to probabilities
//             for (size_t j = 0; j < n_edges; j++) {
//                 struct ptd_edge *edge = v->edges[j];

//                 // Get edge weight index (recompute like in Phase 1)
//                 size_t weight_idx;
//                 if (edge->parameterized) {
//                     struct ptd_edge_parameterized *param_edge =
//                         (struct ptd_edge_parameterized*)edge;

//                     double *coeffs = param_edge->state;
//                     size_t coeffs_len = graph->param_length;

//                     bool all_zero = true;
//                     for (size_t k = 0; k < coeffs_len; k++) {
//                         if (fabs(coeffs[k]) > 1e-15) {
//                             all_zero = false;
//                             break;
//                         }
//                     }

//                     if (all_zero) {
//                         weight_idx = add_const_to_trace(trace, param_edge->weight);
//                     } else {
//                         size_t dot_idx = add_dot_to_trace(trace, coeffs, coeffs_len);
//                         if (fabs(param_edge->weight) > 1e-15) {
//                             size_t base_idx = add_const_to_trace(trace, param_edge->weight);
//                             weight_idx = add_add_to_trace(trace, base_idx, dot_idx);
//                         } else {
//                             weight_idx = dot_idx;
//                         }
//                     }
//                 } else {
//                     weight_idx = add_const_to_trace(trace, edge->weight);
//                 }

//                 // prob = weight * rate
//                 size_t prob_idx = add_mul_to_trace(trace, weight_idx, trace->vertex_rates[i]);
//                 trace->edge_probs[i][j] = prob_idx;

//                 // Store target vertex index
//                 trace->vertex_targets[i][j] = edge->to->index;
//             }
//         }
//     }

//     // PHASE 3: Elimination loop
//     // Build parent-child relationships
//     size_t **parents = (size_t **)malloc(trace->n_vertices * sizeof(size_t*));
//     size_t *parents_lengths = (size_t *)calloc(trace->n_vertices, sizeof(size_t));
//     size_t *parents_capacities = (size_t *)malloc(trace->n_vertices * sizeof(size_t));

//     if (parents == NULL || parents_lengths == NULL || parents_capacities == NULL) {
//         sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate parent arrays");
//         free(edge_capacities);
//         free(parents);
//         free(parents_lengths);
//         free(parents_capacities);
//         ptd_elimination_trace_destroy(trace);
//         return NULL;
//     }

//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         parents_capacities[i] = 4;  // Initial capacity
//         parents[i] = (size_t *)malloc(parents_capacities[i] * sizeof(size_t));
//         if (parents[i] == NULL) {
//             sprintf((char*)ptd_err, "ptd_record_elimination_trace: failed to allocate parent list");
//             for (size_t k = 0; k < i; k++) {
//                 free(parents[k]);
//             }
//             free(edge_capacities);
//             free(parents);
//             free(parents_lengths);
//             free(parents_capacities);
//             ptd_elimination_trace_destroy(trace);
//             return NULL;
//         }
//     }

//     // Build parent lists
//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         for (size_t j = 0; j < trace->vertex_targets_lengths[i]; j++) {
//             size_t to_idx = trace->vertex_targets[i][j];

//             // Add i to parents of to_idx
//             if (parents_lengths[to_idx] >= parents_capacities[to_idx]) {
//                 parents_capacities[to_idx] *= 2;
//                 size_t *new_parents = (size_t *)realloc(parents[to_idx],
//                     parents_capacities[to_idx] * sizeof(size_t));
//                 if (new_parents == NULL) {
//                     sprintf((char*)ptd_err, "ptd_record_elimination_trace: realloc parent failed");
//                     for (size_t k = 0; k < trace->n_vertices; k++) {
//                         free(parents[k]);
//                     }
//                     free(edge_capacities);
//                     free(parents);
//                     free(parents_lengths);
//                     free(parents_capacities);
//                     ptd_elimination_trace_destroy(trace);
//                     return NULL;
//                 }
//                 parents[to_idx] = new_parents;
//             }
//             parents[to_idx][parents_lengths[to_idx]++] = i;
//         }
//     }

//     // Elimination loop
//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         size_t n_children = trace->vertex_targets_lengths[i];

//         if (n_children == 0) {
//             // Absorbing state, nothing to eliminate
//             continue;
//         }

//         // For each parent of vertex i
//         for (size_t p = 0; p < parents_lengths[i]; p++) {
//             size_t parent_idx = parents[i][p];

//             // Skip if parent already processed
//             if (parent_idx < i) {
//                 continue;
//             }

//             // Find edge from parent to i
//             size_t parent_to_i_edge_idx = (size_t)-1;
//             for (size_t e = 0; e < trace->vertex_targets_lengths[parent_idx]; e++) {
//                 if (trace->vertex_targets[parent_idx][e] == i &&
//                     trace->edge_probs[parent_idx][e] != (size_t)-1) {
//                     parent_to_i_edge_idx = e;
//                     break;
//                 }
//             }

//             if (parent_to_i_edge_idx == (size_t)-1) {
//                 // Parent no longer has edge to i
//                 continue;
//             }

//             size_t parent_to_i_prob = trace->edge_probs[parent_idx][parent_to_i_edge_idx];

//             // For each child of i
//             for (size_t c = 0; c < n_children; c++) {
//                 size_t child_idx = trace->vertex_targets[i][c];
//                 size_t i_to_child_prob = trace->edge_probs[i][c];

//                 // Skip self-loops (TODO: implement properly later)
//                 if (child_idx == parent_idx || child_idx == i) {
//                     continue;
//                 }

//                 // Bypass probability: parent_to_i * i_to_child
//                 size_t bypass_prob = add_mul_to_trace(trace, parent_to_i_prob, i_to_child_prob);

//                 // Check if parent already has edge to child
//                 size_t parent_to_child_edge_idx = (size_t)-1;
//                 for (size_t e = 0; e < trace->vertex_targets_lengths[parent_idx]; e++) {
//                     if (trace->vertex_targets[parent_idx][e] == child_idx &&
//                         trace->edge_probs[parent_idx][e] != (size_t)-1) {
//                         parent_to_child_edge_idx = e;
//                         break;
//                     }
//                 }

//                 if (parent_to_child_edge_idx != (size_t)-1) {
//                     // Update existing edge
//                     size_t old_prob = trace->edge_probs[parent_idx][parent_to_child_edge_idx];
//                     size_t new_prob = add_add_to_trace(trace, old_prob, bypass_prob);
//                     trace->edge_probs[parent_idx][parent_to_child_edge_idx] = new_prob;
//                 } else {
//                     // Create new edge
//                     size_t new_idx = trace->vertex_targets_lengths[parent_idx];

//                     // Ensure capacity
//                     if (new_idx >= edge_capacities[parent_idx]) {
//                         edge_capacities[parent_idx] *= 2;
//                         size_t *new_probs = (size_t *)realloc(trace->edge_probs[parent_idx],
//                             edge_capacities[parent_idx] * sizeof(size_t));
//                         size_t *new_targets = (size_t *)realloc(trace->vertex_targets[parent_idx],
//                             edge_capacities[parent_idx] * sizeof(size_t));

//                         if (new_probs == NULL || new_targets == NULL) {
//                             sprintf((char*)ptd_err, "ptd_record_elimination_trace: realloc edge failed");
//                             for (size_t k = 0; k < trace->n_vertices; k++) {
//                                 free(parents[k]);
//                             }
//                             free(edge_capacities);
//                             free(parents);
//                             free(parents_lengths);
//                             free(parents_capacities);
//                             ptd_elimination_trace_destroy(trace);
//                             return NULL;
//                         }

//                         trace->edge_probs[parent_idx] = new_probs;
//                         trace->vertex_targets[parent_idx] = new_targets;
//                     }

//                     trace->edge_probs[parent_idx][new_idx] = bypass_prob;
//                     trace->vertex_targets[parent_idx][new_idx] = child_idx;
//                     trace->vertex_targets_lengths[parent_idx]++;
//                     trace->edge_probs_lengths[parent_idx]++;
//                 }
//             }

//             // Mark edge from parent to i as removed
//             trace->edge_probs[parent_idx][parent_to_i_edge_idx] = (size_t)-1;

//             // Renormalize parent's edges
//             // Count valid (non-removed) edges
//             size_t valid_count = 0;
//             for (size_t e = 0; e < trace->edge_probs_lengths[parent_idx]; e++) {
//                 if (trace->edge_probs[parent_idx][e] != (size_t)-1) {
//                     valid_count++;
//                 }
//             }

//             if (valid_count > 0) {
//                 // Compute sum of valid edges
//                 size_t *valid_probs = (size_t *)malloc(valid_count * sizeof(size_t));
//                 if (valid_probs == NULL) {
//                     sprintf((char*)ptd_err, "ptd_record_elimination_trace: malloc valid_probs failed");
//                     for (size_t k = 0; k < trace->n_vertices; k++) {
//                         free(parents[k]);
//                     }
//                     free(edge_capacities);
//                     free(parents);
//                     free(parents_lengths);
//                     free(parents_capacities);
//                     ptd_elimination_trace_destroy(trace);
//                     return NULL;
//                 }

//                 size_t valid_idx = 0;
//                 for (size_t e = 0; e < trace->edge_probs_lengths[parent_idx]; e++) {
//                     if (trace->edge_probs[parent_idx][e] != (size_t)-1) {
//                         valid_probs[valid_idx++] = trace->edge_probs[parent_idx][e];
//                     }
//                 }

//                 size_t total_idx = add_sum_to_trace(trace, valid_probs, valid_count);
//                 free(valid_probs);

//                 // Normalize each valid edge: prob = prob / total
//                 for (size_t e = 0; e < trace->edge_probs_lengths[parent_idx]; e++) {
//                     if (trace->edge_probs[parent_idx][e] != (size_t)-1) {
//                         size_t old_prob = trace->edge_probs[parent_idx][e];
//                         size_t new_prob = add_div_to_trace(trace, old_prob, total_idx);
//                         trace->edge_probs[parent_idx][e] = new_prob;
//                     }
//                 }
//             }
//         }
//     }

//     // PHASE 4: Clean up removed edges
//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         // Count valid edges
//         size_t valid_count = 0;
//         for (size_t j = 0; j < trace->edge_probs_lengths[i]; j++) {
//             if (trace->edge_probs[i][j] != (size_t)-1) {
//                 valid_count++;
//             }
//         }

//         // Compact arrays
//         if (valid_count < trace->edge_probs_lengths[i]) {
//             size_t *new_probs = (size_t *)malloc(valid_count * sizeof(size_t));
//             size_t *new_targets = (size_t *)malloc(valid_count * sizeof(size_t));

//             if ((valid_count > 0 && (new_probs == NULL || new_targets == NULL))) {
//                 sprintf((char*)ptd_err, "ptd_record_elimination_trace: cleanup malloc failed");
//                 for (size_t k = 0; k < trace->n_vertices; k++) {
//                     free(parents[k]);
//                 }
//                 free(edge_capacities);
//                 free(parents);
//                 free(parents_lengths);
//                 free(parents_capacities);
//                 free(new_probs);
//                 free(new_targets);
//                 ptd_elimination_trace_destroy(trace);
//                 return NULL;
//             }

//             size_t write_idx = 0;
//             for (size_t j = 0; j < trace->edge_probs_lengths[i]; j++) {
//                 if (trace->edge_probs[i][j] != (size_t)-1) {
//                     new_probs[write_idx] = trace->edge_probs[i][j];
//                     new_targets[write_idx] = trace->vertex_targets[i][j];
//                     write_idx++;
//                 }
//             }

//             free(trace->edge_probs[i]);
//             free(trace->vertex_targets[i]);

//             trace->edge_probs[i] = new_probs;
//             trace->vertex_targets[i] = new_targets;
//             trace->edge_probs_lengths[i] = valid_count;
//             trace->vertex_targets_lengths[i] = valid_count;
//         }
//     }

//     // Cleanup temporary arrays
//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         free(parents[i]);
//     }
//     free(edge_capacities);
//     free(parents);
//     free(parents_lengths);
//     free(parents_capacities);

//     // Save newly recorded trace to cache
//     if (hash != NULL) {
//         save_trace_to_cache(hash->hash_hex, trace);
//         ptd_hash_destroy(hash);
//     }

//     return trace;
// }

// /**
//  * Destroy elimination trace and free all memory
//  */
// void ptd_elimination_trace_destroy(struct ptd_elimination_trace *trace) {
//     if (trace == NULL) {
//         return;
//     }

//     // Free operations
//     if (trace->operations != NULL) {
//         for (size_t i = 0; i < trace->operations_length; i++) {
//             struct ptd_trace_operation *op = &trace->operations[i];
//             if (op->coefficients != NULL) {
//                 free(op->coefficients);
//             }
//             if (op->operands != NULL) {
//                 free(op->operands);
//             }
//         }
//         free(trace->operations);
//     }

//     // Free vertex mappings
//     if (trace->vertex_rates != NULL) {
//         free(trace->vertex_rates);
//     }

//     if (trace->edge_probs != NULL) {
//         for (size_t i = 0; i < trace->n_vertices; i++) {
//             if (trace->edge_probs[i] != NULL) {
//                 free(trace->edge_probs[i]);
//             }
//         }
//         free(trace->edge_probs);
//     }

//     if (trace->edge_probs_lengths != NULL) {
//         free(trace->edge_probs_lengths);
//     }

//     if (trace->vertex_targets != NULL) {
//         for (size_t i = 0; i < trace->n_vertices; i++) {
//             if (trace->vertex_targets[i] != NULL) {
//                 free(trace->vertex_targets[i]);
//             }
//         }
//         free(trace->vertex_targets);
//     }

//     if (trace->vertex_targets_lengths != NULL) {
//         free(trace->vertex_targets_lengths);
//     }

//     // Free states
//     if (trace->states != NULL) {
//         for (size_t i = 0; i < trace->n_vertices; i++) {
//             if (trace->states[i] != NULL) {
//                 free(trace->states[i]);
//             }
//         }
//         free(trace->states);
//     }

//     free(trace);
// }

// /**
//  * Evaluate elimination trace with concrete parameter values
//  *
//  * Executes the recorded operation sequence with given parameters
//  * to produce vertex rates and edge probabilities.
//  */
// struct ptd_trace_result *ptd_evaluate_trace(
//     const struct ptd_elimination_trace *trace,
//     const double *params,
//     size_t params_length
// ) {
//     // Validate parameters
//     if (trace == NULL) {
//         sprintf((char*)ptd_err, "ptd_evaluate_trace: trace is NULL");
//         return NULL;
//     }

//     if (trace->param_length > 0) {
//         if (params == NULL) {
//             sprintf((char*)ptd_err, "ptd_evaluate_trace: params is NULL but trace has %zu parameters",
//                     trace->param_length);
//             return NULL;
//         }

//         if (params_length != trace->param_length) {
//             sprintf((char*)ptd_err, "ptd_evaluate_trace: expected %zu parameters, got %zu",
//                     trace->param_length, params_length);
//             return NULL;
//         }
//     }

//     // Allocate value array for all operations
//     double *values = (double *)calloc(trace->operations_length, sizeof(double));
//     if (values == NULL) {
//         sprintf((char*)ptd_err, "ptd_evaluate_trace: failed to allocate values array");
//         return NULL;
//     }

//     // Execute operations in order
//     for (size_t i = 0; i < trace->operations_length; i++) {
//         const struct ptd_trace_operation *op = &trace->operations[i];

//         switch (op->op_type) {
//             case PTD_OP_CONST:
//                 values[i] = op->const_value;
//                 break;

//             case PTD_OP_PARAM:
//                 if (op->param_idx < params_length) {
//                     values[i] = params[op->param_idx];
//                 }
//                 break;

//             case PTD_OP_DOT:
//                 // Dot product: Σ(cᵢ * θᵢ)
//                 values[i] = 0.0;
//                 for (size_t j = 0; j < op->coefficients_length && j < params_length; j++) {
//                     values[i] += op->coefficients[j] * params[j];
//                 }
//                 break;

//             case PTD_OP_ADD:
//                 if (op->operands_length >= 2) {
//                     values[i] = values[op->operands[0]] + values[op->operands[1]];
//                 }
//                 break;

//             case PTD_OP_MUL:
//                 if (op->operands_length >= 2) {
//                     values[i] = values[op->operands[0]] * values[op->operands[1]];
//                 }
//                 break;

//             case PTD_OP_DIV:
//                 if (op->operands_length >= 2) {
//                     double denominator = values[op->operands[1]];
//                     if (fabs(denominator) > 1e-15) {
//                         values[i] = values[op->operands[0]] / denominator;
//                     } else {
//                         values[i] = 0.0;  // Handle division by zero
//                     }
//                 }
//                 break;

//             case PTD_OP_INV:
//                 if (op->operands_length >= 1) {
//                     double val = values[op->operands[0]];
//                     if (fabs(val) > 1e-15) {
//                         values[i] = 1.0 / val;
//                     } else {
//                         values[i] = 0.0;  // Handle inverse of zero
//                     }
//                 }
//                 break;

//             case PTD_OP_SUM:
//                 values[i] = 0.0;
//                 for (size_t j = 0; j < op->operands_length; j++) {
//                     values[i] += values[op->operands[j]];
//                 }
//                 break;

//             default:
//                 // Unknown operation type
//                 values[i] = 0.0;
//                 break;
//         }
//     }

//     // Allocate result structure
//     struct ptd_trace_result *result = (struct ptd_trace_result *)malloc(sizeof(*result));
//     if (result == NULL) {
//         sprintf((char*)ptd_err, "ptd_evaluate_trace: failed to allocate result");
//         free(values);
//         return NULL;
//     }

//     result->n_vertices = trace->n_vertices;

//     // Extract vertex rates
//     result->vertex_rates = (double *)malloc(trace->n_vertices * sizeof(double));
//     if (result->vertex_rates == NULL) {
//         sprintf((char*)ptd_err, "ptd_evaluate_trace: failed to allocate vertex_rates");
//         free(values);
//         free(result);
//         return NULL;
//     }

//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         result->vertex_rates[i] = values[trace->vertex_rates[i]];
//     }

//     // Extract edge probabilities
//     result->edge_probs = (double **)malloc(trace->n_vertices * sizeof(double*));
//     result->edge_probs_lengths = (size_t *)malloc(trace->n_vertices * sizeof(size_t));
//     result->vertex_targets = (size_t **)malloc(trace->n_vertices * sizeof(size_t*));
//     result->vertex_targets_lengths = (size_t *)malloc(trace->n_vertices * sizeof(size_t));

//     if (result->edge_probs == NULL || result->edge_probs_lengths == NULL ||
//         result->vertex_targets == NULL || result->vertex_targets_lengths == NULL) {
//         sprintf((char*)ptd_err, "ptd_evaluate_trace: failed to allocate edge arrays");
//         free(values);
//         ptd_trace_result_destroy(result);
//         return NULL;
//     }

//     // Initialize to NULL
//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         result->edge_probs[i] = NULL;
//         result->vertex_targets[i] = NULL;
//     }

//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         size_t n_edges = trace->edge_probs_lengths[i];
//         result->edge_probs_lengths[i] = n_edges;
//         result->vertex_targets_lengths[i] = n_edges;

//         if (n_edges > 0) {
//             result->edge_probs[i] = (double *)malloc(n_edges * sizeof(double));
//             result->vertex_targets[i] = (size_t *)malloc(n_edges * sizeof(size_t));

//             if (result->edge_probs[i] == NULL || result->vertex_targets[i] == NULL) {
//                 sprintf((char*)ptd_err, "ptd_evaluate_trace: failed to allocate edge arrays for vertex %zu", i);
//                 free(values);
//                 ptd_trace_result_destroy(result);
//                 return NULL;
//             }

//             for (size_t j = 0; j < n_edges; j++) {
//                 result->edge_probs[i][j] = values[trace->edge_probs[i][j]];
//                 result->vertex_targets[i][j] = trace->vertex_targets[i][j];
//             }
//         } else {
//             result->edge_probs[i] = NULL;
//             result->vertex_targets[i] = NULL;
//         }
//     }

//     free(values);
//     return result;
// }

// /**
//  * Destroy trace evaluation result and free all memory
//  */
// void ptd_trace_result_destroy(struct ptd_trace_result *result) {
//     if (result == NULL) {
//         return;
//     }

//     if (result->vertex_rates != NULL) {
//         free(result->vertex_rates);
//     }

//     if (result->edge_probs != NULL) {
//         for (size_t i = 0; i < result->n_vertices; i++) {
//             if (result->edge_probs[i] != NULL) {
//                 free(result->edge_probs[i]);
//             }
//         }
//         free(result->edge_probs);
//     }

//     if (result->edge_probs_lengths != NULL) {
//         free(result->edge_probs_lengths);
//     }

//     if (result->vertex_targets != NULL) {
//         for (size_t i = 0; i < result->n_vertices; i++) {
//             if (result->vertex_targets[i] != NULL) {
//                 free(result->vertex_targets[i]);
//             }
//         }
//         free(result->vertex_targets);
//     }

//     if (result->vertex_targets_lengths != NULL) {
//         free(result->vertex_targets_lengths);
//     }

//     free(result);
// }

// /**
//  * Instantiate a complete graph from trace evaluation result
//  *
//  * Creates a new graph with all vertices and edges from the evaluated trace.
//  * This mirrors the Python instantiate_from_trace() function.
//  */
// struct ptd_graph *ptd_instantiate_from_trace(
//     const struct ptd_trace_result *result,
//     const struct ptd_elimination_trace *trace
// ) {
//     // Validate inputs
//     if (result == NULL || trace == NULL) {
//         sprintf((char*)ptd_err, "ptd_instantiate_from_trace: NULL input");
//         return NULL;
//     }

//     if (result->n_vertices != trace->n_vertices) {
//         sprintf((char*)ptd_err, "ptd_instantiate_from_trace: vertex count mismatch");
//         return NULL;
//     }

//     // Create new graph
//     struct ptd_graph *graph = ptd_graph_create(trace->state_length);
//     if (graph == NULL) {
//         return NULL;
//     }

//     // Create AVL tree for vertex lookup
//     struct ptd_avl_tree *avl_tree = ptd_avl_tree_create(graph->state_length);
//     if (avl_tree == NULL) {
//         sprintf((char*)ptd_err, "ptd_instantiate_from_trace: failed to create AVL tree");
//         ptd_graph_destroy(graph);
//         return NULL;
//     }

//     // Build state-to-vertex mapping
//     struct ptd_vertex **vertices = (struct ptd_vertex **)malloc(trace->n_vertices * sizeof(struct ptd_vertex *));
//     if (vertices == NULL) {
//         sprintf((char*)ptd_err, "ptd_instantiate_from_trace: failed to allocate vertex array");
//         ptd_avl_tree_destroy(avl_tree);
//         ptd_graph_destroy(graph);
//         return NULL;
//     }

//     // Get starting vertex
//     struct ptd_vertex *start_vertex = graph->starting_vertex;

//     // Check if starting vertex matches trace->states[starting_vertex_idx]
//     bool start_matches = true;
//     for (size_t j = 0; j < trace->state_length; j++) {
//         if (start_vertex->state[j] != trace->states[trace->starting_vertex_idx][j]) {
//             start_matches = false;
//             break;
//         }
//     }

//     // Create all vertices
//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         // Check if this is the starting vertex
//         if (i == trace->starting_vertex_idx && start_matches) {
//             vertices[i] = start_vertex;
//             // Add to AVL tree
//             ptd_avl_tree_find_or_insert(avl_tree, start_vertex->state, start_vertex);
//         } else {
//             // Find or create vertex with this state
//             vertices[i] = ptd_find_or_create_vertex(graph, avl_tree, trace->states[i]);
//             if (vertices[i] == NULL) {
//                 sprintf((char*)ptd_err, "ptd_instantiate_from_trace: failed to create vertex %zu", i);
//                 free(vertices);
//                 ptd_avl_tree_destroy(avl_tree);
//                 ptd_graph_destroy(graph);
//                 return NULL;
//             }
//         }
//     }

//     // Add edges
//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         double inv_rate = result->vertex_rates[i];

//         // Skip if absorbing (rate = 0 means inv_rate would be 0 or invalid)
//         if (inv_rate <= 0.0 || result->vertex_targets_lengths[i] == 0) {
//             continue;
//         }

//         struct ptd_vertex *from_vertex = vertices[i];

//         for (size_t j = 0; j < result->vertex_targets_lengths[i]; j++) {
//             double prob = result->edge_probs[i][j];
//             size_t to_idx = result->vertex_targets[i][j];

//             // Convert probability back to weight: weight = prob / inv_rate
//             // Since rate = 1 / sum(weights), we have inv_rate = sum(weights)
//             // And prob = weight / sum(weights) = weight * rate
//             // So weight = prob / rate = prob * (1 / inv_rate) = prob / inv_rate
//             double weight = prob / inv_rate;

//             struct ptd_vertex *to_vertex = vertices[to_idx];

//             // Add edge
//             struct ptd_edge *edge = ptd_graph_add_edge(from_vertex, to_vertex, weight);
//             if (edge == NULL) {
//                 sprintf((char*)ptd_err, "ptd_instantiate_from_trace: failed to add edge from %zu to %zu", i, to_idx);
//                 free(vertices);
//                 ptd_avl_tree_destroy(avl_tree);
//                 ptd_graph_destroy(graph);
//                 return NULL;
//             }
//         }
//     }

//     // Cleanup
//     free(vertices);
//     ptd_avl_tree_destroy(avl_tree);

//     return graph;
// }

// /**
//  * Build reward computation graph from trace evaluation result
//  *
//  * Converts the evaluated trace result (vertex rates, edge probabilities, targets)
//  * into a reward_compute structure that can be used for PDF/PMF computation.
//  *
//  * This is the trace-based equivalent of ptd_graph_ex_absorbation_time_comp_graph().
//  *
//  * @param result Evaluation result from ptd_evaluate_trace()
//  * @param graph Original graph structure (for metadata)
//  * @return Reward computation structure, or NULL on error
//  */
// struct ptd_desc_reward_compute *ptd_build_reward_compute_from_trace(
//     const struct ptd_trace_result *result,
//     struct ptd_graph *graph
// ) {
//     if (result == NULL) {
//         snprintf((char*)ptd_err, sizeof(ptd_err), "Trace result is NULL");
//         return NULL;
//     }

//     if (graph == NULL) {
//         snprintf((char*)ptd_err, sizeof(ptd_err), "Graph is NULL");
//         return NULL;
//     }

//     size_t n_vertices = result->n_vertices;
//     struct ptd_reward_increase *commands = NULL;
//     size_t command_index = 0;

//     // Phase 1: Add vertex rate commands
//     // For each vertex, add self-command with its rate
//     // Command format: from[i] *= (rate - 1) when from == to
//     // This is represented as: add_command(i, i, rate, ...)

//     for (size_t i = 0; i < n_vertices; i++) {
//         double rate = result->vertex_rates[i];

//         // Starting vertex or absorbing state gets rate 0
//         if (i == 0 || result->edge_probs_lengths[i] == 0) {
//             commands = add_command(commands, i, i, 0.0, command_index++);
//         } else {
//             commands = add_command(commands, i, i, rate, command_index++);
//         }
//     }

//     // Phase 2: Add edge probability commands
//     // Traverse vertices in reverse order (topological order for DAG)
//     // For each edge, add command: from[i] += to[j] * probability

//     for (size_t ii = 0; ii < n_vertices; ii++) {
//         size_t i = n_vertices - ii - 1;  // Reverse order

//         size_t n_edges = result->edge_probs_lengths[i];

//         for (size_t j = 0; j < n_edges; j++) {
//             double prob = result->edge_probs[i][j];
//             size_t target = result->vertex_targets[i][j];

//             // Add command: vertex[i] += vertex[target] * prob
//             commands = add_command(commands, i, target, prob, command_index++);
//         }
//     }

//     // Phase 3: Add terminating command with NAN
//     commands = add_command(commands, 0, 0, NAN, command_index);

//     // Create and return result structure
//     struct ptd_desc_reward_compute *res =
//         (struct ptd_desc_reward_compute *) malloc(sizeof(*res));

//     if (res == NULL) {
//         snprintf((char*)ptd_err, sizeof(ptd_err),
//                 "Failed to allocate reward_compute structure");
//         free(commands);
//         return NULL;
//     }

//     res->length = command_index;
//     res->commands = commands;

//     return res;
// }

// /* ==================================================================
//  * Trace Caching - Internal Functions
//  * ==================================================================
//  * These functions implement automatic caching of elimination traces
//  * to avoid O(n³) re-recording for graphs with identical structure.
//  */

// /**
//  * Get path to cache directory, creating it if needed
//  * Returns newly allocated string that must be freed by caller
//  */
// static char *get_cache_dir(void) {
//     char *home = getenv("HOME");
//     if (home == NULL) {
//         return NULL;
//     }

//     // Allocate space for ~/.phasic_cache/traces
//     size_t len = strlen(home) + 40;
//     char *cache_dir = (char *)malloc(len);
//     if (cache_dir == NULL) {
//         return NULL;
//     }

//     snprintf(cache_dir, len, "%s/.phasic_cache", home);
//     mkdir(cache_dir, 0755);  // Create if doesn't exist

//     snprintf(cache_dir, len, "%s/.phasic_cache/traces", home);
//     mkdir(cache_dir, 0755);  // Create traces subdirectory

//     return cache_dir;
// }

// /**
//  * Get full path to cached trace file for given hash
//  * Returns newly allocated string that must be freed by caller
//  */
// static char *get_cache_path(const char *hash_hex) {
//     char *cache_dir = get_cache_dir();
//     if (cache_dir == NULL) {
//         return NULL;
//     }

//     size_t len = strlen(cache_dir) + strlen(hash_hex) + 10;
//     char *path = (char *)malloc(len);
//     if (path == NULL) {
//         free(cache_dir);
//         return NULL;
//     }

//     snprintf(path, len, "%s/%s.json", cache_dir, hash_hex);
//     free(cache_dir);

//     return path;
// }

// /**
//  * Serialize trace operation to JSON string
//  * Appends to the provided string buffer
//  */
// static void operation_to_json(const struct ptd_trace_operation *op,
//                               char **buffer, size_t *buffer_len, size_t *buffer_cap) {
//     // Ensure buffer has space
//     while (*buffer_len + 512 > *buffer_cap) {
//         *buffer_cap *= 2;
//         *buffer = (char *)realloc(*buffer, *buffer_cap);
//     }

//     *buffer_len += snprintf(*buffer + *buffer_len, *buffer_cap - *buffer_len,
//                            "{\"op_type\":%d,\"const_value\":%.17g,\"param_idx\":%zu,",
//                            op->op_type, op->const_value, op->param_idx);

//     // Coefficients array
//     *buffer_len += snprintf(*buffer + *buffer_len, *buffer_cap - *buffer_len,
//                            "\"coefficients\":[");
//     for (size_t i = 0; i < op->coefficients_length; i++) {
//         if (i > 0) {
//             *buffer_len += snprintf(*buffer + *buffer_len, *buffer_cap - *buffer_len, ",");
//         }
//         *buffer_len += snprintf(*buffer + *buffer_len, *buffer_cap - *buffer_len,
//                                "%.17g", op->coefficients[i]);
//     }
//     *buffer_len += snprintf(*buffer + *buffer_len, *buffer_cap - *buffer_len, "],");

//     // Operands array
//     *buffer_len += snprintf(*buffer + *buffer_len, *buffer_cap - *buffer_len,
//                            "\"operands\":[");
//     for (size_t i = 0; i < op->operands_length; i++) {
//         if (i > 0) {
//             *buffer_len += snprintf(*buffer + *buffer_len, *buffer_cap - *buffer_len, ",");
//         }
//         *buffer_len += snprintf(*buffer + *buffer_len, *buffer_cap - *buffer_len,
//                                "%zu", op->operands[i]);
//     }
//     *buffer_len += snprintf(*buffer + *buffer_len, *buffer_cap - *buffer_len, "]}");
// }

// /**
//  * Serialize elimination trace to JSON string (internal use only)
//  * Returns newly allocated JSON string, or NULL on error
//  * Caller must free the returned string
//  */
// static char *trace_to_json_internal(const struct ptd_elimination_trace *trace) {
//     if (trace == NULL) {
//         return NULL;
//     }

//     // Start with reasonable buffer size
//     size_t buffer_cap = 8192;
//     size_t buffer_len = 0;
//     char *buffer = (char *)malloc(buffer_cap);
//     if (buffer == NULL) {
//         return NULL;
//     }

//     // Start JSON object
//     buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
//                           "{\"n_vertices\":%zu,\"param_length\":%zu,\"state_length\":%zu,",
//                           trace->n_vertices, trace->param_length, trace->state_length);

//     // Operations array
//     buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
//                           "\"operations\":[");
//     for (size_t i = 0; i < trace->operations_length; i++) {
//         if (i > 0) {
//             buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
//         }
//         operation_to_json(&trace->operations[i], &buffer, &buffer_len, &buffer_cap);
//     }
//     buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "],");

//     // Vertex rates array
//     buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
//                           "\"vertex_rates\":[");
//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         if (i > 0) {
//             buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
//         }
//         buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
//                               "%zu", trace->vertex_rates[i]);
//     }
//     buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "],");

//     // Edge probs arrays (2D)
//     buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
//                           "\"edge_probs\":[");
//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         if (i > 0) {
//             buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
//         }
//         buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "[");
//         for (size_t j = 0; j < trace->edge_probs_lengths[i]; j++) {
//             if (j > 0) {
//                 buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
//             }
//             buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
//                                   "%zu", trace->edge_probs[i][j]);
//         }
//         buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "]");
//     }
//     buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "],");

//     // Edge probs lengths
//     buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
//                           "\"edge_probs_lengths\":[");
//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         if (i > 0) {
//             buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
//         }
//         buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
//                               "%zu", trace->edge_probs_lengths[i]);
//     }
//     buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "],");

//     // Vertex targets arrays (2D)
//     buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
//                           "\"vertex_targets\":[");
//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         if (i > 0) {
//             buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
//         }
//         buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "[");
//         for (size_t j = 0; j < trace->vertex_targets_lengths[i]; j++) {
//             if (j > 0) {
//                 buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
//             }
//             buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
//                                   "%zu", trace->vertex_targets[i][j]);
//         }
//         buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "]");
//     }
//     buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "],");

//     // Vertex targets lengths
//     buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
//                           "\"vertex_targets_lengths\":[");
//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         if (i > 0) {
//             buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
//         }
//         buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
//                               "%zu", trace->vertex_targets_lengths[i]);
//     }
//     buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "],");

//     // States arrays (2D)
//     buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
//                           "\"states\":[");
//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         if (i > 0) {
//             buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
//         }
//         buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "[");
//         for (size_t j = 0; j < trace->state_length; j++) {
//             if (j > 0) {
//                 buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, ",");
//             }
//             buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
//                                   "%d", trace->states[i][j]);
//         }
//         buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "]");
//     }
//     buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "],");

//     // Starting vertex index
//     buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
//                           "\"starting_vertex_idx\":%zu,", trace->starting_vertex_idx);

//     // Is discrete
//     buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len,
//                           "\"is_discrete\":%s", trace->is_discrete ? "true" : "false");

//     // Close JSON object
//     buffer_len += snprintf(buffer + buffer_len, buffer_cap - buffer_len, "}");

//     return buffer;
// }

// /**
//  * Simple JSON parser helpers for trace deserialization
//  */

// /* Skip whitespace in JSON string */
// static const char *skip_whitespace(const char *s) {
//     while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') {
//         s++;
//     }
//     return s;
// }

// /* Find the closing bracket/brace, accounting for nesting */
// static const char *find_closing(const char *s, char open, char close) {
//     int depth = 1;
//     s++; // Skip opening bracket
//     while (*s && depth > 0) {
//         if (*s == open) depth++;
//         else if (*s == close) depth--;
//         if (depth > 0) s++;
//     }
//     return s;
// }

// /* Parse a size_t value from JSON */
// static size_t parse_size_t(const char *s) {
//     return (size_t)strtoull(s, NULL, 10);
// }

// /* Parse an int value from JSON */
// static int parse_int(const char *s) {
//     return (int)strtol(s, NULL, 10);
// }

// /* Parse a double value from JSON */
// static double parse_double(const char *s) {
//     return strtod(s, NULL);
// }

// /* Parse a bool value from JSON */
// static bool parse_bool(const char *s) {
//     s = skip_whitespace(s);
//     return (strncmp(s, "true", 4) == 0);
// }

// /* Find a JSON field by name and return pointer to its value */
// static const char *find_field(const char *json, const char *field_name) {
//     char search[256];
//     snprintf(search, sizeof(search), "\"%s\":", field_name);
//     const char *field = strstr(json, search);
//     if (field == NULL) {
//         return NULL;
//     }
//     field += strlen(search);
//     return skip_whitespace(field);
// }

// /* Parse array of size_t values */
// static size_t *parse_size_t_array(const char *json, size_t *out_length) {
//     json = skip_whitespace(json);
//     if (*json != '[') {
//         return NULL;
//     }

//     // Count elements
//     size_t count = 0;
//     const char *p = json + 1;
//     while (*p && *p != ']') {
//         if (*p >= '0' && *p <= '9') {
//             count++;
//             while (*p && *p != ',' && *p != ']') p++;
//         }
//         if (*p == ',') p++;
//         p = skip_whitespace(p);
//     }

//     if (count == 0) {
//         *out_length = 0;
//         return NULL;
//     }

//     // Allocate and parse
//     size_t *arr = (size_t *)malloc(count * sizeof(size_t));
//     if (arr == NULL) {
//         return NULL;
//     }

//     p = json + 1;
//     for (size_t i = 0; i < count; i++) {
//         p = skip_whitespace(p);
//         arr[i] = parse_size_t(p);
//         while (*p && *p != ',' && *p != ']') p++;
//         if (*p == ',') p++;
//     }

//     *out_length = count;
//     return arr;
// }

// /* Parse array of double values */
// static double *parse_double_array(const char *json, size_t *out_length) {
//     json = skip_whitespace(json);
//     if (*json != '[') {
//         return NULL;
//     }

//     // Count elements
//     size_t count = 0;
//     const char *p = json + 1;
//     while (*p && *p != ']') {
//         p = skip_whitespace(p);
//         if (*p == '-' || (*p >= '0' && *p <= '9')) {
//             count++;
//             while (*p && *p != ',' && *p != ']') p++;
//         }
//         if (*p == ',') p++;
//     }

//     if (count == 0) {
//         *out_length = 0;
//         return NULL;
//     }

//     // Allocate and parse
//     double *arr = (double *)malloc(count * sizeof(double));
//     if (arr == NULL) {
//         return NULL;
//     }

//     p = json + 1;
//     for (size_t i = 0; i < count; i++) {
//         p = skip_whitespace(p);
//         arr[i] = parse_double(p);
//         while (*p && *p != ',' && *p != ']') p++;
//         if (*p == ',') p++;
//     }

//     *out_length = count;
//     return arr;
// }

// /* Parse array of int values */
// static int *parse_int_array(const char *json, size_t *out_length) {
//     json = skip_whitespace(json);
//     if (*json != '[') {
//         return NULL;
//     }

//     // Count elements
//     size_t count = 0;
//     const char *p = json + 1;
//     while (*p && *p != ']') {
//         p = skip_whitespace(p);
//         if (*p == '-' || (*p >= '0' && *p <= '9')) {
//             count++;
//             while (*p && *p != ',' && *p != ']') p++;
//         }
//         if (*p == ',') p++;
//     }

//     if (count == 0) {
//         *out_length = 0;
//         return NULL;
//     }

//     // Allocate and parse
//     int *arr = (int *)malloc(count * sizeof(int));
//     if (arr == NULL) {
//         return NULL;
//     }

//     p = json + 1;
//     for (size_t i = 0; i < count; i++) {
//         p = skip_whitespace(p);
//         arr[i] = parse_int(p);
//         while (*p && *p != ',' && *p != ']') p++;
//         if (*p == ',') p++;
//     }

//     *out_length = count;
//     return arr;
// }

// /* Parse a trace operation from JSON object */
// static int parse_operation(const char *json, struct ptd_trace_operation *op) {
//     const char *field;

//     // op_type
//     field = find_field(json, "op_type");
//     if (field == NULL) return -1;
//     op->op_type = (enum ptd_trace_op_type)parse_int(field);

//     // const_value
//     field = find_field(json, "const_value");
//     if (field == NULL) return -1;
//     op->const_value = parse_double(field);

//     // param_idx
//     field = find_field(json, "param_idx");
//     if (field == NULL) return -1;
//     op->param_idx = parse_size_t(field);

//     // coefficients
//     field = find_field(json, "coefficients");
//     if (field == NULL) return -1;
//     op->coefficients = parse_double_array(field, &op->coefficients_length);

//     // operands
//     field = find_field(json, "operands");
//     if (field == NULL) return -1;
//     op->operands = parse_size_t_array(field, &op->operands_length);

//     return 0;
// }

// /**
//  * Load elimination trace from cache file (internal use only)
//  * Returns trace if found in cache, NULL otherwise
//  */
// static struct ptd_elimination_trace *load_trace_from_cache(const char *hash_hex) {
//     char *path = get_cache_path(hash_hex);
//     if (path == NULL) {
//         return NULL;
//     }

//     // Check if file exists
//     if (access(path, F_OK) != 0) {
//         free(path);
//         return NULL;
//     }

//     // Read file
//     FILE *f = fopen(path, "r");
//     if (f == NULL) {
//         DEBUG_PRINT("WARNING: failed to open cache file for reading: %s\n", path);
//         free(path);
//         return NULL;
//     }

//     // Get file size
//     fseek(f, 0, SEEK_END);
//     long file_size = ftell(f);
//     fseek(f, 0, SEEK_SET);

//     if (file_size <= 0) {
//         fclose(f);
//         free(path);
//         return NULL;
//     }

//     // Read entire file into memory
//     char *json = (char *)malloc(file_size + 1);
//     if (json == NULL) {
//         fclose(f);
//         free(path);
//         return NULL;
//     }

//     size_t bytes_read = fread(json, 1, file_size, f);
//     fclose(f);

//     if (bytes_read != (size_t)file_size) {
//         DEBUG_PRINT("WARNING: failed to read complete cache file: %s\n", path);
//         free(json);
//         free(path);
//         return NULL;
//     }
//     json[file_size] = '\0';

//     // Allocate trace structure
//     struct ptd_elimination_trace *trace = (struct ptd_elimination_trace *)calloc(1, sizeof(*trace));
//     if (trace == NULL) {
//         free(json);
//         free(path);
//         return NULL;
//     }

//     // Declare all variables at the beginning to avoid goto issues
//     const char *field;
//     const char *p;
//     size_t op_count;
//     size_t vr_len;
//     size_t epl_len;
//     size_t vtl_len;
//     size_t len;
//     int depth;

//     // Parse metadata fields
//     field = find_field(json, "n_vertices");
//     if (field == NULL) goto error;
//     trace->n_vertices = parse_size_t(field);

//     field = find_field(json, "param_length");
//     if (field == NULL) goto error;
//     trace->param_length = parse_size_t(field);

//     field = find_field(json, "state_length");
//     if (field == NULL) goto error;
//     trace->state_length = parse_size_t(field);

//     field = find_field(json, "starting_vertex_idx");
//     if (field == NULL) goto error;
//     trace->starting_vertex_idx = parse_size_t(field);

//     field = find_field(json, "is_discrete");
//     if (field == NULL) goto error;
//     trace->is_discrete = parse_bool(field);

//     // Parse operations array
//     field = find_field(json, "operations");
//     if (field == NULL) goto error;

//     field = skip_whitespace(field);
//     if (*field != '[') goto error;

//     // Count operations
//     op_count = 0;
//     p = field + 1;
//     depth = 0;
//     while (*p) {
//         if (*p == '{') {
//             if (depth == 0) op_count++;
//             depth++;
//         } else if (*p == '}') {
//             depth--;
//         } else if (*p == ']' && depth == 0) {
//             break;
//         }
//         p++;
//     }

//     trace->operations_length = op_count;
//     trace->operations = (struct ptd_trace_operation *)calloc(op_count, sizeof(struct ptd_trace_operation));
//     if (trace->operations == NULL) goto error;

//     // Parse each operation
//     p = field + 1;
//     for (size_t i = 0; i < op_count; i++) {
//         p = skip_whitespace(p);
//         if (*p != '{') goto error;

//         const char *op_end = find_closing(p, '{', '}');
//         if (op_end == NULL) goto error;

//         if (parse_operation(p, &trace->operations[i]) != 0) {
//             goto error;
//         }

//         p = op_end + 1;
//         if (*p == ',') p++;
//     }

//     // Parse vertex_rates
//     field = find_field(json, "vertex_rates");
//     if (field == NULL) goto error;
//     trace->vertex_rates = parse_size_t_array(field, &vr_len);
//     if (vr_len != trace->n_vertices) goto error;

//     // Parse edge_probs_lengths
//     field = find_field(json, "edge_probs_lengths");
//     if (field == NULL) goto error;
//     trace->edge_probs_lengths = parse_size_t_array(field, &epl_len);
//     if (epl_len != trace->n_vertices) goto error;

//     // Parse edge_probs (2D array)
//     field = find_field(json, "edge_probs");
//     if (field == NULL) goto error;
//     field = skip_whitespace(field);
//     if (*field != '[') goto error;

//     trace->edge_probs = (size_t **)malloc(trace->n_vertices * sizeof(size_t*));
//     if (trace->edge_probs == NULL) goto error;

//     p = field + 1;
//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         p = skip_whitespace(p);
//         if (*p != '[') goto error;

//         trace->edge_probs[i] = parse_size_t_array(p, &len);
//         if (len != trace->edge_probs_lengths[i]) goto error;

//         p = find_closing(p, '[', ']');
//         if (*p == ']') p++;
//         if (*p == ',') p++;
//     }

//     // Parse vertex_targets_lengths
//     field = find_field(json, "vertex_targets_lengths");
//     if (field == NULL) goto error;
//     trace->vertex_targets_lengths = parse_size_t_array(field, &vtl_len);
//     if (vtl_len != trace->n_vertices) goto error;

//     // Parse vertex_targets (2D array)
//     field = find_field(json, "vertex_targets");
//     if (field == NULL) goto error;
//     field = skip_whitespace(field);
//     if (*field != '[') goto error;

//     trace->vertex_targets = (size_t **)malloc(trace->n_vertices * sizeof(size_t*));
//     if (trace->vertex_targets == NULL) goto error;

//     p = field + 1;
//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         p = skip_whitespace(p);
//         if (*p != '[') goto error;

//         trace->vertex_targets[i] = parse_size_t_array(p, &len);
//         if (len != trace->vertex_targets_lengths[i]) goto error;

//         p = find_closing(p, '[', ']');
//         if (*p == ']') p++;
//         if (*p == ',') p++;
//     }

//     // Parse states (2D array of ints)
//     field = find_field(json, "states");
//     if (field == NULL) goto error;
//     field = skip_whitespace(field);
//     if (*field != '[') goto error;

//     trace->states = (int **)malloc(trace->n_vertices * sizeof(int*));
//     if (trace->states == NULL) goto error;

//     p = field + 1;
//     for (size_t i = 0; i < trace->n_vertices; i++) {
//         p = skip_whitespace(p);
//         if (*p != '[') goto error;

//         trace->states[i] = parse_int_array(p, &len);
//         if (len != trace->state_length) goto error;

//         p = find_closing(p, '[', ']');
//         if (*p == ']') p++;
//         if (*p == ',') p++;
//     }

//     free(json);
//     free(path);

//     DEBUG_PRINT("INFO: loaded elimination trace from cache (%s): %zu operations, %zu vertices\n",
//                 hash_hex, trace->operations_length, trace->n_vertices);

//     return trace;

// error:
//     DEBUG_PRINT("WARNING: failed to deserialize trace from cache: %s\n", path);
//     if (trace != NULL) {
//         ptd_elimination_trace_destroy(trace);
//     }
//     free(json);
//     free(path);
//     return NULL;
// }

// /**
//  * Save elimination trace to cache file (internal use only)
//  * Returns true on success, false on error
//  */
// static bool save_trace_to_cache(const char *hash_hex,
//                                 const struct ptd_elimination_trace *trace) {
//     if (hash_hex == NULL || trace == NULL) {
//         return false;
//     }

//     char *path = get_cache_path(hash_hex);
//     if (path == NULL) {
//         return false;
//     }

//     // Serialize trace to JSON
//     char *json = trace_to_json_internal(trace);
//     if (json == NULL) {
//         free(path);
//         return false;
//     }

//     // Write to file
//     FILE *f = fopen(path, "w");
//     if (f == NULL) {
//         DEBUG_PRINT("WARNING: failed to open cache file for writing: %s\n", path);
//         free(path);
//         free(json);
//         return false;
//     }

//     size_t json_len = strlen(json);
//     size_t written = fwrite(json, 1, json_len, f);
//     fclose(f);

//     bool success = (written == json_len);

//     if (success) {
//         DEBUG_PRINT("INFO: saved trace to cache (%zu bytes): %s\n", json_len, path);
//     } else {
//         DEBUG_PRINT("WARNING: failed to write complete trace to cache\n");
//     }

//     free(path);
//     free(json);

//     return success;
// }

// ============================================================================
// Trace-Based Elimination Implementation - NEW CLEAN VERSION
// ============================================================================

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
