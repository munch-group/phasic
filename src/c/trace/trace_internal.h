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

#ifndef PHASIC_TRACE_INTERNAL_H
#define PHASIC_TRACE_INTERNAL_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <limits.h>

#include "../../../api/c/phasic.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

// Debug printing macro
#ifndef DEBUG_PRINT
#define DEBUG_PRINT(...) do {} while(0)  // Disabled by default
#endif

// ============================================================================
// Cache I/O (trace_cache.c)
// ============================================================================

/**
 * Get the cache directory path for storing elimination traces.
 * Creates ~/.phasic_cache if it doesn't exist.
 *
 * @param buffer Buffer to store the path
 * @param buffer_size Size of the buffer
 * @return 0 on success, -1 on failure
 */
int get_cache_dir(char *buffer, size_t buffer_size);

/**
 * Load an elimination trace from the disk cache.
 *
 * @param hash_hex Hexadecimal hash string identifying the trace
 * @return Loaded trace structure, or NULL if not found or error
 */
struct ptd_elimination_trace *load_trace_from_cache(const char *hash_hex);

/**
 * Save an elimination trace to the disk cache.
 *
 * @param hash_hex Hexadecimal hash string identifying the trace
 * @param trace The trace structure to save
 * @return true on success, false on failure
 */
bool save_trace_to_cache(const char *hash_hex, const struct ptd_elimination_trace *trace);

// ============================================================================
// JSON Serialization (trace_json.c)
// ============================================================================

/**
 * Serialize an elimination trace to JSON format.
 *
 * @param trace The trace to serialize
 * @return JSON string (caller must free), or NULL on error
 */
char *trace_to_json(const struct ptd_elimination_trace *trace);

/**
 * Deserialize an elimination trace from JSON format.
 *
 * @param json JSON string containing the trace
 * @return Trace structure (caller must call ptd_elimination_trace_destroy), or NULL on error
 */
struct ptd_elimination_trace *json_to_trace(const char *json);

// JSON parsing helpers (internal)
const char *skip_whitespace(const char *s);
const char *find_field(const char *json, const char *field_name);
size_t parse_size_t(const char *s);
double parse_double(const char *s);
int parse_int(const char *s);
bool parse_bool(const char *s);
size_t *parse_size_t_array(const char *json, size_t *out_length);
double *parse_double_array(const char *json, size_t *out_length);
int *parse_int_array(const char *json, size_t *out_length);

// ============================================================================
// Trace Operations (trace_operations.c)
// ============================================================================

/**
 * Ensure the trace has capacity for more operations.
 * Reallocates the operations array if necessary.
 *
 * @param trace The trace to check/resize
 * @param required_capacity Minimum required capacity
 * @return 0 on success, -1 on failure
 */
int ensure_operation_capacity(
    struct ptd_elimination_trace *trace,
    size_t required_capacity
);

/**
 * Add a constant operation to the trace.
 *
 * @param trace The trace to add to
 * @param result_idx Index where result will be stored
 * @param value Constant value
 * @return Operation index, or (size_t)-1 on error
 */
size_t add_const_to_trace(
    struct ptd_elimination_trace *trace,
    size_t result_idx,
    double value
);

/**
 * Add a dot product operation to the trace.
 *
 * @param trace The trace to add to
 * @param result_idx Index where result will be stored
 * @param coefficients Array of coefficients
 * @param operands Array of operand indices
 * @param length Length of both arrays
 * @return Operation index, or (size_t)-1 on error
 */
size_t add_dot_to_trace(
    struct ptd_elimination_trace *trace,
    size_t result_idx,
    const double *coefficients,
    const size_t *operands,
    size_t length
);

/**
 * Add a binary operation to the trace (add, multiply, divide).
 *
 * @param trace The trace to add to
 * @param result_idx Index where result will be stored
 * @param op Operation type (OP_ADD, OP_MUL, OP_DIV)
 * @param operand_a First operand index
 * @param operand_b Second operand index
 * @return Operation index, or (size_t)-1 on error
 */
size_t add_binary_op_to_trace(
    struct ptd_elimination_trace *trace,
    size_t result_idx,
    int op,
    size_t operand_a,
    size_t operand_b
);

/**
 * Add an inverse operation (1/x) to the trace.
 *
 * @param trace The trace to add to
 * @param result_idx Index where result will be stored
 * @param operand Operand index
 * @return Operation index, or (size_t)-1 on error
 */
size_t add_inv_to_trace(
    struct ptd_elimination_trace *trace,
    size_t result_idx,
    size_t operand
);

/**
 * Add a sum operation to the trace.
 *
 * @param trace The trace to add to
 * @param result_idx Index where result will be stored
 * @param operands Array of operand indices
 * @param length Number of operands
 * @return Operation index, or (size_t)-1 on error
 */
size_t add_sum_to_trace(
    struct ptd_elimination_trace *trace,
    size_t result_idx,
    const size_t *operands,
    size_t length
);

#endif // PHASIC_TRACE_INTERNAL_H
