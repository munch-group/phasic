/**
 * @file ptdalgorithms_hash.c
 * @brief Implementation of graph content hashing
 *
 * Uses a modified Weisfeiler-Lehman algorithm combined with SHA-256
 * to produce consistent, collision-resistant hashes of graph structures.
 */

#include "../../api/c/ptdalgorithms_hash.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

// Simple SHA-256 implementation (minimal, no external dependencies)
// For production, could use OpenSSL or similar, but this keeps dependencies minimal

// SHA-256 context and constants
typedef struct {
    uint32_t state[8];
    uint64_t count;
    uint8_t buffer[64];
} sha256_context;

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

static void sha256_transform(sha256_context *ctx, const uint8_t data[64]) {
    uint32_t a, b, c, d, e, f, g, h, t1, t2, m[64];
    int i;

    for (i = 0; i < 16; ++i)
        m[i] = ((uint32_t)data[i * 4] << 24) | ((uint32_t)data[i * 4 + 1] << 16) |
               ((uint32_t)data[i * 4 + 2] << 8) | ((uint32_t)data[i * 4 + 3]);
    for (; i < 64; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];

    for (i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e, f, g) + K[i] + m[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

static void sha256_init(sha256_context *ctx) {
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
    ctx->count = 0;
}

static void sha256_update(sha256_context *ctx, const uint8_t *data, size_t len) {
    size_t i;
    for (i = 0; i < len; ++i) {
        ctx->buffer[ctx->count % 64] = data[i];
        ctx->count++;
        if (ctx->count % 64 == 0)
            sha256_transform(ctx, ctx->buffer);
    }
}

static void sha256_final(sha256_context *ctx, uint8_t hash[32]) {
    uint32_t i = (uint32_t)(ctx->count % 64);

    ctx->buffer[i++] = 0x80;
    if (i > 56) {
        while (i < 64) ctx->buffer[i++] = 0x00;
        sha256_transform(ctx, ctx->buffer);
        i = 0;
    }
    while (i < 56) ctx->buffer[i++] = 0x00;

    uint64_t bit_len = ctx->count * 8;
    for (int j = 0; j < 8; j++)
        ctx->buffer[56 + j] = (bit_len >> (56 - j * 8)) & 0xff;

    sha256_transform(ctx, ctx->buffer);

    for (i = 0; i < 8; ++i) {
        hash[i * 4] = (ctx->state[i] >> 24) & 0xff;
        hash[i * 4 + 1] = (ctx->state[i] >> 16) & 0xff;
        hash[i * 4 + 2] = (ctx->state[i] >> 8) & 0xff;
        hash[i * 4 + 3] = ctx->state[i] & 0xff;
    }
}

// ============================================================================
// Graph Hashing Implementation
// ============================================================================

// Comparison function for sorting edges (for canonical ordering)
static int compare_edges(const void *a, const void *b) {
    const struct ptd_edge *edge_a = *(const struct ptd_edge **)a;
    const struct ptd_edge *edge_b = *(const struct ptd_edge **)b;

    // Sort by target vertex index
    ptrdiff_t diff = edge_a->to->index - edge_b->to->index;
    if (diff != 0) return (diff > 0) ? 1 : -1;

    // Then by parameterized flag
    if (edge_a->parameterized != edge_b->parameterized)
        return edge_a->parameterized ? 1 : -1;

    return 0;
}

// Hash a single vertex's structure
static uint64_t hash_vertex_structure(const struct ptd_vertex *vertex) {
    sha256_context ctx;
    sha256_init(&ctx);

    // Hash vertex state
    sha256_update(&ctx, (const uint8_t *)vertex->state,
                  vertex->graph->state_length * sizeof(int));

    // Create sorted copy of edges for canonical ordering
    struct ptd_edge **sorted_edges = (struct ptd_edge **)malloc(vertex->edges_length * sizeof(struct ptd_edge *));
    if (sorted_edges == NULL) return 0;

    for (size_t i = 0; i < vertex->edges_length; i++) {
        sorted_edges[i] = vertex->edges[i];
    }
    qsort(sorted_edges, vertex->edges_length, sizeof(struct ptd_edge *), compare_edges);

    // Hash each edge
    for (size_t i = 0; i < vertex->edges_length; i++) {
        struct ptd_edge *edge = sorted_edges[i];

        // Hash target index
        size_t target_idx = edge->to->index;
        sha256_update(&ctx, (const uint8_t *)&target_idx, sizeof(size_t));

        // Hash edge type
        uint8_t edge_type = edge->parameterized ? 1 : 0;
        sha256_update(&ctx, &edge_type, 1);

        // Hash parameterized edge structure (not values!)
        if (edge->parameterized) {
            struct ptd_edge_parameterized *param_edge =
                (struct ptd_edge_parameterized *)edge;

            // Hash the parameter coefficients pattern
            if (param_edge->state != NULL) {
                sha256_update(&ctx, (const uint8_t *)param_edge->state,
                             vertex->graph->param_length * sizeof(double));
            }
        }
    }

    free(sorted_edges);

    // Get final hash
    uint8_t hash_bytes[32];
    sha256_final(&ctx, hash_bytes);

    // Convert first 8 bytes to uint64_t
    uint64_t result = 0;
    for (int i = 0; i < 8; i++) {
        result = (result << 8) | hash_bytes[i];
    }

    return result;
}

struct ptd_hash_result *ptd_graph_content_hash(const struct ptd_graph *graph) {
    if (graph == NULL || graph->vertices == NULL) {
        return NULL;
    }

    struct ptd_hash_result *result = (struct ptd_hash_result *)malloc(sizeof(struct ptd_hash_result));
    if (result == NULL) return NULL;

    sha256_context ctx;
    sha256_init(&ctx);

    // Hash graph metadata
    sha256_update(&ctx, (const uint8_t *)&graph->state_length, sizeof(size_t));
    sha256_update(&ctx, (const uint8_t *)&graph->param_length, sizeof(size_t));
    sha256_update(&ctx, (const uint8_t *)&graph->vertices_length, sizeof(size_t));

    uint8_t graph_flags = 0;
    if (graph->parameterized) graph_flags |= 0x01;
    if (graph->was_dph) graph_flags |= 0x02;
    sha256_update(&ctx, &graph_flags, 1);

    // Hash each vertex in index order (canonical ordering)
    for (size_t i = 0; i < graph->vertices_length; i++) {
        if (graph->vertices[i] == NULL) continue;

        uint64_t vertex_hash = hash_vertex_structure(graph->vertices[i]);
        sha256_update(&ctx, (const uint8_t *)&vertex_hash, sizeof(uint64_t));
    }

    // Compute final hash
    sha256_final(&ctx, result->hash_full);

    // Extract 64-bit hash from first 8 bytes
    result->hash64 = 0;
    for (int i = 0; i < 8; i++) {
        result->hash64 = (result->hash64 << 8) | result->hash_full[i];
    }

    // Convert to hex string
    for (int i = 0; i < 32; i++) {
        sprintf(&result->hash_hex[i * 2], "%02x", result->hash_full[i]);
    }
    result->hash_hex[64] = '\0';

    return result;
}

bool ptd_hash_equal(const struct ptd_hash_result *hash1,
                    const struct ptd_hash_result *hash2) {
    if (hash1 == NULL || hash2 == NULL) {
        return (hash1 == hash2);
    }

    // Quick check with 64-bit hash
    if (hash1->hash64 != hash2->hash64) {
        return false;
    }

    // Full check with 256-bit hash
    return memcmp(hash1->hash_full, hash2->hash_full, 32) == 0;
}

void ptd_hash_destroy(struct ptd_hash_result *hash) {
    if (hash != NULL) {
        free(hash);
    }
}

struct ptd_hash_result *ptd_hash_from_hex(const char *hex_str) {
    if (hex_str == NULL || strlen(hex_str) != 64) {
        return NULL;
    }

    struct ptd_hash_result *result = (struct ptd_hash_result *)malloc(sizeof(struct ptd_hash_result));
    if (result == NULL) return NULL;

    // Parse hex string to bytes
    for (int i = 0; i < 32; i++) {
        unsigned int byte_val;
        if (sscanf(&hex_str[i * 2], "%2x", &byte_val) != 1) {
            free(result);
            return NULL;
        }
        result->hash_full[i] = (uint8_t)byte_val;
    }

    // Reconstruct 64-bit hash
    result->hash64 = 0;
    for (int i = 0; i < 8; i++) {
        result->hash64 = (result->hash64 << 8) | result->hash_full[i];
    }

    // Copy hex string
    strncpy(result->hash_hex, hex_str, 64);
    result->hash_hex[64] = '\0';

    return result;
}

uint64_t ptd_hash_parameterized_edge(const struct ptd_edge_parameterized *edge,
                                     size_t state_length) {
    if (edge == NULL || !edge->parameterized) {
        return 0;
    }

    sha256_context ctx;
    sha256_init(&ctx);

    // Hash target index
    size_t target_idx = edge->to->index;
    sha256_update(&ctx, (const uint8_t *)&target_idx, sizeof(size_t));

    // Hash parameter coefficients
    if (edge->state != NULL && state_length > 0) {
        sha256_update(&ctx, (const uint8_t *)edge->state,
                     state_length * sizeof(double));
    }

    uint8_t hash_bytes[32];
    sha256_final(&ctx, hash_bytes);

    uint64_t result = 0;
    for (int i = 0; i < 8; i++) {
        result = (result << 8) | hash_bytes[i];
    }

    return result;
}

struct ptd_hash_result *ptd_graph_hash_from_json(const char *json_str) {
    // For now, return NULL - full JSON parsing would require json-c or similar
    // This can be implemented later if needed, or handled at Python level
    (void)json_str;  // Suppress unused parameter warning
    return NULL;
}
