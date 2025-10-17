/**
 * @file ptdalgorithms_hash.h
 * @brief Graph content hashing for symbolic DAG caching
 *
 * This module provides content-addressed hashing of phase-type distribution graphs.
 * The hash is based solely on graph structure and parameterization pattern, not
 * on actual parameter values, enabling:
 *
 * - Cache lookup for previously computed symbolic DAGs
 * - Distributed sharing of pre-computed models
 * - Content-addressed model repositories
 *
 * The hashing algorithm uses a modified Weisfeiler-Lehman approach that produces
 * consistent hashes for structurally identical graphs while being robust to
 * vertex index permutations.
 */

#ifndef PTDALGORITHMS_HASH_H
#define PTDALGORITHMS_HASH_H

#include "ptdalgorithms.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Hash result structure containing multiple hash representations
 */
struct ptd_hash_result {
    uint64_t hash64;              // 64-bit hash (fast comparison)
    uint8_t hash_full[32];        // Full SHA-256 hash (collision-resistant)
    char hash_hex[65];            // Hexadecimal string representation (null-terminated)
};

/**
 * @brief Compute content hash of a graph structure
 *
 * Computes a deterministic hash based on:
 * - Graph topology (vertices and edges)
 * - Edge types (regular vs parameterized)
 * - Parameter structure (edge_state coefficients)
 * - State space dimensions
 * - Graph properties (discrete vs continuous)
 *
 * The hash is INDEPENDENT of:
 * - Actual edge weights (those are parameters)
 * - Vertex ordering in memory (uses canonical ordering)
 * - Initial probability distribution
 *
 * This enables cache lookup for symbolic DAGs that have the same structure
 * but different parameter values.
 *
 * @param graph Phase-type distribution graph to hash
 * @return Hash result structure (caller must free with ptd_hash_destroy)
 * @return NULL if graph is NULL or invalid
 *
 * @note Hash computation is O(V log V + E) where V = vertices, E = edges
 * @note Thread-safe (uses no global state)
 * @note The hash is consistent across different runs and platforms
 *
 * @example
 * struct ptd_graph *g = ptd_graph_create(2);
 * // ... build graph ...
 *
 * struct ptd_hash_result *hash = ptd_graph_content_hash(g);
 * printf("Graph hash: %s\n", hash->hash_hex);
 *
 * // Check cache
 * if (symbolic_cache_exists(hash->hash_hex)) {
 *     // Load from cache
 * }
 *
 * ptd_hash_destroy(hash);
 */
struct ptd_hash_result *ptd_graph_content_hash(const struct ptd_graph *graph);

/**
 * @brief Compare two hash results for equality
 *
 * @param hash1 First hash result
 * @param hash2 Second hash result
 * @return true if hashes are identical, false otherwise
 *
 * @note Compares 64-bit hashes first for speed, then full hash if needed
 */
bool ptd_hash_equal(const struct ptd_hash_result *hash1,
                    const struct ptd_hash_result *hash2);

/**
 * @brief Destroy hash result and free memory
 *
 * @param hash Hash result to destroy (may be NULL)
 */
void ptd_hash_destroy(struct ptd_hash_result *hash);

/**
 * @brief Compute hash from serialized graph JSON
 *
 * Useful for hashing graphs that have been serialized without reconstructing
 * the full C graph structure.
 *
 * @param json_str Serialized graph in JSON format
 * @return Hash result structure (caller must free with ptd_hash_destroy)
 * @return NULL if json_str is NULL or invalid JSON
 */
struct ptd_hash_result *ptd_graph_hash_from_json(const char *json_str);

/**
 * @brief Create hash result from hexadecimal string
 *
 * Useful for loading cached symbolic DAGs by hash key.
 *
 * @param hex_str Hexadecimal hash string (64 characters)
 * @return Hash result structure (caller must free with ptd_hash_destroy)
 * @return NULL if hex_str is NULL or invalid format
 *
 * @example
 * const char *cached_hash = "a3f2e9c8b1d4...";
 * struct ptd_hash_result *hash = ptd_hash_from_hex(cached_hash);
 * // Use hash for cache lookup...
 * ptd_hash_destroy(hash);
 */
struct ptd_hash_result *ptd_hash_from_hex(const char *hex_str);

/**
 * @brief Compute hash of parameterized edge structure
 *
 * Internal function for hashing individual parameterized edges.
 * Exported for testing purposes.
 *
 * @param edge Parameterized edge to hash
 * @param state_length Length of edge state vector
 * @return 64-bit hash of edge structure
 */
uint64_t ptd_hash_parameterized_edge(const struct ptd_edge_parameterized *edge,
                                     size_t state_length);

#ifdef __cplusplus
}
#endif

#endif // PTDALGORITHMS_HASH_H
