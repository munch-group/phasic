#ifndef MARKOV_CACHE_INTERNAL_H
#define MARKOV_CACHE_INTERNAL_H

#include <markov_cache/markov_cache.h>
#include <sqlite3.h>
#include <zstd.h>
#include <sys/stat.h>

/* ============================================================================
 * Internal Cache Structure
 * ============================================================================ */

typedef struct LRUCache LRUCache;

struct PersistentCache {
    char cache_dir[512];
    sqlite3 *db;
    ZSTD_CDict *cdict;
    ZSTD_DDict *ddict;
    LRUCache *hot_cache;
    
    // Statistics
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t total_chunks;
};

/* ============================================================================
 * Serialization
 * ============================================================================ */

/**
 * Serialize CSR graph to buffer with delta encoding
 */
size_t serialize_csr(const CSRGraph *csr, uint8_t **out_buffer);

/**
 * Deserialize CSR graph from buffer
 */
CSRGraph* deserialize_csr(const uint8_t *buffer, size_t size);

/**
 * Compress data using Zstandard with dictionary
 */
size_t compress_data(const uint8_t *data, size_t size,
                    ZSTD_CDict *dict,
                    uint8_t **out_compressed);

/**
 * Decompress data using Zstandard with dictionary
 */
size_t decompress_data(const uint8_t *compressed, size_t compressed_size,
                      ZSTD_DDict *dict,
                      uint8_t **out_data);

/* ============================================================================
 * Variable-Length Integer Encoding
 * ============================================================================ */

/**
 * Write variable-length integer (LEB128 encoding)
 */
size_t write_varint(uint8_t *buffer, uint64_t value);

/**
 * Read variable-length integer
 */
size_t read_varint(const uint8_t *buffer, uint64_t *value);

/**
 * Get size needed for varint encoding
 */
size_t varint_size(uint64_t value);

/* ============================================================================
 * LRU Cache
 * ============================================================================ */

typedef struct {
    uint8_t orig_hash[32];
    uint8_t dag_hash[32];
} CacheKey;

LRUCache* lru_create(size_t max_size_bytes);
void lru_free(LRUCache *cache);
CSRGraph* lru_get(LRUCache *cache, const CacheKey *key);
void lru_put(LRUCache *cache, const CacheKey *key, CSRGraph *csr);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Create directory recursively
 */
int mkdir_recursive(const char *path);

/**
 * Convert hash to hex string
 */
void hash_to_hex(const uint8_t *hash, char *hex, size_t hex_size);

/**
 * Get file path for chunk
 */
void get_chunk_path(const char *cache_dir,
                   const uint8_t *orig_hash,
                   const uint8_t *dag_hash,
                   char *path,
                   size_t path_size);

/**
 * Compare two cache keys
 */
int cache_key_compare(const CacheKey *a, const CacheKey *b);

#endif /* MARKOV_CACHE_INTERNAL_H */
