#ifndef MARKOV_CACHE_H
#define MARKOV_CACHE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Core Data Structures
 * ============================================================================ */

/**
 * Markov graph representation (original cyclic graph)
 */
typedef struct {
    uint32_t num_nodes;
    uint32_t num_edges;
    uint32_t *edge_src;     // Source node IDs
    uint32_t *edge_dst;     // Destination node IDs
    double *rates;          // Transition rates
} MarkovGraph;

/**
 * DAG result after Gaussian elimination
 */
typedef struct {
    uint32_t num_nodes;
    uint32_t num_edges;
    uint32_t *edge_src;
    uint32_t *edge_dst;
    double *weights;        // Computed weights
    double *moments;        // Computed moments (optional)
    uint32_t num_moments;
} DAGResult;

/**
 * Compressed Sparse Row format (internal representation)
 */
typedef struct {
    uint32_t num_nodes;
    uint32_t num_edges;
    uint32_t *row_ptr;      // Size: num_nodes + 1
    uint32_t *col_idx;      // Size: num_edges
    double *values;         // Size: num_edges
} CSRGraph;

/**
 * Chunk metadata
 */
typedef struct {
    uint8_t orig_hash[32];      // SHA-256 of original graph
    uint8_t dag_hash[32];       // SHA-256 of DAG result
    uint32_t num_nodes;
    uint32_t num_edges;
    uint32_t level;             // Topological level
    uint32_t *node_ids;         // Original node IDs
} ChunkMetadata;

/**
 * Chunk data (loaded from cache)
 */
typedef struct {
    ChunkMetadata metadata;
    CSRGraph *csr_orig;         // Original graph in CSR format
    CSRGraph *csr_dag;          // DAG result in CSR format
    DAGResult *dag_result;      // Full DAG result (if needed)
} ChunkData;

/**
 * Cache query result
 */
typedef struct {
    uint32_t num_cached;            // Number of chunks found in cache
    ChunkData **cached_chunks;
    
    uint32_t num_uncached;          // Number that need computation
    MarkovGraph **uncached_subgraphs;
    uint32_t **uncached_node_ids;
    
    uint32_t *assembly_order;       // Topological order for assembly
} CacheQueryResult;

/**
 * Persistent cache handle
 */
typedef struct PersistentCache PersistentCache;

/* ============================================================================
 * Cache Management API
 * ============================================================================ */

/**
 * Initialize persistent cache
 * 
 * @param cache_dir Directory for cache storage
 * @param hot_cache_size Size of in-memory LRU cache in bytes (0 = no LRU)
 * @return Cache handle or NULL on error
 */
PersistentCache* cache_init(const char *cache_dir, size_t hot_cache_size);

/**
 * Close cache and free resources
 */
void cache_close(PersistentCache *cache);

/**
 * Store a computed chunk in the cache
 * 
 * @param cache Cache handle
 * @param orig Original Markov graph
 * @param dag Computed DAG result
 * @param metadata Chunk metadata (hashes will be computed if not set)
 * @return true on success
 */
bool cache_store(PersistentCache *cache,
                const MarkovGraph *orig,
                const DAGResult *dag,
                const ChunkMetadata *metadata);

/**
 * Load a chunk from cache
 * 
 * @param cache Cache handle
 * @param orig_hash Hash of original graph
 * @param dag_hash Hash of DAG result
 * @return Chunk data or NULL if not found
 */
ChunkData* cache_load(PersistentCache *cache,
                     const uint8_t *orig_hash,
                     const uint8_t *dag_hash);

/**
 * Query cache for a graph and find reusable chunks
 * 
 * @param cache Cache handle
 * @param graph Graph to query
 * @return Query result with cached/uncached chunks
 */
CacheQueryResult* cache_query(PersistentCache *cache,
                              const MarkovGraph *graph);

/**
 * Get cache statistics
 */
void cache_get_stats(PersistentCache *cache,
                    uint64_t *hits,
                    uint64_t *misses,
                    uint64_t *total_chunks);

/**
 * Train compression dictionary from sample graphs
 * 
 * @param cache Cache handle
 * @param samples Array of sample graphs
 * @param num_samples Number of samples
 * @return true on success
 */
bool cache_train_dictionary(PersistentCache *cache,
                            MarkovGraph **samples,
                            uint32_t num_samples);

/* ============================================================================
 * Graph Operations
 * ============================================================================ */

/**
 * Create a new Markov graph
 */
MarkovGraph* markov_graph_create(uint32_t num_nodes);

/**
 * Add an edge to a Markov graph
 */
void markov_graph_add_edge(MarkovGraph *g, uint32_t src, uint32_t dst, double rate);

/**
 * Free a Markov graph
 */
void markov_graph_free(MarkovGraph *g);

/**
 * Create a new DAG result
 */
DAGResult* dag_result_create(uint32_t num_nodes);

/**
 * Free a DAG result
 */
void dag_result_free(DAGResult *g);

/**
 * Compute canonical hash of a graph
 */
void compute_graph_hash(const MarkovGraph *g, uint8_t *hash);

/**
 * Convert Markov graph to CSR format
 */
CSRGraph* markov_to_csr(const MarkovGraph *g);

/**
 * Convert CSR back to Markov graph
 */
MarkovGraph* csr_to_markov(const CSRGraph *csr);

/**
 * Free CSR graph
 */
void csr_free(CSRGraph *csr);

/* ============================================================================
 * Graph Decomposition
 * ============================================================================ */

/**
 * Strongly Connected Component decomposition
 */
typedef struct {
    uint32_t num_sccs;
    uint32_t *node_to_scc;      // Maps node ID to SCC ID
    uint32_t *scc_sizes;        // Size of each SCC
    uint32_t **scc_nodes;       // Nodes in each SCC
} SCCDecomposition;

/**
 * Find strongly connected components
 */
SCCDecomposition* find_sccs(const MarkovGraph *g);

/**
 * Free SCC decomposition
 */
void scc_free(SCCDecomposition *scc);

/**
 * Extract subgraph containing specified nodes
 */
MarkovGraph* extract_subgraph(const MarkovGraph *g,
                             const uint32_t *node_ids,
                             uint32_t num_nodes);

/* ============================================================================
 * Gaussian Elimination (User-Provided)
 * ============================================================================ */

/**
 * Perform Gaussian elimination on a Markov graph to produce a DAG
 * This function should be implemented by the user
 * 
 * @param graph Input Markov graph (may have cycles)
 * @return DAG result after elimination
 */
DAGResult* gaussian_elimination(const MarkovGraph *graph);

/* ============================================================================
 * High-Level Workflow API
 * ============================================================================ */

/**
 * Solve a Markov graph using cache
 * This is the main entry point that:
 * 1. Queries the cache for reusable chunks
 * 2. Computes missing chunks using gaussian_elimination()
 * 3. Stores new results in cache
 * 4. Assembles final solution
 * 
 * @param cache Cache handle
 * @param graph Input graph
 * @param max_chunk_size Maximum size for recursive chunking (0 = no limit)
 * @return Complete DAG solution
 */
DAGResult* solve_with_cache(PersistentCache *cache,
                           const MarkovGraph *graph,
                           uint32_t max_chunk_size);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Free cache query result
 */
void cache_query_result_free(CacheQueryResult *result);

/**
 * Free chunk data
 */
void chunk_data_free(ChunkData *chunk);

/**
 * Print cache statistics to stdout
 */
void cache_print_stats(PersistentCache *cache);

/**
 * Validate that a graph is a DAG (no cycles)
 */
bool is_dag(const MarkovGraph *g);

/**
 * Get library version string
 */
const char* markov_cache_version(void);

#ifdef __cplusplus
}
#endif

#endif /* MARKOV_CACHE_H */
