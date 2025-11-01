#include <markov_cache/markov_cache.h>
#include <markov_cache/internal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <zdict.h>

PersistentCache* cache_init(const char *cache_dir, size_t hot_cache_size) {
    if (!cache_dir) return NULL;
    
    PersistentCache *cache = malloc(sizeof(PersistentCache));
    if (!cache) return NULL;
    
    strncpy(cache->cache_dir, cache_dir, sizeof(cache->cache_dir) - 1);
    cache->cache_dir[sizeof(cache->cache_dir) - 1] = '\0';
    
    // Create cache directory structure
    mkdir_recursive(cache_dir);
    
    char chunks_dir[600];
    snprintf(chunks_dir, sizeof(chunks_dir), "%s/chunks", cache_dir);
    mkdir_recursive(chunks_dir);
    
    // Open SQLite database
    char db_path[600];
    snprintf(db_path, sizeof(db_path), "%s/index.db", cache_dir);
    
    if (sqlite3_open(db_path, &cache->db) != SQLITE_OK) {
        free(cache);
        return NULL;
    }
    
    // Create schema
    const char *schema = 
        "CREATE TABLE IF NOT EXISTS chunks ("
        "  orig_hash BLOB(32) NOT NULL,"
        "  dag_hash BLOB(32) NOT NULL,"
        "  num_nodes INTEGER,"
        "  num_edges INTEGER,"
        "  level INTEGER,"
        "  compressed_size INTEGER,"
        "  uncompressed_size INTEGER,"
        "  filepath TEXT,"
        "  PRIMARY KEY (orig_hash, dag_hash)"
        ");"
        "CREATE INDEX IF NOT EXISTS idx_orig ON chunks(orig_hash);"
        "CREATE INDEX IF NOT EXISTS idx_dag ON chunks(dag_hash);"
        "CREATE INDEX IF NOT EXISTS idx_size ON chunks(num_nodes);";
    
    char *err_msg = NULL;
    if (sqlite3_exec(cache->db, schema, NULL, NULL, &err_msg) != SQLITE_OK) {
        sqlite3_free(err_msg);
        sqlite3_close(cache->db);
        free(cache);
        return NULL;
    }
    
    // Load compression dictionary if it exists
    char dict_path[600];
    snprintf(dict_path, sizeof(dict_path), "%s/dict.zstd", cache_dir);
    
    FILE *f = fopen(dict_path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        size_t dict_size = ftell(f);
        fseek(f, 0, SEEK_SET);
        
        void *dict_buffer = malloc(dict_size);
        fread(dict_buffer, dict_size, 1, f);
        fclose(f);
        
        cache->cdict = ZSTD_createCDict(dict_buffer, dict_size, 3);
        cache->ddict = ZSTD_createDDict(dict_buffer, dict_size);
        
        free(dict_buffer);
    } else {
        cache->cdict = NULL;
        cache->ddict = NULL;
    }
    
    // Initialize hot cache
    if (hot_cache_size > 0) {
        cache->hot_cache = lru_create(hot_cache_size);
    } else {
        cache->hot_cache = NULL;
    }
    
    cache->cache_hits = 0;
    cache->cache_misses = 0;
    cache->total_chunks = 0;
    
    // Count existing chunks
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(cache->db, "SELECT COUNT(*) FROM chunks", -1, &stmt, NULL) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            cache->total_chunks = sqlite3_column_int64(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }
    
    return cache;
}

void cache_close(PersistentCache *cache) {
    if (!cache) return;
    
    if (cache->db) {
        sqlite3_close(cache->db);
    }
    
    if (cache->cdict) {
        ZSTD_freeCDict(cache->cdict);
    }
    
    if (cache->ddict) {
        ZSTD_freeDDict(cache->ddict);
    }
    
    if (cache->hot_cache) {
        lru_free(cache->hot_cache);
    }
    
    free(cache);
}

bool cache_store(PersistentCache *cache,
                const MarkovGraph *orig,
                const DAGResult *dag,
                const ChunkMetadata *metadata) {
    if (!cache || !orig || !dag) return false;
    
    // Compute hashes if not provided
    uint8_t orig_hash[32], dag_hash[32];
    
    if (metadata && metadata->orig_hash[0] != 0) {
        memcpy(orig_hash, metadata->orig_hash, 32);
        memcpy(dag_hash, metadata->dag_hash, 32);
    } else {
        compute_graph_hash(orig, orig_hash);
        
        // Convert DAG to MarkovGraph for hashing
        MarkovGraph dag_mg;
        dag_mg.num_nodes = dag->num_nodes;
        dag_mg.num_edges = dag->num_edges;
        dag_mg.edge_src = dag->edge_src;
        dag_mg.edge_dst = dag->edge_dst;
        dag_mg.rates = dag->weights;
        compute_graph_hash(&dag_mg, dag_hash);
    }
    
    // Check if already exists
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(cache->db,
        "SELECT filepath FROM chunks WHERE orig_hash=? AND dag_hash=?",
        -1, &stmt, NULL) == SQLITE_OK) {
        
        sqlite3_bind_blob(stmt, 1, orig_hash, 32, SQLITE_STATIC);
        sqlite3_bind_blob(stmt, 2, dag_hash, 32, SQLITE_STATIC);
        
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            sqlite3_finalize(stmt);
            return true;  // Already exists
        }
        sqlite3_finalize(stmt);
    }
    
    // Convert to CSR
    CSRGraph *csr_orig = markov_to_csr(orig);
    if (!csr_orig) return false;
    
    // Serialize
    uint8_t *serialized;
    size_t serialized_size = serialize_csr(csr_orig, &serialized);
    if (serialized_size == 0) {
        csr_free(csr_orig);
        return false;
    }
    
    // Compress
    uint8_t *compressed;
    size_t compressed_size = compress_data(serialized, serialized_size,
                                          cache->cdict, &compressed);
    free(serialized);
    
    if (compressed_size == 0) {
        csr_free(csr_orig);
        return false;
    }
    
    // Generate filepath
    char filepath[600];
    get_chunk_path(cache->cache_dir, orig_hash, dag_hash, 
                   filepath, sizeof(filepath));
    
    // Ensure directory exists
    char dirpath[600];
    snprintf(dirpath, sizeof(dirpath), "%s/chunks/%02x/%02x",
             cache->cache_dir, orig_hash[0], orig_hash[1]);
    mkdir_recursive(dirpath);
    
    // Write compressed data
    FILE *f = fopen(filepath, "wb");
    if (!f) {
        free(compressed);
        csr_free(csr_orig);
        return false;
    }
    
    fwrite(compressed, compressed_size, 1, f);
    fclose(f);
    free(compressed);
    
    // Update database
    if (sqlite3_prepare_v2(cache->db,
        "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)",
        -1, &stmt, NULL) == SQLITE_OK) {
        
        sqlite3_bind_blob(stmt, 1, orig_hash, 32, SQLITE_STATIC);
        sqlite3_bind_blob(stmt, 2, dag_hash, 32, SQLITE_STATIC);
        sqlite3_bind_int(stmt, 3, orig->num_nodes);
        sqlite3_bind_int(stmt, 4, orig->num_edges);
        sqlite3_bind_int(stmt, 5, metadata ? metadata->level : 0);
        sqlite3_bind_int(stmt, 6, compressed_size);
        sqlite3_bind_int(stmt, 7, serialized_size);
        sqlite3_bind_text(stmt, 8, filepath, -1, SQLITE_TRANSIENT);
        
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }
    
    csr_free(csr_orig);
    cache->total_chunks++;
    
    return true;
}

ChunkData* cache_load(PersistentCache *cache,
                     const uint8_t *orig_hash,
                     const uint8_t *dag_hash) {
    if (!cache || !orig_hash || !dag_hash) return NULL;
    
    // For now, skip hot cache to avoid ownership issues
    // TODO: Implement proper hot cache with reference counting
    
    // Query database
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(cache->db,
        "SELECT filepath, uncompressed_size, num_nodes, num_edges, level "
        "FROM chunks WHERE orig_hash=? AND dag_hash=?",
        -1, &stmt, NULL) != SQLITE_OK) {
        cache->cache_misses++;
        return NULL;
    }
    
    sqlite3_bind_blob(stmt, 1, orig_hash, 32, SQLITE_STATIC);
    sqlite3_bind_blob(stmt, 2, dag_hash, 32, SQLITE_STATIC);
    
    if (sqlite3_step(stmt) != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        cache->cache_misses++;
        return NULL;
    }
    
    const char *filepath = (const char*)sqlite3_column_text(stmt, 0);
    size_t uncompressed_size = sqlite3_column_int(stmt, 1);
    uint32_t num_nodes = sqlite3_column_int(stmt, 2);
    uint32_t num_edges = sqlite3_column_int(stmt, 3);
    uint32_t level = sqlite3_column_int(stmt, 4);
    
    // Memory-map file
    int fd = open(filepath, O_RDONLY);
    if (fd < 0) {
        sqlite3_finalize(stmt);
        cache->cache_misses++;
        return NULL;
    }
    
    struct stat st;
    fstat(fd, &st);
    
    void *compressed = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (compressed == MAP_FAILED) {
        close(fd);
        sqlite3_finalize(stmt);
        cache->cache_misses++;
        return NULL;
    }
    
    // Decompress
    uint8_t *uncompressed;
    size_t actual_size = decompress_data(compressed, st.st_size,
                                        cache->ddict, &uncompressed);
    
    munmap(compressed, st.st_size);
    close(fd);
    sqlite3_finalize(stmt);
    
    if (actual_size == 0) {
        cache->cache_misses++;
        return NULL;
    }
    
    // Deserialize
    CSRGraph *csr = deserialize_csr(uncompressed, actual_size);
    free(uncompressed);
    
    if (!csr) {
        cache->cache_misses++;
        return NULL;
    }
    
    // Add to hot cache
    if (cache->hot_cache) {
        CacheKey key;
        memcpy(key.orig_hash, orig_hash, 32);
        memcpy(key.dag_hash, dag_hash, 32);
        // Note: LRU cache doesn't own the CSR, just references it
        // We'll manage CSR ownership through ChunkData
    }
    
    // Build result
    ChunkData *chunk = malloc(sizeof(ChunkData));
    memcpy(chunk->metadata.orig_hash, orig_hash, 32);
    memcpy(chunk->metadata.dag_hash, dag_hash, 32);
    chunk->metadata.num_nodes = num_nodes;
    chunk->metadata.num_edges = num_edges;
    chunk->metadata.level = level;
    chunk->metadata.node_ids = NULL;
    chunk->csr_orig = csr;
    chunk->csr_dag = NULL;
    chunk->dag_result = NULL;
    
    cache->cache_hits++;
    return chunk;
}

void chunk_data_free(ChunkData *chunk) {
    if (!chunk) return;
    
    if (chunk->csr_orig) csr_free(chunk->csr_orig);
    if (chunk->csr_dag) csr_free(chunk->csr_dag);
    if (chunk->dag_result) dag_result_free(chunk->dag_result);
    if (chunk->metadata.node_ids) free(chunk->metadata.node_ids);
    
    free(chunk);
}

CacheQueryResult* cache_query(PersistentCache *cache,
                              const MarkovGraph *graph) {
    if (!cache || !graph) return NULL;
    
    CacheQueryResult *result = malloc(sizeof(CacheQueryResult));
    memset(result, 0, sizeof(CacheQueryResult));
    
    // Compute hash of full graph
    uint8_t graph_hash[32];
    compute_graph_hash(graph, graph_hash);
    
    // Check if entire graph is cached
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(cache->db,
        "SELECT dag_hash FROM chunks WHERE orig_hash=?",
        -1, &stmt, NULL) == SQLITE_OK) {
        
        sqlite3_bind_blob(stmt, 1, graph_hash, 32, SQLITE_STATIC);
        
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const uint8_t *dag_hash = sqlite3_column_blob(stmt, 0);
            
            ChunkData *cached = cache_load(cache, graph_hash, dag_hash);
            if (cached) {
                result->num_cached = 1;
                result->cached_chunks = malloc(sizeof(ChunkData*));
                result->cached_chunks[0] = cached;
                result->num_uncached = 0;
                sqlite3_finalize(stmt);
                return result;
            }
        }
        sqlite3_finalize(stmt);
    }
    
    // Not fully cached - decompose into SCCs
    SCCDecomposition *sccs = find_sccs(graph);
    
    result->cached_chunks = malloc(sccs->num_sccs * sizeof(ChunkData*));
    result->uncached_subgraphs = malloc(sccs->num_sccs * sizeof(MarkovGraph*));
    result->uncached_node_ids = malloc(sccs->num_sccs * sizeof(uint32_t*));
    
    // Check each SCC
    for (uint32_t i = 0; i < sccs->num_sccs; i++) {
        MarkovGraph *subgraph = extract_subgraph(graph,
                                                sccs->scc_nodes[i],
                                                sccs->scc_sizes[i]);
        
        uint8_t sub_hash[32];
        compute_graph_hash(subgraph, sub_hash);
        
        // Try to load from cache
        if (sqlite3_prepare_v2(cache->db,
            "SELECT dag_hash FROM chunks WHERE orig_hash=? LIMIT 1",
            -1, &stmt, NULL) == SQLITE_OK) {
            
            sqlite3_bind_blob(stmt, 1, sub_hash, 32, SQLITE_STATIC);
            
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                const uint8_t *dag_hash = sqlite3_column_blob(stmt, 0);
                
                ChunkData *cached = cache_load(cache, sub_hash, dag_hash);
                if (cached) {
                    result->cached_chunks[result->num_cached++] = cached;
                    markov_graph_free(subgraph);
                    sqlite3_finalize(stmt);
                    continue;
                }
            }
            sqlite3_finalize(stmt);
        }
        
        // Not cached - add to uncached list
        result->uncached_subgraphs[result->num_uncached] = subgraph;
        result->uncached_node_ids[result->num_uncached] = 
            malloc(sccs->scc_sizes[i] * sizeof(uint32_t));
        memcpy(result->uncached_node_ids[result->num_uncached],
               sccs->scc_nodes[i],
               sccs->scc_sizes[i] * sizeof(uint32_t));
        result->num_uncached++;
    }
    
    scc_free(sccs);
    return result;
}

void cache_query_result_free(CacheQueryResult *result) {
    if (!result) return;
    
    for (uint32_t i = 0; i < result->num_cached; i++) {
        chunk_data_free(result->cached_chunks[i]);
    }
    free(result->cached_chunks);
    
    for (uint32_t i = 0; i < result->num_uncached; i++) {
        markov_graph_free(result->uncached_subgraphs[i]);
        free(result->uncached_node_ids[i]);
    }
    free(result->uncached_subgraphs);
    free(result->uncached_node_ids);
    free(result->assembly_order);
    
    free(result);
}

void cache_get_stats(PersistentCache *cache,
                    uint64_t *hits,
                    uint64_t *misses,
                    uint64_t *total_chunks) {
    if (!cache) return;
    
    if (hits) *hits = cache->cache_hits;
    if (misses) *misses = cache->cache_misses;
    if (total_chunks) *total_chunks = cache->total_chunks;
}

void cache_print_stats(PersistentCache *cache) {
    if (!cache) return;
    
    uint64_t total_queries = cache->cache_hits + cache->cache_misses;
    double hit_rate = total_queries > 0 ? 
        (double)cache->cache_hits / total_queries * 100.0 : 0.0;
    
    printf("Cache Statistics:\n");
    printf("  Total chunks: %lu\n", cache->total_chunks);
    printf("  Cache hits: %lu\n", cache->cache_hits);
    printf("  Cache misses: %lu\n", cache->cache_misses);
    printf("  Hit rate: %.2f%%\n", hit_rate);
}

bool cache_train_dictionary(PersistentCache *cache,
                            MarkovGraph **samples,
                            uint32_t num_samples) {
    if (!cache || !samples || num_samples == 0) return false;
    
    // Serialize all samples
    size_t total_size = 0;
    uint8_t **buffers = malloc(num_samples * sizeof(uint8_t*));
    size_t *sizes = malloc(num_samples * sizeof(size_t));
    
    for (uint32_t i = 0; i < num_samples; i++) {
        CSRGraph *csr = markov_to_csr(samples[i]);
        sizes[i] = serialize_csr(csr, &buffers[i]);
        total_size += sizes[i];
        csr_free(csr);
    }
    
    // Combine into training data
    uint8_t *training_data = malloc(total_size);
    size_t offset = 0;
    
    for (uint32_t i = 0; i < num_samples; i++) {
        memcpy(training_data + offset, buffers[i], sizes[i]);
        offset += sizes[i];
        free(buffers[i]);
    }
    
    free(buffers);
    
    // Train dictionary
    size_t dict_size = 100 * 1024;  // 100KB
    void *dict_buffer = malloc(dict_size);
    
    size_t trained_size = ZDICT_trainFromBuffer(
        dict_buffer, dict_size,
        training_data, sizes, num_samples);
    
    free(training_data);
    free(sizes);
    
    if (ZSTD_isError(trained_size)) {
        free(dict_buffer);
        return false;
    }
    
    // Save dictionary
    char dict_path[600];
    snprintf(dict_path, sizeof(dict_path), "%s/dict.zstd", cache->cache_dir);
    
    FILE *f = fopen(dict_path, "wb");
    if (!f) {
        free(dict_buffer);
        return false;
    }
    
    fwrite(dict_buffer, trained_size, 1, f);
    fclose(f);
    
    // Update cache dictionaries
    if (cache->cdict) ZSTD_freeCDict(cache->cdict);
    if (cache->ddict) ZSTD_freeDDict(cache->ddict);
    
    cache->cdict = ZSTD_createCDict(dict_buffer, trained_size, 3);
    cache->ddict = ZSTD_createDDict(dict_buffer, trained_size);
    
    free(dict_buffer);
    return true;
}

DAGResult* solve_with_cache(PersistentCache *cache,
                           const MarkovGraph *graph,
                           uint32_t max_chunk_size) {
    if (!cache || !graph) return NULL;
    
    // Query cache
    CacheQueryResult *query = cache_query(cache, graph);
    
    printf("Cache query: %u cached, %u uncached chunks\n",
           query->num_cached, query->num_uncached);
    
    // Solve uncached subgraphs
    DAGResult **new_solutions = NULL;
    if (query->num_uncached > 0) {
        new_solutions = malloc(query->num_uncached * sizeof(DAGResult*));
        
        for (uint32_t i = 0; i < query->num_uncached; i++) {
            printf("Computing Gaussian elimination for chunk %u (%u nodes, %u edges)\n",
                   i, query->uncached_subgraphs[i]->num_nodes,
                   query->uncached_subgraphs[i]->num_edges);
            
            // Call user-provided Gaussian elimination
            new_solutions[i] = gaussian_elimination(query->uncached_subgraphs[i]);
            
            if (new_solutions[i]) {
                // Store in cache
                cache_store(cache, query->uncached_subgraphs[i],
                          new_solutions[i], NULL);
            }
        }
    }
    
    // For now, if entire graph was computed, return that
    // A full implementation would assemble from chunks
    DAGResult *result = NULL;
    if (query->num_uncached == 1 && query->num_cached == 0) {
        result = new_solutions[0];
    } else if (query->num_cached == 1 && query->num_uncached == 0) {
        // Convert cached CSR back to DAG result
        result = dag_result_create(query->cached_chunks[0]->metadata.num_nodes);
        // Would need full implementation here
    } else {
        // Multiple chunks - need assembly logic
        // For now, just recompute the whole thing
        result = gaussian_elimination(graph);
        cache_store(cache, graph, result, NULL);
    }
    
    if (new_solutions) {
        for (uint32_t i = 0; i < query->num_uncached; i++) {
            if (new_solutions[i] && new_solutions[i] != result) {
                dag_result_free(new_solutions[i]);
            }
        }
        free(new_solutions);
    }
    
    cache_query_result_free(query);
    return result;
}
