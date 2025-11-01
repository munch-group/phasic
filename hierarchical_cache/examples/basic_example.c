#include <markov_cache/markov_cache.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Helper function to create a random cyclic Markov graph
MarkovGraph* create_random_graph(uint32_t num_nodes, uint32_t num_edges, unsigned int seed) {
    srand(seed);
    
    MarkovGraph *g = markov_graph_create(num_nodes);
    
    for (uint32_t i = 0; i < num_edges; i++) {
        uint32_t src = rand() % num_nodes;
        uint32_t dst = rand() % num_nodes;
        double rate = (double)rand() / RAND_MAX * 10.0;
        
        markov_graph_add_edge(g, src, dst, rate);
    }
    
    return g;
}

// Helper function to create a graph with known structure (for testing)
MarkovGraph* create_cyclic_graph() {
    MarkovGraph *g = markov_graph_create(10);
    
    // Create a cycle: 0->1->2->3->4->0
    markov_graph_add_edge(g, 0, 1, 1.0);
    markov_graph_add_edge(g, 1, 2, 1.5);
    markov_graph_add_edge(g, 2, 3, 2.0);
    markov_graph_add_edge(g, 3, 4, 2.5);
    markov_graph_add_edge(g, 4, 0, 3.0);
    
    // Add another cycle: 5->6->7->5
    markov_graph_add_edge(g, 5, 6, 0.5);
    markov_graph_add_edge(g, 6, 7, 0.7);
    markov_graph_add_edge(g, 7, 5, 0.9);
    
    // Connect the cycles
    markov_graph_add_edge(g, 2, 5, 1.2);
    markov_graph_add_edge(g, 7, 8, 1.8);
    markov_graph_add_edge(g, 8, 9, 2.2);
    
    return g;
}

void example_basic_usage() {
    printf("\n=== Example 1: Basic Cache Usage ===\n");
    
    // Initialize cache
    PersistentCache *cache = cache_init("./test_cache", 10 * 1024 * 1024);  // 10MB hot cache
    if (!cache) {
        fprintf(stderr, "Failed to initialize cache\n");
        return;
    }
    
    // Create a cyclic graph
    printf("Creating cyclic Markov graph...\n");
    MarkovGraph *graph = create_cyclic_graph();
    printf("  Nodes: %u, Edges: %u\n", graph->num_nodes, graph->num_edges);
    printf("  Is DAG: %s\n", is_dag(graph) ? "Yes" : "No");
    
    // Solve with cache (first time - cache miss)
    printf("\nSolving graph (first time)...\n");
    DAGResult *result1 = solve_with_cache(cache, graph, 0);
    
    if (result1) {
        printf("  Result: %u nodes, %u edges\n", 
               result1->num_nodes, result1->num_edges);
        
        // Verify result is a DAG
        MarkovGraph dag_graph;
        dag_graph.num_nodes = result1->num_nodes;
        dag_graph.num_edges = result1->num_edges;
        dag_graph.edge_src = result1->edge_src;
        dag_graph.edge_dst = result1->edge_dst;
        dag_graph.rates = result1->weights;
        printf("  Result is DAG: %s\n", is_dag(&dag_graph) ? "Yes" : "No");
    }
    
    // Solve same graph again (should be cache hit)
    printf("\nSolving same graph again (should be cached)...\n");
    DAGResult *result2 = solve_with_cache(cache, graph, 0);
    
    if (result2) {
        printf("  Result: %u nodes, %u edges\n",
               result2->num_nodes, result2->num_edges);
    }
    
    // Print cache statistics
    printf("\n");
    cache_print_stats(cache);
    
    // Cleanup
    if (result1) dag_result_free(result1);
    if (result2 && result2 != result1) dag_result_free(result2);
    markov_graph_free(graph);
    cache_close(cache);
}

void example_scc_decomposition() {
    printf("\n=== Example 2: SCC Decomposition ===\n");
    
    MarkovGraph *graph = create_cyclic_graph();
    
    printf("Finding strongly connected components...\n");
    SCCDecomposition *sccs = find_sccs(graph);
    
    printf("  Found %u SCCs:\n", sccs->num_sccs);
    for (uint32_t i = 0; i < sccs->num_sccs; i++) {
        printf("    SCC %u: %u nodes [", i, sccs->scc_sizes[i]);
        for (uint32_t j = 0; j < sccs->scc_sizes[i]; j++) {
            printf("%u", sccs->scc_nodes[i][j]);
            if (j < sccs->scc_sizes[i] - 1) printf(", ");
        }
        printf("]\n");
    }
    
    scc_free(sccs);
    markov_graph_free(graph);
}

void example_cache_reuse() {
    printf("\n=== Example 3: Cache Reuse Across Multiple Graphs ===\n");
    
    PersistentCache *cache = cache_init("./test_cache", 10 * 1024 * 1024);
    
    // Create multiple graphs with overlapping substructures
    printf("Creating and solving 5 random graphs...\n");
    
    for (int i = 0; i < 5; i++) {
        printf("\nGraph %d:\n", i + 1);
        MarkovGraph *graph = create_random_graph(20, 40, 1000 + i);
        
        DAGResult *result = solve_with_cache(cache, graph, 0);
        
        if (result) {
            printf("  Solved: %u nodes, %u edges\n",
                   result->num_nodes, result->num_edges);
            dag_result_free(result);
        }
        
        markov_graph_free(graph);
    }
    
    printf("\n");
    cache_print_stats(cache);
    
    cache_close(cache);
}

void example_csr_conversion() {
    printf("\n=== Example 4: CSR Conversion ===\n");
    
    MarkovGraph *graph = create_cyclic_graph();
    
    printf("Original graph: %u nodes, %u edges\n",
           graph->num_nodes, graph->num_edges);
    
    // Convert to CSR
    printf("Converting to CSR format...\n");
    CSRGraph *csr = markov_to_csr(graph);
    
    printf("  CSR: %u nodes, %u edges\n", csr->num_nodes, csr->num_edges);
    printf("  row_ptr[0] = %u, row_ptr[%u] = %u\n",
           csr->row_ptr[0], csr->num_nodes, csr->row_ptr[csr->num_nodes]);
    
    // Convert back
    printf("Converting back to Markov graph...\n");
    MarkovGraph *graph2 = csr_to_markov(csr);
    
    printf("  Recovered: %u nodes, %u edges\n",
           graph2->num_nodes, graph2->num_edges);
    
    csr_free(csr);
    markov_graph_free(graph);
    markov_graph_free(graph2);
}

void example_hash_computation() {
    printf("\n=== Example 5: Graph Hashing ===\n");
    
    // Create two identical graphs
    MarkovGraph *g1 = markov_graph_create(5);
    markov_graph_add_edge(g1, 0, 1, 1.0);
    markov_graph_add_edge(g1, 1, 2, 2.0);
    markov_graph_add_edge(g1, 2, 3, 3.0);
    
    MarkovGraph *g2 = markov_graph_create(5);
    markov_graph_add_edge(g2, 0, 1, 1.0);
    markov_graph_add_edge(g2, 2, 3, 3.0);  // Different order
    markov_graph_add_edge(g2, 1, 2, 2.0);
    
    // Compute hashes
    uint8_t hash1[32], hash2[32];
    compute_graph_hash(g1, hash1);
    compute_graph_hash(g2, hash2);
    
    printf("Graph 1 hash: ");
    for (int i = 0; i < 8; i++) printf("%02x", hash1[i]);
    printf("...\n");
    
    printf("Graph 2 hash: ");
    for (int i = 0; i < 8; i++) printf("%02x", hash2[i]);
    printf("...\n");
    
    printf("Hashes match: %s\n",
           memcmp(hash1, hash2, 32) == 0 ? "Yes" : "No");
    
    // Create different graph
    MarkovGraph *g3 = markov_graph_create(5);
    markov_graph_add_edge(g3, 0, 1, 1.0);
    markov_graph_add_edge(g3, 1, 2, 2.5);  // Different rate
    markov_graph_add_edge(g3, 2, 3, 3.0);
    
    uint8_t hash3[32];
    compute_graph_hash(g3, hash3);
    
    printf("Graph 3 hash: ");
    for (int i = 0; i < 8; i++) printf("%02x", hash3[i]);
    printf("...\n");
    
    printf("G1 and G3 hashes match: %s\n",
           memcmp(hash1, hash3, 32) == 0 ? "Yes" : "No");
    
    markov_graph_free(g1);
    markov_graph_free(g2);
    markov_graph_free(g3);
}

void example_dictionary_training() {
    printf("\n=== Example 6: Compression Dictionary Training ===\n");
    
    PersistentCache *cache = cache_init("./test_cache", 10 * 1024 * 1024);
    
    // Create training samples
    const int num_samples = 10;
    MarkovGraph **samples = malloc(num_samples * sizeof(MarkovGraph*));
    
    printf("Creating %d training samples...\n", num_samples);
    for (int i = 0; i < num_samples; i++) {
        samples[i] = create_random_graph(15, 30, 2000 + i);
    }
    
    printf("Training compression dictionary...\n");
    bool success = cache_train_dictionary(cache, samples, num_samples);
    
    if (success) {
        printf("  Dictionary trained successfully!\n");
        
        // Test compression with dictionary
        printf("\nTesting compression with trained dictionary...\n");
        MarkovGraph *test_graph = create_random_graph(15, 30, 3000);
        
        DAGResult *result = solve_with_cache(cache, test_graph, 0);
        if (result) {
            printf("  Compression successful\n");
            dag_result_free(result);
        }
        
        markov_graph_free(test_graph);
    }
    
    // Cleanup
    for (int i = 0; i < num_samples; i++) {
        markov_graph_free(samples[i]);
    }
    free(samples);
    
    cache_close(cache);
}

void example_cache_query() {
    printf("\n=== Example 7: Direct Cache Query ===\n");
    
    PersistentCache *cache = cache_init("./test_cache", 10 * 1024 * 1024);
    
    MarkovGraph *graph = create_cyclic_graph();
    
    printf("Querying cache for graph...\n");
    CacheQueryResult *query = cache_query(cache, graph);
    
    printf("  Cached chunks: %u\n", query->num_cached);
    printf("  Uncached chunks: %u\n", query->num_uncached);
    
    if (query->num_uncached > 0) {
        printf("\n  Uncached subgraphs:\n");
        for (uint32_t i = 0; i < query->num_uncached; i++) {
            printf("    Subgraph %u: %u nodes, %u edges\n",
                   i,
                   query->uncached_subgraphs[i]->num_nodes,
                   query->uncached_subgraphs[i]->num_edges);
        }
    }
    
    cache_query_result_free(query);
    markov_graph_free(graph);
    cache_close(cache);
}

int main(int argc, char *argv[]) {
    printf("Markov Cache Library - Complete Example\n");
    printf("========================================\n");
    printf("Version: %s\n", markov_cache_version());
    
    // Run all examples
    example_basic_usage();
    example_scc_decomposition();
    example_cache_reuse();
    example_csr_conversion();
    example_hash_computation();
    example_dictionary_training();
    example_cache_query();
    
    printf("\n=== All Examples Complete ===\n");
    printf("\nCache directory './test_cache' contains persistent data.\n");
    printf("Run the program again to see cache hits!\n");
    
    return 0;
}
