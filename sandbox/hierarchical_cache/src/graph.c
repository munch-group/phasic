#include <markov_cache/markov_cache.h>
#include <stdlib.h>
#include <string.h>

MarkovGraph* markov_graph_create(uint32_t num_nodes) {
    MarkovGraph *g = malloc(sizeof(MarkovGraph));
    if (!g) return NULL;
    
    g->num_nodes = num_nodes;
    g->num_edges = 0;
    g->edge_src = NULL;
    g->edge_dst = NULL;
    g->rates = NULL;
    
    return g;
}

void markov_graph_add_edge(MarkovGraph *g, uint32_t src, uint32_t dst, double rate) {
    // Reallocate arrays
    g->num_edges++;
    g->edge_src = realloc(g->edge_src, g->num_edges * sizeof(uint32_t));
    g->edge_dst = realloc(g->edge_dst, g->num_edges * sizeof(uint32_t));
    g->rates = realloc(g->rates, g->num_edges * sizeof(double));
    
    g->edge_src[g->num_edges - 1] = src;
    g->edge_dst[g->num_edges - 1] = dst;
    g->rates[g->num_edges - 1] = rate;
}

void markov_graph_free(MarkovGraph *g) {
    if (!g) return;
    
    free(g->edge_src);
    free(g->edge_dst);
    free(g->rates);
    free(g);
}

DAGResult* dag_result_create(uint32_t num_nodes) {
    DAGResult *dag = malloc(sizeof(DAGResult));
    if (!dag) return NULL;
    
    dag->num_nodes = num_nodes;
    dag->num_edges = 0;
    dag->edge_src = NULL;
    dag->edge_dst = NULL;
    dag->weights = NULL;
    dag->moments = NULL;
    dag->num_moments = 0;
    
    return dag;
}

void dag_result_free(DAGResult *g) {
    if (!g) return;
    
    free(g->edge_src);
    free(g->edge_dst);
    free(g->weights);
    free(g->moments);
    free(g);
}

MarkovGraph* extract_subgraph(const MarkovGraph *g,
                             const uint32_t *node_ids,
                             uint32_t num_nodes) {
    if (!g || !node_ids || num_nodes == 0) return NULL;
    
    // Create mapping from old node IDs to new ones
    uint32_t *node_map = malloc(g->num_nodes * sizeof(uint32_t));
    memset(node_map, 0xFF, g->num_nodes * sizeof(uint32_t));
    
    for (uint32_t i = 0; i < num_nodes; i++) {
        node_map[node_ids[i]] = i;
    }
    
    // Count edges in subgraph
    uint32_t num_subgraph_edges = 0;
    for (uint32_t i = 0; i < g->num_edges; i++) {
        if (node_map[g->edge_src[i]] != 0xFFFFFFFF &&
            node_map[g->edge_dst[i]] != 0xFFFFFFFF) {
            num_subgraph_edges++;
        }
    }
    
    // Create subgraph
    MarkovGraph *sub = markov_graph_create(num_nodes);
    sub->num_edges = num_subgraph_edges;
    sub->edge_src = malloc(num_subgraph_edges * sizeof(uint32_t));
    sub->edge_dst = malloc(num_subgraph_edges * sizeof(uint32_t));
    sub->rates = malloc(num_subgraph_edges * sizeof(double));
    
    // Copy edges
    uint32_t edge_idx = 0;
    for (uint32_t i = 0; i < g->num_edges; i++) {
        if (node_map[g->edge_src[i]] != 0xFFFFFFFF &&
            node_map[g->edge_dst[i]] != 0xFFFFFFFF) {
            sub->edge_src[edge_idx] = node_map[g->edge_src[i]];
            sub->edge_dst[edge_idx] = node_map[g->edge_dst[i]];
            sub->rates[edge_idx] = g->rates[i];
            edge_idx++;
        }
    }
    
    free(node_map);
    return sub;
}

bool is_dag(const MarkovGraph *g) {
    if (!g) return false;
    
    // Kahn's algorithm for topological sort
    uint32_t *in_degree = calloc(g->num_nodes, sizeof(uint32_t));
    
    // Count in-degrees
    for (uint32_t i = 0; i < g->num_edges; i++) {
        in_degree[g->edge_dst[i]]++;
    }
    
    // Queue for nodes with in-degree 0
    uint32_t *queue = malloc(g->num_nodes * sizeof(uint32_t));
    int queue_start = 0, queue_end = 0;
    
    for (uint32_t i = 0; i < g->num_nodes; i++) {
        if (in_degree[i] == 0) {
            queue[queue_end++] = i;
        }
    }
    
    uint32_t processed = 0;
    
    while (queue_start < queue_end) {
        uint32_t node = queue[queue_start++];
        processed++;
        
        // Reduce in-degree of neighbors
        for (uint32_t i = 0; i < g->num_edges; i++) {
            if (g->edge_src[i] == node) {
                if (--in_degree[g->edge_dst[i]] == 0) {
                    queue[queue_end++] = g->edge_dst[i];
                }
            }
        }
    }
    
    free(in_degree);
    free(queue);
    
    return processed == g->num_nodes;
}

const char* markov_cache_version(void) {
    return "1.0.0";
}
