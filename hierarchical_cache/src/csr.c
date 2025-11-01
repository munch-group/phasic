#include <markov_cache/markov_cache.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t src, dst;
    double rate;
} Edge;

static int edge_compare(const void *a, const void *b) {
    const Edge *ea = (const Edge*)a;
    const Edge *eb = (const Edge*)b;
    
    if (ea->src != eb->src) {
        return (int)ea->src - (int)eb->src;
    }
    return (int)ea->dst - (int)eb->dst;
}

CSRGraph* markov_to_csr(const MarkovGraph *g) {
    if (!g) return NULL;
    
    CSRGraph *csr = malloc(sizeof(CSRGraph));
    if (!csr) return NULL;
    
    csr->num_nodes = g->num_nodes;
    csr->num_edges = g->num_edges;
    
    // Allocate arrays
    csr->row_ptr = malloc((g->num_nodes + 1) * sizeof(uint32_t));
    csr->col_idx = malloc(g->num_edges * sizeof(uint32_t));
    csr->values = malloc(g->num_edges * sizeof(double));
    
    if (!csr->row_ptr || !csr->col_idx || !csr->values) {
        csr_free(csr);
        return NULL;
    }
    
    // Sort edges by source
    Edge *sorted = malloc(g->num_edges * sizeof(Edge));
    for (uint32_t i = 0; i < g->num_edges; i++) {
        sorted[i].src = g->edge_src[i];
        sorted[i].dst = g->edge_dst[i];
        sorted[i].rate = g->rates[i];
    }
    qsort(sorted, g->num_edges, sizeof(Edge), edge_compare);
    
    // Build CSR structure
    uint32_t edge_idx = 0;
    for (uint32_t node = 0; node < g->num_nodes; node++) {
        csr->row_ptr[node] = edge_idx;
        
        while (edge_idx < g->num_edges && sorted[edge_idx].src == node) {
            csr->col_idx[edge_idx] = sorted[edge_idx].dst;
            csr->values[edge_idx] = sorted[edge_idx].rate;
            edge_idx++;
        }
    }
    csr->row_ptr[g->num_nodes] = edge_idx;
    
    free(sorted);
    return csr;
}

MarkovGraph* csr_to_markov(const CSRGraph *csr) {
    if (!csr) return NULL;
    
    MarkovGraph *g = markov_graph_create(csr->num_nodes);
    if (!g) return NULL;
    
    g->num_edges = csr->num_edges;
    g->edge_src = malloc(csr->num_edges * sizeof(uint32_t));
    g->edge_dst = malloc(csr->num_edges * sizeof(uint32_t));
    g->rates = malloc(csr->num_edges * sizeof(double));
    
    if (!g->edge_src || !g->edge_dst || !g->rates) {
        markov_graph_free(g);
        return NULL;
    }
    
    // Convert from CSR format
    uint32_t edge_idx = 0;
    for (uint32_t node = 0; node < csr->num_nodes; node++) {
        for (uint32_t i = csr->row_ptr[node]; i < csr->row_ptr[node + 1]; i++) {
            g->edge_src[edge_idx] = node;
            g->edge_dst[edge_idx] = csr->col_idx[i];
            g->rates[edge_idx] = csr->values[i];
            edge_idx++;
        }
    }
    
    return g;
}

void csr_free(CSRGraph *csr) {
    if (!csr) return;
    
    free(csr->row_ptr);
    free(csr->col_idx);
    free(csr->values);
    free(csr);
}
