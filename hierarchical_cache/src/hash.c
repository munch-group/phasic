#include <markov_cache/markov_cache.h>
#include <markov_cache/internal.h>
#include <openssl/sha.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t src, dst;
    double rate;
} HashEdge;

static int hash_edge_compare(const void *a, const void *b) {
    const HashEdge *ea = (const HashEdge*)a;
    const HashEdge *eb = (const HashEdge*)b;
    
    // Canonical ordering: min(src,dst), max(src,dst), rate
    uint32_t ea_min = ea->src < ea->dst ? ea->src : ea->dst;
    uint32_t ea_max = ea->src < ea->dst ? ea->dst : ea->src;
    uint32_t eb_min = eb->src < eb->dst ? eb->src : eb->dst;
    uint32_t eb_max = eb->src < eb->dst ? eb->dst : eb->src;
    
    if (ea_min != eb_min) {
        return (int)ea_min - (int)eb_min;
    }
    if (ea_max != eb_max) {
        return (int)ea_max - (int)eb_max;
    }
    
    // Compare rates
    if (ea->rate < eb->rate) return -1;
    if (ea->rate > eb->rate) return 1;
    return 0;
}

void compute_graph_hash(const MarkovGraph *g, uint8_t *hash) {
    if (!g || !hash) return;
    
    // Create canonical edge representation
    HashEdge *edges = malloc(g->num_edges * sizeof(HashEdge));
    for (uint32_t i = 0; i < g->num_edges; i++) {
        edges[i].src = g->edge_src[i];
        edges[i].dst = g->edge_dst[i];
        edges[i].rate = g->rates[i];
    }
    
    // Sort for canonical form
    qsort(edges, g->num_edges, sizeof(HashEdge), hash_edge_compare);
    
    // Compute SHA-256
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    
    // Hash number of nodes
    SHA256_Update(&ctx, &g->num_nodes, sizeof(uint32_t));
    
    // Hash sorted edges
    SHA256_Update(&ctx, edges, g->num_edges * sizeof(HashEdge));
    
    SHA256_Final(hash, &ctx);
    
    free(edges);
}
