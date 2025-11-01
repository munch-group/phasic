#include <markov_cache/markov_cache.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

typedef struct {
    uint32_t *index;
    uint32_t *lowlink;
    bool *on_stack;
    uint32_t *stack;
    int stack_top;
    uint32_t current_index;
    
    // Results
    uint32_t *node_to_scc;
    uint32_t num_sccs;
    
    // Adjacency list
    uint32_t **adj_list;
    uint32_t *adj_count;
} TarjanState;

static void tarjan_dfs(uint32_t v, const MarkovGraph *g, TarjanState *state) {
    state->index[v] = state->current_index;
    state->lowlink[v] = state->current_index;
    state->current_index++;
    
    state->stack[++state->stack_top] = v;
    state->on_stack[v] = true;
    
    // Visit neighbors
    for (uint32_t i = 0; i < state->adj_count[v]; i++) {
        uint32_t w = state->adj_list[v][i];
        
        if (state->index[w] == UINT32_MAX) {
            tarjan_dfs(w, g, state);
            if (state->lowlink[w] < state->lowlink[v]) {
                state->lowlink[v] = state->lowlink[w];
            }
        } else if (state->on_stack[w]) {
            if (state->index[w] < state->lowlink[v]) {
                state->lowlink[v] = state->index[w];
            }
        }
    }
    
    // Root of SCC
    if (state->lowlink[v] == state->index[v]) {
        uint32_t scc_id = state->num_sccs++;
        uint32_t w;
        
        do {
            w = state->stack[state->stack_top--];
            state->on_stack[w] = false;
            state->node_to_scc[w] = scc_id;
        } while (w != v);
    }
}

SCCDecomposition* find_sccs(const MarkovGraph *g) {
    if (!g) return NULL;
    
    TarjanState state;
    
    // Initialize state
    state.index = malloc(g->num_nodes * sizeof(uint32_t));
    state.lowlink = malloc(g->num_nodes * sizeof(uint32_t));
    state.on_stack = calloc(g->num_nodes, sizeof(bool));
    state.stack = malloc(g->num_nodes * sizeof(uint32_t));
    state.stack_top = -1;
    state.current_index = 0;
    state.node_to_scc = malloc(g->num_nodes * sizeof(uint32_t));
    state.num_sccs = 0;
    
    for (uint32_t i = 0; i < g->num_nodes; i++) {
        state.index[i] = UINT32_MAX;
    }
    
    // Build adjacency list
    state.adj_count = calloc(g->num_nodes, sizeof(uint32_t));
    
    // Count outgoing edges per node
    for (uint32_t i = 0; i < g->num_edges; i++) {
        state.adj_count[g->edge_src[i]]++;
    }
    
    // Allocate adjacency lists
    state.adj_list = malloc(g->num_nodes * sizeof(uint32_t*));
    for (uint32_t i = 0; i < g->num_nodes; i++) {
        if (state.adj_count[i] > 0) {
            state.adj_list[i] = malloc(state.adj_count[i] * sizeof(uint32_t));
        } else {
            state.adj_list[i] = NULL;
        }
        state.adj_count[i] = 0;  // Reset for filling
    }
    
    // Fill adjacency lists
    for (uint32_t i = 0; i < g->num_edges; i++) {
        uint32_t src = g->edge_src[i];
        uint32_t dst = g->edge_dst[i];
        state.adj_list[src][state.adj_count[src]++] = dst;
    }
    
    // Run Tarjan's algorithm
    for (uint32_t v = 0; v < g->num_nodes; v++) {
        if (state.index[v] == UINT32_MAX) {
            tarjan_dfs(v, g, &state);
        }
    }
    
    // Build result
    SCCDecomposition *result = malloc(sizeof(SCCDecomposition));
    result->num_sccs = state.num_sccs;
    result->node_to_scc = state.node_to_scc;
    result->scc_sizes = calloc(state.num_sccs, sizeof(uint32_t));
    result->scc_nodes = malloc(state.num_sccs * sizeof(uint32_t*));
    
    // Count SCC sizes
    for (uint32_t i = 0; i < g->num_nodes; i++) {
        result->scc_sizes[state.node_to_scc[i]]++;
    }
    
    // Allocate SCC node arrays
    for (uint32_t i = 0; i < state.num_sccs; i++) {
        result->scc_nodes[i] = malloc(result->scc_sizes[i] * sizeof(uint32_t));
        result->scc_sizes[i] = 0;  // Reset for filling
    }
    
    // Fill SCC node arrays
    for (uint32_t i = 0; i < g->num_nodes; i++) {
        uint32_t scc_id = state.node_to_scc[i];
        result->scc_nodes[scc_id][result->scc_sizes[scc_id]++] = i;
    }
    
    // Cleanup state
    free(state.index);
    free(state.lowlink);
    free(state.on_stack);
    free(state.stack);
    free(state.adj_count);
    for (uint32_t i = 0; i < g->num_nodes; i++) {
        free(state.adj_list[i]);
    }
    free(state.adj_list);
    
    return result;
}

void scc_free(SCCDecomposition *scc) {
    if (!scc) return;
    
    for (uint32_t i = 0; i < scc->num_sccs; i++) {
        free(scc->scc_nodes[i]);
    }
    
    free(scc->node_to_scc);
    free(scc->scc_sizes);
    free(scc->scc_nodes);
    free(scc);
}
