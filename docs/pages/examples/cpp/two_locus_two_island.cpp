#include <Rcpp.h>
#include "stdint.h"
#include "stdlib.h"
#include "assert.h"
#include "./../../../phasic/api/c/phasic.h"
#include "./../../../phasic/api/cpp/phasiccpp.h"

#include "./../../../phasic/src/c/phasic.c"

using namespace Rcpp;
using namespace phasic;


/**
 * Struct representing lineage properties corresponding to a state vector index:
 * locus1: number of decendendants at locus 1 for lineages represented by the state vector index.
 * locus2: number of decendendants at locus 2 for lineages represented by the state vector index.
 * population: the current population of lineages represented by the state vector index.
 */
struct properties {
    int locus1;
    int locus2;
    int population;
};


/**
 * Converts a zero-based state vector index to a props
 *
 * @param s Sample size.
 * @param i Index to be converted.
 * @return Container struct for `index`.
 */
static inline properties _index_to_props(int s, int i) {
    assert(i <= s);
    int p = (i-1) / pow((s+1),2);
    properties props;
    props.locus1 = (i - p*pow((s+1),2)) / (s+1);   
    props.locus2 = fmod((i - p*pow((s+1),2)), (s+1));     
    props.population = p + 1;
    return props;
}


/**
 * Same as above, but one-based for exporting to R
 */
// [[Rcpp::export]]
List index_to_props(int s, int i) {
    i  -= 1;
    properties props = _index_to_props(s, i);
    return List::create(Named("locus1") = props.locus1 , _["locus2"] = props.locus2, _["population"] = props.population);
}


/**
 * Converts a lineage properties to a state vector index.
 *
 * @param s Sample size.
 * @param a Number of decendants of lineage at locus one.
 * @param b Number of decendants of lineage at locus two.
 * @param p Index of the population where the lineage is located (1 or 2).
 * @return State vector index.
 */
static inline int _props_to_index(int s, int a, int b, int p) {
    assert(a <= s);
    assert(b <= s);
    assert(p > 0 && p <= 2);
    return((p-1)*pow((s+1),2) + a*(s+1) + b);
}


/**
 * Same as above, but one-based for exporting to R
 */
// [[Rcpp::export]]
int props_to_index(int s, int a, int b, int p)  {
    return(_props_to_index(s, a, b, p)+1);
}


/**
 * Constructs a state space for a two-locus two two-island model with recombination.
 *
 * @param sample_size Sample size.
 * @param N Scaled population size.
 * @param M Scaled migration rate
 * @param R Scaled recombination rate.
 * @return Graph.
 */
// [[Rcpp::export]]
SEXP construct_twolocus_island_graph(int sample_size, 
                                     double N1, double N2, 
                                     double M1, double M2, 
                                     double R,
                                     bool epoques=0) {
// SEXP construct_twolocus_island_graph(int sample_size, double N, double M, double R, bool keep_null_edges=false) {
    
     // number of populations
    const int nr_populations = 2; // needs to be 2
    // state vector length
    const int state_length = nr_populations * pow((sample_size+1), 2);
    const size_t state_size = sizeof(int) * state_length;

    struct ptd_graph *graph = ptd_graph_create((size_t) state_length + ((int) epoques));  // plus one to make room for the epoque label
    struct ptd_avl_tree *avl_tree = ptd_avl_tree_create((size_t) state_length + ((int) epoques));  // plus one to make room for the epoque label

    int *initial_state = (int *) calloc((size_t) state_length, sizeof(int));
    
    initial_state[_props_to_index(sample_size, 1, 1, 1)] = sample_size;
    // initial_state[sample_size*2-1] = sample_size;

    ptd_graph_add_edge(
            graph->starting_vertex,
            ptd_find_or_create_vertex(graph, avl_tree, initial_state),
            1
    );
    free(initial_state);
    int *child_state = (int *) malloc(state_size);

    for (size_t index = 1; index < graph->vertices_length; ++index) {
        struct ptd_vertex *vertex = graph->vertices[index];
        int *state = vertex->state;
        
        int lineages_left = 0;

        for (int i = 0; i < state_length; ++i) {
            lineages_left += state[i];
        }

        if (lineages_left == 0 || lineages_left == 1) {
            // Only one lineage left, absorb
            continue;
        }

        for (int i = 0; i < state_length; ++i) {
            properties props_i = _index_to_props(sample_size, i);
                
            // coalescence
            for (int j = i; j < state_length; ++j) {
                properties props_j = _index_to_props(sample_size, j);
                    
                if (props_i.population != props_j.population) {
                    // different populations
                    continue;
                }
                
//                double edge_state[] = {0, 0, 0, 0, 0}; // N1, N2, M1, M2, R
                double *edge_state = (double *) calloc(5, sizeof(double));
                for (int i = 0; i < 5; i++) edge_state[i] = 0; // N1, N2, M1, M2, R

                double rate;
                if (i == j) {
                    if (state[i] < 2) {
                      continue;
                    }
                    if (props_i.population == 1) {
                        rate = state[i] * (state[i] - 1) / 2 / N1;
                        edge_state[0] = state[i] * (state[i] - 1) / 2;
                    } else {
                        rate = state[i] * (state[i] - 1) / 2 / N2;
                        edge_state[1] = state[i] * (state[i] - 1) / 2;                        
                    }
                } else {
                    if (state[i] < 1 || state[j] < 1) {
                      continue;
                    }
                    if (props_i.population == 1) {
                        rate = state[i] * state[j] / N1;
                        edge_state[0] = state[i] * state[j];                        
                    } else {
                        rate = state[i] * state[j] / N2;
                        edge_state[1] = state[i] * state[j];                                                    
                    }
                }
         
                memcpy(child_state, state, state_size);
                    
                // lineages with index i and j coalesce:  
                child_state[i] = child_state[i] - 1;
                child_state[j] = child_state[j] - 1;

                // coalescene into lineage with index k
                int k = _props_to_index(sample_size, props_i.locus1+props_j.locus1, props_i.locus2+props_j.locus2, props_i.population);
                child_state[k] = child_state[k] + 1;

                struct ptd_vertex *child_vertex = ptd_find_or_create_vertex(
                                graph, avl_tree, child_state
                       );
                // ptd_graph_add_edge(vertex, child_vertex, rate); 
                ptd_graph_add_edge_parameterized(vertex, child_vertex, rate, edge_state);                        
            }
            // migration
            // if (state[i] > 0 && (M > 0 || keep_null_edges)) { // M > 0 to not make zero nonsensical zero weight edges (disable if parameterized)
            if (state[i] > 0 && (M1 > 0 || M2 > 0)) { // M > 0 to not make zero nonsensical zero weight edges (disable if parameterized)

                double rate;
                double *edge_state = (double *) calloc(5, sizeof(double));
                for (int i = 0; i < 5; i++) edge_state[i] = 0; // N1, N2, M1, M2, R
                memcpy(child_state, state, state_size);

                int m;
                if (props_i.population == 1) {
                    m = 2;
                    rate = state[i] * M1;
                    edge_state[2] = state[i];
                } else {
                    m = 1;
                    rate = state[i] * M2;
                    edge_state[3] = state[i];                        
                }

                int k = _props_to_index(sample_size, props_i.locus1, props_i.locus2, m);
                child_state[i] = child_state[i] - 1;
                child_state[k] = child_state[k] + 1;

                struct ptd_vertex *child_vertex = ptd_find_or_create_vertex(
                                graph, avl_tree, child_state
                       );
                // ptd_graph_add_edge(vertex, child_vertex, rate); 
                ptd_graph_add_edge_parameterized(vertex, child_vertex, rate, edge_state);  
            }  
            // recombination
            // if (state[i] > 0 && props_i.locus1 > 0 && props_i.locus2 > 0 && (R > 0 || keep_null_edges)) { // R > 0 to not make zero nonsensical zero weight edges (disable if parameterized)
            if (state[i] > 0 && props_i.locus1 > 0 && props_i.locus2 > 0 && R > 0) { // R > 0 to not make zero nonsensical zero weight edges (disable if parameterized)


      
                double rate = R;
                double *edge_state = (double *) calloc(5, sizeof(double));
                for (int i = 0; i < 5; i++) edge_state[i] = 0; // N1, N2, M1, M2, R
                edge_state[4] = 1;
                
                memcpy(child_state, state, state_size);

                // a lineage with index i recombines to produce lineages with index k and l
                int k = _props_to_index(sample_size, props_i.locus1, 0, props_i.population);
                int l = _props_to_index(sample_size, 0, props_i.locus2, props_i.population);
                child_state[i] = child_state[i] - 1;
                child_state[k] = child_state[k] + 1;
                child_state[l] = child_state[l] + 1;

                // checks:
                properties props_k = _index_to_props(sample_size, k);
                properties props_l = _index_to_props(sample_size, l);                
                assert(props_k.locus1 + props_l.locus1 == props_i.locus1);
                assert(props_k.locus2 + props_l.locus2 == props_i.locus2);

                
                struct ptd_vertex *child_vertex = ptd_find_or_create_vertex(
                                graph, avl_tree, child_state
                       );
                // ptd_graph_add_edge(vertex, child_vertex, rate); 
                ptd_graph_add_edge_parameterized(vertex, child_vertex, rate, edge_state);             
            }             
        }
    }
    free(child_state);

    return Rcpp::XPtr<Graph>(
            new phasic::Graph(graph, avl_tree)
    );
}
