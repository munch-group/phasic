#include <Rcpp.h>
#include "stdint.h"
#include "stdlib.h"
#include "assert.h"
#include "./../../../PtDAlgorithms/api/c/ptdalgorithms.h"
#include "./../../../PtDAlgorithms/api/cpp/ptdalgorithmscpp.h"

#include "./../../../PtDAlgorithms/src/c/ptdalgorithms.c"

using namespace Rcpp;
using namespace ptdalgorithms;


/**
 * Struct representing lineage properties corresponding to a state vector index:
 * locus1: number of decendendants at locus 1 for lineages represented by the state vector index.
 * locus2: number of decendendants at locus 2 for lineages represented by the state vector index.
 * population: the current population of lineages represented by the state vector index.
 */
struct properties {
    int ancestral;
    int derived;
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
    properties props;
    props.derived = i / s;   
    props.ancestral = fmod(i, s);     
    return props;
}


/**
 * Same as above, but one-based for exporting to R
 */
// [[Rcpp::export]]
List index_to_props(int s, int i) {
    assert(i <= 1);
    i  -= 1;
    properties props = _index_to_props(s, i);
    return List::create(Named("ancestral") = props.ancestral , _["derived"] = props.derived);
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
static inline int _props_to_index(int s, int n_anc, int n_der) {
    return(s*n_der+n_anc);
}

/**
 * Same as above, but one-based for exporting to R
 */
// [[Rcpp::export]]
int props_to_index(int s, int n_anc, int n_der)  {
    return(_props_to_index(s, n_anc, n_der)+1);
}

// double bin_frequency(int n, int f) {

//     if (f == 0) {
//         return(0);
//     }
//     if (f == n+2) {
//         return(1);
//     }
//     return(((f-1)+f)/(double)n/2);
// }

// double rel_freq_jump_prob(int n_freqbins, int end_freq_bin, int start_freq_bin, 
//                           double drift_time, double s) {

//     // compute freqs for each bin as vector (maybe cache that)

//     double *frequencies = (double *) calloc((size_t)(n_freqbins+2), sizeof(double));
//     frequencies[0] = 0; // lossed frequency
//     frequencies[n_freqbins+2] = 1; // fixed frequency
//     for (int i = 1; i < n_freqbins+1; ++i) {
//        frequencies[i] = ((i-1)+i)/(double)n_freqbins/2  ;
//     }
//     // compute jump probs from each start_freq_bin to end_freq_bin given time and s
//     double *jump_probs = (double *) calloc(n_freqbins+2, sizeof(double));
//     for (int i = 0; i < n_freqbins; ++i) {
//        jump_probs[i] = 1; // dummy jump prob from frequencies[i] to frequencies[end_freq_bin]
//     }
//     // total jump prob
//     double tot_jumpprob = 0;
//     for (int i = 0; i < n_freqbins; ++i) {
//         tot_jumpprob += jump_probs[i];
//     }
//     double rel_jump_prob = jump_probs[start_freq_bin] / tot_jumpprob;
    
//     free(jump_probs);
//     free(frequencies);
//     return(rel_jump_prob);    
// }


/**
 * Constructs a state space for a Kingman coalescent.
 *
 * @param sample_size Sample size.
 * @param N Scaled population size.
 * @return Graph.
 */
// [[Rcpp::export]]
SEXP construct_coalescent_selection_graph(int sample_size, double N, 
                                     int n_derived, int n_freqbins, double s) {

    fprintf(stderr, "Beginning\n");
    
    // state vector length
    const int state_length = sample_size * 2 + 1;
    const size_t state_size = sizeof(int) * state_length;
    
    struct ptd_graph *graph = ptd_graph_create((size_t) state_length);
    struct ptd_avl_tree *avl_tree = ptd_avl_tree_create((size_t) state_length);

    int freqbin_index = state_length - 1;
    
    fprintf(stderr, "index %d\n", _props_to_index(sample_size, 1, 0));        

    int *initial_state = (int *) calloc((size_t) state_length, sizeof(int));
    
    initial_state[_props_to_index(sample_size, 1, 0)] = sample_size;
    initial_state[freqbin_index] = 1;

    ptd_graph_add_edge(
            graph->starting_vertex,
            ptd_find_or_create_vertex(graph, avl_tree, initial_state),
            1
    );
    free(initial_state);
    int *child_state = (int *) malloc(state_size);

    fprintf(stderr, "Starting\n");
    
    for (size_t index = 1; index < graph->vertices_length; ++index) {
        struct ptd_vertex *vertex = graph->vertices[index];
        int *state = vertex->state;

        // fprintf(stderr, "%zu %zu\n", index, graph->vertices_length);
        
        int lineages_left = 0;

        for (int i = 0; i < state_length-1; ++i) { // -1 becuase last filed is the freqbin
            lineages_left += state[i];
        }

        if (lineages_left == 0 || lineages_left == 1) {
            // Only one lineage left, absorb
            continue;
        }

        int derived_left = 0;
        for (int i = 0; i < state_length-1; ++i) { // -1 becuase last filed is the freqbin
            properties lin_props = _index_to_props(sample_size, i);
            derived_left += state[i] * lin_props.derived;
        }
        
        // lineage one
        for (int i = 0; i < state_length; ++i) {
            properties props_i = _index_to_props(sample_size, i);
                
            // lineage two
            for (int j = i; j < state_length; ++j) {
                properties props_j = _index_to_props(sample_size, j);
                    
                // // frequency bin to jump to
                // for (int f = 0; f < n_freqbins+2; ++f) {
                    
                    // // get derived freuqency for frequency bin in the current state
                    // double current_freq = bin_frequency(n_freqbins, state[freqbin_index]);

                    double coal_rate;
                    
//                     // do not coalesce anc and der before all der have coalesced
//                     if (derived_left > 1) {
//                         // assert that each lienage has only andestral or derived descendants
//                         assert(props_i.ancestral == 0 != props_i.derived == 0 );
//                         assert(props_i.ancestral == 0 != props_i.derived == 0 );

//                         // so we know that each lineage only have one kind of descendants.
//                         // so continue if one is derived lineage 
//                         // and the other is ancestral lineage
//                         if (props_i.ancestral > 1 != props_j.ancestral > 1) {
//                             // different kinds of lineages
//                             continue;
//                         }

//                         if (props_i.ancestral > 1 == props_j.ancestral > 1) {
//                             // both pure ancestral

//                             if (i == j) {
//                                 if (props_i.ancestral < 2) {
//                                   continue;
//                                 }
//                                 coal_rate = props_i.ancestral * (props_i.ancestral - 1) / 2 / (N*(1-current_freq));
//                             } else {
//                                 if (props_i.ancestral < 1 || props_j.ancestral < 1) {
//                                   continue;
//                                 }
//                                 coal_rate = props_i.ancestral * props_j.ancestral / (N*(1-current_freq));
//                             }
                            
//                         } else {
//                             // both pure derived 
//                             assert(props_i.derived > 1 == props_j.derived > 1);
                            
//                             if (i == j) {
//                                 if (props_i.derived < 2) {
//                                   continue;
//                                 }
//                                 coal_rate = props_i.derived * (props_i.derived - 1) / 2 / (N*current_freq);
//                             } else {
//                                 if (props_i.derived < 1 || props_j.derived < 1) {
//                                   continue;
//                                 }
//                                 coal_rate = props_i.derived * props_j.derived / (N*current_freq);
//                             }
                            
//                         }
//                     } else {

                        // all derived coalesced
                        int n_lin_i = props_i.ancestral * props_i.derived;
                        int n_lin_j = props_j.ancestral * props_j.derived;

                        if (i == j) {
                            if (state[i] < 2) {
                              continue;
                            }
                            coal_rate = n_lin_i * (n_lin_i - 1) / 2 / N;
                        } else {
                            if (n_lin_i < 1 || n_lin_j < 1) {
                              continue;
                            }
                            coal_rate = n_lin_i * n_lin_j / N;
                        }
                
                    // }

                    double relative_freq_jump_prob = 1; // rel_freq_jump_prob(n_freqbins, state[freqbin_index], f, 1/coal_rate, s);
                    if (abs(relative_freq_jump_prob) < 1e-12) { // is zero
                        continue;
                    }
                    double rate = coal_rate * relative_freq_jump_prob;
                    
                    memcpy(child_state, state, state_size);

                    // lineages with index i and j coalesce:  
                    child_state[i] = child_state[i] - 1;
                    child_state[j] = child_state[j] - 1;

                    // coalescene into lineage with index k
                    int k = _props_to_index(sample_size, props_i.ancestral+props_j.ancestral,
                                            props_i.derived+props_j.derived);
                    child_state[k] = child_state[k] + 1;
                    
                    // // frequency
                    // child_state[freqbin_index] = f;

                    struct ptd_vertex *child_vertex = ptd_find_or_create_vertex(
                                    graph, avl_tree, child_state
                           );
                    ptd_graph_add_edge(vertex, child_vertex, rate); 
                // }
            }
        }
    }
    free(child_state);

    return Rcpp::XPtr<Graph>(
            new ptdalgorithms::Graph(graph, avl_tree)
    );
}
