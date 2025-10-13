// [[Rcpp::depends(RcppGSL)]]

#include <Rcpp.h>
#include "stdint.h"
#include "stdlib.h"
#include "assert.h"
#include "math.h"

#include "./../../../PtDAlgorithms/api/c/ptdalgorithms.h"
#include "./../../../PtDAlgorithms/api/cpp/ptdalgorithmscpp.h"

#include "./../../../PtDAlgorithms/src/c/ptdalgorithms.c"

#include<stdio.h> 
#include<sys/stat.h>

#include <RcppGSL.h>
#include <gsl/gsl_randist.h>


using namespace Rcpp;
using namespace ptdalgorithms;


/**
 * Struct representing lineage properties corresponding to a state vector index:
 * locus1: number of decendendants at locus 1 for lineages represented by the state vector index.
 * locus2: number of decendendants at locus 2 for lineages represented by the state vector index.
 * population: the current population of lineages represented by the state vector index.
 */
struct properties {
    int is_derived;
    int n_descendants;
};


/**
 * Converts a zero-based state vector index to a props
 *
 * @param s Sample size.
 * @param i Index to be converted.
 * @return Container struct for `index`.
 */
static inline properties _index_to_props(int s, int i) {
    properties props;
    if (i > s) {
        props.is_derived = 1;  
        props.n_descendants = i - s - 1;  
    } else {
        props.is_derived = 0;     
        props.n_descendants = i;     
    }    
    return props;
}


/**
 * Same as above, but one-based for exporting to R
 */
// [[Rcpp::export]]
List index_to_props(int s, int i) {
    i  -= 1;
    properties props = _index_to_props(s, i);
    return List::create(Named("is_derived") = props.is_derived,
                        Named("n_descendants") = props.n_descendants);
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
static inline int _props_to_index(int s, int n_desc, int is_der) {
    return(is_der*(s+1) + n_desc); // check this
}

/**
 * Same as above, but one-based for exporting to R
 */
// [[Rcpp::export]]
int props_to_index(int s, int n_desc, int is_der)  {
    return(_props_to_index(s, n_desc, is_der)+1);
}

    
    
double** trans_matrix(int n, double* frequencies, int n_freqbins, double sel_coef) {
    
    /// compute binomial transition matrix for one generation
    double **m;
    m = (double **) malloc(sizeof(double*) * n_freqbins);
    for(int i = 0; i < n_freqbins; i++) {
        m[i] = (double *) malloc(sizeof(double*) * n_freqbins);
    }
    for (int i=0; i < n_freqbins; i++) {
        for (int j=0; j < n_freqbins; j++) {
            if (i == 0 || i == n_freqbins-1) {
                if (i == j) {
                    m[i][j] = 1;
                } else {
                    m[i][j] = 0;
                }
            } else {
                int x = (int)round(n*frequencies[j]);
                double p = frequencies[i];
                m[i][j] = gsl_ran_binomial_pdf(x, p, n);
            }
        }
    }
    for (int i=0; i < n_freqbins; i++) {
        double s = 0;
        for (int j=0; j < n_freqbins; j++)
            s += m[i][j];
        for (int j=0; j < n_freqbins; j++) {
           m[i][j] /= s;
        }
    }
    
    // to get trans a -> b do: m[a][b]
    
    // for (int i=0; i<n_freqbins; i++) {
    //     for(int j=0; j<n_freqbins; j++) {
    //          fprintf(stderr, "%e ", m[i][j]);
    //     }
    //     fprintf(stderr, "\n");
    // }
    // fprintf(stderr, "\n");
    
    return(m);
}


/**
 * Constructs a state space for a Kingman coalescent.
 *
 * @param sample_size Sample size.
 * @param N Scaled population size.
 * @return Graph.
 */
// [[Rcpp::export]]
SEXP construct_coalescent_selection_graph(int sample_size, int n_derived, int pop_size, int n_freqbins, double sel_coef) {
     
    // If there is only anc or der there can be only one freqbin: 0 or 1
    double *frequencies;
    if (n_derived == 0) {
        fprintf(stderr, "no derived. enforcing single der freq bin = 0)\n");
        n_freqbins = 1;
        frequencies = (double *) calloc((size_t)(n_freqbins), sizeof(double));
        frequencies[0] = 0;
    } else if (n_derived == sample_size) {
        fprintf(stderr, "no derived. enforcing single der freq bin = 1)\n");
        n_freqbins = 1;
        frequencies = (double *) calloc((size_t)(n_freqbins), sizeof(double));
        frequencies[0] = 1;
    } else {
        n_freqbins += 2; // add two fixed freq bins

        // compute freqs for each bin as vector
        frequencies = (double *) calloc((size_t)(n_freqbins+2), sizeof(double));
        frequencies[0] = 0; // lost
        frequencies[n_freqbins-1] = 1; // fixed
        for (int i = 1; i < n_freqbins-1; ++i) {
           frequencies[i] = i/((double)n_freqbins-2) - 1/(double)(n_freqbins-2)/2 ; // center of bin
        }
    }
    
    // fprintf(stderr, "freq bins: ");
    // for (int i=0; i<n_freqbins; i++)
    //     fprintf(stderr, "%f ", frequencies[i]);
    // fprintf(stderr, "\n");

    
    double N = 1;
     
    // state vector length
    const int state_length = (sample_size + 1) * 2 + 1;
    const size_t state_size = sizeof(int) * state_length;
    
    int freqbin_index = state_length - 1;
    
    double der_freq = n_derived / (double)sample_size;
    
    // find freq bin closest to derived freq in sample
    double* diffs = (double *) calloc((double)(n_freqbins), sizeof(double));
    // double* diffs = (double *) malloc(sizeof(double) * n_freqbins);
    int start_freq_bin = 0;
    for (int i = 0; i < n_freqbins; i++) {     
       if(abs(frequencies[i] - der_freq) < abs(frequencies[start_freq_bin] - der_freq)) {    
           start_freq_bin = i;
       }
    }       

    // fprintf(stderr, "startbin: %d\n", start_freq_bin);

    double **freq_trans = trans_matrix(pop_size, frequencies, n_freqbins, sel_coef);

    struct ptd_graph *graph = ptd_graph_create((size_t) state_length);
    struct ptd_avl_tree *avl_tree = ptd_avl_tree_create((size_t) state_length);
    
    int *initial_state = (int *) calloc((size_t) state_length, sizeof(int));
    for (int i=0; i < n_freqbins; ++i) {
        if (n_freqbins > 1 && (i == 0 || i == n_freqbins-1)) {
            // fixed bins now allowed among initial states
            continue;
        }
        
        for (int x=0; x < state_length; x++) {
            initial_state[x] = 0;
        }
        
        initial_state[_props_to_index(sample_size, 1, 0)] = sample_size - n_derived;
        initial_state[_props_to_index(sample_size, 1, 1)] = n_derived;
        initial_state[freqbin_index] = i;

        double ipv_rate = freq_trans[start_freq_bin][i];

        // fprintf(stderr, "%f\n", ipv_rate);
        
        if (abs(ipv_rate) > 1.0e-20) {
            // fprintf(stderr, "ipv %f\n", ipv_rate);

            ptd_graph_add_edge(
                    graph->starting_vertex,
                    ptd_find_or_create_vertex(graph, avl_tree, initial_state),
                    ipv_rate
            );
        }
        
    }
    free(initial_state);

    int *child_state = (int *) malloc(state_size);
    
    for (size_t index = 1; index < graph->vertices_length; ++index) {
        struct ptd_vertex *vertex = graph->vertices[index];
        int *state = vertex->state;
                
        int lineages_left = 0;

        for (int i = 0; i < state_length-1; ++i) {
            lineages_left += state[i];
        }

        if (lineages_left == 0 || lineages_left == 1) {
            // Only one lineage left, absorb
            continue;
        }

        int tot_ancestral = 0;
        int tot_derived = 0;
        for (int i = 0; i < state_length-1; ++i) {
            properties lin_props = _index_to_props(sample_size, i);
            if (lin_props.is_derived) {
                tot_derived += state[i];
            } else {
                tot_ancestral += state[i];
            }
        }
        
        
        int cur_freq_bin = state[freqbin_index];
        
        // loop over freqs other than cur_freq_bin 
        for (int freq_bin = 0; freq_bin < n_freqbins; freq_bin++) {
            if (freq_bin == cur_freq_bin) {
                continue;   
            }

            if (tot_ancestral > 0 && freq_bin == n_freqbins-1) {
                // cannot loose ancestral allele while anc lineages remain
                continue;
            }
            if (tot_derived > 0 && freq_bin == 0) {
                // cannot loose derived allele while der lineages remain
                continue;
            }

            double rate = freq_trans[cur_freq_bin][freq_bin];

            rate *= pop_size; // to make it scale the same as N

            if (abs(rate) < 1.0e-20) {
                continue;
            }

            memcpy(child_state, state, state_size);

            child_state[freqbin_index] = freq_bin;

            struct ptd_vertex *child_vertex = ptd_find_or_create_vertex(
                        graph, avl_tree, child_state
               );
            ptd_graph_add_edge(vertex, child_vertex, rate); 
        }

        
        // lineage one
        for (int i = 0; i < state_length-1; ++i) {
            properties props_i = _index_to_props(sample_size, i);
                
            // lineage two
            for (int j = i; j < state_length-1; ++j) {
                properties props_j = _index_to_props(sample_size, j);
                    
                double coal_rate;
                double freq;
                if (tot_derived > 1) {
                    // there are still derived lineages left
                    if ((props_i.is_derived == 1) != (props_i.is_derived == 1)) { // xor
                        // so if the lineages are not of same kind we skip
                        continue;
                    }
                    
                    if (props_i.is_derived == 1) {
                        // both derived
                        freq = frequencies[state[freqbin_index]];
                    } else {
                        // both ancestral
                        freq = 1 - frequencies[state[freqbin_index]];
                    }                    
                } else {
                    // only one derived lineage remaining wich can coalesce with the ancestral
                    freq = 1;
                }
            
                if (i == j) {
                    if (state[i] < 2) {
                      continue;
                    }
                    coal_rate = state[i] * (state[i] - 1) / 2 / N;
                } else {
                    if (state[i] < 1 || state[j] < 1) {
                      continue;
                    }
                    coal_rate = state[i] * state[j] / N;
                }            
                 
                double rate = coal_rate / freq;
                // fprintf(stderr, "%e %e %e\n", coal_rate, freq, rate);
                                                
                memcpy(child_state, state, state_size);

                // lineages with index i and j coalesce:  
                child_state[i] = child_state[i] - 1;
                child_state[j] = child_state[j] - 1;

                // coalescene into lineage with index k
                int k;
                if (props_i.is_derived * props_j.is_derived != 0) { 
                    // both derived
                    k = _props_to_index(sample_size, props_i.n_descendants + props_j.n_descendants, 1);
                } else {
                    // one or both ancestral
                    k = _props_to_index(sample_size, props_i.n_descendants + props_j.n_descendants, 0);
                }

                child_state[k] = child_state[k] + 1;                

                struct ptd_vertex *child_vertex = ptd_find_or_create_vertex(
                                graph, avl_tree, child_state
                       );
                ptd_graph_add_edge(vertex, child_vertex, rate); 


            }
        }
    }
    free(child_state);
    free(frequencies);
    return Rcpp::XPtr<Graph>(
            new ptdalgorithms::Graph(graph, avl_tree)
    );
}
