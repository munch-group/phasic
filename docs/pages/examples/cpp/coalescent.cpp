// cppimport
#include <pybind11/pybind11.h>
#include <ptdalgorithms.h>

namespace py = pybind11;

using namespace pybind11::literals; // to bring in the `_a` literal

/* Basic C libraries */
#include "stdint.h"
#include "stdlib.h"

/* ----------------- Don't change the code above! ----------------- */

#include "assert.h"
#include "math.h"

#include<stdio.h> 
#include<sys/stat.h>

#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

#include <gmp.h>
#include <gmpxx.h>




/**
 * Struct representing lineage properties corresponding to a state vector index:
 * id_derived: whether the variant is derived. 
 * n_descendants: number of decendendants.
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
 * @param n_desc Number of decendants.
 * @param is_der Whether the variant is derived.
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


int cmpfun (const void * a, const void * b) {
   return ( *(double*)a - *(double*)b );
}

/**
 * Numerically stable summation
 *
 * @param x Vector to sort.
 * @param n Vector length
 * @return Sum.
 */
long double kahan_sum(long double x[], int n) {
    qsort(x, n, sizeof(long double), cmpfun);
    long double s = x[0];
    long double c = 0.0;
    for (int i = 1; i < n; i++) {
        long double y = x[i] - c;
        long double t = s + y;
        c = (t - s) - y;
        s = t;
    }
    return s;
}

void multiply_matrix(double **mat1, double **mat2, double **result, int dim) {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            result[i][j] = 0;
            for (int k = 0; k < dim; ++k) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
}

// Function to raise a matrix to a power
void power_matrix(double **mat, double **result, int dim, int power) {
    // Initialize result matrix as the identity matrix
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            result[i][j] = (i == j) ? 1 : 0;
        }
    }
    double **temp;
    temp = (double **) malloc(sizeof(double*) * dim);
    for(int i = 0; i < dim; i++) {
        temp[i] = (double *) malloc(sizeof(double*) * dim);
    }  
    // Multiply mat by itself 'power' times
    for (int i = 0; i < power; ++i) {      
        multiply_matrix(result, mat, temp, dim);
        // Copy the result back to the result matrix
        for (int j = 0; j < dim; ++j) {
            for (int k = 0; k < dim; ++k) {
                result[j][k] = temp[j][k];
            }
        }        
    }
}


/**
 * Constructs a transition matrix for discrete frequency bins.
 *
 * @param n Population size.
 * @param bins Vector of binned frequencies.
 * @param n_freqbins Nr. of binned frequencies.
 * @param s Selection coeficient.
 * @return matrix.
 */
long double** trans_matrix(int n, int* bins, int n_freqbins, double s, int inclzero) {
    // to get trans a -> b do: m[a][b]

    assert(inclzero == 0 || inclzero == 1);

    s *= -1; // flip sign of s because we are looking backwards in time...

    // compute binomial transition matrix for one generation
    long double **m;
    m = (long double **) malloc(sizeof(long double*) * n_freqbins);
    for(int i = 0; i < n_freqbins; i++) {
        m[i] = (long double *) malloc(sizeof(long double*) * n_freqbins);
    }
    double p;
    for (int i=0; i < n_freqbins; i++) {
        for (int j=0; j < n_freqbins; j++) {
            // no transition from zero freq and only nonzero 
            // transition to zero freq if spcecified
            if (i == 0 || j == 0 && inclzero == 0) {
                m[i][j] = 0;
                continue;
            }
            p = (long double)bins[i] / n;
            p = p*(1+s)/(p*(1+s)+1-p);
            m[i][j] = gsl_ran_binomial_pdf(bins[j], p, n) ;
        }
    }
    // normalize rows to sum to one
    // (except first row with transitions from zero freq that are all zeros)
    for (int i=1; i < n_freqbins; i++) {
        long double s = kahan_sum(m[i], n_freqbins);
        for (int j=0; j < n_freqbins; j++) {
           m[i][j] /= s;
        }
    }

    // for (int i=0; i<n_freqbins; i++) {
    //     for(int j=0; j<n_freqbins; j++) {
    //          fprintf(stderr, "%Le ", m[i][j]);
    //     }
    //     fprintf(stderr, "\n");
    // }
    // fprintf(stderr, "\n");
    
    return(m);
}

double *linspace(double a, double b, int num) {
    double *v = (double *) malloc(sizeof(double*) * num);
    for (int i=0; i < num; i++) {
          v[i] = a + i * (b - a) / (double (num-1));
     }
    return v;
}


/**
 * Constructs a state space for a Kingman coalescent.
 *
 * @param sample_size Sample size.
 * @param N Scaled population size.
 * @return Graph.
 */
// [[Rcpp::export]]

// ptdalgorithms::Graph build(int starting_rabbits, float flooding_left, float flooding_right) {

SEXP construct_coalescent_selection_graph(int sample_size, int n_derived, int pop_size, int n_freqbins, double sel_coef) {

    // scaled population size
    double N = 1;

    // state vector length
    const int state_length = (sample_size + 1) * 2 + 1;
    const size_t state_size = sizeof(int) * state_length;
    
    //// BINS FOR FREQUENCIES: ////////////////////////////////////////////////

    n_freqbins = pop_size;
    
    int *bins = (int *) malloc(sizeof(int) * n_freqbins);
    for (int i=0; i < n_freqbins; i++)
        bins[i] = i;

    for (int i=0; i < n_freqbins-1; i++) assert(bins[i] != bins[i+1]) ;

    // for (int i=0; i < n_freqbins; i++)
    //     fprintf(stderr, "%d\n", bins[i]);


    ///////
    
    // double *bin_probs = linspace(10/((double) pop_size), 1-10/((double) pop_size), n_freqbins);

    // n_freqbins += 1; // extra zero bin
    // int *bins = (int *) malloc(sizeof(int) * n_freqbins);
    // for (int i=0; i < n_freqbins; i++) {
    //     if (i == 0) {
    //         bins[0] = 0;
    //     } else {            
    //         bins[i] = (int) round(gsl_cdf_beta_Pinv(bin_probs[i-1], 0.5, 0.5) * pop_size) ;
    //         // (bin_probs represent the nonzero bins, henxe the i-1
    //     }
    // }
    // // make sure they diff by at least one
    // for (int i=0; i < n_freqbins; i++) {
    //     if (i < (double)n_freqbins/2 && bins[i] == bins[i+1]) {
    //         bins[i] += 1;
    //     }
    //     if (i > (double)n_freqbins/2 && bins[i] == bins[i+1]) {
    //         bins[i] -= 1;
    //     }
    // }
    
    // for (int i=0; i < n_freqbins-1; i++) assert(bins[i] != bins[i+1]) ;

    // for (int i=0; i < n_freqbins; i++)
    //     fprintf(stderr, "%e %d\n", bin_probs[i], bins[i]);

    //// IPV: /////////////////////////////////////////////////////////////////

    // index in the state vector holding the bin
    int state_vector_freq_bin_index = state_length - 1;

    // initial probability vector
    long double *ipv_rates = (long double *) malloc(sizeof(long double*) * n_freqbins);
    for (int i=0; i < n_freqbins; i++) {
        if (i == 0) {
            ipv_rates[i] = 0;
        } else {
            ipv_rates[i] = gsl_ran_beta_pdf(bins[i]/((long double)pop_size), n_derived, sample_size - n_derived);
        }
    }
    // scale ipv rates to unit sum
    long double ipv_sum = kahan_sum(ipv_rates, n_freqbins);
    for (int i=0; i < n_freqbins; i++) {
       ipv_rates[i] /= ipv_sum;
    }        

    // MAYBE INSTEAD ALWAYS USE TRANS INCL ZERO AND ONE. IT SHOULD WORK IF COAL RATE IS IMEDIATE WHEN RATE IS ZERO...

    // ALSO BELOW: WHEN D == 1, ONLY ADD ZERO FREQ STATES
    
    //// FREQUENCY TRANSITION MATRIX: /////////////////////////////////////////

    // frequency transition matrix
    long double **freq_trans = trans_matrix(pop_size, bins, n_freqbins, sel_coef, 0);

    // frequency transition matrix with nonzero probability of transition to zero freq
    long double **freq_trans_incl_zero = trans_matrix(pop_size, bins, n_freqbins, sel_coef, 1);

    //// INITIAL STATE: ///////////////////////////////////////////////////////
    
    // initialize graph
    struct ptd_graph *graph = ptd_graph_create((size_t) state_length);
    struct ptd_avl_tree *avl_tree = ptd_avl_tree_create((size_t) state_length);

    // initial state>
    int *initial_state = (int *) calloc((size_t) state_length, sizeof(int));
    for (int i=0; i < n_freqbins; ++i) {

        // intialize with zeros
        for (int x=0; x < state_length; x++) {
            initial_state[x] = 0;
        }
        // set nr ancestral singleton lineages
        initial_state[_props_to_index(sample_size, 1, 0)] = sample_size - n_derived;
        // set nr derived singleton lineages
        initial_state[_props_to_index(sample_size, 1, 1)] = n_derived;
        // set frequency bin
        initial_state[state_vector_freq_bin_index] = i;

        // fprintf(stderr, "%f\n", ipv_rate);
        // if (abs(ipv_rate) > 1.0e-30) {
        //     fprintf(stderr, "ipv %f\n", ipv_rate);

        ptd_graph_add_edge(
                    graph->starting_vertex,
                    ptd_find_or_create_vertex(graph, avl_tree, initial_state),
                    ipv_rates[i]
            );
    }
    free(initial_state);

    //// STATE SPACE: /////////////////////////////////////////////////////////
    
    // construct state space
    int *child_state = (int *) malloc(state_size);
    for (size_t index = 1; index < graph->vertices_length; ++index) {

        // get vertex
        struct ptd_vertex *vertex = graph->vertices[index];

        // get state
        int *state = vertex->state;

        // get nr of live lineages
        int n = 0; 
        for (int i = 0; i < state_length-1; ++i) {
            n += state[i];
        }

        // if (n == 0 || n == 1) {
        if (n == 1) {
            // only one lineage left
            // vertex has not chilndren, absorb
            continue;
        }

        // get number of live derived lineages
        int d = 0; 
        for (int i = 0; i < state_length-1; ++i) {
            properties lin_props = _index_to_props(sample_size, i);
            if (lin_props.is_derived) {
                d += state[i];
            }
        }
        // get index of the current frequency bin
        int cur_freq_bin = state[state_vector_freq_bin_index];

        if (cur_freq_bin == 0 && d > 1) {
            // no transition from zero bin unles only one derived remains
            continue;
        }

        
        // transitions to other states differing only by frequency bin
        for (int freq_bin = 0; freq_bin < n_freqbins; freq_bin++) {

            if (freq_bin == cur_freq_bin) {
                continue;   
            }

            // CHANGED HERE
            if (freq_bin == 0 && d > 1) {
                // no transition to zero bin unles only one derived remains
                continue;   
            }

            if (freq_bin > 0 && d == 0) {
                // no transitions between non-zero bins if no derived remain
                continue;   
            }

            // CHANGED HERE
            // double rate = freq_trans[cur_freq_bin][freq_bin] * pop_size;
            long double rate;
            if (d > 1) {
                rate = freq_trans[cur_freq_bin][freq_bin];
                rate *= pop_size; // to make it scale the same as N
            } else {
                // compute rate allowing transition to zero freq
                rate = freq_trans_incl_zero[cur_freq_bin][freq_bin];
                rate *= pop_size; // to make it scale the same as N
            }
            
            // get child state vector by copying state vector
            memcpy(child_state, state, state_size);

            // set frequency bin of child state vector
            child_state[state_vector_freq_bin_index] = freq_bin;

            // make child vertex
            struct ptd_vertex *child_vertex = ptd_find_or_create_vertex(
                        graph, avl_tree, child_state
               );
            // add edge to child vertex
            ptd_graph_add_edge(vertex, child_vertex, rate);        
        }
        
        // lineage one
        for (int i = 0; i < state_length-1; ++i) {
            properties props_i = _index_to_props(sample_size, i);
                
            // lineage two
            for (int j = i; j < state_length-1; ++j) {
                properties props_j = _index_to_props(sample_size, j);

                // we cannot coalesce ancestral and derived unless freq is zero
                if (props_i.is_derived != props_j.is_derived && cur_freq_bin != 0) {
                    continue;
                }

                
                long double coal_rate;
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
                
                long double freq;
                if (d > 1) {
                    // // there are still derived lineages left
                    // if ((props_i.is_derived == 1) != (props_j.is_derived == 1)) { // xor
                    //     // so if the lineages are not of same kind we skip
                    //     continue;
                    // }
                    
                    if (props_i.is_derived == 1) {
                        // both derived
                        freq = bins[state[state_vector_freq_bin_index]] / (double)pop_size;
                    } else {
                        // both ancestral
                        freq = (pop_size - bins[state[state_vector_freq_bin_index]]) / (double)pop_size;
                    }                    
                } else {
                    // only one derived lineage remaining wich can coalesce with the ancestral
                    freq = 1;
                }

                // CHANGED HERE
                long double rate = coal_rate / freq;
                // double rate;
                // if (freq == 0) {
                //     rate = 1000000;
                // } else {
                //     rate = coal_rate / freq;
                // }


                
                // get child state vector by copying state vector
                memcpy(child_state, state, state_size);

                // change child state vector to reflect that lineages with index i and j coalesce:  
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

                // make child vertex
                struct ptd_vertex *child_vertex = ptd_find_or_create_vertex(
                                graph, avl_tree, child_state
                       );
                // add edge to child vertex
                ptd_graph_add_edge(vertex, child_vertex, rate); 


            }
        }
    }
    free(child_state);
    free(bins);
    return Rcpp::XPtr<Graph>(
            new ptdalgorithms::Graph(graph, avl_tree)
    );
}
