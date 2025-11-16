// [[Rcpp::depends(RcppGSL)]]

#include <Rcpp.h>
#include "stdint.h"
#include "stdlib.h"
#include "assert.h"
#include "math.h"

#include "./../../../phasic/api/c/phasic.h"
#include "./../../../phasic/api/cpp/phasiccpp.h"

#include "./../../../phasic/src/c/phasic.c"

#include<stdio.h> 
#include<sys/stat.h>

#include <RcppGSL.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

#include <gmp.h>
#include <gmpxx.h>

using namespace Rcpp;
using namespace phasic;


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
float kahan_sum(double x[], int n) {
    qsort(x, n, sizeof(double), cmpfun);
    double s = x[0];
    double c = 0.0;
    for (int i = 1; i < n; i++) {
        double y = x[i] - c;
        double t = s + y;
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

// typedef struct {
//     int num;
//     int index;
// } Number;

// int compare(const void* a, const void* b) {
//     return ((Number*)a)->num - ((Number*)b)->num;
// }

// int* twoSum(int* nums, int numsSize, int target, int* returnSize) {
//     Number numbers[numsSize];
//     for(int i = 0; i < numsSize; i++) {
//         numbers[i].num = nums[i];
//         numbers[i].index = i;
//     }
//     qsort(numbers, numsSize, sizeof(Number), compare);

//     int left = 0, right = numsSize - 1;
//     while(left < right) {
//         int sum = numbers[left].num + numbers[right].num;
//         if(sum == target) {
//             *returnSize = 2;
//             int* result = (int*)malloc(*returnSize * sizeof(int));
//             result[0] = numbers[left].index;
//             result[1] = numbers[right].index;
//             return result;
//         } else if(sum < target) {
//             left++;
//         } else {
//             right--;
//         }
//     }
//     *returnSize = 0;
//     return NULL;
// }

// faster alternative with Fast2Sum....
// function KahanSum2(input)
//     // Prepare the accumulator.
//     var sum = 0.0
//     // A running compensation for lost low-order bits.
//     var c = 0.0
//     // The array input has elements indexed 
//     for i = 1 to input.length do
//         // c is zero the first time around.
//         var y = input[i] + c
//         // sum + c is an approximation to the exact sum.
//         (sum,c) = Fast2Sum(sum,y)
//     // Next time around, the lost low part will be added to y in a fresh attempt.
//     next i
//     return sum
    

/**
 * Constructs a transition matrix for discrete frequency bins.
 *
 * @param n Population size.
 * @param bins Vector of binned frequencies.
 * @param n_freqbins Nr. of binned frequencies.
 * @param s Selection coeficient.
 * @return matrix.
 */
double** trans_matrix(int n, int* bins, int n_freqbins, double s) {


    s *= -1; // flip sign of s because we are looking backwards in time...
    
    // mpf_t x, y;
    // mpf_init(x);     
    // mpf_init2 (y, 256);
    
//     mpf_t test;
//     mpf_add(test, 3.0, 9.9);
    
    /// compute binomial transition matrix for one generation
    double **m;
    m = (double **) malloc(sizeof(double*) * n_freqbins);
    for(int i = 0; i < n_freqbins; i++) {
        m[i] = (double *) malloc(sizeof(double*) * n_freqbins);
    }
    for (int i=0; i < n_freqbins; i++) {
        for (int j=0; j < n_freqbins; j++) {
            // int x = (int)round(n*frequencies[j]);
            int x = bins[j];
            // double p = frequencies[i];
            double p = (double)bins[i] / n;
            p = p*(1+s)/(p*(1+s)+1-p);
            m[i][j] = gsl_ran_binomial_pdf(x, p, n) ;
            // fprintf(stderr, "%d %f %d %f\n", x, p, n, m[i][j]);
        }
    }
    
    for (int i=0; i < n_freqbins; i++) {
        double s = kahan_sum(m[i], n_freqbins);
        for (int j=0; j < n_freqbins; j++) {
           m[i][j] /= s;
        }
    }

    // double **temp;
    // temp = (double **) malloc(sizeof(double*) * n_freqbins);
    // for(int i = 0; i < n_freqbins; i++) {
    //     temp[i] = (double *) malloc(sizeof(double*) * n_freqbins);
    // }      
    // power_matrix(m, temp, n_freqbins, n);
    // m = temp;


    // to get trans a -> b do: m[a][b]
    
    for (int i=0; i<n_freqbins; i++) {
        for(int j=0; j<n_freqbins; j++) {
             fprintf(stderr, "%e ", m[i][j]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
    
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
SEXP construct_coalescent_selection_graph(int sample_size, int n_derived, int pop_size, int n_freqbins, double sel_coef) {
     
    // // compute freqs for each bin as vector
    // int *bins = (int *) calloc((size_t)(n_freqbins+2), sizeof(int));
    // for (int i = 0; i < n_freqbins; ++i) {
    //    int bin_center = (int)round(pop_size*((i+1)/((double)n_freqbins) - 1/(double)(n_freqbins)/2)) ;
    //    bins[i] = bin_center;
    //    fprintf(stderr, "bin %d\n", bin_center);
    // }

    double *bin_probs = linspace(10/((double) pop_size), 1-10/((double) pop_size), n_freqbins);
    // int *bins = (int *) calloc((size_t)(n_freqbins), sizeof(int));
    int *bins = (int *) malloc(sizeof(int) * n_freqbins);
    for (int i=0; i < n_freqbins; i++) {
        bins[i] = (int) round(gsl_cdf_beta_Pinv(bin_probs[i], 0.5, 0.5) * pop_size) ;
        fprintf(stderr, "%e %d\n", bin_probs[i], bins[i]);
    }

    
    // int *bins = (int *) calloc((size_t)(n_freqbins+2), sizeof(int));
    // bins[i] = 0
    // for (int i = 1; i < n_freqbins+1; ++i) {
    //    int bin_center = (int)round(pop_size*((i+1)/((double)n_freqbins) - 1/(double)(n_freqbins)/2)) ;
    //    bins[i] = bin_center;
    //    fprintf(stderr, "bin %d\n", bin_center);
    // }
    // bins[n_freqbins+1] = 0;
    
    // int *frequencies = (int *) calloc((size_t)(n_freqbins+2), sizeof(int));
    // for (int i = 0; i < n_freqbins; ++i) {
    //    double f = (i+1)/((double)n_freqbins) - 1/(double)(n_freqbins)/2 ; // center of bin
    //    frequencies[i] = (int)round(f*pop_size);
    // }

    
    // fprintf(stderr, "freq bins: ");
    // for (int i=0; i<n_freqbins; i++)
    //     fprintf(stderr, "%f ", frequencies[i]);
    // fprintf(stderr, "\n");

    
    double N = 1;
     
    // state vector length
    const int state_length = (sample_size + 1) * 2 + 1;
    const size_t state_size = sizeof(int) * state_length;
    
    int bin_index = state_length - 1;
    
    double der_freq = n_derived / (double)sample_size;
    
    // find freq bin closest to derived freq in sample
    double* diffs = (double *) calloc((double)(n_freqbins), sizeof(double));
    // double* diffs = (double *) malloc(sizeof(double) * n_freqbins);

    // /********/
    // for (int i = 0; i < n_freqbins; i++) {  
    //     diffs[i] = abs(bins[i]/((double)pop_size) - der_freq);
    // }
    // double diff_sum = kahan_sum(diffs, n_freqbins);        
    // /********/
    
    int start_bin = 0;
    for (int i = 0; i < n_freqbins; i++) {     
       if(abs(bins[i] - pop_size*der_freq) < abs(bins[start_bin] - pop_size*der_freq)) {    
           start_bin = i;
       }
    }      

        
    double *ipv_rates = (double *) malloc(sizeof(double*) * n_freqbins);
    for (int i=0; i < n_freqbins; i++) {
        ipv_rates[i] = gsl_ran_beta_pdf(bins[i]/((double)pop_size), n_derived, sample_size - n_derived) ;
    }
    double ipv_sum = kahan_sum(ipv_rates, n_freqbins);
    for (int i=0; i < n_freqbins; i++) {
       ipv_rates[i] /= ipv_sum;
    }        
    fprintf(stderr, "ipv_rates: ");
    for (int i=0; i<n_freqbins; i++)
        fprintf(stderr, "%f ", ipv_rates[i]);
    fprintf(stderr, "\n");

    
    // fprintf(stderr, "startbin: %d\n", start_freq_bin);

    double **freq_trans = trans_matrix(pop_size, bins, n_freqbins, sel_coef);

    struct ptd_graph *graph = ptd_graph_create((size_t) state_length);
    struct ptd_avl_tree *avl_tree = ptd_avl_tree_create((size_t) state_length);
    
    int *initial_state = (int *) calloc((size_t) state_length, sizeof(int));
    for (int i=0; i < n_freqbins; ++i) {
        
        for (int x=0; x < state_length; x++) {
            initial_state[x] = 0;
        }
        
        initial_state[_props_to_index(sample_size, 1, 0)] = sample_size - n_derived;
        initial_state[_props_to_index(sample_size, 1, 1)] = n_derived;
        initial_state[bin_index] = i;
        
        /********/
        // double ipv_rate = 0;
        
        // for (int b = 0; b < n_freqbins; b++) {     
        //     ipv_rate += (1-diffs[b]/diff_sum) * freq_trans[b][i];
        // } 
        /********/

        // double ipv_rate;
        // if (bins[start_bin1] == n_derived) {
        //      ipv_rate = freq_trans[start_bin1][i];
        // } else {
        //     double d1 = abs(bins[start_bin1] - n_derived);
        //     double d2 = abs(bins[start_bin2] - n_derived);
        //     ipv_rate = d1/(d1+d2) * freq_trans[start_bin1][i] +  d2/(d1+d2) * freq_trans[start_bin2][i];
        // }
        
        // double ipv_rate = freq_trans[start_bin][i];
        double ipv_rate = ipv_rates[i];
        
        // fprintf(stderr, "%f\n", ipv_rate);
        // if (abs(ipv_rate) > 1.0e-30) {
        //     fprintf(stderr, "ipv %f\n", ipv_rate);

        ptd_graph_add_edge(
                    graph->starting_vertex,
                    ptd_find_or_create_vertex(graph, avl_tree, initial_state),
                    ipv_rate
            );
//      }
        
    }
    free(initial_state);

    int *child_state = (int *) malloc(state_size);
    
    for (size_t index = 1; index < graph->vertices_length; ++index) {
        struct ptd_vertex *vertex = graph->vertices[index];
        int *state = vertex->state;
                
        int n = 0; // lineages_left

        for (int i = 0; i < state_length-1; ++i) {
            n += state[i];
        }

        if (n == 0 || n == 1) {
            // Only one lineage left, absorb
            continue;
        }

        int d = 0; // derived lineages left
        for (int i = 0; i < state_length-1; ++i) {
            properties lin_props = _index_to_props(sample_size, i);
            if (lin_props.is_derived) {
                d += state[i];
            }
        }
        
        int cur_freq_bin = state[bin_index];
        
        // loop over freqs other than cur_freq_bin 
        for (int freq_bin = 0; freq_bin < n_freqbins; freq_bin++) {

            if (freq_bin == cur_freq_bin) {
                continue;   
            }

            if (n - d > 0 && freq_bin == n_freqbins-1) {
                // cannot loose ancestral allele while anc lineages remain
                continue;
            }
            if (d > 1 && freq_bin == 0) {
                // cannot loose derived allele while der lineages remain
                continue;
            }

            double rate = freq_trans[cur_freq_bin][freq_bin];
            
            rate *= pop_size; // to make it scale the same as N

            // rate += 0.0001;

            if (rate > 0) {
                fprintf(stderr, "rate from bin %d to bin %d: %f\n", cur_freq_bin, freq_bin, rate);
                fprintf(stderr, "%s", "state: ");
                for (int i=0; i<state_length; i++)
                    fprintf(stderr, "%d", state[i]);
                fprintf(stderr, "%s", "\n");
            }
            
            // if (abs(rate) < 1.0e-30) {
            //     continue;
            // }

            memcpy(child_state, state, state_size);

            child_state[bin_index] = freq_bin;

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
                
                double freq;
                if (d > 1) {
                    // there are still derived lineages left
                    if ((props_i.is_derived == 1) != (props_j.is_derived == 1)) { // xor
                        // so if the lineages are not of same kind we skip
                        continue;
                    }
                    
                    if (props_i.is_derived == 1) {
                        // both derived
                        freq = bins[state[bin_index]] / (double)pop_size;
                    } else {
                        // both ancestral
                        freq = (pop_size - bins[state[bin_index]]) / (double)pop_size;
                    }                    
                } else {
                    // only one derived lineage remaining wich can coalesce with the ancestral
                    freq = 1;
                }
                double rate = coal_rate / freq;

                // fprintf(stderr, "%d %d %d %d %f %e %e\n", props_i.is_derived, props_i.is_derived, d, n, freq, coal_rate, rate);
                
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
    free(bins);
    return Rcpp::XPtr<Graph>(
            new phasic::Graph(graph, avl_tree)
    );
}
