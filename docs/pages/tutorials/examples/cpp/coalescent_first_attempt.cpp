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


// def exp_coal(g, N):
//     return 2*N - (g * np.exp(-g/(2*N))) / (1 - np.exp(-g/(2*N)))

// def epoch(demog, h, i):
//     g, N = demog[i]
//     N *= h
//     if i == len(demog)-1:
//         return 2*N
//     return (1-np.exp(-g/(2*N))) * exp_coal(g, N) + np.exp(-g/(2*N)) * (g + epoch(demog, h, i+1))


// epoques are tuples of n_generations_in_epoque and epoque_n (last tuple is open-ended and has None for n_generations): 
//epoques = [(189999, 10000), (5000, 10000), (4000, 10000), (None, 10000)]
// pi = epoch(epoques, 1, 0)
// print(pi / 2) //  should be 10000  .... I could also replace 2N with N in function....


    
    
    
    
    
double*** trans_matrix(int n, int generations, double sel_coef) {
    
    int verbose = 0;
    
    char filename[80];
    sprintf(filename, "matrix_%d_%d_%f.dat", n, generations, sel_coef);
    struct stat buffer;
    int file_missing = stat(filename, &buffer);       

    if (file_missing == 0) {
        FILE *fin = fopen(filename, "rb"); 
        double* array = (double *) malloc(sizeof(double) * (generations+1)*(n+1)*(n+1));        
        fread(array, sizeof(*array), (generations+1)*(n+1)*(n+1), fin);
        
        double*** trans = (double***) malloc(sizeof(double**) * (generations+1));
        for (int gen = 0; gen < generations+1; gen++) {
            trans[gen] = (double**) malloc(sizeof(double*) * (n+1));
            for(int i = 0; i < n+1; i++) { 
                trans[gen][i] = (double*) malloc(sizeof(double) * (n+1));
                for(int j = 0; j < n+1; j++) {
                    // fprintf(stderr, "%d %d %d %d %d\n", gen, i, j, gen*(n+1)*(n+1)+i*(n+1)+j, (generations+1)*(n+1)*(n+1));
                    trans[gen][i][j] = array[gen*(n+1)*(n+1)+i*(n+1)+j];    
                }    
            }
        } 
        free(array);
        return(trans);
    }
    
    // compute binomial coeficients
    int** binom = (int **) malloc(sizeof(int*) * n+1);
    for (int i = 0; i <= n; i++) {
        int num = 1;
        int* row = (int *) malloc(sizeof(int*) * n+1);
        for (int j = 0; j <= i; j++) {
            if (i != 0 && j != 0)
                num = num * (i - j + 1) / j;
                row[j] = num;
        }
        binom[i] = row;
    }
    // to get binom(a, b) do: binom[a][b]
    
    if (verbose) {
        for (int i=0; i<n; i++) {
            for(int j=0; j<=i; j++) {
                 fprintf(stderr, "%d ", binom[i][j]);
            }
            fprintf(stderr, "\n");
        }
    }
    
    /// compute binomial transition matrix for one generation
    double **m;
    m = (double **) malloc(sizeof(double*) * n+1);
    for(int i = 0; i < n+1; i++) {
        m[i] = (double *) malloc(sizeof(double*) * n+1);
    }
    for (int i=0; i < n+1; i++) {
        for (int j=0; j < n+1; j++) {
            if (i == 0) {
                // lost remains lost
                if (j == 0) {
                    m[i][j] = 1;
                } else {
                    m[i][j] = 0;
                }
            } else if (i == n) {
                // fixed remains fixed
                if (j == n) {
                    m[i][j] = 1;
                } else {
                    m[i][j] = 0;
                }
            } else {
                // fprintf(stderr, "%d %d %d %f %f\n", N, j, binom[N][j], pow(i/(double)N, j), pow(1-(i/(double)N), N-j));
                m[i][j] = binom[n][j] * pow(i/(double)n, j) * pow(1-(i/(double)n), n-j);
            }
        }
    }
    // to get trans a -> b do: m[a][b]

    if (verbose) {
    for (int i=0; i<n+1; i++) {
        for(int j=0; j<n+1; j++) {
             fprintf(stderr, "%f ", m[i][j]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
    }

    double*** trans = (double ***) malloc(sizeof(double**) * generations+1);
    trans[0] = m;
    for (int gen = 1; gen < generations+1; gen++) {
        double** gen_m = (double **) malloc(sizeof(double*) * n+1);
        for(int i = 0; i < n+1; i++) {
            gen_m[i] = (double *) malloc(sizeof(double) * n+1);
        }
        trans[gen] = gen_m;
        for(int i = 0; i < n+1; i++) {    
            for(int j = 0; j < n+1; j++) {    
                trans[gen][i][j] = 0;    
                for(int k = 0; k < n+1; k++) { 
                    trans[gen][i][j] += trans[gen-1][i][k] * m[k][j];    
                }    
            }    
        }
    }

    if (verbose) {
        for (int i=0; i<n+1; i++) {
            for(int j=0; j<n+1; j++) {
                 fprintf(stderr, "%f ", trans[generations][i][j]);
            }
            fprintf(stderr, "\n");
        }    
        fprintf(stderr, "\n");
    }
    
    
    double* array = (double*) malloc(sizeof(double) * (generations+1)*(n+1)*(n+1));
    for (int gen = 0; gen < generations+1; gen++) {
        for(int i = 0; i < n+1; i++) {    
            for(int j = 0; j < n+1; j++) {    
                array[gen*(n+1)*(n+1)+i*(n+1)+j] = trans[gen][i][j];    
            }    
        }
    }
    FILE *fout = fopen(filename, "wb"); // save
    // fwrite(trans, sizeof(trans), (generations+1)*(n+1)*(n+1), fout);
    fwrite(array, sizeof(array), (generations+1)*(n+1)*(n+1), fout);
    fclose(fout);
    
    return(trans);
}

double _exp_coal(double g, double N) {
    return(N - (g * exp(-g/(N))) / (1 - exp(-g/(N))));
}

double _epoch(double** demog, int nr_epoques, float h, int i) {
    double g = demog[i][0];
    double N = demog[i][1];
    N *= h;
    if (i == nr_epoques-1) {
        return(N);
    }
    return((1-exp(-g/(N))) * _exp_coal(g, N) + exp(-g/(N)) * (g + _epoch(demog, nr_epoques, h, i+1)));
}

double exp_coal(double cur_freq, double end_freq, double t) {
    int n_epoques = 100000;
    double **epoques = (double **) malloc(sizeof(double*) * n_epoques);
    for (int i=0; i< n_epoques; i++) {
        epoques[i] = (double *) malloc(sizeof(double) * 2);
        epoques[i][0] = t / n_epoques; 
        epoques[i][1] = cur_freq + (end_freq - cur_freq) * i/(double)n_epoques;
    }
    double exp_time = _epoch(epoques, n_epoques, 1, 0);
    for (int i=0; i< n_epoques; i++) {
        free(epoques[i]);
    }
    free(epoques);
    return(exp_time);
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
           frequencies[i] = i/(double)n_freqbins - 1/(double)n_freqbins/2 ; // center of bin
        }
    }
    
    // compute freq transition matrices
    int n = 500; // 1000;  // ~ 3 hours
    int generations = 1000; 
    double time_scaling = n / (double)pop_size;
    double sel_coef_scaling = sel_coef > 0 && pop_size / sel_coef || 0;
    
    time_t seconds;
    seconds = time(NULL);
    double ***freq_trans = trans_matrix(n, generations, 0);
    // fprintf(stderr, "generated matrices in %ld seconds\n", time(NULL) - seconds);

    double N = 1;
    
    // state vector length
    const int state_length = (sample_size + 1) * 2 + 1;
    const size_t state_size = sizeof(int) * state_length;
    
    int freqbin_index = state_length - 1;
    
    struct ptd_graph *graph = ptd_graph_create((size_t) state_length);
    struct ptd_avl_tree *avl_tree = ptd_avl_tree_create((size_t) state_length);
    
    int *initial_state = (int *) calloc((size_t) state_length, sizeof(int));
    for (int i=0; i < n_freqbins; ++i) {
        // fprintf(stderr, "%d\n", i);

        for (int x=0; x < state_length; x++) {
            initial_state[x] = 0;
        }
        
        initial_state[_props_to_index(sample_size, 1, 0)] = sample_size - n_derived;
        initial_state[_props_to_index(sample_size, 1, 1)] = n_derived;
        initial_state[freqbin_index] = i;

        double bin_freq = frequencies[i];
        double der_freq = n_derived / (double)sample_size;
        
        double ipv_rate = freq_trans[0][(int)round(pop_size*der_freq)][(int)round(pop_size*bin_freq)];

        fprintf(stderr, "%f\n", ipv_rate);
        
        if (abs(ipv_rate) > 1.0e-20) {
            fprintf(stderr, "ipv %f\n", ipv_rate);

            ptd_graph_add_edge(
                    graph->starting_vertex,
                    ptd_find_or_create_vertex(graph, avl_tree, initial_state),
                    ipv_rate
            );
        }
    }
    free(initial_state);

    
    int *child_state = (int *) malloc(state_size);
    
    // for (size_t index = 1; index < graph->vertices_length; ++index) {
    for (size_t index = 1; index < graph->vertices_length; ++index) {
        struct ptd_vertex *vertex = graph->vertices[index];
        int *state = vertex->state;
        
        // fprintf(stderr, "%zu %zu\n", index, graph->vertices_length);
        
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
        // fprintf(stderr, "ancestral %d derived %d\n", tot_ancestral, tot_derived);

        
        // lineage one
        for (int i = 0; i < state_length-1; ++i) {
            properties props_i = _index_to_props(sample_size, i);
                
            // lineage two
            for (int j = i; j < state_length-1; ++j) {
                properties props_j = _index_to_props(sample_size, j);
                    
                double coal_rate;
                double freq;
                int is_der;
                if (tot_derived > 1) {
                    // there are still derived lineages left
                    if ((props_i.is_derived == 1) != (props_i.is_derived == 1)) { // xor
                        // so if the lineages are not of same kind we skip
                        continue;
                    }
                    
                    if (props_i.is_derived == 1) {
                        // both derived
                        freq = state[freqbin_index];
                        is_der = 1;
                    } else {
                        // both ancestral
                        freq = 1 - state[freqbin_index];
                        is_der = 0;
                    }                    
                } else {
                    // only one derived lineage remaining wich can coalesce with the ancestral
                    freq = 1;
                    is_der = 0;
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
                 
                int cur_freq_bin = state[freqbin_index];
                double rate;
                for (int freq_bin = 0; freq_bin < n_freqbins; freq_bin++) {
                    
                    double target_freq;
                    if (is_der == 1) {
                        target_freq = frequencies[freq_bin];
                    } else {
                        target_freq = 1 - frequencies[freq_bin];
                    }
                    

                    // I GUESS I NEED TO KEEP THE RATE CONSTANT FOR EACH STATE TO MAKE THE RATE EXPONENTIALLY DISTRIBUTED...
                    
                    // double exp_coal_time = exp_coal(freq, target_freq, 1/coal_rate);
                    // fprintf(stderr, "pool-nielsen: %f %f %f %f\n", freq, coal_rate, target_freq, exp_coal_time);
                    double exp_coal_time = freq/coal_rate;
                    fprintf(stderr, "constant: %f\n", exp_coal_time);

                    // compute relative probabilities of transition to each 
                    // frequency state given the time = 1/mean_freq and s = ....
                    // int g = round(mean_freq * N * time_scaling);  
                    int g = round(exp_coal_time * time_scaling);                 
                    double prob_tot = 0; 
                    int a = round(N*frequencies[freq_bin]);
                    int b = round(N*frequencies[cur_freq_bin]);
                    for (int f = 0; f < n_freqbins; f++) {
                        prob_tot += freq_trans[g][a][b];
                    }
                    int c = round(N*frequencies[freq_bin]);                    
                    double rel_prob_freq_change = freq_trans[g][a][c] / prob_tot;                    
                    // compute final rate
                    // fprintf(stderr, "%f %f %f\n", freq, target_freq, mean_freq);
                    // rate = coal_rate / mean_freq / n_freqbins / rel_prob_freq_change;
                    rate = 1/ exp_coal_time / n_freqbins / rel_prob_freq_change;
                    
                    fprintf(stderr, "%f %f\n", rel_prob_freq_change, rate);

                    if (abs(rate) < 1.0e-20) {
                        continue;
                    }
                    
                    memcpy(child_state, state, state_size);
                    
                    // lineages with index i and j coalesce:  
                    child_state[i] = child_state[i] - 1;
                    child_state[j] = child_state[j] - 1;
                    child_state[freqbin_index] = freq_bin;

                    // if (child_state[i] < 0) {
                    //     char str[80];
                    //     sprintf(str, "Neg value %d", child_state[i]);
                    //     throw std::range_error(str);
                    // }

                    // coalescene into lineage with index k
                    // int k = i+j;
                    int k;
                    if (props_i.is_derived * props_j.is_derived != 0) { 
                        // both derived
                        k = _props_to_index(sample_size, props_i.n_descendants + props_j.n_descendants, 1);
                    } else {
                        // one or both ancestral
                        k = _props_to_index(sample_size, props_i.n_descendants + props_j.n_descendants, 0);
                    }

                    child_state[k] = child_state[k] + 1;                

                    // fprintf(stderr, "i, j, k: %d %d %d\n", i, j, k);

                    struct ptd_vertex *child_vertex = ptd_find_or_create_vertex(
                                    graph, avl_tree, child_state
                           );
                    ptd_graph_add_edge(vertex, child_vertex, rate); 

                }
                    
            }
        }
    }
    free(child_state);
    free(frequencies);
    return Rcpp::XPtr<Graph>(
            new phasic::Graph(graph, avl_tree)
    );
}
