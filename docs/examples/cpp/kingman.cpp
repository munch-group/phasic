/*
 * Clone or download the code, and include these files in the repository!
 * Make SURE that the version of the downloaded code is the same as the
 * installed R library!! Otherwise it may crash randomly.
 *
 * The path is currently ../ as we are in the same repository. This path
 * should be something like [full or relative path to cloned code]/api...
 */
#include "./../../../PtDAlgorithms/api/c/ptdalgorithms.h"

/*
* Including a .c file is very strange usually!
* But the way Rcpp::sourceCpp links is different from what
* you would usually expect. Therefore this is by far
* the easiest way of importing the code.
*/
#include "./../../../PtDAlgorithms/api/cpp/ptdalgorithmscpp.h"
#include "./../../../PtDAlgorithms/src/c/ptdalgorithms.c"

/* This is the binding layer such that R can invoke this function */
#include <Rcpp.h>

/* Basic C libraries */
#include "stdint.h"
#include "stdlib.h"

using namespace std;
using namespace ptdalgorithms;
using namespace Rcpp;

// [[Rcpp::export]]
SEXP generate_kingman_graph(int n, int m) {
// SEXP generate_kingman_graph(int m) {
//   int n = m;

  m--;
  struct ptd_graph *kingman_graph = ptd_graph_create(m + 1);
  struct ptd_avl_tree *avl_tree = ptd_avl_tree_create(m + 1);
  int *initial_state = (int *) calloc(kingman_graph->state_length, sizeof(int));
  initial_state[0] = n;
  ptd_graph_add_edge(
    kingman_graph->starting_vertex,
    ptd_find_or_create_vertex(kingman_graph, avl_tree, initial_state),
    1
  );
  free(initial_state);
  
  int *state = (int *) calloc(kingman_graph->state_length, sizeof(int));
  for (size_t k = 1; k < kingman_graph->vertices_length; k++) {
    R_CheckUserInterrupt();
    struct ptd_vertex *vertex = kingman_graph->vertices[k];
    memcpy(state, vertex->state, kingman_graph->state_length * sizeof(int));
    
    for (int i = 0; i <= m; ++i) {
      for (int j = i; j <= m; ++j) {
        double weight;
        
        if (i == j) {
          if (state[i] < 2) {
            continue;
          }
          
          weight = state[i] * (state[i] - 1) / 2;
        } else {
          if (state[i] < 1 || state[j] < 1) {
            continue;
          }
          
          weight = state[i] * state[j];
        }
        
        int new_index = i + j + 2 - 1;
        
        if (new_index > m) {
          new_index = m;
        }
        
        state[i]--;
        state[j]--;
        state[new_index]++;
        
        struct ptd_vertex *child = ptd_find_or_create_vertex(kingman_graph, avl_tree, state);
        
        state[i]++;
        state[j]++;
        state[new_index]--;
        
        ptd_graph_add_edge(vertex, child, weight);
      }
    }
  }
  free(state);
  
  return Rcpp::XPtr<Graph>(
    new Graph(kingman_graph, avl_tree)
  );
}

// [[Rcpp::export]]
int number_of_edges(SEXP phase_type_graph) {
  Rcpp::XPtr<Graph> graphcpp(phase_type_graph);
  ptd_graph *graph = graphcpp->c_graph();
  int nedges = 0;
  
  for (size_t k = 0; k < graph->vertices_length; ++k) {
    nedges += graph->vertices[k]->edges_length;
  }
  
  return nedges;
}