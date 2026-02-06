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


// find (one of) the absorbing children if any of each state
struct ptd_edge** ptd_graph_vertices_absorbing_edge(
        ptd_graph *graph
) {

  struct ptd_edge **abs_edges = (struct ptd_edge **) calloc(graph->vertices_length, sizeof(*abs_edges));
    
  for (size_t v = 0; v < graph->vertices_length; ++v) {
      abs_edges[v] = NULL;
      for (size_t e = 0; e < graph->vertices[v]->edges_length; ++e) {
          if (graph->vertices[v]->edges[e]->to->edges_length == 0) {
            abs_edges[v] = graph->vertices[v]->edges[e];
            break;
          }
      }      
  }
  return abs_edges;
}

double ptd_graph_laplace_transform(struct ptd_graph *graph, struct ptd_avl_tree *avl_tree, double theta) {

    // get array mapping each vertex to an absorbing edge if it has any
    struct ptd_edge** vertices_absorbing_edge = ptd_graph_vertices_absorbing_edge(graph);

    // turn NULL/edge array into 0/1 array
    double *orig_graph_absorbing_child = (double *) calloc(graph->vertices_length, sizeof(double));
    for (size_t v = 0; v < graph->vertices_length; ++v) {
        if (vertices_absorbing_edge[v]) {
           orig_graph_absorbing_child[v] = 1;
        } else {
           orig_graph_absorbing_child[v] = 0;
        }
    }   

    // for each state, add an absorbing edge with weight theta or add the weight 
    // to an absorbing existing branch
    struct ptd_clone_res cloned = ptd_clone_graph(graph, avl_tree);
    struct ptd_graph *new_graph = cloned.graph;

    struct ptd_edge **vertices_absorbing_edge =  ptd_graph_vertices_absorbing_edge(new_graph);
        
    // find an absorbing state
    struct ptd_vertex *absorbing_vertex;
    for (size_t v = new_graph->vertices_length - 1; v >= 0; --v) {
        if (graph->vertices[v]->edges_length == 0) {
            absorbing_vertex = new_graph->vertices[v];
            break;
        }
    }
    for (size_t v = 0; v < new_graph->vertices_length; ++v) {
        struct ptd_vertex *vertex = new_graph->vertices[v];
        // skip starting and absorbing vertices
        if (vertex == new_graph->starting_vertex || vertex->edges_length == 0) {
            continue;
        }
        if (vertices_absorbing_edge[v]) {
            // add weight to an existing edge to absorbing
            for (size_t e = 0; e < vertex->edges_length; ++e) {
                if (vertices_absorbing_edge[v] == vertex->edges[e]) {
                ptd_edge_update_weight(vertex->edges[e], vertex->edges[e]->weight + theta);
                break;          
                }
            }
        } else {
            // add edge to absorbing
            double *edge_state = (double *) calloc(((int) new_graph->state_length), sizeof(double));         
            for (int j=0; j<new_graph->state_length; ++j) {
                edge_state[j] = 0;
            }
            edge_state[new_graph->state_length-1] = 1;
            // ptd_graph_add_edge_parameterized(vertex, absorbing_vertex, theta, edge_state);
            ptd_graph_add_edge_parameterized(vertex, absorbing_vertex, 1, edge_state);
        }
    }

    double *params = (double *) calloc(new_graph->state_length, sizeof(double));
    for (int i=0; i < new_graph->state_length-1; ++i) {
        params[i] = 1;
    }
    params[new_graph->state_length-1] = theta;
    ptd_graph_update_weight_parameterized(new_graph, params, new_graph->state_length);

    // compute expectation
    double *exp_wait_times = ptd_expected_waiting_time(new_graph, orig_graph_absorbing_child);
    return exp_wait_times[0];
}
