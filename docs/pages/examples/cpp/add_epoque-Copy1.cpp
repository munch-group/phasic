/*
 * Clone or download the code, and include these files in the repository!
 * Make SURE that the version of the downloaded code is the same as the
 * installed R library!! Otherwise it may crash randomly.
 *
 * The path is currently ../ as we are in the same repository. This path
 * should be something like [full or relative path to cloned code]/api...
 */
#include "./../../PtDAlgorithms/api/c/ptdalgorithms.h"

/*
* Including a .c file is very strange usually!
* But the way Rcpp::sourceCpp links is different from what
* you would usually expect. Therefore this is by far
* the easiest way of importing the code.
*/
#include "./../../PtDAlgorithms/api/cpp/ptdalgorithmscpp.h"
#include "./../../PtDAlgorithms/src/c/ptdalgorithms.c"

/* This is the binding layer such that R can invoke this function */
#include <Rcpp.h>

/* Basic C libraries */
#include "stdint.h"
#include "stdlib.h"
// #include "assert.h"


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

// [[Rcpp::export]]
vector<int> absorbing_parents(
        SEXP phase_type_graph
) {
  Rcpp::XPtr<Graph> graphcpp(phase_type_graph);
  ptd_graph *graph = graphcpp->c_graph();
  struct ptd_edge** vertices_absorbing_edge = ptd_graph_vertices_absorbing_edge(graph);

  vector<int> parents;
  for (size_t v = 0; v < graph->vertices_length; ++v) {
      if (vertices_absorbing_edge[v]) {
         parents.push_back(v+1);
      } 
  }
  free(vertices_absorbing_edge);

  return parents;
}


void print_state(int *state, int l) {

    for (int i=0; i<l; i++) {
         fprintf(stderr, "%d ", state[i]);
    }
    fprintf(stderr, "\n");
}

// [[Rcpp::export]]
void add_epoque(SEXP graph, std::vector<double> scalars, double t) {

    Rcpp::XPtr<Graph> graphcpp(graph);
    struct ptd_graph *ptd_graph = graphcpp->c_graph();
    struct ptd_avl_tree *avl_tree = graphcpp->c_avl_tree();

    // compute edge_trans
    std::vector<double> stop_probs = graphcpp->stop_probability(t);
    std::vector<double> acum_visit = graphcpp->accumulated_visiting_time(t);
    double *edge_trans = (double *) calloc(ptd_graph->vertices_length, sizeof(double));
    for (int i=0; i<ptd_graph->vertices_length; ++i) {
        if (stop_probs[i] == 0 && acum_visit[i] == 0) { 
            edge_trans[i] = 0;
        } else {
            edge_trans[i] = stop_probs[i] / acum_visit[i];
        }
    }
    // save number of states in orig graph
    int nr_states = ptd_graph->vertices_length;

    // state buffer
    int *state = (int *) calloc(ptd_graph->state_length, sizeof(int));

    for (size_t i = 0; i < nr_states; ++i) {

        // get vertex
        struct ptd_vertex *vertex = ptd_graph->vertices[i];

        // skip starting and absobing vertices
        if (vertex == ptd_graph->starting_vertex || vertex->edges_length == 0) {
            continue;
        }

        // define the sister state
        // memcpy(state, vertex->state, ptd_graph->state_length * sizeof(int));
        for (int k=0; k < ptd_graph->state_length; ++k) {
            state[k] = vertex->state[k];
        }
        // use current nr of states as epoque label in last state slot of sister state
        state[(ptd_graph->state_length) - 1] = nr_states; 

        // create sister vertex
        struct ptd_vertex *sister_vertex = ptd_find_or_create_vertex(ptd_graph, avl_tree, state);

        // create edge state for edge to the sister state
        double *edge_state = (double *) calloc(((int) scalars.size()), sizeof(double));       
        // double *edge_state = (double *) calloc(((int) scalars.size()), sizeof(*edge_state));   
        for (int j=0; j<scalars.size(); ++j) {
            edge_state[j] = 0;
        }        
        // create edge to to the sister state
        ptd_graph_add_edge_parameterized(vertex, sister_vertex, edge_trans[i], edge_state);   

        // only clone edges for the first eqopuqe of states
        if (vertex->state[(ptd_graph->state_length)-1] != 0) {
            continue;  
        }
        
        for (size_t j = 0; j < vertex->edges_length; ++j) {

                 
            // // get sister state
            // struct ptd_edge_parameterized *edge = ((struct ptd_edge_parameterized *) vertex->edges[j]);

            // edges connecting epoques are already made
            if (vertex->edges[j]->to == sister_vertex) {
                continue;
            }
            // add/find state corresponding edge from sister-state should point to
            
            // memcpy(state, vertex->edges[j]->to->state, ptd_graph->state_length * sizeof(int));
            for (int k=0; k < ptd_graph->state_length; ++k) {
                state[k] = vertex->edges[j]->to->state[k];
            }
            // make transitions in all epoques go to absorb of first epoque (saves states)
            if (vertex->edges[j]->to->edges_length > 0) {
                state[ptd_graph->state_length-1] = nr_states;  // NOT SURE ABOUT THIS...
            }
            // state[ptd_graph->state_length-1] = nr_states;
            struct ptd_vertex *sister_to_vertex  = ptd_find_or_create_vertex(ptd_graph, avl_tree, state);

            // edge params for edge to that state
            double *sister_edge_state = (double *) calloc(((int) scalars.size()), sizeof(double));         

            if (vertex->edges[j]->parameterized) { // THERE SHOULD BE AN ELSE FOR THIS....

           // memcpy(sister_edge_state, vertex->edges[j]->state, ((int) scalars.size()) * sizeof(double));
            for (int k=0; k < scalars.size(); ++k) {
                // sister_edge_state[k] = vertex->edges[j]->state[k];
                sister_edge_state[k] = ((ptd_edge_parameterized *) vertex->edges[j])->state[k];
            }
/////////////////////////
            
            // // weight for edge to that state
            // double weight = 0;
            // for (size_t k = 0; k < scalars.size(); ++k) {
            //     weight += scalars[k] * sister_edge_state[k];
            // }            
            // // add the edge to that state
            // ptd_graph_add_edge_parameterized(
            //     sister_vertex, 
            //     sister_to_vertex, 
            //     weight, 
            //     sister_edge_state);         
/////////////////////////

            struct ptd_edge *new_edge = (struct ptd_edge *) ptd_graph_add_edge_parameterized(
                sister_vertex, 
                sister_to_vertex, 
                vertex->edges[j]->weight, 
                sister_edge_state);         
          
            ptd_edge_update_weight_parameterized(
                    new_edge, &scalars[0], ((int) scalars.size())
            );


/////////////////////////
            }
          }
        }
    
    // free(stop_probs);
    // free(acum_visit);
    // free(state);
    // free(edge_state);

    // ptd_validate_graph(ptd_graph);
    ptd_notify_change(ptd_graph);  
}



            // print_state(vertex->state, ptd_graph->state_length);
            // print_state( vertex->edges[j]->to->state, ptd_graph->state_length);
            // fprintf(stderr, "%d\n\n", vertex->edges[j]->parameterized);

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

