/*
 * Clone or download the code, and include these files in the repository!
 * Make SURE that the version of the downloaded code is the same as the
 * installed R library!! Otherwise it may crash randomly.
 *
 * The path is currently ../ as we are in the same repository. This path
 * should be something like [full or relative path to cloned code]/api...
 */
#include "./../../../phasic/api/c/phasic.h"

/*
* Including a .c file is very strange usually!
* But the way Rcpp::sourceCpp links is different from what
* you would usually expect. Therefore this is by far
* the easiest way of importing the code.
*/
#include "./../../../phasic/api/cpp/phasiccpp.h"
#include "./../../../phasic/src/c/phasic.c"

/* This is the binding layer such that R can invoke this function */
#include <Rcpp.h>

/* Basic C libraries */
#include "stdint.h"
#include "stdlib.h"
// #include "assert.h"


using namespace std;
using namespace phasic;
using namespace Rcpp;


void print_state(int *state, int l) {

    for (int i=0; i<l; i++) {
         fprintf(stderr, "%d ", state[i]);
    }
    fprintf(stderr, "\n");
}

// [[Rcpp::export]]
SEXP kingman_graph(int n, std::vector<double> scalars) {

  int state_length = n + 1;
  struct ptd_graph *ptd_graph = ptd_graph_create(state_length);
  struct ptd_avl_tree *avl_tree = ptd_avl_tree_create(state_length);
  int *initial_state = (int *) calloc(ptd_graph->state_length, sizeof(int));
  initial_state[0] = n;
  ptd_graph_add_edge(
    ptd_graph->starting_vertex,
    ptd_find_or_create_vertex(ptd_graph, avl_tree, initial_state),
    1
  );
  free(initial_state);

  int *state = (int *) calloc(ptd_graph->state_length, sizeof(int));
  for (int k = 1; k < ptd_graph->vertices_length; k++) {

    R_CheckUserInterrupt();
    struct ptd_vertex *vertex = ptd_graph->vertices[k];
    memcpy(state, vertex->state, ptd_graph->state_length * sizeof(int));

    // for (int i=0; i<ptd_graph->state_length; i++) {
    //      fprintf(stderr, "%d ", state[i]);
    // }
    // fprintf(stderr, "\n");
      
    for (int i = 0; i <= n-1; ++i) {
      for (int j = i; j <= n-1; ++j) {
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
        
        if (new_index > n-1) {
          new_index = n-1;
        }
        
        state[i]--;
        state[j]--;
        state[new_index]++;
        
        struct ptd_vertex *child = ptd_find_or_create_vertex(ptd_graph, avl_tree, state);
        
        state[i]++;
        state[j]++;
        state[new_index]--;
        
        // ptd_graph_add_edge(vertex, child, weight);
        // double *scalars = (double *) calloc(1, sizeof(double));
        double *edge_state = (double *) calloc(((int) scalars.size()), sizeof(double));    
          
        edge_state[0] = weight;
        ptd_graph_add_edge_parameterized(vertex, child, weight, edge_state);
      }
    }
  }
  free(state);

   ptd_graph_update_weight_parameterized(ptd_graph, &scalars[0], scalars.size());
  
   return Rcpp::XPtr<Graph>(
    new Graph(ptd_graph, avl_tree)
  );
}

// // [[Rcpp::export]]
// void add_epoque(SEXP graph, std::vector<double> scalars, double t) {

//     Rcpp::XPtr<Graph> graphcpp(graph);
//     struct ptd_graph *ptd_graph = graphcpp->c_graph();
//     struct ptd_avl_tree *avl_tree = graphcpp->c_avl_tree();

//     // compute edge_trans
//     std::vector<double> stop_probs = graphcpp->stop_probability(t);
//     std::vector<double> acum_visit = graphcpp->accumulated_visiting_time(t);
//     double *edge_trans = (double *) calloc(ptd_graph->vertices_length, sizeof(double));
//     for (int i=0; i<ptd_graph->vertices_length; i++) {
//         edge_trans[i] = stop_probs[i] / acum_visit[i];
//     }
    
//     // number of states in orig graph
//     int nr_states = ptd_graph->vertices_length;

//     // state and edge state buffers
//     int *state = (int *) calloc(ptd_graph->state_length, sizeof(int));
    
//     for (size_t i = 0; i < nr_states; ++i) {

//         // get vertex
//         struct ptd_vertex *vertex = ptd_graph->vertices[i];

//         // skip starting and absobing vertices
//         if (vertex == ptd_graph->starting_vertex || vertex->edges_length == 0) {
//             continue;
//         }

//         // add/find sister state
//         memcpy(state, vertex->state, ptd_graph->state_length * sizeof(int));
//         state[(ptd_graph->state_length)-1] = nr_states;
//         struct ptd_vertex *sister_vertex  = ptd_find_or_create_vertex(ptd_graph, avl_tree, state);

//         // edge params for edge to sister state
//         double *edge_state = (double *) calloc(((int) scalars.size()), sizeof(double));         
//         for (int j=0; j<scalars.size(); ++j) {
//             edge_state[j] = 0;
//         }
//         // assert(!(edge_trans[i] != edge_trans[i]));        

//         // add edge to sister
//         ptd_graph_add_edge_parameterized(vertex, sister_vertex, edge_trans[i], edge_state);   

//         // only clone edges for the first eqopuqe of states
//         if (vertex->state[(ptd_graph->state_length)-1] != 0) {
//             continue;  
//         }
//         for (size_t j = 0; j < vertex->edges_length; ++j) {

//             // get sister state
//             struct ptd_edge_parameterized *edge = ((struct ptd_edge_parameterized *) vertex->edges[j]);

//             // edges connecting epoques are already made
//             if (edge->to == sister_vertex) {
//                 continue;
//             }

//             // add/find state corresponding edge from sister-state should point to
//             memcpy(state, edge->to->state, ptd_graph->state_length * sizeof(int));
//             // make transitions in all epoques go to absorb of first epoque (saves states)
//             if (edge->to->edges_length > 0) {
//                 state[ptd_graph->state_length-1] = nr_states;
//             }
//             // state[ptd_graph->state_length-1] = nr_states;
//             struct ptd_vertex *sister_to_vertex  = ptd_find_or_create_vertex(ptd_graph, avl_tree, state);

//             // edge params for edge to that state
//             double *sister_edge_state = (double *) calloc(((int) scalars.size()), sizeof(double));         
//             memcpy(sister_edge_state, edge->state, ((int) scalars.size()) * sizeof(double));
//             // weight for edge to that state
//             double weight = 0;
//             for (size_t k = 0; k < scalars.size(); ++k) {
//                 weight += scalars[k] * sister_edge_state[k];
//             }            
//             // add the edge to that state
//             ptd_graph_add_edge_parameterized(
//                 sister_vertex, 
//                 sister_to_vertex, 
//                 weight, 
//                 sister_edge_state);         
//             }
//         }

//         for (size_t i = 0; i < ptd_graph->vertices_length; ++i) {
//             ptd_vertex *v = ptd_graph->vertices[i];
//             print_state(v->state, ptd_graph->state_length);
//             for (size_t j = 0; j < v->edges_length; ++j) {
//                 ptd_edge *e = v->edges[j];
//                 fprintf(stderr, "\t%f -> ", e->weight);
//                 print_state(e->to->state, ptd_graph->state_length);
//             }
//         }
//         fprintf(stderr, "---\n");

    
//     // free(stop_probs);
//     // free(acum_visit);
//     // free(state);
//     // free(edge_state);

//     // assert(ptd_validate_graph(ptd_graph) == 0);
//     ptd_notify_change(ptd_graph);  
// }



// // [[Rcpp::export]]
// int number_of_edges(SEXP phase_type_graph) {
//   Rcpp::XPtr<Graph> graphcpp(phase_type_graph);
//   ptd_graph *graph = graphcpp->c_graph();
//   int nedges = 0;
  
//   for (size_t k = 0; k < graph->vertices_length; ++k) {
//     nedges += graph->vertices[k]->edges_length;
//   }
  
//   return nedges;
// }


// // [[Rcpp::export]]
// void add_epoque(SEXP graph, std::vector<double> scalars, double t) {

//     Rcpp::XPtr<Graph> graphcpp(graph);
//     struct ptd_graph *ptd_graph = graphcpp->c_graph();
//     struct ptd_avl_tree *avl_tree = graphcpp->c_avl_tree();

//     // compute edge_trans
//     std::vector<double> stop_probs = graphcpp->stop_probability(t);
//     std::vector<double> acum_visit = graphcpp->accumulated_visiting_time(t);
//     double *edge_trans = (double *) calloc(ptd_graph->vertices_length, sizeof(double));
//     for (int i=0; i<ptd_graph->vertices_length; i++) {
//         edge_trans[i] = stop_probs[i] / acum_visit[i];
//     }
    
//     // number of states in orig graph
//     int nr_states = ptd_graph->vertices_length;

//     // state and edge state buffers
//     int *state = (int *) calloc(ptd_graph->state_length, sizeof(int));

//     fprintf(stderr, "epoque: %d\n", nr_states);

    
//     // add a copy of each vertex to the graph
//     for (size_t i = 0; i < nr_states; ++i) {

//         // get orig vertex
//         struct ptd_vertex *vertex = ptd_graph->vertices[i];

//         // skip starting and absorbing vertices
//         if (i == ptd_graph->starting_vertex->index) {
//             // starting vertex
//             continue;
//         }
//         if (vertex->edges_length == 0) {
//             // absorbing state
//             continue;
//         }

//         struct ptd_vertex *sister_vertex;
        
//         // only copy first epoque states
//         if (vertex->state[ptd_graph->state_length-1] == 0) {

//             fprintf(stderr, "orig vertex:");
//             print_state(vertex->state, ptd_graph->state_length);
        
//             // make the state of the sister vertex
//             memcpy(state, vertex->state, ptd_graph->state_length * sizeof(int));
    
//             // label state with epoque
//             state[ptd_graph->state_length-1] = nr_states;
    
//             // add sister vertex to graph
//             sister_vertex = ptd_find_or_create_vertex(ptd_graph, avl_tree, state);

//             fprintf(stderr, "sister:");
//             print_state(sister_vertex->state, ptd_graph->state_length);
    
                
//             // add edges from sister vertex
//             for (size_t j = 0; j < vertex->edges_length; ++j) {
    
//                 // get edge
//                 struct ptd_edge_parameterized *edge = ((struct ptd_edge_parameterized *) vertex->edges[j]);
    
//                 // get child vertex for edge
//                 struct ptd_vertex *to_vertex = ptd_find_or_create_vertex(ptd_graph, avl_tree, edge->to->state);
    
//                 // // only add edges connecting same-epoque states or edges to absorbing
//                 // if (vertex->state[ptd_graph->state_length-1] != to_vertex->state[ptd_graph->state_length-1] & to_vertex->edges_length > 0) {
//                 //     continue;
//                 // }
    
//                 // fprintf(stderr, "from:");
//                 // print_state(state, ptd_graph->state_length);
//                 // fprintf(stderr, "to:");
//                 // print_state(to_vertex->state, ptd_graph->state_length);
    
    
//                 // add child vertex with state labelled with epoque
//                 memcpy(state, to_vertex->state, ptd_graph->state_length * sizeof(int));
//                 state[ptd_graph->state_length-1] = nr_states;
//                 // state[ptd_graph->state_length-1] += nr_states;
//                 struct ptd_vertex *sister_to_vertex = ptd_find_or_create_vertex(ptd_graph, avl_tree, state);
                
//                 // add edge between vertex and child
//                 double *edge_state = (double *) calloc(scalars.size(), sizeof(double));         
//                 memcpy(edge_state, edge->state, ((int) scalars.size()) * sizeof(double));
//                 double weight = 0;
//                 for (size_t k = 0; k < scalars.size(); ++k) {
//                     weight += scalars[k] * ((struct ptd_edge_parameterized *) vertex->edges[j])->state[k];
//                 }
//                 ptd_graph_add_edge_parameterized(sister_vertex, sister_to_vertex, weight, edge_state);         
//             }

//             // // add edge to sister vertex
//             // double *edge_state = (double *) calloc(scalars.size(), sizeof(double));            
//             // memcpy(edge_state, &scalars[0], ((int) scalars.size()) * sizeof(double));
//             // ptd_graph_add_edge_parameterized(vertex, sister_vertex, edge_trans[i], edge_state);
        
//         }

//         // add edges to sister vertex
//         for (size_t l = 0; l < nr_states; ++l) {

//             struct ptd_vertex *from_vertex = ptd_graph->vertices[l];

//             if (from_vertex->index == sister_vertex->index) {
//                 // no edge to self
//                 continue;
//             }
//             if (l == ptd_graph->starting_vertex->index) {
//                 // starting vertex
//                 continue;
//             }
//             if (from_vertex->edges_length == 0) {
//                 // absorbing state
//                 continue;
//             }

            
//             // get sister state of from_vertex
//             memcpy(state, from_vertex->state, ptd_graph->state_length * sizeof(int));
//             state[ptd_graph->state_length-1] = nr_states;
//             // see if that is the sister_vertex
//             if (ptd_find_or_create_vertex(ptd_graph, avl_tree, state)->index != sister_vertex->index) {
//                 continue;
//             }
            

            
//             fprintf(stderr, "\tfrom:");
//             print_state(from_vertex->state, ptd_graph->state_length);
//             fprintf(stderr, "\tto:");
//             print_state(sister_vertex->state, ptd_graph->state_length);

            
//             double *edge_state = (double *) calloc(scalars.size(), sizeof(double));            
//             memcpy(edge_state, &scalars[0], ((int) scalars.size()) * sizeof(double));
//             ptd_graph_add_edge_parameterized(from_vertex, sister_vertex, edge_trans[l], edge_state);
//         }        
        
//     }
    
//     // free(stop_probs);
//     // free(acum_visit);
//     // free(state);
//     // free(edge_state);

//     assert(ptd_validate_graph(ptd_graph) == 0);
//     ptd_notify_change(ptd_graph);  
// }

