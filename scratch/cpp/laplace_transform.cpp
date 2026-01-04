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

ptd_graph* ptd_graph_add_edge_weight_to_absorbing(
        struct ptd_graph *graph, 
        struct ptd_avl_tree *avl_tree, 
        double theta
) {
  struct ptd_clone_res cloned = ptd_clone_graph(graph, avl_tree);
  struct ptd_graph *new_graph = cloned.graph;
    
  // new_graph->state_length += 1;
  // for (size_t v = 0; v < new_graph->vertices_length; ++v) {
  //     int *state = new_graph->vertices[v]->state;
  //     int *new_state = (int *) calloc(new_graph->state_length, sizeof(int));
  //     for (int i=0; i < new_graph->state_length-1; ++i) {
  //         new_state[i] = state[i];
  //     }
  //     new_state[new_graph->state_length-1] = 1; // represent 1*theta weight
  //     new_graph->vertices[v]->state = new_state;
  //     free(state); 
  //     for (int i=0; i < new_graph->vertices[v]->edges_length; ++i) {


  //         // we could also just say it does only work for parameterized state spaces
  //         if (new_graph->vertices[v]->edges[i]->parameterized == true) {
  //             struct ptd_edge_parameterized *edge = new_graph->vertices[v]->edges[i];
  //             double *edge_state = edge->state;
              
  //             double *new_edge_state = (double *) calloc(new_graph->state_length, sizeof(double));
  //             for (int j=0; j < new_graph->state_length-1; ++j) {
  //               new_edge_state[j] = edge_state[j];
  //             } 
  //             new_edge_state[new_graph->state_length-1] = 0;

  //             new_graph->vertices[v]->edges[i]->state = new_edge_state;
  //             free(edge_state);
  //         } else {
  //             struct ptd_edge *edge = new_graph->vertices[v]->edges[i];
  //             double *new_edge_state = (double *) calloc(new_graph->state_length, sizeof(double));

  //             for (int j=0; j < new_graph->state_length-1; ++j) {
  //               new_edge_state[j] = new_graph->vertices[v]->state[j];
  //             } 
  //             new_edge_state[new_graph->state_length-1] = 0;
              
  //             struct ptd_edge_parameterized *new_edge;
  //             new_edge->state = new_edge_state;
  //             new_edge->weight = edge->weight;
  //             new_edge->from = edge->from;
  //             new_edge->to = edge->to;
              
  //             new_graph->vertices[v]->edges[i] = new_edge;
  //             // free(edge_state);              
  //             free(edge);
  //         }
  //     }      
  // }
  // no need to redo avl tree as states maintain sorting order
    
  // struct ptd_avl_tree *cloned_avl_tree = cloned.avl_tree;
  // ptd_avl_tree_create(new_graph->state_length);
  // ptd_avl_tree_destroy(cloned_avl_tree);
  
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

    
  return new_graph;
}

// [[Rcpp::export]]
SEXP add_edge_weight_to_absorbing(
        SEXP phase_type_graph, 
        double theta
) {
    Rcpp::XPtr<Graph> graphcpp(phase_type_graph);
    struct ptd_graph *graph = graphcpp->c_graph();
    struct ptd_avl_tree *avl_tree = graphcpp->c_avl_tree();
    struct ptd_graph *new_graph = ptd_graph_add_edge_weight_to_absorbing(graph, avl_tree, theta);
    return Rcpp::XPtr<Graph>(new Graph(new_graph, avl_tree));
}

double ptd_graph_laplace_transform(
        struct ptd_graph *graph, 
        struct ptd_avl_tree *avl_tree,
        double theta
) {

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
    struct ptd_graph *new_graph = ptd_graph_add_edge_weight_to_absorbing(graph, avl_tree, theta);


    // compute expectation
    double *exp_wait_times = ptd_expected_waiting_time(new_graph, orig_graph_absorbing_child);
    return exp_wait_times[0];
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

// [[Rcpp::export]]
double laplace_transform(SEXP phase_type_graph, double theta) {
  Rcpp::XPtr<Graph> graphcpp(phase_type_graph);
  struct ptd_graph *graph = graphcpp->c_graph();
  struct ptd_avl_tree *avl_tree = graphcpp->c_avl_tree();
  return ptd_graph_laplace_transform(graph, avl_tree, theta);
}

///////////////////////////////

// double* ptd_add_absorbing(ptd_graph *ptd_graph, double theta) {

//   struct ptd_edge **absorbing_children_map = ptd_absorbing_edges(ptd_graph);

//   // find an absorbing state
//   struct ptd_vertex *absorbing_vertex;
//   for (size_t v = ptd_graph->vertices_length - 1; v >= 0; --v) {
//       if (ptd_graph->vertices[v]->edges_length == 0) {
//         absorbing_vertex = ptd_graph->vertices[v];
//         break;
//       }
//   }
//   double *has_absorbing = (double *) calloc(ptd_graph->vertices_length, sizeof(int));

//   for (size_t v = 0; v < ptd_graph->vertices_length; ++v) {
//       struct ptd_vertex *vertex = ptd_graph->vertices[v];
//       // skip starting and absorbing vertices
//       if (vertex == ptd_graph->starting_vertex || vertex->edges_length == 0) {
//         continue;
//       }
//       if (absorbing_children_map[v]) {
//           // add weight to an existing edge to absorbing

//           for (size_t e = 0; e < vertex->edges_length; ++e) {
//             if (absorbing_children_map[v] == vertex->edges[e]) {
//               ptd_edge_update_weight(vertex->edges[e], vertex->edges[e]->weight + theta);
//               break;          
//             }
//           }
//           has_absorbing[v] = 1;
//       } else {
          
//           // add edge to absorbing
//           ptd_graph_add_edge(vertex, absorbing_vertex, theta);

//           // double *param_vec = (double *) calloc(2, sizeof(double));
//           // param_vec[0] =
//           // param_vec[1] = 0;
//           // add_edge_parameterized(vertex, absorbing_vertex, u, param_vec);

//           has_absorbing[v] = 0;
//       }
//   }
//   return has_absorbing;
// }

// double ptd_laplace_transform(ptd_graph *ptd_graph, double theta) {

//     double* has_absorbing = ptd_add_absorbing(ptd_graph, theta);
//     double* exp_wait_times = ptd_expected_waiting_time(ptd_graph, has_absorbing);
//     return exp_wait_times[0];
// }

// // [[Rcpp::export]]
// double laplace_transform(SEXP phase_type_graph, double theta) {
//   Rcpp::XPtr<Graph> graphcpp(phase_type_graph);
//   ptd_graph *ptd_graph = graphcpp->c_graph();
//   return ptd_laplace_transform(ptd_graph, theta);
// }

/////////////////////////////////

// // [[Rcpp::export]]
// double laplace_transform(SEXP phase_type_graph, double theta) {
// // vector<int> laplace_transform(SEXP phase_type_graph, double theta) {
//   Rcpp::XPtr<Graph> graphcpp(phase_type_graph);
//   struct ptd_graph *orig_graph = graphcpp->c_graph();
//   struct ptd_avl_tree *avl_tree = graphcpp->c_avl_tree();

//   struct ptd_clone_res cloned = ptd_clone_graph(orig_graph, avl_tree);
//   struct ptd_graph *graph = cloned.graph;
    
//   struct ptd_edge **absorbing_children_map = (struct ptd_edge **) calloc(graph->vertices_length, sizeof(*absorbing_children_map));
    
//   for (size_t v = 0; v < graph->vertices_length; ++v) {
//       absorbing_children_map[v] = NULL;
//       for (size_t e = 0; e < graph->vertices[v]->edges_length; ++e) {
//           if (graph->vertices[v]->edges[e]->to->edges_length == 0) {
//             absorbing_children_map[v] = graph->vertices[v]->edges[e];
//             break;
//           }
//       }      
//   }
    
//   // find an absorbing state
//   struct ptd_vertex *absorbing_vertex;
//   for (size_t v = graph->vertices_length - 1; v >= 0; --v) {
//       if (graph->vertices[v]->edges_length == 0) {
//         absorbing_vertex = graph->vertices[v];
//         break;
//       }
//   }

//   double *has_absorbing = (double *) calloc(graph->vertices_length, sizeof(double));

//   for (size_t v = 0; v < graph->vertices_length; ++v) {
//       struct ptd_vertex *vertex = graph->vertices[v];
//       // skip starting and absorbing vertices
//       if (vertex == graph->starting_vertex || vertex->edges_length == 0) {
//         continue;
//       }
//       if (absorbing_children_map[v]) {
//           // add weight to an existing edge to absorbing

//           for (size_t e = 0; e < vertex->edges_length; ++e) {
//             if (absorbing_children_map[v] == vertex->edges[e]) {
//               ptd_edge_update_weight(vertex->edges[e], vertex->edges[e]->weight + theta);
//               break;          
//             }
//           }
//           has_absorbing[v] = 1;
//       } else {
          
//           // add edge to absorbing
//           ptd_graph_add_edge(vertex, absorbing_vertex, theta);

//           // double *param_vec = (double *) calloc(2, sizeof(double));
//           // param_vec[0] = 0;
//           // param_vec[1] = 0;
//           // add_edge_parameterized(vertex, absorbing_vertex, u, param_vec);

//           has_absorbing[v] = 0;
//       }
//   }
//   // vector<int> laplace_rewards(has_absorbing, has_absorbing + ptd_graph->vertices_length);

//   return ptd_expected_waiting_time(graph, has_absorbing)[0];
// }












