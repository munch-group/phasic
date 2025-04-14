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
// #include "assert.h"


using namespace std;
using namespace ptdalgorithms;
using namespace Rcpp; 

// Graph *coalescent_graph(int n, int m) {

//   int edge_state_size = 2;

//   m--;
//   struct ptd_graph *kingman_graph = ptd_graph_create(m + 1);
//   struct ptd_avl_tree *avl_tree = ptd_avl_tree_create(m + 1);
//   int *initial_state = (int *) calloc(kingman_graph->state_length, sizeof(int));
//   initial_state[0] = n;
//   ptd_graph_add_edge(
//     kingman_graph->starting_vertex,
//     ptd_find_or_create_vertex(kingman_graph, avl_tree, initial_state),
//     1
//   );
//   free(initial_state);
  
//   int *state = (int *) calloc(kingman_graph->state_length, sizeof(int));
//   for (size_t k = 1; k < kingman_graph->vertices_length; k++) {
//     // R_CheckUserInterrupt();
//     struct ptd_vertex *vertex = kingman_graph->vertices[k];
//     memcpy(state, vertex->state, kingman_graph->state_length * sizeof(int));
    
//     for (int i = 0; i <= m; ++i) {
//       for (int j = i; j <= m; ++j) {
//         double weight;
        
//         if (i == j) {
//           if (state[i] < 2) {
//             continue;
//           }
          
//           weight = state[i] * (state[i] - 1) / 2;
//         } else {
//           if (state[i] < 1 || state[j] < 1) {
//             continue;
//           }
          
//           weight = state[i] * state[j];
//         }
        
//         int new_index = i + j + 2 - 1;
        
//         if (new_index > m) {
//           new_index = m;
//         }
        
//         state[i]--;
//         state[j]--;
//         state[new_index]++;
        
//         struct ptd_vertex *child = ptd_find_or_create_vertex(kingman_graph, avl_tree, state);
        
//         state[i]++;
//         state[j]++;
//         state[new_index]--;
        
//         double *edge_state = (double *) calloc(edge_state_size, sizeof(double));         
//         edge_state[0] = weight;
//         edge_state[2] = 0;

//         ptd_graph_add_edge_parameterized(vertex, child, weight, edge_state);
//       }
//     }
//   }
//   free(state);
//   Graph *graph = new Graph(kingman_graph, avl_tree);
//   return graph;
// }


//////////////////////////////

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

// [[Rcpp::export]]
vector<int> absorbing_states(
        SEXP phase_type_graph
) {
  Rcpp::XPtr<Graph> graphcpp(phase_type_graph);
  ptd_graph *graph = graphcpp->c_graph();
  vector<int> parents;
  for (size_t v = 0; v < graph->vertices_length; ++v) {
      if (graph->vertices[v]->edges_length == 0) {
         parents.push_back(v+1);
      } 
  }
  return parents;
}


void print_state(int *state, int l) {

    for (int i=0; i<l; i++) {
         fprintf(stderr, "%d ", state[i]);
    }
    fprintf(stderr, "\n");
}



// Graph* redo_graph(struct ptd_graph *graph, int edge_state_size, int epoque_label) {

//   // new graph and avl tree
//   struct ptd_graph *new_graph = ptd_graph_create(graph->state_length);
//   struct ptd_avl_tree *avl_tree = ptd_avl_tree_create(graph->state_length);
    
//   // state buffer
//   int *state = (int *) calloc(graph->state_length, sizeof(int));
    
//   // the single absorbing state to use in the new graph
//   struct ptd_vertex *absorbing_vertex;

//   // ADD ALL THE VERTICES /////////////////////////////
    
//   // add all vertices to new graph except absorbing ones
//   for (size_t i = 0; i < graph->vertices_length; i++) {

//     // old vertex
//     struct ptd_vertex *vertex = graph->vertices[i];

//     // skip old starting vertex
//     if (vertex == graph->starting_vertex) {
//         continue;
//     }

//     // skip all absorbing vertices, keep pointer to the last absorbing
//     if (vertex->edges_length == 0) {
//         if (vertex->state[(graph->state_length)-1] == epoque_label) {
//             // keep last absorbing vertex
//             absorbing_vertex = vertex;
//         } else {
//             continue;
//         }
//     }
//     // add verticies to new graph
//     ptd_find_or_create_vertex(new_graph, avl_tree, vertex->state);
//   }      
    
//   // add vertex for the absorbing state for the last epoque
//   ptd_find_or_create_vertex(new_graph, avl_tree, absorbing_vertex->state);

//   // ADD ALL THE EDGES /////////////////////////////

//   // add edges from starting vertex (IPV)
//   for (int j=0; j < graph->starting_vertex->edges_length; ++j) {
//       ptd_graph_add_edge(new_graph->starting_vertex, 
//           graph->starting_vertex->edges[j]->to, graph->starting_vertex->edges[j]->weight);
//   }

//   // loop old vertices again to add edges
//   for (size_t i = 0; i < graph->vertices_length; i++) {

//         // old vertex
//         struct ptd_vertex *vertex = graph->vertices[i];

//         if (vertex == graph->starting_vertex) {
//             continue;
//         }      
//         // new vertex
//         struct ptd_vertex *new_vertex = ptd_find_or_create_vertex(new_graph, avl_tree, vertex->state);

//         for (int j=0; j < vertex->edges_length; ++j) {

//             struct ptd_vertex *child_vertex;
//             if (vertex->edges[j]->to->edges_length == 0) {
//                 child_vertex = absorbing_vertex;
//             // fprintf(stderr, "abs %d\n", child_vertex->index);
//             } else {
//                 child_vertex = vertex->edges[j]->to;
//             // fprintf(stderr, "    %d\n", child_vertex->index);
//             }
//             struct ptd_vertex *new_child_vertex = ptd_find_or_create_vertex(new_graph, avl_tree, child_vertex->state);

//             if (vertex->edges[j]->parameterized) {
//                 double *edge_state = (double *) calloc(edge_state_size, sizeof(double));         
//                 memcpy(edge_state, ((ptd_edge_parameterized *) vertex->edges[j])->state, edge_state_size * sizeof(double));
//                 ptd_graph_add_edge_parameterized(new_vertex, new_child_vertex, vertex->edges[j]->weight, edge_state);
//             } else {
//                 ptd_graph_add_edge(new_vertex, new_child_vertex, vertex->edges[j]->weight);
//             }
//         }
//     }



// // free(state);
// return new Graph(new_graph, avl_tree)   ;

// //     for (int j=0; j < vertex->edges_length; ++j) {
// //         struct ptd_vertex *child_vertex = ptd_find_or_create_vertex(new_graph, avl_tree, vertex->edges[j]->to->state);
// //         if (vertex->edges[j]->parameterized) {
// //             fprintf(stderr, "param\n");
// //             double *edge_state = (double *) calloc(edge_state_size, sizeof(double));         
// //             memcpy(edge_state, ((ptd_edge_parameterized *) vertex->edges[j])->state, edge_state_size * sizeof(double));
// //             ptd_graph_add_edge_parameterized(new_vertex, child_vertex, vertex->edges[j]->weight, edge_state);
// //         } else {
// //             fprintf(stderr, "no param\n");
// //             ptd_graph_add_edge(new_vertex, child_vertex, vertex->edges[j]->weight);
// //         }
// //         }
// //   }
// //   free(state);
// //   return new Graph(new_graph, avl_tree)   ;

// }


// Graph* consolidate_graph(struct ptd_graph *graph) {

//     struct ptd_scc_graph *scc = ptd_find_strongly_connected_components(graph);
//     struct ptd_vertex **vertices =
//             (struct ptd_vertex **) calloc(graph->vertices_length, sizeof(*vertices));
//     struct ptd_scc_vertex **v = ptd_scc_graph_topological_sort(scc);
//     size_t idx = 0;

//     struct ptd_vertex *absorbing_vertex;

//     for (size_t i = 0; i < scc->vertices_length; ++i) {
//         for (size_t j = 0; j < v[i]->internal_vertices_length; ++j) {
//             vertices[idx] = v[i]->internal_vertices[j];
//             vertices[idx]->index = idx;
//             idx++;

//             // FIND LAST ABSORBING STATE
//             if (v[i]->internal_vertices[j]->edges_length == 0) {
//                 absorbing_vertex = v[i]->internal_vertices[j]; // is relpaced if there are more than one, leaving us with the last one I hope...
//             }

//         }
//     }

//     size_t size = graph->vertices_length;

//     struct ptd_graph *new_graph = ptd_graph_create(graph->state_length);
//     struct ptd_avl_tree *avl_tree = ptd_avl_tree_create(graph->state_length);

//     size_t *indices = (size_t *) calloc(size, sizeof(*indices));
//     size_t index = 0;

//     for (size_t k = 0; k < size; ++k) {
//         struct ptd_vertex *vertex = vertices[k];

//         if (graph->starting_vertex != vertex && vertex->edges_length != 0) {
//             indices[vertex->index] = index;

//             // ADD VERTICES THAT ARE NOT STARTING OR ABSORBING
//             ptd_find_or_create_vertex(new_graph, avl_tree, vertex->state);
//             // res->vertices[index] = vertex;

//             index++;
//         }
//     }

//     // ADD AN ABSORBING STATE
//     ptd_find_or_create_vertex(new_graph, avl_tree, absorbing_vertex->state);

//     // res->length = index;

//     for (size_t k = 0; k < graph->vertices_length; ++k) {
//         struct ptd_vertex *vertex = vertices[k];

//         if (vertex->edges_length == 0) {
//             continue;
//         }

//         if (vertex == graph->starting_vertex) {
//             double rate = 0;

//             for (size_t i = 0; i < vertex->edges_length; ++i) {
//                 struct ptd_edge *edge = vertex->edges[i];

//                 rate += edge->weight;
//             }

//             for (size_t i = 0; i < vertex->edges_length; ++i) {
//                 struct ptd_edge *edge = vertex->edges[i];

//                 if (edge->to->edges_length != 0) {

//                     // res->initial_probability_vector[indices[edge->to->index]] = edge->weight / rate;
//                     ptd_graph_add_edge(
//                         new_graph->starting_vertex, 
//                         ptd_find_or_create_vertex(new_graph, avl_tree, edge->to->state), 
//                         edge->weight / rate);

//                 } 
//             }
//             continue;
//         }

//         for (size_t i = 0; i < vertex->edges_length; ++i) {
//             struct ptd_edge *edge = vertex->edges[i];

//             if (edge->to->edges_length != 0) {

//                 // ADD EDGES FROM TO NON-ABSORBING VERTICES
//                 ptd_graph_add_edge(
//                     ptd_find_or_create_vertex(new_graph, avl_tree, vertex->state), 
//                     ptd_find_or_create_vertex(new_graph, avl_tree, edge->to->state), 
//                     edge->weight);
//                 // res->sub_intensity_matrix[indices[vertex->index]][indices[edge->to->index]] += edge->weight;

//             } else {

//                 ptd_graph_add_edge(
//                     ptd_find_or_create_vertex(new_graph, avl_tree, vertex->state), 
//                     ptd_find_or_create_vertex(new_graph, avl_tree, absorbing_vertex->state), 
//                     edge->weight);

//             }

//             // res->sub_intensity_matrix[indices[vertex->index]][indices[vertex->index]] -= edge->weight;
//         }
//     }


//     // ADD EDGES TO ABSORBING STATES (COMPUTE RATES AS )

//     for (size_t i = 0; i < graph->vertices_length; ++i) {
//         graph->vertices[i]->index = i;
//     }

//     free(v);
//     ptd_scc_graph_destroy(scc);
//     free(indices);
//     free(vertices);

//     // return res;
//     return new Graph(new_graph, avl_tree);
// }


// Graph* consolidate_graph(struct ptd_graph *graph) {

//     struct ptd_phase_type_distribution *res = (struct ptd_phase_type_distribution *) malloc(sizeof(*res));

//     if (res == NULL) {
//         return NULL;
//     }

//     res->length = 0;

//     size_t size = graph->vertices_length;

//     res->memory_allocated = size;
//     res->vertices = (struct ptd_vertex **) calloc(size, sizeof(struct ptd_vertex *));

//     if (res->vertices == NULL) {
//         free(res);
//         return NULL;
//     }

//     res->initial_probability_vector = (double *) calloc(size, sizeof(double));

//     if (res->initial_probability_vector == NULL) {
//         free(res->vertices);
//         free(res);
//         return NULL;
//     }

//     res->sub_intensity_matrix = (double **) calloc(size, sizeof(double *));

//     if (res->sub_intensity_matrix == NULL) {
//         free(res->initial_probability_vector);
//         free(res->vertices);
//         free(res);
//         return NULL;
//     }

//     for (size_t i = 0; i < size; ++i) {
//         res->sub_intensity_matrix[i] = (double *) calloc(size, sizeof(double));

//         if ((res->sub_intensity_matrix)[i] == NULL) {
//             for (size_t j = 0; j < i; ++j) {
//                 free(res->sub_intensity_matrix[j]);
//             }

//             free(res->sub_intensity_matrix);
//             free(res->initial_probability_vector);
//             free(res->vertices);
//             free(res);
//             return NULL;
//         }
//     }

//     struct ptd_scc_graph *scc = ptd_find_strongly_connected_components(graph);
//     struct ptd_vertex **vertices =
//             (struct ptd_vertex **) calloc(graph->vertices_length, sizeof(*vertices));
//     struct ptd_scc_vertex **v = ptd_scc_graph_topological_sort(scc);
//     size_t idx = 0;

//     for (size_t i = 0; i < scc->vertices_length; ++i) {
//         for (size_t j = 0; j < v[i]->internal_vertices_length; ++j) {
//             vertices[idx] = v[i]->internal_vertices[j];
//             vertices[idx]->index = idx;
//             idx++;
//         }
//     }

//     size_t *indices = (size_t *) calloc(size, sizeof(*indices));
//     size_t index = 0;

//     for (size_t k = 0; k < graph->vertices_length; ++k) {
//         struct ptd_vertex *vertex = vertices[k];

//         if (graph->starting_vertex != vertex && vertex->edges_length != 0) {
//             indices[vertex->index] = index;
//             res->vertices[index] = vertex;
//             index++;
//         }
//     }

//     res->length = index;

//     for (size_t k = 0; k < graph->vertices_length; ++k) {
//         struct ptd_vertex *vertex = vertices[k];

//         if (vertex->edges_length == 0) {
//             continue;
//         }

//         if (vertex == graph->starting_vertex) {
//             double rate = 0;

//             for (size_t i = 0; i < vertex->edges_length; ++i) {
//                 struct ptd_edge *edge = vertex->edges[i];

//                 rate += edge->weight;
//             }

//             for (size_t i = 0; i < vertex->edges_length; ++i) {
//                 struct ptd_edge *edge = vertex->edges[i];

//                 if (edge->to->edges_length != 0) {
//                     res->initial_probability_vector[indices[edge->to->index]] = edge->weight / rate;
//                 }
//             }

//             continue;
//         }

//         for (size_t i = 0; i < vertex->edges_length; ++i) {
//             struct ptd_edge *edge = vertex->edges[i];

//             if (edge->to->edges_length != 0) {
//                 res->sub_intensity_matrix[indices[vertex->index]][indices[edge->to->index]] += edge->weight;
//             }

//             res->sub_intensity_matrix[indices[vertex->index]][indices[vertex->index]] -= edge->weight;
//         }
//     }

//     for (size_t i = 0; i < graph->vertices_length; ++i) {
//         graph->vertices[i]->index = i;
//     }

// //////////////

//     struct ptd_phase_type_distribution *dist = res;

//     NumericMatrix SIM(dist->length, dist->length);
//     NumericVector IPV(dist->length);

//     for (size_t i = 0; i < dist->length; ++i) {
//         IPV(i) = dist->initial_probability_vector[i];

//         for (size_t j = 0; j < dist->length; ++j) {
//             SIM(i, j) = dist->sub_intensity_matrix[i][j];
//         }
//     }

//     size_t state_length = graph->state_length;
//     NumericMatrix states(dist->length, state_length);

//     for (size_t i = 0; i < dist->length; i++) {
//         for (size_t j = 0; j < state_length; j++) {
//             states(i, j) = dist->vertices[i]->state[j];
//         }
//     }

//     NumericVector indices(dist->length);
//     for (size_t i = 0; i < dist->length; i++) {
//         indices[i] = dist->vertices[i]->index + 1;
//     }



// /////////////////////////////////////

//     int has_rewards = 0;

//     size_t state_space_size = graph->state_length;

//     Graph *cppGraph = new Graph(state_space_size);
//     struct ptd_graph *graph = cppGraph->c_graph();

//     for (int i = 0; i < SIM.nrow(); i++) {
//         if (has_rewards) {
//             int *state = (int *) calloc((size_t) rw.ncol(), sizeof(*state));

//             for (int j = 0; j < rw.ncol(); j++) {
//                 state[j] = rw.at(i, j);
//             }

//             ptd_vertex_create_state(graph, state);
//         } else {
//             ptd_vertex_create(graph);
//         }
//     }

//     struct ptd_vertex *absorbing_vertex = ptd_vertex_create(graph);
//     double sum_outgoing = 0;

//     for (int i = 0; i < SIM.nrow(); i++) {
//         if (IPV[i] != 0) {
//             ptd_graph_add_edge(graph->starting_vertex, graph->vertices[i + 1], IPV[i]);
//             sum_outgoing += IPV[i];
//         }
//     }

//     if (sum_outgoing < 0.99999) {
//         ptd_graph_add_edge(graph->starting_vertex, absorbing_vertex, 1 - sum_outgoing);
//     }

//     for (int i = 0; i < SIM.nrow(); i++) {
//         double s = 0;

//         for (int j = 0; j < SIM.nrow(); j++) {
//             if (i == j) {
//                 continue;
//             }

//             double weight = SIM.at(i, j);

//             if (weight != 0) {
//                 ptd_graph_add_edge(graph->vertices[i + 1], graph->vertices[j + 1], weight);
//                 s += weight;
//             }
//         }

//         double w = -(SIM.at(i, i) + s);

//         if (w >= 0.000001) {
//             ptd_graph_add_edge(graph->vertices[i + 1], absorbing_vertex, w);
//         }
//     }

// ///////////////////////


//     free(v);
//     ptd_scc_graph_destroy(scc);
//     free(indices);
//     free(vertices);

//     ptd_phase_type_distribution_destroy(dist);

// ///////////////////////

//     return cppGraph;

// }

Graph* consolidate_graph(struct ptd_graph *graph) {

    // struct ptd_phase_type_distribution *dist = ptd_graph_as_phase_type_distribution(graph->c_graph());
    struct ptd_phase_type_distribution *dist = ptd_graph_as_phase_type_distribution(graph);

    NumericMatrix SIM(dist->length, dist->length);
    NumericVector IPV(dist->length);

    for (size_t i = 0; i < dist->length; ++i) {
        IPV(i) = dist->initial_probability_vector[i];

        for (size_t j = 0; j < dist->length; ++j) {
            SIM(i, j) = dist->sub_intensity_matrix[i][j];
        }
    }

    size_t state_length = graph->state_length;
    NumericMatrix states(dist->length, state_length);

    for (size_t i = 0; i < dist->length; i++) {
        for (size_t j = 0; j < state_length; j++) {
            states(i, j) = dist->vertices[i]->state[j];
            fprintf(stderr, "%d\n", dist->vertices[i]->state[j]);
        }
    }

    NumericVector indices(dist->length);
    for (size_t i = 0; i < dist->length; i++) {
        indices[i] = dist->vertices[i]->index + 1;
    }

//    ptd_phase_type_distribution_destroy(dist);
    
    // return List::create(Named("states") = states, _["SIM"] = SIM, _["IPV"] = IPV, _["indices"] = indices);

///////////////////////////////////////////////

    Nullable <NumericMatrix> rewards = states;

    NumericMatrix rw;
    bool has_rewards;

    if (IPV.length() <= 0 || IPV.length() != SIM.ncol() || SIM.ncol() != SIM.nrow()) {
        char message[1024];

        snprintf(
                message,
                1024,
                "Failed: IPV must have length > 0, was %i, and SIM must have same dimensions and be square, was %i, %i",
                (int) IPV.length(), (int) SIM.nrow(), (int) SIM.ncol()
        );

        throw std::runtime_error(
                message
        );
    }

    if (rewards.isNotNull()) {
        rw = NumericMatrix(rewards);
        has_rewards = true;

        if (rw.nrow() != SIM.nrow()) {
            char message[1024];

            snprintf(
                    message,
                    1024,
                    "Failed: Rewards must have %i rows, had %i",
                    (int) SIM.nrow(), (int) rw.nrow()
            );

            throw std::runtime_error(
                    message
            );
        }
    } else {
        has_rewards = false;
    }

    size_t state_space_size = has_rewards ? rw.ncol() : 1;

    Graph *cppGraph = new Graph(state_space_size);
    struct ptd_graph *new_graph = cppGraph->c_graph();

    for (int i = 0; i < SIM.nrow(); i++) {
        if (has_rewards) {
            int *state = (int *) calloc((size_t) rw.ncol(), sizeof(*state));

            for (int j = 0; j < rw.ncol(); j++) {
                state[j] = rw.at(i, j);
            }

            ptd_vertex_create_state(new_graph, state);
        } else {
            ptd_vertex_create(new_graph);
        }
    }

    struct ptd_vertex *absorbing_vertex = ptd_vertex_create(new_graph);
    double sum_outgoing = 0;

    for (int i = 0; i < SIM.nrow(); i++) {
        if (IPV[i] != 0) {
            ptd_graph_add_edge(new_graph->starting_vertex, new_graph->vertices[i + 1], IPV[i]);
            sum_outgoing += IPV[i];
        }
    }

    if (sum_outgoing < 0.99999) {
        ptd_graph_add_edge(new_graph->starting_vertex, absorbing_vertex, 1 - sum_outgoing);
    }

    for (int i = 0; i < SIM.nrow(); i++) {
        double s = 0;

        for (int j = 0; j < SIM.nrow(); j++) {
            if (i == j) {
                continue;
            }

            double weight = SIM.at(i, j);

            if (weight != 0) {
                ptd_graph_add_edge(new_graph->vertices[i + 1], new_graph->vertices[j + 1], weight);
                s += weight;
            }
        }

        double w = -(SIM.at(i, i) + s);

        if (w >= 0.000001) {
            ptd_graph_add_edge(new_graph->vertices[i + 1], absorbing_vertex, w);
        }
    }

    return Rcpp::XPtr<Graph>(
            cppGraph
    );



// return new Graph(new_graph, avl_tree);

}


// void add_epoque(Graph* graph, std::vector<double> scalars, double t) {
//     Graph *graphcpp = graph;


// [[Rcpp::export]]
void add_epoque(SEXP graph, std::vector<double> scalars, double t) {
    Rcpp::XPtr<Graph> graphcpp(graph);
    struct ptd_graph *ptd_graph = graphcpp->c_graph();
    struct ptd_avl_tree *avl_tree = graphcpp->c_avl_tree();
// SEXP add_epoque(SEXP graph, std::vector<double> scalars, double t) {
//     Rcpp::XPtr<Graph> graphcpp(graph);
//     struct ptd_clone_res tmp_graph = ptd_clone_graph(graphcpp->c_graph(), graphcpp->c_avl_tree());
//     struct ptd_graph *ptd_graph = tmp_graph.graph;
//     struct ptd_avl_tree *avl_tree = tmp_graph.avl_tree;


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

        // R_CheckUserInterrupt();

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
        // double *edge_state = (double *) calloc(((int) scalars.size()), sizeof(double));       
        double *edge_state = (double *) calloc(((int) scalars.size()), sizeof(*edge_state));   
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
            // if (vertex->edges[j]->to->edges_length > 0) {
            //     state[ptd_graph->state_length-1] = nr_states;  // NOT SURE ABOUT THIS...
            // }
            state[ptd_graph->state_length-1] = nr_states;

            struct ptd_vertex *sister_to_vertex  = ptd_find_or_create_vertex(ptd_graph, avl_tree, state);

            // edge params for edge to that state
            double *sister_edge_state = (double *) calloc(((int) scalars.size()), sizeof(double));         

            if (vertex->edges[j]->parameterized) { // THERE SHOULD BE AN ELSE FOR THIS....

                // memcpy(sister_edge_state, ((ptd_edge_parameterized *) vertex->edges[j]->state), ((int) scalars.size()) * sizeof(double));
                for (int k=0; k < scalars.size(); ++k) {
                    sister_edge_state[k] = ((ptd_edge_parameterized *) vertex->edges[j])->state[k];
                }
                struct ptd_edge *new_edge = (struct ptd_edge *) ptd_graph_add_edge_parameterized(
                    sister_vertex, 
                    sister_to_vertex, 
                    vertex->edges[j]->weight, 
                    sister_edge_state);         
            
                ptd_edge_update_weight_parameterized(new_edge, &scalars[0], ((int) scalars.size()));

            } else {
                ptd_graph_add_edge(
                    sister_vertex, 
                    sister_to_vertex, 
                    vertex->edges[j]->weight);     
            }
            }
          }

    graphcpp->notify_change();

    // free(stop_probs);
    // free(acum_visit);
    // free(state);
    // free(edge_state);

    ptd_validate_graph(ptd_graph);
    // ptd_notify_change(ptd_graph);  

  //   Graph *new_graph = redo_graph(ptd_graph, ((int) scalars.size()), nr_states);
  //   // free(tmp_graph);    
  // return Rcpp::XPtr<Graph>(new_graph);

//     Graph *new_graph = consolidate_graph(ptd_graph);
//     // free(tmp_graph);    
//   return Rcpp::XPtr<Graph>(new_graph);


  // return Rcpp::XPtr<Graph>(
  //   new Graph(ptd_graph, avl_tree)
  // );

}


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

// int main() {
//     int n = 3;
//     int m = 3;
//     Graph *graph = coalescent_graph(n, m);
//     vector<double> scalars(2, 0);
//     scalars[0] =  0.2;
//     add_epoque(graph, scalars, 1);
//     scalars[0] =  0.1;
//     add_epoque(graph, scalars, 2);
//     ptd_graph *ptd_graph = graph->c_graph();
//     double *rewards = NULL;
//     double *waiting_times = ptd_expected_waiting_time(ptd_graph, rewards);
//     return 0;
// }