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


// [[Rcpp::export]]
SEXP reward_zip(SEXP graph, Function callback_fun) {
    Rcpp::XPtr<Graph> graphcpp(graph);
    struct ptd_graph *ptd_graph = graphcpp->c_graph();
    struct ptd_avl_tree *avl_tree = graphcpp->c_avl_tree();
    int state_vector_length = ptd_graph->state_length;

    // probabilities of eaching each vertex in the DAG
    double *vertex_probs = (double *) calloc(ptd_graph->vertices_length, sizeof(double));
    for (size_t i=0; i < ptd_graph->vertices_length; ++i) vertex_probs[0] = 0;
    vertex_probs[0] = 1;

    // get state vector size for bl_graph
    IntegerVector vec;
    for (size_t i=0; i < state_vector_length; ++i) vec.push_back(ptd_graph->vertices[0]->state[i]);     
    IntegerVector bl_vec = callback_fun(vec);
    int bl_state_vector_length = bl_vec.size();
  

// THIS ONLY WORKS ON THE DAG... 
// MAYBE THIS ALGORITHM COULD BE SOMEHOW STORED AS INSTRUCTIONS DOWNSTREAM OF THE DAG CREATION IF THE USER SPECIFIES A CALLBACK?

// Make a function collapse_transform(graph, callback) like reward_transform(graph, rewards) PRODUCE A NEW GRAPH using REWARDS 


// if the graph is acyclic, I can just collapsed and run expectation on the collapsed graph
// if the graph is cyclic, I need to first make the DAG before I can collapse, so there is not getting around the O(n^3) step

// ... unless I can collapse everything but the strongly connected components of a cyclic
// find the strongly connectec components and treat them as "vertices". If that graph is acyclic, I can collapse that

// or maybe, I can somehow get around not knowing the vector_probs, if I track how they are visited when collapsing the graph?...
    
//////////////////////////////////
    
    // int **state_mapping = (int **) calloc(ptd_graph->vertices_length, sizeof(int *));
    
    // for (size_t i=0; i < ptd_graph->vertices_length; ++i) {
        
    //     struct ptd_vertex *vertex = ptd_graph->vertices[i];
    //     if (i == 0) {
    //         IntegerVector vec(bl_state_vector_length); // zeros
    //     } else {                
    //         IntegerVector vec;
    //         for (size_t k=0; k < state_vector_length; ++k)
    //             vec.push_back(vertex->state[k]);     
    //         IntegerVector bl_vec = callback_fun(vec);
    //     }
    //     int *mapped_state = (int *) calloc(bl_state_vector_length, sizeof(int));
    //     std::copy(bl_vec.begin(), bl_vec.end(), mapped_state);            
    //     state_mapping[i] = mapped_state;
    
    // }

/////////////////////////////////

    // create bl_graph
    struct ptd_graph *bl_ptd_graph = ptd_graph_create(bl_state_vector_length);
    struct ptd_avl_tree *bl_avl_tree = ptd_avl_tree_create(bl_state_vector_length);
        
    // lists of rewards
    std::vector<double *> rewards;
    std::vector< std::vector<double> > reward_list;

    // buffers
    int *bl_state = (int *) calloc(bl_state_vector_length, sizeof(int));
    int *bl_child_state = (int *) calloc(bl_state_vector_length, sizeof(int));

    // sort vertices toplogically
    // .....
    
    // traverse graph in toplogical order
    for (size_t i=0; i < ptd_graph->vertices_length; ++i) {

        // get vertex
        struct ptd_vertex *vertex = ptd_graph->vertices[i];

        // get corresponding bl_vertex
        struct ptd_vertex *bl_vertex;
        if (i == 0) {
            bl_vertex = bl_ptd_graph->vertices[0];
        } else {
            ////////////////////////////
            IntegerVector vec;
            for (size_t k=0; k < state_vector_length; ++k)
                vec.push_back(vertex->state[k]);     
            IntegerVector bl_vec = callback_fun(vec);
            // int* bl_state = &bl_vec[0];
            std::copy(bl_vec.begin(), bl_vec.end(), bl_state);
            ////////////////////////////            
            // int *bl_state = state_mapping[vertex->index];           
            ////////////////////////////

            bl_vertex = ptd_find_or_create_vertex(bl_ptd_graph, bl_avl_tree, bl_state);
        }

        // compute total edge weight
        double tot_edge_weight = 0;
        for (size_t j = 0; j < vertex->edges_length; ++j) {
            tot_edge_weight += vertex->edges[j]->weight;
        }  
        // loop edges
        for (size_t j = 0; j < vertex->edges_length; ++j) {

            // get child state
            int *child_state = vertex->edges[j]->to->state;

            // record prob of reaching child vertices
            vertex_probs[vertex->edges[j]->to->index] += vertex_probs[i] * vertex->edges[j]->weight / tot_edge_weight;

            // get bl_child_state
            ////////////////////////////
            IntegerVector child_vec;
            for (size_t k=0; k < state_vector_length; ++k)
                child_vec.push_back(child_state[k]);
            IntegerVector bl_child_vec = callback_fun(child_vec);
            // int* bl_child_state = &bl_child_vec[0];
            std::copy(bl_child_vec.begin(), bl_child_vec.end(), bl_child_state);
            ////////////////////////////            
            // bl_child_state = state_mapping[vertex->edges[j]->to->index];           
            ////////////////////////////

            // get bl_child_vertex
            struct ptd_vertex *bl_child_vertex = ptd_find_or_create_vertex(bl_ptd_graph, bl_avl_tree, bl_child_state);

            // make sure reward_list is long enough
            for (size_t k=0; k < ptd_graph->vertices_length; ++k) {
                if (reward_list.size() > bl_child_vertex->index) {
                    break;
                }
                std::vector<double> vec(state_vector_length, 0.0);                
                reward_list.push_back(vec);
            }

            // update rewards
            std:vector<double> vec = reward_list[bl_child_vertex->index];            
            for (size_t k=0; k < state_vector_length; ++k) {
                vec[k] = vec[k] + child_state[k] * vertex->edges[j]->weight * vertex_probs[i];
            }
            reward_list[bl_child_vertex->index] = vec;
            
            // if bl_edge exists, add edge->weight to that edge
            int bl_edge_index = -1;
            for (size_t k = 0; k < bl_vertex->edges_length; ++k) {
                if (bl_vertex->edges[k]->to->index == bl_child_vertex->index) {
                    bl_edge_index = k;
                    break;
                }
            }
            if (bl_edge_index > -1) {
                // if so, add edge->weight to that edge
                bl_vertex->edges[bl_edge_index]->weight += vertex_probs[i] * vertex->edges[j]->weight;
            } else {
                // else add the edge
                ptd_graph_add_edge(bl_vertex, bl_child_vertex, vertex_probs[i] * vertex->edges[j]->weight);  
            }

        }
    }

    free(vertex_probs);
    free(bl_state);
    free(bl_child_state);

    
    std::vector<double> concat_rewards;
    int n = bl_ptd_graph->vertices_length;
    for (auto vec: reward_list) {
        double total = 0;
        for (auto& x : vec)
             total += x;
        if (total > 0) {
            for (size_t i=0; i < state_vector_length; ++i) { 
                vec[i] = n * vec[i] / total;
            }
        }
        concat_rewards.insert( concat_rewards.end(), vec.begin(), vec.end() );
        --n;
    }
    NumericMatrix reward_matrix(
        state_vector_length,    
        bl_ptd_graph->vertices_length,
        concat_rewards.begin());

    List L = List::create(
             Rcpp::XPtr<Graph>(
                new Graph(bl_ptd_graph, bl_avl_tree)
             ),
    reward_matrix
        );
    // Setting element names
    L.attr("names") = CharacterVector::create("graph", "rewards");
            
    return L;
    
}



    // // array to vector
    // std::vector<int> v;
    // for (int i = 0; i < N; i++) 
    //     v.push_back(arr[i]); 

    // // vector to array
    // int *arr = (int) calloc(n, sizeof(int));
    // std::copy(v.begin(), v.end(), arr);

                // Rcout << "The value of v : " << child_state[k] * vertex->edges[j]->weight << "\n";
// Rprintf("the value of %f \n", child_state[k] * vertex->edges[j]->weight);
// REprintf("the value of %f \n", child_state[k] * vertex->edges[j]->weight);