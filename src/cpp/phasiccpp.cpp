/*
 * MIT License
 *
 * Copyright (c) 2021 Tobias Røikjer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdexcept>
#include <sstream>
#include <stack>
#include <cerrno>
#include <cstring>
#include <vector>
#include "phasiccpp.h"

/* While it seems very strange to have this in a C file, the R code
 * has very strange linking behavior, and we therefore sometimes include
 * the same C file...
 */

#ifndef PTDALGORITHMS_PTDCPP_C
#define PTDALGORITHMS_PTDCPP_C

static void assert_same_length(std::vector<int> state, struct ptd_graph *graph) {
    if (state.size() != graph->state_length) {
        std::stringstream message;
        message << "Vector `state` argument must have same size as graph state length. Was '";
        message << state.size() << "' expected '" << graph->state_length << "'" << std::endl;

        throw std::invalid_argument(message.str());
    }
}

static int *force_same_length(std::vector<int> state, struct ptd_graph *graph) {
    int *res = (int *) calloc(graph->state_length, sizeof(int));
  
    for (size_t i = 0; i < state.size(); ++i) {
        res[i] = state[i];
    }
  
    return res;
}

phasic::Vertex phasic::Graph::create_vertex(std::vector<int> state) {
    int *c = force_same_length(state, c_graph());

    Vertex res = create_vertex(c);

    return res;
}


phasic::Vertex phasic::Graph::create_vertex(const int *state) {
    struct ptd_vertex *c_vertex = ptd_vertex_create_state(c_graph(), (int*)state);
  
    Vertex vertex = phasic::Vertex(*this, c_vertex);
    notify_change();

    return vertex;
}

phasic::Vertex *phasic::Graph::create_vertex_p(std::vector<int> state) {
    int *c = force_same_length(state, c_graph());

    Vertex *res = create_vertex_p(c);

    return res;
}


phasic::Vertex *phasic::Graph::create_vertex_p(const int *state) {
    struct ptd_vertex *c_vertex = ptd_vertex_create_state(c_graph(), (int*)state);

    Vertex *vertex = new phasic::Vertex(*this, c_vertex);
    notify_change();
    
    return vertex;
}

phasic::Vertex phasic::Graph::find_vertex(std::vector<int> state) {
    int *c = force_same_length(state, c_graph());

    Vertex res = find_vertex(c);

    free(c);

    return res;
}


phasic::Vertex phasic::Graph::find_vertex(const int *state) {
    struct ptd_avl_node *node = ptd_avl_tree_find(this->rf_graph->tree, state);

    if (node == NULL) {
        throw std::runtime_error(
                "No such vertex found\n"
        );
    }

    return phasic::Vertex(*this, (struct ptd_vertex *) node->entry);
}

phasic::Vertex *phasic::Graph::find_vertex_p(std::vector<int> state) {
    int *c = force_same_length(state, c_graph());

    Vertex *res = find_vertex_p(c);

    free(c);

    return res;
}

phasic::Vertex *phasic::Graph::find_vertex_p(const int *state) {
    struct ptd_avl_node *node = ptd_avl_tree_find(this->rf_graph->tree, state);

    if (node == NULL) {
        throw std::runtime_error(
                "No such vertex found\n"
        );
    }

    return new phasic::Vertex(*this, (struct ptd_vertex *) node->entry);
}

bool phasic::Graph::vertex_exists(std::vector<int> state) {
    assert_same_length(state, this->rf_graph->graph);

    return vertex_exists(&state[0]);
}

bool phasic::Graph::vertex_exists(const int *state) {
    struct ptd_avl_node *node = ptd_avl_tree_find(this->rf_graph->tree, state);

    return (node != NULL);
}

phasic::Vertex phasic::Graph::find_or_create_vertex(std::vector<int> state) {
    assert_same_length(state, this->rf_graph->graph);

    return find_or_create_vertex(&state[0]);
}

phasic::Vertex phasic::Graph::find_or_create_vertex(const int *state) {
    notify_change();

    return phasic::Vertex(*this, ptd_find_or_create_vertex(c_graph(), c_avl_tree(), state));
}

phasic::Vertex *phasic::Graph::find_or_create_vertex_p(std::vector<int> state) {
    assert_same_length(state, this->rf_graph->graph);

    return find_or_create_vertex_p(&state[0]);
}

phasic::Vertex *phasic::Graph::find_or_create_vertex_p(const int *state) {
    notify_change();

    return new phasic::Vertex(*this, ptd_find_or_create_vertex(c_graph(), c_avl_tree(), state));
}

phasic::Vertex phasic::Graph::starting_vertex() {
    return Vertex(*this, this->rf_graph->graph->starting_vertex);
}

phasic::Vertex *phasic::Graph::starting_vertex_p() {
    return new Vertex(*this, this->rf_graph->graph->starting_vertex);
}

std::vector<phasic::Vertex> phasic::Graph::vertices() {
    std::vector<Vertex> vec;

    for (size_t i = 0; i < c_graph()->vertices_length; ++i) {
        vec.push_back(Vertex(*this, c_graph()->vertices[i]));
    }

    return vec;
}

std::vector<phasic::Vertex *> phasic::Graph::vertices_p() {
    std::vector<Vertex *> vec;

    for (size_t i = 0; i < c_graph()->vertices_length; ++i) {
        vec.push_back(new Vertex(*this, c_graph()->vertices[i]));
    }

    return vec;
}

phasic::Vertex phasic::Graph::vertex_at(size_t index) {
    return Vertex(*this, c_graph()->vertices[index]);
}

phasic::Vertex *phasic::Graph::vertex_at_p(size_t index) {
    return new Vertex(*this, c_graph()->vertices[index]);
}

size_t phasic::Graph::vertices_length() {
    return c_graph()->vertices_length;
}

bool phasic::Graph::parameterized() {
    return c_graph()->parameterized;
}

phasic::PhaseTypeDistribution phasic::Graph::phase_type_distribution() {
    struct ptd_phase_type_distribution *matrix = ptd_graph_as_phase_type_distribution(this->rf_graph->graph);

    if (matrix == NULL) {
        char msg[1024];

        snprintf(msg, 1024, "Failed to make sub-intensity matrix: %s \n", std::strerror(errno));

        throw new std::runtime_error(
                msg
        );
    }

    return PhaseTypeDistribution(*this, matrix);
}

void phasic::Vertex::add_edge(Vertex &to, double weight) {
    if (this->vertex == to.vertex) {
        throw new std::invalid_argument(
                "The edge to add is between the same vertex\n"
        );
    }

    // EDGE MODE LOCKING: Lock to CONSTANT mode on first non-IPV edge with scalar syntax
    // IPV (starting vertex) edges don't affect mode locking
    if (this->vertex != this->vertex->graph->starting_vertex) {
        if (this->vertex->graph->edge_mode == PTD_EDGE_MODE_UNLOCKED) {
            // First non-IPV edge: lock to CONSTANT mode
            this->vertex->graph->edge_mode = PTD_EDGE_MODE_CONSTANT;
        } else if (this->vertex->graph->edge_mode == PTD_EDGE_MODE_PARAMETERIZED) {
            // Graph is locked to PARAMETERIZED, reject scalar syntax
            throw std::runtime_error(
                "Cannot mix constant and parameterized edges. "
                "Graph mode is PARAMETERIZED (locked by first non-IPV edge using array syntax). "
                "This edge uses scalar syntax. "
                "Use add_edge(vertex, [coefficients]) for parameterized edges."
            );
        }
    }

    graph.notify_change();

    // Constant edge: single-element coefficient array
    double coeff = weight;
    struct ptd_edge *result = ptd_graph_add_edge(this->vertex, to.vertex, &coeff, 1);

    if (result == NULL) {
        throw std::runtime_error((char *) ptd_err);
    }
}


void phasic::Vertex::add_edge_parameterized(Vertex &to, double weight, std::vector<double> edge_state) {
    if (this->vertex == to.vertex) {
        throw new std::invalid_argument(
                "The edge to add is between the same vertex\n"
        );
    }

    // EDGE MODE LOCKING: Lock to PARAMETERIZED mode on first non-IPV edge with array syntax
    // IPV (starting vertex) edges don't affect mode locking
    if (this->vertex != this->vertex->graph->starting_vertex) {
        if (this->vertex->graph->edge_mode == PTD_EDGE_MODE_UNLOCKED) {
            // First non-IPV edge: lock to PARAMETERIZED mode
            this->vertex->graph->edge_mode = PTD_EDGE_MODE_PARAMETERIZED;
        } else if (this->vertex->graph->edge_mode == PTD_EDGE_MODE_CONSTANT) {
            // Graph is locked to CONSTANT, reject array syntax
            throw std::runtime_error(
                "Cannot mix constant and parameterized edges. "
                "Graph mode is CONSTANT (locked by first non-IPV edge using scalar syntax). "
                "This edge uses array syntax. "
                "Use add_edge(vertex, scalar) for constant edges."
            );
        }
    }

    size_t state_length = edge_state.size();
    double *state = (double *) calloc(state_length, sizeof(*state));

    for (size_t i = 0; i < state_length; ++i) {
        state[i] = edge_state[i];
    }

    graph.notify_change();

    // Unified API: use coefficient array directly
    struct ptd_edge *result = ptd_graph_add_edge(this->vertex, to.vertex, state, state_length);

    free(state);

    if (result == NULL) {
        throw std::runtime_error((char *) ptd_err);
    }
}


phasic::Vertex phasic::Vertex::add_aux_vertex(double rate) {
    // Create all-zero state vector
    size_t state_len = this->vertex->graph->state_length;
    std::vector<int> zero_state(state_len, 0);

    // create the aux vertex
    Vertex aux = graph.create_vertex(zero_state);

    // Edge 1: FROM aux TO this vertex with constant weight 1.0
    // Create edge manually to bypass validation (always constant weight 1.0)
    struct ptd_edge *edge1 = (struct ptd_edge *)malloc(sizeof(*edge1));
    if (edge1 == NULL) {
        throw std::runtime_error("Failed to allocate edge");
    }

    edge1->to = this->vertex;
    edge1->weight = 1.0;
    edge1->coefficients_length = 0;  // No coefficients - pure constant
    edge1->coefficients = NULL;
    edge1->should_free_coefficients = false;

    // Add edge to aux vertex's edge list
    struct ptd_edge **new_edges = (struct ptd_edge **)realloc(
        aux.vertex->edges,
        (aux.vertex->edges_length + 1) * sizeof(struct ptd_edge *)
    );
    if (new_edges == NULL) {
        free(edge1);
        throw std::runtime_error("Failed to allocate edge array");
    }
    aux.vertex->edges = new_edges;
    aux.vertex->edges[aux.vertex->edges_length] = edge1;
    aux.vertex->edges_length++;

    // Edge 2: FROM this vertex TO aux with given rate (constant)
    // Use normal add_edge for proper validation
    this->add_edge(aux, rate);

    graph.notify_change();

    return aux;
}


phasic::Vertex phasic::Vertex::add_aux_vertex(std::vector<double> rate_coeffs) {
    // Create all-zero state vector
    size_t state_len = this->vertex->graph->state_length;
    std::vector<int> zero_state(state_len, 0);

    // create the aux vertex
    Vertex aux = graph.create_vertex(zero_state);

    // Edge 1: FROM aux TO this vertex with constant weight 1.0
    // Create edge manually to bypass validation (always constant weight 1.0)
    struct ptd_edge *edge1 = (struct ptd_edge *)malloc(sizeof(*edge1));
    if (edge1 == NULL) {
        throw std::runtime_error("Failed to allocate edge");
    }

    edge1->to = this->vertex;
    edge1->weight = 1.0;
    edge1->coefficients_length = 0;  // No coefficients - pure constant
    edge1->coefficients = NULL;
    edge1->should_free_coefficients = false;

    // Add edge to aux vertex's edge list
    struct ptd_edge **new_edges = (struct ptd_edge **)realloc(
        aux.vertex->edges,
        (aux.vertex->edges_length + 1) * sizeof(struct ptd_edge *)
    );
    if (new_edges == NULL) {
        free(edge1);
        throw std::runtime_error("Failed to allocate edge array");
    }
    aux.vertex->edges = new_edges;
    aux.vertex->edges[aux.vertex->edges_length] = edge1;
    aux.vertex->edges_length++;

    // Edge 2: FROM this vertex TO aux with given rate (parameterized)
    // Use normal add_edge_parameterized for proper validation
    this->add_edge_parameterized(aux, 0.0, rate_coeffs);

    graph.notify_change();

    return aux;
}


std::vector<int> phasic::Vertex::state() {
    return std::vector<int>(
            this->vertex->state,
            this->vertex->state + this->vertex->graph->state_length
    );
}

std::vector<phasic::Edge> phasic::Vertex::edges() {
    std::vector<Edge> vector;

    for (size_t i = 0; i < this->vertex->edges_length; ++i) {
        Edge edge_i(
                this->vertex->edges[i]->to,
                this->vertex->edges[i],
                graph,
                this->vertex->edges[i]->weight
        );

        vector.push_back(edge_i);
    }

    return vector;
}

std::vector<phasic::ParameterizedEdge> phasic::Vertex::parameterized_edges() {
    std::vector<ParameterizedEdge> vector;

    for (size_t i = 0; i < this->vertex->edges_length; ++i) {
        // Include edges with coefficient arrays (parameterized in unified interface)
        // This includes single-parameter edges (coefficients_length == 1)
        if (this->vertex->edges[i]->coefficients_length >= 1) {
            ParameterizedEdge edge_i(
                    this->vertex->edges[i]->to,
                    this->vertex->edges[i],
                    graph,
                    this->vertex->edges[i]->weight,
                    this->vertex->edges[i]->coefficients
            );

            vector.push_back(edge_i);
        }
    }

    return vector;
}

// phasic::Graph phasic::Graph::expectation_dag(std::vector<double> rewards) {
//     struct ptd_clone_res res = ptd_graph_expectation_dag(this->c_graph(), &rewards[0]);

//     if (res.graph == NULL) {
//         throw std::runtime_error((char *) ptd_err);
//     }

//     return Graph(res.graph, res.avl_tree);
// }

// phasic::Graph *phasic::Graph::expectation_dag_p(std::vector<double> rewards) {
//   struct ptd_clone_res res = ptd_graph_expectation_dag(this->c_graph(), &rewards[0]);
  
//   if (res.graph == NULL) {
//     throw std::runtime_error((char *) ptd_err);
//   }
  
//   return new Graph(res.graph, res.avl_tree);
// }

phasic::Graph phasic::Graph::reward_transform(std::vector<double> rewards) {
    struct ptd_graph *res = ptd_graph_reward_transform(this->c_graph(), &rewards[0]);

    if (res == NULL) {
        throw std::runtime_error((char *) ptd_err);
    }

    return Graph(res);
}

phasic::Graph *phasic::Graph::reward_transform_p(std::vector<double> rewards) {
  struct ptd_graph *res = ptd_graph_reward_transform(this->c_graph(), &rewards[0]);
  
  if (res == NULL) {
    throw std::runtime_error((char *) ptd_err);
  }
  
  return new Graph(res);
}

void phasic::Graph::update_weights_parameterized(std::vector<double> scalars) {
    ptd_graph_update_weights(
            this->c_graph(),
            &scalars[0],
            scalars.size()
    );

    // Check if error occurred and throw exception
    if (ptd_err[0] != '\0') {
        std::string error_msg((const char*)ptd_err);
        ptd_err[0] = '\0';  // Clear error
        throw std::runtime_error(error_msg);
    }

    notify_change();
}

#endif