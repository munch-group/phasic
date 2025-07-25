#include "user_graph_api.h"
#include <algorithm>
#include <queue>
#include <set>
#include <stdexcept>
#include <iostream>

// Graph implementation
int Graph::add_vertex(const std::vector<int>& state) {
    // Check if state already exists (deduplication)
    auto it = state_to_vertex_.find(state);
    if (it != state_to_vertex_.end()) {
        return it->second;
    }
    
    // Create new vertex
    Vertex vertex;
    vertex.state = state;
    vertex.id = next_vertex_id_++;
    
    vertices_.push_back(vertex);
    state_to_vertex_[state] = vertex.id;
    
    return vertex.id;
}

void Graph::add_edge(int from_vertex, int to_vertex, double rate) {
    if (from_vertex >= vertices_.size() || to_vertex >= vertices_.size()) {
        throw std::invalid_argument("Invalid vertex ID in add_edge");
    }
    
    vertices_[from_vertex].edges.emplace_back(to_vertex, rate);
}

void Graph::set_absorption_rate(int vertex, double rate) {
    if (vertex >= vertices_.size()) {
        throw std::invalid_argument("Invalid vertex ID in set_absorption_rate");
    }
    
    vertices_[vertex].absorption_rate = rate;
}

const std::vector<int>& Graph::get_state(int vertex_id) const {
    if (vertex_id >= vertices_.size()) {
        throw std::invalid_argument("Invalid vertex ID in get_state");
    }
    return vertices_[vertex_id].state;
}

const std::vector<std::pair<int, double>>& Graph::get_edges(int vertex_id) const {
    if (vertex_id >= vertices_.size()) {
        throw std::invalid_argument("Invalid vertex ID in get_edges");
    }
    return vertices_[vertex_id].edges;
}

double Graph::get_absorption_rate(int vertex_id) const {
    if (vertex_id >= vertices_.size()) {
        throw std::invalid_argument("Invalid vertex ID in get_absorption_rate");
    }
    return vertices_[vertex_id].absorption_rate;
}

void Graph::to_matrices(std::vector<double>& transition_matrix, 
                       std::vector<double>& exit_rates,
                       std::vector<double>& initial_distribution) const {
    int n = vertices_.size();
    
    // Initialize matrices
    transition_matrix.assign(n * n, 0.0);
    exit_rates.assign(n, 0.0);
    initial_distribution.assign(n, 0.0);
    
    // Set initial distribution (assuming first vertex is initial state)
    if (n > 0) {
        initial_distribution[0] = 1.0;
    }
    
    // Fill transition matrix and exit rates
    for (int i = 0; i < n; ++i) {
        const auto& vertex = vertices_[i];
        
        // Add transitions
        double total_rate = 0.0;
        for (const auto& edge : vertex.edges) {
            int j = edge.first;
            double rate = edge.second;
            transition_matrix[i * n + j] = rate;
            total_rate += rate;
        }
        
        // Add absorption rate
        exit_rates[i] = vertex.absorption_rate;
        total_rate += vertex.absorption_rate;
        
        // Normalize to get transition probabilities
        if (total_rate > 0) {
            for (const auto& edge : vertex.edges) {
                int j = edge.first;
                transition_matrix[i * n + j] /= total_rate;
            }
            exit_rates[i] = total_rate;
        }
    }
}

// UserConfig implementation
double UserConfig::get_param(const std::string& name, double default_val) const {
    auto it = custom_params.find(name);
    return (it != custom_params.end()) ? it->second : default_val;
}

// GraphBuilderRegistry implementation
std::map<std::string, GraphBuilder>& GraphBuilderRegistry::get_builders() {
    static std::map<std::string, GraphBuilder> builders;
    return builders;
}

void GraphBuilderRegistry::register_builder(const std::string& name, GraphBuilder builder) {
    get_builders()[name] = builder;
}

GraphBuilder GraphBuilderRegistry::get_builder(const std::string& name) {
    auto& builders = get_builders();
    auto it = builders.find(name);
    if (it == builders.end()) {
        throw std::runtime_error("Graph builder '" + name + "' not found");
    }
    return it->second;
}

bool GraphBuilderRegistry::has_builder(const std::string& name) {
    return get_builders().find(name) != get_builders().end();
}

std::vector<std::string> GraphBuilderRegistry::list_builders() {
    std::vector<std::string> names;
    for (const auto& pair : get_builders()) {
        names.push_back(pair.first);
    }
    return names;
}

// Utility functions
Graph apply_discretization(const Graph& continuous_graph, double mutation_rate, 
                          const UserConfig& config) {
    Graph discrete_graph = continuous_graph;  // Copy
    
    int original_vertex_count = continuous_graph.vertex_count();
    
    // Add mutation transitions for each vertex with positive state values
    for (int i = 0; i < original_vertex_count; ++i) {
        const auto& state = continuous_graph.get_state(i);
        
        // For each position in state vector that can mutate
        for (size_t j = 0; j < state.size(); ++j) {
            if (state[j] > 0) {
                // Create mutation state (decrement this position)
                std::vector<int> mutation_state = state;
                mutation_state[j] -= 1;
                
                // Add mutation vertex and transition
                int mutation_vertex = discrete_graph.add_vertex(mutation_state);
                discrete_graph.add_edge(i, mutation_vertex, mutation_rate * state[j]);
            }
        }
    }
    
    return discrete_graph;
}

Graph normalize_graph(const Graph& graph) {
    Graph normalized = graph;  // Copy
    
    // Graph::to_matrices already handles normalization
    // This function can add additional normalization logic if needed
    
    return normalized;
}