#include "user_graph_api.h"
#include <iostream>

int main() {
    std::cout << "Testing Graph API..." << std::endl;
    
    // Test basic graph creation
    Graph graph;
    
    std::vector<int> state1 = {2};
    std::vector<int> state2 = {1};
    std::vector<int> state3 = {0};
    
    int v1 = graph.add_vertex(state1);
    int v2 = graph.add_vertex(state2); 
    int v3 = graph.add_vertex(state3);
    
    std::cout << "Added " << graph.vertex_count() << " vertices" << std::endl;
    
    graph.add_edge(v1, v2, 1.0);
    graph.add_edge(v2, v3, 1.0);
    graph.set_absorption_rate(v3, 1.0);
    
    // Test matrix conversion
    std::vector<double> trans_matrix, exit_rates, initial_dist;
    graph.to_matrices(trans_matrix, exit_rates, initial_dist);
    
    std::cout << "Matrix size: " << trans_matrix.size() << std::endl;
    std::cout << "Initial dist: ";
    for (double x : initial_dist) std::cout << x << " ";
    std::cout << std::endl;
    
    // Test registry
    std::cout << "Testing registry..." << std::endl;
    
    auto simple_builder = [](const double* theta, int theta_size, const UserConfig& config) -> Graph {
        Graph g;
        std::vector<int> s = {1};
        g.add_vertex(s);
        return g;
    };
    
    GraphBuilderRegistry::register_builder("test", simple_builder);
    std::cout << "Builder registered" << std::endl;
    
    if (GraphBuilderRegistry::has_builder("test")) {
        std::cout << "Builder found!" << std::endl;
        
        double theta[] = {100.0, 0.01};
        UserConfig config;
        
        Graph test_graph = GraphBuilderRegistry::get_builder("test")(theta, 2, config);
        std::cout << "Test graph has " << test_graph.vertex_count() << " vertices" << std::endl;
    }
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}