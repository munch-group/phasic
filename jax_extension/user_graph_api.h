#ifndef USER_GRAPH_API_H
#define USER_GRAPH_API_H

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <memory>

// Forward declarations
struct UserConfig;
class Graph;

/**
 * Simple Graph API for user-defined graph construction
 * Users only need to implement graph structure, not PMF computation
 */
class Graph {
private:
    struct Vertex {
        std::vector<int> state;
        std::vector<std::pair<int, double>> edges;  // target_vertex_id, rate
        double absorption_rate = 0.0;
        int id;
    };
    
    std::vector<Vertex> vertices_;
    int next_vertex_id_ = 0;
    std::map<std::vector<int>, int> state_to_vertex_;  // For deduplication
    
public:
    /**
     * Add a vertex with given state vector
     * Returns vertex ID (automatically deduplicated)
     */
    int add_vertex(const std::vector<int>& state);
    
    /**
     * Add transition edge from source to target vertex with given rate
     */
    void add_edge(int from_vertex, int to_vertex, double rate);
    
    /**
     * Set absorption rate for a vertex (exit from the system)
     */
    void set_absorption_rate(int vertex, double rate);
    
    /**
     * Get vertex count
     */
    int vertex_count() const { return vertices_.size(); }
    
    /**
     * Get state vector for vertex
     */
    const std::vector<int>& get_state(int vertex_id) const;
    
    /**
     * Get outgoing edges for vertex
     */
    const std::vector<std::pair<int, double>>& get_edges(int vertex_id) const;
    
    /**
     * Get absorption rate for vertex
     */
    double get_absorption_rate(int vertex_id) const;
    
    /**
     * Convert to transition matrix representation for PMF computation
     */
    void to_matrices(std::vector<double>& transition_matrix, 
                    std::vector<double>& exit_rates,
                    std::vector<double>& initial_distribution) const;
};

/**
 * Configuration structure passed to user graph builders
 */
struct UserConfig {
    int nr_samples = 3;
    double mutation_rate = 0.0;
    bool apply_discretization = true;
    std::map<std::string, double> custom_params;
    
    // Helper to get custom parameter with default
    double get_param(const std::string& name, double default_val = 0.0) const;
};

/**
 * User graph builder function signature
 */
using GraphBuilder = std::function<Graph(const double* theta, int theta_size, const UserConfig& config)>;

/**
 * Registry for user-defined graph builders
 */
class GraphBuilderRegistry {
private:
    static std::map<std::string, GraphBuilder>& get_builders();
    
public:
    /**
     * Register a graph builder function
     */
    static void register_builder(const std::string& name, GraphBuilder builder);
    
    /**
     * Get registered builder by name
     */
    static GraphBuilder get_builder(const std::string& name);
    
    /**
     * Check if builder exists
     */
    static bool has_builder(const std::string& name);
    
    /**
     * List all registered builders
     */
    static std::vector<std::string> list_builders();
};

/**
 * Apply discretization to continuous graph (mutation process)
 */
Graph apply_discretization(const Graph& continuous_graph, double mutation_rate, 
                          const UserConfig& config);

/**
 * Normalize graph rates (make rows sum to exit rate)
 */
Graph normalize_graph(const Graph& graph);

#endif // USER_GRAPH_API_H