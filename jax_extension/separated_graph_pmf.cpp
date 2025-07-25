#include "user_graph_api.h"
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <limits>

// Forward declarations
void deserialize_config(const char* data, UserConfig& config);
std::string serialize_config(const UserConfig& config);

extern "C" {

/**
 * Separated JAX primitive: user provides graph, system computes PMF
 * Inputs:
 *   - theta: parameter vector
 *   - times: time points for PMF evaluation
 *   - builder_name: name of registered graph builder
 *   - config_data: serialized UserConfig
 */
__attribute__((visibility("default")))
void jax_separated_graph_pmf(void* out_ptr, void** in_ptrs) {
    // Parse inputs
    double* theta = reinterpret_cast<double*>(in_ptrs[0]);
    int64_t* times = reinterpret_cast<int64_t*>(in_ptrs[1]);
    int64_t* dims = reinterpret_cast<int64_t*>(in_ptrs[2]); // [theta_size, n_times]
    char* builder_name = reinterpret_cast<char*>(in_ptrs[3]);
    char* config_data = reinterpret_cast<char*>(in_ptrs[4]);
    double* output = reinterpret_cast<double*>(out_ptr);
    
    int64_t theta_size = dims[0];
    int64_t n_times = dims[1];
    
    try {
        // Deserialize config
        UserConfig config;
        deserialize_config(config_data, config);
        
        // Get user's graph builder
        std::string builder_name_str(builder_name);
        if (!GraphBuilderRegistry::has_builder(builder_name_str)) {
            throw std::runtime_error("Unknown graph builder: " + builder_name_str);
        }
        
        GraphBuilder builder = GraphBuilderRegistry::get_builder(builder_name_str);
        
        // 1. Call user's graph construction
        Graph user_graph = builder(theta, theta_size, config);
        
        // 2. Apply post-processing (discretization, normalization)
        Graph processed_graph = user_graph;
        if (config.apply_discretization && config.mutation_rate > 0) {
            processed_graph = apply_discretization(user_graph, config.mutation_rate, config);
        }
        processed_graph = normalize_graph(processed_graph);
        
        // 3. Convert to matrix form for PMF computation
        std::vector<double> transition_matrix, exit_rates, initial_dist;
        processed_graph.to_matrices(transition_matrix, exit_rates, initial_dist);
        
        // 4. Compute PMF using standard algorithm
        int n_states = initial_dist.size();
        std::vector<double> current_dist = initial_dist;
        std::vector<double> temp_dist(n_states);
        
        for (int64_t time_idx = 0; time_idx < n_times; ++time_idx) {
            int64_t target_time = times[time_idx];
            
            // Reset to initial distribution
            current_dist = initial_dist;
            
            // Apply transition matrix target_time times
            for (int64_t step = 0; step < target_time; ++step) {
                std::fill(temp_dist.begin(), temp_dist.end(), 0.0);
                
                // Matrix-vector multiplication: temp_dist = current_dist * transition_matrix
                for (int i = 0; i < n_states; ++i) {
                    for (int j = 0; j < n_states; ++j) {
                        temp_dist[j] += current_dist[i] * transition_matrix[i * n_states + j];
                    }
                }
                
                std::swap(current_dist, temp_dist);
            }
            
            // Compute PMF: probability of absorption at this time
            double pmf = 0.0;
            for (int i = 0; i < n_states; ++i) {
                // For discrete case: use transition to absorbing state
                // For now, use exit rate as proxy for absorption probability
                pmf += current_dist[i] * (exit_rates[i] / (1.0 + exit_rates[i]));
            }
            
            output[time_idx] = pmf;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in separated graph PMF: " << e.what() << std::endl;
        // Fill with NaN to indicate error
        for (int64_t i = 0; i < n_times; ++i) {
            output[i] = std::numeric_limits<double>::quiet_NaN();
        }
    }
}


} // extern "C"

/**
 * Serialize UserConfig for passing to C++
 */
std::string serialize_config(const UserConfig& config) {
    std::string result;
    result += std::to_string(config.nr_samples) + ",";
    result += std::to_string(config.mutation_rate) + ",";
    result += std::string(config.apply_discretization ? "1" : "0") + ",";
    
    // Add custom parameters
    for (const auto& pair : config.custom_params) {
        result += pair.first + "=" + std::to_string(pair.second) + ";";
    }
    
    return result;
}

/**
 * Deserialize UserConfig from string
 */
void deserialize_config(const char* data, UserConfig& config) {
    std::string str(data);
    size_t pos = 0;
    
    // Parse nr_samples
    size_t comma = str.find(',', pos);
    config.nr_samples = std::stoi(str.substr(pos, comma - pos));
    pos = comma + 1;
    
    // Parse mutation_rate
    comma = str.find(',', pos);
    config.mutation_rate = std::stod(str.substr(pos, comma - pos));
    pos = comma + 1;
    
    // Parse apply_discretization
    comma = str.find(',', pos);
    config.apply_discretization = (str.substr(pos, comma - pos) == "1");
    pos = comma + 1;
    
    // Parse custom parameters
    if (pos < str.length()) {
        std::string params_str = str.substr(pos);
        size_t param_pos = 0;
        
        while (param_pos < params_str.length()) {
            size_t semicolon = params_str.find(';', param_pos);
            if (semicolon == std::string::npos) break;
            
            std::string param = params_str.substr(param_pos, semicolon - param_pos);
            size_t eq = param.find('=');
            if (eq != std::string::npos) {
                std::string key = param.substr(0, eq);
                double value = std::stod(param.substr(eq + 1));
                config.custom_params[key] = value;
            }
            
            param_pos = semicolon + 1;
        }
    }
}