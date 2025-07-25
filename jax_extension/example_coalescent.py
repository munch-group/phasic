"""
Example: User-defined coalescent model using separated architecture
"""

import jax
import jax.numpy as jnp
from separated_graph_python import register_graph_builder, GraphConfig

# Define coalescent graph construction in C++
coalescent_cpp_code = """
    Graph graph;
    
    // Extract parameters
    double pop_size = theta[0];
    double mutation_rate = config.mutation_rate;
    int nr_samples = config.nr_samples;
    
    // Initial state: all samples in one lineage
    std::vector<int> initial_state(nr_samples + 1, 0);
    initial_state[nr_samples] = 1;  // One group of size nr_samples
    
    // Build coalescent tree using breadth-first search
    std::queue<std::vector<int>> to_process;
    to_process.push(initial_state);
    
    while (!to_process.empty()) {
        auto state = to_process.front();
        to_process.pop();
        
        int vertex_id = graph.add_vertex(state);
        
        // Check for coalescent transitions
        for (int group_size = 2; group_size <= nr_samples; ++group_size) {
            if (state[group_size] > 0) {
                // Coalescence within this group
                auto new_state = state;
                new_state[group_size] -= 1;       // One less group of this size
                new_state[group_size - 1] += 1;   // One more group of size-1
                
                // Coalescent rate: binomial coefficient * rate per pair
                double rate = state[group_size] * group_size * (group_size - 1) / (2.0 * pop_size);
                
                int target_vertex = graph.add_vertex(new_state);
                graph.add_edge(vertex_id, target_vertex, rate);
                
                to_process.push(new_state);
            }
        }
        
        // Check if this is a terminal state (only singletons left)
        bool is_terminal = true;
        for (int i = 2; i <= nr_samples; ++i) {
            if (state[i] > 0) {
                is_terminal = false;
                break;
            }
        }
        
        if (is_terminal) {
            // Set absorption rate (coalescent process complete)
            graph.set_absorption_rate(vertex_id, 1.0);
        }
    }
    
    return graph;
"""

# Register the coalescent model
coalescent_pmf = register_graph_builder("coalescent", coalescent_cpp_code)

# Example usage
if __name__ == "__main__":
    print("Testing separated coalescent model...")
    
    # Parameters: [population_size, mutation_rate]
    theta = jnp.array([1000.0, 0.01])
    
    # Time points to evaluate PMF
    times = jnp.array([1, 2, 3, 4, 5])
    
    # Configuration
    config = GraphConfig(
        nr_samples=3,
        mutation_rate=theta[1],  # Use parameter value
        apply_discretization=True
    )
    
    # Test basic PMF computation
    print("\\n1. Basic PMF computation:")
    try:
        pmf_vals = coalescent_pmf(theta, times, config)
        print(f"PMF values: {pmf_vals}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with JAX transformations
    print("\\n2. Testing with JAX JIT:")
    
    def log_likelihood(theta, data):
        pmf_vals = coalescent_pmf(theta, data, config)
        return jnp.sum(jnp.log(jnp.maximum(pmf_vals, 1e-12)))
    
    jit_log_likelihood = jax.jit(log_likelihood)
    
    try:
        loglik = log_likelihood(theta, times)
        print(f"Log-likelihood: {loglik}")
        
        loglik_jit = jit_log_likelihood(theta, times)
        print(f"JIT Log-likelihood: {loglik_jit}")
    except Exception as e:
        print(f"JIT Error: {e}")
    
    # Test gradients
    print("\\n3. Testing gradients:")
    try:
        grad_fn = jax.grad(log_likelihood)
        gradients = grad_fn(theta, times)
        print(f"Gradients: {gradients}")
    except Exception as e:
        print(f"Gradient Error: {e}")
    
    # Test batching (SVGD-style)
    print("\\n4. Testing batching (SVGD particles):")
    
    particles = jnp.array([
        [1000.0, 0.01],
        [500.0, 0.02], 
        [2000.0, 0.005]
    ])
    
    try:
        batch_loglik = jax.vmap(lambda p: log_likelihood(p, times))(particles)
        print(f"Batch log-likelihoods: {batch_loglik}")
        
        batch_grads = jax.vmap(lambda p: jax.grad(lambda theta: log_likelihood(theta, times))(p))(particles)
        print(f"Batch gradients shape: {batch_grads.shape}")
    except Exception as e:
        print(f"Batching Error: {e}")
    
    print("\\nTest complete!")