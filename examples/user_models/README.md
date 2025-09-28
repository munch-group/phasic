# User-Defined C++ Models for PtDAlgorithms

This directory contains example C++ models that can be loaded using `Graph.load_cpp_model()`.

## Available Models

- **simple_exponential.cpp** - Exponential distribution (1 parameter: rate)
- **erlang_distribution.cpp** - Erlang distribution (2 parameters: rate, stages)
- **birth_death_process.cpp** - Birth-death process (2 parameters: birth rate, death rate)
- **mm1_queue.cpp** - M/M/1 queue (3 parameters: arrival rate, service rate, max queue size)
- **rabbit_flooding.cpp** - Rabbit flooding simulation (3 parameters: initial rabbits, flood rate left, flood rate right)

## Two Approaches for Loading Models

### 1. JAX-Compatible Approach (Default)
```python
from ptdalgorithms import Graph
import jax
import jax.numpy as jnp

# Load model with full JAX support
model = Graph.load_cpp_model("simple_exponential.cpp")

# Use with JAX transformations
theta = jnp.array([1.0])
times = jnp.array([0.5, 1.0, 1.5, 2.0])

# Basic evaluation
pdf = model(theta, times)

# JIT compilation
jit_model = jax.jit(model)
pdf_fast = jit_model(theta, times)

# Automatic differentiation
grad_fn = jax.grad(lambda p: jnp.sum(model(p, times)))
gradient = grad_fn(theta)

# Vectorization
batch_params = jnp.array([[0.5], [1.0], [1.5]])
vmap_model = jax.vmap(lambda p: model(p, times))
batch_pdfs = vmap_model(batch_params)
```

**Best for:**
- Parameter optimization
- Research requiring gradients
- Integration with JAX/ML libraries
- Bayesian inference

### 2. FFI Approach (Foreign Function Interface)
```python
from ptdalgorithms import Graph
import numpy as np

# Load model with FFI for efficiency
builder = Graph.load_cpp_model("simple_exponential.cpp", use_ffi=True)

# Build graph once
theta = np.array([1.0])
graph = builder(theta)

# Use many times without rebuilding
for t in [0.5, 1.0, 1.5, 2.0]:
    pdf = graph.pdf(t, granularity=100)
    # Process result
```

**Best for:**
- Monte Carlo simulations with fixed parameters
- Real-time/production systems
- Large-scale simulations
- Maximum performance

## Performance Comparison

| Approach | Build Graph | Evaluation | 1000 calls |
|----------|-------------|------------|------------|
| JAX-Compatible | Every call | Fast | ~9 seconds |
| FFI | Once | Very fast | ~0.004 seconds |

**Speedup with FFI: ~2000x for repeated evaluations with same parameters**

## Writing Your Own C++ Model

Create a C++ file that includes `"user_model.h"` and implements:

```cpp
#include "user_model.h"
#include <vector>

ptdalgorithms::Graph build_model(const double* theta, int n_params) {
    ptdalgorithms::Graph g(state_dimension);

    // Get starting vertex
    auto start = g.starting_vertex();

    // Create states (use std::vector<int>, not initializer lists!)
    std::vector<int> state0 = {0};
    auto v0 = g.find_or_create_vertex(state0);

    // Add transitions
    start->add_edge(v0, 1.0);

    // Add more states and transitions...

    return g;
}
```

**Important:** Always use `std::vector<int>` for state vectors, not initializer lists like `{0}`.

## Examples

For detailed examples, see:
- **examples/load_cpp_model_simple.py** - Basic usage of both approaches
- **examples/jax_compatible_example.py** - Comprehensive JAX approach demonstration
- **examples/ffi_approach_example.py** - Comprehensive FFI approach demonstration
- **examples/approach_comparison.py** - Side-by-side comparison with benchmarks

## Quick Decision Guide

**Use JAX-Compatible when:**
- You need automatic differentiation
- Integrating with JAX ecosystem
- Parameters change frequently
- Research and experimentation

**Use FFI when:**
- Parameters are fixed or change rarely
- Performance is critical
- Real-time applications
- Production systems

Both approaches use the same C++ model files - you can switch by changing one parameter!