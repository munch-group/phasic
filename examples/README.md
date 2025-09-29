# PtDAlgorithms Examples

This directory contains examples demonstrating various features of the PtDAlgorithms library, with a focus on loading user-defined C++ models.

## 📚 C++ Model Loading Examples

### Quick Start
- **`load_cpp_model_simple.py`** - Simple introduction to both approaches

### Interactive Notebook
- **`cpp_models_showcase.ipynb`** 🎯 - Jupyter notebook with interactive examples using the Rabbit Flooding model
  - Side-by-side comparison of JAX and FFI approaches
  - Interactive visualizations
  - Performance benchmarks
  - Parameter optimization examples

### Deep Dives
- **`jax_compatible_example.py`** - Comprehensive demonstration of JAX-compatible approach
  - JIT compilation
  - Automatic differentiation
  - Vectorization with vmap
  - Integration with JAX ecosystem

- **`ffi_approach_example.py`** - Comprehensive demonstration of FFI approach
  - Build once, use many times pattern
  - Performance optimization
  - Monte Carlo simulations
  - Real-time applications

### Comparisons
- **`approach_comparison.py`** - Direct side-by-side comparison
  - Performance benchmarks
  - Feature comparison table
  - Use case recommendations
  - Decision guide

## 🔧 User Models

The `user_models/` directory contains example C++ models:
- `simple_exponential.cpp` - Exponential distribution
- `erlang_distribution.cpp` - Erlang distribution
- `birth_death_process.cpp` - Birth-death process
- `mm1_queue.cpp` - M/M/1 queue
- `rabbit_flooding.cpp` - Rabbit flooding simulation

See `user_models/README.md` for details on writing your own models.

## 🚀 Getting Started

### Option 1: Run the Jupyter Notebook (Recommended)
```bash
jupyter notebook cpp_models_showcase.ipynb
```

### Option 2: Run Python Scripts
```bash
# Quick introduction
python load_cpp_model_simple.py

# Full comparison
python approach_comparison.py

# Deep dive into JAX features
python jax_compatible_example.py

# Deep dive into FFI efficiency
python ffi_approach_example.py
```

## 📊 Performance Summary

| Approach | Best For | Performance |
|----------|----------|-------------|
| **JAX-Compatible** | Gradients, optimization, research | Rebuilds graph each call |
| **FFI** | Production, Monte Carlo, fixed parameters | ~2000x faster for repeated evals |

Both approaches use the same C++ model files - switch with one parameter: `use_ffi=True`!

## 📖 Documentation

For more information, see the [main documentation](../README.md) and the [C++ API documentation](../api/cpp/README.md).