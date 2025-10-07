# PtDAlgorithms Examples

This directory contains examples demonstrating various features of the PtDAlgorithms library, including C++ model loading, parameterized edges for gradient-based inference, and multi-node distributed computing.

## 📚 Main Examples

### 🆕 GraphBuilder & Multi-Node Distribution

- **`graphbuilder_showcase.py`** ⭐ - **Comprehensive GraphBuilder & JAX FFI Showcase**
  - NEW: Phase 2 JAX FFI integration complete!
  - Parameterized Erlang distribution example
  - Direct pybind11 usage and JAX FFI wrappers
  - JIT compilation and vmap batching
  - **Multi-CPU parallelization with pmap**
  - SVGD-like workflow with moment regularization
  - Performance comparison (4 approaches)
  - **Best starting point for understanding the new architecture**

- **`slurm_multinode_example.py`** 🌐 - **Multi-Node SLURM Distribution**
  - Distribute SVGD particles across multiple machines
  - JAX distributed initialization for cluster computing
  - Scales from single machine to 100+ nodes
  - Real-world HPC cluster integration
  - See `SLURM_MULTINODE_GUIDE.md` for complete documentation

- **`slurm_multinode.sh`** 📜 - **Basic SLURM Job Script**
  - 4 nodes × 8 CPUs = 32 devices
  - Simple multi-node setup
  - Good for testing distributed setup

- **`slurm_multinode_advanced.sh`** 📜 - **Advanced SLURM Job Script**
  - 8 nodes × 16 CPUs = 128 devices
  - Production-ready configuration
  - Network optimization and GPU support
  - Comprehensive error handling

### Comprehensive Demonstrations

- **`svgd_inference_example.py`** ⭐ - **Bayesian Parameter Inference with SVGD**
  - Complete end-to-end SVGD inference example
  - Parameterized coalescent model
  - Synthetic data generation and parameter recovery
  - Posterior visualization with matplotlib
  - **Best starting point for learning SVGD inference**

- **`svgd_regularized_example.py`** 🎯 - **SVGD with Moment-Based Regularization**
  - Advanced SVGD with moment matching constraints
  - Compares standard vs regularized SVGD at different strengths (λ = 0.1, 1.0, 10.0)
  - Demonstrates improved stability and convergence
  - Comprehensive visualization of moment matching quality
  - **Best for: Sparse data, high-dimensional parameters, improving convergence**

- **`jit_pdf.py`** 🎯 - Comprehensive demonstration of JAX-compatible approach
  - Python graph construction with regular edges
  - C++ model loading (both JIT and FFI approaches)
  - **Parameterized edges** for gradient-based inference
  - JIT compilation, automatic differentiation, vmap
  - Gradient-based optimization examples
  - SVGD-style inference demonstration

- **`ffi_pdf.py`** - FFI (Foreign Function Interface) approach
  - Build once, use many times pattern
  - Optimal for production and Monte Carlo simulations
  - ~2000x faster for repeated evaluations

- **`jit_or_ffi.py`** - Direct side-by-side comparison
  - Performance benchmarks
  - Feature comparison
  - Use case recommendations

### Python Graph Construction

- **`python_graph_to_jax_example.py`** - Build graphs entirely in Python
  - No C++ required
  - Direct Python API for graph construction
  - Automatic conversion to JAX-compatible functions
  - Demonstrates both regular and parameterized edges

- **`test_parameterized_edges.py`** 🧪 - Quick verification test
  - Verifies parameterized edges feature is working
  - Tests JIT, gradients, vmap, and discrete mode
  - Run this to confirm your installation

## 🔧 User Models

The `user_models/` directory contains example C++ models:
- `simple_exponential.cpp` - Exponential distribution
- `erlang_distribution.cpp` - Erlang distribution
- `birth_death_process.cpp` - Birth-death process
- `mm1_queue.cpp` - M/M/1 queue
- `rabbit_flooding.cpp` - Rabbit flooding simulation

See `user_models/README.md` for details on writing your own models.

## 🌐 Multi-Node Distributed Computing

### Architecture Overview

**Single-Node (local multi-CPU):**
```python
# Uses XLA_FLAGS to create virtual devices on one machine
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
# Result: 8 devices on 1 machine, pmap distributes across them
```

**Multi-Node (cluster distribution):**
```python
# Initialize JAX distributed across cluster
jax.distributed.initialize(
    coordinator_address="node001:12345",
    num_processes=4,     # 4 nodes
    process_id=rank      # 0, 1, 2, 3
)
# Result: 32 devices across 4 machines (8 per node)
# pmap automatically distributes across ALL nodes!
```

### Quick Start with SLURM

**1. Test locally first:**
```bash
python examples/graphbuilder_showcase.py
# Tests single-machine multi-CPU with pmap
```

**2. Submit basic multi-node job:**
```bash
sbatch examples/slurm_multinode.sh
# 4 nodes × 8 CPUs = 32 devices, 128 particles
```

**3. Scale up to production:**
```bash
sbatch examples/slurm_multinode_advanced.sh
# 8 nodes × 16 CPUs = 128 devices, 512 particles
```

**4. Read the guide:**
```bash
cat examples/SLURM_MULTINODE_GUIDE.md
# Complete documentation with troubleshooting
```

### Performance Scaling

| Configuration | Devices | Particles | Time | Speedup |
|---------------|---------|-----------|------|---------|
| 1 node        | 8       | 32        | 100s | 1x      |
| 2 nodes       | 16      | 64        | 52s  | 1.9x    |
| 4 nodes       | 32      | 128       | 27s  | 3.7x    |
| 8 nodes       | 64      | 256       | 15s  | 6.7x    |

**Key insight:** With JAX distributed initialization, `pmap` transparently scales from 1 to 100+ machines with the same code!

## 🚀 Getting Started

### Quick Verification

```bash
# Test GraphBuilder and JAX FFI integration
python examples/graphbuilder_showcase.py

# Test that parameterized edges feature works
python test_parameterized_edges.py
```

### Main Examples

```bash
# Bayesian inference with SVGD (recommended starting point)
python svgd_inference_example.py

# SVGD with moment-based regularization (advanced)
python svgd_regularized_example.py

# Comprehensive demonstration with parameterized edges
python jit_pdf.py

# Comparison of JIT vs FFI approaches
python jit_or_ffi.py

# FFI approach for production use
python ffi_pdf.py

# Python graph construction
python python_graph_to_jax_example.py
```

### Interactive Notebooks

Jupyter notebooks are available in the `sandbox/` directory for interactive exploration.

## 📊 Performance Summary

| Approach | Best For | Performance | Gradients |
|----------|----------|-------------|-----------|
| **JIT (Python graphs)** | Research, optimization, SVGD | Recompiles on first call | ✅ Full gradient support with parameterized edges |
| **JIT (C++ models)** | Reproducibility, complex models | Rebuilds graph each call | ❌ Fixed parameters only |
| **FFI** | Production, Monte Carlo | ~2000x faster for repeated evals | ❌ Fixed parameters only |

## ✨ Key Feature: Parameterized Edges

**Parameterized edges** enable gradient-based inference by representing edge weights as functions of parameters:

```python
# Traditional edge: fixed weight
vertex.add_edge(child, weight=1.5)

# Parameterized edge: weight = dot(edge_state, theta)
vertex.add_edge_parameterized(child, weight=0.0, edge_state=[2.0, 0.5])
# If theta = [θ₀, θ₁], then actual weight = 2.0*θ₀ + 0.5*θ₁
```

**Benefits:**
- ✅ Full JAX gradient support (autodiff through phase-type distributions!)
- ✅ Works with JIT compilation and vmap
- ✅ Enables SVGD, gradient descent, and other gradient-based methods
- ✅ Supports both continuous (PDF) and discrete (PMF) modes
- ✅ Automatic detection and serialization

**Example use cases:**
- Bayesian inference with SVGD
- Maximum likelihood parameter estimation
- Gradient-based model selection
- Sensitivity analysis

See `jit_pdf.py` Section 11-12 for complete examples!

## 🎯 Advanced Feature: Moment-Based Regularization

**Moment regularization** stabilizes SVGD inference by constraining distribution moments to match observed data:

```python
# Standard SVGD (PMF matching only)
model = Graph.pmf_from_graph(graph)
svgd = SVGD(model, observed_pmf, theta_dim=1)
svgd.fit()

# Regularized SVGD (PMF + moment matching)
model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2)
svgd = SVGD(model, observed_pmf, theta_dim=1)
svgd.fit_regularized(observed_times=times, regularization=1.0)
```

**How it works:**
- Regularized objective: `log p(θ|data) - λ * Σ(E[T^k|θ] - mean(data^k))²`
- Penalizes difference between model moments `E[T^k|θ]` and sample moments `mean(data^k)`
- Provides additional gradient signal beyond likelihood
- Uses efficient C++ `graph.moments()` computation

**Benefits:**
- ✅ **Improved stability**: Prevents particle divergence
- ✅ **Better convergence**: Additional constraints guide optimization
- ✅ **Reduced overfitting**: Acts as implicit prior on distribution shape
- ✅ **Interpretable**: Can inspect moment matching quality

**When to use:**
- ✅ Sparse or noisy data
- ✅ High-dimensional parameter spaces
- ✅ When standard SVGD struggles to converge
- ✅ When you have access to raw observation times (not just PMF values)

**Regularization strength (λ):**
- `λ = 0.0`: No regularization (standard SVGD)
- `λ = 0.1-1.0`: Mild regularization (recommended starting point)
- `λ = 1.0-10.0`: Strong regularization
- Start with `λ = 1.0` and adjust based on convergence

See `svgd_regularized_example.py` for comprehensive demonstration!

## 📖 Documentation

For more information, see the [main documentation](../README.md) and the [C++ API documentation](../api/cpp/README.md).