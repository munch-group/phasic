# phasic - Quick Reference

**Version:** 0.22.0
**Paper:** [Røikjer, Hobolth & Munch (2022)](https://doi.org/10.1007/s11222-022-10155-6) - Statistics and Computing
**Repository:** https://github.com/munch-group/phasic
**Contact:** Kasper Munch (kaspermunch@birc.au.dk)

---

## Overview

**phasic** is a high-performance library for computing with **phase-type distributions** using graph-based algorithms. Phase-type distributions model the time until absorption in continuous or discrete-time Markov chains on finite state spaces.

### Key Innovation

Traditional matrix-based methods become computationally infeasible for systems with thousands of states. This library uses **graph-based algorithms** that:
- Execute **10-100x faster** than matrix methods for sparse systems
- Use dramatically less memory (O(n+m) vs O(n²))
- Scale to large state spaces (500,000+ states)
- Support iterative construction of complex models

### Primary Applications

- **Population genetics**: Coalescent models, site frequency spectra
- **Queuing theory**: Service time modeling, system reliability
- **Survival analysis**: Time-to-event modeling
- **Bayesian inference**: Efficient likelihood computation for MCMC/SVGD

---

## Development

- Always use pixi environment.
- Use "pixi run install-dev" for development install.
- Do not ever implement silent fallbacks. Code should work as specified or fail.


## Key Concepts

### Phase-Type Distributions

A **continuous phase-type (PH) distribution** represents the time until absorption in a continuous-time Markov chain:

- **PH(α, S)** where α = initial probability vector, S = sub-intensity matrix
- PDF: f(t) = α · exp(S·t) · s* (forward algorithm, Algorithm 4)
- Moments: E[Tᵏ] computed via reward transformation (Algorithm 2)

**Discrete phase-type (DPH)** distributions model number of jumps until absorption.

### Graph Representation

**Vertices** = states, **Edges** = transitions with rates/probabilities

- **Parameterized edges**: weight depends on `weight_mode`:
  - `'linear'` (default): weight = c₁θ₁ + c₂θ₂ + ... + cₙθₙ
  - `'log'`: weight = (c₁θ₁)(c₂θ₂)···(cₙθₙ) (multiplicative, computed in log-space)
  - `'callback'`: weight = callback(θ, coefficients) (arbitrary Python function)
- **Graph elimination** (Algorithm 3): Converts cyclic → acyclic via Gaussian elimination on graph
- **Sparse graphs**: Only store actual transitions, not full n×n matrix

### Trace-Based Elimination (Phases 1-4)

**Phase 1-2**: Record elimination operations as linear trace
**Phase 3**: SVGD integration with JAX (jit/grad/vmap)
**Phase 4**: ✅ Exact phase-type likelihood using C forward algorithm
**Phase 5 Week 3**: ✅ Forward algorithm PDF gradients in C
**Phase 5 (continuation)**: (In progress) JAX FFI gradients for full autodiff support

**Advantage**: Record once (O(n³)), evaluate many times (O(n)) → 5-10x faster than symbolic DAG for SVGD workloads

### JAX Integration

**Pattern**: Serialize graph → C++ builder → `jax.pure_callback` → JAX compatible

Supports:
- `jax.jit`: JIT compilation
- `jax.grad`: Automatic differentiation (via custom VJP)
- `jax.vmap`: Batching over parameters
- `jax.pmap`: Multi-device parallelization

---

## Python API Patterns

### Building Graphs

```python
from phasic import Graph
import numpy as np

# Callback-based construction
def coalescent_callback(state):
    n = state[0]
    if n <= 1:
        return []
    rate = n * (n - 1) / 2
    # For parameterized: return (next_state, [coeff_vector])
    return [(np.array([n - 1]), [rate])]

g = Graph(
    state_length=1,
    callback=coalescent_callback,
    parameterized=True,  # Enable parameterized edges
    nr_samples=5
)

# Or manual construction
g = Graph(1)
v0 = g.starting_vertex()
v1 = g.find_or_create_vertex([1])
v0.add_edge(v1, [2.0, 0.5])
# Edge weight = 2.0*θ[0] + 0.5*θ[1]
```

### Weight Modes

Parameterized edges support three weight computation modes that flow through the full JAX/FFI/SVGD pipeline:

```python
# Linear (default): weight = Σ c_k θ_k
g.weight_mode = 'linear'

# Log/multiplicative: weight = Π(c_k θ_k)
# Useful when coefficients and parameters represent factors
g.weight_mode = 'log'

# Callback: weight = callback(theta, coefficients)
# For arbitrary non-linear weight functions
def resistance_weight(theta, coeffs):
    return np.exp(-np.dot(coeffs, theta))

g.weight_callback = resistance_weight  # Automatically sets weight_mode='callback'
```

All three modes work with `pmf_from_graph`, `pmf_and_moments_from_graph`, `graph.svgd()`, `jax.grad`, and `jax.vmap`. Log mode uses the C++ FFI path (fast, multi-core). Callback mode uses `jax.pure_callback` (sequential, Python overhead per evaluation).

### Computing PDF/PMF

```python
# Direct C++ call (fast, not JAX-differentiable)
pdf_value = graph.pdf(time, granularity=0)  # granularity=0 → auto
pmf_value = graph.dph_pmf(jumps)

# Vectorized
times = np.array([0.5, 1.0, 1.5])
pdf_values = graph.pdf(times, granularity=100)

# JAX-compatible (for gradients)
import jax.numpy as jnp
from phasic.ffi_wrappers import compute_pmf_ffi

structure_json = graph.serialize()
theta = jnp.array([1.0, 0.5])
times = jnp.linspace(0.1, 5.0, 100)
pdf = compute_pmf_ffi(structure_json, theta, times, discrete=False, granularity=100)

# Works with JAX transformations
jitted = jax.jit(compute_pmf_ffi, static_argnums=(0, 3, 4))
grad_fn = jax.grad(lambda t: jnp.sum(compute_pmf_ffi(structure_json, t, times, False, 100)))
```

### Trace Elimination Workflow

```python
from phasic.trace_elimination import (
    record_elimination_trace,
    evaluate_trace_jax,
    instantiate_from_trace,
    trace_to_log_likelihood
)

# 1. Record trace (once, ~ms for 67 vertices)
trace = record_elimination_trace(graph, param_length=2)

# 2. Evaluate with concrete parameters (fast, O(n))
result = evaluate_trace_jax(trace, jnp.array([1.0, 2.0]))
# Returns: {'vertex_rates': ..., 'edge_probs': ..., 'vertex_targets': ...}

# 3. Instantiate concrete graph from trace
concrete_graph = instantiate_from_trace(trace, np.array([1.0, 2.0]))
pdf = concrete_graph.pdf(times, granularity=0)

# 4. For SVGD: exact phase-type likelihood
observed_times = np.array([1.5, 2.3, 0.8])
log_lik = trace_to_log_likelihood(trace, observed_times, reward_vector=None, granularity=0)

# Use with SVGD
from phasic import SVGD
svgd = SVGD(log_lik, theta_dim=2, n_particles=100, n_iterations=1000)
results = svgd.fit()
```

### SVGD Inference

```python
# High-level SVGD API
from phasic import Graph

# Build parameterized model
model = Graph.pmf_from_graph(graph, discrete=False)

# Run SVGD
results = Graph.svgd(
    model=model,
    observed_data=observations,
    theta_dim=2,
    n_particles=100,
    n_iterations=1000,
    learning_rate=0.01
)

print(f"Posterior mean: {results['theta_mean']}")
print(f"Posterior std: {results['theta_std']}")
```

### Multivariate Phase-Type Models (2D Observations & Rewards)

**New in v0.21.4**: Support for multivariate phase-type distributions where each feature dimension has its own reward vector.

**Updated in v0.22.22**: Reward matrix shape changed to `(n_features, n_vertices)` for more intuitive indexing.

```python
from phasic import Graph
import jax.numpy as jnp

# Create parameterized graph
graph = Graph(model_callback)

# Create multivariate model
model = Graph.pmf_and_moments_from_graph_multivariate(
    graph,
    nr_moments=2,
    discrete=False
)

# Setup multivariate data
n_times = 100
n_features = 3  # e.g., 3 marginal distributions
n_vertices = graph.vertices_length()

# 2D observations: (n_times, n_features)
observed_data = jnp.array([
    [obs_feature_0, obs_feature_1, obs_feature_2],
    ...
])  # Shape: (100, 3)

# 2D rewards: (n_features, n_vertices) - UPDATED v0.22.22
# Each row defines the complete reward vector for one marginal
rewards_2d = jnp.array([
    [r0_feat0, r1_feat0, r2_feat0, ...],  # Feature 0 reward vector
    [r0_feat1, r1_feat1, r2_feat1, ...],  # Feature 1 reward vector
    [r0_feat2, r1_feat2, r2_feat2, ...],  # Feature 2 reward vector
])  # Shape: (3, n_vertices)

# Run SVGD with multivariate model
svgd_result = graph.svgd(
    observed_data=observed_data,
    theta_dim=2,
    n_particles=100,
    n_iterations=1000,
    rewards=rewards_2d  # Pass 2D rewards
)

# Or use SVGD directly
from phasic import SVGD
svgd = SVGD(
    model=model,
    observed_data=observed_data,
    theta_dim=2,
    n_particles=100,
    rewards=rewards_2d
)
svgd.optimize()
```

**Key Features:**
- **Independent computation**: Each feature dimension computed separately with its reward vector
- **Log-likelihood**: Sum over all observation elements: `Σᵢⱼ log(PMF[i,j])`
- **Moment regularization**: Moments aggregated across features (mean)
- **NaN handling**: Missing observations (NaN) skipped in likelihood; NaN PMF for valid obs raises error
- **Backward compatible**: 1D rewards work exactly as before

**Shape Convention (v0.22.22):**
- Rewards: `(n_features, n_vertices)` - each row is one feature's reward vector
- Observations: `(n_times, n_features)`
- PMF output: `(n_times, n_features)` for 2D rewards, `(n_times,)` for 1D
- Moments output: `(n_features, nr_moments)` for 2D rewards, `(nr_moments,)` for 1D

**Migration from v0.21.4:**
```python
# Old shape (v0.21.4)
rewards_old = jnp.ones((n_vertices, n_features))  # Column per feature

# New shape (v0.22.22)
rewards_new = rewards_old.T  # Transpose: row per feature
```

### Sparse Observation Format (NEW)

For multivariate models where different features have different numbers of observations, use `SparseObservations` instead of dense NaN-padded arrays. This avoids NaN propagation through JAX gradients.

```python
from phasic import SparseObservations, dense_to_sparse, SVGD

# Option 1: Convert from dense NaN-padded array
dense_obs = jnp.array([
    [1.0, np.nan, 3.0],   # Feature 0 has value, Feature 1 missing, Feature 2 has value
    [1.5, 2.0, np.nan],   # etc.
])
sparse_obs = dense_to_sparse(dense_obs)

# Option 2: Create directly with pre-computed slices
# (Required for JAX JIT compatibility)
sparse_obs = SparseObservations(
    values=jnp.array([1.0, 1.5, 2.0, 3.0]),  # All valid values, grouped by feature
    features=jnp.array([0, 0, 1, 2]),         # Feature index for each value
    n_features=3,
    slices=((0, 2), (2, 3), (3, 4))           # (start, end) indices per feature
)

# Use in SVGD
svgd = SVGD(
    model=model,
    observed_data=sparse_obs,  # Works just like dense observations
    theta_dim=2,
    rewards=rewards_2d
)
svgd.optimize()
```

**Key Benefits:**
- No NaN propagation through JAX callbacks
- Memory efficient for very sparse observation patterns
- Supports unequal observation counts per feature
- `dense_to_sparse()` automatically computes slices

### Reward Transformation

```python
# Transform for higher moments or multivariate distributions
rewards = np.array([1.0, 2.0, 0.5, ...])  # One per vertex
transformed_graph = graph.reward_transform(rewards)

# For k-variate phase-type (v0.22.22+ shape):
reward_matrix = np.array([
    [r1_1, r2_1, r3_1, ...],  # Marginal 1: reward vector across all vertices
    [r1_2, r2_2, r3_2, ...],  # Marginal 2: reward vector across all vertices
    ...
    [r1_k, r2_k, r3_k, ...],  # Marginal k: reward vector across all vertices
])
# Each row = one marginal's complete reward vector
```

### HexGrid Spatial Models

`HexGrid` creates hexagonal grids within geographic boundaries, producing `Property` objects that compose with `StateIndexer` for spatial phase-type models.

```python
from phasic import Graph, HexGrid, StateIndexer, Property

# Create grid from shapefile
grid = HexGrid.from_shapefile('africa.shp', hex_size=5)

# Grid properties compose with other state properties
indexer = StateIndexer(
    cell=grid.properties() + [Property('lineage', min_value=1, max_value=2)]
)

# Transition function receives grid as second argument
def migration(state, grid, indexer=None, rate=None):
    transitions = []
    # ... use grid.neighbors(row, col), indexer.cell.p2i(), etc.
    return transitions

# Build graph via standard Graph constructor
graph = Graph(migration, ipv=[(initial, 1)], grid=grid, indexer=indexer, rate=1.0)

# Or via HexGrid.build_graph for simpler models
graph = grid.build_graph(transition_fn, indexer.cell)

# Map results back to grid
grid_values = grid.map_to_grid(graph, indexer.cell, values)
```

Key methods: `grid.neighbors(r, c)`, `grid.coords_to_rowcol(x, y)`, `grid.valid_cells()`, `grid.properties()`.

---

## Important Implementation Details

### Graph Elimination (Algorithm 3)

**Purpose**: Convert cyclic graph → acyclic graph for moment computation

**Algorithm**: Gaussian elimination on graph structure
- Iterate through vertices in order
- For each vertex i being eliminated:
  - For each parent → child pair:
    - Add bypass edge (or update existing)
    - Renormalize probabilities
  - Remove edges to eliminated vertex

**Complexity**: O(n³) one-time, enables O(n²) moment computation

### Forward Algorithm (Algorithm 4)

**Purpose**: Compute exact PDF/PMF via uniformization

**Used by**: `graph.pdf(time, granularity)`, `graph.dph_pmf(jumps)`

**Algorithm** (continuous):
1. Discretize time using uniformization (granularity = 2 × max_rate by default)
2. Simulate discrete-time chain via dynamic programming
3. Track probability mass at each vertex over time
4. Sum mass reaching absorbing states

**Complexity**: O(t · n² · g) where t = time, n = vertices, g = granularity

**Note**: This is the **exact** phase-type PDF computation, not an approximation

### JAX FFI Integration Pattern

**Location**: `src/phasic/ffi_wrappers.py`

**Pattern**:
```python
def compute_pmf_fallback(structure_json, theta, times, discrete, granularity):
    # Wrap C++ call with jax.pure_callback
    result_shape = jax.ShapeDtypeStruct(times.shape, jnp.float64)

    return jax.pure_callback(
        lambda theta_jax, times_jax: _compute_pmf_impl(
            structure_json,
            np.asarray(theta_jax),
            np.asarray(times_jax),
            discrete,
            granularity
        ),
        result_shape,
        theta,
        times,
        vmap_method='sequential'  # Enable vmap
    )
```

**Key**: `pure_callback` allows JAX jit/vmap while calling C++ code

### Performance Characteristics

**Graph vs Matrix** (sparse systems):
- PDF computation: 10-100x faster
- Memory: 100-1000x less (O(n+m) vs O(n²))
- Scales to 500K+ states (vs 10K for matrices)

**Trace vs Symbolic DAG** (repeated evaluation):
- Setup: ~0.5x time (faster to record trace)
- Evaluation: 5-10x faster per parameter vector
- Break-even: ~6 evaluations
- Ideal for SVGD: 100-1000 evaluations

**Phase 3 Targets** (1000 SVGD evaluations):
- 37 vertices: <5 min ✓ (actual: ~2s)
- 67 vertices: <30 min ✓ (actual: ~5s)

---

## Phase 4: Exact Phase-Type Likelihood ✅ COMPLETE

**Status**: ✅ Implemented October 2025

Upgraded `trace_to_log_likelihood()` from exponential approximation to exact phase-type likelihood using forward algorithm (Algorithm 4).

**Key Changes**:
- Use `instantiate_from_trace()` + `graph.pdf()` for exact PDF computation
- Add `granularity` parameter for accuracy control (default=100)
- Reward vector support: falls back to exponential with warning (exact support planned)
- Performance: 4.7ms per evaluation for simple models, well under targets

**Accuracy Improvement**:
- Erlang(3) distribution: Difference of 1.33 in log-likelihood vs exponential
- Critical for multi-stage phase-type distributions
- Exact computation essential for correct Bayesian inference

**Usage**:
```python
trace = record_elimination_trace(graph, param_length=2)
observed_times = np.array([1.5, 2.3, 0.8, 1.2])

# Exact likelihood with granularity control
log_lik = trace_to_log_likelihood(trace, observed_times, granularity=100)

# Use with SVGD
from phasic import SVGD
svgd = SVGD(log_lik, theta_dim=2, n_particles=100, n_iterations=1000)
results = svgd.fit()
```

**Performance**:
- 5-10× slower than exponential approximation
- Still meets Phase 3 targets with margin
- 67-vertex model: ~50s for 1000 evaluations (target: <2 min)

---

## Phase 5 Week 3: Forward Algorithm PDF Gradients ✅ COMPLETE

**Status**: ✅ Implemented October 2025

Implemented machine-precision gradient computation for phase-type distribution PDFs using uniformization-based forward algorithm.

**Key Features**:
- Exact PDF and gradient computation with error ≤ 2.05e-16
- Two API workflows: direct parameter passing and integrated parameter update
- Zero API signature changes - full backward compatibility with C++/R/Python
- Minimal code changes for gradient support

**API Functions**:

1. **`ptd_graph_pdf_with_gradient()`** - Direct parameter passing:
```c
int ptd_graph_pdf_with_gradient(
    struct ptd_graph *graph,
    double time,
    size_t granularity,
    const double *params,
    size_t n_params,
    double *pdf_value,
    double *pdf_gradient
);
```

2. **`ptd_graph_pdf_parameterized()`** - Integrated workflow:
```c
// Set parameters first
ptd_graph_update_weight_parameterized(graph, theta, n_params);

// Compute PDF + gradients using stored parameters
int ptd_graph_pdf_parameterized(
    struct ptd_graph *graph,
    double time,
    size_t granularity,
    double *pdf_value,
    double *pdf_gradient  // NULL for PDF-only
);
```

**Architecture Solution**:
Preserves original edge coefficients for gradient computation while allowing `update_weight_parameterized()` to store concrete weights for fast PDF computation.

**Test Results**:
- `test_single_exp_grad.c`: error = 0.00e+00 ✓
- `test_c_pdf_parameterized.c`: error ≤ 2.05e-16 ✓

**Performance**: ~4-5ms per PDF+gradient evaluation for simple models

**Documentation**: See `PHASE5_WEEK3_SOLUTION.md` for complete implementation details

**Python API (via trace)**:
```python
from phasic.trace_elimination import (
    record_elimination_trace,
    instantiate_from_trace
)

# 1. Build parameterized graph
graph = Graph(1)
# ... add parameterized edges ...

# 2. Record trace
trace = record_elimination_trace(graph, param_length=1)

# 3. Instantiate with concrete parameters
theta = np.array([2.0])
concrete_graph = instantiate_from_trace(trace, theta)

# 4. Compute PDF (gradient via finite differences)
pdf = concrete_graph.pdf(time=1.0, granularity=100)
```

**Key Insights**:
- Uniformization relates discrete jumps to continuous time: `dt = 1/λ`
- PMF is instantaneous absorption probability (with Poisson weighting)
- PDF = PMF / dt = PMF * λ (NOT PMF * granularity)
- Zeroing absorbed probability is critical to avoid cumulation

**Performance**:
- Single PDF+gradient evaluation: ~4-5ms for small models
- Suitable for gradient-based optimization (SVGD, HMC, etc.)

**Next Steps (Phase 5 continuation)**:
- JAX FFI integration for full autodiff (Week 4)
- Extend to reward-transformed graphs (Week 5)
- Benchmark on larger models (100+ vertices)

---

## Quick Reference

### Logging

**phasic** provides a unified logging system that integrates Python and C/C++ code logging into a single consistent interface.

**Default Behavior**: Logging is configured at WARNING level by default, so only important messages are shown unless explicitly enabled.

**Environment Variables**:
```bash
# Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
export PHASIC_LOG_LEVEL=DEBUG

# Write logs to file (in addition to console)
export PHASIC_LOG_FILE=/path/to/logfile.log

# Force colored output on/off (auto-detected by default)
export PHASIC_LOG_COLOR=1  # or 0 to disable
```

**Python API**:
```python
from phasic.logging_config import set_log_level, get_logger

# Enable debug logging for entire package
set_log_level('DEBUG')

# Enable debug logging for specific module
set_log_level('DEBUG', module='trace_elimination')

# Get logger for your module
logger = get_logger(__name__)
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
```

**Logger Hierarchy**:
- `phasic` - Root logger for entire package
- `phasic.c` - All C/C++ code logs appear here
- `phasic.module_name` - Module-specific loggers (e.g., `phasic.trace_elimination`)

**Examples**:
```python
# Example 1: Debug cache operations
from phasic.logging_config import set_log_level
set_log_level('DEBUG')

# Now you'll see detailed logs about:
# - Cache hits/misses
# - Hash computation
# - Trace serialization/deserialization
# - Graph operations

# Example 2: Silence all logging
from phasic.logging_config import disable_logging
disable_logging()

# Example 3: View only errors
set_log_level('ERROR')
```

**C Logging** (for developers):
```c
#include "phasic_log.h"

PTD_LOG_DEBUG("Processing vertex %d with rate %f", v_idx, rate);
PTD_LOG_INFO("Cache hit for hash %s", hash_hex);
PTD_LOG_WARNING("Parameter out of range: %d", param_idx);
PTD_LOG_ERROR("Failed to allocate memory for %zu bytes", size);
```

**Key Features**:
- Thread-safe logging from C code
- Automatic integration of C logs into Python logging hierarchy
- Colored console output (when terminal supports it)
- Zero overhead when logging is disabled
- File and console output simultaneously

**Implementation Details**:
- Python: `src/phasic/logging_config.py` - Unified logging configuration
- C API: `src/c/phasic_log.h/c` - Thread-safe C logging with callback mechanism
- Bridge: `src/cpp/phasic_pybind.cpp` - pybind11 bridge connecting C to Python logging
- Strategic logging in: `phasic_hash.c` (hash computation), `trace_cache.c` (cache operations)

### Full API Documentation

- **C API**: See `api/c/phasic.h` (all C functions with comments)
- **C++ API**: See `api/cpp/phasiccpp.h` (object-oriented wrapper)
- **Python API**: Use `help(Graph)` or docstrings in code

### Key Files

**Core Implementation:**
- `src/c/phasic.c` - Core C algorithms
- `src/c/phasic_symbolic.c` - Symbolic expression system

**Python Modules:**
- `src/phasic/__init__.py` - Graph class, SVGD, main API
- `src/phasic/hex_grid.py` - HexGrid spatial grid utilities
- `src/phasic/trace_elimination.py` - Trace recording and evaluation (Phases 1-4)
- `src/phasic/ffi_wrappers.py` - JAX FFI integration (Phase 5 in progress)
- `src/phasic/svgd.py` - SVGD implementation

**Tests:**
- `tests/test_trace_recording.py` - Phase 1 tests
- `tests/test_trace_jax.py` - Phase 2 JAX integration tests
- `tests/test_trace_svgd_benchmark.py` - Phase 3 performance benchmarks
- `tests/test_trace_exact_likelihood.py` - Phase 4 exact likelihood tests

### Build and Install

```bash
# Development install
pip install -e .

# With JAX support
pip install -e .[jax]

# Using pixi (recommended)
pixi install
pixi run test
```

---

*Last updated: 2026-04-08*
- When creating a markdown file summarizing changes made, please prompt to add changed and new files to git, commit them with a message from the markdown file, but do not add the markdown file itself. Prompt only once and then do git add, commit, and push without prompting further.
