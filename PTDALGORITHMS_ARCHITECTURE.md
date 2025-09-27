# PtDAlgorithms Project Architecture

## Overview

PtDAlgorithms is a multi-language library for efficient phase-type distribution algorithms using graph-based approaches. The library provides unified APIs across Python, R, and C/C++, with a core C implementation and language-specific bindings.

## Project Structure

```
PtDalgorithms/
├── api/                    # Public API headers
│   ├── c/                  # C API (ptdalgorithms.h)
│   └── cpp/                # C++ API (ptdalgorithmscpp.h)
├── src/                    # Source implementations
│   ├── c/                  # Core C implementation
│   ├── cpp/                # C++ wrapper and Python bindings
│   ├── ptdalgorithms/      # Python package
│   ├── Makevars*           # R compilation configuration
│   └── RcppExports.cpp     # R bindings (auto-generated)
├── R/                      # R package files
├── jax_extension/          # JAX integration with separated architecture
├── docs/                   # Documentation and examples
├── tests/                  # Test suites for all languages
└── build configuration files (CMakeLists.txt, pyproject.toml, DESCRIPTION)
```

## Component Architecture

### 1. Core C Layer (`src/c/`, `api/c/`)

**Purpose**: Provides the foundational graph algorithms and data structures for phase-type distributions.

**Key Components**:
- **Graph structures**: `ptd_graph`, `ptd_vertex`, `ptd_edge`
- **AVL tree**: `ptd_avl_tree` for efficient vertex lookup by state vectors
- **Algorithms**: Normalization, reward transforms, probability computations
- **Distribution contexts**: For iterative probability calculations

**Key Files**:
- `api/c/ptdalgorithms.h`: Complete C API (422 lines)
- `src/c/ptdalgorithms.c`: Core implementation
- `src/c/ptdalgorithms.h`: Internal C header

**Data Structures**:
```c
struct ptd_graph {
    size_t vertices_length;
    struct ptd_vertex **vertices;
    struct ptd_vertex *starting_vertex;
    size_t state_length;
    bool parameterized;
    // ... reward computation graphs
};

struct ptd_vertex {
    size_t edges_length;
    struct ptd_edge **edges;
    struct ptd_graph *graph;
    size_t index;
    int *state;  // State vector
};

struct ptd_edge {
    struct ptd_vertex *to;
    double weight;
    bool parameterized;
};
```

### 2. C++ Wrapper Layer (`src/cpp/`, `api/cpp/`)

**Purpose**: Provides modern C++ interfaces with RAII, STL integration, and exception handling.

**Key Components**:
- **Graph class**: Reference-counted wrapper around C graph
- **Vertex, Edge classes**: Type-safe wrappers
- **STL integration**: Vector-based interfaces for states and rewards
- **Distribution contexts**: C++ wrappers for probability calculations

**Key Files**:
- `api/cpp/ptdalgorithmscpp.h`: C++ API (1146 lines)
- `src/cpp/ptdalgorithmscpp.cpp`: C++ implementation
- `src/cpp/ptdalgorithmscpp.h`: Internal C++ header

**Key Classes**:
```cpp
namespace ptdalgorithms {
    class Graph {
        // Reference-counted wrapper around ptd_graph
        std::vector<double> expected_waiting_time(std::vector<double> rewards);
        Vertex create_vertex(std::vector<int> state);
        Graph reward_transform(std::vector<double> rewards);
        // ... many more methods
    };
    
    class Vertex {
        void add_edge(Vertex &to, double weight);
        std::vector<int> state();
        // ...
    };
}
```

### 3. Python Bindings (`src/cpp/ptdalgorithmscpp_pybind.cpp`)

**Purpose**: Provides Python access via pybind11 with NumPy integration and Pythonic APIs.

**Key Components**:
- **pybind11 bindings**: Wraps C++ classes for Python
- **NumPy integration**: Eigen matrices for efficient array operations
- **Python Graph class**: Extends C++ Graph with Python-specific methods
- **JAX integration**: Experimental JAX custom operations

**Key Files**:
- `src/cpp/ptdalgorithmscpp_pybind.cpp`: pybind11 bindings
- `src/ptdalgorithms/__init__.py`: Python API and extensions
- `src/ptdalgorithms/plot.py`: Visualization utilities

**Python Extensions**:
```python
class Graph(_Graph):
    def __init__(self, state_length=None, callback=None, **kwargs):
        # Constructor supporting callback-based graph construction

    def discretize(self, reward_rate, skip_states=[], skip_slots=[]):
        # Creates discrete distribution from continuous one

    def plot(self, *args, **kwargs):
        # Graphviz-based visualization

# JAX Integration (via separated_graph_python.py)
def compute_pmf_jax(theta, config):
    # JAX-compatible PMF computation with gradients
    return pmf_values
```

### 4. R Bindings (`src/RcppExports.cpp`, `R/`)

**Purpose**: Provides R access via Rcpp with R-native data structures.

**Key Components**:
- **Rcpp bindings**: Auto-generated from C++ annotations
- **R data types**: Integration with R vectors, lists, and data frames
- **R package structure**: Standard R package with documentation

**Key Files**:
- `src/RcppExports.cpp`: Auto-generated Rcpp bindings
- `R/package.r`: R package namespace definitions
- `DESCRIPTION`: R package metadata
- `src/Makevars`: R compilation configuration

**Build Process**:
- Uses Rcpp's `compileAttributes()` to generate bindings
- Links against C++ layer via standard R package mechanisms
- Configured via `src/Makevars` for include paths

### 5. JAX Extension (`jax_extension/`)

**Purpose**: Production-ready JAX integration for GPU-accelerated computations with automatic differentiation.

**Architecture**: Separated concerns - user-defined C++ graph builders with JAX-compatible PMF computation

**Key Components**:
- **User Graph API**: C++ interface for custom graph construction
- **Separated PMF Engine**: Decoupled computation engine
- **JAX Integration**: JIT compilation and gradient support
- **Batching**: Support for SVGD and multi-chain inference
- **HDF5 Caching**: Precomputed graph serialization

**Key Files**:
- `jax_extension/user_graph_api.h/cpp`: User-implementable C++ API
- `jax_extension/separated_graph_pmf.cpp`: Core PMF computation engine
- `jax_extension/separated_graph_python.py`: Python-JAX bridge
- `jax_extension/test_graph_api.cpp`: C++ API tests
- `jax_extension/example_coalescent.py`: Real-world usage example
- `jax_extension/SEPARATED_ARCHITECTURE.md`: Detailed architecture docs

**User Graph API Example**:
```cpp
Graph build_my_graph(const double* theta, int theta_size, const UserConfig& config) {
    Graph graph;
    double pop_size = theta[0];

    // Domain-specific graph construction
    int vertex_id = graph.add_vertex({config.nr_samples});
    graph.add_edge(vertex_id, target_vertex, rate);
    graph.set_absorption_rate(terminal_vertex, 1.0);

    return graph;
}
```

## Build Systems

### 1. C/C++ Core (CMake)

**Configuration**: `CMakeLists.txt`

**Targets**:
- `libptdalgorithms`: C shared library
- `libptdalgorithmscpp`: C++ shared library  
- `ptdalgorithmscpp_pybind`: Python extension module

**Dependencies**:
- Eigen3 (linear algebra)
- pybind11 (Python bindings)

**Build Process**:
```bash
cmake . && make
```

### 2. Python Package (scikit-build-core)

**Configuration**: `pyproject.toml`

**Build System**: scikit-build-core with CMake backend

**Dependencies**:
- pybind11 >= 2.10.0
- eigen
- graphviz, numpy, seaborn (runtime)

**Build Process**:
```bash
pip install -e .          # Editable install
python -m build           # Distribution build
```

**Integration**: Uses CMake configuration to build extension modules

### 3. R Package (Standard R)

**Configuration**: `DESCRIPTION`

**Build System**: Standard R package with Rcpp

**Dependencies**:
- Rcpp (build and runtime)
- R >= 2.0.0

**Build Process**:
```bash
R CMD INSTALL .           # Standard install
devtools::install()       # Development install
```

**Compilation**: Configured via `src/Makevars` with C++ flags

### 4. JAX Extension (Custom Makefile)

**Configuration**: `jax_extension/Makefile`

**Build System**: Custom Makefile with specific flags

**Dependencies**:
- HDF5 libraries
- C++17 compiler
- Boost (via pixi environment)
- Eigen3 (matrix operations)

**Build Process**:
```bash
cd jax_extension && make           # Build shared library
./test_graph_api                   # Run C++ tests
python simple_python_test.py       # Run Python integration tests
```

**Integration with JAX**:
- Compiles to shared library loaded by ctypes
- Provides JAX custom operations via `jax.pure_callback`
- Supports automatic differentiation through finite differences
- Enables batched computation for ensemble methods

## Language Integration Patterns

### State Representation
- **C**: `int *state` arrays
- **C++**: `std::vector<int> state`
- **Python**: Python lists/NumPy arrays
- **R**: Integer vectors

### Memory Management
- **C**: Manual allocation/deallocation
- **C++**: Reference counting with RAII
- **Python**: Automatic via pybind11
- **R**: Automatic via Rcpp

### Error Handling
- **C**: Error codes and global error buffer (`ptd_err`)
- **C++**: C++ exceptions (`std::runtime_error`)
- **Python**: Python exceptions (translated from C++)
- **R**: R errors (translated from C++)

### API Consistency
All language interfaces provide equivalent functionality:
- Graph construction and manipulation
- Vertex and edge operations  
- Probability computations (PDF/CDF/PMF)
- Reward transformations
- Random sampling
- Normalization

## Data Flow

### Standard Flow
1. **Graph Construction**: User creates graph in preferred language
2. **State Management**: AVL tree maintains state-to-vertex mapping
3. **Algorithm Execution**: Core C algorithms process graph structure
4. **Result Translation**: Results converted to appropriate language types
5. **Memory Cleanup**: Language-specific cleanup (RAII, GC, reference counting)

### JAX Extension Flow
1. **Registration**: User registers C++ graph builder at compile time
2. **JAX Call**: Python passes parameters to JAX-wrapped function
3. **Graph Build**: C++ constructs graph from parameters
4. **PMF Computation**: Separated engine computes probabilities
5. **Gradient**: JAX computes gradients via automatic differentiation
6. **Batching**: Multiple parameter sets processed in parallel

## Development Workflow

### Dependencies
- **Core**: CMake, C/C++ compiler, Eigen
- **Python**: pybind11, NumPy, scikit-build-core
- **R**: Rcpp, devtools
- **JAX**: HDF5, Boost
- **Environment**: Pixi for unified dependency management (pixi.toml)

### Testing
- **Python**: pytest in `tests/`
- **R**: testthat in `tests/testthat/`
- **C++**: Basic tests in `test/`

### Documentation  
- **API docs**: Quarto-based documentation in `docs/`
- **Examples**: Multi-language examples in `docs/examples/`
- **Notebooks**: Jupyter notebooks in `docs/pages/`

This architecture provides a robust, multi-language scientific computing library with consistent APIs while leveraging each language's strengths and ecosystems.