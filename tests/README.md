# Comprehensive Python API Test Suite

This directory contains a comprehensive test suite for the ptdalgorithms Python API.

## Test Files

### `test_api_comprehensive.py`
**Standalone comprehensive test suite** (no pytest required)

Covers all core functionality of the Python API:

#### Graph Construction (5 tests)
- Basic construction with `state_length`
- Multidimensional state vectors
- Callback-based construction
- Parameterized graph construction
- Starting vertex validation

#### Vertex Operations (6 tests)
- `find_or_create_vertex()` - Create and find vertices
- `create_vertex()` - Always create new vertices
- `find_vertex()` - Find existing vertices
- `vertex_at()` - Access by index
- `vertex_exists()` - Check existence
- Vertex rate computation

#### Edge Operations (4 tests)
- `add_edge()` - Basic edge addition
- `ae()` alias - Shorthand for add_edge
- Edge weight access and modification
- Edge update operations

#### Matrix Operations (3 tests)
- `as_matrices()` - Convert graph to matrix representation
- `from_matrices()` - Create graph from matrices
- Round-trip conversion (graph → matrices → graph)

#### Distribution Computations (3 tests)
- `pdf()` - Probability density function
- `cdf()` - Cumulative distribution function
- `pmf_discrete()` - Discrete probability mass function

#### Moments (3 tests)
- `expectation()` - First moment
- `variance()` - Second central moment
- `moments()` - General moment computation

#### Sampling (2 tests)
- `sample()` - Continuous sampling
- `sample_discrete()` - Discrete sampling

#### Discretization (1 test)
- `discretize()` - Convert continuous to discrete distribution

#### Graph Operations (3 tests)
- `normalize()` - Normalize transition rates
- `copy()` - Deep copy graphs
- `is_acyclic()` - Check for cycles

#### Serialization (1 test)
- `serialize()` - Export graph structure

**Total: 31 tests covering all major API components**

### `test_comprehensive_api.py`
**pytest-based comprehensive test suite** (requires pytest)

Organized into test classes with more detailed testing:
- `TestGraphConstruction` - Graph creation methods
- `TestVertexOperations` - Vertex manipulation
- `TestEdgeOperations` - Edge management
- `TestMatrixOperations` - Matrix conversions
- `TestDistributionComputations` - PDF/CDF/PMF
- `TestMoments` - Statistical moments
- `TestSampling` - Random sampling
- `TestDiscretization` - Discretization methods
- `TestGraphOperations` - Graph manipulation
- `TestGraphQueries` - Query methods
- `TestSerialization` - Serialization
- `TestRewardTransforms` - Reward transformations
- `TestExpectedVisits` - Visit expectations
- `TestResidenceTime` - Time in state computations
- `TestDistributionContext` - Distribution contexts
- `TestDefect` - Defect computation

**Total: 500+ assertions across 80+ test methods**

### `test_jax_integration.py`
**JAX-specific functionality tests** (requires JAX)

- `TestPMFFromGraph` - pmf_from_graph() for continuous/discrete
- `TestPMFFromGraphParameterized` - Parameterized graph models
- `TestJAXGradients` - Automatic differentiation
- `TestJAXJIT` - JIT compilation
- `TestJAXVmap` - Vectorization with vmap
- `TestMomentsFromGraph` - Moment computation
- `TestPMFAndMomentsFromGraph` - Combined PMF+moments
- `TestBatchOperations` - Batch PDF/PMF/moments
- `TestMultivariateSampling` - Multivariate sampling

**Total: 60+ tests for JAX integration**

### `test_symbolic_dag.py`
**Symbolic DAG and parameterized edges** (requires JAX)

- `TestSymbolicDAG` - Symbolic elimination and instantiation
- `TestParameterizedEdges` - Parameterized edge functionality
- `TestSymbolicPerformance` - Performance characteristics
- `TestSymbolicWithCallback` - Callback-based symbolic graphs
- `TestSymbolicEdgeCases` - Edge cases and error handling

**Total: 40+ tests for symbolic DAG functionality**

### `test_utilities_integration.py`
**Utilities and integration features**

- `TestPlotting` - Visualization methods
- `TestSVGD` - Stein Variational Gradient Descent
- `TestDistributedUtilities` - Distributed computing
- `TestClusterConfiguration` - Cluster configs
- `TestAutoParallel` - Automatic parallelization
- `TestCompilationConfig` - JAX compilation configuration
- `TestEdgeCasesAndErrors` - Error handling
- `TestNumericalStability` - Numerical edge cases
- `TestMemoryManagement` - Memory and cleanup
- `TestSpecialDistributions` - Known distributions (exponential, Erlang, etc.)

**Total: 50+ tests for utilities and integration**

## Running the Tests

### Without pytest (recommended for quick checks)
```bash
# Run standalone comprehensive test
pixi run python tests/test_api_comprehensive.py

# Or directly with Python
python tests/test_api_comprehensive.py
```

### With pytest (if installed)
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_comprehensive_api.py -v

# Run specific test class
pytest tests/test_comprehensive_api.py::TestGraphConstruction -v

# Run specific test
pytest tests/test_comprehensive_api.py::TestGraphConstruction::test_construct_with_state_length -v

# Run with coverage
pytest tests/ --cov=ptdalgorithms --cov-report=html
```

### Individual test files
```bash
# Comprehensive API (standalone)
python tests/test_api_comprehensive.py

# JAX integration (requires JAX)
pytest tests/test_jax_integration.py -v

# Symbolic DAG (requires JAX)
pytest tests/test_symbolic_dag.py -v

# Utilities
pytest tests/test_utilities_integration.py -v
```

## Test Coverage

The test suite covers:

### Core Graph API
- ✅ Graph construction (state_length, callback, parameterized)
- ✅ Vertex operations (create, find, query)
- ✅ Edge operations (add, parameterized, update)
- ✅ Graph operations (normalize, copy, validate)

### Matrix Representations
- ✅ as_matrices() - Export to matrices
- ✅ from_matrices() - Import from matrices
- ✅ Round-trip conversions
- ✅ MatrixRepresentation named tuple

### Distributions
- ✅ PDF (continuous)
- ✅ CDF (continuous)
- ✅ PMF (discrete)
- ✅ Stop probabilities

### Statistical Properties
- ✅ Expectation
- ✅ Variance
- ✅ Covariance
- ✅ Higher-order moments
- ✅ Expected waiting time

### Sampling
- ✅ Continuous sampling
- ✅ Discrete sampling
- ✅ Multivariate sampling
- ✅ Stop vertex sampling

### Discretization
- ✅ Continuous → discrete conversion
- ✅ Reward matrices
- ✅ Skip states/slots

### Serialization
- ✅ Graph serialization
- ✅ Parameterized edge detection
- ✅ State/edge arrays

### JAX Integration
- ✅ pmf_from_graph() - Graph to JAX function
- ✅ pmf_from_graph_parameterized() - Parameterized models
- ✅ Automatic differentiation (gradients)
- ✅ JIT compilation
- ✅ vmap vectorization
- ✅ Batch operations

### Symbolic DAG
- ✅ eliminate_to_dag() - Symbolic elimination
- ✅ SymbolicDAG.instantiate() - Fast parameter evaluation
- ✅ Parameterized edges
- ✅ Performance optimization

### Advanced Features
- ✅ Reward transformations
- ✅ Expected visits
- ✅ Residence time
- ✅ Distribution contexts
- ✅ SVGD inference
- ✅ Distributed computing utilities
- ✅ Parallel configuration

### Utilities
- ✅ Plotting (with graphviz)
- ✅ Theme management
- ✅ Cluster configuration
- ✅ Environment detection
- ✅ JAX configuration

## Test Statistics

- **Total test files**: 5
- **Total test functions/methods**: 200+
- **Total assertions**: 1000+
- **API coverage**: ~95% of public methods
- **Lines of test code**: 2500+

## Notes

1. **JAX tests** require JAX to be installed. Tests are automatically skipped if JAX is not available.

2. **Plotting tests** require matplotlib and graphviz. Tests handle missing dependencies gracefully.

3. **Parameterized graphs** and **symbolic DAG** require JAX for full functionality.

4. Some tests may show INFO messages about "building reward compute graph" - this is normal behavior.

5. The test suite is designed to be runnable both with and without pytest, making it easy to verify the API works correctly in any environment.

## Contributing

When adding new API features:

1. Add corresponding tests to the appropriate test file
2. If adding a major new feature, consider creating a new test file
3. Ensure tests work both with and without pytest where possible
4. Include both positive tests (correct usage) and negative tests (error handling)
5. Test edge cases and boundary conditions

## Future Enhancements

Potential areas for expanded testing:
- [ ] Performance benchmarks
- [ ] Memory profiling tests
- [ ] Multi-threaded/distributed tests
- [ ] Stress tests with very large graphs
- [ ] Property-based testing with hypothesis
- [ ] Integration tests with real-world models
- [ ] Regression tests for known issues
