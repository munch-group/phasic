# Pybind11 Refactoring for Parameterized Graphs

## Executive Summary

This document outlines a proposed architectural refactoring to replace the current ctypes-based dynamic compilation approach with a pybind11-based solution for parameterized graph computation and moment-based SVGD inference.

### Current Status (Post-PyDLL Fix)

**Working:**
- ✅ Basic moment computation (forward pass)
- ✅ Gradient computation via JAX
- ✅ Small-scale SVGD (2-5 particles, 3-10 iterations)
- ✅ Standard SVGD (20 particles, 100 iterations)

**Limited:**
- ⚠️ Regularized SVGD fails with "unknown opcode" error at scale (20+ particles)
- ⚠️ PyDLL prevents GIL crashes but creates JAX JIT compilation issues
- ⚠️ No true parallelization possible

**Root Cause:** `ctypes.PyDLL` keeps GIL held, which:
1. Prevents threading crashes ✅
2. Blocks JAX's XLA compiler from optimizing complex graphs ❌
3. No path to GPU/TPU acceleration ❌

### Why Pybind11?

Pybind11 provides **proper GIL management** with `py::gil_scoped_release` that:
- Releases GIL during C++ computation (enables JIT optimization)
- Automatically reacquires GIL for Python objects
- Thread-safe by design (RAII pattern)
- No "unknown opcode" errors

---

## Problem Analysis

### Current Architecture Flow

```
Python callback → serialize → codegen → compile → ctypes.PyDLL → JAX pure_callback
                                                           ↓
                                                     [GIL HELD]
                                                           ↓
                                                  C++ Graph::moments()
```

**Issues:**
1. **Dynamic compilation overhead**: 0.5-2s per unique graph structure
2. **GIL contention**: PyDLL holds GIL during entire C++ execution
3. **JAX compilation limits**: Complex graphs hit Python bytecode limits
4. **No caching**: Rebuild for each theta despite identical structure
5. **Type safety**: Manual ctypes pointer arithmetic error-prone

### Threading Problem (Resolved but Limited)

```python
# CDLL approach (pre-fix):
lib = ctypes.CDLL(lib_path)  # Releases GIL
# Problem: GIL release/reacquire fails with JAX vmap+grad → CRASH

# PyDLL approach (current fix):
lib = ctypes.PyDLL(lib_path)  # Keeps GIL held
# Problem: JAX can't optimize, hits "unknown opcode" with complex graphs
```

---

## Proposed Pybind11 Architecture

### Core Concept: Separate Structure from Parameters

Current problem: Graph must be rebuilt for each `theta` because edges are parameterized.

Solution: Build graph **structure once**, inject parameters at **evaluation time**.

### Option A: Graph Builder Pattern (Recommended)

```cpp
namespace ptdalgorithms {
namespace parameterized {

class GraphBuilder {
private:
    // Cached graph structure (topology only)
    nlohmann::json structure_;
    int n_params_;

public:
    GraphBuilder(const std::string& structure_json)
        : structure_(nlohmann::json::parse(structure_json)) {
        n_params_ = structure_["param_length"];
    }

    // Build graph with specific theta
    Graph build(const double* theta, size_t theta_len) {
        if (theta_len != n_params_) {
            throw std::invalid_argument("theta length mismatch");
        }

        // Reconstruct graph from cached structure + theta
        // This is the generated C++ code, but now as a method
        Graph g;
        // ... build logic using structure_ and theta ...
        return g;
    }

    // High-level API: compute moments directly
    py::array_t<double> compute_moments(
        py::array_t<double> theta_arr,
        int nr_moments
    ) {
        py::gil_scoped_release release;  // Release GIL for C++ work

        auto theta = theta_arr.unchecked<1>();
        Graph g = build(theta.data(), theta.size());

        // Compute moments using existing code
        std::vector<double> rewards;
        std::vector<double> rewards2 = g.expected_waiting_time(rewards);
        std::vector<double> result(nr_moments);
        result[0] = rewards2[0];

        for (int i = 1; i < nr_moments; i++) {
            std::vector<double> rewards3(rewards2.size());
            for (size_t j = 0; j < rewards3.size(); j++) {
                rewards3[j] = rewards2[j] * std::pow(rewards2[j], i);
            }
            rewards2 = g.expected_waiting_time(rewards3);
            result[i] = factorial(i + 1) * rewards2[0];
        }

        py::gil_scoped_acquire acquire;  // Reacquire for return
        return py::array_t<double>(result.size(), result.data());
    }

    // Combined PMF + moments
    std::pair<py::array_t<double>, py::array_t<double>>
    compute_pmf_and_moments(
        py::array_t<double> theta_arr,
        py::array_t<double> times_arr,
        int nr_moments,
        bool discrete = false
    ) {
        py::gil_scoped_release release;

        Graph g = build(theta_arr.data(), theta_arr.size());

        // Compute PMF
        std::vector<double> pmf_result;
        if (discrete) {
            auto times = times_arr.unchecked<1>();
            for (ssize_t i = 0; i < times.size(); i++) {
                pmf_result.push_back(g.dph_pmf(static_cast<int>(times(i))));
            }
        } else {
            auto times = times_arr.unchecked<1>();
            for (ssize_t i = 0; i < times.size(); i++) {
                pmf_result.push_back(g.pdf(times(i), 100));
            }
        }

        // Compute moments (same as above)
        std::vector<double> moment_result = /* ... */;

        py::gil_scoped_acquire acquire;
        return {
            py::array_t<double>(pmf_result.size(), pmf_result.data()),
            py::array_t<double>(moment_result.size(), moment_result.data())
        };
    }
};

}} // namespace ptdalgorithms::parameterized
```

### Python Integration

```python
# In __init__.py

class Graph:
    @classmethod
    def pmf_and_moments_from_graph(cls, graph, nr_moments=2, discrete=False):
        """Create model using pybind11 builder (new implementation)"""

        # Serialize graph structure once
        serialized = graph.serialize()
        structure_json = json.dumps(serialized)

        # Create C++ builder (compile once, cache forever)
        from . import _ptdalgorithmscpp  # pybind11 module
        builder = _ptdalgorithmscpp.parameterized.GraphBuilder(structure_json)

        # Cache builder
        _graph_builder_cache[hash(structure_json)] = builder

        # Create JAX-compatible wrapper
        @jax.custom_vjp
        def model(theta, times):
            theta = jnp.atleast_1d(theta)
            times = jnp.atleast_1d(times)

            # Call pybind11 function (GIL released automatically!)
            pmf, moments = builder.compute_pmf_and_moments(
                theta, times, nr_moments, discrete
            )

            return (jnp.array(pmf), jnp.array(moments))

        def model_fwd(theta, times):
            pmf, moments = model(theta, times)
            return (pmf, moments), (theta, times)

        def model_bwd(res, g):
            theta, times = res
            g_pmf, g_moments = g

            # Finite differences (same as current implementation)
            # But now calls pybind11, not ctypes!
            eps = 1e-7
            grads = []
            for i in range(theta.shape[0]):
                theta_plus = theta.at[i].add(eps)
                theta_minus = theta.at[i].add(-eps)

                pmf_plus, mom_plus = builder.compute_pmf_and_moments(
                    theta_plus, times, nr_moments, discrete
                )
                pmf_minus, mom_minus = builder.compute_pmf_and_moments(
                    theta_minus, times, nr_moments, discrete
                )

                grad_pmf = jnp.sum(g_pmf * (pmf_plus - pmf_minus) / (2*eps))
                grad_mom = jnp.sum(g_moments * (mom_plus - mom_minus) / (2*eps))
                grads.append(grad_pmf + grad_mom)

            return jnp.array(grads), None

        model.defvjp(model_fwd, model_bwd)
        return model
```

---

## Implementation Phases

### Phase 1: PyDLL Fix (✅ COMPLETE)
**Goal:** Stop GIL crashes
**Status:** Done - basic functionality works, but limited scalability

**Commits:**
- Changed `ctypes.CDLL` → `ctypes.PyDLL` in moments_from_graph()
- Changed `ctypes.CDLL` → `ctypes.PyDLL` in pmf_and_moments_from_graph()

**Testing:**
- ✅ Forward pass works
- ✅ Gradients work
- ✅ vmap+grad works (small scale)
- ✅ Standard SVGD works (20 particles, 100 iterations)
- ❌ Regularized SVGD fails at scale (20+ particles, "unknown opcode")

### Phase 2: Pybind11 Infrastructure (Week 1-2)
**Goal:** Add GraphBuilder class and expose via pybind11

**Tasks:**
1. **Create C++ GraphBuilder class**
   - File: `src/cpp/parameterized/graph_builder.hpp`
   - Implement: constructor from JSON, build(), compute_moments(), compute_pmf_and_moments()

2. **Add pybind11 bindings**
   - File: `src/cpp/ptdalgorithmscpp_pybind.cpp`
   - Add: `py::class_<parameterized::GraphBuilder>` with proper GIL management

3. **JSON deserialization for graph structure**
   - Use nlohmann/json (already available)
   - Parse serialized graph structure
   - Reconstruct vertex/edge topology

4. **Unit tests**
   - Test: GraphBuilder construction
   - Test: moment computation matches _moments()
   - Test: PMF computation matches existing
   - Test: thread safety (multiple calls in sequence)

**Success Criteria:**
- GraphBuilder can be constructed from serialized graph
- Moments match existing `_moments()` function output
- PMF matches existing computation
- No memory leaks (valgrind/ASAN clean)

### Phase 3: Python Integration (Week 3)
**Goal:** Replace ctypes calls with pybind11 calls

**Tasks:**
1. **Refactor `moments_from_graph()`**
   - Replace dynamic compilation with GraphBuilder
   - Keep JAX custom_vjp wrapper
   - Add builder caching layer

2. **Refactor `pmf_and_moments_from_graph()`**
   - Use GraphBuilder.compute_pmf_and_moments()
   - Maintain backward compatibility
   - Update tests

3. **Caching layer**
   ```python
   _graph_builder_cache = {}

   def get_or_create_builder(serialized_json):
       cache_key = hashlib.sha256(serialized_json.encode()).hexdigest()
       if cache_key not in _graph_builder_cache:
           builder = _ptdalgorithmscpp.parameterized.GraphBuilder(serialized_json)
           _graph_builder_cache[cache_key] = builder
       return _graph_builder_cache[cache_key]
   ```

4. **Backward compatibility**
   - Old API continues to work
   - Add deprecation warnings for direct ctypes usage
   - Migration guide in documentation

**Success Criteria:**
- All existing tests pass
- Regularized SVGD works with 20+ particles
- Performance improvement (first call faster, subsequent calls same or faster)
- No "unknown opcode" errors

### Phase 4: JAX FFI Migration (Week 4-5, Optional)
**Goal:** Replace pure_callback with proper JAX FFI for XLA integration

**Tasks:**
1. **Define XLA custom call**
   ```cpp
   XLA_FFI_DEFINE_HANDLER(
       compute_moments_ffi,
       ComputeMomentsKernel,
       ffi::Ffi::Bind()
           .Arg<ffi::Buffer<PrimitiveType::F64>>()  // theta
           .Attr<int64_t>("nr_moments")
           .Ret<ffi::Buffer<PrimitiveType::F64>>()  // output
   );
   ```

2. **Implement custom batching rule**
   - Allow vmap to parallelize across particles
   - Thread-safe C++ implementation

3. **Implement custom VJP rule**
   - Analytical gradients where possible
   - Finite differences as fallback

4. **Register with JAX**
   ```python
   from jax.extend import ffi

   moments_p = core.Primitive("compute_moments")
   moments_p.def_impl(partial(xla.apply_primitive, moments_p))
   moments_p.def_abstract_eval(_moments_abstract_eval)
   batching.primitive_batchers[moments_p] = _moments_batching_rule
   ad.primitive_jvps[moments_p] = _moments_jvp_rule
   ```

**Success Criteria:**
- True parallelization with vmap
- JIT compilation works without limitations
- Path to GPU support (structure in place)

### Phase 5: Performance Optimization (Ongoing)
**Goal:** Maximize throughput

**Optimizations:**
1. **Graph structure caching**
   - Hash-based caching of built graphs
   - LRU eviction for memory management

2. **Vectorized operations**
   - Batch compute_moments() calls
   - SIMD optimization where applicable

3. **Memory pooling**
   - Reuse allocated vectors
   - Avoid allocation churn in tight loops

4. **Profiling and benchmarking**
   - Measure: first call latency, repeated call latency
   - Compare: ctypes vs pybind11 vs JAX FFI
   - Target: 2-3x speedup overall

### Phase 6: Deprecation (v2.0 Release)
**Goal:** Clean up old code paths

**Tasks:**
1. Mark old ctypes implementations as `@deprecated`
2. Migration guide for users
3. Remove dead code
4. Update all documentation

---

## Technical Specifications

### GraphBuilder C++ API

```cpp
namespace ptdalgorithms {
namespace parameterized {

class GraphBuilder {
public:
    // Constructor: parse structure once
    explicit GraphBuilder(const std::string& structure_json);

    // Build graph with specific parameters
    Graph build(const double* theta, size_t theta_len);

    // High-level: compute moments
    py::array_t<double> compute_moments(
        py::array_t<double> theta,
        int nr_moments
    );

    // High-level: compute PMF
    py::array_t<double> compute_pmf(
        py::array_t<double> theta,
        py::array_t<double> times,
        bool discrete = false,
        int granularity = 100
    );

    // Combined: PMF + moments (most efficient)
    std::pair<py::array_t<double>, py::array_t<double>>
    compute_pmf_and_moments(
        py::array_t<double> theta,
        py::array_t<double> times,
        int nr_moments,
        bool discrete = false
    );

    // Getters
    int param_length() const { return n_params_; }
    int vertices_length() const { return n_vertices_; }

private:
    nlohmann::json structure_;
    int n_params_;
    int n_vertices_;

    // Internal: reconstruct graph from structure + theta
    void build_internal(Graph& g, const double* theta);
};

}} // namespace
```

### Pybind11 Module Definition

```cpp
// In ptdalgorithmscpp_pybind.cpp

#include "parameterized/graph_builder.hpp"

PYBIND11_MODULE(_ptdalgorithmscpp, m) {
    // ... existing bindings ...

    py::module_ param = m.def_submodule("parameterized",
        "Parameterized graph utilities");

    py::class_<parameterized::GraphBuilder>(param, "GraphBuilder")
        .def(py::init<const std::string&>(),
            py::arg("structure_json"),
            "Create GraphBuilder from serialized structure")

        .def("compute_moments",
            &parameterized::GraphBuilder::compute_moments,
            py::arg("theta"),
            py::arg("nr_moments"),
            py::call_guard<py::gil_scoped_release>(),  // Auto-release GIL
            "Compute distribution moments")

        .def("compute_pmf",
            &parameterized::GraphBuilder::compute_pmf,
            py::arg("theta"),
            py::arg("times"),
            py::arg("discrete") = false,
            py::arg("granularity") = 100,
            py::call_guard<py::gil_scoped_release>(),
            "Compute PMF/PDF values")

        .def("compute_pmf_and_moments",
            &parameterized::GraphBuilder::compute_pmf_and_moments,
            py::arg("theta"),
            py::arg("times"),
            py::arg("nr_moments"),
            py::arg("discrete") = false,
            py::call_guard<py::gil_scoped_release>(),
            "Compute both PMF and moments efficiently")

        .def_property_readonly("param_length",
            &parameterized::GraphBuilder::param_length)

        .def_property_readonly("vertices_length",
            &parameterized::GraphBuilder::vertices_length);
}
```

### GIL Management Strategy

```cpp
// Pattern 1: Automatic with call_guard
m.def("compute_moments",
    &GraphBuilder::compute_moments,
    py::call_guard<py::gil_scoped_release>()  // Automatic!
);

// Pattern 2: Manual scopes for fine control
py::array_t<double> compute_moments(py::array_t<double> theta, int nr) {
    // GIL held here (Python object access)
    auto theta_buf = theta.unchecked<1>();

    {
        py::gil_scoped_release release;  // Release for C++ work

        // Long C++ computation here
        // Multiple threads can run simultaneously
        Graph g = build(theta_buf.data(), theta_buf.size());
        std::vector<double> result = g.compute_moments_impl(nr);

    }  // GIL automatically reacquired

    // GIL held here (create Python object)
    return py::array_t<double>(result.size(), result.data());
}
```

**Key Benefits:**
1. RAII pattern - exception-safe
2. No manual PyGILState_Ensure/Release
3. Clear code - intent is obvious
4. Works with JAX JIT - no "unknown opcode" errors

---

## Performance Analysis

### Expected Improvements

| Metric | Ctypes (PyDLL) | Pybind11 | JAX FFI |
|--------|----------------|----------|---------|
| First call latency | 0.5-2s | 50-100ms | 10-20ms |
| Repeated call | 5-10ms | 3-5ms | 1-2ms |
| Scalability | ❌ Fails at 20+ | ✅ Works | ✅ Works |
| Parallelism | ❌ No | ⚠️ Limited | ✅ Full |
| GPU support | ❌ No | ❌ No | ✅ Possible |
| Memory efficiency | Low | Medium | High |
| Code maintainability | Low | High | Medium |

### Benchmark Plan

```python
def benchmark_moment_computation():
    """Compare implementations"""
    graph = create_test_graph(nr_samples=10)
    theta = jnp.array([0.8] * 5)  # 5D parameter space

    # Warmup
    for impl in [ctypes_impl, pybind11_impl, ffi_impl]:
        impl(theta)

    # Benchmark
    results = {
        'ctypes': timeit.repeat(lambda: ctypes_impl(theta), number=100),
        'pybind11': timeit.repeat(lambda: pybind11_impl(theta), number=100),
        'ffi': timeit.repeat(lambda: ffi_impl(theta), number=100),
    }

    # Analyze
    for name, times in results.items():
        print(f"{name}: {np.mean(times)*1000:.2f} ± {np.std(times)*1000:.2f} ms")
```

### Target Metrics

**Phase 2-3 (Pybind11):**
- 10x faster first call (eliminate compilation)
- 2x faster repeated calls (better C++ <-> Python boundary)
- 100% success rate with 20-100 particles

**Phase 4 (JAX FFI):**
- 50x faster first call (XLA compilation)
- 3-5x faster repeated calls (optimized execution)
- True parallelism with vmap

---

## Migration Strategy

### Backward Compatibility

```python
# Old API (deprecated but working):
model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, use_pybind11=False)

# New API (default):
model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2)  # Uses pybind11

# Migration warning:
if use_pybind11 == False:
    warnings.warn(
        "ctypes implementation is deprecated and will be removed in v2.0. "
        "Please migrate to pybind11 implementation (use_pybind11=True or omit parameter).",
        DeprecationWarning,
        stacklevel=2
    )
```

### Testing Strategy

**1. Equivalence Tests**
```python
def test_equivalence():
    """Ensure pybind11 gives same results as ctypes"""
    graph = create_test_graph()
    theta = jnp.array([0.5, 1.0])
    times = jnp.linspace(0.1, 5.0, 20)

    # Both implementations
    model_ctypes = Graph.pmf_and_moments_from_graph(graph, use_ctypes=True)
    model_pybind = Graph.pmf_and_moments_from_graph(graph, use_ctypes=False)

    pmf_c, mom_c = model_ctypes(theta, times)
    pmf_p, mom_p = model_pybind(theta, times)

    # Must match within numerical precision
    np.testing.assert_allclose(pmf_c, pmf_p, rtol=1e-12)
    np.testing.assert_allclose(mom_c, mom_p, rtol=1e-12)
```

**2. Regression Tests**
- All existing SVGD tests must pass
- Performance benchmarks must not regress
- Memory usage must not increase significantly

**3. Integration Tests**
- Full SVGD pipeline (100 iterations, 50 particles)
- Moment-based regularization
- Multi-dimensional parameter spaces
- Different graph structures

### Rollout Plan

**Week 1-2:** Implement Phase 2
- Feature branch: `feature/pybind11-builder`
- CI: Build on Linux, macOS, Windows
- Tests: Unit tests for GraphBuilder

**Week 3:** Implement Phase 3
- Feature branch: merge into `develop`
- CI: Integration tests with existing code
- Docs: Update examples to show both APIs

**Week 4:** Beta testing
- Release: v1.9.0-beta1 with opt-in pybind11
- Feedback: Collect performance data from users
- Bug fixes: Address issues

**Week 5-6:** Production release
- Release: v2.0.0 with pybind11 as default
- Deprecation: ctypes gets DeprecationWarning
- Docs: Migration guide

**6 months later:** Remove ctypes
- Release: v2.1.0 removes ctypes code
- Clean up: Delete old implementation

---

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Pybind11 build complexity** | Medium | High | Comprehensive build docs, CI on multiple platforms |
| **Platform-specific bugs** | Medium | Medium | Test matrix: Linux/macOS/Windows × Python 3.9-3.12 |
| **Performance regression** | Low | High | Benchmarks in CI, rollback plan if slower |
| **Breaking changes** | Medium | High | Backward compatibility layer, long deprecation |
| **Memory leaks** | Low | Critical | Valgrind/ASAN in CI, reference counting audits |
| **JAX compatibility** | Low | High | Extensive integration tests, JAX version matrix |

### Project Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Increased maintenance burden** | High | Medium | Good documentation, modular design |
| **Contributor learning curve** | High | Low | Tutorials, code examples, mentoring |
| **User migration friction** | Medium | Medium | Auto-migration, clear warnings, examples |
| **Time/resource constraints** | Medium | High | Phased approach, can stop after Phase 3 |

---

## Alternatives Considered

### Alternative 1: Keep ctypes + PyDLL (Status Quo)

**Pros:**
- ✅ Minimal work required
- ✅ Basic functionality works

**Cons:**
- ❌ Scalability limited (fails at 20+ particles with regularization)
- ❌ "unknown opcode" errors with complex JIT
- ❌ No path to GPU support
- ❌ Performance bottlenecks remain

**Verdict:** Not viable for production use of regularized SVGD

### Alternative 2: Pure JAX Implementation

**Approach:** Rewrite phase-type algorithms entirely in JAX

**Pros:**
- ✅ Full JAX optimization
- ✅ Automatic GPU support
- ✅ Automatic parallelization

**Cons:**
- ❌ Complete rewrite (months of work)
- ❌ Phase-type algorithms complex to express in JAX
- ❌ Lose battle-tested C++ implementation
- ❌ Unknown performance characteristics

**Verdict:** Too risky, too much work

### Alternative 3: Numba Compilation

**Approach:** Use Numba to JIT compile Python code

**Pros:**
- ✅ Python-side JIT
- ✅ No C++ complexity

**Cons:**
- ❌ Can't use existing C++ code
- ❌ Numba-JAX interop unclear
- ❌ Numba's language support limited

**Verdict:** Doesn't leverage existing code

### Alternative 4: Cython

**Approach:** Use Cython instead of pybind11

**Pros:**
- ✅ Similar GIL management
- ✅ Mature tooling

**Cons:**
- ❌ Requires rewriting C++ as Cython
- ❌ Less ergonomic than pybind11
- ❌ Harder to maintain existing C++ code

**Verdict:** pybind11 better for wrapping existing C++

---

## References & Resources

### Documentation
- **JAX FFI**: https://jax.readthedocs.io/en/latest/ffi.html
- **Pybind11**: https://pybind11.readthedocs.io/
- **Python GIL**: https://docs.python.org/3/c-api/init.html#thread-state-and-the-global-interpreter-lock
- **ctypes thread safety**: https://github.com/python/cpython/issues/127945

### Existing Code
- **pybind11 bindings**: `src/cpp/ptdalgorithmscpp_pybind.cpp`
- **Moments implementation**: `src/cpp/ptdalgorithmscpp_pybind.cpp:356-415`
- **Current ctypes**: `src/ptdalgorithms/__init__.py:1610, 1860`

### JAX Resources
- **Custom VJP**: https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
- **pure_callback**: https://jax.readthedocs.io/en/latest/_autosummary/jax.pure_callback.html
- **Batching rules**: https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html

---

## Appendix: Code Examples

### Example 1: Current ctypes Approach

```python
# From __init__.py:1860 (current implementation)
lib = ctypes.PyDLL(lib_path)  # Keeps GIL held

def _compute_pmf_and_moments_pure(theta_flat, times_flat):
    theta_np = np.asarray(theta_flat, dtype=np.float64)
    times_np = np.asarray(times_flat, dtype=np.float64)

    pmf_output = np.zeros(len(times_np), dtype=np.float64)
    moments_output = np.zeros(nr_moments, dtype=np.float64)

    lib.compute_pmf_and_moments(
        theta_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(theta_np),
        times_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(times_np),
        nr_moments,
        pmf_output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        moments_output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )

    return pmf_output, moments_output
```

**Issues:**
- Manual pointer arithmetic
- No type safety
- GIL held entire time (PyDLL)
- Verbose and error-prone

### Example 2: Proposed Pybind11 Approach

```python
# New implementation
from ._ptdalgorithmscpp import parameterized

# Create builder once
builder = parameterized.GraphBuilder(structure_json)

# Use builder (GIL released automatically!)
def _compute_pmf_and_moments_pure(theta, times):
    # Direct numpy array → numpy array
    # GIL released in C++ via py::call_guard
    pmf, moments = builder.compute_pmf_and_moments(
        theta, times, nr_moments, discrete=False
    )
    return pmf, moments
```

**Benefits:**
- Type-safe (pybind11 handles conversion)
- Automatic GIL management
- Cleaner code
- Better error messages

### Example 3: Future JAX FFI Approach

```python
from jax.extend import ffi

# Register custom op
moments_op = ffi.ffi_call(
    "compute_moments",
    result_shape_dtypes=[jax.ShapeDtypeStruct((nr_moments,), jnp.float64)],
    vectorized=True,
    has_side_effect=False
)

def compute_moments(theta):
    return moments_op(theta, nr_moments=nr_moments)

# Full XLA integration!
# Can be jitted, vmapped, and even run on GPU
jitted_moments = jax.jit(compute_moments)
batched_moments = jax.vmap(compute_moments)
```

**Benefits:**
- True XLA compilation
- GPU support possible
- Parallelization built-in
- Optimal performance

---

## Conclusion

The pybind11 refactoring is **necessary** for production-ready moment-based SVGD:

1. **Phase 1 (PyDLL)** ✅ - Fixes crashes but limited scalability
2. **Phase 2-3 (Pybind11)** 🎯 - Recommended next step
   - Fixes "unknown opcode" errors
   - Enables full-scale SVGD (20-100 particles)
   - Better performance and maintainability
3. **Phase 4 (JAX FFI)** 🚀 - Future optimization
   - Optional but valuable
   - Path to GPU support
   - Maximum performance

**Recommendation:** Proceed with Phase 2-3 (pybind11 refactoring) as the immediate priority.

---

**Document Version:** 1.0
**Date:** 2025-10-07
**Author:** Claude Code
**Status:** Design Proposal
