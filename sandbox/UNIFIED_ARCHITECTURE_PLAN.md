# Unified Architecture Plan: Trace-Based System with FFI + Distributed Computing

## Executive Summary

Unify GraphBuilder and Trace elimination into a single TraceEvaluator that:
- ✅ Supports fast re-parameterization (trace-based)
- ✅ Supports reward transformation (for multivariate)
- ✅ Works with JAX FFI (for distributed computing via pmap)
- ✅ Transparent to Python API (no breaking changes)

## Current State

### Model API (GraphBuilder via FFI)
```python
# FFI mode - SUPPORTS pmap/distributed
model = Graph.pmf_and_moments_from_graph(graph, use_ffi=True)
pmf, moments = model(theta, times, rewards=None)  # ← JAX FFI call → C++

# Flow:
# 1. JAX FFI registered: jax.ffi.register_ffi_target("ptd_compute_pmf_and_moments", ...)
# 2. FFI call: ffi_fn(theta, times, rewards, structure_json=..., ...)
# 3. C++ handler: GraphBuilder::compute_pmf_and_moments(theta, times, rewards)
# 4. XLA manages distribution across devices (pmap works!)
```

**Problem**: Rewards parameter exists (line 637 in ffi_wrappers.py) but C++ ignores it for PDF computation.

### Trace API (Python mode)
```python
# pure_callback - DOES NOT support pmap/distributed
trace = record_elimination_trace(graph, param_length=2)
log_lik = trace_to_log_likelihood(trace, observed_data, use_cpp=False)
ll = log_lik(theta)  # ← pure_callback → Python → C++

# Flow:
# 1. Trace evaluation: Pure JAX ops (jitable)
# 2. Graph instantiation: pure_callback → Python
# 3. PDF computation: graph.pdf() → C++
# 4. pure_callback breaks pmap (executes on host CPU)
```

**Problem**: pure_callback is an escape hatch that breaks distributed computing.

### Trace API (C++ compiled mode)
```python
# Standalone C++ - DOES NOT support pmap/distributed
log_lik = trace_to_log_likelihood(trace, observed_data, use_cpp=True)
ll = log_lik(theta)  # ← pure_callback → compiled .so

# Flow:
# 1. Generate C++ code with embedded trace
# 2. Compile to shared library (one-time)
# 3. Load via ctypes/cffi
# 4. Call via pure_callback
# 5. pure_callback breaks pmap
```

**Problem**: Not integrated with XLA, can't participate in device placement.

---

## Unified Solution: TraceEvaluator via FFI

### Key Insight
**Trace evaluation can be registered as a JAX FFI target**, just like GraphBuilder is!

```python
# Register trace evaluator with JAX FFI
jax.ffi.register_ffi_target(
    "ptd_evaluate_trace",
    evaluate_trace_capsule,  # C++ function pointer
    platform="cpu",
    api_version=1
)
```

This enables:
- Fast trace evaluation (O(n) vs O(n³))
- Reward support (built into evaluation)
- Distributed computing (pmap/jit work)
- Transparent API (same as GraphBuilder)

---

## Implementation Plan

### Phase 1: C++ TraceEvaluator with Rewards

**File**: `src/cpp/trace_evaluator.hpp` (new)

```cpp
namespace phasic {
namespace trace {

class TraceEvaluator {
public:
    // Store trace operations and structure
    struct TraceData {
        std::vector<Operation> operations;
        std::vector<int> vertex_rate_indices;
        std::vector<std::vector<int>> edge_prob_indices;
        std::vector<std::vector<int>> vertex_targets;
        std::vector<std::vector<int>> states;
        int n_vertices;
        int state_length;
        int param_length;
    };

private:
    TraceData trace_;

public:
    // Constructor from JSON (same as GraphBuilder)
    explicit TraceEvaluator(const std::string& trace_json);

    // Core evaluation: trace ops → graph
    Graph evaluate(const double* params, size_t param_len,
                  const double* rewards = nullptr,  // ← REWARD SUPPORT
                  size_t rewards_len = 0);

    // Combined PMF + moments (like GraphBuilder)
    std::pair<py::array_t<double>, py::array_t<double>>
    compute_pmf_and_moments(
        py::array_t<double> theta,
        py::array_t<double> times,
        int nr_moments,
        bool discrete,
        int granularity,
        py::object rewards_obj  // ← REWARD SUPPORT
    );
};

}} // namespace phasic::trace
```

**Key feature**: Rewards applied during evaluation, before graph construction:

```cpp
Graph TraceEvaluator::evaluate(const double* params, const double* rewards, ...) {
    // Step 1: Execute trace operations
    std::vector<double> values(trace_.operations.size());
    for (size_t i = 0; i < trace_.operations.size(); i++) {
        values[i] = eval_operation(trace_.operations[i], values, params);
    }

    // Step 2: Extract vertex rates
    std::vector<double> vertex_rates(trace_.n_vertices);
    for (size_t i = 0; i < trace_.n_vertices; i++) {
        double base_rate = values[trace_.vertex_rate_indices[i]];

        // REWARD TRANSFORMATION: Scale rate by reward
        if (rewards != nullptr && rewards_len > 0) {
            vertex_rates[i] = base_rate * rewards[i];
        } else {
            vertex_rates[i] = base_rate;
        }
    }

    // Step 3: Extract edge probabilities (unchanged)
    // ...

    // Step 4: Build graph from scaled rates
    return build_graph_from_rates(vertex_rates, edge_probs, ...);
}
```

### Phase 2: FFI Registration

**File**: `src/cpp/ffi_handlers.cpp` (modify existing)

```cpp
// Add trace evaluator FFI handler
XLA_FFI_Error* ptd_evaluate_trace_and_compute_pmf_ffi(
    XLA_FFI_CallFrame* call_frame
) {
    // Extract inputs from XLA buffers
    auto theta_buf = ...;
    auto times_buf = ...;
    auto rewards_buf = ...;  // ← REWARD SUPPORT

    // Extract attributes (static data)
    auto trace_json = call_frame->attrs["trace_json"];  // ← TRACE STRUCTURE
    auto granularity = call_frame->attrs["granularity"];
    auto discrete = call_frame->attrs["discrete"];
    auto nr_moments = call_frame->attrs["nr_moments"];

    // Create or get cached TraceEvaluator
    static std::unordered_map<std::string, TraceEvaluator> evaluator_cache;
    if (evaluator_cache.find(trace_json) == evaluator_cache.end()) {
        evaluator_cache[trace_json] = TraceEvaluator(trace_json);
    }
    auto& evaluator = evaluator_cache[trace_json];

    // Evaluate trace with rewards
    Graph g = evaluator.evaluate(theta_buf, rewards_buf);

    // Compute PDF on reward-scaled graph (CORRECT!)
    std::vector<double> pmf_vals = compute_pdf_batch(g, times_buf, granularity);

    // Compute moments (rewards already applied to graph)
    std::vector<double> moments = compute_moments(g, nr_moments);

    // Write results to XLA output buffers
    write_output(call_frame, pmf_vals, moments);
    return nullptr;  // Success
}

// Register with JAX
PYBIND11_MODULE(phasic_pybind, m) {
    // ... existing registrations ...

    m.def("get_trace_evaluator_ffi_capsule", []() {
        return py::capsule(
            reinterpret_cast<void*>(&ptd_evaluate_trace_and_compute_pmf_ffi),
            "xla._CUSTOM_CALL_TARGET"
        );
    });
}
```

### Phase 3: Python FFI Wrapper

**File**: `src/phasic/ffi_wrappers.py` (add new function)

```python
def compute_pmf_and_moments_from_trace_ffi(
    trace_json: str,
    theta: jax.Array,
    times: jax.Array,
    nr_moments: int,
    discrete: bool = False,
    granularity: int = 0,
    rewards: jax.Array = None
) -> tuple[jax.Array, jax.Array]:
    """
    Compute PMF and moments from trace using JAX FFI.

    Supports:
    - jax.jit: JIT compilation
    - jax.vmap: Batching over parameters
    - jax.pmap: Multi-device parallelization  ← DISTRIBUTED COMPUTING
    - Rewards: For multivariate phase-type
    """
    _register_ffi_targets()  # Registers both GraphBuilder and TraceEvaluator

    # Handle optional rewards
    if rewards is None:
        rewards = jnp.array([], dtype=jnp.float64)
    else:
        rewards = jnp.asarray(rewards, dtype=jnp.float64)

    # Call JAX FFI target for trace evaluation
    ffi_fn = jax.ffi.ffi_call(
        "ptd_evaluate_trace_and_compute_pmf",  # ← NEW FFI TARGET
        (jax.ShapeDtypeStruct(times.shape, jnp.float64),
         jax.ShapeDtypeStruct((nr_moments,), jnp.float64)),
        vmap_method="expand_dims"  # ← ENABLES pmap
    )
    pmf_result, moments_result = ffi_fn(
        theta,       # Buffer: parameters
        times,       # Buffer: time points
        rewards,     # Buffer: reward vector
        trace_json=trace_json,              # Attr: trace structure (static)
        granularity=np.int32(granularity),
        discrete=np.bool_(discrete),
        nr_moments=np.int32(nr_moments)
    )
    return pmf_result, moments_result
```

### Phase 4: Unified Model API

**File**: `src/phasic/__init__.py` (modify `pmf_and_moments_from_graph`)

```python
@classmethod
def pmf_and_moments_from_graph(cls, graph, nr_moments=1, discrete=False,
                               use_ffi=True, param_length=None):
    """
    Create JAX-compatible model from parameterized graph.

    NOW USES: TraceEvaluator (fast re-parameterization + rewards)
    BEFORE: GraphBuilder (slow re-parameterization, broken rewards)
    """
    # Record trace from graph (one-time O(n³))
    from .trace_elimination import record_elimination_trace
    trace = record_elimination_trace(graph, param_length=param_length)

    # Serialize trace to JSON
    trace_json = json.dumps(trace_to_json(trace))

    if use_ffi:
        # FFI MODE: TraceEvaluator via XLA (fast, distributed)
        from .ffi_wrappers import compute_pmf_and_moments_from_trace_ffi

        def model(theta, times, rewards=None):
            return compute_pmf_and_moments_from_trace_ffi(
                trace_json, theta, times, nr_moments, discrete, 0, rewards
            )
    else:
        # FALLBACK: Python trace evaluation (compatible, slower)
        from .trace_elimination import evaluate_trace_jax, instantiate_from_trace

        def model(theta, times, rewards=None):
            # ... existing pure_callback implementation ...
            pass

    # Return JAX-compatible model with custom VJP
    return _wrap_with_vjp(model)
```

### Phase 5: Migration Path

**Step 1**: Implement TraceEvaluator C++ class
- Can reuse existing trace evaluation code
- Add reward scaling logic
- Test standalone (no FFI yet)

**Step 2**: Add FFI handler and registration
- Register "ptd_evaluate_trace_and_compute_pmf"
- Test with simple jit/vmap

**Step 3**: Test distributed computing
```python
# Test pmap works with TraceEvaluator
model = Graph.pmf_and_moments_from_graph(graph, use_ffi=True)

# Distributed across 8 devices
@jax.pmap
def compute_batch(theta_batch):
    return model(theta_batch, times, rewards=rewards)

result = compute_batch(theta_particles)  # ← Should work!
```

**Step 4**: Update Model API to use TraceEvaluator
- Change `pmf_and_moments_from_graph` to record trace
- Keep GraphBuilder as fallback for now
- Deprecate later

**Step 5**: Update Trace API to use FFI
- Modify `trace_to_log_likelihood` to use FFI when available
- Keep compiled C++ mode as ultra-fast option
- Keep Python mode for debugging

---

## Benefits of Unified Approach

| Feature | GraphBuilder (old) | TraceEvaluator (new) |
|---------|-------------------|---------------------|
| **Re-parameterization** | O(n³) each time | O(n) each time ✅ |
| **Reward support** | ❌ Broken | ✅ Built-in |
| **FFI / distributed** | ✅ Yes | ✅ Yes |
| **pmap support** | ✅ Yes | ✅ Yes |
| **JIT benefit** | Minimal | Minimal |
| **API transparency** | ✅ Yes | ✅ Yes |

---

## Success Criteria

1. ✅ Model API uses TraceEvaluator transparently
2. ✅ Trace API uses TraceEvaluator via FFI
3. ✅ Rewards work correctly (multivariate SVGD: 1-2% error)
4. ✅ `pmap` works for distributed computing
5. ✅ Fast re-parameterization (< 1ms per trace eval)
6. ✅ Backward compatible (existing code works)
7. ✅ Single code path (eliminate GraphBuilder duplication)

---

## Timeline

- **Phase 1** (C++ TraceEvaluator): 2-3 days
- **Phase 2** (FFI registration): 1 day
- **Phase 3** (Python wrapper): 1 day
- **Phase 4** (Model API migration): 1 day
- **Phase 5** (Testing + docs): 2 days

**Total**: 7-9 days

---

## Risk Assessment

**Low risk**:
- FFI infrastructure already exists and works
- Trace evaluation code already exists
- Just combining existing pieces differently

**Fallback**:
- Keep GraphBuilder for backward compatibility
- Make TraceEvaluator opt-in initially
- Gradual migration, not big-bang replacement

---

## Key Architectural Decision

**Use JAX FFI for trace evaluation, NOT pre-compiled C++**

Why:
- FFI integrates with XLA device placement
- Enables `pmap` / distributed computing
- Single registration, works everywhere
- No compilation overhead (trace cached internally)

The compiled C++ mode can remain as an ultra-optimization for single-machine SVGD, but FFI mode should be the primary path.
