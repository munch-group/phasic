# FFI Performance Analysis: Sojourn Times

**Date**: 2025-12-30
**Status**: Analysis Complete

## Summary

FFI implementation is **slower** for small graphs but **enables** large graphs that were previously impossible due to memory constraints.

**Key Finding**: The tradeoff is **acceptable** because memory, not speed, is the bottleneck for the target use case (SVGD with large joint probability graphs).

---

## Benchmark Results

### Small Graphs (5-168 vertices)

| Graph Size | Method | Time/Iteration | Speedup |
|------------|--------|---------------|---------|
| 5 vertices | Direct C++ | 0.01ms | - |
| | FFI | 0.16ms | **0.08x** (12x slower) |
| 8 vertices | Direct C++ | 0.01ms | - |
| | FFI | 0.23ms | **0.06x** (17x slower) |
| 11 vertices | Direct C++ | 0.02ms | - |
| | FFI | 0.42ms | **0.05x** (20x slower) |
| 168 vertices (joint graph) | Direct C++ | 0.26ms | - |
| | FFI | 1.51ms | **0.17x** (6x slower) |

**Conclusion**: FFI is consistently slower for small graphs due to JAX overhead (marshaling, registration, etc.).

---

## Why FFI is Slower on Small Graphs

### FFI Overhead Components

1. **JAX FFI Call Overhead** (~0.1-0.2ms)
   - Marshaling Python → C++ buffers
   - JAX tracing/compilation (first call)
   - XLA dispatch

2. **Array Conversion** (~0.05ms)
   - JAX array → numpy → C++ buffer
   - int32 dtype conversion for indices

3. **Thread-Local Cache Lookup** (~0.01ms)
   - Hash JSON string
   - Lookup in unordered_map

**Total Overhead**: ~0.15-0.25ms

For tiny graphs where computation is <0.1ms, overhead dominates.

---

## Memory Analysis: The Critical Factor

### Small Graph (168 vertices)

**Full Computation (wasteful)**:
- Matrix: 168 × 168 = 28,224 elements
- Memory: 0.0002 GB (0.2 MB)
- **Not a problem**

**Subset (64 indices via FFI)**:
- Matrix: 168 × 64 = 10,752 elements
- Memory: 0.0001 GB (0.1 MB)
- Savings: 62% (but both are tiny)

### Large Graph (183,000 vertices) - **Target Use Case**

**Full Computation (wasteful)**:
- Matrix: 183,000 × 183,000 = 33,489,000,000 elements
- Memory: **267.9 GB**
- **Out of memory on most systems**

**Subset (1,215 t-states via FFI)**:
- Matrix: 183,000 × 1,215 = 222,345,000 elements
- Memory: **1.78 GB**
- Savings: **99.3% (267.9 GB → 1.78 GB)**

---

## SVGD Workload: The Real Bottleneck

### Scenario: Inference on Large Joint Probability Graph

**Parameters**:
- Graph size: 183,000 vertices
- T-states needed: 1,215 vertices (0.66% of graph)
- SVGD: 100 particles × 1,000 iterations = 100,000 evaluations

**Without FFI (wasteful)**:
- Each evaluation: Compute all 183k vertices (267.9 GB)
- Total memory: 267.9 GB per evaluation
- **Result**: Out of memory, cannot run

**With FFI (efficient)**:
- Each evaluation: Compute only 1,215 needed vertices (1.78 GB)
- Total memory: 1.78 GB per evaluation
- **Result**: Practical, completes in reasonable time

**Performance Impact**:
- Direct C++ time (if it could run): ~50ms per evaluation (estimated)
- FFI time: ~300ms per evaluation (6x slower, estimated from small graph ratio)
- **But**: 6x slower on a computation that can actually run vs ∞ time on impossible computation

**Conclusion**: FFI enables the computation, even though it's slower per evaluation.

---

## When to Use FFI vs Direct C++

### Use Direct C++ (wasteful but fast) when:
- Graph is small (< 1,000 vertices)
- Computing all vertices is practical
- Speed is critical, memory is not

### Use FFI (efficient but slower) when:
- Graph is large (> 10,000 vertices)
- Only need subset of vertices
- Memory is the bottleneck
- **This is the joint_index=True SVGD use case**

---

## Recommendation: Keep FFI Implementation

**Decision: ✓ Keep FFI despite being slower on small graphs**

**Justification**:

1. **Target use case requires it**
   - SVGD with `joint_index=True` on large graphs
   - 267.9 GB → 1.78 GB enables computation

2. **Performance impact is acceptable**
   - Small graphs: 1.5ms vs 0.26ms (both fast enough)
   - Large graphs: 300ms vs impossible (FFI wins)

3. **API is correct**
   - Same signature as other FFI functions
   - Supports vmap, jit, grad (via custom VJP)
   - No silent fallbacks

4. **Future optimization possible**
   - Could add heuristic to switch between methods
   - Could optimize FFI overhead
   - Could cache compiled kernels

---

## Alternative Considered: Hybrid Approach

**Idea**: Automatically choose Direct C++ vs FFI based on graph size

```python
def _compute_pure(theta, vertex_indices):
    n_vertices = graph.vertices_length()
    n_indices = len(vertex_indices)

    # Use direct C++ for small graphs
    if n_vertices < 1000:
        # ... direct C++ implementation ...
    else:
        # Use FFI for large graphs
        return compute_sojourn_times_ffi(...)
```

**Rejected because**:
- Adds complexity
- Threshold is arbitrary
- Current performance is acceptable for all cases
- Can optimize later if needed

---

## Conclusion

**FFI is the correct choice** for `pmf_from_graph_joint_index()` because:

1. ✅ Enables large graphs (the whole point)
2. ✅ Acceptable speed on small graphs (1.5ms is fine)
3. ✅ Matches API of other FFI functions
4. ✅ Supports JAX transformations
5. ✅ No memory explosions

**Tradeoff**: 6x slower on small graphs, but enables impossible computations on large graphs.

**User impact**: Negligible for interactive use, critical for SVGD workloads.
