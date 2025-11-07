# Hierarchical SCC Caching - Implementation Roadmap

**Date**: 2025-11-06
**Current Status**: Phase 3a Complete
**Next Steps**: Phase 3b - Full SCC-Level Caching

---

## Unimplemented Features by Category

### 🔴 Critical (Phase 3b Core)

#### 1. Trace Stitching Algorithm
**Status**: Not Implemented (raises `NotImplementedError`)
**Location**: `src/phasic/hierarchical_trace_cache.py:348`
**Complexity**: High
**Estimated Effort**: 1-2 weeks

**What It Does**: Merges SCC traces into full graph trace

**Why Critical**: Enables SCC-level caching and cross-graph reuse

**Requirements**:
- Operation index remapping across SCCs
- Vertex index remapping from SCC subgraph → full graph
- Handle boundary edges between SCCs
- Preserve topological ordering
- Update `vertex_rates`, `edge_probs`, `vertex_targets`

**Algorithm Outline**:
```python
def stitch_scc_traces(scc_graph, scc_trace_dict):
    merged = EliminationTrace()
    op_remap = {}  # SCC op_idx → merged op_idx
    vertex_remap = {}  # SCC vertex_idx → merged vertex_idx

    for scc in scc_graph.sccs_in_topo_order():
        scc_trace = scc_trace_dict[scc.hash()]

        # Remap and append operations
        offset = len(merged.operations)
        for op in scc_trace.operations:
            new_op = remap_operation(op, op_remap, offset)
            merged.operations.append(new_op)

        # Remap vertex indices
        for i, orig_idx in enumerate(scc.internal_vertex_indices()):
            vertex_remap[orig_idx] = len(merged.states)

        # Append states
        merged.states = np.vstack([merged.states, scc_trace.states])

        # Remap vertex_rates
        for v_idx, op_idx in enumerate(scc_trace.vertex_rates):
            merged_v_idx = vertex_remap[v_idx]
            merged_op_idx = op_remap.get(op_idx, op_idx + offset)
            merged.vertex_rates[merged_v_idx] = merged_op_idx

        # Remap edge_probs and vertex_targets
        # ... similar logic ...

    return merged
```

**Testing Requirements**:
- Compare stitched trace vs direct computation (same graph)
- Verify operation semantics preserved
- Test with various topologies (linear, cycles, multiple SCCs)
- Validate boundary edges handled correctly

---

#### 2. Trace Serialization/Deserialization
**Status**: Stubs Only (return None/False)
**Location**: `src/phasic/hierarchical_trace_cache.py:54, 63`
**Complexity**: Medium
**Estimated Effort**: 1-2 days

**What It Does**: Save/load traces to/from disk cache

**Why Critical**: Enables persistent caching across sessions

**Options**:

**Option A: Pickle (Simplest)**
```python
def _save_trace_to_cache(hash_hex, trace):
    cache_file = _get_cache_path(hash_hex)
    with open(cache_file, 'wb') as f:
        pickle.dump(trace, f)
    return True

def _load_trace_from_cache(hash_hex):
    cache_file = _get_cache_path(hash_hex)
    if not cache_file.exists():
        return None
    with open(cache_file, 'rb') as f:
        return pickle.load(f)
```

**Option B: JSON (More Robust)**
```python
def trace_to_dict(trace):
    return {
        'operations': [op_to_dict(op) for op in trace.operations],
        'vertex_rates': trace.vertex_rates.tolist(),
        'edge_probs': trace.edge_probs,
        'vertex_targets': trace.vertex_targets,
        'states': trace.states.tolist(),
        # ... other fields ...
    }

def trace_from_dict(data):
    trace = EliminationTrace()
    trace.operations = [op_from_dict(op) for op in data['operations']]
    # ... restore fields ...
    return trace
```

**Option C: Use Existing C Cache (Best)**
```python
# Leverage existing C-level trace_cache.c functions
def _save_trace_to_cache(hash_hex, trace):
    # Convert to C struct
    c_trace = trace_to_c_struct(trace)
    success = ptd_save_trace_to_cache(hash_hex, c_trace)
    ptd_trace_destroy(c_trace)
    return success
```

**Recommendation**: Start with Pickle (Option A) for speed, migrate to C cache (Option C) for consistency

---

#### 3. Enable SCC Subdivision
**Status**: Logic Disabled (Phase 3a simplified)
**Location**: `src/phasic/hierarchical_trace_cache.py:421-424`
**Complexity**: Low (integration only)
**Estimated Effort**: 1 day

**What It Does**: Actually decompose large graphs into SCCs and compute in parallel

**Current Code**:
```python
# Step 2: Compute trace directly (no subdivision in Phase 3a)
trace = record_elimination_trace(graph, param_length)
```

**Phase 3b Code**:
```python
# Step 2: Check if graph is large enough to subdivide
if graph.vertices_length() >= min_size:
    # Collect missing work units
    work_units = collect_missing_traces_batch(graph, param_length, min_size)

    # Compute in parallel
    if work_units:
        scc_traces = compute_missing_traces_parallel(work_units, parallel_strategy)
    else:
        scc_traces = {}

    # Get SCC decomposition
    scc_graph = graph.scc_decomposition()

    # Build complete trace dict
    all_scc_traces = {}
    for scc in scc_graph.sccs_in_topo_order():
        scc_hash = scc.hash()
        if scc_hash in scc_traces:
            all_scc_traces[scc_hash] = scc_traces[scc_hash]
        else:
            cached = _load_trace_from_cache(scc_hash)
            if cached is None:
                raise RuntimeError(f"SCC trace not found: {scc_hash}")
            all_scc_traces[scc_hash] = cached

    # Stitch together
    trace = stitch_scc_traces(scc_graph, all_scc_traces)
else:
    # Small graph: compute directly
    trace = record_elimination_trace(graph, param_length)
```

**Prerequisites**: Trace stitching must work first

---

### 🟡 Important (Phase 3b Enhancement)

#### 4. Parallel SCC Computation (vmap/pmap)
**Status**: Functions Exist but Not Used
**Location**: `src/phasic/hierarchical_trace_cache.py:231-291`
**Complexity**: Medium
**Estimated Effort**: 2-3 days

**What It Does**: Distribute SCC trace computation across CPUs/GPUs

**Current Status**:
- `compute_missing_traces_parallel()` implemented
- Supports vmap, pmap, sequential strategies
- **Not used**: Phase 3a computes full graph directly

**What Needs Work**:
1. Test vmap strategy (single machine, multi-CPU)
2. Test pmap strategy (multi-GPU/multi-machine)
3. Handle JAX device management
4. Add padding/sharding for pmap
5. Error handling for failed workers

**Testing Requirements**:
- Verify parallel results match sequential
- Benchmark speedup vs sequential
- Test device failures and recovery

---

#### 5. Graph.deserialize() Method
**Status**: Not Implemented
**Location**: N/A (needs creation)
**Complexity**: Medium
**Estimated Effort**: 1 day

**What It Does**: Reconstruct graph from serialized dict

**Why Needed**: Required by `compute_trace_work_unit()` for cross-machine work distribution

**Current Issue**:
```python
# In compute_trace_work_unit()
graph_dict = json.loads(graph_json)
graph = Graph.deserialize(graph_dict)  # ← Does not exist!
```

**Implementation**:
```python
@classmethod
def deserialize(cls, data: Dict[str, np.ndarray]) -> 'Graph':
    """
    Reconstruct graph from serialized dict

    Parameters
    ----------
    data : dict
        Output from Graph.serialize()

    Returns
    -------
    Graph
        Reconstructed graph
    """
    state_length = data['state_length']
    param_length = data['param_length']

    # Create empty graph
    graph = cls(state_length)

    # Restore vertices by state
    state_to_vertex = {}
    for i, state in enumerate(data['states']):
        v = graph.find_or_create_vertex(state)
        state_to_vertex[tuple(state)] = v

    # Restore starting vertex edges
    start = graph.starting_vertex()
    for to_idx, weight in data['start_edges']:
        to_state = tuple(data['states'][to_idx])
        to_vertex = state_to_vertex[to_state]
        start.add_edge(to_vertex, weight)

    # Restore regular edges
    for from_idx, to_idx, weight in data['edges']:
        from_state = tuple(data['states'][from_idx])
        to_state = tuple(data['states'][to_idx])
        from_vertex = state_to_vertex[from_state]
        to_vertex = state_to_vertex[to_state]
        from_vertex.add_edge(to_vertex, weight)

    # Restore parameterized edges
    if param_length > 0:
        for edge_data in data['param_edges']:
            from_idx = int(edge_data[0])
            to_idx = int(edge_data[1])
            coeffs = edge_data[2:]

            from_state = tuple(data['states'][from_idx])
            to_state = tuple(data['states'][to_idx])
            from_vertex = state_to_vertex[from_state]
            to_vertex = state_to_vertex[to_state]
            from_vertex.add_edge_parameterized(to_vertex, 0.0, coeffs)

    return graph
```

**Testing**: Verify `serialize() → deserialize()` roundtrip produces identical graph

---

#### 6. Parameterized Edge Support in SCC Extraction
**Status**: Not Implemented (TODO comment)
**Location**: `api/cpp/scc_graph.cpp:174`
**Complexity**: Medium
**Estimated Effort**: 1-2 days

**What It Does**: Handle parameterized edges when extracting SCC as standalone graph

**Current Code**:
```cpp
// TODO: Add support for parameterized edges in future
// Parameterized edges are not handled in this initial implementation
```

**Why Needed**: For parameterized graphs, SCCs may have parameterized internal edges

**Implementation** (in `SCCVertex::as_graph()`):
```cpp
// Copy parameterized edges
for (size_t i = 0; i < scc_vertex_->internal_vertices_length; ++i) {
    struct ptd_vertex* orig_vertex = scc_vertex_->internal_vertices[i];
    Vertex from_vertex = scc_graph.find_vertex(vertex_state_map[orig_vertex]);

    for (size_t j = 0; j < orig_vertex->parameterized_edges_length; ++j) {
        struct ptd_edge_parameterized* edge = orig_vertex->parameterized_edges[j];

        // Only copy if target is also in this SCC
        auto it = vertex_state_map.find(edge->to);
        if (it != vertex_state_map.end()) {
            Vertex to_vertex = scc_graph.find_vertex(it->second);

            // Get edge state (coefficient vector)
            size_t param_length = parent_scc_graph_->original_graph().param_length();
            std::vector<double> coeffs;
            for (size_t k = 0; k < param_length; ++k) {
                coeffs.push_back(edge->edge_state[k]);
            }

            from_vertex.add_edge_parameterized(to_vertex, edge->base_weight, coeffs);
        }
    }
}
```

**Challenge**: Need to get `param_length` from original graph

---

### 🟢 Nice to Have (Polish & Optimization)

#### 7. Performance Benchmarks
**Status**: Not Implemented
**Complexity**: Medium
**Estimated Effort**: 2-3 days

**What To Measure**:
1. **Cache Hit Rate**
   - Full graph hits
   - Partial SCC hits (Phase 3b)
   - Cache size vs hit rate tradeoff

2. **Computation Time**
   - Phase 3a (full graph)
   - Phase 3b (SCC subdivision)
   - Speedup factor

3. **Memory Usage**
   - Peak memory during computation
   - Cache storage overhead
   - Memory vs speed tradeoff

4. **Scalability**
   - Graph size (10-1000 vertices)
   - SCC count
   - Repeated evaluation (SVGD workload)

**Test Cases**:
- Small graph (10-20 vertices)
- Medium graph (50-100 vertices)
- Large graph (500-1000 vertices)
- Coalescent models (realistic)
- Synthetic graphs (stress test)

**Benchmark Script**:
```python
import time
import numpy as np
from phasic import Graph

def benchmark_caching(graph_sizes, n_repeats=10):
    results = []

    for n in graph_sizes:
        # Build graph
        graph = Graph(callback=coalescent_model, nr_samples=n)

        # Time first computation (cache miss)
        start = time.time()
        trace = graph.compute_trace(hierarchical=True)
        first_time = time.time() - start

        # Time repeated computation (cache hit)
        hit_times = []
        for _ in range(n_repeats):
            start = time.time()
            trace = graph.compute_trace(hierarchical=True)
            hit_times.append(time.time() - start)

        results.append({
            'n_vertices': graph.vertices_length(),
            'first_time': first_time,
            'hit_time_mean': np.mean(hit_times),
            'speedup': first_time / np.mean(hit_times)
        })

    return results
```

---

#### 8. Comprehensive Automated Testing
**Status**: Tests Written but Don't Run (cleanup crashes)
**Complexity**: High (requires fixing crashes)
**Estimated Effort**: 1 week

**Current State**:
- 15 unit tests in `tests/test_scc_api.py`
- 18 integration tests in `tests/test_hierarchical_cache.py`
- 14 standalone tests in `test_implementation.py`
- **All fail due to cleanup crash**

**What Needs Work**:
1. **Fix Cleanup Crashes**
   - Investigate destructor ordering
   - Add explicit cleanup
   - Test with valgrind/ASan

2. **CI/CD Integration**
   - Add to GitHub Actions
   - Automated regression testing
   - Coverage reporting

3. **Edge Case Testing**
   - Empty graphs
   - Single-vertex graphs
   - Disconnected components
   - Very large graphs

4. **Stress Testing**
   - Long-running operations
   - Memory limits
   - Concurrent access

---

#### 9. Documentation Improvements
**Status**: Good but Could Be Better
**Complexity**: Low
**Estimated Effort**: 1-2 days

**What Exists**:
- ✅ API docstrings
- ✅ Implementation plans
- ✅ Completion summaries
- ✅ Testing status

**What's Missing**:
- User guide with examples
- Performance tuning guide
- Troubleshooting guide
- API reference (auto-generated)

**Create**:
1. `docs/hierarchical_caching_guide.md` - User guide
2. `docs/performance_tuning.md` - Optimization tips
3. `docs/api_reference.md` - Complete API reference
4. Update main README with hierarchical caching section

---

#### 10. Error Handling & Validation
**Status**: Minimal
**Complexity**: Low
**Estimated Effort**: 1 day

**Current Issues**:
- Exceptions not always informative
- No input validation in many places
- Silent failures possible

**Improvements Needed**:

**Input Validation**:
```python
def get_trace_hierarchical(graph, param_length=None, min_size=50, parallel_strategy='auto'):
    # Validate inputs
    if not isinstance(graph, Graph):
        raise TypeError(f"graph must be Graph, got {type(graph)}")

    if min_size < 1:
        raise ValueError(f"min_size must be positive, got {min_size}")

    if parallel_strategy not in ['auto', 'vmap', 'pmap', 'sequential']:
        raise ValueError(f"Invalid strategy: {parallel_strategy}")

    # ... rest of function ...
```

**Better Error Messages**:
```python
try:
    hash_result = phasic_hash.compute_graph_hash(graph)
except Exception as e:
    raise RuntimeError(
        f"Failed to compute graph hash: {e}\n"
        f"Graph has {graph.vertices_length()} vertices"
    ) from e
```

---

## Prioritized Implementation Plan

### Phase 3b: Core SCC-Level Caching

**Goal**: Enable true hierarchical caching with SCC subdivision

#### Milestone 1: Trace Stitching (2-3 weeks)
- [ ] Week 1: Design and implement stitching algorithm
  - [ ] Operation remapping
  - [ ] Vertex remapping
  - [ ] Boundary edge handling
- [ ] Week 2: Testing and debugging
  - [ ] Unit tests for stitching
  - [ ] Compare stitched vs direct
  - [ ] Test various topologies
- [ ] Week 3: Integration and validation
  - [ ] Integrate with get_trace_hierarchical
  - [ ] End-to-end testing
  - [ ] Performance validation

#### Milestone 2: Serialization & Subdivision (1 week)
- [ ] Day 1-2: Trace serialization/deserialization
  - [ ] Implement pickle-based caching
  - [ ] Test save/load roundtrip
- [ ] Day 3: Graph.deserialize() method
  - [ ] Implement deserialization
  - [ ] Test serialize/deserialize roundtrip
- [ ] Day 4-5: Enable SCC subdivision
  - [ ] Uncomment subdivision logic
  - [ ] Integration testing
  - [ ] Parameter tuning (min_size)

#### Milestone 3: Parallel Computation (1 week)
- [ ] Day 1-2: Test vmap strategy
  - [ ] Verify correctness
  - [ ] Benchmark speedup
- [ ] Day 3-4: Test pmap strategy
  - [ ] Multi-device testing
  - [ ] Handle device failures
- [ ] Day 5: Optimization
  - [ ] Tune batch sizes
  - [ ] Profile performance

**Total Phase 3b Estimate**: 5-6 weeks

---

### Phase 3c: Polish & Production Readiness

**Goal**: Production-ready system with comprehensive testing

#### Milestone 4: Testing Infrastructure (2 weeks)
- [ ] Week 1: Fix cleanup crashes
  - [ ] Investigate destructor ordering
  - [ ] Add explicit cleanup
  - [ ] Verify fix with test suite
- [ ] Week 2: Comprehensive testing
  - [ ] Run full test suite
  - [ ] Add edge case tests
  - [ ] CI/CD integration

#### Milestone 5: Performance & Optimization (1 week)
- [ ] Day 1-2: Benchmarking
  - [ ] Implement benchmark suite
  - [ ] Compare Phase 3a vs 3b
  - [ ] Measure cache effectiveness
- [ ] Day 3-4: Optimization
  - [ ] Profile bottlenecks
  - [ ] Optimize hot paths
  - [ ] Tune parameters
- [ ] Day 5: Documentation
  - [ ] Performance guide
  - [ ] Tuning recommendations

#### Milestone 6: Documentation & Examples (1 week)
- [ ] Day 1-2: User guide
  - [ ] Getting started
  - [ ] Common workflows
  - [ ] Troubleshooting
- [ ] Day 3-4: API reference
  - [ ] Complete API docs
  - [ ] Code examples
  - [ ] Migration guide
- [ ] Day 5: Polish
  - [ ] Review all docs
  - [ ] Add diagrams
  - [ ] Final edits

**Total Phase 3c Estimate**: 4 weeks

---

## Overall Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 3b | 5-6 weeks | None (can start now) |
| Phase 3c | 4 weeks | Phase 3b complete |
| **Total** | **9-10 weeks** | - |

---

## Quick Wins (Can Do Immediately)

### Priority: High, Effort: Low

1. **Error Handling** (1 day)
   - Add input validation
   - Better error messages
   - Type checking

2. **Graph.deserialize()** (1 day)
   - Needed for cross-machine work
   - Simple implementation
   - High value

3. **Pickle-Based Serialization** (1 day)
   - Enable disk caching
   - Simple implementation
   - Immediate benefit

4. **Documentation Polish** (1 day)
   - User guide
   - Examples
   - Quick reference

**Total Quick Wins**: 4 days

---

## Success Metrics

### Phase 3b Success Criteria

- ✅ Trace stitching produces correct results (verified against direct computation)
- ✅ SCC subdivision enabled and tested
- ✅ Cache hit rate >50% for typical workloads
- ✅ Speedup ≥2x for graphs with reusable SCCs
- ✅ Parallel computation works (vmap/pmap)
- ✅ All tests pass

### Phase 3c Success Criteria

- ✅ No cleanup crashes
- ✅ CI/CD integrated
- ✅ Test coverage >80%
- ✅ Performance benchmarks documented
- ✅ User guide complete
- ✅ Production-ready

---

## Recommendations

### Immediate Next Steps

1. **Start with Quick Wins** (1 week)
   - Graph.deserialize()
   - Pickle serialization
   - Error handling
   - Documentation

2. **Then Tackle Phase 3b** (5-6 weeks)
   - Trace stitching (hardest part)
   - Enable subdivision
   - Test parallel computation

3. **Finally Polish** (4 weeks)
   - Fix cleanup crashes
   - Comprehensive testing
   - Performance optimization

### Alternative: Incremental Approach

If 9-10 weeks is too long, consider incremental delivery:

**v0.23.0** (Quick Wins): 1 week
- Graph.deserialize()
- Pickle serialization
- Better docs

**v0.24.0** (Phase 3b Core): 3-4 weeks
- Trace stitching
- SCC subdivision
- Basic testing

**v0.25.0** (Parallel + Polish): 2-3 weeks
- Parallel computation
- Performance optimization
- Comprehensive testing

This allows delivering value incrementally while working toward full Phase 3b.

---

**Created**: 2025-11-06
**Status**: Planning Document
**Owner**: Kasper Munch
