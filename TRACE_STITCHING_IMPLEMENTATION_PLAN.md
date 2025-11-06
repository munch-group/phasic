# Trace Stitching Implementation Plan

**Date**: 2025-11-06
**Status**: Ready for Implementation
**Target**: Complete Phase 3b hierarchical caching

---

## Implementation Strategy

### Goals
1. Implement `stitch_scc_traces()` in `hierarchical_trace_cache.py`
2. Support both constant and parameterized edges
3. Handle all edge cases (single SCC, disconnected components, etc.)
4. Write comprehensive tests
5. Integrate with `get_trace_hierarchical()`

### Timeline
- **Week 1**: Core stitching implementation (Days 1-5)
- **Week 2**: Testing and edge cases (Days 6-10)
- **Week 3**: Integration and validation (Days 11-15)

---

## Implementation Steps

### Step 1: Core Data Structures (Day 1)

**File**: `src/phasic/hierarchical_trace_cache.py`

**Add helper classes**:
```python
@dataclass
class VertexMapping:
    """Maps vertices between SCC subgraph and original/merged graphs"""
    scc_idx: int
    scc_v_idx: int
    orig_v_idx: int
    merged_v_idx: int
    state: np.ndarray

@dataclass
class OperationMapping:
    """Maps operations from SCC traces to merged trace"""
    scc_idx: int
    scc_op_idx: int
    merged_op_idx: int
```

**Rationale**: Explicit mappings make debugging easier, can optimize later if needed.

### Step 2: Vertex Mapping Function (Day 1)

```python
def _build_vertex_mappings(
    scc_graph: 'SCCGraph',
    scc_traces: Dict[str, 'EliminationTrace']
) -> Tuple[Dict[Tuple[int, int], int], Dict[int, int]]:
    """
    Build vertex mappings for stitching.

    Returns
    -------
    vertex_to_original : Dict[(scc_idx, scc_v_idx), orig_v_idx]
        Maps SCC vertex indices to original graph indices
    original_to_merged : Dict[orig_v_idx, merged_v_idx]
        Maps original graph indices to merged trace indices
    """
    original_graph = scc_graph.original_graph()
    sccs = scc_graph.sccs_in_topo_order()

    vertex_to_original = {}
    original_to_merged = {}
    next_merged_idx = 0

    for scc_idx, scc in enumerate(sccs):
        scc_trace = scc_traces[scc.hash()]

        for scc_v_idx in range(scc_trace.n_vertices):
            # Get state from SCC trace
            state = scc_trace.states[scc_v_idx]

            # Find in original graph (user's simplification!)
            orig_vertex = original_graph.find_vertex(state)
            orig_v_idx = orig_vertex.index()

            # Record mapping
            vertex_to_original[(scc_idx, scc_v_idx)] = orig_v_idx

            # Assign merged index if not already assigned
            if orig_v_idx not in original_to_merged:
                original_to_merged[orig_v_idx] = next_merged_idx
                next_merged_idx += 1

    return vertex_to_original, original_to_merged
```

**Key insight**: Using `find_vertex(state).index()` eliminates need for pointer mappings!

### Step 3: Operation Remapping Function (Day 2)

```python
def _remap_operation(
    op: 'Operation',
    op_offset: int
) -> 'Operation':
    """
    Remap operation indices by adding offset.

    Parameters
    ----------
    op : Operation
        Original operation from SCC trace
    op_offset : int
        Offset to add to operation indices

    Returns
    -------
    Operation
        Remapped operation for merged trace
    """
    from .trace_elimination import Operation, OpType

    if op.op_type == OpType.CONST:
        # No remapping needed
        return Operation(op_type=OpType.CONST, value=op.value)

    elif op.op_type == OpType.PARAM:
        # No remapping needed (references parameter array)
        return Operation(op_type=OpType.PARAM, operands=op.operands.copy())

    elif op.op_type == OpType.DOT:
        # Remap operands, preserve coefficients
        return Operation(
            op_type=OpType.DOT,
            coefficients=op.coefficients.copy(),
            operands=[idx + op_offset for idx in op.operands]
        )

    elif op.op_type in [OpType.ADD, OpType.MUL, OpType.DIV]:
        # Remap both operands
        return Operation(
            op_type=op.op_type,
            operands=[idx + op_offset for idx in op.operands]
        )

    elif op.op_type == OpType.INV:
        # Remap single operand
        return Operation(
            op_type=OpType.INV,
            operands=[op.operands[0] + op_offset]
        )

    elif op.op_type == OpType.SUM:
        # Remap all operands
        return Operation(
            op_type=OpType.SUM,
            operands=[idx + op_offset for idx in op.operands]
        )

    else:
        raise ValueError(f"Unknown operation type: {op.op_type}")
```

**Key principle**: Only remap indices that reference other operations, not external data (CONST values, PARAM indices).

### Step 4: Core Stitching Function (Days 2-3)

```python
def stitch_scc_traces(
    scc_graph: 'SCCGraph',
    scc_trace_dict: Dict[str, 'EliminationTrace']
) -> 'EliminationTrace':
    """
    Stitch SCC traces into unified full-graph trace.

    Algorithm:
    1. Build vertex mappings
    2. Initialize merged trace
    3. Process each SCC in topological order:
       a. Remap and append operations
       b. Copy and remap vertex data
    4. Add boundary edges
    5. Return merged trace

    Parameters
    ----------
    scc_graph : SCCGraph
        Graph decomposed into SCCs
    scc_trace_dict : Dict[str, EliminationTrace]
        Traces for each SCC (keyed by SCC hash)

    Returns
    -------
    EliminationTrace
        Unified trace for full graph
    """
    from .trace_elimination import EliminationTrace, Operation, OpType
    import numpy as np

    original_graph = scc_graph.original_graph()
    sccs = scc_graph.sccs_in_topo_order()

    # Edge case: empty graph
    if len(sccs) == 0:
        raise ValueError("Cannot stitch empty SCC graph")

    # Get first trace for metadata
    first_trace = next(iter(scc_trace_dict.values()))

    # Step 1: Build vertex mappings
    vertex_to_original, original_to_merged = _build_vertex_mappings(
        scc_graph, scc_trace_dict
    )

    # Step 2: Initialize merged trace
    n_vertices_merged = len(original_to_merged)

    merged = EliminationTrace(
        operations=[],
        vertex_rates=np.zeros(n_vertices_merged, dtype=np.int64),
        edge_probs=[[] for _ in range(n_vertices_merged)],
        vertex_targets=[[] for _ in range(n_vertices_merged)],
        states=np.zeros((n_vertices_merged, first_trace.state_length), dtype=first_trace.states.dtype),
        starting_vertex_idx=0,  # Will be set later
        n_vertices=n_vertices_merged,
        state_length=first_trace.state_length,
        param_length=first_trace.param_length,
        reward_length=first_trace.reward_length,
        is_discrete=first_trace.is_discrete,
        metadata={}
    )

    # Step 3: Process each SCC in topological order
    op_remap = {}  # (scc_idx, scc_op_idx) -> merged_op_idx

    for scc_idx, scc in enumerate(sccs):
        scc_hash = scc.hash()
        scc_trace = scc_trace_dict[scc_hash]

        # 3a. Remap and append operations
        op_offset = len(merged.operations)

        for scc_op_idx, operation in enumerate(scc_trace.operations):
            # Remap operation indices
            new_op = _remap_operation(operation, op_offset)
            merged.operations.append(new_op)

            # Record mapping
            op_remap[(scc_idx, scc_op_idx)] = op_offset + scc_op_idx

        # 3b. Copy and remap vertex data
        for scc_v_idx in range(scc_trace.n_vertices):
            orig_v_idx = vertex_to_original[(scc_idx, scc_v_idx)]
            merged_v_idx = original_to_merged[orig_v_idx]

            # Copy state
            merged.states[merged_v_idx] = scc_trace.states[scc_v_idx]

            # Remap vertex_rates
            scc_rate_op_idx = scc_trace.vertex_rates[scc_v_idx]
            if (scc_idx, scc_rate_op_idx) in op_remap:
                merged.vertex_rates[merged_v_idx] = op_remap[(scc_idx, scc_rate_op_idx)]

            # Remap edge_probs and vertex_targets
            for j, scc_edge_op_idx in enumerate(scc_trace.edge_probs[scc_v_idx]):
                # Remap edge operation
                merged_edge_op_idx = op_remap[(scc_idx, scc_edge_op_idx)]
                merged.edge_probs[merged_v_idx].append(merged_edge_op_idx)

                # Remap target vertex
                scc_target_v_idx = scc_trace.vertex_targets[scc_v_idx][j]
                orig_target_v_idx = vertex_to_original[(scc_idx, scc_target_v_idx)]
                merged_target_v_idx = original_to_merged[orig_target_v_idx]
                merged.vertex_targets[merged_v_idx].append(merged_target_v_idx)

    # Step 4: Add boundary edges
    _add_boundary_edges(merged, scc_graph, vertex_to_original, original_to_merged)

    # Step 5: Set starting vertex
    starting_state = original_graph.starting_vertex().state()
    starting_orig_idx = original_graph.find_vertex(starting_state).index()
    merged.starting_vertex_idx = original_to_merged[starting_orig_idx]

    return merged
```

### Step 5: Boundary Edge Addition (Day 3)

```python
def _add_boundary_edges(
    merged: 'EliminationTrace',
    scc_graph: 'SCCGraph',
    vertex_to_original: Dict[Tuple[int, int], int],
    original_to_merged: Dict[int, int]
) -> None:
    """
    Add boundary edges (edges crossing SCC boundaries).

    Modifies merged trace in-place.
    """
    from .trace_elimination import Operation, OpType

    original_graph = scc_graph.original_graph()
    sccs = scc_graph.sccs_in_topo_order()

    # Build set of internal vertices for each SCC
    scc_vertex_sets = []
    for scc_idx, scc in enumerate(sccs):
        internal_states = set()
        scc_subgraph = scc.as_graph()
        for v_idx in range(scc_subgraph.vertices_length()):
            v = scc_subgraph.get_vertex(v_idx)
            internal_states.add(tuple(v.state()))
        scc_vertex_sets.append(internal_states)

    # Iterate through SCCs and find boundary edges
    for scc_idx, scc in enumerate(sccs):
        current_scc_states = scc_vertex_sets[scc_idx]

        # Check each vertex in this SCC
        for (map_scc_idx, scc_v_idx), orig_v_idx in vertex_to_original.items():
            if map_scc_idx != scc_idx:
                continue

            # Get original vertex
            orig_vertex = original_graph.get_vertex(orig_v_idx)
            merged_v_idx = original_to_merged[orig_v_idx]

            # Check each outgoing edge
            for edge_idx in range(orig_vertex.edges_length()):
                edge = orig_vertex.get_edge(edge_idx)
                target_state = tuple(edge.target.state())

                # Is this a boundary edge?
                if target_state not in current_scc_states:
                    # Get target merged index
                    target_orig_vertex = original_graph.find_vertex(edge.target.state())
                    target_orig_idx = target_orig_vertex.index()
                    target_merged_idx = original_to_merged[target_orig_idx]

                    # Create operation for edge weight
                    if edge.is_parameterized():
                        # Parameterized edge: DOT operation
                        coeffs = list(edge.edge_state)
                        param_indices = list(range(len(coeffs)))

                        edge_op = Operation(
                            op_type=OpType.DOT,
                            coefficients=coeffs,
                            operands=param_indices
                        )
                    else:
                        # Constant edge: CONST operation
                        edge_op = Operation(
                            op_type=OpType.CONST,
                            value=edge.weight
                        )

                    # Add operation and update trace
                    op_idx = len(merged.operations)
                    merged.operations.append(edge_op)
                    merged.edge_probs[merged_v_idx].append(op_idx)
                    merged.vertex_targets[merged_v_idx].append(target_merged_idx)
```

**Key principle**: Boundary edge weights come from original graph, accessed during this step.

### Step 6: Edge Case Handling (Day 4)

**Edge cases to handle**:

1. **Single SCC**: Should work without special handling
2. **Disconnected components**: Need to verify all vertices reachable
3. **Empty SCC traces**: Validate before stitching
4. **Missing SCC traces**: Raise informative error
5. **Parameterized vs constant graphs**: Support both

```python
def stitch_scc_traces(scc_graph, scc_trace_dict):
    # Validate inputs
    if not scc_trace_dict:
        raise ValueError("scc_trace_dict is empty")

    sccs = scc_graph.sccs_in_topo_order()

    # Check all SCC traces present
    for scc in sccs:
        if scc.hash() not in scc_trace_dict:
            raise ValueError(f"Missing trace for SCC {scc.hash()}")

    # ... rest of implementation
```

### Step 7: Testing (Days 5-7)

**Test files to create**:

1. **`tests/test_trace_stitching_unit.py`**: Unit tests for helpers
   - `test_vertex_mapping_simple()`
   - `test_vertex_mapping_multiple_sccs()`
   - `test_operation_remapping_const()`
   - `test_operation_remapping_param()`
   - `test_operation_remapping_dot()`
   - `test_operation_remapping_binary()`

2. **`tests/test_trace_stitching_integration.py`**: Integration tests
   - `test_stitch_single_scc()` - should be identity
   - `test_stitch_two_sccs_constant()`
   - `test_stitch_two_sccs_parameterized()`
   - `test_stitch_boundary_edges()`
   - `test_stitch_vs_direct_computation()`

3. **`tests/test_trace_stitching_edge_cases.py`**: Edge cases
   - `test_empty_scc_graph()`
   - `test_missing_scc_trace()`
   - `test_disconnected_graph()`

**Key validation strategy**:
```python
def test_stitch_vs_direct_computation():
    """Compare stitched trace vs direct computation"""
    # Build graph
    graph = Graph(callback=coalescent_callback, parameterized=True)

    # Direct trace
    trace_direct = record_elimination_trace(graph)

    # SCC decomposition and stitching
    scc_graph = graph.scc_decomposition()
    scc_traces = {}
    for scc in scc_graph.sccs_in_topo_order():
        scc_subgraph = scc.as_graph()
        scc_traces[scc.hash()] = record_elimination_trace(scc_subgraph)
    trace_stitched = stitch_scc_traces(scc_graph, scc_traces)

    # Evaluate both with same parameters
    params = np.array([1.0, 2.0])
    result_direct = evaluate_trace_jax(trace_direct, params)
    result_stitched = evaluate_trace_jax(trace_stitched, params)

    # Compare
    np.testing.assert_allclose(
        result_direct['vertex_rates'],
        result_stitched['vertex_rates']
    )
    # ... compare edge_probs and vertex_targets
```

### Step 8: Integration with Hierarchical Caching (Days 8-9)

**Modify `get_trace_hierarchical()`**:

```python
def get_trace_hierarchical(
    graph: 'Graph',
    param_length: int = 0,
    min_scc_size: int = 10
) -> 'EliminationTrace':
    """
    Get trace with hierarchical SCC-level caching.

    New parameters
    --------------
    min_scc_size : int
        Minimum SCC size to subdivide (default: 10)
    """
    from . import hash as phasic_hash

    # Check full graph cache first
    full_hash = phasic_hash.compute_graph_hash(graph)
    trace = _load_trace_from_cache(full_hash.hash_hex)
    if trace is not None:
        return trace

    # Try SCC subdivision if graph is large enough
    if graph.vertices_length() >= min_scc_size * 2:
        scc_graph = graph.scc_decomposition()
        sccs = scc_graph.sccs_in_topo_order()

        # Only subdivide if we have multiple SCCs
        if len(sccs) > 1:
            scc_trace_dict = {}

            # Get trace for each SCC (with caching)
            for scc in sccs:
                scc_hash = scc.hash()

                # Try cache first
                scc_trace = _load_trace_from_cache(scc_hash)

                if scc_trace is None:
                    # Compute and cache
                    scc_subgraph = scc.as_graph()
                    scc_trace = record_elimination_trace(
                        scc_subgraph,
                        param_length=param_length
                    )
                    _save_trace_to_cache(scc_hash, scc_trace)

                scc_trace_dict[scc_hash] = scc_trace

            # Stitch SCC traces
            trace = stitch_scc_traces(scc_graph, scc_trace_dict)

            # Cache full trace
            _save_trace_to_cache(full_hash.hash_hex, trace)
            return trace

    # Fall back to direct computation
    trace = record_elimination_trace(graph, param_length=param_length)
    _save_trace_to_cache(full_hash.hash_hex, trace)
    return trace
```

### Step 9: Documentation (Day 10)

**Add to `TRACE_STITCHING_DESIGN.md`**:
- Implementation notes
- Performance results
- Known limitations
- Usage examples

**Update `hierarchical_trace_cache.py` docstrings**:
- Complete function documentation
- Parameter descriptions
- Examples

---

## Testing Strategy

### Unit Tests (Days 5-6)
- Test each helper function independently
- Mock SCC graphs and traces for predictable scenarios
- Verify mappings are correct

### Integration Tests (Day 7)
- Test full stitching pipeline
- Compare stitched vs direct computation
- Test with real models (coalescent, queuing)

### Property Tests (Day 8)
- All vertex indices valid
- All operation indices valid
- Edge probabilities sum to 1.0
- No dangling references
- State vectors preserved

### Performance Tests (Day 9)
- Benchmark stitching overhead
- Compare cached vs uncached performance
- Verify sub-linear scaling with cache hits

---

## Success Criteria

✅ **Functional**:
- [ ] Stitch simple 2-SCC graph correctly
- [ ] Stitch complex multi-SCC graph
- [ ] Handle boundary edges correctly
- [ ] Support parameterized graphs
- [ ] Pass all unit tests
- [ ] Pass integration tests vs direct computation

✅ **Performance**:
- [ ] Stitching overhead < 10% of trace recording time
- [ ] Cache hits provide 10x+ speedup
- [ ] No memory leaks

✅ **Code Quality**:
- [ ] Comprehensive docstrings
- [ ] Type hints on all functions
- [ ] Logging for debugging
- [ ] Error handling with informative messages

---

## Risk Mitigation

### Risk 1: Complex Boundary Edge Logic
**Mitigation**: Extensive testing, visualization tools for debugging

### Risk 2: Performance Overhead
**Mitigation**: Profile early, optimize hot paths, consider Cython if needed

### Risk 3: Edge Cases Breaking Correctness
**Mitigation**: Property-based testing, fuzzing with random graphs

---

## Implementation Checklist

### Week 1: Core Implementation
- [ ] Day 1: Implement vertex mapping helpers
- [ ] Day 2: Implement operation remapping
- [ ] Day 3: Implement core stitching function
- [ ] Day 3: Implement boundary edge addition
- [ ] Day 4: Add edge case handling
- [ ] Day 5: Write unit tests

### Week 2: Testing
- [ ] Day 6: Write integration tests
- [ ] Day 7: Test with real models
- [ ] Day 8: Property-based testing
- [ ] Day 9: Performance benchmarks
- [ ] Day 10: Documentation

### Week 3: Integration
- [ ] Day 11: Integrate with `get_trace_hierarchical()`
- [ ] Day 12: End-to-end testing
- [ ] Day 13: Performance validation
- [ ] Day 14: Edge case fixes
- [ ] Day 15: Final documentation and release

---

**Status**: Ready to implement ✅
**Estimated Time**: 10-15 days
**Risk Level**: Medium (well-designed, but complex algorithm)
