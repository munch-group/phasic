# Trace Stitching Algorithm Design

**Date**: 2025-11-06
**Status**: Design Phase
**Goal**: Merge SCC traces into full graph trace

---

## Problem Statement

Given:
- A graph decomposed into SCCs via `scc_decomposition()`
- Individual traces for each SCC subgraph
- SCCs in topological order

Produce:
- Single unified trace for the full graph
- Preserves correctness of elimination algorithm
- Maintains operation dependencies

---

## Key Challenges

### 1. **Operation Index Remapping**
Each SCC trace has operations indexed 0..N. When stitching:
- Operations from SCC #2 must be renumbered to avoid conflicts with SCC #1
- All references to operation indices must be updated

### 2. **Vertex Index Remapping**
Each SCC trace indexes vertices 0..M within that SCC. When stitching:
- Need to map SCC subgraph vertex indices → original graph vertex indices
- SCC subgraph may reorder vertices differently than original graph

### 3. **Boundary Edges**
Edges between SCCs exist in original graph but not in SCC subgraphs:
- SCC extraction only includes internal edges
- Need to restore cross-SCC edges in final trace

### 4. **Topological Ordering**
SCCs must be processed in topological order:
- Dependencies: If SCC A → SCC B, process A before B
- This is guaranteed by `sccs_in_topo_order()`

---

## Data Structures

### Input: SCC Traces
```python
scc_trace_dict: Dict[str, EliminationTrace]
# Key: SCC hash
# Value: Trace for that SCC subgraph

scc_graph: SCCGraph
# Condensation graph with topological ordering
```

### Output: Stitched Trace
```python
merged_trace: EliminationTrace
# Combined trace for full graph
```

### Intermediate Mappings
```python
# Operation remapping: SCC op_idx → merged op_idx
op_remap: Dict[Tuple[int, int], int]  # (scc_idx, op_idx) → merged_op_idx

# Vertex remapping: SCC vertex → original graph vertex
vertex_to_original: Dict[Tuple[int, int], int]  # (scc_idx, scc_v_idx) → orig_v_idx

# Original vertex → merged vertex index
original_to_merged: Dict[int, int]  # orig_v_idx → merged_v_idx
```

---

## Algorithm Overview

```python
def stitch_scc_traces(scc_graph: SCCGraph,
                     scc_trace_dict: Dict[str, EliminationTrace]) -> EliminationTrace:
    """
    Stitch SCC traces into full graph trace

    Steps:
    1. Build vertex mappings (SCC → original → merged)
    2. For each SCC in topological order:
       a. Remap and append operations
       b. Update operation indices
       c. Append states
       d. Remap vertex_rates
       e. Remap edge_probs and vertex_targets
    3. Add boundary edges between SCCs
    4. Return merged trace
    """
```

---

## Detailed Algorithm

### Step 1: Build Vertex Mappings

```python
# Build SCC vertex → original vertex mapping
vertex_to_original = {}
for scc_idx, scc in enumerate(scc_graph.sccs_in_topo_order()):
    for scc_v_idx in range(scc.size()):
        # Get original vertex pointer from SCC
        orig_vertex_ptr = scc.internal_vertices[scc_v_idx]

        # Find this vertex's index in original graph
        orig_v_idx = find_vertex_index_in_original_graph(orig_vertex_ptr)

        vertex_to_original[(scc_idx, scc_v_idx)] = orig_v_idx
```

**Challenge**: How to get `find_vertex_index_in_original_graph()`?

**Solution**: Build mapping from original graph:
```python
# Before stitching, build original vertex → index mapping
orig_graph = scc_graph.original_graph()
vertex_ptr_to_idx = {}
for v_idx in range(orig_graph.vertices_length()):
    vertex_ptr = get_vertex_ptr_at_index(v_idx)
    vertex_ptr_to_idx[vertex_ptr] = v_idx
```

### Step 2: Initialize Merged Trace

```python
merged = EliminationTrace(
    operations=[],
    vertex_rates=np.zeros(orig_graph.vertices_length(), dtype=int),
    edge_probs=[[] for _ in range(orig_graph.vertices_length())],
    vertex_targets=[[] for _ in range(orig_graph.vertices_length())],
    states=np.zeros((orig_graph.vertices_length(), state_length)),
    starting_vertex_idx=orig_graph.starting_vertex_index(),
    n_vertices=orig_graph.vertices_length(),
    state_length=state_length,
    param_length=param_length,
    reward_length=reward_length,
    is_discrete=is_discrete
)
```

### Step 3: Process Each SCC in Topological Order

```python
op_remap = {}
original_to_merged = {}

for scc_idx, scc in enumerate(scc_graph.sccs_in_topo_order()):
    scc_trace = scc_trace_dict[scc.hash()]

    # 3a. Remap and append operations
    op_offset = len(merged.operations)

    for scc_op_idx, operation in enumerate(scc_trace.operations):
        # Remap operands in this operation
        new_op = remap_operation(operation, op_remap, op_offset)
        merged.operations.append(new_op)

        # Record mapping: (scc_idx, scc_op_idx) → merged_op_idx
        op_remap[(scc_idx, scc_op_idx)] = op_offset + scc_op_idx

    # 3b. Build merged vertex indices
    for scc_v_idx in range(scc_trace.n_vertices):
        orig_v_idx = vertex_to_original[(scc_idx, scc_v_idx)]

        if orig_v_idx not in original_to_merged:
            merged_v_idx = len([v for v in original_to_merged.values()])
            original_to_merged[orig_v_idx] = merged_v_idx
        else:
            merged_v_idx = original_to_merged[orig_v_idx]

        # 3c. Copy state
        merged.states[merged_v_idx] = scc_trace.states[scc_v_idx]

        # 3d. Remap vertex_rates
        scc_rate_op_idx = scc_trace.vertex_rates[scc_v_idx]
        merged_rate_op_idx = op_remap[(scc_idx, scc_rate_op_idx)]
        merged.vertex_rates[merged_v_idx] = merged_rate_op_idx

        # 3e. Remap edge_probs and vertex_targets
        for j, scc_edge_op_idx in enumerate(scc_trace.edge_probs[scc_v_idx]):
            merged_edge_op_idx = op_remap[(scc_idx, scc_edge_op_idx)]
            merged.edge_probs[merged_v_idx].append(merged_edge_op_idx)

            # Remap target vertex
            scc_target_v_idx = scc_trace.vertex_targets[scc_v_idx][j]
            orig_target_v_idx = vertex_to_original[(scc_idx, scc_target_v_idx)]
            merged_target_v_idx = original_to_merged[orig_target_v_idx]
            merged.vertex_targets[merged_v_idx].append(merged_target_v_idx)
```

### Step 4: Add Boundary Edges

```python
# Boundary edges: edges in original graph that cross SCC boundaries
# These are NOT in SCC subgraph traces

for scc in scc_graph.sccs_in_topo_order():
    # For each SCC, check outgoing edges to other SCCs
    for outgoing_scc_edge in scc.outgoing_scc_edges():
        target_scc = outgoing_scc_edge.to

        # Find vertices in source SCC that have edges to target SCC
        for v in scc.internal_vertices:
            orig_v_idx = get_original_index(v)
            merged_v_idx = original_to_merged[orig_v_idx]

            # Check original graph for edges to target SCC
            for edge in get_edges_from_vertex(v):
                if edge.to in target_scc.internal_vertices:
                    # This is a boundary edge!
                    orig_target_idx = get_original_index(edge.to)
                    merged_target_idx = original_to_merged[orig_target_idx]

                    # Add edge to merged trace
                    # Need to create operation for edge weight/probability
                    edge_op_idx = create_edge_operation(merged, edge)
                    merged.edge_probs[merged_v_idx].append(edge_op_idx)
                    merged.vertex_targets[merged_v_idx].append(merged_target_idx)
```

**Challenge**: How to handle parameterized boundary edges?

**Solution**:
- For constant edges: Create CONST operation
- For parameterized edges: Create DOT operation with coefficients

---

## Operation Remapping Logic

```python
def remap_operation(op: Operation,
                   op_remap: Dict,
                   op_offset: int) -> Operation:
    """
    Remap operation indices in operands

    For DOT operation with operands [3, 7, 12]:
    - If these reference other operations in same SCC:
      - Remap using op_offset: [3+offset, 7+offset, 12+offset]

    For PARAM operation:
    - No remapping needed (references parameter array, not operations)

    For binary ops (ADD, MUL, DIV):
    - Remap both operands
    """
    new_op = Operation(op_type=op.op_type)

    if op.op_type == OpType.CONST:
        new_op.value = op.value

    elif op.op_type == OpType.PARAM:
        new_op.operands = op.operands  # No remapping (param index)

    elif op.op_type == OpType.DOT:
        new_op.coefficients = op.coefficients
        new_op.operands = [idx + op_offset for idx in op.operands]

    elif op.op_type in [OpType.ADD, OpType.MUL, OpType.DIV]:
        new_op.operands = [idx + op_offset for idx in op.operands]

    elif op.op_type == OpType.INV:
        new_op.operands = [op.operands[0] + op_offset]

    elif op.op_type == OpType.SUM:
        new_op.operands = [idx + op_offset for idx in op.operands]

    return new_op
```

---

## Correctness Guarantees

### 1. **Operation Dependencies Preserved**
- Topological order ensures dependencies processed first
- Operation remapping maintains relative order within each SCC

### 2. **Vertex Mapping Correct**
- Each original vertex mapped to exactly one merged vertex
- State vectors preserved

### 3. **Edge Structure Preserved**
- Internal SCC edges copied from SCC traces
- Boundary edges restored from original graph

### 4. **Arithmetic Semantics Preserved**
- Operation types unchanged
- Coefficients and constants unchanged
- Only indices updated

---

## Testing Strategy

### Unit Tests

```python
def test_operation_remapping():
    """Test that operation indices are correctly remapped"""
    # Create operations with dependencies
    # Remap with offset
    # Verify operands updated correctly

def test_vertex_remapping():
    """Test vertex index mapping SCC → original → merged"""
    # Build simple graph with 2 SCCs
    # Verify mapping preserves vertex identity

def test_boundary_edges():
    """Test that cross-SCC edges are correctly added"""
    # Create graph with edges between SCCs
    # Verify boundary edges present in stitched trace
```

### Integration Tests

```python
def test_stitched_vs_direct():
    """Compare stitched trace vs direct computation"""
    # Compute trace directly: trace_direct = record_elimination_trace(graph)
    # Compute via stitching: trace_stitched = stitch_scc_traces(...)

    # Evaluate both with same parameters
    # result_direct = evaluate_trace(trace_direct, params)
    # result_stitched = evaluate_trace(trace_stitched, params)

    # Compare vertex_rates
    # assert np.allclose(result_direct['vertex_rates'], result_stitched['vertex_rates'])

    # Compare edge probabilities
    # ...
```

### Property Tests

```python
def test_trace_validity():
    """Verify stitched trace satisfies invariants"""
    # All operation indices valid
    # All vertex indices valid
    # Sum of edge probabilities = 1.0 for each vertex
    # No dangling references
```

---

## Performance Considerations

### Time Complexity
- Building mappings: O(V) where V = vertices in original graph
- Processing SCCs: O(S * (O + E)) where S = num SCCs, O = operations per SCC, E = edges per SCC
- Adding boundary edges: O(B) where B = boundary edges
- **Total**: O(V + S*O + S*E + B) ≈ O(V + O_total + E_total)

Same complexity as direct computation, but with better cache locality for repeated graphs.

### Space Complexity
- Mappings: O(V + O_total)
- Merged trace: O(V + O_total + E_total)
- **Total**: O(V + O + E)

Same as single trace, no duplication.

---

## Implementation Plan

### Week 1: Core Implementation
- **Day 1-2**: Implement vertex mapping logic
- **Day 3-4**: Implement operation remapping
- **Day 5**: Initial integration test

### Week 2: Boundary Edges & Testing
- **Day 1-2**: Implement boundary edge handling
- **Day 3-4**: Unit tests for all components
- **Day 5**: Integration tests (stitched vs direct)

### Week 3: Polish & Integration
- **Day 1-2**: Edge case handling (single SCC, disconnected, etc.)
- **Day 3-4**: Integrate with get_trace_hierarchical()
- **Day 5**: End-to-end testing with real models

---

## Next Steps

1. ✅ Read and understand trace structure (DONE)
2. ✅ Design algorithm (DONE - this document)
3. **NOW**: Implement vertex mapping (start here)
4. Implement operation remapping
5. Implement boundary edge handling
6. Write tests
7. Integrate with hierarchical caching

---

## Open Questions

1. **Parameterized boundary edges**: How to extract coefficients from original graph?
   - **Answer**: Need to access `ptd_edge_parameterized` from original graph vertices

2. **Starting vertex**: Which SCC contains it?
   - **Answer**: The starting vertex is in exactly one SCC, find it during vertex mapping

3. **Absorbing states**: Do they exist in SCC subgraphs?
   - **Answer**: Yes, SCCs can have absorbing states (vertices with no outgoing edges)

4. **Cache key for stitched trace**: Use full graph hash or combination of SCC hashes?
   - **Answer**: Full graph hash (already implemented in Phase 3a)

---

**Status**: Design complete, ready for implementation ✅
**Estimated Implementation Time**: 2-3 weeks
**Risk Level**: Medium-High (complex algorithm, many edge cases)

