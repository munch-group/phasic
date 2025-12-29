# Vertex Indices Implementation - Supporting Duplicate States

**Date**: 2025-12-27
**Status**: ✅ Complete

## Summary

Added `vertex_indices` field to serialization and trace structures to properly support graphs with duplicate states (e.g., trash loops, absorbing states). This fixes the fundamental limitation where state-based vertex lookup would fail when multiple vertices share the same state vector.

## Problem Statement

### Root Cause

The original implementation used **state-based vertex lookup** for cross-graph mapping:

```python
# OLD APPROACH (BROKEN with duplicate states)
state_to_idx = {}
for i, v in enumerate(vertices_list):
    state_tuple = tuple(v.state())
    state_to_idx[state_tuple] = i  # ← Overwrites if duplicate!
```

This approach fails when graphs contain multiple vertices with identical states, such as:
- **Trash loops**: Multiple all-zeros states `[0,0,0,0]`
- **Absorbing states**: Multiple terminal states with same configuration
- **Sister vertices**: Same state appearing in different SCCs

### Manifestation

Two critical failures:

1. **Serialization/Deserialization**: `RuntimeError: Multiple edges to the same vertex!`
   - State-based lookup maps all duplicate-state vertices to a single index
   - Edges to different vertices get collapsed together
   - Reconstruction creates duplicate edges

2. **SCC Stitching**: Cross-graph vertex mapping would fail
   - Metadata regeneration fallback uses state-based lookup
   - Cannot distinguish between vertices with duplicate states
   - Silent data corruption or incorrect trace merging

## Solution: Explicit Vertex Identity

Added `vertex_indices` array to preserve **vertex identity** separate from **state values**:

```python
# NEW APPROACH (CORRECT with duplicate states)
vertex_indices = np.array([v.index() for v in vertices_list], dtype=np.int32)
# Maps enumeration position → original graph vertex index
```

This provides a **unique identifier** for each vertex independent of its state, enabling:
- Correct serialization/deserialization with duplicate states
- Proper cross-graph vertex mapping in SCC stitching
- Preservation of graph structure semantics

## Implementation Details

### 1. EliminationTrace Dataclass

**File**: `src/phasic/trace_elimination.py`

**Changes**:
- Added `vertex_indices: np.ndarray` field to dataclass (line 156)
- Updated docstring to document the field (lines 130-133)
- Populated in trace creation (line 793)

```python
@dataclass
class EliminationTrace:
    """
    ...
    vertex_indices : np.ndarray
        Original vertex indices from source graph (n_vertices,)
        Maps enumeration position → original graph vertex index
        Essential for graphs with duplicate states (e.g., trash loops)
    ...
    """
    states: np.ndarray = field(default_factory=lambda: np.array([]))
    vertex_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    # ...
```

### 2. Trace Recording

**File**: `src/phasic/trace_elimination.py`

**Changes**:
- Extract vertex indices during state extraction (lines 443-446)
- Pass to EliminationTrace constructor (line 793)

```python
# Extract states and vertex indices
state_length = graph.state_length()
states = np.zeros((n_vertices, state_length), dtype=np.int32)
vertex_indices = np.zeros(n_vertices, dtype=np.int32)
for i, v in enumerate(vertices_list):
    states[i, :] = v.state()
    vertex_indices[i] = v.index()

# Create trace
trace = EliminationTrace(
    # ...
    states=states,
    vertex_indices=vertex_indices,
    # ...
)
```

### 3. Graph Serialization

**File**: `src/phasic/__init__.py`

**Changes**:
- Extract vertex indices alongside states (lines 1669-1678)
- Add to serialization output dictionary (line 1781)

```python
# Extract states and create vertex index mapping
states = np.zeros((n_vertices, state_length), dtype=np.int32)
vertex_indices = np.zeros(n_vertices, dtype=np.int32)

for i, v in enumerate(vertices_list):
    state = v.state()
    states[i, :] = state
    vertex_indices[i] = v.index()
    vertex_idx_to_enum[v.index()] = i

return {
    'states': states,
    'vertex_indices': vertex_indices,  # ← NEW
    # ...
}
```

### 4. SCC Stitching Fallback

**File**: `src/phasic/hierarchical_trace_cache.py`

**Changes**:
- Check for `vertex_indices` attribute first (line 1683)
- If present, use it directly (line 1685)
- If absent, validate no duplicate states before fallback (lines 1691-1702)
- Raise descriptive error if duplicates found (lines 1698-1702)

```python
# Infer ordered_vertices: prefer vertex_indices, fallback to state matching
if hasattr(scc_trace, 'vertex_indices') and len(scc_trace.vertex_indices) > 0:
    # Modern approach: use vertex identity from trace
    ordered_vertices = scc_trace.vertex_indices.tolist()
    logger.debug("Using vertex_indices from trace (%d vertices)", len(ordered_vertices))
else:
    # Legacy fallback: match states (FAILS with duplicate states)
    logger.warning("Trace missing vertex_indices - using state matching fallback")

    # Check for duplicate states in trace (would cause ambiguity)
    trace_states = [tuple(scc_trace.states[i]) for i in range(scc_trace.n_vertices)]
    if len(trace_states) != len(set(trace_states)):
        from collections import Counter
        state_counts = Counter(trace_states)
        duplicates = {s: c for s, c in state_counts.items() if c > 1}
        logger.error("Trace has duplicate states but no vertex_indices: %s", duplicates)
        raise ValueError(
            "Cannot use state-based vertex matching with duplicate states. "
            "Trace must include vertex_indices field. "
            f"Found duplicate states: {duplicates}"
        )

    # State-based matching (original code - only if no duplicates)
    # ... existing fallback code ...
```

## Key Design Decisions

### 1. Backward Compatibility

**Kept state-based fallback** for legacy traces without `vertex_indices`:
- Allows old cached traces to still work
- Validates no duplicate states before use
- Raises clear error if ambiguity detected
- Gradual migration path for existing codebases

### 2. Fail-Fast Error Handling

**Explicit validation** prevents silent data corruption:
- Check for duplicate states when using fallback
- Descriptive error messages guide users to solution
- No ambiguous behavior with duplicate states

### 3. Semantic Preservation

**States remain clean** - no type indicators or pollution:
- `graph.states().T` can still be used directly as rewards
- State vectors preserve their original meaning
- No workarounds that leak implementation details

## Testing

### Basic Functionality

```python
from phasic import Graph
import numpy as np

# Graph with duplicate states
g = Graph(2)
start = g.starting_vertex()
v1 = g.find_or_create_vertex([0, 0])
v2 = g.create_vertex([0, 0])  # Duplicate!
v3 = g.create_vertex([1, 0])

# Test serialization
data = g.serialize()
assert 'vertex_indices' in data
assert data['vertex_indices'].shape == (4,)
assert np.array_equal(data['vertex_indices'], [0, 1, 2, 3])

# Test trace recording
from phasic.trace_elimination import record_elimination_trace
trace = record_elimination_trace(g, param_length=0)
assert hasattr(trace, 'vertex_indices')
assert trace.vertex_indices.shape == (4,)
```

### Coalescent Model

```python
# Test with real coalescent model (from examples/test.py)
def coalescent(state):
    transitions = []
    for i in range(state.size):
        for j in range(i, state.size):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[i+j+1] += 1
            transitions.append((new, [state[i]*(state[j]-same)/(1+same)]))
    return transitions

nr_samples = 4
base_graph = Graph(coalescent, ipv=[nr_samples]+[0]*(nr_samples-1))

# Serialization works
data = base_graph.serialize()
assert 'vertex_indices' in data

# No NaN warnings
exp = base_graph.expectation()  # ✓ No warnings!
```

## Impact

### Fixed Issues

1. ✅ **Serialization with duplicate states** - No more duplicate edge errors
2. ✅ **SCC stitching robustness** - Proper cross-graph mapping with fail-fast validation
3. ✅ **State semantics preservation** - `graph.states().T` works directly as rewards
4. ✅ **NaN handling** - Combined with earlier `0 × ∞ = 0` fix, trash states work correctly

### API Compatibility

**Zero breaking changes**:
- Serialization adds new field, doesn't remove old ones
- Trace recording automatically populates vertex_indices
- SCC stitching falls back gracefully for old traces
- Existing code continues to work unchanged

### Performance

**Negligible overhead**:
- One additional integer array allocation
- No computational cost (just copying indices)
- Same O(n) memory as states array

## Related Work

This implementation completes the optimization of `joint_index=True` mode:

1. **JOINT_INDEX_OPTIMIZATION.md** - 3-35x speedup using `expected_sojourn_time()`
2. **This document** - Duplicate state support for trash loops
3. **Earlier work** - NaN handling for `0 × ∞` in elimination trace

Together, these changes enable:
- Fast joint probability computations
- Robust handling of reward transformations
- Support for arbitrary graph structures (including trash states)

## Files Modified

1. `src/phasic/trace_elimination.py` - EliminationTrace dataclass, trace recording
2. `src/phasic/__init__.py` - Graph.serialize() method
3. `src/phasic/hierarchical_trace_cache.py` - SCC stitching fallback

**Total changes**: 3 files, ~50 lines added/modified

---

**Result**: Vertex identity is now properly preserved through serialization and trace structures, enabling correct handling of graphs with duplicate states while maintaining full backward compatibility.
