# instantiate_from_trace() Bug Fix

## Problem

`instantiate_from_trace()` was returning PDF=0.0 for all parameter values and observation times, even though trace evaluation was correct.

## Root Causes Identified

### 1. Wrong starting_vertex_idx (FIXED)

**Bug**: Trace recording was using state lookup to find starting vertex:
```python
start_state = tuple(graph.starting_vertex().state())
starting_vertex_idx = state_to_idx[start_state]
```

**Problem**: Multiple vertices can have the same state after elimination! For example:
- Vertex 0: state=[0] (functional starting vertex with outgoing edges)
- Vertex 2: state=[0] (absorbing vertex with rate=0)

The `state_to_idx` dictionary would map state `[0]` to vertex 2 (last one added), even though vertex 0 was the actual starting vertex.

**Additional Discovery**: `graph.starting_vertex()` returns a sentinel vertex that is NOT in the `graph.vertices()` list. So we can't use object identity to find it.

**Fix**: Use convention that vertex 0 in the vertices list is always the starting vertex:
```python
# The starting vertex is always the first vertex in the elimination order (index 0)
starting_vertex_idx = 0
```

**Location**: `src/phasic/trace_elimination.py` line 733

### 2. state_to_vertex collision in instantiate_from_trace() (FIXED)

**Bug**: `instantiate_from_trace()` was using state-to-vertex mapping:
```python
state_to_vertex = {}
for i in range(trace.n_vertices):
    state = tuple(trace.states[i].tolist())
    if state not in state_to_vertex:
        v = graph.find_or_create_vertex(trace.states[i].tolist())
        state_to_vertex[state] = v
```

**Problem**: Same as above - vertices 0 and 2 both have state `[0]`, so only one vertex was created!

**Fix**: Use index-to-vertex mapping instead:
```python
idx_to_vertex = {}
# Get or create starting vertex
start_idx = trace.starting_vertex_idx
start_vertex = graph.starting_vertex()
idx_to_vertex[start_idx] = start_vertex

# Create all other vertices
for i in range(trace.n_vertices):
    if i not in idx_to_vertex:
        v = graph.find_or_create_vertex(trace.states[i].tolist())
        idx_to_vertex[i] = v
```

**Location**: `src/phasic/trace_elimination.py` lines 1145-1166

## Test Results

### Before Fix
```
PDF evaluation at t=0.5:
     θ |       Direct |        Trace |    Error %
------------------------------------------------------
  0.50 |     0.391112 |     0.000000 |    100.00%
  1.00 |     0.611117 |     0.000000 |    100.00%
  1.50 |     0.715264 |     0.000000 |    100.00%
  2.00 |     0.743203 |     0.000000 |    100.00%
  2.50 |     0.723046 |     0.000000 |    100.00%
  3.00 |     0.674429 |     0.000000 |    100.00%
```

### After Fix
```
PDF evaluation at t=0.5:
     θ |       Direct |        Trace |    Error %
------------------------------------------------------
  0.50 |     0.391112 |     0.391112 |      0.00%
  1.00 |     0.611117 |     0.611117 |      0.00%
  1.50 |     0.715264 |     0.715264 |      0.00%
  2.00 |     0.743203 |     0.743203 |      0.00%
  2.50 |     0.723046 |     0.723046 |      0.00%
  3.00 |     0.674429 |     0.674429 |      0.00%
```

✅ **Trace-based PDF now works perfectly!**

## Remaining Issue: SVGD Still Fails

SVGD tests still show 97-98% error even though trace-based PDF is now correct:
```
Test 1: Simple Exponential (θ = 2.0, n = 10000)
  True θ:      2.000
  SVGD θ_mean: 0.052
  Relative error: 97.4% ✗
```

**Why?** SVGD uses `GraphBuilder` (C++) to deserialize graphs from JSON, NOT traces. GraphBuilder likely has the same starting_vertex_idx bug and needs to be fixed separately.

## Next Steps

1. ✅ Fix trace recording to use starting_vertex_idx = 0
2. ✅ Fix instantiate_from_trace() to use index-to-vertex mapping
3. ⏳ Fix GraphBuilder (C++) to correctly handle starting_vertex_idx
4. ⏳ Re-run SVGD tests to verify fix

## Files Modified

- `src/phasic/trace_elimination.py` - Fixed starting_vertex_idx lookup and instantiate_from_trace mapping
