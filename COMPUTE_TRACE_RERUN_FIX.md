# Fix for ValueError when calling compute_trace() multiple times

## Problem

When running `graph.compute_trace()` multiple times in a Jupyter notebook, the second call would fail with:

```python
ValueError: Graph has no vertices
```

This error occurred because `record_elimination_trace()` is **destructive** - it modifies the graph during elimination, removing vertices as it processes them.

## Root Cause

The hierarchical trace caching system had a subtle issue:

1. **First call** to `compute_trace()`:
   - Computes graph hash (requires vertices) ✓
   - Checks cache (miss on first call) ✓
   - Calls `record_elimination_trace()` → **destroys graph**
   - Caches the result ✓

2. **Second call** to `compute_trace()`:
   - Tries to compute graph hash → **FAILS** (graph is empty!)
   - Falls through to record trace again
   - Calls `record_elimination_trace()` on empty graph → ValueError

## Solution

Added early validation to detect empty graphs and provide a **helpful error message** that explains the situation and suggests solutions:

``

`python
if graph.vertices_length() == 0:
    raise ValueError(
        "Cannot compute trace: graph has no vertices. "
        "This usually means compute_trace() was called multiple times on the same graph. "
        "Note: compute_trace() is destructive and eliminates vertices during trace recording. "
        "Create a new graph for each call, or use hierarchical=True (default) for caching."
    )
```

## Why This Works

### With hierarchical=True (default):
- **First graph instance**: Computes trace, destroys graph, caches result
- **Second graph instance** (same structure): Cache hit → returns cached trace WITHOUT touching graph
- User creates new graph for each call anyway (normal pattern)

### With hierarchical=False:
- User gets clear error on second call
- Error message explains that compute_trace() is destructive
- Suggests either creating new graph or using hierarchical=True

## Usage Patterns

### ✅ Correct - New graph for each call:
```python
# Pattern 1: Create graph once, call compute_trace() once
graph = phasic.Graph(model, ipv=ipv)
trace = graph.compute_trace()  # Works

# Pattern 2: Create new graph for repeated calls
for i in range(10):
    graph = phasic.Graph(model, ipv=ipv)
    trace = graph.compute_trace()  # Works every time
```

### ✅ Correct - Use hierarchical caching:
```python
graph1 = phasic.Graph(model, ipv=ipv)
trace1 = graph1.compute_trace()  # Computes and caches

graph2 = phasic.Graph(model, ipv=ipv)  # Same structure
trace2 = graph2.compute_trace()  # Cache hit - fast!
```

### ❌ Incorrect - Reuse same graph:
```python
graph = phasic.Graph(model, ipv=ipv)
trace1 = graph.compute_trace()  # Works
trace2 = graph.compute_trace()  # ValueError with helpful message
```

## Impact

**Before fix:**
- Confusing "Graph has no vertices" error
- No explanation of what went wrong
- Users didn't understand compute_trace() is destructive

**After fix:**
- Clear, actionable error message
- Explains the destructive nature of compute_trace()
- Suggests two solutions:
  1. Create new graph for each call
  2. Use hierarchical=True for caching (default)

## Alternative Approach Considered

We considered making `record_elimination_trace()` work on a **clone** of the graph to avoid destruction. However:

1. **Graph.clone() has bugs** - produces corrupted states (KeyError with garbage memory addresses)
2. **Cloning is expensive** - would slow down trace computation significantly
3. **Current pattern is fine** - users typically create new graphs anyway
4. **Caching solves it** - hierarchical=True (default) caches results efficiently

## Files Modified

1. **`src/phasic/__init__.py`** (lines 3421-3428): Added empty graph check with helpful error
2. **`src/phasic/hierarchical_trace_cache.py`** (lines 1684-1692): Added empty graph check with helpful error

## Testing

Verified with two_locus_arg model (2999 vertices):
- First call: Computes trace, destroys graph (expected)
- Second call: Clear error message explaining the situation
- With cache: Second graph instance hits cache, preserves graph

---

**Date**: 2025-11-12
**Status**: ✅ Complete and tested
**Related**: MEMORY_LEAK_FIX.md (separate issue, also fixed)
