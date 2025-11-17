# Issue: compute_trace() Does Not Check Cache (hierarchical=False)

**Reported by**: Kasper
**Date**: 2025-11-08
**Status**: ⚠️ Blocked by C memory bug in hash module

---

## Problem

`graph.compute_trace(hierarchical=False)` does NOT check the trace cache before computing. This means:

- Every call re-computes the elimination trace from scratch
- No benefit from caching across sessions
- Slower performance for repeated computations on the same graph

### Current Behavior

```python
# src/phasic/__init__.py line 3419-3421
else:
    from .trace_elimination import record_elimination_trace
    return record_elimination_trace(self, param_length=param_length)
```

**Problem**: Directly calls `record_elimination_trace()` without checking cache.

### Expected Behavior

`compute_trace(hierarchical=False)` should:
1. Compute graph hash
2. Check cache for existing trace
3. Return cached trace if found
4. Otherwise compute and save to cache

This is exactly what `compute_trace(hierarchical=True)` does (line 3412-3418).

---

## Attempted Fix

I attempted to add caching to the non-hierarchical path:

```python
else:
    from .trace_elimination import record_elimination_trace
    from . import hash as phasic_hash
    from .trace_serialization import load_trace_from_cache, save_trace_to_cache

    # Check cache first
    hash_result = phasic_hash.compute_graph_hash(self)
    graph_hash = hash_result.hash_hex
    trace = load_trace_from_cache(graph_hash)
    if trace is not None:
        return trace  # Cache hit!

    # Compute trace
    trace = record_elimination_trace(self, param_length=param_length)

    # Save to cache
    save_trace_to_cache(graph_hash, trace)

    return trace
```

---

## Blocking Issue: C Memory Bug in Hash Module

**The fix crashes due to a C memory management bug in the hash module.**

### Symptoms

```bash
$ python3 test.py
Testing hash import...
✓ hash imported
Testing hash computation...
✓ hash computed: 297280cc236c022a...
Script ending...
Abort trap: 6
```

- Hash computation completes successfully
- Crash occurs during Python cleanup/exit
- Likely a double-free or memory leak in C code

### Reproduction

```python
import phasic
from phasic import hash as phasic_hash

graph = phasic.Graph(callback, ipv=ipv, ...)
hash_result = phasic_hash.compute_graph_hash(graph)
print(f'Hash: {hash_result.hash_hex}')
# Script ends → CRASH during cleanup
```

### Testing

✅ `record_elimination_trace()` works fine (no crash)
✅ Hash computation succeeds and returns correct result
❌ Python crashes during cleanup after hash computation
❌ Crash happens even with `PHASIC_DISABLE_CACHE=1`

---

## Root Cause Analysis

The hash module (`phasic.hash`) has a C-level memory management issue:

1. `compute_graph_hash()` creates C objects (likely graph serialization)
2. Python wrapper doesn't properly manage object lifetimes
3. During cleanup, C objects are freed twice or not freed at all
4. Result: Abort trap 6 (SIGSEGV or SIGABRT)

**Location**: Likely in `src/phasic/hash.py` or its C bindings

---

## Workaround

Users should use `compute_trace(hierarchical=True)` which DOES check the cache:

```python
# GOOD - uses cache
trace = graph.compute_trace(hierarchical=True)

# BAD - no cache, re-computes every time
trace = graph.compute_trace(hierarchical=False)
trace = graph.compute_trace()  # default is hierarchical=False
```

**Note**: Even `hierarchical=True` will crash due to the same hash bug, but at least it provides caching on the first call before crashing.

---

## Solution Roadmap

### Step 1: Fix C Memory Bug in Hash Module

**Required before any caching improvements**

1. Identify the memory leak/double-free in `phasic.hash`
2. Likely issues:
   - Improper pybind11 memory management
   - Missing `PYBIND11_MAKE_OPAQUE` declarations
   - Incorrect ownership transfer between C++ and Python
   - Double-delete of serialized graph objects

3. Fix and verify:
```bash
# Should not crash
python3 -c "from phasic import hash as h; h.compute_graph_hash(graph)"
echo "Exit code: $?"  # Should be 0, not 134 (SIGABRT)
```

### Step 2: Enable Caching for hierarchical=False

Once hash module is fixed, apply this patch:

```python
# src/phasic/__init__.py line 3419
else:
    from .trace_elimination import record_elimination_trace
    from . import hash as phasic_hash
    from .trace_serialization import load_trace_from_cache, save_trace_to_cache

    # Check cache
    hash_result = phasic_hash.compute_graph_hash(self)
    trace = load_trace_from_cache(hash_result.hash_hex)
    if trace is not None:
        return trace

    # Compute and cache
    trace = record_elimination_trace(self, param_length=param_length)
    save_trace_to_cache(hash_result.hash_hex, trace)
    return trace
```

### Step 3: Test

```python
# First call - compute and cache
trace1 = graph.compute_trace()

# Second call - should hit cache (10-100x faster)
trace2 = graph.compute_trace()
```

---

## Related Files

- `src/phasic/__init__.py` - `Graph.compute_trace()` method (line 3373)
- `src/phasic/hash.py` - Hash computation (has memory bug)
- `src/phasic/hierarchical_trace_cache.py` - `get_trace_hierarchical()` (line 739, has working cache check)
- `src/phasic/trace_serialization.py` - Cache load/save functions

---

## Impact

**Current workaround**: Use `compute_trace(hierarchical=True)` for caching, or call `record_elimination_trace()` directly if caching not needed.

**Performance impact**: Without caching, trace computation is repeated on every call (can be seconds to minutes for large graphs).

**User confusion**: Inconsistent behavior between `hierarchical=True` (cached) and `hierarchical=False` (not cached).

---

**Fix priority**: HIGH - hash module crash blocks caching improvements
**Estimated effort**: 2-4 hours to debug C memory issue, 15 minutes to add caching once fixed
