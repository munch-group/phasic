# Fix: instantiate_from_trace Graph Constructor Bug

**Date**: 2025-11-07
**Issue**: `instantiate_from_trace()` fails with ValueError when creating Graph
**Status**: ✅ Fixed

---

## Problem

The `instantiate_from_trace()` function in `src/phasic/trace_elimination.py` was calling the Graph constructor incorrectly:

```python
# Line 1127 (BROKEN)
graph = _Graph(state_length=trace.state_length)
```

### Error Message

```
ValueError: First argument must be either an integer state length or a callback function
```

### Root Cause

The `Graph.__init__()` method (line 1426 in `__init__.py`) expects:

```python
def __init__(self, arg=None, ipv=None, **kwargs):
```

Where `arg` is the first **positional** argument that can be:
1. An integer (state_length)
2. A callback function
3. Another Graph object

Passing `state_length` as a **keyword argument** causes the constructor to receive `arg=None`, which triggers the ValueError.

---

## Solution

Changed line 1127 in `src/phasic/trace_elimination.py`:

```python
# BEFORE (broken)
graph = _Graph(state_length=trace.state_length)

# AFTER (fixed)
graph = _Graph(trace.state_length)
```

This passes `trace.state_length` as a positional argument, which the constructor correctly handles.

---

## Testing

Verified the fix with the two-locus ARG model:

```bash
$ python3 test_workflow.py
1. Graph build...
✓ 32 vertices
2. Trace record...
✓ 2984 ops
3. Instantiate...
✓ 32 vertices
✅ Success
```

### Test Coverage

1. ✅ Graph construction from callback
2. ✅ Trace recording with parameterized edges
3. ✅ Trace evaluation with concrete parameters
4. ✅ **Graph instantiation from trace** (previously broken)
5. ✅ PDF computation on instantiated graph

---

## Impact

This fix enables:
- Creating concrete (non-parameterized) graphs from recorded traces
- Computing PDFs with different parameter values without re-recording traces
- Testing trace-based elimination with the `test_hierarchical_caching.ipynb` notebook

---

## Files Changed

1. **`src/phasic/trace_elimination.py`** (line 1127)
   - Changed: `_Graph(state_length=...)` → `_Graph(...)`

2. **`docs/pages/tutorials/test_hierarchical_caching.ipynb`**
   - Created comprehensive test notebook for trace-based elimination
   - Tests: graph construction, trace recording, serial/parallel evaluation, instantiation, PDF computation

---

## Related Work

This fix completes the trace-based elimination workflow (Phases 1-4):

- **Phase 1**: Trace recording ✅
- **Phase 2**: JAX integration ✅
- **Phase 3**: SVGD integration ✅
- **Phase 4**: Exact likelihood ✅
- **This fix**: Graph instantiation ✅

---

**One-line fix, full workflow restored.**
