# Queue Implementation Bug Fix

**Date**: 2025-11-02
**Status**: ✅ FIXED

---

## Problem

The `reward_transform()` function was crashing when called after trace recording was enabled. The user reported:
> "The reward transformation somehow gets the indexing (or order) of vertices wrong"

Test case that crashed:
```python
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=4)
graph.update_parameterized_weights(theta)
rewards = graph.states().T[:-2]
rev_trans = phasic.Graph(graph.reward_transform(rewards[0]))  # CRASH HERE
```

---

## Root Cause

**File**: `src/c/phasic.c`

The queue implementation (lines 186-231) didn't properly maintain the `queue->tail` pointer:

1. **queue_enqueue** (lines 186-207):
   - Traversed the entire linked list to find the tail (O(n) operation)
   - Never set `queue->tail` field

2. **queue_dequeue** (lines 209-226):
   - Didn't clear `queue->tail` when dequeueing the last element

3. **queue_empty** (lines 228-231):
   - Checked `queue->ll == NULL` instead of `queue->tail == NULL`

The `struct ptd_queue` has a `tail` field (line 81), and other parts of the code expect it to be maintained correctly. The original commented-out implementation (lines 6246-6278) showed the correct behavior.

---

## Solution

Fixed all three queue functions to properly maintain the `tail` pointer:

### 1. queue_enqueue (lines 186-207)

**Before**:
```c
if (queue->ll == NULL) {
    queue->ll = node;
} else {
    // Find tail - O(n) traversal!
    struct ptd_ll *tail = queue->ll;
    while (tail->next != NULL) {
        tail = tail->next;
    }
    tail->next = node;
}
// queue->tail NOT SET
```

**After**:
```c
if (queue->tail != NULL) {
    queue->tail->next = node;
} else {
    queue->ll = node;
}

queue->tail = node;  // Maintain tail pointer
```

**Improvement**: O(n) → O(1) enqueue operation

### 2. queue_dequeue (lines 209-226)

**Before**:
```c
queue->ll = node->next;
free(node);
// queue->tail NOT CLEARED when empty
```

**After**:
```c
queue->ll = node->next;

if (queue->tail == node) {
    queue->tail = NULL;  // Clear tail when dequeueing last element
}

free(node);
```

### 3. queue_empty (lines 228-231)

**Before**:
```c
return (queue->ll == NULL) ? 1 : 0;
```

**After**:
```c
return (queue->tail == NULL) ? 1 : 0;
```

**Note**: Matches original implementation at line 6277

---

## Test Results

### Before Fix
```bash
$ python test_crash.py
Creating graph...
Updating weights...
Computing expectation...
Expectation: 6.15
Getting states...
Rewards shape: (3, 6)
Applying reward transform...
[CRASH - kernel died]
```

### After Fix
```bash
$ python test_crash.py
Creating graph...
Updating weights...
Computing expectation...
Expectation: 6.15
Getting states...
Rewards shape: (3, 6)
Applying reward transform...
Done!
```

**SUCCESS** ✓

---

## Impact

**Fixed functions**:
- `graph.reward_transform(rewards)` - No longer crashes
- `graph.expectation()` - Still works (already fixed NAN bug separately)
- `graph.expected_waiting_time()` - Still works

**Performance improvement**:
- Queue enqueue: O(n) → O(1) per operation
- Critical for SCC/topological sort in large graphs

**No breaking changes**:
- All existing tests still pass
- API unchanged
- Backward compatible

---

## Files Modified

- `src/c/phasic.c` (lines 186-231)
  - queue_enqueue: 13 lines → 12 lines (simplified and faster)
  - queue_dequeue: 12 lines → 15 lines (added tail clearing)
  - queue_empty: 1 line changed (check tail instead of ll)

---

## Related Fixes

This session also fixed:

1. **NAN expectation bug** (separate issue):
   - Removed incorrect NAN terminator in `ptd_build_reward_compute_from_trace` (line 10447-10461)
   - Changed `res->length = cmd_idx + 1` → `res->length = cmd_idx`

---

## Conclusion

The reward_transform crash was caused by incorrect queue implementation in the new utility functions. The traditional path was correct (as user indicated). The fix:

1. ✅ Maintains `queue->tail` pointer correctly
2. ✅ Improves performance (O(n) → O(1) enqueue)
3. ✅ No API changes
4. ✅ All tests pass

**Implementation complete**: 2025-11-02
**Status**: ✅ PRODUCTION READY
