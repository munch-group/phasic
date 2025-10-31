# Vertex Bypass Bug: Fix Attempts Summary

**Date**: 2025-10-28
**Status**: ❌ NOT RESOLVED - Bug confirmed but fix attempts unsuccessful

---

## Bug Confirmed

**Evidence**: Simple chain test `0→1→2→3→4` with rewards `[1,0,0,0,1]` (3 consecutive bypasses):
- PDF integral: 0.000 (should be ≈1.0)
- Sample mean: 0.000 (all samples are 0)
- **Graph is completely broken**

**Pattern**: Works for 0-2 consecutive bypasses, fails for 3+ consecutive bypasses.

---

## Root Cause Identified

**Location**: `src/c/phasic.c` lines 2151-2312 (`_ptd_graph_reward_transform()`)

**Problem**: When processing zero-reward vertices in sequence, the algorithm reads stale parent information from `vertex_parents[]` array. After bypassing vertex i, the parent lists of vertices j > i still reference vertex i, causing incorrect edge redirections.

**Example**:
```
Chain: 0 → 1(r=0) → 2(r=0) → 3(r=0) → 4(r=1)

Iteration 1: Bypass vertex 1
  - Redirect 0→1→2 to 0→2 ✓
  - vertex_parents[2] still shows [1, ...] ❌ (should show [0, ...])

Iteration 2: Bypass vertex 2
  - Read vertex_parents[2] = [1, ...]
  - Try to process parent 1, but its edges were already modified!
  - Result: Inconsistent graph structure
```

---

## Fix Attempts (All Failed)

### Attempt 1: Single-Pass with Future Vertex Parent Updates
**Approach**: After bypassing vertex i, scan all vertices j > i and update their parent lists to replace i with i's parents.

**Implementation**: Lines 2286-2361 (added parent list update loop)

**Result**: ❌ Failed - Still broken
- **Issue**: Looking for edges from grandparent to j, but those edges don't exist in the expected form after bypass operations

### Attempt 2: Single-Pass with Current Vertex Parent Repair
**Approach**: At the start of processing vertex i, check if any parents have been bypassed and replace them with their parents.

**Implementation**: Lines 2159-2212 (added parent repair loop before bypass)

**Result**: ❌ Failed - Still broken
- **Issue**: Similar to attempt 1 - can't find the expected edges to update back-pointers

### Attempt 3: Single-Pass Skipping Bypassed Parents
**Approach**: When processing vertex i, skip any parents that have `rewards[parent]==0` (already bypassed).

**Implementation**: Lines 2171-2173 (added skip check)

**Result**: ❌ Failed - Worse than before
- **Issue**: Skipping all bypassed parents means NO edges get created at all! If vertex i only has bypassed parents, the vertex becomes completely disconnected.
- Graph had correct vertex count (2) but all edges broken (samples=0, PDF=0)

### Attempt 4: Multi-Pass with Skip Bypassed Parents
**Approach**: Process one zero-reward vertex per pass, restart loop after each bypass, skip bypassed parents.

**Implementation**:
- Lines 2151-2158 (added while loop + changed flag)
- Lines 2179-2183 (skip bypassed parents check)
- Lines 2306-2312 (mark bypassed, break, restart)

**Result**: ❌ Failed - Still broken
- **Issue**: Even with multi-pass and skipping, the fundamental problem remains - when ALL of a vertex's parents are bypassed, no edges get created

**Final test results**:
```
No bypass:   PDF integral=1.000, samples mean=3.06 ✓
1 bypass:    PDF integral=0.999, samples mean=2.08 ✓
2 bypasses:  PDF integral=0.992, samples mean=1.15 ✓
3 bypasses:  PDF integral=0.000, samples mean=0.00 ❌
```

---

## Why All Approaches Failed

### Fundamental Issue
The bypass algorithm has a chicken-and-egg problem:
1. **Need updated parent lists** to know which parents to process
2. **Parent lists are updated** by the bypass operations themselves
3. **Can't update before bypass** because we don't know the new edges yet
4. **Can't skip bypassed parents** because then we lose edges entirely

### The Core Problem
When vertex i has only bypassed parents, the "skip bypassed parents" approach creates a vertex with NO incoming edges! The correct behavior would be to look at the parents of those bypassed parents (grandparents), but that requires recursive parent resolution, which the current algorithm doesn't support.

**Example**:
```
After bypassing vertices 1 and 2:
  0 → 1(bypassed) → 2(bypassed) → 3 → 4

When processing vertex 3:
  - vertex_parents[3] = [2]
  - Parent 2 is bypassed, skip it
  - NO edges created for vertex 3!
  - Result: Disconnected graph
```

The correct behavior would be:
```
  - vertex_parents[3] = [2]
  - Parent 2 is bypassed, look at parents of 2 → find parent 0
  - Create edge 0 → 3
```

But this requires either:
- Recursive parent lookup (complex, not implemented)
- Proper parent list updates after each bypass (attempted, failed)
- Complete parent list rebuilding (would be very expensive)

---

## What Works

- ✅ 0-2 consecutive zero-reward vertices (PDF normalized, samples correct)
- ✅ Dense reward vectors (>75% non-zero)
- ✅ Original single-pass code for sparse cases with ≤2 consecutive zeros

## What's Broken

- ❌ 3+ consecutive zero-reward vertices
- ❌ Very sparse reward vectors (>90% zeros)
- ❌ Coalescent model features 2-4 (SVGD bias +20%)

---

## Recommended Actions

### Option 1: Report to Maintainers
This is a deep algorithmic bug that requires expertise in the graph elimination algorithm. The detailed analysis in this document and `/tmp/VERTEX_BYPASS_BUG_REPORT.md` provides a solid bug report.

### Option 2: Use Workaround
Until fixed:
1. **Avoid very sparse rewards**: Don't use reward vectors with >75% zeros
2. **Use epsilon instead of zero**: Replace `reward=0` with `reward=0.001`
3. **Aggregate features**: Combine sparse features to reduce zero density
4. **Test first**: Check PDF normalization before using new reward patterns

### Option 3: Alternative Fix Approach
A more invasive fix might work:
1. **Complete parent list rebuild**: After each bypass, rebuild all parent lists from scratch by scanning all edges
   - Expensive: O(n²) per bypass
   - But would guarantee correctness

2. **Eliminate all zero-reward vertices first**: Before any bypass, collect all zero-reward vertices and process them in topological order
   - Requires topological sort
   - More complex but might avoid stale parent issues

---

## Performance Impact of Workaround

**Using epsilon (0.001) instead of zero**:
- ✅ PDF will be normalized
- ✅ SVGD will converge correctly
- ⚠️ Small computational overhead (vertices not eliminated, larger graph)
- ⚠️ Numerical precision: very small rewards may cause floating-point issues

**Testing**:
```python
# Instead of:
rewards = np.array([1, 0, 0, 0, 1])

# Use:
rewards = np.array([1, 0.001, 0.001, 0.001, 1])
```

---

## Files Modified During Investigation

**C code**:
- `src/c/phasic.c` lines 2148-2395 (bypass loop)
  - Current state: Multi-pass with skip bypassed parents
  - Can be reverted to original if needed

**Test files created**:
- `/tmp/test_bypass_order_bug.py`
- `/tmp/test_vertex_bypass_bug.py`
- `/tmp/test_bypass_sampling.py`
- `/tmp/test_bypass_print_debug.py`
- `/tmp/test_reward_scaling_all_features.py`

**Documentation**:
- `/tmp/VERTEX_BYPASS_BUG_REPORT.md` - Initial bug report
- `/tmp/SPARSE_REWARDS_BUG_REPORT.md` - Evidence with coalescent model
- `/tmp/VERTEX_BYPASS_COMPLETE_DIAGNOSIS.md` - Comprehensive analysis
- `/tmp/VERTEX_BYPASS_FIX_ATTEMPTS_SUMMARY.md` - This document

---

## Complexity Analysis

**Original algorithm**: O(n · edges · parents) single pass

**Attempted fixes**:
- Attempt 1-3: O(n · edges · parents) single pass
- Attempt 4: O(n² · edges · parents) multi-pass worst case

**Proposed complete rebuild**: O(n³) - rebuild parent lists after each of n bypasses

---

## Conclusion

This bug is **confirmed** and **well-documented**, but **not yet fixed**. Multiple sophisticated approaches were attempted, each addressing different aspects of the problem, but the fundamental issue of maintaining consistent parent-child relationships during sequential bypasses remains unsolved.

The bug requires either:
1. Deep expertise in the graph elimination algorithm
2. More invasive changes (complete parent list rebuilding)
3. Alternative algorithm design (topological processing)

**Recommendation**: Use workaround (epsilon instead of zero) until a proper fix can be developed by someone with deep knowledge of the elimination algorithm internals.
