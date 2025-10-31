# Vertex Bypass Bug - Complete Debugging Package

**Date**: 2025-10-28
**Bug Status**: Confirmed, not fixed
**Test Case**: Chain graph with 3+ consecutive zero-reward vertices

---

## Quick Start

1. **Run diagnostic script**:
   ```bash
   python /tmp/diagnose_vertex_bypass.py
   ```

2. **Read analysis**:
   - `/tmp/VERTEX_BYPASS_BUG_REPORT.md` - Initial bug report with evidence
   - `/tmp/VERTEX_BYPASS_COMPLETE_DIAGNOSIS.md` - Detailed technical analysis
   - `/tmp/VERTEX_BYPASS_FIX_ATTEMPTS_SUMMARY.md` - All fix attempts and why they failed

3. **Add C debugging** (see `/tmp/C_DEBUGGING_GUIDE.md`):
   - Add printf statements to `src/c/phasic.c`
   - Rebuild and run diagnostic script
   - Analyze output to see exact execution flow

---

## Files in This Package

| File | Purpose |
|------|---------|
| `diagnose_vertex_bypass.py` | Python diagnostic script with detailed output |
| `VERTEX_BYPASS_BUG_REPORT.md` | Initial bug discovery with test evidence |
| `VERTEX_BYPASS_COMPLETE_DIAGNOSIS.md` | Comprehensive technical analysis |
| `VERTEX_BYPASS_FIX_ATTEMPTS_SUMMARY.md` | All 4 fix attempts and failure analysis |
| `C_DEBUGGING_GUIDE.md` | Where to add printf debugging in C code |
| `test_bypass_order_bug.py` | Simple test demonstrating the bug |
| `test_vertex_bypass_bug.py` | Detailed test with multiple bypass patterns |
| `test_bypass_sampling.py` | Tests PDF vs sampling behavior |

---

## The Bug in One Sentence

**When 3+ consecutive vertices have reward=0, the bypass algorithm reads stale parent pointers from `vertex_parents[]` array that still reference previously-bypassed vertices, causing incorrect edge redirections and a broken graph.**

---

## Critical Code Location

**File**: `src/c/phasic.c`
**Function**: `_ptd_graph_reward_transform()`
**Lines**: ~2151-2312 (with attempted fixes) or ~2151-2294 (original)

**The Problem Loop**:
```c
for (size_t i = 0; i < vertices_length; ++i) {
    if (rewards[i] != 0) continue;

    // Process parents of vertex i
    for (size_t p = 0; p < vertex_parents_length[i]; ++p) {
        struct arr_p me_to_parent = vertex_parents[i][p];
        struct ptd_vertex *parent_vertex = me_to_parent.p;  // ← STALE!

        // If parent_vertex was bypassed in iteration j < i,
        // this pointer is now invalid for edge operations
    }
}
```

---

## Data Structures Involved

```c
// Built once at initialization (lines 2135-2146)
// NEVER updated during bypass operations!
struct arr_p *vertex_parents[vertices_length];  // Parent lists
size_t vertex_parents_length[vertices_length];  // Number of parents

// Updated during bypass operations
struct arr_c *vertex_edges[vertices_length];    // Edge lists
size_t vertex_edges_length[vertices_length];    // Number of edges

// Tracking array (added in attempted fixes)
bool *bypassed[vertices_length];  // Which vertices have been bypassed
```

---

## Test Results

### Working Cases (PDF integral ≈ 1.0, samples > 0):
- ✅ No bypasses: `[1, 1, 1, 1, 1]`
- ✅ 1 bypass: `[1, 1, 0, 1, 1]`
- ✅ 2 bypasses: `[1, 1, 0, 0, 1]`

### Broken Cases (PDF integral = 0, samples = 0):
- ❌ 3 bypasses: `[1, 0, 0, 0, 1]`
- ❌ 4+ bypasses: Any pattern with ≥3 consecutive zeros

**Graph Structure**: Correct (2 vertices: 0 and 4)
**Graph Edges**: Broken (no valid path from 0 to 4)

---

## What I Tried (All Failed)

### Attempt 1: Update future vertices' parent lists after each bypass
**Result**: ❌ Can't find edges in expected form to update back-pointers

### Attempt 2: Repair current vertex's parent list before bypass
**Result**: ❌ Same edge-finding issue

### Attempt 3: Skip bypassed parents during processing
**Result**: ❌ Creates disconnected vertices when all parents are bypassed

### Attempt 4: Multi-pass with skip bypassed parents
**Result**: ❌ Fundamental issue remains - no edges created for all-bypassed case

**Why They All Fail**: Chicken-and-egg problem
- Need updated parent lists to know which parents to process
- Parent lists are updated by the bypass operations themselves
- Skipping bypassed parents = losing edges entirely
- Can't find grandparent edges to create proper connections

---

## Recommended Fix Approaches

### Option A: Recursive Parent Resolution (Cleanest)

When encountering a bypassed parent, recursively resolve to find the first non-bypassed ancestor:

```c
struct ptd_vertex *resolve_parent(size_t vertex_idx, size_t parent_idx) {
    struct ptd_vertex *p = vertex_parents[vertex_idx][parent_idx].p;
    size_t p_idx = p->index;

    // Follow chain of bypassed parents
    while (bypassed[p_idx] && vertex_parents_length[p_idx] > 0) {
        // Get first parent of bypassed vertex
        p = vertex_parents[p_idx][0].p;
        p_idx = p->index;
    }

    return p;
}
```

Use this when reading parent: `parent_vertex = resolve_parent(i, p);`

**Complexity**: O(n) per parent lookup, O(n² · parents) total
**Pros**: Clean, conceptually simple
**Cons**: Assumes certain graph structure

### Option B: Complete Parent List Rebuild (Most Reliable)

After each bypass, rebuild all parent lists from scratch:

```c
void rebuild_parent_lists() {
    // Clear all parent lists
    for (size_t i = 0; i < vertices_length; ++i) {
        vertex_parents_length[i] = 0;
    }

    // Rebuild from edges
    for (size_t i = 0; i < vertices_length; ++i) {
        if (bypassed[i]) continue;

        for (size_t j = 1; j < vertex_edges_length[i] - 1; ++j) {
            struct arr_c *edge = &vertex_edges[i][j];
            size_t child_idx = edge->to->index;

            // Add i as parent of child
            vertex_parents[child_idx][vertex_parents_length[child_idx]].p = vertices[i];
            vertex_parents[child_idx][vertex_parents_length[child_idx]].arr_c_index = j;
            edge->arr_p_index = vertex_parents_length[child_idx];
            vertex_parents_length[child_idx]++;
        }
    }
}
```

Call after each bypass (before `break;`)

**Complexity**: O(n² · edges) worst case
**Pros**: Guaranteed correct, no assumptions
**Cons**: Expensive for large graphs

### Option C: Topological Order Processing (Most Elegant)

Process vertices in reverse topological order (from absorbing states backward):

```c
// Compute topological order
size_t *topo_order = topological_sort(graph);

// Process in reverse order
for (size_t idx = 0; idx < vertices_length; ++idx) {
    size_t i = topo_order[vertices_length - 1 - idx];

    if (rewards[i] != 0) continue;

    // All children of i have already been processed
    // So their parent lists are up-to-date
    // ... bypass logic ...
}
```

**Complexity**: O(n + edges) for topo sort, O(n · parents) for bypass
**Pros**: Elegant, avoids stale parent issue
**Cons**: Need to implement topological sort, assumes DAG

---

## Recommended: Option A (Recursive Resolution)

**Reasoning**:
- Simplest to implement
- Minimal code changes
- O(n) overhead acceptable for typical graph sizes
- Works for all cases

**Implementation**:
1. Add `resolve_parent()` function
2. Replace `parent_vertex = vertex_parents[i][p].p` with `parent_vertex = resolve_parent(i, p)`
3. Test thoroughly

---

## How to Debug Your Fix

1. **Add printf debugging** (see `C_DEBUGGING_GUIDE.md`)

2. **Run diagnostic script**:
   ```bash
   python /tmp/diagnose_vertex_bypass.py 2>&1 | tee debug.log
   ```

3. **Check for these patterns** in output:
   - Stale parent references (parent shows bypassed vertex)
   - Skipped edge creation (no edges for vertex with all-bypassed parents)
   - Incorrect final graph (vertex 0 and 4 exist but not connected)

4. **Verify fix**:
   ```python
   # All these should pass:
   assert pdf_integral > 0.9
   assert sample_mean > 0
   assert graph_transformed.vertices_length() == 2
   ```

5. **Test edge cases**:
   - 4+ consecutive bypasses
   - All vertices bypassed except endpoints
   - Non-consecutive bypasses (should already work)

---

## Workaround Until Fixed

Use epsilon instead of zero for sparse rewards:

```python
# Instead of:
rewards[rewards == 0] = 0

# Use:
epsilon = 0.001
rewards[rewards == 0] = epsilon
```

**Impact**:
- ✅ Avoids bypass bug entirely
- ✅ PDF properly normalized
- ✅ SVGD converges correctly
- ⚠️ Slightly larger graphs (vertices not eliminated)
- ⚠️ Small numerical overhead

---

## Testing Checklist

After implementing fix:

- [ ] Run `/tmp/diagnose_vertex_bypass.py` - should show ✓ for all checks
- [ ] Run `/tmp/test_bypass_order_bug.py` - PDF integral > 0.9 for all cases
- [ ] Run `/tmp/test_bypass_sampling.py` - sample mean > 0 for all cases
- [ ] Test coalescent model (nr_samples=6) - SVGD bias < 5% for all features
- [ ] Run existing test suite - no regressions
- [ ] Performance test - check O(n²) doesn't cause issues for large graphs

---

## Contact / Questions

This debugging package created by Claude (AI assistant) on 2025-10-28.

Original investigation started from SVGD convergence issues with sparse reward features in coalescent models.

For questions about:
- **The bug**: See `VERTEX_BYPASS_BUG_REPORT.md`
- **Technical details**: See `VERTEX_BYPASS_COMPLETE_DIAGNOSIS.md`
- **Fix attempts**: See `VERTEX_BYPASS_FIX_ATTEMPTS_SUMMARY.md`
- **C debugging**: See `C_DEBUGGING_GUIDE.md`
- **Python testing**: Run `diagnose_vertex_bypass.py`

---

**Good luck with the fix! The bug is well-understood - it just needs the right implementation approach.**
