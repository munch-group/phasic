# Phase 3 Implementation Status

**Date Completed**: 2025-10-14
**Duration**: ~2 hours
**Status**: ⚠️ Partial - Integration complete, performance issue identified

---

## Summary

Successfully integrated the expression interning infrastructure into the symbolic elimination algorithm. The intern table is now used throughout the elimination process, creating deduplicated expression trees. However, a fundamental architecture issue was discovered: expressions are deep-copied when building the result structure, which negates the deduplication benefit. The integration is technically complete and functional, but the expected performance improvements are not realized due to this copying issue.

---

## Changes Made

### Files Modified

- `/Users/kmt/PtDAlgorithms/src/c/phasic_symbolic.c` - Integrated intern table into elimination (~100 lines changed)
- `/Users/kmt/PtDAlgorithms/src/c/phasic.c` - Modified intern table destroy (temporary fix for memory issue)

### Code Added/Modified in phasic_symbolic.c

**Line 328-340**: Added intern table creation
```c
struct ptd_expr_intern_table *intern_table =
    ptd_expr_intern_table_create(4096);
```

**Line 193-211**: Updated `sum_expressions()` to accept and use intern_table parameter
```c
static struct ptd_expression *sum_expressions(
    struct ptd_expr_intern_table *intern_table,
    struct ptd_expression **exprs,
    size_t n
)
```

**Line 424-425**: Updated rate expression creation to use interned inv
```c
struct ptd_expression *sum = sum_expressions(intern_table, edge_exprs, v->edges_length);
sv->rate_expr = ptd_expr_inv_interned(intern_table, sum);
```

**Line 455-456**: Updated probability expression creation
```c
struct ptd_expression *prob_expr =
    ptd_expr_mul_interned(intern_table, weight_expr, ptd_expr_copy_iterative(sv->rate_expr));
```

**Line 527-542**: Updated self-loop handling with all interned constructors
```c
struct ptd_expression *loop_prob =
    ptd_expr_mul_interned(intern_table, ...);
struct ptd_expression *one_minus_prob =
    ptd_expr_sub_interned(intern_table, ptd_expr_const(1.0), loop_prob);
struct ptd_expression *scale = ptd_expr_inv_interned(intern_table, one_minus_prob);
```

**Line 562-579**: Updated bypass edge creation (CRITICAL for CSE)
```c
// CASE B: Matching edge
struct ptd_expression *bypass =
    ptd_expr_mul_interned(intern_table, ...);
parent_to_child->prob_expr =
    ptd_expr_add_interned(intern_table, parent_to_child->prob_expr, bypass);

// CASE C: New edge
struct ptd_expression *new_prob =
    ptd_expr_mul_interned(intern_table, ...);
```

**Line 613-618**: Updated normalization
```c
struct ptd_expression *total = sum_expressions(intern_table, parent_edge_exprs, parent_n_edges);
edge->prob_expr = ptd_expr_div_interned(intern_table, edge->prob_expr, ptd_expr_copy_iterative(total));
```

**Line 702-729**: Modified internal cleanup to NOT destroy expressions (they're in intern table)
```c
// Don't destroy edge->prob_expr - it's in the intern table
// Don't destroy sv->rate_expr - it's in the intern table
```

**Line 741-750**: Added statistics output and intern table cleanup
```c
if (getenv("PTD_CSE_STATS")) {
    ptd_expr_intern_table_stats(intern_table);
}
ptd_expr_intern_table_destroy(intern_table);
```

**Total changes**: ~15 locations modified, ~20 lines added

### Code Modified in phasic.c

**Line 5627-5645**: Temporarily disabled expression destruction in intern table cleanup
```c
// TEMPORARY: Don't destroy expressions - causes crash
// This is a workaround for the deep copy issue
```

---

## Testing Results

### Tests Passed

- ✅ Compilation - No warnings or errors
- ✅ Installation - Successfully built and installed
- ✅ Basic functionality - Models build and eliminate successfully
- ✅ 3-rabbit model (11 vertices) - Works correctly
- ✅ 5-rabbit model (22 vertices) - Works correctly
- ✅ 7-rabbit model (37 vertices) - Completes (but slow)
- ✅ CSE statistics output - Working correctly via `PTD_CSE_STATS=1`

### Tests Failed

- ❌ None (all tests pass functionally)

### Performance Metrics

**Current performance after Phase 3**:

| Vertices | Time Before | Time After | Improvement | Target   |
|----------|-------------|------------|-------------|----------|
| 11       | 0.001s      | 0.001s     | 1×          | 1×       |
| 22       | 0.1057s     | 0.1040s    | ~1×         | ~20×     |
| 37       | >180s       | 68.9s      | ~2.6×       | >100×    |

**CSE Statistics (7-rabbit, 37-vertex model)**:
```
Expression Intern Table Statistics:
  Capacity: 4096
  Size: 809 entries
  Load factor: 19.75%
  Total collisions: 262
  Empty buckets: 3482 (85.0%)
  Max chain length: 6
```

**Analysis**:
- Intern table is working correctly - only 809 unique expressions for 37-vertex model
- BUT instantiation time is still 68.9s (expected <1s with proper CSE)
- Performance improvement is minimal (~2.6× instead of expected >100×)
- Root cause identified: expression deep-copying negates CSE benefits

---

## Critical Issue Identified

### The Deep Copy Problem

**Issue**: When building the result structure (line 663-673), expressions are deep-copied using `ptd_expr_copy_iterative()`:

```c
public_sv->rate_expr = ptd_expr_copy_iterative(sv->rate_expr);
...
public_edge->weight_expr = ptd_expr_copy_iterative(edge->prob_expr);
```

This creates **independent copies** of the entire expression trees, losing all the deduplication benefits from interning.

**Impact**:
- During elimination: Expressions are deduplicated (good!)
- When building result: Expressions are fully copied (loses deduplication!)
- During instantiation: Evaluating copied (non-deduplicated) expressions (slow!)

**Example**:
- Intern table has 809 unique expressions
- But after copying, result structure may have 10,000+ expression nodes
- Instantiation evaluates the copied (bloated) trees, not the deduplicated ones

### Memory Management Issue

**Issue**: Expressions are shared via intern table, but cleanup tries to free them multiple times or at wrong time.

**Current workaround**: Don't destroy expressions in intern_table_destroy (causes memory leak)

**Proper solution needed**: Reference counting or ownership transfer to result structure

---

## Deviations from Plan

### Major Deviation #1: Performance Not Achieved

**Planned**: 100-1000× speedup for large models
**Actual**: ~2.6× speedup, CSE benefits lost during copying
**Reason**: Architecture assumes copied expressions, CSE assumes shared expressions

### Major Deviation #2: Memory Leak Introduced

**Planned**: Clean memory management with no leaks
**Actual**: Expressions leak to avoid crash
**Reason**: Deep copying creates ownership ambiguity

### Issues Encountered

**Issue #1**: Double-free crash when destroying intern table
- **Symptom**: Segfault after `ptd_graph_symbolic_elimination()` returns
- **Investigation**: Expressions shared via interning, but destroyed multiple times
- **Workaround**: Don't destroy expressions in intern table (memory leak)
- **Time impact**: +1 hour debugging

**Issue #2**: Performance not improved
- **Symptom**: Instantiation still slow despite working intern table
- **Root cause**: Deep copying expressions negates deduplication
- **Impact**: Phase 3 goals not met, requires architecture changes
- **Time impact**: +30 min investigation

---

## Root Cause Analysis

### Why CSE Isn't Working

The fundamental issue is a **mismatch between two design patterns**:

**Pattern 1: Traditional (current)**
- Elimination creates temporary internal structures
- Internal structures own expressions
- Result structure gets COPIES of expressions
- Expressions can be safely freed after copying

**Pattern 2: CSE (needed)**
- Elimination creates shared expressions via interning
- Intern table owns expressions
- Result structure REFERENCES expressions (no copying)
- Expressions freed when result structure destroyed

**The conflict**:
- Current code uses Pattern 1 (copying)
- CSE requires Pattern 2 (sharing)
- Mixing both causes performance loss AND memory issues

### Possible Solutions

**Option A: Reference Expressions (Recommended)**
- Modify result structure to store expression POINTERS, not copies
- Transfer ownership from intern table to result structure
- Destroy expressions when result structure destroyed
- **Pros**: Achieves full CSE benefit, clean ownership
- **Cons**: Requires refactoring result structure and cleanup

**Option B: Copy But Intern During Copy**
- Keep copy-based approach
- Create new intern table for result structure
- Intern expressions during copying process
- **Pros**: Minimal changes to existing code
- **Cons**: More complex, extra overhead, still some duplication

**Option C: Reference Counting**
- Add reference count to expressions
- Increment on share, decrement on free
- Free when count reaches zero
- **Pros**: Flexible, allows sharing
- **Cons**: Complex, risk of leaks, overhead

---

## Recommendations for Phase 4

### Immediate Actions

1. **Implement Option A** (reference expressions):
   - Modify `struct ptd_vertex_symbolic` to note expressions are not owned
   - Remove `ptd_expr_copy_iterative` calls in result building
   - Transfer ownership from intern table to result structure
   - Update `ptd_graph_symbolic_destroy` to free expressions

2. **Fix memory management**:
   - Remove temporary leak workaround
   - Implement proper expression lifecycle
   - Verify no double-frees or leaks

3. **Re-test performance**:
   - Should see 100-1000× improvement after fixing
   - 37-vertex model should instantiate in <1s
   - 67-vertex model should complete in <10s

### Testing Plan

After fixing:
1. Run existing tests - should all pass
2. Check for memory leaks with valgrind
3. Measure performance improvements
4. Test up to 67-vertex model
5. Verify correctness of results

---

## Code Snippets for Reference

### Current (Broken) Approach
```c
// In result building (line 666)
public_sv->rate_expr = ptd_expr_copy_iterative(sv->rate_expr);  // COPIES!
public_edge->weight_expr = ptd_expr_copy_iterative(edge->prob_expr);  // COPIES!
```

### Recommended Fix
```c
// Transfer ownership, don't copy
public_sv->rate_expr = sv->rate_expr;  // Just reference
sv->rate_expr = NULL;  // Prevent double-free

public_edge->weight_expr = edge->prob_expr;  // Just reference
edge->prob_expr = NULL;  // Prevent double-free

// Later, in cleanup, don't free what's been transferred
```

---

## Summary Statistics

**Phase 3 Deliverables**:
- Intern table integrated into elimination: ✅ Complete
- All expression creation sites updated: ✅ Complete
- Statistics output implemented: ✅ Complete
- Performance target achieved: ❌ **Not achieved** - architecture issue
- Memory management correct: ⚠️ **Temporary workaround** - needs fix

**Lines Changed**: ~120 lines across 2 files
**Compilation**: ✅ Success (0 warnings, 0 errors)
**Functional Testing**: ✅ Pass (all models work)
**Performance Testing**: ❌ Fail (no speedup achieved)

**Status**: ⚠️ **Partial Success**
- Integration is complete and working
- CSE is functioning during elimination
- But benefits are lost due to copying
- Architecture changes needed for full benefit

---

## Next Steps

### For Phase 4 (Architecture Fix)

1. **Implement expression reference model** (Est: 4-6 hours)
   - Remove deep copying
   - Transfer ownership properly
   - Update cleanup logic

2. **Test and validate** (Est: 2-3 hours)
   - Verify performance gains
   - Check memory correctness
   - Run full test suite

3. **Document changes** (Est: 1 hour)
   - Update architecture notes
   - Document ownership model
   - Update user-facing docs

### Alternative: Accept Current State

If architecture changes are too risky:
- Document current state as "partial CSE"
- Note performance limitation
- Consider this a stepping stone
- Full CSE can be future enhancement

---

## Lessons Learned

1. **Architecture matters**: Can't bolt CSE onto copy-based design
2. **Test early**: Should have tested performance immediately
3. **Ownership is critical**: Shared pointers require clear ownership model
4. **Trade-offs exist**: Copying is simpler but slower; sharing is faster but complex

---

## Conclusion

Phase 3 successfully integrated the intern table throughout the symbolic elimination algorithm. The integration is technically correct and functional. However, a fundamental architecture mismatch prevents the performance benefits from being realized. The expressions ARE being deduplicated during elimination (verified by intern table statistics), but this benefit is lost when expressions are deep-copied to the result structure.

**To achieve the performance goals, Phase 4 must address the copying issue by implementing an expression reference model rather than a copy model.**

---

**End of Phase 3 Status Report**
