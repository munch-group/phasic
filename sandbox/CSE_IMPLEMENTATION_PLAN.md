# CSE Implementation Plan: Expression Tree Optimization

**Status**: Ready for Implementation
**Priority**: Critical - blocks large models (>30 vertices)
**Estimated Duration**: 12 days
**Author**: Claude Code Analysis
**Date**: 2025-10-14

---

## Quick Start

**For Claude**: This plan is designed for phase-by-phase implementation in separate conversations.

**To begin Phase N**:
1. Start a fresh conversation
2. Attach this plan: `CSE_IMPLEMENTATION_PLAN.md`
3. Attach previous phase status: `CSE_PHASE{N-1}_STATUS.md` (if applicable)
4. Say: `Implement Phase {N} of CSE_IMPLEMENTATION_PLAN.md`

**At phase completion**:
- Write comprehensive status report to `CSE_PHASE{N}_STATUS.md`
- Use template: `CSE_PHASE_STATUS_TEMPLATE.md`
- Document all changes, test results, and deviations
- See "How to Use This Plan" section below for detailed guidance

**Files**:
- Plan: `CSE_IMPLEMENTATION_PLAN.md` (this file)
- Template: `CSE_PHASE_STATUS_TEMPLATE.md`
- Status reports: `CSE_PHASE{N}_STATUS.md` (N = 1-5)

---

## Table of Contents

### Planning & Overview
- [Executive Summary](#executive-summary)
- [How to Use This Plan](#how-to-use-this-plan-phase-by-phase-implementation) ⭐
- [Current Performance Analysis](#current-performance-analysis)
- [Solution Architecture](#solution-architecture)

### Implementation Phases
- [Phase 1: Algebraic Simplification](#phase-1-algebraic-simplification-days-1-2) (Days 1-2)
- [Phase 2: Expression Interning Infrastructure](#phase-2-expression-interning-infrastructure-days-3-5) (Days 3-5)
- [Phase 3: Integration into Elimination](#phase-3-integration-into-elimination-days-6-7) (Days 6-7)
- [Phase 4: Testing and Validation](#phase-4-testing-and-validation-days-8-10) (Days 8-10)
- [Phase 5: Documentation and Polish](#phase-5-documentation-and-polish-days-11-12) (Days 11-12)

### Reference
- [Risk Management](#risk-management)
- [Success Metrics](#success-metrics)
- [Timeline and Resources](#timeline-and-resources)

---

## Executive Summary

### Problem

Symbolic Gaussian elimination creates exponentially growing expression trees, causing timeouts for models >30 vertices. The 67-vertex rabbit model (10 rabbits) that should instantiate in <1 second currently times out after 3+ minutes.

### Root Cause

`ptd_expr_copy_iterative()` creates full deep copies during bypass edge construction in the elimination algorithm. Without **Common Subexpression Elimination (CSE)**, identical sub-expressions are duplicated throughout the tree, causing O(exp(n)) growth instead of the intended O(n²).

**Critical Code**: `/Users/kmt/PtDAlgorithms/src/c/ptdalgorithms_symbolic.c:541-554, 593`

### Solution

Implement **CSE via expression interning** with hash table + **algebraic simplification** to achieve true O(n²) complexity.

### Expected Outcome

```
Model Size    | Before        | After (Projected) | Speedup
--------------|---------------|-------------------|----------
11 vertices   | 0.001s        | 0.001s           | 1×
22 vertices   | 0.12s         | 0.005s           | 24×
37 vertices   | >180s         | 0.05s            | >3600×
67 vertices   | Timeout       | 0.3s             | ∞
100 vertices  | Timeout       | 1.5s             | ∞
```

---

## How to Use This Plan: Phase-by-Phase Implementation

### Overview

This plan is designed to be implemented **phase by phase in separate conversations**. Each phase should be completed in a fresh conversation to maximize token availability and maintain clean context.

### Starting Each Phase

**Create a new conversation** and provide Claude with:

1. **The plan document** - Upload or reference `CSE_IMPLEMENTATION_PLAN.md`
2. **Previous phase status** (if applicable) - Attach the status report from the previous phase
3. **Clear instruction** - Tell Claude exactly which phase to implement

#### Example Phase Start Messages

**Phase 1:**
```
Implement Phase 1 of CSE_IMPLEMENTATION_PLAN.md

Phase 1: Algebraic Simplification (Days 1-2)
- Modify ptd_expr_mul(), ptd_expr_add(), ptd_expr_div(), ptd_expr_sub()
- Add constant folding and identity elimination
- Test with 5-rabbit model

When complete, write a status report to CSE_PHASE1_STATUS.md
```

**Phase 2:**
```
Implement Phase 2 of CSE_IMPLEMENTATION_PLAN.md

Previous status: See CSE_PHASE1_STATUS.md

Phase 2: Expression Interning Infrastructure (Days 3-5)
- Add ptd_expr_hash() and ptd_expr_equal()
- Implement intern table data structure
- Create interned expression constructors
- Add to header file

When complete, write a status report to CSE_PHASE2_STATUS.md
```

**Phase 3:**
```
Implement Phase 3 of CSE_IMPLEMENTATION_PLAN.md

Previous status: See CSE_PHASE2_STATUS.md

Phase 3: Integration into Elimination (Days 6-7)
- Add intern table to symbolic elimination
- Update all expression creation sites
- Update normalization
- Add statistics output

When complete, write a status report to CSE_PHASE3_STATUS.md
```

**Phase 4:**
```
Implement Phase 4 of CSE_IMPLEMENTATION_PLAN.md

Previous status: See CSE_PHASE3_STATUS.md

Phase 4: Testing and Validation (Days 8-10)
- Run all test suites
- Execute full rabbits notebook
- Perform stress tests
- Check for memory leaks

When complete, write a status report to CSE_PHASE4_STATUS.md
```

**Phase 5:**
```
Implement Phase 5 of CSE_IMPLEMENTATION_PLAN.md

Previous status: See CSE_PHASE4_STATUS.md

Phase 5: Documentation and Polish (Days 11-12)
- Update CLAUDE.md with CSE section
- Update tutorial notebook
- Create migration guide
- Update test suite configuration

When complete, write a final status report to CSE_PHASE5_STATUS.md
```

### Ending Each Phase

At the end of each phase, Claude must write a **comprehensive status report** to a file:

**File naming**: `CSE_PHASE{N}_STATUS.md` where N is the phase number (1-5)

**Template**: Copy from `CSE_PHASE_STATUS_TEMPLATE.md` and fill in

**Required sections in status report**:

```markdown
# Phase {N} Implementation Status

**Date Completed**: YYYY-MM-DD
**Duration**: X days
**Status**: ✅ Complete / ⚠️ Partial / ❌ Blocked

---

## Summary

Brief 2-3 sentence summary of what was accomplished.

---

## Changes Made

### Files Modified

List all files changed with brief description:
- `/path/to/file1.c` - Added algebraic simplification to ptd_expr_mul()
- `/path/to/file2.h` - Declared new intern table functions
- etc.

### Code Added

Summary of major code additions:
- Added ~150 lines for expression hashing
- Implemented intern table with 300 lines
- Created 5 new functions

### Code Modified

Summary of modifications to existing code:
- Updated ptd_expr_add() with simplification rules
- Modified symbolic elimination to use intern table
- etc.

---

## Testing Results

### Tests Passed
- ✅ test_evaluate_performance.py 5 - 22-vertex model: 0.05s (2× faster)
- ✅ Unit tests all passing
- etc.

### Tests Failed
- ❌ None / List any failures

### Performance Metrics

Current performance after this phase:
| Vertices | Time Before | Time After | Improvement |
|----------|-------------|------------|-------------|
| 11       | 0.001s      | 0.001s     | 1×          |
| 22       | 0.12s       | 0.06s      | 2×          |

---

## Deviations from Plan

### Changes to Original Plan

Document any deviations:
- **Changed**: Used FNV-1a hash instead of simple multiplicative hash
  - **Reason**: Better distribution, fewer collisions
  - **Impact**: No change to timeline

- **Skipped**: Some optional optimization
  - **Reason**: Not needed for performance target
  - **Impact**: Saves 0.5 days

### Issues Encountered

Document problems and resolutions:
- **Issue**: Compilation error in ptd_expr_hash()
  - **Solution**: Added missing header include
  - **Time impact**: +1 hour

### Risks Identified

Any risks for future phases:
- Intern table might need larger capacity for 100+ vertex models
- Memory usage needs monitoring in Phase 4

---

## Next Steps

What needs to happen in the next phase:
1. Start Phase {N+1} in new conversation
2. Attach this status report
3. Specific prerequisites or setup needed

---

## Code Snippets (Optional)

Key code additions that might be useful for reference:

```c
// Example: Hash function signature
uint64_t ptd_expr_hash(const struct ptd_expression *expr) {
    // ... implementation
}
```

---

## Notes for Future Phases

Any important information for later:
- Intern table uses 4096 buckets - might need tuning
- Hash function handles commutativity of ADD/MUL
- Remember to free intern table at end of elimination
```

### Phase Completion Checklist

Before writing the status report, verify:

- [ ] All code changes compile without warnings
- [ ] All code changes committed to git (if using version control)
- [ ] Tests run and pass (or failures documented)
- [ ] Performance metrics collected
- [ ] Any deviations from plan documented
- [ ] Next phase prerequisites identified

### Continuing After Issues

If a phase cannot be completed:

1. **Document what was accomplished** in the status report
2. **Mark status as** ⚠️ Partial or ❌ Blocked
3. **Clearly describe the blocker** with error messages, logs, etc.
4. **Propose solution** or alternative approach
5. **Estimate impact** on timeline

Example:
```markdown
**Status**: ⚠️ Partial - 80% complete

## Blocker

Hash function causes segfault on large expressions (>1000 nodes).

### Error Details
```
Segmentation fault (core dumped)
Backtrace: ptd_expr_hash() -> stack overflow
```

### Proposed Solution
Change to iterative hash computation instead of recursive.
Estimate: +1 day to implement and test.

### Alternative
Use simpler hash (pointer XOR) as temporary workaround.
May increase collisions but allows progress.
```

### Status Report Storage

All status reports should be stored in:
```
/Users/kmt/PtDAlgorithms/CSE_PHASE{N}_STATUS.md
```

After final phase (Phase 5), create a summary document:
```
/Users/kmt/PtDAlgorithms/CSE_IMPLEMENTATION_COMPLETE.md
```

This summary should include:
- Links to all 5 phase status reports
- Overall timeline (planned vs actual)
- Final performance metrics
- Lessons learned
- Future optimization opportunities

---

## Current Performance Analysis

### Empirical Measurements

**Test**: Rabbit island model with parameterized edges

| Vertices | Construction | Elimination | Instantiation | Total   | Notes          |
|----------|--------------|-------------|---------------|---------|----------------|
| 11       | 0.00s        | 0.00s       | 0.001s        | 0.001s  | ✓ Working      |
| 22       | 0.00s        | 0.21s       | 0.12s         | 0.33s   | ✓ Slow         |
| 37       | 0.00s        | 2.0s        | >180s         | Timeout | ✗ Failed       |
| 67       | 0.00s        | ~5s         | Timeout       | Timeout | ✗ Target model |

**Bottleneck**: Instantiation phase, specifically `ptd_expr_evaluate_iterative()` called on exponentially large expression trees.

### Expression Growth Pattern

During elimination of vertex `v` with parents P and children C:

```
For each (parent u ∈ P, child w ∈ C):
    Create bypass: expr(u→v) * expr(v→w)
    Each multiplication copies both sub-expressions
```

**Example Growth**:
- Vertex 10 eliminated: Expression `E10` appears in bypass edges for 5 parents
- Each parent gets `copy(E10) * copy(E_child)`
- Result: 5 full copies of E10 subtree
- When parent eliminated: Each of those 5 copies gets copied again
- **Exponential cascade**: Depth 10 vertex expression copied 2^10 = 1024 times

### Why O(exp(n)) Instead of O(n²)

**Intended design**: O(n³) elimination (n² edges, n iterations), then O(n) instantiation per parameter vector.

**Actual behavior**: Expression tree size grows exponentially, making instantiation O(exp(n)).

**Key insight**: Without CSE, elimination creates a **tree** representation when it should create a **DAG** representation.

---

## Solution Architecture

### Two-Pronged Approach

1. **Algebraic Simplification** (quick win, 2-5× improvement)
   - Constant folding: `c1 * c2 → c3`
   - Identity elimination: `x * 1 → x`, `x + 0 → x`
   - Zero propagation: `x * 0 → 0`

2. **Expression Interning** (main solution, 100-1000× improvement)
   - Hash table mapping structural hash → canonical expression
   - Before creating expression, check if identical exists
   - Share single instance across all references

### Why Interning?

**Alternatives considered**:
- Reference counting: Risk of cycles, invasive changes
- DAG representation: Requires major refactor, breaks API
- Lazy evaluation: Complex, doesn't address root cause

**Interning advantages**:
- Automatic deduplication
- Minimal API changes (internal only)
- Clean separation of concerns
- No risk of memory leaks or cycles

---

## Phase 1: Algebraic Simplification (Days 1-2)

### Goal

Reduce expression size by 2-5× through constant folding and identity elimination.

### Files to Modify

- `/Users/kmt/PtDAlgorithms/src/c/ptdalgorithms.c`

### Implementation

#### 1.1 Modify `ptd_expr_mul()` (line ~4900)

```c
struct ptd_expression *ptd_expr_mul(struct ptd_expression *left,
                                     struct ptd_expression *right) {
    // Simplification: 0 * x = 0
    if (left->type == PTD_EXPR_CONST && left->const_value == 0.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }
    if (right->type == PTD_EXPR_CONST && right->const_value == 0.0) {
        ptd_expr_destroy_iterative(left);
        return right;
    }

    // Simplification: 1 * x = x
    if (left->type == PTD_EXPR_CONST && left->const_value == 1.0) {
        ptd_expr_destroy_iterative(left);
        return right;
    }
    if (right->type == PTD_EXPR_CONST && right->const_value == 1.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }

    // Constant folding: c1 * c2 = c3
    if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
        double result = left->const_value * right->const_value;
        ptd_expr_destroy_iterative(left);
        ptd_expr_destroy_iterative(right);
        return ptd_expr_const(result);
    }

    // Original allocation
    struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate memory for multiplication expression");
    }
    expr->type = PTD_EXPR_MUL;
    expr->left = left;
    expr->right = right;
    return expr;
}
```

#### 1.2 Modify `ptd_expr_add()` (line ~4882)

```c
struct ptd_expression *ptd_expr_add(struct ptd_expression *left,
                                     struct ptd_expression *right) {
    // Simplification: 0 + x = x
    if (left->type == PTD_EXPR_CONST && left->const_value == 0.0) {
        ptd_expr_destroy_iterative(left);
        return right;
    }
    if (right->type == PTD_EXPR_CONST && right->const_value == 0.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }

    // Constant folding: c1 + c2 = c3
    if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
        double result = left->const_value + right->const_value;
        ptd_expr_destroy_iterative(left);
        ptd_expr_destroy_iterative(right);
        return ptd_expr_const(result);
    }

    // Original allocation
    struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate memory for addition expression");
    }
    expr->type = PTD_EXPR_ADD;
    expr->left = left;
    expr->right = right;
    return expr;
}
```

#### 1.3 Modify `ptd_expr_div()` (line ~4916)

```c
struct ptd_expression *ptd_expr_div(struct ptd_expression *left,
                                     struct ptd_expression *right) {
    // Simplification: 0 / x = 0 (x != 0)
    if (left->type == PTD_EXPR_CONST && left->const_value == 0.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }

    // Simplification: x / 1 = x
    if (right->type == PTD_EXPR_CONST && right->const_value == 1.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }

    // Constant folding: c1 / c2 = c3
    if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
        if (right->const_value == 0.0) {
            DIE_ERROR(1, "Division by zero in constant folding");
        }
        double result = left->const_value / right->const_value;
        ptd_expr_destroy_iterative(left);
        ptd_expr_destroy_iterative(right);
        return ptd_expr_const(result);
    }

    // Original allocation
    struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate memory for division expression");
    }
    expr->type = PTD_EXPR_DIV;
    expr->left = left;
    expr->right = right;
    return expr;
}
```

#### 1.4 Modify `ptd_expr_sub()` (line ~4933)

```c
struct ptd_expression *ptd_expr_sub(struct ptd_expression *left,
                                     struct ptd_expression *right) {
    // Simplification: x - 0 = x
    if (right->type == PTD_EXPR_CONST && right->const_value == 0.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }

    // Constant folding: c1 - c2 = c3
    if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
        double result = left->const_value - right->const_value;
        ptd_expr_destroy_iterative(left);
        ptd_expr_destroy_iterative(right);
        return ptd_expr_const(result);
    }

    // Original allocation
    struct ptd_expression *expr = (struct ptd_expression *) calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate memory for subtraction expression");
    }
    expr->type = PTD_EXPR_SUB;
    expr->left = left;
    expr->right = right;
    return expr;
}
```

### Testing Phase 1

**Quick test**:
```bash
pixi run pip install -e . --force-reinstall --no-deps
pixi run python test_evaluate_performance.py 5
```

**Expected**: ~2× speedup for 22-vertex model (0.12s → ~0.06s instantiation)

---

## Phase 2: Expression Interning Infrastructure (Days 3-5)

### Goal

Implement CSE hash table for automatic deduplication of expressions.

### Files to Modify

- `/Users/kmt/PtDAlgorithms/src/c/ptdalgorithms.c` (add ~300 lines)
- `/Users/kmt/PtDAlgorithms/api/c/ptdalgorithms.h` (declare new functions)

### Step 2.1: Expression Hashing

Add after line 5300 in `ptdalgorithms.c`:

```c
/**
 * Compute structural hash of expression tree
 *
 * Uses FNV-1a-like hash with type and value mixing.
 * For commutative operations (ADD, MUL), sorts child hashes for consistency.
 */
uint64_t ptd_expr_hash(const struct ptd_expression *expr) {
    if (expr == NULL) return 0;

    uint64_t hash = 14695981039346656037ULL;  // FNV offset basis
    const uint64_t prime = 1099511628211ULL;  // FNV prime

    // Mix in type
    hash ^= (uint64_t)expr->type;
    hash *= prime;

    switch (expr->type) {
        case PTD_EXPR_CONST: {
            // Hash double value by reinterpreting bits
            uint64_t bits;
            memcpy(&bits, &expr->const_value, sizeof(uint64_t));
            hash ^= bits;
            hash *= prime;
            break;
        }

        case PTD_EXPR_PARAM:
            hash ^= expr->param_index;
            hash *= prime;
            break;

        case PTD_EXPR_DOT:
            hash ^= expr->n_terms;
            hash *= prime;
            for (size_t i = 0; i < expr->n_terms; i++) {
                hash ^= expr->param_indices[i];
                hash *= prime;

                uint64_t coeff_bits;
                memcpy(&coeff_bits, &expr->coefficients[i], sizeof(uint64_t));
                hash ^= coeff_bits;
                hash *= prime;
            }
            break;

        case PTD_EXPR_INV:
            hash ^= ptd_expr_hash(expr->left);
            hash *= prime;
            break;

        case PTD_EXPR_ADD:
        case PTD_EXPR_MUL:
        case PTD_EXPR_DIV:
        case PTD_EXPR_SUB: {
            uint64_t left_hash = ptd_expr_hash(expr->left);
            uint64_t right_hash = ptd_expr_hash(expr->right);

            // Commutative operations: sort hashes for consistency
            if (expr->type == PTD_EXPR_ADD || expr->type == PTD_EXPR_MUL) {
                if (left_hash > right_hash) {
                    uint64_t tmp = left_hash;
                    left_hash = right_hash;
                    right_hash = tmp;
                }
            }

            hash ^= left_hash;
            hash *= prime;
            hash ^= right_hash;
            hash *= prime;
            break;
        }
    }

    return hash;
}

/**
 * Check structural equality of two expressions
 *
 * Performs deep comparison, handling commutativity of ADD and MUL.
 */
bool ptd_expr_equal(const struct ptd_expression *a, const struct ptd_expression *b) {
    if (a == b) return true;
    if (a == NULL || b == NULL) return false;
    if (a->type != b->type) return false;

    switch (a->type) {
        case PTD_EXPR_CONST:
            return a->const_value == b->const_value;

        case PTD_EXPR_PARAM:
            return a->param_index == b->param_index;

        case PTD_EXPR_DOT:
            if (a->n_terms != b->n_terms) return false;
            for (size_t i = 0; i < a->n_terms; i++) {
                if (a->param_indices[i] != b->param_indices[i]) return false;
                if (a->coefficients[i] != b->coefficients[i]) return false;
            }
            return true;

        case PTD_EXPR_INV:
            return ptd_expr_equal(a->left, b->left);

        case PTD_EXPR_ADD:
        case PTD_EXPR_MUL:
            // Commutative: check both orderings
            return (ptd_expr_equal(a->left, b->left) && ptd_expr_equal(a->right, b->right)) ||
                   (ptd_expr_equal(a->left, b->right) && ptd_expr_equal(a->right, b->left));

        case PTD_EXPR_DIV:
        case PTD_EXPR_SUB:
            // Non-commutative: order matters
            return ptd_expr_equal(a->left, b->left) && ptd_expr_equal(a->right, b->right);
    }

    return false;
}
```

### Step 2.2: Intern Table Data Structure

Add after hash functions:

```c
/**
 * Expression intern table entry (linked list for collision handling)
 */
struct ptd_expr_intern_entry {
    struct ptd_expression *expr;
    uint64_t hash;
    struct ptd_expr_intern_entry *next;
};

/**
 * Expression intern table for CSE
 *
 * Hash table mapping expression structure → canonical instance.
 * Multiple references to identical expressions share single instance.
 */
struct ptd_expr_intern_table {
    struct ptd_expr_intern_entry **buckets;
    size_t capacity;
    size_t size;
    size_t collisions;  // Statistics
};

/**
 * Create intern table with specified capacity
 */
struct ptd_expr_intern_table *ptd_expr_intern_table_create(size_t capacity) {
    struct ptd_expr_intern_table *table =
        (struct ptd_expr_intern_table *)malloc(sizeof(struct ptd_expr_intern_table));

    if (table == NULL) {
        DIE_ERROR(1, "Failed to allocate intern table");
    }

    table->capacity = capacity;
    table->size = 0;
    table->collisions = 0;
    table->buckets = (struct ptd_expr_intern_entry **)
        calloc(capacity, sizeof(struct ptd_expr_intern_entry *));

    if (table->buckets == NULL) {
        free(table);
        DIE_ERROR(1, "Failed to allocate intern table buckets");
    }

    return table;
}

/**
 * Intern an expression (returns existing if found, otherwise adds to table)
 *
 * IMPORTANT: If existing expression found, destroys input and returns existing.
 * Caller must not use input pointer after calling this function.
 */
struct ptd_expression *ptd_expr_intern(
    struct ptd_expr_intern_table *table,
    struct ptd_expression *expr
) {
    if (expr == NULL || table == NULL) return expr;

    uint64_t hash = ptd_expr_hash(expr);
    size_t bucket = hash % table->capacity;

    // Search for existing expression
    struct ptd_expr_intern_entry *entry = table->buckets[bucket];
    bool first = true;
    while (entry != NULL) {
        if (entry->hash == hash && ptd_expr_equal(entry->expr, expr)) {
            // Found existing - destroy input and return existing
            ptd_expr_destroy_iterative(expr);
            return entry->expr;
        }
        if (!first) table->collisions++;
        first = false;
        entry = entry->next;
    }

    // Not found - add to table
    struct ptd_expr_intern_entry *new_entry =
        (struct ptd_expr_intern_entry *)malloc(sizeof(struct ptd_expr_intern_entry));

    if (new_entry == NULL) {
        DIE_ERROR(1, "Failed to allocate intern table entry");
    }

    new_entry->expr = expr;
    new_entry->hash = hash;
    new_entry->next = table->buckets[bucket];
    table->buckets[bucket] = new_entry;
    table->size++;

    return expr;
}

/**
 * Destroy intern table
 *
 * Note: Does NOT destroy expressions themselves - they may still be referenced.
 * Only frees table structure.
 */
void ptd_expr_intern_table_destroy(struct ptd_expr_intern_table *table) {
    if (table == NULL) return;

    for (size_t i = 0; i < table->capacity; i++) {
        struct ptd_expr_intern_entry *entry = table->buckets[i];
        while (entry != NULL) {
            struct ptd_expr_intern_entry *next = entry->next;
            free(entry);
            entry = next;
        }
    }
    free(table->buckets);
    free(table);
}

/**
 * Print intern table statistics (for debugging/profiling)
 */
void ptd_expr_intern_table_stats(const struct ptd_expr_intern_table *table) {
    if (table == NULL) return;

    printf("Expression Intern Table Statistics:\n");
    printf("  Capacity: %zu\n", table->capacity);
    printf("  Size: %zu entries\n", table->size);
    printf("  Load factor: %.2f%%\n", 100.0 * table->size / table->capacity);
    printf("  Total collisions: %zu\n", table->collisions);

    // Compute chain length distribution
    size_t max_chain = 0;
    size_t empty_buckets = 0;
    size_t chain_lengths[10] = {0};  // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9+

    for (size_t i = 0; i < table->capacity; i++) {
        size_t chain_len = 0;
        struct ptd_expr_intern_entry *e = table->buckets[i];
        while (e) {
            chain_len++;
            e = e->next;
        }

        if (chain_len == 0) {
            empty_buckets++;
        } else {
            size_t idx = chain_len < 9 ? chain_len : 9;
            chain_lengths[idx]++;
        }

        if (chain_len > max_chain) max_chain = chain_len;
    }

    printf("  Empty buckets: %zu (%.1f%%)\n", empty_buckets,
           100.0 * empty_buckets / table->capacity);
    printf("  Max chain length: %zu\n", max_chain);
    printf("  Chain length distribution:\n");
    for (size_t i = 1; i < 10; i++) {
        if (chain_lengths[i] > 0) {
            printf("    Length %zu: %zu buckets\n",
                   i < 9 ? i : 9, chain_lengths[i]);
        }
    }
}
```

### Step 2.3: Interned Expression Constructors

Add wrapper functions that use intern table:

```c
/**
 * Create addition expression with interning
 */
struct ptd_expression *ptd_expr_add_interned(
    struct ptd_expr_intern_table *table,
    struct ptd_expression *left,
    struct ptd_expression *right
) {
    // Apply simplifications first (from Phase 1)
    if (left->type == PTD_EXPR_CONST && left->const_value == 0.0) {
        ptd_expr_destroy_iterative(left);
        return right;
    }
    if (right->type == PTD_EXPR_CONST && right->const_value == 0.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }
    if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
        double result = left->const_value + right->const_value;
        ptd_expr_destroy_iterative(left);
        ptd_expr_destroy_iterative(right);
        return ptd_expr_const(result);
    }

    // Create expression
    struct ptd_expression *expr = (struct ptd_expression *)calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate addition expression");
    }
    expr->type = PTD_EXPR_ADD;
    expr->left = left;
    expr->right = right;

    // Intern if table provided
    if (table != NULL) {
        return ptd_expr_intern(table, expr);
    }
    return expr;
}

/**
 * Create multiplication expression with interning
 */
struct ptd_expression *ptd_expr_mul_interned(
    struct ptd_expr_intern_table *table,
    struct ptd_expression *left,
    struct ptd_expression *right
) {
    // Apply simplifications first
    if (left->type == PTD_EXPR_CONST && left->const_value == 0.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }
    if (right->type == PTD_EXPR_CONST && right->const_value == 0.0) {
        ptd_expr_destroy_iterative(left);
        return right;
    }
    if (left->type == PTD_EXPR_CONST && left->const_value == 1.0) {
        ptd_expr_destroy_iterative(left);
        return right;
    }
    if (right->type == PTD_EXPR_CONST && right->const_value == 1.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }
    if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
        double result = left->const_value * right->const_value;
        ptd_expr_destroy_iterative(left);
        ptd_expr_destroy_iterative(right);
        return ptd_expr_const(result);
    }

    // Create expression
    struct ptd_expression *expr = (struct ptd_expression *)calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate multiplication expression");
    }
    expr->type = PTD_EXPR_MUL;
    expr->left = left;
    expr->right = right;

    // Intern if table provided
    if (table != NULL) {
        return ptd_expr_intern(table, expr);
    }
    return expr;
}

/**
 * Create division expression with interning
 */
struct ptd_expression *ptd_expr_div_interned(
    struct ptd_expr_intern_table *table,
    struct ptd_expression *left,
    struct ptd_expression *right
) {
    // Apply simplifications first
    if (left->type == PTD_EXPR_CONST && left->const_value == 0.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }
    if (right->type == PTD_EXPR_CONST && right->const_value == 1.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }
    if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
        if (right->const_value == 0.0) {
            DIE_ERROR(1, "Division by zero in constant folding");
        }
        double result = left->const_value / right->const_value;
        ptd_expr_destroy_iterative(left);
        ptd_expr_destroy_iterative(right);
        return ptd_expr_const(result);
    }

    // Create expression
    struct ptd_expression *expr = (struct ptd_expression *)calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate division expression");
    }
    expr->type = PTD_EXPR_DIV;
    expr->left = left;
    expr->right = right;

    // Intern if table provided
    if (table != NULL) {
        return ptd_expr_intern(table, expr);
    }
    return expr;
}

/**
 * Create subtraction expression with interning
 */
struct ptd_expression *ptd_expr_sub_interned(
    struct ptd_expr_intern_table *table,
    struct ptd_expression *left,
    struct ptd_expression *right
) {
    // Apply simplifications first
    if (right->type == PTD_EXPR_CONST && right->const_value == 0.0) {
        ptd_expr_destroy_iterative(right);
        return left;
    }
    if (left->type == PTD_EXPR_CONST && right->type == PTD_EXPR_CONST) {
        double result = left->const_value - right->const_value;
        ptd_expr_destroy_iterative(left);
        ptd_expr_destroy_iterative(right);
        return ptd_expr_const(result);
    }

    // Create expression
    struct ptd_expression *expr = (struct ptd_expression *)calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate subtraction expression");
    }
    expr->type = PTD_EXPR_SUB;
    expr->left = left;
    expr->right = right;

    // Intern if table provided
    if (table != NULL) {
        return ptd_expr_intern(table, expr);
    }
    return expr;
}

/**
 * Create inversion expression with interning
 */
struct ptd_expression *ptd_expr_inv_interned(
    struct ptd_expr_intern_table *table,
    struct ptd_expression *child
) {
    // Simplification: inv(const) = const(1/c)
    if (child->type == PTD_EXPR_CONST) {
        if (child->const_value == 0.0) {
            DIE_ERROR(1, "Division by zero in constant inversion");
        }
        double result = 1.0 / child->const_value;
        ptd_expr_destroy_iterative(child);
        return ptd_expr_const(result);
    }

    // Create expression
    struct ptd_expression *expr = (struct ptd_expression *)calloc(1, sizeof(*expr));
    if (expr == NULL) {
        DIE_ERROR(1, "Failed to allocate inversion expression");
    }
    expr->type = PTD_EXPR_INV;
    expr->left = child;

    // Intern if table provided
    if (table != NULL) {
        return ptd_expr_intern(table, expr);
    }
    return expr;
}
```

### Step 2.4: Update Header

Add to `/Users/kmt/PtDAlgorithms/api/c/ptdalgorithms.h` (in expression section):

```c
// Expression interning for CSE
struct ptd_expr_intern_table;

struct ptd_expr_intern_table *ptd_expr_intern_table_create(size_t capacity);
void ptd_expr_intern_table_destroy(struct ptd_expr_intern_table *table);
struct ptd_expression *ptd_expr_intern(struct ptd_expr_intern_table *table,
                                        struct ptd_expression *expr);
void ptd_expr_intern_table_stats(const struct ptd_expr_intern_table *table);

// Interned expression constructors
struct ptd_expression *ptd_expr_add_interned(struct ptd_expr_intern_table *table,
                                              struct ptd_expression *left,
                                              struct ptd_expression *right);
struct ptd_expression *ptd_expr_mul_interned(struct ptd_expr_intern_table *table,
                                              struct ptd_expression *left,
                                              struct ptd_expression *right);
struct ptd_expression *ptd_expr_div_interned(struct ptd_expr_intern_table *table,
                                              struct ptd_expression *left,
                                              struct ptd_expression *right);
struct ptd_expression *ptd_expr_sub_interned(struct ptd_expr_intern_table *table,
                                              struct ptd_expression *left,
                                              struct ptd_expression *right);
struct ptd_expression *ptd_expr_inv_interned(struct ptd_expr_intern_table *table,
                                              struct ptd_expression *child);

// Expression utilities
uint64_t ptd_expr_hash(const struct ptd_expression *expr);
bool ptd_expr_equal(const struct ptd_expression *a, const struct ptd_expression *b);
```

### Testing Phase 2

**Unit test** (create `test_cse_unit.py`):
```python
from ctypes import CDLL, c_void_p, c_uint64, c_bool
import os

lib = CDLL("libptdalgorithms.so")

def test_expr_hashing():
    """Test that identical expressions have same hash"""
    # This requires exposing test functions or using Python API
    pass

def test_intern_table():
    """Test basic interning functionality"""
    pass
```

---

## Phase 3: Integration into Elimination (Days 6-7)

### Goal

Use intern table throughout symbolic elimination algorithm.

### Files to Modify

- `/Users/kmt/PtDAlgorithms/src/c/ptdalgorithms_symbolic.c`

### Step 3.1: Add Intern Table to Elimination

Modify `ptd_graph_symbolic_elimination()` function (line 299):

```c
struct ptd_graph_symbolic *ptd_graph_symbolic_elimination(struct ptd_graph *graph) {
    // ... existing parameter detection code ...

    DEBUG_PRINT("INFO: Starting symbolic elimination (param_length=%zu, vertices=%zu)\n",
                param_length, graph->vertices_length);

    // CREATE INTERN TABLE (key change)
    // Use 4096 buckets initially (can tune based on profiling)
    struct ptd_expr_intern_table *intern_table =
        ptd_expr_intern_table_create(4096);

    if (intern_table == NULL) {
        DIE_ERROR(1, "Failed to create expression intern table");
    }

    // ... rest of existing setup code ...
```

### Step 3.2: Update sum_expressions Helper

Modify helper function (line 193):

```c
/**
 * Sum an array of expressions (with interning)
 */
static struct ptd_expression *sum_expressions_interned(
    struct ptd_expr_intern_table *intern_table,
    struct ptd_expression **exprs,
    size_t n
) {
    if (n == 0) {
        return ptd_expr_const(0.0);
    }
    if (n == 1) {
        return ptd_expr_copy_iterative(exprs[0]);
    }

    struct ptd_expression *sum = ptd_expr_copy_iterative(exprs[0]);
    for (size_t i = 1; i < n; i++) {
        sum = ptd_expr_add_interned(intern_table, sum,
                                     ptd_expr_copy_iterative(exprs[i]));
    }
    return sum;
}
```

### Step 3.3: Update Rate Expression Creation

Modify initial rate expression setup (line 437):

```c
// OLD:
struct ptd_expression *prob_expr =
    ptd_expr_mul(weight_expr, ptd_expr_copy_iterative(sv->rate_expr));

// NEW:
struct ptd_expression *prob_expr =
    ptd_expr_mul_interned(intern_table, weight_expr,
                          ptd_expr_copy_iterative(sv->rate_expr));
```

### Step 3.4: Update Self-Loop Handling

Modify self-loop scale (lines 507-521):

```c
// Self-loop probability
struct ptd_expression *loop_prob =
    ptd_expr_mul_interned(
        intern_table,
        ptd_expr_copy_iterative(parent_to_me_expr),
        ptd_expr_copy_iterative(me_to_child->prob_expr)
    );

struct ptd_expression *one_minus_prob =
    ptd_expr_sub_interned(intern_table, ptd_expr_const(1.0), loop_prob);

struct ptd_expression *scale =
    ptd_expr_inv_interned(intern_table, one_minus_prob);

if (self_loop != NULL) {
    self_loop->prob_expr =
        ptd_expr_mul_interned(intern_table, self_loop->prob_expr, scale);
}
```

### Step 3.5: Update Bypass Edge Creation (CRITICAL)

Modify Case B (line 540):

```c
// CASE B: Matching edge - add bypass probability
// OLD:
struct ptd_expression *bypass =
    ptd_expr_mul(
        ptd_expr_copy_iterative(parent_to_me_expr),
        ptd_expr_copy_iterative(me_to_child->prob_expr)
    );
parent_to_child->prob_expr =
    ptd_expr_add(parent_to_child->prob_expr, bypass);

// NEW:
struct ptd_expression *bypass =
    ptd_expr_mul_interned(
        intern_table,
        ptd_expr_copy_iterative(parent_to_me_expr),
        ptd_expr_copy_iterative(me_to_child->prob_expr)
    );
parent_to_child->prob_expr =
    ptd_expr_add_interned(intern_table, parent_to_child->prob_expr, bypass);
```

Modify Case C (line 550):

```c
// CASE C: New edge
// OLD:
struct ptd_expression *new_prob =
    ptd_expr_mul(
        ptd_expr_copy_iterative(parent_to_me_expr),
        ptd_expr_copy_iterative(me_to_child->prob_expr)
    );

// NEW:
struct ptd_expression *new_prob =
    ptd_expr_mul_interned(
        intern_table,
        ptd_expr_copy_iterative(parent_to_me_expr),
        ptd_expr_copy_iterative(me_to_child->prob_expr)
    );
```

### Step 3.6: Update Normalization

Modify edge normalization (line 588-593):

```c
// Compute sum of edge probabilities
struct ptd_expression *total = sum_expressions_interned(
    intern_table, parent_edge_exprs, parent_n_edges);

// Normalize each edge
edge = parent->first_edge->next;
while (edge != parent->last_edge) {
    edge->prob_expr = ptd_expr_div_interned(
        intern_table,
        edge->prob_expr,
        ptd_expr_copy_iterative(total)
    );
    edge = edge->next;
}
```

### Step 3.7: Update Rate Computation

Modify rate expression building (line 639):

```c
// OLD:
public_sv->rate_expr = ptd_expr_copy_iterative(sv->rate_expr);

// NEW:
public_sv->rate_expr = ptd_expr_copy_iterative(sv->rate_expr);
// Note: rate_expr already uses interned expressions during construction
```

### Step 3.8: Cleanup and Statistics

Add before return (after line 700):

```c
    // Print intern table statistics (for profiling)
    if (getenv("PTD_CSE_STATS")) {
        ptd_expr_intern_table_stats(intern_table);
    }

    // Clean up intern table
    // Note: Expressions are still referenced by result structure, so don't destroy them
    ptd_expr_intern_table_destroy(intern_table);

    return result;
}
```

### Testing Phase 3

**Test with environment variable**:
```bash
export PTD_CSE_STATS=1
pixi run python test_evaluate_performance.py 5
```

Expected output:
```
Expression Intern Table Statistics:
  Capacity: 4096
  Size: 1234 entries
  Load factor: 30.13%
  Total collisions: 156
  Empty buckets: 2862 (69.9%)
  Max chain length: 4
```

**Performance test**:
```bash
pixi run python test_evaluate_performance.py 7
```

Expected: Should complete in <10 seconds (was timing out before)

---

## Phase 4: Testing and Validation (Days 8-10)

### Goal

Verify correctness, performance, and stability.

### 4.1 Unit Tests

Create `tests/test_cse_correctness.py`:

```python
#!/usr/bin/env python3
"""
Unit tests for CSE implementation correctness
"""
import numpy as np
from ptdalgorithms import Graph

def construct_rabbit_model(n_rabbits):
    """Standard rabbit model constructor"""
    g = Graph(state_length=2)
    initial_state = [n_rabbits, 0]
    g.starting_vertex().add_edge(g.find_or_create_vertex(initial_state), 1)

    index = 1
    while index < g.vertices_length():
        vertex = g.vertex_at(index)
        state = vertex.state()

        if state[0] > 0:
            child_state = [state[0] - 1, state[1] + 1]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [state[0], 0, 0]
            )
            child_state = [0, state[1]]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [0, 1, 0]
            )

        if state[1] > 0:
            child_state = [state[0] + 1, state[1] - 1]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [state[1], 0, 0]
            )
            child_state = [state[0], 0]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [0, 0, 1]
            )

        index += 1

    return g


def test_numerical_accuracy():
    """Verify CSE doesn't change numerical results"""
    print("Testing numerical accuracy...")

    for n_rabbits in [3, 5, 7]:
        print(f"  {n_rabbits} rabbits...", end=" ")

        g = construct_rabbit_model(n_rabbits)
        g.update_parameterized_weights([1.0, 2.0, 4.0])

        dag = g.eliminate_to_dag()

        # Test multiple parameter sets
        params_list = [
            [0.5, 0.1, 0.1],
            [1.0, 0.5, 0.5],
            [2.0, 1.0, 1.0],
            [0.1, 0.05, 0.02],
        ]

        for params in params_list:
            inst = dag.instantiate(params)
            ewt = inst.expected_waiting_time()

            # Basic sanity checks
            assert len(ewt) == inst.vertices_length()
            assert all(x >= 0 for x in ewt), f"Negative waiting time: {ewt}"
            assert all(np.isfinite(x) for x in ewt), f"Non-finite values: {ewt}"

            # Check absorbing state has EWT = 0
            # (assuming last state is absorbing)
            # assert ewt[-1] == 0.0 or ewt[-1] < 1e-10

        print("✓")

    print("✓ Numerical accuracy tests passed")


def test_determinism():
    """Verify CSE produces deterministic results"""
    print("Testing determinism...")

    results = []
    for run in range(3):
        g = construct_rabbit_model(5)
        g.update_parameterized_weights([1.0, 2.0, 4.0])
        dag = g.eliminate_to_dag()
        inst = dag.instantiate([0.5, 0.1, 0.1])
        ewt = inst.expected_waiting_time()
        results.append(ewt)

    # All runs should produce identical results
    for i in range(1, len(results)):
        assert np.allclose(results[0], results[i]), \
            f"Run {i} differs from run 0"

    print("✓ Determinism test passed")


def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("Testing edge cases...")

    # Empty model
    g = Graph(state_length=1)
    g.update_parameterized_weights([1.0])
    dag = g.eliminate_to_dag()
    inst = dag.instantiate([0.5])

    # Single vertex
    g = Graph(state_length=1)
    v = g.find_or_create_vertex([0])
    g.starting_vertex().add_edge(v, 1.0)
    dag = g.eliminate_to_dag()
    inst = dag.instantiate([])

    print("✓ Edge cases passed")


if __name__ == "__main__":
    test_numerical_accuracy()
    test_determinism()
    test_edge_cases()
    print("\n✓ All correctness tests passed!")
```

### 4.2 Performance Tests

Create `tests/test_cse_performance.py`:

```python
#!/usr/bin/env python3
"""
Performance benchmarks for CSE implementation
"""
import time
import numpy as np
from ptdalgorithms import Graph

def construct_rabbit_model(n_rabbits):
    """Standard rabbit model constructor"""
    g = Graph(state_length=2)
    initial_state = [n_rabbits, 0]
    g.starting_vertex().add_edge(g.find_or_create_vertex(initial_state), 1)

    index = 1
    while index < g.vertices_length():
        vertex = g.vertex_at(index)
        state = vertex.state()

        if state[0] > 0:
            child_state = [state[0] - 1, state[1] + 1]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [state[0], 0, 0]
            )
            child_state = [0, state[1]]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [0, 1, 0]
            )

        if state[1] > 0:
            child_state = [state[0] + 1, state[1] - 1]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [state[1], 0, 0]
            )
            child_state = [state[0], 0]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [0, 0, 1]
            )

        index += 1

    return g


def test_scaling():
    """Test O(n²) scaling"""
    print("Testing performance scaling...")
    print("=" * 70)

    results = []
    test_sizes = [3, 5, 7, 10, 15]

    for n_rabbits in test_sizes:
        expected_vertices = ((n_rabbits + 1) * (n_rabbits + 2)) // 2
        print(f"\n{n_rabbits} rabbits ({expected_vertices} vertices):")

        # Construction
        start = time.time()
        g = construct_rabbit_model(n_rabbits)
        construct_time = time.time() - start
        actual_vertices = g.vertices_length()
        print(f"  Construction: {construct_time:.3f}s ({actual_vertices} vertices)")

        # Set parameters
        g.update_parameterized_weights([1.0, 2.0, 4.0])

        # Elimination
        start = time.time()
        dag = g.eliminate_to_dag()
        elim_time = time.time() - start
        print(f"  Elimination:  {elim_time:.3f}s")

        # Instantiation (10 runs)
        times = []
        for _ in range(10):
            start = time.time()
            inst = dag.instantiate([0.5, 0.1, 0.1])
            times.append(time.time() - start)
        inst_time = np.mean(times)
        inst_std = np.std(times)
        print(f"  Instantiation: {inst_time:.4f}s ± {inst_std:.4f}s (avg of 10)")
        print(f"  Total:        {elim_time + inst_time:.4f}s")

        results.append({
            'n_rabbits': n_rabbits,
            'vertices': actual_vertices,
            'construct_time': construct_time,
            'elim_time': elim_time,
            'inst_time': inst_time,
            'total_time': elim_time + inst_time
        })

    print("\n" + "=" * 70)
    print("Performance Summary:")
    print("=" * 70)
    print(f"{'Rabbits':<10} {'Vertices':<10} {'Total':<12} {'Inst':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['n_rabbits']:<10} {r['vertices']:<10} "
              f"{r['total_time']:<12.4f} {r['inst_time']:<12.4f}")

    # Check scaling
    print("\n" + "=" * 70)
    print("Scaling Analysis:")
    print("=" * 70)
    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]

        time_ratio = curr['inst_time'] / prev['inst_time']
        vertex_ratio = curr['vertices'] / prev['vertices']

        # O(n²) would give time_ratio ≈ vertex_ratio²
        expected_ratio = vertex_ratio ** 2

        print(f"{prev['vertices']} → {curr['vertices']} vertices:")
        print(f"  Time ratio:     {time_ratio:.2f}×")
        print(f"  Vertex ratio:   {vertex_ratio:.2f}×")
        print(f"  O(n²) expects:  {expected_ratio:.2f}×")

        # Check if within reasonable bounds (< 2.5× quadratic)
        if time_ratio > expected_ratio * 2.5:
            print(f"  ⚠️  WARNING: Worse than O(n²)!")
        else:
            print(f"  ✓  OK")

    # Final assessment
    print("\n" + "=" * 70)
    largest = results[-1]
    if largest['inst_time'] < 2.0:
        print(f"✓ SUCCESS: {largest['vertices']}-vertex model instantiates in "
              f"{largest['inst_time']:.3f}s")
    else:
        print(f"⚠️  WARNING: {largest['vertices']}-vertex model takes "
              f"{largest['inst_time']:.3f}s (target: <2s)")

    return results


def test_batch_instantiation():
    """Test performance of batch instantiation"""
    print("\n" + "=" * 70)
    print("Batch Instantiation Performance:")
    print("=" * 70)

    g = construct_rabbit_model(7)
    g.update_parameterized_weights([1.0, 2.0, 4.0])
    dag = g.eliminate_to_dag()

    # Generate 100 random parameter sets
    np.random.seed(42)
    param_sets = [
        [np.random.uniform(0.1, 2.0) for _ in range(3)]
        for _ in range(100)
    ]

    start = time.time()
    for params in param_sets:
        inst = dag.instantiate(params)
    elapsed = time.time() - start

    print(f"100 instantiations: {elapsed:.3f}s ({elapsed/100:.4f}s each)")

    if elapsed / 100 < 0.1:
        print("✓ Batch instantiation performance acceptable")
    else:
        print("⚠️  Batch instantiation slower than expected")


if __name__ == "__main__":
    results = test_scaling()
    test_batch_instantiation()
    print("\n✓ All performance tests complete!")
```

### 4.3 Integration Test: Full Notebook

**CRITICAL**: Run the complete tutorial notebook:

```bash
cd /Users/kmt/PtDAlgorithms
pixi run jupyter nbconvert --to notebook --execute \
    docs/pages/tutorials/rabbits_full_py_api_example.ipynb \
    --output test_output/rabbits_full_test.ipynb
```

**Success criteria**:
- ✅ All cells execute without errors
- ✅ All outputs match expected values
- ✅ Execution completes in <5 minutes total
- ✅ No memory errors or warnings

### 4.4 Stress Tests

Create `tests/test_cse_stress.py`:

```python
#!/usr/bin/env python3
"""
Stress tests for large models
"""
import time
from ptdalgorithms import Graph

def construct_rabbit_model(n_rabbits):
    """Standard rabbit model constructor"""
    g = Graph(state_length=2)
    initial_state = [n_rabbits, 0]
    g.starting_vertex().add_edge(g.find_or_create_vertex(initial_state), 1)

    index = 1
    while index < g.vertices_length():
        vertex = g.vertex_at(index)
        state = vertex.state()

        if state[0] > 0:
            child_state = [state[0] - 1, state[1] + 1]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [state[0], 0, 0]
            )
            child_state = [0, state[1]]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [0, 1, 0]
            )

        if state[1] > 0:
            child_state = [state[0] + 1, state[1] - 1]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [state[1], 0, 0]
            )
            child_state = [state[0], 0]
            vertex.add_edge_parameterized(
                g.find_or_create_vertex(child_state), 0, [0, 0, 1]
            )

        index += 1

    return g


def test_67_vertex_model():
    """Test 67-vertex model (was timing out before CSE)"""
    print("=" * 70)
    print("TEST: 67-Vertex Model (10 rabbits)")
    print("=" * 70)
    print("\nThis was the model that originally timed out.")
    print("Target: Complete in <60 seconds\n")

    n_rabbits = 10

    print("Constructing model...")
    start = time.time()
    g = construct_rabbit_model(n_rabbits)
    construct_time = time.time() - start
    print(f"✓ Constructed in {construct_time:.2f}s ({g.vertices_length()} vertices)")

    print("\nSetting parameters...")
    g.update_parameterized_weights([1.0, 2.0, 4.0])

    print("Running symbolic elimination...")
    start = time.time()
    dag = g.eliminate_to_dag()
    elim_time = time.time() - start
    print(f"✓ Elimination complete in {elim_time:.2f}s")
    print(f"  DAG vertices: {dag.vertices_length}")
    print(f"  Is acyclic: {dag.is_acyclic}")

    print("\nInstantiating with parameters...")
    start = time.time()
    inst = dag.instantiate([0.5, 0.1, 0.1])
    inst_time = time.time() - start
    print(f"✓ Instantiation complete in {inst_time:.4f}s")

    total_time = construct_time + elim_time + inst_time
    print(f"\n{'='*70}")
    print(f"Total time: {total_time:.2f}s")

    # Thresholds
    assert elim_time < 60, f"Elimination too slow: {elim_time:.2f}s"
    assert inst_time < 1.0, f"Instantiation too slow: {inst_time:.4f}s"

    print("✓ 67-vertex model works within time limits!")
    print("=" * 70)


def test_100_vertex_model():
    """Push to 100+ vertices"""
    print("\n" + "=" * 70)
    print("TEST: 136-Vertex Model (15 rabbits)")
    print("=" * 70)
    print("\nStress test with larger model.")
    print("Target: Complete in <3 minutes\n")

    n_rabbits = 15

    print("Constructing model...")
    start = time.time()
    g = construct_rabbit_model(n_rabbits)
    construct_time = time.time() - start
    print(f"✓ Constructed in {construct_time:.2f}s ({g.vertices_length()} vertices)")

    print("\nSetting parameters...")
    g.update_parameterized_weights([1.0, 2.0, 4.0])

    print("Running symbolic elimination...")
    start = time.time()
    dag = g.eliminate_to_dag()
    elim_time = time.time() - start
    print(f"✓ Elimination complete in {elim_time:.2f}s")

    print("\nInstantiating...")
    start = time.time()
    inst = dag.instantiate([0.5, 0.1, 0.1])
    inst_time = time.time() - start
    print(f"✓ Instantiation complete in {inst_time:.4f}s")

    total_time = construct_time + elim_time + inst_time
    print(f"\nTotal time: {total_time:.2f}s")

    if total_time < 180:
        print("✓ 136-vertex model works!")
    else:
        print(f"⚠️  136-vertex model slow: {total_time:.2f}s")

    print("=" * 70)


def test_memory_usage():
    """Monitor memory usage for large models"""
    import psutil
    import os

    print("\n" + "=" * 70)
    print("Memory Usage Test")
    print("=" * 70)

    process = psutil.Process(os.getpid())
    initial_mem = process.memory_info().rss / 1024 / 1024  # MB

    print(f"Initial memory: {initial_mem:.1f} MB")

    # Build large model
    g = construct_rabbit_model(15)
    g.update_parameterized_weights([1.0, 2.0, 4.0])

    mem_after_build = process.memory_info().rss / 1024 / 1024
    print(f"After build: {mem_after_build:.1f} MB (+{mem_after_build - initial_mem:.1f} MB)")

    # Eliminate
    dag = g.eliminate_to_dag()

    mem_after_elim = process.memory_info().rss / 1024 / 1024
    print(f"After elimination: {mem_after_elim:.1f} MB (+{mem_after_elim - mem_after_build:.1f} MB)")

    # Instantiate
    inst = dag.instantiate([0.5, 0.1, 0.1])

    mem_after_inst = process.memory_info().rss / 1024 / 1024
    print(f"After instantiation: {mem_after_inst:.1f} MB (+{mem_after_inst - mem_after_elim:.1f} MB)")

    peak_mem = mem_after_inst
    if peak_mem < 2000:  # 2GB threshold
        print(f"✓ Memory usage acceptable: {peak_mem:.1f} MB")
    else:
        print(f"⚠️  High memory usage: {peak_mem:.1f} MB")

    print("=" * 70)


if __name__ == "__main__":
    test_67_vertex_model()
    test_100_vertex_model()
    test_memory_usage()
    print("\n✓ All stress tests complete!")
```

### 4.5 Memory Leak Check

```bash
# Install valgrind if not present
# brew install valgrind  # (macOS may have issues, use Linux if possible)

# Run with valgrind
valgrind --leak-check=full --show-leak-kinds=all \
    --track-origins=yes \
    pixi run python tests/test_cse_stress.py
```

**Success criteria**:
- No "definitely lost" blocks
- No "indirectly lost" blocks
- All memory properly freed

---

## Phase 5: Documentation and Polish (Days 11-12)

### Goal

Document changes and update examples.

### 5.1 Update CLAUDE.md

Add new section to `/Users/kmt/PtDAlgorithms/CLAUDE.md`:

```markdown
## Common Subexpression Elimination (CSE)

### Overview

The symbolic elimination system uses **expression interning** to implement Common Subexpression Elimination (CSE). This prevents exponential growth of expression trees during Gaussian elimination.

### Architecture

**Intern Table**: Hash table mapping `hash(expression) → canonical_expression`

**Key Functions**:
- `ptd_expr_hash()`: Computes structural hash (FNV-1a-based)
- `ptd_expr_equal()`: Deep structural equality check (handles commutativity)
- `ptd_expr_intern()`: Returns existing expression or adds new one

### Usage in Elimination

```c
// Create intern table
struct ptd_expr_intern_table *table = ptd_expr_intern_table_create(4096);

// Use interned constructors
struct ptd_expression *sum = ptd_expr_add_interned(table, left, right);
struct ptd_expression *product = ptd_expr_mul_interned(table, a, b);

// Cleanup (doesn't destroy expressions, just table structure)
ptd_expr_intern_table_destroy(table);
```

### Performance Impact

**Before CSE**:
- 22 vertices: 0.12s instantiation
- 37 vertices: >180s (timeout)
- 67 vertices: timeout

**After CSE**:
- 22 vertices: 0.005s instantiation (24× faster)
- 37 vertices: 0.05s (>3600× faster)
- 67 vertices: 0.3s (∞ faster - was impossible)

### Debugging

Enable statistics:
```bash
export PTD_CSE_STATS=1
python your_script.py
```

Output:
```
Expression Intern Table Statistics:
  Capacity: 4096
  Size: 1234 entries
  Load factor: 30.13%
  Max chain length: 4
```

### Design Decisions

**Why not reference counting?**
- Risk of reference cycles
- More invasive changes
- Harder to debug memory leaks

**Why not DAG representation?**
- Major API breaking change
- Extensive refactoring required
- Intern table achieves same goal with minimal changes

**Hash function choice:**
- FNV-1a for simplicity and speed
- Commutative ops sorted for consistency
- 64-bit hash: collision risk ~1 in 10^9 for 1M expressions
```

### 5.2 Update Tutorial Notebook

Modify `/Users/kmt/PtDAlgorithms/docs/pages/tutorials/rabbits_full_py_api_example.ipynb`:

**Change 1**: Increase rabbit count from 5 to 10:
```python
# OLD:
n_rabbits = 5  # Limited due to performance

# NEW:
n_rabbits = 10  # Now works thanks to CSE!
```

**Change 2**: Add performance comparison cell:
```python
# Add new cell:
"""
## Performance Comparison

Let's demonstrate the CSE performance improvement by testing different model sizes:
"""

import time
import pandas as pd
import matplotlib.pyplot as plt

results = []
for n in [3, 5, 7, 10]:
    g = construct_rabbit_model(n)
    g.update_parameterized_weights([1.0, 2.0, 4.0])

    start = time.time()
    dag = g.eliminate_to_dag()
    inst = dag.instantiate([0.5, 0.1, 0.1])
    elapsed = time.time() - start

    results.append({
        'Rabbits': n,
        'Vertices': g.vertices_length(),
        'Time (s)': elapsed
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df['Vertices'], df['Time (s)'], 'o-', linewidth=2, markersize=8)
ax.set_xlabel('Number of Vertices', fontsize=12)
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_title('Symbolic Elimination + Instantiation Performance', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n✓ The 10-rabbit model ({df.iloc[-1]['Vertices']} vertices) " +
      f"completes in {df.iloc[-1]['Time (s)']:.2f}s!")
```

### 5.3 Create Migration Guide

Create `/Users/kmt/PtDAlgorithms/docs/CSE_MIGRATION.md`:

```markdown
# CSE Implementation Migration Guide

## What Changed

### Internal Changes

1. **Expression constructors now use interning**
   - `ptd_expr_add()`, `ptd_expr_mul()`, etc. now check intern table
   - Identical expressions share single instance

2. **Algebraic simplification added**
   - `x * 0 → 0`, `x * 1 → x`, `x + 0 → x`
   - Constant folding: `c1 * c2 → c3`

3. **New data structures**
   - `struct ptd_expr_intern_table`: Hash table for CSE
   - Intern table created/destroyed within symbolic elimination

### API Compatibility

✅ **100% backward compatible**
- All public API unchanged
- No changes to Python bindings
- Existing code works without modification

### Performance Changes

**Dramatic improvements for large models:**
- Small models (<20 vertices): No change or slight improvement
- Medium models (20-50 vertices): 10-100× faster
- Large models (>50 vertices): Previously impossible, now feasible

### Memory Usage

**Before**: O(exp(n)) due to expression duplication
**After**: O(n²) with CSE deduplication

**Typical savings**: 50-90% reduction in peak memory for large models

## Migration Steps

### No action required!

Your existing code will automatically benefit from CSE. No changes needed.

### Optional: Enable Statistics

```bash
export PTD_CSE_STATS=1
python your_script.py
```

### Optional: Disable CSE (for comparison)

```bash
export PTD_DISABLE_CSE=1
python your_script.py
```

## Troubleshooting

### "Expression hash collision" warning

Extremely rare (<1 in 10^9 for typical models). If seen:
1. Report as bug with model details
2. Workaround: `export PTD_DISABLE_CSE=1`

### Performance regression

If you observe slower performance after upgrade:
1. Check model size (CSE has ~5% overhead for small models)
2. Verify correct installation: `pip install --force-reinstall`
3. Report issue with benchmark script

### Memory leak

CSE should not introduce leaks. If suspected:
1. Run with valgrind
2. Compare `PTD_DISABLE_CSE=1` vs `PTD_DISABLE_CSE=0`
3. Report with reproduction steps

## Performance Benchmarks

Run included benchmark:
```bash
python tests/test_cse_performance.py
```

Expected output:
```
Rabbits    Vertices   Total        Inst
----------------------------------------------
3          11         0.0010       0.0010
5          22         0.2600       0.0050
7          37         1.2500       0.0500
10         67         5.3000       0.3000
15         136        16.2000      1.2000
```

## Support

Questions or issues: https://github.com/munch-group/ptdalgorithms/issues
```

### 5.4 Update Test Suite

Add to existing test infrastructure:

```bash
# In pyproject.toml or test configuration
[tool.pytest.ini_options]
markers = [
    "cse: Tests specific to CSE implementation",
    "slow: Tests that take >10 seconds",
]

# Run CSE-specific tests
pixi run pytest -m cse

# Run all tests including slow ones
pixi run pytest -m "slow"
```

---

## Risk Management

### Fallback Mechanism

Keep ability to disable CSE:

```c
// In ptd_graph_symbolic_elimination():
bool use_cse = (getenv("PTD_DISABLE_CSE") == NULL);

struct ptd_expr_intern_table *intern_table = use_cse ?
    ptd_expr_intern_table_create(4096) : NULL;

// In expression constructors:
if (intern_table != NULL) {
    return ptd_expr_intern(intern_table, expr);
}
return expr;  // Non-interned fallback
```

### Debugging Aids

```c
// Add DEBUG_PRINT for intern operations
#ifdef PTD_DEBUG
#define INTERN_DEBUG_PRINT(...) fprintf(stderr, __VA_ARGS__)
#else
#define INTERN_DEBUG_PRINT(...)
#endif

// In ptd_expr_intern():
INTERN_DEBUG_PRINT("Interning %s expression (hash=%lu)\n",
                   expr_type_name(expr->type), hash);
if (existing) {
    INTERN_DEBUG_PRINT("  Found existing (saved copy)\n");
}
```

### Validation Tests

Add assertion checks (disabled in production):

```c
#ifdef PTD_VALIDATE_CSE
// After interning, verify hash consistency
uint64_t hash1 = ptd_expr_hash(expr);
uint64_t hash2 = ptd_expr_hash(result);
assert(hash1 == hash2 && "Hash changed after interning");
#endif
```

---

## Success Metrics

### Must-Have (Blocking Issues)

- ✅ All existing tests pass
- ✅ Full notebook executes successfully
- ✅ 67-vertex model instantiates in <1 second
- ✅ No memory leaks (valgrind clean)
- ✅ Numerical results unchanged

### Should-Have (Quality)

- ✅ 100-vertex model completes in <2 minutes
- ✅ Memory usage <2GB for 100-vertex model
- ✅ O(n²) scaling verified empirically
- ✅ Intern table load factor 20-40%

### Nice-to-Have (Polish)

- ✅ Comprehensive documentation
- ✅ Performance comparison plots
- ✅ Debugging utilities
- ✅ Statistics collection

---

## Timeline and Resources

### Timeline

| Phase | Duration | Dependencies | Risk Level |
|-------|----------|--------------|------------|
| 1     | 2 days   | None         | Low        |
| 2     | 3 days   | Phase 1      | Medium     |
| 3     | 2 days   | Phase 2      | High       |
| 4     | 3 days   | Phase 3      | Medium     |
| 5     | 2 days   | Phase 4      | Low        |
| **Total** | **12 days** | | |

### Critical Path

Phase 2 → Phase 3 → Phase 4

Phases 1 and 5 can proceed in parallel if resources available.

### Checkpoints

- **End of Phase 1**: 2× speedup for 22-vertex model
- **End of Phase 2**: Unit tests pass, intern table functional
- **End of Phase 3**: 37-vertex model completes (was timing out)
- **End of Phase 4**: 67-vertex model works, all tests pass
- **End of Phase 5**: Documentation complete, ready for release

---

## Conclusion

This implementation plan provides a systematic approach to implementing Common Subexpression Elimination via expression interning. The solution is:

- **Effective**: Achieves O(n²) complexity as intended
- **Minimal**: Internal changes only, no API breakage
- **Testable**: Comprehensive test suite ensures correctness
- **Maintainable**: Clear code with debugging support

**Expected Outcome**: Enable 67+ vertex models that were previously impossible, with 100-1000× performance improvements.

---

**Next Steps**: Review this plan, then start implementation in a fresh conversation with:

```
Implement Phase 1 of CSE_IMPLEMENTATION_PLAN.md
```
