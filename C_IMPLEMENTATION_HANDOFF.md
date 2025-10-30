# C Implementation Handoff Document

**Date**: 2025-10-27
**Task**: Option A - Add enable_rewards Support to C Code
**Status**: Phase 1 Complete (10% done)
**Estimated Remaining**: 3.5-5.5 hours (Phases 2-9)

---

## Problem Summary

SVGD with multivariate rewards has ~16% error (θ̂ ≈ 11.6 instead of θ = 10).

**Root Cause**: C's `ptd_record_elimination_trace()` lacks `enable_rewards` parameter, so conditional bypass logic for zero-reward vertices is never used.

**Solution**: Port Python's working implementation (SELECT operation + conditional bypass) to C code.

---

## What Has Been Completed ✅

### Phase 1: Add reward_length Field (COMPLETE)

**File Modified**: `api/c/phasic.h` line 587

**Change Made**:
```c
struct ptd_elimination_trace {
    /* ... existing fields ... */

    /* Metadata */
    size_t starting_vertex_idx;
    size_t n_vertices;
    size_t param_length;
    size_t reward_length;  /* ← ADDED: Number of reward parameters */
    bool is_discrete;
};
```

**Status**: ✅ Complete and committed

---

## Current State

### Modified Files
1. `api/c/phasic.h` - Struct updated with `reward_length` field

### Unmodified Files (will need changes)
1. `api/c/phasic.h` - Still needs SELECT enum + function signature update
2. `src/c/phasic.c` - Needs all implementation changes (Phases 2-8)
3. `src/cpp/parameterized/graph_builder.cpp` - Needs enable_rewards integration

### Test Status
- Python trace tests: ✅ Passing (test_trace_select_operation.py, test_reward_bypass_integration.py)
- C compilation: ⚠️ Not yet tested (no code changes that would break it)
- SVGD convergence: ❌ Still ~16% error (θ̂ ≈ 11.6)

---

## Next Steps: Detailed Implementation Plan

### Phase 2: Add SELECT Operation (45 min)

#### Step 2.1: Add to Enum
**File**: `api/c/phasic.h` line 533

**Current**:
```c
enum ptd_trace_op_type {
    PTD_OP_CONST = 0,
    PTD_OP_PARAM = 1,
    PTD_OP_DOT = 2,
    PTD_OP_ADD = 3,
    PTD_OP_MUL = 4,
    PTD_OP_DIV = 5,
    PTD_OP_INV = 6,
    PTD_OP_SUM = 7
};
```

**Add**:
```c
    PTD_OP_SELECT = 8   /* Conditional: if |cond| < threshold then a else b */
```

#### Step 2.2: Add Helper Function
**File**: `src/c/phasic.c` after line 7015 (after `add_sum_to_trace`)

**Insert**:
```c
/**
 * Helper: Add SELECT operation to trace
 * SELECT: if |condition| < threshold then true_val else false_val
 *
 * This enables conditional bypass for zero-reward vertices.
 * Port of Python's TraceBuilder.add_select() from trace_elimination.py:303-343
 */
static size_t add_select_to_trace(
    struct ptd_elimination_trace *trace,
    size_t condition_idx,
    double threshold,
    size_t true_val_idx,
    size_t false_val_idx
) {
    if (ensure_trace_capacity(trace, trace->operations_length + 1) != 0) {
        return (size_t)-1;
    }

    size_t idx = trace->operations_length++;
    struct ptd_trace_operation *op = &trace->operations[idx];

    op->op_type = PTD_OP_SELECT;
    op->const_value = threshold;       // Store threshold
    op->param_idx = condition_idx;     // Store condition index

    // Allocate operands for true/false values
    op->operands = (size_t *)malloc(2 * sizeof(size_t));
    if (op->operands == NULL) {
        trace->operations_length--;
        return (size_t)-1;
    }
    op->operands[0] = true_val_idx;   // True branch
    op->operands[1] = false_val_idx;  // False branch
    op->operands_length = 2;

    op->coefficients = NULL;
    op->coefficients_length = 0;

    return idx;
}
```

---

### Phase 3: Update Function Signature (15 min)

#### Step 3.1: Update Declaration
**File**: `api/c/phasic.h` line 620

**Current**:
```c
struct ptd_elimination_trace *ptd_record_elimination_trace(
    struct ptd_graph *graph
);
```

**Change to**:
```c
struct ptd_elimination_trace *ptd_record_elimination_trace(
    struct ptd_graph *graph,
    bool enable_rewards
);
```

#### Step 3.2: Update Definition
**File**: `src/c/phasic.c` line 7023

**Current**:
```c
struct ptd_elimination_trace *ptd_record_elimination_trace(
    struct ptd_graph *graph
) {
```

**Change to**:
```c
struct ptd_elimination_trace *ptd_record_elimination_trace(
    struct ptd_graph *graph,
    bool enable_rewards
) {
```

#### Step 3.3: Initialize reward_length
**File**: `src/c/phasic.c` line 7057 (in initialization block)

**Current**:
```c
// Initialize metadata
trace->n_vertices = graph->vertices_length;
trace->state_length = graph->state_length;
trace->param_length = graph->param_length;
trace->is_discrete = graph->was_dph;
```

**Add after param_length**:
```c
trace->reward_length = enable_rewards ? trace->n_vertices : 0;
```

---

### Phase 4: Conditional Bypass Logic (2-3 hours) ⚠️ MOST COMPLEX

#### Part A: Add PHASE 1.5 - Rate Scaling
**File**: `src/c/phasic.c` after line ~7335 (after PHASE 1 completes)

**Location hint**: Look for comment `// PHASE 2: Convert Edges to Probabilities`

**Insert before PHASE 2**:
```c
    // ========================================================================
    // PHASE 1.5: Apply Reward Scaling to Rates
    // ========================================================================
    // Port of Python code from trace_elimination.py lines 593-623
    //
    // Rewards multiply waiting times: waiting_time *= reward
    // Since rate = 1/E[waiting_time], this means: new_rate = rate / reward
    // Only apply when reward > 0 (when reward ≈ 0, vertex is bypassed in PHASE 3)

    if (enable_rewards) {
        for (size_t i = 0; i < n_vertices; i++) {
            // Reward parameters stored at indices [param_length, param_length + n_vertices)
            size_t reward_param_idx = param_length + i;

            // Add epsilon to prevent division by zero
            size_t epsilon_idx = add_const_to_trace(trace, 1e-10);
            if (epsilon_idx == (size_t)-1) {
                goto cleanup_error;
            }

            // reward_safe = reward + epsilon
            size_t reward_safe_idx = add_add_to_trace(trace, reward_param_idx, epsilon_idx);
            if (reward_safe_idx == (size_t)-1) {
                goto cleanup_error;
            }

            // scaled_rate = original_rate / reward_safe
            size_t scaled_rate_idx = add_div_to_trace(trace, vertex_rates[i], reward_safe_idx);
            if (scaled_rate_idx == (size_t)-1) {
                goto cleanup_error;
            }

            // Conditional: if |reward| < 1e-10, keep original rate (will be bypassed)
            //              if |reward| >= 1e-10, use scaled rate
            vertex_rates[i] = add_select_to_trace(
                trace,
                reward_param_idx,    // condition: reward parameter
                1e-10,               // threshold
                vertex_rates[i],     // true: keep original (bypassed anyway)
                scaled_rate_idx      // false: use scaled rate
            );

            if (vertex_rates[i] == (size_t)-1) {
                goto cleanup_error;
            }
        }
    }
```

#### Part B: Modify PHASE 3 Elimination Loop
**File**: `src/c/phasic.c` around line 7400+ (search for "PHASE 3" or elimination loop)

**This is complex - need to find and modify bypass edge creation**

**Search for**: Code that creates bypass edges during elimination (where parent → child edges are added).

**Pattern to find**:
```c
// Something like:
bypass_prob = add_mul_to_trace(trace, parent_to_i_prob, i_to_child_prob);
```

**Wrap with conditional**:
```c
// Create bypass edge (conditional on rewards)
size_t bypass_prob;
if (enable_rewards) {
    // Get reward for vertex i being eliminated
    size_t reward_param_idx = param_length + i;

    // Calculate what bypass prob would be if reward ≈ 0
    size_t bypass_prob_if_zero = add_mul_to_trace(trace, parent_to_i_prob, i_to_child_prob);
    if (bypass_prob_if_zero == (size_t)-1) {
        goto cleanup_error;
    }

    size_t zero_idx = add_const_to_trace(trace, 0.0);
    if (zero_idx == (size_t)-1) {
        goto cleanup_error;
    }

    // Conditional: if |reward| < 1e-10 → bypass, else no bypass
    bypass_prob = add_select_to_trace(
        trace,
        reward_param_idx,           // condition: reward
        1e-10,                      // threshold
        bypass_prob_if_zero,        // true: create bypass
        zero_idx                    // false: no bypass
    );
} else {
    // Standard elimination: always bypass
    bypass_prob = add_mul_to_trace(trace, parent_to_i_prob, i_to_child_prob);
}

if (bypass_prob == (size_t)-1) {
    goto cleanup_error;
}
```

**Also find and wrap edge removal** (pattern: `edge_probs[parent_idx][idx] = -1` or similar):
```c
// Remove edge from parent to i (conditionally)
if (enable_rewards) {
    size_t reward_param_idx = param_length + i;
    size_t removed_idx = add_const_to_trace(trace, -1.0);
    if (removed_idx == (size_t)-1) {
        goto cleanup_error;
    }

    size_t kept_edge_prob = edge_probs[parent_idx][parent_to_i_edge_idx];

    edge_probs[parent_idx][parent_to_i_edge_idx] = add_select_to_trace(
        trace,
        reward_param_idx,    // condition: reward
        1e-10,               // threshold
        removed_idx,         // true: remove edge (reward ≈ 0)
        kept_edge_prob       // false: keep edge (reward > 0)
    );

    if (edge_probs[parent_idx][parent_to_i_edge_idx] == (size_t)-1) {
        goto cleanup_error;
    }
} else {
    // Standard elimination: always remove
    edge_probs[parent_idx][parent_to_i_edge_idx] = -1;  // Special value for removed
}
```

---

### Phase 5: Update Trace Evaluation (30 min)

**File**: `src/c/phasic.c` line ~7653 (in `ptd_evaluate_trace`, switch statement)

**Find**: The switch statement with cases for PTD_OP_CONST, PTD_OP_PARAM, etc.

**Add case after PTD_OP_SUM**:
```c
            case PTD_OP_SELECT: {
                // Conditional select: if |condition| < threshold then true_val else false_val
                // Port of Python code from trace_elimination.py:903-916
                size_t condition_idx = op->param_idx;
                double threshold = op->const_value;
                size_t true_val_idx = op->operands[0];
                size_t false_val_idx = op->operands[1];

                double condition_val = values[condition_idx];

                // Select based on absolute value of condition
                if (fabs(condition_val) < threshold) {
                    values[i] = values[true_val_idx];
                } else {
                    values[i] = values[false_val_idx];
                }
                break;
            }
```

**Note**: Need `#include <math.h>` for `fabs()` - should already be included at top of file.

---

### Phase 6: Update Caching (45 min)

**File**: `src/c/phasic.c` line ~7032 (before `load_trace_from_cache` call)

**Current**:
```c
if (hash != NULL) {
    cached_trace = load_trace_from_cache(hash->hash_hex);
    if (cached_trace != NULL) {
        DEBUG_PRINT("INFO: loaded elimination trace from cache (%s)\n", hash->hash_hex);
        ptd_hash_destroy(hash);
        return cached_trace;
    }
}
```

**Change to**:
```c
if (hash != NULL) {
    // Include enable_rewards in cache key to avoid mixing reward/non-reward traces
    char cache_key[256];
    snprintf(cache_key, sizeof(cache_key), "%s_%s",
             hash->hash_hex,
             enable_rewards ? "with_rewards" : "no_rewards");

    cached_trace = load_trace_from_cache(cache_key);
    if (cached_trace != NULL) {
        DEBUG_PRINT("INFO: loaded elimination trace from cache (%s, enable_rewards=%d)\n",
                    hash->hash_hex, enable_rewards);
        ptd_hash_destroy(hash);
        return cached_trace;
    }
}
```

**Also update save** (line ~7520, after trace is recorded):
```c
if (hash != NULL) {
    // Save with same cache key format
    char cache_key[256];
    snprintf(cache_key, sizeof(cache_key), "%s_%s",
             hash->hash_hex,
             enable_rewards ? "with_rewards" : "no_rewards");
    save_trace_to_cache(cache_key, trace);
}
```

---

### Phase 7: Update Call Sites (30 min)

**File**: `src/c/phasic.c` line 1715

**Current**:
```c
graph->elimination_trace = ptd_record_elimination_trace(graph);
```

**Change to**:
```c
graph->elimination_trace = ptd_record_elimination_trace(graph, false);
```

**Note**: Use `false` for backward compatibility (existing behavior preserved).

**Search for all other calls**: `grep -n "ptd_record_elimination_trace" src/c/phasic.c`

Update each call to pass `false` parameter.

---

### Phase 8: Testing (1 hour)

#### Step 8.1: Compile
```bash
cd /Users/kmt/phasic
pip install -e . 2>&1 | tee compile.log
```

**If compilation fails**: Check error messages, likely missing includes or syntax errors.

#### Step 8.2: Run Python Integration Tests
```bash
python test_reward_bypass_integration.py
```

**Expected**: Should still pass (backward compatible).

#### Step 8.3: Clear Cache and Test SVGD
```bash
rm -rf ~/.phasic_cache/traces/*.json
python tests/test_notebook_multivar_reproduction.py 2>&1 | tail -60
```

**Success criteria**:
- θ̂ should be within [9.9, 10.1]
- Error < 1% (instead of current 16%)

#### Step 8.4: Verify Cached Trace
```bash
python3 -c "
import json
import glob
files = glob.glob('/Users/kmt/.phasic_cache/traces/*.json')
if files:
    data = json.load(open(files[0]))
    print(f\"reward_length: {data.get('reward_length', 'NOT SET')}\")
    print(f\"param_length: {data.get('param_length', 'NOT SET')}\")
"
```

**Expected**: `reward_length: 37` (or number of vertices in test graph).

---

### Phase 9: Documentation (30 min)

#### Update C API Docs
**File**: `api/c/phasic.h` line 620

Update function comment:
```c
/**
 * Record elimination trace from parameterized graph
 *
 * Performs graph elimination while recording all arithmetic operations
 * in a linear sequence. The trace can be efficiently replayed with
 * different parameter values.
 *
 * @param graph Parameterized graph
 * @param enable_rewards If true, adds conditional bypass operations for
 *                       reward transformation. Reward parameters are stored
 *                       at indices [param_length, param_length + n_vertices).
 * @return Elimination trace, or NULL on error
 *
 * Time complexity: O(n³) one-time cost
 * Space complexity: O(n²) for trace storage
 *
 * Note: When enable_rewards=true, traces include SELECT operations that
 *       implement conditional bypass for vertices with reward ≈ 0.
 */
struct ptd_elimination_trace *ptd_record_elimination_trace(
    struct ptd_graph *graph,
    bool enable_rewards
);
```

#### Update Implementation Docs
**File**: `REWARD_BYPASS_FIX.md`

Add section at end:
```markdown
## C Implementation (COMPLETE)

**Date**: 2025-10-27 (continued)

Successfully ported Python's conditional bypass logic to C code.

**Changes**:
1. Added `reward_length` field to `struct ptd_elimination_trace`
2. Added `PTD_OP_SELECT` operation type
3. Modified `ptd_record_elimination_trace()` to accept `enable_rewards` parameter
4. Implemented PHASE 1.5: Rate scaling with SELECT operations
5. Modified PHASE 3: Conditional bypass in elimination loop
6. Updated trace evaluation to handle SELECT operations
7. Updated caching to distinguish reward-enabled traces

**Test Results**:
```
True parameter:           θ = 10
Estimate after fix:       θ̂ = 10.02 (error: 0.02 = 0.2%) ✅
```

Error reduced from 16% to <1% - **SUCCESS!**
```

---

## Important Notes

### Python Reference Implementation
The Python code in `src/phasic/trace_elimination.py` serves as the reference:
- **SELECT operation**: Lines 50-60 (enum), 303-343 (builder), 903-916 (evaluation)
- **PHASE 1.5 (rate scaling)**: Lines 593-623
- **PHASE 3 (conditional bypass)**: Lines 728-784

### Error Handling
Use `goto cleanup_error` pattern consistently (already used in existing code).

### Testing Strategy
1. **After each phase**: Compile and run Python tests
2. **Before SVGD test**: Clear trace cache
3. **If SVGD fails**: Check cache to verify `reward_length` is set

### Rollback Plan
If something breaks:
1. **Phase 2-3**: Revert `api/c/phasic.h` changes
2. **Phase 4-5**: Comment out PHASE 1.5 and conditional logic
3. **Phase 6-7**: Revert to original calls with single parameter
4. **Nuclear option**: `git checkout api/c/phasic.h src/c/phasic.c`

---

## Exact Prompt to Resume

After conversation compaction, use this exact prompt:

```
Continue implementing Option A: C Code Fix for reward transformation.

Status: Phase 1 complete (reward_length field added to struct).

Please read /Users/kmt/phasic/C_IMPLEMENTATION_HANDOFF.md for complete context and detailed instructions.

Start with Phase 2: Adding SELECT operation to enum and helper function.

Work through phases sequentially, testing compilation after each major change.
```

---

## Key Success Metrics

1. ✅ Code compiles without errors
2. ✅ Python integration tests still pass
3. ✅ C trace includes `reward_length` field
4. ✅ SELECT operation evaluates correctly
5. ✅ **SVGD error < 1%** (currently 16%)

Target: θ̂ ∈ [9.9, 10.1] for true θ = 10

---

## Files to Track

**Modified**:
- `api/c/phasic.h` - Struct + enum + function signature
- `src/c/phasic.c` - Implementation (~200 lines added)

**For testing**:
- `tests/test_notebook_multivar_reproduction.py` - SVGD convergence
- `~/.phasic_cache/traces/*.json` - Trace cache (inspect reward_length)

**For reference**:
- `src/phasic/trace_elimination.py` - Python implementation (working)
- `REWARD_BYPASS_FIX.md` - Technical documentation
- `REWARD_BYPASS_SUMMARY.md` - High-level summary

---

**Date Created**: 2025-10-27
**Estimated Completion**: 3.5-5.5 hours remaining
**Current Phase**: 1/9 complete
