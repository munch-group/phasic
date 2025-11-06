# Reward Bypass Fix: Complete Investigation Summary

**Date**: 2025-10-27
**Issue**: SVGD with multivariate rewards converges to θ̂ ≈ 11.6 instead of true θ = 10 (~16% error)

---

## ✅ What Was Successfully Completed

### 1. Python Trace System - FULLY WORKING

**Implemented** (all tests passing):
- SELECT operation for conditional logic (lines 50-60, 303-343 in trace_elimination.py)
- Conditional bypass in PHASE 3: vertices with reward ≈ 0 are bypassed (lines 728-784)
- Rate scaling in PHASE 1.5: rates divided by rewards when reward > 0 (lines 593-623)
- Removed incorrect reward multiplication from PHASE 2 (lines 620-629, 664-673)

**Test results**:
```python
# test_reward_bypass_integration.py - ALL PASSING ✓
Test 1: Basic reward bypass (reward=0 vs reward=1) ✓
Test 2: Rate scaling (reward=2 → rate halved) ✓
Test 3: JAX compatibility ✓
```

**Files modified**:
- `src/phasic/trace_elimination.py` - Core implementation
- `tests/test_trace_select_operation.py` - Unit tests (5/5 passing)
- `test_reward_bypass_integration.py` - Integration tests (3/3 passing)

### 2. Root Cause Analysis - COMPLETE

**Identified**: C code's `ptd_record_elimination_trace()` (line 7023 in phasic.c) lacks `enable_rewards` parameter.

**Evidence chain**:
```
SVGD
  → GraphBuilder (C++)
    → compute_moments_impl()
      → expected_waiting_time(rewards)
        → C elimination algorithm
          → ptd_record_elimination_trace(graph)  ❌ NO enable_rewards
            → Traces without conditional bypass
              → 16% systematic bias
```

**Proof**:
```bash
# Cached trace inspection
$ python3 -c "import json; print(json.load(open('trace.json')).get('reward_length'))"
NOT SET  # ← Should be n_vertices for reward-enabled traces
```

---

## ❌ What Doesn't Work Yet

### GraphBuilder / SVGD Still Has 16% Error

**Test results** (after all Python fixes):
```
True parameter:           θ = 10
Without regularization:   θ̂ = 11.6045 (error: 1.6045 = 16%)
With regularization:      θ̂ = 11.6321 (error: 1.6321 = 16%)
```

**Why**: C code records traces without `enable_rewards`, so conditional bypass logic is never used by SVGD.

---

## 🔧 The Fix Required: Modify C Code

### Option A: Add enable_rewards to C (Recommended)

**Estimated effort**: 4-6 hours
**Complexity**: Medium-High (C code modifications)

**Required changes**:

1. **Find and modify C trace struct** (location TBD - needs investigation):
   ```c
   struct ptd_elimination_trace {
       size_t n_vertices;
       size_t state_length;
       size_t param_length;
       size_t reward_length;  // ← ADD THIS FIELD
       bool is_discrete;
       // ... rest of fields
   };
   ```

2. **Update function signature** (src/c/phasic.c line 7023):
   ```c
   // OLD:
   struct ptd_elimination_trace *ptd_record_elimination_trace(
       struct ptd_graph *graph
   );

   // NEW:
   struct ptd_elimination_trace *ptd_record_elimination_trace(
       struct ptd_graph *graph,
       bool enable_rewards  // ← ADD THIS PARAMETER
   );
   ```

3. **Implement SELECT operation in C** (new code needed):
   ```c
   // Add to trace operation types
   #define PTD_TRACE_OP_SELECT 8  // or next available number

   // Record SELECT operations during elimination
   // Similar to Python's add_select() implementation
   ```

4. **Add conditional bypass logic** (modify elimination loop):
   ```c
   // In elimination algorithm, check reward_length > 0
   if (enable_rewards) {
       // Add SELECT for bypass probability
       // Add SELECT for edge removal
       // Match Python implementation in trace_elimination.py lines 728-784
   }
   ```

5. **Update trace caching** (src/c/phasic.c around line 7032):
   ```c
   // Include enable_rewards in cache hash
   // Separate cache entries for reward-enabled vs non-reward traces
   ```

6. **Update C++ GraphBuilder** (src/cpp/parameterized/graph_builder.cpp line 210):
   ```cpp
   // When computing moments with rewards, pass enable_rewards=true
   rewards2 = g.expected_waiting_time(rewards, enable_rewards=true);
   ```

---

## Alternative Approaches

### Option B: Full Python Implementation

**Estimated effort**: 6-8 hours
**Complexity**: Medium

**Strategy**: Compute moments entirely in Python, bypass C++ entirely when rewards used

**Pros**:
- No C code changes needed
- Proven correct (Python traces work)

**Cons**:
- Slower than C
- Need to implement absorption time calculation in Python

### Option C: Hybrid Routing

**Estimated effort**: 2-3 hours
**Complexity**: Low

**Strategy**:
```python
if rewards is not None and any(rewards < epsilon):
    # Use Python trace path (has zero-reward bypass)
    return python_trace_evaluation(...)
else:
    # Use GraphBuilder (C code OK for uniform scaling)
    return graphbuilder_evaluation(...)
```

**Pros**: Quick fix, best performance when possible
**Cons**: Doesn't fix root cause, complex routing logic

---

## Current Status

### Working:
✅ Python trace system (SELECT, conditional bypass, rate scaling)
✅ Integration tests proving correctness
✅ Root cause fully identified and documented

### Not Working:
❌ SVGD/GraphBuilder (uses C code without enable_rewards)
❌ 16% systematic bias persists in parameter estimation

### Next Steps:

**Immediate** (to complete Option A):
1. Locate C trace struct definition
2. Add `reward_length` field
3. Update `ptd_record_elimination_trace()` signature
4. Implement SELECT operation in C
5. Add conditional bypass to C elimination algorithm
6. Update caching and GraphBuilder integration
7. Test with SVGD

**Quick workaround** (Option C):
- Implement hybrid routing in `pmf_and_moments_from_graph()`
- Detect zero rewards → route to Python traces
- ~2-3 hours to implement and test

---

## Files Reference

### Modified (Python trace system - working):
- `src/phasic/trace_elimination.py`
- `tests/test_trace_select_operation.py`
- `test_reward_bypass_integration.py`
- `REWARD_BYPASS_FIX.md` (detailed documentation)

### Need modification (C code - Option A):
- `src/c/phasic.c` - Core C implementation
- `src/cpp/parameterized/graph_builder.cpp` - C++ wrapper
- Location TBD - C trace struct definition

### Test files:
- `tests/test_notebook_multivar_reproduction.py` - SVGD convergence test (currently failing)

---

## Key Insights

1. **Python implementation is correct** - Integration tests prove the algorithm works
2. **C code is the blocker** - No `enable_rewards` parameter → no conditional bypass
3. **Two code paths exist**:
   - ✅ Python: `record_elimination_trace(..., enable_rewards=True)` - WORKS
   - ❌ C: `ptd_record_elimination_trace(graph)` - MISSING FEATURE
4. **Fix is well-defined** - We know exactly what needs to change
5. **Trade-offs clear** - C fix (proper) vs Python workaround (quick)

---

**Recommendation**: Proceed with **Option A** (C code fix) for proper long-term solution, or **Option C** (hybrid routing) for quick short-term workaround.
