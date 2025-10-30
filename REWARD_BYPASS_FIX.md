# Reward Bypass Implementation for Trace-Based Elimination

## Problem

SVGD inference with multivariate rewards was converging to θ̂ ≈ 11.6-11.8 instead of true θ = 10 (~16-18% error).

Investigation revealed that the trace-based elimination system was incorrectly handling rewards by multiplying edge probabilities instead of implementing proper vertex bypass.

## Root Cause

In `src/phasic/trace_elimination.py`, PHASE 2 was multiplying edge probabilities by rewards:

```python
# BUG: This made vertices with reward=0 absorbing states
prob_idx = builder.add_mul(prob_idx, reward_idx)
```

**Correct behavior** (from C++ implementation):
- Rewards multiply WAITING TIMES, not transition probabilities
- Vertices with reward ≈ 0 should be BYPASSED (edges redirected through them to children)
- Vertices with reward > 0 should have rates scaled by reward

## Solution Implemented

### 1. Added SELECT Operation (✅ COMPLETE)

Added conditional operation to trace system for dynamic behavior based on parameter values.

**Files modified**:
- `src/phasic/trace_elimination.py`:
  - Line 50-60: Added `OpType.SELECT` enum
  - Line 303-343: Added `TraceBuilder.add_select()` method
  - Line 903-916: Added SELECT case in `evaluate_trace()`
  - Line 1369-1388: Added SELECT case in `evaluate_trace_jax()`
- `tests/test_trace_select_operation.py`: Comprehensive unit tests (all passing ✅)

### 2. Removed Incorrect Reward Multiplication (✅ COMPLETE)

**Files modified**:
- `src/phasic/trace_elimination.py`:
  - Lines 620-629: Removed buggy reward multiplication for regular edges
  - Lines 664-673: Removed buggy reward multiplication for parameterized edges

### 3. Added Conditional Bypass Logic to PHASE 3 (✅ COMPLETE)

**Files modified**:
- `src/phasic/trace_elimination.py`:
  - Lines 728-747: Conditional bypass edge creation
    - If reward ≈ 0: create bypass edges (parent → child via eliminated vertex)
    - If reward > 0: no bypass (zero bypass probability)
  - Lines 766-784: Conditional edge removal
    - If reward ≈ 0: remove edge to vertex (vertex bypassed)
    - If reward > 0: keep edge to vertex (vertex not bypassed)

### 4. Added Rate Scaling for Non-Zero Rewards (✅ COMPLETE)

**Files modified**:
- `src/phasic/trace_elimination.py`:
  - Lines 593-623: PHASE 1.5 - Apply reward scaling to rates
    - For reward > 0: new_rate = original_rate / reward
    - For reward ≈ 0: keep original rate (will be bypassed anyway)
    - Uses epsilon guard to prevent division by zero

### 5. Integration Tests (✅ COMPLETE)

**File created**: `test_reward_bypass_integration.py`

Tests verify:
1. ✅ Vertices with reward=0 are bypassed correctly
2. ✅ Rate scaling works correctly (reward=2 → rate halved)
3. ✅ JAX compatibility maintained
4. ✅ Probability distributions sum to 1.0

All integration tests passing!

## Current Status

### ✅ What Works
- SELECT operation implementation complete and tested
- Trace-based evaluation correctly handles rewards with conditional bypass
- Integration tests confirm correct behavior
- Error improved from ~18% to ~12% when trace cache was cleared

### ⚠️ Outstanding Issue - ROOT CAUSE IDENTIFIED

**Problem**: SVGD still shows ~12% error because traces are being recorded WITHOUT `enable_rewards=True`.

**Root cause identified**:
1. C++ GraphBuilder DOES use Python's trace system (confirmed by cache hit messages)
2. But it records traces WITHOUT the `enable_rewards=True` flag
3. Cached trace inspection shows: `reward_length: NOT SET`
4. Therefore, our conditional bypass logic in PHASE 1.5 and PHASE 3 is never executed!

**Evidence**:
```bash
$ python3 -c "import json; data = json.load(open('~/.phasic_cache/traces/aa79...baf.json')); \
  print('reward_length:', data.get('reward_length', 'NOT SET'))"
reward_length: NOT SET
```

**Fix required**:
- Option A: Modify C++ GraphBuilder binding to pass `enable_rewards=True` when recording traces
- Option B: Create Python wrapper that uses `record_elimination_trace(graph, param_length=..., enable_rewards=True)` directly
- Option C: Modify trace recording to auto-detect if rewards will be used

## Test Results

### Before Fixes
- θ̂ ≈ 11.8, error = 1.8 (~18%)

### After Fixes (with fresh trace cache)
- θ̂ ≈ 11.2, error = 1.2 (~12%)
- **33% improvement** but still not converging correctly

### Target
- Error < 1% (θ̂ within [9.9, 10.1])

## Files Modified

1. **src/phasic/trace_elimination.py**: Core trace recording and evaluation
   - Added SELECT operation (OpType, builder method, evaluation)
   - Removed incorrect reward multiplication
   - Added conditional bypass logic in PHASE 3
   - Added rate scaling in PHASE 1.5

2. **tests/test_trace_select_operation.py**: Unit tests for SELECT operation (NEW)

3. **test_reward_bypass_integration.py**: Integration tests (NEW)

## Key Insights

1. **Rewards affect waiting times, not transitions**: `new_rate = rate / reward`
2. **Zero-reward bypass**: Vertices with reward ≈ 0 must be bypassed via elimination
3. **SELECT enables conditionals**: Dynamic behavior based on parameter values at evaluation time
4. **Trace caching**: Must clear `~/.phasic_cache/traces/` when implementation changes
5. **C++ vs Python evaluation**: Current SVGD uses C++ GraphBuilder, not Python traces

## Attempted Fix: Python Trace Integration (INCOMPLETE)

**Date**: 2025-10-27 (continued)

### What Was Attempted

Modified `pmf_and_moments_from_graph()` to route reward-based computations through Python's trace system with `enable_rewards=True`.

**Files modified**:
- `src/phasic/__init__.py`:
  - Added helper functions `_compute_pmf_from_concrete_graph()` and `_compute_moments_from_trace_with_rewards()` (lines 2898-2967)
  - Added `_compute_pure_with_trace()` function (lines 3109-3129)
  - Modified callback routing (line 3189-3191)

### Why It Didn't Work

**Root cause**: C code's `ptd_record_elimination_trace()` doesn't have `enable_rewards` parameter.

**The problem chain**:
1. GraphBuilder calls C++'s `compute_moments_impl()`
2. That calls `g.expected_waiting_time(rewards)`
3. That triggers C's elimination algorithm which uses cached traces
4. Traces are recorded via `ptd_record_elimination_trace(graph)` - NO enable_rewards parameter
5. Without enable_rewards, traces don't have conditional bypass logic
6. Result: 16% error persists

**Evidence**:
```c
// src/c/phasic.c line 7023
struct ptd_elimination_trace *ptd_record_elimination_trace(
    struct ptd_graph *graph  // ← No enable_rewards parameter!
)
```

### Test Results After Attempted Fix

```
True parameter:           θ = 10
Without regularization:   θ̂ = 11.6045 (error: 1.6045 = 16%)
With regularization:      θ̂ = 11.6321 (error: 1.6321 = 16%)
```

No improvement - still 16% error (same as before).

### What Works vs What Doesn't

✅ **Python trace system** (`record_elimination_trace(..., enable_rewards=True)`):
- Integration tests show correct behavior
- Rate scaling works (reward=2 → rate halved)
- Conditional bypass works (reward=0 → vertex bypassed)

❌ **C/C++ GraphBuilder** (used by SVGD):
- Uses `ptd_record_elimination_trace()` without enable_rewards
- Traces lack conditional bypass logic
- Results in ~16% systematic bias

### The Real Fix Required

**Option A: Modify C code** (proper but complex):
1. Add `enable_rewards` parameter to `ptd_record_elimination_trace()`
2. Add `reward_length` field to `struct ptd_elimination_trace`
3. Implement SELECT operation in C (conditional bypass logic)
4. Update trace struct serialization/deserialization
5. Update caching system to distinguish reward-enabled traces

**Option B: Full Python moments computation** (alternative):
1. Implement `compute_moments_via_elimination()` in Python
2. Use trace evaluation to compute absorption times
3. Avoid C++ code entirely when rewards are used
4. Slower but correct

**Option C: Hybrid approach** (pragmatic):
1. Detect when rewards have zeros
2. If all rewards > epsilon: use GraphBuilder (C code OK for uniform scaling)
3. If any reward ≈ 0: fall back to Python trace evaluation
4. Best of both worlds but more complex routing logic

### Status

**Current state**: Partial implementation completed but not functional
**Remaining work**: Requires C code changes (Option A) or full Python implementation (Option B)
**Estimated effort**:
- Option A: 4-6 hours (C code + testing)
- Option B: 6-8 hours (Python moments + testing)
- Option C: 2-3 hours (hybrid routing + testing)

## Date

2025-10-27
