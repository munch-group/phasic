# Debug Folder: Vertex Bypass Bug Investigation

**Date**: 2025-10-28
**Bug**: Vertex bypass in `reward_transform()` fails for 3+ consecutive zero-reward vertices
**Status**: Confirmed but not fixed

---

## 🚀 Quick Start

**Run the main diagnostic**:
```bash
cd /Users/kmt/phasic
python debug/diagnose_vertex_bypass.py
```

This will show you the complete bug behavior with diagnostic output.

---

## 📁 File Organization

### 📘 Main Documentation (Read in Order)

1. **`VERTEX_BYPASS_DEBUGGING_PACKAGE.md`** ⭐ **START HERE**
   - Master document with overview
   - Quick fix recommendations
   - Complete roadmap

2. **`VERTEX_BYPASS_BUG_REPORT.md`**
   - Initial bug discovery
   - Test evidence showing the bug
   - Simple explanation of root cause

3. **`VERTEX_BYPASS_COMPLETE_DIAGNOSIS.md`**
   - Detailed technical analysis
   - Line-by-line code examination
   - Data structure breakdowns

4. **`VERTEX_BYPASS_FIX_ATTEMPTS_SUMMARY.md`**
   - All 4 fix attempts that I tried
   - Why each approach failed
   - Lessons learned

5. **`SPARSE_REWARDS_BUG_REPORT.md`**
   - How bug was discovered (SVGD convergence issues)
   - Coalescent model test results
   - Pattern analysis

### 🔧 C Debugging Guide

**`C_DEBUGGING_GUIDE.md`**
- Exact printf statements to add to C code
- Line numbers and locations
- What to look for in debug output
- Expected vs broken patterns

### 🧪 Test Scripts

**Main Diagnostic**:
- `diagnose_vertex_bypass.py` - Comprehensive diagnostic with detailed output

**Simple Tests**:
- `test_bypass_order_bug.py` - Demonstrates bug with different bypass counts
- `test_vertex_bypass_bug.py` - Tests multiple reward patterns
- `test_bypass_sampling.py` - Compares PDF vs sampling behavior
- `test_bypass_print_debug.py` - Graph structure inspection

**Validation Tests**:
- `test_reward_scaling_all_features.py` - Verifies reward scaling works correctly

---

## 🐛 The Bug Explained

### What's Broken

Chain graph `0→1→2→3→4` with rewards `[1, 0, 0, 0, 1]` (3 consecutive zero-reward vertices):
- **Result**: PDF integral = 0, samples = 0 (completely broken)
- **Why**: Stale parent pointers in `vertex_parents[]` array

### The Pattern

| Bypasses | Rewards | PDF Integral | Status |
|----------|---------|--------------|--------|
| 0 | `[1,1,1,1,1]` | 1.000 | ✅ Works |
| 1 | `[1,1,0,1,1]` | 0.999 | ✅ Works |
| 2 | `[1,1,0,0,1]` | 0.992 | ✅ Works |
| 3 | `[1,0,0,0,1]` | 0.000 | ❌ **BROKEN** |

### Root Cause

In `src/c/phasic.c`, function `_ptd_graph_reward_transform()`:

```c
// Lines 2135-2146: Build parent lists ONCE
for (size_t i = 0; i < vertices_length; ++i) {
    // Build vertex_parents[] array
}

// Lines 2160-2312: Process zero-reward vertices
for (size_t i = 0; i < vertices_length; ++i) {
    if (rewards[i] != 0) continue;

    // Read parent information - BUT IT'S STALE!
    for (size_t p = 0; p < vertex_parents_length[i]; ++p) {
        struct ptd_vertex *parent = vertex_parents[i][p].p;  // ← STALE!
        // After bypassing vertex j < i, this pointer may reference
        // a bypassed vertex whose edges have been modified
    }
}
```

**Problem**: `vertex_parents[]` is built once at initialization and never updated during bypass operations. After bypassing vertex 1, `vertex_parents[2]` still shows vertex 1 as parent (should be vertex 0).

---

## 💡 Recommended Fix

**Option A: Recursive Parent Resolution** (Cleanest)

Add to `src/c/phasic.c` around line 2160:

```c
struct ptd_vertex *resolve_parent(size_t vertex_idx, size_t parent_idx,
                                   bool *bypassed,
                                   struct arr_p **vertex_parents,
                                   size_t *vertex_parents_length) {
    struct ptd_vertex *p = vertex_parents[vertex_idx][parent_idx].p;
    size_t p_idx = p->index;

    // Follow chain of bypassed parents to find first non-bypassed ancestor
    while (bypassed[p_idx] && vertex_parents_length[p_idx] > 0) {
        p = vertex_parents[p_idx][0].p;
        p_idx = p->index;
    }

    return p;
}
```

Then replace line ~2176:
```c
// OLD:
struct ptd_vertex *parent_vertex = me_to_parent.p;

// NEW:
struct ptd_vertex *parent_vertex = resolve_parent(i, p, bypassed,
                                                    vertex_parents,
                                                    vertex_parents_length);
```

**Alternative fixes** described in `VERTEX_BYPASS_DEBUGGING_PACKAGE.md`:
- Option B: Complete parent list rebuild (most reliable but expensive)
- Option C: Topological order processing (most elegant but complex)

---

## 🔍 How to Debug

1. **Add C debugging** (see `C_DEBUGGING_GUIDE.md`):
   ```c
   printf("\n=== BYPASSING VERTEX %zu ===\n", i);
   printf("  Parent %zu: vertex %zu (bypassed=%s)\n", p, parent_idx,
          bypassed[parent_idx] ? "YES" : "NO");
   ```

2. **Rebuild**:
   ```bash
   pip install -e . --no-build-isolation
   ```

3. **Run diagnostic**:
   ```bash
   python debug/diagnose_vertex_bypass.py 2>&1 | tee debug_output.log
   ```

4. **Look for**:
   - Stale parent references
   - Skipped edge creation
   - Missing final edge (0→4)

---

## ⚡ Workaround (Temporary Solution)

Until fixed, use epsilon instead of zero:

```python
epsilon = 0.001
rewards[rewards == 0] = epsilon
```

This avoids the bug entirely (vertices aren't bypassed).

---

## ✅ Testing After Fix

Run all tests to verify:

```bash
# Main diagnostic (should show all ✓)
python debug/diagnose_vertex_bypass.py

# Specific test patterns
python debug/test_bypass_order_bug.py
python debug/test_bypass_sampling.py

# Reward scaling validation
python debug/test_reward_scaling_all_features.py
```

**Success criteria**:
- PDF integral > 0.9 for all bypass patterns
- Sample mean > 0 for all patterns
- Graph structure: 2 vertices with valid edge 0→4
- No regressions in existing tests

---

## 📊 Impact

**Affects**:
- Sparse reward vectors (>75% zeros)
- Coalescent models with many features
- Site frequency spectra
- Any phase-type model with long runs of zero rewards

**Severity**: CRITICAL
- Causes 20% bias in SVGD parameter estimates
- Silent failure (no error, just wrong results)
- Breaks PDF normalization

**Workaround available**: Yes (use epsilon instead of zero)

---

## 📝 Current Code State

The C code in `src/c/phasic.c` currently has my attempted fixes:
- Multi-pass bypass loop (`while (changed)`)
- `bypassed[]` array for tracking eliminated vertices
- Skip bypassed parents check

**These fixes don't work** because they create the "all-bypassed parent" problem (no edges created when all parents are bypassed).

**You can**:
- Keep the multi-pass structure (helps for other edge cases)
- Add recursive parent resolution on top
- Or revert to original and implement Option B/C

---

## 🤝 Contributing

When you fix this:
1. Test with all scripts in this folder
2. Add regression test for 3+ bypasses
3. Update documentation with the fix approach used
4. Consider adding a comment in C code referencing this debug folder

---

## 📧 Questions?

This debug package was created by Claude (AI assistant) during investigation of SVGD convergence issues.

For detailed explanations, see the documentation files above. The bug is well-understood - it just needs the right implementation.

**Good luck with the fix!**
