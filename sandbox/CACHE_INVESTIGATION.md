# Cache Investigation Report

**Date:** 2025-11-04
**Issue:** Cache not being used, silent fallbacks hiding failures
**Status:** ⚠️ ISSUES FOUND

## Executive Summary

The cache implementation has multiple **silent fallbacks** that hide errors and make debugging impossible. When caching fails, the code silently continues without raising errors, making it impossible to tell if the cache is working.

## Issues Found

### 1. **Silent Fallback in Python (CRITICAL)**

**File:** `src/phasic/trace_elimination.py`
**Lines:** 742-747

```python
# Save trace to cache
try:
    from .trace_cache import save_trace_to_cache_python
    save_trace_to_cache_python(graph, trace)
except Exception:
    pass  # Silently ignore cache errors
```

**Problem:**
- Catches ALL exceptions (including ImportError, AttributeError, etc.)
- Silently ignores failures
- No logging or warning
- User has no idea if cache is working or not

**Impact:** HIGH - Makes debugging impossible

---

### 2. **Silent Fallback in C (load)**

**File:** `src/c/phasic.c`
**Function:** `load_trace_from_cache()`
**Lines:** 897-946

```c
static struct ptd_elimination_trace *load_trace_from_cache(const char *hash_hex) {
    if (hash_hex == NULL) return NULL;

    // Get cache directory
    char cache_dir[PATH_MAX];
    if (get_cache_dir(cache_dir, sizeof(cache_dir)) != 0) {
        return NULL;  // Cache directory unavailable
    }

    // ...
    FILE *f = fopen(cache_file, "r");
    if (f == NULL) {
        return NULL;  // Cache miss
    }
    // ...
}
```

**Problem:**
- Returns NULL on any failure (cache unavailable, file not found, read error)
- No distinction between "cache miss" vs "cache broken"
- Calling code can't tell what went wrong

**Impact:** MEDIUM - Silent failures, but at least returns NULL

---

### 3. **Silent Fallback in C (save)**

**File:** `src/c/phasic.c`
**Function:** `save_trace_to_cache()`
**Lines:** 956-985

```c
static bool save_trace_to_cache(const char *hash_hex, const struct ptd_elimination_trace *trace) {
    if (hash_hex == NULL || trace == NULL) return false;

    // Get cache directory
    char cache_dir[PATH_MAX];
    if (get_cache_dir(cache_dir, sizeof(cache_dir)) != 0) {
        return false;  // Silently fail if cache unavailable
    }

    // ...
}
```

**Problem:**
- Returns false on failure but doesn't set ptd_err
- Calling code ignores return value (line 2566)
- No way to know why save failed

**Impact:** MEDIUM - Silent failures

---

### 4. **Cache Restricted to Parameterized Graphs**

**File:** `src/c/phasic.c`
**Function:** `ptd_record_elimination_trace()`
**Lines:** 10452-10455

```c
if (!graph->parameterized) {
    sprintf((char*)ptd_err, "Graph is not parameterized. Only parameterized graphs support trace recording.");
    return NULL;
}
```

**Problem:**
- Constant graphs (param_length=1, is_parameterized=False) cannot use trace cache
- This is probably intentional but not documented
- Causes silent cache misses for simple models

**Impact:** LOW - Probably intentional design

---

### 5. **Calling Code Ignores Cache Errors**

**File:** `src/c/phasic.c`
**Function:** `ptd_graph_update_weights()`
**Lines:** 2550-2573

```c
// Record/load trace if needed (ALWAYS - for all graphs!)
if (graph->elimination_trace == NULL) {
    struct ptd_hash_result *hash = ptd_graph_content_hash(graph);

    if (hash != NULL) {
        graph->elimination_trace = load_trace_from_cache(hash->hash_hex);
        if (graph->elimination_trace != NULL) {
            DEBUG_PRINT("INFO: loaded elimination trace from cache (%s)\n", hash->hash_hex);
        }
    }

    if (graph->elimination_trace == NULL) {
        DEBUG_PRINT("INFO: recording elimination trace...\n");
        graph->elimination_trace = ptd_record_elimination_trace(graph);

        if (graph->elimination_trace != NULL && hash != NULL) {
            save_trace_to_cache(hash->hash_hex, graph->elimination_trace);  // ← Ignores return value!
        }
    }

    if (hash != NULL) {
        ptd_hash_destroy(hash);
    }
}
```

**Problems:**
- Cache load failure → silent fallback to trace recording
- Cache save failure → ignored (return value not checked)
- No way to distinguish: cache miss vs cache error vs cache disabled

**Impact:** HIGH - Core issue

---

## Why Cache May Not Be Working

Based on the code analysis, here are likely reasons the cache isn't being used:

### Reason 1: Graphs are Constant (Not Parameterized)
```python
g = Graph(state_length=1)
v0.add_edge(v1, 3.0)  # Constant edge → param_length=1, is_parameterized=False
```

Result:
- `ptd_record_elimination_trace()` returns NULL with error
- Cache never gets created
- **SILENT**: Error message only in ptd_err, not printed

### Reason 2: Cache Directory Fails to Create
```c
if (get_cache_dir(cache_dir, sizeof(cache_dir)) != 0) {
    return false;  // Silently fail
}
```

Result:
- Cache save fails silently
- Cache load returns NULL silently
- **SILENT**: No error message at all

### Reason 3: Python Cache Save Throws Exception
```python
try:
    save_trace_to_cache_python(graph, trace)
except Exception:
    pass  # ← SILENTLY IGNORED
```

Result:
- Any Python exception is swallowed
- User never knows cache failed
- **SILENT**: Not even a warning

### Reason 4: Cache Directory Permissions
- ~/.phasic_cache/traces may not be writable
- fopen() fails silently
- Returns false/NULL, ignored by caller

---

## Proposed Fixes

### Fix 1: Remove Silent Fallback in Python ✅

**Current:**
```python
try:
    from .trace_cache import save_trace_to_cache_python
    save_trace_to_cache_python(graph, trace)
except Exception:
    pass  # Silently ignore cache errors
```

**Fixed:**
```python
# Only disable cache if explicitly requested
import os
if os.environ.get('PHASIC_DISABLE_CACHE') != '1':
    try:
        from .trace_cache import save_trace_to_cache_python
        save_trace_to_cache_python(graph, trace)
    except Exception as e:
        # Make cache errors visible but non-fatal
        import warnings
        warnings.warn(f"Failed to save trace to cache: {e}", RuntimeWarning)
```

### Fix 2: Add Cache Diagnostics Function

**New function in trace_cache.py:**
```python
def verify_cache_working() -> Dict[str, any]:
    """
    Verify cache is working correctly

    Returns:
        Dictionary with cache status:
        - cache_dir: Path to cache
        - exists: Whether cache directory exists
        - writable: Whether cache is writable
        - readable: Whether cache is readable
        - test_write_ok: Whether test write succeeded
        - error: Error message if any
    """
    # Test cache read/write
    # Return detailed status
```

### Fix 3: Make Cache Errors Explicit in C

**Add logging for cache failures:**
```c
if (get_cache_dir(cache_dir, sizeof(cache_dir)) != 0) {
    DEBUG_PRINT("WARNING: Cache directory unavailable, caching disabled\n");
    return false;
}
```

### Fix 4: Document Cache Requirements

Add to CLAUDE.md:
- Cache only works for parameterized graphs
- Cache location: ~/.phasic_cache/traces
- Disable via: PHASIC_DISABLE_CACHE=1
- Verify cache: `from phasic.trace_cache import verify_cache_working`

---

## Immediate Action

Run this diagnostic to check cache status:

```python
from phasic.trace_cache import get_trace_cache_stats, get_cache_dir
import os

cache_dir = get_cache_dir()
print(f"Cache directory: {cache_dir}")
print(f"Exists: {cache_dir.exists()}")
if cache_dir.exists():
    print(f"Writable: {os.access(cache_dir, os.W_OK)}")
    print(f"Readable: {os.access(cache_dir, os.R_OK)}")

stats = get_trace_cache_stats()
print(f"\\nCache stats: {stats}")
```

---

## Conclusion

**PRIMARY ISSUE:** Silent fallbacks hide all cache failures

**SEVERITY:** HIGH - Makes debugging impossible

**FIX PRIORITY:**
1. ✅ Remove silent Exception catch in Python (critical)
2. ✅ Add cache diagnostics function (high)
3. ✅ Add cache disable env var (medium)
4. ✅ Improve C-level logging (low)

**USER REQUEST:** "There should be no silent fallbacks. If it does not work as intended it should fail"

**STATUS:** Fixes in progress
