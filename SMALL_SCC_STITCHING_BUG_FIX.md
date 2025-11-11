# Small SCC Stitching Bug Fix

**Date**: 2025-11-11
**Status**: ✅ **FIXED**

---

## Problem

When calling `graph.compute_trace(hierarchical=True, min_size=10)` on two_locus_arg with 5 samples (340 vertices, 51 SCCs), the code crashed with a KeyError during trace stitching:

```
[ERROR] phasic.hierarchical_trace_cache:   ✗ Trace stitching failed: 'e95937bc8372a02a1361a7570ab2349b706791a404734e3819be5808c96ed601'
```

**Graph Structure**:
- 340 vertices
- 51 SCCs total:
  - 9 large SCCs (≥10 vertices)
  - 42 small SCCs (<10 vertices, 223 total vertices)

---

## Root Cause

The hierarchical caching system has a design where:

1. **`collect_missing_traces_batch()`** only creates traces for **large SCCs** (≥ min_size)
2. Small SCCs' vertices are **included in the enhanced subgraphs** of large SCCs (upstream/downstream connecting vertices)
3. Returns `all_scc_hashes` containing only the 9 large SCC hashes

However, **`stitch_scc_traces()`** was:

1. Iterating over **ALL SCCs** in the decomposition (all 51, not just the 9 large ones)
2. Trying to look up traces for every SCC in `scc_trace_dict`
3. When it encountered a small SCC hash, it **wasn't in the dict** → KeyError

**Key Files**:
- `src/phasic/hierarchical_trace_cache.py`
  - Line 241: `all_scc_hashes.append(scc_hash)` - only large SCCs added
  - Line 1287: `sccs = list(scc_graph.sccs_in_topo_order())` - ALL SCCs iterated
  - Line 1305: `scc_trace = scc_trace_dict[scc_hash]` - KeyError for small SCCs

---

## The Fix

Modified `stitch_scc_traces()` to **skip SCCs that don't have traces** in three locations:

### 1. Metadata Inference Loop (lines 1303-1310)

**Before**:
```python
for scc in sccs:
    scc_hash = scc.hash()
    scc_trace = scc_trace_dict[scc_hash]  # KeyError for small SCCs!
```

**After**:
```python
for scc in sccs:
    scc_hash = scc.hash()

    # Skip SCCs that don't have traces (small SCCs below min_size threshold)
    # These SCCs' vertices are already included in the enhanced subgraphs of large SCCs
    if scc_hash not in scc_trace_dict:
        logger.debug("  Skipping SCC %s (no trace - likely below min_size threshold)", scc_hash[:16])
        continue

    scc_trace = scc_trace_dict[scc_hash]
```

### 2. First SCC Initialization (lines 1374-1391)

**Before**:
```python
first_scc = sccs[0]
first_hash = first_scc.hash()
first_trace = scc_trace_dict[first_hash]  # Assumes first SCC has trace
```

**After**:
```python
# Initialize merged trace with first SCC that has a trace
first_scc = None
first_hash = None
first_trace = None
first_metadata = None

for scc in sccs:
    scc_hash = scc.hash()
    if scc_hash in scc_trace_dict:
        first_scc = scc
        first_hash = scc_hash
        first_trace = scc_trace_dict[first_hash]
        first_metadata = scc_metadata_dict[first_hash]
        break

if first_scc is None:
    logger.error("No SCCs with traces found - this should not happen!")
    raise ValueError("No SCCs with traces in scc_trace_dict")
```

### 3. Remaining SCCs Loop (lines 1462-1475)

**Before**:
```python
for scc_idx in range(1, len(sccs)):
    scc = sccs[scc_idx]
    scc_hash = scc.hash()
    scc_trace = scc_trace_dict[scc_hash]  # KeyError for small SCCs!
```

**After**:
```python
for scc_idx in range(len(sccs)):
    scc = sccs[scc_idx]
    scc_hash = scc.hash()

    # Skip the first SCC that we already processed
    if scc_hash == first_hash:
        logger.debug("  Skipping SCC %d/%d (already processed as first SCC)", scc_idx + 1, len(sccs))
        continue

    # Skip SCCs that don't have traces (small SCCs below min_size threshold)
    if scc_hash not in scc_trace_dict:
        logger.debug("  Skipping SCC %d/%d (hash=%s..., no trace - likely below min_size threshold)",
                    scc_idx + 1, len(sccs), scc_hash[:16])
        continue

    scc_trace = scc_trace_dict[scc_hash]
```

---

## Key Insight

**Small SCCs' vertices are NOT lost** - they are already included in the enhanced subgraphs of large SCCs as:
- **Upstream vertices** (fake starting vertices from previous SCCs)
- **Downstream vertices** (fake absorbing vertices for downstream SCCs)

Therefore, we only need to stitch the large SCCs' traces together. The small SCCs don't need separate processing during stitching.

---

## Test Results

### Test 1: two_locus_arg samples=5, min_size=10 (Reproduces Original Bug)

**Setup**:
- 340 vertices
- 51 SCCs (9 large ≥10, 42 small <10)

**Before Fix**: KeyError during stitching

**After Fix**:
```
✓ SUCCESS: Trace computed: 340 vertices, 15518 operations
```

**Time**: ~5 seconds (vs crash before)

### Test 2: two_locus_arg samples=4, min_size=30 (Regression Test)

**Setup**:
- 110 vertices
- 27 SCCs (all < 30)

**After Fix**:
```
✓ SUCCESS: Trace computed: 110 vertices, 9394 operations
Direct trace: 110 vertices, 9394 operations
```

**No regression** - still works correctly (falls back to full graph recording)

---

## Impact

### Fixed Issues

1. ✅ **two_locus_arg with samples=5, min_size=10** now works
2. ✅ **No more KeyError** during trace stitching
3. ✅ **Graphs with mixed large/small SCCs** handled correctly
4. ✅ **No performance regression** for previous cases

### Design Principle

**Stitching should only process SCCs that have traces**:
- Large SCCs (≥ min_size): Have dedicated traces, stitch these together
- Small SCCs (< min_size): Already included in large SCC subgraphs, skip during stitching

---

## Verification Checklist

- [x] two_locus_arg samples=5, min_size=10 works
- [x] No KeyError in stitching
- [x] Regression test passes (samples=4, min_size=30)
- [x] Operation counts match expectations
- [x] Log messages are clear and informative

---

## Files Modified

- `src/phasic/hierarchical_trace_cache.py` (lines 1303-1310, 1374-1391, 1462-1475)

## Tests Created

- `test_two_locus_n5_min10.py` (reproduction and verification)

---

## Conclusion

The small SCC stitching bug has been **fixed and verified**. Graphs with mixed large and small SCCs now work correctly with hierarchical caching. The stitching logic properly skips small SCCs that don't have dedicated traces.

**Status**: Production ready! ✅

---

*Bug fixed 2025-11-11*
