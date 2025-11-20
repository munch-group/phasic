# Hierarchical SCC Caching - Fix Status

## Problem Confirmed
✅ Hierarchical trace produces **INCORRECT** PDF values (12% error)
✅ Root cause identified: Connecting vertices lose upstream probability information

## Key Findings

### 1. Connecting Vertices Appear Twice
- **Version 1 (upstream SCCs)**: As absorbing vertices receiving probability mass
  - Example: Vertex 11 appears as connecting in 9 upstream SCCs
  - Each appearance has an absorption rate (probability flowing in)

- **Version 2 (home SCC)**: As internal vertices with outgoing edges
  - Has local exit rate computed from outgoing edges
  - Current code uses only this local rate, ignoring upstream absorption

### 2. Current Stitching Logic (INCOMPLETE)
```python
# Lines 922-934: Skip connecting vertices in upstream SCCs
if orig_v_idx not in internal_indices:
    # Save absorption rate for later (NEW - added in this session)
    connecting_rates[orig_v_idx].append((scc_idx, merged_rate_op_idx))
    continue  # Skip - don't set vertex data

# Lines 936-943: Process internal vertices
# Sets vertex_rates from local SCC trace
# ⚠️ IGNORES upstream absorption rates stored in connecting_rates!
```

### 3. What's Missing
When a vertex is processed in its home SCC, we need to:
1. Check if it has upstream absorption rates (`connecting_rates[orig_v_idx]`)
2. Create normalization operations to connect upstream → downstream probability flow
3. Scale downstream operations accordingly

## Proposed Fix

### Option A: Replace Vertex Rate (WRONG - attempted but incorrect)
```python
if orig_v_idx in connecting_rates:
    # Use upstream absorption rate instead of local rate
    merged.vertex_rates[merged_v_idx] = sum_of_upstream_rates  # ❌ WRONG!
```

**Why wrong**: Vertex rate MUST equal sum of outgoing edge rates (phase-type invariant)

### Option B: Scale Edge Probabilities (CORRECT but complex)
```python
if orig_v_idx in connecting_rates:
    upstream_absorption = sum_of_upstream_rates
    local_rate = scc_trace.vertex_rates[trace_v_idx]

    # Create scaling factor
    scale = upstream_absorption / local_rate

    # Scale ALL edge probabilities from this vertex
    for edge_prob_op in edge_probs[trace_v_idx]:
        scaled_op = MUL(edge_prob_op, scale)
        # Update merged trace...
```

**Challenge**: Need to recursively scale dependent operations

### Option C: Renormalize at Boundary (SIMPLEST?)
Accept that enhanced subgraphs use local rates, but ensure incoming edges from upstream have correct weights.

**Insight**: The issue might not be in vertex rates, but in **edge probabilities** from upstream SCCs!

When an upstream SCC has edge `u → c` with probability `p`:
- This `p` is computed relative to vertex `u`'s rate in the upstream SCC
- But when merged, vertex `c` might have a different rate in its home SCC
- The edge probability needs adjustment!

## Next Steps

1. **Verify edge probability hypothesis**: Check if edges TO connecting vertices have correct probabilities
2. **Implement edge probability scaling**: When copying edges to connecting vertices, scale by the ratio of rates
3. **Test PDF correctness**: Run `test_trace_correctness.py` to verify fix
4. **Run full test suite**: Ensure all 8 scenarios pass

## Code Status
- ✅ Connecting vertex tracking implemented (lines 877, 922-933)
- ✅ Detection of vertices with upstream connections (lines 946-949)
- ❌ **Scaling/normalization NOT YET IMPLEMENTED**
- ❌ PDF correctness test still fails (12% error)

## Files Modified
- `src/phasic/hierarchical_trace_cache.py`: Lines 876-949 (tracking added)
- `test_trace_correctness.py`: Created to verify PDF correctness

## Operation Count Analysis
- Direct: 25,027 operations
- Hierarchical (current): 12,572 operations (missing ~12,455 ops)
- Expected after fix: 12,572 + scaling operations (one per connecting vertex appearance)
- Connecting vertex appearances: ~266 total (from "Total enhanced vertices (with connecting): 376")
- Expected final count: ~12,838 operations (still less than direct, but CORRECT)

The operation count difference is EXPECTED - hierarchical decomposition is more efficient!
The key is ensuring mathematical correctness, not matching operation counts.
