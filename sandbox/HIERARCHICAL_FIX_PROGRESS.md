# Hierarchical SCC Caching Fix - Current Progress

## Problem
Hierarchical trace produces incorrect PDF values (was 12% error, now varies 10-600% depending on approach).

## Approaches Tried

### Attempt 1: Replace Vertex Rate with Upstream Rate (FAILED)
**Approach**: Save `vertex_rates[connecting_v]` from upstream SCCs, use to replace local rate.

**Result**: PDFs = 0.0 (total failure)

**Root Cause**: `vertex_rates[connecting_v]` in upstream SCC is from elimination trace, often = 0 for absorbing vertices. This is NOT the flow to the vertex!

---

### Attempt 2: Compute Upstream Flow, Replace Rate + Scale Probs (PARTIAL)
**Approach**:
1. Compute upstream flow = sum(edge_prob * source_rate) for all edges TO connecting vertex
2. Replace vertex rate with upstream_flow
3. Scale edge probabilities by (upstream_flow / local_rate)

**Result**: 10% error

**Issue**: Scaling both rate AND probabilities causes edge_rate = edge_prob * rate to be scaled TWICE:
- new_edge_rate = (old_prob * scale) * upstream_flow = old_prob * (upstream_flow/local_rate) * upstream_flow = old_prob * upstream_flow²/local_rate
- Should be: new_edge_rate = old_prob * upstream_flow

---

### Attempt 3: Compute Upstream Flow, Replace Rate Only (WORSE!)
**Approach**:
1. Compute upstream flow = sum(edge_prob * source_rate)
2. Replace vertex rate with upstream_flow
3. Keep edge probabilities unchanged

**Result**: 612% error at t=5.0

**Issue**: Unknown - this SHOULD be correct mathematically!

---

## Current Implementation

### Flow Computation (lines 921-973)
```python
# For each connecting vertex in upstream SCC:
incoming_flow_ops = []
for src_vertex in internal_vertices:
    for edge_to_connecting_vertex:
        # flow = edge_prob * source_rate
        flow_op = MUL(edge_prob_op, source_rate_op)
        incoming_flow_ops.append(flow_op)

# Total flow = sum of all incoming flows
if len(incoming_flow_ops) == 1:
    total_flow = incoming_flow_ops[0]
else:
    total_flow = SUM(incoming_flow_ops)

connecting_rates[orig_v_idx].append(total_flow_op)
```

###  Rate Replacement (lines 989-1013)
```python
if orig_v_idx in connecting_rates:
    # Sum flows from all upstream SCCs
    if len(upstream_flows) == 1:
        upstream_total = upstream_flows[0]
    else:
        upstream_total = SUM(upstream_flows)

    # Replace vertex rate
    merged.vertex_rates[merged_v_idx] = upstream_total
```

## Hypothesis for Attempt 3 Failure

Possible issues:
1. **Flow computation is wrong**: Maybe `edge_prob * source_rate` isn't the right formula?
2. **Missing normalization**: Maybe edge probs in local SCC don't sum to 1?
3. **Absorbing vertex handling**: Maybe connecting vertices in enhanced subgraphs don't work as expected?
4. **Initial probability**: Maybe initial probability vector is affected?

## Next Steps

1. **Verify flow computation formula**:
   - In elimination traces, is edge flow really = `edge_prob * vertex_rate`?
   - Or is it stored differently?

2. **Check edge probability sums**:
   - Do edge probabilities from a vertex sum to 1.0?
   - Or do they need renormalization?

3. **Test with simple case**:
   - Create minimal 2-SCC example
   - Manually verify flow computation
   - Check if approach works for simple case

4. **Review phase-type theory**:
   - Double-check mathematical foundation
   - Ensure understanding of how elimination affects rates/probs

## Files Modified
- `src/phasic/hierarchical_trace_cache.py`: Lines 921-1050 (flow computation and stitching)

## Test Results
- Direct trace: Always correct
- Hierarchical trace:
  - Attempt 1: 0% (total failure)
  - Attempt 2: ~10% error
  - Attempt 3: ~600% error

## Operation Counts
- Direct: 25,027 operations
- Hierarchical (current): 14,052 operations (includes flow computation)
