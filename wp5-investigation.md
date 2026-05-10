# WP-5 follow-up investigation: Python pipeline + Option B viability

## Python pipeline reality check

I investigated `src/phasic/hierarchical_trace_cache.py` (the
existing Python pipeline that the working agreements cited as
the reference) and ran it on the toy graphs.

**Finding: the Python pipeline does NOT handle SCCs with internal
cycles.** It bails with:

```
RuntimeError: Trace-based elimination cannot handle the cycle
(parent=5 → i=4 → parent=5): self-loop correction 1/(1 − q) is
not implemented.
```

This error fires on toy-base (which has 2-cycles inside SCC₁ and
SCC₂). Toy-A (fully acyclic, all singleton SCCs) works fine.

So the Python pipeline's "cross-graph reuse" claim only applies
to graphs whose SCCs are all singletons (i.e. the parent graph
itself is acyclic). For cyclic graphs — the case this whole
branch was motivated by — there is no working Python reference.

## What the Python pipeline does that we can still learn from

Reading `_build_scc_subgraph` (`hierarchical_trace_cache.py:1047+`)
and `stitch_scc_traces` (`:1619+`):

- Each SCC's "enhanced subgraph" includes upstream-vertices and
  downstream-vertices as **real vertices with their actual parent
  state vectors**. Downstream vertices have NO outgoing edges in
  the subgraph (they're absorbing leaves).
- Stitching identifies **sister vertices** by state across SCC
  pairs: a downstream-leaf in SCC `i`'s subgraph and an
  upstream-vertex in SCC `j`'s subgraph (where `i → j` in the
  condensation) are sisters when they share a state vector.
- Stitching builds a merged trace by copying operations and
  remapping vertex indices, with sister-vertex collapsing.
- The merged trace is **re-evaluated** at use time — not reused
  as cached numerical output.

So the Python pipeline caches the **structural elimination
trace**, then re-evaluates it for each θ. The trace acts like a
compiled computation graph; evaluation is fast (O(trace
length)) but it's not "skipping the elimination" in the C-path
sense.

This is closer to my **Option A** from `wp5-experiments.md`
than to Option B — the cache stores the symbolic structure, and
elimination math (the actual cycle/self-loop math) happens at
build time, not at use time.

For our C path with cyclic SCCs, the equivalent would be:
**cache the per-SCC PRC's command stream + an extension that
captures inter-SCC dataflow, then re-execute at compose time.**

## Option B details

The plan is workable if the synthetic graph has **one synthetic
absorbing vertex per downstream-connecting vertex**, so each
external out-channel has its own slot to receive a downstream
value at compose time.

### Math

For an SCC with downstream-connecting vertex `d_j` and external
edges `(d_j → w_1)`, `(d_j → w_2)`, …, `(d_j → w_k)` in the
parent:

- In the synthetic graph, replace each parent edge `(d_j → w_i)`
  with `(d_j → s_abs_for_d_j_to_w_i)`, with the placeholder
  weight set to the parent's external edge weight `w(d_j, w_i)`.
- Each `s_abs_for_d_j_to_w_i` is a distinct absorbing vertex.
- The synthetic graph's elimination produces commands of the
  form `result[d_j] += (w(d_j, w_i) / rate(d_j)) * result[s_abs_for_d_j_to_w_i]`.
- At compose time, set `result[s_abs_for_d_j_to_w_i] =
  result[w_i_in_parent]` (which was computed by the downstream
  SCC, processed earlier in reverse-topo order).

### Cross-parent invariance

Two parents with the same SCC and same external-edge fan-out
shape (same `(d_j, w_i)` pairs at the structural level) produce
the same synthetic graph and same cache hash. Different external
edge weights → different synthetic edge weights → different
elimination commands' literal multipliers, but the **structure**
(command sequence, MEM offsets, EDGE references) is identical.

Wait — this last claim needs verification. Edge weights are
referenced by pointer (`&edge->weight`) in the PRC, so changing
weights at replay time is fine. But **the literal `multiplier`
in `add_command_param_p` is `1` (per `phasic.c:7237`), and the
edge weight enters as `*toT`** — which is read at replay time.
So edge weights flow through at replay; the symbolic structure
is parent-invariant.

✓ Cross-parent reuse works in this design IF the synthetic
graph topology (vertex count, edge structure modulo weights)
is parent-invariant.

### Cross-parent invariance with multi-edge fan-out

If parent P has SCC with `d_j → w_1` (one external edge) and
parent P' has SCC with `d_j → w_1, w_2` (two), the synthetic
graphs have different absorbing-vertex counts — different
topologies, different hashes, no reuse.

Two responses:
1. **Accept this as a limitation.** Cross-parent reuse only
   works when external fan-out shape matches. Many phase-type
   models (e.g. coalescents) have this property naturally.
2. **Canonicalize the fan-out.** Always create exactly one
   absorbing vertex per `(d_j, downstream-target-position)`
   pair, where target-position is "which slot in d_j's external
   out-edges." This keeps topology stable when fan-out shapes
   match, but doesn't help for mismatched cases.

I'll go with response 1 — accept the limitation for this
branch, document it, and revisit later if needed.

### Cycle handling

The C eliminator already handles within-SCC cycles correctly
via Gaussian elimination's self-loop normalization. Synthetic
graphs (with internal cycles + Type A/C placeholder edges) are
just regular graphs to the C eliminator — it doesn't know or
care that they came from an SCC decomposition.

WP-1 already verified this: the synthetic graphs for every toy
variant pass through the C eliminator and produce finite
results. Cyclic SCCs are not special.

So Option B works on cyclic SCCs out of the box, without the
self-loop limitation that broke the Python pipeline.

## Plan revision

### Change to WP-1's synthetic graph constructor

Currently WP-1 produces ONE synthetic absorbing vertex per
SCC, with all downstream-connecting vertices' Type C edges
pointing at it.

For Option B, I need **one synthetic absorbing vertex per
external-out-channel** (i.e. one per `(downstream_connecting,
parent_external_target)` pair). Each gets its own placeholder
edge with the parent's external edge weight as its (live)
weight.

This is a structural change to WP-1. It affects:
- `ptd_scc_build_synthetic_graph`: constructs more absorbing
  vertices.
- `ptd_scc_synthetic_metadata`: needs to record which absorbing
  vertex corresponds to which external channel.
- The synthetic graph's hash: more vertices means different
  hash, so existing cache files become stale. Acceptable —
  bump format revision or just clear the cache directory.
- Tests: WP-1 + WP-2 + WP-3 + WP-4 tests all need updating.

### Change to WP-5 composer

The composer:
1. For each SCC in reverse-topo order:
   1a. Get/compute the per-SCC PRC.
   1b. Build the per-channel `result_for_channel` value:
       `external_table[channel] = result_in_parent[downstream_target]`
   1c. Translate per-SCC PRC commands and append to parent PRC,
       resolving EXTERNAL pointers against `external_table`.
2. Synthesize cross-SCC `NEW_ADD` back-substitution commands
   for parent edges that cross SCC boundaries — these mirror
   what the monolithic eliminator's back-substitution would
   have emitted. The exact set is determined by walking the
   parent's external edges.

Actually — re-thinking this. If the per-SCC PRC's commands
correctly compute `result[d_j]` using the per-channel
`result[s_abs_for_channel]` values injected via EXTERNAL,
then the per-SCC PRC's back-substitution `NEW_ADD` commands
already include the cross-SCC propagation via the
"absorbing vertex" stand-ins. No separate boundary commands
needed.

Let me verify this on toy-base manually before writing the
composer.

## Next step

Test the math hypothesis empirically:
1. Manually construct the "Option B" synthetic graph for SCC₂
   (one absorbing per channel; weights = parent's external
   weights).
2. Run elimination on it.
3. Verify that with `result[s_abs_for_each_channel] =
   result[downstream_in_parent]`, the per-SCC `result[A]` and
   `result[B]` match monolithic's 3.64 and 3.8.

If the math checks out, proceed to implement WP-5 with the
revised WP-1 synthetic-graph structure. If it doesn't,
reconsider.

## Empirical confirmation (2026-05-10)

Constructed the Option B synthetic graph for SCC₂ and ran the
elimination. Two configurations:

1. **Standard Option B** (Type C weights = parent values; no
   phantom; absorbing vertices have result=0):
   - `result[A] = 2.64` ❌ (monolithic wants 3.64)
   - Confirms that just setting Type C weights right isn't
     enough — we need to inject downstream values.

2. **Option B with phantom absorbers** (each `s_abs_for_(d,w)`
   has an outgoing edge to a single phantom-absorbing vertex
   with weight `1/result[w_in_parent]`, so `result[s_abs] =
   1/rate = result[w_in_parent]`):
   - `result[A] = 3.64` ✓ (matches monolithic)
   - `result[B] = 3.8`  ✓ (matches monolithic)

**Option B works mathematically.** The trick: in a phase-type
graph, `result[v] = 1/rate(v) + Σ (prob_v_to_child) ·
result[child]`. For an absorbing-with-one-child vertex, this
reduces to `result[v] = 1/rate(v)`. Setting `rate(v) =
1/desired_result` makes `result[v] = desired_result`. So the
phantom-absorber chain injects the downstream value via the
synthetic graph's normal elimination mechanics.

## Revised WP-5 design

### Modified synthetic graph (replaces WP-1's structure)

Per-SCC synthetic graph contains, in order:

1. Synthetic source vertex (index 0).
2. Internal vertices, in canonical 5-part order
   (upstream-connecting, internal-only,
   downstream-connecting).
3. **One synthetic absorbing vertex per external out-channel**
   `(d_j, w)`, NOT one per SCC. Each has:
   - Type C edge from `d_j` with parent's external edge weight
     as the (live) edge weight.
   - One outgoing edge to a single phantom-absorbing vertex,
     with weight `1/downstream_result[w]` (set at compose time).
4. **One phantom-absorbing vertex** at the very end (single
   true absorbing vertex; result = 0 by structural fact).

### Cross-parent invariance

Hash key: SCC's content (internal structure + count and
identity of external out-channels per `d_j`). Two parents with
the same SCC structure and the same external-fan-out pattern
share the cache.

The cache breaks if external fan-out shape differs. For
phase-type models where each downstream-connecting vertex has
one external out-edge per (d_j, fixed-target) pair (typical for
state-vector-based models), this works naturally.

### Compose algorithm

Walk SCCs in reverse-topological order:

1. For SCC `s`:
   1a. Get/compute the synthetic graph + PRC.
   1b. Look up the parent's external out-edges from each `d_j`.
   1c. For each `(d_j, w)` channel:
       - Set the synthetic Type C edge weight `d_j →
         s_abs_for_(d_j,w)` to the parent's edge weight (live
         value at current θ).
       - Set the synthetic phantom edge weight `s_abs_for_(d_j,w)
         → phantom` to `1/result[w]`, where `result[w]` is the
         already-computed parent result for downstream vertex `w`.
   1d. Run the synthetic graph's elimination via the cached
       PRC, producing `result[v]` for each internal vertex.
   1e. Copy each `result[v]` into the parent-wide result vector
       at parent index `meta.parent_indices[v_synth]`.

2. After all SCCs are processed, the parent-wide result vector
   has `result[v]` for every parent vertex.

### What this means for the existing WP-1..WP-4

The modified synthetic graph differs from WP-1's. Specifically,
WP-1 has one synthetic absorbing per SCC; Option B has one per
external out-channel + one phantom. This means:

- **WP-1 needs revision** — `ptd_scc_build_synthetic_graph` needs
  to produce the new topology.
- **WP-2 (canonical hashing) needs revisiting** — the new
  topology has different invariance properties; cross-parent
  reuse only works when external fan-out shapes match.
- **WP-3 (EXTERNAL pointer)** turns out to be unnecessary again —
  the per-channel weights and phantom rates can be set directly
  on the synthetic graph's edge slots before each compose, and
  the eliminator's pointer-to-edge-weight scheme handles the
  rest. WP-3's machinery is dead weight (but harmless).
- **WP-4 (per-SCC cache)** keeps its API but caches the new
  topology.

### Estimated impact

- WP-1 modification: ~150 LOC (more vertices, more edges per
  external channel; metadata extension).
- WP-2 tests update: rerun on new topology; cross-parent
  invariance test conditions may be more nuanced.
- WP-4 tests update: regenerate cache files; verify cross-parent
  reuse still works for fan-out-matching parents.
- WP-5 implementation: ~400 LOC. Most of the work is bookkeeping
  for the per-channel weight injection.

Total work: maybe 800 LOC of churn across WP-1..WP-5, plus
substantial test rework. But the result is a working
hierarchical pipeline with cross-graph reuse on cyclic SCCs —
which is the original goal of the branch.

I'll proceed with this plan unless the user wants to discuss.

