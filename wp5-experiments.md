# WP-5 experiments — composition feasibility

## Summary

WP-5 (composition of per-SCC PRCs into a parent-level PRC) is
**not a straightforward plumbing task**. Three experiments
revealed that the original design assumption — that
per-SCC PRCs could be cached as standalone artefacts and
composed by appending command streams — is wrong.

## Experiment 5a — monolithic vs per-SCC command streams

Dumped the monolithic eliminator's command stream on toy-base
(84 commands across 6 vertices) and the per-SCC PRCs (13 + 47 +
43 + small commands across 4 SCCs).

**Finding:** the per-SCC command counts add up to roughly the
right total, but the *structure* differs:
- Monolithic emits cross-SCC `NEW_ADD` back-substitution
  commands (commands 75-83) that propagate `result[]` values
  across SCC boundaries.
- Per-SCC PRCs emit `NEW_ADD` commands only within each SCC
  (or to/from the synthetic source/absorbing).

A naive concatenation of per-SCC streams misses these cross-SCC
back-substitution commands.

## Experiment 5b — per-SCC standalone numerical results

Ran each SCC's synthetic graph through the eliminator
independently (as a standalone graph), and compared the
resulting `expected_waiting_time` per parent vertex against the
monolithic.

**Finding:**

| Vertex | Monolithic | Per-SCC standalone |
|---|---|---|
| v0 (s) | 2.32 | 1.0 |
| v1 (A) | 3.64 | ~1.0 |
| v2 (B) | 3.8 | ~1.0 |
| v3 (C) | 1.0 | 1.0 |
| v4 (D) | 1.0 | 1.0 |

The downstream SCC (`{C,D}`) gets correct results in isolation.
Upstream SCCs (`{A,B}` and `{s}`) get *wrong* results in
isolation: their per-SCC computation assumes synthetic
absorbing has `result = 0`, but in the parent, the actual
downstream is not absorbing — its `result[]` is non-zero (e.g.
`result[C] = 1.0`).

This is the central reason composition can't be a simple stream
concatenation. The per-SCC PRC needs values from downstream
SCCs at replay time.

## Experiment 5c — feasibility of post-replay correction

I sketched an approach: at composition time, set the synthetic
graph's Type C edge weights to the parent's external edge
weights (rather than the placeholder 1.0), and inject
downstream `result[w]` values into the synthetic absorbing
vertex via some mechanism.

**Findings:**
1. Modifying synthetic Type C edge weights at replay time is
   feasible — the eliminator emits `multiplierptr` references
   into edge weight slots, so changing `edge->weight` after
   elimination affects the replay numerically. The structural
   commands themselves don't change.
2. **However**, modifying Type C weights changes the rate at
   `d_j`. For the synthetic graph's elimination to remain
   correct after weight modification, the structural assumption
   `rate_synth(d_j) = sum of all synthetic edge weights at d_j`
   must continue to hold. It does — the rate is computed by
   summing edge weights at replay time (per `add_command_param_p`
   summing `*toT` values into `rates[i]`).
3. **The harder problem:** how to inject downstream
   `result[w]` values. The synthetic absorbing vertex has
   `result[s_abs] = 0` by elimination structure (no outgoing
   edges, so its mass goes nowhere). To make it equal a
   downstream `result[w]`, we'd need to either:

   - (a) Change the synthetic graph topology so the absorbing
     vertex has an outgoing edge whose dynamics produce
     `result = result[w]`. This requires a *different*
     synthetic graph per parent (because `result[w]` differs
     across parents), defeating cross-graph cache reuse.
   - (b) Modify the back-substitution commands at replay time to
     inject `result[w]` directly. This requires understanding
     and editing the PRC's command stream — substantially more
     than just changing edge weights.
   - (c) Change the composition strategy entirely — eliminate
     all SCCs in one pass on a "stitched" parent graph that's
     reconstructed from per-SCC subgraphs, accepting that the
     final elimination is per-parent and the cache only saves
     the *per-SCC structure* (which can be thought of as
     "pre-eliminated subgraphs").

## What this means for WP-5

The original WP-5 plan (concatenate command streams + index
translation) cannot work. Three viable directions, in order of
my current preference:

### Option A: cache per-SCC *symbolic structure*, not the eliminated PRC

Re-frame the cache: per-SCC entries store the synthetic graph's
*topology + canonical vertex ordering* (cheap to recompute),
NOT the eliminated PRC. At parent compute time, the composer:

1. For each SCC in reverse-topo order, *re-run the eliminator*
   on the synthetic graph with **parent-correct** Type C edge
   weights and a downstream-result injection mechanism.
2. Emit per-vertex `result[v_parent]` into a parent-wide result
   array.

The cache savings come from skipping the canonical-ordering and
synthetic-graph-construction work, not from skipping
elimination. This is much less ambitious than the original
plan, but it's correct and implementable.

The main downside: this isn't really "caching the elimination" —
it's caching the SCC subgraph structure. The O(n³) work happens
on every compute, just per-SCC instead of monolithically. Still
a win for parallelism (each SCC can be eliminated on its own
thread) but not for repeated computes.

### Option B: cache the elimination but include downstream-injection capability

Keep WP-3's EXTERNAL pointer scheme but extend it to support
"this command's `multiplierptr` is a downstream-result value
that the composer fills in at replay time." This is essentially
what EXTERNAL was supposed to do; it just needs to also affect
the back-substitution commands, not just the rate computation.

This is the path closest to the original WP-5 plan but requires
understanding the back-substitution structure carefully and
identifying which commands need `EXTERNAL`-bound `multiplierptr`
values at composition time.

### Option C: drop the cache; focus on parallelism only

Concede that cross-graph reuse via cached PRCs is too complex
and instead aim for *within-process parallelism* — eliminate
SCCs in parallel within one compute, using the existing
monolithic algorithm but with thread-level parallelism over
independent SCC level sets. The current branch's WP-1 through
WP-4 still help (synthetic graphs, hashing, format extension) —
they just become infrastructure for a different goal.

This loses the "single goal: parallel SCC elimination AND
cross-graph reuse" framing from §1.3 of the reference doc.

## Recommendation

I'd pause WP-5 and discuss the path forward with the user.
Specifically: the original "single goal" framing in the
reference doc may need revision, because the experiments show
that cross-graph reuse of a *fully eliminated PRC* is
substantially harder than the WP-1 hashing experiments
suggested. The cache invariance result in WP-2 was about graph
*content* hashes, not about eliminations — and the elimination
of an SCC depends on values from its downstream SCCs, which
break the parent-independence assumption.

The honest answer to "is this branch on track?" is:
- WP-1 through WP-4 are solid and tested. They produce a
  working synthetic-graph + cache infrastructure.
- WP-5 is harder than expected. Without WP-5, the per-SCC
  cache doesn't actually accelerate any user-facing operation
  — there's no way to use a cached per-SCC PRC to compute the
  parent's `expected_waiting_time`.
- WP-7 (integration) depends on WP-5.

We should discuss before proceeding.
