# WP-5 — composition: parent PRC from per-SCC PRCs (detailed plan)

**Branch:** `hierar-elimin-cache`
**Status:** drafted
**Predecessors:** WP-4 (commit `839224f`)
**Successor:** WP-7 (integration into
`ptd_precompute_reward_compute_graph`).

## 1. Goal

A C function `ptd_compose_scc_prcs` that takes a parent graph, a
list of per-SCC PRCs (each with its synthetic graph and metadata),
and produces a single parent-level PRC numerically equivalent to
running the monolithic eliminator on the parent.

```c
struct ptd_desc_reward_compute_parameterized *
ptd_compose_scc_prcs(
        const struct ptd_graph *parent,
        const struct ptd_scc_graph *scc_graph,
        struct ptd_graph **per_scc_synth,           /* len == n_sccs */
        struct ptd_scc_synthetic_metadata **per_scc_meta,  /* len == n_sccs */
        struct ptd_desc_reward_compute_parameterized **per_scc_prc);
        /* len == n_sccs */
```

Output: a parent-level `ptd_desc_reward_compute_parameterized`
whose commands, when replayed against the parent's edge weights,
produce the same `expected_waiting_time` as the existing monolithic
eliminator.

This is the WP that makes the cross-graph cache *useful* —
without composition, the per-SCC PRCs are isolated artifacts
nobody can consume.

## 2. Why this WP is the most delicate

Three subtleties that must be solved correctly:

### 2.1 The per-SCC PRC is in synthetic-graph index space

Each per-SCC PRC's `from`/`to` fields are **original-graph
parent-vertex indices** (per Experiment 2 finding — the eliminator
records via `original_indices[]`). But the SCC's "original graph"
*is* its synthetic graph. So the PRC's indices range over
`[0, n_synth)`, which is a different space from the parent
graph's `[0, n_parent)`.

The composer must translate each command's `from` and `to` from
synthetic-graph indices to parent-graph indices. The synthetic
source (index 0) and synthetic absorbing (index n_synth - 1) have
no parent counterpart and need special handling — see §3.4.

### 2.2 The pointer fields reference synthetic-graph edge weights

Each PRC command's `fromT`, `toT`, `multiplierptr` are pointers.
Three pointer kinds appear after WP-3's load:
- **MEM** pointers into the per-SCC mem buffer (working scratch).
- **EDGE** pointers to synthetic-graph edge `weight` slots.
- **EXTERNAL** pointers to external-table slots (placeholder
  edges' "external coefficient" stand-ins).

For the composed parent PRC, all three need to be re-resolved to
addresses that make sense in the parent's elimination context:
- MEM pointers stay in the per-SCC scratch (the composer doesn't
  flatten mem buffers across SCCs — each SCC owns its own).
- EDGE pointers to *internal* synthetic edges (those whose
  parent counterpart is also internal to the SCC) must be
  re-pointed to the parent's matching edge weight slots.
- EXTERNAL pointers must be re-pointed to slots holding values
  derived from the parent's external edges.

§3.5 below describes the pointer translation in detail.

### 2.3 The composition order matters

The composer walks SCCs in reverse-topological (sink-first) order
(per §3.5 of the reference doc, verified empirically in the WP-1
experiments). At each step:
- Eliminate downstream SCCs first; their absorption-probability
  results become inputs to upstream SCCs.
- The "absorption probability" produced by SCC `s` for its
  external-output direction `d` is a value the composer must
  compute and store somewhere upstream SCCs can read.

This is what the EXTERNAL pointer kind is for: upstream SCCs'
PRCs reference external slots, and the composer fills those
slots with values from downstream SCCs' computations.

## 3. Algorithm

### 3.1 Inputs and bookkeeping

Inputs:
- Parent graph `parent`.
- SCC decomposition `scc_graph` (n_sccs SCCs, in reverse-topo order).
- Per-SCC: synthetic graph `synth[i]`, metadata `meta[i]`, PRC
  `prc[i]`. Caller (WP-7) computes these via
  `ptd_scc_get_or_compute_prc`.

Bookkeeping that the composer builds:
- `parent_to_scc[v_parent]` — for each parent vertex, which SCC
  contains it.
- `scc_synth_to_parent[i][v_synth]` — for each SCC `i`, mapping
  from synthetic-vertex index to parent-vertex index. Available
  from `meta[i]->parent_indices`.
- `external_table[i]` — for each SCC `i`, the array that backs its
  EXTERNAL pointers. Sized to the SCC's anchor count. Populated
  by the composer per §3.6.

### 3.2 Output structure

Single `ptd_desc_reward_compute_parameterized` with:
- A flat `commands[]` array — concatenation of all per-SCC
  commands, with `from`/`to` translated and pointers
  re-resolved.
- A `mem` chain — all SCC mem buffers concatenated. Each SCC's
  MEM pointer offsets shift by the cumulative offset of preceding
  SCCs.
- A `memr[]` array — for each parent vertex, a pointer to the
  flat mem buffer at the slot holding its rate. Built from
  per-SCC `memr[]` translated through the synth-to-parent
  mapping.

### 3.3 Algorithm outline

```
1. Build parent_to_scc[v_parent].
2. Walk SCCs in scc_graph order (which is reverse-topological).
3. For each SCC i:
   3a. Allocate external_table[i] (size = anchor count from
       ptd_scc_collect_external_anchors).
   3b. Fill external_table[i] with values derived from the parent's
       external edges and from upstream SCCs' computations
       (see §3.6).
   3c. Append translated copies of prc[i]'s commands to the parent's
       commands list.
4. Build the parent's memr[] by walking parent vertices and
   resolving each via the per-SCC memr[] translation.
5. Return the assembled PRC.
```

Step 3b is where the inter-SCC dataflow actually happens. The
external_table values *are* the inter-SCC interface.

### 3.4 Synthetic source / absorbing handling

Synthetic source (synth-index 0) and synthetic absorbing
(synth-index n_synth-1) have `parent_indices[k] = SIZE_MAX`. They
do not appear in the parent graph. Their meaning in the per-SCC
PRC:

- **Synthetic source's edges** carry external-input weights via
  the EXTERNAL anchors (Type A edges). These don't appear as
  *vertices* in the composition; the PRC's commands referencing
  them are translated to reference the parent vertex that the
  EXTERNAL anchor's value flows into.

  Wait — that's not quite right. Let me think again.

  The synthetic source is a *real vertex* in the synthetic graph,
  with synth-index 0. The eliminator processes it like any other
  vertex. So the PRC's commands include statements like
  `result[0] = result[k] * something` where `0` is the synthetic
  source. Those commands need translation.

  But result[0] in the synthetic graph corresponds to "expected
  absorption time starting at the synthetic source" — which in the
  parent graph corresponds to "expected absorption time given mass
  enters the SCC via its upstream-connecting interface." That value
  is consumed by *upstream* SCCs (those that route into this SCC's
  upstream-connecting vertices).

  So the synthetic source's `result[0]` isn't a parent vertex — it's
  an inter-SCC interface value. Two options for handling it:
  - (a) Allocate a scratch slot per SCC for `result[0]`, and have
    upstream SCCs' EXTERNAL pointers reference it.
  - (b) Translate `result[0]` to mean "the parent's
    expected-absorption-from-this-SCC's-entry-point value"
    accumulated into the parent's result vector at the appropriate
    upstream-connecting parent-vertex index.

  Option (b) collapses the synthetic source and the upstream-
  connecting vertex into the parent's representation. It's simpler
  if there's exactly one upstream-connecting vertex; complicated
  otherwise (multiple upstream-connecting vertices, the synthetic
  source distributes mass across them and `result[0]` is a
  weighted sum).

  Actually — neither is quite right. Let me re-think the math.

### 3.5 Mathematical model — the right composition

For a graph with absorption time `T`, the per-vertex result the
eliminator computes is:

  `result[v] = E[T | start at v]`

For an SCC `S` with upstream-connecting vertices `u_1, ..., u_p`
and downstream-connecting vertices `d_1, ..., d_q`, and synthetic
source `s_src`, synthetic absorbing `s_abs`:

  `synth_result[u_k] = E[time inside S | enter at u_k]
                        + Σ_j P[exit S via d_j | enter at u_k]
                          · result_of_downstream_via_d_j`

The synthetic source distributes "incoming mass" across `u_k`
according to its placeholder edge weights. The synthetic absorbing
collects "outgoing mass" from `d_j`. Both are bookkeeping
artifacts — they don't carry semantic value of their own.

In the **parent's** elimination, we need:

  `parent_result[u_k]`

for each upstream-connecting vertex of each SCC. This is what the
synthetic graph's elimination computes as `synth_result[u_k]`,
*provided* the synthetic absorbing's edges from `d_j` carry the
right "downstream value" — i.e. the expected absorption time
starting at the parent vertex that `d_j` connects to outside
the SCC.

So the EXTERNAL pointers' job is:
- Each Type C placeholder edge (`d_j → s_abs`) has a coefficient
  whose value should be the **parent's `result` for the parent
  vertex on the other side of the corresponding parent-graph
  external edge from `d_j`**, weighted by the parent's external
  edge weight.

Hmm wait, that's not quite right either. The Type C placeholder
in the synthetic graph contributes `placeholder_weight ·
result[d_j]` to `result[s_abs]`. We don't *want* `result[s_abs]`
to be meaningful in the parent — what we want is for
`result[d_j]` itself (in the synthetic) to equal the parent's
`result[d_j]`.

The way the existing eliminator achieves correctness on the whole
parent graph is: every command of the form
`result[v] += result[w] * weight` accumulates the contribution
from `w` to `v`'s expected absorption time, weighted by the
transition probability from `v` to `w` (or the edge weight, in
the rate normalisation). For the parent eliminator, **all** edges
are "real" parent edges carrying real coefficients.

For the per-SCC eliminator, edges that exit the SCC (parent-graph
edges from `d_j` to outside) become Type C placeholder edges to
`s_abs`. The placeholder weight (1.0) doesn't reflect the parent's
external edge weight. So `result[d_j]` in the synthetic ends up
including a contribution `1.0 * result[s_abs]`, where
`result[s_abs] = 0` because absorbing vertices have no outgoing.
That makes `result[d_j]` (in the synthetic) **incomplete** — it
misses the contribution that would come from "leaving through the
parent's external edge to a downstream-non-absorbing vertex."

The fix: at composition time, fill `external_table` with values
that, when multiplied by 1.0 (the placeholder weight) and added
to `result[d_j]`, produce the correct contribution from leaving
through the external edge. Specifically:

  `external_table[Type C anchor for d_j → external vertex w] =
       (parent edge weight d_j→w) · result[w_in_parent]`

But wait — `result[w_in_parent]` is something we're *computing*.
It's the result for the parent vertex on the other side of the
external edge. In a reverse-topological walk, that vertex's SCC
has already been processed (sink-first), so its result is known.

OK so the algorithm is roughly:

```
For each SCC i in reverse-topological order:
  Build external_table[i] using:
    - Type A anchors: feed in the parent's external edge weight
      (the weight from the parent vertex in some upstream SCC u
      to this SCC's upstream-connecting vertex). But this feeds
      INTO the SCC, not OUT. The Type A placeholder edge in the
      synthetic graph adds weight * result[u_k] to
      result[s_src]. The "parent's edge weight from external
      vertex u to u_k" is what we want to inject. But result[u_k]
      is what we're computing — we *want* the synthetic's
      result[u_k] to equal the parent's result[u_k]. The Type A
      edge gives us result[s_src] = Σ (Type A weight) · result[u_k],
      which is "expected absorption starting at the synthetic
      source given the input distribution." That isn't what we
      need from the parent's perspective.

      Actually — the synthetic source is *not* the parent's
      starting vertex (in general). It's a bookkeeping vertex we
      added. The parent's starting vertex is a real parent vertex,
      possibly in a different SCC.

      Conclusion: result[s_src] doesn't translate to anything in
      the parent. Same for result[s_abs]. The composer should
      simply *drop* the commands that write to or read from
      synthetic source / absorbing, and rely on result[u_k] and
      result[d_j] (which are real parent vertices) to carry the
      cross-SCC interface values.
```

This is getting complex. Let me step back.

### 3.6 The simpler composition strategy: don't compose, just stitch

Re-reading §3.5 of the reference doc and looking at what
`hierarchical_trace_cache.py` actually does in
`stitch_scc_traces`, the right framing is:

**"Composition" is concatenation of per-SCC command sequences,
with index translation. The cross-SCC dataflow flows through
shared parent vertex indices, NOT through inter-SCC scratch
slots.**

Specifically:
- Each SCC's PRC computes `result[v_parent]` for every
  internal vertex `v_parent` of that SCC. The parent vertex
  indices are used directly (synth-graph index → parent-graph
  index translation via `meta->parent_indices`).
- For an upstream-connecting vertex `u_k` of SCC `i`, and a
  downstream-connecting vertex `d_j` of an upstream SCC that has
  an edge to `u_k`: the parent's external edge weight is what
  flows into `u_k` from `d_j` via a parent-graph command of the
  form `result[d_j] += result[u_k] * weight(d_j, u_k)`.

  This command isn't in either SCC's PRC — it crosses the SCC
  boundary. The composer must **synthesize** these
  cross-boundary commands from the parent graph's external
  edges.

So the composer:
1. For each SCC `i` in reverse-topo order:
   1a. Translate `prc[i]`'s commands and append to the output:
       - Translate `from`, `to` from synth-index to parent-index
         via `meta[i]->parent_indices`.
       - Drop commands that touch synthetic source (synth-index 0)
         or synthetic absorbing (last index): those vertices have
         `parent_indices[k] = SIZE_MAX` and don't exist in the
         parent.
       - Translate MEM pointers via mem-buffer offset shifts.
       - Translate EDGE pointers to synthetic-graph edges:
         - If the edge is internal-to-internal: re-point to the
           parent edge's weight slot.
         - If the edge is a Type A or Type C placeholder: drop the
           command (the cross-SCC contribution is added by the
           synthesized boundary command in step 2).
       - EXTERNAL pointers: rare in this scheme; drop the
         command if the SCC's PRC has any.
   1b. Synthesize cross-boundary commands using the parent's
       external edges:
       For each parent edge `(v, w)` where `v` and `w` are in
       different SCCs:
       Append a command `result[v] += result[w] * (weight(v,w) *
       rate_factor[v])`. The rate factor is the same one the
       eliminator computes for `v` based on its total outgoing
       rate.

That last step (1b) is where the architecture becomes... not the
elegant "EXTERNAL pointer scheme" the WP-3 plan envisioned. The
EXTERNAL pointers turn out to be unnecessary if we synthesize
boundary commands at composition time.

**Major mid-WP-5 reconsideration:** WP-3's EXTERNAL pointer scheme
is the wrong machinery for this composition. The right machinery
is:

- Per-SCC PRCs encode only **internal** elimination (no Type A or
  Type C edges in the saved commands). The synthetic graph still
  needs Type A and Type C edges *to make the elimination work
  correctly* (the eliminator needs a complete graph), but the
  resulting commands referencing those edges shouldn't be saved
  to the per-SCC PRC.
- Composition appends per-SCC commands and synthesizes boundary
  commands from the parent's external edges.

Two options to proceed:
1. Keep WP-3's EXTERNAL pointer scheme and write the composer to
   use it. Possible but more complex than necessary.
2. Bypass EXTERNAL: composer drops Type A / Type C-related
   commands during translation and synthesizes boundary commands.
   Simpler.

**Decision: option 2.** The EXTERNAL machinery from WP-3 was
correct in isolation but turns out to be unnecessary for
composition. WP-3 stays (it's already implemented and tested);
the composer just doesn't rely on it.

This means EXTERNAL pointers will remain in the cache files but
go unused at compose time. Two minor consequences:
- The cache files are slightly larger than they need to be (the
  EXTERNAL pointer kind costs a few bytes per command vs MEM/EDGE).
  Acceptable.
- WP-7 / future work could clean this up by saving without
  EXTERNAL anchors at all (just drop the placeholder commands
  before save). Out of scope for this branch.

### 3.7 The synthesized boundary commands

For each parent edge `(v, w)` where `v` and `w` are in different
SCCs, the composer emits commands equivalent to what the
monolithic eliminator would have emitted for that edge.

Looking at the monolithic eliminator's per-edge command pattern
(`phasic.c:7236+`):

```c
commands = add_command_param_p(
        commands,
        rates[i],                      // accumulator slot for v's rate
        &(vertex->edges[j]->weight),   // edge weight pointer
        1,                              // multiplier (literal)
        command_index++
);
```

This is a `P`-type command: `*fromT += *toT * multiplier`. Where
`fromT = rates[i]` (mem slot) and `toT = &edge->weight` (parent
edge weight). Multiplier is 1.0.

There's also the back-substitution step that produces the
final result-update commands. Looking more carefully at the
monolithic eliminator — the elimination produces a sequence of
commands, and the final step is a back-substitution loop that
emits:

```c
commands = add_command_param(
        commands,
        original_indices[vertex->index],
        original_indices[child->c->index],
        child->weight,
        command_index++
);
```

This is a `NEW_ADD` command: `result[from] += result[to] *
*multiplierptr`. The `*multiplierptr` is the elimination output's
edge weight (a pointer into the mem chain).

For composition, the synthesized boundary commands should mirror
this back-substitution pattern: for each cross-SCC parent edge
`(v, w)` with weight `w_vw`, emit commands that propagate
`result[w]` into `result[v]` weighted by the appropriate
elimination-derived factor.

**This is where the math gets non-trivial and I need to consult
the monolithic eliminator more carefully before writing the
composer.**

## 4. The honest pre-WP-5 unknown

I've been writing pseudocode for §3.5–§3.7 and finding that I
don't actually understand the boundary-command synthesis well
enough to implement it without experiments. The reference doc
§4.5 sketched composition, but the details about how to
synthesize boundary commands depend on facts about the
monolithic eliminator's command structure that I haven't yet
verified.

**Therefore: WP-5 needs an experiment phase before implementation.**
Specifically:

**Experiment 5a:** trace the monolithic eliminator on a 2-SCC
toy graph (e.g. toy-base SCC₁ ∪ SCC₂) and dump the command
stream. Compare to the per-SCC PRCs produced for SCC₁ and SCC₂.
Identify exactly which commands appear in the monolithic stream
that are *not* in either per-SCC stream — those are the
boundary commands the composer must synthesize.

**Experiment 5b:** verify that synthesizing the boundary
commands as proposed (mirror the monolithic per-edge pattern,
parameterised by the parent edge's weight) produces a numerically
correct result on the toy.

Only after these two experiments should I commit to a composer
implementation.

## 5. Revised plan

1. **Run experiments 5a + 5b.** Document findings in
   `wp5-experiments.md`. Cap at 1–2 hours of investigation.
2. **Update this WP-5 plan** with the empirically-grounded
   algorithm.
3. **Implement the composer** based on what the experiments
   revealed. Likely simpler than the speculative version above.
4. **Tests:** numerical equivalence with the monolithic
   eliminator on every toy variant (this is the WP-5 gate that
   §5 of the reference doc calls "the most important single
   test in the branch").

## 6. Why I'm pausing here

I've been too willing to write speculative algorithms in the
plan section. The composer correctness depends on the exact
command-emission behaviour of the monolithic eliminator, which
the WP-1 experiments touched on (Experiment 2 confirmed the
output is index-addressable) but didn't dig into the
command-by-command structure.

Per the working agreements: "No guessing about how existing code
behaves. Read it, or run it, or both." The current speculative
plan §3 violates that. Better to run the experiments first.

## 7. File layout (preliminary)

To be confirmed after experiments:

| Path | Likely change |
|---|---|
| `api/c/phasic.h` | +30 lines: `ptd_compose_scc_prcs` decl. |
| `src/c/scc_synthetic.c` | +400 lines (estimate; depends on experiment findings). |
| `src/cpp/phasic_pybind.cpp` | +50 lines: pybind binding for tests. |
| `tests/pytest/test_scc_compose.py` | new: numerical equivalence tests. |

Estimate: 600 LOC, 1–2 days of work. The experiments determine
whether it lands closer to 400 LOC (clean reuse of monolithic
eliminator pattern) or 800 LOC (substantial new bookkeeping).
