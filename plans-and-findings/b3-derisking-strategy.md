# B3 de-risking strategy — replacing the mixed-scale-broken FD gradient

Branch: `fd-b3-derisk` (off master `c97dccb4`). This is a **de-risking strategy**
(experiments + decision gates to run *before* committing to a B3 implementation),
not an implementation plan. Every claim is source-grounded.

## 1. The pinned defect (recap)

Every SVGD-facing gradient is a `custom_vjp` whose backward re-runs the opaque
C/C++ forward at θ±eps, central difference, **`eps=1e-7` absolute, same step for
every parameter** (moments `__init__.py:~6786`, pdf `~:3681`, daisy `~:4649/:4880`,
reward-visit `ffi_wrappers.py:1296`). An absolute step cannot serve mixed
parameter scales; the defect is pinned by `tests/pytest/inference/test_fd_gradient_mixed_scale.py`
(exact-oracle: 4–9% error at θ=[1,1e-8]; daisy path **hard-crashes** — the −1e-7
step drives a ~1e-8 rate negative and the FFI aborts the backward, which kills the
SVGD run).

## 2. Why FD was chosen (grounded — the SCC/cyclic reason)

**Every production forward runs through opaque C/C++** (XLA-FFI `GraphBuilder`, the
C `parameterized_reward_compute_graph` PRC tape, or C uniformization) — none is a
JAX-traceable computation, so `custom_vjp` must hand-supply a backward, and FD is
the only backward that needs nothing but forward evaluations.

The one pure-JAX differentiable object, `evaluate_trace_jax` (`trace_elimination.py:1475`),
is **not usable as-is** for three independent reasons:
1. **It refuses cyclic graphs.** `record_elimination_trace` raises `RuntimeError`
   on the self-loop that Gaussian elimination of any cycle produces
   (`trace_elimination.py:845-853`); the 1/(1−q) geometric self-loop correction
   "is not implemented" (the C PRC tape *does* apply it, `phasic.c:~4274`). A
   cyclic subgraph is an SCC of size>1 — and coalescent/recombination models are
   cyclic. This is the retirement reason cited at `__init__.py:1391-1422`
   ("numerical bugs on cyclic graphs … RuntimeError on non-parameterised graphs")
   and the divergence evidence in `tests/pytest/failing_tests.md:53-57`
   (`expectation()` 1.61 vs 21.11 on a cyclic graph; a `scc.hash()` segfault).
2. **It doesn't compute a distribution quantity.** It emits the *eliminated
   chain's* `{vertex_rates, edge_probs, vertex_targets}` (`trace_elimination.py:1611-1615`),
   not a moment/pmf/sojourn; the reduction to a scalar currently happens via
   `instantiate_from_trace` → rebuild a Graph → query **in C++** (opaque again).
3. **It is unwired.** Zero production callers; the `hierarchical`/`cache_trace`
   kwargs that gated it are forced off (`__init__.py:1422`).

**The SCC decompose/cache/stitch is numeric C, opaque to JAX.** The production
moments path routes to C++ `super()` using the "Stage A0-cached PRC graph"
(`__init__.py:2122-2138`); per-SCC compute + stitch is numeric C with OpenMP
(`scc_compose.c:138-259`, stitching numeric doubles via `1/parent_result[...]`);
the cache stores the serialized PRC tape/results, not a JAX graph. (A *separate*
Python `hierarchical_trace_cache.stitch_scc_traces` merges symbolic traces
differentiably — `hierarchical_trace_cache.py:1641-1668` — but each per-SCC trace
comes from the cyclic-refusing recorder, so it can't produce a trace for a real
multi-vertex SCC. It is retired/dead.)

**Bottom line:** exact AD is blocked at the *availability of differentiable raw
material* on cyclic (target-domain) graphs — not at a JAX-composition subtlety.
So the naive plan ("Tier 1 = replace FD with `evaluate_trace_jax`", tmp.md:37)
is not viable without a full cyclic-elimination rewrite.

## 3. Per-quantity feasibility (from the investigation)

| Quantity | Production forward | Diff. trace available? | Exact-AD blocker |
|---|---|---|---|
| moments | C++ PRC / FFI (`__init__.py:2137`, `6934`) | only via retired cyclic-refusing recorder | cyclic-refuse; reduction in C++; not recorded at grad site |
| sojourn / joint-index | C `ptd_expected_sojourn_time_subset` **adjoint** tape (`phasic.c:10181-10264`) | tape in C (diff. structure), not JAX | adjoint is w.r.t. reward SEED, not θ (yet) |
| continuous PDF | C++ uniformization FFI (`__init__.py:3730`, `3652`) | no (Poisson series) | not trace-expressible |
| discrete PMF | same FFI | no | same |
| **daisy-chain** | per-epoch `stop_probability` uniformization FFI (`__init__.py:4615-4663`) | no | uniformization + epoch loop in C++ |
| reward-visit-prob | FFI `pure_callback` (`ffi_wrappers.py:1255`) | no | opaque FFI |

## 4. The three fixes, re-ranked by feasibility (this inverts tmp.md's ordering)

- **Tier 2 — scale-aware / clamped FD step.** Cheapest, backward-only, independent.
  Turns the *crash* (highest-severity symptom) into a bounded approximate gradient.
  Not a full accuracy fix, but removes the SVGD-killing abort.
- **Tier 3 — C reverse-mode θ-adjoint.** The **durable** answer and the only one that
  works on cyclic (target) graphs today. Strong precedent: the sojourn adjoint
  (`c340bedc`, `phasic.c:10197-10264`) already walks the PRC command tape in reverse
  (`adjoint[to] += adjoint[from]*mult`) w.r.t. the reward seed; a θ-adjoint is the
  same tape walked in reverse chained through the edge coefficients.
- **Tier 1 — exact AD via a (rewritten) trace.** *Riskiest.* Requires (a) a cyclic
  elimination recorder with self-loop correction, (b) a JAX-side reduction to the
  scalar quantity, (c) `use_log` threading. Only covers moments/sojourn/joint-index
  (never the uniformized PDF/daisy). Likely dominated by Tier 3 for the cyclic case.

## 5. De-risking experiments (each: hypothesis → build/measure → decision gate)

Ordered cheapest/most-decisive first. None requires committing to an implementation.

### DR-D — Tier-2 scale-aware step: does it stop the daisy crash? (cheapest, independent)
- **Hypothesis:** a per-parameter *relative* step `eps_i = max(rel·|θ_i|, floor)` (or a
  clamp keeping every perturbed rate > 0) turns the mixed-scale crash into a
  finite gradient matching the scale-matched reference.
- **Build/measure (branch spike):** in `_autodiff_bwd`/`_per_obs_bwd` (and the moments FD)
  swap the absolute eps for a relative/clamped step; re-run `test_fd_gradient_mixed_scale.py`
  — does `test_daisy_fd_correct_at_mixed_scale` stop crashing and match the ref? Sweep
  the regime grid vs the scale-matched oracle; confirm the forward is untouched
  (backward-only) and existing daisy/moments gates stay green.
- **First sub-task (important): find out WHY the relative step was already reverted**
  — `12a30a78 revert(numerical): roll back the relative FD step to master's absolute step`.
  Read that commit + whatever test/parity drove the revert; the revert reason is a
  landmine this DR must clear (likely forward-parity or a specific gate, not accuracy).
- **Gate:** stops the crash + matches ref within a stated tolerance + no gate regression →
  ship as an independent stop-gap (removes the SVGD-killing abort). Else document why.

### DR-A — Tier-1 make-or-break: can a cyclic elimination trace be recorded correctly? (forward-only, no AD)
- **Hypothesis:** adding the C tape's self-loop correction (drop the eliminated vertex +
  `1/(1−q)` geometric term) to `record_elimination_trace` makes the trace forward
  reproduce the C forward on a **cyclic** parameterized graph.
- **Build/measure:** prototype the correction on a minimal cyclic fixture; compare the
  trace forward (recorded→reduced) to the C/FFI forward to ~1e-13, reusing
  `test_gate_trace_ffi_equivalence.py`'s parity backbone (it currently *skips* cyclic for
  the trace engine — this DR is exactly un-skipping it).
- **Gate:** correction is tractable + matches → Tier 1 has a path (proceed to DR-B). If it
  needs a deep elimination rewrite or won't match → **Tier 1 is dead for the target
  (cyclic) domain → commit to Tier 3.** This single experiment decides Tier 1 vs Tier 3.

### DR-B — Tier-1 AD on an ACYCLIC fixture (only if DR-A is promising)
- **Hypothesis:** `evaluate_trace_jax` + a JAX-side reward/sojourn reduction gives an
  exact gradient where FD is wrong.
- **Build/measure:** on an acyclic parameterized fixture (where the recorder already works),
  wire `evaluate_trace_jax` + reduction; compare `jax.grad` to FD and to the closed-form/
  central-diff oracle at mixed scale (θ~[1e-8,1]); confirm forward parity vs FFI.
- **Gate:** trace-AD exact where FD is 4–9% wrong → Tier-1 accuracy proven for acyclic
  (cyclic still gated on DR-A).

### DR-C — use_log threading (Tier-1 landmine, cheap)
- Confirm a log-mode graph's Tier-1 gradient is correct only with `use_log=True`
  (`trace_elimination.py:969-989`; `test_trace_requires_use_log`). Pin it so a future
  auto-detect can't silently differentiate the linear function.

### DR-E — Tier-3 θ-adjoint from the sojourn exemplar (the durable path; works on cyclic)
- **Hypothesis:** the reverse-mode sojourn adjoint (`phasic.c:10197-10264`) extends to a
  θ-gradient by walking the same PRC tape in reverse, accumulating d/d(edge-weight) and
  chaining to dθ through the `DOT` edge coefficients — O(n) memory, one reverse pass,
  cyclic-correct (the tape already encodes the self-loop correction).
- **Build/measure:** a θ-adjoint over the PRC tape (C) on a small **cyclic** parameterized
  graph; gate vs FD and a central-diff oracle (~1e-13), reusing `test_sojourn_subset_adjoint.py`'s
  pattern (adjoint vs independent forward vs `PHASIC_SOJOURN_FORWARD=1` legacy).
- **Separate sub-lead (do NOT trust yet):** `ptd_graph_pdf_with_gradient` (`phasic.c:11805`,
  declared `api/c/phasic.h:1397`) is an *unwired, forward-mode* PDF gradient with a red-flag
  "the lambda gradient term should be SUBTRACTED … the minus sign gives correct results"
  (`phasic.c:11935-11941`), zero callers. Re-derive + independently test before any PDF
  adjoint relies on it; forward-mode also doesn't scale (O(n·n_params)).
- **Gate:** θ-adjoint matches the oracle on cyclic graphs → this is the production fix for
  moments/sojourn/joint-index. Daisy/PDF adjoints follow the same template (harder — the
  uniformization series + epoch loop, including the per-epoch IPV projection, must be
  reverse-differentiated).

### DR-F — performance guard (any tier)
- Measure gradient wall-time of the candidate fix vs `GraphBuilder`-forward + FD across
  graph sizes, especially large SCC-decomposed graphs where the C composer/cache is the
  reason the trace path was abandoned. A correct-but-slow fix that regresses SVGD on real
  models is not shippable.

## 6. Decision tree / recommended sequencing

```
DR-D (scale-aware step)  ──►  stops the crash?  ──► YES ─► ship stop-gap (independent)
        (cheap, backward-only)                     └─ NO ─► document the 12a30a78 blocker

DR-A (cyclic recorder forward)  ──►  reproduces C forward?
        ├─ YES ─► DR-B (acyclic AD) + DR-C (use_log) ─► Tier-1 for moments/sojourn/joint-index
        └─ NO  ─► Tier-1 dead ─► DR-E (Tier-3 θ-adjoint) as the production fix

DR-E (Tier-3 θ-adjoint)  ──►  the durable answer for the cyclic target domain,
        pursue in parallel with DR-D regardless of DR-A.
```

**Recommendation:** run **DR-D** (crash stop-gap) and **DR-A** (the Tier-1 make-or-break)
first and in parallel — they are cheap and each independently decisive. DR-D can ship the
highest-severity fix (no more SVGD-killing aborts) on its own; DR-A tells us whether the
"exact-AD via trace" dream is alive or whether **Tier-3 C adjoint (DR-E)** is the real
production fix (it is the only cyclic-correct exact route today). Treat Tier-1 as
contingent on DR-A.

## 7. The de-risk harness (already ~built — B0/B1/B2 done)

Reuse, don't rebuild: `test_gate_trace_ffi_equivalence.py` (forward parity TRACE==pybind==FFI;
`test_trace_requires_use_log`; currently skips cyclic → DR-A un-skips it),
`test_sojourn_subset_adjoint.py` (Tier-3 exemplar's gate), `test_gate_daisy_chain_joint_probs.py`
(daisy value + the loose 5e-2 gradient pin — a weak spot to tighten),
`test_fd_gradient_mixed_scale.py` (**this session** — the mixed-scale correctness target the
fix must beat; its two strict-xfails XPASS when the fix lands), `test_gate_ffi_vs_pybind.py`,
`test_gate_scc_ordering.py`, `inference/test_trace_jax_compat.py`. The referenced
`audit-fd-step-remediation-plan.md` is absent from the tree; this document supersedes it.

## 8. Risks / non-negotiables

- **Forward parity is sacred.** Any new backward must leave the forward bit-identical; a
  new *forward* (Tier-1 trace) must match the FFI within granularity tolerance
  (`test_gate_trace_ffi_equivalence.py`).
- **use_log (F-004).** Tier-1 must thread `use_log`; log-mode is silent-wrong otherwise.
- **Cyclic is the domain, not the edge case.** Any tier that only handles acyclic graphs
  is not a production fix.
- **Additive / opt-in.** Prefer a flag selecting the analytic backward with FD fallback,
  so the swap is reversible and can be gated per-quantity.
- **SCC-ordering split-brain** (`scc_compose.c:324-373` Kahn vs `scc_graph.cpp:70-81` stored
  Tarjan; 0% native coverage) — only relevant if a Tier-1 build consumes the Python SCC
  ordering; note it, don't touch it blindly.
