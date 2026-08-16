# Deferred-1 implementation plan — hierarchical/SCC two-level exact adjoint (v3, P3(a) scope)

**Status: DRAFT v3 — BOTH refuters' findings folded (technical: 3
MAJOR F1-F3 + F4-F7; process: 1 CRITICAL F-P1 + 5 MAJOR F-P2..P6 +
F-P7..P11 — all corrected below). AWAITING USER SIGN-OFF. NO code is
authorized by this document.**

**Gate-P status [F-P11]:** P1 SATISFIED (Batch-0 shared core shipped,
now five kinds). P2 SATISFIED in its design-coordination reading
(Batch C shipped the output-side binp exit; the input-side seed +
graphless supply is exactly I1's remaining piece — read-confirmed by
both refuters). P3 DECIDED (P3(a), user checkpoint 2026-08-15).
**Authorization to plan:** user checkpoint 2026-08-15 ("Activate" — the
three-unit decision round), at the plan-of-record's default **P3(a)
E[T]-class scope**.
**Design-of-record chain:** `deferred-1-hierarchical-scc-adjoint-plan.md`
(v2, reviewed) → `b3-d1-derisk-findings.md` (E0-E4 all complete) → this
plan. The verified math reference is
`experiments/dr_d1_e1_twolevel_adjoint.py` (commit `739147dc`): value
parity vs the real C hierarchical path 0..8e-16, dense random-cotangent
VJP oracle ≤2.4e-15, assembled Jacobian vs FD-of-the-composer ≤7.3e-10,
on all 6 toy fixtures × 4 θ.

## 1. Scope (hard boundaries, all decided)

- **Quantity: the rewards-free first-moment vector** (E[T]-class; what
  `ptd_compose_scc_prcs` actually produces) — P3(a), user-decided.
  `nr_moments >= 2` stays monolithic-only (the composer itself is
  first-moment-only; extending it is P3(b), out of scope).
- **`weight_mode='linear'`, continuous, no rewards, no was_dph** — the
  composer is silently linear-only (`scc_compose.c:187/:315`,
  `use_log=false` unconditional) and rewards-free; every other mode is
  declined LOUDLY at the entry point.
- **Default paths byte-identical, but NOT purely additive [F-P2 —
  the v2 "Additive only" banner obscured what is being approved].
  TWO SHIPPED-CODE MODIFICATIONS are part of this unit and are
  explicit sign-off items:**
  (i) **I1 modifies `ptd_b3_moments_core`'s internals** — the seeding
  blocks and input-value supply sit inside the shared core that IS the
  body of all five shipped gradient functions; however the entry is
  cut, this is surgery on gate-verified shipped code (verified by
  per-conversion byte-identity + golden gates, but verification is
  not authorization — approval requested here).
  (ii) **I2 threads a retention flag through the shipped composer**
  (the E4 finding: the tape to retain is built by the INNER
  `ptd_expected_waiting_time` call inside `ptd_compose_scc_one`,
  destroyed per-call today — capturing it is flag-threading, not a
  free-standing entry).
  The default composition path and every existing gradient entry stay
  byte-identical (gated); the E2 guard (micro-batch
  `b3/d1-e2-synthetic-guard`) stays in force — the new two-level
  entry is the ONLY legitimate gradient consumer of synthetic-graph
  structures, consuming retained tapes + binding tables, never
  placeholder coefficients.
- **Adjoint pass is SERIAL v1** (E4: the source-first traversal is
  order-incompatible with the forward's sink-first parallel levels — a
  separate pass over retained artifacts is structurally forced;
  parallelizing it later would reuse the level machinery in reverse).

## 2. The algorithm (from the E1-verified reference)

Forward (retention mode ON): compose exactly as today (sink-first
levels, OpenMP), additionally retaining per SCC [artifact list
corrected per review F3]:
- its offset tape;
- a **per-tape-input BINDING TABLE** — for each tape input: its kind
  (Type-B internal / Type-C / phantom; Type-A is NEVER injected — the
  composer's override step touches only Type-C and phantom,
  `scc_compose.c:191-226`, Type-A stays at construction weight 1.0),
  and its binding: a COPY of the real coefficients for Type-B (these
  live on the synth graph's verbatim-copied edges, `scc_synthetic.c:
  864-886` — channel metadata alone cannot supply them), the parent
  edge reference for Type-C, the parent target vertex for phantom,
  the constant value otherwise. This table is REQUIRED for memory
  safety, not just convenience: the offset tape's `inputs[]` are LIVE
  POINTERS into the synth graph's edge structs (`phasic.c:11010`;
  `scc_synthetic.c:1170-1175`) and the synth is destroyed at
  `scc_compose.c:262` — retaining the tape without snapshotting the
  input values/bindings leaves dangling pointers even on fresh builds;
- the `parent_result` values consumed by its phantom weights at
  compose time.
Absorbing SCCs [review F6]: their `parent_result` stays 0 by calloc
convention (`scc_compose.c:290-292`) and any cotangent arriving at an
absorbing SCC is DROPPED (derivative of a constant) — the reference's
explicit rule, stated here so the C implementer carries it.

Reverse (new, serial): source-first over the condensation DAG with ONE
parent-length cotangent accumulator (E1(ii): parent vertex t's cotangent
is consumed by the unique SCC containing t, fed by the caller's seed
plus one phantom-rule routing per upstream channel; source-first order
guarantees completeness). Per SCC:

1. **Cotangent-seeded reverse over the retained tape** (E1(i): the
   per-SCC VJP is the shipped stage-1 mechanism with a non-one-hot
   seed), producing pre-contraction per-tape-input adjoints.
2. **Contraction routing**, three-way by input kind:
   - real θ-edges → contract via the SCC's true coefficients into
     `J_out` (never placeholder coefficients — the E2 landmine);
   - **phantom** (w = 1/parent_result[t]): route
     `adj(parent_result[t]) += −(1/x²)·adj(w)` for x>0; at the 1e300
     clamp the derivative is 0 and the cotangent is DROPPED — the clamp
     fires on EVERY fixture (1-4×), it is the NORMAL absorption
     mechanism, not an edge case (E1 headline finding);
   - **Type-C** (parent-edge-carried weights) → contract via the
     PARENT edge's real coefficients into `J_out`;
   - **Type-A** (source placeholders): adjoint is STRUCTURALLY ZERO
     (the source's M-column is diagonal-only, perturbation-verified
     exactly 0.0) — assert; a nonzero Type-A adjoint is a HARD ERROR
     (bookkeeping bug), never silently contracted.
3. **Start-edge θ-constancy** (E1): start-vertex edges carry
   coefficient arrays but `update_weights` skips the start vertex — the
   outer pass must NEVER contract start-edge coefficients; `dθ` slots
   fed only by start edges remain exactly 0 and double as a pollution
   sentinel in every gate.

Conditioning/decline design (E3 INVERSION — the measured direction):
naive per-SCC condition numbers OVER-decline by 20+ decades
(phantom-weight scale mixing imports the parent's scale); the default
gate is the PHANTOM-EXCLUDED per-SCC aggregate, calibrated against the
whole-graph statistic where the monolithic tape exists — decision
point D-2 below (re-scoped per review F1).

## 3. Batches (each gets its own detailed micro-plan + review at execution)

- **I1 — the P2 exit on the shared core** (the Batch-0 comment-marked
  "[seeding block 1/2 — Deferred-1 marker]" generalizes): an ADDITIVE
  core entry accepting (a) a caller-supplied cotangent seed vector
  (input side — the shipped functions hard-code a one-hot seed at
  vertex 0) and (b) the pre-contraction per-tape-input adjoint exit
  (output side — Batch C's binp-exit class, already shipped for the
  K-moment case). Existing five kinds byte-identical (per-conversion
  byte-identity gates, the Batch-0 method). Gate: seeded-VJP vs the E1
  Python reference's stage-1 on the toy fixtures, machine precision.
  **E2-guard interaction (designed, not accidental):** the shipped
  guard declines `graph->synthetic` at the top of the shared core. The
  seeded entry is the two-level adjoint's INNER mechanism and operates
  on a RETAINED per-SCC tape; its signature must carry the tape and
  the θ-edge coefficient bindings (from the binding table) explicitly,
  NOT the synthetic graph as its contraction source — so the guard
  stays intact for every existing entry and the inner call is
  structurally exempt by construction (never via a bypass flag). I1's
  micro-plan must show this signature explicitly, and must treat
  GRAPHLESS INPUT-VALUE SUPPLY as a first-class signature element
  alongside the seed [review F4]: the true generalization axis is
  seeding + input-value/coefficient supply (the marked block controls
  only the seed; the entry must also bypass the live `inputs[]`
  dereference, the graph-length read, and input-spec validation).
  Flip side [review F7]: being graphless, the new entry gets ZERO
  protection from the E2 guard — correctness of the inner path rests
  entirely on the binding table's kind classification plus the Type-A
  hard-error, and I1's micro-plan must say so.
- **I2 — retention mode in the composer** (B1): opt-in flag threaded
  through the shipped composer (sign-off item (ii) above); v1 FORCES
  FRESH BUILDS under retention (the rev-3 mmap cache-HIT tape's
  buffer lifetime is tied to the mapping — v2 territory); retains
  tape + binding table + consumed parent_result values. Gates:
  retention OFF byte-identical (golden run); retained artifacts
  reproduce the E1 reference's inputs exactly on the toy fixtures;
  **[F-P5] retention peak-RSS + Σ L_scc measured against monolithic
  at the two-locus n=8 fixture HERE, before any I3 code exists** —
  D-3's stop-and-report criterion fires at I2 if retention memory
  exceeds monolithic, so the late-stop scenario is defused early;
  **[F-P6] the E0 record's uninvestigated `twolocus:8 hier-omp1`
  cell error is TRIAGED here** (harness artifact vs a real
  single-thread composer defect) before any hierarchical-path gate
  is trusted.
- **I3 — per-SCC VJP + outer DAG reverse** (B2+B3): new C function,
  signature CARRIES AN EXPLICIT COTANGENT SEED [review F2]:
  `ptd_hier_ewt_vjp_theta(graph, theta, cotangent[n], J_out[P])`-shaped
  (final name at I3's micro-plan) — a seeded VJP over the composed
  E[T] vector, NOT an n×P Jacobian materialization (callers needing a
  row seed one-hot). Gates [restored to the design-of-record's
  mandatory oracle set — review F2; the shipped
  `_moments_grad_theta(1)` output is a 1×P gradient of the moment at
  vertex 0 only (`phasic.c:11007`, `:11124-11126`), so it validates
  ONE cotangent direction and CANNOT be the sole gate]:
  (a) e₀ seed vs `_moments_grad_theta(1)` (pinning the e₀-vs-e_start
  convention explicitly); (b) ≥5 RANDOM cotangent seeds per fixture×θ
  vs the E1 Python reference's adjoint (machine precision) AND vs FD
  of the C composer; (c) Type-A hard-error path exercised (fault
  injection); (d) dθ[0]-sentinel clean; (e) forward-path determinism
  unchanged under OpenMP (the adjoint itself is serial).
  On a stop fired at I2's memory gate, I1 remains merged as validated
  infrastructure with a NAMED future consumer (the master plan §5
  third-consumer note); I3 is never written [F-P5].
- **I4 — wiring + production-scale gates**: pybind validation surface
  first (`Graph._hier_moment0_grad_theta`, underscore-private); the
  public opt-in kwarg decision is D-1 below. Gates: E0's two-locus n=8
  fixture — hierarchical gradient == monolithic gradient (parity) +
  peak-RSS comparison (the unit's raison d'être: monolithic gradient
  measured 96.5s/3.7GB at n=8, ~50GB-class at n=10); n=9/10 measured
  ONLY under the HARDENED memory protocol below, NEVER a parallel
  agent.
  **[F-P1 — CRITICAL correction: the E0 harness's "protocol" is a
  TIME-BOX ONLY (subprocess timeout + post-hoc ru_maxrss readout; no
  rlimit, no watchdog anywhere in `dr_d1_e0_scale.py`) — a time-box
  does not stop a fast 50GB allocation, which is precisely how the
  incident outran human reaction. I4's scale cells therefore REQUIRE,
  specified in I4's micro-plan, all of:**
  (a) child-side `resource.setrlimit(RLIMIT_DATA, cap)` (+RLIMIT_AS
  for Linux portability; on Darwin RLIMIT_DATA is the one malloc
  respects) at subprocess entry;
  (b) a parent-side RSS watchdog (psutil, ~1s poll) that SIGKILLs the
  child at a named threshold (12 GB default) and records the cell as
  a MEMORY-WALL DATA POINT — that is the measurement, not an error;
  (c) staged escalation with hard abort of the remaining ladder on
  any kill (never proceed n=9→n=10 after a kill);
  (d) disk hygiene: tape/cache writes pointed at scratch, per-cell
  cleanup, free-disk check first (the incident also wrote a ≥2.5 GB
  partial on-disk tape).

**Process wiring [F-P3, F-P4, F-P9, F-P10]:** branches
`b3/d1-i1-seeded-core` … `b3/d1-i4-wiring` cut from POST-GUARD-MERGE
master; G0 per sub-batch = the then-current baseline stamp, starting
from the 11th (post-guard) stamp — each sub-batch runs the FULL
ladder G0-G5 and squash-merges separately (up to four baseline
regenerations; the gate cost is accepted for revertability).
Worktree + isolated pixi env per the process §3.3 C-work mandate (no
in-tree deviation — this unit is too large for the guard
micro-batch's exception). G2 targeted map rows: SCC/hierarchical
files, the moments-core exact-grad files,
`test_synthetic_scc_guard.py`. `src/c/phasic.c` is CRLF —
binary-mode edits with count asserts (the Batch-0 M5 rule). If the
guard micro-batch's remaining gates go red, this unit BLOCKS until
the guard is resolved (I1 edits the same function the guard just
touched).

**Sizing honesty [F-P7]:** realistically a SMALL MULTIPLE of the
discrete/was_dph batch (that was 287 lines of C, one new function,
one plan/review cycle; this is four plan/review/gate cycles + core
surgery + composer retention + binding tables + a new two-level
reverse pass + wiring + new fixtures). Expected wall-clock: several
working sessions across multiple days; the I3/I4 parity gates pay
~96.5s per monolithic n=8 gradient call, and each sub-batch's
chunked G3 is a ~40-min suite run. I1 and I2 land first (I1's future
consumer beyond I3 is the master plan §5 third-consumer note); I3 is
the math; I4 is where the E0 value is proven end-to-end.

## 4. Decision points presented at sign-off (defaults proposed)

- **D-1 exposure surface:** v1 ships the private pybind entry + gates
  only (validation-first), NO public kwarg — public wiring (e.g.
  `pmf_and_moments_from_graph(..., parallel_elimination=True)` reaching
  the two-level adjoint for nr_moments=1) is its own follow-up decision
  once I4's parity + memory numbers are in hand. [default: private-only]
- **D-2 decline gate [RE-SCOPED per review F1 — the v1 default was a
  design hole]:** the "whole-graph statistic" is a statistic of the
  MONOLITHIC tape's multiplier stream (`ptd_dbg_tape_needs_mpfr`,
  populated only by replaying a built tape) — uncomputable exactly at
  the A1 forcing scale where the monolithic tape cannot be built. E3's
  evidence says the whole-graph statistic is TRUSTWORTHY, not that it
  is computable hierarchically. [default: a PHANTOM-EXCLUDED aggregate
  over the retained per-SCC multiplier streams — global pmax/pmin over
  the union of per-SCC streams with phantom-tainted commands excluded
  (identifiable from the binding table) — CALIBRATED against the
  whole-graph statistic at sizes where the monolithic tape exists
  (n≤8, which the I3/I4 parity gates build anyway; the E3 fixture's
  cross-SCC 1e12-1e14 cases, which the whole-graph gate correctly did
  NOT decline, are the calibration floor). Decline fallback at scale:
  FD-of-the-HIERARCHICAL-composer (2P compose calls — computable where
  monolithic is not), logged.]
- **D-3 retention memory budget:** retention holds every SCC's tape
  simultaneously (the adjoint needs them in reverse order). EXPECTED
  to be the same order as the monolithic tape in total, but [review
  F5] E0's record contains NO Σ L_scc-vs-L_mono measurement — this is
  an expectation to be MEASURED at I4, not a grounded claim; if I4
  measures retention pushing peak RSS above monolithic, the unit's
  value claim fails and we stop and report. [default: proceed,
  measure at I4, hard stop-and-report criterion]

## 5. Risks carried forward

- The E1 reference validated 6 toy fixtures; no TRAP-vertex fixture
  exists anywhere in the suite (the standing ledger gap) — I3's
  micro-plan must add one (an infinite-E[T] SCC) or explicitly decline
  trap graphs loudly.
- Non-singleton start SCCs need the surrogate-source rule (E1 note) —
  I3 must either implement or decline them loudly; the toy set contains
  singleton-start fixtures only.
- The pybind `scc_decomposition()`-on-temporary dangling footgun (E1
  note) is adjacent-but-separate — ROUTED to the master plan §16b
  ledger as item 12 at this plan's commit [F-P8]; not gated here.
- The composer's silent linear-only weight recompute is a PRE-EXISTING
  hazard recorded in master §16b — this unit inherits linear-only scope
  and must not be blamed for (or silently fix) that value-side bug.
