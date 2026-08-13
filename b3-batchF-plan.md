# Batch F plan — D6 `lax.cond`/`vmap` static-dispatch redesign (joint-index)

**Status: v2, 2026-08-13 — the mandated D6.1 adversarial review is DONE
(two refuters, both SOUND-WITH-CORRECTIONS; one CRITICAL each, all folded
below). Cleared for implementation EXCEPT: step F0 is a go/no-go
experiment — if the raise message is not legible under jit, STOP and
present the failure-mode decision to the user before any wiring.**
Design-of-record: `b3-joint-index-plan.md` §D6 (:612-857), as AMENDED by
this document (the amendments below override the sketch where they
conflict). Branch: `b3/batchF-static-dispatch`, worktree
`../phasic-batchF`. Baseline: ledger @ `d2cca7ab` (1885/0/84/24; HEAD
`678ada3e` is docs-only beyond it — G0 satisfied, recorded).

## Design amendments (review-mandated; override the D6 sketch)

1. **Probe index set = `union(all_terminal_indices, [0])`, not `[0]`**
   [JAX-review CRITICAL 1]. The final isfinite sweep in
   `ptd_sojourn_grad_theta_subset` (`phasic.c:11501`) is INDEX-dependent:
   trap/deficit-sink-adjacent rows can be structurally non-finite at every
   theta (the CLAUDE.md-flagged class), so a `[0]`-only probe can commit a
   model whose FIRST real call (whose union always includes
   `all_terminal`) then raises with a misleading "theta-specific" message.
   Probing the construction-known union makes the probe exact for the
   fixed part of every future index set; length check becomes
   `len(raw) == len(probe_union) * P`. Runtime `vi` can still add rows
   (non-baked) — the residual, reflected in the message (amendment 3).
2. **F0 pre-implementation go/no-go: the jitted-raise experiment**
   [JAX-review MAJOR 2]. `dr_lax_cond_vmap_derisk.py` never tested `jit`;
   SVGD's real composition is `vmap(jit(grad(...)))` (`svgd.py:6394/6520`
   → `:4145`, pmap variant `:4137`). Extend the de-risk with
   `vmap(jax.jit(jax.grad(f)))` and `jax.jit(jax.vmap(jax.grad(f)))`
   (+ pmap if devices allow), asserting the original message text is
   findable in `str(e)`. **If not legible → STOP; the raise failure mode
   is a user decision to revisit.** Also record: SVGD itself cannot
   swallow the exception (verified: bare optimize loop, sanitizer operates
   on arrays only).
3. **Raise-path accuracy** [JAX-review MAJOR 3]. (a) Validate
   `vertex_indices` bounds in the callback BEFORE calling C — out-of-range
   indices get their own accurate error, not the conditioning message.
   (b) The committed-decline `RuntimeError` lists the residual causes:
   theta-conditioning (MPFR gate), non-finite Jacobian rows at the
   REQUESTED vertices, or allocation failure — no more "this is
   theta-specific" as fact. (c) Document the pre-existing NaN-theta
   `ValueError` path (`update_weights` validation) in the docstring.
   (No C changes: distinct C return codes were considered and declined —
   shipped-C modification needs its own approval; ledger-noted.)
4. **Probe false-negative logging** [JAX-review MINOR 6]: the
   probe-failure INFO log distinguishes (where possible) out-of-scope
   structure vs ill-conditioning at the reference point, and the docstring
   names `theta=ones` as the reference so users can reason about it.
5. **F1 scope includes the docstring rewrite**: `:7770-7796` (the
   "falls back per call, no exception" and lax.cond cost narrative become
   false) — plus the stale test-module docstring
   (`test_exact_grad_joint_index.py:4-11`, wrong about the default since
   D5) [process-review 3; JAX-review 8].

## Existing-test fate table (process-review CRITICAL 1 — G1 is defined by this)

- **Breaks by design → rewritten**:
  `test_exact_grad_falls_back_to_fd_at_extreme_condition` becomes the
  F2(c) raise test (keep its direct C-level decline-forcing precondition).
- **Keep passing (11)**: `cd_branching`, `unsorted_duplicated_subset`,
  `ffi_and_clone_agree_on_primal`, `theta_dim_override_declines`,
  `cd_native_dph`, `vmap_matches_fd`, `default_path_uses_fd`,
  `exact_grad_false_logs`, `fixed_mask`, `was_dph_declines`,
  `baked_mode_declines`.
- **Keep passing, pre-verify**: `cd_mixed_scale` — confirm via the direct
  C call that exact genuinely succeeds at its theta (a latent decline that
  FD-fallback used to mask would now raise).

## Implementation steps

- **F0** — the jitted-raise de-risk extension (amendment 2). GO/NO-GO.
- **F1** — wiring per D6 + amendments 1/3/4/5. Python-only; the sketch's
  variable drift noted by review (current code names `exact_tbm`, builds
  `_fixed_keep` inline with dtype at `:8319-8325`) — carry the current
  constructions, don't paste the sketch.
- **F2** — tests (mechanisms corrected per both reviews):
  (a) probe-success + spy proving the FD branch is never traced under
  `vmap`: monkeypatch `phasic.ffi_wrappers.compute_sojourn_times_ffi`
  BEFORE model construction (the from-import binds then) and count
  trace-time calls (committed-exact traces ~2; FD adds 4·P) — the
  `_fd_theta_bar` closure itself has no seam [JAX-review MAJOR 4];
  (b) probe-failure → whole-model FD, logged: force via
  `PHASIC_CONDITION_THRESHOLD≈0` set BEFORE construction (the gate reads
  env per call — verified `phasic.c:10656-10658`); restore env after;
  skip on non-MPFR builds [both reviews — the D6.3 discretize() fixture
  CANNOT reach the probe: was_dph is statically excluded first];
  (c) committed decline → `RuntimeError`, message findable, under `grad`,
  `vmap(grad)`, AND the F0 jitted compositions: construct at default
  threshold, then set `PHASIC_CONDITION_THRESHOLD≈0` (or
  `PHASIC_FORCE_MPFR=1`); restore env; MPFR-build skip;
  (d) `exact_grad=False` byte-identity: golden gradients captured from the
  main checkout's `d2cca7ab` install (the worktree pattern keeps it as
  reference); FD is deterministic pure-JAX;
  (e) out-of-range `vertex_indices` → the new accurate error (amendment 3a).

## Gate ladder

- **G0:** recorded above (`d2cca7ab` + docs-only = `678ada3e`).
- **G1:** F2 suite green + the fate table holds exactly (no other test
  changes state).
- **G2:** `inference/test_exact_grad_joint_index.py`,
  `tests/pytest/test_joint_index_callback.py`,
  `inference/test_optimized_joint_index.py`,
  `tests/pytest/test_gate_daisy_chain_joint_probs.py`,
  `inference/test_jax_integration.py` (ledger subset),
  `inference/test_fd_gradient_mixed_scale.py` (process-mandated
  always-run) [process-review 4; paths corrected].
- **G3:** chunked full suite vs ledger.
- **G4:** two diff refuters (wiring fidelity vs amended design; test
  adequacy/process).
- **G5:** merge review; **CLAUDE.md joint-index section rewrite** (the
  lax.cond cost paragraph and follow-up list change) [process-review 3];
  tracker (F unblocks D.3 + E); master-plan §8/§15 tick; **the deferred
  default-flip decision gets its vehicle: presented to the user as an
  explicit post-merge choice** (D6's "D9" pointer is dangling —
  process-review 5); §16b item 2 (sojourn MPFR comment) explicitly
  DECLINED here (no C edits), stays ledgered for Batch E's docs pass.

## Cross-batch notes (review-amended)

- **Batch E** (baked, next on this wiring): must probe the ACTUAL baked
  union (`uniq ∪ all_terminal`) — construction-known, making the probe
  exact for baked mode; recorded for E's plan [JAX-review 9].
- **D.3** (svgd plumbing): inherits F0's jit findings AND must state the
  raise blast radius (one bad particle halts the whole cloud — consistent
  with the recorded user decision, but stated, not implied).
- CC-2/Deferred-4: the env-var semantics F2 relies on
  (`PHASIC_CONDITION_THRESHOLD`/`PHASIC_FORCE_MPFR`, per-call getenv) are
  the same knobs Deferred-4 Phase 1 might extend — pin note recorded.

## Adversarial plan-review record (2026-08-13; = D6.1, first execution)

JAX/design refuter: SOUND-WITH-CORRECTIONS — CRITICAL 1 (probe index set)
→ amendment 1; MAJOR 2 (jit untested) → F0 go/no-go; MAJOR 3 (decline
taxonomy: index-dependent `indices[r]>=n`, env-dependent allocs, NaN-theta
ValueError; size guard confirmed static; mmap latch static) → amendment 3;
MAJOR 4 (spy seam) → F2(a) mechanism; MINOR 5 (fixture infeasible; env
forcing verified) → F2(b/c); MINOR 6 (false-negative logging) → amendment
4; finding 7: state-consistency after a raised backward VERIFIED SOUND (no
correction needed); 8 (anchors: `_baked` exclusion `:7951-7957`, cond
`:8327`; docstring obligations) → folded; 9 (E/D.3 fits) → cross-batch
notes. Process refuter: SOUND-WITH-CORRECTIONS — CRITICAL 1 (test-fate
table) → §above; 2 (fixture) → F2(b); 3 (CLAUDE.md + module docstring) →
G5/F1; 4 (G2 gap) → folded; 5 (dangling D9) → G5 decision-point; 6-12
(anchors, spy, byte-identity procedure, tracker staleness, G0 note,
pinning, probe false-negative) → folded. Tracker hygiene items fixed in
the same pass as this plan's v2 commit.

## Merge review (G5) — 2026-08-13

**All gates green; squash-merged to master.**

- **F0 (go/no-go):** GO — the diagnostic marker survives legibly in
  `XlaRuntimeError` under `vmap(jit(grad))` and `jit(vmap(grad))`
  (`dr_batchF_jit_raise_derisk.py`; pmap variant declined — no
  multi-device host, recorded here per the amendment's conditional).
- **G1:** 17/17 file tests (13 pre-F, all accounted for in the fate table
  — the earlier "+3 vs +4" confusion was a narrative slip in MY tally
  message, not a table defect: 11 keep + 1 pre-verify + 1 rewritten = 13;
  16 after F2, 17 after the G4 fold-ins). Golden `exact_grad=False`
  bit-identity vs the pre-F master install: PASS, max-abs-diff 0.0.
- **G2:** green — exactly the 9 ledgered sources-on failures.
- **G3:** chunked, **1888 / 0 / 84 / 24** = ledger 1885 + 3 net new tests.
- **G4:** wiring reviewer SOUND-WITH-CORRECTIONS (mechanism fully
  verified: zero residual traced predicates; committed-path math
  byte-identical minus the cond; all 5 amendments confirmed in source);
  tests/process reviewer SOUND-WITH-CORRECTIONS. All folded (`05fcc5be`):
  vmap-wrapped spy; `jit(vmap(grad))` in the raise test; the NEW
  vmap out-of-range test — which settled the reviewers' composed finding
  empirically (the batched forward FFI silently NaN-fills bad indices, a
  PRE-EXISTING gap, so the backward bounds check is the live defense
  under vmap and fires correctly); probe-success + `cd_mixed_scale`
  direct-C preconditions as evidence artifacts; docstring/comment
  accuracy fixes.

**Deviations / decisions queued:**
1. **Default-flip (the dangling "D9"): OPEN USER DECISION, presented
   post-merge** — with the vmap double-cost fixed, should
   `exact_grad` default to True? Trade recorded in the docstring: exact
   correctness + hard-stop declines vs FD-favoured cost at P=2.
2. §16b item 2 (sojourn MPFR-comment correction): DECLINED here (no C
   edits in this batch); remains ledgered for Batch E's docs pass.
3. The batched forward FFI's silent NaN-fill on bad indices is a
   PRE-EXISTING robustness gap now documented by the new test —
   ledger-noted (same family as §16b item 8).
4. Unblocks: D.3 (Phase 1b) and Batch E (Phase 2).
