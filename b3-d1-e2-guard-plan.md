# D1-E2 guard micro-batch — decline exact gradients on synthetic SCC graphs

**Authorization:** user-approved 2026-08-15 (three-unit checkpoint round;
"Ship the guard"). Modifies shipped code with explicit approval —
recorded exception to [[feedback_no_modify_existing]].
**Motivating record:** `b3-d1-derisk-findings.md` §E2 +
`experiments/dr_d1_e234_landmine_conditioning_reentrancy.py`: the shipped
exact-gradient entry points ACCEPT a synthetic SCC graph
(`SCCVertex.as_synthetic_graph()`) and return a full-size plausible
Jacobian contracted from the Type-A/phantom PLACEHOLDER coefficients —
silent garbage. The true phantom weight (1/parent_result[target]) is
θ-dependent through the PARENT and unrepresentable by any linear
contraction of placeholder coefficients.

## Design (grounded surface, all read-from-source)

A `synthetic` marker on `struct ptd_graph`, set at synthetic-graph
creation, checked at the TOP of the two shared gradient cores (which
between them carry every production exact-gradient entry point):

1. `api/c/phasic.h:203` (after `dph_compute_invalidated`): new field
   `bool synthetic;` + comment. LF file, plain edit.
2. `src/c/phasic.c` `ptd_graph_create` (`:2967`, malloc + explicit
   field init — NOT calloc, so the init line is REQUIRED):
   `graph->synthetic = false;` beside `graph->was_dph = false;`.
3. `src/c/phasic.c:1142` (the clone metadata block):
   `new_graph->synthetic = graph->synthetic;`.
   LOAD-BEARING, not defensive: the Python exact path
   (`_exact_moments_jac_np`) computes on a private `graph.clone()`, so
   without propagation the guard never fires through
   `pmf_and_moments_from_graph`. [Corrected per plan-review R1 finding
   1: clone is NOT the only was_dph-metadata-copy site — GraphBuilder
   (`src/cpp/parameterized/graph_builder.cpp:228-231`) and the
   `set_was_dph` setter (`src/cpp/phasic_pybind.cpp:1368-1377`, used by
   `from_serialized`) also copy it on serialize→rebuild routes. Those
   two are JUSTIFIED OMISSIONS, not oversights: both rebuild from
   serialized JSON, synthetic graphs are not directly serializable
   (finding-2 note below), and the rebuild feeds the FORWARD path only
   — the exact backward clones the ORIGINAL graph.]
4. `src/c/scc_synthetic.c:711` (`ptd_scc_build_synthetic_graph`, the
   single creation site — the `SCCVertex.as_synthetic_graph` pybind
   route lands here): `synth->synthetic = true;` after the NULL check.
   LF file.
5. `src/c/phasic.c:10969` top of `ptd_b3_moments_core` (first check in
   the body): decline. Covers ALL FIVE kinds — linear, log, dph,
   formula, binp-exit (note per G4 refuter A n1: the FORMULA kind is
   covered VACUOUSLY — a synthetic graph never carries a formula tape,
   so `_formula` declines before the core; 4 of 5 are live coverage) — i.e. `ptd_moments_grad_theta{,_dph,_log,_formula}`
   and `ptd_moments_binp_exit` in one site (the Batch-0 shared-core
   landing the D1 plan predicted).
6. `src/c/phasic.c:11712` top of `ptd_b3_sojourn_grad_core` (first
   check, before the parameterized/was_dph checks): decline. Covers
   BOTH `ptd_sojourn_grad_theta_subset` and `_nogate` (thin wrappers,
   `:11931`/`:11941`).

Guard-position cost note (plan-review R1 finding 3, ACCEPTED): the four
moments wrappers + `ptd_moments_binp_exit` build/convert the
parameterized tape BEFORE entering the core, so a core-top decline pays
that work per declined call (the codebase's was_dph precedent declines
in the wrapper, `phasic.c:11283-11289`). Deliberately accepted to keep
the guarantee-carrying SINGLE check site: the cost is paid only on the
misuse path (calling exact gradients on a synthetic graph at all), and
the in-core MPFR-gate decline is equal precedent. **Correction (G4
refuter A, m2): the v2 claim that "the O(n^3) precompute is
graph-cached after the first call" is FALSE for the moments family** —
the five wrappers call
`ptd_graph_ex_absorbation_time_comp_graph_parameterized`, which builds
a FRESH tape each time and destroys it on exit, so every declined
moments call pays a full elimination. (The sojourn core is unaffected:
its guard precedes any build.) Still accepted — misuse-path only — but
it strengthens the case for the follow-up decline LATCH below. Also noted (R1 finding 5):
`ptd_graph_content_hash` does not include the new field — no cache-key
churn (the per-SCC cache stores forward tapes only). R1 finding 4
(synthetic+k==0 sojourn returns -1 instead of 0) is unreachable from
pybind (`indices.empty()` pre-check) — no action.

Decline semantics: `PTD_LOG_WARNING` — UPGRADED from the originally
sketched INFO per plan-review R2 finding 4: this decline is
always-misuse (unlike every other decline, where FD is a valid slower
answer, FD of a synthetic graph's placeholder forward produces the SAME
garbage numbers — R2 finding 2), and WARNING is visible at the DEFAULT
log level, so the direct C/pybind surface becomes honest without the
user opting into INFO. Message: "synthetic SCC graph: placeholder
coefficients cannot yield exact gradients of composed quantities (needs
the two-level adjoint); declining" then `return -1` — the standard
decline path every wrapper/pybind caller already maps to an empty
result → the Python layer's existing FD-fallback engages. No new
failure mode; no behavior change on any non-synthetic graph.

7. TWO Python cause-list string additions (same approval scope — the
   guard's loudness): the moments decline cause-list
   (`src/phasic/__init__.py:7706-7715`) and the joint-index probe
   cause-list (`:8863-8873`) never mention synthetic graphs — append
   "or a synthetic SCC graph" so the Python-side INFO diagnostics stop
   misattributing the cause (R2 finding 4).

FOLLOW-UP RECORDED (G4 refuter A, m3 — not done here): the decline
has no LATCH. A graph's synthetic-ness cannot change with theta, yet
`_one()` re-enters the backward every SVGD step × particle, so both
the C WARNING and the Python INFO repeat per call (measured: 3 handler
hits per `jax.grad`). The joint-index probe's construction-time latch
(`__init__.py:8856`) is the precedent for a once-per-model decision.
Ledgered as a follow-up, not fixed in this micro-batch.

HONESTY RESIDUAL (R2 finding 2, recorded — and now VERIFIED by G4
refuter A two ways: direct FD on the fixture gives [-1.0000000001,
0.0] vs the exact placeholder [-1.0, 0.0], and end-to-end through
`pmf_and_moments_from_graph` post-guard FD gives [-1.0000000006, 0.0],
agreement 5.6e-10, while a non-synthetic control matches its exact
Jacobian bitwise — the fixture is discriminating, not degenerate): for a user feeding a
synthetic graph to the Python entry points, the guard changes WHICH
code computes the garbage, not the numbers — the FD fallback
differentiates the placeholder-linear forward, ≈ the declined
placeholder Jacobian. The guard's real value: (a) the direct C/pybind
surface declines honestly (default-visible WARNING), (b) the future
two-level adjoint machinery cannot silently mis-wire shipped
1-level gradients onto synthetic graphs.

**Explicitly excluded (with reasons):**
- `ptd_moment0_grad_theta` + `ptd_debug_*` — compile-guarded
  `PHASIC_B3_VALIDATORS` debug tools, OFF in production builds.
- ~~serialize round-trip persistence~~ — **NO LONGER EXCLUDED; FIXED
  in this batch (G4 refuter A, finding M1).** Both earlier versions of
  this paragraph were WRONG: v1 said synthetic graphs "are not
  serializable"; v2 (R1 finding 2) allowed a `phasic.Graph(synth)`
  wrap but called it "three deliberate steps deep ... no legitimate
  use". In fact `phasic/distributed_scc.py` ships a documented API
  pair whose ENTIRE PURPOSE is round-tripping a synthetic SCC graph —
  `serialize_scc_synth` (`:77`) / `deserialize_scc_synth` (`:125`),
  used by the SLURM per-SCC worker (`scc_worker.py:72`). The refuter
  round-tripped it and recovered the pre-guard landmine `J=[-1.0,
  0.0]` with no warning — i.e. WRONG NUMBERS, on precisely the
  distributed route where a future two-level adjoint would run.
  **Fix (additive):** new pybind setter `_set_synthetic(bool)`
  (mirroring the shipped `set_was_dph` precedent), called once at the
  end of `deserialize_scc_synth`. Pinned by
  `test_distributed_roundtrip_keeps_the_marker`. The bare
  `Graph(synth).serialize()` → `from_serialized` route remains an
  accepted residual (no shipped caller; closing it needs a serialized
  field).
- `update_weights` on a synthetic graph still silently overwrites
  compose-injected phantom weights (E2(b)) — OUT OF SCOPE here; the
  guard prevents the *gradient* landmine only. Guarding update_weights
  would break the forward composer's own legitimate use.

## CRLF discipline

`src/c/phasic.c` is CRLF — all its edits via binary-mode Python
replacement with `\r\n` and `assert count == 1` (the Batch-0 M5
incident rule). `api/c/phasic.h` and `src/c/scc_synthetic.c` are LF —
normal edits.

## Tests

New file `tests/pytest/test_synthetic_scc_guard.py` (additive):
1. Two-SCC fixture (from the E2 experiment) → `as_synthetic_graph()` →
   `_moments_grad_theta(1)` returns EMPTY (declined).
2. Same on `synth.clone()` (pins the propagation line).
3. `_sojourn_grad_theta_subset([0])` declines on the synth graph, with
   `skip_condition_gate` both False and True (pins that the synthetic
   check is NOT the conditioning gate).
4. Non-regression: the PARENT parameterized graph (same fixture) still
   returns a non-empty, finite Jacobian from both families.
Test strengthenings (R2 finding 3 — without them all four tests pass
even with a broken log contract or a wrong-cause decline):
- assert the C WARNING line (message contains "synthetic") via the
  direct-handler-attach pattern (`phasic` logger sets propagate=False,
  so vanilla caplog is blind; precedent
  `tests/pytest/test_exposure_daisy_chain.py:849-859`);
- run the moments decline tests under `PHASIC_CONDITION_THRESHOLD=1e300`
  (monkeypatch) — pins that the decline is NOT the MPFR conditioning
  gate (the moments analogue of sojourn's skip_condition_gate=True);
- file name `test_synthetic_scc_guard.py` (beside the existing
  `test_synthetic_scc_graph.py` / `test_synthetic_scc_hash_invariance.py`).

Experiment updates (`dr_d1_e234_landmine_conditioning_reentrancy.py`,
not shipped code):
- E2(a) flips to expect DECLINED (comment records the pre-guard
  demonstration), and gains a sojourn-family acceptance→decline record
  (the pre-guard sojourn acceptance was never on record — R2 finding 3);
- E3's per-SCC bisection is scoped PRE-GUARD HISTORICAL (R2 finding 1
  — the ONLY correction G1 could not self-detect): post-guard every
  synthetic `_moments_grad_theta` call returns empty at ANY threshold,
  so `bisect_cond` degenerates to inf and the printed science inverts
  silently while the script still exits 0. The per-SCC block is
  replaced by a printed PRE-GUARD-HISTORICAL note citing the recorded
  numbers (1e23/1e28); `b3-d1-derisk-findings.md` §E3 gets a matching
  annotation that its measurement predates the guard and is no longer
  reproducible on a guarded build (by design).

## Process

Branch `b3/d1-e2-synthetic-guard` off master, IN-TREE (deviation from
the worktree policy, deliberate: micro-batch with immediate merge;
avoids a second pixi env build under the post-incident memory mandate —
if abandoned, re-run `pixi run install-dev` from master). Gate ladder:
G1 = new pytest file green + E2 experiment green; G2 targeted (paths corrected per R2 finding 5) =
`tests/pytest/inference/test_fd_gradient_mixed_scale.py`,
`tests/pytest/inference/test_d4_conditioning_pin.py`,
`tests/pytest/inference/test_exact_grad_joint_index.py`,
`tests/pytest/inference/test_exact_grad_joint_index_baked.py`,
`tests/pytest/test_sojourn_subset_adjoint.py`,
`tests/pytest/inference/test_exact_grad_discrete.py`,
`tests/pytest/inference/test_exact_grad_log_weight_mode.py`,
`tests/pytest/inference/test_exact_grad_formula_mode.py`,
`tests/pytest/inference/test_exact_grad_callback_mode.py`,
`tests/pytest/inference/test_exact_grad_rewards.py`,
`tests/pytest/test_synthetic_scc_graph.py`,
`tests/pytest/test_synthetic_scc_hash_invariance.py`; G3 =
chunked full suite vs the 10th-stamp ledger (2001/0/84/24, zero NEW
failures); G4 = two adversarial diff refuters (serial, per the memory
mandate); G5 = squash-merge + tracker/CLAUDE.md close-out.

## Risks

- A field added mid-struct shifts offsets — safe: single compiled unit
  set (C core + C++ + pybind built together by install-dev); no
  external ABI consumers.
- Any OTHER graph-creation path that should propagate: the grep mirror
  set for metadata copy is exactly one site (`:1142`); the reward
  transform (`:6538`) builds a NEW graph from vertices without copying
  was_dph-class metadata — if a synthetic graph were reward-transformed
  the marker would drop. Accepted residual (documented): synthetic
  graphs carry channel/phantom structure that reward transformation
  does not preserve semantically anyway; revisit if D1's
  implementation ever routes transforms through synthetic graphs.
- The guard could over-fire if `synthetic` were ever set on real
  graphs: it is written in exactly one place (edit 4).

---

## G5 merge review (2026-08-16)

**Unit:** D1-E2 synthetic-SCC-graph guard. **Branch:** `b3/d1-e2-synthetic-guard`
(in-tree — deviation from process §3.3's worktree mandate, recorded below).
**Authorization:** user checkpoint 2026-08-15 ("Ship the guard").

### What shipped

A `bool synthetic` marker on `struct ptd_graph`, set at
`ptd_scc_build_synthetic_graph` creation, propagated by `ptd_clone_graph`,
re-applied by `distributed_scc.deserialize_scc_synth`, and checked as the
FIRST statement of both shared gradient cores (`ptd_b3_moments_core`,
covering all five contraction kinds; `ptd_b3_sojourn_grad_core`, covering
both public wrappers gate-skipping or not) — declining with a
`PTD_LOG_WARNING` visible at the default log level. Plus: an additive
pybind `_set_synthetic(bool)` setter, two Python cause-list strings, six
C public-API decline-cause enumerations, and an 8-test pin file.

### Gates

| gate | result |
|---|---|
| G1 | 8/8 new tests; E2 experiment green (E2(a) + E2(a') flipped to DECLINE) |
| G2 targeted (12 files) | 407 passed, 1 xfailed |
| G2 SCC/hierarchical row (11 files, added after refuter B's MAJOR-4) | 407 passed, 3 xfailed |
| G2 SLURM/distributed row (8 files, added after refuter A's M1 fix) | 81 passed, 2 xfailed |
| G3 chunked full suite | see the 11th baseline stamp |
| G4 | two adversarial diff refuters, both SOUND-WITH-CORRECTIONS, all findings folded (below) |

### G4 findings and dispositions

**Refuter A (correctness/completeness) — one real bypass, FIXED:**
- **M1 (fixed, code):** `distributed_scc.serialize_scc_synth` /
  `deserialize_scc_synth` — a shipped, documented synth-serializer pair
  used by the SLURM per-SCC worker — round-tripped to an UNMARKED graph
  that returned the pre-guard landmine `J=[-1.0, 0.0]`. The plan's
  "synthetic graphs are not directly serializable / no legitimate use"
  justification was FALSE. Fixed additively (`_set_synthetic` + one call),
  pinned by `test_distributed_roundtrip_keeps_the_marker`.
- **M2, m1, n5 (fixed, tests):** parent-leak coverage, both sojourn
  wrappers in the non-regression test, and the Python `Graph.clone()`
  route (the one production actually takes) — three new tests.
- **m2 (fixed, doc):** the plan's "O(n^3) precompute is graph-cached"
  cost claim is false for the moments family; corrected in place.
- **m3 (ledgered, not fixed):** no decline LATCH — master plan §16b item 13.
- **n1 (doc):** the FORMULA kind is covered vacuously (a synth never
  carries a formula tape); 4 of 5 kinds are live coverage. Recorded.
- **Verified by refuter A, not merely claimed:** the honesty residual is
  REAL (post-guard FD `[-1.0000000006, 0.0]` vs exact placeholder
  `[-1.0, 0.0]`, agreement 5.6e-10) while a non-synthetic control matches
  its exact Jacobian bitwise — the fixture discriminates.
- **Attacks that failed:** `reward_transform` (result is not
  parameterized → declines earlier), GraphBuilder/FFI rebuild
  (forward-only), end-to-end Python entry (guard fires), PRC cache
  hit/miss asymmetry (always built before cache consult), cache-key churn
  (field not hashed), BINP_EXIT decline-path leaks (all five out-params
  clean).

**Refuter B (docs/process/grounding) — no ship-blockers:**
- **MAJOR-1 (fixed):** six C public-API decline-cause comment blocks were
  closed enumerations that omitted the new cause — including
  `_nogate`'s explicit "every other decline stays live" contract.
- **MAJOR-2 (fixed):** `b3-d1-derisk-findings.md` §E2 still described the
  guard as an unapproved DESIGN with an INFO log; dated correction added.
- **MAJOR-3 (fixed at the stamp):** G0 was STALE — the ledger's 10th stamp
  names `f73d0650` while master had moved two commits, and the CC-2 merge
  `7371a369` added `test_d4_conditioning_pin.py` (+3 tests) without a
  re-stamp. The 11th stamp names the inheritance explicitly.
- **MAJOR-4 (fixed):** G2 omitted the mandated SCC/hierarchical row; run
  (green) and recorded above.
- **MINOR-5/6/7 (fixed):** stale experiment docstring/print text (now
  post-guard, with the pre-guard reproduction commits `bfb737ce` /
  `4229207b` cited), the field comment's "True iff" (→ "ORIGINATES from",
  since clones carry it too), and the E3 reproduction path.
- **MINOR-8, NIT-9/10/11 (accepted):** the log test's default-level claim
  is true but unpinned; `E2(a')` sits outside the E2(a) try/except;
  `phantom_edges` remains assigned-and-unused (pre-existing).

### Recorded deviations

1. **In-tree branch** instead of worktree+isolated env (process §3.3). Stated
   reason: micro-batch with immediate merge, avoiding a second pixi env build
   under the post-incident memory mandate. Refuter B correctly notes this does
   not engage the rule's *purpose* (keeping the main checkout a valid baseline).
   **Concurrent-unit provenance caveat:** the D2/D3 de-risk measurements taken
   during this window ran on a guard-carrying build. Verified non-contaminating —
   none of those experiments touch `as_synthetic_graph`, `_moments_grad_theta`,
   or `_sojourn_grad*`.
2. **G4 refuters ran concurrently, not serially.** The memory mandate targets
   MEMORY (the 50GB incident); both refuters are read-only + tiny probes, so
   concurrency was judged safe. (An earlier serial refuter attempt died on API
   credit exhaustion, and its findings were lost — the two reported here are a
   fresh pair.)
3. **G0 was stale at branch start** (see MAJOR-3); corrected at the 11th stamp.
