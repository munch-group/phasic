# Deferred 4 — MPFR conditioning-floor: decision-tree plan (sweep → gate → full adjoint)

**Status: v2 — adversarially reviewed 2026-08-11 (dedicated technical refuter:
SOUND-WITH-CORRECTIONS; cross-plan conflict refuter: COMPATIBLE-WITH-CORRECTIONS);
all findings folded in, see §8. Planning-only — no code changes of any kind
are authorized by this document.** Unlike Deferred 1-3, part of this
unit is *already scheduled*: the master plan's Phase 1 contains the "Deferred-4
DR-A re-confirmation" cheap check (§14, §15). **This plan is the detailed
design of that scheduled check plus the decision tree hanging off its outcome
— it does not schedule the check twice.** Everything beyond Phase 0 below is
conditional and expected, on current evidence, NOT to run.

**Grounding.** Primary source:
`atlas/plan-feasibility-callback-and-conditioning-floor.md` Job B ("CCF §n"),
master plan §14 and risk items 9 and 17, `b3-experiment-findings.md` (DR-A),
`experiments/dr_a_cyclic_analytic_gradient.py`. Key corrected fact (master
plan §14): DR-A's historical "conditioning floor" was measured on a
**discarded `jnp.linalg.solve` prototype**, not the shipped adjoint; the one
shipped-adjoint check done during master-plan review (single fixture) showed
the existing MPFR **decline gate fires exactly where corruption would begin**
— no silently-wrong zone observed there. This plan's Phase 0 broadens that
single-fixture check into real evidence.

---

## 1. The decision tree (the whole unit on one page)

```
Phase 0 (scheduled; master-plan Phase 1 cheap check, detailed here):
  broadened DR-A sweep of the SHIPPED adjoints vs. a high-precision oracle
        │
        ├─ outcome CLEAN (gate fires before corruption, everywhere tested)
        │     → PARK the unit. Deliverables: sweep report; the never-written
        │       "documented-regime" pin test; a CLAUDE.md note. No C changes.
        │
        └─ outcome GAP (a silently-wrong zone exists somewhere)
              → Phase 1: gradient-specific decline metric (small; additive
                gate + a one-line consult insertion in the shared core, see
                §3; after Batch 0). Decline → logged FD.
                    │
                    ├─ regime rare in practice → STOP here (expected).
                    │
                    └─ telemetry/user evidence shows the regime is hit often
                       AND decline-to-FD is unacceptably lossy
                          → Phase 2: full MPFR adjoint (large; after
                            Batches 0+A+B(+C) on the final shared core).
```

## 2. Phase 0 — the broadened sweep (scheduled work; experiments only)

**Question:** does any (fixture × mode × moment-order × θ) combination exist
where a shipped **moments-family** exact-gradient function
(`ptd_moments_grad_theta`/`_log`/`_dph` — the sojourn function is excluded
per CCF Job B's explicit scope, "the three moment functions only"; an
optional small sojourn slice may be added but is not owed [review F10])
returns a **silently wrong** Jacobian (no decline, no NaN) beyond an
accuracy floor FD would not also hit?

**Method (no `src/` changes):**
1. **Oracle:** an `mpmath`-based **dense high-precision reference**, NOT a
   tape replay — the recorded tape is not readable from Python without new
   bindings, and writing those would violate this phase's no-code rule.
   Build `alpha`/`S(θ)` from `serialize()`, compute the moment vector and its
   analytic θ-Jacobian by dense high-precision linear algebra (200+ bits;
   resolvent-derivative identities, no matrix exponential —
   [[feedback_avoid_matrix_exp]] respected), with the value/Jacobian
   *semantic mapping* to this library's `moments(K)` convention calibrated on
   benign cases: the oracle must agree with the shipped values and shipped
   exact Jacobians to ~1e-13 on well-conditioned fixtures, and with
   `jax.jacobian` of a float64 dense reference, before it is trusted on any
   disputed point. Being tape-independent makes it *more* independent than a
   tape replay would be (it cannot share a tape-level bug with the code under
   test). Pure-Python experiment code (`experiments/dr_d4_mpfr_oracle.py`).
   Limitation to record honestly: a dense oracle cannot supply per-command
   `s0`/`s1` statistics, and **neither can a `PHASIC_B3_VALIDATORS=ON`
   build** — the validators expose exactly three aggregate bindings
   (`_debug_fwdmode_grad`/`_debug_reverse_grad`/`_moment0_grad_theta`,
   per-edge or per-θ outputs only, `phasic_pybind.cpp:1904-1914`), no
   tape/command-level data [review F1 — the v1 validators claim is
   RETRACTED]. The §2.4 statistics therefore come from:
   (i) the **actual** existing-gate condition number per swept point,
   recovered exactly by **bisecting `PHASIC_CONDITION_THRESHOLD`** (the gate
   reads it from env on every call, `phasic.c:10655-10659`) against observed
   decline/no-decline — this fully serves the "does the existing metric
   separate GAP points" ROC question; and
   (ii) candidate s0/s1-style metrics only as a **labeled approximation**
   via an independent Python elimination re-derivation — fidelity limits
   stated: the real tape's multipliers depend on the ordering choice
   (`use_dyn_ordering`), the diagonal `-1` convention, and self-loop
   corrections, so this is a proxy of the gate's statistic, not the
   statistic itself [review F4] — with true command-level measurement
   deferred to Phase 1's own in-C mini-plan if a GAP is found.
   **Additionally [review F2]:** the sweep's `_dph`/log slices need their
   own oracle mathematics, which the continuous resolvent identities do NOT
   cover. The discrete recipe: a continuous-style moment chain on the
   probability matrix followed by the θ-independent Stirling/binomial K×K
   correction map (`phasic.c:11087-11119`, mirroring
   `graph_builder.cpp:694`); for `was_dph`, the renorm quotient rule
   `∂p_e/∂θ_j = (c_e^j − p_e·Σ_{e'}c_{e'}^j)/S_v` (`:11128-11141`); log mode
   uses `w_e = Π(c_kθ_k)`. And `serialize()` carries NEITHER `weight_mode`
   NOR `was_dph` (`_graph_serialize.py:26-37`) — the oracle builder must
   take both from the live graph's attributes, never from the serialized
   dict. If any slice's oracle cannot be calibrated to the benign-case
   agreement bar, that slice is verified against closed-form/`jax.jacobian`
   float64 references only, and the report says so explicitly.
2. **Sweep axes:** DR-A's cyclic fixture + >=3 further topologies (chain,
   branching, coalescent-class); linear + log modes; native-DPH/`was_dph`
   where in scope (`_dph`); moments K=1..3; rewards once Batch A lands
   (re-run the relevant slice then — noted as a follow-on, not a blocker);
   θ sweeps crossing the corruption threshold continuously (the regime DR-A
   probed: one component → 1e-8).
3. **Classification per point:** (a) exact returned & correct; (b) declined →
   FD (log line observed — verifying [[feedback_no_silent_fallbacks]] end to
   end); (c) exact returned & wrong beyond tolerance = **GAP**. Also record
   FD's own error at the same points (context for "is decline-to-FD lossy").
   **Sweep protocol requirements [review F9]:** record the build config
   (`HAVE_MPFR` — on a non-MPFR build the gate returns 0 unconditionally,
   so a CLEAN verdict there is meaningless); keep `PHASIC_FORCE_MPFR` and
   `PHASIC_CONDITION_THRESHOLD` unset except where bisection sets the latter
   deliberately; raise the phasic gradient logger to INFO — the decline
   lines are invisible at the default WARNING level (`__init__.py:6953`),
   so classification (b) is unobservable without this.
4. **Gate-metric adequacy analysis (master-plan risk 9):** for every swept
   point, record the existing gate's condition number — recovered via the
   §2.1(i) threshold bisection — AND candidate gradient-specific statistics
   (min |s1|, max quotient-term magnitude |s0/s1²|) via the §2.1(ii)
   labeled-approximation re-derivation [provenance corrected per review F3 —
   the v1 "from the oracle replay" wording contradicted §2.1]. If GAP points
   exist, test whether the existing metric separates them (ROC-style); this
   decides whether Phase 1 needs a *new* metric or a threshold change
   [CCF Job B §Q2 point 5 — plausible under-detection, unproven; citation
   corrected, no "§Q2.5" exists].

**Deliverables:** sweep report (`b3-d4-sweep-findings.md`); a **proposed**
regression pin + CLAUDE.md note, presented for approval at report time — the
no-code rule means they ship only with user sign-off, and the pin must
skip/xfail on non-MPFR builds [cross-plan F9, review F9]; go/park
recommendation. **Cost-class note [review F8]:** this design makes the
master plan's "[cheap check]" a small project (order: days, not the "ten
minutes" the single-fixture version took) — broadening beyond the
closed-form DR-A fixture requires the oracle; the master plan's Phase-1
line now says so and names this plan as design-of-record.

## 3. Phase 1 — gradient-specific decline metric (conditional: GAP only)

- **Shape:** a second gate function computed from stage-0 snapshots (the
  O(L) scan of |s1| / quotient magnitudes — cheap, same pass that already
  computes `nm[]` statistics), consulted alongside `ptd_dbg_tape_needs_mpfr`.
  `ptd_dbg_tape_needs_mpfr` itself is NOT modified — its semantics are
  shared with **four** shipped call sites (`:10783/:10960/:11221` plus the
  production sojourn function at `:11529` [review F5 — count corrected]).
  **Honesty about "additive" [review F5]:** consulting a second gate still
  requires a one-line insertion inside the shipped (or Batch-0-extracted)
  function bodies — a small modification of existing code, flagged for
  approval in Phase 1's own mini-plan, not zero-modification.
  **Implementation brief, recorded now so it is not rediscovered as a bug
  [review F11]:** `s0`/`s1` are `malloc`'d, not `calloc`'d, and written only
  for command types 2/3/5 (`phasic.c:10757-10758`, `:10772-10775`) — any
  "min |s1|" scan must index only assigned entries, never the whole array.
- **Sequencing:** after Batch 0, so the insertion is one site in the shared
  core, not three copies — the same argument the master plan makes for every
  shared-core change. **Concurrency caveat [cross-plan F3]:** Phase 1 fires
  on the same "after Batch 0" trigger as Phase 3's Batch A → B/C; if any of
  them is in flight when Phase 1 activates, coordinate explicitly (rebase
  discipline per master §4's A/B note) — "no concurrent edits" holds only
  if this ordering is actively managed, not by construction. If Batch 0
  were somehow abandoned, this becomes three identical insertions and D.2's
  backport lesson applies (fix must land in all siblings; the atlas
  documented exactly this failure pattern).
- **Validator-drift guard [review F6]:** if Phase 1 ships, mirror the new
  gate into `ptd_moment0_grad_theta` (or annotate the validator's
  divergence) — otherwise the compile-gated validator re-accepts regimes
  production declines, recreating exactly the backport-drift pattern D.2
  exists to fix.
- **Calibration coverage [cross-plan F9]:** thresholds calibrated on
  Phase 0's sweep cannot cover formula/callback modes (those contraction
  variants do not exist pre-B/C); schedule a "formula/callback slice after
  Batches B/C" re-calibration follow-on, mirroring the rewards-after-A
  slice.
- **Threshold calibration:** directly from Phase 0's sweep data (the metric
  must separate GAP from clean points with margin; no magic constants without
  data provenance).
- **Failure semantics:** decline → logged FD (INFO), identical to the
  existing per-theta declines — no behavior change for any θ outside the
  regime; gate tests assert both the decline and the log line.
- **Size:** small (days), but still gets its own mini-plan + adversarial
  review per standing practice.

## 4. Phase 2 — full MPFR adjoint (conditional: strong evidence only; expected never)

Recorded so the cost is never re-litigated from scratch [CCF §Q2]:
- **Self-consistency requires stages 0-2 all upgraded** — stage-0 snapshots
  (`s0`/`s1`) feed the cancellation-prone stage-2 quotient terms; upgrading
  only the backward accumulation would not fix the root cause.
- **Cost profile:** up to O(L) `mpfr_t` values (vs. the primal's O(n)
  economy), each needing `mpfr_init2`/`mpfr_clear`; ~14 arrays change type;
  the function family currently has zero NULL-checked allocations and no size
  guard — those guards are a hard prerequisite, not a nicety.
- **Consistency constraints:** trigger under the same condition and precision
  as the primal's own escalation for that θ (else the fwd/bwd mismatch the
  current gate exists to prevent is reintroduced).
- **Sequencing:** only after Batches 0, A, B (and C if scheduled) have landed
  on the shared core — one MPFR rewrite of the final skeleton, not three-plus
  rewrites of soon-to-change functions (master plan §14).
- **De-risk inputs already in hand by then:** the Phase-0 mpmath oracle (the
  numerics prototype) and E0-style L-statistics (tape lengths on production
  graphs — coordinate with Deferred 1's E0 measurement, which records L
  incidentally) sizing the memory cost with numbers, not adjectives.
- **Decision owner:** user, on Phase-1 evidence; this plan deliberately does
  not argue for Phase 2.

## 5. Interaction / conflict analysis

| Against | Interaction | Resolution |
|---|---|---|
| Master-plan Phase 1 cheap check | **Same work as Phase 0 here** — this plan is its design, not a duplicate. The identity is now two-directional [cross-plan F4]: the master plan's §15 line names this plan as design-of-record, so a master-only session cannot half-execute a genuinely-cheap re-run and orphan the rest | When Phase 0 runs, tick the master-plan Phase-1 item; one execution; the rewards/formula-callback follow-on slices stay tracked here |
| Batch 0 | Phase 1's insertion point should be the shared core (sequence after); Phase 0 is code-free and independent | Phase 0 anytime post-sign-off; Phase 1 post-Batch-0 |
| Batch A (rewards) | Same three functions; Phase 0's rewards slice runs *after* A lands (noted follow-on). Phase 2 (if ever) strictly after A. **Phase 1 and Batch A fire on the same "after Batch 0" trigger — coordinate explicitly if concurrent [cross-plan F3]** | Ordering actively managed, not "by construction"; rebase discipline per master §4 |
| Batch B/C | **Phase 1 (gate-consult insertion) AND Phase 2 both touch the shared core** [review F7 — the v1 "Phase 2 only" cell was wrong]; same trigger-collision caveat as Batch A | Phase 1 coordinated against in-flight A/B/C; Phase 2 after B (and C if scheduled) |
| Batch D.2 | D.2 duplicates the *existing* gate into a validator; unaffected by Phase 1's additive second metric (validator is compile-gated, zero production exposure) | None |
| Batches E/F/G/H, Deferred 2/3 | No overlap (sojourn/uniformization machinery). `ptd_dbg_tape_needs_mpfr`'s call in the sojourn function is **live as a conservatism knob on MPFR builds** (it causes real FD declines); it is "inert" only w.r.t. the fwd/bwd-consistency *rationale* (no MPFR primal exists for sojourn) and on non-MPFR builds [review F10 — wording corrected]. Phase 1 does not touch it [CCF Job B risks] | None |
| Deferred 1 | Shares the incidental L-statistics measurement (E0 there); AND if Phase 2 ever rewrites the shared core, Deferred-1's inner level built on that core inherits it — conditional sequencing coupling, mirrored from Deferred-1's matrix [cross-plan F12] | Reuse L data if available; revisit the coupling only if both units activate |
| `feedback_no_modify_existing` | Phase 0 code-free; Phase 1 = a new gate function PLUS a one-line consult insertion in the shared core — a small modification flagged for approval in its mini-plan [review F5]; Phase 2 explicitly flagged as the full-rewrite exception requiring its own approval | Phase 1 approval via its mini-plan; Phase 2 needs explicit user approval on this ground alone |
| `feedback_no_silent_fallbacks` | The entire unit exists to convert a (hypothetical) silent-wrong zone into loud decline+log | Aligned |

## 6. Risks specific to this plan

1. **The mpmath oracle could itself be wrong** — mitigation: benign-case
   cross-validation against both the shipped adjoint and `jax.jacobian`
   before trusting it anywhere disputed ("verify the verifier").
2. **Sweep coverage is finite** — a clean sweep proves the tested regimes,
   not universality; the report must state coverage honestly and the pin test
   keeps the tested regime locked.
3. **Phase-1 metric false positives** would needlessly push healthy θ to FD —
   calibration must bound the false-positive rate on the sweep's clean points
   (target: zero declines on currently-passing gate fixtures).
4. **Interaction ordering with Batch A's rewards:** the rewards slice of the
   sweep cannot run until A lands; if A slips, the sweep report must mark that
   slice pending rather than silently omitting it.
5. **mpmath cost envelope [review F11]:** dense LU at 200+ bits is
   pure-Python O(n³) per point × P Jacobian solves × the full sweep grid —
   set a fixture-size budget up front (n <= ~200 unless measured cheap;
   "coalescent-class" is otherwise unbounded).
6. **θ-grid resolution [review F11]:** "crossing the threshold continuously"
   needs a stated strategy — coarse sweep, then bisect the decline boundary
   to localize any GAP zone; without it a narrow silent-wrong band could sit
   between grid points.

## 7. Handoff

**State snapshot (2026-08-11):** master = `cadf1ca4` (local, unpushed); master
plan awaiting sign-off; the single-fixture shipped-adjoint check (done during
master-plan review) was clean — gate fired where corruption would begin; the
historical DR-A floor belongs to a discarded prototype, not the shipped code.
`ptd_dbg_tape_needs_mpfr` at src/c/phasic.c:10643, called at :10783/:10960/
:11221 + the sojourn function's call at :11529 (live conservatism knob on
MPFR builds; no fwd/bwd-consistency rationale — no MPFR primal exists for
sojourn). Unit = Phase 0 scheduled (master-plan Phase 1), everything else
conditional.

**Copy-paste prompt for the executing session:**
> Read `/Users/kmt/phasic/deferred-4-mpfr-conditioning-floor-plan.md`,
> `atlas/plan-feasibility-callback-and-conditioning-floor.md` (Job B), and
> master plan §14. Confirm the master plan was signed off; if not, stop and
> report. Execute Phase 0 exactly as §2 specifies, on a branch
> (`derisk/d4-mpfr-sweep`), experiments only, no `src/` changes; build and
> cross-validate the mpmath oracle before using it on disputed points. Write
> `b3-d4-sweep-findings.md`, adversarially review the findings (refute the
> oracle first), then present the go/park recommendation. Any regression pin
> or CLAUDE.md note is a PROPOSAL in the report — do not commit either
> without explicit approval. Do not start Phase 1 without a GAP outcome and
> explicit user approval.

## 8. Adversarial review record (2026-08-11)

One dedicated technical refuter (verdict: **SOUND-WITH-CORRECTIONS**; 2 MAJOR
F1-F2, 9 MINOR F3-F11) + one cross-plan conflict refuter (this plan:
**COMPATIBLE-WITH-CORRECTIONS**). All findings folded into v2:

- **F1** the v1 `PHASIC_B3_VALIDATORS` fallback claim RETRACTED — the
  validators expose only three aggregate bindings, no tape/command data.
  Replaced by (i) threshold-bisection recovery of the actual gate statistic
  and (ii) a labeled-approximation Python re-derivation, with true
  command-level measurement deferred to Phase 1's mini-plan (§2.1, §2.4).
- **F2** the `_dph`/log oracle mathematics spelled out (Stirling/binomial
  map, was_dph renorm quotient, log product formula); `serialize()` carries
  neither `weight_mode` nor `was_dph` — oracle builder reads them from live
  graph attributes; de-scoping fallback stated (§2.1).
- **F3 / cross-F9(i)** "from the oracle replay" provenance contradiction
  fixed (§2.4).
- **F4** Python-re-derived multiplier spread labeled a proxy; the real gate
  statistic comes from bisection (§2.1, §2.4).
- **F5** caller count corrected to four; Phase 1 acknowledged as a one-line
  modification of the shared core, approval via its mini-plan (§3, §5).
- **F6** validator-drift guard added (mirror the new gate into
  `ptd_moment0_grad_theta` or annotate) (§3).
- **F7 / cross-F3** the "Phase 2 only" matrix cell corrected; Phase-1 vs
  in-flight A/B/C trigger collision called out with a coordination rule
  (§3, §5).
- **F8 / cross-F4** cost-class change vs. the master plan's "[cheap check]"
  stated; master plan §15 amended to name this plan design-of-record (§2).
- **F9** build/env protocol added (HAVE_MPFR, env hygiene, INFO logging);
  pin skips on non-MPFR builds (§2.3, deliverables, §7 prompt).
- **F10** the Question scoped to the moments family (sojourn excluded per
  CCF); "inert sojourn call" wording corrected everywhere (live conservatism
  knob; rationale-inert only) (§2, §5, §7).
- **F11** risks added: mpmath cost envelope, θ-grid bisection strategy, and
  the s0/s1 malloc/partial-write indexing trap recorded in Phase 1's brief
  (§3, §6).
- **cross-F9(ii)** phantom "[CCF §Q2.5]" citation fixed (§2.4);
  **cross-F9(iii)** pin/CLAUDE.md-note deliverables reframed as proposals
  requiring approval; **cross-F12** the Deferred-1 conditional coupling
  mirrored into §5.

Survived attack (review-verified at source): the DR-A history claims
(prototype-vs-shipped provenance; the single-fixture clean check); the
Phase-2 cost profile (point-for-point vs CCF Job B §Q2, NULL-check absence
re-verified); the decision-tree architecture; the dense-oracle carve-out
under `feedback_avoid_matrix_exp` (experiment-only, resolvent-based,
DR-A-precedented); the master-plan §14/§15/risk-9/risk-17 citations; the
"thrice-recommended pin" provenance; the state snapshot.
