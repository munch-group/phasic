# Deferred 2 — daisy-chain intermediate-epoch exact gradient: de-risk & activation plan

**Status: v2 — adversarially reviewed 2026-08-11 (dedicated technical refuter:
SOUND-WITH-CORRECTIONS; cross-plan conflict refuter: the D2×D3 E4 gate was
CONFLICTING as written, now fixed); all findings folded in, see §9.
Planning-only — no code changes of any kind are authorized by this document.** The master plan
(`b3-exact-gradient-master-plan.md` §12) states Deferred 2 needs "its own
dedicated de-risking pass … **before** any implementation plan is even
written." This document is that de-risking pass's plan (plus activation gates
and a conditional sketch to be re-detailed from findings, per
`feedback_derisk_and_reevaluate`) — it is deliberately NOT an implementation
plan.

**Grounding.** Primary source: `atlas/plan-feasibility-pdf-gradient-daisy-chain.md`
(2026-08-05; cited as "PDF §n"), master plan §10 (Batch H) and §12, risk items
4, 8, and 15 [citation corrected per cross-plan review F10 — risk 6 is E/F
sequencing, unrelated]. [PDF §n] = read-from-source findings of that document;
[reasoned] = this plan's inference, to be confirmed in de-risk.

---

## 1. Goal and why it matters

**Goal (conditional):** exact theta-gradients for the *intermediate* epochs of
the daisy-chain (epoch / time-inhomogeneous) model — the `Graph.svgd()`
default for multi-epoch work, which today has **zero** exact-gradient coverage
(SVGD leaf 1; 100% FD, the exact defect class B3 exists to fix — atlas
headline finding). Batch H covers only the final epoch's contribution; this
unit completes the chain.

**Value test at activation time:** after Batch H lands, measure how much of
the total gradient error/instability on real epoch fits is attributable to the
remaining FD intermediate-epoch terms (using the E3 oracle below). If the
final-epoch exact term already removes the practical defect (plausible when
intermediate epochs are short/benign), this unit's priority drops and it can
stay parked — that is a legitimate outcome.

## 2. The three novel prerequisites (why this is not batch-sized)

1. **Granularity must be pinned θ-independently.** `granularity=0` (the
   current default in `daisy_chain_joint_probs` and `Graph.svgd`) resolves to
   an integer-cast, θ-dependent value re-derived per call (current C
   mechanics, synced to source per review F6: `max_rate` initialized to 512,
   `granularity = (int64_t)(max_rate·2.0)`, a never-binding 1000 floor, plus
   an overflow guard — `phasic.c:12648-12692`), making the embedded DTMC
   `P(θ) = I + Q(θ)/λ` change *identity* discontinuously as θ crosses
   thresholds. FD tolerates this (needs only eps-local smoothness, holds
   a.e.); an exact gradient is meaningless without a pinned λ [PDF §5.3].
   **Failure surfacing is NOT currently loud on the path that matters
   [review F1 — the v1 claim here was wrong]:** the C context-create does
   validate `rate/granularity <= 1.0001` per vertex and errors
   (`phasic.c:12703-12726`), and the C++ wrapper throws — but the daisy FFI
   handlers catch the exception per batch element, write a **NaN row**, and
   return `Success()` (`graph_builder_ffi.cpp:1828-1840`), and the sojourn
   handler — the shipped `final_read='sojourn'` DEFAULT that SVGD actually
   uses — does so with **no log line at all** (`:2065-2082`, `:2148`). Today
   a violated pinned λ would therefore surface as an unexplained NaN
   loss/gradient for that particle. **Building the loud path is new work E1
   must cost** — in-file precedent: the negative-rate `InvalidArgument`
   escalation pattern (`:1887-1896`, `:2137-2146`). The policy question is
   what to pin and how to fail (see E1).
   **Exact-mode consistency requirement (explicit, per review F2):** in
   exact mode the primal FORWARD value must be evaluated at the same pinned
   λ as the gradient — gradient-at-pinned-λ against forward-at-auto-λ is
   exactly the fwd/bwd-inconsistency defect class CLAUDE.md already flags at
   rate-blowup, and is forbidden here.
2. **Jacobian, not gradient, chained across every epoch.** Per-epoch state is
   the IPV vector (`n_ipv` = the JSP graph's `_ipv_target_indices` count), so
   the per-epoch primitive is the Jacobian of the collapsed output w.r.t. both
   `ipv_in` (n_ipv × n_ipv) and `theta_epoch` (n_ipv × n_params), composed
   epoch-by-epoch [PDF §5.4]. Batch H builds this composition skill for one
   hop; this unit compounds it across all hops. **Wiring surface, corrected
   per review F3:** what is reusable is the external `custom_vjp` *shape*
   (residual = `theta_flat`; cotangent-dot-product VJP) — but there are
   THREE separate FD `custom_vjp` sites, and the fwd AND **bwd** bodies must
   both be replaced (replacing only `_forward` still yields FD gradients):
   `daisy_chain_joint_probs`'s wrapper (`__init__.py:10340-10361`) and
   `_daisy_chain_svgd_model`'s no-exposure (`:4669-4692`) and exposure
   (`:4900-4932`) branches — the latter two wrapping `custom_vmap`-decorated
   cores that fuse particle batches into 2D FFI calls.
   **`Graph.svgd(epoch_starts=…)` and `Graph.epoch_model` reach ONLY the
   `_daisy_chain_svgd_model` sites** — the leaf-1 goal is delivered there,
   not in `daisy_chain_joint_probs` (whose own comment says the SVGD entry
   deliberately does not route through it). The exact-VJP × `custom_vmap`
   composition is its own de-risk item (E3(v)). (IPV remains an internal
   chain quantity — never an SVGD parameter, [[feedback_ipv_not_optimized]].)
3. **Backprop-through-time cost regime, unprecedented here.** Per-epoch
   `stop_probability(dt)` is ≈λ·dt applications of the same matrix (the
   ceil-like `while (time > ctx->time)` loop — see §4-E3 for the exact
   semantics; k in the thousands to tens of thousands). Naive reverse-mode is
   `O(k·n)` memory (infeasible at production n without `O(√k)`
   checkpointing at ~2× compute); forward-mode tangent propagation is
   history-free at `O(n·(n_params + n_ipv))` memory but scales compute with
   seed count. **No profiling data exists for either on realistic
   `n_ipv`/`k`/`n`** — an empirical question, not a judgment call
   [PDF §5.5; master-plan risk 8].

## 3. Activation gates

- **A1:** Batch H shipped, gate-verified, adversarially reviewed (hard
  prerequisite — supplies the Jacobian primitive shape and the one-hop
  composition pattern).
- **A2:** the §1 value test shows intermediate-epoch FD still causes a real
  accuracy/robustness problem after H.
- **A3:** the de-risk phase below has completed and its two open decisions
  (granularity policy; forward vs. checkpointed-reverse) are made and signed
  off.

## 4. De-risk phase (experiments; branch-only, no shipped-code changes)

- **E1 — granularity-pinning policy study.** On >=2 real epoch fixtures (the
  `test_lrt_at.py` coalescent epoch fixture; one larger model):
  1. Measure `max_exit_rate(θ)` variation across the prior's support and along
     actual SVGD particle trajectories (record per-iteration λ that auto mode
     would have chosen).
  2. Quantify forward-value error vs. pinned λ (convergence curves of
     `stop_probability` against a high-λ reference) to size the safety margin.
  3. Evaluate candidate policies:
     - (i) **explicit user-supplied granularity required for exact mode**
       (cleanest w.r.t. [[feedback_no_silent_fallbacks]]: no hidden magic;
       value-path default `granularity=0` untouched);
     - (ii) pin once at model construction from a probe θ + margin factor,
       raise loudly if any later θ violates the `<=1.0001` validation;
     - (iii) pin per-fit from the prior's upper support.
     Deliverable: a recommendation with failure semantics for each.
     **Constraint on the option space [review F4]:** "per-particle
     decline-to-FD" at the JAX level is FORECLOSED by B3's static-dispatch
     commitment — the D6 record (`b3-joint-index-plan.md:641-650`)
     establishes there is no way to both skip a branch under `vmap` and fall
     back per-particle, and the user already chose **raise** for the
     analogous late-θ failure in the joint-index redesign (`:689-705`).
     Feasible semantics: (a) raise (precedented), (b) a host-side FD
     fallback inside the bwd callback (new, unprecedented machinery — cost
     it explicitly if proposed), (c) per-call dispatch accepting the known
     vmap double-cost. The λ-pinning POLICY itself is a single decision
     shared with Deferred 3, recorded in the E4 interface note
     [cross-plan F2].
  4. Confirm [reasoned]: pinning affects only the *exact-grad opt-in* path;
     the existing FD/value path keeps `granularity=0` semantics byte-identical.
- **E2 — cost-model measurement.** Instrument (out-of-band, no src changes)
  real epoch fixtures for `n_ipv`, `k`, `n`, edges; prototype both modes in
  the E3 JAX reference; produce a decision table: forward-mode vs.
  checkpointed-reverse crossover as a function of `(n_params + n_ipv)` vs.
  `√k`-checkpoint overhead. Include at least one two-locus-scale case where
  `n_ipv` is large (tens-hundreds), since that regime is where forward-mode's
  seed count hurts [PDF §5.5].
- **E3 — end-to-end differentiable oracle.** A pure-JAX reference of the full
  daisy chain on small models: dense `P(θ)` per epoch, the special
  **instantaneous t=0 IPV redistribution step** (the primal's real initial
  condition — the exact step `ptd_graph_pdf_with_gradient` got wrong, PDF
  §1/§2.5), the step loop matching the C stepping semantics EXACTLY — the C
  loop is `while (time > ctx->time) step()` (ceil-like, with creation-time
  probe-step mechanics), NOT `k = floor(λ·dt)`; take it from source
  (`phasiccpp.h:1519-1541`, `phasic.c:12734-12749`), never from this plan's
  shorthand [review F7] — and collapse/projection between epochs modeled
  *exactly* as the FFI loop does (update_ipv → update_weights →
  stop_probability → collapse → project [PDF §5.1/§5.4]). Gates:
  1. value parity vs. production `daisy_chain_joint_probs` at matched pinned
     granularity (target ~1e-12; any mismatch means the reference mis-models
     the collapse/projection step — fix before proceeding). When the
     production exact mode is later implemented, its FORWARD must run at
     this same pinned λ (the §2.1 consistency requirement);
  2. `jax.grad` of the reference == FD of the reference (benign θ);
  3. FD-vs-oracle divergence demonstrated at mixed-scale θ (motivating case,
     mirroring the pinned FD-defect test's regime);
  4. the reference machinery is SHARED with Deferred 3's E1/E2 JAX reference
     — one implementation, two scopes [cross-plan F11];
  5. exact-VJP × `custom_vmap` composition probe: differentiate the
     reference under the same `custom_vmap` batching pattern the production
     `_daisy_chain_svgd_model` sites use (§2.2) — the actual hard JAX
     problem here, cousin of the Batch-F lesson [review F3].
  The collapse/projection Jacobian is a fixed linear selection/aggregation
  [review-verified at source: pair-sum collapse + index selection, no
  renormalization (`graph_builder_ffi.cpp:1853-1881`, `:2068-2077`);
  `update_ipv` is a direct write (`phasic.c:5909-5913`) — E3 still verifies
  by construction rather than assuming].
- **E4 — shared-primitive + λ-policy decision with Deferred 3.** The
  per-epoch tangent stepper (forward-mode propagation of d(state)/dθ and
  d(state)/d(ipv) through the DTMC power iteration, including the special
  first step) is *plausibly* the same primitive Deferred 3's route (i) needs
  at single-epoch scope — a working hypothesis, not established [review F5]:
  D3 needs the absorbed-mass pmf/cdf harvest readout with θ-seeds only; D2
  needs the full state-vector readout with θ+IPV seeds; and if D3's de-risk
  selects route (ii) (Poisson mixture), the sharing premise lapses entirely
  and E4 reduces to a written no-sharing note [cross-plan F2].
  **Ownership rule [cross-plan F1 — the v1 "joint gate" deadlocked under
  single-unit activation]:** E4 is owned by whichever unit activates FIRST,
  and the interface note is written against BOTH plans' requirements
  regardless of the other's activation status — a stepper built by one unit
  must either satisfy or explicitly decline (user-signed) the parked unit's
  seed/state/readout requirements. The note also records the SINGLE
  λ-pinning policy decision both units share [cross-plan F2]. Deliverable:
  a one-page interface note (mirrored in Deferred 3's plan §4-E4).

**Re-evaluation checkpoint:** after E1-E4, write the real implementation plan
from findings; adversarial review; sign-off. The sketch below is a shape
forecast only.

## 5. Conditional implementation sketch (post-de-risk, re-detail first)

- **B1 — C tangent stepper (possibly shared with Deferred 3).** A new
  context/step pair propagating tangent state alongside `prob` for a set of
  seed directions (θ-columns + IPV-columns), preserving the instantaneous
  first step; linear weight mode only at first — `log` is already rejected by
  the daisy path today [PDF §5.1], and `callback`/`formula` must be rejected
  loudly, not linearized. Additive: new functions only, existing
  `ptd_probability_distribution_*` untouched ([[feedback_no_modify_existing]]).
  **Discrete scoping [cross-plan F13]:** the chain inherits the sojourn
  adjoint's documented `was_dph=True` exclusion via Batch H
  (`Graph.discretize()` chains out of scope; native-DPH status follows
  whatever scope H ships) — stated here so the exclusion is explicit, not
  inherited silently.
- **B2 — per-epoch Jacobian assembly + chain composition.** Compose per-epoch
  (n_ipv×n_ipv, n_ipv×n_params) blocks across the epoch boundary exactly as
  Batch H established for the final hop; accumulate a VJP for the full chain.
  If E2 picks checkpointed-reverse instead, B1/B2 restructure accordingly —
  which is precisely why implementation planning waits for E2.
- **B3 — wiring.** By this unit's own sequencing, Batches H and G will
  already have shipped an exact final-epoch branch and an exact-grad kwarg on
  the `_daisy_chain_svgd_model` sites — so B3 is an **extension of shipped
  H/G code, i.e. an explicit modification requiring user approval under
  `feedback_no_modify_existing`** (or an additive restructure preserving the
  H-era branch with the off-path byte-identical, per B4's gate); the kwarg's
  name/default are owned by H's and G's implementation plans, not re-decided
  here [cross-plan F7]. Replace the fwd AND bwd bodies at the `custom_vjp`
  sites that actually carry the SVGD chain (§2.2 — the
  `_daisy_chain_svgd_model` sites, plus `daisy_chain_joint_probs`'s wrapper
  for the public path), behind the existing external VJP shape; failure
  semantics per the E1 decision (loud); **construction-time static dispatch,
  no per-call `lax.cond`** — under `vmap(grad(...))` a batched predicate
  computes both branches (the D6/Batch-F lesson, empirically confirmed in
  `experiments/dr_lax_cond_vmap_derisk.py`; master plan §8, §15 Phase 1b).
- **B4 — gates.** E3's oracle promoted to pytest on small models; exact==FD on
  benign θ; exact correct (vs. oracle) at the mixed-scale regime where FD is
  pinned-wrong; end-to-end `Graph.svgd`/`epoch_model` regression
  (bit-identical when `exact_grad` is off); granularity-violation failure
  semantics tested.

**Sizing honesty:** multi-week even after de-risk; per-batch adversarial
review mandatory (this codebase's track record: the real defects were found
by review, not by implementations' own tests).

## 6. Interaction / conflict analysis

| Against | Interaction | Resolution |
|---|---|---|
| Batch H | Hard prerequisite (A1). H's primitive shape (extend `ptd_sojourn_grad_theta_subset` vs. new function) is explicitly unresolved in the master plan (§10); this unit consumes whatever H ships | Re-verify interface fit when H's design is drafted — same instruction the master plan already gives for E/H |
| Batch E / F | F's static-dispatch lesson is a design constraint on B3 (adopted). No line overlap: E/F live in `pmf_from_graph_joint_index`'s wiring; this unit lives in `daisy_chain_joint_probs`/FFI | None beyond adopting the pattern |
| Batch G (leaf 1) | G's daisy plumbing is blocked on H (master plan §9); when this unit lands, G's leaf-1 kwarg gains full-chain meaning. No double-scheduling: G plumbs, this unit implements | Sequence: H → G(leaf 1, final-epoch-exact) → this unit upgrades the same kwarg's coverage |
| Batches 0/A/B/C | No overlap — different C machinery entirely (uniformization stepping vs. elimination-tape replay) [PDF §8(a),(b)]. The Batch-0 skeleton is very likely not a candidate home for the stepper [reasoned — PDF §8(b) hedges "may not even be a candidate"; carry its requested two-line note when Batch B is scoped] | None |
| Batch D / Deferred 4 | No overlap (SVGD-plumbing Tier 1 and the MPFR gate live in the moments family) [added for completeness, review F9] | None |
| Deferred 3 | Shared tangent-stepper primitive (E4). Also shares the granularity/λ-pinning *decision* — make it once, consistently | Joint design gate E4; do not implement two steppers |
| Deferred 1 | Orthogonal — JSP/daisy graphs are monolithic elimination targets throughout [PDF §8(d)] | None |
| `feedback_avoid_matrix_exp` | Differentiates the *actual* power-iteration primal; no expm/Krylov introduced | Respected by design |
| `feedback_ipv_not_optimized` | IPV Jacobians are internal chain links; SVGD's parameter surface unchanged | Respected |
| `feedback_no_modify_existing` / `no_silent_fallbacks` | B1's stepper is additive; **B3 extends shipped H/G code — an explicit modification requiring user approval** [cross-plan F7]; failure semantics loud per E1 | B1 respected by construction; B3 flagged for approval at activation; E1 decides the exact failure mode |

## 7. Risks specific to this plan

1. **E3 value-parity may be hard to hit** if the FFI epoch loop has
   undocumented details (ordering, normalization, exposure handling) — that is
   the point of gating on parity first; a reference that doesn't match the
   primal to ~1e-12 disqualifies itself as an oracle.
2. **Pinned-λ accuracy trade:** a λ safely above the whole run's max rate
   inflates k (cost) and changes the value slightly vs. auto mode; E1's
   convergence curves must show the value error at the chosen margin is below
   the fit's statistical noise, or exact mode's value would differ from FD
   mode's value (a cross-mode divergence needing explicit documentation).
3. **Exposure interaction:** the daisy path bakes exposure handling
   internally; the E3 reference initially models the no-exposure case — the
   exposure-bearing variant needs its own parity check before B3 wires it
   (mirrors master-plan risk 4's shape).
4. **Step-count discretization of epoch boundaries:** the C stepping loop is
   `while (time > ctx->time) step()` (ceil-like — NOT `k = floor(λ·dt)`;
   sync any oracle to source, `phasiccpp.h:1519-1541` [review F7]); step
   counts are integer-quantized in λ·dt; d/dθ is unaffected (dt is
   θ-independent) [reasoned — E3 should confirm no hidden θ-dependence via
   `t_eval` resolution; review verified t_eval is probe-θ-resolved at
   construction and baked as a float].

## 8. Handoff

**State snapshot (2026-08-11):** master = `cadf1ca4` (local, unpushed);
master plan awaiting sign-off; Batch H not started (Phase 1 item);
`final_read='sojourn'` is the shipped default; FD mixed-scale defect pinned in
`test_fd_gradient_mixed_scale.py` (commit `9e7d2132`). This unit is parked
pending gates A1-A3.

**Copy-paste prompt for the executing session** (rewritten per review F8 —
the v1 prompt hard-stopped ALL experiments on Batch H, though E1/E2/E4 do not
depend on it):
> Read `/Users/kmt/phasic/deferred-2-daisy-intermediate-epoch-plan.md` and
> `/Users/kmt/phasic/atlas/plan-feasibility-pdf-gradient-daisy-chain.md` in
> full. Confirm the master plan was signed off and the user has explicitly
> authorized this unit's de-risk; if not, stop and report. E1, E2, and E4 may
> run before Batch H ships (they do not depend on H's primitive); the §1
> value test and gate A2 require H; implementation planning requires A1-A3.
> Run the de-risk on a branch (`derisk/daisy-intermediate-exact`),
> experiments only, no `src/` changes. E4 is a JOINT deliverable with
> `deferred-3-pdf-gradient-revival-plan.md` — owned by whichever unit
> activates first, written against BOTH plans' requirements. Then write the
> implementation plan from findings, adversarially review it, and present
> for sign-off before any code.

## 9. Adversarial review record (2026-08-11)

One dedicated technical refuter (verdict: **SOUND-WITH-CORRECTIONS**; 4 MAJOR
F1-F4, 5 MINOR F5-F9) + one cross-plan conflict refuter (D2×D3 judged
CONFLICTING as written on the E4 gate — CRITICAL cross-F1 — now fixed). All
findings folded into v2:

- **F1** the "fails loudly" claim was WRONG on the path that matters: daisy
  FFI handlers convert the C granularity-violation error into a NaN row +
  `Success()`, unlogged on the default sojourn handler. §2.1 corrected;
  building the loud path is now costed E1 work (precedent: the negative-rate
  `InvalidArgument` pattern). Also recorded in the master plan's §16b ledger
  as an independent observability gap.
- **F2** the forward-at-pinned-λ consistency requirement made explicit
  (§2.1, E3 gate 1).
- **F3** wiring surface corrected: three `custom_vjp` sites + two
  `custom_vmap` cores; SVGD leaf 1 reaches only the `_daisy_chain_svgd_model`
  sites; fwd AND bwd bodies replaced; new E3(v) composition probe (§2.2,
  §4-E3, §5-B3).
- **F4** E1's failure-semantics option space constrained by the D6 record
  (no per-particle JAX-level FD fallback under static dispatch; user
  precedent = raise) (§4-E1).
- **F5 / cross-F1 / cross-F2** E4 rewritten: same-primitive is a working
  hypothesis (different readouts/seed sets); ownership = whichever unit
  activates first, note written against both plans' requirements regardless;
  single shared λ-policy decision; route-(ii) lapse clause (§4-E4).
- **F6** auto-granularity mechanics synced to current C (§2.1).
- **F7** stepping semantics corrected (`while (time > ctx->time)`, not
  floor) (§4-E3, §7 risk 4).
- **F8** handoff prompt no longer hard-stops E1/E2/E4 on Batch H (§8).
- **F9** Batch-0-skeleton claim re-hedged as [reasoned]; Batch D/Deferred-4
  matrix rows added (§6).
- **cross-F7** B3 acknowledged as a modification of shipped H/G code
  requiring approval; kwarg name/default owned by H/G (§5-B3, §6).
- **cross-F10** grounding citation fixed (risks 4/8/15, not 6); "§5-E4" →
  "§4-E4".
- **cross-F13** explicit `was_dph` scoping statement added (§5-B1).

Survived attack (review-verified at source): granularity's end-to-end
kwarg→JSON→FFI→C flow; collapse/projection linearity; t_eval θ-independence;
the n_ipv/ipv_work Jacobian-shape claims; H/G sequencing vs master §9/§10/§15;
exposure deferral validity on the chosen fixture; the handoff snapshot
(including that `final_read='sojourn'` IS the shipped default — the reviewer
confirmed commit `9a80ac45` is an ancestor of master).
