# B3 exact-gradient master plan — sequencing across all remaining paths

**Status: REVISED post-review — three refute-tasked adversarial passes
completed and folded in (§17 items 1-2); deferred-unit design-of-record
amendments + §16b ledger added 2026-08-11 after the four deferred-unit plans'
own adversarial reviews. AWAITING USER SIGN-OFF. No implementation may start
on any batch until the user explicitly signs off this document.**

**Scope.** Every remaining path that could plausibly carry an exact
theta-adjoint gradient, EXCEPT `bffg.py` (explicitly out of scope per user
instruction) and, explicitly rather than by omission, the two separate
FD-only paths `Graph.moments_from_graph` (its exact gradient — D.1 fixes
only its vmap crash) and `method_of_moments` (scipy-internal FD) — tracked
in §16b, not inventoried here (amendment 2026-08-11, cross-plan review F8). This document supersedes ad-hoc batch-by-batch planning: it is
the single source of truth for what to build, in what order, and why — so
that no batch is designed in ignorance of another's constraints, the way the
D6 joint-index redesign was nearly built on the false premise that
`Graph.svgd()` reaches it (it doesn't — see `project_exact_fd_atlas` memory).

**Grounding.** Every claim below traces to one of six feasibility documents in
`/Users/kmt/phasic/atlas/` (`plan-feasibility-rewards-support.md`,
`-formula-mode.md`, `-svgd-plumbing.md`, `-hierarchical-scc.md`,
`-pdf-gradient-daisy-chain.md`, `-callback-and-conditioning-floor.md`), all
produced 2026-08-05 by independent agents each instructed to re-derive claims
from a fresh read of current source rather than trust CLAUDE.md/memory/each
other, plus the broader `exact-fd-atlas-SUMMARY.md` (2026-08-04). Every one of
the six re-verified its own starting hypotheses against source and flagged
corrections where the hypotheses (drawn from CLAUDE.md/memory) were wrong or
imprecise — those corrections are called out explicitly below wherever they
change this plan's conclusions.

---

## 1. Batch inventory (13 units of work)

| # | Batch | Size | Status |
|---|---|---|---|
| 0 | Reverse-tape skeleton extraction (prerequisite refactor) | Medium | Not started |
| A | Rewards support in the moments adjoint | Small-Medium | Not started |
| B | `weight_mode='formula'` exact gradient | Medium | Not started |
| C | `weight_mode='callback'` exact gradient (Job A only) | Small-Medium | Not started |
| D | SVGD plumbing, Tier 1 (mechanical, no dependencies) | Small | Not started |
| E | Joint-index baked/dedup-mode exact gradient (scatter-add) | Small-Medium | Not started |
| F | D6 `lax.cond`/`vmap` redesign for joint-index | Small-Medium | Planned + de-risked, **not implemented** |
| G | SVGD plumbing, Tier 3 (blocked leaves) | Trivial once unblocked | Blocked on A, H |
| H | Daisy-chain final-epoch exact gradient | Small-Medium | Not started |
| — | *Deferred 1:* Hierarchical/SCC two-level adjoint | Large | Deferred |
| — | *Deferred 2:* Daisy-chain intermediate-epoch exact gradient | Large (multi-week) | Deferred, own initiative |
| — | *Deferred 3:* `ptd_graph_pdf_with_gradient` revival | Large | Deferred — do not trust/reuse current code without full re-verification |
| — | *Deferred 4:* MPFR conditioning-floor full adjoint | Large | Full adjoint deferred; a cheap decline-gate check is scheduled in Phase 1 (§14, §15) |

Sections 2-14 detail each. Section 15 gives the consolidated dependency graph
and phased sequence. Section 16 is the risk register. Section 17 states what
"done" looks like for this planning document itself.

---

## 2. Batch 0 — reverse-tape skeleton extraction (the shared prerequisite)

**Why this exists as its own batch, not folded into whichever lands first:**
four independent investigations (rewards-support, formula-mode,
hierarchical-scc, callback-mode) converged unprompted on the same conclusion:
`ptd_moments_grad_theta` (`src/c/phasic.c:10738-10881`), `_dph`
(`:11142-11338`), `_log` (`:10917-11063`) share a genuinely
byte-for-byte-identical ~110-line stage-0/1/2 core, confirmed by direct
side-by-side reading (formula-mode doc §4 table):

| Block | linear | log | dph |
|---|---|---|---|
| Stage-0 forward tape walk | 10766-10779 | 10946-10959 | 11207-11220 |
| MPFR gate call | 10783-10788 | 10960-10965 | 11221-11227 |
| Forward moment chain (`a_0=ones` seed) | 10790-10801 | 10966-10977 | 11228-11239 |
| Per-`outk` reverse chain | 10811-10829 | 10986-11004 | 11248-11266 |
| Stage-2 param-tape reverse | 10830-10847 | 11005-11021 | 11267-11283 |

Every one of Batches A, B, C, and *Deferred 1* would otherwise touch this
same block a second, third, fourth, or fifth time — reproducing a pattern
this codebase has already hit twice (the `coefficients_length==0` guard and
the MPFR gate were each fixed in one function and never backported to
`ptd_moment0_grad_theta`, confirmed still missing there — see Batch D.2).

**What's NOT identical (must survive extraction, not be flattened away):**
1. Signatures differ — `ptd_moments_grad_theta` takes no `theta`/`theta_len`
   at all; `_log`/`_dph` add both. A shared helper must accept `theta`
   unconditionally (linear ignores it).
2. `_dph` has an entire extra **pre-pass** (`Sv`/`SigmaCv` precompute + mixed
   constant/parameterized decline, `11149-11179`) and an extra
   **post-pass** (`ptd_dph_correct_discrete_moment_grad`, `11094-11119`,
   invoked `11326-11328`) — neither belongs inside the shared core; both stay
   as pre/post wrappers `_dph` calls around it.
3. Contraction guard conditions differ: linear only skips
   `coefficients_length==0` (`10868`); log/dph additionally skip the
   starting-vertex edge (`11046`, `11313`).
4. Linear has **no `was_dph` guard of its own** — today's safety rests
   entirely on Python-side routing (`pmf_and_moments_from_graph`'s
   `_effective_discrete` dispatch) never calling it on a `was_dph` graph. Cheap
   to close opportunistically during extraction (an explicit
   `assert !graph->was_dph` in the linear contraction case), not required.

**Design (open choice, not yet decided — flagged explicitly, not assumed
settled):** the formula-mode feasibility document's own risk list states
this interface design "is not yet decided... a design call for whoever
implements it, not resolved here." The candidate this plan leans toward is
an `enum`-dispatched `switch` inside one shared contraction function
(simpler than function-pointer/`void*ctx` plumbing; matches this codebase's
existing dispatch idiom, e.g. `ptd_pdf`'s `discrete`/`is_discrete` switch),
but whoever implements Batch 0 should treat this as a real decision to make
at implementation time, not a foregone conclusion — in particular Batches B
and C (§4, §5) each add a *new* contraction variant on top of whatever shape
is chosen here, so the choice should be validated against both of their
needs before committing. Keep `_dph`'s pre/post passes as callers-of, not
part of, the shared core.

**Verification:** re-run `dr_moments_jac_gate.py`, `dr_dph_moments_jac_gate.py`,
`dr_log_mode_moments_jac_gate.py` as a byte-identical-output check before and
after. **Baseline confirmed fresh this session (2026-08-05, no code changes):
ALL THREE PASS** — this is the diff target for the refactor.

**Not a hard technical blocker for anything** (a 4th/5th near-duplicate copy
*could* be written without it, the way `_dph`/`_log` were each built from
`ptd_moments_grad_theta`) — but strongly recommended first given the
convergent, independently-discovered evidence above. Zero user-facing change;
this batch is pure internal restructuring, gated purely by the three existing
jac-gates staying value-identical.

---

## 3. Batch A — rewards support in the moments adjoint

> **MERGED 2026-08-14 (`798ddcaa`, squash of `b3/batchA-rewards`).** Shipped
> as planned with two deviations recorded in `b3-batchA-plan.md`'s dated
> amendments: (1) dph/discrete rewards were REFUTED by direct computation
> at plan review (not implemented-then-gated) — the c2d correction requires
> U/P commutation, broken by reward scaling; permanent static decline with
> an INFO log; (2) the svgd 1-D-rewards opt-out was BUNDLED in by user
> decision (R29 1-D arm relaxed; Batch G leaf 3 delivered here, G.2
> shrinks to the 2-D/multivariate leaf). Gates: rewardless byte-identity
> BITWISE cross-install; rewards vs primal-FD 1e-11..2e-10 incl. extreme
> scales; all-ones == rewardless bitwise; log-mode leg 1.4e-10; G3
> 1957/0/84/24 = ledger+6; two G4 refuters SOUND-WITH-CORRECTIONS (no
> shipped-code defect, corrections folded `1ee12b3f`).

**Headline correction to the task's own starting hypothesis (verified
empirically against the live package, not assumed):** the fix is **not**
"seed `a_0` with `rewards`." `Graph.moments(power, rewards)`
(`api/cpp/phasiccpp.h:650-671`) re-multiplies by the *original* `rewards`
vector at **every** stage transition (`a_j = replay(a_{j-1} .* rewards)`), not
just at `a_0`. A naive seed-only patch reproduces the correct reward-weighted
**first** moment but is **silently wrong** for the second and higher moments
— confirmed by direct execution: naive gives `2.5`, ground truth is `3.5`,
for a concrete 4-vertex test graph.

**The correct fix — exactly two lines changed per function (6 lines total):**
- `for (v) out[v]=seed[v];` → `out[v] = seed[v] * (rewards ? rewards[v] : 1.0);`
  (linear `:10795`, dph `:11233`, log `:10971`)
- `for (v) bar_out[v]=adj[v];` → `bar_out[v] = adj[v] * (rewards ? rewards[v] : 1.0);`
  (linear `:10827`, dph `:11264`, log `:11002`) — the reverse-mode VJP of an
  elementwise-scale map is multiplication by the same scale; `rewards` is
  theta-independent so no new `dm[]`/theta term is needed here.
- **Nothing else changes**: `na/nb/nm` extraction, the MPFR gate (already
  reward-blind identically to the primal's own condition-number pre-scan —
  not a new gap), stage-2 contraction, and the `was_dph`/log correction
  branches are all unaffected — verified by reading every line that touches
  the tape/theta pipeline in all three functions.

**Threading:** new `(rewards, rewards_len)` parameter on all three C
signatures (decline `-1` if `rewards_len != 0 && rewards_len !=
graph->vertices_length`, matching the existing `theta_len` decline
convention already used by `_dph`/`_log`), the corresponding C++ header
methods (`api/cpp/phasiccpp.h:560-599`) and pybind `.def()` sites
(`src/cpp/phasic_pybind.cpp:1915-1930`). Python: `_exact_moments_jac_np`/
`_one` needs `rewards` threaded across the `pure_callback` boundary as a
genuine per-call array (not closed over at construction) — directly
precedented by `pmf_from_graph_joint_index`'s existing
`_exact_sojourn_jac_np(theta_np, vertex_indices_np)` pattern. `model_bwd`'s
`_rewards_provided` guard (`__init__.py:7559-7567`) changes from
unconditional decline to a real dispatch.

**Free side effect:** `pmf_and_moments_from_graph_multivariate` needs **no
code change of its own** — it already calls `model_1d(..., rewards=reward_j)`
per feature with a concrete non-`None` reward slice; fixing the 1D case
automatically un-breaks the multivariate wrapper's exact-path reachability.

**Open risk (flagged, not resolved):** whether
`ptd_dph_correct_discrete_moment_grad`'s combinatorial continuous→discrete
correction remains valid unchanged under reward-weighting is
plausible-by-analogy (the value-level analog's validity argument is
graph/vector-independent) but **not independently re-derived or numerically
gated** in the feasibility pass. **Action: a dedicated `dr_*.py` gate
(reward-weighted discrete moments, exact vs FD) must be written and pass
before Batch A's `_dph` support ships** — this is exactly the shape of defect
(rewards silently wrong for `nr_moments>=2`) this batch's own headline finding
already demonstrates is easy to get subtly wrong.

**Sequencing constraint (hard, line-level, not soft):** this batch's two
changed lines are inside Batch 0's exact extraction target, in all three
functions. **Must not run blind-parallel with Batch 0.** Land Batch 0 first
(preferred — Batch A then touches one helper instead of three near-duplicate
call sites) or, if Batch A must go first, design the `× rewards` hook into
Batch 0's shared skeleton from day one.

---

## 4. Batch B — `weight_mode='formula'` exact gradient

> **MERGED 2026-08-14 (`c6cc38b9`, squash of `b3/batchB-formula`; G4 fold
> `d6bb0c99`).** Shipped as planned in `b3-batchB-plan.md` v1+v2 (the v2
> amendment is the binding record — its plan review found two
> load-bearing corrections BEFORE any code: the theta-dimension
> decoupling contract, resolved as ALIGNED-graphs scope with a static
> decline for the lazy-decoupled class [full decoupled support =
> ledgered follow-up], and the POW rule corrected to the two-term
> adjoint [the factored form below is REFUTED at a=0 — kept here
> unedited as the historical design note]). The Wengert-list reverse
> pass with zero-propagate through comparisons superseded the
> "stop early" option below (unguarded tapes CAN reach the C executor
> via _set_weight_tape/from_serialized). Evidence: formula ==
> linear-exact BITWISE at mixed scale; two G4 refuters SOUND /
> SOUND-WITH-CORRECTIONS (independent oracle 24/24 at ≤3.3e-15; zero
> memory drift); G3 1975/0/84/24 = ledger+12. was_dph exclusion
> CONFIRMED load-bearing by direct repro (silently computes), per this
> section's own demand below.

**Confirmed architecture: two separate VMs, not one op set to extend.** The
elimination tape (`P/PP/INV/OM/DIV/ZERO/NEW_ADD`, 7 types,
`src/c/phasic.c:3638-3640`) only ever reads an edge's *current weight* as an
opaque free variable — proven already by the fact that `_log`/`_dph` reuse
stage-0/1/2 verbatim regardless of how the weight was computed. The
weight-formula tape (`ptd_weight_tape`, `PTD_WF_PUSH_THETA=0..SELECT=22`,
`phasic.c:5085-5093`, mirroring `weight_formula.py`'s `OPCODES` dict
integer-for-integer) computes *that weight* from `(theta, e->coefficients)`
one level up, *before* the elimination tape runs at all. **No new elimination
tape op types are needed; no changes to `off->input_specs` are needed** — it
already resolves `(sp.v, sp.e)` → `struct ptd_edge*`, from which
`e->coefficients` is directly available, exactly what a formula-tape autodiff
pass needs as its `c<j>` inputs.

**What's actually needed: a new, independent autodiff pass over
`ptd_weight_tape`**, following the same snapshot-then-replay idiom the
elimination tape's own `PP`/`DIVIDE` reverse cases already use
(`s0[i]/s1[i]` at `10772/10775`, reversed at `10840/10843`). Differentiation
rules for the 12 real (non-comparison/boolean) opcodes:
`ADD/SUB` (pass-through ±1), `MUL` (product rule, needs the other operand's
snapshotted forward value), `DIV` (quotient rule, both operands snapshotted),
`POW(a,b)` (general two-sided rule — **cannot** assume a theta-independent
exponent; `t0**t1` is syntactically legal), `NEG`, `EXP/LOG/SQRT/LOGISTIC`
(standard 1-arg chain rules). The remaining 11 opcodes
(`EQ/NE/LT/GT/LE/GE/AND/OR/NOT/SELECT`'s condition) contribute **exactly
zero** gradient and need **no** sub-adjoint propagation into their operand
subtrees at all — guaranteed by the compiler's own static
theta-independence guard (it statically rejects any formula putting `theta`
into a comparison/boolean/select-condition operand at assignment time). This
is the single fact that keeps this batch's math tractable: no
branch/subgradient handling is ever needed.

**Where it runs:** once per tape-input edge, before the per-`outk` loop
(mirrors `_dph`'s one-time `Sv`/`SigmaCv` precompute pattern) — `dw_e/dtheta`
doesn't depend on `outk`. Store as an `ni × P` array indexed identically to
the contraction loop's existing `k`. Contraction step is then **identical
shape** to linear's/dph's non-`was_dph` contraction, substituting the new
per-edge derivative vector for `e->coefficients[j]`.

**Design choice made explicitly (not deferred as ambiguous):**
differentiate the **full** `graph->weight_tape` directly via
`e->coefficients`, not the pre-specialized `wf_residuals[]` (which uses a
different iteration order that doesn't line up with `input_specs` and would
need a nontrivial reverse-mapping) — correctness-first; treat the residual-
tape route as a future speed optimization only if profiling shows it's
needed.

**Decline conditions:** `was_dph` almost certainly excluded, by analogy to
`_log`'s own caution — **needs its own direct repro before being asserted**,
not assumed by analogy (this is exactly the class of assumption CLAUDE.md's
B3 section flags adversarial review exists to catch). MPFR gate inherited
unchanged (orthogonal — elimination-tape conditioning, not formula-tape
numerics). The formula tape's own non-finite outputs (log of non-positive,
etc.) need **no new decline logic** — they fall through to the existing
final `isfinite` sweep over `J_out`, as long as the new contraction doesn't
early-return before it.

**Dependency:** built on Batch 0's refactored core as a 4th switch case (not
a hard blocker, but CLAUDE.md + three independent investigations all
recommend Batch 0 land first). No conflict with Batch A directly (different
lines — Batch A touches the seed/adjoint-scale lines, Batch B adds a new
contraction case), but both are consumers of the same shared core, so
whichever of A/B lands second on top of Batch 0 should rebase cleanly rather
than reintroduce a duplicate.

**Risks flagged:** `POW`'s general rule needs its own dedicated de-risk
script (`jax.jacobian` cross-check) before trusting it. No existing gate
quantifies FD's mixed-scale unreliability specifically for formula mode yet
(`test_weight_formula_svgd.py` currently exercises only FD) — a concrete
repro mirroring `dr_log_mode_moments_jac_gate.py`'s mixed-scale cases should
be an early step of implementation, strengthening the motivating case.

---

## 5. Batch C — `weight_mode='callback'` exact gradient (Job A only)

> **MERGED 2026-08-14 (`35a17364`, squash of `b3/batchC-callback`; G4 fold
> `88e5cc68`) — PHASE 3 COMPLETE with this merge.** Shipped per
> `b3-batchC-plan.md` v1+v2: option (b) below (the pre-contraction binp
> exit, `PTD_B3_BINP_EXIT`, the shared core's 5th consumer) + Python
> matmul contraction against the construction-jitted jax.grad of the
> callback. The v2 review REFUTED the v1 theta-dim restriction —
> update_weights(callback=) skips the length check BY DESIGN
> (phasiccpp.cpp:1879-1884), so decoupled graphs are SUPPORTED, not
> declined. Non-JAX-native callbacks = the permanent FD boundary
> (option 2 below, adopted); the analytic-derivative-callback opt-in
> (option 1) is ledgered in §16b. The "not verified by execution" risk
> below was discharged at plan review (jit(grad) probed end-to-end
> through a custom_vjp mimic under all five transform compositions);
> the coefficients_length risk is inert for callback (the exit exports
> full per-input coefficient vectors; no fixed-P read). The joint-index
> callback path remains its own follow-up (unchanged below).

**Current state: the callback receives concrete numpy, never a JAX tracer,
by construction — `_apply_weight_callback` (`__init__.py:735-807`) does
`float(callback(theta, coeffs))`, a hard cast that destroys any JAX
autodiff structure regardless of how the callback is internally written**
(confirmed across all four call sites: `pmf_from_graph`, `pmf_and_moments_
from_graph`, its `cdf_zero` companion, `pmf_from_graph_joint_index`).

**Is this fixable, or a real mathematical wall? Fixable for JAX-native
callbacks — and this is sound chain-rule math, not a hopeful analogy.**
Stage 2 of `ptd_moments_grad_theta` (the *only* stage referencing `theta` at
all) computes `J_out[j] += binp[k] * e->coefficients[j]`, i.e. `d(moment)/
d(theta_j) = Σ_e (d(moment)/d(w_e)) · (d(w_e)/d(theta_j))` — where
`e->coefficients[j]` stands in for `∂w_e/∂θ_j` **because that partial
happens to be a graph-topology constant for linear mode.** Reverse-mode
composition does not care whether that local Jacobian came from a constant
or from `jax.grad` a moment earlier — for `weight_mode='callback'`,
`∂w_e/∂θ_j = (∂f/∂θ_j)(θ, c_e)` is still just a number for a given θ, and the
same contraction formula is exactly correct.

**What's actually missing is not pure Python — a small, genuinely new C exit
point is required.** Two options assessed:
- (a) a new mutator to overwrite `edge->coefficients[]` on a live graph after
  construction — **not recommended** (one of two options assessed, not an
  outright rejection): no such mutator exists today (coefficients are set
  once at `add_edge`), and adding one touches persistent-graph memory
  ownership, a part of the codebase already flagged as having zero
  NULL-checked allocations — real, if modest, new memory-safety surface.
- (b) **recommended**: a new, small function that runs the *existing*
  stage-0/1 code verbatim and returns the **pre-contraction** per-tape-input
  adjoint (`dm[]`/`binp[]`) *before* stage 2 runs, so the theta-contraction
  happens in Python as a plain matmul against a
  `jax.vmap(jax.grad(weight_callback))`-computed Jacobian matrix. Reuses the
  expensive O(n³) elimination-adjoint machinery (stages 0-1) completely
  unchanged; only a thin new "stage-2 exit" is needed — architecturally this
  is the exact same "make stage-2 pluggable" idea Batch 0's refactor already
  buys, making Batch C a natural **5th** switch-case consumer of the shared
  core (formula mode being the 4th).

**Third-consumer note (added 2026-08-11, from the Deferred-1 plan's
adversarial review):** a potential third consumer of this exit exists —
Deferred 1's per-SCC VJP (`deferred-1-hierarchical-scc-adjoint-plan.md`
§4-P2), which needs the pre-contraction exit **plus caller-supplied cotangent
seeding** (an input-side entry point; the shipped functions hard-code a
one-hot seed at vertex 0). Validate the exit's shape against that
requirement, or record explicitly why it is declined — do not let the exit
be designed blind to it.
**Batch-0 decision (2026-08-13, `d2cca7ab`):** the cotangent-seed entry was
DECLINED for Batch 0 (different output semantics — E[T]-vector VJP vs K
moments); the core's seeding block (target selection + factorial seed) is
comment-marked as one section, and stage-1 seeding composes orthogonally
with any future stage-2 exit — a future seed parameter touches only that
section. Recorded per this note's requirement.

**Primal side already works, no change needed:**
`Graph.update_weights(theta, callback=fn)` (`__init__.py:1910-1991` →
`phasiccpp.cpp:1857-1926`) already sets the private clone's primal edge
weights correctly for stage 0 — zero new code needed there.

**`pmf_from_graph_joint_index`'s callback path is a separate piece of
work** — different C function (`ptd_sojourn_grad_theta_subset`), different
stage split — needs its **own** new exit point; fixing the moments path does
not fix this one for free. Track as a follow-up sub-item, not assumed bundled.

**Non-JAX-native callbacks:** no general autodiff is possible through an
arbitrary black-box Python function. Two options, and this batch recommends
the second as default: (1) require the user to also supply an explicit
analytic-derivative callback (pushes correctness risk onto the user, a wrong
derivative silently produces a wrong "exact" gradient with no self-check,
unlike FD) as an opt-in for advanced users only, or (2) **leave non-JAX-native
callbacks FD-only permanently** — a legitimate, honest scope boundary for what
is explicitly the "arbitrary Python escape hatch" mode. Determining whether a
given callback *is* JAX-native needs an explicit opt-in flag or a runtime
probe (try `jax.grad`, catch the resulting tracer-conversion error) — small,
not free.

**Risks not yet resolved (carried forward from the source document's own
risk list, not previously surfaced in this plan):**
- The `jax.grad`/`jax.vmap`-over-a-toy-callback composition described above
  was reasoned through as standard first-order JAX usage but **was not
  verified by execution** in the feasibility pass — no JAX-version-specific
  quirk has been ruled out empirically.
- The stage-2 contraction's existing linear-mode code already reads
  `e->coefficients[0..P-1]` **without checking `coefficients_length >= P`**
  — a pre-existing, unguarded risk that a new callback-mode contraction
  would inherit unchanged if implemented the same way, not something this
  batch needs to fix but worth guarding against introducing more of.

**Sequencing:** after Batch 0 (shares the "new exit point on shared core"
investment — doing it before Batch 0 means writing a throwaway one-off exit
point that gets redesigned when the refactor lands). If Batch 0 is not
imminent, Batch C is still viable standalone, just slightly more expensive.
Explicitly **not** the MPFR conditioning-floor work (Job B of the same
feasibility document) — that is *Deferred 4*, tracked separately below.

**Tracked but not yet scheduled — a second, separate piece of work:**
`pmf_from_graph_joint_index`'s own callback-mode path (a structurally
different C function, `ptd_sojourn_grad_theta_subset`, with its own stage
split) needs its **own** new exit point; fixing this batch's moments-path
exit point does not deliver it for free. No urgency signal was found for
this sub-item; sequence after Batch C lands, if/when wanted (see Section 15).

---

## 6. Batch D — SVGD plumbing, Tier 1 (mechanical, zero dependencies)

Four independent, self-contained items. None depends on any other batch in
this document; all can run in parallel with everything else, including each
other.

**D.1 — `Graph.moments_from_graph` vmap crash (execution-confirmed real
bug).** `_compute_moments_pure` (`__init__.py:6767-6779`) has **no `ndim`
check** before calling `lib.compute_moments`, unlike its three working
siblings (`_exact_moments_jac_np`, which branches on the complementary
`th.ndim==1` case and falls through to a batch loop otherwise;
`_compute_pmf_and_moments_cached` and `_compute_cdf_zero_cached`, which both
branch on `theta_np.ndim==2` directly) — all three siblings detect the
batched case and loop, by one framing or the other. Under `vmap` (i.e. any
real SVGD run), `theta_flat` arrives 2-D, `len(theta_np)` silently returns
the batch size instead of `n_params`, and the ctypes call proceeds against a
mis-shaped buffer. **Fix: add the identical ndim-detection-and-loop pattern
the three siblings already use** — no change needed to the compiled backend
or to `ShapeDtypeStruct` declarations. `moments_fn_bwd`'s FD loop uses
ordinary vmap-transparent `jnp` ops, which is *reasoned* (not
execution-verified in the feasibility pass) to mean the bug is isolated to
this one host callback and doesn't recur elsewhere in the backward pass —
worth a quick confirming test during implementation rather than treating as
already settled.

**D.2 — `ptd_moment0_grad_theta`'s missing guards** (validator-only,
`PHASIC_B3_VALIDATORS`-gated, confirmed zero production exposure — single
caller, itself gated identically, no pybind/Python/test surface). Two
guards, confirmed missing, present in the shipped successor
`ptd_moments_grad_theta`:
- NULL-check for `coefficients_length==0` (`:10868` in the successor) —
  **trivial, self-contained one-line copy-paste**, no wrinkle.
- MPFR gate (`ptd_dbg_tape_needs_mpfr`, called at `:10783` in the successor)
  — **not** a plain copy-paste: the natural place to add it,
  `ptd_dbg_reverse_tape` (`:10529-10612`), is a shared helper with a *second*
  caller (`ptd_debug_reverse_grad`, a different validator) — inserting there
  changes that function's behavior too. **Recommendation: duplicate the
  small gate-check locally inside `ptd_moment0_grad_theta` instead**,
  accepting minor duplication, per this project's explicit
  `feedback_no_modify_existing` preference over touching shared, already-
  shipped code for a second function beyond the one in scope.

**D.3 — `Graph.svgd()` leaf 2b plumbing (joint-index + `exposure` set).**
`_bake_obs = observed_data if exposure is None else None` (`:6092`) means
the static baked exclusion (`:7892-7898`) does **not** fire when `exposure`
is set. Plumbing an `exact_grad` kwarg through `Graph.svgd()` to this leaf
specifically makes it structurally reachable **today, with no other batch
needed** — the one piece of real, immediate, dependency-free value in SVGD
plumbing. *(Needs a dedicated de-risk pass confirming the exact path is
correct in combination with the exposure-scaling wrapper before shipping —
not yet independently tested; see risk register.)*

> **AMENDMENT 2026-08-13 (process §6 Class D — new information
> invalidating this section's premise; found by BOTH refuters of the
> D.3 plan, `b3-batchD3-plan.md` review record): the reachability claim
> above is FALSE.** Shipped validator rule R9
> (`svgd_config.py:805-820`, shipped 2026-05-15 `f6fcbce7`, test-pinned)
> statically rejects exposure + joint-prob + no-epochs BEFORE model
> construction, on cost grounds, directing users to `epoch_starts=[0.0]`
> (the daisy route). The exposure arm of the leaf-2 call site is dead
> code for `'joint_prob'` graphs under `Graph.svgd`; the atlas line this
> section leaned on was a model-builder-level statement. §16 risk 4's
> "de-risk the exposure-wrapper interaction" is therefore moot AS
> SCOPED (the composition is unreachable). The only surviving variant
> (`joint_stop_prob` + exposure) rides a probable R9 classifier hole —
> §16b item 9. D.3's disposition is a USER DECISION (relax R9 /
> re-scope to jsp / fold into Batch G's single-epoch daisy route, which
> Batch H's internal-exposure + `exact_final_grad` machinery now makes
> strictly better-shaped). See `b3-batchD3-plan.md`.

**D.4 — `Graph.svgd()` leaf 5 plumbing (no rewards).** Already reachable
today via the callee's own default (`exact_moment_grad=True`). Plumbing only
adds the ability to explicitly force `exact_moment_grad=False` through
`Graph.svgd()`, which isn't currently possible — low value, cheap, bundle
opportunistically.

**Explicitly NOT part of this batch (see Batch G):** leaves 3/4
(rewards-bearing) and leaf 1 (daisy-chain) — plumbing these now would ship a
kwarg that is dynamically inert (leaves 3/4, blocked on Batch A) or
impossible to give any meaning to at all (leaf 1, blocked on Batch H /
*Deferred 2*). Per this project's no-silent-fallback principle, do not ship
those kwargs until their blockers land, or ship with an explicit logged
no-op warning if shipped early for API-shape reasons.

---

## 7. Batch E — joint-index baked/dedup-mode exact gradient

**The claimed fix (`b3-joint-index-plan.md`'s own framing — "a scatter-add of
the upstream cotangent by the inverse-index map before the quotient rule") is
verified correct by direct derivation, not taken on faith.** Forward:
`sojourn_probs[i] = uniq_probs[inverse_idx[i]]` is a **gather**; its VJP is a
**scatter-add**: `g_uniq[u] = scatter_add(g_visits, inverse_idx,
size=n_unique)` — one JAX primitive (`jnp.zeros(n_unique).at[inverse_idx].
add(g_visits)`). The **existing, unchanged** quotient-rule math then applies
verbatim at unique-index granularity, using the **existing**
`_exact_sojourn_jac_np(theta, _uniq_idx_jnp)` call with the *static*
`_uniq_idx_jnp` (already computed and closed over at construction) in place
of the runtime `_vi_norm`. **No new C code, no new FFI call shape, no new
callback signature** — reuses 100% of existing exact-path machinery.
`_fd_theta_bar` (the FD backward) needs **zero changes** — it already treats
the whole forward closure, gather included, as an opaque black box.

**Free bonus found in this pass:** for baked mode, the callback's
`union_idx = np.union1d(vi, _jix_all_terminal_np)` computation (currently
recomputed on every gradient call, because `vi` is a genuine per-call value
in the non-baked case) becomes **fully static** — both operands are fixed at
construction — so it, `obs_pos`, and `all_pos` can all be hoisted, a natural
efficiency win specific to baked mode that falls out of the same change.

**Critical interaction with Batch F (D6), the most important finding of
this batch's feasibility pass:** building baked-mode support on the
**current** `lax.cond`-based wiring means inheriting the exact defect D6
exists to fix — under real SVGD `vmap(grad(loss))(particles)` usage,
`lax.cond`'s predicate is always batched, so **both branches compute** (FD
*and* exact, every step), confirmed empirically this session
(`experiments/dr_lax_cond_vmap_derisk.py`). Two sequencing options:
- **(i) — recommended:** do Batch F (D6's static-if/construction-time-probe
  redesign) first, then build Batch E directly into the corrected wiring.
  D6's own design is already index-set-agnostic (the probe-and-commit
  pattern doesn't care whether the index set is baked or not), making this
  very likely the lower-total-effort path.
- (ii): build Batch E first on today's `lax.cond` wiring (matching the
  non-baked pattern for consistency), accept the known inefficiency, let a
  later D6-equivalent pass fix both together.

**No dependency on Batch 0, A, B, or C** — this is a structurally separate C
function (`ptd_sojourn_grad_theta_subset`, forward-mode) not part of the
linear/log/dph reverse-mode trio at all.

**New test needed:** exact-vs-FD parity specifically for baked mode (not
covered by the existing D5-batch joint-index tests, which were scoped
non-baked-only).

---

## 8. Batch F — D6 `lax.cond`/`vmap` redesign

**Confirmed status: planned and de-risked (commits `4f5936d0`, `7633a895`),
NOT implemented** — `_jix_probed_ok` (the design's construction-time probe
latch) has zero matches in current source. `pmf_from_graph_joint_index`
today is still exactly the pre-D6 state: `exact_grad: bool = False`
(`:7656`), plain `jax.lax.cond` at `:8268`, static `_baked` exclusion at
`:7892-7898`.

This batch is the redesign itself: replace the runtime `lax.cond`
FD-vs-exact branch (which, per the confirmed `vmap` semantics above, always
computes both branches under real SVGD usage) with a construction-time probe
that commits to a plain Python `if` — decided once, outside any `vmap`/`jit`
trace, per the previously-reviewed D6 plan. This document does not re-derive
D6's own internal design (already reviewed and de-risked in the prior
session) — it is included here only for its sequencing role relative to
Batch E (§7) and its position in the overall dependency graph. No dependency
on Batches 0/A/B/C/D; can proceed independently of all of them.

---

## 9. Batch G — SVGD plumbing, Tier 3 (blocked leaves)

Not schedulable now; listed for completeness and to make the blocking
relationship explicit rather than silent.

- **Leaf 2 (joint-index, baked, no `exposure` — the default and most common
  case; previously missing from this plan's inventory, added on review).**
  This is the leaf `Graph.svgd()` reaches by default whenever a joint-prob
  graph has no `epoch_starts` and no `exposure`. It already gets a clean,
  logged FD-decline today (the static `_baked` exclusion,
  `__init__.py:7892-7898`), so plumbing a kwarg through to it is *safe* at
  any time — but its plumbing is **low-value until Batch E lands**: per the
  SVGD-plumbing feasibility document's own per-leaf table, "once B [Batch E]
  (and ideally D6 [Batch F]) land, leaf-2 plumbing becomes genuinely valuable
  for the first time." **Gate: ship leaf-2's `Graph.svgd()` kwarg
  pass-through alongside Batch E (Phase 2, §15), not as an isolated Tier-1
  item** — shipping it standalone earlier would add a kwarg that reaches a
  branch matching neither Batch A's nor Batch E's payoff yet.
- **Leaves 3 (multivariate, 2D rewards) and 4 (1D rewards):** plumbing
  itself is a one-line pass-through in both cases, but **dynamically inert**
  until Batch A ships — `pmf_and_moments_from_graph`'s `_rewards_provided`
  guard forces FD on every call where `rewards is not None`, regardless of
  any kwarg's value, and the multivariate wrapper always passes concrete
  per-feature rewards. **Gate: do not implement until Batch A lands.**
- **Leaf 1 (daisy-chain):** plumbing is not merely low-value but
  **impossible** — `_daisy_chain_svgd_model`'s two `@jax.custom_vjp` sites
  are FD-only by construction with no exact variant to select at all. **Gate:
  blocked until Batch H (or the larger *Deferred 2* initiative) ships
  something to plumb into.**

---

## 10. Batch H — daisy-chain final-epoch exact gradient

**What makes this different from the rest of the daisy-chain problem (see
*Deferred 2*):** `final_read='sojourn'` (today's **default**) already reads
the final epoch via `joint_sojourn_graph()` — an exact, granularity-free
**elimination** solve, not uniformization at all. But the current code still
wraps this in the *identical* bulk-FD `custom_vjp` as `final_read='stopprob'`
regardless (`__init__.py:10312-10363`) — **the final epoch's own internal
exactness is not currently exploited by the gradient at all.**

**This batch:** give the final epoch's contribution an exact gradient by
reusing/extending the already-shipped `ptd_sojourn_grad_theta_subset`
adjoint, composed across the epoch/IPV boundary. Crucially, this does
**not** require either of the two things that make the rest of daisy-chain
hard: **no granularity-pinning decision needed** (this read path doesn't use
uniformization at all) and **no backprop-through-time / checkpointing
decision needed** (it's a single closed-form elimination solve, not a
`k`-step DTMC unroll).

**The one genuinely new primitive shape, even for this small piece:** the
natural per-epoch gradient primitive is not a scalar (as every other shipped
B3 adjoint produces) but a **Jacobian** of the collapsed output vector w.r.t.
both `ipv_in` (`n_ipv × n_ipv`) and `theta_epoch` (`n_ipv × n_params`) —
because chaining epochs by hand requires composing these Jacobians
epoch-by-epoch to back-propagate a cotangent from the final epoch to any
earlier epoch's θ. This is a strictly larger primitive than anything B3 has
built so far, but it slots into an **already-existing external shape**:
`_autodiff_bwd` in `daisy_chain_joint_probs` (`__init__.py:10347-10361`)
already treats the whole multi-epoch chain as one opaque cotangent-dot-
product VJP function — an exact implementation replaces `_forward`'s AND
`_autodiff_bwd`'s bodies behind the unchanged external VJP shape (replacing
only `_forward` would still yield FD gradients; corrected 2026-08-11 per the
deferred-2 plan's review F3, see that plan's §2.2 — note also that
`Graph.svgd`'s leaf 1 reaches the `_daisy_chain_svgd_model` custom_vjp
sites, not this one).

**Why this should be attempted before the (deferred) intermediate-epoch
work:** smaller, reuses fully-verified machinery, and succeeding at
Jacobian-composition across one IPV boundary is a direct prerequisite skill
for composing it across several (the exact thing *Deferred 2* needs).

**Dependencies:** none on Batches 0/A/B/C (different C function entirely,
not part of the linear/log/dph family). Relationship to Batch E is **not
fully resolved and should be re-checked once each batch's concrete design is
chosen, not assumed clean**: Batch E needs only the existing
per-observation-row Jacobian shape at a different index granularity (a pure
consumer, no change to `ptd_sojourn_grad_theta_subset` itself), whereas this
batch needs a strictly bigger `ipv × theta` Jacobian — if Batch H's
implementation requires *extending* `ptd_sojourn_grad_theta_subset`'s
existing signature/behavior (rather than adding a wholly new function), that
would be a real interface change Batch E's own call site would need to
tolerate. Neither source document settles which shape Batch H's
implementation will actually take. Can run in parallel with everything else
in this document; listed after Batch D/F only for review-bandwidth reasons,
not a hard ordering requirement — but re-verify the E/H relationship once
Batch H's design is drafted, before both are in flight simultaneously.

**Must be de-risked and adversarially reviewed on its own** before merging,
per this project's standing practice — this is new primitive shape (Jacobian,
not gradient) that nothing else in B3 has needed yet.

---

## 11. Deferred 1 — hierarchical/SCC two-level adjoint

> **Design-of-record (added 2026-08-11):**
> `deferred-1-hierarchical-scc-adjoint-plan.md` — the adversarially-reviewed
> de-risk & activation plan for this unit. Its E0 extends (not replaces) this
> section's cheap usage check; annotate here when E0 runs.

**Headline verdict: this is new math, not new plumbing, and nothing today is
blocked by its absence.**

`parallel_elimination=True` (`PHASIC_HIERAR_ELIMINATION` env var) and
`exact_moment_grad=True` are **completely orthogonal today** — confirmed by
direct trace: none of the four exact-gradient C functions (the three
moments functions plus `ptd_sojourn_grad_theta_subset`) ever touches the
SCC/composer machinery under any configuration; all four always rebuild a
full monolithic tape on every call. Turning on hierarchical mode today gives
exact-gradient computation **zero benefit** and nothing warns about it.

**Why "point the existing adjoint at a hierarchically-built tape" cannot
work:** the composer's parent-level output (`ptd_compose_scc_prcs`) is a
**plain numeric `double*` array**, not a tape — the only symbolic artifacts
produced (per-SCC parameterized compute graphs) are destroyed after each
composition call, never assembled into a whole-graph symbolic structure.
Worse, the cross-SCC "channel" edge weights are overwritten with **numeric
doubles**, one of which is `1.0 / parent_result[target]` — the reciprocal of
a *different SCC's own numeric output*. The existing tape-input encoding
(`PTD_PCG_PTR_MEM/_EDGE/_EXTERNAL`) can only express linear-in-theta
quantities; it has no representation for "this input's derivative requires
running a different elimination's own reverse-mode adjoint and applying
`d(1/x)/dx`." A correct adjoint needs a genuinely new **two-level
reverse-mode structure**: an inner level (reusable, built on Batch 0's core)
differentiating each per-SCC tape, plus an entirely new outer level — a
second reverse-mode pass over the SCC condensation DAG itself, run
source-first (opposite of the sink-first value-composition order) —
operating at SCC-block granularity, something with no existing analogue
anywhere in this codebase.

**A concrete illustration of why this is a landmine, not just a missing
feature:** the synthetic SCC graphs' channel edges carry **placeholder**
coefficients (`[1.0, 0, 0, ...]`) and ordinary `PTD_PCG_PTR_EDGE` kind. None
of the four existing functions' structural guards would catch a naive
attempt to run them directly on a post-composition synthetic graph — they'd
silently compute `d(weight)/dtheta[0]=1` for a channel edge whose true
dependency is either an arbitrary linear combination or a nonlinear
cross-SCC reciprocal chain. A plausible-looking, silently wrong Jacobian
entry — exactly the defect class adversarial review exists to catch, so this
must never be attempted as a "just try it" shortcut.

**Rationale for deferral:**
1. Large (new mathematical content, not a bug fix or cache-plumbing fix).
2. **No source-code or test path combines the two flags** — confirmed by
   grep (`exact_moment_grad` never co-occurs with
   `parallel`/`hierar`/`scc` anywhere in `src/phasic/*.py` or `tests/`).
   **Correction from adversarial review: this is narrower than "no
   tested/documented workflow" (the original wording here, now retracted as
   false) — two shipped tutorial notebooks DO configure both**
   (`docs/pages/tutorial/distributed.ipynb` and
   `docs/pages/popgen/coalescent-derived.ipynb`, both call
   `configure(parallel_elimination=True, ...)` and later run real
   `.svgd()` calls without turning the flag back off). In the one notebook
   traced in depth, this happens to have no live consequence — its SVGD
   calls all use `weight_mode='callback'` and/or `rewards=...`, both of
   which independently force FD today regardless of `parallel_elimination`
   (Batches A/C's own scope) — but the underlying claim that no *documented*
   workflow does this was simply wrong, and should not be repeated as
   evidence of urgency-free status without this caveat. **Separately, and
   independent of any B3 batch:** `distributed.ipynb`'s prose and
   `src/phasic/profile.py`'s recommendation logic (`__init__.py:8497`) both
   describe `parallel_elimination=True` as broadly beneficial for
   `Graph.svgd()` workflows without ever noting it gives zero benefit to
   the gradient computation itself (and, per the hierarchical-scc feasibility
   document's own §2, no benefit even to the *forward* moments computation
   whenever `nr_moments>=2`, i.e. the normal mean+variance case) — a
   documentation-accuracy fix worth filing as its own small, unrelated
   follow-up, not gated on any batch in this plan.
3. Real, cheap, non-code-writing prerequisite regardless: needs Batch 0's
   skeleton extraction to land first (its inner level would reuse that core;
   attempting it before risks a 4th/5th near-duplicate needing to be
   unwound later).
4. **Open scale question, addressed by an executable check, not the
   originally-proposed one.** This plan originally proposed "grep production
   model configs / recent SVGD run logs" — **confirmed, on review, that
   neither artifact exists anywhere in this repo to grep**, making that
   wording hand-wavy rather than actionable. The real substitute (already run
   once during review, see Phase 1's `[cheap check]` line in §15): grep
   `docs/`, `tests/`, and git history for co-occurrence of
   `parallel_elimination=True` and `.svgd(`/`.expectation(`/`.variance(`
   calls. The one pass done so far found two tutorials configuring both
   (point 2 above) but no case where hierarchical mode's absence-of-benefit
   actually matters for an exact-gradient-eligible call. This does not fully
   answer "does any real model *require* hierarchical mode to complete at
   all" (vs. merely for speed) — that remains open and would need a
   dedicated performance/scale investigation, not a grep, if it's ever
   decided to be worth answering. If such a model is ever found to exist,
   this batch jumps to a blocking gap; absent that, it stays at the back of
   the queue.
5. Numeric (non-gradient) correctness of the underlying hierarchical path
   itself is only tested at toy scale (the 6 `toy_model.BUILDERS` fixtures —
   the "5 canonical" phrasing elsewhere came from a stale docstring,
   corrected 2026-08-11) — no
   production-scale or `was_dph`/native-DPH coverage exists at all,
   independent of the gradient question. Any future work here needs new
   fixtures first regardless.
6. **Two additional open risks, not previously listed here:** (a) whether a
   per-SCC condition number (computed on a small synthetic subgraph by the
   composer's own MPFR escalation) correctly reflects whole-graph
   conditioning is unverified — a future per-SCC adjoint would inherit this
   same question for its own MPFR gate; (b) the composer already runs
   per-SCC eliminations inside nested OpenMP (`#pragma omp parallel for` over
   same-level SCCs, each of which may itself spawn nested work) — a future
   per-SCC adjoint added inside that loop would need to be reentrant/
   thread-safe under that regime, untested territory since nothing
   gradient-related runs there today.

**Position:** after Batch 0 lands and the usage-check (point 4) resolves —
not scheduled as active engineering work in this plan.

---

## 12. Deferred 2 — daisy-chain intermediate-epoch exact gradient

> **Design-of-record (added 2026-08-11):**
> `deferred-2-daisy-intermediate-epoch-plan.md` — the adversarially-reviewed
> de-risk & activation plan. Its conditional implementation sketch is
> explicitly non-binding per its own status header and is re-detailed only
> after the de-risk pass — it is not the "implementation plan" this
> section's directive forbids writing pre-de-risk.

**Confirmed: the claimed dependency on `ptd_graph_pdf_with_gradient`
(*Deferred 3*) is refuted.** `stop_probability(dt)` (the primal daisy-chain
uses for every epoch) and `ptd_graph_pdf_with_gradient` implement
**different algorithms** — the former is a fixed-`k` Euler power-iteration
of the embedded DTMC `P(θ)=I+Q(θ)/λ` at a caller-chosen `λ=granularity`
(confirmed: `Graph::pdf`/`Graph::stop_probability` both drive the same
`ptd_probability_distribution_context_create`/`_step` mechanism, stepping
forward until `time>=context->time`, no Poisson mixture at all); the latter
implements a genuinely different, classically-exact Poisson-mixture formula
at a fixed `λ=max_exit_rate(θ)`. **Fixing or reusing the latter would
transfer zero code and zero formulas to the former** — only a reusable
*architectural pattern* (forward-mode tangent propagation alongside the
value, discarding history per step) carries over, and either team could
independently rediscover it.

**Three novel prerequisites, none shared with any other batch in this
document, each requiring its own resolution before implementation can even
be designed:**
1. **Granularity must be pinned to a θ-independent constant.**
   `granularity=0` (the current default in both `daisy_chain_joint_probs`
   and `Graph.svgd`) resolves to an **integer-cast, θ-dependent** value
   (`2*max(512, max_rate(θ))`), re-derived fresh from the current θ every
   call — meaning the embedded DTMC's *identity* changes discontinuously as
   θ crosses integer-granularity thresholds under the current default. Any
   exact gradient is meaningless without first deciding (and validating) a
   fixed granularity policy for the whole SVGD run — FD sidesteps this today
   only because it needs local smoothness in an eps-neighborhood, which
   holds almost everywhere; an exact treatment cannot paper over it.
2. **A Jacobian-not-gradient primitive, chained across a matrix, at every
   epoch** — the same shape Batch H needs for the (simpler) final-epoch
   case, but here compounded across every intermediate epoch, with `ipv_work`
   genuinely a vector (not scalar) threaded epoch-to-epoch.
3. **Cost regime with no precedent in this codebase: backprop-through-time.**
   Every other B3 batch differentiates a fixed-size elimination tape
   (`O(n)` per replay). Daisy-chain's per-epoch `stop_probability(dt)` is `k`
   (thousands to tens of thousands) repeated applications of the same
   matrix. Naive reverse-mode storage is `O(k·n_vertices)` — infeasible at
   this codebase's stated production scale without gradient-checkpointing
   (`O(√k)` checkpoints, ~2× compute). Forward-mode (matching the existing
   `compute_pmf_with_gradient` architecture) avoids the memory blowup at the
   cost of scaling with seed count (`n_params + n_ipv` per epoch, both θ- and
   IPV-directions). **No profiling data exists for this cost model at all**
   — whether forward-mode or checkpointed-reverse-mode wins depends on
   model-specific `n_ipv`/`k`, an empirical question, not a judgment call.

**Rationale for deferral:** not a batch-sized unit of work in the sense of
the other batches in this document — a genuinely separate, multi-week
follow-up initiative needing its own dedicated de-risking pass (branch
experiments per `feedback_derisk_and_reevaluate`) to resolve the granularity
policy and the cost-model question **before** any implementation plan is
even written, let alone before code is.

**Position:** own follow-up initiative, explicitly not scheduled as part of
this master plan's implementation phases. Batch H (the final-epoch subset)
should be attempted first regardless, both because it's independently
valuable and because it builds the Jacobian-composition skill this larger
piece needs.

---

## 13. Deferred 3 — `ptd_graph_pdf_with_gradient` revival

> **Design-of-record (added 2026-08-11):**
> `deferred-3-pdf-gradient-revival-plan.md` — the adversarially-reviewed
> de-risk & activation plan (reframed as "exact PMF/PDF-term gradient"; the
> oracle-first requirement below is its E1-E5). Its conditional
> implementation sketch is explicitly non-binding per its own status header
> — it is not the "implementation plan" this section's directive forbids
> writing pre-de-risk.

**Recommend: do not trust or reuse this code without fully re-verifying
every stage against a working oracle — treat as requiring the same
from-scratch re-derivation either way, not as a quick patch.** (Revised on
review from an earlier, stronger "abandon, do not build on this code"
framing: three of the four bugs below are individually describable as
*localized* patches — porting in the primal's already-correct IPV
instantaneous-redistribution step, adding a missing quotient-rule term to
two existing DP update lines, and aligning one branch condition — not a
different recursion shape. Whether that means patching the existing file in
place or writing a fresh implementation is a real implementation-time
choice; the evidence found does not by itself force full abandonment, and
the effort estimate below is the same either way, since the hard part — the
missing chain-rule re-derivation — is unavoidable under both framings. What
*is* clear: nothing in the current file can be trusted as-is or reused
without this re-verification.) Confirmed broken in (at least) four
independent, structural ways by compiling and running a throwaway C harness
directly against the built library (not just static reading):

1. **Forward value itself is wrong by ~72%** on a trivial 2-state
   Exponential(θ) test graph against both the closed form and the real
   primal `g.pdf()` (which correctly converges as granularity→∞). Root
   cause: the function never performs the primal's special "instantaneous,
   unscaled" redistribution of the starting vertex's edges at t=0 (the IPV
   convention) — it treats the IPV edges as ordinary decaying rates from
   `k=0` instead, a fundamentally wrong initial condition, not a rounding
   difference.
2. **The DP recursion under-counts, not over-counts, `λ(θ)`'s own
   θ-dependence at every one of `k` steps** — every off-diagonal and
   self-loop update differentiates `X/λ` treating `λ` as constant, when
   `λ=max_exit_rate(θ)` is not constant whenever the max-rate vertex has
   parameterized edges (the common case). The function's own in-code
   comment frames its final correction term as fixing "double-counting" —
   the actual direction of error is the opposite, and a single
   end-of-computation subtraction structurally cannot repair a per-step,
   k-compounding omission. **Confirmed: flipping the "empirically
   determined" sign does not fix it either** — neither sign variant matches
   the function's own finite-difference check of its own forward output,
   let alone the true closed-form gradient.
3. **A second, independent bug:** the outer function's λ/λ-gradient
   dispatch condition (`coefficients_length >= n_params && n_params > 0`)
   disagrees with the inner DP's own parameterized/constant dispatch
   condition (`coefficients_length > 1`) whenever `n_params==1` — confirmed
   empirically: for a single-parameter graph, the reported PDF value **never
   changes** as θ is varied, while the reported **gradient does change and is
   nonzero** — a function reporting a nonzero derivative for an input it is
   provably not using.
4. **`granularity` is an entirely dead parameter** — confirmed by grep (only
   appears in a log string) and by execution (identical output across a
   100,000× range of granularity values) — directly contradicting its own
   header doc.

**Zero callers anywhere** (pybind, C++, Python, tests) — confirmed by grep;
there is no partial work to build on for either PMF/PDF gradients generally
or daisy-chain specifically (refuted dependency, see *Deferred 2*).

**If PMF/PDF-at-time-t gradients are ever wanted:** treat as a genuinely new
initiative, either re-deriving the Poisson-mixture formula correctly from
scratch (viable in principle — a fixed `λ=max_exit_rate` Poisson mixture is
classically exact) or consciously choosing to differentiate the *actual*
primal algorithm (fixed-`k` Euler power-iteration) instead — comparable in
scope to the discrete/`was_dph` moments batch (which itself needed a
dedicated MPFR precision gate), roughly **1.5-3 weeks**. **Like *Deferred
2*, this requires its own dedicated de-risking pass (branch experiments per
`feedback_derisk_and_reevaluate`) — establishing a working oracle and
independently re-deriving every chain-rule term against it — before any
implementation plan is even written**, not merely "MPFR-style de-risking
and adversarial review" bolted onto an implementation attempt; this batch
has zero working partial implementation to iterate from, unlike Batches
A/B/C, which each extend or mirror an already-correct sibling. Not scheduled
in this plan.

---

## 14. Deferred 4 — MPFR conditioning-floor full adjoint

> **PARKED 2026-08-15 (user decision at the CC-2 / Phase-0 report;
> `b3-d4-sweep-findings.md`, branch `derisk/d4-mpfr-sweep`).** The
> broadened sweep came back CLEAN: 64 points, zero silently-wrong
> Jacobians against an exact-rational oracle (calibrated 1e-16-class);
> declines gate-driven, logged, and 3-4 decades early on the swept
> fixtures. Phase 1 (new decline metric) and Phase 2 (full MPFR
> adjoint) are unjustified by evidence; the regression pin
> (`inference/test_d4_conditioning_pin.py`, non-MPFR-skipping) guards
> the verdict. Revisit only on telemetry showing frequent declines
> where FD's ~1e-7 fallback error is unacceptable. The sojourn family's
> separate gate evidence (Batch E/H) is out of this unit's scope and
> unaffected.

> **Design-of-record (added 2026-08-11):**
> `deferred-4-mpfr-conditioning-floor-plan.md` — the adversarially-reviewed
> decision-tree plan. Its Phase 0 IS this plan's Phase-1 "[cheap check]"
> item (one execution ticks both); note its design makes that item a small
> project (~days, an independent high-precision oracle is required beyond
> the closed-form DR-A fixture), not the "ten minutes" of the single-fixture
> version. The post-Batch-A rewards slice and post-B/C formula/callback
> slice remain tracked there.

**Confirmed: today's decline gate (`ptd_dbg_tape_needs_mpfr`) is
decline-only** — it never escalates the adjoint itself to MPFR, it only
mirrors the primal's condition-number pre-scan and forces `-1` (FD fallback)
above threshold. **Is a mechanical translation feasible?** Mostly yes for the
elementary arithmetic (every operator in stages 0-2 has a direct MPFR
equivalent, already in active use elsewhere in this file for the primal's
own MPFR path — no new numerical-methods research needed there) — **but**
there is a real, non-trivial asymmetry with the primal's own MPFR economy
that makes "mechanical" not mean "cheap":

The primal's MPFR-A path only upgrades **O(n)** per-vertex accumulators,
trusting the O(L) trace's `multiplier` constants as already-safe doubles.
The adjoint's stage-0 snapshots (`s0[i]`/`s1[i]`) feed **directly** into
stage-2's cancellation-prone quotient-rule terms (`s0[i]/s1[i]²`-type
expressions — exactly the "one rate ~1e-8" scenario the conditioning-floor
docs describe). If stage 0 stays double-precision, those snapshots are
already corrupted before stage 1/2 ever run in MPFR — a self-consistent
adjoint would need **stage 0 as well as stages 1-2** upgraded, meaning up to
`O(L)` individual `mpfr_t` values (not `O(n)`), each requiring its own
`mpfr_init2`/`mpfr_clear` pair (MPFR's C API has no operator overloading) —
a real, possibly severe, memory/performance cost the primal's own escalation
never pays. Compounding this: all three functions currently have **zero
NULL-checked allocations and no size guard**, so this would be a
substantially larger correctness-engineering burden than "swap `double` for
`mpfr_t`" suggests. There's also an unresolved open question: the existing
gate's condition-number metric was designed for the *primal value's*
stability (elimination-multiplier spread), not the *gradient's*
sub-dominant-component sensitivity — it's plausible this metric
under-detects the gradient-specific regime even when it correctly flags the
primal, meaning "reuse the same metric with a stricter threshold" may not,
by itself, catch the regime the conditioning-floor docs found.

**Rationale for deferral, and the cheap alternative substituted — corrected
on review, this is weaker evidence than originally stated.** The original
framing here ("FD is unambiguously worse in this exact regime — a historical
finding, not in dispute") overstates what the cited historical finding
(`experiments/dr_a_cyclic_analytic_gradient.py`, the "DR-A" repro) actually
established: DR-A's own code predates the `exact_moment_grad=True` default
flip and never invokes the shipped C adjoint (`ptd_moments_grad_theta`) at
all — its "conditioning floor" result (sub-dominant gradient corrupted at
θ≈1e-8) is a property of a **discarded `jnp.linalg.solve` prototype oracle**,
explicitly marked in DR-A's own text as "NOT the production route." Nobody
had, until this plan's own adversarial review, run the actually-shipped
adjoint against DR-A's fixture. **That check was done during review** (one
fixture, a small θ sweep): the shipped adjoint's existing MPFR decline gate
fires cleanly starting exactly where corruption would otherwise begin — no
observed silently-wrong zone on this fixture, in contrast to the gate-less
prototype DR-A actually measured. This is reassuring, not alarming, but it
is a single fixture, not a systematic sweep across topologies, `was_dph`,
or higher moments — it does not fully resolve the concern, only shows it is
concretely, cheaply checkable (about ten minutes of work). **Revised
scheduling: add this re-confirmation as an explicit Phase 1 cheap-check item
(§15)** — re-run DR-A's fixture through the shipped adjoint across a broader
sweep (more topologies, `was_dph`, log-mode, higher moments) before deciding
whether a new gradient-specific condition-number metric is even needed, not
after. If the broader sweep also comes back clean, the "extend the decline
gate" work below may turn out to be unnecessary rather than merely cheap.

Independent of that re-confirmation's outcome, building a full MPFR adjoint
speculatively before confirming how often production models actually land in
this regime remains a disproportionate investment relative to a documented
edge case. **If the broader sweep does turn up a gap, the recommended
action is: extend the existing decline gate with a gradient-specific
condition-number metric** (implemented as an ADDITIVE second gate consulted
alongside `ptd_dbg_tape_needs_mpfr`, per the deferred-4 plan §3 — not a
modification of the existing gate function, whose semantics are shared by
four shipped call sites; wording clarified 2026-08-11) (over the recorded
`s1[i]` magnitudes / quotient-
rule term magnitudes, not just the primal's multiplier-spread metric) so
this regime produces a clean, logged FD fallback instead of a
silently-inaccurate "exact" result — matching `feedback_no_silent_fallbacks`.
It remains an open, unverified question (flagged, not resolved, by this
plan) whether even a new metric would reliably catch a gradient-specific
regime the primal-value-oriented metric might miss — see risk register item
9. Only pursue the full MPFR adjoint if telemetry/user reports later show
this regime is hit often enough in practice to justify the cost profiled
above; if greenlit, sequence it after Batches A and B have landed on Batch
0's shared/refactored core (one MPFR rewrite of the final shared skeleton,
not three separate rewrites of soon-to-change functions).

**Position:** the DR-A re-confirmation cheap-check IS scheduled, in Phase 1
(§15) — this is the one piece of Deferred-4 that is active work in this
plan. The gate-tightening design itself (if the broader sweep shows it's
needed) and any full MPFR adjoint remain explicitly not scheduled beyond
that check.

---

## 15. Consolidated dependency graph and phased sequence

**Phases below denote dependency order, not strict time-boxing.** An item
may start as soon as its own specific prerequisites are satisfied — it does
not need to wait for every other item in an earlier-numbered phase to
finish. Where an item's gate is "Batch X" rather than "Phase N as a whole,"
that is stated explicitly.

```
Phase 1 (mutually independent — start anytime, no gate):
  D.1  moments_from_graph vmap fix        [MERGED 2026-08-13, 164e2758]
  D.2  ptd_moment0_grad_theta guards      [MERGED 2026-08-13, 164e2758]
  D.4  Graph.svgd() leaf-5 plumbing       [MERGED 2026-08-13, 164e2758]
  0    reverse-tape skeleton extraction (zero dependencies -- placed here,
       not gated behind A/B/C, because nothing blocks starting it; it has
       no user-facing value until A/B/C consume it, but there is no reason
       to delay *starting* it)
       gate: dr_moments_jac_gate.py + dr_dph_moments_jac_gate.py +
             dr_log_mode_moments_jac_gate.py byte-identical before/after
  F    D6 lax.cond/vmap redesign             [MERGED 2026-08-13, eaf86e82]
  E    joint-index baked-mode exact gradient [MERGED 2026-08-14, c475a78c
       -- + svgd leaf-2 exact_grad plumbing (s9's alongside-E gate) + the
       exact_grad_decline='fd' per-particle fallback (user decision
       2026-08-14, recorded in b3-batchE-plan.md's dated amendment);
       s16 risk item 5 DISCHARGED (scatter-add numerically gated: parity
       2.5e-16 vs the shipped exact path + oracle suite); Phase 2
       complete -- s16b reviewed at the phase boundary: remaining items
       1/4/5/6/7/8 statuses unchanged, none owed by Batch A]
  G.1  public svgd exact_final_grad plumbing [MERGED 2026-08-13, 0c052cfe
       -- Batch G leaf 1 + the folded Batch D.3's user value + R30 + the
       R9 jsp fix; b3-batchG1-plan.md v2 + merge review + findings.
       Leaves 3/4 remain gated on Batch A]
  H    daisy-chain final-epoch exact gradient [MERGED 2026-08-13, ecd708fc
       -- full de-risk + 3 review cycles (plan v1→v2, v3→v3.1, G4 diff);
       b3-batchH-plan.md + b3-batchH-findings.md; unblocks G leaf 1 and
       satisfies Def-2's "Batch H shipped" gate]
  [cheap check] Deferred-1 co-occurrence check -- REVISED from the original
       "grep production configs / SVGD run logs" wording, which does not
       correspond to anything that exists in this repo (confirmed on
       review). Executable substitute: grep docs/tutorials/tests for
       co-occurrence of `parallel_elimination=True` and `.svgd(`/
       `.expectation(`/`.variance(`. Already run once during review: two
       tutorial notebooks configure both, with no live consequence found in
       the one traced in depth (see §11 point 2) -- re-run as new tutorials
       are added; does not by itself change Deferred-1's priority.
  [cheap check] Deferred-4 DR-A re-confirmation -- re-run the DR-A cyclic
       fixture (experiments/dr_a_cyclic_analytic_gradient.py) through the
       actually-SHIPPED `ptd_moments_grad_theta` (not the discarded
       jnp.linalg.solve prototype DR-A originally used), across a broader
       theta sweep and additional topologies/was_dph/log-mode/higher-moment
       fixtures than the single case checked during review (see §14).
       Resolves whether Deferred-4's gate-tightening design is even needed.
       DESIGN-OF-RECORD (2026-08-11): deferred-4-mpfr-conditioning-floor-
       plan.md §2 — executing its Phase 0 ticks this item (one execution);
       its oracle-based design makes this a ~days mini-project, not minutes;
       the post-Batch-A rewards slice and post-B/C formula/callback slice
       remain tracked there.

Phase 1b (gate: Batch F specifically, not all of Phase 1):
  D.3  Graph.svgd() leaf-2b plumbing -- MUST NOT ship ahead of Batch F (or
       must ship with an explicit logged no-op/inefficiency warning): D.3's
       target (`pmf_from_graph_joint_index`'s `jax.lax.cond` dispatch,
       `__init__.py:8268`) is exactly the code Batch F replaces. Shipping
       D.3 first lets `exact_grad=True` pay FD *and* exact cost on every
       SVGD step for this leaf, silently, under real `vmap(grad(loss))`
       usage -- the identical defect class this plan already flags and
       mitigates for leaves 3/4 (§6), previously missed for this leaf
       (found on review).

Phase 2 (gate: Batch F):
  E    joint-index baked-mode scatter-add (built on F's corrected wiring)
  --   leaf 2 (joint-index, baked, no exposure -- the default case; missing
       from the original inventory, added on review, see §9) -- ship its
       Graph.svgd() kwarg pass-through alongside E, since leaf 2's plumbing
       has no real payoff before E lands.

Phase 3 (gate: Batch 0 specifically -- A first, then B/C in parallel):
  A    rewards support [MERGED 2026-08-14, 798ddcaa -- 1-D rewards in the
       exact moments adjoint (per-stage re-scale + adjoint VJP in
       ptd_b3_moments_core; linear+log wrappers take (rewards,
       rewards_len)); dph+rewards REFUTED by direct computation (c2d
       correction needs U/P commutation, broken by reward scaling; 2nd
       moments provably wrong) -- permanent static decline, NOT a feature
       gap, so the planned "_dph reward-weighted case" gate became a
       both-sub-kinds CONTRACT check in dr_batchA_i1_gate.py (c);
       BUNDLED (user decision 2026-08-14): svgd 1-D-rewards leaf now
       forwards exact_moment_grad, R29 1-D arm relaxed (leaf 3 of Batch G
       delivered here; G.2 shrinks to the 2-D/multivariate leaf).
       B/C note: the shared core's contraction signature now carries
       (rewards, rewards_len) before the kind enum -- a 4th/5th variant
       must thread it (or pass NULL, 0 explicitly and decline rewards
       with a log, never silently).
       Two G4 refuters SOUND-WITH-CORRECTIONS (no shipped-code defect);
       b3-batchA-plan.md + b3-batchA-findings.md]
  B    formula-mode exact gradient [MERGED 2026-08-14, c6cc38b9 --
       Wengert-list autodiff over the weight-formula tape as the 4th
       core kind; ALIGNED-theta-dim scope (lazy-decoupled class = static
       decline + ledgered follow-up); POW two-term adjoint; rewards
       inherited via the core hooks (gated); the planned "in parallel"
       with C was resolved STRICTLY SERIAL per process s3.4 -- C is now
       unblocked and must build on this merged core (4 kinds).
       b3-batchB-plan.md v1+v2 + merge review; b3-batchB-findings.md]
  C    callback-mode exact gradient, Job A only [MERGED 2026-08-14,
       35a17364 -- the binp exit as the 5th core consumer; decoupled
       theta SUPPORTED (v1 restriction refuted at review); JAX-native-
       under-jit probe; non-JAX-native = permanent FD boundary.
       *** PHASE 3 COMPLETE: the moments adjoint is exact across ALL
       FOUR weight modes (linear cont+disc, log, formula, callback). ***
       b3-batchC-plan.md v1+v2 + merge review; b3-batchC-findings.md]
  --   tracked, not yet scheduled: pmf_from_graph_joint_index's OWN
       callback-mode exact gradient (a second, separate piece of work from
       C -- see §5) -- sequence after C if/when wanted, no urgency found.

Phase 4 (gate: Batch A for leaves 3/4; gate: Batch H, already satisfied in
Phase 1, for leaf 1 -- so leaf 1's G-work may start as soon as H completes,
independent of Phase 3's timeline):
  G    SVGD plumbing leaves 3/4 (needs A) and leaf 1 (needs H)
       [ALL LEAVES DELIVERED — leaf 1 by G.1 (0c052cfe)+H; leaf 2 by
       E (c475a78c); 1-D rewards by A's bundle (798ddcaa); the
       2-D/multivariate leaf by G.2 (f73d0650, 2026-08-15: full
       symmetry + uniform rejection + R32). BATCH G CLOSED;
       PHASE 4 COMPLETE — every planned batch of the program shipped.]

Not scheduled in this plan (deferred, own future initiatives):
  Deferred 1  hierarchical/SCC two-level adjoint
  Deferred 2  daisy-chain intermediate-epoch exact gradient
  Deferred 3  ptd_graph_pdf_with_gradient revival (do not trust/reuse as-is;
              re-verification-or-rewrite, own de-risking pass required)
  Deferred 4  MPFR conditioning-floor FULL adjoint only -- the DR-A
              re-confirmation cheap-check IS scheduled, in Phase 1 above
```

**Cross-batch conflict matrix** (only cells with a real interaction shown;
blank = confirmed no interaction by at least one feasibility document):

| | 0 | A | B | C | D | E | F | H |
|---|---|---|---|---|---|---|---|---|
| **0** | — | same 2 lines×3 fns | shared core, 4th case | shared core, 5th case | | | | |
| **A** | | — | both consume shared core (no direct overlap) | both consume shared core | leaves 3/4 depend on A | | | |
| **D** | | | | | — | leaf-2 plumbing depends on E | **D.3 must sequence after F (§6, §7) -- found on review, not in original matrix** | leaf-1 plumbing depends on H |
| **E** | | | | | see D | — | **must sequence vs F (§7)** | shares underlying C fn w/ H -- **RESOLVED @ H merge `ecd708fc`: H kept the C signature unchanged (wrapper) and added only a new additive symbol; E's planned consumer is unaffected (re-verified at H's G5)** |
| **F** | | | | | see D | see E | — | |
| **H** | | | | | see D | see E | | — |

**Why this ordering and not another:** Phase 1 items were independently
confirmed dependency-free by every relevant feasibility document — deferring
them would gain nothing, and Batch 0 is included here (not delayed to its
own later phase) precisely because it, too, has zero unmet dependencies;
its *value* only materializes once A/B/C consume it in Phase 3, but there is
no reason to delay *starting* it. D.3 is the one Phase-1 exception, held to
Phase 1b specifically because its target code is exactly what Batch F
replaces (found on review — the original draft placed it in Phase 1
unconditionally, which was wrong). E is gated on F specifically (not on
0/A/B/C) per the joint-index feasibility document's explicit recommendation,
so it is placed in its own phase keyed only to F's completion, letting it
proceed independently of the Phase 3 timeline; leaf-2's plumbing rides along
with E since that is the earliest point at which it becomes valuable.

---

## 16. Consolidated risk register

1. **`ptd_dph_correct_discrete_moment_grad`'s validity under reward-
   weighting** (Batch A) — plausible-by-analogy, not independently verified.
   Gate before shipping `_dph`'s rewards support.
2. **`was_dph` exclusion for formula mode** (Batch B) — asserted by analogy
   to `_log`, needs its own direct repro, not yet done.
3. **`POW`'s two-sided differentiation rule** (Batch B) — needs a dedicated
   `jax.jacobian` cross-check before trusting it.
4. **Leaf 2b's interaction with the exposure-scaling wrapper** (Batch D.3) —
   no dedicated test exists yet confirming the exact path is correct in
   combination with `_wrap_model_with_exposure`; needs its own de-risk pass,
   not assumed safe by transitivity.
5. **Batch E's scatter-add derivation is algebraically verified but not yet
   numerically gated** against a live FD/central-difference oracle — a
   future implementation pass must do this before merging, per standing
   practice.
6. **E/F sequencing is a genuine judgment call**, not fully resolved here —
   option (i) (F before E) is recommended but option (ii) remains viable if
   baked-mode correctness is wanted sooner than the D6 redesign.
   **RESOLVED 2026-08-13: option (i) happened — Batch F merged first
   (`eaf86e82`); E unblocked with the F merge review's probe-the-actual-
   baked-union requirement.** (Note: the Batch H plan's v1 mis-cited this
   risk as the E/H interface question; that question lives in §10's
   Dependencies paragraph and the §15 conflict matrix — corrected in H v2.)
7. **Deferred-1's scale question is unresolved** — whether any real model
   requires hierarchical mode for tractability (not just speed) would
   materially change its priority; a cheap usage check is recommended before
   any future prioritization decision, independent of this plan's phases.
8. **Deferred-2's cost-model question has zero profiling data** — whether
   forward-mode or checkpointed-reverse-mode wins depends on real,
   model-specific `n_ipv`/`k`, not yet measured anywhere.
9. **Deferred-4's condition-number metric adequacy is unconfirmed** — it's
   possible even a new gradient-specific metric under-detects the regime;
   needs re-confirmation against the historical DR-A repro before the
   gate-tightening fix is finalized.
10. **The historical "fix applied to one sibling, never backported" pattern**
    (MPFR gate, `coefficients_length` guard — both fixed once, never
    backported to `ptd_moment0_grad_theta`) is exactly the failure mode
    Batch 0 exists to structurally prevent for future fixes — but it is a
    reason to prioritize Batch 0 highly within Phase 1/3, not a reason to
    delay any batch further.
11. **D.1's "bug isolated to one host callback" claim** (Batch D.1) is
    reasoned from ordinary vmap-transparency of `jnp` ops, not
    execution-verified — worth a quick confirming test during
    implementation rather than treating as already settled (see §6).
12. **Batch 0's contraction-dispatch interface (enum-switch vs.
    function-pointer) is an open design decision**, not yet resolved by any
    source document — whoever implements Batch 0 should validate the choice
    against both Batch B's and Batch C's contraction needs before
    committing (see §2) — and against Deferred-1's cotangent-seeded per-SCC
    VJP requirement (§5's third-consumer note;
    `deferred-1-hierarchical-scc-adjoint-plan.md` §4-P2; added 2026-08-11).
    **RESOLVED 2026-08-13 (Batch 0, `d2cca7ab`):** enum-dispatched switch
    chosen and validated (B: core-internal pre-outk stage; C: exit added
    when C lands, options recorded in `b3-batch0-plan.md`; Deferred-1:
    declined with the orthogonality record at §5).
13. **Two hierarchical/SCC-specific risks, relevant only if Deferred 1 is
    ever un-deferred:** (a) whether a per-SCC MPFR condition number
    correctly reflects whole-graph conditioning is unverified; (b) a future
    per-SCC adjoint would run inside the composer's existing nested-OpenMP
    loop, untested territory for reentrancy (see §11 point 6).
14. **Two callback-mode (Batch C) risks carried from the source document,
    not previously listed here:** the `jax.grad`/`jax.vmap` composition
    was reasoned through but not execution-verified against a toy callback;
    the stage-2 contraction's pre-existing unguarded
    `coefficients_length >= P` assumption would be inherited by a new
    callback-mode contraction built the same way (see §5).
15. **D.3/Batch-F gating (§15, Phase 1b) depends on this plan actually being
    followed** — the risk this mitigates (silent FD+exact double cost under
    `vmap` for the joint-index+`exposure` leaf) is real and code-confirmed,
    not hypothetical; if D.3 is ever implemented by someone working from an
    earlier cached copy of this plan (or from Section 6 in isolation without
    Section 15's gating note), the mitigation is lost. Flag this
    prominently in Batch D.3's own eventual implementation plan, not just
    here.
16. **Deferred-1's documentation-accuracy gap** (`distributed.ipynb`,
    `profile.py`'s recommendation logic) overstating `parallel_elimination`'s
    benefit for SVGD/gradient workflows is a real, independent, small
    follow-up — not gated on any batch in this plan, but worth not losing
    track of (see §11 point 2).
17. **Deferred-4's DR-A re-confirmation (Phase 1 cheap-check) is currently
    single-fixture** — the review pass that ran it checked one topology
    across one theta sweep; a clean result there does not yet generalize to
    `was_dph`/log-mode/higher-moment cases, which the Phase-1 item explicitly
    calls for broadening to before treating the question as resolved (see
    §14).
18. **Deferred-3's "patch in place vs. rewrite from scratch" is an
    unresolved implementation-time choice** (softened from an earlier
    "abandon" verdict on review) — the evidence found does not by itself
    settle which is cheaper; whoever picks this up should make that call
    explicitly rather than defaulting to either extreme (see §13).

---

## 16b. Unscheduled follow-ups ledger (added 2026-08-11, per cross-plan review)

Items no batch or deferred unit owns, listed so they cannot silently fall
through (each needs an explicit owner/vehicle when picked up; none is gated
on any batch here):

1. **Risk-16 doc fix** — `distributed.ipynb` + `profile.py` overstate
   `parallel_elimination`'s benefit for SVGD/gradient workflows. Standalone
   micro-task, any time.
2. **CLAUDE.md joint-index MPFR-comment correction** — the comment in
   `ptd_sojourn_grad_theta_subset` claims a rationale that doesn't transfer.
   Standalone one-liner; bundle with any Batch E/F-adjacent docs pass.
   **CLOSED 2026-08-13 @ Batch H merge `ecd708fc`: the comment is corrected
   in situ in the rewritten core (the function was being edited under the
   gate-opt-out user decision, so Batch F's "no C edits" decline reason no
   longer applied), citing the H0 evidence: this path's primal has no MPFR
   fallback, and the gate declined 100% of realistic calls while its lifted
   answers matched an fp64 oracle to ~1e-13. That evidence is ALSO a dated
   INPUT to Deferred-4/CC-2 (threshold semantics): Def-4's scope currently
   EXCLUDES the sojourn slice, so its owner must decide whether to widen —
   flagged here, not silently absorbed. Post-opt-out, Def-4 pinning no
   longer affects the daisy `exact_final_grad` path (it skips the gate);
   it still governs the default sojourn/joint-index gate.**
3. **Offset-tape-conversion caching** — `ptd_pcg_convert_to_offset` runs
   fresh per call (O(commands)); Batches E and H put the sojourn function on
   per-SVGD-step / per-epoch hot paths that inherit this cost silently.
   Evaluate during Batch H's design. **CLOSED 2026-08-13 @ Batch H merge:
   evaluated per the mandate (H1(a), `experiments/dr_batchH_cost.py`) and
   DECLINED with evidence — the whole adjoint call containing the
   conversion is 1.0-1.3% of the FD backward it replaces, stable across a
   37× graph-size range (n=514 → 18,910); caching could recover at most
   that. Re-open only if a future consumer's profile contradicts this.**
4. **Rate-blowup fwd/bwd inconsistency** (CLAUDE.md flagged) — still
   unscheduled; the Deferred-3 plan adopts fwd/bwd consistency as a
   requirement for its own new path but does not fix the moments-path gap.
5. **`moments_from_graph` / `method_of_moments` exact gradients** —
   explicitly out of this plan's scope (see the amended Scope paragraph);
   only D.1's vmap-crash fix touches the former.
6. **NEW (2026-08-11, found by the Deferred-1 plan's adversarial review):
   the hierarchical composer silently recomputes ALL weights linear** —
   `ptd_graph_update_weights(..., use_log=false)` unconditionally at
   `scc_compose.c:187`/`:315`, and the formula `weight_tape` is never
   copied to synthetic graphs. A `weight_mode='log'` parameterized graph
   under `parallel_elimination=True` gets linear-weight NUMERIC results
   with no error (rewards-free first-moment path). Pre-existing
   silent-wrong-answer hazard; needs its own guard/doc micro-fix (decline
   or warn in the hierarchical gate), independent of any gradient work.
7. **NEW (2026-08-12, Batch D Tier-1 triage): `pmf_from_cpp`'s callback
   likely shares the D.1 vmap bug** — its pure_callback wrapper uses
   `vmap_method='expand_dims'` with no ndim handling
   (`__init__.py:4174-4183`, read-confirmed by the G4 reviewer; the atlas
   flagged it as "probable sibling, unconfirmed"). Execution probe deferred:
   `pmf_from_cpp(cpp_file, ...)` needs a generated C++ model-file fixture,
   beyond the batch's bounded-probe budget. Fix would mirror D.1's loop
   (Class B; own micro-batch with a cpp-file fixture).
8. **NEW (2026-08-11, found by the Deferred-2 plan's adversarial review):
   daisy FFI handlers swallow C-level context-create failures** — the
   exception becomes a NaN row + `Success()`, with no log line at all on
   the default sojourn handler (`graph_builder_ffi.cpp:2065-2082`, `:2148`;
   contrast the loud negative-rate escalation at `:1887-1896`). A
   robustness/observability gap worth a small logging fix on its own; the
   full loud-path design is costed in the Deferred-2 plan's E1.
9. **NEW (2026-08-13, found by BOTH refuters of the D.3 plan): R9's
   graph-kind classifier hole** — `_check_R9_exposure_with_vanilla_
   joint_prob_unsupported` (`svgd_config.py:805-820`) tests only
   `graph_kind == 'joint_prob'`, but a `joint_stop_prob_graph()` output
   carries the base-graph indexer and enters the SAME svgd joint-index
   branch — so exposure + jsp + no-epochs PASSES validation into a
   configuration R9's cost rationale equally condemns, and R1
   simultaneously forbids the `epoch_starts` remedy R9's message
   prescribes. Exactly the classifier-hole class R29 had (fixed at
   D.4's G4). No test composes jsp + exposure anywhere; intent
   undeterminable from source. Class B candidate (silent acceptance of
   a config the rule means to reject); fix is a two-token kind-set
   change + one pinning test, but needs the D.3-disposition user
   decision first (the fix direction depends on it). **CLOSED 2026-08-13
   @ Batch G.1 merge `0c052cfe`: kind-AWARE fix shipped — R9 gains a
   jsp arm whose message does NOT prescribe the R1-forbidden
   epoch_starts remedy (points at the source joint-prob graph);
   joint_prob arm byte-identical (programmatically verified); pinned
   tests both arms; R30's no-epochs branch got the same kind-aware
   treatment so the trap class is closed in both rules.**
10. **CLOSED 2026-08-15 @ Batch G.2 merge `f73d0650` (uniform-rejection
    user decision):** the 1-D model now rejects 2-D rewards LOUDLY on
    all three compute paths — and the closure resolved MORE than this
    item recorded: the FFI path's silent-garbage output (features
    beyond the first dropped) is fixed by the same guard, and the
    callback path's accidental working 2-D support was retired by
    recorded decision (wrapper value-identical by probe). Original
    item: **direct 2-D rewards on the
    1-D `pmf_and_moments_from_graph` leaf fail in the FORWARD** with a
    shape-contract error (`Expected: (2, 4), Actual: (2, 2)` — the
    pure_callback result spec doesn't account for the feature axis).
    The production 2-D route (`pmf_and_moments_from_graph_multivariate`,
    per-feature 1-D slices) works and — post-A — engages the exact
    gradient per feature. Nothing user-facing routes 2-D rewards at the
    1-D leaf directly; still, the forward should either support or
    loudly reject them. Natural vehicle: Batch G.2 (the 2-D/multivariate
    leaf pass) or a standalone micro-fix. Also recorded in
    `b3-batchA-findings.md`.
11. **NEW (2026-08-14, Batch C close-out): the analytic-derivative-
    callback opt-in** — for NON-JAX-native weight callbacks, feasibility
    Q4 option 1 (user supplies `weight_callback_grad(theta, coeffs) ->
    (P,)`) slots into the SAME binp-exit contraction Batch C shipped;
    declined as a default (a wrong user derivative is silently wrong
    with no self-check), available as an explicit opt-in if ever
    requested. Also ledgered here: the joint-index callback exit
    (master §5 records it) and the batched-vmap W optimization (D-C2's
    recorded option, b3-batchC-findings.md).

---

## 17. What "done" looks like for this document

This master plan is complete once:
1. ✅ **Done.** Submitted to adversarial review: three independent parallel
   passes, each tasked to refute — not confirm — one dimension
   (sequencing/conflicts, grounding fidelity against source code, deferral
   rationale + bffg scope boundary).
2. ✅ **Done.** Findings from all three passes incorporated into this
   revision — every finding above "cosmetic" severity was either fixed
   (claim reworded, missing dependency/risk added, phase resequenced) or is
   explicitly carried forward as an open, flagged risk (Section 16) where no
   source document resolves it outright. None were silently dropped.
3. ✅ **Done — SIGNED OFF by the user 2026-08-11** (as amended: three-pass
   review folded in + the 2026-08-11 deferred-unit design-of-record
   amendments and §16b ledger). Execution governed by
   `b3-execution-process.md` (documentation/branch/gate/git/deferral
   policy) and tracked live in `b3-execution-tracker.md`. Precondition
   before any implementation code: the clean-baseline test ledger
   (`b3-test-baseline.md`) recorded at this commit.

**No implementation code is to be written for any batch — including Phase 1
"mechanical, zero-dependency" items — until that sign-off occurs**, per the
user's explicit, twice-stated directive that the whole plan must be
validated before any code changes for ANY path. When each batch's turn
comes, it gets its own detailed implementation plan (design, concrete diff
sketch, new gate scripts) written and reviewed separately, per the user's
own chosen granularity ("master sequencing plan now; detailed per-batch
plans as each starts").
