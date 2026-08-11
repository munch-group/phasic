# Deferred 3 — PMF/PDF-at-time-t exact gradient (the `ptd_graph_pdf_with_gradient` question): de-risk & activation plan

**Status: v2 — adversarially reviewed 2026-08-11 (dedicated technical refuter:
SOUND-WITH-CORRECTIONS; cross-plan conflict refuter: the D2×D3 E4 gate was
CONFLICTING as written, now fixed); all findings folded in, see §9.
Planning-only — no code changes of any kind are authorized by this document.** The master plan (§13) requires
"establishing a working oracle and independently re-deriving every chain-rule
term against it — **before** any implementation plan is even written." This
document plans exactly that de-risk pass, the route decision it must produce,
and a conditional implementation sketch to be re-detailed from findings.

**Grounding.** Primary source: `atlas/plan-feasibility-pdf-gradient-daisy-chain.md`
("PDF §n"; its §§1-4, 6 are this unit's evidence base), master plan §13 and
risk item 18. [PDF §n] = read-from-source/measured findings; [reasoned] = this
plan's inference, to be confirmed in de-risk.

---

## 1. Reframing: what this unit actually is

The unit is **not** "repair the dead function." It is: *exact θ-gradients for
the time-indexed PMF/PDF term of the likelihood.* That term is FD in every
shipped model and stays FD under every master-plan batch (the B3 moments/
sojourn adjoints cover the moment-regularization and sojourn-likelihood terms;
"pmf stays FD" is explicit in the shipped batches). `ptd_graph_pdf_with_gradient`
is merely a broken, zero-caller artifact adjacent to this goal [PDF §2-§4].

**Do-not-trust baseline (measured, not assumed):** the existing function has
four independent structural defects — (1) forward value ~72% wrong (missing
the primal's special instantaneous t=0 IPV redistribution); (2) per-step
under-counting of λ(θ)'s θ-dependence, uncorrectable by its end-of-run
"double-counting" term (neither sign works); (3) self-inconsistent
parameterized/constant dispatch at `n_params==1` (PDF value provably ignores
θ while the gradient is nonzero); (4) `granularity` a dead parameter
[PDF §2.2-§2.5, §3]. Zero callers anywhere [PDF §4]. Per master-plan risk 18,
patch-vs-rewrite stays an implementation-time choice, but **nothing in the
current file may be trusted or reused without full re-verification**; this
plan's working assumption is greenfield with the old file left untouched
(zero callers ⇒ zero risk in leaving it; removing it is a separate user
decision, [[feedback_no_modify_existing]]).

## 2. The route decision (the single most load-bearing choice)

Two mathematically legitimate targets exist; they differ in *which quantity*
gets differentiated:

- **Route (i) — differentiate the actual primal.** The shipped
  `pdf`/`stop_probability` is a fixed-`k` Euler power iteration of
  `P(θ)=I+Q(θ)/λ` at caller-chosen λ=granularity, with the special
  instantaneous t=0 IPV step [PDF §1]. Forward-mode tangent propagation
  through exactly that recursion yields the gradient **of the value the
  library actually returns**, at any finite granularity — full fwd/bwd
  consistency (a property CLAUDE.md's rate-blowup follow-up records as
  currently *violated* on the moments path — flagged there as a gap; this
  unit adopts it as a hard design requirement, it does not inherit it as
  established practice [cross-plan F8]). Requires λ pinned θ-independently.
  **Pinning is NOT automatic today at the model level:** `pmf_from_graph` /
  `pmf_and_moments_from_graph` expose no granularity control and the FFI PMF
  call sites hardcode `granularity=0` (verified `__init__.py:3740`, `:7086`,
  `:7162`; the audit's F-004 *section* notes the hardcoding as a methodology
  trap, and **F-005** is the auto-granularity finding proper [review F4 —
  attribution corrected]). Note auto-λ is θ-dependent **only above the 512
  floor** (`max_rate` initialized to 512, `phasic.c:12648`): every model
  with max exit rate <= 512 runs at a constant λ=1024, which includes
  essentially all small closed-form fixtures — so if the de-risk wants to
  *demonstrate* (not just assert) the auto-mode pathology, one fixture must
  cross max-rate 512 over its θ-range [review F4]. The exact-mode path
  therefore needs an explicit θ-independent λ control on its own path — an
  explicit pinned-granularity kwarg on a new entry point is one candidate
  mechanism (= Deferred 2's E1 policy (i)); **the final mechanism is the
  single shared λ-policy decision recorded in E4, not pre-committed here**
  [cross-plan F2]. Its forward must evaluate at that same pinned λ as the
  gradient — meaning the exact-opt-in model's value differs by O(1/λ) from
  the default auto-λ value, a documented cross-mode difference to manage
  (same shape as Deferred 2's risk 2), never a silent change to the
  existing paths. Cost `O(P·k·edges)` forward-mode.
  **Working hypothesis [reasoned, review F2]: this is the same
  tangent-stepper primitive Deferred 2 needs**, at single-epoch scope with a
  different readout (absorbed-mass pmf/cdf harvest vs. full state vector)
  and seed set (θ-only vs. θ+IPV) — PDF §5.2 itself establishes only that
  the forward-mode *pattern* transfers from the dead code; primitive
  identity is E4's decision, not a fact.
- **Route (ii) — a correctly-derived Poisson-mixture primitive.** At a
  **pinned** λ ≥ max exit rate over the θ-range visited, the Poisson mixture
  is classically exact in t and `dλ/dθ = 0`, which *eliminates by
  construction* the entire per-step λ-chain-term class that sank the dead
  code (its defect 2) [reasoned from PDF §2.1-§2.2: the missing terms are all
  ∂/∂λ terms]. Trade-offs: it is a *new, additive* value+gradient pair whose
  value differs from `g.pdf` at finite granularity (a deliberate,
  documented cross-path difference to manage under the strict-xfail
  conventions — NOT a silent change to the existing primal, which stays
  untouched); the λ ≥ max-rate invariant must be validated per call and fail
  loudly; Poisson-tail truncation needs its own error bound.

**Recommendation (to be confirmed by de-risk, not pre-committed):** route (i),
because (a) fwd/bwd consistency with the shipped value; (b) the shared
primitive with Deferred 2 halves total cost if both activate *and E4
confirms the primitive identity*; (c) it keeps
exactly one uniformization semantics in the codebase
([[feedback_avoid_matrix_exp]]-adjacent simplicity argument). Route (ii)
remains the fallback if E2 shows the Euler gradient's O(1/λ) bias is
practically material at affordable λ.

## 3. Activation gates

- **A1:** master plan signed off, and the user confirms exact PMF/PDF-term
  gradients are actually wanted (the moments+sojourn exact terms may already
  deliver the practical SVGD benefit; measure before building — see E0).
- **A2:** the de-risk phase below completed; route chosen; and the E4
  interface note exists — written against BOTH units' requirements
  **regardless of Deferred 2's activation status** (ownership: whichever
  unit activates first; a stepper built by this unit must satisfy or
  explicitly decline, user-signed, Deferred 2's θ+IPV-seed / state-vector
  requirements) [cross-plan F1 — the v1 "only if Deferred 2 is also active"
  conditional deadlocked the joint gate under single-unit activation]. If
  route (ii) is selected, the sharing premise lapses and E4 reduces to a
  written no-sharing note [cross-plan F2].

## 4. De-risk phase (experiments; branch-only, no shipped-code changes)

- **E0 — value measurement (runs AFTER E1, which supplies its oracle;
  Batch-H-independent [review F1 — the v1 "run E0 after H" sequencing was
  inert: H lives in the disjoint daisy family]).** On >=2 real SVGD fits
  using `pmf_from_graph` (pure-PMF likelihood — its gradient is 100% FD)
  and `pmf_and_moments_from_graph` (moments term exact by default, PMF term
  FD — the separable `grad_i = grad_pmf_i + grad_moments_i` structure at
  `__init__.py:7609-7618` makes the PMF term's FD error directly isolable),
  quantify how much of the total gradient error — vs. a full-model oracle:
  E1's pdf oracle for the PMF term + the shipped exact moments Jacobian for
  the moments term — comes from the FD PMF term at benign and mixed-scale θ.
  The daisy-side residual-FD measurement is owned by **Deferred 2's §1 value
  test** — cross-referenced, not duplicated here. If the PMF term's FD error
  is immaterial where it matters, park the unit (legitimate outcome; report
  as such).
- **E1 — oracle battery.** Closed forms (Exponential, Erlang, hypoexponential
  d(pdf)/dθ) + a dense JAX reference for arbitrary small graphs (build
  `Q(θ)`, compute pdf via the same Euler recursion in JAX incl. the special
  first step; autodiff it), covering cyclic graphs; **linear weight mode
  only, for both parameterized and constant edges; continuous-only pending
  the risk-3 discrete decision** (log/formula/callback out of scope v1,
  rejected loudly — mirroring the daisy path's existing log rejection)
  [review F6 — the v1 "both weight modes' linear case" wording was
  self-contradictory]. The JAX reference machinery is shared with
  Deferred 2's E3 — one reference implementation, two scopes
  [cross-plan F11].
- **E2 — route (i) probe.** Validate the JAX Euler reference against the
  *production* `g.pdf` at matched granularity (parity gate ~1e-12); then
  measure gradient-vs-λ convergence: does d(pdf)/dθ converge O(1/λ) like the
  value, and is the bias at practical λ below FD's own error? Output: the
  quantitative case for/against route (i).
- **E3 — route (ii) probe.** Same reference machinery, Poisson mixture at
  pinned λ: verify exactness vs. closed forms; quantify tail-truncation error
  vs. the 6σ rule; verify `dλ/dθ=0` makes the gradient exact; characterize
  failure when λ < max rate (must be detectable, loud). **Name the density
  alignment explicitly in the derivation** (from review): the phase-type
  density aligns as `f(t) = λ·Σ_k Poisson(k;λt)·π_{k+1}` (mass absorbed at
  step k+1) — an off-by-one class the closed-form gate must be constructed
  to catch, and a term the E5 dossier must state, not assume.
- **E4 — shared-primitive + λ-policy interface note with Deferred 2** (joint
  deliverable; see that plan's §4-E4). Ownership: whichever unit activates
  first; written against BOTH plans' requirements regardless of the other's
  activation status [cross-plan F1]. Scope: (a) one C tangent stepper vs.
  two (default: one [reasoned] — the readout and seed-set deltas recorded
  explicitly, per §2); (b) the SINGLE λ-pinning policy decision both units
  share — this plan's §2 kwarg is one candidate mechanism, final mechanism
  set here [cross-plan F2]; (c) if route (ii) is chosen, a written
  no-sharing note.
- **E5 — chain-rule re-derivation dossier.** For the chosen route, a written
  derivation of every term (the master plan's explicit requirement), checked
  term-by-term against the E1/E2 oracle by zeroing-out experiments (drop one
  term, confirm the oracle mismatch appears where predicted). This dossier is
  the input to the implementation plan and its adversarial review.

**Re-evaluation checkpoint:** route decision + go/park; then write the real
implementation plan from findings; adversarial review; sign-off.

## 5. Conditional implementation sketch (post-de-risk, re-detail first)

- **B1 — C tangent stepper** (shared with Deferred 2 if both active; see E4).
  Additive new functions; existing `ptd_probability_distribution_*` and the
  dead `ptd_graph_pdf_with_gradient` untouched.
- **B2 — bindings + model wiring.** Mirror the established B3 machinery
  (private clone, `pure_callback` with F64 dtype, decline → logged FD,
  [[feedback_no_silent_fallbacks]]), with **construction-time static
  dispatch — no per-call `lax.cond`** (Batch-F lesson). **API shape is an
  explicit OPEN decision for the implementation plan [review F3]:** (a) a
  new entry point (additive; an `exact_grad=False`-default kwarg would be
  incoherent there — a new exact-purpose builder shipping FD-by-default is
  pointless) vs. (b) an `exact_grad`-style kwarg on the existing
  `pmf_from_graph`/`pmf_and_moments_from_graph` (touches a shipped
  signature — carries the `feedback_no_modify_existing` approval flag).
  Each option's no-modify implication is weighed at activation, not
  pre-decided here. Discrete dispatch explicitly scoped (native DPH in
  scope v1 or excluded loudly — decide in the implementation plan from E1
  coverage; `was_dph` excluded v1, matching the sojourn adjoint's
  precedent).
- **B3 — gates.** E1 battery as pytest; exact==FD benign-θ parity;
  mixed-scale correctness (vs. oracle) where FD is pinned-wrong; granularity
  semantics tests (pinned-λ validation failure is loud); cross-path value
  gates unchanged (the existing primal's outputs must be bit-identical
  pre/post — trivially true since nothing existing changes).

**Sizing honesty:** 1.5-3 weeks standalone [PDF §6]; materially less for the
C stepper if shared with Deferred 2.

## 6. Interaction / conflict analysis

| Against | Interaction | Resolution |
|---|---|---|
| Deferred 2 | Plausibly the same new primitive (E4 working hypothesis; readout/seed deltas recorded); same λ-pinning decision — made ONCE in the E4 note | E4 owned by whichever unit activates first, written against both plans' requirements regardless of the other's status [cross-plan F1]; route-(ii) selection lapses the sharing |
| Batch H | None structurally (H uses the elimination-based sojourn read, no uniformization) [PDF §5.6]; E0 is H-independent (its fixtures are the pmf family; the daisy-side measurement belongs to Deferred 2 §1) [review F1] | None — do not sequence E0 on H |
| Batches 0/A/B/C | No overlap — elimination-tape machinery vs. uniformization stepping [PDF §8(a),(b)]; the Batch-0 skeleton is very likely not a candidate home for the stepper [reasoned — PDF §8(b) hedges "may not even be a candidate"; carry its requested two-line note when Batch B is scoped, review F5] | None |
| Batches D/E/F/G | No line overlap. F's static-dispatch lesson adopted in B2 | None |
| Deferred 1 | Orthogonal [PDF §8(d)] | None |
| Deferred 4 | Orthogonal (different numerical machinery; no MPFR path exists in uniformization stepping). If a conditioning-style gate is ever wanted for the stepper, it is a new question, not Deferred 4's | None |
| `pmf_from_graph_parameterized` revival (CLAUDE.md) | Untouched; that is a disabled Builder-API question, not a gradient question | None |
| `feedback_no_modify_existing` | Greenfield additive under this plan's WORKING ASSUMPTION (§1); master-plan risk 18 keeps patch-in-place open — that route would modify shipped (dead) code and needs explicit user approval [review F7]; a B2 option-(b) kwarg likewise (§5-B2) | Respected under the greenfield assumption; conditional approval flags recorded for the alternatives |

## 7. Risks specific to this plan

1. **The route decision could be forced both ways** (E2 shows Euler-gradient
   bias material AND E3 shows pinned-λ infeasible for wandering particles) —
   then the honest outcome is "exact PMF gradient is not practically
   attainable under current primal semantics," documented and parked.
2. **JAX-reference parity with `g.pdf`** may expose primal quirks (the 512
   floor, `floor(λt)` indexing off-by-ones) — E2 must match them exactly or
   the oracle is invalid; budget iteration time for this.
3. **Discrete/DPH scope creep:** `dph_pmf` gradients are a distinct recursion;
   v1 should scope continuous-only or native-DPH-only after E1 evidence, and
   must exclude the rest loudly.
4. The dead function's presence invites future confusion — mitigation: the
   implementation plan should propose (as a user decision) a one-line header
   deprecation comment or removal, explicitly flagged as a modification
   requiring approval.

## 8. Handoff

**State snapshot (2026-08-11):** master = `cadf1ca4` (local, unpushed); master
plan awaiting sign-off; `ptd_graph_pdf_with_gradient` (src/c/phasic.c:13090)
confirmed broken 4 ways, zero callers; no gradient variant of the real primal
exists anywhere [PDF §4]. Unit parked pending gates A1-A2.

**Copy-paste prompt for the executing session:**
> Read `/Users/kmt/phasic/deferred-3-pdf-gradient-revival-plan.md` and
> `/Users/kmt/phasic/atlas/plan-feasibility-pdf-gradient-daisy-chain.md`
> (§§1-4, 6) in full. Confirm the master plan was signed off and gate A1
> holds; if not, stop and report. Run the de-risk on a branch
> (`derisk/pdf-exact-gradient`) in the order E1 → E0 → E2/E3 → E5
> (E0 consumes E1's oracle; none of it depends on Batch H), experiments
> only, no `src/` changes; treat the existing `ptd_graph_pdf_with_gradient`
> as reference-only evidence, never as a base to build on. E4 is a JOINT
> deliverable with `deferred-2-daisy-intermediate-epoch-plan.md` — owned by
> whichever unit activates first, written against BOTH plans' requirements.
> Then write the implementation plan from findings (including the E5
> derivation dossier), adversarially review it, and present for sign-off
> before any code.

## 9. Adversarial review record (2026-08-11)

One dedicated technical refuter (verdict: **SOUND-WITH-CORRECTIONS**; 1 MAJOR
F1, 6 MINOR F2-F7) + one cross-plan conflict refuter (D2×D3 judged
CONFLICTING as written on the E4 gate — CRITICAL cross-F1 — now fixed). All
findings folded into v2:

- **F1** the v1 "run E0 after Batch H" sequencing was inert (H lives in the
  disjoint daisy family; E0's pmf-family fixtures carry no sojourn term) and
  double-booked the daisy-side measurement with Deferred 2's §1 value test.
  E0 rewritten: H-independent, runs after E1 (its oracle source), daisy-side
  measurement cross-referenced to Deferred 2 (§4-E0, §6).
- **F2 / cross-F1 / cross-F2** the "literally the same primitive" claim
  retagged as a [reasoned] working hypothesis with the readout/seed deltas
  stated; E4 ownership fixed (whichever unit activates first, note written
  against both plans' requirements regardless); the λ-policy made a single
  shared E4 decision instead of a pre-committed kwarg (§2, §3-A2, §4-E4).
- **F3** the §2/§5-B2 API-shape disagreement resolved as an explicit open
  decision (new entry point vs. kwarg-on-existing, each with its no-modify
  implication) (§5-B2).
- **F4** the 512-floor qualification added (auto-λ θ-independent below max
  rate 512 — a demonstration fixture must cross it); F-004→F-005
  attribution corrected (§2).
- **F5** the Batch-0-skeleton claim re-hedged as [reasoned]; PDF §8(b)'s
  requested two-line note preserved (§6).
- **F6** E1's scope sentence de-ambiguated (linear-only, param+constant
  edges, continuous-only pending risk 3) (§4-E1).
- **F7** the no-modify matrix row made conditional on the greenfield
  assumption (§6).
- **cross-F8** the rate-blowup "established as desirable" register inverted
  to match CLAUDE.md (a *violated* property adopted here as a requirement)
  (§2).
- **cross-F11** shared JAX reference machinery with Deferred 2's E3 (§4-E1).
- Review addition: the route-(ii) density off-by-one (`π_{k+1}` alignment)
  named as an explicit E3/E5 derivation item (§4-E3).

Survived attack (review-verified at source): the central "PMF term is FD
everywhere and under every master-plan batch" claim (verbatim "keeping FD
for pmf" at `__init__.py:7613-7617`; joint-index's sojourn adjoint is a
different, non-time-indexed term); route (ii)'s dλ/dθ=0 elimination argument
(term-by-term against the DP code); all route-(i) facts (no granularity
kwarg; hardcoded call sites; `Graph.pdf` accepts explicit granularity; the
special t=0 step); the four-defect do-not-trust baseline; E0's isolability;
discrete scoping; process compliance (every experiment reachable via public
API).
