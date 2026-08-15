# Deferred-2 de-risk findings (daisy-chain intermediate-epoch exact gradient)

**Plan:** `deferred-2-daisy-intermediate-epoch-plan.md` §3 (gates
A1-A3), §4 (E1-E4).
**Branch note:** run serially on the consolidated branch
`derisk/d1-scc-adjoint` (post-50GB-incident memory-safety mandate; the
plan named `derisk/daisy-intermediate-exact`) — recorded deliberately.
**Artifacts:** `experiments/dr_d2_a2_value_test.py`,
`experiments/dr_d2_e1_lambda_study.py` (2026-08-15).

## Headline: gate A2's answer is two-sided — at benign θ the remaining intermediate-epoch FD error is NEGLIGIBLE (~6e-8), but at mixed-scale θ the shipped backward does not merely lose accuracy, it CRASHES (absolute-eps FD probes θ below zero → FFI rate-validation raise). D2's value proposition is small-θ ROBUSTNESS, not benign accuracy.

## A2 — the §1 value test (COMPLETE)

Fixture: the `test_lrt_at` coalescent class at nr=3 (JSP 56 vertices,
n_ipv 36), 2 epochs, `final_read='sojourn'` (the shipped default),
granularity 2048, flat θ = [e0_th, e0_mu, e1_th, e1_mu]. Reference =
Richardson-extrapolated RELATIVE-step central differences of the
production primal (valid wherever the primal is smooth; stated
limitation: still FD — at truly pathological θ both estimators fail
together; the mixed point here is the moderately-mixed regime).

**Benign (θ0=1e-4):** shipped-FD per-slot relative error —
e0_th 5.68e-08, e0_mu 3.90e-08, e1_th 1.23e-07, e1_mu 7.96e-09.
The INTERMEDIATE-epoch slots (worst 5.7e-8) are no worse than the
final-epoch slots; FD error is uniformly ~1e-7-class. At benign scale
the plan §1's "final-epoch exact term already removes the practical
defect" clause is effectively met — there is no practical defect left
to remove *at this scale*.

**Mixed-scale (θ0=1e-8):** the reference computes fine (gradient spans
1.7e-4 .. 7.7e7 — a genuinely stiff point), but the shipped
`jax.grad` RAISES:
`DaisyChainSojournFfiImpl: theta produced a NEGATIVE transition rate
(most negative edge weight: -0.000000)`. Mechanism (grounded): the
shipped backward is absolute-step FD with eps=1e-7
(`src/phasic/ffi_wrappers.py:1296` for the public primitive;
`src/phasic/__init__.py:5060` `eps_local = 1e-7` for the svgd model
sites) — its probe θ−eps goes negative whenever any slot sits below
1e-7, and the FFI's negative-rate validation (correctly) refuses.
Production consequence: with `exact_final_grad=True` the final block
is exact, but every EARLIER epoch's slots still use these full-chain
absolute-eps FD probes — a particle whose intermediate-epoch rate
drops below ~1e-7 kills the fit with this raise (or, pre-Batch-H,
kills any multi-epoch gradient at that θ). An exact intermediate-epoch
gradient removes the probe entirely.

Debugging record (for honesty about the session's path): the raise was
first misattributed to the forward and to a −0.0 sojourn-structure
coefficient; direct probes showed both `sojourn` and `stopprob`
forwards succeed at these θ, and `-0.000000` is the `%f`-printed tiny
negative (−θ_eps·coeff ≈ −2.7e-7) from the backward's probe. The
committed test records the crash as a measured outcome (try/except)
rather than dying on it.

## E1 — granularity/λ study (COMPACT run; item 3's policy weigh-in below)

`dr_d2_e1_lambda_study.py`, same fixture:

- **Item 1 (λ variation):** max exit rate over θ0 ∈ {1e-8 .. 1e2}
  ranges 1 .. 300 — NEVER above the 512 floor, so auto-granularity is
  CONSTANT (1024) across the whole prior-scale grid: on
  coalescent-scale models the θ-dependent-DTMC-identity risk (plan
  §2.1) is already neutralized by the floor. It engages only when
  rates exceed 512 (θ0 ≳ 1e2 here). Live-SVGD-trajectory recording
  remains un-run (scoped as remaining E1 work).
- **Item 2 (pinned-λ value error):** vs a g=524288 reference, the
  benign-θ primal's relative error is FLAT at ~1.04e-11 for every
  g ∈ {512 .. 65536} — already converged at the floor; pinning costs
  ~1e-11, far below any fit's statistical noise (plan §7 risk 2
  satisfied on this fixture class).
- **Item 3 (policy):** the measurements favor option (ii)
  (construction-time pin from a probe θ + margin, loud raise on
  violation) — near-free on this model class because the floor already
  dominates; the loud path still must be BUILT (the F1 gap: today the
  daisy FFI swallows the C granularity-violation into an unlogged NaN
  row). Failure semantics constrained to raise / host-side-FD /
  per-call-dispatch per the D6 record (plan F4). Final decision at
  activation sign-off; recorded as the SINGLE shared λ-policy in the
  E4 note.
- **Item 4:** untouched-value-path confirmation is deferred to
  implementation gates (B4 byte-identity), as planned.

## E4 — shared-primitive + λ-policy note (COMPLETE via lapse)

Deferred 3's de-risk SELECTED route (ii) (Poisson mixture) ⇒ the
shared-tangent-stepper premise LAPSES (cross-plan F2). The full joint
note — no-sharing statement + the shared λ-pinning evidence and
recommendation from BOTH units' measurements — lives in
`b3-d3-derisk-findings.md` §E4 (written once, referenced here per the
one-page-note deliverable). If D2 activates, its B1 stepper is built
against D2's requirements alone.

## E2 / E3 — remaining de-risk work (scoped honestly; NOT done)

- **E2 (cost model):** n_ipv/k/n instrumentation + forward-mode vs
  checkpointed-reverse decision table, including a large-n_ipv
  two-locus-scale case. Note the A2 fixture gives first data points:
  n_ipv=36, k ≈ g·dt = 1024 (auto) per epoch — forward-mode seed count
  (n_params + n_ipv ≈ 38) looks tractable at this scale; the decision
  table still needs the large case.
- **E3 (end-to-end differentiable oracle, 5 gates):** the JAX
  reference machinery exists in spirit in D3's committed Euler
  reference (`dr_d3_e123_pdf_routes.py` — stepping semantics already
  parity-gated to 0..6e-15 against production `pdf`), but the daisy
  chain's collapse/projection/epoch composition and the
  exact-VJP × `custom_vmap` composition probe (E3(v) — the actual hard
  JAX problem) are UNBUILT. Gate 4 (share with D3) lapsed with
  route (ii).

## Activation-gate status

- **A1** (Batch H shipped + reviewed): SATISFIED (`ecd708fc`).
- **A2** (intermediate-epoch FD still a real problem after H):
  **SATISFIED in the robustness sense** — crash at sub-eps θ slots;
  NOT satisfied in the benign-accuracy sense (~6e-8 residual). The
  checkpoint decision should weigh how often production particles
  visit sub-1e-7 rate scales (the B3 program's pinned mixed-scale
  defect test says the regime is real).
- **A3** (de-risk complete + the two open decisions signed off):
  **PARTIAL** — λ-policy evidence in hand and recommendation written;
  forward-vs-checkpointed-reverse undecided (needs E2's large case);
  E3 oracle unbuilt.

## Re-evaluation checkpoint OUTCOME (user-decided 2026-08-15)

**GO — finish the de-risk**: run the remaining E2 cost model
(including a large-n_ipv case, sized cautiously per the memory
mandate) and the E3 end-to-end JAX oracle with the E3(v)
exact-VJP × custom_vmap composition probe; then the build-vs-park
decision returns to the user with A3's two open decisions. The
relative-step-FD stopgap and park options were declined.
