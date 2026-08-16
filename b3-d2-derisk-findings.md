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

## E2 — cost model (COMPLETE 2026-08-16, post-checkpoint GO): CHECKPOINTED-REVERSE WINS, decisively — the plan's forward-mode lean is REFUTED

`experiments/dr_d2_e2_cost_model.py`, staged coalescent JSP ladder
(builds ≤0.2 s, no gradient pipeline constructed — structure only):

| nr | n (JSP) | n_ipv | k = λ·dt | seeds = P+n_ipv | 2√k | winner |
|---|---|---|---|---|---|---|
| 3 | 56 | 36 | 512 | 38 | 45.3 | forward |
| 4 | 232 | 164 | 512 | 166 | 45.3 | **checkpoint** |
| 5 | 936 | 676 | 512 | 678 | 45.3 | **checkpoint** |
| 6 | 3832 | 2804 | 512 | 2806 | 45.3 | **checkpoint** |

Memory (doubles): forward O(n·seeds) 2.1e3 → **1.08e7**; checkpointed
O(n·√k) 1.3e3 → **8.7e4**; naive reverse O(k·n) 2.9e4 → 1.96e6.

**The mechanism:** `n_ipv` grows ~4× per added sample (36 → 164 → 676
→ 2804) while `k` is CONSTANT at 512 — because λ is pinned at the
auto-granularity floor (E1's finding) and `k = λ·dt` therefore does
not grow with the model at all. Forward-mode's cost scales with the
seed count (which is n_ipv-dominated); checkpointing's scales with
√k (fixed). So the crossover is passed at **nr=4** — the A2 toy
fixture (nr=3) is the ONLY size where forward-mode wins, and by
nr=5 forward-mode's memory exceeds even NAIVE reverse.

**Decision A3(ii) — forward-mode vs checkpointed-reverse — ANSWERED:
checkpointed-reverse, for every production size.** (The plan
recorded forward-mode as attractive because it is history-free; that
reasoning holds only where the seed count is small, which on this
model family means the toy fixture alone.) Consequence for any
implementation plan: B1's "tangent stepper" shape is the WRONG
primitive — the primitive to build is a checkpointed reverse stepper
(~2× compute, O(n·√k) memory).

## E3(v) — the exact-VJP × custom_vmap composition probe (COMPLETE 2026-08-16): **GO — the blocking risk is RETIRED**

`experiments/dr_d2_e3v_vmap_composition.py`. The plan called gate (v)
"the actual hard JAX problem", and it is the one that could have
blocked the unit regardless of the C math: production's FD backward
gets batching for free (its probes re-enter the `custom_vmap`'d core),
and Batch H's exact final-epoch term dodges the question a second way
(its block is a construction-time numpy CONSTANT that vmap simply
broadcasts). An exact INTERMEDIATE-epoch gradient can do neither — it
is θ-dependent per particle, so it must be a PER-CALL host callback
inside the backward under `vmap(grad(loss))(particles)`, which nothing
in the shipped codebase does today.

The probe rebuilds the production skeleton exactly (`custom_batching.
custom_vmap` core with a 1-D path + a fused batched rule, `custom_vjp`
on top, a numpy-only "FFI" that ASSERTS it never receives a 3-D
buffer) and swaps the FD backward for an exact `jax.pure_callback`
adjoint. All five probes PASS:

| probe | result |
|---|---|
| P1 `vmap(grad(loss))` == analytic truth | **0.00e+00** (bitwise) |
| P2 adjoint callback sees ONE particle/call | ndim==1, 3 calls / 3 particles |
| P5 forward stayed 2-D + FUSED rule fired | only shape seen: (12, 3) = 3 particles × 4 unique |
| P3 `jit(vmap(grad(loss)))` == truth | **0.00e+00** |
| P4 shipped-style FD backward (sanity) | 4.13e-11 |

**Design note the implementation must carry:** the per-call adjoint
callback needs `vmap_method='sequential'` on `jax.pure_callback` —
that is what makes the callback observe one particle at a time (P2)
instead of a batched buffer it would have to unpack. With it, the
composition is not merely workable but EXACT and jit-clean.

## E3 gates 1-3 — remaining work (scoped honestly; NOT done)
- **E3 (end-to-end differentiable oracle, 5 gates):** the JAX
  reference machinery exists in spirit in D3's committed Euler
  reference (`dr_d3_e123_pdf_routes.py` — stepping semantics already
  parity-gated to 0..6e-15 against production `pdf`), but the daisy
  chain's collapse/projection/epoch-composition ORACLE (gates 1-3:
  value parity vs production at matched pinned λ; grad == FD of the
  reference at benign θ; FD-vs-oracle divergence at mixed scale) is
  UNBUILT. Gate 4 (share with D3) lapsed with route (ii); **gate 5
  (E3(v)) is DONE and GO — see the section above**. Gates 1-3 are
  oracle-CONSTRUCTION work whose decision value has already been
  delivered by A2/E1/E2/E3(v), so they belong to the implementation
  plan's first gate — and their spec is fully grounded below.

**E3's reference SPEC is now grounded (2026-08-16)** — the remaining
work is implementing it, not discovering it. Read from
`src/cpp/parameterized/graph_builder_ffi.cpp:2047-2095`
(`DaisyChainSojournFfiImpl`, the shipped `final_read='sojourn'` path
SVGD actually uses), the per-epoch loop is exactly:

1. `ptd_graph_update_ipv(gj, ipv_work, n_ipv)`;
2. `ptd_graph_update_weights(gj, theta_epoch, param_length, false)`;
3. negative-rate check (`most_negative_edge_weight < kMinEdgeWeight`
   ⇒ the whole batch element NaN-fills — this is the A2 crash's
   forward twin);
4. `raw = gj->stop_probability(dt_epoch, granularity)`;
5. **collapse**: for each non-aux vertex v, `p = raw[v] +
   raw[t_to_aux[v]]` when v has an aux partner, written to
   `collapsed[collapsed_pos[v]]` — a pure pair-sum + reindex, NO
   renormalization (confirming the plan's linearity assumption at
   source);
6. **gather**: `ipv_work[k] = collapsed[collapsed_pos[
   ipv_target_indices[k]]]`.

Final epoch: gather `sojourn_ipv[k] = ipv_work[sojourn_jsp_gather[k]]`
(mass summed for the Batch-H handoff), then update_ipv/update_weights
on the sojourn graph and the granularity-free sojourn read. Every
index map (`aux_set`, `t_to_aux`, `collapsed_pos`,
`ipv_target_indices`, `sojourn_jsp_gather`) is θ-independent and
computed at model-build time — so the collapse/projection Jacobian is
the fixed linear map the plan assumed, now VERIFIED at source rather
than reasoned.

## Activation-gate status

- **A1** (Batch H shipped + reviewed): SATISFIED (`ecd708fc`).
- **A2** (intermediate-epoch FD still a real problem after H):
  **SATISFIED in the robustness sense** — crash at sub-eps θ slots;
  NOT satisfied in the benign-accuracy sense (~6e-8 residual). The
  checkpoint decision should weigh how often production particles
  visit sub-1e-7 rate scales (the B3 program's pinned mixed-scale
  defect test says the regime is real).
- **A3** (de-risk complete + the two open decisions signed off):
  **NEARLY COMPLETE (2026-08-16)** — both open decisions now have
  evidence-backed answers: (1) λ-policy = construction-time pin +
  loud raise (near-free on this model family, E1); (2) mode =
  CHECKPOINTED-REVERSE, measured, crossover passed at nr=4 (E2). The
  BLOCKING JAX risk (E3 gate 5, exact-VJP × custom_vmap) is also
  retired GO. Only E3 gates 1-3 (the oracle itself) remain unbuilt —
  oracle-construction work whose decision value has already been
  delivered, and whose spec is now grounded from the FFI source, so
  the implementation plan owns it as its first gate.

**D2 DE-RISK BOTTOM LINE (2026-08-16): every question that could
change the DECISION has been answered.** Build case = the shipped
multi-epoch backward CRASHES at sub-1e-7 θ slots (A2) and the exact
path removes the probe entirely; shape = checkpointed-reverse (E2),
λ pinned at construction (E1), per-call adjoint callback with
`vmap_method='sequential'` (E3(v)); benign-scale accuracy gains
nothing (A2). The build-vs-park decision is ready for the user.

## Re-evaluation checkpoint OUTCOME (user-decided 2026-08-15)

**GO — finish the de-risk**: run the remaining E2 cost model
(including a large-n_ipv case, sized cautiously per the memory
mandate) and the E3 end-to-end JAX oracle with the E3(v)
exact-VJP × custom_vmap composition probe; then the build-vs-park
decision returns to the user with A3's two open decisions. The
relative-step-FD stopgap and park options were declined.
