# Deferred-1 de-risk findings (hierarchical/SCC two-level adjoint)

**Plan:** `deferred-1-hierarchical-scc-adjoint-plan.md` §5 (E0-E4).
**Branch:** `derisk/d1-scc-adjoint`, base master `7371a369`.
**Prerequisite state at execution (2026-08-15):** P1 satisfied (Batch 0
core shipped, now five kinds); P2 satisfied (Batch C's binp exit IS the
output-side pre-contraction surface; the input-side cotangent-seed
entry remains future work per the Batch-0 decision — the seeding block
stays comment-marked and orthogonal); P3 open (user decision at
activation; default recommendation P3(a) E[T]-class scope).

## E4 — reentrancy survey (static read; COMPLETE)

The forward composer parallelizes `ptd_compose_scc_one` over SCCs
within a topological LEVEL (`scc_compose.c:517`); levels serial.
Shared state: `parent_result` (per-SCC-owned slots + earlier-level
reads — safe by the level barrier), per-worker err slabs, a `__thread`
TLS reentrancy guard with a documented per-worker bump, atomic
telemetry. **Decisive finding: the adjoint's source-first traversal is
ORDER-INCOMPATIBLE with the forward sink-first loop — a separate pass
over retained per-SCC artifacts is structurally forced** (the plan's
"separate serial pass first" default is not a preference but the only
shape). Retention must capture the tape built by the INNER
`ptd_expected_waiting_time` call inside `ptd_compose_scc_one`
(destroyed per-call today).

## E2 — landmine demonstration + guard design (COMPLETE)

`experiments/dr_d1_e234_landmine_conditioning_reentrancy.py`:
- **Demonstrated:** `SCCVertex.as_synthetic_graph()` yields a
  first-class `Graph`; the shipped `_moments_grad_theta(1)` ACCEPTS it
  and returns a full-size plausible Jacobian (`J=[-1.0, 0.0]` on the
  two-SCC fixture) contracted from the Type-A/phantom PLACEHOLDER
  coefficients. The true phantom weight under hierarchical semantics
  (1/parent_result[target] = 0.2857 on the fixture) is θ-dependent
  through the PARENT — unrepresentable by any linear contraction of
  placeholder coefficients. `update_weights` on the synth graph
  silently re-derives every parameterized edge as c·θ, overwriting
  compose-injected semantics.
*(Dated correction 2026-08-16: this section describes the guard as a
DESIGN awaiting approval and specifies an INFO log. Both are
superseded — the guard SHIPPED on 2026-08-16 (user-approved
2026-08-15), logging at **WARNING** (deliberately upgraded: this
decline is always-misuse and the FD fallback produces the same wrong
numbers, so it must be visible at the default level), with the marker
also re-applied across the distributed serialize/deserialize boundary
(`distributed_scc.deserialize_scc_synth` — a shipped synth serializer
the guard plan had wrongly assumed did not exist; found by G4 refuter
A). Pinned by `tests/pytest/test_synthetic_scc_guard.py` (8 tests).
Record: `b3-d1-e2-guard-plan.md`.)*

- **Guard design (as originally drafted — see the dated correction
  above for what actually shipped):** a `synthetic` marker on `ptd_graph` set
  at `as_synthetic_graph` creation; ONE decline check at the top of
  `ptd_b3_moments_core` (covers all five kinds in one site) + the two
  sojourn entries; INFO log naming the two-level-adjoint requirement.
  Pinned-test-to-be: E2(a) flips to declined. Worth shipping
  REGARDLESS of whether Deferred 1 activates (protects users from
  mis-wiring A/B/C-era gradients onto synth graphs).

## E3 — per-SCC vs whole-graph conditioning (COMPLETE; risk 13a INVERTED)

On the engineered fixture (cross-SCC rate ratios 1e12/1e14, benign
within each SCC): the WHOLE-graph gate condition stays ~1e1 (benign —
the monolithic tape does not mix the scales into its conditioning
statistic, consistent with the D4 sweep's large-θ findings), while the
per-SCC synthetic graph's gate condition explodes to **1e23/1e28**
(the phantom weight 1/parent_result imports the parent's scale into
the SCC's tape). **Master risk 13a asked whether a per-SCC gate
UNDER-detects; the measured direction is the OPPOSITE — a per-SCC gate
would OVER-decline through phantom-weight scale mixing.**
*(Post-guard annotation, 2026-08-15: this measurement PREDATES the
E2 guard and is no longer reproducible on a guarded build — the
per-SCC bisection called `_moments_grad_theta` on synthetic graphs,
which now decline at any threshold by design. The experiment's per-SCC
arm is scoped PRE-GUARD HISTORICAL; the numbers above are the record.)* A future
two-level adjoint's decline design must therefore gate on the
whole-graph statistic (or a phantom-excluded per-SCC statistic), not
naive per-SCC condition numbers.

## E0 / E1 — running (agent-executed; results appended on completion)

## E1 — two-level adjoint reference vs oracles (agent-executed; COMPLETE, GO)

`experiments/dr_d1_e1_twolevel_adjoint.py` (commit `739147dc`). All 6
`toy_model.BUILDERS` × 4 θ: value parity vs the REAL C hierarchical
path 0..8e-16; dense random-cotangent VJP oracle (5 seeds × 4 θ per
fixture) ≤2.4e-15; shipped e_start binding ≤8.2e-16; full assembled
Jacobian vs FD-of-the-C-composer ≤7.3e-10 (FD noise floor). The four
plan questions: (i) per-SCC cotangent-seeded reverse IS the stage-1
mechanism with a different seed (math-level confirmation; the C entry
point for non-one-hot seeds remains the P2 exit to build); (ii) one
parent-length cotangent accumulator suffices — parent vertex t's
cotangent is consumed by the unique SCC containing t, fed by the
caller seed + one phantom-rule routing per upstream channel,
source-first order guaranteeing completeness; (iii) NOTHING besides
Type-C/phantom weights couples SCCs (generic-FD-vs-composer test);
(iv) Type-A placeholders receive STRUCTURALLY zero adjoint (the
source's M-column is diagonal-only) — perturbation-verified exactly
0.0 at three placeholder values; a nonzero Type-A adjoint can only be
a bookkeeping bug and should hard-error.

**Findings the C design must carry:** the 1e300 clamp branch fires on
EVERY fixture (1-4×) — it is the NORMAL absorption mechanism, its
cotangent-drop exactly right; starting-vertex edges are θ-CONSTANT
despite carrying coefficient arrays (update_weights skips the start
vertex) — the outer pass must never contract start-edge coefficients
(dθ[0] ≡ 0 doubles as a pollution sentinel); the start vertex accrues
no holding time and Type-A synth edges are start edges of the synth
graph (never rescaled); PYBIND FOOTGUN: `scc_decomposition()` on a
temporary dangles (borrows the parent) and compose on a dead parent
returns plausible garbage silently — a candidate guard/doc item;
non-singleton start SCCs would need the surrogate-source rule; no
trap-vertex toy fixture exists (the known ledger gap).

## E0 — scale/necessity measurement (COMPLETE at safe sizes; gate A1 ANSWERED)

Harness `experiments/dr_d1_e0_scale.py` (commit `53c21f4e`); serial
per-cell subprocesses, per-op 120s time-box; full table in the
harness's `d1e0_results.json` (scratch). Two-locus ARG ladder
(vertices / param edges): n=6 → 1,044 / 8.1k; n=8 → 8,407 / 92k;
n=9 → 22,653 / 286k; n=10 → 59,522 / 852k.

**Measured (n=8, the largest safely-measured size):**
- primal monolithic: 0.68s build + 0.061s op, peak RSS 3.5 GB
- primal hierarchical: 0.40s + 0.44s, peak RSS **1.6 GB** — HALF the
  monolithic memory (the decomposition's memory benefit is real at
  this scale; wall-clock comparable)
- **exact-gradient pipeline (`_moments_grad_theta(1)`): 96.5 s,
  RSS 3.7 GB** (dyn-ordering variant 96.1 s — no rescue) — ~1,500×
  the primal op cost at n=8
- Stage-A2 cache changes nothing for the gradient (its tape is
  private per call, as the plan stated).

**The wall (forensic, from the interrupted first run — deliberately
NOT re-measured):** at n=10 the gradient/tape pipeline wrote a
**≥2.5 GB partial on-disk tape** and consumed ~50 GB RAM before the
user had to kill the session. n=9 was skipped in the completion run
by caution (extrapolation: tens of minutes, 10-20 GB class).

**Gate A1 verdict: a forcing model EXISTS for the GRADIENT pipeline**
— two-locus models in the n≈2e4-6e4-vertex range (nr_samples 9-10,
squarely production-relevant for this library's domain) are
effectively excluded from exact gradients by the monolithic tape's
memory/time scaling, while the hierarchical PRIMAL demonstrably halves
memory at n=8. The primal alone shows no hard wall at measured sizes
(mono still fits at n=8; n=10 unmeasured for the primal). A1 is
therefore satisfied in the strong form the plan named: "a model whose
primal fits monolithically can still OOM/timeout in the gradient
function."

One cell error recorded: `twolocus:8 hier-omp1` errored (single-thread
OMP variant; not investigated — does not affect the A1 verdict).

## Re-evaluation checkpoint OUTCOME (user-decided 2026-08-15)

Presented as the consolidated three-unit decision round (with D2/D3):

- **ACTIVATED.** Write the full implementation plan at P3(a)
  E[T]-class scope (plan -> adversarial review -> sign-off before any
  code), per the plan's re-evaluation checkpoint.
- **E2 guard APPROVED for shipping** as an independent micro-batch
  (modifies shipped code with explicit user approval): the `synthetic`
  marker on `ptd_graph` + one decline check at the top of
  `ptd_b3_moments_core` + the two sojourn entries; pinned test = E2(a)
  flips to declined; normal gate ladder + adversarial diff review.
