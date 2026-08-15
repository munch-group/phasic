# Deferred-3 de-risk findings (PDF-at-time-t exact gradient)

**Plan:** `deferred-3-pdf-gradient-revival-plan.md` §4 (E0-E5).
**Branch note:** all three deferred units' de-risk artifacts were run
serially on the SINGLE consolidated branch `derisk/d1-scc-adjoint`
(memory-safety mandate after the 2026-08-15 50GB incident: no parallel
heavy agents, one working tree). The plan named
`derisk/pdf-gradient-revival`; the consolidation is deliberate and
recorded here.
**Artifact:** `experiments/dr_d3_e123_pdf_routes.py` (commit
`1b2de9fd`); all gates re-verified 2026-08-15.

## Headline: route (ii) — Poisson mixture at pinned λ — SELECTED; the plan's provisional route-(i) recommendation is REFUTED by measurement

## Production stepping semantics (grounded from source before any probe)

Read from `phasiccpp.h` / `phasic.c` (the E2 reference was built against
these, then gate-verified): a special INSTANTANEOUS first step
redistributes the IPV (the `priv3=1` creation-time probe step); each
subsequent step applies `p[to] += p[from]·w/g` (the embedded DTMC
`P = I + Q/λ`); the pdf readout is `absorbed_mass·g`; production
`_pdf[0]` duplicates step 1; auto granularity is `2·max(512, max
exit rate)`.

## E1 — oracle battery (COMPLETE)

Closed forms: Exponential, Erlang-3, hypoexponential-2, plus a cyclic
4-vertex fixture (no closed form; cross-route consistency used there).
The cyclic fixture needed `add_aux_vertex_constant(0.15)` — mixed
scalar/parameterized edges are refused by construction.

## E2 — route (i) probe (COMPLETE; quantitative case AGAINST)

JAX Euler reference vs production `g.pdf` at matched g=1024 — parity
PASS on all four fixtures: expo 0.0, erlang3 2.57e-16, hypo2 0.0,
cyclic4 5.94e-15 (tol 1e-10; the `_pdf[0]`-duplicates-step-1
convention had to be modeled — an initial rel~1e-3 mismatch was this
off-by-one, fixed as `series[max(idx-1, 0)]`).

Route (i) = differentiate the Euler stepper as-is. Its gradient bias
vs the Exponential closed form decays only O(1/λ):

| λ | value bias | grad bias |
|---|---|---|
| 1024 | 1.71e-03 | 1.43e-02 |
| 4096 | 2.37e-04 | 1.26e-03 |
| 16384 | 1.07e-04 | 8.92e-04 |

At practical λ the gradient bias (~9e-4) sits ORDERS above FD-class
error (~5e-8) — route (i) would replace FD with something *less*
accurate. REFUTED.

## E3 — route (ii) probe (COMPLETE; ALL GATES PASS)

Density alignment (the review-mandated explicit statement):
`f(t) = λ·Σ_k Poisson(k; λt)·π_{k+1}` — absorption mass harvested at
step k+1. The deliberately mis-aligned variant errs by 2.07e-02 — the
closed-form gate CATCHES the off-by-one class it was constructed for.

- expo: value rel 1.63e-11, gradient rel 1.79e-10 (dλ/dθ=0 ⇒ the
  gradient is exact through the π_k products alone);
- erlang3: value 7.46e-11, gradient 1.36e-11;
- cyclic4: vs high-λ route-(i) reference 4.88e-05 (tol 1e-3);
  gradient vs FD-of-the-mixture 2.04e-10.

Failure mode characterized: λ below the max exit rate produces
NEGATIVE intermediate probabilities and a silently corrupt value
(measured: λ=2 vs rate=10 → value 0.01171, wrong). DETECTABLE by a
`p < 0` check inside the stepper — **the implementation must add that
check as the loud path** (plan §4-E3's "must be detectable, loud").

## E4 — shared-primitive + λ-policy note (COMPLETE via the lapse clause)

Route (ii) is selected ⇒ per cross-plan F2 the D2/D3
shared-tangent-stepper premise LAPSES: route (ii) differentiates the
Poisson-mixture weights and the π_k harvest sequence — no
tangent-propagating stepper exists in its design, so there is nothing
for Deferred 2's θ+IPV-seed/state-vector requirements to share.
**No-sharing note: if Deferred 2 activates, its B1 stepper is built
against its own requirements alone; no joint primitive is owed to
Deferred 3.**

The SINGLE shared λ-pinning policy decision (both plans' E4 item (b)) —
evidence now on the table from both sides:
- D2-E1 (compact, `experiments/dr_d2_e1_lambda_study.py`): on the
  coalescent-scale JSP fixture the auto-granularity FLOOR (g=1024)
  binds across the entire prior-scale grid θ0 ∈ [1e-8, 1e2] (max exit
  rate ≤ 512 everywhere) — the DTMC identity is already effectively
  pinned on this model class — and the pinned-λ value error is flat at
  ~1e-11 from g=512 up.
- D3-E3: route (ii) REQUIRES a pinned λ by construction; violation
  (λ < max rate) is loudly detectable via the p<0 check.

Joint recommendation (final decision at activation sign-off, not
here): construction-time pin from a probe θ with a margin factor —
which on coalescent-scale models coincides with the existing floor and
costs ~1e-11 in value — with loud raise on violation (D2 side: the
`<=1.0001` validation escalated InvalidArgument-style instead of the
current NaN-row swallow, plan F1; D3 side: the p<0 stepper check).

## Remaining de-risk work (scoped honestly; NOT done)

- **E0 — value measurement on real SVGD fits** (how much of the total
  gradient error is the FD PMF term, benign + mixed-scale; parks the
  unit if immaterial). Requires real fits; deferred to the activation
  decision.
- **E5 — chain-rule re-derivation dossier** for route (ii) with
  term-zeroing checks. This is the input to any implementation plan.

## Activation-gate status

- A1 (user confirms PDF-term exact gradients are wanted): **OPEN — the
  checkpoint question.**
- A2 (de-risk complete + route chosen + E4 note): route chosen (ii),
  E4 note above; E0/E5 outstanding ⇒ **PARTIAL**. If the user says GO,
  E0+E5 are the next de-risk work; if PARK, this document plus the
  committed experiment are the complete record.

## Re-evaluation checkpoint OUTCOME (user-decided 2026-08-15)

**GO — finish E0 + E5**: run the E0 value measurement on real SVGD
fits (the explicit park-if-immaterial test) and write the E5
chain-rule re-derivation dossier for route (ii); then the build-vs-park
decision returns to the user on that evidence.
