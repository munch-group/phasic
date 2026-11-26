# Deferred-4 Phase 0 — conditioning-sweep findings (CC-2)

**Design-of-record:** `deferred-4-mpfr-conditioning-floor-plan.md` §2
(adversarially reviewed 2026-08-11; F1-F10 folded). **Executed
2026-08-15** on branch `derisk/d4-mpfr-sweep`, master base `a31d76cb`
(post-G.2 — the full five-kind core; every planned B3 batch shipped).
**Artifacts:** `experiments/dr_d4_exact_oracle.py` (the oracle),
`experiments/dr_d4_oracle_calibration.py` (the trust bar),
`experiments/dr_d4_phase0_sweep.py` (the sweep; row dump in the session
scratchpad, tallies reproduced below).

## Verdict: **CLEAN → PARK** (the decision tree's first branch)

No GAP point exists anywhere in the sweep: at every point where a
shipped moments-family exact-gradient function returns a Jacobian, it
matches an **exact-rational** oracle to 1e-16..1e-9; every decline is
gate-driven, happens BEFORE any corruption, and reaches the user as a
logged FD fallback. Phase 1 (a gradient-specific decline metric) has no
evidence to justify it; Phase 2 (full MPFR adjoint) remains "expected
never".

## One deliberate substitution vs the design-of-record

The plan named an mpmath 200-bit dense oracle; **mpmath is not in the
pixi env**, and adding a dependency would violate the no-changes rule
in spirit. Substituted: **exact rational arithmetic**
(`fractions.Fraction`) — INFINITELY precise for rational inputs
(strictly stronger than 200 bits), zero new dependencies, equally
tape-independent. Exactness covers linear, log (the WEIGHT is a plain
rational product; the library's log-space evaluation is an
implementation detail), rewards, and rational formulas; transcendental
weights would fall to the plan's float64-reference fallback clause
(none were needed for this sweep's scope: the three moments functions).

## Calibration (the §2.1 trust bar — required before any disputed point)

Oracle vs shipped primal AND shipped exact Jacobians on benign
fixtures: chain2 / branchy / cyclic linear **2.8e-16 .. 6.3e-16**;
log **exactly 0.0**; rewards slice **2.0e-16 / 1.3e-16**; vs
`jax.jacobian` of an independent float64 dense reference **1.0e-17**.
Bar (~1e-13) beaten by three+ orders everywhere.

## Sweep (64 points; HAVE_MPFR=True verified by force-probe)

Axes: 5 fixtures (chain2, branchy, cyclic [the DR-A class],
coalescent3 [Kingman-style], log2) × θ ratio paths 1 → 1e-14 (small
component) and 1e2 → 1e8 (large component) × K=1..3 per call + a
rewards slice (branchy/cyclic × benign + mixed θ — the Batch-A
follow-on the plan scheduled).

**Tallies: correct=56 · declined=7 · degraded=0 · GAP=0.**

- **Exact-path accuracy where it returns:** 1e-17..1e-15 in every
  benign and small-component regime; worst observed anywhere:
  1.6e-9 (cyclic, θ=[1e8, 1]) and 9.4e-11 (branchy, θ=[1e8, 1]) —
  at those same points FD is 29-100% WRONG.
- **FD context (is decline-to-FD lossy?):** at the 7 declined points
  FD's error is 4e-8..7e-7 (mildly lossy, usable); in the LARGE-θ
  direction FD catastrophically fails (0.3–1.0 relative at θ₀≥1e6)
  while the exact path stays correct AND the gate correctly does NOT
  fire (cond ≈ 1e1 — the failure is FD's step-size problem, not tape
  conditioning; the exact path is the cure, no gate involvement
  needed).
- **Gate-metric adequacy (master risk 9):** the bisection-recovered
  condition number tracks the θ ratio cleanly (cond ≈ ratio⁻¹·10);
  every point below the default threshold (1e12) is CORRECT — the
  metric's ROC against GAP points is vacuously perfect (no GAP
  exists). The declines begin exactly at cond > 1e12.
- **Gate conservatism quantified (the H-batch question, now with an
  exact oracle):** lifting the threshold at all 7 declined points, the
  exact path is STILL correct to **1e-16..6.9e-14** — the gate is
  conservative by 3-4 decades on the moments family for these
  fixtures. (Batch E's 34-144% lifted-gate errors live on the SOJOURN
  function family — excluded from this sweep per the CCF Job B scope
  — and are not contradicted by this finding.)
- **No-silent-fallbacks verified end-to-end (classification (b)):**
  the raw-wrapper sweep necessarily sees no logs (the logging lives in
  the model layer — recorded as a sweep artifact, not a defect); the
  model-level probe at a declined point produced exactly ONE INFO
  decline line and a finite FD-backed gradient.

## Proposed deliverables requiring user sign-off (the no-code rule)

1. **Regression pin test** (proposed, not shipped): a pytest cell
   pinning (i) exact==oracle-grade at a benign + a mixed-scale point
   (via a frozen oracle value), (ii) the decline at θ=[1,1e-12]-class
   WITH its INFO line, (iii) skip/xfail on non-MPFR builds (the gate
   is inert there — a CLEAN assertion would be meaningless).
2. **CLAUDE.md note** (proposed): the B3 known-gaps section's
   conditioning items can now cite this sweep: the moments-family gate
   is verified-conservative (no silent-wrong zone found; declines are
   3-4 decades early on small fixtures) — distinct from the sojourn
   family's E-batch evidence.
3. **PARK Deferred 4** (recommendation): Phase 1/2 unjustified by
   evidence; revisit only if telemetry/user reports show frequent
   declines where FD's ~1e-7 fallback error is unacceptable — and note
   the sweep's bonus finding that the LARGE-θ regime (where FD
   catastrophically fails and no gate fires) is already fully served
   by the shipped exact path.
