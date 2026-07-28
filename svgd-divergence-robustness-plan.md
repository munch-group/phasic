# Plan — SVGD divergence robustness (deferred issue #1)

Source analysis: `deferred-svgd-divergece-fix.md`. Branch: `fix/svgd-robustness-and-lrt`
(stacked on the Category-A fixes).

## Problem (grounded at HEAD)

A single SVGD particle can wander during gradient ascent to a θ whose implied
transition rate is astronomically large (~1e31). The native PMF build then raises
`RuntimeError: Maximum outgoing rate (...) is too large to build a phase-type
distribution ... model diverged or ... rescaling` (`src/c/phasic.c:11392`; a
sibling message at `:11897` "...too large to compute..."; a legacy
"...Increase the granularity" at `:11437`). That RuntimeError crosses the
`jax.pure_callback` boundary (`__init__.py:6983`, callback body `:6956`, native
call `_compute_pmf_and_moments_cached` `:6900` → `builder.compute_pmf_and_moments`)
and **aborts the entire optimize() run** — one bad particle out of N kills the fit.

Repro tests (known-flaky; a stochastic particle triggers it):
`inference/test_model_selection.py::test_log_likelihood_independent_of_regularization`,
`inference/test_svgd_exposure.py::test_exposure_shifts_posterior_inverse_to_alpha`.

## Scope decision

Implement **fix A (fail-soft callback)** as the core — highest leverage, smallest
blast radius, behavior-preserving (the `try` only fires on a θ that *already*
crashed). Evaluate **fix B.1 (per-step φ trust-region)** only if de-risking shows
particles still wander pathologically after A. Defer C (bounded transforms) and
D (diagnostics/policy knob) — larger surface, not required to make the fits robust.

## Batch 1 — fail-soft callback (fix A)

- Add a module helper `_is_rate_blowup(exc)` matching the three rate-blowup
  strings only ("too large to build a phase-type", "too large to compute a
  phase-type", "Increase the granularity") — NARROW, so genuine bugs still surface.
- In `_compute_pmf_and_moments_cached` (`__init__.py:6900`): wrap the native call
  **per-θ** — in the batched loop (one penalty for the offending particle only, not
  the whole batch) and in the unbatched branch. On `_is_rate_blowup`, return a
  finite penalty matching the declared `pure_callback` shapes: `pmf = full(shape,
  _PMF_FLOOR)` with `_PMF_FLOOR = 1e-300` (so `log(pmf)` is finite-negative, not
  `-inf`), `moments = zeros(nr_moments)`. Re-raise anything else.
- The FD/analytic gradient path reaches the same callback at its probe points, so
  a blown-up θ yields finite (≈0 or restoring) gradients rather than propagating
  `nan` — no separate gradient wrap needed. Verify this holds.

**Gate 1:**
- Both repro tests complete without `XlaRuntimeError` (run several times — flaky).
- A normal (non-diverging) fit is **bit-identical** before/after (prove the `try`
  is inert on the happy path): compare `get_results()` on a fixed-seed non-tied fit.
- A crafted "one diverged particle" fit finishes and its finite particles converge
  (the penalty particle is pulled back or stays at a finite bad loss).

## Batch 2 — per-step φ trust-region (fix B.1), CONDITIONAL

Only if Batch-1 de-risking shows a particle still reaching pathological φ often
enough to matter. Cap the per-particle φ-update L2 norm at a generous
`trust_radius` (never triggers on normal steps → behavior-preserving). Place it in
the SVGD update step in `svgd.py`. Decision recorded from Batch-1 findings.

**Gate 2 (if taken):** normal fits bit-identical (trust radius never binds);
divergence-prone fit stays bounded (max implied rate below the feasibility ceiling).

## Adversarial review (both batches)

Reviewer told to REFUTE: (a) prove the fail-soft is NARROW (a genuine non-rate
RuntimeError still propagates); (b) prove happy-path bit-identity; (c) attack the
penalty shapes (multivariate/2D rewards, batched vmap) for a shape mismatch vs the
declared `pure_callback` shapes; (d) confirm the FD-gradient at a penalized θ is
finite, not `nan`; (e) confirm no other reachable callback path in the two tests
still crashes.

## Risks / notes

- `feedback_no_change_svgd`: fix A is in `__init__.py` (callback), not the SVGD
  update math — minimal SVGD surface. B.1 touches `svgd.py`; keep it
  behavior-preserving and gated on de-risking.
- Penalty must match the exact `pure_callback` declared shapes or JAX errors — the
  main implementation hazard; covered by Gate 1 + review (c).
