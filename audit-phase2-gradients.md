# Audit — Phase 2: Gradient correctness across all FD sites

> **STATUS: this report was itself audited.** A 15-agent adversarial pass (5 claims × 3
> independent lenses, every reviewer required to run code and told to default to REFUTED)
> **broke several of its original conclusions**, and 5 independent hunters found 12 further
> defects, all of which survived their own refutation. The full verdict is in
> `audit-phase2-adversarial-verdict.md`.
>
> This document has been rewritten to state what is actually true. The corrections are
> called out explicitly — an audit that quietly edits its own errors is worthless.

---

## VERDICT (revised)

**The FD machinery is correctly *implemented* — probe points, divisor, indexing and
cotangent contraction are all right, and it reproduces closed forms to ≤2.6e-10 at
well-scaled θ ≈ O(1).**

**But the refactor does not achieve its central goal.** At the parameter magnitudes it was
written for, the gradient it produces is numerical noise. And the negative-rate bug it was
written to prevent is still reachable — by a route the refactor's design does not address.

**Recommendation: do NOT sign off.** See G1 and G2.

---

## G1 — HIGH — the relative FD step makes the gradient GARBAGE at small θ

*(This is the correction to this report's own original "all 12 sites are correct" verdict,
and it is independently the hunters' finding N5.)*

The step is `h = max(_FD_REL_STEP·|θᵢ|, _FD_MIN_STEP)` = `max(1e-6·|θᵢ|, 1e-15)`
(`__init__.py:687-688`, `:783-795`).

For a **mixed-scale** θ — a small θᵢ next to an O(1) θⱼ, which is exactly the coalescent
case (a per-generation mutation rate ~1e-8 beside a coalescent rate ~1) — measured against
the **exact** derivative of the closed form:

| θ₀ (with θ₁ = 2.0) | library ∂/∂θ₀ | exact | rel err |
|---|---|---|---|
| 1e-2 | −0.09997 | −0.10107 | 1.1e-02 |
| 1e-7 | −0.09479 | −0.09903 | 4.3e-02 |
| **1e-8** | **−0.03053** | −0.09903 | **6.9e-01** |
| 1e-9 | −0.06939 | −0.09903 | 3.0e-01 |
| **1e-12** | **+0.09714** | −0.09903 | **2.0e+00 — SIGN FLIPPED** |
| 0.0 | +0.11102 | −0.09903 | 2.1e+00 |

**Mechanism (arithmetic, confirmed):** at θ₀ = 1e-9 the relative term `1e-6·1e-9` *is* 1e-15,
so the step **is the floor**. But ∂f/∂θ₀ stays O(1) (θ₁ dominates the rate), so:

```
theta0=1e-06  h=1.0e-12  |delta_f|~1.98e-13  roundoff(f)~5.74e-16   SNR = 345
theta0=1e-08  h=1.0e-14  |delta_f|~1.98e-15  roundoff(f)~5.74e-16   SNR = 3.4
theta0=1e-09  h=1.0e-15  |delta_f|~1.98e-16  roundoff(f)~5.74e-16   SNR = 0.3   <-- noise
```

The difference quotient falls **below the roundoff floor of f**. The hunters proved causality
by monkeypatching the floor 1e-15 → 1e-6: the error collapses from 53% to ~3e-7.

**The bitter part:** the OLD absolute step (1e-7) was *well*-conditioned here (SNR ≈ 2e8). It
was only wrong because it probed `θ₀ − 1e-7 < 0`. **The refactor traded a sign-flip for an
accuracy collapse — at 1e-8, the exact magnitude its own docstring cites as the motivation**
("a per-generation mutation rate is routinely 1e-8").

A **relative** step is well-scaled for the *argument*; it does not keep Δf above the
*function's* roundoff. Those coincide only when sensitivity scales as 1/θ — false in
mixed-scale models. Control: scaling **both** components together gives only 1.9e-03 error,
which is why balanced-θ testing (all this report originally did) sees nothing.

**Fix:** the step must be relative to a scale that keeps Δf above noise —
`h ≈ cbrt(eps)·max(|θᵢ|, θ_scale)` — or use complex-step / exact AD on the small-θ branch.
Do **not** simply raise the floor: `'log'` has no floor at all (by design, for strict
positivity) and is quantized identically.

## G2 — HIGH — a negative RATE is still reachable, with strictly positive θ

*(This is the corrected form of the original P2-1. **The original repro was WRONG** — see
"Corrections" below.)*

Edge coefficients may be **signed**. With coefficients `[1, -1]` the weight is `t0 - t1`, so a
**strictly positive** θ produces a **negative rate**:

```
theta        = [1.0, 2.0]            <- both strictly POSITIVE
edge weights = [1.0, -1.0, 5.0]      <- one is NEGATIVE
pmf_from_graph = [-0.937, -1.474, -2.256]     <- returned with NO ERROR
jax.grad       = [ 7.147, -7.874]             <- finite, plausible, GARBAGE
```

This defeats every mitigation the refactor relies on: **no FD probe is involved**, and
`positive_params=True` / softplus keeps θ positive but **cannot** keep `dot(c, θ)` positive.
`Graph.svgd()` on a signed-coefficient model is therefore **exposed** — contrary to this
report's original claim that SVGD was safe.

**Corrected mechanism** (the original was wrong on all three counts):

| | |
|---|---|
| The hole is **exactly one handler** | `ComputePmfFfiImpl`, `graph_builder_ffi.cpp:91` — the **only** one missing the `most_negative_edge_weight` guard its siblings already apply at `:1551, :1785, :1832` |
| Root enabler | the negative-weight validation in `ptd_graph_update_weights` is **commented out** — `src/c/phasic.c:5733-5755` |
| `_check_weight` does **not** validate positivity | `__init__.py:821-838` checks only `np.isfinite`. **The docstring at `:758` claiming it "loudly rejects any negative weight" is FALSE** and must be fixed — any fix built on that sentence rests on a false mechanism. |

**Fix: validate the WEIGHT (`dot(c,θ) > 0`), not the sign of θ.** A "θ must be positive" check
is both wrong (negative θ is legal in linear mode) and would not close this hole.

---

## What DOES hold

* **The FD implementation is correct at well-scaled θ.** All three reviewers re-derived the
  closed forms from scratch (hypoexponential, Geom⊗Geom convolution, `t0/(t0+t1)`,
  `t1/(t0+t1)`) and independently reproduced ≤2.6e-10 agreement at θ ≈ O(1). Probe points,
  the `denom = hi - lo` divisor, indexing and cotangent contraction are all right.
* **Probe positivity:** `linear` floored at 1e-15 for every θ including 0; `log` sign-preserving
  and non-zero-crossing down to θ=1e-30. (Batch A's `log` fix is sound.)
* **12/12 FD sites were exercised**, proven by instrumenting `_fd_probe_points` to record each
  caller's `file:line`. *(Reviewers reached only 5–6 sites each and flagged the other 7 as
  unverified-by-them; the instrumentation record stands, but no reviewer independently
  reproduced those 7.)*

## Corrections to this report's original claims

| original claim | status |
|---|---|
| "all 12 FD sites produce CORRECT gradients" (FOUNDATIONAL) | **OVER-CLAIMED.** True only at θ ≈ O(1). See G1. |
| P2-1 repro: `theta=[-0.5, 1.0] → [-0.2254, -0.4266]` | **WRONG.** Those coefficients give weights (2.0, 1.5) — both positive; pmf is `[0.533, 0.633, 0.527]`. I conflated two different test graphs. Corrected repro in G2. |
| "linear is the only mode with no validation; `_check_weight` guards callback" | **WRONG MECHANISM.** `_check_weight` only tests `isfinite`. The real asymmetry is one FFI handler. |
| "jax.grad and SVGD (positive_params=True) are safe from the negative rate" | **WRONG.** Signed coefficients defeat both. |
| "formula/callback RAISE on the same input where linear doesn't" | **APPLES-TO-ORANGES.** I compared linear `c·θ` against a *product* formula. Like-for-like, all three agree. |
| P2-3: mechanism = preconditioner probes the user's small θ | **WRONG for 2 of 3 sites.** `_find_moment_matching_reference` (`svgd.py:3402`) *discards* the user's θ and snaps to a grid. Only `ProbabilityJacobianPreconditioner` matches. Conclusion survives by a different route (the refined grid contains an exact 0.0). Also: preconditioning is **ON BY DEFAULT** — I said it was default-safe. |
| P2-5(c): "no DPH row-sum breach is possible" | **REFUTED as stated.** True only for weights homogeneous of degree ≤1 in θᵢ. `weight_formula="c0*t0**200"` at a row sum of exactly 1.0 breaches on the plus probe (forward OK, `jax.grad` raises). |
| P2-2 (callback/formula unfloored probe) | **Stands, scope widened:** `weight_mode=None` also takes the unfloored branch; and at θ == 1e-15 exactly the failure is **silent** (zero-weight forward accepted), loud only below. |
| P2-5(a) cdf_zero, P2-5(b) exposure retraction | **Both CONFIRMED** by all three reviewers. The dismissals were right. |
| "independent FD with a different step" as a check | **NOT independent** — it reuses the same forward. Its 7.2e-08 number is uninformative (any two consistent central differences agree to ~1e-8). Only the closed forms are load-bearing. |

---

## 12 further defects found by the hunters (all survived refutation, 0 dropped)

Full detail in `audit-phase2-adversarial-verdict.md`. Headlines:

**HIGH**
* **N1 — `discrete=True` + rewards applies the CONTINUOUS reward transform**, then reads
  `dph_pmf`. Wrong PMF *and* wrong gradient (rel err **10.6% / 31.6%**). `graph_builder.cpp:741`
  and `graph_builder_ffi.cpp:675, :717`. Verified against phasic's *own*
  `reward_transform_discrete`.
* **N2 — DPH moments are CTMC moments.** `compute_moments_impl` has no `discrete` parameter.
  Exact invariant: the returned `m₂` is always too large by exactly `E[N]`. `E[N]` itself is
  right, which masks it. `graph_builder.cpp:485`.
* **N3 — tied "slave" parameters are exported as the 0.0 sentinel.** `svgd_step` re-tiles
  `fixed_values` every iteration, destroying the master→slave copy. Measured: a fitted value
  **4 orders of magnitude off** (`theta_mean[2] = 0.693 = softplus(0)`), `std = 0.0`. The
  likelihood and gradient are fine — **only the exported results are poisoned**, and
  `SVGD.summary()` masks it. `svgd.py:4093-4103`.
* **N4 — reward-vector length never validated → out-of-bounds heap read.** Non-deterministic
  results (3 distinct values over 8 identical calls); eager-vs-jit gradients differing by ~3e6.
  `_validate_rewards` exists but is not wired into `pmf_and_moments_from_graph*`.
* **N5 — the `_FD_MIN_STEP` floor** — same defect as **G1**.

**MEDIUM**
* **N8** — `moments_from_graph` cannot be `vmap`ped (`vmap_method='expand_dims'` on a
  scalar-only callback; batch size is passed as `n_params`). Fails loudly.
* **N9** — `grad(jit(f))` on `daisy_chain_joint_probs` raises (tracer leaked into jaxpr consts).
  `jit(grad(f))` is a workaround; the SVGD path is unaffected.
* **N6** — the dead C API `ptd_graph_pdf_with_gradient` returns a gradient that is not the
  derivative of its own PDF (and its forward treats the IPV edge as a rated stage). No in-tree
  callers, but the docs advertise it as "machine-precision".

**LOW**
* **N12** — `pmf_and_moments_from_graph` reads `n_features` from the wrong rewards axis; **no**
  2D orientation works on the default path. One-character fix at `__init__.py:7004`.
* **N11** — FD plus-probe vs the DPH `rate > 1.0001` guard (the P2-5(c) refutation).
* **N7, N10** — dead/broken C entry points.
* **`exposure_param_index` is only bounds-checked**, never checked for a *live* coefficient
  slot → a bit-exact silent no-op. (This is what fooled me into briefly claiming exposure was
  ignored.) Violates the repo's own "no silent fallbacks" rule.

---

## Verification provenance (who actually reproduced what)

Stated explicitly, because this report was itself wrong once and the reader deserves to know
how much weight each claim carries:

| finding | provenance |
|---|---|
| **G1** (relative step → garbage at small θ) | found by a hunter; **personally reproduced** by the auditor (table above, plus the SNR arithmetic); causally proven by a third agent's monkeypatch of the floor |
| **G2** (negative rate via signed coefficients) | found by a reviewer; **personally reproduced** by the auditor (`pmf = [-0.937, -1.474, -2.256]`, no error) |
| **N1** (discrete + rewards uses the continuous transform) | found by a hunter, re-reproduced by an independent refuter on a different graph, **and spot-checked by the auditor**: the library's output is *bit-identical* to `reward_transform` (continuous) followed by `pdf_discrete` — which is the bug |
| **N2, N3, N4, N6–N12** | found by a hunter and independently re-reproduced by a separate refuting agent (each with executed output). **Not personally re-run by the auditor** — relayed. Treat as high-confidence but verify before acting. |
| every original P2-x correction | 3 independent reviewers, each required to run code |

## Recommendation

**Do not sign off.** The refactor fixes a real bug (the sign-flip) but:

1. **G1** — it does not deliver a usable gradient at the parameter magnitudes it targets. At a
   mutation rate of 1e-8 the gradient is 69% wrong; at 1e-12 the sign is wrong. This needs a
   properly conditioned step before the change can be said to work.
2. **G2** — the negative-rate hole is still open via signed coefficients, and `positive_params`
   does not close it.
3. **N1–N4** are HIGH-severity silent-wrong-number bugs on live paths, independent of this
   refactor but now documented with repros.

The methodology lesson, stated plainly: **this report originally passed the refactor because it
tested only balanced θ ≈ [1, 2] — the one regime where the change is a no-op.** Test the regime
the change exists for.
