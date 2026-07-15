# VERDICT REPORT — Adversarial Verification of the `phasic` Gradient Audit

## 1. Verdict table

| Claim | Lens A (correctness) | Lens B (reach/severity) | Lens C (file:line/mechanism) | **MAJORITY** | Confidence |
|---|---|---|---|---|---|
| **P2-1** — linear mode silently accepts non-positive edge weight | PARTIALLY_CORRECT | PARTIALLY_CORRECT | PARTIALLY_CORRECT | **PARTIALLY_CORRECT** (unanimous) | High |
| **P2-3** — 3 unconverted FD sites in `svgd.py` preconditioners | CONFIRMED | PARTIALLY_CORRECT | PARTIALLY_CORRECT | **PARTIALLY_CORRECT** (2–1) | High |
| **P2-2** — no FD floor in callback/formula mode | PARTIALLY_CORRECT | PARTIALLY_CORRECT | PARTIALLY_CORRECT | **PARTIALLY_CORRECT** (unanimous) | High |
| **P2-4** — "ALL 12 converted FD sites produce CORRECT gradients" | PARTIALLY_CORRECT | PARTIALLY_CORRECT | CONFIRMED | **PARTIALLY_CORRECT** (2–1) | High |
| **P2-5** — non-findings (cdf_zero, exposure, DPH row-sum) | PARTIALLY_CORRECT | PARTIALLY_CORRECT | PARTIALLY_CORRECT | **PARTIALLY_CORRECT** (unanimous) | High |

No UNRESOLVED splits. **Nothing was unanimously CONFIRMED.** Every one of the five audit claims took damage.

**Evidence quality note:** all 15 refutation reports carried executed output with real numbers and named scratchpad scripts. I found no verdict returned on pure code-reading, so nothing needs discounting on that basis. The one verdict I *do* discount is **P2-4 Lens C's "CONFIRMED"**, which is internally inconsistent: its own corrections section reports 1.5e-3 relative error at θ=1e-14 and **18% at θ=1e-15**, i.e. it measured the same refutation the other two lenses did, then labelled the claim CONFIRMED anyway. Treated as a de-facto PARTIALLY_CORRECT.

---

## 2. Disagreements and dissents (the interesting cases)

### P2-1 — unanimous verdict, but the three disagree on **severity**, and the split matters

All three agree on the same set of falsifications of the audit's *supporting detail*:

- **The audit's repro does not reproduce.** At `theta=[-0.5, 1.0]` the two edge weights are `2(-0.5)+3(1)=2.0` and `1(-0.5)+2(1)=1.5` — **both strictly positive**. All three lenses independently measured `pmf = [0.62736217, 0.52726014]`. The audit's quoted negative pmf `[-0.2254, -0.4266]` never appears. The auditor conflated *negative θ* with *negative rate*.
- **The "formula and callback both RAISE on the same input" differential is apples-to-oranges.** The auditor compared linear `c·θ` against a *product* formula `c0*t0*t1`. With a like-for-like formula (`c0*t0+c1*t1`) or a dot-product callback, all three modes return the identical `[0.627, 0.527]` and nothing raises.
- **The cited mechanism is wrong.** `_check_weight` (`src/phasic/__init__.py:821-838`) tests **only `np.isfinite(w)`** — Lens A, B and C all called it with negative weights (-1.8e-15, -2.0, -1e-16) and got **no raise**. The callback mode's negative-weight rejection actually comes from the C guard `src/c/phasic.c:4921` ("Edge weight evaluates to non-positive value … must be strictly larger than 0"), reached only because callback takes the slow Python-callback path.

**Where they split — severity:**

- **Lens A: downgrade from HIGH.** Reasoning: the linear *moments* path already raises the C guard, so "no guard anywhere" is false; the residual is a missing-validation gap on one path reachable only by handing in a deliberately invalid θ.
- **Lens C: MEDIUM.** It localises the hole precisely: `ComputePmfFfiImpl` (`graph_builder_ffi.cpp:91`) is the **only** handler missing the `most_negative_edge_weight` guard its siblings at `:1551, :1785, :1832` already apply; root enabler is the **commented-out** negative-weight validation in `ptd_graph_update_weights` (`src/c/phasic.c:5744-5755`).
- **Lens B: HIGH is defensible** — and it is the only lens that found the route that actually matters.

**I side with Lens B on severity, with Lens C on location.** Lens B's counterexample defeats both of the audit's own "this is safe" arguments *and* A and C's downgrade arguments: **signed edge coefficients are legal in linear mode.** With coefficients `[1.0, -1.0]` and a **strictly positive** `theta=[1, 2]` (so `positive_params=True`/softplus never helps, and no FD floor is involved):

```
weights: -1, 5
pmf  = [-1.30393895, -2.25634704]      <-- negative "probabilities", no error
value = -3.560285993558394
grad  = [ 5.75000118, -6.22559885]     <-- jax.grad returns finite, plausible garbage
dot(c,θ)==0 -> pmf [0., 0.]            <-- also silent
```

A and C both restricted their exposure analysis to "a direct call with an invalid θ", which is exactly the framing the audit used. Neither tried signed coefficients. Under Lens B's route an ordinary SVGD run with the default `positive_params=True` can descend into the region and keep optimising on nonsense. **Net severity: HIGH-for-signed-coefficient models, MEDIUM otherwise. The fix must validate the WEIGHT, not θ** — a naive "θ must be positive" check (which is what the audit as written invites) is both wrong and would not close the hole.

### P2-3 — Lens A dissents (CONFIRMED); B and C refute the mechanism

**Lens A's position:** all three sites exist, use absolute `eps=1e-5`, are not routed through `_fd_probe_points`, and the negative probe reaches the model with no error — it measured `moments = [-2.09523810e+05, …]` (a **negative mean**) at `theta=[-9e-6, 1e-6]` and all three preconditioners returning a scaling vector with no exception.

**Lens B and C's refutation:** the claimed exploit path is wrong for **2 of the 3 sites**. `MomentJacobianPreconditioner` (`svgd.py:3523`) and `FisherPreconditioner` (`svgd.py:3713`) both call `_PreconditionerBase._find_moment_matching_reference` (`svgd.py:3402`) *first*, which **discards the user's `theta_ref`** and snaps each dimension onto the grid `[-2, -1, 0] ∪ linspace(0.5, upper, 12)` (`svgd.py:3432-3436`). Measured: `IN=[1e-6, 2e-6] → OUT=[1.818, 0.0]` (Lens C), `→ [0.5, 0.5]` (Lens B), so the sub-1e-5 rate the claim worries about is gone before the first probe. Only `ProbabilityJacobianPreconditioner` — which explicitly opts out of the ref search (`svgd.py:3627`) — behaves as the claim describes, and Lens B demonstrated it end-to-end (`theta = -9e-6, -8e-6` passed to the model, scaling returned with no error).

**Lens C additionally refutes the *consequence*:** it reports `moments_from_graph` at `theta=[-9e-6, 2e-6]` **RAISES** `Edge weight evaluates to non-positive value '-0.000012'`, and `compute_scaling` does not wrap its model calls in try/except — so the failure is a **loud crash**, not silent corruption.

**Who is right? Both, and the apparent contradiction resolves cleanly.** Lens A used `Graph.pmf_and_moments_from_graph` (the combined FFI handler); Lens C used `Graph.moments_from_graph`. **Per P2-1 Lens C, `ComputePmfFfiImpl` is exactly the one handler missing the guard** — so the pmf-bearing entry point returns silent garbage while `moments_from_graph` raises. Lens A's negative mean and Lens C's exception are both real, on different entry points. This cross-validates the P2-1 localisation.

**On the mechanism, B and C win**: two independent agents reproduced the ref-search overwrite. But note **Lens A and Lens C both got a refined ref containing an exact `0.0` component** (`[1.515, 0.0]` and `[1.818, 0.0]`), so at those two sites the `.add(-1e-5)` probe *still* goes to `-1e-5` — just from θ=0, not from the user's small rate. The claim's mechanism is wrong; its conclusion (negative probes reach the model) survives by a different route.

**All three lenses independently flag a hazard the audit missed and that is larger than the one it reported:** with `param_transform=None`, `_find_moment_matching_reference` evaluates the model at **θ = -2.0 and -1.0 directly** — negative rates, no FD involved, no error. This is not one of the "15 FD sites", so converting the 3 remaining sites to a relative step **would not fix it**.

**Reachability correction in the claim's favour, which the auditor missed:** preconditioning is **ON BY DEFAULT** (`svgd.py:5553-5554`, `preconditioner=None → 'jacobian'`). Confirmed by Lens B and C.

### P2-2 — unanimous PARTIALLY_CORRECT; severity split 2–1 (MEDIUM vs LOW)

Headline confirmed by all three, bit-for-bit: `callback`/`formula` (and, per Lens C, **`weight_mode=None` too — the claim omits it**) share the unfloored branch; minus-probe = exactly `0.0` at θ=1e-15 and `-9e-16` at θ=1e-16, while `linear` floors at `+1e-15`. The defence ("a formula may legitimately be `exp(t0)`, where flooring would be catastrophic") was independently verified by all three: at θ₀=-3 the unfloored probe returns the exact analytic gradient (-30.1283 vs analytic -30.1283), and a linear-style floor would relocate the minus probe from -3.000003 to +1e-15.

Three common corrections:
1. **Docstring at `__init__.py:758` is factually false.** It claims "`_check_weight` loudly rejects any negative weight". It rejects only non-finite values. The loud failure is supplied incidentally by the native layer.
2. **The boundary point is silent, not loud.** At θ exactly 1e-15 the minus-probe weight is **zero**, the weight validator does not fire, and the continuous pdf path accepts the degenerate graph and silently degrades to a one-sided difference. Lens B reports the alternative at that point is the cryptic `Computation produced NaN at vertex 2 (command 4) - numerical catastrophe`, which names neither weights nor θ.
3. **"Loudness" is not a property of the modes.** For a multi-term weight (`c0*t0 + c1*t1`) the negative probe produces a still-positive weight, nothing raises, and `jax.grad` returns a **wrong-sign** gradient (+0.18 / +0.35 at θ₀=1e-15/1e-16 vs true ≈ -0.33). All three note this is *not* caused by the missing floor — floored `linear` is equally bad there — it is the `_FD_MIN_STEP` cancellation defect, i.e. **new finding #12**.

Severity: Lens A and C say MEDIUM stands; Lens B says LOW ("the failure window is θ≤1e-15, where the model is already degenerate: E[T]~1e15, grad~1e28"). **I keep MEDIUM**, on the strength of correction (2) — there is a genuinely silent sub-case at the boundary, which Lens B's LOW rating does not account for.

### P2-4 — the audit's foundational claim does not hold, and this is the most consequential correction in the report

All three lenses independently re-derived closed forms from scratch (hypoexponential E[T], E[T²]; discrete pmf as a Geom⊗Geom convolution; `rvp = t0/(t0+t1)`; `cdf_zero = t1/(t0+t1)`; the log product-weight truth) and **the implementation passes at O(1) θ**: max relative error 2.43e-10 / 1.82e-10 / 4.16e-11 — reproducing the auditor's 2.6e-10 headline. Probe points, denominators, indexing, cotangent contraction are all correct. That much survives.

**But the audit validated only at balanced θ ≈ [1, 2] — precisely the regime where the relative-step refactor changes nothing.** Two lenses measured the regime the refactor exists for:

| θ (mixed scale, t₁=1.0) | library FD | exact | rel. err |
|---|---|---|---|
| 1e-4 | -0.472166750 | -0.472167597 | 1.79e-06 |
| 1e-7 | -0.473510120 | -0.472222168 | 2.73e-03 |
| 1e-8 | -0.471844785 | -0.472222217 | 7.99e-04 |
| 1e-9 | -0.444089210 | -0.472222222 | **5.96e-02** |
| 1e-15 | -0.555111512 | -0.472222222 | **1.76e-01** |

Lens A's control — feeding the library's **own** probe points into an **exact closed-form f** — degrades identically, proving this is inherent to `_FD_REL_STEP=1e-6` and not a solver bug. `1e-8` is the exact value the refactor's own docstring cites ("a per-generation mutation rate is routinely 1e-8") as its motivation.

**Precise condition (synthesis across lenses, none of them stated it alone):** the degradation requires **mixed scales** — a small θᵢ next to an O(1) θⱼ. Lens C scaled *both* components down (θ=[1e-9, 2e-9]) and saw only 1.4e-10; Lenses A and B held t₁=2.0/1.0 and saw 6e-2. `reward_visit_probability` is immune because `p = t0/(t0+t1)` is scale-homogeneous in t₀ — which is why the auditor's rvp cell looked reassuring. **This is the same defect as new finding #12** (`fd-min-step-floor`), independently confirmed there with a causal monkeypatch.

**Coverage is also over-claimed.** Neither Lens A nor Lens C could reach all 12 sites: Lens A exercised 5 (`3806, 6629, 7131, 7188, ffi_wrappers:1325`), Lens C 6. Sites `3680, 4179, 4620, 4864, 6888, 7638, 9735` (user `build_model`, daisy-chain ±exposure, epoch/joint sojourn) are unverified by anyone.

**Methodology:** all three agree evidence-check (a) — "independent forward-only central difference, max rel. err 7.2e-08" — is **not independent** (it reuses the same forward) and its number is uninformative (a 1e-4 CD has O(h²)≈1e-8 truncation, so ~7e-8 is what *any* two consistent central differences give). Only the closed forms are load-bearing.

### P2-5 — (a) and (b) survive; the DPH sub-claim is refuted by all three

- **(a) `cdf_zero`** — confirmed by all three. The zero gradient on the path-forced chain is **topological** (the forward is identically 0). On a branching graph the same code path recovers `t1/(t0+t1)` exactly and its derivative to ~2e-11. Dismissal correct. *Caveat for re-runners (Lens B): the branch must be at an **interior** vertex — edges out of the starting vertex are IPV and are not re-weighted by θ.*
- **(b) exposure** — confirmed by all three. Exposure multiplies `theta[exposure_param_index]` by α (`__init__.py:4692-4703`); the observed no-op was a **dead coefficient slot** in the test graph (bit-identical `0.0e+00` diff), while live slots move the forward by 2–3e-2 (or 8e-2). Retraction correct, for the stated reason.
- **(c) "no breach is possible" — REFUTED as stated, by all three, with executed counterexamples.** The argument only holds for weights that are degree-≤1 homogeneous in θᵢ (linear/log). In formula/callback mode the elasticity is unbounded:
  - `weight_formula="c0*t0**200"` at a row sum of exactly 1.0 → plus probe row sum 1.00010001 → **forward OK, `jax.grad` RAISES** `outgoing rate <= 1. Is '1.000100'`.
  - `weight_formula="c0*exp(t0-300)"` at θ₀=300 → same loud failure from a forward-valid DPH.
  - Lens C also shows even in linear mode the headroom is thinner than claimed: row sums in (1.000099, 1.0001] are **accepted** by the forward but breach on the plus probe.

  Low severity (loud, contrived to reach), but the universal wording is false. This is the same mechanism as new finding #11.

---

## 3. Corrections list (explicit)

**P2-1**
- **Repro WRONG.** `theta=[-0.5, 1.0]` gives rates **(2.0, 1.5)** and `pmf=[0.62736217, 0.52726014]`. Use signed coefficients `[1,-1]` with `theta=[1,2]` (weight −1 → `pmf=[-1.304, -2.256]`), or `theta=[-5,1]` (weights −7,−3 → `pmf=[147.2, 5479.0]`).
- **Differential WRONG.** Formula and callback do **not** raise at that θ when they compute the same weight. The audit compared against a different (product) weight.
- **Mechanism WRONG.** `_check_weight` (`src/phasic/__init__.py:821-838`) checks only `np.isfinite`. Callback's guard is `src/c/phasic.c:4921`.
- **"No guard anywhere" WRONG.** `moments_from_graph` raises; the daisy-chain/joint FFI handlers guard linear via `most_negative_edge_weight` (`graph_builder_ffi.cpp:1241`, used at `:1551, :1785, :1832`). **The hole is exactly one handler: `ComputePmfFfiImpl` (`graph_builder_ffi.cpp:91`).** Root enabler: the negative-weight validation in `ptd_graph_update_weights` is **commented out** (`src/c/phasic.c:5733-5755`).
- **Symptom mis-stated.** The failure is out-of-range/blow-up pmf (25.7, 279.0) *and*, via signed coefficients, genuinely negative pmf. Not "negative θ".
- **Reachability UNDERSTATED** (the audit said grad/SVGD are safe). They are not — see §2.

**P2-3**
- **file:line:** the `eps = 1e-5` assignments are at `svgd.py:3541, 3645, 3744`. The audit cited `3556/3660/3762` (the difference-quotient lines, same method bodies).
- **Mechanism WRONG for 2 of 3 sites** (Moment, Fisher — the ref search overwrites `theta_ref`). Only `ProbabilityJacobianPreconditioner` matches the stated mechanism.
- **Reachability UNDERSTATED:** preconditioning is **ON BY DEFAULT** (`svgd.py:5553-5554`).
- **Missing hazard:** `_find_moment_matching_reference` (`svgd.py:3402-3436`) evaluates the model at θ = **-2.0, -1.0, 0.0** directly when `param_transform=None`. Larger blast radius, not an FD site.
- **Severity:** MEDIUM → **LOW-MEDIUM** for the eps-specific defect (kernel scaling only, non-default flags); the umbrella issue stays MEDIUM.

**P2-2**
- **Scope:** `weight_mode=None` also takes the unfloored branch (`__init__.py:788-795`); the claim names only callback/formula.
- **Boundary:** at θ == 1e-15 the failure is **silent** (zero-weight forward accepted, one-sided FD), not loud. Loud only for θ < 1e-15.
- **`_check_weight` docstring at `__init__.py:758` is false** and must be fixed.
- Severity **MEDIUM stands**.

**P2-4**
- **Drop the word "CORRECT" and the FOUNDATIONAL severity from the general statement.** Restate: *"the FD sites are correctly implemented (probe points, denominator, indexing, cotangent contraction) and reproduce closed forms to ≤2.6e-10 **at well-scaled θ ≈ O(1)**."*
- **Add:** gradients are roundoff-destroyed for a small parameter sitting next to an O(1) parameter — 8.0e-4 at θ=1e-8, 6.0e-2 at θ≤1e-9, 1.8e-1 at θ→0 — i.e. exactly the regime the refactor was written to defend.
- **Demote evidence (a)** (the 1e-4 CD cross-check): not independent, uninformative number.
- **Coverage:** only 5–6 of 12 sites were actually exercised by anyone. State which.
- **Nit (Lens C):** the discrete-pmf cell cannot have run at θ=[1,2] on the stated chain (outgoing rate 8 > 1 → the C layer raises).

**P2-5**
- **Narrow the DPH statement to `linear`/`log` only.** In formula/callback a breach IS possible and was triggered three times.
- **Numeric:** the plus-probe factor is **1.000001** (`_FD_REL_STEP=1e-6`, `__init__.py:687`), not 1.0000006.
- **`phasic.c:6679` mis-cited as "the" bound.** The same `rate > 1.0001` guard is duplicated at `6788, 10906, 10978, 11096, 11182` (+`11418` for granularity); `6679` sits in `ptd_graph_dph_reward_transform` (rewards path only), so a rewards-free discrete run breaches a different one.
- **(b) leaves a live hazard the audit closed too fast:** nothing validates that `exposure_param_index` points at a **live** coefficient slot (`svgd.py:5123` / `svgd_config.py:774-786` are bounds checks only). Pointing exposure at an all-zero slot is a bit-exact **silent** no-op — a "no silent fallbacks" violation.

---

## 4. NEW findings that survived adversarial verification

**12 hunted, 12 CONFIRMED, 0 REFUTED, 0 PARTIALLY_CORRECT.** Every new finding came with an independent re-repro by the refuting agent (different graphs/θ in several cases). Zero dropped.

| # | Finding | Verified severity | Notes from the verifier |
|---|---|---|---|
| N1 | **`discrete=True` + rewards applies the CONTINUOUS reward transform**, then reads `dph_pmf` → wrong PMF and wrong gradient | **HIGH** (stands) | Verifier used a *different* graph and phasic's **own** `reward_transform_discrete` as reference. Model reproduces the continuous transform **bit-for-bit**; correct DPH has P(N=2)=0, model returns 0.105. Gradient rel. err **[0.106, 0.316]**. Scope **wider** than reported: also `graph_builder_ffi.cpp:675, :717`. **Moments are wrong too, by a separate route** (6.25/59.03 vs true 6.25/49.44) — the fix is not just "call `dph_reward_transform`". |
| N2 | **DPH moments are CONTINUOUS moments** — `compute_moments_impl` has no `discrete` parameter | **HIGH** (stands) | Verifier built its own DPH, used phasic's own `dph_pmf` (mass 0.9999999999999999) as truth. Returned m₂ matches `2·a·N²·1` to **1e-15**; true DPH is `a(I+P)N²·1`. **Exact invariant: returned m₂ is always too large by exactly E[N]** — cleaner than the reported "16%". E[N] agrees, which masks the bug. |
| N3 | **Tied 'slave' params exported as the 0.0 sentinel** — `svgd_step` re-tiles `fixed_values` every iteration, destroying the master→slave copy | **HIGH** (stands) | Independently reproduced: `theta_init` row0 `[-9.117, -9.210, -9.117, -9.210]` (correct) → after `optimize()` `[-9.841, -9.210, **0.0**, -9.210]`. `theta_mean[2] = 0.693147181 = softplus(0)` — **4 orders of magnitude off**, with `theta_std[2] = 0.0`. Corruption also in `svgd.history`. Likelihood/gradient are fine; **only the exported results are poisoned**. `SVGD.summary()` masks it. |
| N4 | **Reward-vector length never validated** on the parameterized path → OOB heap read | **HIGH** (stands) | Confirmed: non-deterministic results (3 distinct over 8 identical calls with heap churn), eager-vs-jit gradients differing by ~3e6 / NaN. `_validate_rewards` exists and is wired into `reward_transform` (`__init__.py:3160`) but **not** into `pmf_and_moments_from_graph*`. Also accepts **longer** vectors (silently truncated). |
| N5 | **`_FD_MIN_STEP = 1e-15` floor → gradient is ULP quantization noise for small θ** | **HIGH** (stands) | **Causal proof**: monkeypatching the min step 1e-15 → 1e-6 collapses the error from 53% to ~3e-7 at θ=1e-9. Gradient takes only 9 discrete values over [1e-12, 1e-8], wandering **-0.1665 … -0.0555** (3× swing) while truth is flat at -0.11806. **This is the same defect as the P2-4 correction.** Verifier adds: `log` mode has **no floor at all** and is quantized identically — blast radius one mode wider than reported. |
| N6 | **`ptd_graph_pdf_with_gradient` returns a gradient that is not the derivative of its own PDF** ("empirically determined" minus sign) | **MEDIUM-HIGH** (↓ from HIGH) | Verified against a **central FD of the routine's own `pdf_value`** — self-consistency, so no approximation defence applies. Sign flips at t=0.5 and t=1.0; granularity-invariant across 1e3→4e5, so not discretization. `rc==0` always. **Downgraded because it has zero in-tree callers** — external C-API consumers only. Root cause is *not* the sign: `pmf_grad` itself is wrong; flipping to `+` gives [0.7145, 0.3818] vs true [0.5297, 0.1046]. |
| N7 | **`compute_pmf_with_gradient` misclassifies length-1 coefficient edges as constant** → θ ignored entirely for every 1-parameter model, yet a nonzero gradient is returned | **LOW-MEDIUM** (↓ from MEDIUM) | PDF **bit-for-bit identical** at θ = 1, 2, 5. Convention clash confirmed: `coefficients_length > 1` (`phasic.c:11652`) vs `>= n_params` (`:11836`) vs `== 0` in the live `ptd_graph_update_weights` (`:5637`). The two halves of the same routine disagree. Downgraded: same dead entry point as N6. |
| N8 | **`moments_from_graph` unusable under vmap** — `vmap_method='expand_dims'` on a scalar-only callback | **MEDIUM** (stands) | Disambiguated with B=4, nr_moments=3: `Expected: (4, 3), Actual: (3,)` — proving the callback got the whole batch and passed `len(theta_np)=4` (batch size!) as `n_params`. Control: `vmap` over `pmf_from_graph` works on the same graph. Fail-loud, so no silent corruption. |
| N9 | **`daisy_chain_joint_probs`: `grad(jit(f))` raises TypeError** (DynamicJaxprTracer leaked into jaxpr consts) | **MEDIUM** (stands) | Reproduced with the repo's own test fixture. **Causal isolation:** custom_vjp closing over a numpy const → OK; closing over a jnp array made in-trace → raises. Passing a concrete out-of-trace jnp array makes `grad(jit(f))` work and gives bit-identical gradients. Trigger is input-dependent; `jit(grad(f))` is a workaround. SVGD path unaffected. |
| N10 | **`Graph.moments(power, discrete=True)` always crashes** — `super().moments_discrete` doesn't exist | **LOW-MEDIUM** (↓ from MEDIUM) | `AttributeError: 'super' object has no attribute 'moments_discrete'`; `grep moments_discrete src/cpp/phasic_pybind.cpp` → nothing. Dead branch: no input returns a value. Downgraded because `expectation_discrete`/`variance_discrete` work (2.065 measured); only power > 2 is genuinely unavailable. |
| N11 | **FD plus-probe pushes a DPH row sum out of the tolerance band phasic itself accepts** — forward evaluates, `jax.grad` raises | **LOW** (stands) | Reproduced: row sum 1.0000999 → forward OK, grad raises `outgoing rate <= 1. Is '1.000100'`. Same mechanism as the P2-5(c) refutation. **Correction:** the vulnerable band (1.000099, 1.0001] is the *widest possible* case (θᵢ scaling the entire row); generally narrower. |
| N12 | **`pmf_and_moments_from_graph` reads `n_features` from the wrong rewards axis** on the default path | **LOW** (stands) | Verifier tried **both** orientations — both fail with mirror-image shape errors. **No 2D rewards orientation works on the default path** (unless n_features == n_vertices, where it would silently mis-interpret instead of crashing). One-character fix: `shape[1]` → `shape[0]` at `__init__.py:7004`. |

**Cross-finding note the verifiers surfaced:** N6's verifier reports the C PDF matches a **3-phase** hypoexponential (start-edge counted as a rate stage), while N7's verifier reports it *fails* to match a **2-phase** closed form (start edge as IPV, phasic's actual semantics: 0.6277 vs 0.8503). Both are consistent: **the dead C routine treats the IPV edge as an ordinary rated transition**, adding a spurious exponential stage. N6 stands regardless (the gradient is inconsistent with the routine's own forward), but if `ptd_graph_pdf_with_gradient` is ever revived, its **forward is wrong too**.

---

## 5. RECOMMENDED FIX LIST (by severity)

### HIGH — silent wrong numbers on live, default code paths

1. **`discrete=True` + rewards uses the continuous reward transform.**
   `src/cpp/parameterized/graph_builder.cpp:741` (also `:706, :904, :931, :1100`) and `src/cpp/parameterized/graph_builder_ffi.cpp:675, :717`.
   Consult `discrete` **before** the transform; call `dph_reward_transform` (`api/cpp/phasiccpp.h:1196`) — which the library's own Python dispatch already does at `src/phasic/__init__.py:3168-3177`. Gradient error measured **10.6% / 31.6%**.

2. **DPH moments are CTMC moments.**
   `src/cpp/parameterized/graph_builder.cpp:485` (`compute_moments_impl`), called for `discrete=True` at `:764, :790, :934, :970`.
   Add a `discrete` branch. Returned m₂ = `2·a·N²·1`; correct = `a(I+P)N²·1` = returned − E[N]. **Not** fixed by (1) alone.

3. **`_FD_MIN_STEP = 1e-15` destroys the gradient at small θ** — *this is also the correction to the audit's own P2-4 "all clean" verdict.*
   `src/phasic/__init__.py:688` and `:783-795`.
   Measured: 1.3% error at θ=1e-8, **53%** at 1e-9, 88% at 0, with a 3× discontinuous wander. Requires **mixed scales** (small θᵢ beside an O(1) θⱼ). Do **not** just raise the floor to 1e-6 globally: `log` mode has no floor at all and needs a different remedy (an absolute floor there breaks its strict-positivity invariant). Correct step is `~cbrt(eps) · max(|θᵢ|, θ_scale)`, or complex-step/exact AD on the small-θ branch.

4. **Tied slave parameters exported as the 0.0 sentinel.**
   `src/phasic/svgd.py:4093-4103` (the per-iteration `jnp.tile(fixed_values, …)` overwrite) destroying the copy at `src/phasic/__init__.py:6000-6018`; sentinel written at `:4464-4471`.
   Re-apply the master→slave scatter **inside `svgd_step`'s expansion** (after the tile). A post-`optimize` repair must also cover `svgd.history`.

5. **Reward-vector length never validated on the parameterized path (OOB heap read).**
   `src/phasic/__init__.py` `pmf_and_moments_from_graph` / `_multivariate` (no `_validate_rewards` call) + `src/cpp/parameterized/graph_builder.cpp:663-676` (derives `n_vertices` from the rewards array).
   Assert **exact** equality `len(rewards_row) == vertices_length()` per feature row at the pybind entry. Also fix the contradictory docstring at `:7697` → `(n_features, n_vertices)`.

6. **`ComputePmfFfiImpl` missing the non-positive-weight guard (P2-1, corrected).**
   `src/cpp/parameterized/graph_builder_ffi.cpp:91` — add the `most_negative_edge_weight` check its siblings already apply at `:1551, :1785, :1832`. Consider un-commenting the validation in `ptd_graph_update_weights` (`src/c/phasic.c:5733-5755`).
   **Validate the WEIGHT `dot(c,θ) > 0`, not the sign of θ** — negative θ is legal in linear mode, and the reachable case is *positive θ + signed coefficients* → `pmf = [-1.304, -2.256]`, `grad = [5.75, -6.23]`, no error.

### MEDIUM

7. **Preconditioners feed unconstrained/negative θ to the solver.**
   `src/phasic/svgd.py:3402-3436` (`_find_moment_matching_reference` grid literally begins `[-2.0, -1.0, 0.0]`, applied raw when `param_transform=None`) — **the dominant route, not the FD probe.** Also convert the three unfloored FD sites at `svgd.py:3541, 3645, 3744` to `_fd_probe_points`. Note preconditioning is **on by default**.

8. **`moments_from_graph` cannot be vmapped.** `src/phasic/__init__.py:6608` (`vmap_method='expand_dims'`) + the scalar-only callback at `:6589-6601`. Either vectorize the callback or use `vmap_method='sequential'` (the pattern already used at `:3661, :6821, :7459`).

9. **`grad(jit(f))` tracer leak in `daisy_chain_joint_probs`.** `src/phasic/__init__.py:9584` — keep `initial_ipv` as a plain `np.ndarray` in the closure; `ffi_wrappers.py:1201` already converts at call time.

10. **Fix the false docstring at `src/phasic/__init__.py:758`** ("`_check_weight` loudly rejects any negative weight"). It rejects only non-finite values (`:821-838`). Any fix built on that sentence rests on a false mechanism.

11. **Bind `moments_discrete`** in `src/cpp/phasic_pybind.cpp`, or dispatch power 1/2 to the existing `_expectation_discrete`/`_variance_discrete`. `src/phasic/__init__.py:2148` currently always raises `AttributeError`.

### LOW

12. **`pmf_and_moments_from_graph` rewards axis.** `src/phasic/__init__.py:7004` — `rewards.shape[1]` → `shape[0]` (and `:7006`). One-character fix; **no** 2D orientation currently works on the default path.

13. **FD plus-probe vs the DPH `rate > 1.0001` guard.** `src/phasic/__init__.py:794` vs `src/c/phasic.c:11182` (and the five duplicate guards at `6679, 6788, 10906, 10978, 11096`). Forward-valid models inside the slack band cannot be differentiated. Also reachable in formula/callback mode from a row sum of *exactly* 1.0 via a high-elasticity weight (`c0*exp(t0-300)`, `c0*t0**200`).

14. **`exposure_param_index` is only bounds-checked** (`svgd.py:5123`, `svgd_config.py:774-786`), never checked for a live coefficient slot → bit-exact silent no-op. Violates the repo's "no silent fallbacks" rule.

15. **Dead C API `ptd_graph_pdf_with_gradient`** (`src/c/phasic.c:11805`, conversion block `11930-11943`; declared `api/c/phasic.h:1397`): gradient is not the derivative of its own PDF (sign flips), length-1 coefficient edges are treated as constants (θ ignored entirely for 1-param models), and the IPV edge is treated as a rated stage. Either fix all three or **remove it from the public header and from `docs/cpp_api/c_distributions.qmd:137` / `docs/mathref/14_distribution_computation.qmd:322`**, which currently advertise it as producing "machine-precision accuracy". Docs line numbers there are stale.