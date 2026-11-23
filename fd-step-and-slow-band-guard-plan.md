# Positive finite-difference step + slow-evaluation guard: plan

**Status: v2, 2026-11-23. v1 went through adversarial review by three
independent agents (numerics, implementation, guard); every finding is
dispositioned in section 10. Experiments ran on master `a07cd1fd` and on a
throwaway prototype (branch `exp/fd-step-prototype`, worktree
`/Users/kmt/phasic-epochprior`, never to be merged). NO shipped code
changed. Decisions in section 8 need sign-off before any batch starts.**

Follows `epoch-fit-prior-and-fd-floor-plan.md` (its Batch 1 "FD probes never
leave the positive domain" and Batch 4 "slow-band guard").

Every number below was measured unless marked "read from source". Scripts
and logs: `experiments/fd_step_guard/` (`e*.py`, `*.log`; reviewers'
scripts in `review_*/`). Timings: this Mac (arm64, where long double ==
double).

---

## 1. Summary

**F — finite-difference step.** Every theta-space FD backward perturbs a
parameter by an ABSOLUTE 1e-7. For a free component theta_i < 1e-7 the minus
probe is a negative rate. Today, by path and weight mode:

| path | linear weights | formula / log weights |
|---|---|---|
| epoch (daisy) S5-S8 | **hard error** (the user-reported `NEGATIVE transition rate`) | rejected earlier (linear only) |
| joint-index S9 (default joint-graph fit) | **silently wrong**: 4.6e15x at 1e-9, NaN at 1e-7, 7% at 1e-6 (k <= 10 mutations) | **error** (`weight_formula produced a non-finite or negative edge weight`) |
| `pmf_and_moments` C3 (the main SVGD path) | **silently wrong**: 3.4e3x at 1e-9, 40x at 1e-8, 34% at 1e-7, 0.33% at 1e-6 (clamp makes it one-sided with h = 1e-7) | no error (clamp keeps the probe >= 1e-9), same inaccuracy |
| `pmf_from_graph` S2 etc. | silently evaluated at the negative rate | **error**, even for FIXED slots at S1-S4/S11 (they do not skip fixed dims) |

Realistic unscaled rates (mutation ~1e-8 per bp per generation) live here.

**S — slow band.** Uniformization (continuous pdf/cdf, the epoch path's
intermediate epochs) costs ~4 ns per edge per step, steps = granularity x
time, granularity = max(1000, int(2 x max_rate)) once max_rate > 512.
`stop_probability` has no cap; `pdf` has a 1e9-step cap that fires after
hours, and costs 16 B of memory per step (measured: 1 GB at 4e7 steps on
a 3-vertex graph). One epoch evaluation at 1e5 / 1e6 x the normal rate took
37 s / 365 s (2,808 vertices); SVGD repeats it (1 + 2P) times per affected
particle per iteration.

**Proposed.**
- **F:** for 0 < theta_i < 1e-7/c use a central step h = c*theta_i
  (c = 1e-3 recommended, D1); everywhere else keep exactly today's code.
  Bit-identical for theta_i >= 1e-7/c (prototype-verified through
  jit(vmap(grad)), including at the boundary); fixes the crash, the silent
  errors and the formula/log crashes; same number of forward calls.
- **S:** (1) a memory bound: lower `pdf`'s step cap from 1e9 to 5e7
  (~0.8 GB per graph) and add the same cap to `stop_probability` /
  `accumulated_visiting_time`; (2) a time bound relative to the fit's own
  starting cost (K x the initial particles' predicted work), installed by
  SVGD/MCMC for the duration of a fit. Both route through the EXISTING
  fail-soft class (MCMC rejects; SVGD's divergence handling applies), and
  SVGD reports any particle that ends a fit in the penalty (it measurably
  never comes back, §5.3). The epoch path, which has no fail-soft, raises
  with the native cause instead of today's anonymous "NaN PMF".

---

## 2. Issue F: inventory (read from source, master `a07cd1fd`; confirmed complete by two reviewers)

| id | site | file:line | step today | reachable from Graph.svgd |
|---|---|---|---|---|
| S1 | `pmf_from_graph` bwd, callback mode | `__init__.py:3858` | ±1e-7, fixed dims NOT skipped | user models passed to `SVGD` |
| S2 | `pmf_from_graph` bwd | `__init__.py:3980` | same | same |
| S3 | `pmf_from_cpp` bwd | `__init__.py:4369` | same | same |
| S4 | `moments_from_graph` bwd | `__init__.py:7762` | same | same (JIT path needs the source tree, §9) |
| S5 | epoch FD, no exposure | `__init__.py:5070` | ±1e-7 | **yes** |
| S6 | epoch FD, no exposure, `exact_final_grad` non-final slots | `__init__.py:5108` | ±1e-7 | yes |
| S7 | epoch FD, exposure | `__init__.py:5411` | ±1e-7 | yes |
| S8 | epoch FD, exposure, `exact_final_grad` | `__init__.py:5462` | ±1e-7 | yes |
| S9 | joint-index `_fd_theta_bar` | `__init__.py:9656` | ±1e-7 | **yes (default joint-graph fit)** |
| S10 | `daisy_chain_joint_probs` (public, legacy) | `__init__.py:11799` | ±1e-7 | public API |
| S11 | `_rvp_bwd` | `ffi_wrappers.py:1301` | ±1e-7, fixed not skipped | yes (zero-inflation via `zero_inflation.py:174-197` when `use_ffi=True`) |
| C1 | `cdf_zero` bwd, callback | `__init__.py:8424` | +1e-7 / clamp `max(theta-1e-7, 1e-9)`, divide by actual diff | yes |
| C2 | `cdf_zero` bwd | `__init__.py:8706` | same | yes |
| C3 | `pmf_and_moments` model bwd | `__init__.py:8832` | same | **yes (main path)** |
| H1 | joint-index exact decline fallback (host numpy) | `__init__.py:9308` | `max(1e-6|t|, 1e-10)` | yes (`exact_grad=True`) |
| P1-P3 | SVGD preconditioner Jacobians | `svgd.py:3995, 4099, 4199` | ±1e-5 | phi-space (always valid) when a transform is set; **theta-space** when `param_transform is None` (positive_params=False): probes go negative for theta < 1e-5 and the epoch forward then raises |

Not FD: scipy least squares in `method_of_moments.py:482/498`,
`probability_matching.py:449` (bounded); `mcmc.py`'s 1e-6 is proposal
jitter. Only the epoch FFI checks negative linear rates
(`graph_builder_ffi.cpp:1604`, tolerance -1e-12); formula/log modes reject
non-positive weights natively (`graph_builder.cpp:607-620`, `phasic.c:5706`).
SVGD zeroes non-finite gradient elements per particle (`svgd.py:4439-4440`);
finite-but-wrong values pass through.

## 3. Proposed FD rule

```
if theta_i <= 0, or c*theta_i >= 1e-7, or c*theta_i == 0 (subnormal):
    EXACTLY today's code at this site
else:                       # 0 < theta_i < 1e-7/c
    h = c*theta_i; probes theta_i ± h (both > 0); central difference
H1: h = 1e-10 if t == 0 else max(1e-6|t|, min(1e-10, c|t|))   (unchanged for |t| >= 1e-10/c)
P1-P3 (only when param_transform is None): same rule with 1e-5 in place of 1e-7
```

- **Bit-identity above the threshold.** Division by a traced `2*h` instead
  of the constant `2*eps` differs by ~1e-15 (XLA turns division by a
  constant into a multiplication), so each non-clamp site selects its
  unchanged old expression when the rule falls through
  (`jnp.where(h == 1e-7, <old>, <new>)`, h selected from the literal, x64
  forced — reliable). Verified bit-identical on the prototype under
  jit(vmap(grad)) and plain grad, in mixed batches, and at the boundary
  (3.3333e-5 x (1+1e-12), 3.34e-5, 4e-5, 1e-4, 1e-3) at the epoch,
  joint-index and pmf_and_moments sites (c = 3e-3; recheck at the chosen c
  in Batch 0). Clamp sites C1-C3 divide by the computed difference and are
  bit-identical without the select.
- **theta_i <= 0 keeps today's code** (deliberate residual: at exactly
  theta = 0 the linear non-epoch paths still probe -1e-7 silently; only
  reachable with positive_params=False or direct calls). Found necessary:
  with ±1e-7 at theta = 0 a `sqrt(t1)` callback model hit NaN where the
  clamp probes 1e-9 (`test_callback_non_finite_gradient_declines_per_theta`).
- **Scope: the lower bound at zero only.** Upper bounds (BetaPrior's
  sigmoid clamp at 1-1e-9, formula weights like `1 - t1`, negative linear
  coefficients) are not protected — today or after. Documented.
- **Helper location:** a new leaf module (`src/phasic/_fd_step.py`)
  imported by `__init__.py`, `ffi_wrappers.py` and `svgd.py`
  (`__init__.py:395` imports `ffi_wrappers` before any helper defined later
  in `__init__` exists).

### 3.1 Accuracy (prototype, through the shipped jax.grad; references: exact gradients where they exist, Richardson FD with positive probes otherwise)

Two gradient shapes near zero:

- **Power law** (p ∝ theta^k: a rate with k observed events, a coalescence
  rate in a pdf). Central relative-step error = (k-1)(k-2)/6 x c^2,
  confirmed to two digits (C3, k = n-1: 9.0e-6 / 1.35e-4 / 4.59e-4 at
  n = 5 / 12 / 20, c = 3e-3). The SAME error exists today at theta just
  above 1e-7/c (h/theta = c), so the rule makes the error uniform, it does
  not add a new one there.

  | k | c = 3e-3 | c = 1e-3 |
  |---|---|---|
  | 10 | 1.1e-4 | 1.2e-5 |
  | 20 | 5.1e-4 | 5.7e-5 |
  | 50 | 3.5e-3 | 3.9e-4 |
  | 100 | 1.5e-2 | 1.6e-3 |

  (Against today: 4.6e15x / 3.4e3x at 1e-9.) The pdf forward is only
  ~2.5e-3 accurate (CLAUDE.md), so c = 1e-3 keeps the FD error below the
  forward's own error up to k ~ 100.
- **Smooth through zero** (an O(1)-scaled effect, e.g. an epoch-1
  coalescence rate). Roundoff ≈ eps_mach |f| / (h |f'|) grows as h
  shrinks: in theta-space the rule is LESS accurate than today here
  (C3 migration: 3.2e-7 → 8.5e-5 at 1e-9, c = 3e-3; S5 non-final slot:
  2.5e-4 at 1e-9). But SVGD consumes the phi-space gradient
  theta x dL/dtheta (softplus), where the absolute error is
  <= 2e-13 on every smooth slot measured — negligible at any c. So c should
  be chosen for the power-law shape alone: **c = 1e-3** (D1).

Other prototype results (c = 3e-3): epoch final slots vs the exact
gradient <= 7.2e-8; H1 8.1e-3 → 7.2e-6 at mu = 1e-9; vmap(grad) timings
unchanged (epoch 21.0 → 19.2 ms, epoch+exposure 52.0 → 56.4, joint-index
1.1 → 1.1, pdf 91.9 → 91.9; noise); the end-to-end epoch fit with
particles started at 5e-8 and 1e-9 fails today with the user's error and
completes with the rule.

**Residual, not addressed:** in that fit the near-floor particles did not
move in 40 iterations. The one at 1e-9 sits on the positivity floor
`max(softplus(phi), 1e-9)` (`svgd.py:5926`), where d(theta)/d(phi) = 0;
the one at 5.2e-8 has d(theta)/d(phi) ≈ theta. Initial particles no longer
start there (`GaussPrior(lower=0.0)`).

### 3.2 Granularity jumps (E11) — why the step is NOT made relative everywhere

A fully relative step (h = c|theta| for all theta) would also fix the
large-component error at mixed scale (E10: d E[T]/d theta_0 at
theta = [1, 1e-8] is 4.3% wrong today and under the capped rule, 7.0e-6
with a fully relative step: E[T] ≈ 1e8 is set by the small parameter, so a
1e-7 step on theta_0 = 1 is roundoff-dominated). Rejected: above max exit
rate 512 the automatic granularity int(2 x max_rate) (`phasic.c:13037`)
makes the computed pdf a step function of theta (log-likelihood jumps of
~2e-4: 12 per ±0.1% of theta at scale 200, 120 at 2000), and a relative
step spans many jumps: 25-34% error. The absolute 1e-7 step rarely contains
one (0 in 600 random thetas, worst 1.4e-5), though it is catastrophically
wrong when it does (6.5e5x at a round theta sitting on a boundary). The
capped rule's step is always <= 1e-7, so it never spans more jumps than
today. On the moments the default `exact_moment_grad=True` already gives
the exact gradient; on the pdf the large-component residual remains.

### 3.3 Tests that pin today's behaviour

12 files mentioning FD steps / mixed scale / negative rates, run on the
prototype (154 passed) vs master. Changes:

- `inference/test_fd_gradient_mixed_scale.py::test_daisy_fd_correct_at_mixed_scale`:
  strict xfail now XPASSes → remove the marker (its docstring asks for it).
- `::test_fd_second_moment_gradient_broken_without_fix`: FD 2nd-moment
  error now 1.8e-5 (asserts > 1e-2) → rewrite to assert correctness.
- `::test_fd_gradient_still_broken_at_mixed_scale_without_fix` still
  passes, correctly (the large component, §3.2).
- `test_likelihood_correctness.py::test_loglik_gradient_degrades_only_below_1e8`
  is unaffected by the rule, and does not measure what it claims: its
  "shipped" loss is `log(pmf + 1e-10)` (line 233) while the closed form has
  no 1e-10. The regulariser alone predicts 2.105e-2 / 0.1686 at
  l2 = 1e-8 / 1e-9; the test measures 2.1087e-2 / 0.16875 on master AND on
  the prototype. The CLAUDE.md figure "2.6e-2 relative gradient error at
  theta=[1,1e-8]" is this artefact. → fix the test (same regulariser in
  the closed form, or none in either) and correct CLAUDE.md (D3).
- The θ <= 0 fix was re-gated on the final prototype: same 12 files, 155
  passed, 4 xfailed, and only the two intended flips above fail.

---

## 4. Issue S: measurements

### 4.1 Where cost depends on rates (read from source; coverage confirmed by review)

Every inference caller of uniformization goes through `Graph::pdf` / `cdf`
/ `stop_probability` / `accumulated_visiting_time`
(`api/cpp/phasiccpp.h:1569, 1675, 1709`) with automatic granularity:
pybind GraphBuilder pmf / multivariate / zero-inflation paths, the FFI pmf
and pmf_and_moments handlers, both epoch FFIs (sojourn and stop-probability
final reads), C++ `add_epoch`, `state_probability`, Python `auto_t` /
`next_ipv` / `daisy_chain_joint_probs`. Discrete DPH cost scales with jumps,
not rates. The user step iterator and `ptd_random_sample_stop_vertex(time)`
(O(rate x time), user API) are out of scope. Elimination paths are
rate-independent except that the moments elimination switches to MPFR
when its condition number exceeds 1e12 (`phasic.c:10069-10083`) — a
conditioning cost, out of scope.

### 4.2 Cost and memory

| path | model | rate scale | time per evaluation |
|---|---|---|---|
| `stop_probability` (dt 0.15) | joint n=6, 2,808 v / 6,730 e | 1e3 / 1e4 / 1e5 / 1e6 | 0.27 / 2.9 / 27.7 / 271 s |
| epoch forward (2 epochs) | same | 1e3 .. 1e6 | 0.44 / 3.8 / 37 / 365 s |
| `pmf_and_moments` (t <= 2.5) | island n=10, 482 v / 2,707 e | 1e2 / 1e3 / 1e4 | 0.27 / 2.6 / 26 s |

- Cost = 4.0 ± 0.3 ns x edges x steps on both paths, independent of
  edges per vertex (4.29 / 4.31 / 4.05 ns at 1 / 4 / 16 edges per vertex,
  up to 64k edges). x86-64 Linux runs this loop in 80-bit long double
  (`phasic.c:12943-12975`): unmeasured there.
- `pdf` appends to `_pdf`/`_cdf` every step (`phasiccpp.h:1608-1613`):
  325 / 656 / 1,021 MB peak at 2e6 / 1e7 / 4e7 steps (3-vertex graph,
  1.6 s for 4e7). The v1 claim "memory flat" was wrong. The FFI batch loops
  build one Graph per particle inside OpenMP (`graph_builder_ffi.cpp:425,
  709`), multiplying by the thread count. `stop_probability` stores nothing
  per step.
- Stepping continues from a cached context: 100 ascending times cost the
  same as one call to the largest time. Any bound must use the target
  horizon (granularity x requested time), not the steps still to go,
  or it depends on call order (sorted plot: never fires; reversed: fires).

### 4.3 What happens to a particle whose evaluation fails fast (E6 + review)

One particle started where the native build fails (rate 1e19), 12
particles, 60 iterations:

- **pdf path (`pmf_and_moments`, pybind):** `_is_rate_blowup` turns the
  error into the fail-soft penalty (`_PMF_FLOOR`). The particle stayed at
  exactly 1e19; `theta_mean` 8.3e17, `theta_std` 2.8e18; no warning.
  Review extended it: started at 1e10 it moved ~0.5 per iteration (ended
  at 9.99999997e9); at 1e12, not visibly. A return from the slow band would
  take >= 1e4 iterations. This is the shipped "fix A" (deliberate: one
  diverged particle must not kill a fit; pinned by
  `test_svgd_divergence_robustness.py:117-137`). For random-walk MCMC
  (`mcmc.py:536-560`) the penalty is a correct rejection.
- **Epoch path:** the FFI catches the native exception and writes NaN
  (`graph_builder_ffi.cpp:2152-2161`; the stop-probability variant does the
  same at 1922-1934); the fit then aborts with "Model returned NaN PMF
  values for valid observations" — loud but without the cause.

---

## 5. Proposed guard

### 5.1 Memory bound (all callers, always on)

Lower `pdf`'s existing step cap (`phasiccpp.h:1595`) from 1e9 to **5e7
steps** (~0.8 GB per graph, D6) and apply the same target-horizon check to
`stop_probability` and `accumulated_visiting_time` (which store nothing per
step, so there it is a time bound only). Keep the existing message: it
matches `_is_rate_blowup`, so on the pdf paths the existing fail-soft
applies unchanged — this only moves its threshold down.

### 5.2 Time bound relative to the fit (D4)

An absolute bound cannot separate a legitimately expensive model from a
diverging particle: at production size (>= ~6e5 edges) "W = 1e10
edge-steps" reduces to "max exit rate <= 8,192", which legitimate models
exceed (large-n coalescent C(n,2)·theta, small islands, reward transforms
with fractional rewards — the latter divide rates by the reward). The
v1 per-call W = 1e10 also left the measured 28-37 s cases unguarded.

Instead: when SVGD or MCMC starts, compute each initial particle's
predicted work, host-side (max exit rate from `update_weights`, the
horizon = max observed time or each epoch's length, edges = the DPH
context's existing edge count, `phasic.c:12861-12868`), and install a cap
of **K x the median** (K = 100, D4) in native code for the duration of
the fit (a process-global `std::atomic<double>` in a core translation unit
— not thread_local, not FFI-only; restored on exit, including on error).
Check it, like the memory bound, against the target horizon before
stepping, only with automatic granularity. Direct user calls are bounded by
the memory bound only.

### 5.3 What happens when it fires (D5)

- **pdf paths:** the message is phrased to match `_is_rate_blowup`, so it
  joins the existing fail-soft: MCMC rejects; SVGD carries the particle as
  fix A does today — but now fast instead of minutes per evaluation.
- **SVGD reporting (D7):** at the end of a fit, any particle whose final
  log-likelihood is at the penalty floor is reported in a warning naming
  the particle, its theta and why; `summary()` flags it. (Today: silent.)
- **Epoch path (no fail-soft exists):** both epoch FFIs record the native
  cause per particle with the existing `BatchError` (not a std::string
  under `omp critical` — bad_alloc hazard, `graph_builder_ffi.cpp:62-90`)
  and return it as the `ffi::Error` after the loop. Messages <= 479
  characters (`kMsgCap`), the actionable part first.
- **JIT `pmf_from_cpp`:** compiles its own copy of the C/C++ sources and
  its `extern "C" compute_pmf` has no try/catch
  (`__init__.py:1043-1080, 4282-4290`): any native exception, including
  today's 1e9 cap, is `std::terminate`. Add the catch (returns the
  fail-soft error) in Batch 2; it never sees the fit-relative cap.

---

## 6. Batches (each: implement on a branch, test, memory-capped gate, report)

### Batch 0 — remaining de-risking (no shipped code)

1. Prototype: switch to the chosen c; apply the rule to S1-S4, S10, S11,
   C1, C2, H1, P1-P3; per site and weight mode (linear, formula, log):
   bit-identity at and above the threshold, and accuracy per shape (power
   law against (k-1)(k-2)c^2/6; smooth in phi-space).
2. Full gate on the prototype vs master (memory-capped); list every
   changed test.
3. Guard calibration: initial-particle predicted work and max exit rate at
   the true parameters of the reference models
   (`test_notebook_model_recovery.py`), the 45,704-vertex joint graph, and
   a reward-transformed model with fractional rewards; confirm K = 100 does
   not fire on any of them across a full fit.

### Batch 1 — FD rule (sites per D2)

- `_fd_step.py` helper; per site the old expression is kept verbatim and
  selected outside the changed range.
- New `test_fd_positive_step.py`, per site: (a) bit-identity vs a frozen
  copy of today's formula at 1e-7/c x (1+1e-12), 2x, 10x and at theta <= 0,
  under grad and jit(vmap(grad)); (b) power-law fixtures within
  1.5 x (k-1)(k-2)c^2/6 of the reference, smooth fixtures within 1e-10
  absolute in phi-space, at theta in {1e-9, 1e-8, 1e-7, 1e-6, 1e-5};
  (c) probes positive: the epoch FFI's negative-rate error and the
  formula-mode rejection as oracles (the FFI forwards cannot be spied),
  a pure_callback model where they can; (d) formula and log weight modes no
  longer raise at 1e-8 (S2, S9, and fixed slots at S1-S4/S11); (e) the epoch
  fit from near-floor particles completes.
- Flip the pins (§3.3) and fix the regulariser test (D3); CLAUDE.md:
  replace the "absolute FD step 1e-7" statements and the 2.6e-2 claim.

### Batch 2 — memory bound + fit-relative time bound

- Native: target-horizon checks in the four methods, the lowered pdf cap,
  the atomic fit cap, epoch-FFI `BatchError` plumbing (both variants), the
  `pmf_from_cpp` catch.
- Python: SVGD/MCMC compute and install the cap (restored in `finally`);
  `phasic.configure(fit_work_multiple=K, uniformization_step_cap=...)`
  (names D8; `None` disables the time bound).
- Tests: the time bound fires fast (< 1 s) at a 1e5-scale particle that
  today takes ~30 s, on pdf, stop_probability and an epoch SVGD fit (with
  the cause in the message); a fit at legitimate high rates (> 8,192, e.g.
  a reward-transformed model) does not trip it; results independent of
  time order (sorted vs reversed); the memory bound at 5e7 steps;
  `pmf_from_cpp` no longer terminates; fix-A pin on a graph with > 10 edges.

### Batch 3 — SVGD reporting of penalised particles (D7)

---

## 7. Risks

- The FD change alters gradients wherever a free component is in
  (0, 1e-7/c): by design. Batch 0 step 2 finds every test that moves.
- `jnp.where` evaluates both branches; both are finite for every
  theta > 0 above subnormal, and the backward is never itself
  differentiated (checked: no hessian/jacfwd of these models).
- Lowering the pdf cap to 5e7 makes the existing fail-soft fire at ~20x
  lower rates on small graphs; legitimate evaluations needing > 5e7 steps
  (e.g. rates > 1e7 at t = 2.5) then need an explicit cap (configurable).
- A process-global fit cap is wrong if two fits run concurrently in one
  process (threads); documented, and the cap is restored on exit.

## 8. Decisions

- **D1 — step rule constant.** c = 1e-3 [recommended: power-law error
  <= 3.9e-4 up to k = 50, below the forward's own 2.5e-3; smooth-shape cost
  negligible in phi-space] / c = 3e-3 (the prototyped value; 3.5e-3 at
  k = 50) / other. (A fully relative step is rejected, §3.2.)
- **D2 — sites.** All of S1-S11, C1-C3, H1, plus P1-P3 when
  `param_transform is None` [recommended] / without P1-P3 / epoch only.
- **D3 — pinned tests.** Flip the two mixed-scale pins; fix the
  regulariser artefact in `test_likelihood_correctness.py` and correct the
  CLAUDE.md paragraph built on it [recommended] / leave the latter.
- **D4 — time bound.** Fit-relative cap K x median initial predicted work,
  K = 100 [recommended] / absolute work cap / none (memory bound only).
- **D5 — when the time bound fires.** Join the existing fail-soft on pdf
  paths and raise with the cause on the epoch path [recommended] / raise
  everywhere (MCMC then aborts instead of rejecting; fix A is overridden).
- **D6 — memory bound.** 5e7 steps (~0.8 GB per graph) [recommended] /
  keep 1e9 (~16 GB) / other.
- **D7 — penalised particles at the end of an SVGD fit.** Warning naming
  them + `summary()` flag [recommended] / raise / leave silent.
- **D8 — option names** as in Batch 2, or other.

## 9. Side findings (not fixed here)

- Automatic granularity makes the continuous pdf and the intermediate
  epochs discontinuous in theta above max exit rate 512 (§3.2). Rounding
  granularity to a coarse grid or freezing it within an FD pair would
  remove this; its own plan.
- `Graph.moments_from_graph`'s JIT path could not be built from the pixi
  install (it needs the source tree under the env's `lib/python3.13/src`).
- `exact_grad=True` on the joint-index path was 0.8% wrong at mu = 1e-9:
  that was H1's 1e-10 step floor (fixed by Batch 1).

## 10. Adversarial review of v1 (three independent agents) — dispositions

Numerics reviewer:
1. "Every other path silently evaluates at the negative rate" — false for
   formula/log (they raise). **Accepted**, §1 table, Batch 1 test (d).
2. Worst case 6.3e-5 is fixture-specific; error = (k-1)(k-2)c^2/6.
   **Accepted**, §3.1, test tolerances per shape, D1 now c = 1e-3.
3. Smooth-shape theta-space regression undisclosed. **Accepted**, §3.1
   (negligible in phi-space, measured <= 2e-13).
4. Granularity-jump corollary mis-stated; the capped step can never span
   more jumps than today. **Accepted**, §3.2; Batch 0 check dropped.
5. Upper bounds not protected. **Accepted** as documented scope (§3).
6. P1-P3 are theta-space when positive_params=False. **Accepted**, D2.
7. Subnormal theta → h = 0. **Accepted**, rule falls back.
8. Circular import. **Accepted**, leaf module.
9. Floor particle explanation. **Accepted**, §3.1 residual.
10. Two reference artefacts in v1 logs (e2 joint-index "exact" = H1;
    e2b epoch-1 Richardson roundoff). **Accepted**; neither number is
    cited.

Implementation reviewer:
1. `test_likelihood_correctness` pin measures its regulariser.
   **Accepted** (read from source, line 233), D3.
2. = numerics 2. **Accepted.**
3. Bit-identity untested at the boundary — reviewer ran it: identical.
   **Accepted**, Batch 1 test (a) includes it.
4. = numerics 6. 5. = numerics 8. **Accepted.**
6. Prototype also changed S6/S8; θ <= 0 fix not re-gated. **Accepted**,
   re-gated (§3.3).
7. H1 at t == 0. **Accepted**, §3.
8. = numerics 1. 9. = numerics 5. **Accepted.**
10. Probe spy impossible on FFI. **Accepted**, test (c) uses oracles.
11. θ <= 0 "bit-identical" would freeze a silent case. **Accepted**: named
    as a deliberate residual, not a desired behaviour.

Guard reviewer:
1. The guard would pre-empt fix A on any graph with > ~10 edges; MCMC
   needs rejection. **Accepted**: redesigned to join the fail-soft (D5).
2. W = 1e10 per call leaves the 28-37 s cases unguarded and does not bound
   an iteration. **Accepted**: fit-relative cap (D4) + test at 1e5 scale.
3. steps_to_go undercounts cached calls. **Accepted**: target horizon.
4. Absolute cap = rate cap 8,192 at production scale. **Accepted**: the
   reason for the fit-relative design.
5. pdf memory grows 16 B/step. **Accepted and re-measured myself**
   (1,021 MB at 4e7 steps); memory bound §5.1.
6. Epoch plumbing: both FFIs; BatchError; 479-char cap. **Accepted.**
7. Config in a core TU; `pmf_from_cpp` terminate. **Accepted**, §5.2/5.3.
8. Edge count already exists. **Accepted.**
9. MPFR elimination cost; sampler. **Accepted** as out-of-scope notes.
10. Reward transforms inflate rates. **Accepted**, Batch 0 step 3 and a
    Batch 2 test.
11. 4 ns is arm64-specific. **Accepted**, noted; unmeasured on x86.

## 11. Copy-paste prompt for a fresh session

> Read `/Users/kmt/phasic/fd-step-and-slow-band-guard-plan.md` (v2) and
> CLAUDE.md. My decisions on D1-D8 are: <fill in>. Do Batch 0 on a branch
> (no shipped code), report, then one batch at a time: implement, add the
> tests, run the gates through `.git/claude-scratch/runcap.py`, and report
> before the next batch. Do not change any notebooks.

State at writing: master `a07cd1fd`; prototype branch
`exp/fd-step-prototype` (uncommitted edits to `src/phasic/__init__.py` in
worktree `/Users/kmt/phasic-epochprior`, env var `PHASIC_FD_RULE` =
old / rel / default=new, `PHASIC_FD_REL` = c), throwaway.
