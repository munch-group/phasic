# Remediation plan — the finite-difference gradient

**Spike complete. The step size is not the thing to fix.** The conclusion is architectural
and is backed by measurements, all reproduced in this session.

---

## 1. What the spike established

### The tension that made this unfixable

Measured on the test chain at `theta = [1e-8, 2.0]` (a mutation rate beside a coalescent
rate — the case the refactor exists for), against the exact derivative of the closed form:

| scheme | h | lowest probe | rel err | |
|---|---|---|---|---|
| **library**: central, `h = 1e-6·\|θ\|` | 1e-14 | 1.0e-08 | **1.3e-02** | garbage |
| central, `h = 1e-6` **absolute** | 1e-06 | **−9.9e-07** | 3.0e-10 | accurate but **probes negative** |
| **one-sided 2nd-order**, `h = 1e-5` | 1e-05 | **1.0e-08** | **3.1e-10** | **accurate AND never negative** |

A well-conditioned step for a mixed-scale parameter must be **larger than θ itself** — the
sensitivity scale is O(1) while |θ| is 1e-8. A **central** difference therefore *necessarily*
probes `θ − h < 0`. **Central FD cannot be both well-conditioned and positivity-preserving.**
That is why the refactor was trapped: it could have accuracy or positivity, and it chose
positivity, silently.

**A one-sided (forward-only) difference escapes the trap**: it never decreases θ, so it needs
**no floor, no clamp, and no per-weight-mode rule at all** — `_fd_probe_points`' entire
mode-specific apparatus (and the `log`/`callback`/`formula` special cases) disappears.

### Exact AD is available today, is exact, and is 40× faster

`trace_elimination.evaluate_trace_jax` is pure-JAX, so `jax.grad` differentiates it exactly.

```
theta0=1e+00   exact AD=-0.0712500000  truth=-0.0712500000  relerr=0.0e+00
theta0=1e-08   exact AD=-0.1180555549  truth=-0.1180555549  relerr=0.0e+00
theta0=1e-12   exact AD=-0.1180555556  truth=-0.1180555556  relerr=0.0e+00
```

Relative error **exactly zero**, at the θ where FD is 53% wrong. And on a real coalescent
graph:

```
coalescent n=5 (8 vertices)   FD backward 2.695 ms | exact AD (jit) 0.057 ms   47x FASTER
coalescent n=7 (16 vertices)  FD backward 2.937 ms | exact AD (jit) 0.076 ms   39x FASTER
```

FD costs `2·n_params` forward solves; reverse-mode costs ~1. **The FD backward is not a
performance optimisation — it is slower *and* wrong.**

### But exact AD does not reach everything (Q1)

`evaluate_trace_jax` is the **only** JAX-native compute in the library, and the trace records
the **elimination**. So:

| quantity | algorithm | exact AD today? |
|---|---|---|
| moments / `expected_waiting_time` | elimination | **YES** |
| `expected_sojourn_time` (`joint_index`) | elimination | **YES** |
| `reward_visit_probability` / backward probabilities | elimination (linear solve) | **YES** |
| pdf (continuous) | **uniformization** | no |
| pmf (discrete, `dph_pmf`) | DPH stepping (α Pⁿ) | no |
| `stop_probability` | **uniformization** | no |
| daisy chain / joint probs | composition of the above | no |
| `pmf_from_cpp` | **user C++ — irreducibly opaque** | **never** |

### MPFR cannot serve as the oracle (Q2) — NEGATIVE RESULT

`PHASIC_FORCE_MPFR=always PHASIC_MPFR_BITS=256` produced **byte-identical** results to f64
(`moments(1) = 0.4166666654861111` both ways), and a high-precision central difference at
`h = 1e-20` returned exactly `0.0` — pure f64 cancellation. **MPFR does not engage on the
parameterized path.** Reviving it as an oracle is its own project; do not budget on it.

**The oracle is instead:** (a) closed forms, (b) `evaluate_trace_jax` exact AD — which is
itself an exact reference for every elimination-derived quantity, on *any* graph, not just
ones with closed forms. That is nearly as good as MPFR would have been, and it works today.

---

## 2. The plan

### B0 — pin the bug *(DONE)*

`tests/pytest/test_fd_step_conditioning.py` — asserts the gradient against the exact
derivative across the mixed-scale regime. Marked `xfail(strict=True)`, so the suite stays
green now and **fails the moment the bug is fixed**, forcing whoever fixes it to remove the
marker. The fix is proven by this test, not by assertion. Currently: 4 passed, 4 xfailed.

The healthy regime (θ well-scaled, and balanced-θ) is kept as **passing controls** — a fix
that breaks them is not a fix.

### B1 — validate the weight (audit G2) — **DONE**

> **Implemented, and it landed differently from the original fix list.** The list said to
> patch `ComputePmfFfiImpl` (`graph_builder_ffi.cpp:91`) as "the only handler missing the
> guard". Reading the code, that was **too narrow**: `builder->build(...)` is called from
> **seven** handlers, and the guard existed in only the two daisy-chain ones. So the check
> went where they all converge — `GraphBuilder::validate_edge_weights`, called after every
> weight update — which covers **all nine FFI handlers and the pybind path from one place**.
>
> The real asymmetry turned out to be in `GraphBuilder::compute_weight`: FORMULA throws on
> `w < 0`, LOG throws on a non-positive product, and **LINEAR — the default — had no check
> at all**. It does now.
>
> **The C validation was deliberately NOT un-commented** (`src/c/phasic.c:5733-5755`), as
> the fix list had suggested. It signals errors via `ptd_err` and returns `void` from inside
> the update loop, so enabling it would leave weights half-updated unless every caller
> checks. The C++ layer throws properly and the handlers already catch `std::exception`.
>
> It validates `dot(c, θ) > 0`, **not** the sign of θ — the reachable case is signed
> coefficients with a strictly positive θ, where a `θ >= 0` check does nothing.
>
> **Gates:** `tests/pytest/test_negative_rate_guard.py` (9/9). Forward parity **199/199
> bit-identical, 0 value diffs, 0 new raises** across the Phase-1 matrix — the guard is
> genuinely raise-only. Exactly-zero weights (which legally remove an edge) do not trip it.
> The false `_check_weight` docstring (`__init__.py:758`) is fixed, with a note recording
> that it *was* false so nobody rebuilds on it.
>
> #### ⚠️ BEHAVIOUR CHANGE — for the release notes
>
> **`positive_params=False` on a linear-weight graph now RAISES where it previously returned
> silent garbage.** With θ unconstrained, SVGD's init and its gradient steps hand the model a
> negative θ → a negative rate → previously, negative "probabilities" computed without
> complaint.
>
> This surfaced as two failures in `inference/test_svgd_exposure.py`, and they were **real**:
> the tests were comparing two garbage values and finding them equal. Both do their actual
> assertion at a *positive* θ = [2.0, 1.0]; only SVGD's construction-time validation probe
> was straying into the invalid domain. Fixed by supplying a positive `theta_init` — the
> idiom **that file already used** in `test_exposure_multivariate_matches_rescaled_theta`
> ("Provide positive theta_init so the model-validation block doesn't try to evaluate at
> negative theta"). No assertion was weakened.
>
> The broader point stands and should be documented: **`positive_params=False` with linear
> weights and an unconstrained prior is not a sound configuration** — nothing keeps
> `dot(c, θ)` positive. It was always broken; it now fails loudly.

*(original plan text follows)*

A gradient of a garbage forward is meaningless, so this is a precondition.

* Add the `most_negative_edge_weight` guard to **`ComputePmfFfiImpl`**
  (`graph_builder_ffi.cpp:91`) — the **only** handler missing it; its siblings already apply
  it at `:1551, :1785, :1832`.
* Consider un-commenting the validation in `ptd_graph_update_weights`
  (`src/c/phasic.c:5733-5755`).
* **Validate `dot(c,θ) > 0`, NOT the sign of θ.** Negative θ is legal in linear mode; the
  reachable failure is *positive θ + signed coefficients* → `pmf = [-0.937, -1.474, -2.256]`
  with no error.
* Fix the **false docstring** at `__init__.py:758` — `_check_weight` (`:821-838`) only tests
  `isfinite`, not positivity. Any fix built on that sentence rests on a false mechanism.

**Gate:** a test asserting a raise for signed-coefficient negative weight; forward parity
elsewhere unchanged.

### B2 — the verification harness, built BEFORE the fix — **DONE**

> **Implemented** as `tests/pytest/_gradient_oracle.py` (shared machinery) +
> `tests/pytest/test_gradient_regime_grid.py` (**B3's acceptance gate**: 20 passed,
> 4 strict-xfail — when the FD backward is replaced, those four must be deleted and the grid
> must go fully green).
>
> * **`REGIME_GRID`** — balanced × **mixed-scale** θ from 1e0 to 1e-12. The mixed half is the
>   axis whose absence let the bug ship *and* survive the first audit pass.
> * **Oracle = `E[T^k] = k! · α (−S)^-k 1`**, built in JAX from the graph's own weight
>   matrix; `jnp.linalg.solve` has an exact VJP, so `jax.grad` of it is exact at any θ.
>   Validated forward against the library to **≤1e-15** on a forced chain *and* a
>   **branching** graph.
>
>   ⚠️ **The FIRST oracle I shipped here was WRONG, and its own "is the oracle an oracle"
>   test passed anyway.** It used `jax.grad(sum(vertex_rates))` from `evaluate_trace_jax`.
>   But `vertex_rates` is the per-vertex `1/exit_rate` — it is **not weighted by the
>   probability of visiting each vertex** — so it equals `E[T]` (+const) only on a
>   **path-forced** graph where every vertex is visited w.p. 1. On a branching graph it gives
>   `sum = 2.833` where `E[T] = 1.0`, and a gradient of `[-1.111, -0.361]` where the truth is
>   `[-0.333, -0.333]`. It passed its self-test **only because that test used the forced
>   chain**. This is the *same* trap that produced two false findings during the audit, and I
>   walked into it building the very thing meant to prevent it. `TestTheOracleItself` now runs
>   on the **branching** graph too, where a wrong oracle cannot hide.
>
>   **Accuracy limit** (documented, so no one asserts past it): the oracle solves
>   `(−S)x = 1`, so its error is ~`eps · cond(−S)`. A graph with a tiny exit rate is
>   ill-conditioned — at θ₀=1e-8 the branching graph's exit rate is 1e-8, `cond ~ 1e8`, and
>   it loses ~8 digits (measured 1.5e-8). Still **six orders** better than the FD error it
>   detects (1e-2 … 5e-1), so it is a valid reference — but tolerances must respect it.
> * **No independent-FD oracle is offered, deliberately.** It re-uses the same forward, so
>   agreement with the library's FD is not independent evidence. That was a real
>   methodological error in the first audit and it should not be re-importable.
> * **MPFR was evaluated and REJECTED** (recorded in the module docstring so nobody retries
>   it): `PHASIC_FORCE_MPFR=always` gives byte-identical f64 results and a 1e-20
>   high-precision difference returns exactly `0.0`. It does not engage on the parameterized
>   path.
> * **`fd_sites_hit()`** — records each caller's `file:line`, so "all FD sites covered" is
>   *proven*, not asserted. Includes a test that the recorder still works (otherwise every
>   claim built on it would be vacuous), and a test that the exact-AD path touches **zero**
>   FD sites — the whole point of B3.
> * The graph zoo carries the traps as comments: use `branching()` for anything
>   reward/visit/zero-inflation shaped, because on a path-forced chain those are
>   topologically constant in θ and a test on them proves nothing (this produced two false
>   "bugs" during the audit).

*(original plan text follows)*

The thing that would have caught this is not more cases — it is **the right axis**.

* **Regime grid as a first-class axis of every gradient test:** θ magnitude 1e0 … 1e-15
  **× balanced vs MIXED scale**. The mixed-scale axis is what hid this bug from the refactor
  *and* from the first audit pass.
* **Oracle:** closed forms + `evaluate_trace_jax` exact AD (works on any graph).
* **Cross-backend differential:** FFI vs pybind vs trace-JAX must agree.
* **Site-coverage instrumentation:** monkeypatch `_fd_probe_points` to record each caller's
  `file:line`, so "all sites tested" is *proven*, not assumed. (This worked; keep it.)
* **Property-based generation** over graph shapes, weight modes, **signed coefficients**, and
  θ magnitudes.

### B3 — replace the backward, tier by tier, one site at a time

**Tier 1 — exact AD (free, exact, 40× faster).** For moments, sojourn, `joint_index`,
`reward_visit_probability`: `custom_vjp` with `fwd` = the fast FFI, `bwd` = `jax.grad`
through `evaluate_trace_jax`. *Gate: relerr ≤ 1e-12 across the whole regime grid.*

**Tier 2 — one-sided difference (universal, cheap, positivity-preserving by construction).**
For everything Tier 1 cannot reach — uniformized pdf, dph pmf, `stop_probability`, the daisy
chain, and `pmf_from_cpp` (which can *never* be AD'd). A 2nd-order forward scheme
(`(-3f(θ) + 4f(θ+h) − f(θ+2h)) / 2h`) measured **3.1e-10** at θ=1e-8 while never probing
below θ. This *also* deletes the whole `_fd_probe_points` mode-specific apparatus and closes
P2-2 (callback/formula probes) and N11 (DPH plus-probe breach) for free.

Step rule for Tier 2 is the one real design question: `h` must track the **sensitivity**
scale, not `|θ|`. Start from `h = cbrt(eps)·max(|θᵢ|, θ_scale)` and *tune it against the
regime grid*, which now exists.

**Tier 3 — adjoint FFI handlers (the long-term right answer).** Analytic adjoint of the
uniformization / elimination in C++, giving exact gradients at FFI speed for the Tier-2
quantities. Bigger job; do it once Tier 1+2 have stopped the bleeding. Note the project rule:
**adjoint of elimination/uniformization, not a dense matrix exponential.**

**Also convert the 3 preconditioner FD sites** (`svgd.py:3541, 3645, 3744`) — and fix the
larger hazard beside them: `_find_moment_matching_reference` (`svgd.py:3402-3436`) evaluates
the model at **θ = −2.0, −1.0** directly when `param_transform=None`. That is not an FD site
and is a bigger blast radius than the probe.

### B4 — adversarial verification

Multi-agent refutation of the fix, exactly as in this audit: independent reviewers, each told
to **default to REFUTED** and required to run code. That pass caught errors in my own report
that three careful solo passes did not. Make it standard, not exceptional.

---

## 3. Ordering

```
B1 (weight validation)  ──┐
                          ├──> B3 Tier 1 (exact AD)  ──> B3 Tier 2 (one-sided) ──> B4
B2 (harness + oracle) ────┘                                     │
                                                                └──> B3 Tier 3 (adjoint FFI)
```

B1 and B2 are independent and can go in parallel. **Do not start B3 before B2 exists** — the
regime grid is the only thing that will tell you whether the fix worked, and its absence is
precisely why this bug shipped.

## 4. What this fixes

| audit finding | fixed by |
|---|---|
| **G1 / N5** — relative step → ULP-noise gradients at small θ | B3 Tier 1 + Tier 2 |
| **G2** — negative rate accepted silently (signed coefficients) | B1 |
| **P2-2** — callback/formula probes reach θ ≤ 0 | B3 Tier 2 (one-sided never decreases θ) |
| **N11** — FD plus-probe breaches the DPH row-sum bound | B3 Tier 2 (step no longer inflates rows past the guard) — verify |
| **P2-3** — 3 unconverted preconditioner FD sites | B3, plus the `_find_moment_matching_reference` fix |
| the `log` floor question, the mode-specific probe rules | deleted entirely by Tier 2 |

Not fixed here, and tracked separately: **N1** (discrete + rewards uses the continuous reward
transform), **N2** (DPH moments are CTMC moments), **N3** (tied slaves exported as the 0.0
sentinel), **N4** (unvalidated reward length → OOB heap read). Those are independent
silent-wrong-number bugs — see `audit-phase2-adversarial-verdict.md`.
