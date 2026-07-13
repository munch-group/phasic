# Numerical audit — tentative findings

**Read the calibration section first. It is the most important part of this document.**

This is the residue of a long audit that produced a large number of confident claims,
**many of which turned out to be wrong**. What survives here is only what I personally
reproduced *and* what an adversarial multi-agent review failed to break. Everything else has
been deleted rather than softened.

Status: the relative FD step was **rolled back** (`12a30a78`). The FD gradient is an **open
problem**. Nothing in this document should be built on without re-verifying it first.

---

## 0. Calibration — how to read this

Claims are tagged:

| tag | means |
|---|---|
| **[R]** | **Reproduced.** I ran it; an adversarial reviewer independently ran it; it checks against a **closed form** or the library's own reference implementation. Re-runnable in seconds. |
| **[H]** | **Hypothesis.** Consistent with the evidence, not established. Do not act on it without testing. |
| **[X]** | **Retracted.** I claimed this and it is WRONG. Listed so nobody re-derives it. |

**The single failure mode of this audit**, and the reason to distrust its conclusions:
**every wrong claim came from verifying in one convenient case and assuming generality.**
The convenient cases were: a *path-forced* chain, *balanced* θ, a *callback/formula that was
arithmetically identical to linear*, and an *exposure index pointing at a dead coefficient
slot*. Each one silently satisfies the property under test for reasons unrelated to the code
being correct.

---

## 1. [R] The finite-difference gradient is broken — in BOTH step regimes

This is the core result and the only one I would defend.

### 1a. Absolute step (`eps = 1e-7`, master, and what the tree is on now)

For any rate below 1e-7 the minus-probe `θᵢ − 1e-7` is **negative**. Solvers accept a
negative rate without complaint. Original symptom: SVGD detonating at iteration 37 with
φ → ±5.9e22 and 2115 NaNs. A per-generation mutation rate is routinely **1e-8**, i.e. inside
the broken band.

*Reproduce:* `jax.grad` of `moments_from_graph` at `θ = [1e-8, 1.0]` returns a finite number
with no warning.

### 1b. Relative step (`h = max(1e-6·|θᵢ|, 1e-15)`, reverted in `12a30a78`)

Fixes 1a — but a relative step is well-scaled for the **argument** and does nothing to keep
the **function difference** above f's roundoff floor. With a small θᵢ beside an O(1) θⱼ
("mixed scale" — a mutation rate next to a coalescent rate, i.e. the library's core use case),
∂f/∂θᵢ stays O(1) while h collapses onto the 1e-15 floor:

```
theta_0=1e-06  h=1.0e-12  |delta_f|~1.98e-13  roundoff(f)~5.74e-16   SNR = 345
theta_0=1e-08  h=1.0e-14  |delta_f|~1.98e-15  roundoff(f)~5.74e-16   SNR = 3.4
theta_0=1e-09  h=1.0e-15  |delta_f|~1.98e-16  roundoff(f)~5.74e-16   SNR = 0.3   <-- noise
```

Measured against the **exact derivative of a closed form** (hypoexponential on a 3-state chain,
`E[T] = 1/(2t₀+3t₁) + 1/(t₀+2t₁)`):

| θ₀ (θ₁ = 2.0) | library | exact | rel err |
|---|---|---|---|
| 1e-2 | −0.09997 | −0.10107 | 1.1e-02 |
| **1e-8** | **−0.03053** | −0.09903 | **6.9e-01** |
| 1e-9 | −0.06939 | −0.09903 | 3.0e-01 |
| **1e-12** | **+0.09714** | −0.09903 | **SIGN FLIPPED** |

*Independently confirmed by three adversarial agents, who also showed the degradation is
inherent to the step (feeding the library's own probe points into an exact closed-form `f`
degrades identically) and requires **mixed scales** — scaling both components together gives
only ~1.9e-03.*

### 1c. [R] **The structural result: no central-difference step can fix this.**

A well-conditioned step for a mixed-scale parameter must be **larger than θᵢ itself** (the
sensitivity scale is O(1) while |θᵢ| is 1e-8). A **central** difference then *necessarily*
probes `θ − h < 0`. Measured at θ = [1e-8, 2.0]:

| scheme | h | lowest probe | rel err |
|---|---|---|---|
| central, `h = 1e-6·\|θ\|` (the reverted step) | 1e-14 | 1.0e-08 | **1.3e-02** |
| central, `h = 1e-6` **absolute** | 1e-06 | **−9.9e-07** | 3.0e-10 |
| **one-sided 2nd-order**, `h = 1e-5` | 1e-05 | **1.0e-08** | **3.1e-10** |

**Central FD cannot be both well-conditioned and positivity-preserving.** That is why the
refactor was trapped: it could have accuracy or positivity, and it chose positivity, quietly.

*Caveat:* the one-sided row is a single measurement on one chain. See **[H1]**.

---

## 2. [R] A negative transition rate is accepted silently

With **signed** edge coefficients, a **strictly positive** θ produces a negative weight.
`positive_params=True` / softplus keeps *θ* positive but cannot keep `dot(c, θ)` positive, and
**no finite-difference probe is involved**:

```
coefficients [1, -1],  theta = [1.0, 2.0]      (both strictly positive)
edge weights: [1.0, -1.0, 5.0]                 no exception
g.pdf(0.5)   = -1.303938951367671              <-- negative "probability"
g.moments(2) = [-0.8, 1.68]
```

**Scope is NOT established.** I verified this on the direct `Graph.update_weights` / pybind
path. I made a confident claim about which FFI handlers are and are not guarded, **and it was
wrong** — see **[X7]**. Treat the set of affected paths as unknown.

---

## 3. [R] An exception inside the FFI's OpenMP loop **aborts the process**

`GraphBuilder::build()` throws (non-positive `log` product; negative `formula` weight). It is
called from inside `#pragma omp parallel for` in several handlers. An exception cannot leave an
OpenMP structured block → `std::terminate`:

```
libc++abi: terminating due to uncaught exception of type std::invalid_argument: ...
Abort trap: 6
```

Uncatchable, no traceback, no checkpoint. Fires even at `OMP_NUM_THREADS=1` (the
`if(batch_size > 1)` clause still opens a single-thread region). **SVGD vmaps over particles,
so `batch_size > 1` is the normal path.**

**This is PRE-EXISTING** — reproducible on `log` mode alone, with no changes to the tree.
*Caveat:* observed on macOS/clang only; behaviour may differ by compiler/runtime.

**Consequence for any future fix:** adding *any* validation that throws from `build()` will
convert a silent wrong answer into a **process kill** unless OpenMP exception safety is fixed
first. The daisy-chain handlers already do this correctly (record under `omp critical`, report
after the loop); the others do not.

---

## 4. Hypotheses — plausible, NOT established

**[H1] The fix is architectural, not a better step.** A one-sided (forward-only) difference
never decreases θ, so it is positivity-preserving *by construction* — no floor, no clamp, and
**no per-weight-mode rules at all**. Measured once: 3.1e-10 at θ=1e-8 with the probe never
below θ. *Untested across graph shapes, weight modes, and quantities. Truncation is O(h) (or
O(h²) for the 3-point form) rather than O(h²) central — the trade needs measuring.*

**[H2] Exact gradients are obtainable and would remove the whole bug class.**
`E[T^k] = k!·α(−S)^{-k}·1` is exactly differentiable in JAX (`jnp.linalg.solve` has an exact
VJP). Its forward matched the library to ≤1e-15 on a chain **and** a branching graph, and its
gradient matched closed forms on both. **But an adversarial review found seven graph/mode
classes on which it silently lies** (defective IPV, non-linear weight modes, `log`, discrete,
unreachable states, and it mutates the graph it measures). So: *a* correct exact gradient
exists for continuous linear-weight non-defective graphs. Generality is **not** established,
and it does not cover the uniformized pdf, `stop_probability`, the daisy chain, or
`pmf_from_cpp` (opaque user C++).

**[H3] Weight validation belongs in `ptd_graph_update_weights` (C), not `GraphBuilder`.**
Every path — pybind, per-thread-cached FFI reuse, the builders — converges there. That is
where the (currently commented-out) validation already sits, `src/c/phasic.c:5733-5755`. I
placed a guard in `GraphBuilder` instead and it missed several live paths (**[X7]**). *The
awkwardness of the C error convention (`void` return + `ptd_err`, callers must check) is the
actual job, not a reason to put the guard elsewhere.*

**[H4] MPFR cannot serve as a high-precision oracle.** `PHASIC_FORCE_MPFR=always
PHASIC_MPFR_BITS=256` produced **byte-identical f64 results**, and a 1e-20 high-precision
difference returned exactly `0.0` (pure cancellation). It does not appear to engage on the
parameterized path. *Recorded so nobody spends a day on it. Not investigated further.*

---

## 5. [X] Claims I made that are WRONG — do not re-derive these

| # | I claimed | Reality |
|---|---|---|
| X1 | "All 12 FD sites produce correct gradients" | Tested only at **balanced θ ≈ [1,2]** — the one regime where the relative step is a no-op. At mixed scales: 69% error, sign flip. |
| X2 | `pmf_from_graph(g)(θ=[-0.5, 1.0])` returns a negative PMF | **Does not reproduce.** Those coefficients give weights (2.0, 1.5) — both positive. I conflated two different test graphs. The real route is signed coefficients (§2). |
| X3 | "`_check_weight` loudly rejects any negative weight" | **False.** `__init__.py:821` only tests `np.isfinite`. Never positivity. |
| X4 | "`linear` is the only weight mode with no validation" | Right conclusion, **wrong mechanism**. |
| X5 | "`cdf_zero`'s gradient is identically zero → bug" | **Test artifact.** On a path-forced chain every trajectory visits the rewarded vertex, so `cdf_zero = P(never visit) = 0` *topologically*. On a branching graph it is exact. |
| X6 | "`exposure` is silently ignored" | **Retracted.** `exposure_param_index=1` addressed a coefficient that is identically **zero** in the coalescent graph. Against a live slot, exposure works. |
| X7 | "A guard in `GraphBuilder` covers all nine FFI handlers and the pybind path" | **False.** `Graph.update_weights` (pybind) and `ComputeSojournTimesFfiImpl` (per-thread graph cache + direct `ptd_graph_update_weights`) both bypass `GraphBuilder` entirely. |
| X8 | "The negative-rate guard is raise-only and safe" | **It caused process aborts** (§3) and **fired on valid models** (an exactly-zero weight whose float64 dot is −1.86e-09 at popgen coefficient/θ scales — the `-1e-12` absolute tolerance must be relative). |
| X9 | The gradient oracle `jax.grad(sum(vertex_rates))` from `evaluate_trace_jax` | **WRONG.** `vertex_rates` is per-vertex `1/exit_rate`, **not weighted by visit probability**, so it equals `E[T]` only on a path-forced graph. On a branching graph: forward 2.833 vs truth 1.0; gradient `[-1.111, -0.361]` vs `[-0.333, -0.333]`. **Its own "is the oracle an oracle" test passed, because that test used the forced chain.** |
| X10 | "No DPH row-sum breach is possible under the plus probe" | **Refuted.** True only for weights homogeneous of degree ≤1 in θᵢ. `weight_formula="c0*t0**200"` at a row sum of exactly 1.0 breaches. |
| X11 | Various `file:line` citations | Went stale after edits and were not re-derived. **Re-grep before trusting any line number in any audit doc.** |

---

## 6. Methodology — the part most worth keeping

Each of these is a lesson paid for with a wrong conclusion above.

1. **A path-forced chain (`s→a→b→absorb`) visits every vertex with probability 1.** Any quantity
   that depends on visit probability looks correct even when the code ignores visit
   probabilities entirely. → **Always use a BRANCHING graph** for anything reward / visit /
   zero-inflation / sojourn shaped. (Cost: X5, X9, and a false mode-inertness finding.)
2. **Balanced θ hides FD conditioning.** Scaling all components together keeps the step
   well-conditioned. → **Make mixed-scale θ a first-class test axis.** (Cost: X1 — and it is why
   the bug shipped in the first place.)
3. **A `callback`/`formula` that is arithmetically identical to `linear` proves nothing.** →
   Make them genuinely non-linear (e.g. `c0*t0*t1`).
4. **Verify the verifier.** An oracle/harness must be validated on a case where a *wrong*
   version cannot pass. A self-test that shares the instrument's blind spot is worthless.
   (Cost: X9.)
5. **`isfinite` / `> 0` / "does not raise" tests pass on wrong answers.** Every bug here produced
   finite, plausible, wrong numbers. **Assert VALUES against an independent reference.**
6. **Two estimators agreeing that a CONSTANT has gradient 0 proves nothing.** Assert the
   gradient is non-trivial.
7. **An "independent" finite difference that re-uses the same forward is not independent.** Its
   agreement with the library's FD is a restatement of the implementation.
8. **Adversarial multi-agent review, with agents told to default to REFUTED and required to run
   code, caught what three careful solo passes did not.** For work of this complexity it is not
   a luxury tier.

---

## 7. What I would do next (and would test before believing)

1. **Fix OpenMP exception safety first** (§3). Until then, any validation that throws turns a
   silent wrong answer into a process kill. This is a pre-existing bug worth fixing on its own.
2. **Then** add weight validation at the C layer (**[H3]**), validating `dot(c, θ)` — **not** the
   sign of θ — with a **relative** tolerance (**[X8]**).
3. **Then** the FD step, as an architectural change (**[H1]** / **[H2]**), gated by a harness whose
   regime grid includes **mixed scales** and whose oracle is validated on a **branching** graph.

Do not build step 3 on this document. Re-derive it.
