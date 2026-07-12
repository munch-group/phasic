# Numerical Refactor — Hand-off

**Status: TENTATIVELY CORRECT. Not signed off.**

Everything below was verified against *targeted probes*, not against phasic's full
flag surface. The probes that were run all pass, and several long-standing bugs were
found and fixed along the way — but the sample of code paths exercised is far smaller
than the set of paths the library actually has. Treat every claim here as *"held up
under the checks that were run"*, not *"proven"*.

The work started as a single bug fix and kept widening because each fix exposed the
next latent bug. That is exactly the pattern that means it needs an independent,
systematic re-verification — which is what §9 is for.

---

## 1. READ THIS FIRST — why "diff against the old code" is the WRONG test

The refactor **deliberately changes numerical output**. Every finite-difference
gradient in the library now uses a *relative* step (`1e-6·|θ|`) instead of a fixed
*absolute* step (`1e-7`). That changes gradient values everywhere, which changes SVGD
trajectories, which changes posterior particles.

So a re-verification **cannot** work by comparing outputs against `git stash` / `HEAD~`.
Old and new *should* differ. The old values were, in an identifiable regime, simply
**wrong** (sign-flipped and inflated by ~12 orders of magnitude).

Correctness must therefore be established against **independent references**:

- closed forms (Erlang / NegBinomial / geometric / hypoexponential on small chains),
- **cross-backend agreement** (XLA-FFI vs pybind vs ctypes-JIT computing the same thing),
- analytic derivatives where they exist,
- convergence behaviour (e.g. continuous PDF error must fall as `1/granularity`).

Forward (non-gradient) values *should* be unchanged. Those **can** be diffed against the
old code, and that is a cheap, high-value check.

---

## 2. State snapshot (a fresh session has zero context)

- Repo: `/Users/kmt/phasic`, branch `master`, **nothing committed** — all changes are in
  the working tree.
- Build: `pixi run install-dev`. **Non-editable** install (`pip install … .`, no `-e`) —
  the package is a *copy* under `.pixi/envs/default/lib/python3.13/site-packages/phasic`.
  **Editing `src/` has no effect until you re-run `pixi run install-dev`.**
- Three compute paths need the C/C++ sources on disk and therefore require
  **`PHASIC_SOURCE_DIR=/Users/kmt/phasic`** (or an editable install):
  `moments_from_graph`, `pmf_from_cpp`, `pmf_from_graph_parameterized`.
  Without it they raise / skip. `_get_package_dir()` (`src/phasic/__init__.py:545`) honours it.
- The full test suite takes **~8 hours serially** on this machine (no `pytest-xdist`).
  A 21-target subset covering the touched paths runs in ~32 min — that subset is the
  gate that was actually used. See `§5`.
- `tests/pytest/failing_tests.md` documents **pre-existing** failures. Full-suite green
  was never a valid gate.

---

## 3. What changed — 7 files (4 source, 3 test), nothing committed

| file | change |
|---|---|
| `src/phasic/__init__.py` | `_fd_probe_points` helper (**:702**) + constants (**:687-693**); 11 FD backward passes rewired; `_compile_wrapper_library` build fixes; moments recursion fix; discrete `normalize()` removal |
| `src/phasic/ffi_wrappers.py` | 12th FD site `_rvp_bwd` (**:1325**); `_weight_mode_of` helper |
| `src/phasic/svgd.py` | `_check_negative_pmf` (**:356**) + `_PMF_LOG_OFFSET`/`_PMF_NEG_TOL` (**:347/:353**); wired into both branches of `_log_lik_from_pmf` (**:5887, :5951**) |
| `src/cpp/parameterized/graph_builder_ffi.cpp` | negative-rate rejection: `kMinEdgeWeight` (**:1238**), `most_negative_edge_weight` (**:1241**), checked in both daisy handlers |
| `tests/pytest/test_daisy_chain_c_path.py` | `test_grad_matches_finite_diff` rewritten; `test_fd_backward_never_probes_a_negative_rate` added |
| `tests/pytest/inference/test_jax_integration.py` | `TestMomentsFromGraphValues` added |
| `tests/pytest/test_gate_ffi_vs_pybind.py` | `test_g1_discrete_pmf_is_exact_not_discretised`, `test_g1_discrete_pmf_actually_depends_on_theta` added |

### The 12 finite-difference sites (all now route through `_fd_probe_points`)

`src/phasic/__init__.py`: **3646, 3774, 4147, 4587, 4830, 6577, 6833, 7076, 7133, 7565, 9648**
`src/phasic/ffi_wrappers.py`: **1325**

The step rule (`__init__.py:702`):

```python
h  = max(_FD_REL_STEP * |θ_i|, _FD_MIN_STEP)      # 1e-6 relative, 1e-15 floor
lo = θ_i - h ;  if nonneg: lo = max(lo, _FD_MIN_THETA)   # 1e-15, STRICTLY positive
hi = max(θ_i + h, lo + h)                          # guarantees hi > lo
denom = hi - lo                                    # the ACTUAL separation, not 2h
```

> **CORRECTED (audit Phase 1, finding F1).** The original text here said "`'log'` (θ is a
> log-scale, legitimately negative)". **That was wrong.** `'log'` means
> `weight = Π(cᵢ·θᵢ)` computed in log-space (`__init__.py:1765`,
> `phasic_pybind.cpp:1494`), and the C layer **raises** if any `(cᵢ·θᵢ)` product is
> non-positive (`phasic.c:5712`). θ under `'log'` must be **strictly positive** — a
> *stricter* constraint than `'linear'`, which tolerates θ = 0. The probe rule is now
> per-mode, not a boolean:
>
> - **`'linear'`** — minus-probe floored at `_FD_MIN_THETA` (strictly positive).
> - **`'log'`** — purely **multiplicative** probe `θᵢ·(1 ± _FD_REL_STEP)`: sign-preserving
>   and zero-avoiding at any magnitude, no absolute floor. (The old `nonneg=False` left a
>   hole at θ ≤ 1e-15, where the minus-probe crossed zero and `jax.grad` raised even though
>   the forward was valid.)
> - **`'callback'` / `'formula'` / `pmf_from_cpp`** — no floor; the user's expression may be
>   `c0*t0` or `exp(t0)` and its validity domain is not ours to assume.
>
> `_fd_probe_points(theta, i, weight_mode)` now takes the mode, not a `nonneg` bool.

---

## 4. Bugs found — five, all pre-existing

| # | bug | evidence | fixed? | confidence |
|---|---|---|---|---|
| 1 | **Absolute FD step drives θ negative.** `θ − 1e-7 < 0` for any rate below 1e-7 (routine: a per-generation mutation rate is ~1e-8). Solvers accept a negative rate **silently** and return negative "probabilities" (min −0.42). Gradient flips sign and inflates ~1e6. | Reproduced end-to-end: SVGD blew up at iteration 37, φ → ±5.9e22, `NaN count: 2115` — byte-identical to the user's error. | yes | **high** — reproduced and the fix demonstrably prevents it |
| 2 | **`moments_from_graph` returns wrong moments.** Recursion did `rewards3[j] = rewards2[j] * pow(rewards2[j], i)` — raising each entry to a power instead of feeding the previous waiting-time vector back in as rewards. | E[T²] = **10** on Erlang(2,1) where the truth is **6**. After fix: `[2, 6, 24, 120]` = (n+1)!, and agrees with the FFI path. | yes | **high** — closed form + cross-backend |
| 3 | **`_compile_wrapper_library` cannot compile.** (a) missing `-I src/c` (`api/c/phasic.h` includes `"phasic_log.h"`, which lives in `src/c/`); (b) source list missing `scc_synthetic.c`, `scc_compose.c`, `api/cpp/scc_graph.cpp`; (c) `.c` files compiled with `g++` as C++ (`scc_compose.c` has a `goto` past an initialiser — legal C, illegal C++). | Three successive compile/link failures. CMake's canonical include/source set (`CMakeLists.txt:33,56`) was used as the reference. | yes | **medium** — it now builds and the outputs cross-check, but this is build machinery I rewrote and only exercised on this machine/toolchain |
| 4 | **`pmf_from_cpp(discrete=True)` destroys the distribution.** Wrapper called `g.normalize()` — the *continuous* normalize — which rescales every vertex's outgoing weights to sum to 1. In a DPH the weights **are** the per-step transition probabilities, so this turns the chain into a deterministic walk. | Returned `[1,0,0,0,0]` (i.e. `P(T=2)=1`) instead of NegBinomial; **gradient identically 0** because the rescaling divides θ out. FFI handler never normalizes and is correct. After fix: bit-identical to FFI (0.000e+00), exact to 1.7e-15. | yes | **high** — closed form + bit-identity with the FFI path |
| 5 | **`pmf_from_graph_parameterized` is broken.** `AttributeError: 'NoneType' object has no attribute 'ShapeDtypeStruct'` at `__init__.py:535` — module-level `jax` is `None` (JAX symbols are lazily imported). Same function also requests `jnp.float32` while the FFI requires F64. | Surfaced once the JIT compile started working. | **NO** | it has no FD backward, so nothing in this refactor touches it. Left alone deliberately. |

**Why bugs 2–4 survived**: their only tests live in
`tests/pytest/inference/test_jax_integration.py`, which has a **module-level `skipif`**
that skips the entire file when the C sources aren't on disk — i.e. always, under a
non-editable install. And even those tests only assert `isfinite` and `> 0`, which a
numerically wrong answer passes happily. Both the moments bug and the discrete-normalize
bug would have passed them.

---

## 5. What WAS verified

- **Gradient correctness, all 12 FD sites.** Each compared against an *independent*
  forward-only finite difference using a **different step size** (so agreement is
  meaningful, not tautological). Agreement: **5e-10 … 2e-08** relative, across θ from
  1.0 down to 1e-12. One site hits an analytic value exactly (−7 at θ=(1,1) for
  `moments_from_graph`).
- **No probe reaches a non-positive rate** — verified down to θ = 0 and θ = 1e-30.
- **Discrete PMF is still exact** — `dph_pmf` matches closed-form NegBinomial to
  **1.7e-15** (vs the continuous PDF's ~4.9e-4 uniformization error, which converges
  first-order in granularity: 5.0e-4 → 5.0e-5 → 5.0e-6 → 5.0e-7).
- **Cross-backend bit-identity**: JIT `pmf_from_cpp` vs FFI `pmf_from_graph` → **0.000e+00**.
- **The original failure is fixed.** The 120-particle repro now completes all 100
  iterations with φ bounded in [−17.9, −8.8] and every particle sitting in what used to
  be the fatal regime. Previously: ±5.9e22 and `NaN count: 2115` at iteration 37.
- **Regression**: a 21-target subset covering every touched path, run **three times**
  across the change set (after the FD rewrite; after the build + moments fixes; after the
  discrete fix) — **522 passed, 0 failed** every time. Skips went 47 → 49 (the two new
  `TestMomentsFromGraphValues` tests, which skip without `PHASIC_SOURCE_DIR`).
  `test_gate_ffi_vs_pybind.py` is **not** in that subset, so the two new discrete tests
  were run separately: **7 passed, 1 xfailed** for that file.
  The exact subset is listed in the scratch file `gate_tests.txt`; it is
  `test_svgd.py`, `inference/`, the daisy/epoch files, the svgd/joint-index/weight-formula
  files, `test_method_of_moments.py`, `test_mcmc.py`, and the gate files.

---

## 6. What was NOT verified — the honest gaps

1. **The flag surface is largely unprobed.** Gates were run on a handful of
   configurations. phasic's compute surface is a *combinatorial matrix*
   (`discrete` × `weight_mode` × backend × `use_ffi`/`use_cache` × `rewards` ×
   `fixed`/`tied` × `joint_index` × `epoch_starts` × `final_read` × `exposure` × …).
   Most cells were never touched. **This is the main reason for the audit in §9.**
2. **The full test suite was never run to completion** (~8 h serially). The 21-target
   subset is a proxy, not a proof.
3. **`_compile_wrapper_library` was rewritten** (C/C++ split compile + link). It works
   on this machine/toolchain. It is untested on Linux, and it is slower (7 compile steps
   instead of 1).
4. **`pmf_from_graph_parameterized` remains broken** (bug 5).
5. **19 pre-existing failures** in `inference/test_jax_integration.py` are now *visible*
   when `PHASIC_SOURCE_DIR` is set (the module used to skip entirely). They are not mine
   — deprecated test API (`param_length == 0`) plus bug 5 — but they are now unmasked.
   The default suite is unchanged (module still skips).
6. **Weak tests were not swept.** An unknown number of existing tests assert only
   `isfinite` / `> 0` / shape, and would pass on wrong answers.

---

## 7. Three methodological traps that already bit me — do not repeat them

1. **A tautological gradient check proves nothing.** Comparing the library's FD against
   a hand-rolled FD *using the same step size* is a restatement of the implementation,
   not a test. The original `test_grad_matches_finite_diff` did exactly this (it
   hard-coded `eps = 1e-7` and asserted machine-precision agreement). **Always use a
   different step for the reference.**
2. **Two estimators agreeing on a *constant* prove nothing.** My first gate on
   `pmf_from_cpp(discrete=True)` *passed* — because the forward was constant in θ, so
   both estimators agreed the gradient was 0. It was masking a total distribution
   collapse. **Always assert the gradient is non-trivial.**
3. **`isfinite` / `> 0` tests pass on wrong answers.** This is precisely how the moments
   bug (E[T²] = 10 instead of 6) survived. **Assert VALUES against a reference.**

---

## 8. The KEY structural fact the audit should exploit

**The refactor cannot have changed any forward (non-gradient) value.**

This was checked mechanically: every added/changed line in the Python sources lives in
exactly one of

- a `*_bwd` (custom_vjp backward) function,
- a closure-scope constant feeding a `*_bwd` (`fd_nonneg = …`),
- `_compile_wrapper_library` or the JIT wrapper C++ strings,
- a **raise-only** guard (`_check_negative_pmf`, the C++ negative-rate check) — these can
  only *raise*, never alter a returned value. (`jnp.log(pmf + 1e-10)` became
  `jnp.log(pmf + _PMF_LOG_OFFSET)` where `_PMF_LOG_OFFSET == 1e-10` — same number.)

Therefore, on **every path that worked before**, forward values must be **bit-identical**
to `HEAD`. The only intended forward-value changes are:

| path | why it changed |
|---|---|
| `moments_from_graph` | bug 2 — moment recursion was wrong. Previously **uncompilable** anyway. Note `use_ffi` is **inert** here (`:6441` only gates an eager `_ensure_jax_active`; `:6451` imports jax regardless), so **both** settings take the JIT path — the whitelist covers the entry point in full, not just `use_ffi=False`. |
| `pmf_from_cpp(discrete=True)` | bug 4 — `normalize()` collapsed the DPH. Previously **uncompilable** anyway. |
| the 3 JIT paths generally | bug 3 — previously **uncompilable**. No "before" exists. |

**Audit Phase 1 confirmed this claim** (`audit-phase1-forward-parity.md`): 167 cells
bit-identical vs HEAD, **0 forward-value differences, 0 unintended new raises**.

Batch A of the follow-up (`audit-f1-f2-fix-plan.md`) adds **two deliberate** new raises,
which are *not* regressions — they convert silently-wrong answers into loud errors:

| path | new behaviour | why |
|---|---|---|
| `moments_from_graph` (weight_mode ≠ `'linear'`) | raises `ValueError` | it JIT-generates a **linear** `build_model`, so a `log`/`callback`/`formula` graph silently got LINEAR moments (E[T]=0.325 where the truth was 0.75). Use `pmf_and_moments_from_graph`, which honours every mode. |
| `pmf_from_graph_joint_index` / `daisy_chain_joint_probs` (weight_mode `'log'`) | raises `ValueError` | the sojourn + daisy FFI handlers hardcode `/*use_log=*/false` (`graph_builder_ffi.cpp:887, 941, 1528, 1782, 1827`), so a `'log'` graph silently got LINEAR weights. `callback`/`formula` are honoured and unaffected. |

**This gives the audit a cheap, exhaustive Phase 1**: diff forward values against `HEAD`
across the whole flag matrix. Anything that differs, outside the table above, is a
regression. No reasoning needed per cell — just a diff.

Gradients are the opposite: they **should** differ, and must be validated against
*independent references*, never against `HEAD`.

---

## 9. THE AUDIT PROMPT (copy-paste, phased)

Run the phases **in separate sessions**. Each is self-contained and each ends with a
written artifact the next phase reads, so you can stop after any phase without losing
work. Phase 1 is by far the cheapest and catches the most; do not skip it.

> **Budget note.** Phase 1 is mostly mechanical (a diff harness + a matrix sweep) and is
> cheap. Phases 2–3 are reasoning-heavy per cell. Phase 4 is a broad sweep. If budget is
> tight, run **1 → 3 → 2 → 4**; Phase 3 catches whole-distribution collapses that Phase 2
> can miss.

---

### PHASE 1 — Forward-value parity (cheapest, highest yield)

```text
ultracode

You are auditing an uncommitted numerical refactor in /Users/kmt/phasic (branch master).
Read /Users/kmt/phasic/numerical-refactor-handoff-plan.md FIRST — it is the full
state snapshot. You have zero prior context; trust nothing you have not read.

BUILD: `pixi run install-dev` after EVERY source change (the install is a non-editable
COPY into .pixi/envs/default/.../site-packages/phasic — editing src/ alone does nothing).
Three compute paths (moments_from_graph, pmf_from_cpp, pmf_from_graph_parameterized)
additionally require PHASIC_SOURCE_DIR=/Users/kmt/phasic.

CLAIM UNDER TEST: the refactor changed ONLY gradient (backward-pass) code and added
raise-only guards. Therefore every FORWARD value must be BIT-IDENTICAL to git HEAD,
except on three paths listed in §8 of the hand-off (moments_from_graph,
pmf_from_cpp(discrete=True), and the JIT paths generally — all of which were previously
uncompilable or provably wrong).

DO THIS:
1. Enumerate phasic's public compute surface and its FLAG MATRIX. Do not guess — read the
   code. At minimum: Graph.svgd(); pmf_from_graph; pmf_from_graph_parameterized;
   pmf_from_cpp; pmf_from_graph_joint_index; moments_from_graph;
   pmf_and_moments_from_graph(+_multivariate); joint_prob_graph / joint_stop_prob_graph /
   joint_sojourn_graph / daisy_chain_joint_probs; reward_visit_probability.
   Flags to cross: discrete, weight_mode (linear|log|callback|formula), use_ffi, use_cache,
   granularity (incl. 0=auto), rewards, fixed, tied, theta_dim, joint_index, epoch_starts,
   final_read (sojourn|stopprob), daisy_chain_t_eval (numeric|None|'auto'), exposure(+index),
   nr_moments, discrete joint graphs, zero-inflation/partial reward coverage.
   Record which combinations are VALID, which RAISE, and which are SILENTLY OVERRIDDEN
   (e.g. joint_index forces discrete=True) — silent overrides are correctness traps.

2. Build a harness that evaluates the FORWARD output of every valid cell on a small,
   deterministic model, and dumps it to a JSON keyed by the flag combination.

3. Produce the same dump from git HEAD. Use `git worktree add` to a temp dir — DO NOT
   `git stash` and DO NOT `git checkout -- .` (there are uncommitted notebooks in the tree
   that must not be touched). Build the worktree separately.

4. Diff. Every difference outside the three whitelisted paths is a REGRESSION — report it
   with the flag combination, both values, and the code path.

5. Also report every cell that RAISES on one side and not the other.

Write findings to /Users/kmt/phasic/audit-phase1-forward-parity.md, including the full
flag matrix you derived (later phases reuse it).
```

---

### PHASE 2 — Gradient correctness across all 12 FD sites

```text
ultracode

Continue the audit of /Users/kmt/phasic. Read numerical-refactor-handoff-plan.md and
audit-phase1-forward-parity.md (for the flag matrix) FIRST. Zero prior context.

The refactor rewrote EVERY finite-difference backward pass in the library (12 sites) to
use a RELATIVE step (1e-6*|theta|, floor 1e-15) instead of an absolute one (1e-7), and
to floor the minus-probe at a strictly-positive 1e-15 for rate-typed theta.
Sites: src/phasic/__init__.py:3646,3774,4147,4587,4830,6577,6833,7076,7133,7565,9648
and src/phasic/ffi_wrappers.py:1325. The helper is __init__.py:_fd_probe_points (:702).

GRADIENTS ARE *SUPPOSED* TO DIFFER FROM HEAD. Do NOT compare against HEAD. Validate
against INDEPENDENT references.

For EVERY FD site, and for EVERY flag combination that reaches it (use the Phase-1 matrix):

  a) Compare jax.grad through the library's custom_vjp against an INDEPENDENT
     forward-only finite difference that uses a DIFFERENT step size. Agreement between two
     estimators with the SAME step is a tautology, not a test — the original
     test_grad_matches_finite_diff made exactly that mistake.

  b) ASSERT THE GRADIENT IS NON-TRIVIAL. Two estimators agreeing that a CONSTANT function
     has gradient 0 proves nothing. This already produced a false PASS during the refactor,
     masking a total distribution collapse in pmf_from_cpp(discrete=True). If |grad| is ~0
     everywhere, the forward probably does not depend on theta at all — investigate, do not
     pass.

  c) Verify no probe reaches a non-positive rate. Sweep theta down to 0 and 1e-30.
     For weight_mode='log' (theta is a log-scale) verify the probe is NOT floored and a
     negative theta stays negative.

  d) For DISCRETE models: a DPH row sum is a probability and phasic VALIDATES (<=1.0001)
     rather than normalising. The PLUS probe grows weights — verify it never breaches the
     bound, including at a row sum of exactly 1.0.

  e) Where an analytic derivative exists (small chains: Erlang, hypoexponential,
     NegBinomial, geometric), check against it, not just against another FD.

Report every mismatch with theta, the flag combination, both values, and file:line.
Write to /Users/kmt/phasic/audit-phase2-gradients.md.
```

---

### PHASE 3 — Distribution invariants (catches whole-distribution collapses)

```text
ultracode

Continue the audit of /Users/kmt/phasic. Read numerical-refactor-handoff-plan.md and
audit-phase1-forward-parity.md FIRST. Zero prior context.

Verify these INVARIANTS hold on every path and flag combination that can reach them.
Each of these catches a class of bug that a gradient check cannot:

1. DISCRETE PMF IS EXACT. dph_pmf(jumps) takes no granularity and steps the chain exactly.
   On a 2-phase chain with both transition probabilities p, the jump count is
   NegBinomial(2,p): P(T=n) = (n-1) p^2 (1-p)^(n-2). Require MACHINE PRECISION
   (rtol 1e-12), not "isfinite". Do this for every discrete-capable entry point:
   pmf_from_graph (both the linear/FFI branch AND the callback branch),
   pmf_and_moments_from_graph, pmf_from_cpp, pmf_from_graph_parameterized.
   A bug of exactly this kind was found: pmf_from_cpp(discrete=True) called the CONTINUOUS
   normalize(), which rescales each vertex's outgoing weights to sum to 1 and collapses the
   DPH to a DETERMINISTIC walk (P(T=n_phases)=1). Its gradient was identically zero.

2. THE PMF ACTUALLY DEPENDS ON THETA. Assert out(theta_a) != out(theta_b). This is the
   direct guard against (1)'s failure mode.

3. CONTINUOUS PDF CONVERGES. Its error vs closed form must fall ~10x per 10x granularity
   (first-order). At the default auto-granularity = max(2*max_rate, 1000) expect ~5e-4 on a
   rate-1 chain. If it does NOT converge, the uniformization is broken.

4. CROSS-BACKEND IDENTITY. The same model computed via XLA-FFI, via pybind GraphBuilder,
   and via the ctypes JIT wrapper must agree (bit-identical where the algorithm is
   identical; to ~1 ulp otherwise). Enumerate every FFI handler in
   src/cpp/parameterized/graph_builder_ffi.cpp and find its Python caller; flag any handler
   with NO caller and any Python entry point with no test.

5. MOMENTS ARE RAW MOMENTS. E[T^n] for Erlang(2, rate=1) is (n+1)!, i.e. [2, 6, 24, 120].
   moments_from_graph previously returned E[T^2] = 10. Check moments_from_graph AND
   pmf_and_moments_from_graph AND the *_multivariate variant against closed form.

6. PROBABILITIES ARE NON-NEGATIVE AND SUM CORRECTLY. Joint-prob tables should sum to ~1
   (report the deficit). No PMF value may be negative.

Write to /Users/kmt/phasic/audit-phase3-invariants.md.
```

---

### PHASE 4 — Weak-test sweep and gap closure

```text
ultracode

Continue the audit of /Users/kmt/phasic. Read numerical-refactor-handoff-plan.md and the
three audit-phase*.md reports FIRST. Zero prior context.

TWO JOBS.

JOB A — find the WEAK TESTS. Sweep tests/pytest/** for tests that assert only
`isfinite`, `> 0`, `.shape`, `len(...)`, or merely "does not raise". Such a test PASSES ON A
WRONG ANSWER. This is not hypothetical: moments_from_graph returned E[T^2]=10 instead of 6
for years, and its tests (inference/test_jax_integration.py::TestMomentsFromGraph) asserted
only isfinite and > 0 — so they passed.
For each weak test: state what it SHOULD assert (a value, against a closed form or a
cross-backend reference) and rewrite it. Run the suite after each batch; commit on green.

JOB B — close the coverage gaps found in Phases 1–3. Write real tests for every uncovered
flag combination, asserting VALUES.

CONSTRAINTS:
- The full suite takes ~8 HOURS serially (no pytest-xdist). Do not attempt a full-green
  gate. Use a targeted subset covering the paths you touch, and say which subset you used.
- tests/pytest/failing_tests.md documents PRE-EXISTING failures. Classify every failure as
  pre-existing or new before reporting it. Never claim a pre-existing failure as a
  regression, and never "fix" a test by weakening its assertion.
- Some paths are SOURCE-DEPENDENT (need PHASIC_SOURCE_DIR). Tests for them belong in a
  module guarded on source availability, and they will SKIP in a default run — say so.

Write to /Users/kmt/phasic/audit-phase4-tests.md.
```

---

## 10. Appendix — how to reproduce the original failure

The bug that started this. On the pre-refactor code, this blows up at ~iteration 37 with
`XlaRuntimeError → ValueError: Model returned NaN PMF values … NaN count: 2115`:

- `docs/pages/tutorial/svgd-joint-prob.ipynb`, the cell using
  `joint_prob_graph(..., discrete=False)` + `svgd(..., joint_index=True,
  epoch_starts=[0, 0.05371094, 0.15234375], daisy_chain_t_eval='auto')`.
- Root cause: `epoch_starts` are in the wrong time units (they are ~0.05 and ~0.10
  *generations* against a rate of ~1e-4/generation, so epochs 1–2 carry **no information**
  — expected coalescences ≈ 5.4e-05). Their particles drift downward until θ < 1e-7, at
  which point the old absolute FD step probed a **negative rate**.
- Note the notebook cell is *also* mis-specified independently of the library bug: it picks
  up a stale `mutation_rate = 1e-4` from an earlier toy cell instead of the msprime
  `mut_rate = 1e-8`, and drops the `exposure=tree_spans, exposure_param_index=1` that the
  preceding cell correctly used. Fixing the library stops the crash; it does not make that
  cell a correct model. The intended fix on the user side is to rescale to **coalescent
  time units** so θ ≈ 1.
- Also: `daisy_chain_t_eval='auto'` is a **no-op** under the default `final_read='sojourn'`
  — the sojourn FFI handler never reads `t_eval` — yet `svgd()` still runs the expensive
  probe and discards the result (`__init__.py`, the `_resolve_daisy_chain_t_eval` call).
  Worth fixing or documenting.
