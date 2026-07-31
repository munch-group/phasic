# B3 exact-gradient — MERGE REVIEW & HANDOFF

Branch: **`fd-b3-experiments`** (tip `505dc2ab`). Base: master `c673be83`.
Everything below is on the branch; master's install was untouched throughout
(built in an isolated worktree pixi env).

---

## 1. TL;DR — what this delivers

Replaces the **finite-difference** `custom_vjp` backward for phase-type **moments**
with an **exact reverse-mode θ-adjoint** over phasic's elimination tape, fixing the
mixed-scale FD gradient defect. Exposed as an **opt-in** kwarg
`Graph.pmf_and_moments_from_graph(..., exact_moment_grad=True)`; **default off =
FD path byte-identical to before.**

Scope shipped: **continuous · weight_mode='linear' · monolithic** moment **vector**
(all `nr_moments`). Exact under `jax.grad` AND `jax.vmap(jax.grad)` (SVGD-safe),
MPFR-safe (declines → FD when the primal would use MPFR). Everything out of scope
(discrete/`was_dph`, log/formula, joint-index, hierarchical SCC) **falls back to FD**
— correct, just not accelerated/de-noised.

The previously `xfail(strict=True)` mixed-scale pin
`tests/pytest/inference/test_fd_gradient_mixed_scale.py::test_fd_gradient_correct_at_mixed_scale`
is **flipped** (now passes): exact grad matches the closed-form oracle to rtol=1e-9
at θ=[1,1e-8] where FD was 9% wrong (359% wrong on the 2nd moment).

---

## 2. Merge mechanics

- **Clean auto-merge onto master `c673be83`.** Master's only post-base commit
  (`c673be83`, "Removed mistaken check for rate <= 1") edits `__init__.py` at
  ~lines 2734/2797 (`discretize`); my `__init__.py` edits are at ~6826/6931/7460
  (`pmf_and_moments_from_graph`). Non-adjacent → no conflict. No other source file
  overlaps.
- **Rebuild after merge:** the install is a *copy*, so `pixi run install-dev` is
  required after merging (Python edits included). Set `PHASIC_SOURCE_DIR=/Users/kmt/phasic`
  for FFI/JIT tests.
- **Remote:** `origin` = `git@github.com:munch-group/phasic`. Not pushed (solo
  review — kept local). If you ever want CI/PR: `git push origin fd-b3-experiments`.

### Recommended merge (solo, clean single master commit)
The branch has ~16 batch/de-risk commits — a valuable record, but noisy for master.
Rather than rewrite branch history (risky, and interactive rebase isn't available
here), **squash at merge** so master gets ONE clean commit while the branch keeps
its full trail:
```bash
cd /Users/kmt/phasic
git worktree prune                          # after the scratch worktree is gone
git checkout master
git merge --squash fd-b3-experiments        # stages the whole net diff, no commit yet
git commit -F - <<'MSG'
feat(gradient): exact reverse-mode moment gradient replacing FD custom_vjp (B3)

Opt-in Graph.pmf_and_moments_from_graph(..., exact_moment_grad=True): replaces the
finite-difference moments backward with an exact reverse-mode theta-adjoint over the
elimination tape (continuous/linear/monolithic moment vector), fixing the mixed-scale
gradient defect. Exact under grad+vmap, MPFR-safe, default off = FD unchanged. Flips
the strict-xfail mixed-scale pin. De-risk validators compile-guarded behind
PHASIC_B3_VALIDATORS (off). See B3-MERGE-REVIEW.md.
MSG
pixi run install-dev
```
Alternatives: `git merge --no-ff fd-b3-experiments` keeps the batch history on master;
plain `git merge` fast-forwards all 16 commits in. Squash is recommended for a clean
master. (If you specifically want the *branch* rewritten into 2-3 commits instead,
that needs a `git reset --soft <base>` + re-commit by file group — say the word.)

---

## 3. What SHIPS (the production diff — review focus, ~150 lines)

| Area | Symbol / change | File:line |
|---|---|---|
| C core | `ptd_moments_grad_theta` — exact Jacobian `d[m_0..m_{K-1}]/dθ` via the moment-chain reverse + edge→θ contraction | `src/c/phasic.c:10725` |
| C core | `ptd_dbg_tape_needs_mpfr` — MPFR/conditioning safety gate (`#ifdef HAVE_MPFR`) | `src/c/phasic.c:10633` |
| Header | decl of `ptd_moments_grad_theta` | `api/c/phasic.h` |
| C++/pybind | `Graph::moments_grad_theta` + `_moments_grad_theta` binding | `api/cpp/phasiccpp.h`, `src/cpp/phasic_pybind.cpp` |
| Python | `exact_moment_grad=False` kwarg | `src/phasic/__init__.py:6826` |
| Python | exact-grad setup (`_exact_grad_enabled`, `_exact_moments_jac_np` host callback over `graph.clone()`) | `:6931`, `:6938` |
| Python | `model_bwd`: `_exactJ` pure_callback → `_exact_tbm = J^T·g_moments` → swap the moments FD term (FD fallback on NaN) | `:7460–7515` |
| Tests | pin flip + 2nd-moment exact + FD-baseline pins | `tests/pytest/inference/test_fd_gradient_mixed_scale.py:58,101,122` |

**Algorithm (the reverse):** θ → edge weights → **elimination trace** (the `_off`
two-tier tape) → Q, reversed. Stays entirely in the native graph/trace framework —
no sub-generator matrix `T`, no matrix inverse, no matrix exponential. The moment
recurrence `a_{j+1}=ewt(a_j)`, `m_k=(k+1)!·a_{k+1}[0]` replays the *same* numeric
tape with a new seed, so the reverse chains a seed-adjoint across replays; a shared
`dm[]` feeds one stage-2 param-tape reverse per output moment; the edge→θ step
contracts `dQ/dw_e` against `edge->coefficients` (linear Jacobian).

**Two subtleties baked in** (a reviewer should confirm these in the C):
1. `add_command` stores `multiplier-1` for **diagonal** (`from==to`) numeric
   commands (the identity term) — applied to primal/snapshot/stage-1 transpose,
   NOT to the glue (the `-1` is constant, `d/dw = 1`).
2. The `mult==0` primal-skip must **not** skip the gradient: at a diagonal weight
   exactly 1 the stored multiplier is 0 but its derivative isn't; `dm_c` is emitted
   regardless while the transpose is a no-op.

---

## 4. Scaffolding (de-risk validators) — COMPILE-GUARDED behind `PHASIC_B3_VALIDATORS`

These proved the math on the real tape and remain runnable oracles. They are now
**fully compiled out of production builds** via `#ifdef PHASIC_B3_VALIDATORS`
(CMake option **OFF by default**, `CMakeLists.txt`). In a normal build they add
**zero** code — no functions, no `_dbg_off_clean` struct field, no `precompute`
hooks, no pybind methods. Verified both build modes (see §5).

| Symbol (guarded) | Role | File |
|---|---|---|
| `ptd_debug_fwdmode_grad` / `ptd_dbg_run_tape` | Batch-0 forward-mode validator | `phasic.c` |
| `ptd_debug_reverse_grad` / `ptd_dbg_reverse_tape` | Batch-1 reverse validator | `phasic.c` |
| `ptd_moment0_grad_theta` | Batch-2 first-moment (superseded by `ptd_moments_grad_theta`) | `phasic.c` |
| `ptd_dbg_acquire_clean_off` + `_dbg_off_clean` field + SELFCHECK/STASH hooks in `ptd_precompute_reward_compute_graph` + create-init + `graph_destroy`-free | env-stash clean-`_off` acquisition (validators only) | `phasic.c`, `phasic.h` |
| `_debug_fwdmode_grad` / `_debug_reverse_grad` / `_moment0_grad_theta` bindings + C++ methods | pybind | `phasic_pybind.cpp`, `phasiccpp.h` |

**Build the validators** (for CI / re-validating the adjoint against the real tape):
```bash
CMAKE_ARGS="-DPHASIC_B3_VALIDATORS=ON" pixi run install-dev
```
Then `experiments/dr_realtape_validator.py`, `dr_reverse_adjoint_gate.py`,
`dr_moment0_theta_gate.py` become runnable (they call the guarded pybind methods).
`dr_moments_jac_gate.py` and `dr_mpfr_gate_test.py` use only production symbols and
run in **either** build.

The production path (`ptd_moments_grad_theta` + the **un**guarded
`ptd_dbg_tape_needs_mpfr` MPFR gate) is fully self-contained — it calls no
scaffolding, only existing core functions. **Kept (not stripped)** because the
remaining B3 work (discrete/log) reuses this validation harness; strip in the final
B3 cleanup once coverage is complete.

---

## 5. Correctness evidence (all green on branch tip)

- **Build-free reference interpreters vs JAX autodiff:**
  `experiments/dr_twotier_full_adjoint.py` 218/218 (full 2-tier reverse);
  `experiments/dr_moment_chain_adjoint.py` 230/230 (moment-chain seed-adjoint).
- **Real-C-tape gates** (run in the worktree env, all "ALL PASS"):
  `dr_realtape_validator` (fwd-mode==native+CD+closed-form),
  `dr_reverse_adjoint_gate` (reverse==fwd-mode oracle, 2-cycle+3-cycle × regime grid),
  `dr_moment0_theta_gate` (dθ==closed-form incl θ=[1,1e-8]),
  `dr_moments_jac_gate` (moment-vector Jacobian==closed-form+θ-CD),
  `dr_mpfr_gate_test` (declines at cond~1e13, stays finite, exact when benign).
- **Pin file:** 7 passed, 1 xfailed (the daisy-chain FD, explicitly out of scope).
- **No regression:** default-path pmf_and_moments tests (`test_gate_persistent_graph_reuse`,
  `test_svgd_fixed_fd_skip`, `test_reward_validation`) — 30 passed, identical on
  master vs branch.
- **Pre-existing failures (NOT regressions):** 9 in
  `tests/pytest/inference/test_jax_integration.py` fail at the existing
  `param_length == 0` check (before any B3 code) on the branch's older fixtures;
  **master SKIPS these same tests**. Confirmed by running them on master (skipped)
  vs branch (fail-at-6913). The test suite is known partially-broken — do not gate
  on full-green.

---

## 6. Risks / non-negotiables

- **Forward parity is sacred:** the primal (moments/pmf values) is bit-identical;
  the adjoint is additive and only active when `exact_moment_grad=True`.
- **Opt-in default off:** existing callers unchanged. The mixed-scale pin flips
  because its model opts in (matches the pin author's "when the fix lands" intent).
- **Host callback (not FFI):** the exact grad runs via `jax.pure_callback` over a
  private `graph.clone()` — correct under `grad`+`vmap`, but a Python/host hop per
  backward. A native FFI gradient handler is a perf option (§7), not a correctness
  need.
- **Scaffolding surface:** ~380 lines + a `void*` field in `phasic.c` (§4). Inert
  normally; decide whether to strip.

---

## 7. Review checklist

- [ ] `ptd_moments_grad_theta`: the two subtleties (§3) present; `target=0` (start
      vertex); `#ifdef HAVE_MPFR` gate declines correctly; free-list matches allocs.
- [ ] `model_bwd`: default path (`_exact_tbm is None`) is byte-identical to old FD;
      `J^T·g_moments` orientation; NaN → `_exact_ok=False` → FD fallback.
- [ ] `graph.clone()` cost acceptable at model-build time; the callback's per-θ
      `update_weights` on the clone is not aliasing the user's graph.
- [ ] Scaffolding is compile-guarded (§4): confirm a default `pixi run install-dev`
      exposes `_moments_grad_theta` but NOT `_debug_*`/`_moment0_grad_theta`, and the
      `_dbg_off_clean` field/precompute hooks compile out.
- [ ] Default build: pin file passes (7 passed / 1 xfailed) + `dr_moments_jac_gate`
      + `dr_mpfr_gate_test` "ALL PASS"; spot-check a discrete/log model still uses FD
      (correct, unaccelerated gradient).
- [ ] Validator build (`CMAKE_ARGS="-DPHASIC_B3_VALIDATORS=ON"`): all 5 gate scripts
      + the 2 reference interpreters "ALL PASS".

---

## 8. HANDOFF — continue the remaining work in a NEW session

### 8.1 How to resume (the worktree is session-scratch; the BRANCH persists)
The isolated worktree at `…/scratchpad/experiments-wt` will not survive this
session, but **branch `fd-b3-experiments` (tip `505dc2ab`) is in `/Users/kmt/phasic`'s
git** with every commit. To resume:
```bash
cd /Users/kmt/phasic
git worktree prune                       # drop the stale scratch worktrees
git worktree list                        # confirm experiments-wt/base-wt are gone
# Option A (simplest, if master no longer needs to stay pristine):
git checkout fd-b3-experiments && pixi run install-dev
# Option B (keep master usable in parallel): make a fresh worktree for the branch
git worktree add ../phasic-b3 fd-b3-experiments
cd ../phasic-b3 && pixi install && pixi run install-dev
```
All findings docs (`b3-*.md`, this file) and gate scripts (`experiments/dr_*.py`)
are ON the branch. Re-run the gates after building to confirm the baseline.

### 8.2 State snapshot
- Done & validated: Batch 0 (differentiability) → 1 (reverse adjoint) → 2 (1st
  moment, pin flipped) → 3 (moment vector / higher moments + MPFR gate). Continuous
  / linear / monolithic. Opt-in. grad+vmap exact. No regression.
- Commits (newest first): `505dc2ab` destroy-free, `d0a5e1e0` docs,
  `b1f7fa0a` MPFR gate, `15a6e681` moment-vector wiring, `9cdaa173` moment-vector C,
  `5b9dd770` 1st-moment wiring (pin flip), `85df40c0` 1st-moment C, `bc6b4d7a`
  reverse adjoint, `3c9b5c0d`/`4cf79903`/`02c5cdb5` Batch-0, `33ff39ec` plan.
- Reference/plan docs on branch: `b3-c-theta-adjoint-plan.md` (the master plan +
  9 adversarial amendments), `b3-batch{0,1,2,3}-*findings.md`,
  `b3-batch3-mpfr-and-discrete-derisk.md` (the discrete math, below).

### 8.3 Remaining work (each its own de-risk; ordered by value)
1. **Discrete / `was_dph`** (biggest coverage gain — needed for DPH/joint-prob SVGD).
   Currently excluded (FD). Three parts (full math in
   `b3-batch3-mpfr-and-discrete-derisk.md`):
   - **Renorm edge→θ Jacobian** (sibling coupling): discrete graphs renormalize
     `p_e = w_e/S_v` (`update_weights`, `phasic.c:5772+`), so the contraction needs
     `∂p_e/∂θ_j = (c_e^j − p_e·Σ_{e'∈out(v)} c_{e'}^j)/S_v`. Needs `S_v = Σ c_{e'}·θ`
     → **the C function must take θ** (new `ptd_moments_grad_theta_dph(graph, K, theta, theta_len, J)`).
   - **Discrete moment correction** (`continuous_to_discrete_moments`,
     `graph_builder.cpp:694`): a θ-independent LINEAR map `C`, so
     `d(discrete m)/dθ = C · d(continuous m)/dθ` (port `d_factorial`/`d_binomial`/
     `d_stirling2` into `phasic.c`).
   - **Wire + de-risk** vs native `moments(K, discrete=True)` central-diff at benign
     scale on `_erlang().discretize(0.5)` (+ NegBinomial closed form). Relax the
     Python `not discrete` gate to call the `_dph` variant.
   Size ≈ Batch-2+3 combined. De-risk in Python first (mirror the two corrections),
   then C, then wire.
2. **log/formula weight modes.** log: `∂w_e/∂θ_j = w_e/θ_j`. **Caution:** memory
   flags a pre-existing bug that `moments_from_graph`/`joint_index` silently IGNORE
   `weight_mode` — investigate before trusting the log path. formula = a separate
   bytecode tape (`ptd_weight_tape`) needing its own small adjoint.
3. **joint-index.** Its forward IS the transpose walk → reverse-over-reverse; a
   distinct derivation (still straight-line-linear). Own de-risk.
4. **hierarchical SCC.** Reverse through θ-dependent phantom-weight stitching
   (`scc_compose.c`) — or keep refusing it (`PHASIC_HIERAR_ELIMINATION`) and document.

### 8.4 Cross-cutting decisions (yours)
- **Make exact the DEFAULT** (vs opt-in `exact_moment_grad`)? It's now correct +
  MPFR-safe + vmap-safe for the in-scope case, with FD fallback for the rest.
  Flipping the default would fix gradients for all callers automatically (a behavior
  change — currently deliberately opt-in per the plan).
- **Native FFI gradient handler** vs the host `pure_callback` (perf only).
- **Strip the scaffolding** (§4) before merge, or keep behind a debug guard.

### 8.5 Copy-paste prompt for the new session
> Continue the B3 exact-gradient work on branch `fd-b3-experiments` in
> `/Users/kmt/phasic`. Read `B3-MERGE-REVIEW.md` and
> `b3-batch3-mpfr-and-discrete-derisk.md` on the branch first. The continuous/linear/
> monolithic moment-vector gradient is done, validated (grad+vmap), opt-in via
> `exact_moment_grad`, MPFR-safe, pin flipped, no regression. Resume by pruning the
> stale scratch worktrees and building the branch (§8.1). Then implement the
> **discrete/`was_dph`** case next (§8.3 item 1): de-risk the renorm edge→θ Jacobian
> + the linear discrete-moment correction in Python vs native `moments(K,
> discrete=True)` central-diff on `_erlang().discretize(0.5)`, then port to a new
> `ptd_moments_grad_theta_dph(graph, K, theta, theta_len, J)` and relax the Python
> `not discrete` gate. Keep master's install untouched only if you still need master
> in parallel; otherwise build the branch directly. Never `git add -A` (rewrites
> deps + notebooks). Follow the batch → de-risk → gate → commit rhythm.
