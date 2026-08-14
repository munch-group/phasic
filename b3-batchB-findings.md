# Batch B findings

**Plan:** `b3-batchB-plan.md` v1 + v2 (binding amendment, two-refuter
review folded). **Branch:** `b3/batchB-formula` (base `ae217b0e`).

## G0

Branch base `ae217b0e` (Batch A close-out). Delta above the seventh
ledger stamp (`798ddcaa`): docs-only (`ae217b0e` close-out; plan commits
`ad08dd27`/`ac971d74` land on master after the branch was cut and do not
touch code). Worktree `../phasic-batchB`, own pixi env, install verified
(imports, Batch-A rewards kwarg present).

## Plan-review de-risks (resolved by refuter probes, recorded in plan v2)

- **D-B1 POW**: two-term adjoint form matches `jax.jacobian` at every
  probed domain edge (a=0 × b∈{0.5,1,2,3}; a<0 integer b; theta-in-
  exponent; theta-indep non-leaf exponent). v1's factored form REFUTED
  at a=0. → gate/pytest cells.
- **D-B2 was_dph arm**: was_dph×formula silently computes renormalized
  weights → the C `was_dph` decline is LOAD-BEARING. Native-DPH arm:
  `is_discrete` is a free Python attribute, invisible to C (confirmed
  below, D-B6.4) → Python `_effective_discrete` gate only.
- **D-B4 skip set**: under formula, `update_weights` recomputes exactly
  {non-start edges with coefficients_length>0} (`phasic.c:5653/:5662`
  precede the tape branch) — the planned contraction skip set is
  IDENTICAL to the primal's frozen set. The cl==0 skip is
  CORRECTNESS-load-bearing (tape evaluates fine with coeff=NULL when
  the formula uses no c<j> → unskipped constant edge = wrong-but-finite
  J, sweep-invisible). Aux-constant-edge fixture mandated in G1(b).

## Pre-implementation de-risks D-B5 + D-B6 (`experiments/dr_batchB_d5_d6_derisk.py`, pre-B build, ALL PASS)

- **D-B5** (`sqrt(t0 - c0)` fixture): the `+ 0.5` variant is the
  constructible gate fixture — weight 0.5 at the sqrt boundary
  (t0==c0==1), primal moments finite [2.0, 8.0], while sqrt's inner
  gradient is inf there. Negative domain (t0<c0): `update_weights`
  REJECTS (NaN weight → RuntimeError) — so at exactly t0=1.0 an
  FD *fallback* would raise on its minus probe. **Test-design
  consequence:** assert the non-finite-gradient decline at the WRAPPER
  level (direct `_moments_grad_theta_formula` → size 0) at t0=1.0;
  model-level cells stay at t0 comfortably inside the domain. The
  pure-sqrt (weight exactly 0) variant raises in the primal
  ("computation produced NaN … numerical catastrophe") — never use it.
- **D-B6.1 lazy-decoupled class** (coeffs [2.0,0.5] lock C
  param_length=2; formula `t0*c0 + c1` → tape n_theta=1):
  `serialize()['param_length']` stays **2** (the C value — the v2 §A
  predicate is therefore **model-resolved theta_dim (n_theta / explicit
  theta_dim kwarg) != `_exact_graph.param_length()`**, NOT the
  serialized field). Forward with theta_dim=1 works; FD gradient works
  (grad=[-1.72036740055...]); svgd front door works;
  `clone().update_weights([0.7])` RAISES RuntimeError — the exact
  hazard the static decline prevents. Post-B contract: this class →
  static INFO decline → FD (behavior unchanged from today, plus the
  log).
- **D-B6.2 canonical `set_param_length(2)` + 3-coeff edges**:
  param_length==n_theta==2, update_weights([2]) works → ALIGNED, exact
  engages post-B.
- **D-B6.3 plain aligned**: param_length==n_theta==2 → exact engages.
- **D-B6.4 native-DPH×formula**: constructible via the free Python
  attribute; C cannot see it (no struct field) — the Python gate is
  the only defense, as v2 §B specifies.

## I1+I2 (C + bindings, commit `3b925dbe`)

`ptd_wf_grad_theta` (static, Wengert list per plan v2 §D; POW two-term
adjoint §C; zero-propagate §D) + `PTD_B3_FORMULA` + core pre-outk stage
+ contraction case (skip set §E) + public wrapper (declines §B) +
header/C++/pybind (the A pattern). CRLF/LF verified all four files.

**Micro-gates (`dr_batchB_i1_gate.py`) ALL PASS:**
- (a) byte-identity vs FRESH pre-B goldens: lin_plain / lin_rw /
  log_plain / log_rw / dph_plain all `identical=True` (the rewards-
  bearing linear/log paths bitwise-gated for the first time, v2 §G).
- (b1) formula vs LINEAR exact on the linear-equivalent formula:
  rel **0.00e+00** (bitwise) at benign AND MIXED (θ=[1,1e-8]) scale —
  the FD-independent motivating cell.
- (b2) pow-mix K=1..3 (label corrected at G4 fold: `(t0*c0)**c1` --
  a constant-exponent POW, NOT exp()): 6.7e-11 / 8.7e-11 / 1.6e-10. (b3) t0**t1:
  1.4e-10. (b4) select: 5.7e-11. (b5) aux-constant-edge skip-set
  discriminator: 1.1e-10.
- (c) rewards×formula 1.8e-10; all-ones == rewardless BITWISE.
- (d) declines 0/0/0/0 (was_dph / no-tape / theta-len / rewards-len).
- (e) POW at t0=0: J exactly 0 (b=2), finite nonzero (b=1), exactly 0
  (b=3); sqrt-boundary wrapper decline size 0 (the D-B5 contract).
- Standing chain on the B build: 3 jac-gates ALL PASS +
  `dr_batchA_i1_gate.py check` vs a FRESH pre-B golden ALL PASS.
- Gate-authoring lesson (mirrors D-B6): the (b2)/(b3) fixtures
  initially hit the lazy-decoupled raise themselves — fixed with the
  canonical `set_param_length` pattern + coefficient padding
  (`cl >= param_length` is enforced by add_edge).

## I3+I4 (Python + tests, commit `5322220d`)

Dispatch: `_formula_scope_ok` = formula ∧ continuous ∧ ALIGNED
(`param_length == graph.param_length()`); two new static INFO declines
(discrete×formula "continuous-only"; lazy-decoupled naming
param_length vs theta dimension); generic message now names
'linear'/'log'/'formula' and callback-only-out-of-scope; formula
callback arm (plain `update_weights(t)`, log=False — probe-verified
correct for formula); try/except → NaN→FD around the whole callback
body (v2 §A defense-in-depth); per-theta decline log gains the
formula-tape non-finite cause. Shipped text: pmf docstring, svgd
docstring, R29 comment, kwarg-file docstring (all per plan §7 + v2 §H).

**`test_exact_grad_formula_mode.py` 12/12** (~58s): twin parity
rtol 1e-12 + central-diff 1e-6 + spy; mixed-scale twin parity 1e-12;
t0**t1 / pow-mix / select vs central-diff 1e-5; rewards engage
(spy rw_len=5, size 4) + all-ones bitwise; discrete decline
(construction-time capture — first-run lesson: the static decline logs
at model BUILD, capture must wrap it); lazy-decoupled decline +
finite FD grad; out-of-scope log GONE + spy; vmap/jit bitwise; svgd
front door ≥6 spy calls, no static-decline logs; multivariate
per-feature vs central-diff 1e-5.

## G2 (expanded map per v2 §G, verbatim)

`test_svgd_config.py test_gate_moments_3way.py
inference/test_jax_integration.py inference/test_exact_grad_discrete.py
inference/test_exact_grad_rewards.py
inference/test_exact_grad_log_weight_mode.py
inference/test_fd_gradient_mixed_scale.py
inference/test_multivariate_correctness.py test_weight_formula_svgd.py
test_weight_formula_residual.py test_weight_formula_theta_dim.py
test_gate_weight_formula_conformance.py
inference/test_svgd_exact_moment_grad_kwarg.py
inference/test_svgd_exact_moment_grad_rewards.py`
→ **182 passed, 30 skipped, 3 xfailed, 0 failed** (72.76s). Note:
test_jax_integration contributed no failures on this run (its 9
ledgered sources-on failures did not manifest; compared against the
ledger, not assumed green — zero NEW failures is the gate).

## G3 (full suite, 32 chunks, worktree, verbatim)

Union check OK (157 collected files == 32-group union; an output file
per group). Summed: **1975 passed / 0 failed / 84 skipped / 24
xfailed / 0 xpassed / 0 errors** = the seventh ledger stamp's 1963 +
Batch B's 12 new tests exactly. Two sleep-kill interruptions resumed
from preserved per-group outputs (no green group re-run); one
false-alarm "hang" in group aa was the machine sleeping mid-window
(re-ran clean: 48 passed in 149s; no stale lock, verified).

## G4 adversarial diff review (two refuters, 2026-08-14)

**Wiring/math refuter: SOUND** (zero CRITICAL/MAJOR). Its independent
from-first-principles oracle (own jnp tape interpreter from OPCODES +
a `M_k = k!·α(UΔr)^k·1` phase-type oracle differentiated with
jax.jacobian, oracle itself validated vs the primal to 6.4e-16):
**24/24 case×theta combinations at J_rel ≤ 3.3e-15**, covering
fan-out/shared subexpressions, same-theta-4×, nested select, logistic/
exp/log chains, NEG chains, theta-in-base+exponent POW, DIV-heavy,
rewards-with-zero, mixed scale, depth-254 nesting. Memory: 8000-call
loops on five paths, RSS delta 0.00 MB. n_theta<P: dw and J columns
exactly 0. n_theta>P: structural decline, never a crash. Contraction
switch: only removed line in the whole C diff is the enum; LINEAR/LOG/
DPH source-byte-identical; cross-install byte-identity reproduced with
the refuter's own fixtures. Skip-set bonus probe: constant-edge graph
vs oracle 5.2e-16. Minors: direction-misleading decline text (FOLDED:
direction-neutral message); dead-select-arm permanent-FD conservatism
(documented below); deep-formula parser recursion limit (pre-existing).

**Tests/process refuter: SOUND-WITH-CORRECTIONS** (no shipped defect;
it probed the uncovered rules itself: DIV 2.4e-11, LOG 2.4e-10, EXP
2.8e-10, LOGISTIC 1.9e-10, a<0-integer-POW 3.1e-10, (t0-3)**2 at 3
exactly 0, non-leaf exponent 7.5e-11 — all pass). Vacuousness attacks:
under simulated regressions cells 1/6 and all four engagement cells
FAIL as they must; all log-filter strings are live. Golden provenance
independently reproduced bitwise. G3 arithmetic recomputed from the 32
preserved outputs: exact.

**Folded (this commit):**
- pytest cells: DIV/EXP/LOG (B-G4-2), aux-constant edge (B-G4-1, the
  v2 §E pytest half), POW domain edges — (t0-3)**2 at 3 exact zero,
  a<0 integer b, non-leaf exponent (B-G4-3); gate (e2) POW-opcode
  b=0.5 boundary decline.
- Multivariate HARD pin [-11.508732477887918, -5.563485415672725]
  (B-G4-4).
- Front-door svgd spy counts full-size SUCCESSES (≥ n_particles), not
  calls (B-G4-8).
- Direction-neutral theta-dim decline message (wiring MINOR 1); stale
  "Scope: weight_mode='linear', monolithic" comment updated (B-G4-9).
- Findings label fix (b2 "pow-mix"); this register.

**mcmc note (B-G4-5, v2 §H):** `Graph.mcmc` builds
`pmf_and_moments_from_graph` models at the default exact_moment_grad —
post-B its formula-mode models construct the exact machinery. Inert:
mcmc.py contains zero jax.grad/jacfwd/jacrev (Metropolis is
gradient-free; refuter-verified by grep).

**Documented conservatism (wiring MINOR 2):** a formula whose UNCHOSEN
select arm has a non-finite intermediate declines the exact path at
every theta (the full tape is differentiated; the primal prunes dead
arms via wf_residuals) → permanent FD for such formulas, matching
jnp.where's NaN convention (plan v2 §D). A residual-tape-based gradient
is the ledgered future optimization (feasibility §5).

## Plan-deviation register (B-G4-7)

1. Commit granularity: I1+I2 (`3b925dbe`) and I3+I4 (`5322220d`) vs
   v2 §I "committing per item" — deliberate coupling (C+bindings are
   unbuildable apart; dispatch+tests reviewed together), same class as
   A's ledgered MINOR.
2. v1 G1(e) MPFR-decline repro SUBSTITUTED by the POW/sqrt boundary
   cells (B-G4-6): no formula-mode MPFR-triggering fixture was
   constructible (the G4 wiring refuter also failed to construct one);
   the MPFR gate is stage-0 in the kind-independent shared core,
   already gated for linear via `dr_mpfr_gate_test.py`. Recorded, not
   silently dropped.
3. Gate fixtures initially hit the lazy-decoupled raise themselves
   (fixed with set_param_length + padding); the discrete-decline test
   initially captured logs AFTER construction (fixed) — both recorded
   at I-phase.
4. 15 pytest cells vs plan §8's "≈9" (12 at I4 + 3 at G4 fold — the
   POW domain edges are one function with three sub-cases; the
   multivariate pin and spy-floor were edits to existing cells; the
   plan said "recorded exactly at G1 time", done at each stage).
   Expected post-fold full-suite arithmetic: 7th stamp 1963 + 15 =
   **1978**.

## 8th ledger stamp (main checkout, post-merge, 2026-08-14)

Measured 32-group run (union == 157 files): raw **1977 / 1 / 84 / 24**;
the single failure (`test_exact_grad_joint_index.py::
test_default_path_uses_fd`, a Batch-F surface B never touched; the
assertion caught exact_grad=True producing the FD gradient — the safe
whole-model-FD fallback, bitwise-identical arrays, not a wrong number)
was UNREPRODUCED across three re-runs (alone + full group ×2, 61
passed each) → closed as a stochastic transient per the G.1 precedent.
Effective stamp: **1978 / 0 / 84 / 24** = 1963 + 15 exact.
