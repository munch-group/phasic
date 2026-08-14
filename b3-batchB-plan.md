# Batch B — `weight_mode='formula'` exact moment gradient (plan v1)

**Master plan:** §4 (signed off 2026-08-11). **Feasibility:**
`atlas/plan-feasibility-formula-mode.md` (read in full; §5 is the design
sketch this plan concretizes). **Process:** `b3-execution-process.md`
(gate ladder G0–G5; two plan refuters BEFORE implementation; de-risk on
the branch before the implementation commit). **Base:** master
`ae217b0e` (Batch A close-out; seventh ledger stamp 1963/0/84/24).
**Branch/worktree:** `b3/batchB-formula` in `../phasic-batchB` (own pixi
env, `pixi run --manifest-path` / cd + install-dev).

## 0. What changed since the feasibility doc (it predates Batches 0 and A)

1. **Batch 0 landed** (`d2cca7ab`): the shared stage-0/1/2 core
   `ptd_b3_moments_core` exists (`src/c/phasic.c:10800`), enum-dispatched
   (`enum ptd_b3_contract { PTD_B3_LINEAR, PTD_B3_LOG, PTD_B3_DPH }`,
   `:10796`). Its header comment PRE-DECLARES this batch's extension
   point: "Batch B: PTD_B3_FORMULA + internal pre-outk dw/dtheta stage"
   (`:10794`). So Batch B is a 4th switch case + one new pre-outk stage,
   NOT a 4th ~150-line copy — the feasibility doc's §4 refactor
   recommendation is already satisfied.
2. **Batch A landed** (`798ddcaa`): the core signature carries
   `(const double *rewards, size_t rewards_len)` before the kind enum,
   with validation and the two reward hooks INSIDE the core (per-stage
   seed re-scale + adjoint-side VJP). These hooks are
   contraction-independent, so a formula kind inherits reward support
   through the core FOR FREE — but the wrapper must thread the pair and
   the gate must verify the combination (never assume inheritance;
   [[feedback_never_assume_verify_adversarially]]).

## 1. Scope

- **In:** continuous graphs with `weight_mode='formula'`;
  `pmf_and_moments_from_graph(..., exact_moment_grad=True)` (and the
  default-True path) computes the exact reverse-mode moment Jacobian
  instead of FD. 1-D rewards supported (core hooks + wrapper threading
  + gate cell). MPFR gate inherited unchanged. Per-theta declines →
  existing NaN→FD fallback with the existing INFO log.
- **Out (static declines, INFO-logged):** discrete/was_dph × formula
  (evidence via de-risk D-B2, not analogy — see §4); 2-D rewards on the
  1-D leaf (unchanged from A); `weight_mode='callback'` (Batch C).
- **Not touched:** R29/svgd_config rules (no new leaf; formula on the
  moments leaves simply becomes genuinely honored instead of
  inert-with-INFO — text updates only, §6), `moments_from_graph`,
  `method_of_moments`, joint-index/daisy paths.

## 2. New C surface (all additive)

1. **`ptd_wf_grad_theta(...)` — static formula-tape autodiff** (new,
   ~120 lines): given `(const struct ptd_weight_tape *tape, const double
   *theta, size_t n_theta, const double *coeffs, size_t n_coeff, double
   *dw_out /* P */)`, run the stack machine forward ONCE recording every
   pushed value (snapshot array sized `tape->n_ops`-ish, mirroring the
   `s0/s1` snapshot-then-replay idiom at `phasic.c:10772/10840`), then a
   reverse pass over a parallel adjoint stack accumulating
   `d(result)/d(theta_i)` into `dw_out`. Rules (feasibility §5, master
   §4): ADD/SUB ±1 pass-through; MUL product rule; DIV quotient rule;
   **POW general two-sided rule `d(a^b)=a^b·(b/a·da + ln(a)·db)`**
   (theta-in-exponent is syntactically legal — dedicated de-risk D-B1);
   NEG; EXP/LOG/SQRT/LOGISTIC 1-arg chain rules; comparisons/booleans/
   NOT: zero gradient AND no adjoint propagation into operand subtrees
   (guaranteed safe by the compiler's static theta-independence guard —
   `weight_formula.py` rejects theta under comparison/boolean/
   select-condition at assignment; the C tape mirrors the compiled
   Python tape integer-for-integer, `phasic.c:5085-5093`); SELECT:
   condition theta-independent, chosen branch's adjoint passes through,
   unchosen dropped. Non-finite intermediates (log of ≤0, DIV by 0,
   invalid POW) are NOT new decline logic — they flow into `dw_out` and
   fall through to the core's existing final `isfinite` sweep over
   `J_out`; the implementation MUST NOT early-return before the sweep
   (feasibility §5's explicit caution; de-risk D-B5 confirms).
2. **`PTD_B3_FORMULA` enum value + core changes:** (a) a pre-outk stage
   (placed at the Batch-0 pre-declared marker): iterate
   `off->input_specs[k]` once, resolve `(sp.v, sp.e)` →
   `struct ptd_edge*`, call `ptd_wf_grad_theta` per eligible tape-input
   edge, store an `ni × P` heap array `dw[k*P+j]` (freed by the core;
   NULL-checked; skipped entirely for the other three kinds); (b) a
   contraction case `J_out[outk*P+j] += binp[k] * dw[k*P+j]` with the
   SAME hygiene guards as log (skip `coefficients_length==0` — BUT see
   the D-B4 grounding question below — and skip starting-vertex edges,
   with log's unreachability caveat carried over verbatim).
3. **`ptd_moments_grad_theta_formula(graph, nr_moments, const double
   *theta, size_t theta_len, const double *rewards, size_t rewards_len,
   double *J_out)`** — public thin wrapper following `_log`'s shape
   (owns ptape/off; validates theta_len == param_length; declines -1
   when `graph->weight_tape == NULL`, `graph->was_dph`, or
   `graph->is_discrete` — continuous-only scope) + `api/c/phasic.h`
   decl + C++ `moments_grad_theta_formula(int, std::vector<double>
   theta, std::vector<double> rewards = {})` in `api/cpp/phasiccpp.h` +
   pybind `_moments_grad_theta_formula` with `py::arg("rewards") =
   std::vector<double>()` (the A pattern exactly).

**CRLF discipline:** `src/c/phasic.c` and `api/cpp/phasiccpp.h` are
CRLF → binary-mode byte replaces with `\r\n` and assert-count==1 per
replacement; `api/c/phasic.h` and `phasic_pybind.cpp` are LF.

## 3. Python changes (`src/phasic/__init__.py`)

- `pmf_and_moments_from_graph`: the static formula decline arm becomes a
  live route: `_exact_is_formula` (mirroring `_exact_is_log`), callback
  calls `_exact_graph._moments_grad_theta_formula(_exact_K, t.tolist(),
  rewards=_rw_list)`. The private clone's weight state: mirror the log
  lesson (Batch A G4 found the log wrapper reads the graph's CURRENT
  weights — the callback must `update_weights` the clone per theta, or
  the wrapper must be verified self-sufficient; GROUNDING ITEM for
  implementation, decided by reading `ptd_graph_update_weights`'s
  formula path `phasic.c:5636-5694` and the existing log callback's
  handling). Static declines: discrete×formula keeps FD with a truthful
  INFO message (new text — currently formula declines with the generic
  out-of-scope message); 2-D rewards arm unchanged.
- `_exact_moments_jac_np` gains the formula branch (dispatch order:
  discrete-decline → 2-D-decline → formula/log/dph/linear).

## 4. De-risk experiments (branch, BEFORE the implementation commit; each
   its own `experiments/dr_batchB_*.py` with recorded verdicts)

- **D-B1 POW rule** (`dr_batchB_pow_derisk.py`): the standalone rule
  (pure Python mirror of the planned C reverse pass over compiled
  tapes from `weight_formula.compile_formula`) vs `jax.jacobian` on:
  `t0**2`, `2**t0`, `t0**t1`, `(c0*t0)**0.5`, `exp(t0)**t1`, POW at
  base<0 with integer exponent, base=0 edges. GO = machine-precision
  agreement everywhere finite + NaN/inf agreement on the invalid domain.
- **D-B2 was_dph/discrete × formula repro** (`dr_batchB_wasdph_repro.py`):
  does `discretize()` + formula fail loudly elsewhere, or silently
  compute (the log-batch D1 lesson: the exclusion must be shown
  load-bearing or shown unnecessary — never assumed by analogy)? Also
  native-DPH (`is_discrete=True`) + formula. Outcome fixes the wrapper's
  decline set and its comment's truthfulness.
- **D-B3 mixed-scale FD defect repro for formula** (motivating gate,
  feasibility §7.4): FD vs closed-form/independent oracle at
  θ=[1, 1e-8]-class on a formula-mode fixture — quantifies the defect
  this batch fixes (no such gate exists; `test_weight_formula_svgd.py`
  is FD-only).
- **D-B4 GROUNDING: which edges does formula-mode `update_weights`
  actually recompute?** Read (and probe) `phasic.c:5636-5694`: if ONLY
  `coefficients_length>0` edges are tape-evaluated, the linear-style
  skip is correct for formula; if ALL parameterized-marked edges (or
  any other set) are, the skip set must match EXACTLY that set — a
  formula `t0*2.0` references no `c<j>`, so "no coefficients ⇒
  theta-independent" is NOT a valid inference for formula mode in
  general. The contraction's skip set and the primal's recompute set
  MUST be proven identical, not assumed (this is the batch's sharpest
  correctness edge; found at plan-drafting time).
- **D-B5 non-finite fall-through** (`dr_batchB_nonfinite_probe.py` or a
  gate cell): a formula whose gradient (not value) is non-finite at
  some theta (e.g. `sqrt(t0)` at t0→0, `log(t0*c0)` with a zero-c0
  edge... chosen after D-B4) → the wrapper returns -1 via the isfinite
  sweep (Python falls back to FD with the per-theta log), never
  garbage.

**Re-evaluate the plan after de-risks** ([[feedback_derisk_and_reevaluate]]):
v2 amendment records each verdict and any design change (especially
D-B4's skip-set resolution and D-B2's decline set) BEFORE implementation.

## 5. Gates

- **G1 micro-gates** (`dr_batchB_i1_gate.py`, dump/check):
  (a) rewardless byte-identity for linear/log/dph vs a pre-B golden
  (the A pattern — dump on the pre-B main install, check on the B
  build; the core is EDITED by the new switch case, so cross-install
  byte-identity on the untouched kinds is the regression gate) + the
  three standing jac-gates (`dr_moments_jac_gate.py`,
  `dr_dph_moments_jac_gate.py`, `dr_log_mode_moments_jac_gate.py`) ALL
  PASS on the B build;
  (b) formula vs FD-of-the-PRIMAL (`update_weights(theta)` +
  `moments(K, rewards?)`) at K=1..3 on ≥2 fixtures with DISTINCT
  per-edge weights (the log-batch lesson: numerically-indistinguishable
  edges mask index-mapping bugs) covering MUL/DIV/EXP/LOG/POW/SELECT
  ops, incl. a mixed-scale θ case (D-B3's fixture) and a
  theta-in-exponent case;
  (c) rewards × formula: mixed rewards vs primal-FD + all-ones ==
  rewardless BITWISE;
  (d) decline contract: was_dph/native-DPH (per D-B2), wrong theta_len,
  wrong rewards_len, no-weight-tape graph;
  (e) MPFR-decline repro (conditioning-triggering fixture → -1 → FD).
- **G2 targeted map** (process doc): the svgd config/validation row +
  moments-rewards row + `test_weight_formula_*.py` +
  `test_gate_weight_formula_conformance.py` +
  `test_exact_grad_log_weight_mode.py` + `test_fd_gradient_mixed_scale.py`
  + `test_multivariate_correctness.py` + the new file.
- **G3** chunked full suite vs ledger 1963/0/84/24 (7th stamp), `-rf`,
  groups enumerated from split output, union + per-group-output checks.
- **G4** two adversarial diff refuters (mandate: independent numeric
  probes; attack the POW rule, the D-B4 skip-set proof, the snapshot
  indexing of the reverse pass, adjoint-stack correctness for SELECT,
  rewards threading, CRLF bytes, callers of the core).
- **G5** merge review + squash-merge from the main checkout + close-out
  (8th ledger stamp; tracker; master §4 banner + §15 tick; CLAUDE.md;
  process-map row for the new test file; memory).

## 6. Tests (new file `tests/pytest/inference/test_exact_grad_formula_mode.py`,
   mirroring `test_exact_grad_log_weight_mode.py`'s structure)

Cells (each value-asserted against the manual central-diff of the
model's OWN forward, tolerances = measured actuals at implementation):
1. formula (MUL/ADD only) grad vs central-diff, K=2.
2. formula with POW(theta-in-exponent) vs central-diff.
3. formula with SELECT (runtime condition on c<j>) vs central-diff —
   both branches exercised across edges.
4. mixed-scale θ=[1,1e-8]-class: exact vs closed-form/central-diff
   (the motivating defect, from D-B3).
5. rewards×formula: 1-D rewards engage (spy floor ≥1 on
   `_moments_grad_theta_formula` + no "finite differences" logs,
   measured-zero fixture) + all-ones == rewardless bitwise.
6. discrete×formula declines with the truthful INFO log (per D-B2).
7. engagement flip: formula + exact_moment_grad=True previously logged
   the out-of-scope decline — that log must be GONE (live-filter
   discipline from A's G4: grep the CURRENT message text, and pair with
   the positive spy assertion — never a deleted string).
8. vmap/jit composition: vmap(grad) == per-particle bitwise,
   jit(vmap(grad)) == vmap(grad).
9. svgd front-door smoke: formula-mode graph, `svgd(...)` default kwarg
   — completes finite, exact engages (spy), no static-decline logs
   (the E lesson: real particle-init distribution).
Fate table: NO existing tests modified except log-absence text updates
if any grep the formula out-of-scope message (enumerate at
implementation; expected: none — verify by grep).

## 7. Shipped-text list (updates that MUST land in the same commit)

- `pmf_and_moments_from_graph` docstring: coverage sentence gains
  formula; the FD-cause list drops "formula" from the out-of-scope set
  (callback remains).
- `Graph.svgd` docstring `exact_moment_grad` section: "formula/callback
  weight modes decline STATICALLY" → callback-only (+ formula now
  covered).
- `svgd_config.py` R29 leading comment: the "formula/callback
  precedent" sentence (added at A's G4) must be reworded — post-B the
  precedent's example is callback (+ discrete+rewards), formula is no
  longer inert.
- CLAUDE.md: B3-moments section formula mention + the leaf-5
  formula/callback exception text (same reword); tracker + master plan
  per G5.

## 8. Ledger arithmetic

New pytest file ≈ 9 cells (+ any parametrize) ⇒ expected G3 =
1963 + N_new, recorded exactly at G1 time. No fate-table breaks
expected (verify by grep at implementation).

## 9. Risks / open questions carried into review

1. D-B4 (skip set vs recompute set) — the sharpest correctness edge;
   unresolved until grounded.
2. POW domain edges (base ≤ 0): rule produces NaN via ln(a) — matches
   JAX? D-B1 decides; the isfinite sweep is the backstop either way.
3. Snapshot sizing for the reverse pass: the stack machine's
   intermediate count ≤ n_ops, but the adjoint pass needs per-op operand
   VALUES (not just final stack) — design records a full push-trace
   (value per op index), not a stack snapshot; refuters check indexing.
4. The clone's per-theta weight state for the formula wrapper (the log
   G4 lesson) — grounded at implementation (§3).
5. `wf_residuals[]` specialized tapes deliberately NOT used
   (correctness-first, different iteration order; future optimization —
   feasibility §5).
6. Formula-tape evaluation cost: the pre-outk stage runs once per call
   (ni × tape-length × P-ish) — no benchmark gate planned (the moments
   adjoint replaces a 2P-call FD loop; net win expected as with
   log/linear); flag only if G4 refuters find a pathological case.
