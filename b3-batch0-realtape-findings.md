# Batch-0 real-C-tape confirmation (isolated worktree env) — COMPLETE

A non-shippable C validator `ptd_debug_fwdmode_grad` (phasic.c) + pybind
`Graph._debug_fwdmode_grad`, built in the ISOLATED worktree env (master's
install untouched). It replays the REAL `_off` elimination tape (the exact tape
native replays) with a parallel forward-mode tangent and a self-contained
central difference, on the CYCLIC graph — the topology whose self-loop made the
differentiable *trace* path refuse, which is why FD was originally chosen.

## Result: BOTH gates pass — the real tape is a complete, correct differentiable trace

`experiments/dr_realtape_validator.py` (run in the worktree env), across scales:

| theta        | forward E[T] == native | forward-mode grad vs oracle |
|--------------|------------------------|-----------------------------|
| [1, 1]       | 3 == 3  ✓              | == central-diff (1.4e-10)   |
| [1, 0.5]     | 5 == 5  ✓              | == central-diff (9.0e-11)   |
| [2, 1]       | 2.5 == 2.5 ✓           | == central-diff (9.0e-11)   |
| [1, 1e-3]    | 2001 == 2001 ✓         | == central-diff (3.8e-8)    |
| [1, 1e-6]    | 2000001 == 2000001 ✓   | == CLOSED-FORM (exact); CD off 2.0e-5 |

- **Forward faithful:** the `_off` tape forward equals native `expected_waiting_time`
  EXACTLY at every scale (n_inputs=4, all operands resolved MEM/INPUT).
- **Differentiates correctly:** the forward-mode `dE[T]/d(edge weight)` matches the
  central difference at benign scales, and the exact closed-form gradient at
  extreme mixed scale.
- **B3 thesis demonstrated in-situ:** at spread=1e6 the central difference (the FD
  baseline) is already off by 2e-5 on the dominant gradient while the analytic
  forward-mode is exact — this is exactly the defect B3 replaces.

Route (i) (θ-adjoint over the real elimination tape) is confirmed sound on the
REAL C executor, not just the Python reference interpreter.

## TWO semantic subtleties the C θ-adjoint (Batch-1) MUST honor

Both were found the hard way while making forward==native, and both are real
traps for the reverse adjoint:

1. **Diagonal identity subtraction.** `add_command()` (phasic.c:6904-6914) stores,
   for a DIAGONAL command (`from==to`), `multiplier - 1` (the `I - ...` term of
   Gaussian elimination); off-diagonal stores the multiplier as-is. Any replay /
   adjoint that reconstructs the numeric command stream from the parameterized
   tape must apply this `-1` on diagonals. (The `-1` is a constant, so it does
   not change the multiplier's tangent/adjoint — but it DOES change the primal,
   hence the guard below.)

2. **`mult==0` primal-skip must NOT skip the gradient.** Native's forward replay
   skips commands with `multiplier==0` (adds 0; also dodges 0*inf=nan). But a
   DIAGONAL whose weight is exactly 1 has stored multiplier `1-1==0` while its
   derivative `d(weight)/dθ` is nonzero. Skipping the whole command there drops
   the gradient term `res[to] * mult_dot`, zeroing a real component of the
   gradient (observed: dE/d(w_Babs) read 0 instead of -2 at theta=[1,1]).
   Correct rule: skip the PRIMAL on `mult==0`, but ALWAYS apply the
   tangent/adjoint (guarding only the genuine inf*0 case). Notably this trap
   bites at specific theta where a diagonal crosses exactly 1 — a mixed-scale-
   adjacent correctness hazard the reverse pass must replicate.

## Mechanics that made the validator faithful (for the Batch-1 harness)
- Clean pre-execution mem: the `_off` executor writes `*fromT` into `mem_base`
  IN PLACE, so any conversion/replay after an executor run reads post-exec mem
  (the pitfall at phasic.c:2060). The validator forces a fresh param-tape
  rebuild (`dph_compute_invalidated=true`), converts at the pre-executor
  SELFCHECK point (:2069), and stashes the CLEAN `_off` (skipping the `_oo`
  self-comparison, which would itself dirty `mem_base`). Env-gated
  (`PHASIC_DBG_STASH_OFF`), worktree-only.
- Operands resolve to EITHER `mem_base` OR `inputs[]` (edge weights), and the
  executor mutates whichever `*fromT` points to — so the replay keeps LOCAL
  mutable copies of both mem and the input values, never the graph's real edges.

## Status
- Isolated worktree build env: working (master install untouched).
- Reference interpreter (dr_twotier): verified vs JAX autodiff (218/218).
- Real-tape forward + differentiability: CONFIRMED (both gates, all scales).
- Next: Batch-1 — port the verified reverse θ-adjoint (dr_twotier_full_adjoint)
  to C, honoring the two subtleties above, gated on cyclic fixtures against this
  validator (forward-mode oracle) and the FD baseline.
