# B3 make-or-break experiment findings (DR-A, DR-D)

Branch `fd-b3-experiments` off master `422fca53`. Run as standalone scripts
against the installed build (no `install-dev`, no production edits). Reproductions:
`scratchpad/dr_a.py` (DR-A), `git show 12a30a78` (DR-D).

## DR-D — Tier 2 (scale-aware FD step): DEAD END. Settled by the existing revert.

The relative step was already implemented (`d69919f2`) and **reverted** (`12a30a78`).
The revert message is decisive (grounded, with numbers):

- A relative step `h = max(1e-6·|θ_i|, 1e-15)` fixes the negative-rate CRASH but does
  **nothing** to keep the function DIFFERENCE above f's roundoff floor. At mixed scale,
  `dfdθ_i` stays O(1) while `h` collapses to the 1e-15 floor, so the quotient becomes
  **ULP quantisation noise**: measured `SNR = 0.3` at θ=1e-9; **69% error at θ=1e-8;
  SIGN FLIP at θ=1e-12** (vs the exact derivative of a 3-state hypoexponential).
- It "traded a loud failure (negative rate → NaN → SVGD detonates) for a silent one
  (a plausible, finite, WRONG gradient) … the step was not the right lever either way."

**Verdict:** no central-difference step — absolute, relative, or clamped — fixes the
mixed-scale gradient. Tier 2 is not a viable accuracy fix, and its only benefit
(crash → silent-wrong) was explicitly judged not an improvement. **Drop Tier 2 from
B3.** (A crash-guard that RAISES a clear error — not a silent wrong gradient — is the
only defensible FD-side change, and `_check_negative_pmf` already partially does that.)

## DR-A — is an ANALYTIC gradient feasible + cyclic-correct + mixed-scale-robust? YES, mostly.

Cyclic CTMC (SCC{A,B}): `start→A ; A↔B (rate θ0) ; B→absorb (rate θ1)`.
Closed form `E[T]=1/θ0 + 2/θ1`. Analytic gradient via the sub-generator solve
`E[T]=1ᵀ(−S)⁻¹α` (`jnp.linalg.solve`, cyclic-correct, exact VJP) vs the production FD.

| θ | analytic dθ0 relerr | FD dθ0 relerr | analytic dθ1 relerr | FD dθ1 relerr |
|---|---|---|---|---|
| [1, 1] | 0 | 3e-9 | 0 | 3e-9 |
| [1, 1e-4] | **0 (exact)** | **1.1e-1 (11% WRONG)** | 2e-13 | 1e-6 |
| [1, 1e-8] | 7 (sub-dominant, corrupted) | 2e7 (garbage) | **1e-8 (exact)** | 9e-2 (9% wrong) |

Forward (all scales): analytic == phasic native == closed form exactly → the linear
solve is **cyclic-correct** (the inverse absorbs the cycle; no self-loop issue).

**Two findings:**
1. **Analytic AD decisively beats FD.** At θ=[1,1e-4] it is EXACT on the component FD
   gets 11% wrong; at extreme scale it nails the DOMINANT gradient (1e-8) where FD is 9%
   wrong AND the daisy path hard-crashes. This is the right direction for B3.
2. **A residual CONDITIONING limit exists even for exact AD** (honest caveat): the
   SUB-dominant gradient component is corrupted at EXTREME mixed scale (θ1≈1e-8), because
   the sub-generator is near-singular (`det ~ θ_small`, condition ~1/θ1); the solve's VJP
   inherits `cond·machine_eps ~ 1e-8`, which buries a sub-dominant signal of ~5e-9. This
   is a float64 precision floor of the LINEAR ALGEBRA — **not** an FD-step artifact — and
   the trace/adjoint compute the same algebra, so they inherit it. It is a far NARROWER
   failure than FD's (kicks in ~θ<1e-6 on the sub-dominant term only, vs FD failing at
   1e-4 and crashing at 1e-8), but it means "exact AD" is not unconditionally exact at
   extreme scale; it must be characterised (possible mitigations: a scaled/log basis for
   the solve, or accept it as far beyond FD's usable range).

**DR-A.2 — Tier-1 trace blocker confirmed:** `record_elimination_trace` on the cyclic
graph raises `RuntimeError: Trace-based elimination cannot handle the cycle (parent=2 →
i=1 → parent=2): self-loop correction 1/(1−q) is [not implemented]`. The C PRC tape
applies this correction (`add_command(parent,parent,1/(1−q))`); the Python trace does
not. **Engineering gap, not a fundamental one** — DR-A.1 proves the underlying math is
differentiable and cyclic-correct.

## Consequences for B3 (update to the strategy)

- **Tier 2 (FD step): dropped** — no step works (DR-D). Keep only a clear-error crash
  guard, never a silent-wrong gradient.
- **Exact AD is the fix** (Tier 1 trace or Tier 3 C adjoint) — proven feasible + cyclic-
  correct + far better than FD (DR-A.1). The make-or-break "is exact AD possible on cyclic
  graphs" = **YES**.
- **Tier 1 vs Tier 3 is now an ENGINEERING/scalability choice**, not a feasibility one:
  - Tier 1 needs the cyclic self-loop correction added to `record_elimination_trace`
    (mirror the C tape) + a JAX reduction to the scalar; then `evaluate_trace_jax` +
    `jax.grad`. Sparse/scalable. Blocker is bounded engineering.
  - Tier 3 extends the working sojourn C adjoint to θ; already cyclic-correct via the tape;
    O(n) memory. The durable answer for large graphs.
  - `linalg.solve` (DR-A's prototype) is NOT the production route — dense O(n³), violates
    the avoid-matrix-exp constraint — it was only the feasibility oracle.
- **New required work item regardless of tier:** characterise + document the sub-dominant
  conditioning floor at extreme mixed scale (DR-A finding 2), and decide whether to
  mitigate (scaled basis) or bound it (state the regime where any float64 gradient of a
  sub-dominant term is unreliable). The mixed-scale pin `test_fd_gradient_mixed_scale.py`
  should gain an analytic-AD comparison so the fix's real accuracy envelope is measured,
  not assumed.

## Recommended next step
Pick the tier: **Tier 3 (C θ-adjoint)** is the lower-risk durable path (extends a shipped,
tested adjoint; cyclic-correct today; scalable). **Tier 1** is attractive for the pure-JAX
story but needs the cyclic-trace fix first. Either way, next de-risk is a small
**θ-adjoint / cyclic-trace prototype on this same cyclic fixture**, gated against DR-A's
analytic oracle across the regime grid.
