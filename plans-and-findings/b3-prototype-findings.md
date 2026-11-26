# B3 prototype findings — θ-adjoint / cyclic-trace, gated against the DR-A oracle

Branch `fd-b3-experiments`. Standalone prototype (`experiments/dr_proto_theta_adjoint.py`),
no `install-dev`, no production edits. Prototypes the two exact-gradient routes on
CYCLIC graphs and gates them against the DR-A analytic oracle AND the true
closed-form gradient AND the production FD.

## What was prototyped

phasic's existing trace is JUMP-CHAIN (edge probabilities) and its own comment
(`trace_elimination.py:831-853`) says the naive `1/(1−q)` self-loop fix is
"insufficient on its own" — a full fix needs "a true Gaussian elimination that
drops vertex i". So the prototype implements exactly that: expected absorption
time `E[T]=1ᵀ(−S)⁻¹α` via **rate-matrix Gaussian elimination recorded as a linear
op trace**, then two exact gradient routes, both cyclic-correct by construction:

- **Tier-1 (cyclic-trace):** the elimination op-sequence in plain JAX → `jax.grad`
  (JAX reverse-mode over the trace).
- **Tier-3 (θ-adjoint):** the SAME elimination recorded on a scalar tape → a
  hand-written reverse-mode adjoint (init `adj[out]=1`, walk the tape in reverse
  with the local +,−,×,÷ transposes, chain leaf adjoints through the edge
  coefficients). This is the C-adjoint template.

Fixtures: a 2-state 2-cycle (the DR-A graph) and a 3-state 3-cycle.

## Results (2-state cycle; relerr vs the TRUE closed-form gradient)

| θ | Tier-1 trace | Tier-3 adjoint | oracle (solve) | FD (production) |
|---|---|---|---|---|
| [1, 1] | 0 | 0 | 0 | 2.8e-9 |
| [1, 1e-4] | **0 (exact)** | **0 (exact)** | 0 | **1.15e-1 (11.5% WRONG)** |
| [1, 1e-8] dominant dθ1 | 1.2e-8 | 1.2e-8 | 1.2e-8 | 9.1e-2 |
| [1, 1e-8] sub-dom dθ0 | 3.0 (cond. floor) | 3.0 (cond. floor) | 3.0 (cond. floor) | **2.2e7 (garbage)** |

3-state 3-cycle: Tier-1 and Tier-3 exact (0–1e-15) on all components at every θ
tested (this fixture's E[T] stays O(1), so no term dominates → no conditioning
floor; FD is 1e-7…1e-9 there). Forward: trace == solve == phasic native exactly
in every case → **cycles are handled correctly** (2-cycle and 3-cycle).

## Verdict — θ-adjoint / cyclic-trace is VALIDATED as the B3 fix

1. **Both routes work on cyclic graphs.** Forward-correct vs native; gradient
   exact vs the true closed form at realistic scales — including θ=[1,1e-4] where
   FD is already 11.5% wrong — and non-crashing where the daisy FD aborts.
2. **Tier-1 == Tier-3 == oracle to machine precision.** The hand-written adjoint
   matches JAX autodiff and the linear-solve oracle exactly → the **Tier-3 C
   adjoint approach is de-risked** (the reverse tape walk + coefficient chaining
   is correct), and Tier-1 (pure-JAX elimination trace) is equally valid.
3. **They crush FD** — exact where FD is 11.5% wrong, and ~7 orders of magnitude
   better at extreme scale where FD is garbage and the daisy path crashes.
4. **Residual conditioning floor is method-independent and inherent** (honest
   caveat, confirming DR-A #2): at extreme mixed scale where E[T] is dominated by
   a ~1e-8 rate, the SUB-dominant gradient is corrupted (relerr 3) IDENTICALLY for
   the trace, the adjoint, AND the linear-solve oracle — it is the near-singular
   sub-generator (cond ~ 1/θ_small ~ 1e8 × machine-eps buries a signal of ~5e-9),
   a float64 precision floor of the linear algebra, not a defect of any method.
   It is a NARROW regime (only when one rate is ~1e-8 AND makes E[T] huge) and
   still ~7 orders milder than FD. Characterise + document it; do not treat it as
   a blocker. Possible future mitigation: a scaled/log basis for the elimination.

## Implications for the B3 implementation

- **The fix is exact AD via a true (vertex-dropping) Gaussian-elimination trace or
  a reverse-mode θ-adjoint over the PRC tape — NOT patching the jump-chain trace**
  with `1/(1−q)` (the code comment and this prototype agree that is the wrong
  lever). Both prototyped routes are correct; the choice is scalability:
  - **Tier-3 C θ-adjoint over the existing PRC tape** — reuses the shipped sojourn
    adjoint machinery, O(n) memory, cyclic-correct, scales to large graphs. The
    prototype's hand-written adjoint is a direct 1:1 template. **Recommended.**
  - **Tier-1 pure-JAX elimination trace** — attractive for the JAX story but needs
    a from-scratch rate-matrix elimination recorder (the existing jump-chain trace
    is the wrong basis) and is dense/O(n³) unless a sparse recorder is written.
- **Regardless of tier:** add the conditioning-floor characterisation to the
  gate; strengthen `test_fd_gradient_mixed_scale.py` with an exact-AD comparison
  (its two strict-xfails should XPASS once the fix lands, EXCEPT the extreme
  sub-dominant point which no float64 method resolves — pin that regime as a
  documented conditioning limit, not an xfail on the fix).

## Recommended next step
Scope the **Tier-3 C θ-adjoint** (extend the sojourn adjoint `phasic.c:10197-10264`
to a θ-gradient over the PRC tape), gated on cyclic parameterized fixtures against
this prototype's analytic oracle + the FD baseline. This prototype is the reference
implementation of the adjoint math to port to C.
