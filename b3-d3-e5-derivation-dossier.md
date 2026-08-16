# Deferred-3 E5 — chain-rule derivation dossier for route (ii)

**Plan:** `deferred-3-pdf-gradient-revival-plan.md` §4-E5 ("a written
derivation of every term ... checked term-by-term against the E1/E2
oracle by zeroing-out experiments").
**Grounding:** every convention below is taken from the gate-validated
reference `experiments/dr_d3_e123_pdf_routes.py` (parity vs production
`g.pdf` 0..6e-15 on four fixtures; closed-form gates), not from memory.
**Status:** derivation + term inventory COMPLETE; the zeroing runs are
recorded in §5 (executed with `experiments/dr_d3_e5_term_zeroing.py`).

## 1. Setup and notation

Continuous phase-type on n vertices. Transient set T = vertices with
out-edges; absorbing set A = the rest. Rate matrix R(θ):
`R[i,j] = Σ_e w_e` over edges i→j, with `w_e = c_e·θ` (linear mode,
P parameters) or `w_e = const` (constant edges, θ-independent).
Initial mass α over the start-edge targets (start-edge weights
normalized to 1; production's special INSTANTANEOUS first step from
the start vertex equals α itself because the start vertex is emptied
at t=0 — reference `jax_pdf_series` docstring).

Uniformized DTMC at PINNED λ (λ ≥ max exit rate, θ-independent by
construction — the route's defining choice):

- one step: `p' = p + (inflow − outflow)/λ` on T, then the mass that
  landed in A is HARVESTED: `π_step = Σ_A p'` and A's entries are
  zeroed (reference `step`).
- In matrix form on T: `P̃(θ) = I_T + Q_TT(θ)/λ`,
  absorption vector `a(θ) = Q_TA(θ)·1/λ`;
  `p_j = p_{j−1}·P̃`, `π_j = p_{j−1}·a` (mass absorbed at step j).
- Index convention (measured, the off-by-one the gate catches at
  2.07e-2): the scan's `pi[k]` (0-based) is π_{k+1}.

## 2. The density

```
f(t) = λ · Σ_{k=0}^∞ Poisson(k; λt) · π_{k+1}
```

Validated: expo 1.63e-11 / erlang3 7.46e-11 vs closed forms;
cyclic4 4.88e-5 vs a high-λ Euler reference (tol 1e-3).

## 3. The gradient — every term, named

With λ pinned, `dλ/dθ = 0`, so the Poisson weights `Poisson(k; λt)`
carry NO θ-dependence — that is the entire point of the pinning and
what makes the gradient exact (measured: 1.79e-10 / 1.36e-11 /
2.04e-10 on expo/erlang3/cyclic4):

```
df/dθ_r = λ · Σ_k Poisson(k; λt) · dπ_{k+1}/dθ_r
```

with the tangent recursion (forward-mode, seed r = one θ column):

- **T0 (initialization):** `dp_0/dθ_r = dα/dθ_r`. In the validated
  scope this is ZERO — the reference fixtures' start edges are
  constant. **Implementation-time obligation:** production start/IPV
  edges CAN be parameterized; the implementation must either seed
  `dα/dθ_r` (extending the recursion below unchanged) or statically
  decline θ-dependent start edges LOUDLY. Not covered by any E3 gate.
- **T1 (propagation):**
  `dp_j = dp_{j−1}·P̃ + p_{j−1}·dP̃_r`, where
  `dP̃_r = (dQ_TT/dθ_r)/λ` — in linear mode `dQ[i,j]/dθ_r = Σ_e c_e^r`
  over edges i→j, diagonal `−Σ` of the row's outgoing coefficient sums
  (plus NO constant-edge contribution).
- **T2 (harvest):**
  `dπ_j = dp_{j−1}·a + p_{j−1}·da_r`, `da_r = (dQ_TA/dθ_r)·1/λ`.
- **T3 (Poisson weights):** ZERO by pinning. (If λ were auto-derived
  from θ, this term is `Σ_k ∂Poisson(k;λt)/∂λ·(dλ/dθ)·π + …` — the
  route-(i) bias family. Forbidden by design; the implementation must
  never re-derive λ from θ inside exact mode — the D2-plan §2.1
  fwd/bwd-consistency requirement, shared λ-policy note in
  `b3-d3-derisk-findings.md` §E4.)

Cost shape: forward-mode with P seeds over the shared π-tangent
sequence — `O(n·P)` memory, no checkpointing; the dπ sequence is
computed ONCE and reused for every observation time t (only the
Poisson weights change with t). This is also the C-implementation
shape (mirrors the joint-index forward-mode precedent).

## 4. Truncation and failure semantics

- **Truncation:** K = λt + 6√(max(λt,1)) + 10 (the reference's rule).
  Value tail: ≤ λ·P(Pois(λt) > K) since π_j ≤ 1. Gradient tail: |dπ_j|
  grows at most LINEARLY in j (a j-fold product with one dP̃ insertion
  per summand), and the Poisson tail decays super-exponentially, so
  the same K keeps the gradient tail subdominant — confirmed
  empirically by the 1e-10-class gradient parities at the rule's K.
- **λ below the max exit rate:** P̃ has negative diagonal entries;
  π_j can go negative and the value silently corrupts (measured:
  0.01171 at λ=2 vs rate 10). The stepper MUST carry a `p < 0` check
  that raises (loud path), per the E3 finding and the plan's
  "detectable, loud" requirement.

## 5. Term-zeroing verification protocol (E5's second half)

`experiments/dr_d3_e5_term_zeroing.py`: an explicit-tangent
implementation of §3 (independent of `jax.grad` — it IS the
C-implementation prototype), gated first by parity vs `jax.grad` of
the intact mixture, then each term dropped in turn:

- **Z-parity:** explicit tangent == jax.grad(mixture) on expo/erlang3/
  cyclic4 (machine precision expected).
- **Z1 (drop `p_{j−1}·dP̃` inside T1):** predicted mismatch wherever
  the chain takes ≥1 transient step (all fixtures).
- **Z2 (drop `p_{j−1}·da` in T2):** predicted mismatch on every
  fixture (the direct absorption channel).
- **Z3 (mis-align π index):** ALREADY BUILT AND MEASURED — the
  off_by_one_wrong gate, 2.07e-2 (E3 record).
- Each dropped term must move the answer OUT of tolerance where
  predicted — a zeroing that changes nothing would mean the dossier
  names a term the implementation doesn't actually need (or the gate
  is too weak).

RESULTS (measured 2026-08-16; predictions above written first):

| fixture | Z-parity (value / grad rel) | Z1 grad rel | Z2 grad rel |
|---|---|---|---|
| expo | 1.82e-14 / 4.02e-15 | 1.01e+01 | 1.11e+01 |
| erlang3 | 2.16e-14 / 2.12e-14 | 5.22e-01 | 4.78e-01 |
| cyclic4 | 1.98e-14 / 1.83e-14 | 2.55e-01 | 8.84e-01 |

ALL PASS: the explicit forward-mode tangent recursion (the
C-implementation prototype) reproduces `jax.grad` of the intact
mixture at ~2e-14 on every fixture, and each dropped term moves the
gradient out of tolerance (25%..1000% relative) exactly where §5
predicted — every named term is load-bearing, and the dossier names
no term the implementation doesn't need. E5 is COMPLETE; with Z3
(the E3 off-by-one gate, 2.07e-2) all three zeroing classes are
covered.
