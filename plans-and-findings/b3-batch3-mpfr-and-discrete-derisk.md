# Batch-3 — MPFR gate (DONE) + discrete/was_dph de-risk (scoped, NOT implemented)

## MPFR/conditioning safety gate — DONE (commit b1f7fa0a)
`ptd_moments_grad_theta` declines (→ FD fallback) when the primal would use MPFR
(condition = max|mult|/min|mult| over the numeric tape > threshold, default 1e12,
or PHASIC_FORCE_MPFR), gated `#ifdef HAVE_MPFR`. Mirrors ptd_expected_waiting_time
exactly. Verified: well-conditioned → exact (J size 4); θ=[1,1e-13] (cond ~1e13)
→ declines (size 0); end-to-end jax.grad stays FINITE (no NaN); benign stays
exact; pin unaffected. HAVE_MPFR confirmed compiled in this build.

## Discrete / was_dph — DE-RISKED, math derived, NOT yet implemented
The discrete case is currently EXCLUDED correctly: the Python wiring gates on
`not discrete`, so discretized (was_dph) graphs stay on FD. No bug — just not
covered. Extending it is additive but is a LARGE increment with THREE parts:

### Part 1 — the renorm edge→θ Jacobian (sibling coupling)
`ptd_graph_update_weights` renormalises each vertex's out-edges for `was_dph`
graphs (phasic.c:5772+): p_e = w_e / S_v, S_v = Σ_{e'∈out(v)} w_{e'}, w_e = c_e·θ.
So the tape runs on the RENORMALISED weights p_e, and my chain reverse already
gives dQ/dp_e (the tape adjoint differentiates w.r.t. the current edge weights).
But the edge→θ contraction must use the QUOTIENT Jacobian, not c_e:
    ∂p_e/∂θ_j = (c_e^j − p_e · Σ_{e'∈out(v)} c_{e'}^j) / S_v .
This COUPLES sibling edges (the Σ over out(v) and S_v). Needs S_v per vertex,
which requires θ (S_v = Σ c_{e'}·θ) — so the C function must TAKE θ (the current
functions read only edge->weight = p_e). New signature:
`ptd_moments_grad_theta_dph(graph, nr_moments, theta, theta_len, J)`.

### Part 2 — the discrete moment correction (θ-independent LINEAR map)
`GraphBuilder::continuous_to_discrete_moments` (graph_builder.cpp:694) maps the
continuous moment vector m → discrete moments via a fixed lower-triangular matrix
C of factorial/binomial/Stirling-2 coefficients (u[j]=m[j-1]/j!, F[r]=r!·Σ
binom(r-1,i)(−1)^i u[r-i], out[k]=Σ stirling2(k,r) F[r]). Since C is θ-independent,
    d(discrete m)/dθ = C · d(continuous m)/dθ .
So: chain-reverse on the renormalised tape → dQ/dp_e per moment → contract with the
renorm Jacobian (Part 1) → d(continuous m)/dθ → apply C (Part 2) → d(discrete m)/dθ.
Requires porting d_factorial / d_binomial / d_stirling2 into phasic.c (small).

### Part 3 — wiring + gates
- Python: relax the `not discrete` gate to also enable discrete (linear) via the
  new θ-taking C function; the discrete branch of pmf_and_moments returns discrete
  moments so the callback must call the DPH gradient variant.
- De-risk: validate d(discrete m)/dθ vs native `moments(K, discrete=True)`
  central-difference at BENIGN scale on a `_erlang().discretize(0.5)` fixture, and
  vs a NegBinomial closed form where available (mirrors the was_dph cross-path
  gate in the DPH plan).

**Estimate:** comparable to Batch-2+3 combined (θ-passing + sibling-coupled
contraction + correction matrix + de-risk). Recommended as the next FOCUSED
increment rather than tail-ended onto this session.

## Other remaining Batch-3 items (each own de-risk)
- log/formula weight-mode Jacobian (∂w/∂θ = w_e/θ_j for log; a bytecode-tape
  adjoint for formula) — NOTE the memory flags a pre-existing bug: moments_from_graph
  / joint_index silently IGNORE weight_mode, so log needs care.
- joint-index: its forward IS the transpose walk → reverse-over-reverse.
- hierarchical SCC: θ-dependent phantom-weight stitching.

## Cross-cutting decisions (for the user)
- Make the exact path the DEFAULT (vs opt-in `exact_moment_grad`)? Now correct for
  continuous/linear/monolithic moments, MPFR-safe, higher-order, vmap-safe.
- Native FFI gradient handler (perf) vs the current correct + vmap-safe pure_callback.
- Merge `fd-b3-experiments` (still UNMERGED; master untouched throughout).
