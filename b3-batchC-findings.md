# Batch C findings

**Plan:** `b3-batchC-plan.md` v1 + v2 (binding amendment, two-refuter
review folded). **Branch:** `b3/batchC-callback` (base `769fe3a1`,
worktree `../phasic-batchC`, own pixi env, install verified — the
formula binding is present, confirming post-B content).

## G0

Branch base `769fe3a1` (Batch B close-out). Delta above the eighth
ledger stamp (`c6cc38b9`): docs-only (close-out + C plan commits
`584b3d3b`/`b25334b2`, which live on master after the branch cut and
touch no code).

## Plan-review de-risks (resolved by refuter probes, recorded in v2)

- **D-C1 (jax-over-callback by execution)**: RESOLVED — jnp-native
  works under grad/jit(grad)/vmap(grad)/jit(vmap(grad))/vmap(jit(grad))
  end-to-end through a custom_vjp mimic; numpy callback →
  TracerArrayConversionError; float()-inside → ConcretizationTypeError;
  data-dependent branching → passes PLAIN grad but raises
  TracerBoolConversionError under the DEPLOYED jit(grad) → v2 §B
  mandates probing with the jitted transform over all edges
  (construction success ⇒ per-call success; "JAX-native UNDER JIT").
  Non-finite gradients RETURN inf (no raise) → the J finiteness check
  is the right net.
- **D-C2 (W strategy/cost)**: SIMPLIFIED — coefficient lengths are
  provably uniform in-scope (serialize() raises on inconsistent
  lengths); one jit trace + ~60 µs warm evals; measure the ADD-ON cost
  at I-phase (v2 §E: the exit does not replace the pmf FD loop).
- **D-C3 pre-arm (the chain-rule math, BEFORE any C)**: PASSED by the
  design refuter's probe — FD-binp (marker-perturbing
  update_weights(callback=) per edge) ∘ jax.grad-W on a branchy
  K=3/P=2 aux-coeff graph: **1.2e-10** vs the shipped
  `_moments_grad_theta` (linear-equivalent callback; primal parity
  exact) and **4.1e-10** vs theta-FD (nonlinear exp/ratio).
  Post-implementation D-C3 re-verifies with the real exit.
- **D-C4 (repurposed by v2 §A)**: the decoupled classes are SUPPORTED,
  not declined — `update_weights(theta, callback=)` accepts len-4/
  len-1/len-0 theta by documented design (`phasiccpp.cpp:1879-1884`);
  the B crash class is ABSENT; the binp exit is theta-blind. Durable
  branch probe: `dr_batchC_d5_derisk.py` (theta-length freedom, lazy
  aux-coeff idiom param_length lock, clone-with-short-theta all PASS).
- **Identity/order**: input_specs k-order = first-use order in the
  elimination stream ≠ serialize()'s vertex-major order, and flips
  under use_dyn_ordering → the (v,e)/frozen export is REQUIRED (v2 §C);
  no existing API exposes tape-input order.

## D-C5 (discrete × callback exclusion evidence, run on the branch)

- **was_dph (discretize()) × callback: SILENTLY COMPUTES** with a
  strictly-positive callback (moments [2.75, 15.625] at t=0.3,
  [1.571..., 5.224...] at t=0.6) → the exit wrapper's `was_dph`
  decline is LOAD-BEARING, matching the log/formula precedents.
  First-run lesson: a callback mapping the discretize-created aux
  edge's coefficients to weight 0 raises the positivity check — a
  fixture artifact, not mode evidence; the probe uses `c0·t0 + 0.1`.
- **native DPH (`is_discrete=True`) × callback: silently computes**
  (C cannot see the attribute — no struct field) → the Python
  `_effective_discrete` static gate is the ONLY defense, exactly the
  formula/log situation. Both arms recorded in
  `experiments/dr_batchC_d5_derisk.py` (ALL PASS).

## I1+I2 (C exit + bindings, commit `ff0e9d91`)

`PTD_B3_BINP_EXIT` + core `binp_exit` out-param (four existing call
sites gain a NULL) + the copy case + `ptd_moments_binp_exit`
(C-allocating single-call form per v2 §C: binp K×ni + (v,e) + C-side
frozen flag; was_dph decline load-bearing per D-C5) + C++
`moments_binp_exit` (ONE tuple: binp, per-input coefficient vectors
extracted from the same (v,e) resolution, frozen) + pybind. CRLF/LF
verified.

**Micro-gates (`dr_batchC_i1_gate.py`) ALL PASS (first run):**
(a) byte-identity vs 7 FRESH pre-C goldens (lin/log/formula ×
plain+rewards + dph_plain) — all `identical=True`;
(b1) exit+matmul on the linear-equivalent callback vs the LINEAR exact
twin: rel **0.00e+00** at benign AND MIXED (θ=[1,1e-8]) scale;
(b2) nonlinear K=1..3: 4.0e-10 / 2.2e-10 / 2.2e-10;
(b3) LAZY aux-coefficients idiom (len 3 > n_theta 2, no
set_param_length — the v2 §A supported class): 3.2e-11;
(c) rewards 1.8e-10 + all-ones BITWISE; (d) was_dph + rewards-len
declines (empty-tuple contract); (e) frozen flags (3 of 4 inputs on
the aux-constant fixture) + skip-aware composition 1.1e-10.
Standing chain on the C build: 3 jac-gates + A gate + B gate (all vs
fresh pre-C goldens) ALL PASS.

## I3+I4 (Python + tests, commit `e6d3b613`)

Dispatch: `_callback_scope_ok` = continuous ∧ JAX-native-under-jit
(the DEPLOYED `jax.jit(jax.grad(f))` probed over ALL param edges'
coefficient vectors at construction — v2 §B); NO theta-dim predicate
(v2 §A); static declines discrete×callback and non-JAX-native (the
latter keeps the "weight_mode" + "finite differences" tokens the
pre-existing out-of-scope pin greps — fate break resolved by message
design, test green with a docstring rework only). The `_one` arm:
clone-update with the build-time-captured callback → exit → skip-aware
W (frozen rows excluded) → `J = binp @ W` → explicit non-finite
decline log. Per-arm update_weights restructure (each kind sets its
own weight rule). Shipped text per plan §7 + v2 §G.

**I3 smoke:** model-level twin parity max abs diff **0.0**;
non-JAX-native decline logged once, FD finite.
**`test_exact_grad_callback_mode.py` 12/12** (+ the reworked discrete
file 12/12): twin parity 1e-12 + mixed scale; nonlinear; LAZY-decoupled
ENGAGES (the anti-B cell); rewards + all-ones bitwise; non-JAX-native
PERMANENT decline; theta-branching declines AT CONSTRUCTION (the
deployed-transform probe cell — the v2 §B design proof); discrete
decline; engagement flip; vmap/jit bitwise; svgd front door with the
success-floor (explicit theta_dim per the stated precondition);
multivariate (hard pin to be measured/added at G4 fold per precedent).

## G2 (expanded map incl. the full callback inventory, verbatim)

17 files (moments-core row + svgd-config row + B formula row + the
callback inventory from the plan-review enumeration):
**200 passed, 31 skipped, 3 xfailed, 0 failed** (88.15s).

## G3 (full suite, 32 chunks, worktree, verbatim)

Union check OK (158 collected files == 32-group union; output per
group). Summed: **1990 passed / 0 failed / 84 skipped / 24 xfailed /
0 xpassed / 0 errors** = the eighth ledger stamp's 1978 + Batch C's 12
new tests exactly. Three sleep-kill interruptions resumed from
preserved per-group outputs (no green group re-run).

## G4 adversarial diff review (two refuters, 2026-08-14)

**Wiring/math refuter: SOUND-WITH-CORRECTIONS** (no wrong gradient,
crash, or leak). Own oracle on a branchy 6-vertex fixture: 8 attack
classes all pass (same-theta-everywhere 1.9e-11 with unused component
EXACTLY 0; theta-ignoring callback grad exactly 0.0; jnp.where 2e-10;
sigmoid/tanh 6.8e-10; aux coefficients 2.2e-10; rewards×nonlinear
1e-9; **theta LONGER than C param_length builds+engages** 1e-9; K=1/3).
Per-arm update_weights restructure verified in source (all five arms)
AND by executing the five mode suites (60 passed). Memory: 13,000
calls across success/decline paths, +0.00 MB. The deployed-jit probe
catches COEFFICIENT-branching too (declines at construction).
Cross-mode isolation (incl. both-properties-set: last-setter wins,
dispatch/forward agree). svgd assertions non-vacuous.
**Correction folded (LOW-1): the "frozen-row non-finite lawfully
ignored" claim was FALSE as implemented** (unmasked inf·0=NaN would
decline) — the matmul is now MASKED to the engaged columns
(`binp[:, eng] @ W[eng]`), making the stated semantics real and
matching the C kinds' skip-before-accumulate. Conservative-direction
defect only; no test could hit it (needs a trap-vertex fixture — the
known pre-existing repo gap).

**Tests/process refuter: SOUND-WITH-CORRECTIONS** (no shipped defect;
suite non-vacuous under BOTH simulated regressions — exit-always-
declines and scope-never-True; G3 arithmetic independently recomputed:
1990/0/84/24 exact). Folded: the mandated-but-swapped non-finite-net
pytest cell (its own probe confirmed the net works:
"non-finite callback gradient" log + finite FD); cells 3/12
construction-inside-capture (proven blind to a dispatch regression
otherwise); cell 9 success-size spy; the multivariate HARD pin
[-11.508732477887918, -5.563485415672725]; the canonical
set_param_length-class engagement cell (D-C4 gap); gate (d)
non-parameterized decline; the D-C2 measurement (below).

## D-C2 add-on cost (v2 §E, measured at G4 fold)

update+exit+W+matmul per call: **5.7 ms at ni≈10, 9.7 ms at ni≈100**
(K=2, P=2; per-edge jitted-grad dispatch ~40µs dominates the ni
scaling; the C exit itself is the same O(n³) stage-0/1 the other kinds
run). This is an ADD-ON to the pmf FD loop (which still runs), same
economics as linear/log/formula — justification is mixed-scale
accuracy, not speed. The batched-vmap W option remains the recorded
future optimization if profiling ever shows the per-edge dispatch
mattering at production ni.

## Plan-deviation register (G4)

1. pybind tuple is `(binp, coeffs, frozen)` — not v2 §C's literal
   `(binp, v, e, frozen)`; sanctioned by v1 §2.3's grounding item
   (the C++ layer extracts coefficients from the same (v,e) pass,
   removing Python's index-mapping surface). C-level v/e out-params
   exist as specified.
2. W coefficients fetched per-call from the exit tuple + per-row loop
   (vs §D's "fetched once at construction + stacked matrix/vmap") —
   simpler, measured acceptable (D-C2 above); the construction-fetch +
   vmap form is the recorded optimization.
3. Commits I1+I2 / I3+I4 (2, not 4) — the A/B-ledgered coupling class.
4. Plan-§6 cells 4+8 merged into one; the §B branching cell added.
5. Gate (d) initially omitted the non-parameterized probe (folded in).
6. G1(e) as-planned (non-finite net) was initially swapped for the
   frozen-flag cell — both now exist (pytest cell + gate (e)).
7. mv hard pin deferred to fold (A/B precedent) — now in.
8. The per-arm update_weights restructure touches existing arm lines
   ([[feedback_no_modify_existing]] boundary case): disclosed at I3,
   behavior-preserving (each arm sets exactly the weight rule it had),
   G3-clean, and refuter-verified arm-by-arm + by suite execution.

## mcmc note (v2 §H)

`Graph.mcmc` builds these models at the default exact_moment_grad —
post-C a callback graph under mcmc runs the construction probe (INFO
logs; gradients inert — mcmc is gradient-free, zero jax.grad in
mcmc.py). mcmc auto-infers theta_dim = param_length() without
_resolve_inference_theta_dim, so the probe's theta there has
coefficient length — harmless (probe correctness is P-independent).

## Quasi-dead arm note (wiring INFO-i)

The "no weight_callback set" branch of the non-JAX-native decline
message logs and is immediately followed by the builder's own
ValueError (~:7739) — no model ever exists carrying that message;
kept as defense-in-depth.

## 9th ledger stamp (main checkout, post-merge, 2026-08-15)

Measured 32-group run (union == 158 files): raw **1990 / 2 / 84 / 24**;
total 1992 = eighth stamp 1978 + Batch C's 14 exact. Both failures
environment-caused (machine throttled/sleeping, ~2-4× slower than the
morning run on the same groups): the scc cpu>wall timing invariant
(passed solo) and a pure pytest-timeout wall-clock kill on the
exposure SVGD test (no assertion failed; its group ran green at 507 s
on identical content in the worktree G3). Effective stamp:
**1992 / 0 / 84 / 24**. Full evidence chain in the ledger entry.
