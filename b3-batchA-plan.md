# Batch A plan — rewards support in the moments adjoint

**Status: DRAFT v1, pending the mandated two-refuter adversarial plan
review.** Design of record: master plan §3 (the headline finding — the
fix is NOT seed-only: `Graph.moments` re-multiplies by rewards at EVERY
stage transition, so a naive seed patch is silently wrong for
nr_moments≥2, verified 2.5-vs-3.5 in the feasibility pass; the correct
fix is exactly two scale lines) + Batch 0's landed extraction (the two
lines live ONCE in `ptd_b3_moments_core`, at the PRE-MARKED hooks:
`src/c/phasic.c:10858` "[Batch-A rewards hook 1/2: seed-scale line]"
and `:10892` "[hook 2/2: adjoint-scale line]"; the core's signature
note anticipates this extension). Branch: `b3/batchA-rewards`,
worktree `../phasic-batchA` (NATIVE change → own pixi env; main
checkout install = pre-A golden). Baseline: ledger sixth stamp @
`c475a78c` (1947/0/84/24, empty; next full run 1951 — four tests added
at E's G4 fold, post-G3). Findings: `b3-batchA-findings.md`.
Experiments: `experiments/dr_batchA_*.py`.

## What A delivers (plain language)

`Graph.pmf_and_moments_from_graph` models called WITH `rewards` today
always decline the exact moment gradient (`__init__.py:7988-7997`:
INFO log → FD — the documented Batch-D-era guard). A makes the exact
reverse-mode moment Jacobian reward-aware, so rewards-bearing moments
models (including every `pmf_and_moments_from_graph_multivariate`
feature slice — the FREE side effect, master §3: the multivariate
wrapper already passes concrete per-feature rewards and needs zero
changes of its own) get exact gradients. This unblocks Batches B and C
(formula/callback modes build on the same core) and Batch G.2 (svgd
leaves 3/4 plumbing — NOT part of A, see scope 5).

## The mathematics (master §3, verified there empirically)

The reward-weighted moment chain is `a_j = replay(a_{j-1} .* rewards)`
— rewards re-scale at EVERY transition. In the core's stage-0/stage-1
structure this is exactly two lines, at the pre-marked hooks:
- seed: `out[v] = seed[v]` → `out[v] = seed[v] * (rewards ? rewards[v] : 1.0)`
- adjoint: `bar_out[v] = adj[v]` → `bar_out[v] = adj[v] * (rewards ? rewards[v] : 1.0)`
  (the reverse-mode VJP of an elementwise scale is the same scale;
  rewards is theta-independent → NO new theta/`dm[]` term).
Nothing else changes: tape extraction, the MPFR gate (already
reward-blind identically to the primal's own pre-scan — not a new
gap), stage-2 contraction, the was_dph/log branches (but see the DPH
RISK below).

## Scope

1. **C (`src/c/phasic.c` + `api/c/phasic.h`):**
   - `ptd_b3_moments_core` gains `(const double *rewards,
     size_t rewards_len)`; the two hook lines apply the scale; decline
     `-1` if `rewards_len != 0 && rewards_len != graph->vertices_length`
     (the existing theta_len decline convention).
   - The three public wrappers (`ptd_moments_grad_theta`,
     `ptd_moments_grad_theta_dph`, `ptd_moments_grad_theta_log`) gain
     the same trailing parameter pair — a SIGNATURE EXTENSION of
     shipped C functions, explicitly specified by master §3 ("new
     (rewards, rewards_len) parameter on all three C signatures");
     every caller updated in the same commit (the C++ wrappers are the
     only callers — verified at implementation, grep'd by the review).
   - **DPH RISK (master §3's flagged open risk, the batch's one
     genuine unknown):** whether
     `ptd_dph_correct_discrete_moment_grad`'s combinatorial
     continuous→discrete correction remains valid under
     reward-weighting is plausible-by-analogy but NOT re-derived. A
     dedicated gate (`experiments/dr_batchA_dph_rewards_gate.py`:
     reward-weighted discrete moments, exact vs FD vs a dense oracle,
     nr_moments 1..3) MUST pass before `_dph` rewards ships. **Defined
     NO-GO path (not a stop): if the gate fails, the `_dph` wrapper
     declines rewards (`rewards_len != 0 → -1` in that wrapper only),
     the batch ships linear+log rewards, and the dph gap is ledgered
     with the gate's numbers.**
   - Micro-gates (`experiments/dr_batchA_i1_gate.py`): (a)
     rewards=NULL/empty byte-identity vs the pre-A install on the
     three existing jac-gate fixtures (the Batch-0/H cross-install
     dump/check pattern); (b) linear+log rewards correctness vs a
     dense-JAX oracle AND tight FD at nr_moments 1..3, INCLUDING the
     headline class (a fixture where seed-only would give the wrong
     2nd moment — assert the exact path matches the true value, not
     the seed-only value); (c) the dph gate above; (d) rewards_len
     mismatch declines to -1 in all three wrappers.
2. **C++ (`api/cpp/phasiccpp.h:560-599`) + pybind
   (`src/cpp/phasic_pybind.cpp:1915-1930`):** the three
   `moments_grad_theta*` methods and `.def()`s gain a default-empty
   `rewards` vector (`std::vector<double> rewards = {}` /
   `py::arg("rewards") = ...`) — source-compatible; existing callers
   unchanged.
3. **Python (`pmf_and_moments_from_graph`):**
   - `_exact_moments_jac_np(theta_np)` →
     `_exact_moments_jac_np(theta_np, rewards_np)`: rewards threaded
     across the `pure_callback` boundary as a genuine per-call array
     (the joint-index `(theta, vertex_indices)` precedent; NOT closed
     over — the model contract passes rewards per call).
   - The `_rewards_provided` decline branch (`:7988-7997`) becomes a
     REAL dispatch: exact-with-rewards when enabled, same
     `jnp.where(_exact_ok, ...)` blending as the rewardless path. The
     INFO decline log is REMOVED for the supported cases; the
     was_dph-gate outcome may retain it for dph+rewards if the NO-GO
     path fires (message names the dph gate's finding).
   - Multivariate: no code change; a test proves each feature slice's
     exact path engages.
4. **Fate table:** the guard's decline log is load-bearing in any test
   that pins "does not yet support rewards" — grep at plan-review time
   (the reviewers pre-fill it); expected: `test_exact_grad_rewards.py`
   (name suggests it pins the CURRENT decline behavior — its tests
   flip BY DESIGN and are rewritten to pin the new exact-with-rewards
   behavior, enumerated per-test in v2). Everything else unchanged.
5. **Out of scope, stated:** svgd leaves 3/4 plumbing + R29's rewards
   message update (Batch G.2, gated on A per master §9); the moments
   leaf's compute-both-branches dispatch architecture (`jnp.where`
   blending — a D6-class redesign, pre-existing, untouched by A);
   `Graph.moments_from_graph`/`method_of_moments` (separate FD-only
   paths, CLAUDE.md-documented, out of scope since the master plan).

## Implementation order

I1 C core+wrappers+headers with micro-gates (a)(d) → the dph gate (c)
decides the `_dph` arm → I2 C++/pybind → I3 Python dispatch → I4 tests
(`tests/pytest/inference/test_exact_grad_rewards_supported.py`, new
file; + the enumerated rewrites in the fate-table file):
  1. Exact-with-rewards vs dense oracle + tight FD (linear), nr_moments
     1..3, incl. the seed-only-would-be-wrong fixture.
  2. Same for log mode; dph per the gate outcome (test asserts
     whichever contract shipped: exact or documented decline).
  3. Multivariate: per-feature exact engagement (spy/log) + gradient
     parity vs FD.
  4. rewards=None byte-identity vs the no-rewards path (bitwise, same
     install) + the cross-install golden (micro-gate (a)).
  5. rewards_len-mismatch declines loudly at the wrapper level; the
     Python layer's shape validation (existing) still fires first at
     model level.
  6. fixed_mask × rewards × exact.
  7. vmap/jit composition on the rewards path (the leaf's
     expand_dims callback — batched theta with per-call rewards).
- Gates: G0 (ledger @ `c475a78c` + enumerated delta); G1 = micro-gates
  + I4 + fate table; G2 = the moments-core row (3 jac-gates +
  `test_gate_moments_3way.py` + `inference/test_jax_integration.py`
  ledgered subset) + `inference/test_exact_grad_rewards.py` +
  `inference/test_multivariate_correctness.py` + always-run
  mixed-scale; G3 chunked `-rf` with preserved outputs (the E
  amendment) vs ledger (expect 1951 + new); G4 two refuters; G5 merge
  review + ledger re-stamp + tracker + master §3/§15 ticks + B/C/G.2
  unblock notes + CLAUDE.md + memory + install rebuild + process-map
  row addition for the new test file.

## Risks

1. The DPH correction under rewards — the gate decides; NO-GO path
   defined (linear+log ship regardless).
2. Signature extension of three shipped C functions — sanctioned by
   the design of record; byte-identity micro-gate (a) is the proof the
   rewardless path is untouched; a red gate = STOP + the additive
   fallback shape (new `_rw` entry points) + back to the user.
3. The `test_exact_grad_rewards.py` fate flip — enumerated per-test in
   v2 after the reviewers' grep.
4. The moments leaf's `vmap_method='expand_dims'` callback with a
   SECOND array argument (rewards): the joint-index precedent used
   'sequential' — whether expand_dims composes with the added arg is
   verified by a de-risk probe BEFORE I3 (a 10-line jaxpr check), not
   assumed.
