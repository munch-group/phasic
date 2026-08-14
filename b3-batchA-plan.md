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

## Adversarial plan-review record (2026-08-14; v1 → v2, folded as this amendment)

Both refuters SOUND-WITH-CORRECTIONS. **Binding v2 amendments:**

1. **DPH arm: REFUTED, pre-committed NO-GO** [design MAJOR 1, by direct
   numpy probe]: the c2d correction rests on U/P commutation, which
   reward-scaling (UΔr) breaks — 2nd moments provably wrong
   (r=[2,1,3]: chain+c2d 58.311 vs true 51.644; r=[0,2,1]: 13.289 vs
   11.956; 1st moments always agree — the headline trap class).
   SHIPPED CONTRACT: linear+log support rewards; `_dph` declines
   `rewards_len != 0` (C wrapper -1 = defense in depth) AND the Python
   layer adds a STATIC dph+rewards decline branch with a truthful INFO
   log naming this refutation [process MAJOR 2: "may retain" → MUST;
   the C-level-only decline would surface through `_one`'s misleading
   conditioning message and pay the C attempt per call]. The dph gate
   (`dr_batchA_dph_rewards_gate.py`) ships as a CONFIRMING artifact
   (integer rewards only — the discrete transform throws on
   fractional/negative [design MAJOR 2a]; decision anchored on
   FD-of-the-PRIMAL, never a chain+c2d oracle that would mirror the
   wrong formula [2b]; both was_dph and native-DPH sub-kinds [2c]).
2. **Exact dispatch restricted to `rewards.ndim == 1`** [design MAJOR
   3]: the 1-D leaf officially accepts 2-D rewards through
   `_compute_pure`; the exact Jacobian shape/contraction and the
   pybind vector cast are 1-D-only. 2-D keeps FD + the INFO log; the
   multivariate wrapper slices per-feature 1-D and is where the free
   side effect genuinely lives (verified).
3. **The svgd blast radius is REAL and needs a fold-time USER
   DECISION** [process CRITICAL 1 + design MINOR 5]: svgd's rewards
   leaves build the model with the callee DEFAULT
   `exact_moment_grad=True`; post-A every `Graph.svgd(rewards=...)`
   run flips FD→exact silently, and R29 rejects ANY explicit value on
   rewards leaves (opt-out locked until G.2). Options to the user:
   bundle G.2's R29-relaxation + forwarding into A (closes the
   asymmetry at merge) vs accept the window vs suppress until G.2.
   Also: a front-door svgd+rewards smoke sampling the ACTUAL
   particle-init distribution joins the gates (the E lesson); the
   svgd docstring's "with rewards the exact path currently declines"
   clause goes stale (G5 shipped-text list); cost framing corrected
   [process MINOR 6]: the FD loop is LOAD-BEARING for the pmf gradient
   (no exact pmf path — Deferred 3), so post-A rewards runs get exact
   parity with rewardless leaf-5 runs, not a new double-cost.
4. **Test cells added** [process MAJOR 3/4]: rewards=all-ones ≡
   rewardless exact (bitwise-class sensitivity check on both hooks);
   rewards=all-zeros (degenerate, exercises the isinf-skip lines);
   contract statement: exact follows the primal's acceptance domain
   (continuous: non-negative reals; discrete: non-negative integers —
   enforced upstream); the vacuous same-install "rewards=None vs
   omitted" cell REPLACED by the all-ones cell; tolerances
   anchored-to-measured at authoring (the standing discipline);
   multivariate engagement check gets an absolute pin.
5. **MPFR parenthetical corrected** [design MINOR 4]: with rewards the
   FORWARD runs on the reward-TRANSFORMED tape while the gate scans
   the original tape — a new (edge-case) fwd/bwd surface at extreme
   reward scales; one extreme-reward-scale case added to micro-gate
   (b); realistic SVGD rewards (0/1/small ints) unaffected.
6. **Fate table (pre-filled by both reviews)**: zero literal-string
   pins; `inference/test_exact_grad_rewards.py::test_exact_grad_with_
   rewards_logs_why_fd_is_used` flips BY DESIGN → rewritten (linear:
   exact engages, no decline log; dph: the refutation decline log);
   `...matches_fd_and_forward_central_diff` stays green but its
   docstring/tolerances are reworked (post-A it compares genuinely
   different paths; keep the central-diff oracle, loosen the FD leg);
   `...without_rewards_unaffected` MUST stay green (live rewardless
   guard); R29 pins unchanged (until the user's decision on 3);
   seeded svgd+rewards accuracy tests may shift numerically — G3
   watches, merge review records any.
7. **G2 additions** [design MINOR 6, process MINOR 9]:
   `inference/test_exact_grad_discrete.py`,
   `inference/test_exact_grad_log_weight_mode.py` (wrapper-owning),
   `inference/test_svgd_exact_moment_grad_kwarg.py` +
   the new svgd smoke (behavior-reach); process-map rows at G5 for the
   new/rewritten files.
8. Anchors/nits: pybind span `:1915-1933`; the rewards_len decline
   lives in the CORE (better than the wrapper-level theta convention —
   stated as such); `:7988-7996`; G0 records the docs-only delta
   (`95da8e35`).
9. **Verified by review (recorded so implementation doesn't
   re-litigate)**: hook placement is per-stage correct for all three
   contraction kinds (hook 1 inside the K-loop; hook 2 the per-stage
   VJP; `dm` inherits the scale via post-scale `st[]` snapshots; log
   branch and stage-2 need nothing); expand_dims + unbatched second
   arg WORKS (probed; the leaf's own forward already does 3-arg
   expand_dims with the disambiguation pattern to reuse — 'sequential'
   switch unnecessary and avoided); the three C functions' only
   callers are the C++ wrappers; svgd rewards is a genuine model
   argument (not a closure).
