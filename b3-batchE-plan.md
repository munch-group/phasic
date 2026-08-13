# Batch E plan — joint-index BAKED-mode exact gradient + svgd leaf-2 plumbing

**Status: DRAFT v1, pending the mandated two-refuter adversarial plan
review.** Design of record: master plan §7 (the scatter-add derivation,
verified there by direct derivation; sequencing option (i) happened —
Batch F merged first, so E builds directly into the probe-and-commit
wiring) + the F merge review's requirement ("E probes the ACTUAL baked
union") + master §9 leaf 2's gate ("ship leaf-2's `Graph.svgd()` kwarg
pass-through **alongside Batch E**" — satisfied by doing both in this
batch). Branch: `b3/batchE-baked-exact`, worktree `../phasic-batchE`
(Python-only; worktree keeps the pre-E install as golden reference).
Baseline: ledger fifth stamp @ `0c052cfe` (1917/0/84/24, empty; next
full run 1919 — two tests added at G.1's G4 fold, post-G3). G0
enumerates the master delta at branch time. Findings:
`b3-batchE-findings.md`. Experiments: `experiments/dr_batchE_*.py`.

## What E delivers (plain language)

`Graph.svgd()` on a joint-probability graph with NO epochs and NO
exposure — **the default and most common joint-prob case** — bakes the
observations into the model (`observed_indices`), which today
STATICALLY excludes the exact gradient (`__init__.py:8287-8293`: INFO
log → FD). E implements the baked exact backward (no new C, no new FFI
shape) and plumbs a public `Graph.svgd(exact_grad=...)` kwarg to this
leaf, so the most common joint-prob SVGD usage can opt into exact
gradients.

## The mathematics (from master §7, verified there; restated)

Baked forward (`__init__.py:8463-8497`): the callback computes
`uniq_sojourn` at the STATIC unique indices; `uniq_probs =
uniq_sojourn / sum(all_sojourn)`; `sojourn_probs =
uniq_probs[_inverse_idx_jnp]` — a GATHER. Its VJP is a SCATTER-ADD:
`g_uniq = jnp.zeros(n_unique, dtype).at[_inverse_idx_jnp].add(g_visits)`
(one JAX primitive). The EXISTING quotient rule (`:8701-8727`) then
applies verbatim at unique granularity: replace the runtime `_vi_norm`
with the static `_uniq_idx_jnp` in the two FFI forwards and the
Jacobian callback, and contract `exact_tbm = d_uniq_probs_exact.T @
g_uniq`. `_fd_theta_bar` needs ZERO changes (it treats the whole
forward, gather included, as a black box).

## Scope

1. **Model level (`pmf_from_graph_joint_index`):**
   - Remove the `elif _baked:` static exclusion (`:8287-8293`); baked
     mode joins the `_jix_exact_enabled = True` arm (the OTHER
     exclusions — was_dph, weight-mode, theta_dim-override — keep
     firing first, unchanged).
   - **Probe index set = `union(_uniq_idx_np, all_terminal_np)`** (the
     F merge-review requirement): both operands are construction-known,
     so the probe is EXACT for baked mode — there are NO runtime index
     rows, hence no residual "index-dependent decline the probe cannot
     see" class at all. (The F probe's `[0]` member existed for the
     runtime-rows residual; for baked it is subsumed — stated, and the
     review checks the reasoning.) Probe length check
     `len(raw) == len(probe_union) * P`.
   - Backward baked branch: scatter-add + quotient at unique
     granularity as above. The `vertex_indices` bounds check is
     construction-time for baked (indices validated when
     `observed_indices` is mapped — re-assert cheaply at construction,
     not per-call). The committed-decline RAISE semantics and message
     are IDENTICAL to non-baked (same callback, same causes); the
     conditioning gate stays DEFAULT-ON for this path (the Batch-H
     opt-out was the daisy caller's, user-scoped; jpg start IPVs are
     clean vectors, and the H de-risk's boundary mapping showed the
     gate's realistic-theta declines need WIDE-dynamic-range IPVs —
     the joint-index fixtures probe fine at ones AND typical theta;
     the E de-risk experiment measures this on the real fixture before
     implementation).
   - The "free bonus" hoist (master §7): in baked mode the callback's
     `union_idx`/`obs_pos`/`all_pos` are static — hoisted to
     construction. NON-baked callback code path stays byte-identical.
   - Docstring: the baked-exclusion sentence in the `exact_grad`
     docstring (written at F) becomes stale — updated.
2. **svgd leaf-2 plumbing (`Graph.svgd(exact_grad: Optional[bool] =
   None)`):**
   - Forwarded at the leaf-2 call site (the `pmf_from_graph_joint_index`
     call, post-G.1 `__init__.py:6438-6443` region) only when not None
     (None = not forwarded, byte-identical default — the
     G.1/D.4 contract, incl. the forwarding-discrimination test).
   - **New rule R31** (`_check_R31_exact_grad_leaf_scope`), R29/R30
     discipline (any non-None rejected off-leaf): accepted =
     joint-prob-KIND graph (`graph_kind in ('joint_prob',
     'joint_stop_prob')` — jsp graphs route to the same leaf; the
     R29/R9 classifier-hole lesson applied on day one), no
     `epoch_starts` (message points at `exact_final_grad`), no
     exposure (unreachable anyway — R9 rejects exposure+no-epochs on
     both kinds; R31's presence-check is belt-and-braces and its
     message defers to R9's). Rejected: epochs; moments/rewards
     leaves. Weight-mode/was_dph pre-emption: NOT config-level for
     this leaf (unlike R30's daisy checks, the joint-index builder's
     own decline ladder is probe-and-log, not raise — an explicit
     True on e.g. a formula-mode jpg would probe-fail → whole-model FD
     + INFO log, which is the F contract, NOT silent inertness;
     stated in the R31 docstring and the svgd docstring).
     [REVIEW QUESTION: is probe-fail→FD+log acceptable for an
     EXPLICIT svgd-level True, or should R31 pre-empt weight-mode
     like R30 does? R30's precedent says pre-empt; the F contract
     says log-and-FD is the model-level meaning of exact_grad=True.
     The plan proposes: pre-empt `effective_weight_mode != 'linear'`
     at R31 for symmetry with R30 (same fields already exist), leave
     was_dph/theta_dim to the builder's logged declines.]
   - Ledger entry (`LEDGER_OPTION_ORDER`), `from_svgd_call` named
     param, `SvgdConfig` field.
   - `svgd.py` token (`_GRAPH_SVGD_ONLY_KWARGS` + `'exact_grad'`):
     same one-token shape the user approved for D.4 and G.1 — ASKED
     at fold time again (the file carries a standing do-not-modify
     rule; two precedents, but each approval was explicit).
   - Shipped-text updates (flagged): R30's no-epochs message clause
     "svgd plumbing for that leaf is not available" becomes FALSE →
     points at `exact_grad`; the `Graph.svgd` docstring paragraph
     (post-G.1) gains the new kwarg's parameter entry (blast-radius
     statement included — same committed-raise semantics; plus the
     probe-fail→FD+log nuance and the forward-mode P-scaling cost
     note from the F docstring, so svgd users see the same trade the
     model-level docstring states).
3. **Out of scope, stated:** leaves 3/4 (rewards — Batch A); the
   joint-index `exact_grad` DEFAULT stays False at both levels
   (user decision `04775b63`; svgd None = not forwarded); non-baked
   behavior byte-identical; `Graph.mcmc` untouched.

## E0 — de-risk experiment (pre-implementation)

`experiments/dr_batchE_baked_derisk.py` on the current install:
- (i) **Oracle check of the scatter-add derivation on real numbers**:
  build the baked model's forward pieces directly (pybind sojourn at
  uniq + normalization + gather), compute the proposed backward
  (scatter-add + quotient at uniq via `_sojourn_grad_theta_subset`),
  and compare against `jax.jacobian` of a dense-JAX replica AND
  against tight central FD of the baked forward. Target ≤1e-9.
- (ii) **Probe/gate reality on the real fixture**: at theta=ones AND
  typical theta (1e-4-scale), the C adjoint at
  `union(uniq, all_terminal)` COMPUTES on the jpg fixture (the H
  gate-decline finding was driven by wide-dynamic-range epoch
  handoff IPVs; jpg start IPVs are clean — measure, don't assume).
  If typical-theta declines: STOP, user decision (the F committed-
  raise contract would make baked exact_grad=True raise on first
  real call — same shape as the H gate finding).
- (iii) **Front-door smoke**: `Graph.svgd(obs)` (no kwarg) on the
  fixture reaches the baked leaf (spy on
  `phasic.ffi_wrappers.compute_sojourn_times_ffi`) — the D.3 lesson.
- GO/NO-GO: (i) ≤1e-9 AND (ii) computes at both thetas AND (iii)
  reaches the leaf.

## Implementation (after E0 GO)

- **I1** model-level baked exact branch + probe union + hoist +
  docstring; **I2** svgd_config (R31, field, param, ledger, R30
  message update) + `Graph.svgd` (kwarg, threading, forwarding,
  docstring) + svgd.py token (post-approval); **I3** tests
  (`tests/pytest/inference/test_svgd_exact_grad_leaf2_kwarg.py` for
  the svgd level, additions to
  `inference/test_exact_grad_joint_index.py`'s file? NO — new file
  `inference/test_exact_grad_joint_index_baked.py` for the model
  level, keeping the F file untouched per the fate table):
  1. Baked exact-vs-oracle parity (E0(i) productionized; benign +
     mixed-scale theta; ≥1e3 improvement floor vs FD).
  2. Baked exact-vs-FD parity at tolerance anchored to measured
     actuals at authoring time (the G.1 G4 discipline — measure
     FIRST, then set tolerance with ~100x headroom, comment the
     measured number truthfully).
  3. Trace-time spy: committed baked model traces ZERO FD ops under
     `vmap(jit(grad))`; ABSOLUTE delta pinned vs the FD model
     (measured at authoring; nonzero-floor guard).
  4. Duplicated observations: scatter-add correctness with heavy
     duplication (n_obs >> n_unique) — gradient equals the
     dedup-weighted expectation (oracle).
  5. Probe exactness: a committed baked model's first real call
     cannot hit an index-dependent decline (the probe set == the
     call set; assert via a spy that the probe and call unions are
     identical).
  6. Committed-decline raise legibility under `vmap(jit(grad))`
     (monkeypatched decline, F test-8(b) mechanism).
  7. fixed_mask × baked exact (fixed slots exactly 0.0).
  8. R31 rejections (epochs → names exact_final_grad; moments;
     rewards 1-D/2-D; + the accepted-leaf None-sweep incl. jsp kind);
     forwarding discrimination (explicit False arrives, None absent);
     ledger default/user; constructor-guard test (post-approval);
     front-door spy test (svgd-built baked model commits);
     weight-mode pre-empt cell per the review-resolved R31 question.
  9. Cross-install golden: svgd-built baked model gradient (no
     kwarg), dump pre-E / check post-E, bitwise
     (`experiments/dr_batchE_golden.py`).
- **Existing-test fate table:** `test_exact_grad_joint_index.py::
  test_baked_mode_declines` (the F fate table's keep-passing entry)
  **BREAKS BY DESIGN** — baked no longer declines; rewritten as
  `test_baked_mode_commits` (probe succeeds, INFO log gone, exact
  engaged) IN THE F FILE (one existing file touched, enumerated
  here). Everything else — the other 18 joint-index tests, the G.1
  svgd suite (19), daisy/epoch files, svgd-config files — must pass
  unchanged. Any other deviation = G1 failure.

## Gates

- **G0:** E0 GO + ledger @ `0c052cfe` + enumerated master delta.
- **G1:** I3 suites + the fate table holds exactly (one enumerated
  rewrite, nothing else).
- **G2:** joint-index/sojourn row (process map, as amended at G.1:
  `test_joint_index_callback.py`, `inference/test_optimized_joint_index.py`,
  `test_gate_daisy_chain_joint_probs.py`,
  `inference/test_exact_grad_joint_index.py`), the svgd-config row
  (all six files incl. G.1's), always-run
  `inference/test_fd_gradient_mixed_scale.py`; sources-on runs treat
  the 9 ledgered `test_jax_integration` failures per the ledger
  addendum.
- **G3:** chunked (`-rf` per the adopted amendment) vs ledger @
  `0c052cfe` (expect 1919 + new).
- **G4:** two diff refuters. **G5:** merge review (shipped-text
  statement: the F-file test rewrite, R30 message, F docstring baked
  sentence); ledger re-stamp; tracker; master §7/§9-leaf-2/§15 ticks;
  CLAUDE.md (the joint-index "deferred" list loses baked mode);
  memory; install rebuild.

## Risks

1. The R31 weight-mode pre-emption question (scope 2) — resolved by
   this plan review, not mid-batch.
2. The E0(ii) gate-decline measurement is the one genuine unknown
   (the H finding's transfer surface); NO-GO path defined.
3. The F-file single-test rewrite is the batch's only existing-test
   change — byte-diff discipline on the rest of that file.
4. Non-baked byte-identity: golden = the F suite's own
   `exact_grad=False`/non-baked tests + the cross-install golden.
