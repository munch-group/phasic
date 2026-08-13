# Batch G.1 plan — public `Graph.svgd(exact_final_grad=...)` (daisy leaf 1) + the D.3 fold

**Status: DRAFT v1, pending the mandated two-refuter adversarial plan
review.** Design of record: master plan §9 leaf 1 (gate "blocked until
Batch H ships something to plumb into" — SATISFIED, `ecd708fc`) + the
Batch H merge review item 6 (G leaf 1 unblocked) + the D.3 disposition
(user decision 2026-08-13, `b3-batchD3-plan.md`: D.3's user value ships
HERE; the R9 jsp hole is fixed HERE; D.3's review corrections transfer
HERE). Branch: `b3/batchG1-svgd-daisy-plumbing`, worktree
`../phasic-batchG1` (Python-only, worktree for the cross-install
golden). Baseline: ledger fourth stamp @ `ecd708fc` (1899/0/84/24,
empty; next full run 1900). Master HEAD is docs-only above the stamp
(close-out + D.3 records) — G0 enumerates the commits explicitly.
Findings: `b3-batchG1-findings.md`. Smoke experiment:
`experiments/dr_batchG1_frontdoor.py`.

## What G.1 delivers (plain language)

Batch H shipped exact final-epoch gradients for epoch models as an
INTERNAL kwarg (`_daisy_chain_svgd_model(exact_final_grad=False)`).
G.1 makes it public: `Graph.svgd(..., epoch_starts=[...],
exact_final_grad=True)`. Because the "final epoch" is the WHOLE model
when `epoch_starts=[0.0]`, this is also the delivery vehicle for
exposure users (the D.3 fold): `svgd(obs, exposure=..., epoch_starts=
[0.0], exact_final_grad=True)` = batched exposure (one OpenMP FFI call
per gradient evaluation, deduped by unique exposure value) + fully
exact gradients (every theta slot is a final-epoch slot).

## Scope (explicit; no silent fallbacks)

1. **Kwarg:** `exact_final_grad: Optional[bool] = None` on `Graph.svgd`.
   `None` (default) = not forwarded — the internal default (False)
   governs, byte-identical to today. `True`/`False` = forwarded to
   `_daisy_chain_svgd_model` at the leaf-1 call site
   (`__init__.py:6369-6382`, which currently omits the kwarg).
2. **New rule R30 (`_check_R30_exact_final_grad_leaf_scope`)** —
   explicit `exact_final_grad` (any non-None, R29 discipline) is
   REJECTED off the epoch leaf: no `epoch_starts` → message names the
   joint-index situation (its own `exact_grad` lives on
   `pmf_from_graph_joint_index`; baked svgd plumbing = Batch E's
   companion, per master §9 leaf 2); rewards/moments leaves → R29's
   domain. `final_read='stopprob'` + explicit True: the INTERNAL
   builder raises `ValueError` at construction (H's guard) — R30
   PRE-EMPTS it with a config-level `SvgdConfigError` naming
   `final_read='sojourn'` (validation-time beats construction-time; the
   H guard stays as belt-and-braces). Weight-mode formula/callback +
   explicit True: same pre-emption question — DECIDED IN REVIEW whether
   R30 checks weight mode (config carries graph handle?) or defers to
   H's construction guard with a documented two-stage failure surface.
3. **R9 jsp-hole fix (§16b item 9, user-sanctioned via the D.3 fold):**
   `_check_R9...` extends from `graph_kind == 'joint_prob'` to
   `graph_kind in ('joint_prob', 'joint_stop_prob')` + a pinning test.
   Consequence (stated, intended): jsp + exposure has NO route at all
   (R1 already forbids `epoch_starts` on jsp) — honest "unsupported"
   beats the current silent pass-through into an untested composition.
   This MODIFIES a shipped rule: the existing TestR9 case (joint_prob)
   keeps passing; only NEW rejections are added. Fate table lists it.
4. **Shipped-text updates (flagged, G4-reviewed):**
   (a) R29's message clause "with ``epoch_starts`` no exact epoch-model
   gradient exists yet" (`svgd_config.py:1121-1123` region) becomes
   FALSE when G.1 ships → updated to point at `exact_final_grad`. If
   any R29 test regexes that clause, updating the regex is the ONE
   sanctioned existing-test text change — enumerated in the fate table
   after a grep at implementation start.
   (b) The `Graph.svgd` docstring paragraph (`__init__.py:5744-5751`)
   gains the plumbing + R30 scope + the exposure route.
   (c) **Blast-radius statement (inherited F/H obligation), verbatim:**
   with `exact_final_grad=True`, a residual per-theta decline in the
   committed backward RAISES (H's user-decided semantics); under SVGD
   one declining particle halts the ENTIRE cloud mid-optimization.
5. **Low-level `SVGD(model=..., exact_final_grad=...)` constructor:**
   PROPOSED to mirror D.4's precedent (a helpful `TypeError` naming
   `Graph.svgd`) — this touches `svgd.py`, protected by the standing
   no-change-SVGD preference; D.4's identical one-token change was
   user-approved. FLAGGED: implement only with the same explicit
   user approval at the review/decision point; otherwise record
   "bare TypeError accepted" and skip.
6. **H's internal kwarg default stays False; no default flip** — the
   public kwarg is opt-in (None), consistent with the recorded
   joint-index default decision (`04775b63`) and H's docstring.

## G0-smoke (pre-implementation; the D.3 front-door lesson)

`experiments/dr_batchG1_frontdoor.py`, on the CURRENT install (no new
kwarg needed): verify TODAY's front door reaches the daisy builder in
the two target shapes — (a) `Graph.svgd(obs, exposure=...,
exposure_param_index=..., epoch_starts=[0.0])` and (b) multi-epoch
no-exposure — construct via `Graph.svgd`, run one gradient evaluation,
assert finite, and assert (via the H trace-count spy on
`phasic.ffi_wrappers.compute_daisy_chain_sojourn_ffi`, patched before
the call) that the models are daisy-built. Purpose: prove the
plumbing target is REACHABLE from the public API before writing any
code — the exact failure D.3's plan died of. GO/NO-GO: both shapes
construct and evaluate. NO-GO → back to the user with findings.

## Implementation

- **I1 — svgd_config:** `exact_final_grad` field on `SvgdConfig`
  (`svgd_config.py:292` region), `from_svgd_call` param + passthrough
  (`:513`, `:588-590`), `LEDGER_OPTION_ORDER` entry (`:122-129`),
  R30 per scope 2, R9 extension per scope 3, R29 message per scope 4a.
- **I2 — `Graph.svgd`:** signature + validator threading + ledger +
  forwarding at `:6369-6382` (only when not None); docstring per scope
  4b/4c.
- **I3 — tests** (`tests/pytest/inference/test_svgd_exact_final_grad_kwarg.py`,
  mirroring the D.4 suite's shape):
  1. R30 rejections: no-epochs joint-prob (baked), rewards 1-D AND 2-D,
     plain moments leaf, jsp graph — each `SvgdConfigError` with the
     documented message; `None` never raises anywhere (all leaves).
  2. R9 jsp pinning: jsp + exposure + no epochs REJECTED (closes §16b
     item 9); existing joint_prob rejection unchanged.
  3. Front-door single-epoch + exposure + `exact_final_grad=True`:
     constructs, one SVGD gradient step finite, trace-count spy shows
     ZERO FD perturbation ops (all slots exact at n_epochs==1) with a
     nonzero-floor guard on the spy.
  4. Front-door multi-epoch (`[0, 0.5]`) no-exposure, True: final-epoch
     FD ops gone, earlier-slot FD present (H test-8 arithmetic,
     derived factor + floor guard).
  5. Explicit False == no-kwarg default: BITWISE-equal gradients of the
     svgd-built model at fixed theta.
  6. Ledger/effective_options: status `'default'` (None) vs `'user'`
     (explicit) — D.4 precedent (`test_svgd_exact_moment_grad_kwarg.py:
     178-188` shape).
  7. Cross-install golden: `jax.grad` of the svgd-BUILT model (no
     kwarg) at fixed theta, dumped under the pre-G.1 install, bitwise
     under the branch install (H micro-gate (a2) template; a SINGLE
     model-gradient call, NOT an SVGD run — the D.3 review's MINOR 5).
  8. Blast radius: monkeypatched decline (H test-8(b) mechanism)
     through the FRONT-DOOR model raises legibly under
     `vmap(jit(grad))`.
  9. Tied + exposure + exact (multi-epoch, `tied=[(0,[0,1])]`):
     front-door gradient finite + parity vs the H oracle's tied
     column-sum rule — closes the H merge review's recorded
     "no exposure+tied combination test" gap.
  10. Constructor guard per scope 5's decision (or its recorded skip).
- **Existing-test fate table:** NO existing test changes state EXCEPT
  (enumerated): (i) any R29 test regexing the updated message clause
  (grep at implementation start; regex-only updates); (ii) TestR9 gains
  a case, existing cases untouched. Everything else — daisy/epoch
  files, joint-index files, svgd config/exposure/parity/assumptions
  files — must pass unchanged.

## Gates

- **G0:** smoke GO + ledger @ `ecd708fc` + enumerated docs-only master
  delta.
- **G1:** I3 suite + fate table holds exactly.
- **G2:** daisy/epoch row (`test_epoch_sojourn_finalread.py`,
  `inference/test_lrt_at.py`, `inference/test_epoch_model.py`),
  `test_gate_daisy_chain_joint_probs.py`,
  `inference/test_exact_grad_daisy_final.py` (H's suite — shared
  machinery), the svgd config/validation set (`test_svgd_config.py`,
  `inference/test_svgd_exact_moment_grad_kwarg.py`,
  `inference/test_svgd_exposure.py`, `inference/test_svgd_api_parity.py`,
  `test_svgd_assumptions.py`), always-run
  `inference/test_fd_gradient_mixed_scale.py`. (A dated process-doc
  amendment adding an "svgd config/validation" G2 row is proposed at
  G5 — the map predates the rule suites.)
- **G3:** chunked full suite vs ledger @ `ecd708fc` (expect 1900 + I3).
- **G4:** two diff refuters (config/wiring fidelity; tests/process).
- **G5:** merge review incl. shipped-text-edit statement (R9/R29
  messages); ledger re-stamp; tracker (G.1 merged; D.3 value
  delivered); master §9/§15 tick + §16b item 9 closure; CLAUDE.md;
  memory; install rebuild.

## Risks

1. R30's weight-mode/final_read pre-emption depth (scope 2) — the
   review decides how much the config layer can see vs. deferring to
   H's construction guards with a documented two-stage surface.
2. R29-message regex exposure — enumerated at implementation start,
   not discovered at G3.
3. The R9 extension makes jsp+exposure fully unsupported — intended
   and stated, but the review should confirm no shipped test or doc
   RELIES on the current silent pass-through.
4. Config-classifier ordering (jsp checked before indexer — the R29/R9
   hole class): R30 must classify jsp EXPLICITLY, not inherit the hole.
