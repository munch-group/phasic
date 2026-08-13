# Batch G.1 plan — public `Graph.svgd(exact_final_grad=...)` (daisy leaf 1) + the D.3 fold

**Status: v2, 2026-08-13 — the mandated two-refuter adversarial plan
review is DONE (both SOUND-WITH-CORRECTIONS; all findings folded below;
review record at the end). ONE user decision surfaced at fold time
(scope 5, the svgd.py one-token guard) — asked before implementation;
everything else is bound. Cleared for the G0 front-door smoke, then
implementation.** Design of record: master plan §9 leaf 1 (gate
SATISFIED by Batch H, `ecd708fc`) + the D.3 disposition (user decision
2026-08-13: D.3's value ships here; R9 jsp hole fixed here; D.3 review
corrections transferred here). Branch: `b3/batchG1-svgd-daisy-plumbing`,
worktree `../phasic-batchG1`. The G0 smoke runs IN-BATCH on this branch
(justified: it is a reachability probe on the CURRENT install adding
only an `experiments/` script — the F0 precedent; no `derisk/*` branch
needed for a no-code-change probe [process-review MAJOR 1b]). Baseline:
ledger fourth stamp @ `ecd708fc` (1899/0/84/24, empty; next full run
1900); master delta above the stamp verified docs-only by both
reviewers (6 commits, .md files only). Findings:
`b3-batchG1-findings.md`. Smoke: `experiments/dr_batchG1_frontdoor.py`.

## What G.1 delivers (plain language)

Batch H shipped exact final-epoch gradients for epoch models as an
INTERNAL kwarg. G.1 makes it public: `Graph.svgd(...,
epoch_starts=[...], exact_final_grad=True)`. With `epoch_starts=[0.0]`
the "final epoch" is the WHOLE model, so this is also the delivery
vehicle for exposure users (the D.3 fold): batched exposure (one fused
OpenMP FFI call per forward, deduped by unique exposure value) + fully
exact gradients. Cost honesty [design-review MINOR 8]: the exact
BACKWARD is a host-side sequential loop over unique exposure values per
particle (H's measured ~7.4%-of-FD figure is the model-level number);
the docstring carries this caveat rather than overselling "one call".

## Scope (all bound post-review; no silent fallbacks)

1. **Kwarg:** `exact_final_grad: Optional[bool] = None` on `Graph.svgd`.
   `None` = not forwarded (internal default False governs —
   byte-identical). `True`/`False` = forwarded at the daisy call site
   (`__init__.py:6369-6382`). Forwarding of explicit False is
   CONTRACTUAL and tested by a call-kwargs spy (see I3 test 11) — a
   `if exact_final_grad:`-style bug that forwards only True would be
   invisible to bitwise tests [process-review MAJOR 4].
2. **New rule R30** — explicit `exact_final_grad` (any non-None, R29
   discipline) rejected off the accepted set. Accepted set =
   `has_epoch_starts` (which, via R1, already implies a continuous
   joint-prob graph — verified equal to the daisy call site's reach-set
   [design-review MINOR 6]). Pre-emption depth DECIDED [design-review
   MAJOR 3, option (b), chosen for a uniform validation-time failure
   surface]: `from_svgd_call` gains (i) `final_read: str = 'sojourn'`
   param + config field + call-site pass-through (`__init__.py:
   6074-6094`) — WITHOUT this the stopprob check has NO DATA PATH and
   dies silently in `**_unused` [design-review MAJOR 1]; (ii) an
   `effective_weight_mode` field computed as the COMBINED form
   (`'formula'` if the `weight_formula` kwarg is given else
   `'callback'` if the `callback` kwarg is given else
   `graph._weight_mode`) — the combined form is REQUIRED because
   validation runs before the per-call overrides flip the graph's mode
   [design-review MAJOR 3]. R30 then rejects, with epochs + explicit
   True: `final_read='stopprob'`; `effective_weight_mode != 'linear'`.
   H's construction guards remain belt-and-braces. **Rule-order
   shadowing, stated:** R1/R9/R16/R21 precede R30 in `_RULES` and fire
   first in several explicit-value cells (e.g. standard graph + epochs
   → R1); their messages are reasonable, the shadowing is accepted, and
   I3 test 1 is written ONLY against cells where R30 actually speaks
   [design-review MINOR 6]. Message texts are BATCH-FREE (no "Batch E"
   labels — the staleness class G.1 is itself fixing) and name the
   `epoch_starts=[0.0]` route where relevant [process-review MINOR 7].
3. **R9 jsp-hole fix (§16b item 9, user-sanctioned via the D.3 fold),
   kind-AWARE:** extend the kind check to `('joint_prob',
   'joint_stop_prob')` AND fork the message [both reviews MAJOR]: the
   jsp arm must NOT prescribe `epoch_starts` (R1 forbids it on jsp —
   the trap §16b item 9 itself records); jsp text = "unsupported — use
   the source joint-prob graph with epoch_starts=[0.0, ...]"; the
   joint_prob arm's text is UNCHANGED so the existing regex
   (`test_svgd_config.py:190`, `"vanilla joint-prob"`) stays green.
   Pinning test regexes the jsp-specific text.
4. **Shipped-text updates (flagged; G5 shipped-text statement):**
   (a) The stale clause exists in TWO places [process-review MINOR 8]:
   R29's message tail (`svgd_config.py:1120-1124` region — keep the
   FIRST sentence so the one regexing test,
   `test_svgd_exact_moment_grad_kwarg.py:117-122`, needs ZERO changes
   [both reviews, fate table pre-filled]) and the svgd docstring at
   `__init__.py:5746`. Both point at `exact_final_grad` after G.1.
   (b) The svgd docstring gains: a proper PARAMETER entry for the new
   kwarg; the R30 scope; the exposure route; **final-epoch-only
   exactness stated explicitly** (earlier epochs stay FD — the
   Deferred-2 boundary made user-visible); the cost caveat (§above);
   and the **blast-radius statement** (inherited F/H obligation): a
   residual per-theta decline in the committed backward RAISES — one
   declining particle halts the ENTIRE cloud mid-optimization.
   (c) The builder's "INTERNAL for now" markers (`__init__.py:
   4332-4333`, `:4571-4573`) become stale at ship — updated
   [design-review MINOR 7].
5. **Low-level `SVGD(model=..., exact_final_grad=...)` guard:** the
   seam exists (`_GRAPH_SVGD_ONLY_KWARGS`, `svgd.py:4531-4535`, already
   holding `'exact_moment_grad'` from the user-approved D.4 change;
   helpful-TypeError helper `:4538-4562`). Adding `'exact_final_grad'`
   is ONE token in the protected svgd.py. **USER DECISION, asked at
   fold time (not mid-batch) [process-review MAJOR 5]** — I3 test 10
   binds to the answer. Without the token the constructor still raises
   loudly via the helper's generic branch (downside is message
   quality, not silence). **DECIDED 2026-08-13 (user, fold-time): YES
   — add the token. Test 10 binds to the D.4-shape guard test.**
6. **Out of scope, stated:** `Graph.epoch_model` (`__init__.py:
   5360-5475`) builds the same daisy model without the kwarg —
   FreeEpochModel/LRT users cannot opt in via G.1; deliberate
   (follow-up candidate, not ledgered — it is derivable from this plan)
   [design-review MINOR 9]. Leaves 2/3/4 untouched (E/A gates). H's
   internal default stays False; the public kwarg is opt-in None.

## G0-smoke (pre-implementation; the D.3 front-door lesson)

`experiments/dr_batchG1_frontdoor.py` on the CURRENT install (no new
kwarg): (a) `Graph.svgd(obs, exposure=..., exposure_param_index=...,
epoch_starts=[0.0])` and (b) multi-epoch no-exposure — both must
construct THROUGH `Graph.svgd`, run one gradient evaluation, finite,
with the daisy spy proving daisy-built (patch
`phasic.ffi_wrappers.compute_daisy_chain_sojourn_ffi` before the call —
binds because the builder imports at construction, verified
`:4742-4745`/`:4996-4999`). Reviewer-verified feasibility:
`epoch_starts=[0.0]` passes builder validation and t_eval resolution;
scalar/per-obs exposure handled at `:6356-6368`. GO/NO-GO: both shapes
work. NO-GO → user.

## Implementation

- **I1 — svgd_config:** `exact_final_grad` field (`:292` region) +
  `from_svgd_call` param + passthrough (`:513`, `:588-590`) +
  `LEDGER_OPTION_ORDER` entry (`svgd_config.py:124-131`) + **the
  `final_read` param/field/pass-through and `effective_weight_mode`
  field (scope 2 — named explicitly here so they cannot fall into
  `**_unused` silently [design-review MAJOR 1])** + R30 + the
  kind-aware R9 fix + the R29-message tail edit.
- **I2 — `Graph.svgd`:** signature, validator threading (incl.
  `final_read=final_read` at `:6074-6094`), ledger, forwarding at
  `:6369-6382` (only when not None), docstring per scope 4.
- **I3 — tests** (`tests/pytest/inference/test_svgd_exact_final_grad_kwarg.py`):
  1. R30 rejections — ONLY cells where R30 speaks [design-review MINOR
     6]: no-epochs joint-prob (baked), rewards 1-D AND 2-D, plain
     moments leaf; each `SvgdConfigError`, documented message.
  2. R9: jsp + exposure + no epochs REJECTED with the jsp-specific
     message (regex-pinned); existing joint_prob rejection unchanged.
  3. Front-door single-epoch + exposure + True: constructs, one SVGD
     gradient step finite, spy shows ZERO FD ops (nonzero-floor guard),
     AND a coarse value check: True-vs-False gradients agree to a
     loose rtol (~1e-3 — FD is inaccurate, not wrong; catches gross
     mis-plumbing) [process-review MINOR 13].
  4. Front-door multi-epoch no-exposure, True: final-slot FD gone,
     earlier-slot FD present (H trace-count arithmetic + floor).
  5. Explicit False == no-kwarg: BITWISE-equal model gradients.
  6. Ledger/effective_options: `'default'` (None) vs `'user'`
     (explicit) — mechanism verified sound by review (auto-record via
     the introspection loop; `record_user_or_default`).
  7. Cross-install golden: single svgd-BUILT model-gradient call (no
     kwarg), dumped under the pre-G.1 install, bitwise under the
     branch install.
  8. Blast radius: forced decline through the FRONT-DOOR model raises
     legibly under `vmap(jit(grad))` (H test-8(b) mechanism).
  9. Tied + exposure + exact (multi-epoch, `tied=[(0,[0,1])]`):
     front-door parity vs the H oracle's tied column-sum rule,
     tolerance ANCHORED TO MEASURED ACTUALS at authoring time (the H
     G4 discipline; target ~1e-6 with 100×+ headroom) [process-review
     MINOR 10].
  10. Constructor guard per the scope-5 user answer (or its recorded
      skip).
  11. Forwarding-discrimination [process-review MAJOR 4]: wrap-spy on
      `Graph._daisy_chain_svgd_model` asserting explicit False arrives
      as `exact_final_grad=False` in the call kwargs (and None arrives
      as ABSENT).
  12. Pre-emption branches [both reviews]: epochs + `final_read=
      'stopprob'` + True → `SvgdConfigError` (NOT the H `ValueError`);
      epochs + `weight_formula=...` + True → `SvgdConfigError`
      (the combined-form snapshot check).
  13. `None` never raises: PARAMETRIZED sweep over {moments,
      rewards-1D, rewards-2D, joint-prob no-epochs, jsp, epoch leaf}
      [process-review MINOR 12].
  14. Front-door `fixed=[(idx, val)]` + single-epoch + exposure + True:
      pinned slot's gradient is exactly 0.0 (the fixed-wins precedence
      — untested even in H's suite at svgd level) and the rest finite;
      plus a scalar-exposure case (broadcast at `:6363-6368`,
      K_unique=1) [both reviews; D.3 fold items restored].
- **Existing-test fate table (pre-filled by review; both reviewers):**
  ZERO existing-test changes expected. The one R29-message regex
  (`test_svgd_exact_moment_grad_kwarg.py:117-122`) survives because the
  first sentence is retained; the R9 regex (`test_svgd_config.py:190`)
  survives because the joint_prob arm is unchanged; no closed-list
  `LEDGER_OPTION_ORDER` assertions exist. Any deviation = G1 failure
  and a STOP (it means the message edits went beyond the bound scope).

## Gates

- **G0:** smoke GO + ledger @ `ecd708fc` + enumerated docs-only delta.
- **G1:** I3 suite (14 tests) + fate table holds exactly.
- **G2:** daisy/epoch row (`test_epoch_sojourn_finalread.py`,
  `inference/test_lrt_at.py`, `inference/test_epoch_model.py`),
  `test_gate_daisy_chain_joint_probs.py`,
  `inference/test_exact_grad_daisy_final.py`, the svgd set
  (`test_svgd_config.py`, `inference/test_svgd_exact_moment_grad_kwarg.py`,
  `inference/test_svgd_exposure.py`, `inference/test_svgd_api_parity.py`,
  `test_svgd_assumptions.py`), always-run
  `inference/test_fd_gradient_mixed_scale.py`. (Process-doc G2-row
  amendment proposed at G5.)
- **G3:** chunked full suite vs ledger @ `ecd708fc` (expect 1900 + I3).
- **G4:** two diff refuters. **G5:** merge review incl. the
  shipped-text-edit statement (R9 fork, R29 tail, docstring, builder
  markers); ledger re-stamp; tracker (status vocabulary fixed to
  process §2 terms [process-review MINOR 11]); master §9/§15 tick +
  §16b item 9 closure; CLAUDE.md; memory; install rebuild.

## Risks

1. The `final_read`/weight-mode threading MUST go through named
   `from_svgd_call` parameters — anything reaching `**_unused` is
   silently dead; I3 test 12 is the canary.
2. The R9 fork must leave the joint_prob arm byte-identical (regex
   pinned); the jsp arm is NEW text.
3. Front-door spy counts may include construction-time probes
   (`daisy_chain_probe_theta` / prior matching) — test authoring
   measures the baseline count empirically first (H discipline).

## Adversarial plan-review record (2026-08-13; v1 → v2)

Design/wiring refuter: SOUND-WITH-CORRECTIONS — MAJOR 1 (final_read
has NO data path into validation; `**_unused` silent-death footgun;
test missing) → scope 2 / I1 / test 12; MAJOR 2 (R9 jsp arm's message
prescribes what R1 forbids; fork the message, pin the jsp text, keep
joint_prob arm regex-stable) → scope 3; MAJOR 3 (weight-mode question
RESOLVED: combined-form snapshot required because validation precedes
the per-call mode overrides; option (b) chosen) → scope 2 / test 12;
MINOR 4 (fate table pre-filled: zero regex changes needed; ledger
mechanism verified; anchor `:124-131`) → fate table; MINOR 5 (fixed=/
scalar-exposure dropped in transfer; fixed-wins untested even in H) →
test 14; MINOR 6 (rule-order shadowing stated; accept-set == reach-set
verified; from_model_call no-op verified) → scope 2; MINOR 7 (builder
INTERNAL markers) → scope 4c; MINOR 8 (cost-sentence honesty) →
delivery §; MINOR 9 (epoch_model out of scope, stated) → scope 6.
Verified: all anchors; jsp routing + no reliance on the pass-through;
smoke feasibility incl. [0.0] validation + t_eval; n_epochs==1 empty-FD
claim; tied+exposure legality; blast-radius mechanism; D.4 guard
genuinely in svgd.py. Process/tests refuter: SOUND-WITH-CORRECTIONS —
MAJOR 1 (two D.3 corrections dropped; smoke branch naming) → test 14 +
header justification; MAJOR 2 (R9 misleading message) → scope 3;
MAJOR 3 (untested rule branches) → test 12; MAJOR 4 (forwarding-
discrimination probe) → test 11 + scope 1; MAJOR 5 (svgd.py guard =
fold-time user question) → scope 5, ASKED; MAJOR 6 (weight-mode
bindable now) → scope 2; MINORs 7-13 (batch-free messages; both stale
sites + param entry + final-epoch-only statement; regex grep carried;
test-9 tolerance; tracker vocabulary; None sweep; test-3 value check)
→ folded. Verified: G0/ledger/tracker claims; fold-carried items;
H obligations; mandate fidelity; authorization chains; anchors; spy
mechanics; naming; R30 number free.
