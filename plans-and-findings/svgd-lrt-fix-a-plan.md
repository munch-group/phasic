# Plan — Fix A: make the canonical `likelihood_ratio_test` accept tied-vs-free (and any provably-equivalent) SVGD fits

## Status of the branch stack (context)

Off `master` (`2f60ea37`):
- `chore/disable-pmf-from-graph-parameterized` (`f7ea75ce`) — shelved the broken Builder-based JAX pmf path. *Unmerged.*
- `fix/from-serialized-find-vertex` (`abb91b48`) — **regression fix** found while de-risking this plan: `from_serialized` rebuilt vertices with `create_vertex` (no AVL insert), breaking `find_vertex` on any cache-reloaded graph and crashing `joint_prob_graph` on a warm `~/.phasic_cache/graphs`. Fixed + adversarially reviewed (verdict CORRECT). *Unmerged; this plan branches off it.*
- `plan/svgd-lrt-canonical` — this plan (+ eventual Batch-1 implementation).

## Goal

The deferred `deferred-svgd-lr-bug` is already satisfied against its own acceptance criteria by fixes **C** (`likelihood_ratio_test_at`) and **D** (SVGD kwarg guard + docstring), committed `2524bc34`. Fix A is the "most correct" follow-up you selected: make the **canonical** discoverable function

```python
res = phasic.likelihood_ratio_test(full, nested)
```

work directly for a **tied-vs-free epoch** pair (and generally for any pair whose two model callables are provably the *same likelihood function*), instead of forcing users to know about the `_at` variant.

Non-goal (see "Rejected: A-move"): moving tying out of the baked model into an SVGD-level θ-transform. De-risk proved it unnecessary.

## De-risk findings (evidence this plan is built on)

Ran a live tied-vs-free epoch pair (the `test_lrt_at.py` coalescent fixture: `epoch_starts=[0,0.5]`, `fixed=[(1,mu)]`, `nested = tied=[(0,[0,1])]`).

1. **The tied fit is already properly nested in the free fit** via the existing `fixed_mask`/`degrees_of_freedom` machinery — NO new nesting math needed:
   - `full` (free): `theta_dim=4`, `dof=2`, `fixed_mask=[0,1,0,1]`, `_tying_info={}`.
   - `nested` (tied): `theta_dim=4`, `dof=1`, `fixed_mask=[0,1,1,1]` (slave slot 2 folded into `fixed`), `_tying_info={2:0}`.
   - `nested.fixed_mask ⊇ full.fixed_mask` (superset holds); shared mu-slot values match; `df = 2−1 = 1`.
   - `_check_nested_fixed(full, nested, strict=True)` **passes**.
2. **The only blocker is the identity guard.** `likelihood_ratio_test(full, nested, strict=False)` **already** returns the correct result — `df=1, statistic=0.0192, p=0.890, ll_full=−8.585, ll_nested=−8.595` — **matching `likelihood_ratio_test_at` to machine precision**. The sole difference is a spurious `RuntimeWarning` from the `full.model is nested.model` guard (`model_selection.py:414`).
3. **SVGD is agnostic to tying during optimization** (subagent-confirmed): the model bakes `_apply_tying` in *after* `param_transform`; SVGD only reads `_tying_info` post-hoc to re-tie result surfaces. Moving tying into SVGD would touch the forward path *and* every post-hoc re-tie site — large and risky.

Conclusion: fix A's LRT goal is a **small, additive `model_selection`-only change**, reusing fix C's already-reviewed runtime consistency check. **No SVGD change** (honours the standing "don't modify SVGD" preference).

## Approach (batched, test-gated)

> Rebuild reality: `model_selection.py` is pure Python but the install is a *copy* — `pixi run install-dev` after edits. Set `PHASIC_SOURCE_DIR=/Users/kmt/phasic`. Never `git add -A` / `git stash` / touch `.ipynb`. Keep the graph cache in mind (the regression fix makes warm-cache runs safe).

### Batch 1 (core, recommended) — recognise provably-equivalent model pairs in `likelihood_ratio_test`

**Mechanism (revised per plan review — SOUND-WITH-GAPS).** Branch in `likelihood_ratio_test` on model identity; on the different-callable branch keep BOTH the structural check and the runtime check, and delegate the statistic/df to `likelihood_ratio_test_at` (which is refine-free and consistency-validated):

- `src/phasic/model_selection.py`:
  1. Keep the ordered pre-checks unchanged — `_check_fitted` (`:397`), `theta_dim` (`:400`), `n_observations` (`:406`).
  2. Replace the hard identity block (`:414-422`) with a branch:
     ```python
     if full.model is nested.model:
         _check_nested_fixed(full, nested, strict)          # existing structural path
         # ... existing df / statistic (refine-capable) / p_value below, unchanged ...
     else:
         # different callables (e.g. tied-vs-free epochs): prove nesting BOTH ways.
         if refine:
             raise ValueError(                              # AMENDMENT 3
                 "refine=True is not supported when the two fits use different "
                 "model callables (refining a tied MAP is unreliable); omit refine.")
         _check_nested_fixed(full, nested, strict)          # AMENDMENT 2: structural
         return likelihood_ratio_test_at(                   # runtime consistency + refine-free
             full, nested, strict=strict, consistency_atol=consistency_atol)
     ```
  3. Add `consistency_atol: float = 1e-4` to `likelihood_ratio_test`'s signature (used only on the different-callable branch; default matches `_at`).
  - Rationale for the amendments (all from the plan review):
    - **AMENDMENT 2 — keep `_check_nested_fixed` on the different-callable branch.** Delegating to `_at` alone proves nesting only via a single-point consistency check + dof-diff `df`, which would WRONGLY ACCEPT a non-superset pair (e.g. `full` fixes `{0}`, `nested` fixes `{1,2}`) whose likelihoods coincide at the nested MAP — with a meaningless `df`. The de-risk proved `_check_nested_fixed(full, nested, strict=True)` PASSES for the tied-vs-free case, so keeping it costs nothing and closes the hole (the same-model branch already had this protection).
    - **AMENDMENT 3 — reject `refine=True` on the different-callable branch.** `likelihood_ratio_test` computes `ll_nested = nested.log_likelihood(refine=refine)` (`:436`); `_at` deliberately omits `refine` because refining a tied MAP is unreliable (`_at` docstring `:515-517`). Delegating to `_at` makes the different-callable statistic refine-free and computed at the SAME MAP the consistency check validated. Reject `refine=True` explicitly rather than silently ignore.
  - Docstring: both fits must be the SAME likelihood function — either the *same* `model` object (fast structural path) OR provably equivalent (verified at runtime); point tied-vs-free users here rather than only at `_at`.

**Policy change + test updates (must surface).** This changes `likelihood_ratio_test`'s contract for *different-callable* pairs from "always reject on identity" to "accept iff provably equivalent AND fixed-mask-nested." **THREE** existing tests pin the OLD policy and must be updated:
- `tests/pytest/inference/test_lrt_at.py::test_lrt_at_tied_vs_free_epoch` (`:159-162`) — **[plan-review MISS, now included]** currently asserts `pytest.raises(ValueError): likelihood_ratio_test(full, nested)` (identity guard). Under Batch 1 this pair is ACCEPTED, so the raise no longer fires. Update lines 160-161 to assert **acceptance** (`df=1`, statistic ≥ 0, == `_at` to machine precision). This becomes the headline canonical-accept test. (Without this edit `test_lrt_at.py` is 5/6, not 6/6 — correcting the earlier draft's false claim.)
- `tests/pytest/inference/test_model_selection.py::test_lrt_strict_model_identity_mismatch` (`:380`) — builds the model twice (equivalent) and asserts a `"SAME model callable"` raise. Update to assert it now **accepts** (correct `df`, statistic ≥ 0).
- `::test_lrt_strict_false_warns_instead` (`:394`) — asserts a `RuntimeWarning` for separate builds under `strict=False`. Update: equivalent separate builds no longer warn.
- The other five raise-tests are unaffected (verified by the plan review): `test_lrt_theta_dim_mismatch_raises`/`test_lrt_n_mismatch_raises` fire on the pre-branch `:400`/`:406` guards; `test_lrt_identical_fixed_mask_raises`/`test_lrt_nested_not_superset_raises`/`test_lrt_shared_fixed_value_mismatch_raises` are same-model `_check_nested_fixed` cases. `test_lrt_at_agrees_with_lrt_on_shared_model` and `test_repr_aic_bic_lrt` are same-model → green. No internal callers: `compare()` uses only `aic`/`bic`; the only `src` mention of the canonical function is a docstring "See Also" (`svgd.py:7085`).

**New tests (pin the new contract):**
- **[headline]** Canonical `likelihood_ratio_test(full, nested)` on the tied-vs-free epoch pair (strict, default) returns `df=1` and equals `likelihood_ratio_test_at` to machine precision — via the `test_lrt_at_tied_vs_free_epoch` update above.
- Canonical `likelihood_ratio_test` on a genuinely NON-equivalent different-callable pair (different data, per `test_lrt_at_rejects_inconsistent_pair`) still **raises** in strict / warns in non-strict — the consistency gate is real.
- Canonical `likelihood_ratio_test` on a non-superset different-callable pair (AMENDMENT-2 hole) is **rejected** by `_check_nested_fixed`.
- `likelihood_ratio_test(different_callable_pair, refine=True)` raises the AMENDMENT-3 error.

**Gate:** `PHASIC_SOURCE_DIR=/Users/kmt/phasic pixi run pytest tests/pytest/inference/test_model_selection.py tests/pytest/inference/test_lrt_at.py -q` green (with the THREE updates); the tied-vs-free canonical result matches `_at`.

### Batch 2 (optional, separable) — public reusable free daisy-chain epoch model builder

Today there is **no public** way to build a daisy-chain (epoch) model object; `Graph.svgd(epoch_starts=...)` builds it internally via the private `_daisy_chain_svgd_model` and fits immediately. The plain path, by contrast, has public `Graph.pmf_and_moments_from_graph` that returns a reusable model. This batch closes that asymmetry:

- Add a public builder (name TBD, e.g. `Graph.pmf_from_graph_joint_index_epochs(...)` or `Graph.epoch_model(...)`) that returns the **free** (untied) daisy-chain model callable + `theta_dim` + broadcast `prior` + the metadata tags (`_handles_exposure_internally`, `_handles_particle_vmap`, `_precondition_output`, `_tying_info={}`), by wrapping `_daisy_chain_svgd_model(user_tied=None, ...)` plus the `Graph.svgd` preamble it needs (observed-index translation, `t_eval` resolution incl. `'auto'`, exposure array). Reuse, don't duplicate, that preamble.
- Value: build once → reuse for multiple fits and for `SVGD.log_likelihood(theta=…)`; makes the epoch API symmetric with the plain path.
- **Not required for the LRT goal** (Batch 1 already makes canonical LRT work for tied-vs-free fit via `Graph.svgd`). Recommend as a follow-up unless you want it now; it is medium-sized (mirrors a non-trivial preamble) and deserves its own de-risk + review.

### Rejected — A-move ("move tying into an SVGD-level θ-transform")

The doc's literal A. **Rejected** because: (a) de-risk shows the LRT goal is fully met without it; (b) it is the largest/riskiest change — it must apply the tying scatter in the SVGD forward *and* preserve every post-hoc `_tying_info` re-tie site (`optimize`, `get_results`, `map_estimate_from_particles`, `summary`) and the FD-VJP slave→master gradient routing; (c) it modifies SVGD, against the standing preference. Documented here so the decision is explicit and revisitable.

## Verification (end-to-end)

- Batch 1: the updated `test_lrt_at_tied_vs_free_epoch` goes fail→pass (canonical now accepts, `df=1`, == `_at` to machine precision); non-equivalent pair still rejected; non-superset pair rejected by `_check_nested_fixed`; `refine=True` on a different-callable pair raises; the THREE updated policy tests + all unaffected raise-tests green.
- No SVGD, FFI, or graph-construction code touched → no rebuild-sensitive numerical surface beyond `model_selection` (still needs `install-dev` — copy install).
- Adversarial review of the Batch-1 diff sized to its (small) complexity: refute that (i) the different-callable branch can accept a non-equivalent OR non-superset pair, (ii) delegating to `_at` changed any result, (iii) any same-model raise-test regressed, (iv) the `refine` rejection and `consistency_atol` default are appropriate.

## Risks / notes

- **Policy change** to a public function's contract (identity → provable-equivalence + fixed-mask-nested). Deliberate and the point of fix A; flagged for approval; **THREE** tests updated (incl. the plan-review-caught `test_lrt_at_tied_vs_free_epoch`).
- The different-callable branch relies on `map_estimate_from_particles()` re-tying slaves in constrained space (F-010) so the nested MAP is well-formed — already in place and tested.
- `consistency_atol=1e-4` inherited from `_at`; the tied-vs-free pair matched to machine precision (gap = 0.0), far inside tolerance.
- Keep `likelihood_ratio_test_at` as a public function (back-compat); after Batch 1 the canonical function delegates to it on the different-callable branch. Removing it is out of scope.
- **Docs (AMENDMENT 4):** `docs/pages/tutorial/model-selection.ipynb` lives in the `pixi run test` path (`pyproject.toml:207` converts notebooks→scripts→pytest) and LRTs a tied pair. It is ALREADY pre-broken by fix D at an earlier cell (`SVGD(model=…, epoch_starts=…, tied=…)` → `TypeError` from the fix-D kwarg guard, `svgd.py:4441`), so Batch 1 does not regress it. Updating that notebook to the supported pattern (`Graph.svgd(epoch_starts=…, tied=…)` + canonical `likelihood_ratio_test`) is a doc follow-up, out of Batch 1's core scope and requiring care given the notebook-safety constraints (do not clobber the uncommitted notebook conversion artifacts).
