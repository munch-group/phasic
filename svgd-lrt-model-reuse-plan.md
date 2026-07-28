# Plan — LRT / model reuse for epoch (daisy-chain) joint-prob SVGD (deferred issue #2)

Source analysis: `deferred-svgd-lr-bug.md`. Branch: `fix/svgd-robustness-and-lrt`.

## Problem (grounded at HEAD)

A nested likelihood-ratio test across a tied-vs-free **epoch daisy-chain
joint-prob** pair is impossible today because:
- `likelihood_ratio_test(full, nested)` requires `full.model is nested.model`
  (identity guard, `model_selection.py:413`) and expresses nesting via `fixed`.
- Tying is baked *into* the model callable (`_daisy_chain_svgd_model`,
  `__init__.py:5242`; `_apply_tying` every forward call), so a tied model and a
  free model are **different callables** → the identity guard can never hold, and
  "same model + extra `fixed`" cannot encode "tied vs free" (different layers).
- The direct `SVGD(model=…)` path rejects `epoch_starts`/`tied` with a **bare
  `TypeError`** (`SVGD.__init__`, `svgd.py:4801`; epoch/tying options are
  pre-model-construction concerns consumed by `Graph.svgd`).

`SVGD.log_likelihood(theta=…)` (`svgd.py:6961`) already evaluates the model at a
supplied **constrained** `(theta_dim,)` vector without re-applying
`param_transform` — the exact primitive needed.

## Scope decision

The doc's option **A** (refactor tying out of the baked model) is the "most
correct" but a large, high-regression-risk change — especially right after the A1
tying-export fix. Implement the pragmatic, lower-risk **C + D**, which meets every
§7 acceptance criterion:

- **C** — a first-class one-model-two-theta LRT that evaluates BOTH fits on ONE
  model's likelihood, so the `full.model is nested.model` obstruction disappears.
- **D** — actionable `SVGD.__init__` error for pre-construction kwargs + docstring
  scoping.

**Key enabler (must de-risk):** after the A1 fix, a tied fit's *constrained* MAP
already has the tied (slave) columns equal to their masters — i.e. the coalescent
rate is equal across epochs — so it **is** a valid free-layout `theta_nested`. The
free model evaluated there returns the constrained (nested) likelihood.

## Batch 1 — de-risk the enabler (no code)

Build a small tied-vs-free epoch joint-prob pair (2 epochs). Verify, with running
code:
1. `free.log_likelihood(theta=v)` accepts a flat `n_epochs×param_length` vector and
   evaluates the daisy-chain likelihood (open question 1 in the doc).
2. `nested.get_results()['theta_mean']` / `map_estimate_from_particles()` (the tied
   constrained MAP) has slave≡master (coal equal across epochs) and, fed to
   `free.log_likelihood(theta=…)`, gives `LL_nested ≤ LL_full` (no inversion) —
   i.e. the tied point really is a constrained sub-case of the free model.
3. `full.degrees_of_freedom - nested.degrees_of_freedom` is the correct df (>0).

If any fails, revisit scope before writing C.

## Batch 2 — fix C: `likelihood_ratio_test_at`

Add to `src/phasic/model_selection.py`:

```
likelihood_ratio_test_at(full, nested, *, strict=True) -> LikelihoodRatioResult
```

- Validate: both fitted; `full.theta_dim == nested.theta_dim`; same
  `n_observations`; `full.degrees_of_freedom > nested.degrees_of_freedom`.
- `LL_full    = full.log_likelihood(theta=<full MAP>)`   (full model at its MAP)
- `LL_nested  = full.log_likelihood(theta=<nested MAP>)` (full model at the tied
  point — the crux: nested's constrained MAP evaluated on the FULL likelihood)
- `df = k_full - k_nested`; `stat = max(0, 2*(LL_full - LL_nested))`;
  reject / warn on LL inversion (`strict`); `p = chi2.sf(stat, df)`.
- Return the SAME result object/shape as `likelihood_ratio_test` (reuse its dataclass
  and fields: statistic, df, p_value, LL_full, LL_nested, k_full, k_nested).
- NOTE loudly that this deliberately does NOT require `full.model is nested.model`
  — it evaluates BOTH thetas on `full.model`, which is what makes tied-vs-free work.

**Gate 2:** a tied-vs-free epoch pair → correct df, `stat ≥ 0`, no inversion,
`p ∈ [0,1]`; the statistic matches a hand-assembled
`2*(full.log_likelihood(theta=full_map) - full.log_likelihood(theta=nested_map))`;
and, as a consistency check on an ORDINARY (non-epoch) `fixed`-nested pair built on
one shared model, `likelihood_ratio_test_at` agrees with the existing
`likelihood_ratio_test` to numerical tolerance.

## Batch 3 — fix D: ergonomics

- `SVGD.__init__`: capture unexpected pre-construction kwargs (`epoch_starts`,
  `tied`, `joint_index`, `discrete`, `daisy_chain_t_eval`, `exposure_*` if
  Graph.svgd-only) and raise a **helpful** `TypeError` naming them and pointing to
  `Graph.svgd(...)` / `theta_dim=n_epochs*param_length` — instead of the bare
  "unexpected keyword argument". Preserve the current signature for all real params.
- Scope the `tied`/`epoch_starts` example blocks in the `SVGD` class docstring so
  they clearly reference `Graph.svgd` and don't read as direct-path params.

**Gate 3:** `SVGD(model, data, epoch_starts=[...])` raises a `TypeError` whose
message names `epoch_starts` and `Graph.svgd`; real-kwarg construction unaffected;
the previously-passing SVGD construction tests still pass.

## Adversarial review

Reviewer told to REFUTE: (a) is `LL_nested` really the constrained likelihood on
the FULL model (not the tied model's own baked likelihood)? construct a case where
the tied MAP is NOT slave≡master and show the guard/df is wrong; (b) LL-inversion
and df edge cases (df=0, k_nested≥k_full); (c) does `likelihood_ratio_test_at`
agree with `likelihood_ratio_test` on an ordinary shared-model `fixed`-nested pair?
(d) does the new `SVGD.__init__` error break any legitimate kwarg? (e) any path
where `full.log_likelihood(theta=nested_map)` silently re-applies a transform or
tying and returns the wrong LL.

## Risks / notes

- Correctness hinges on Batch-1 enabler (tied MAP = valid free `theta_nested`).
  If a tied fit's MAP were NOT slave≡master (e.g. pre-A1), C would be wrong — A1
  is a prerequisite and is on this branch's base.
- We are NOT doing option A (tying-out-of-model refactor); note it as future work.
  C fully serves the LRT use case at far lower risk.
