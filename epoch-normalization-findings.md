# Epoch normalisation — Batch 0 findings and implementation record

Plan: `epoch-normalization-plan.md` v2 (approved 2026-11-19 with D2a and
the other recommendations: D1 modify the epoch builder + listed tests,
D3 no runtime warning, D4 the tot_reward_limit single-epoch guard is its
own ticket). Branch `fix/epoch-normalization`, worktree
`/Users/kmt/phasic-epochnorm`, branched from master `57596461`.

## Batch 0 de-risk (script `experiments/dr_epochnorm_checks.py`)

Run against the modified code (Python-only change; package copy with the
two edited modules overlaid). All results as printed by the script.

| check | result |
|---|---|
| A n=2 one epoch vs closed form, reward_limit and tot_reward_limit, no exposure and exposure {0.5, 1, 3} | max rel err 2.2e-16 (all four) |
| B n=2 two epochs (c 2->7, mu 0.7->1.5, t1 0.5), alpha 1 | 2.79e-4 (gran 0), 2.86e-5 (1e4), 2.86e-6 (1e5) — first order |
| B same, alpha 3 (exposure branch) | 4.14e-4, 4.23e-5, 4.23e-6 |
| C n=3 closed form vs discrete table (raw, reward_limit 6) | 6.7e-16 over 49 rows |
| C n=3 normalised, per-feature / total / both caps | 7.8e-16 / 4.4e-16 / 3.3e-16; outcome sets equal the closed-form allowed sets (9 / 6 / 8) |
| D gradients, 1 epoch, no exposure | FD vs cf 8.4e-10, exact vs cf 3.0e-10, exact vs FD 1.1e-9 |
| D 1 epoch, exposure (quotient rule in exact block) | 2.4e-9, 3.0e-10, 2.7e-9 |
| D 2 epochs, no exposure (gran 1e5) | 3.6e-6, 3.6e-6 (granularity error), exact vs FD 9.8e-10 |
| D 2 epochs, exposure | 2.1e-6, 2.1e-6, exact vs FD 4.5e-9 |
| D vmap(grad) vs loop | <= 2.5e-9 everywhere |
| E deterministic expected-log-likelihood gate (mu fixed, alphas 0.3/1/4) | argmax rel err 1.1e-8; raw variant misses by 1.0e0, shared-Z by 2.7e-1, Z-at-alpha-1 by 7.4e-1 (detectable by >1e6x the tolerance) |
| F test.py model, tied 2 epochs, expected log-lik argmax | 5.000000 (rel err 8.4e-10); raw maximiser was 6.377 |

Closed forms (two-epoch n=2 formula from the statistics reviewer) were
verified against brute-force integration to 1e-14 during planning.

## Implementation (all in `src/phasic/__init__.py` unless noted)

* `_make_exact_final_block`: inner function renamed `block_with_rows`,
  returning `(r_v * soj * mass, J)`; `block` returns `[1]` (Jacobian
  value unchanged) and carries `block.with_rows`. Zero-mass branch
  returns zero rows and zero J.
* No-exposure `model()`: `joint_probs[pos] / jnp.sum(joint_probs)`
  (outside the custom_vjp); `model._raw_forward` seam.
* Exposure branch: `_normalise_rows` applied in `_per_obs_core` and its
  vmap rule; `_efg_block_exp_np` returns `(rows, J)`; the exact-final
  override applies `Jn = (J - q S) / Z` per unique alpha with a `Z == 0`
  guard (keeps the exact zero); `model._raw_forward` seam (not
  vmappable, documented).
* D2a: new rule R33 in `svgd_config.py` (after R30, so existing tests
  expecting R30's message for stopprob + exact_final_grad still match);
  `Graph.epoch_model` raises `ValueError` for `final_read='stopprob'`.
* Docstrings: `svgd` (`epoch_starts` paragraph, new `final_read` entry),
  `epoch_model`.
* `CLAUDE.md`: new section on the conditional likelihood and the
  deferred overflow efficiency gain.

## Tests

* New `tests/pytest/test_epoch_normalization.py` (independent closed
  forms, gradients incl. vmap, zero-mass exposure, deterministic gate,
  reproducer model, replicate MLE from an independent simulator with
  exposure-correlated dropout, D2a rejection).
* Changed (plan §8): `tests/pytest/inference/test_exact_grad_daisy_final.py`
  — the dense oracle now normalises over all t-vertices; the "plain FD"
  identity reference is the central difference of the RAW forward pushed
  through the same quotient rule (rtol 1e-12 kept), plus a check that the
  forward equals the normalised raw forward.
* Subset baseline (unmodified branch, 25 files, 436 tests): 428 passed,
  3 skipped, 5 xfailed, 0 failed (49:54).
* Same subset + the new file on the modified branch (reinstalled):
  445 passed (= 428 + 17 new), 3 skipped, 5 xfailed, 0 failed (51:42).
  GATE PASSED: no new failures; the §8 "may change" tests
  (`test_svgd_exact_final_grad_kwarg.py:153, 263`,
  `test_fd_gradient_mixed_scale.py:225`, `test_exposure_daisy_chain.py:822`)
  all still pass unmodified.
* Both runs end with a segfault AFTER pytest prints its summary (at
  interpreter shutdown) — present in the baseline too, so pre-existing
  and unrelated to this change; noted for a separate look.
* Not run: the full 2000+ test suite (the subset covers every file that
  touches joint-prob / epoch / daisy / exposure / LRT / svgd_config).
