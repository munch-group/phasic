# Batch G.1 findings — gate evidence and deviation register

**Plan of record:** `b3-batchG1-plan.md` v2. **Branch:**
`b3/batchG1-svgd-daisy-plumbing` (worktree `../phasic-batchG1`), cut
from master `6897e612` (docs-only above the ledger stamp `ecd708fc`).

## G0 — front-door smoke (`experiments/dr_batchG1_frontdoor.py`): GO

Run on the PRE-G.1 install (main checkout), verbatim results:
shape (a) single-epoch + exposure — constructed+ran: True; particles
finite: True; daisy spy calls: 10 (daisy-built: True). Shape (b)
multi-epoch no-exposure — constructed+ran: True; particles finite:
True; daisy spy calls: 16 (daisy-built: True). **GO.**

## Golden gate (`experiments/dr_batchG1_golden.py`)

Dump under the pre-G.1 main-checkout install: gradient
`[0.011699451794481803, -0.004499810946164241, 467.77576429971964,
-179.91303673462977]` at theta `[1e-4, 1e-4, 5e-5, 1.3e-4]`
(2-epoch svgd-built model, no kwarg). Check under the branch install:
**identical=True (bitwise, np.array_equal)**. Independently
RE-EXECUTED from scratch by the G4 config/wiring refuter (fresh dump
under the verified-pre-G.1 install): identical=True again.

## Gate tallies

- **G1:** 17/17 at implementation (`d18c1794`); **19/19 after the G4
  fold** (`000c8392` adds the jsp-message test and the rewards2d
  None-sweep cell).
- **G2 + fate table:** 146 passed / 3 skipped / 1 xfailed across the
  plan's named files — ZERO existing-test flips (the R9/R29 message
  edits were regex-safe exactly as the pre-filled fate table
  predicted). The two pinned regex files re-run by a G4 refuter:
  72 passed / 3 pre-existing skips.
- **G3:** chunked (31 groups of ≤5 files), summed:
  **1917 / 0 / 84 / 24** = the ledger's expected 1900 baseline + 17
  new tests; skips/xfails identical to the ledger. See the flake
  record below.
- **G4:** two refuters, both SOUND-WITH-CORRECTIONS, zero shipped-code
  defects; all corrections folded (`000c8392`). Independent probes by
  the refuters: exact-vs-tight-FD gradient rel err 1.9e-8 / 7.5e-8
  through the public plumbing; spy-count deltas measured (8→16,
  20→28); tolerance actuals measured (rel ~1e-6).

## G3 flake record

First pass: two single-test failures, one in group ac, one in group
ad; names NOT captured (the first-pass command lacked `-rf` — an
instrumentation gap, see the process amendment below). Both groups
green on identical-command re-runs (ac twice: once in an
exit-0 background pass, once documented foreground 19 passed /
26 skipped; ad re-run split: 9 + 34 + 5 passed / 5 skipped = the full
group). Group manifests (reconstructed from the split, marked as
reconstruction): ac = {test_jax_integration, test_log_lik_from_pmf,
test_log_likelihood_zero_inflated, test_log_prob_unified_bit_identity,
test_lrt_at}; ad = {test_manual_vs_trace_graph, test_mcmc_accuracy,
test_model_selection, test_moments_from_graph_vmap,
test_multivariate_correctness} — the suite's stochastic SVGD/MCMC
accuracy tests live in exactly these groups; the new G.1 file is in
group ah. A dedicated `-rf`-instrumented naming pass of ac+ad was run
post-G4: result recorded below when complete.

- **Naming-pass result (`-rf`, both groups in one run, 22:00 min):
  67 passed / 31 skipped / ZERO failures.** Both first-pass failures
  are unreproduced across two subsequent identical-scope runs —
  CLOSED as stochastic flakes (candidate manifests above); counts
  discipline met (final tally 1917/0/84/24 = ledger + new tests).

**Process amendment proposed at G5 (G4 tests/process refuter):**
chunked G3 runner commands always pass `-rf` so first-pass failures
are named — the actual defect here was instrumentation, not the suite.

## Deviation register (all recorded per process §6; folded or triaged)

1. Plan I3 test 7 (cross-install golden) shipped as
   `experiments/dr_batchG1_golden.py` — a pytest test cannot span two
   installs; the H micro-gate (a2) template. Result above.
2. Plan test 9's oracle-column-sum comparison shipped as FD-front-door
   parity (tolerance now 1e-4, anchored to the G4-measured 1e-6):
   the oracle variant is covered at the builder level by
   `test_exact_grad_daisy_final.py`'s tied test, composed with the
   verbatim-forwarding proof (test 11). Recorded, not re-implemented.
3. Tests 3/4 shipped relative spy-count claims at G4; the fold pinned
   the ABSOLUTE deltas (8 in both shapes) per the plan's zero-FD
   intent.
4. R30 is deliberately STRICTER than the plan's letter: any non-None
   value (explicit False included) is rejected on stopprob/non-linear
   cells — the R29 discipline; tested post-fold; the H construction
   guard's advice text aligned ("Drop exact_final_grad").
5. The None-sweep's jsp and rewards2d cells are validation-level (a
   full fit needs different fixtures); reasons stated in-code.
