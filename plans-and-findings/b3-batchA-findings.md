# Batch A findings

**Plan:** `b3-batchA-plan.md` v2 (+ the fold decision: svgd opt-out
bundled). **Branch:** `b3/batchA-rewards`.

## I1 micro-gates (`dr_batchA_i1_gate.py`; dump = pre-A install)

- **(a) rewardless byte-identity: PASS** (chain4 + branchy fixtures,
  exact array equality cross-install).
- **(b) rewards vs primal-FD** (the decision anchor is FD of the
  PRIMAL `Graph.moments(k, rewards)` — never a chain-formula oracle):
  chain4 mixed 1.1e-11 / ones 3.9e-11 / extreme(1e6..1e-6 scales)
  3.1e-11; branchy 7.9e-12 / 8.0e-12 / 2.0e-10 — ALL PASS at K=3.
  **all-ones == rewardless: BITWISE (both fixtures)** — the sharpest
  cheap sensitivity check on both hook lines.
- **(c) dph REFUTED contract: the `_dph` wrapper DECLINES any
  rewards** (the plan-review probe's numbers: chain+c2d 58.311 vs true
  51.644 — U/P non-commutation under reward scaling).
- **(d) rewards_len mismatch: all three wrappers decline (0-size).**

## Implementation notes

- Two hook lines applied in `ptd_b3_moments_core` (per-stage rescale,
  APPLIED comments cite the headline finding + the dm-inheritance
  verification); core-level rewards_len validation; three wrapper
  signatures extended per master §3; C++/pybind default-empty rewards.
- Python: per-call rewards threading through the expand_dims callback
  (leading-axis collapse per the probe); static dph+rewards decline
  with the refutation log; 2-D rewards keep FD with a log (the exact
  contraction is 1-D-only); svgd 1-D-rewards leaf forwards
  exact_moment_grad (bundle decision) + R29's 1-D arm relaxed (2-D
  keeps rejection, G.2 message).
- **Pre-existing defect found (NOT Batch A's)**: a direct 2-D-rewards
  call on the 1-D leaf fails in the FORWARD with a shape-contract
  error — verified identical on the pre-A install; the production 2-D
  route (the multivariate wrapper, per-feature 1-D slices) works and
  now engages exact per feature. LEDGERED at the merge review.

## Suite state at commit

`test_exact_grad_rewards.py` 6/6 (fate flip: decline-log test →
engages-silently + discrete-refutation-log + all-ones + multivariate
tests); `test_svgd_exact_moment_grad_kwarg.py` 1-D cell flipped to
honored (rename `test_rewards_1d_honored`); new
`test_svgd_exact_moment_grad_rewards.py` 3/3 (forwarding
discrimination, front-door default-runs-exact with real particle
init, 2-D still rejected).

## Gate ladder record (G4-fold completion — the G4 process review found
## these missing per house convention, m4)

- **G0**: baseline fresh — branch base `54d0c086`; every commit on
  master above the sixth ledger stamp (`c475a78c`) is docs-only
  (independently re-verified by the G4 wiring refuter).
- **G1** (batch tests, pre-fold): the three batch files 15 passed /
  3 skipped (the skips are pre-existing environment-conditional JIT
  cells gated on sources-on-disk, not vacuous — refuter-verified).
- **G2** (targeted map, pre-fold): 36 passed / 27 skipped / 1 xfail.
  No verbatim chunk output was preserved (process miss, G4 finding);
  the post-fold G2 re-run below is recorded verbatim instead.
- **G3** (full suite, 32 chunks): **1957 / 0 / 84 / 24** = ledger
  1951 + Batch A's 6 net new tests, zero failures, zero flakes.
  Per-chunk full `-rf` outputs preserved (`$SP/a_g3_out_*.txt`).
- **bf-chunk incident**: `split -l 5` over 156 collected files made 32
  groups (aa..bf) but only 31 (aa..be) were initially run; caught by
  tally arithmetic (1954 ≠ 1951+6), bf run (3 passed), total corrected.
  The G4 process refuter independently verified: union of the 32 group
  files == the 156-file list exactly, an output file exists for every
  group, summed tallies = 1957/0/84/24. Process amendment routed to
  `b3-execution-process.md` §4 (chunk list enumerated from split output
  on disk; union check + per-group output existence before tallies).

## Plan-deviation register (G4 finding M5 — none previously recorded)

1. Planned new file `test_exact_grad_rewards_supported.py` was never
   created — its cells were folded into the existing fate-table file
   `test_exact_grad_rewards.py` (defensible: one file per surface).
2. Planned `dr_batchA_dph_rewards_gate.py` was never created — the dph
   contract check was folded into `dr_batchA_i1_gate.py` section (c),
   which originally checked only a CONTINUOUS graph (non-discriminating)
   while its header claimed sub-kind coverage. CORRECTED at G4-fold:
   (c) now proves rewardless-computes / rewards-declines on BOTH
   discrete sub-kinds (was_dph via discretize(), native DPH).
3. Gate (b)'s dense-JAX-oracle leg was dropped; FD-of-the-PRIMAL is the
   sole (and independent) anchor. Docstring corrected to say so.
4. Plan I4/v2 test cells missing at commit, ALL ADDED at G4-fold with
   probe-measured actuals: log-mode model-level (I4.2), fixed_mask ×
   rewards (I4.6), vmap/jit composition (I4.7), all-zeros rewards
   (amendment 4), the multivariate absolute pin (amendment 4), K=1..3
   at model level (parametrize), the svgd True leg (MINOR 8).
5. The headline seed-only-wrong check shipped as a print, not an
   assert — the mixed-rewards FD leg discriminates seed-only scaling
   implicitly (its Jacobian differs), so substance survives; recorded.

## G4 adversarial diff review (two refuters, 2026-08-14)

Both verdicts: **SOUND-WITH-CORRECTIONS — no shipped-code defect.**
Every attacked code path survived independent numeric probes: C hook
placement (per-stage, inside the j=1..K loop), adjoint-side VJP, dm
scaling inheritance, rewards_len validation order, CRLF discipline,
pybind positional-caller safety, empty-array callback arg under
vmap/jit/jit∘vmap (bitwise), cross-install byte-identity (refuter
regenerated the golden itself), linear+log+multivariate numerics vs
independent oracles, R29 scope, forwarding seam.

Corrections folded (this commit):
- svgd docstring rewards clause (mandated shipped-text item that never
  shipped — M1 in both reviews).
- Stale `model_bwd` contract comment asserting rewards are NOT
  supported, directly above the dispatch that supports them (M2).
- `pmf_and_moments_from_graph` docstring FD-cause list + R29 leading
  comment + both test-file module docstrings (m3).
- Vacuous front-door log assertion (grepped a deleted string; passed
  under a simulated full-decline regression) → live static-decline
  filter + spy-count engagement floor (M2-tests/MINOR 7).
- All-ones test → bitwise equality (measured 0.0; doubles as the
  engagement guard — FD differs at ~1e-9, probe Q2) (m1).
- matches_fd tolerances → measured actuals (rel ≤5.6e-10 vs FD,
  ≤9.9e-11 vs central-diff; was a guessed 1e-4) (m2).
- The log-mode+rewards evidence gap (zero coverage anywhere; both
  refuters probed it PASS independently) → gate (b2) + pytest cell,
  measured exact [-27.0, -13.5], rel 2.5e-11 (M4/MINOR 5).
- Gate (c) discriminating sub-kind rewrite (MINOR 4).
- Missing plan cells added (MINOR 6, deviation register item 4).

**R29 discrete+rewards disposition (wiring M3, decided this fold,
flagged for user veto at merge review):** for an effectively-discrete
model with 1-D rewards, an explicit `exact_moment_grad` is accepted by
R29 but permanently inert (True → static refutation decline → FD;
False → FD). DOCUMENTED rather than rejected: R29 polices LEAF routing
only; builder-level static declines that depend on graph properties
(formula/callback modes on leaf 5 — the pre-existing precedent —
'log'+was_dph, discrete+rewards) are accepted-but-INFO-logged.
A config-layer arm could not be complete anyway: `Graph.svgd`'s
call-time `discrete=` override is not part of `SvgdConfig` (its
`is_discrete` is graph-derived), so the arm would miss `discrete=True`
on a continuous graph while the builder's INFO log catches every
route. Reversing this (adding the reject arm) is purely additive
strictness and remains available.

Fold test delta: `test_exact_grad_rewards.py` 6 → 12 tests
(parametrize K ±2, log-mode +1, all-zeros +1, fixed_mask +1,
vmap/jit +1); other files unchanged in count. Expected full-suite
tally after fold: **1963 / 0 / 84 / 24** (= G3's 1957 + 6). The fold's
only `src/` changes are comments/docstrings (no behavior), so the
pre-fold G3 remains valid for all untouched files; the touched test
files + targeted G2 map re-run post-fold (recorded below).

## Post-fold verification (verbatim, 2026-08-14)

- **Micro-gate** (`dr_batchA_i1_gate.py check` vs the pre-A golden):
  `ALL MICRO-GATES PASS` — (a) chain4/branchy `identical=True`;
  (b) rels 1.13e-11 / 3.94e-11 / 3.10e-11 / 7.93e-12 / 7.97e-12 /
  1.98e-10, ones==rewardless True both fixtures (unchanged from the
  original run, digit-for-digit); **(b2) NEW log/mixed rel=1.40e-10
  PASS, log ones==rewardless True**; **(c) NEW was_dph(discretize)
  rewardless size=4 / rewards size=0 PASS, native_dph 4 / 0 PASS**;
  (d) 0/0/0.
- **Batch files** (the three): `21 passed, 3 skipped` = pre-fold 15/3
  + the 6 fold tests, zero failures.
- **G2 map re-run** (`test_svgd_config.py`,
  `test_exact_grad_discrete.py`, `test_exact_grad_log_weight_mode.py`,
  `test_fd_gradient_mixed_scale.py`, `test_multivariate_correctness.py`):
  `99 passed, 1 xfailed`, zero failures.
- **Two gate-authoring lessons hit while adding (b2)/(c)** (bugs in the
  NEW gate code, not in shipped code; both now documented in-gate):
  the private `_moments_grad_theta_log` wrapper reads the graph's
  CURRENT weight state for the tape — direct callers must
  `update_weights(theta, log=True)` first or silently get the
  stale-weights Jacobian (the model callback handles this internally);
  and `discretize()` ADDS a parameter slot (the P=1 erlang becomes
  P=2), so wrapper theta length must be `param_length()` of the
  discretized graph.
