# B3 test baseline ledger

**Re-stamped 2026-08-15 (ninth) · master `35a17364` (Batch C merge) ·
verified 1992 / 0 / 84 / 24 via a MEASURED post-merge chunked run in the
MAIN checkout (32 groups, `-rf`, union == 158 collected files, output
per group, freshly rebuilt install) = eighth stamp's 1978 + Batch C's
14 tests. The raw run showed 1990/2; BOTH failures are
environment-caused, not code (the machine spent the run in an
aggressively-sleeping/throttled state, ~2-4× slower than the morning's
eighth-stamp run on the same groups): (1)
`test_scc_parallelism_smoke.py::test_cpu_time_exceeds_wall_time_on_warm_path`
asserts cpu>wall — machine sleep mid-test breaks exactly that invariant;
PASSED on solo re-run. (2) `test_svgd_exposure.py::
test_exposure_shifts_posterior_inverse_to_alpha` failed as a PURE
`pytest-timeout` wall-clock kill (>600 s; the test normally runs
~150-300 s and its whole 5-file group took 507 s in the worktree G3 on
IDENTICAL merged content, where it PASSED) — reproduced under
caffeinate only because throttling persists; no assertion ever failed.
Re-verify trivially on an awake machine. Known-failure ledger EMPTY.**

*(Previous stamp:)* **2026-08-14 (eighth) · master `c6cc38b9` (Batch B merge) ·
verified 1978 / 0 / 84 / 24 via a MEASURED post-merge chunked run in the
MAIN checkout (32 groups, `-rf`, union == 157 collected files, output
per group, freshly rebuilt install) = seventh stamp's 1963 + Batch B's
15 tests. The raw run showed 1977/1: a single first-pass failure in
`test_exact_grad_joint_index.py::test_default_path_uses_fd` (a Batch-F
surface untouched by B — the failing assertion saw exact_grad=True fall
back to FD, arrays bitwise-identical, i.e. the SAFE fallback, not a
wrong number) UNREPRODUCED across three re-runs (test alone + its full
5-file group twice, 61 passed each) — closed as a stochastic transient
per the fifth stamp's precedent. Known-failure ledger EMPTY.**

*(Previous stamp:)* **2026-08-14 (seventh) · master `798ddcaa` (Batch A merge) ·
verified 1963 / 0 / 84 / 24 via a MEASURED post-merge chunked run in the
MAIN checkout (32 groups, `-rf`, freshly rebuilt install) — the first
stamp run under the A-G4 process amendment: groups enumerated from the
split output on disk, union == the 156 collected files verified, an
output file confirmed per group before tallies. Arithmetic check: sixth
stamp's expectation 1951 + Batch A's 6 net new at G3 + the G4 fold's 6
(rewards file 6 → 12) = 1963, exact match. Run was sleep-killed twice
and resumed from preserved per-group outputs (no green group re-run).
Incidental `jax._src.callback` ERROR log lines inside passing tests are
expected (decline-raise legibility tests exercise those paths).
Known-failure ledger EMPTY.**

*(Previous stamp:)* **2026-08-14 (sixth) · master `c475a78c` (Batch E merge) ·
verified 1947 / 0 / 84 / 24 via the batchE worktree's chunked run (31
groups, `-rf`) on content identical to the merged tree — ledger 1919
(fifth stamp's expectation) + 28 new E tests at G3 time; the G4 fold
added 4 more tests, so the next full run will show 1951. One ab-group
first-pass transient re-ran green twice (record in
`b3-batchE-findings.md`; the chunk runner's tail-1 initially discarded
the `-rf` names — amendment strengthened in the process doc).
Known-failure ledger EMPTY.**

*(Previous stamp:)* **2026-08-13 (fifth) · master `0c052cfe` (Batch G.1 merge) ·
verified 1917 / 0 / 84 / 24 via the batchG1 worktree's chunked run (31
groups, `-rf` now mandatory per the adopted process amendment) on
content identical to the merged tree — ledger 1900 (fourth stamp's
expectation) + 17 new G.1 tests at G3 time; the G4 fold added 2 more
tests, so the next full run will show 1919. Two first-pass single-test
transients (groups ac/ad) were unreproduced across two re-runs
including a dedicated `-rf` naming pass (67/31/0) — closed as
stochastic flakes, full record in `b3-batchG1-findings.md`.
Known-failure ledger EMPTY.**

*(Previous stamp:)* **2026-08-13 (fourth) · master `ecd708fc` (Batch H merge) ·
verified 1899 / 0 / 84 / 24 via the batchH worktree's chunked full-suite
run (31 groups) on content identical to the merged tree — ledger 1889
(third stamp's 1888 + its one post-G3 test) + 10 new Batch H tests. The
G4 fold added one more test (n_epochs==1) AFTER the G3 run, so the next
full run will show 1900. Known-failure ledger EMPTY.**

*(Previous stamp:)* **2026-08-13 (third) · master `eaf86e82` (Batch F merge) ·
verified 1888 / 0 / 84 / 24 via the batchF worktree's chunked run on
content identical to the merged tree — ledger 1885 + 3 net new tests
(joint-index file 13 → 16 at F2, 17 after G4 fold-ins with one more added
post-G3; next full run will show 1889). Known-failure ledger EMPTY.**

*(Previous stamp:)* **2026-08-13 (second) · master `d2cca7ab` (Batch 0 merge) ·
verified 1885 / 0 / 84 / 24 via the batch0 worktree's chunked full-suite
run on content identical to the merged tree — exact match to the previous
stamp; ledger unchanged (empty).**

*(Previous stamp:)* **2026-08-13 · master `164e2758` (Batch D Tier-1 merge) ·
verified counts: 1885 passed / 0 failed / 84 skipped / 24 xfailed** —
obtained from the batch worktree's chunked full-suite run on content
identical to the merged tree (12 sub-runs, same command semantics; a single
background run is killed by machine sleep on this host, see the batch merge
review). Exact baseline arithmetic vs the 2026-08-11 stamp: +6 new
no-source-dir tests passed, +8 new source-gated skips, xfail map unchanged
at 24. Known-failure ledger remains EMPTY.

*(Original stamp, superseded: 2026-08-11 · master `cadf1ca4` · rebuilt via
`pixi run install-dev` immediately before the run.)*

## Command and result

```
pixi run pytest tests/pytest/ -q --tb=no -rf
→ 4 failed, 1875 passed, 76 skipped, 24 xfailed, 701 warnings in 2624.97s (0:43:44)
```

followed by a test-only alignment (below), after which:

```
pixi run pytest tests/pytest/test_graph.py::TestDiscretize -q
→ 5 passed
```

**Effective baseline: 0 failed / 1879 passed / 76 skipped / 24 xfailed.**

## Known-failure ledger: EMPTY

The only 4 failures in the full run were `test_graph.py::TestDiscretize::
test_rate_{zero,one,negative,greater_than_one}` — stale tests pinning the
`discretize(rate)` contract that commit `c673be83` ("Removed mistaken check
for rate <= 1", 2026-07-30, user-authored) deliberately changed: rate must
now be `> 0` (new message "rate must be larger than 0"); rates `>= 1` are
accepted. Not regressions. The four tests were aligned to the new contract
(test-file-only edit, `tests/pytest/test_graph.py::TestDiscretize`; the
edit cannot affect other tests, so the full-run arithmetic 1875+4 holds
without a second full run). **The alignment edit is currently uncommitted**
— re-stamp this ledger's commit hash when it lands.

## Interpretation rules for gate G3 (see `b3-execution-process.md` §4)

- **Failures:** any failure at all is NEW (ledger is empty) → gate fails.
- **24 xfailed:** these are the deliberate strict-xfail cross-path pin map
  (Q1 trace-refuses-cyclic/formula ×6, Q5/Q6a/Q6b serialize-roundtrip,
  Q10 SCC ordering, Q11a sampler seeding, Q-G4-1/2 formula VM, and
  friends). A new **XPASS** is not a win — it means a refactor silently
  unified a pinned divergence: investigate before merging (the F-006 audit
  rule). A *vanished* xfail (test removed/renamed) is likewise a finding.
- **76 skipped:** environment/feature-gated skips; the count is pinned —
  a materially changed skip count needs an explanation in the merge review.
- **Warnings (701):** not gated, but a new warning *class* from touched
  code is worth a line in the merge review.

## The sources-on universe (addendum 2026-08-12)

The baseline command runs WITHOUT `PHASIC_SOURCE_DIR`, so tests requiring
the C/C++ sources on disk (JIT-compile paths) sit in the 76-skip bucket.
Setting `PHASIC_SOURCE_DIR` un-skips them and exposes **9 pre-existing
failures, all in `inference/test_jax_integration.py`**
(TestMomentsFromGraph ×2, TestPMFAndMomentsFromGraph ×2,
TestBatchOperations ×3, TestMultivariateSampling ×2) — differentially
confirmed identical on untouched master `19b86d71` (2026-08-12), matching
the long-documented "9 test_jax_integration failures are PRE-EXISTING
(fail at param_length check before B3 code)". Rules:
- **G3 is defined in the no-source-dir universe** (baseline-identical
  command). Never compare a sources-on run against this ledger's counts.
- G2 runs that need `PHASIC_SOURCE_DIR` treat exactly these 9 as
  ledgered known-failures; a 10th sources-on failure is NEW.

## Scope

`tests/pytest/` only. The `pixi run test` notebook-conversion path is
excluded from the baseline: `docs/pages/tutorial/model-selection.ipynb` is
known-pre-broken (fix-D kwarg guard; tracked in
`b3-execution-tracker.md` → non-B3 residuals) and notebook execution is not
an instrument for B3 gates.

## Regeneration

Regenerate at every merge to master (process §4.1/§7.5): re-run the full
command, update the counts + commit hash here, and re-derive the
interpretation rules if the pin map changed. Native C/C++ changes require
`pixi run install-dev` before the run (copy install — a stale install
invalidates the ledger).
