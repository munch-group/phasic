# B3 test baseline ledger

**Re-stamped 2026-08-13 (second) · master `d2cca7ab` (Batch 0 merge) ·
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
