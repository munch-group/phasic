# B3 test baseline ledger

**Pinned 2026-08-11 · master `cadf1ca4` · rebuilt via `pixi run install-dev`
immediately before the run (install log: session scratchpad
`baseline-install.log`).**

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
