# Batch G.2 findings

**Plan:** `b3-batchG2-plan.md` v1 + v2 (two USER DECISIONS: full
symmetry 2026-08-15; uniform 2-D rejection 2026-08-15). **Branch:**
`b3/batchG2-multivariate` (in-place per process §3.3 — pure-Python
batch; commits on the branch per §5.3).

## G0

Branch base = master `4bfd9f5b` (plan v2). Ninth ledger stamp
1992/0/84/24 at `35a17364`; two docs-only commits above it at branch
time (`7e1e162c` incl. the already-recorded caveat-clearing ledger
note, `4bfd9f5b`).

## De-risk disposition

D-G2.2 was RESOLVED at plan review (both refuters probed all three
compute paths: pybind crash text pinned, FFI silent-garbage shape,
callback value-parity vs the wrapper; trace-time raise legibility
proven under jit/vmap on the existing guard). D-G2.1 (forwarded-False
⇒ zero exact calls) folded into the I1 smoke: **False → 0 exact
calls; default → 2 full-size successes; parity rel 7.2e-10** — the
seam is real.

## Implementation (commits `09a201b1` + `21f6...`-series narrowing)

Wrapper kwarg forwarded to the SINGLE internal build site; svgd 2-D
leaf splat (None-absent); uniform ndim>=2 rejection at the shared
`_check_rewards_len` (one edit site covers all three compute paths);
R29 2-D arm retired + comment 4th rewrite + field comment + stale R31
message fixed; NEW rule R32; orientation doc fixes ((n_features,
n_vertices) — the svgd + wrapper docstrings contradicted the
validator and the wrapper's own slicing); backward 2-D arm
re-commented as unreachable defense-in-depth.

**I1 smoke:** wrapper False/default spy 0-vs-2; 2-D-on-1-D →
actionable ValueError; R32 fires with the actionable route.

## R32 predicate narrowed at G1 (review-claim correction — deviations register item 1)

The plan-review claim "only SparseObservations completes with 2-D
rewards" was ITSELF too broad: the G2 map's first run failed 5 tests
proving **dense 2-D observations (n_times, n_features) + 2-D rewards
is a WORKING, shipped-test-covered multivariate route** (incl. with
exposure: `test_multivariate.py::TestSVGDMultivariate` ×2,
`test_svgd_exposure.py::test_multivariate_exposure_*` ×2,
`test_svgd_config.py::TestR11_ExposureWith2DRewardsWarns`). R32 now
rejects exactly the genuinely-dead construction — dense **1-D**
observations (`observation_kind == '1d_times'`) + 2-D rewards (passes
validation then dies in `_log_lik_from_pmf`) — with both actionable
routes named. The exposure × 2-D question thereby resolves
differently than the v2 §B/§E notes predicted: the SPARSE+exposure
route is rejected by the existing sparse-exposure rule, while the
DENSE-2-D+exposure route works and keeps R11's warning (all pinned by
the shipped suite). The impossibility test cell pins sparse+exposure
and 1-D-dense+2-D; the working dense-2-D route is pinned by the five
pre-existing tests above.

## Further deviations register

2. The exposure×2-D "newly-reachable cell" shipped as a
   transitive-impossibility/working-route pin pair rather than the v2
   §E warn-and-complete cell (which was factually unbuildable — see
   the narrowing note).
3. Commit granularity: implementation landed as one feat commit + one
   fix commit (narrowing) rather than per-item.

## G1 (batch files, post-narrowing)

`test_exact_grad_multivariate_kwarg.py` (8 cells) +
`test_svgd_exact_moment_grad_kwarg.py` (fate flip → honored_sparse +
R32 pin) + `test_svgd_exact_moment_grad_rewards.py` (fate flip →
honored_with_spy): **18 passed / 3 skipped** first post-narrowing run;
25 passed / 3 skipped including the five recovered map tests.

## G2 (expanded map incl. the four multivariate files, verbatim)

15 files: **187 passed, 2 skipped, 1 xfailed, 0 failed** (636.95s).

## G3 (full suite, 32 chunks, in-place branch, verbatim)

Union check OK (159 collected files == 32-group union; output per
group; `-rf`). Summed: **2001 passed / 0 failed / 84 skipped / 24
xfailed / 0 xpassed / 0 errors** = the ninth ledger stamp's 1992 +
G.2's 9 net new tests exactly (new file 8 cells; kwarg file +1 net —
the rejection pin became honored_sparse + the R32 pin; rewards file
0 net). Several sleep-kill interruptions resumed from preserved
per-group outputs; no green group re-run.

## Dated correction (2026-08-15, at G4 fold)

The implementation-record line "commits `09a201b1` + `21f6...`-series
narrowing" cited a wrong hash: the R32 narrowing is **`714f878d`**.

## Consumer notes (v1 §1/§E mandate, added at fold)

- `Graph.mcmc` builds the multivariate wrapper kwarg-less → inherits
  default True ≡ pre-batch construction; mcmc is gradient-free (zero
  jax.grad), so the kwarg is inert there.
- `method_of_moments.py:389` likewise builds it kwarg-less → default
  True; scipy computes its own FD Jacobian regardless (CLAUDE.md §16b
  item 5 unchanged).

## G4 adversarial diff review (two refuters, 2026-08-15)

**Both SOUND-WITH-CORRECTIONS, zero shipped-code/behavior defects.**
Wiring lane: own central-diff oracle default/True rel 9.5e-11,
default==True bitwise, False→0 exact calls under grad/vmap(grad)/
jit(vmap(grad)); svgd three states on the sparse route (None ≡ True
bitwise, ledger rows exact, False→0 calls); the R32 matrix fires on
exactly (1d_times, 2d) across 16 cells + the SVGD-class entry; the
uniform rejection 20/20 across three paths × four transforms with 1-D
rewards unaffected; vacuousness sims prove the guard structure (each
simulated regression fails exactly the cells designed to catch it);
the pre-G.2 baseline re-verified on an edited package COPY (pybind
crash / FFI silent garbage / callback value-parity — the
migration-lossless claim execution-verified). Process lane:
independently re-derived every gate record (G1 re-executed 18/3; G2
outputs verified; G3 union+tally recomputed = 2001/0/84/24 = 1992+9
cell-verified); judged the R32 narrowing defect-avoidance restoring
shipped behavior wrongly declared dead at plan review — adequately
recorded, re-presented here. FOLDED (all text): wrapper docstring
gains the kwarg (+fixed_mask) Parameters entries; both flipped files'
module docstrings + the sparse-only-route comments corrected to the
post-narrowing truth; the A-era engagement test gains its closure
note; the exposure-test prose rewritten; R32 renamed
(_check_R32_2d_rewards_reject_dense_1d_observations) + remedy order
leads with the universal dense_to_sparse; the two legacy vmap
comments marked unreachable-since-G.2. Known text quirks accepted:
R32's degenerate-ndim message (0-d/3-d obs classify as 1d_times —
those constructions were broken anyway); honored_sparse is
ledger-level by design (the spy cells are the deep guard);
zero-regularization svgd cannot distinguish exact/FD by particles
(pre-existing property of all moments leaves — the spy/ledger cells
are the honest proof, as shipped).
