# Batch H de-risk findings (H0-H2)

**Plan of record:** `b3-batchH-plan.md` v2 (two-refuter review folded).
**Branch:** `derisk/batchH-final-epoch` (created off master `52d688ff`).
**G0 record:** ledger `b3-test-baseline.md` third stamp @ `eaf86e82`
(1888/0/84/24, known-failure ledger empty; next full run expected 1889).
Master commits above the stamp at branch time (`127ed6ee`, `e06ea0c6`,
`4bc24fb7`, `52d688ff` + docs commits between) are docs-only except
`e06ea0c6`'s install rebuild note (no source change) — verified via
`git log --stat eaf86e82..52d688ff` at branch creation; recorded here
explicitly per plan v2's G0 requirement, not assumed.

## H0 — pure-JAX one-hop oracle (`experiments/dr_batchH_oracle.py`)

**Status: DONE 2026-08-13 — ALL GATED CHECKS PASS.** Fixture: the
`test_lrt_at.py` epoch model (jpg n=514, jsp n=730, sg n=512, n_ipv=510,
n_t=216, P=2, epoch_starts=[0, 0.5]).

| check | criterion | result |
|---|---|---|
| (i) tier-1: production FFI vs pybind step-by-step replication | ≤1e-12 | **2.2e-16 PASS** — also validates the H2(a) Python-replication handoff route at machine precision |
| (i) tier-2: production vs dense-JAX oracle (expm chain + linear solve) | measured | 4.7e-15 (uniformization at this fixture is fp-exact; granularity sweep 512/4096/32768 all ~1e-15) |
| (ii) linearity of the SHIPPED final read in raw handoff ipv | fp round-off | **1.2e-15 PASS**; normalization probe: `soj(2x)==soj(x)` EXACTLY → C normalizes internally, the mass rescale restores linearity — the claimed mechanism verified precisely |
| (iii) exact block incl. r_v product-rule term vs `jax.jacobian` | ≤1e-10 | **2.7e-13 / 1.4e-13 PASS** (oracle- and production-handoff) |
| (iv) composed Jacobian, final-epoch cols vs oracle | ≥10× better than FD | **5.0e+05× (benign), 3.6e+05× (mixed-scale)** — FD final-col error 4.0e-06 / 1.0e-07, composed 8.0e-12 / 2.8e-13; earlier-col FD error 2.4e-10 / 1.2e-11 (the final epoch owned ~all of FD's error on this fixture) |
| (iv) tied variant (master = flat cols 0+P summed, the scatter-VJP rule) | recorded | 2.4e-10 / 9.3e-12 — consistent with flat-col errors |

r_v term coverage note: at the FFI/Jacobian level ALL slots are free
(the SVGD fixture's `fixed=[(1, mu)]` pins happen above the custom_vjp),
so the mutation-slot column exercises the r_v product-rule term without
a fixture rewrite — the amendment-1 masking concern is discharged.

**MAJOR NEW FINDING — the MPFR conditioning gate declines 100% of
realistic calls on this fixture, and its declined answers are accurate
anyway.** Diagnosis chain (`scratchpad dbg_decline2-6`, boundary mapped):
theta=ones never declines (the Batch-F-style probe WOULD pass); at the
fixture's realistic coalescent theta (~1e-4) the default
`PHASIC_CONDITION_THRESHOLD` declines once handoff entries reach ~1e-8
dynamic range; the TRUE production handoff has NO zeros/negatives and
min-nonzero entry **5.4e-148** (range ~1e148), so the gate declines at
ANY threshold below ~1e300. With the gate lifted (1e300), the computed
Jacobians match the fp64 oracle to ~1e-13 at exactly those points —
i.e., on the sojourn path the gate is pure over-conservatism at
realistic coalescent scales (hard evidence for §16b item 2's
"MPFR-comment/rationale doesn't transfer", and a direct input to CC-2/
Deferred-4 threshold semantics). Consequence for v3 design: probe-and-
commit at theta=ones would commit and then RAISE on the first real call;
the failure-mode design MUST NOT be gate-governed-with-raise. Options
for v3 (user decision, per plan): lift/re-scope the gate for this path,
or floor the handoff (mass below ~1e-15 contributes nothing to final_jp
at fp64), or per-call FD fallback. H2(d)'s census: 4/4 realistic
(iii)/(iv) evaluations declined at default threshold.

## H1 — cost/instrument study (`experiments/dr_batchH_cost.py`)

**Status: DONE 2026-08-13.** Warm medians (N=20), three fixture sizes
(nr_samples=4/5/6 → jpg n=514 / 3,106 / 18,910 — a 37× range):

| quantity | n=514 | n=3,106 | n=18,910 |
|---|---|---|---|
| fused FFI forward | 7.3 ms | 49.6 ms | 332 ms |
| handoff extraction (pybind replication) | 3.9 ms | 25.1 ms | 163 ms |
| exact adjoint call (contains offset conversion) | 0.60 ms | 4.4 ms | 34.9 ms |
| full-FD backward (2·4 FFI calls) | 58.7 ms | 397 ms | 2,659 ms |
| **exact-block add-on / FD backward** | **7.6%** | **7.4%** | **7.4%** |
| adjoint / FD backward | 1.0% | 1.1% | 1.3% |

**(a) §16b item 3 decision: DECLINE conversion caching, with evidence.**
The entire adjoint call — conversion included — is 1.0-1.3% of the FD
backward it replaces, stable across the size range; caching the
conversion can recover at most that. The plan's "prohibitive" bound
(add-on ≤50% of one FD backward) is met by 7×; the composed backward is
in fact a net SPEEDUP (it removes 2·P_final FFI calls ≈ 4×t_ffi, ~4×
the add-on's cost). Cold-vs-warm adjoint difference is ~10-15% (the RAW
tape cache works; no O(n³) rebuild observed).
**(b) J_ipv instruments: SKIPPED (contingency not triggered)** — H0(iv)
leaves the ipv_bar variant nothing to buy (earlier-col FD error is
already 4 orders below the final-col FD error the primary variant
eliminates).

## H2 — wiring-point study

**Status: DONE 2026-08-13** (no separate script needed — items landed in
H0/H1 + code reads):

- **(a) handoff route: Python replication (option ii) — VALIDATED and
  costed.** Tier-1 parity 2.2e-16 (H0); cost ≈ half of one fused FFI
  forward (H1), inside the 7.4% total add-on. No new native surface, no
  §16b-item-8 handler-family exposure; NaN from the pybind
  stop_probability path raises Python-side (loud by construction).
- **(b) composition point**: the two `_daisy_chain_svgd_model`
  custom_vjp backwards (`_autodiff_bwd` `__init__.py:4676-4690`,
  `_per_obs_bwd` `:4907-4930`); final-epoch slots leave the FD loop,
  earlier slots keep full-chain FD; external VJP shape unchanged
  (master §10 authorization). Flat-slot convention verified in H0
  (tied variant = scatter-VJP column sums, checked numerically).
- **(c) exposure cost model**: per backward per particle the exact path
  adds K_unique × (handoff extraction + adjoint) ≈ K_unique × 7.4% of
  one no-exposure FD backward; the FD baseline it replaces also scales
  with K_unique, so the RATIO is K-invariant. Recorded; v3 must still
  compose the final-epoch theta slice with the pre-FFI exposure scaling
  (chain-rule factor per unique alpha).
- **(d) decline census: answered by the H0 gate finding** — 100% of
  realistic-theta calls decline at the default conditioning threshold on
  this fixture (not a rare tail event; a per-SVGD-trajectory count is
  moot at 100%). See the H0 MAJOR FINDING block.
- **(e) upstream weight-mode rejection: NONE exists.** The svgd
  theta-dim resolution explicitly SKIPS formula inference under
  `epoch_starts` (docstring `__init__.py:5205-5207`) and formula mode is
  deliberately propagated into the sojourn graph (`:10141`) for C-side
  evaluation — so non-linear weight modes reach the daisy path today,
  and the v3 wiring MUST carry its own loud linear-only guard
  (joint-index precedent `:7941-7947`).

## Go/no-go tally (plan v2 §Go/no-go) — **GO**

- H0(i) ≤1e-10: **2.2e-16** ✔ · H0(ii) linearity: **1.2e-15** ✔ ·
  H0(iii) ≤1e-10: **1.4e-13** ✔ · H0(iv) ≥10×: **3.6e5×** ✔ ·
  H1(a) recorded: **decline caching, evidence above** ✔ · H2(a)
  validated: **tier-1 2.2e-16** ✔. No NO-GO condition fired.
- **GO to v3 implementation planning, WITH one open user decision**
  (plan v2 anticipated it; the numbers are now in hand): the v3
  dispatch/failure-mode design cannot be gate-governed-with-raise —
  the default conditioning gate declines 100% of realistic calls while
  its answers match the fp64 oracle to ~1e-13. The options + evidence
  go to the user before v3 is drafted (v3's central design input).

## I1 micro-gate results (2026-08-13, worktree `../phasic-batchH`, branch `b3/batchH-final-epoch`)

`experiments/dr_batchH_i1_gate.py` (dump under the pre-H master install,
check under the new build):

- **(a) flag=0 byte-identity: PASS 6/6 after the G4 fold** (originally
  4/4 on the sojourn-graph fixture only — the G4 tests/process refuter
  flagged the narrowing vs the plan's "joint-index fixtures AND the H0
  sg fixture"; two `jix_*` cases on the joint-prob graph — the
  joint-index consumer's graph type — were added and the goldens
  re-dumped from the pre-H install: all identical). **(a2), added at
  the G4 fold: the daisy False-path GRADIENT cross-install golden (the
  plan's I3 test-2 commitment, initially substituted by in-install
  checks — G4 MAJOR 2) is bitwise IDENTICAL between the pre-H install
  and the new build.** New nuance recorded: theta=ones with the REAL
  handoff ALSO declines on the pre-H build (the ~1e148 ipv dynamic
  range alone crosses the default threshold, independent of theta) —
  consistent with, and sharpening, the H0 gate finding.
- **(b) flag=1 at the H0 declining point: COMPUTES, matches a dense-JAX
  oracle to 1.5e-16** (tol 1e-10). Default path still declines there.
- **(c) trap fixture (manually built parameterized trap cycle): with
  the gate skipped the adjoint still DECLINES.** (G4 reword: the script
  proves the decline, not its mechanism — no instrumentation excludes
  other residual causes, and the gate script cannot go red on a
  computed-finite trap result. The pytest suite still has no trap
  fixture; the CLAUDE.md-flagged trap-class gap narrows but stays
  open.)
- **(d) subnormal-mass probe: mass=1e-300 and 1e-308 both DECLINE**
  (not garbage, not a crash). **I2 branch-width decision: the
  zero-branch stays exact-`mass == 0.0` only; subnormal-but-nonzero
  mass routes to the residual-decline RAISE with its diagnostic** —
  defensible because such a particle's forward/loss is already
  degenerate, and the raise names the cause.

## Appendix (dated) — decisions taken; v3 review folded

- **2026-08-13: the open user decision above is TAKEN** (master
  `04775b63`, merged here via `cc5a936d`): the conditioning gate gets an
  ADDITIVE C-side opt-out for the Batch-H caller (default behavior
  unchanged everywhere else; chosen over env-var wrapping,
  raise-on-refusal, and parking). In the same sitting: the joint-index
  `exact_grad` default STAYS False, with the trade-off documented in
  its docstring (closes the Batch F merge-review queued item 1).
- **2026-08-13: plan v3 drafted (`a5b03d64`), two-refuter adversarially
  reviewed, all findings folded → v3.1 (cleared for implementation).**
  Notable review corrections to de-risk-era statements: production's
  FORWARD NaN-fills at a zero handoff (the H0 scripts never exercised
  mass==0 — the zero-mass branch's rationale is corrected in the plan);
  the C adjoint DECLINES at a zero IPV; subnormal-mass behavior is
  unmeasured → new I1 micro-gate (d) probes it. The CC-2/Def-4 note in
  plan §scope is superseded: the opt-out path no longer reads
  `PHASIC_CONDITION_THRESHOLD`; the H0 gate evidence is routed to
  Def-4/CC-2 at G5 as a dated note.
