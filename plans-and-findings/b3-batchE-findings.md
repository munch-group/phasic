# Batch E findings

**Plan:** `b3-batchE-plan.md` v2. **Branch:** `b3/batchE-baked-exact`.

## E0 (pre-implementation): GO

(i) proposed baked backward vs shipped non-baked exact: 1.4e-12 (main),
8.1e-13 (n_unique==1), 2.2e-16 (n_obs==1); vs tight FD: FD-limited
1e-4. (ii) gate sweep {ones, 1e-2, 1e-4, mixed[1e-4,5e-2]}: ALL
COMPUTE; trap disposition: the fixture HAS 2 non-finite sojourn rows,
but they lie OUTSIDE the baked index sets (terminal vertices are
finite) — and baked probe set == call set, so committed models can
never touch them. (iii) front-door smoke: baked leaf engaged (spied
dedup lengths 30 obs → 1 unique).

## Implementation state (committed up to the HALT)

Model-level baked exact: DONE, 13/13 new tests + the F-file fate
rewrite green (30 passed across both joint-index files; baked-exact
vs non-baked-exact parity 2.5e-16). svgd plumbing (R31 + kwarg +
token): DONE; suite 13/15 — the two failing tests exposed the HALT
finding below.

## HALT FINDING (2026-08-14): SVGD particle init × conditioning gate × committed-raise

Running a REAL svgd fit with exact_grad=True on the continuous
coalescent fixture: the default particle initialization (log-scale,
sd=5.0) creates particles at extreme theta ratios, where the C
conditioning gate DECLINES → the committed exact path RAISES → the
first fit dies. E0(ii)'s sweep missed it (moderate scales only; the
plan's STOP condition fires now instead). Probe at the actual wild
thetas from the crashed fits:

| theta | default gate | gate lifted (H's skip_condition_gate) |
|---|---|---|
| [1e-6, 2.9e-5] | DECLINE | computes, matches tight FD to 1.0e-8 |
| [385, 1e-6] | DECLINE | computes finite, 34% OFF tight FD |
| [1e-8, 1e3] | OK | matches to 3.3e-8 |
| [1e3, 1e-8] | DECLINE | computes finite, 144% OFF tight FD |

Interpretation: at moderate dynamic range the gate is over-conservative
(the H finding transfers); at theta ratios ≳1e8 the gate flags GENUINE
fp breakdown — lifting it there returns silently unreliable numbers.
Neither raise-always nor lift-always is right for the svgd path.
USER DECISION requested (options in the tracker/ask): host-side
per-particle FD fallback on decline (possible inside the pure_callback,
unlike JAX-level fallback which the D6 record rules out) vs keep the
raise vs lift the gate. The MODEL-level baked exact path (direct
users, F semantics) is unaffected by this choice and is complete.

## Appendix (dated 2026-08-14) — decision outcome + final gate evidence

- **Decision (user): host-side per-particle FD fallback + WARNING** for
  the svgd entry (`exact_grad_decline='fd'`); model-level default
  'raise' keeps the F contract. Rationale + probe table above.
- **G1: 45 passed** at implementation; **49 passed** after the G4 fold
  (value-level fallback test, exact_grad_decline ValueError, 2-D
  rejection, jsp kind guard added).
- **G2: 156 passed / 3 skipped / 1 xfailed** (12-file surface incl. the
  G.1 suite, daisy/epoch files, svgd-config set, mixed-scale).
- **G3 (chunked, 31 groups, -rf): 1947 / 0 / 84 / 24** = ledger 1919 +
  28 new collected tests; skips/xfails ledger-identical. ONE first-pass
  transient in group ab (contains the modified F file + jax-heavy
  files); re-ran GREEN twice (46/7/1 both times); name not captured on
  the first pass because the chunk runner's tail -1 discarded the -rf
  lines (instrumentation slip, fixed mid-run; the G5 process amendment
  strengthens the wording: preserve full chunk outputs until tallies
  are recorded).
- **Cross-install golden** (`dr_batchE_golden.py`): svgd-built baked
  model gradient, no kwarg: dumped under the pre-E install
  `[89.98509261325792, -89.98500267487107]` at theta [1e-4, 1e-4];
  branch install: **identical=True (bitwise)**.
- **G4**: both refuters SOUND-WITH-CORRECTIONS, zero live numeric
  defects; all corrections folded (see the plan's G4 record).
