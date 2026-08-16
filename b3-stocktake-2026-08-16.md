# Stock-take: is the exact-gradient programme aligned with what phasic is for?

Written 2026-08-16 at the user's request, after the scale requirement
(20k-60k vertices typical, 200k-600k required) was stated for the first
time and immediately falsified several of the programme's working
assumptions. This document takes stock; it does not decide anything.
Plans are re-opened after the Q&A that follows it.

---

## 1. What was built

The B3 programme set out to replace finite-difference gradients with
exact analytic theta-adjoints, because FD uses one absolute step
(eps=1e-7) for every parameter and therefore fails at mixed parameter
scales — 4-9% error at theta=[1,1e-8] on the pinned test, and a hard
crash on the daisy-chain path.

Every planned batch shipped. In order: the shared moments core (Batch
0); rewards (A); the four weight modes — linear, log, formula, callback
(B, C, and predecessors); SVGD plumbing on every leaf (D, G.1, G.2);
the joint-index sojourn adjoint and its baked/dedup mode (E, F); the
daisy final-epoch exact gradient (H). Plus, this session, a guard that
stops the exact-gradient entry points from silently accepting synthetic
SCC graphs and returning plausible-but-wrong Jacobians.

Test ledger at the eleventh stamp: 2012 passed, 0 failed, 84 skipped,
24 xfailed. Every batch went through a plan review, gate ladder, and
two adversarial diff refuters. That process worked: essentially every
real defect in this programme was found by review rather than by the
implementation's own tests, including two in shipped code and, this
week, a guard bypass across the SLURM serialisation boundary.

## 2. What the scale check revealed

Measured 2026-08-16 with a memory-capped ladder on the two-locus ARG
model — the library's headline application domain:

- At 1,044 vertices and a benign theta=[2.0, 5.0], the exact moments
  gradient DECLINES. The cause is the MPFR conditioning gate, not
  scale or topology: lifting the threshold makes it accept, and the
  answer it was refusing is accurate to 8.5e-13 against a Richardson
  reference. So on this model class the shipped default silently
  delivers finite differences — after paying the full exact-path setup.
- At 8,407 vertices a gradient call takes ~95 seconds and ~5 GB, and
  still declines.
- At 22,653 vertices — the low end of the stated typical range — no
  gradient call of either family completed within a 15-minute box.
- 200,000-600,000 vertices is not within reach of the monolithic path
  by any margin.

The honest summary: the programme delivers exact gradients on toy and
small-coalescent models, which is where all of its validation lived,
and does not currently deliver them on the models the library exists to
serve.

## 3. Why this was not caught earlier

Not because of sloppiness in any individual batch — the reviews were
genuinely adversarial — but because of a scoping gap that no batch was
responsible for noticing:

1. The scale requirement was never written down. It is not in
   CLAUDE.md, the master plan, or any batch plan. Now recorded in both
   CLAUDE.md and memory.
2. Every gate was a correctness gate. "Zero new failures against the
   pinned ledger" and per-batch oracles say nothing about whether the
   feature runs on a production-sized model.
3. The conditioning-gate sweep (CC-2, which led to parking Deferred 4)
   used toy fixtures, where the gate is conservative but harmless. On a
   real two-locus graph it bites. The park decision was therefore taken
   on unrepresentative evidence.
4. Batch H found this exact problem for the sojourn family — "the gate
   declines 100% of realistic coalescent-scale calls" — and shipped an
   opt-out. Nobody asked whether the moments family had the same
   disease. It does.

Point 4 is the most uncomfortable: the evidence was already in the tree
and was not generalised.

## 4. What remains under the current plans

Deferred 1, hierarchical/SCC two-level adjoint. De-risk complete and
all-GO; implementation plan drafted (v4) and hardened by two refuters;
not started. This is the only designed route to exact gradients above
about 10k vertices. Its plan now carries re-scoped gates (parity where
the monolithic oracle exists, feasibility across 20-60k, a measured
extrapolation toward 600k) and an upgraded design question: retain all
per-SCC tapes versus recompute them on demand, since retain-all
plausibly does not fit at the target scale.

Deferred 2, daisy intermediate-epoch exact gradient. De-risk complete.
Both open design decisions answered by measurement: lambda pinned at
construction, and checkpointed-reverse rather than forward-mode (the
crossover is passed at four samples). The blocking JAX composition risk
is retired. No implementation plan written. Its build case is a crash,
not accuracy: below theta~1e-7 the shipped backward drives a probe
negative and the FFI refuses.

Deferred 3, exact PMF/PDF-term gradient. De-risk complete. Route
selected by measurement (Poisson mixture at pinned lambda; the
alternative was refuted). Derivation dossier written and verified
term-by-term. Its value test came back "not immaterial" at mixed scale
(335x error) and "nothing to gain" at benign scale. No implementation
plan written.

Deferred 4, MPFR conditioning floor. Parked with sign-off — but section
2 above reopens its cheap end. Not the full MPFR adjoint, which remains
unjustified, but a moments-family gate opt-out mirroring the one the
sojourn family already ships.

Unscheduled ledger (master plan section 16b): fourteen items, of which
these remain open — 1 (documentation overstating parallel_elimination),
4 (rate-blowup forward/backward inconsistency), 5 (moments_from_graph
and method_of_moments are still FD-only paths), 6 (the hierarchical
composer silently recomputes weights as linear, so log-mode plus
parallel elimination is a silent wrong answer), 7 (a pmf_from_cpp
callback issue), 8 (daisy FFI swallows context-create failures as an
unlogged NaN row), 11 (analytic-derivative-callback opt-in), 12 (pybind
scc_decomposition dangling-temporary footgun), 13 (the new guard has no
decline latch, and declined moments calls still pay a full tape build),
14 (a serialise round-trip drops the synthetic marker).

Item 6 deserves promotion in light of the scale finding: it is a silent
wrong answer on exactly the path large models must use.

## 5. The alignment question, stated plainly

The programme optimised for correctness-in-the-small with unusual rigor
and never asked whether the result runs at production size. Three
things follow, and they are the substance of the Q&A:

- Reach versus exactness. If typical models cannot use exact gradients
  at all, then reach — making them run — dominates exactness on
  quantities that already work at small size. That ordering would put
  the conditioning-gate opt-out first (cheap, immediate), Deferred 1
  second (the only scale route), and Deferred 2/3 behind them.
- What "works" means. A 95-second gradient at 8k vertices is either
  fine or fatal depending on whether it is called once or thousands of
  times inside an SVGD loop. Nothing in the programme's records states
  the iteration budget for a real fit.
- Which quantities actually matter. The programme spread effort evenly
  across moments, sojourn, PMF, and epochs. It is not recorded which of
  these real analyses depend on, so the effort allocation was never
  checked against use.

## 6. Documents updated alongside this one

- CLAUDE.md — new "Scale targets" section, stating the ranges and the
  measured consequence, plus the memory-measurement safety rule.
- Memory — project_model_scale_targets (new), and the B3 project memory
  updated with the guard, the D1 plan status, and the D2/D3 outcomes.
- b3-scale-reality-findings.md — the measurements, the gate diagnosis,
  and the macOS rlimit correction.
- b3-d1-implementation-plan.md — re-scoped gates, upgraded retention
  decision, corrected memory protocol.
- b3-execution-tracker.md — deferred-unit rows brought current.
