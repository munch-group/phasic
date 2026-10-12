# phasic — project overview

Written 2026-10-10 as the single entry point to the ~70 plan, findings and
review files in the repository root; updated 2026-10-12 with the October
model-building consistency work (§1d) and the Stage-3 status correction. Everything here is a summary of those
files (named in each entry) plus git history; nothing new was measured to
write it. When a file and this overview disagree, the file is the record —
and this overview should be corrected.

Convention from here on: every ID (batch letter, deferred number, tier,
gate, rule) is written together with a few words saying what it is.

---

## 1. Where the project stands (October 2026)

Master is pushed and unchanged since 17 August 2026. New work since
10 October is on branch **`fix/constant-parameterized-consistency`**
(not pushed, not merged; see §1d). Uncommitted in the working
tree: your re-run tutorial notebook outputs and edits, and your
test-design notebook `tests/pytest/notebook_model_recovery.ipynb`.

### 1a. The question the project is currently about: does inference recover the truth?

Work in mid-August turned from "make gradients exact" to "does SVGD fitting
actually recover known parameter values on the tutorial models". The test
file `tests/pytest/test_notebook_model_recovery.py` fits five models derived
from the `docs/pages/tutorial/svgd-*` notebooks to data simulated at known
parameters, from priors that exclude the true value. Four cases fail and are
recorded as expected failures ("pins"). Whether each failure is a real
defect or just too few iterations was settled in
`b3-budget-vs-defect-findings.md`:

| Model | What goes wrong | Verdict |
|---|---|---|
| Coalescent, one rate (43 vertices) | Point estimate fine, but the posterior is too narrow and misses the truth; more iterations make it narrower, not better | Defect |
| Joint-probability coalescent (168 vertices) | Stuck ~54% below the true rate; particles pile up at the 1e-9 lower bound | Defect |
| Two-locus ARG, coalescence + recombination | Coalescence recovered; recombination ~86% off, and more compute shrinks the interval around the wrong value | Defect |
| Two-island (186 vertices) | Off at the suite's budget, steadily improving with more iterations | Budget-limited, not a defect |

The mechanism behind the three defects is not known. Candidates named in the
records: kernel bandwidth, preconditioner, step-size schedule.

Two further facts limit how precise any fit can be:

- **The default likelihood is only ~2.5e-3 accurate** (relative), and that
  error biases the estimate by the same amount regardless of sample size.
  Cause: the default discretisation granularity (1024 steps) in the
  continuous PDF. Record: `b3-mixed-scale-defect-settled.md`; guarded by
  `tests/pytest/test_likelihood_correctness.py`.
- **Recombination is only identifiable at larger rates**: at 0.05 the
  likelihood is flat over a 25-fold range, so the test uses 0.5.

Your notebook (`tests/pytest/notebook_model_recovery.ipynb`, untracked) is a
first draft of a new test design with MAP-accuracy and trajectory-smoothness
checks. It was copied from an older version of the test file, so it carries
outdated wording for the pins.

### 1b. The exact-gradient programme ("B3") — finished, but aimed at the wrong target

From late July to mid-August most effort went into replacing
finite-difference gradients (estimating a derivative from nearby
evaluations, which fails when parameters differ hugely in scale, e.g. 1e-8
next to 1) with exact analytic gradients. Every planned unit of that work
shipped and passed review. A stock-take on 16 August
(`b3-stocktake-2026-08-16.md`) then found two problems:

1. **Scale.** Production models are 20k–60k vertices, and the library must
   handle 200k–600k. The exact gradients were only ever tested up to ~8k
   vertices. At 22.6k vertices no gradient call finished within 15 minutes;
   on two-locus models a safety check makes the exact path quietly hand
   back finite differences anyway, already at ~1k vertices.
   Record: `b3-scale-reality-findings.md`.
2. **Wrong quantity.** The programme made the gradient of *moments* exact,
   but the default SVGD likelihood uses the *PDF at the observed times*,
   whose gradient is still finite differences everywhere in the library.
   Record: `b3-likelihood-path-map.md`.

### 1c. Open items, in plain language

1. Explain the three inference defects in 1a. Not started.
2. **Exact gradients for big models, by splitting them into components**
   ("Deferred 1", plan `b3-d1-implementation-plan.md`). Large graphs are
   already processed as strongly connected components (clusters of states
   that can reach each other), which are then stitched together. This plan
   would make the gradient follow the same split. It is the only designed
   route to exact gradients above ~10k vertices. Exploratory experiments
   all came back positive. The plan is written and reviewed but **not
   signed off by you**. Its header says v3; the stock-take calls it v4.
3. **Two cheap fixes, not built:**
   - Let the moments gradient skip its numerical-conditioning safety check.
     That check is what makes it fall back to finite differences on
     two-locus models, and the sojourn-time gradient already has such an
     opt-out.
   - Cache the moments recording (tape) between calls. Measured cost
     difference: 3,062x.
4. **Recombination recovery from richer data** (several summary statistics,
   or joint probabilities, instead of only absorption times). Not yet made
   to converge.
5. **Exact gradient of the PDF itself** ("Deferred 3",
   `deferred-3-pdf-gradient-revival-plan.md`, `b3-d3-derisk-findings.md`).
   This is the only item that would make the *default* fit exact. The
   method is chosen (Poisson-mixture route) and derived on paper. No
   implementation plan yet.
6. **Exact gradient for the earlier time epochs** of time-inhomogeneous
   models ("Deferred 2", `deferred-2-daisy-intermediate-epoch-plan.md`,
   `b3-d2-derisk-findings.md`). Exploratory work is done. The main reason
   to build it: the current backward pass crashes for parameters below
   ~1e-7. No implementation plan yet.
7. **High-precision (MPFR) gradients** ("Deferred 4"). Parked by your
   decision. Its cheap end is item 3a above.
8. **Loose ends** (the master plan's §16b list). Most important: log-weight
   models combined with parallel elimination give a **silently wrong
   answer**, because the parallel composer always treats weights as linear
   (§16b item 6). Others: documentation overstating parallel-elimination
   benefits, a forward/backward mismatch when a rate blows up, and failures
   in the multi-epoch FFI that become unlogged NaNs. Full list in
   `b3-stocktake-2026-08-16.md` §4.
9. **Leftovers outside the gradient programme** (end of
   `b3-execution-tracker.md`):
   - `model-selection.ipynb` uses an outdated, broken pattern.
   - A random tie-break in the epoch model is untested.
   - The old cache format needs hardening.

10. **Finish the model-building consistency branch** (§1d): one full-suite
    run on the branch (needs your go-ahead), then your merge and push.
11. **Notebook edits for you** (§1d): five notebooks use patterns the branch
    now refuses or changes.
12. **One `discretize()`, owned by C++** (`discretize-cpp-ssot-plan.md`, v2,
    reviewed, all nine decisions made): the next piece of the Stage-3
    refactor (§1e). Starts after item 10.

### 1d. Model-building consistency (10–12 October)

Started from your question why a coalescent built with float rates came
back parameterized. Plan `constant-parameterized-consistency-plan.md`
(record in its §13), with Part 1 detailed in
`graph-cache-constant-roundtrip-plan.md`. Branch
`fix/constant-parameterized-consistency`, one commit per part:

| Part | What it does | Commit |
|---|---|---|
| Plans | the two plans + this overview | `50f42033` |
| 1 — cache round trip | constant graphs no longer come back parameterized from `serialize()`/the graph cache (they were silently θ-scaled); cache version 1.0 → 1.1, old cache deleted | `553dc412` |
| 2 — one guard | every entry point that takes θ (svgd, mcmc, method_of_moments, joint_* builders, add_epoch, …) refuses a constant model with one message | `e7b7679a` |
| 3a — edge arrays | memory-safety fix: four rate edges + `add_edge_constant` + one more edge on a vertex crashed Python | `54533414` |
| 3 — lossless widening | layout widening no longer drops edges, merges aux vertices, truncates coefficients or drops metadata; refuses where undefined (non-linear weight modes, ragged coefficients, constant edges in discretized/joint graphs, epochs on discrete graphs) | `2181b9ff` |
| 4 — discretize callback | `discretize(rate=callable)` returns one number for any graph; on a parameterized graph it goes into a new θ slot | `f67e175d` |

Every part passed its own new tests and the existing tests touching it; an
adversarial diff review ran for each (for Parts 2 and 3a only after
committing — recorded). Not yet done: one full-suite run on the finished
branch.

**Notebooks to edit (yours):** `parametrization.ipynb` cell 23 (now
raises: drop the reserved `[rate, 0]` slot, callback returns
`nr_lineages`); `joint-probability.ipynb` cell 3 and
`time-inhomogeneous.ipynb` cell 32 (rates as lists); `popgen/coalescent-derived.ipynb`
cells 8→26 and `popgen/old-moran-coalescent-derived.ipynb` cells 19→37
(`joint_prob_graph` on a discretized graph; never run, already broken
before the branch).

**Left open (recorded in the plan's §13):** constant edges in discrete and
joint graphs are refused rather than supported (needs a C change);
`joint_prob_graph(reward_rates_callback=...)` has been broken since
21 May 2026 (calls a commented-out method); per-state rate APIs not yet on
the new rules (legacy `_discretize_inplace`, C++ `discretize`,
`joint_prob_graph`'s `mutation_rate`); `expectation()` on a discretized
graph counts all steps, the mutation count needs `graph.rewards` passed
explicitly.

### 1e. The Stage-3 refactor is only partly done

The July refactor plan (`stage3-refactor-plan.md`) is the larger plan in
force: one implementation per capability, with the C++ API as the single
source of truth (its decision Q4) and Python delegating. Executed so far:
WS-A (safety-net test gates), WS-B (dead-code removal) and part of WS-C
(splitting the `Graph` class), merged at `9de5d61a`. Not started: WS-D
(thinning the pybind layer), WS-E (the C++ single-source promotions), WS-F
(splitting the C core), WS-G (doctrine document and CI guards). Item 12
above is WS-E's step 11 (discretize), proposed as its first promotion.

---

## 2. How the work unfolded (timeline)

| When | Work stream | What it was | Status |
|---|---|---|---|
| early Jul | Hardening fixes (H1–H10, C1–C3) | Ten bug fixes and three cache-security fixes (untrusted pickles, tar members) | Done |
| 5–9 Jul | Refactor, Stages 1–3 | Remove dead code; build a test safety net; split the 12k-line `Graph` class into modules | Stages 1–2 done; Stage 3 partly (WS-A, WS-B, part of WS-C; merged `9de5d61a`); WS-D–G not started (§1e) |
| 15–19 Jul | Differential audit | Compare post-refactor behaviour with the pre-refactor commit `3082ebc6`; log every divergence | Done; findings F-001… in `audit-differential-findings.md` |
| 17 Jul | Category-A fixes (A1–A3) | The three audit bugs reachable by users | Done |
| 27–30 Jul | SVGD robustness + likelihood-ratio tests | Stop runaway particles from crashing a fit; make nested LRTs work for tied-vs-free epoch models; public `Graph.epoch_model` | Done |
| 30 Jul–4 Aug | B3 exploration | Prototype exact gradients; first exact moments gradient (continuous, then discrete); log weight mode; joint-index (sojourn) gradient | Done |
| 4 Aug | Exact/FD atlas | Map of which code paths have exact gradients and which `Graph.svgd()` can actually reach | Reference |
| 11–15 Aug | B3 master plan, Batches 0, A–H | Systematic rollout of exact gradients across weight modes, rewards and SVGD routes | All shipped |
| 15–16 Aug | Deferred 1–4 exploration | Experiments for the four big postponed items | Done; decisions pending |
| 16 Aug | Stock-take + scale check | Discovery that B3 doesn't reach production size or the default likelihood | See 1b |
| 16–17 Aug | Inference correctness | Recovery tests on the tutorial models; budget-vs-defect verdicts | See 1a |
| 10–12 Oct | Model-building consistency | Constant vs parameterized models, lossless widening, one-number discretize callback, edge-array memory fix | On branch, full suite pending (§1d) |
| 12 Oct | discretize → C++ plan | One C++ `discretize()` + widening and layout standards | Plan v2, decisions made (§1c item 12) |

---

## 3. Label decoder

Several schemes reuse the same letters. **Collisions are marked ⚠.**

| Label | Meaning | Defined in |
|---|---|---|
| **B3** | The whole exact-gradient programme. Originally step 3 of a five-step plan B0–B4 for fixing finite-difference gradients (B0 pin the bug with a test, B1 guard against negative rates, B2 build a verification harness, B3 replace the FD backward passes, B4 adversarial verification) | `b3-derisking-strategy.md`; decoder in `tmp.md` |
| **Batch 0** | Merge three near-identical C gradient functions into one shared core | `b3-batch0-plan.md` |
| **Batch A** ⚠ | (Aug) Exact moments gradient with `rewards`. ⚠ In July "Batch A / A1–A4" was an unrelated audit-fix batch, and "A1–A3" are also the Category-A bug fixes | `b3-batchA-plan.md` |
| **Batch B / C** | Exact gradient for `weight_mode='formula'` / `'callback'` | `b3-batchB-plan.md`, `b3-batchC-plan.md` |
| **Batch D (D.1–D.4)** ⚠ | D.1 fix a crash of `moments_from_graph` under `jax.vmap`; D.2 add safety checks to a debug-only C function; D.3 pass an option through to joint-probability SVGD (cancelled, folded into G.1); D.4 pass `exact_moment_grad` through `Graph.svgd()`. ⚠ Not the same as D1 = Deferred 1 | `b3-batchD-tier1-plan.md`, `b3-batchD3-plan.md` |
| **Batch E** | Exact gradient for joint-probability models with de-duplicated observations ("baked" mode) | `b3-batchE-plan.md` |
| **Batch F** / **D6** | Remove a JAX construct (`lax.cond`) that, under `vmap`, computed both the exact and the FD gradient. D6 was its name inside the joint-index plan | `b3-batchF-plan.md` |
| **Batch G (G.1, G.2)** ⚠ | Expose exact-gradient options through `Graph.svgd()` (G.1: multi-epoch final epoch; G.2: multi-feature rewards). ⚠ Not gates G0–G5, nor audit findings G1/G2 | `b3-batchG1-plan.md`, `b3-batchG2-plan.md` |
| **Batch H** ⚠ | Exact gradient for the final epoch of multi-epoch (daisy-chain) models. ⚠ Not hardening fixes H1–H10 | `b3-batchH-plan.md` |
| **Batch 1–3** (Jul) ⚠ | Early B3 steps: tape adjoint, first exact gradient, higher moments. ⚠ "Batch 1/2" also name the LRT fix and the `epoch_model` builder | `b3-batch{1,2,3}-*-findings.md`, `batch2-free-epoch-model-plan.md` |
| **Deferred 1 (D1, Def-1)** | Exact gradients for big models split into strongly connected components | `deferred-1-…-plan.md`, `b3-d1-*.md` |
| **D1-E2 guard** | Small shipped fix from Deferred 1 experiment E2: exact-gradient code now refuses placeholder "synthetic" component graphs, where it used to return plausible but wrong numbers | `b3-d1-e2-guard-plan.md` |
| **Deferred 2 (D2)** | Exact gradient for the earlier epochs of multi-epoch models | `deferred-2-…-plan.md`, `b3-d2-derisk-findings.md` |
| **Deferred 3 (D3)** | Exact gradient of the PDF/PMF at observed times | `deferred-3-…-plan.md`, `b3-d3-*.md` |
| **Deferred 4 (D4)** | High-precision (MPFR) gradients for badly conditioned cases; parked | `deferred-4-…-plan.md`, `b3-d4-sweep-findings.md` |
| **E0–E5** | Numbered experiments *inside* each deferred plan; the numbers restart in every plan | the deferred plans |
| **CC-1, CC-2** | "Cheap checks" in the master plan. CC-2 = the conditioning sweep that led to parking Deferred 4 | `b3-exact-gradient-master-plan.md` §15 |
| **Phase 1–4** ⚠ | (B3) Ordering groups of batches in the master plan; all complete. ⚠ In July: the four phases of the numerical audit (forward values, gradients, invariants, weak tests) | master plan §15 / `tmp.md` |
| **Tier 1–3** ⚠ | (B3) Three candidate strategies: exact differentiation via the trace / smarter FD step (abandoned) / C adjoint (chosen). ⚠ In the recovery tests, Tiers 0–2 mean sampler check / likelihood-peak check / full SVGD fit. ⚠ "Batch D Tier-1" and "SVGD plumbing Tier 3" mean yet another thing | `b3-derisking-strategy.md`; test file docstring |
| **G0–G5** | Checklist every batch passed: baseline current, own tests, targeted tests, full suite vs ledger, adversarial diff review, merge record | `b3-execution-process.md` §4 |
| **§16b items 1–14** | The master plan's list of unscheduled loose ends | master plan §16b |
| **R9, R29–R32** | Numbered argument-validation rules in `svgd_config.py` that reject an option where it would have no effect | `src/phasic/svgd_config.py` |
| **Leaf 1–5** | Branches of `Graph.svgd()`'s internal routing: 1 multi-epoch, 2 joint-probability, 3 multi-feature rewards, 4 single rewards, 5 plain moments | `atlas/exact-fd-atlas-svgd-reachability.md` |
| **DR-A, DR-D** | Early "de-risk" experiments: DR-A an analytic gradient oracle, DR-D a scale-aware FD step (dead end) | `b3-experiment-findings.md` |
| **F-001…** ⚠ | Audit findings. ⚠ Also "F1–F7" (audit phase-1 findings), "F-P1…" (reviewer findings inside plans) | `audit-differential-findings.md` |
| **N1–N12, G1–G2, P2-x** | Defects found during the July audit (N), its headline findings (G), its early claims (P2) | `tmp.md` |
| **WS-A … WS-G** | Workstreams of the Stage-3 refactor: A safety-net test gates, B dead-code removal, C splitting Python god-objects (the Graph class, svgd.py), D thinning the pybind layer, E C++ single-source promotions, F splitting the C core, G doctrine document + CI guards | `stage3-refactor-plan.md` §7 |
| **Stage-3 steps 0–20** | The refactor's ordered commit sequence; step 11 = discretize to C++ | `stage3-refactor-plan.md` §8 |
| **Q1–Q13, Q-CODEGEN** | Decisions recorded in the refactor plan; Q4 = "C++ API is a product, its ports become the single source of truth"; Q13 = freeze the public C symbols | `stage3-refactor-plan.md` §9 |
| **Part 1, 2, 3a, 3, 4** | The model-building consistency work: cache round trip / one guard for constant models / edge-array memory fix / lossless widening / one-number discretize callback | `constant-parameterized-consistency-plan.md` |
| **P1–P4, O1–O8** | Widening standard: P = what the shared widening primitive guarantees; O = what each operation (discretize, add_epoch, joint_prob_graph) must check, change and record | `discretize-cpp-ssot-plan.md` §3 |
| **Fix A / C / D** | Fixes for the nested likelihood-ratio test on epoch models | `svgd-lrt-fix-a-plan.md` |
| **Pins 1–4** | The four expected-failure recovery tests in 1a | `b3-budget-vs-defect-findings.md` |
| **Ledger stamps** (e.g. "11th stamp 2012/0/84/24") | Recorded full-suite results: passed / failed / skipped / expected-fail | `b3-test-baseline.md` |
| **SOUND-WITH-CORRECTIONS** etc. | Verdicts from adversarial review agents | — |

---

## 4. File index

**Current = read for present work. History = record of finished work; keep, but no need to read.**

### Start here
| File | What it is | |
|---|---|---|
| `PROJECT-OVERVIEW.md` | This file | Current |
| `CLAUDE.md` | Build, test, architecture notes and the long list of known gaps | Current |
| `b3-stocktake-2026-08-16.md` | Is the gradient work aligned with what the library is for? (No: scale and wrong quantity) | Current |
| `b3-budget-vs-defect-findings.md` | Verdicts on the four failing recovery tests | Current |
| `b3-likelihood-path-map.md` | What each inference method actually computes, traced through the source | Current |

### Inference correctness (Aug 16–17)
| File | What it is | |
|---|---|---|
| `b3-mixed-scale-defect-settled.md` | FD gradient error at mixed scales is smaller than thought; the 2.5e-3 likelihood error is the bigger problem | Current |
| `b3-scale-reality-findings.md` | Measurements of the exact gradients at 1k–22k vertices; tape growth per model type | Current |
| `b3-inference-accuracy-table.md` | Generated table: true vs inferred parameters with seven convergence checks | Current |

### Exact-gradient programme (B3): overview and process
| File | What it is | |
|---|---|---|
| `b3-execution-tracker.md` | Status table for every batch (dense; see decoder above) | History, partly current |
| `b3-exact-gradient-master-plan.md` | 1,482-line plan sequencing Batches 0, A–H and Deferred 1–4 | History |
| `b3-execution-process.md` | Branching, review and test-gate rules used for every batch | Reference |
| `b3-test-baseline.md` | Full-suite results ledger | Reference |
| `exact-fd-atlas-SUMMARY.md` + `atlas/` (13 files) | Map of exact vs FD gradient coverage, and feasibility studies | History |
| `b3-derisking-strategy.md` | The original exploration plan for replacing FD gradients | History |

### B3: per-batch plans and findings (all shipped)
`b3-batch0-plan.md`, `b3-batchA-*`, `b3-batchB-*`, `b3-batchC-*`,
`b3-batchD-tier1-plan.md`, `b3-batchD3-plan.md`, `b3-batchE-*`,
`b3-batchF-plan.md`, `b3-batchG1-*`, `b3-batchG2-*`, `b3-batchH-*` —
meanings in the decoder. History.

### B3: early work, late July – early August (shipped)
`b3-experiment-findings.md`, `b3-prototype-findings.md`,
`b3-c-theta-adjoint-plan.md`, `b3-batch0-realtape-findings.md`,
`b3-batch1-reverse-adjoint-findings.md`, `b3-batch2-exact-grad-findings.md`,
`b3-batch3-higher-moments-findings.md`, `b3-batch3-mpfr-and-discrete-derisk.md`,
`B3-MERGE-REVIEW.md` (first exact moments gradient),
`b3-discrete-theta-adjoint-plan.md` + `B3-DISCRETE-MERGE-REVIEW.md`
(discrete models), `b3-log-weight-mode-plan.md`, `b3-joint-index-plan.md`
(sojourn-time gradient). History.

### B3: deferred items (open decisions)
| File | What it is | |
|---|---|---|
| `deferred-1-hierarchical-scc-adjoint-plan.md`, `b3-d1-derisk-findings.md`, `b3-d1-implementation-plan.md` | Component-split gradients for big models: exploration plan, results, implementation plan awaiting sign-off | Current |
| `b3-d1-e2-guard-plan.md` | The shipped synthetic-graph guard | History |
| `deferred-2-daisy-intermediate-epoch-plan.md`, `b3-d2-derisk-findings.md` | Earlier-epoch gradients | Current |
| `deferred-3-pdf-gradient-revival-plan.md`, `b3-d3-derisk-findings.md`, `b3-d3-e5-derivation-dossier.md` | PDF gradient: plan, results, derivation | Current |
| `deferred-4-mpfr-conditioning-floor-plan.md`, `b3-d4-sweep-findings.md` | High-precision gradients (parked) | History |

### Model-building consistency and the C++ discretize (October)
| File | What it is | |
|---|---|---|
| `constant-parameterized-consistency-plan.md` | Parts 1–4: constant vs parameterized models, lossless widening, discretize callback; §13 is the implementation record | Current |
| `graph-cache-constant-roundtrip-plan.md` | Part 1 in detail: constant graphs coming back parameterized from the graph cache | Current |
| `discretize-cpp-ssot-plan.md` | One `discretize()` owned by C++, with the widening and layout standards; v2, decisions made | Current |
| `stage3-refactor-plan.md` | The larger refactoring plan still in force (doctrine, workstreams, commit sequence, decisions) | Current |

### Refactor and audit (July)
`stage2-coverage-safety-net-handoff.md`,
`stage3-execution-handoff.md` (what of Stage 3 was executed), `deferred-notebook-triage.md`,
`audit-situation-map.md` (also the map of all compute paths — still useful),
`audit-differential-findings.md`, `category-A-fix-plans.md`. History.

### SVGD robustness and model selection (late July)
`deferred-svgd-divergece-fix.md`, `svgd-divergence-robustness-plan.md`,
`deferred-svgd-lr-bug.md`, `svgd-lrt-model-reuse-plan.md`,
`svgd-lrt-fix-a-plan.md`, `batch2-free-epoch-model-plan.md`. History.

### Other
| File | What it is |
|---|---|
| `sojourn-fix.md` | Cut memory of sojourn-time computation from O(n·k) to O(n) (shipped) |
| `tree_toplogy_encoding.md` | Encoding tree topologies as states |
| `tmp.md` (git-ignored) | An earlier decoder of the July audit labels |
| `experiments/` (58 scripts) | Scripts behind the findings files (`dr_*.py`) |

---

## 5. Suggested tidy-up (not done)

- Move every "History" file into `docs/history/` (or `notes/archive/`), so
  that the repo root holds only this overview, `CLAUDE.md`, `README.md` and
  the "Current" files.
- Give new plans descriptive names (for example `pdf-gradient-plan.md`)
  instead of letter-number IDs.
- Trim the B3 sections of `CLAUDE.md`, which are now mostly history, to a
  pointer to this file.
