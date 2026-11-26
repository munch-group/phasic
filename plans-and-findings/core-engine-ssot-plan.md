# Core-engine refactor: C/C++ as the single source of every core computation

**Status: v1, 2026-11-26. NOT reviewed. No code changed.** This is a
master plan: each phase gets its own detailed sub-plan, reviewed before
any code (as the exact-gradient master plan did). It supersedes the
unexecuted parts of [stage3-refactor-plan.md](stage3-refactor-plan.md)
(July 2026), which stays as history; its doctrine and the decisions listed
in §2 carry over.

## 0. Copy-paste prompt for a fresh session

> Read `plans-and-findings/core-engine-ssot-plan.md` in full, then
> `OPEN-ITEMS.md` (Priorities section) and `CLAUDE.md`. Find the first
> phase whose status is not DONE. If it has no reviewed sub-plan, write one
> in `plans-and-findings/core-ssot-p<N>-<name>-plan.md` and put it through
> adversarial review (one Opus reviewer per round; after a clean sign-off,
> two more independent reviewers) before any code. Implement in a worktree
> on its own branch; gate with the suite comparison (no new failures) plus
> the phase's bit-identity gates; merge with `--no-ff`; run
> `pixi run install-dev` in the main tree; update this plan's status and
> `OPEN-ITEMS.md`. Memory-cap every run; one heavy run at a time; never run
> reviewers alongside test runs. Never edit notebooks (list cells for the
> user). Never `git add -A`. The user pushes.

## 1. Goal and definition of done

**Goal (user, 2026-11-26):** every core computation has exactly one
implementation, in C or C++, so the engine is easier to maintain and to
develop safely. Python delegates to it. The inference layer stays Python.

**Core computation** = everything that builds, transforms or evaluates a
phase-type graph:

- state-space construction from a callback;
- state layout and encoding (the index codec, §2 R2);
- the joint-probability family (`joint_prob_graph`, `joint_stop_prob_graph`,
  `joint_sojourn_graph`, `joint_prob_table`) and the graph construction of
  the epoch (daisy-chain) models;
- discretize, add_epoch, layout widening, reward and Laplace transforms;
- moments, expectation, variance, covariance; pdf/cdf/pmf and other
  distribution read-outs; sampling;
- serialize / from_serialized and the cache file formats;
- weight formulas (compile and evaluate);
- elimination (all variants), SCC decomposition and composition, the FFI
  compute handlers.

**Stays Python:** SVGD, MCMC, priors, `svgd_config`, `probability_matching`,
`method_of_moments`, model selection, plotting; the JAX wrapping
(`custom_vjp`, `pure_callback`, the FD backward rules of Plan A) and the
JAX trace REPLAY (jit/grad need it; the July plan's sanctioned dual path);
orchestration and infrastructure (the graph file cache and its key,
SLURM/distributed orchestration, hierarchical cache management, `profile`
as an advisory tool).

**Done means:**

1. Every row of the §3 scope table marked "C/C++" has one implementation,
   bound to Python, with Python delegating; the other copies are deleted.
2. The public Python API is unchanged (same names, signatures, results),
   except where a row says otherwise and the user approved it.
3. Every move was gated bit-identical (or within a listed, approved
   tolerance; §4).
4. A C++ user can build any model the Python API can, including a joint-
   probability model from an index-codec layout, and hand a parameterized
   graph to Python (§2 R8).
5. The doctrine document and CI guards exist (phase 11).

## 2. Decisions (recorded)

From the user, 2026-11-26 ("otherwise I will go with your recommendations"):

- **R1** A new master plan (this file) replaces the July plan's remaining
  workstreams.
- **R2** StateIndexer, option B: C++ owns layout and encoding (an
  `IndexCodec`: mixed-radix encode/decode, slot offsets, append/entity
  order, loading from the existing `StateIndexer.to_dict()` format). The
  Python `StateIndexer` keeps its user API (Property objects, dict views,
  `StateVector`) and delegates the rules to the codec. A full C++
  StateIndexer (option A) can grow from this later if C++ users need it.
- **R3** Weight formulas: the C++ compiler becomes the single source;
  Python `compile_formula` delegates (it may survive as a test oracle).
- **R4** The generated-C++ moments channel (`moments_from_graph` writing and
  compiling per-model C++) is retired: `moments_from_graph` keeps its
  signature and computes through the compiled library.
- **R5** `profile` stays Python (advisory, not a computation); the unbound
  C++ `Graph::profile` port is deleted.
- **R6** No deliberate numeric change during the refactor. Plan B (true
  uniformization) and exact PDF gradients come after it.
- **R7** `pmf_from_cpp` and `load_cpp_builder` (user-written C++ models)
  keep working through the refactor, including on a normal install
  (today they compile the whole phasic C/C++ source into each model and
  fail without a source checkout; OI-058).
- **R8** New deliverable: a parameterized C++ model route. A user's C++
  code builds a parameterized `phasic::Graph` once and hands it to Python,
  where the full machinery applies (FFI, exact gradients, caching, svgd).
- **R9** After the refactor, decide whether the old per-theta C++ route
  (rebuild the graph for every theta, FD gradients) is still worth keeping.
- **R10** Plan A ([fd-domain-steps-plan.md](fd-domain-steps-plan.md)) is
  done before phase 1.

Carried over from the July plan (still valid): the nine-rule delegation
doctrine (§2 there); complete-then-promote for every port (reach parity →
bind and gate → delegate and delete, each its own commit); the public C
symbols are frozen (no removals without approval; additions fine); keep the
`_mpfr` elimination variant; persistent per-thread FFI graphs.

Decided by the user, 2026-11-26:

- **R11 (phase 3)** The moment recurrence lives in `phasic::Graph::moments`
  (the owner over the C waiting-time kernel; already equivalent to the
  bound copy). The pybind `_moments` family, `GraphBuilder::compute_moments_impl`
  and the generated channel delegate to it or are deleted.
- **R12 (phase 9)** Elimination traces are recorded in C (the GraphBuilder
  engine already handles cycles and formula mode, which the Python
  recorder refuses; OI-044). Python keeps only the JAX replay of a
  recorded trace; the Python recorder in `trace_elimination.py` is deleted
  once the C recording feeds every consumer (hierarchical cache, JAX
  replay).

Open, to be settled in the named phase's sub-plan (recommendation given):

- **O3 (phase 8)** How user C++ models build on a normal install: link
  against an installed phasic library plus shipped headers (recommended:
  no per-model recompile of the whole engine), or ship the sources.

## 3. Scope table (re-baselined on master `f4eeea41`, 2026-11-26)

| Capability | Implementations today | Target | Phase |
|---|---|---|---|
| State-space construction from a callback | four builders in `phasic_pybind.cpp:431-660` (plain/parameterized × build/extend) + `phasic::Graph::from_callback` / `extend` (`phasiccpp.cpp:196`, `:414`) | one: `phasic::Graph::from_callback` / `extend`, bound | 2 |
| Moments / expectation / variance / covariance | pybind `_moments` family (bound, `phasic_pybind.cpp:1832`); `GraphBuilder::compute_moments_impl` (`graph_builder.cpp:644`); `phasic::Graph::moments` (`phasiccpp.h:731`, unbound); generated C++ string (`moments_from_graph`, `__init__.py:7873`) | `phasic::Graph::moments` (R11); GraphBuilder delegates; generated channel retired (R4) | 3 |
| Matrix export / import | bound lambdas (`phasic_pybind.cpp:3295`, `:3381`); `phasic::Graph::as_matrices` / `from_matrices` (unbound); orphan Eigen class | `phasic::Graph` methods, bound | 3 |
| State layout and encoding | Python `StateIndexer` / `PropertySet` (`state_indexing.py`, 2,266 lines) | C++ `IndexCodec`; Python ergonomics delegate (R2) | 4 |
| Joint-probability family | Python: `joint_prob_graph` (`__init__.py:10703-11342`), `joint_stop_prob_graph` (`:11343`), `joint_sojourn_graph` (`:11548`), `joint_prob_table` (`:12366`) | C++, using the codec; Python wrappers keep signatures | 5 |
| Epoch (daisy-chain) graph construction | Python, inside `_daisy_chain_svgd_model` (`__init__.py:4665`) and `daisy_chain_joint_probs` | graph construction in C++; JAX model wrapping stays Python | 5 |
| serialize / from_serialized | Python (`_graph_serialize.py`), `GraphBuilder::parse_structure`, `phasic::Graph::serialize` / `from_serialized` (`phasiccpp.cpp:1317`, `:1407`; lacks `is_aux`) | one C++ owner + schema version constant; GraphBuilder reads the same schema | 6 |
| Elimination-cache file formats | rev-3 loader (hardened), rev-1/2 loader (unhardened, OI-079), per-SCC files | rev-3 only (or rev-1/2 hardened), invariants enforced in code (OI-087) | 6 |
| add_epoch | Python (live, `__init__.py:3246`); `phasic::Graph::add_epoch` port (unbound) | C++ port promoted after parity (OI-025) | 7 |
| Weight formulas | Python `compile_formula` (live) + C tape VM (evaluator); C++ `wf_*` compiler (`phasiccpp.cpp:937-1316`, unbound, untested) | C++ compiler single source (R3), C VM evaluator; semantics of OI-048 decided | 7 |
| FFI compute handlers | 9 handlers in `graph_builder_ffi.cpp` re-implementing the times loop | one shared `compute_*_core` both pybind and FFI call; failures reported (OI-028) | 7 |
| discretize, widening, theta layout | C++, bound (`_discretize`, `_widen_layout`, `_theta_layout`) | done; residual parity fix OI-033 | — |
| Reward / Laplace transforms, sampling, pdf/cdf read-outs | C core, bound | keep; samplers unified (phase 9) | 9 |
| User C++ models | `pmf_from_cpp` / `load_cpp_builder` compile all of phasic into each model | work on a normal install (R7, O3); new parameterized route (R8) | 8 |
| SCC topological order | C Kahn (`scc_compose.c:355-379`) vs C++ trust-Tarjan (`scc_graph.cpp:70`), both bound | one provider (C Kahn) (OI-045) | 9 |
| Conditioned samplers | `ptd_random_sample_path_conditioned` / `_fixed` (`phasic.c:12362`, `:12451`) | one loop with an RNG hook (OI-047) | 9 |
| Elimination variants | `ptd_graph_ex_absorbation_time_comp_graph{,_dyn,_mpfr,_parameterized,_parameterized_dyn}` (`phasic.c:7270-9073`) | one driver with strategy hooks; keep `_mpfr` | 9 |
| Trace op enum | `phasic.h` vs `trace_elimination.py:53` (Python has `SUB`, C lacks it) | one asserted constant | 9 |
| `profile` | Python `profile.py` (live); C++ `Graph::profile` (unbound) | Python only; delete the port (R5) | 10 |

## 4. Safety net and gates

- **Before each move, a gate** that compares the current implementation
  with the new one bit for bit on serialized graphs and read-outs:
  - phase 2: construction from callbacks (every callback shape, plain and
    parameterized, with and without ipv, extend);
  - phase 3: all four moment surfaces (incl. rewards, discrete) and
    matrix export;
  - phase 4: the codec vs the Python indexer (property-based: random
    layouts, append, multi-property sets, round trips);
  - phase 5: serialized joint graphs over a model matrix (coalescent,
    two-island, two-locus ARG; discrete and continuous; `reward_limit`,
    `tot_reward_limit`, `reward_only`, the reward-rates callback, epochs,
    exposure) plus the 35 existing test files that build joint graphs;
  - phase 6: serialize round trips (incl. `is_aux`, constant edges, theta
    layout) and old-file loading;
  - phase 7: weight-formula tapes (needs a C++ tape reader first) and
    add_epoch;
  - phase 9: the existing SCC ordering gate flips; elimination variants.
- **Tolerance exceptions** are allowed only where a move replaces one
  arithmetic order by another (e.g. retiring the generated moments
  channel, unifying moment recurrences). Each is listed in its sub-plan
  with the measured difference and approved by the user before merging.
- **Suite comparison** (no new failures vs the previous merge) on every
  merge, with the gate tooling in `scripts/` and the broad suite in CI
  (prerequisites, phase 0).
- **Reproducibility first** (phase 0): bit-identity gates cannot work while
  some read-outs differ run to run (OI-023, OI-022, OI-015).

## 5. Phases

Each phase: detailed sub-plan → review → implement → gates → merge.
Status column maintained here.

| # | Phase | Contents | Status |
|---|---|---|---|
| 0 | Prerequisites | Plan A; register Tier 1 (CI runs the broad suite OI-062, gate tooling into `scripts/` OI-067, plot-test and flaky-test fixes OI-060/061, commit/prune/push OI-110/111/113); Tier 2 reproducibility (OI-023, OI-022, OI-015); Tier 3 core contracts (discrete read-out rule OI-021, `defect()` semantics OI-031, parameterized flag of derived graphs OI-026, C++ discretize rate checks OI-033, discrete reward-transform validation and leak OI-019/020) | not started |
| 1 | Inventory check and gates | verify §3 against master at that time; build the phase 2-3 gates; decide the gate-tolerance policy details | not started |
| 2 | pybind restructure | split the module into `register_*()` functions; one construction path (`phasic::Graph::from_callback` / `extend`) replacing the four pybind builders; `scc_decomposition` keep-alive (OI-030); OpenMP spawn failure raises (OI-034) | not started |
| 3 | Moments and matrices | one moment recurrence in `phasic::Graph::moments` (R11); GraphBuilder delegates; `moments_discrete`; retire the generated moments channel (R4; `moments_from_graph` keeps its signature; tolerance exception listed); matrix export/import promoted | not started |
| 4 | Index codec | C++ `IndexCodec` + bindings (batch encode/decode to avoid per-call overhead); Python `StateIndexer` delegates layout and encoding; property-based gate | not started |
| 5 | Joint-probability family in C++ | product construction with caps, `reward_only`, trash and t-vertex rules, degenerate-outcome merge, multi-property rule; the reward-rates callback calls into Python per state (as construction from callbacks does); `joint_stop_prob_graph`, `joint_sojourn_graph`, `joint_prob_table`; epoch graph construction | not started |
| 6 | serialize and cache formats | one C++ owner + schema version; `is_aux`; GraphBuilder reads the same schema; OI-053, OI-054, OI-083; rev-1/2 loader removed or hardened and `pull_cache` made consistent (OI-079); cache invariants in code (OI-087) | not started |
| 7 | add_epoch, formulas, FFI core | add_epoch promoted (OI-025); C++ tape reader, formula gate, C++ compiler as single source, select/pow semantics decided (OI-048, OI-051); FFI handlers on one compute core, failures reported (OI-028) | not started |
| 8 | User C++ models | O3; `pmf_from_cpp` / `load_cpp_builder` work on a normal install (R7, OI-058, OI-029); parameterized C++ model route with a documented entry point, an example and a test against the equivalent Python model (R8) | not started |
| 9 | C-internal unification | one SCC topological order (OI-045, OI-046, OI-035, OI-049); one conditioned sampler (OI-047); one elimination driver (keep `_mpfr`); trace op enum; trace recording in C, Python keeps only the JAX replay (R12, OI-044); split `phasic.c` into units (keep the `phasic.h` amalgamation used by the JIT include) | not started |
| 10 | Python cleanup | delete the now-dead Python implementations and C++ ports (`profile`, the external-pointer per-SCC scheme OI-050); split the (now smaller) `Graph` class; check `bffg.py` for duplication (OI-086) | not started |
| 11 | Doctrine and guards | `docs/architecture/boundary.md`; cross-boundary version constants; reachability and lock-step CI guards; regenerate API docs (OI-090) | not started |

**Order rationale.** Phase 0 makes gates trustworthy. Phases 2-3 are low
risk and shrink the pybind layer that later phases bind into. The codec
(4) must exist before the joint family (5). serialize (6) follows 5 so the
joint metadata goes into the one schema once. User C++ models (8) need the
unified Graph, serialize and codec. C-internal work (9) comes late because
it is the riskiest and benefits most from the gates built earlier. Splitting
the Python `Graph` class waits until phase 10, when delegation has shrunk
it, so code moves once.

**Parallelism.** One heavy run at a time (memory), so phases run one after
another; small independent register items may fill gaps.

## 6. Risks

- **Performance of the codec boundary.** Per-call Python→C++ encoding can
  be slower than today's Python for tiny calls; the codec exposes batch
  calls, and joint construction runs entirely in C++.
- **Hidden behaviour in Python code being ported** (joint family:
  degenerate merges, caps, metadata attached to the returned graph). The
  phase 5 gate compares serialized graphs and the attached attributes.
- **Bit identity across arithmetic-order changes** (moments, generated
  channel): handled by listed, approved tolerance exceptions only.
- **Scale.** Constructions must stay usable at 200k-600k vertices; every
  phase that ports construction measures time and memory at the target
  scale, memory-capped.
- **Long programme, diversions.** New defects found along the way go into
  `OPEN-ITEMS.md`; only those that block the current phase are fixed in it.

## 7. Out of scope

Plan B and exact PDF gradients (R6; OI-002, OI-005); all inference-layer
items; new features beyond R8; notebook edits (the user's).

## 8. Open items owned by this plan

Phase 0: OI-015, 019, 020, 021, 022, 023, 026, 031, 033, 060, 061, 062,
067, 110, 111, 113. Phases 2-11: OI-025, 028, 029, 030, 034, 035, 044, 045,
046, 047, 048, 049, 050, 051, 053, 054, 058, 065, 077, 079, 083, 086, 087,
090. Decision OI-009 (scope) is settled by §1-§2; OI-007 is "continue"
under this plan.
