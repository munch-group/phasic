# Core-engine refactor: C/C++ as the single source of every core computation

**Status: v2, 2026-11-26. v1 (`b9178dab`) went through three independent
adversarial reviews (Opus; all CHANGES REQUIRED, none rejecting the goal);
their findings and the user's decisions on them are folded in here
(dispositions in §9). v2 is NOT yet re-reviewed. No code changed.**
This is a master plan: each phase gets its own detailed sub-plan, reviewed
before any code. It supersedes the unexecuted parts of
[stage3-refactor-plan.md](stage3-refactor-plan.md) (July 2026), which stays
as history; its doctrine and the decisions listed in §2 carry over.
Review reports: `experiments/core_ssot/review{1,2,3}/REPORT.md`.

## 0. Copy-paste prompt for a fresh session

> Read `plans-and-findings/core-engine-ssot-plan.md` in full (including §9
> and the review reports it names), then `OPEN-ITEMS.md` (Priorities) and
> `CLAUDE.md`. Find the first phase in §5 whose status is not DONE. If it
> has no reviewed sub-plan, write one in
> `plans-and-findings/core-ssot-p<N>-<name>-plan.md` and put it through
> adversarial review (one Opus reviewer per round; after a clean sign-off,
> two more independent reviewers) before any code. Sub-plans and reviews
> may run ahead of implementation; implementation is strictly sequential.
> Implement in a worktree on its own branch; gate with §4; merge with
> `--no-ff`; run `pixi run install-dev` in the main tree; update §5's status
> and `OPEN-ITEMS.md`. Memory-cap every run; one heavy run at a time; never
> run reviewers alongside test runs. Never edit notebooks (list cells for
> the user). Never `git add -A`. The user pushes.

## 1. Goal and definition of done

**Goal (user, 2026-11-26):** every core computation has exactly one
implementation, in C or C++, so the engine is easier to maintain and to
develop safely. Python delegates to it. The inference layer stays Python.

**Core computation** = everything that builds, transforms or evaluates a
phase-type graph:

- state-space construction from a callback; state layout and encoding (the
  index codec);
- the graph's own state (§3 "graph state": discreteness, weight mode and
  formula, layout, joint and epoch metadata);
- the joint-probability family and the epoch forward step;
- discretize, add_epoch, layout widening, reward and Laplace transforms;
- moments and the discrete correction; distribution read-outs; sampling;
- serialize and the cache file formats;
- weight formulas; elimination (all variants); SCC decomposition and
  composition; the FFI compute handlers.

**Stays Python:**
- **inference:** SVGD, MCMC, priors, `svgd_config`,
  `probability_matching`, `method_of_moments`, model selection, plotting;
- **the JAX wrapping:** `custom_vjp`, `pure_callback`, the FD backward
  rules of Plan A;
- **orchestration and infrastructure:** the graph file cache and its key,
  SLURM/distributed orchestration, `profile` as an advisory tool.

**Done means:**

1. Every row of §3 marked "C/C++" has one implementation, bound to Python,
   with Python delegating. Other copies are deleted, except the sanctioned
   exceptions listed in §2.
2. **Public Python API unchanged:**
   - names, signatures and results;
   - exception types, and the messages tests match;
   - warnings;
   - callback argument types (e.g. numpy state arrays);
   - attributes on returned graphs;
   - serialize keys;
   - pickling;
   - graph-cache keys.

   Each exception is listed in a sub-plan and approved by the user. The
   removals approved under R12' are separate.
3. Every move was gated (§4); the gates still exist after the old copy is
   deleted (frozen golden fixtures).
4. A C++ user can build every model the Python API can except
   `weight_mode='callback'` (arbitrary Python): constant, parameterized,
   formula-mode, indexer-based, joint-probability and epoch models. They
   can hand any of them to Python as serialized bytes (R13), where the full
   machinery applies.
5. Performance at target scale is no worse than before (benchmark gate,
   §4); the doctrine document and CI guards exist.

## 2. Decisions (recorded)

**From the user, 2026-11-26:**

- **R1** A new master plan (this file) replaces the July plan's remaining
  workstreams.
- **R2** StateIndexer, option B: C++ owns layout and encoding (an
  `IndexCodec`). The Python `StateIndexer` keeps its user API and delegates
  the rules. The codec also has a native constructor (properties, sets,
  slots, append) for C++ users, not only loading from `to_dict()`.
- **R3** Weight formulas: the C++ compiler becomes the single source;
  Python `compile_formula` survives only as a test oracle.
- **R4** The generated-C++ moments channel is retired. `moments_from_graph`
  keeps its signature and computes through the compiled library; its
  `use_ffi` argument is deprecated (a warning, then removal), never
  silently ignored.
- **R5** `profile` stays Python; the unbound C++ port is deleted.
- **R6** No deliberate numeric change during the refactor, except R14 and
  the tolerance exceptions approved under §4. Plan B and exact PDF
  gradients come after the refactor.
- **R7** `pmf_from_cpp` and `load_cpp_builder` keep working through the
  refactor. Their defects (OI-089: /tmp path, stale-library key, missing
  sources) are fixed in phase 0, and their internals move onto the R13
  hand-off in phase 11.
- **R8** New deliverable: a C++ user builds a parameterized model once and
  hands it to Python (via R13).
- **R9** After the refactor, decide whether the per-theta C++ route is
  still worth keeping.
- **R10** Plan A ([fd-domain-steps-plan.md](fd-domain-steps-plan.md)) is
  done before phase 0.
- **R11 (revised after review)** The moment recurrence and the discrete
  correction become C kernels: `ptd_moments` and
  `ptd_continuous_to_discrete_moments`. `phasic::Graph::moments` is the
  public C++ API over them. Python, GraphBuilder, the pybind discrete
  functions and the FFI handlers call them.
  - **Sanctioned exception:** the exact-gradient core's forward chain
    (`ptd_b3_moments_core`, `phasic.c:11082-11100`), which needs
    per-command snapshots. A drift gate compares it with `ptd_moments`.
- **R12' (replaces R12)** The Python trace pipeline is deleted after a
  check that nothing live needs it. It covers the recorder, replay,
  `EliminationTrace`, `compute_trace`, the hierarchical trace cache and its
  stitching, the trace serialization, and the trace-cache statistics. The
  C parameterized tape is already the live path (§3). The list of public
  names to remove goes to the user before deletion. The unused C port of
  the Python trace format (`ptd_record_elimination_trace`,
  `ptd_evaluate_trace`) is in the frozen public C header; its removal is
  asked for separately.
- **R13 (new; settles O3)** A graph built by user C++ code reaches Python
  only as the versioned serialized graph (bytes); no live pointer crosses
  between compiled copies of the core. The extension's build structure
  stays (each binary links its own static copy).
- **R14 (new)** Approved exception to R6: constant-rate elimination, the
  reward transform and `_dyn` order edges by memory address today
  (`arr_c_cmp`, `phasic.c:6088-6094`, used at `:6327`, `:7414`, `:7784`,
  `:8180`; it also never returns 0). This is non-reproducible across
  instances (measured: 3 distinct `expectation()` bit patterns over 6
  instances, `experiments/core_ssot/review3/probe_determinism.py`) and the
  likely cause of OI-022/OI-023. It is fixed in phase 0 by ordering by
  index, as the parameterized path already does (`ptd_edge_order_cmp`,
  `:8553`); last digits change.

**Carried over from the July plan:** the nine-rule delegation doctrine (C
owns numeric kernels; C++ a facade over them; pybind marshalling only);
complete-then-promote for every move (reach parity → bind and gate →
delegate and delete, each its own commit); public C symbols frozen
(additions fine, removals need approval); keep the `_mpfr` variant.
"Persistent per-thread FFI graphs" is NOT carried over unexamined; see
phase 10.

## 3. Scope table (master `b9178dab`, re-checked by three reviews)

### Live versus retired elimination paths (read from source)

- **Live:** the C parameterized tape. `ptd_precompute_reward_compute_graph`
  (`phasic.c:1915`) records once and is replayed after `update_weights`.
  It feeds expectation/moments/variance, sojourn times, the exact-gradient
  cores and the FFI (GraphBuilder); it is cached in memory, in rev-3 files,
  and per component for the C composer.
- **Retired:**
  - the Python trace, reachable only via deprecated `compute_trace`
    (`__init__.py:10465`) and `_ensure_trace`, whose callers are commented
    out (`:2556-2691`);
  - `evaluate_trace_jax`, with no production caller;
  - the hierarchical trace cache's Python stitching
    (`hierarchical_trace_cache.py:1641`), used only by these.

  All are deleted under R12'.

### Capabilities

| Capability | Implementations today | Target | Phase |
|---|---|---|---|
| Construction from a callback | four pybind builders (`phasic_pybind.cpp:431-660`; parameterized IPV edges, numpy state argument, zero-probability IPV accepted; leak `new Graph` without delete `:434-480`) + `phasic::Graph::from_callback` / `extend` (`phasiccpp.cpp:196`, `:414`; constant IPV only, rejects <= 0) + Python IPV validation (`_validate_ipv`, the `_callback` wrapper) | one path in `phasic::Graph`, after parity | 2 |
| Moment recurrence | pybind `_moments` (`phasic_pybind.cpp:1832`; 32-bit `fac()`, wrong for k >= 13, OI-088); `GraphBuilder::compute_moments_impl` (`graph_builder.cpp:644`); `phasic::Graph::moments` (`phasiccpp.h:731`; `long` factorial, wrong for k >= 21); generated C++ (`moments_from_graph`, `__init__.py:7772`); gradient-core forward chain (`phasic.c:11082`) | `ptd_moments` (C) + `phasic::Graph::moments` API (R11) | 3 |
| Discrete moment correction | Python `_continuous_to_discrete_moments` (`__init__.py:644-687`); `GraphBuilder::continuous_to_discrete_moments` (`graph_builder.cpp:826`; FFI `:548`, `:575`, `:772`); C `ptd_dph_*` (`phasic.c:11552-11600`); pybind `_expectation/_variance/_covariance_discrete` (`phasic_pybind.cpp:348`, `:355`, `:399`) | `ptd_continuous_to_discrete_moments` (C) | 3 |
| Matrix export / import | bound lambdas (`:3295`, `:3381`); `phasic::Graph` port (unbound); orphan Eigen class | `phasic::Graph`, bound | 3 |
| Generated-C++ graph emitter | `_generate_cpp_from_graph` (`__init__.py:968-1095`): `moments_from_graph` (R4) and `pmf_from_graph` on constant graphs (`:4027` via `pmf_from_cpp` `:4234`) | retired; constant `pmf_from_graph` routed through the compiled library | 3 |
| Callback-mode weights | `_apply_weight_callback` rewrites the serialized dict in Python (`:893`; used `:4056`, `:8536`, `:8609`, `:9677`) | stays Python (callback mode is arbitrary Python), documented boundary | — |
| Graph state | Python attributes: `is_discrete`, `_weight_mode`, `_weight_formula(_tape)`, `_weight_callback`, `_indexer`, `_base_indexer`, joint metadata (`__init__.py:11332-11340`), epoch metadata (`:10307-10322`); `clone()` copies a subset (`:10291-10312`); C++ knows only `was_dph` (`phasiccpp.h:1123-1125`, `:191`); discrete dispatch in Python (`:2575-2580`, `:2826`, ~`:2900`) | a C++-owned metadata record on `phasic::Graph`, carried by clone, copy and serialize | 4 |
| serialize / from_serialized | Python `_graph_serialize.py` (carries weight_mode, is_discrete, is_aux only, `:217-242`); `GraphBuilder::parse_structure`; `phasic::Graph::serialize` (`phasiccpp.cpp:1317`, `:1407`; no `is_aux`); the generated-C++ emitter | one C++ owner; versioned schema carrying the graph state; GraphBuilder reads it | 4 |
| State layout and encoding | Python `StateIndexer` (`state_indexing.py`, 2,266 lines) | C++ `IndexCodec` (R2) | 5 |
| joint_prob_graph | Python (`__init__.py:10703-11342`; per-vertex loop over the C AVL lookup) | C++ | 6a |
| joint_stop_prob_graph, joint_sojourn_graph, EpochContext, epoch forward step | Python: `:11343`, `:11548` (returns `EpochContext`, `:12433`); `_probe_daisy_t_eval` (`:12055-12186`); `EpochContext.next_ipv` / `_collapse_one` / `auto_t` (`:12594-12684`); r_v read `_get_joint_probs` (`:12321-12326`) vs C++ FFI (`graph_builder_ffi.cpp:2222-2235`); t-vertex discovery walk (`:12296-12303`) | C++ (one copy each) | 6b |
| joint_prob_table, observation mapping, reward-coverage walks | Python (`:12366`; `_graph_reward_validation.py:13-62`; `absorbing_state_rewards`) | C++ | 6c |
| Elimination-cache file formats | rev-3 (hardened); rev-1/2 loader (unhardened, OI-079); per-SCC files; pybind `cache_file_path` hard-codes `$HOME/.phasic_cache` (`phasic_pybind.cpp:3706-3712`) | rev-3 only (or rev-1/2 hardened); invariants in code (OI-087); one cache-path rule | 7 |
| add_epoch | Python (`__init__.py:3246`); C++ port (unbound) | C++ after parity (OI-025) | 8 |
| Weight formulas | Python `compile_formula` + C tape VM; C++ `wf_*` compiler (`phasiccpp.cpp:937-1316`, unbound, untested) | C++ compiler (R3), C VM | 9 |
| FFI compute handlers | 9 handlers in `graph_builder_ffi.cpp` (2,523 lines); structure JSON copied and keyed by full string per call (`:272-293`, `:342-347`, per batch element `:1542`); per-thread builder and graph caches each building their own tape (`:103`, `:300-310`) | one compute core; models keyed by a content hash computed once; read-only tape shared across threads; failures reported (OI-028) | 10 |
| User C++ models | `pmf_from_cpp` (`__init__.py:4600`, Python JIT driver `_compile_wrapper_library` `:1096`) and `load_cpp_builder` (pybind JIT driver `phasic_pybind.cpp:5141-5268`, compiles `phasic.c` as C++ without SCC sources, MPFR or OpenMP; shares a graph pointer across copies) | R13 bytes hand-off; one JIT driver; R8 deliverable | 11 |
| SCC topological order | C Kahn (`scc_compose.c:355-379`) vs C++ trust-Tarjan (`api/cpp/scc_graph.cpp:70`) | one provider (C Kahn); tolerance exception if bits change | 12 |
| Conditioned samplers | `phasic.c:12362`, `:12451` | one loop with an RNG hook (OI-047) | 12 |
| Elimination variants | five builders `phasic.c:7270-9073` (constant, `_dyn`, `_mpfr` `:8036`, parameterized, parameterized `_dyn`) | one driver, compile-time dispatch; keep `_mpfr` | 12 |
| Python trace pipeline | `trace_elimination.py`, `hierarchical_trace_cache.py`, `trace_serialization.py`, `trace_cache.py`, `compute_trace`, `_ensure_trace` | deleted (R12') | 13 |
| discretize, widening, theta layout | C++, bound | done; residual OI-033 in phase 0 | — |
| `profile` | Python (live); C++ port (unbound) | Python only (R5) | 15 |

## 4. Gates and safety net

**Before each move:**
- **Golden fixtures.** Freeze the current implementation's outputs as
  pinned fixtures (serialized graphs incl. metadata and dict order,
  read-outs, error types and messages). The precedent is
  `test_gate_discretize_pins.py`. They outlive the deleted copy.
- **Branch coverage.** Measure the gate corpus's branch coverage of the
  Python code being replaced, and list uncovered branches in the sub-plan.
  For the joint family, examples include `reward_only` merging with several
  absorbing base states, add_epoch'd and discretized bases, constant-edge
  refusal, and formula/log bases.

**Bit identity:**
- A/B on the same platform.
- Ported translation units compile with `-ffp-contract=off`, or the
  sub-plan lists each expression allowed to contract.
- Every bit-identity gate also re-runs with the heap perturbed (catches
  address dependence, R14).
- CI runs the gates on arm64 and x86.
- Tolerance exceptions only for listed arithmetic-order or fusion changes,
  with the measured difference, approved by the user.

**Performance and memory** (new). A memory-capped benchmark per family:
coalescent ~60k vertices, two_island ~20k, two-locus as large as the cap
allows. It records elimination time, tape length, peak RSS, and time per
SVGD iteration through the FFI. A regression above 5% fails. A
construction benchmark at 60k and 600k states covers callback builds.

**Native checklist (phases 2, 3, 6, 8-12):**
- the thread-local teardown test plus a long-run shutdown smoke test with
  XLA worker threads;
- an OpenMP run with batch > 1;
- a test where a Python callback raises;
- a malloc-scribble run, and one ASan/UBSan CI job (Linux-only
  uninitialised-memory history);
- new native caches use the live-pointer pattern.

**API stability.** The phase-2 exception-mapping layer (C++ exception →
the same Python exception type and message); pickle round trips and
old-pickle loading; graph-cache key stability tests.

**Version constants:** each phase that changes a cross-boundary format
introduces and asserts its version constant in that phase (serialize
schema, cache files, SLURM exchange), not at the end.

**Suite comparison** (no new failures against the previous merge) on every
merge; gate tooling in `scripts/`; the broad suite in CI.

## 5. Phases

Each: sub-plan → review → implement → gates → merge. Sub-plans may run
ahead; implementation is sequential. Size is relative (S / M / L / XL), from
review 2's estimates.

| # | Phase | Contents | Size | Status |
|---|---|---|---|---|
| — | Plan A | [fd-domain-steps-plan.md](fd-domain-steps-plan.md) | M | not started |
| 0a | Prerequisites | CI runs the broad suite (OI-062); gate tooling into `scripts/` (OI-067); plot-test and flaky-test fixes (OI-060/061); commit, prune, push (OI-110/111/113) | S | not started |
| 0b | Determinism | R14 (index order instead of address order); confirm and fix OI-022, OI-023, OI-015; heap-perturbation harness | M | not started |
| 0c | Cheap core fixes | factorial overflow OI-088; user-C++ JIT defects OI-089 (secure directory, phasic version in the cache key, correct source list); C++ discretize rate checks OI-033; discrete reward-transform validation and leak OI-019/020 | S | not started |
| 1 | Inventory, gates, baselines | re-check §3 on master; frozen golden fixtures for phases 2-3; benchmark baseline; tests and a measured baseline for `pmf_from_cpp` / `load_cpp_builder` (none exist); FP-contract policy; arm64 + x86 CI gates; ASan job; record supported platforms (Windows is built by `pypi-release.yml:47-51`, but the JIT routes hard-code `/tmp`, `g++` and POSIX mmap) | M | not started |
| 2 | pybind restructure and construction | split into `register_*()`; C++ exception-mapping layer; one construction path, complete-then-promote: parity first (parameterized IPV edges, zero-probability entries, numpy state argument, the three IPV validations reconciled; every user-visible difference listed for approval); fix the builder leak; recheck OI-030 (may already be done, `phasic_pybind.cpp:3743-3763`); OpenMP spawn failure raises (OI-034) | M | not started |
| 3 | Moments and matrices | `ptd_moments`, `ptd_continuous_to_discrete_moments`, `phasic::Graph::moments` (R11); all copies delegate; drift gate against the gradient core; gate k >= 13 and k >= 21; matrices; retire the generated-C++ emitter (R4, constant `pmf_from_graph` route); `use_ffi` deprecation; OI-026 (parameterized flag of derived graphs) and OI-031 (`defect()` semantics) decided here | M | not started |
| 4 | Graph state and serialize schema | C++-owned metadata record (discreteness and the one discrete rule OI-021, weight mode and formula tape, layout, joint and epoch metadata, t-vertex maps); carried by clone, copy and serialize; one versioned serialize owner (binary format considered after measuring JSON at 600k); GraphBuilder reads it; OI-053, OI-054, OI-083 | L | not started |
| 5 | Index codec | exact codec API listed; native constructor; codec → StateIndexer direction (for the reward-rates callback); error-message parity; codec handle NOT in the instance `__dict__` (`callback_hash.py:149-157`, `:653-720`; a module-level cache keyed by `to_dict()`), so graph-cache keys stay identical (OI-055 checked); pickle and old-pickle tests; scalar call latency no worse than today (2.4 / 3.8 µs) inside 60k and 600k callback builds | L | not started |
| 6a | joint_prob_graph in C++ | index-ordered containers only (no pointer keys, no unordered iteration); caps, `reward_only`, trash and t-vertex rules, degenerate merge, multi-property rule; reward-rates callback: two ABIs (Python objects via the codec→StateIndexer direction; a `std::function` for C++ users), GIL held, single-threaded, never inside OpenMP; ~53 test files | L | not started |
| 6b | Epoch family in C++ | `joint_stop_prob_graph`, `joint_sojourn_graph`, `EpochContext`, the epoch forward step (IPV projection, t/aux collapse, auto_t, r_v read, t-vertex discovery) with the FFI using the same code | L | not started |
| 6c | Tables and walks | `joint_prob_table`, observation mapping, reward-coverage and absorbing-state walks | M | not started |
| 7 | Cache formats | rev-1/2 loader removed or hardened and `pull_cache` consistent (OI-079); cache invariants in code (OI-087); one cache-path rule; version constants | M | not started |
| 8 | add_epoch | port promoted after parity (OI-025) | M | not started |
| 9 | Weight formulas | C++ tape reader; formula gate; C++ compiler as single source; select/pow semantics decided (OI-048, OI-051) | M | not started |
| 10 | FFI compute core | one compute core for 9 handlers; content-hash model keys; read-only tape shared across threads; failures reported (OI-028); memory-capped gate at 60k with OpenMP threads = cores | L | not started |
| 11 | User C++ models | R13 bytes hand-off; one JIT driver; `pmf_from_cpp` / `load_cpp_builder` internals onto it (OI-029, OI-058); R8 example and a test against the equivalent Python model | M | not started |
| 12 | C-internal unification | one SCC order (OI-045, OI-046, OI-035); one conditioned sampler (OI-047); one elimination driver with compile-time dispatch or LTO, benchmark gate, MPFR gate scripts rerun | XL | not started |
| 13 | Python trace pipeline | confirm no live consumer; user approves the removal list; delete (R12'); OI-044 and OI-049 close with it | M | not started |
| 14 | phasic.c split | translation units; constraints are the JIT source lists (two drivers until phase 11, one after) and the Rcpp example (`docs/pages/popgen/coalescent_with_freqbins.cpp:12`); nothing includes the old amalgamation header | L | not started |
| 15 | Python cleanup | delete dead implementations and ports (`profile`, the external-pointer SCC scheme OI-050); split the (smaller) `Graph` class; check `bffg.py` (OI-086) | L | not started |
| 16 | Doctrine and guards | `docs/architecture/boundary.md`; reachability and lock-step CI guards; regenerate API docs (OI-090). Sub-plans cite the July §2 doctrine until then | S | not started |

**Order rationale:**
- Phase 0 makes the gates trustworthy (determinism first).
- Phase 1 builds the baselines everything else is judged against.
- Phases 2-3 are the smallest real moves and remove three of four moment
  copies.
- Graph state and the schema (4) must exist before anything that writes
  metadata (5, 6).
- The codec (5) precedes the joint family (6).
- Formats (7) settle once the schema exists.
- User C++ models (11) need the schema, codec and joint family.
- C-internal work (12-14) comes late because it is the riskiest and
  benefits most from earlier gates.
- The Python `Graph` class is split last, when delegation has shrunk it.

## 6. Risks

- **Half-delegated states.** For example, the Python `StateIndexer`
  delegating while Python `joint_prob_graph` still runs. Each phase leaves
  every path working and gated; the codec latency gate covers the 5→6
  interval.
- **Hidden behaviour in ported Python.** Covered by frozen fixtures,
  branch coverage and metadata comparison.
- **Memory at scale on the FFI path:** threads × (graph + tape) today
  (estimated ~49 GB per thread for two_island at 60k); phase 10 fixes it.
- **Performance regressions from unification and the split:** the
  benchmark gate.
- **ABI drift:** `ptd_graph` layout depends on `PHASIC_B3_VALIDATORS`
  (`phasic.h:263-269`). With R13, no struct crosses compiled copies; the
  JIT cache key includes the phasic version (phase 0c).
- **Long programme with diversions.** New defects go to `OPEN-ITEMS.md`;
  only blockers of the current phase are fixed in it.

## 7. Out of scope

Plan B and exact PDF gradients (R6; OI-002, OI-005); inference-layer items;
features beyond R8; notebook edits (the user's).

## 8. Open items owned by this plan

- **Phase 0:** OI-015, 019, 020, 022, 023, 033, 060, 061, 062, 067, 088,
  089, 110, 111, 113.
- **Later phases:** OI-021, 025, 026, 028, 029, 030, 031, 034, 035, 044,
  045, 046, 047, 048, 049, 050, 051, 053, 054, 055, 056, 058, 065, 077,
  079, 080, 083, 086, 087, 090.
- **Decisions:** OI-009 is settled by §1-§2; OI-007 is "continue" under
  this plan.

## 9. Review dispositions (v1 → v2)

Reports: `experiments/core_ssot/review{1,2,3}/REPORT.md`.

- **Graph metadata had no owner.** All three reviews (R1 B1, R2 B1,
  R3 M2) → the new phase 4 (graph state and schema) before codec and joint
  work; done-criterion 4 via R13.
- **R12 aimed at retired consumers.** R1 M4, R3 B1 → user decision R12',
  phase 13.
- **R11 left C copies.** R1 M1, R3 M1 → user decision: C kernels with the
  C++ API, a sanctioned gradient-core exception and a drift gate.
- **Factorial overflow.** R1 M2 → OI-088 in phase 0c; gates at k >= 13
  and k >= 21.
- **Phase 2 was not low risk.** R1 M3, R2 M5 → phase 2 rewritten as
  complete-then-promote with an exception-mapping layer.
- **Codec versus cache key and pickle.** R1 M5, R2 M2 → phase 5.
- **Joint/epoch rows mis-scoped; duplicates missed.** R1 M6, R2 M4 →
  §3 rows and phases 6a/6b/6c.
- **Generated C++ wider than R4.** R1 M7 → §3 rows, phase 3.
- **Gates lost with the oracle; branches missed.** R1 M8, R2 M6 → §4
  frozen fixtures and branch coverage.
- **FP settings and platforms.** R1 M9, R2 M6 → §4.
- **No baseline for user C++ models, and a corruption hazard.** R2 B2 →
  phase 0c (OI-089) and phase 1 tests.
- **Packaging.** R2 M1, R3 M2 → user decision R13 (bytes hand-off, build
  structure unchanged); the platform statement goes to phase 1.
- **Version constants.** R2 M3 → §4, per phase.
- **Phases too large.** R2 M4 → 0a/0b/0c, 6a/6b/6c, 7-10 separated, and
  12/13/14 separated.
- **API stability.** R2 M5 → done-criterion 2 and §4.
- **Half-delegated states.** R2 M7 → phase 5 latency gate and the
  codec→StateIndexer direction.
- **FFI memory growth with threads.** R3 M3 → phase 10.
- **No performance gate.** R3 M4 → §4 and phase 12.
- **Address-ordered edges.** R3 M5 → user decision R14, phase 0b.
- **Concurrency and lifetime.** R3 M6 → §4 native checklist; the leak
  goes to phase 2.
- **Joint determinism and the scale path.** R3 M7 → phase 6a and §4.
- **Stale references.** Minors corrected: `scc_graph.cpp` path,
  `moments_from_graph` line, test-file counts (53-55), the `_mpfr`
  variant, the OI-030 recheck, the amalgamation constraint, the
  sequencing note, the §8 omissions.
