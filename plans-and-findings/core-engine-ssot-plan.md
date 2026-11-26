# Core-engine refactor: C/C++ as the single source of every core computation

**Status: v2.1, 2026-11-26. v1 (`b9178dab`) went through three independent
adversarial reviews, v2 (`256c788a`) through a fourth (Opus; all CHANGES
REQUIRED, none rejecting the goal); their findings and the user's decisions
on them are folded in here (dispositions in §9). v2.1 is NOT re-reviewed.
Two decisions are pending with the user (§2, D15, D16). No code changed.**

This is a master plan: each phase gets its own detailed sub-plan, reviewed
before any code. It supersedes the unexecuted parts of
[stage3-refactor-plan.md](stage3-refactor-plan.md) (July 2026), which stays
as history; its doctrine and the decisions listed in §2 carry over, except
its §4a item 3 ("`evaluate_trace_jax` is not dead code"), retracted by R12'.
Review reports: `experiments/core_ssot/review{1,2,3,4}/REPORT.md`.

Phase numbers changed in v2.1 (trace deletion moved to phase 2, add_epoch
ahead of the joint family); §9 maps old to new.

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
- the graph's own state (discreteness, weight mode and formula, layout,
  joint and epoch metadata);
- the joint-probability family and the epoch forward step;
- discretize, add_epoch, layout widening, reward and Laplace transforms;
- moments, variance, covariance and their discrete forms; distribution
  read-outs; sampling;
- serialize and the cache file formats;
- weight formulas; elimination (all variants); SCC decomposition and
  composition; the FFI compute handlers.

**Stays Python:**

- **inference:** SVGD, MCMC, priors, `svgd_config`, `probability_matching`,
  `method_of_moments`, model selection, plotting;
- **the JAX wrapping:** `custom_vjp`, `pure_callback`, the FD backward rules
  of Plan A;
- **callback-mode weights** (`weight_mode='callback'`, arbitrary Python);
- **orchestration and infrastructure:** the graph file cache and its key,
  SLURM/distributed orchestration, `profile` as an advisory tool.

**Done means:**

1. Every row of §3 marked "C/C++" has one implementation, bound to Python,
   with Python delegating. Other copies are deleted, except the sanctioned
   exceptions in §2.
2. **Public Python API unchanged.** This covers:
   - names, signatures and results;
   - exception types, and the messages tests match;
   - warnings;
   - callback argument types (e.g. numpy state arrays);
   - attributes on returned graphs;
   - pickling;
   - graph-cache keys;
   - serialize: existing keys keep their meaning; new keys are versioned
     additions.

   Each exception is listed in a sub-plan and approved by the user. The
   removals approved under R12' are separate.
3. Every move was gated (§4); the gates outlive the deleted copies (frozen
   golden fixtures).
4. A C++ user can build every model the Python API can, except
   `weight_mode='callback'`. That includes constant, parameterized,
   formula-mode, indexer-based, joint-probability and epoch models, and
   any of them can be handed to Python as serialized bytes (R13), where the
   full machinery applies.
5. Performance at target scale is no worse than before (benchmark gate,
   §4). The doctrine document and CI guards exist.

## 2. Decisions

**User, 2026-11-26:**

- **R1** This master plan replaces the July plan's remaining workstreams.
- **R2** StateIndexer option B: C++ owns layout and encoding (an
  `IndexCodec`) with a native constructor for C++ users. The Python
  `StateIndexer` keeps its user API and delegates the rules. Its layout
  encoding is `StateIndexer.to_dict()`.
- **R3** Weight formulas: the C++ compiler is the single source; Python
  `compile_formula` survives only as a test oracle.
- **R4** The generated-C++ moments channel is retired.
  `moments_from_graph` computes through the compiled library; its
  `use_ffi` argument is deprecated (a warning, then removal), never
  silently ignored.
- **R5** `profile` stays Python; the unbound C++ port is deleted.
- **R6** No deliberate numeric change during the refactor, except R14 and
  approved tolerance exceptions (§4). Plan B and exact PDF gradients come
  after.
- **R7** `pmf_from_cpp` and `load_cpp_builder` keep working. Their defects
  are fixed in phase 0c: OI-089 (the /tmp path, the stale-library key,
  missing sources) and OI-116 (`normalize()`).
- **R8** Deliverable: a C++ user builds a parameterized model once and
  hands it to Python (via R13).
- **R9** After the refactor, decide whether the per-theta C++ route is
  still worth keeping, including its per-call cost.
- **R10** Plan A ([fd-domain-steps-plan.md](fd-domain-steps-plan.md)) goes
  first.
- **R11 (revised)** The moment recurrence and the discrete correction are
  C kernels: `ptd_moments` and `ptd_continuous_to_discrete_moments`.
  `phasic::Graph::moments` is the public C++ API over them. Variance,
  covariance and their discrete and rewarded forms get C kernels too, or
  are listed as sanctioned in phase 4.
  - **Sanctioned exception:** the exact-gradient core's forward chain
    (`ptd_b3_moments_core`, `phasic.c:11082-11100`). A drift gate covers
    it, but only on tapes that do not switch to MPFR, and with the same
    reward scaling.
- **R12'** Delete the Python trace pipeline after a liveness check
  (phase 2). The list of public names to remove goes to the user first.
  The unused C port of the trace format (`ptd_record_elimination_trace`,
  `ptd_evaluate_trace`, in the frozen public header) is asked about at the
  same time.
- **R13** Graphs built by user C++ code reach Python only as versioned
  serialized bytes; no live pointer crosses compiled copies.
  - **Scope (refined from review 4, for confirmation, D16):** this applies
    to build-once models (R8) and to `load_cpp_builder`, whose pointer
    sharing is the only real crossing today (`phasic_pybind.cpp:5262`).
  - `pmf_from_cpp` computes inside its JIT library and returns only
    numbers, so it already satisfies R13. It keeps doing so.
  - Routing it through bytes would add serialize + parse per theta and per
    FD probe: measured ~2.2 s at 37k vertices with today's Python
    serializer, `experiments/core_ssot/review4/probe_serialize_cost.log`.
- **R14** Approved exception to R6: address-ordered edges become
  index-ordered (phase 0b).

**Pending user decisions:**

- **D15 (phase 1)** A build-wide `-ffp-contract=off` (no fused multiply-add)
  so bit identity survives code moving between files and C/C++ ports of
  Python arithmetic. It changes last digits wherever the compiler fuses
  today, so it is an R6 exception. Baselines are taken after the flip.
  - **Alternative:** keep today's flags and gate only same-file moves
    bitwise; tolerance gates elsewhere.
- **D16** The R13 scope refinement above.

**Carried over from the July plan:**

- the nine-rule delegation doctrine: C owns numeric kernels, C++ is a
  facade, pybind only marshals;
- complete-then-promote for every move, in its own commits (reach parity,
  bind and gate, delegate and delete);
- the public C symbols are frozen: additions fine, removals need approval;
- keep the `_mpfr` variant;
- the §4a invariants: an eager path for every forward property, and the
  daisy-chain gate `test_gate_daisy_chain_joint_probs.py`.

"Persistent per-thread FFI graphs" is re-examined in phase 11.

## 3. Scope table (master `256c788a`)

### Live versus retired elimination paths (read from source)

- **Live:** the C parameterized tape. `ptd_precompute_reward_compute_graph`
  (`phasic.c:1915`) records once and the tape is replayed after
  `update_weights`. It feeds expectation/moments/variance, sojourn times,
  the exact-gradient cores and the FFI (GraphBuilder). It is cached in
  memory, in rev-3 files, and per component for the C composer.
- **Retired:** the Python trace, reachable only via the deprecated
  `compute_trace` (`__init__.py:10465`) and `_ensure_trace`, whose callers
  are commented out (`:2556-2691`); `evaluate_trace_jax`, with no
  production caller; the hierarchical trace cache's Python stitching. All
  are deleted in phase 2.
  - One live helper sits in `trace_serialization.py`: `is_cache_disabled`
    gates the C reward-compute cache (`:85`, re-exported by
    `cache.py:48-55`). It is relocated, not deleted.

### Capabilities

| Capability | Implementations today | Target | Phase |
|---|---|---|---|
| Python trace pipeline | `trace_elimination.py`, `hierarchical_trace_cache.py`, `trace_serialization.py`, `trace_cache.py`, `compute_trace`, `_ensure_trace`, `Graph._trace` / `_cache_trace` / `_trace_dirty` (`__init__.py:2128-2152`) | deleted (R12'); `is_cache_disabled` relocated | 2 |
| Construction from a callback | four pybind builders (`phasic_pybind.cpp:431-660`). They accept parameterized IPV edges, pass a numpy state argument and accept zero-probability IPV; they leak (`new Graph` without delete, `:434-480`). Also `phasic::Graph::from_callback` / `extend` (`phasiccpp.cpp:196`, `:414`; constant IPV only, rejects <= 0) and Python IPV validation | one path in `phasic::Graph`, after parity | 3 |
| Moment recurrence | pybind `_moments` (`phasic_pybind.cpp:1832`; 32-bit `fac()`, OI-088); `GraphBuilder::compute_moments_impl` (`graph_builder.cpp:644`); `phasic::Graph::moments` (`phasiccpp.h:731`; `long` factorial, wrong for k >= 21); the generated C++; the gradient-core chain | `ptd_moments` + `phasic::Graph::moments` (R11) | 4 |
| Variance, covariance (continuous, discrete, rewarded) | pybind `_variance`, `_covariance`, `_variance_discrete`, `_covariance_discrete` (`phasic_pybind.cpp:284-420`); `phasic::Graph` copies (`phasiccpp.h:779-880`). The rewarded discrete variance is `momentsr[1] - momentsr[0]^2 - momentsrr[0]` (`phasic_pybind.cpp:381-388`), not the Stirling correction | C kernels, or listed as sanctioned | 4 |
| Discrete moment correction | Python `_continuous_to_discrete_moments` (`__init__.py:644-687`); `GraphBuilder::continuous_to_discrete_moments` (`graph_builder.cpp:826`; FFI `:548`, `:575`, `:772`); C `ptd_dph_*` (`phasic.c:11552-11600`) | `ptd_continuous_to_discrete_moments` | 4 |
| Rewarded moments | two legitimate algorithms: the FFI reward-transforms first, then corrects (`graph_builder.cpp:990-1020`); pybind and `phasic::Graph` scale the recurrence by rewards | one rule per call site, recorded in phase 4 | 4 |
| Discreteness of a graph | Python `is_discrete` attribute; C knows only `was_dph`; the Python dispatch (`__init__.py:2575-2580`, `:2826`, ~`:2900`) | a C-owned flag plus one discrete rule (OI-021), settled in phase 4 before the moments API is public | 4 |
| Matrix export / import | bound lambdas (`:3295`, `:3381`); `phasic::Graph` port; orphan Eigen class | `phasic::Graph`, bound | 4 |
| Generated-C++ graph emitter | `_generate_cpp_from_graph` (`__init__.py:968-1095`): `moments_from_graph` (R4) and constant `pmf_from_graph` (`:4027` → `pmf_from_cpp` `:4234`) | retired; constant `pmf_from_graph` routed through the extension, with a named A/B gate (the JIT copy is compiled without MPFR or OpenMP) | 4 |
| Graph state | Python attributes (`is_discrete`, `_weight_mode`, `_weight_formula(_tape)`, `_weight_callback`, `_indexer`, `_base_indexer`, joint metadata `__init__.py:11332-11340`, epoch metadata `:10307-10322`). The C++ graph already holds `_weight_formula_src`, `_epoch_state_index`, `_n_epochs`, `_theta_blocks`, `_theta_names` and `_callback` (`phasiccpp.h:275-281`), and the copy constructor shares `rf_graph` (`:254-266`). `clone()` copies a subset (`__init__.py:10291-10312`) | one C++-owned record (reconciling both sources), carried by clone, copy and serialize | 5 |
| serialize / from_serialized | Python `_graph_serialize.py` (`:217-242`); `GraphBuilder::parse_structure`; `phasic::Graph::serialize` (`phasiccpp.cpp:1317`, `:1407`); consumers `graph_cache.py:211`, `:347`, `distributed_scc.serialize_scc_synth` (`distributed_scc.py:77`), `compute_repository` | one versioned C++ owner, hardened as a trust boundary (R13) | 5 |
| State layout and encoding | Python `StateIndexer` (2,266 lines) | C++ `IndexCodec` (R2) | 6 |
| add_epoch | Python (`__init__.py:3246`); C++ port (unbound) | C++ after parity (OI-025) | 7 |
| joint_prob_graph | Python (`__init__.py:10703-11342`) | C++ | 8a |
| Epoch family and forward step | `joint_stop_prob_graph` (`:11343`); `joint_sojourn_graph` (`:11548`, returns `EpochContext` `:12433`); `_probe_daisy_t_eval` (`:12055-12186`); `EpochContext.next_ipv` / `_collapse_one` / `auto_t` (`:12594-12684`); the r_v read (`:12321-12326`) vs the C++ FFI (`graph_builder_ffi.cpp:2222-2235`); t-vertex discovery (`:12296-12303`) | C++, one copy each | 8b |
| Tables and walks | `joint_prob_table` (`:12366`); observation mapping; `_graph_reward_validation.py:13-62`; `absorbing_state_rewards` | C++ | 8c |
| Elimination-cache formats | rev-3; rev-1/2 loader (unhardened, OI-079); per-SCC files; pybind `cache_file_path` hard-codes `$HOME/.phasic_cache` (`phasic_pybind.cpp:3706-3712`) | rev-3 only (or hardened); invariants in code (OI-087); one cache-path rule | 9 |
| Weight formulas | Python `compile_formula` + C tape VM; C++ `wf_*` (`phasiccpp.cpp:937-1316`, unbound) | C++ compiler (R3), C VM | 10 |
| FFI compute handlers and tape sharing | 9 handlers (`graph_builder_ffi.cpp`, 2,523 lines). Structure JSON is copied and keyed by the full string per call (`:272-293`, `:342-347`, per batch element `:1542`). Per-thread builder and graph caches each build their own tape (`:103`, `:300-310`). The offset tape binds writable `mem_base` and `inputs[]` to one graph's `&edge->weight` (`phasic.c:1888-1900`) | one compute core; content-hash keys; a C replay API with a per-thread arena and input binding so one read-only tape is shared (C API change) | 11 |
| User C++ models | `pmf_from_cpp` (computes in its JIT library); `load_cpp_builder` (pybind JIT driver `phasic_pybind.cpp:5141-5268`; shares a pointer; returns `_Graph`); two JIT drivers | R13 for build-once and `load_cpp_builder` (D16); one JIT driver; R8 | 12 |
| SCC order, samplers, elimination variants | C Kahn (`scc_compose.c:355-379`) vs trust-Tarjan (`api/cpp/scc_graph.cpp:70`); samplers `phasic.c:12362`, `:12451`; five builders `phasic.c:7270-9073` incl. `_mpfr` | one each; compile-time dispatch; keep `_mpfr` | 13 |
| Installed orphan libraries and headers | `site-packages/lib/liblibphasic*`, `site-packages/include/` (`CMakeLists.txt:48-52`, `:74-78`); nothing links them | decided in phase 1 (stop installing them, or document them for C++ users) | 1 |
| discretize, widening, theta layout | C++, bound | done; residual OI-033 in phase 0c | — |
| `profile` | Python (live); C++ port (unbound) | Python only (R5) | 15 |

## 4. Gates and safety net

**Before each move:**

- **Golden fixtures.** Freeze the current outputs as pinned fixtures
  (serialized graphs with metadata and dict order, read-outs, error types
  and messages); the precedent is `test_gate_discretize_pins.py`. They
  outlive the deleted copy.
- **Branch coverage.** Measure the gate corpus's branch coverage of the
  Python being replaced. Every uncovered branch gets a fixture or a user
  waiver.

**Bit identity:**

- A/B on the same platform, after D15 is applied.
- Every bit-identity gate also re-runs with the heap perturbed.
- CI runs the gates on arm64 and x86.
- Tolerance exceptions only for listed sources, with the measured
  difference, approved by the user. Listed sources are arithmetic-order
  changes, compiler fusion if D15 is declined, numpy pairwise reductions
  (for example `__init__.py:12588`), and libm differences.

**Performance and memory.** A memory-capped benchmark per family:

- coalescent at ~60k and ~200k vertices; two_island at ~20k; two-locus as
  large as the cap allows;
- records elimination time, tape length, peak RSS, and time per SVGD
  iteration through the FFI;
- construction benchmarks at 60k and 600k states;
- protocol: N = 5 runs, median, fixed machine, no other heavy process; a
  median regression above 5% fails.

**Native checklist** (phases 3, 4, 5, 6, 8, 9, 11, 12, 13, 14):

- the thread-local teardown test plus a long-run shutdown smoke test with
  XLA worker threads;
- an OpenMP run with batch > 1;
- a test where a Python callback raises;
- a malloc-scribble run;
- one ASan/UBSan and one TSan CI job;
- new native caches use the live-pointer pattern;
- phase 5 checks clone/copy metadata and GraphBuilder reading the schema;
- phase 6 runs the native codec inside 600k callback builds;
- phases 5, 9 and 12 fuzz every loader of untrusted bytes (serialize, the
  cache files) under ASan;
- phase 14 checks every relocated thread-local and static.

**API stability.** The phase-3 exception-mapping layer; pickle round trips
and old-pickle loading; graph-cache key stability; a read rule for
unversioned (v0) graph-cache entries.

**Version constants.** Each phase that changes a cross-boundary format
(serialize schema, cache files, SLURM exchange) introduces and asserts its
version constant in that phase. Later phases may bump them.

**Suite comparison.** No new failures against the previous merge, on every
merge; gate tooling in `scripts/`; the broad suite in CI.

## 5. Phases

Each: sub-plan → review → implement → gates → merge. Sub-plans may run
ahead; implementation is sequential. Sizes are relative (S / M / L / XL).

| # | Phase | Contents | Size | Status |
|---|---|---|---|---|
| — | Plan A | [fd-domain-steps-plan.md](fd-domain-steps-plan.md) | M | not started |
| 0a | Prerequisites | CI runs the broad suite (OI-062); gate tooling into `scripts/` (OI-067); plot-test and flaky-test fixes (OI-060/061); commit, prune, push (OI-110/111/113) | S | not started |
| 0b | Determinism (R14) | Address order → index order everywhere: the `arr_c_cmp` sorts (`phasic.c:6088-6094`, used at `:6327`, `:7414`, `:7784`, `:8180`), the pointer comparisons in the merge loops (`:6489`, `:7535`, `:7926`, `:8301`) and the pointer-arithmetic sentinels (`:6187`/`:6247`, `:7275`/`:7324`, `:7643`/`:7696`, `:8052`), all in the `ptd_vtx_order_key` form (`:8541`). Confirm and close OI-022, OI-023, and OI-015 (time-boxed; the FFI rewarded path reward-transforms per theta, `graph_builder.cpp:993`). Heap-perturbation harness. Listed change: seeded `sample()` sequences on reward-transformed graphs change. | M | not started |
| 0c | Cheap core fixes | OI-088 factorial; OI-089 JIT path, key and sources; OI-116 `load_cpp_builder` normalize; OI-033 C++ discretize checks; OI-019/020 discrete reward transform | S | not started |
| 1 | Inventory, gates, baselines | re-check §3 on master; D15 applied (if approved), then baselines; golden fixtures for phases 2-4; benchmark protocol and baseline; tests and a measured baseline for `pmf_from_cpp` / `load_cpp_builder` (after 0c); arm64 + x86 CI gates; ASan and TSan jobs; supported platforms recorded (Windows wheels are built, `pypi-release.yml:47-51`, but the JIT routes assume `/tmp`, `g++` and POSIX mmap); the orphan installed libraries decided | M | not started |
| 2 | Delete the Python trace pipeline (R12') | confirm no live consumer; user approves the removal list; ask about the C trace port; delete; relocate `is_cache_disabled`; OI-044, OI-049, OI-083 close | M | not started |
| 3 | pybind restructure and construction | split into `register_*()`; C++ exception-mapping layer; one construction path via complete-then-promote. First parity: parameterized IPV edges, zero-probability entries, numpy state argument, the three IPV validations reconciled, every user-visible difference listed for approval. Then fix the builder leak, recheck OI-030 (`phasic_pybind.cpp:3743-3763`), and make an OpenMP spawn failure raise (OI-034) | M | not started |
| 4 | Moments, variance, covariance, discreteness | the C-owned discreteness flag and OI-021 (one discrete rule) decided first; `ptd_moments`, `ptd_continuous_to_discrete_moments`, variance/covariance kernels or sanctioned list; `phasic::Graph::moments` API; rewarded-moments rule per call site; drift gate (non-MPFR tapes, same reward scaling); gates at k >= 13 and k >= 21; matrices; retire the generated-C++ emitter (R4) with the constant `pmf_from_graph` A/B gate; `use_ffi` deprecation; OI-026, OI-031 | L | not started |
| 5 | Graph state and serialize schema | one C++-owned record reconciling the Python attributes and the existing C++ fields; carried by clone, copy and serialize. Schema rules: formula SOURCE plus a tape version (phase 10 may change tape semantics); layout as `StateIndexer.to_dict()`; joint and epoch fields may be added in phases 7-8 with a version bump; v0 graph-cache entries are read (or rejected) by an explicit rule. Consumers updated (graph cache, `distributed_scc`, `compute_repository`, GraphBuilder); `from_serialized` hardened and fuzzed; binary format considered after measuring at 600k; OI-053, OI-054 | L | not started |
| 6 | Index codec | exact API; native constructor; codec → StateIndexer direction; error-message parity; the codec handle kept OUT of the instance `__dict__` (`callback_hash.py:149-157`, `:653-720`), so graph-cache keys are unchanged (OI-055 checked); pickle tests; scalar call latency no worse than today (2.4 / 3.8 µs) inside 60k and 600k callback builds | L | not started |
| 7 | add_epoch | port promoted after parity (OI-025), before the joint family so it gates on the final epoch-metadata producer | M | not started |
| 8a | joint_prob_graph in C++ | index-ordered containers only; caps, `reward_only`, trash and t-vertex rules, degenerate merge, multi-property rule. The reward-rates callback has two ABIs (Python objects via codec → StateIndexer; a `std::function`), holds the GIL, runs single-threaded and never inside OpenMP. ~53 test files | L | not started |
| 8b | Epoch family in C++ | `joint_stop_prob_graph`, `joint_sojourn_graph`, `EpochContext`, the epoch forward step, with the FFI using the same code; the July §4a invariants kept | L | not started |
| 8c | Tables and walks | `joint_prob_table`, observation mapping, reward-coverage and absorbing-state walks | M | not started |
| 9 | Cache formats | rev-1/2 loader removed or hardened and `pull_cache` consistent (OI-079; ASan + fuzz corpus); cache invariants in code (OI-087); one cache-path rule; version constants | M | not started |
| 10 | Weight formulas | C++ tape reader; formula gate; C++ compiler as single source; select/pow semantics decided (OI-048, OI-051) | M | not started |
| 11 | FFI compute core and shared tape | one compute core for 9 handlers; content-hash model keys; a C replay API with a per-thread arena and input binding so the read-only tape is shared (requires phase 9's OI-087 invariants; the exact-gradient cores in its gates); failures reported (OI-028); the §4a invariants; a memory-capped gate at 60k with OpenMP threads = cores | XL | not started |
| 12 | User C++ models | R13 bytes hand-off for build-once models and `load_cpp_builder` (D16; its return type changes from `_Graph` to `Graph`, listed); one JIT driver; `pmf_from_cpp` stays in-JIT with the version key (OI-029, OI-058); R8 example and a test against the equivalent Python model | M | not started |
| 13 | C-internal unification | one SCC order (OI-045, OI-046, OI-035; tolerance exception if bits change); one conditioned sampler (OI-047); one elimination driver with compile-time dispatch or LTO; benchmark gate; MPFR gate scripts rerun | XL | not started |
| 14 | phasic.c split | translation units, all with the D15 flags; constraints are the JIT source list(s) and the Rcpp example (`docs/pages/popgen/coalescent_with_freqbins.cpp:12`) | L | not started |
| 15 | Python cleanup | delete dead implementations and ports (`profile`, the external-pointer SCC scheme OI-050); split the (smaller) `Graph` class; check `bffg.py` (OI-086) | L | not started |
| 16 | Doctrine and guards | `docs/architecture/boundary.md`; reachability and lock-step CI guards; regenerate API docs (OI-090). Sub-plans cite the July §2 doctrine until then | S | not started |

**Order rationale:**

- Phase 0 makes the gates trustworthy.
- Phase 1 builds the baselines everything is judged against.
- Phase 2 removes dead code that later phases would otherwise have to keep
  working.
- Phases 3-4 are the smallest real moves; phase 4 settles discreteness
  before the moments API becomes public.
- The graph state and schema (5) precede everything that writes metadata
  (6-8).
- The codec (6) and add_epoch (7) precede the joint family (8).
- Cache formats (9) settle after the schema, and precede the shared-tape
  work (11) that depends on their invariants.
- User C++ models (12) need the schema, codec and joint family.
- C-internal work (13-14) comes late because it is the riskiest.
- The Python `Graph` class is split last, once delegation has shrunk it.

## 6. Risks

- **Half-delegated states.** For example: Python attributes and the C++
  record coexist from phase 5 to phase 8; the Python `StateIndexer`
  delegates while Python `joint_prob_graph` still runs. Each phase leaves
  every path working and gated; the codec latency gate covers phases 6-8.
- **Hidden behaviour in ported Python.** Covered by frozen fixtures, branch
  coverage (fixture or waiver) and metadata comparison.
- **FFI memory at scale.** Today it is threads × (graph + tape); phase 11
  fixes it.
- **Performance regressions.** Covered by the benchmark gate.
- **ABI drift.** The `ptd_graph` layout depends on `PHASIC_B3_VALIDATORS`
  (`phasic.h:263-269`). Under R13 no struct crosses compiled copies, and
  the JIT cache key carries the phasic version (0c).
- **Untrusted bytes** (R13, cache files): fuzzed loaders.
- **Long programme with diversions.** New defects go to `OPEN-ITEMS.md`;
  only blockers of the current phase are fixed in it.

## 7. Out of scope

Plan B and exact PDF gradients (R6; OI-002, OI-005); inference-layer items;
features beyond R8; notebook edits (the user's).

## 8. Open items owned by this plan

- **Phase 0:** OI-015, 019, 020, 022, 023, 033, 060, 061, 062, 067, 088,
  089, 110, 111, 113, 116.
- **Later phases:** OI-021, 025, 026, 028, 029, 030, 031, 034, 035, 044,
  045, 046, 047, 048, 049, 050, 051, 053, 054, 055, 056, 058, 065, 077,
  079, 080, 083, 086, 087, 090.
- **Decisions:** OI-009 is settled by §1-§2; OI-007 is "continue" under
  this plan.

## 9. Review dispositions

**Phase renumbering v2 → v2.1:**

| Phase | v2 | v2.1 |
|---|---|---|
| trace deletion | 13 | 2 |
| pybind | 2 | 3 |
| moments | 3 | 4 |
| graph state | 4 | 5 |
| codec | 5 | 6 |
| add_epoch | 8 | 7 |
| joint | 6a-c | 8a-c |
| cache formats | 7 | 9 |
| formulas | 9 | 10 |
| FFI | 10 | 11 |
| user C++ | 11 | 12 |
| C-internal | 12 | 13 |
| split | 14 | 14 |
| cleanup | 15 | 15 |
| doctrine | 16 | 16 |

**Reviews of v1 (reports 1-3)**, with v2 phase numbers:

- **Graph metadata had no owner.** All three reviews → the graph-state and
  schema phase.
- **R12 aimed at retired consumers** → R12'.
- **R11 left C copies** → C kernels.
- **Factorial overflow** → OI-088.
- **Phase 2 was not low risk** → complete-then-promote plus
  exception mapping.
- **Codec versus cache key and pickle** → the codec phase.
- **Joint/epoch rows mis-scoped** → rows plus 6a-c.
- **Generated C++ wider than R4** → rows.
- **Gates lost with the oracle; FP settings** → §4.
- **No user-C++ baseline; corruption hazard** → OI-089, phase 1.
- **Packaging** → R13.
- **Version constants** → per phase.
- **Phases too large** → split.
- **API stability** → done-criterion 2.
- **Half-delegated states** → latency gate.
- **FFI memory** → FFI phase.
- **No performance gate** → benchmark gate.
- **Address-ordered edges** → R14.
- **Concurrency and lifetime** → native checklist.
- **Joint determinism** → index-ordered containers.

**Review of v2 (report 4) → v2.1:**

- **M1** (moments API frozen before discreteness) → the discreteness flag
  and OI-021 settled at the start of phase 4.
- **M2** (variance/covariance, rewarded formulas, MPFR) → new §3 rows, R11
  wording, the per-call-site rule, the drift gate scoped to non-MPFR tapes.
- **M3** (R13 cost for `pmf_from_cpp`) → R13 scope refinement, pending as
  D16.
- **M4** (`load_cpp_builder` normalize) → OI-116 in 0c.
- **M5** (FP-contract rule incoherent) → D15 plus tolerance sources.
- **M6** (shared tape is a C replay redesign) → stated in phase 11, after
  phase 9.
- **M7** (gates) → 200k benchmark point and protocol, fixture-or-waiver,
  checklist phases, fuzzing.
- **M8** (trace deletion early) → phase 2; `is_cache_disabled` relocated;
  OI-083 closes with it.
- **M9** (schema rules) → phase 5.
- **Part A gaps** → the C++ metadata fields acknowledged; July §4a item 3
  retracted; the serialize-keys criterion corrected; `load_cpp_builder`'s
  return type listed; the orphan installed libraries row; the graph-cache
  v0 read rule; TSan.
- **Minors** → m1: the R14 scope in 0b; m2: add_epoch before the joint
  family; m3: the §4a invariants; m4: drift-gate reward scaling.
