# Core-engine refactor: C/C++ as the single source of every core computation

**Status: v2.11, 2026-11-26. Fifteen adversarial reviews so far (Opus):
three of v1 (`b9178dab`), one each of v2 (`256c788a`), v2.2 (`f326cae7`),
v2.3 to v2.10, two of v2.1 (`900b8016`). v2.10 moves all
`load_cpp_builder` work (its source rule, cache key, secure directory,
tests and OI-116) into phase 0d, since it only compiles from 0d. All said CHANGES REQUIRED; none
rejected the goal. Findings and the user's decisions are folded in
(dispositions in §9). The user approved D15, D16 and D17 on 2026-11-26
(§2; D18 on the hand-off format added later the same day). v2.11 is NOT re-reviewed. One decision is pending with the user (D19, §2). No code changed.**

This is a master plan: each phase gets its own detailed sub-plan, reviewed
before any code. It supersedes the unexecuted parts of
[stage3-refactor-plan.md](stage3-refactor-plan.md) (July 2026), which stays
as history; its doctrine and the decisions listed in §2 carry over, except
its §4a item 3 ("`evaluate_trace_jax` is not dead code"), retracted by R12'.
Review reports: `experiments/core_ssot/review{1..15}/REPORT.md`.

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
> and `OPEN-ITEMS.md`; hand the user the phase's list of affected notebook
> cells and the result of the notebook execute-smoke (§4). Memory-cap every run; one heavy run at a time; never
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
- **callback-mode weights** (`weight_mode='callback'`, arbitrary Python).
  The callable stays a Python-side attribute; it is not part of the C++
  graph-state record. Its mechanism rewrites the serialize() dict per theta
  (`_apply_weight_callback`, `__init__.py:893-966`), so the public dict
  format is load-bearing (phase 5). Today a graph-cache round trip of a
  callback-mode graph raises (`__init__.py:4045`); that is pinned.
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
   - pickling of the objects that pickle today (`StateIndexer` and its
     parts, which phase 6 touches; SVGD/MCMC result objects); `Graph` does
     NOT pickle today (`TypeError: cannot pickle 'Graph' object`, measured,
     `experiments/core_ssot/review9/p_pickle_wrap.log`), and that is pinned
     so no phase makes it half-picklable;
   - `Graph(g)` wrap semantics: a wrap shares the underlying C graph, but
     Python-level state such as `is_discrete` is reset on the wrap and the
     SOURCE is unchanged (`__init__.py:1990-1995`; tested,
     `test_theta_layout.py:94-98`);
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
  are fixed in phases 0c and 0d: OI-089 (the /tmp path, the stale-library key,
  missing sources) and OI-116 (`normalize()`). Phase 0c fixes
  `pmf_from_cpp` (installed sources, source rule, cache key, secure
  directory); everything for `load_cpp_builder` (the same three plus its
  compile fix, OI-116 and the bytes hand-off) lands in 0d, because that
  route only compiles from 0d.
  - **No struct crosses compiled copies, from phase 0d on (reviews 9 and
    10).** `load_cpp_builder` compiles the live source tree
    (`phasic_pybind.cpp:5143-5147`, `:5192-5200`) and shares the resulting
    graph POINTER with the extension (the copy at about `:5267`). That pointer carries
    structs partly defined privately in `phasic.c` (`ptd_weight_tape`,
    `phasic.c:5106`; the offset-tape and cache structs `:1871-1891`) and
    C++ members (`std::string`, `std::function`) whose layout depends on
    `PHASIC_B3_VALIDATORS` (`phasic.h:263-270`) and on the compiler's
    standard-library ABI. No hash of sources can detect all of that.
  - **The fix:** in phase 0d, `load_cpp_builder`'s hand-off moves to plain
    bytes, ahead of the rest of R13 (a timing refinement of D16), so the
    layout/ABI hazard disappears rather than being detected and no layout
    fingerprint is needed.
  - **Format (user decision D18, 2026-11-26): a flat, versioned BINARY
    buffer rebuilt in C++.** JSON was rejected because the builder runs
    once per parameter value. Review 11 measured the JSON round trip for a
    lean 600k-vertex graph at 3.84 s (Python serialize) plus 2.65 s
    (`from_serialized`, a Python loop) per call, against ~35 µs today
    (`experiments/core_ssot/review11/p_bytes_cost_600k.log`). The buffer:
    - is plain arrays behind a header with a format version constant;
    - is WRITTEN by a new C++ emitter in the JIT copy and REBUILT by a C++
      reader in the extension, with no Python loops (native O(V));
    - carries everything a C++-built graph carries: states (including
      duplicates) and vertex indices; regular, parameterized, constant and
      starting-vertex (incl. parameterized IPV) edges; `is_aux`;
      `was_dph`; weight mode; the formula source and tape; the theta
      layout (`_theta_blocks`, `_theta_names`); epoch metadata
      (`_epoch_state_index`, `_n_epochs`, `_base_param_length`);
    - stores doubles bitwise, so infinities and NaNs survive and there is
      no precision loss;
    - comes back from the JIT copy through an exported free function,
      because allocators differ between copies;
    - preserves edge ORDER. Today's serializers reorder interleaved edge
      kinds (`_graph_serialize.py:550-660`), and both drop a constant edge
      parallel to a parameterized one (`phasiccpp.cpp:1372`). The buffer
      fixes the first; the second is recorded as a known limit in the 0d
      test, or fixed if it is cheap.
  - **The seed of phase 5.** This emitter/reader pair is the seed of phase
    5's single C++ serialize owner. Phase 5 extends it (and keeps the
    public Python `serialize()` dict), so phase 5 does not inherit three
    serializers.
  - **Return type.** `load_cpp_builder` then returns a `Graph`, not the raw
    `_Graph`. Listed for approval with its behaviour differences: `Graph`
    overrides `pdf` (raises `ValueError` for NaN or negative times,
    `TypeError` for a non-integer granularity, dispatches on
    `is_discrete`, `__init__.py:2763-2776`), `sample` (reward-coverage
    checks and warnings), `normalize` (cache invalidation) and others
    (`:2334`, `:2521`, `:2584-2736`, `:2850`, `:3102`).
    `isinstance(g, _Graph)` still holds. Also listed: a C++ graph's vertex
    lookup tree may not hold every vertex or may map a duplicate state to
    a later vertex (`phasic::Graph::create_vertex` does not insert into it,
    `phasiccpp.cpp:80-86`), so the buffer carries a per-vertex "in lookup
    tree" bit and the reader rebuilds the tree accordingly (the
    duplicate-state test model checks `find_vertex`). Fields such as
    `edge_mode`, `param_length_locked` and `parameterized` are
    reconstructed through the normal add-edge path, not copied;
    `current_params` and `weight_version` start fresh; the CURRENT edge
    weights are carried explicitly, separately from the coefficients. Also
    lost and listed: the C++
    construction callback (`_callback`, `phasiccpp.h:1929`) that the copy
    constructor shares today (`:280`), so a C++ `extend()` on a returned
    graph raises.
  - **Python-side metadata (review 12).** A `Graph(g)` wrap resets
    `is_discrete`, `_weight_mode` and `_weight_formula_tape`
    (`__init__.py:1989-2003`), and `serialize()` and the FFI route on them
    (`_graph_serialize.py:217`, `:247-253`), so a formula-mode C++ graph
    would silently reach the FFI as linear. The Python `load_cpp_builder`
    wrapper therefore sets `is_discrete` (from `was_dph`), `_weight_mode`,
    `_weight_formula` and `_weight_formula_tape` from the buffer header,
    and, when `_n_epochs > 0`, the epoch attributes `_epoch_state_index`,
    `_n_epochs` and `_base_param_length` (Python `add_epoch` decides "first
    epoch" and the base slot count from these Python attributes,
    `__init__.py:3334`, `:3338-3341`; without them a C++-built epoch model
    would be extended as if it were epoch 0, a silently wrong model);
    the tape dict comes from a C getter. `phasic::Graph` has no
    weight-mode field (only `_weight_formula_src`, `phasiccpp.h:1920`), so
    the emitter derives weight mode as formula iff a formula tape or source
    is present, otherwise linear. Log mode is not representable by a C++
    model and is stated as such. The tape dict comes from a NEW C getter
    (only `_set_weight_tape` and `_clear_weight_tape` are bound today,
    `phasic_pybind.cpp:1593-1615`; `ptd_weight_tape` is private,
    `phasic.c:5106`) that reproduces `compile_formula`'s dict shape
    exactly. The wrapper must NOT set `weight_formula = src`, which would
    recompile with the Python compiler and could diverge on select/pow
    (OI-048) from the tape the graph actually evaluates. A native-DPH C++
    model comes back with `is_discrete=False` (C++ has no discreteness flag
    until phase 4; only `was_dph` is known): listed among the return-type
    differences until phase 4, with its remedy (assign
    `g.is_discrete = True`, public API) and, offered in the approval list,
    an additive `discrete=` keyword on `load_cpp_builder` mirroring
    `pmf_from_cpp`.
  - **Today's route does not compile (review 12, measured,
    `experiments/core_ssot/review12/p_today_compile.log`):** the command at
    `phasic_pybind.cpp:5192-5200` fails (`phasic_log.h` not found; then
    missing SCC and log symbols; `scc_compose.c` and `scc_synthetic.c` are
    not valid C++). It builds only when the C files are compiled as C, as
    `pmf_from_cpp` already does (`__init__.py:1163-1193`). The compile fix
    lands in 0d TOGETHER with the bytes hand-off, so the pointer-sharing
    route is never re-enabled (in a `PHASIC_B3_VALIDATORS` build it would
    let the extension's `ptd_graph_destroy` read `_dbg_off_clean` past a
    JIT-allocated struct, `phasic.c:4648-4653`). Nobody can
    depend on its current behaviour.
  - **Per-call cost and its bound.** The baseline is today's route with
    only the compile fixed and no `normalize()`. That build is never
    merged (the compile fix lands only with the hand-off), so the baseline
    is the bare `build_model` time measured INSIDE the JIT copy
    (review 12 measured `build_model` plus normalize: ~0.025 s at
    60k and ~0.28 s at 600k vertices, peak 122 MB,
    `p_native_build.log`). Acceptance: emit plus rebuild <= 1× the
    `build_model` time at 60k, 200k and 600k, with peak RSS recorded. The
    per-call cost (roughly 2× today's broken-route equivalent) is an
    exception to §4's 5% rule and done-criterion 5: decision D19, pending.
  - **Hand-off order:** emit, free the JIT graph, then rebuild, so peak
    memory is one graph plus the buffer. The buffer can approach the
    graph's size for epoch and joint models, whose coefficient vectors have
    length `param_length` (`phasic.h:279`).
  - **Trust and scope:** the buffer is in-process only, never persisted or
    cached, so the v0 read rule, phase 9 and §4's public-dict rule do not
    apply. The reader is not exposed as a Python callable taking arbitrary
    bytes; it validates the header, lengths and bounds before any
    allocation. A version mismatch is a hard error.
  - **Sources on a normal install (review 10, measured,
    `experiments/core_ssot/review10/p_pkgdir.log`):** both routes fail
    today without a source checkout. `pmf_from_cpp` needs
    `PHASIC_SOURCE_DIR` (`__init__.py:1138-1150`); `load_cpp_builder` uses
    the compile-time `__FILE__` and ignores `PHASIC_SOURCE_DIR`. The
    package installs only part of the source set (`CMakeLists.txt:365-386`).
    Phase 0c (for `pmf_from_cpp`) and phase 0d (for `load_cpp_builder`,
    which only compiles from 0d) therefore:
    - installs the complete JIT source set into `phasic/include`;
    - gives both routes ONE source-resolution rule: the installed copy by
      default, with `PHASIC_SOURCE_DIR` as an explicit override honoured by
      both;
    - keys each library cache on hash(user source, the whole installed
      JIT source set including headers (`phasic.h`, `phasic_log.h`,
      `phasiccpp.h`, `scc_graph.h`, `user_model.h`), compile flags, and
      the compiler's path and version), so a changed tree or compiler
      recompiles (staleness only: no struct crosses, so no refusal is
      needed);
    - states the supported install modes (wheel, conda, editable) in R7.
      `PHASIC_SOURCE_DIR` is a DEVELOPER-ONLY override; in addition the
      0d buffer header carries a hash of the emitter/reader source, so a
      mismatched source tree is a clear error rather than a silent
      misreading.
  - **`pmf_from_cpp`** exchanges only arrays through ctypes
    (`__init__.py:1213-1266`). Its behaviour is unchanged apart from
    working on a normal install; it gets no new refusal.
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
  **Amended by D17:** the whole of `trace_elimination.py`
  (`record_elimination_trace`, `evaluate_trace`, `evaluate_trace_jax`,
  `instantiate_from_trace`, `EliminationTrace`) survives as a tests-only
  oracle under `tests/` (no public import path, imports rewritten,
  `trace_elimination.py:44`, `:449`, `:1412`) until phase 13 merges; the
  rest (hierarchical trace cache, trace serialization, trace-cache
  statistics, `compute_trace`, `_ensure_trace`) is deleted in phase 2.
  - **The oracle's limits:** it refuses cyclic graphs
    (`trace_elimination.py:844-852`) and formula mode (`:442`), so it covers
    acyclic linear and log graphs only. Its `instantiate_from_trace` path
    calls `.moments()`, so in phase 4 it checks elimination, not the moment
    recurrence. The cyclic and formula elimination cells are covered by
    phase 1's golden fixtures instead.
  The unused C port of the trace format (`ptd_record_elimination_trace`,
  `ptd_evaluate_trace`, in the frozen public header) is asked about at the
  same time.
- **R13** Graphs built by user C++ code reach Python only as versioned
  serialized bytes; no live pointer crosses compiled copies.
  - **Scope (refined from review 4; D16 approved):** this applies
    to build-once models (R8) and to `load_cpp_builder`, whose pointer
    sharing is the only real crossing today (`phasic_pybind.cpp`, about `:5267`).
  - `pmf_from_cpp` computes inside its JIT library and returns only
    numbers, so it already satisfies R13. It keeps doing so.
  - Routing it through bytes would add serialize + parse per theta and per
    FD probe: measured ~2.2 s at 37k vertices with today's Python
    serializer, `experiments/core_ssot/review4/probe_serialize_cost.log`.
- **R14** Approved exception to R6: address-ordered edges become
  index-ordered (phase 0b).

**Pending decision D19 (must be decided before phase 0d's sub-plan is
written):** accept the 0d per-call cost of `load_cpp_builder`'s bytes
hand-off (bound: emit + rebuild <= 1× the native build time, i.e. roughly
2× a native build per call) as an exception to §4's 5% rule and
done-criterion 5. The route does not compile today, so no user depends on
its current speed. No bytes hand-off can meet the 5% rule; the
alternative if D19 is rejected is that `load_cpp_builder` raises a clear
`NotImplementedError` until phase 12.

**Decision D18 (user, 2026-11-26):** `load_cpp_builder` hands its graph over as a flat versioned BINARY buffer rebuilt in C++ (R7), not JSON.

**Decisions D15-D17 (APPROVED by the user, 2026-11-26):**

- **D15 (phase 1), APPROVED.** A build-wide `-ffp-contract=off` (no fused multiply-add)
  so bit identity survives code moving between files and C/C++ ports of
  Python arithmetic. It changes last digits wherever the compiler fuses
  today, so it is an R6 exception. Baselines are taken after the flip;
  one benchmark protocol run (N = 5, median, §4) is made just before and
  one just after it, and the
  performance difference is recorded with the exception (63 fused
  multiply-adds removed; the cost is unmeasured).
  - **Alternative:** keep today's flags and gate only same-file moves
    bitwise; tolerance gates elsewhere.
  - **Evidence (review 6):** the arm64 extension holds 63 fused
    multiply-add instructions (in the waiting time, both elimination
    builders, the reward transform, the moments gradient and the sojourn
    kernels); x86 builds hold none. So without D15, arm64 and x86 cannot be
    bit-identical, and neither can the R11 drift gate or any C++ port of
    Python arithmetic. Reviews 5 and 6 both recommend approving.
  - **Scope:** the extension, and also the JIT compile commands
    (`__init__.py:1179`, `:1187`; 0d's rewritten `load_cpp_builder` compile command) and the Rcpp
    example; the MSVC `/fp:contract` setting for the Windows wheels is
    recorded.
- **D16, APPROVED.** The R13 scope refinement above (reviews 5 and 6 both
  recommend approving).
- **D17 (from review 6), APPROVED as recommended** (tests-only oracle until phase 13). The Python trace recorder is today the only
  independent elimination oracle in the gate corpus: 20 test files use it,
  including `test_gate_trace_ffi_equivalence.py`,
  `test_gate_elimination_bit_identity.py` and
  `inference/test_nan_observations_correctness.py`. Deleting it in phase 2
  removes that oracle just before phases 3, 4 and 13 (one elimination
  driver) need it.
  - **Recommendation:** amend R12' so that `trace_elimination.py` survives
    as a TESTS-ONLY oracle (moved under `tests/`, no public import path),
    like `compile_formula` under R3, until phase 13 merges. Everything
    else in the pipeline is deleted in phase 2.
  - **Alternative:** phase 2 converts every oracle test into frozen
    fixtures over a broad corpus (cyclic, log mode, discrete, n >= 1k) and
    names a replacement oracle (exact-rational or MPFR).

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
  are deleted in phase 2, except `trace_elimination.py`, which moves under
  `tests/` as the oracle (D17) until phase 13.
  - **Still reachable in public API (review 6, read from source):**
    - imported when `phasic` is imported (`__init__.py:416`, the package
      exports `get_trace_cache_stats` and `print_trace_cache_info`; `:438`,
      `EliminationTrace`);
    - `get_all_cache_stats()` (not deprecated; its result has a `'trace'`
      key, `model_export.py:236-241`), `print_all_cache_info()` and
      `clear_caches()`;
    - the deprecated `phasic.cache.clear_trace_cache`, `trace_cache_info`
      and `clear_all_caches`;
    - the deprecated `Graph(cache_trace=...)` keyword
      (`__init__.py:1812-1827`), and the `phasic.trace_elimination` import
      path.

    Phase 2's removal list names all of them. The change to
    `get_all_cache_stats()`'s result shape is a listed exception to
    done-criterion 2.
  - **`is_cache_disabled`** (`trace_serialization.py`) does NOT gate the C
    reward-compute cache (the C side reads the environment itself). It is
    called only inside `trace_serialization.py` (`:273`, `:366`, `:454`),
    reported by `cache.py:181`, re-exported (`cache.py:48-55`) and tested
    (`test_cache_module.py:146`). Its relocation is a cache-info
    compatibility item. It returns True when `_HAS_C_BINDINGS` is false, a
    flag set by importing the C trace getters (`:56-67`, `:100`).
  - **Import order:** `cache.py` imports `trace_serialization`, which loads
    `phasic_pybind` early. `graph_cache.py:83-86` documents an import-order
    constraint around `OMP_NUM_THREADS`. Phase 2 keeps that load order, or
    changes it deliberately as a listed item.
  - **The pybind `_c_trace_*` and `_c_load_trace_from_cache` bindings**
    (`phasic_pybind.cpp:3885-3953`) go into the same question as the C
    trace port.

### Capabilities

| Capability | Implementations today | Target | Phase |
|---|---|---|---|
| Python trace pipeline | `trace_elimination.py`, `hierarchical_trace_cache.py`, `trace_serialization.py`, `trace_cache.py`, `compute_trace`, `_ensure_trace`, `Graph._trace` / `_cache_trace` / `_trace_dirty` (set at `__init__.py:1924`, `:1998`, `:10304-10311`; `_ensure_trace` at `:2128-2152`) | deleted (R12'); `is_cache_disabled` relocated | 2 |
| Construction from a callback | four pybind builders (`phasic_pybind.cpp:431-660`). They accept parameterized IPV edges, pass a numpy state argument and accept zero-probability IPV; they leak (`new Graph` without delete, `:434-480`). Also `phasic::Graph::from_callback` / `extend` (`phasiccpp.cpp:196`, `:414`; constant IPV only, rejects <= 0) and Python IPV validation | one path in `phasic::Graph`, after parity | 3 |
| Moment recurrence | pybind `_moments` (`phasic_pybind.cpp:1832`; 32-bit `fac()`, OI-088); `GraphBuilder::compute_moments_impl` (`graph_builder.cpp:644`); `phasic::Graph::moments` (`phasiccpp.h:731`; `long` factorial, wrong for k >= 21); the generated C++; the gradient-core chain | `ptd_moments` + `phasic::Graph::moments` (R11) | 4 |
| Variance, covariance (continuous, discrete, rewarded) | pybind `_variance`, `_covariance`, `_variance_discrete`, `_covariance_discrete` (`phasic_pybind.cpp:284-420`); `phasic::Graph` copies (`phasiccpp.h:779-880`). The rewarded discrete variance is `momentsr[1] - momentsr[0]^2 - momentsrr[0]` (`phasic_pybind.cpp:393`), not the Stirling correction | C kernels, or listed as sanctioned | 4 |
| Discrete moment correction | Python `_continuous_to_discrete_moments` (`__init__.py:644-687`); `GraphBuilder::continuous_to_discrete_moments` (`graph_builder.cpp:826`, called `:982`, `:1020`, `:1045`; FFI `:548`, `:575`, `:772`, `:845`); C `ptd_dph_*` (`phasic.c:11552-11600`) | `ptd_continuous_to_discrete_moments` | 4 |
| Rewarded moments | two legitimate algorithms: the FFI reward-transforms first, then corrects (`graph_builder.cpp:990-1020`); pybind and `phasic::Graph` scale the recurrence by rewards | one rule per call site, recorded in phase 4 | 4 |
| Discreteness of a graph | Python `is_discrete` attribute; C knows only `was_dph`; the Python dispatch (`__init__.py:2575-2580`, `:2826`, ~`:2900`) | a C++-owned flag (per `phasic::Graph` object) plus one discrete rule (OI-021), settled in phase 4 before the moments API is public; C kernels take discreteness as an argument | 4 |
| Matrix export / import | bound lambdas (`:3295`, `:3381`); `phasic::Graph` port; orphan Eigen class | `phasic::Graph`, bound | 4 |
| Generated-C++ graph emitter | `_generate_cpp_from_graph` (`__init__.py:968-1095`): `moments_from_graph` (R4) and constant `pmf_from_graph` (`:4027` → `pmf_from_cpp` `:4234`) | retired; constant `pmf_from_graph` routed through the extension, with a named A/B gate (the JIT copy is compiled without MPFR or OpenMP) | 4 |
| Graph state | Python attributes (`is_discrete`, `_weight_mode`, `_weight_formula(_tape)`, `_weight_callback`, `_indexer`, `_base_indexer`, joint metadata `__init__.py:11332-11340`, epoch metadata `:10319-10320` in `_WIDEN_CARRIED_ATTRS`). The C++ graph already holds `_weight_formula_src`, `_epoch_state_index`, `_n_epochs`, `_theta_blocks`, `_theta_names` and `_callback` (`phasiccpp.h:275-281`), and the copy constructor shares `rf_graph` (`:254-266`). `clone()` copies a subset (`__init__.py:10291-10312`) | one C++-owned record (reconciling both sources), carried by clone, copy and serialize. Fields that are Python objects stay NAMED Python-side attributes, each with its clone/copy/pickle/widen rule: `_weight_callback`; `_base_callback`, `_base_callback_kwargs` (`__init__.py:3506-3507`, `_WIDEN_CARRIED_ATTRS` `:10316-10320`); `_callback`, `_callback_kwargs` (`:1921`, `:3503`, `:10365`; reconciled with the C++ graph's own `_callback`, `phasiccpp.h:280`, `:1926`); `_joint_prob_source` (a reference to another Graph, `:11532`, read `:11947`); `_rewarded_props` (`:11332`, `:12353`); `_t_aux_map`, `_t_vertex_indices`, `_ipv_target_indices`, `_ipv_target_states` (`:11527-11536`, `:11692-11694`; these may instead become record fields, decided in the phase-5 sub-plan); `_graph_cache_entry`, `_last_theta`, `_last_ipv`; and, until phase 6, `_indexer` / `_base_indexer` / `_joint_prob_base_graph_indexer` (the Python indexer is authoritative, the record stores its `to_dict()`, and a consistency assertion runs on serialize) | 5 |
| serialize / from_serialized | Python `_graph_serialize.py` (`:217-242`; reader `:700-736` has no version check); `GraphBuilder::parse_structure`; `phasic::Graph::serialize` (`phasiccpp.cpp:1317`, `:1407`). Consumers of the dict: `graph_cache.py:211`, `:347`; `distributed_scc.serialize_scc_synth` (`distributed_scc.py:77`); `compute_repository`; `_apply_weight_callback` (`__init__.py:893-966`, called `:4056`, `:8536`, `:8609`, `:9677`; relies on keys, numpy dtypes and edge order); `__init__.py:692-703`, `:984-991`; `ffi_wrappers.py:103`, `:142`, `:561`; `bffg.py:318`, `:492`; `zero_inflation.py:147`; `probability_matching.py:296`. The structure JSON is an XLA attribute embedded in the HLO (15.8 MB at 37k vertices, ~480 B/vertex) | one versioned C++ owner, hardened as a trust boundary (R13). The public `serialize()` keeps returning today's dict (keys, dtypes, edge order); any binary format, or passing a registered handle to the FFI instead of the JSON attribute, is internal only | 5 |
| Graph-metadata reads in the Python layers that stay | the inference layer routes on private attributes read with getattr defaults, so a moved or renamed attribute misroutes silently (e.g. a log-mode graph treated as linear, which `__init__.py:7857` records as a wrong answer with no error): `svgd_config.py:469`, `:471`, `:686`, `:759`, `:764`; `svgd.py:1861`; `probability_matching.py:320`; 29 more in `__init__.py` (e.g. `:762`, `:7861`, `:9298`, `:9335`) | a census at the start of phase 5; each read becomes a strict accessor or a property over the C++ record; a routing gate (§4) | 5 |
| State layout and encoding | Python `StateIndexer` (class ~1,280 lines, `state_indexing.py:680-1959`; module 2,266) | C++ `IndexCodec` (R2) | 6 |
| add_epoch | Python (`__init__.py:3246`); C++ port (unbound) | C++ after parity (OI-025) | 7 |
| joint_prob_graph | Python (`__init__.py:10703-11342`) | C++ | 8a |
| Epoch family and forward step | `joint_stop_prob_graph` (`:11343`); `joint_sojourn_graph` (`:11548`, returns `EpochContext` `:12433`); `_probe_daisy_t_eval` (`:12055-12186`); `EpochContext.next_ipv` / `_collapse_one` / `auto_t` (`:12594-12684`); the r_v read (`:12321-12326`) vs the C++ FFI (`graph_builder_ffi.cpp:2222-2235`); t-vertex discovery (`:12296-12303`) | C++, one copy each | 8b |
| Tables and walks | `joint_prob_table` (`:12366`); observation mapping; `_graph_reward_validation.py:13-62`; `absorbing_state_rewards` | C++ | 8c |
| Elimination-cache formats | rev-3; rev-1/2 loader (unhardened, OI-079); per-SCC files; pybind `cache_file_path` hard-codes `$HOME/.phasic_cache` (`phasic_pybind.cpp:3706-3712`) while the C side honours `PHASIC_CACHE_DIR` (`phasic.c:3886-3910`); `clear_caches()` ignores `PHASIC_CACHE_DIR` (OI-024) | rev-3 only (or hardened); invariants in code (OI-087); one cache-path rule | 9 |
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
- **Corpus axes.** Every golden corpus spans weight mode {linear, log,
  formula, callback} × graph kind {continuous, native DPH, was_dph, joint,
  epoch} × rewards {none, 1-D, 2-D} × {exposure, no exposure} × {plain,
  SCC/hierarchical route}. Each cell is present or waived by name (OI-065
  is tracked here).
- **Refusal-matrix fixture** (phase 1). It EXTENDS Plan A's Batch 0
  refusal grid (Plan A runs first) and must agree with it on the
  OVERLAPPING cells (same weight mode × graph kind × rewards × builder →
  same refuse/accept; the two use different probe values). For weight mode
  × graph kind × rewards {none, 1-D, 2-D} × read-out path (including
  `pmf_from_cpp` and `load_cpp_builder`) × probe {-1e-12, -1e-9, -5e-8,
  -1e-7, 0}, record whether the forward refuses and with which message.
  Phase 0c, which precedes it, checks against Plan A's Batch 0 grid, and
  the phase-1 baseline is taken after 0d (which changes
  `load_cpp_builder`'s return type and with it `pdf`'s refusals). Every
  phase that moves a refusal
  reproduces it, or lists the change for approval and updates Plan A's
  `model_refuses()` in the same commit:
  0c (discrete reward-transform validation), 3, 4 (one discrete rule:
  row-sum validation on more read-outs), 7, 8a (was_dph producer;
  mutation-rate checks), 8b, 10, 11, 12, 13. Phase 5 replaces the
  implementation of Plan A's `model_refuses(build_info, has_rewards)` by a
  C++ function `refuses(record, has_rewards)` on the per-object graph-state
  record, with `pmf_from_cpp` refusing by constant. Plan A's second
  derivation of the flag, inside the `reward_visit_probability` backward
  (from its `structure_json`), is routed through the same function in
  phase 5.
- **Routing gate** (frozen in phase 1; run on EVERY merge after phase 1,
  including 13-15, which split the Graph class and delete dead code). The
  corpus includes `Graph(g)` wraps of each kind and graphs returned by
  `load_cpp_builder`: a linear, a formula-mode, a was_dph and an epoch one
  (the phase-1 baseline is taken after 0d). On every graph kind, run `_classify_graph_kind`, the
  effective-weight-mode snapshot and the `probability_matching` /
  `method_of_moments` dispatch, and compare with the frozen result (phase
  4 changes the discreteness flag these read with defaults,
  `svgd_config.py:686`, `:759`; phase 0d changes `load_cpp_builder`'s
  return type).
- **Notebooks.** Each sub-plan ships the generated list of affected
  notebook cells (a scan like `experiments/core_ssot/review5/nbscan.py`,
  moved to `scripts/`). Each merge runs a memory-capped execute-smoke of
  the affected notebooks on read-only copies in the worktree. They are
  never edited; failing cells go to the user. Today nothing runs them:
  `pixi run test` only converts them (`pixi.toml:209`), and pytest
  collects `test_*.py` only.

**Bit identity:**

- A/B on the same platform, after D15 is applied.
- Every bit-identity gate also re-runs with the heap perturbed by
  interleaved allocations inside one process (separate processes probably
  keep the same relative address order).
- CI runs the gates on arm64 and x86.
- Tolerance exceptions only for listed sources, with the measured
  difference, approved by the user. Listed sources are arithmetic-order
  changes, numpy pairwise reductions
  (for example `__init__.py:12588`), and libm differences.

**Performance and memory.** A memory-capped benchmark per family:

- coalescent at ~60k and ~200k vertices; two_island at ~20k; two-locus as
  large as the cap allows;
- joint_prob_graph construction and SVGD iteration at ~45k and at ~600k
  or as large as the memory cap allows, recorded (discrete, the production
  default); an epoch (daisy) model with
  exposure; a formula-mode model;
- records elimination time, tape length, peak RSS, and time per SVGD
  iteration through the FFI;
- construction benchmarks at 60k and 600k states (measured: most of
  today's construction time is Python, the user callback 57-59% and
  phasic's per-transition Python wrapper most of the rest, so the codec
  latency gate matters more than native construction time);
- protocol: N = 5 runs, median, fixed machine, no other heavy process; a
  median regression above 5% fails.

**Native checklist** (phases 0d, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14;
for 0d, which precedes the phase-1 CI jobs, the ASan and malloc-scribble
runs and a small header/length fuzz smoke of the reader are run locally):

- the thread-local teardown test plus a long-run shutdown smoke test with
  XLA worker threads;
- an OpenMP run with batch > 1;
- a test where a Python callback raises;
- a malloc-scribble run;
- one ASan/UBSan and one TSan CI job;
- new native caches use the live-pointer pattern;
- phase 5 checks clone/copy metadata and GraphBuilder reading the schema;
- phase 6 runs the native codec inside 600k callback builds;
- phases 5, 9, 10 and 12 fuzz every loader of untrusted bytes (serialize,
  the cache files, formula source carried in serialized bytes) under ASan;
- phase 14 checks every relocated thread-local and static.

**API stability.** The phase-3 exception-mapping layer; pickle round trips
and old-pickle loading of the objects that pickle today (and the pinned
`Graph` pickling `TypeError`); `Graph(g)` wrap semantics (source unchanged,
wrap reset); graph-cache key stability; a read rule for
unversioned (v0) graph-cache entries. Phase 0c also makes TODAY's
serialize reader reject unknown versions and keys, so a rolled-back build
cannot silently accept a newer entry.

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
| 0a | Prerequisites | CI runs the broad suite (OI-062); gate tooling into `scripts/` (OI-067); the notebook scan script into `scripts/` (0b and 0c already change notebook-visible numbers); plot-test and flaky-test fixes (OI-060/061); commit, prune, push (OI-110/111/113) | S | not started |
| 0b | Determinism (R14) | Address order → index order everywhere: the `arr_c_cmp` sorts (`phasic.c:6088-6094`, used at `:6327`, `:7414`, `:7784`, `:8180`), the pointer comparisons in the merge loops (`:6489`, `:7535`, `:7926`, `:8301`) and the pointer-arithmetic sentinels (`:6187`/`:6247`, `:7275`/`:7324`, `:7643`/`:7696`, `:8052`/`:8101`), all in the `ptd_vtx_order_key` form (`:8541`). Confirm and close OI-022, OI-023, and OI-015 (time-boxed; the FFI rewarded path reward-transforms per theta, `graph_builder.cpp:993`). Heap-perturbation harness (in-process). **Listed changes:** last digits of constant-model read-outs; the EDGE ORDER and `serialize()` output of reward-transformed graphs (measured address-dependent today: 3 orders in 16 instances, `experiments/core_ssot/review6/r14b.log`), and any content hash or cache key computed from them; seeded `sample()` sequences on reward-transformed graphs. | M | not started |
| 0c | Cheap core fixes | OI-088 factorial; OI-118 the `phasic::Vertex` / `Graph` copy-assignment segfault (`phasiccpp.h:~1840-1900`, root cause in copy assignment); OI-089's source and cache parts for `pmf_from_cpp` per R7 (the complete JIT source set installed, one source-resolution rule with `PHASIC_SOURCE_DIR` as override, a cache key over every compiled source file and the flags, a secure cache directory; tests on `pmf_from_cpp`: a changed source tree recompiles, it works on a non-editable install); OI-033 C++ discretize checks; OI-019/020 discrete reward transform; today's serialize reader treats unknown versions or keys as a cache MISS for graph-cache entries (a clear error elsewhere), after enumerating every producer of dicts fed to `from_serialized` (`serialize_scc_synth` `distributed_scc.py:77`, `compute_repository`, `_apply_weight_callback`) | S (relative to 0d; M in absolute terms) | not started |
| 0d | `load_cpp_builder` bytes hand-off (D18) | `load_cpp_builder`'s compile fixed (C files compiled as C, with `-fvisibility=hidden` exporting only the entry, free and test-walk functions, plus `-Wl,-Bsymbolic` on Linux, so a host that loaded another phasic copy globally cannot interpose its symbols) in the same phase as the hand-off; its source rule, cache key and secure directory (as for `pmf_from_cpp` in 0c); OI-116 (no `normalize()`; the wrapper template, `phasic_pybind.cpp:5172-5186`, regenerated); tests: a changed source tree recompiles, it works on a non-editable install; an INDEPENDENT oracle: a test-only function exported by the JIT copy writes a canonical C-level walk of the SOURCE graph BEFORE it is freed (vertex index, state, is_aux, in-tree bit, each edge's target, weight bits and coefficient bits in array order, IPV incl. parameterized start edges, was_dph, param_length, formula source, theta blocks and names, epoch fields, and the formula tape's ops and constant bits), plus one comparison of the post-`update_weights` edge weights at a fixed theta on both graphs, bitwise (both copies compiled with the same explicit `-ffp-contract` setting; a model-level read-out such as `expected_waiting_time` is NOT used, because the extension may switch to MPFR above condition 1e12 or under `PHASIC_FORCE_MPFR`, `phasic.c:10121-10133`, and the JIT copy has no MPFR or OpenMP); the same walk compiled into the extension runs on the rebuilt graph, and the two are compared field by field; each test model also asserts its expected header values from its definition (a was_dph model returns `is_discrete` True, a formula model returns formula); the C++ emitter in the JIT copy and the C++ reader in the extension (unbound, reachable only through the JIT function's buffer); version constant; exported free function; hand-off order emit → free → rebuild; the Python wrapper sets the metadata (R7); returns a `Graph` (differences listed for approval, incl. the lost `_callback`); per-call benchmark against today's route with the compile fixed, bound emit + rebuild <= 1× build time (D19); field-by-field gate: exact edge order against a C-level walk of the rebuilt graph, an idempotence check (emit, rebuild, emit in the extension, byte-compare), theta layout via `_theta_layout`, epoch fields as the Python attributes the wrapper sets (no pybind getter exists today; one may be added in 0d) and an epoch test model built with C++ `add_epoch`, for which Python `add_epoch` on the returned graph must equal Python `add_epoch` on the equivalent Python-built model, the Python attributes, and against Python `serialize()` only the fields it carries (order compared modulo its documented reordering); test models include a formula-mode, a was_dph, a named-slot and a duplicate-state graph and use the C++ vertex-chain idiom (OI-118 is fixed in 0c; 0d's reader and wrapper avoid `Graph` assignment regardless); the native checklist (ASan, malloc-scribble, fuzz smoke) | M | not started |
| 1 | Inventory, gates, baselines | re-check §3 on master; D15 applied (including the JIT commands), then baselines; the refusal-matrix fixture; the routing-gate baseline; golden fixtures for the cyclic and formula elimination cells (outside the D17 oracle's reach); the corpus axes (OI-065); codec latency baseline (the phase-6 gate is relative to it); golden fixtures for phases 2-4; benchmark protocol and baseline; tests and a measured baseline for `pmf_from_cpp` / `load_cpp_builder` (after 0d); arm64 + x86 CI gates; ASan and TSan jobs; supported platforms recorded (Windows wheels are built, `pypi-release.yml:47-51`, but the JIT routes assume `/tmp`, `g++` and POSIX mmap); the orphan installed libraries decided | M | not started |
| 2 | Delete the Python trace pipeline (R12') | removal list for the user: every public name in §3, the `phasic.trace_elimination` import path, `Graph(cache_trace=...)`, the public properties `Graph.cache_trace` (`__init__.py:2179`), `Graph.hierarchical` (`:2202`) and `Graph.trace_valid` (`:2213`, all referenced in `docs/api/Graph.qmd`), the change to `get_all_cache_stats()`'s result, and the affected notebook cells (`trace-and-jax-caching-c-path`, `svgd-joint-prob`, `benchmarks/python/test_hierarchical_caching` cells 2, 8, 10, 12, 14, 18); ask about the C trace port and the `_c_trace_*` bindings; `trace_elimination.py` moves under `tests/` as a tests-only oracle (D17) and is deleted after phase 13 merges; relocate `is_cache_disabled` (redefining its condition); a disposition list (rewrite / delete / keep) for the 24 test files that import the pipeline (`experiments/core_ssot/review8/imports.txt`) plus `test_gate_moments_3way.py:52` (asserts `ga.cache_trace is False`) and `inference/test_hierarchical_graph.py` (uses `compute_trace`), e.g. `test_modeling_compose.py:434-457` (add_epoch moments via `get_trace_hierarchical`, rewritten onto `record_elimination_trace`, an oracle input for phase 7), `test_trace_cache.py`, `test_slurm_cpu_count.py`, `test_pickle_cache_trust.py`, `test_c_trace_reload_param.py`; OI-049, OI-080 (`_pmap_file_cache` lives only in `hierarchical_trace_cache.py:471`, `:542`) and OI-083 close; OI-044 becomes a recorded limitation of the oracle | M | not started |
| 3 | pybind restructure and construction | split into `register_*()` (keeping the 0d binding's exception types and messages) (the D17 oracle tests are in this phase's gate: they use the re-registered `_Graph` methods); C++ exception-mapping layer; one construction path via complete-then-promote. First parity: parameterized IPV edges, zero-probability entries, numpy state argument, the three IPV validations reconciled, every user-visible difference listed for approval. Then fix the builder leak (check whether it explains OI-017's per-fit memory growth), recheck OI-030 (`phasic_pybind.cpp:3743-3763`), and make an OpenMP spawn failure raise (OI-034). Parity also covers phasic's per-transition Python wrapper (`__init__.py:1317`): validation by `assert` (an `AssertionError`, which disappears under `python -O`), two stdout prints, int coercion; the sub-plan decides whether the wrapper is part of the one construction path | M | not started |
| 4 | Moments, variance, covariance, discreteness | the C++-owned (per-object) discreteness flag and OI-021 (one discrete rule, including OI-078's discretized `expectation()`) decided first; `is_discrete` becomes a property WITH A SETTER over the C flag in this phase. The flag's HOME is per `phasic::Graph` OBJECT, copied in the C++ copy constructor like `_weight_formula_src` (`phasiccpp.h:254-266`), NOT in the shared `ptd_graph`: a wrap shares the `ptd_graph`, and a flag stored there would let `Graph(g)`'s reset silently flip the SOURCE to continuous (measured semantics today: the wrap is reset, the source is unchanged, `experiments/core_ssot/review9/p_pickle_wrap.log`). The desync fixture includes wrap-then-assign on both source and wrap (assignment is public API: 13 assignments in `src/phasic`, 15 in tests, e.g. `__init__.py:11327`, `_graph_serialize.py:~713`), so there is never a second discreteness store; its relation to `was_dph` (a distinct latch) is spelled out; a desync fixture assigns the attribute and checks the read-outs and the routing gate; `ptd_moments`, `ptd_continuous_to_discrete_moments`, variance/covariance kernels or sanctioned list; `phasic::Graph::moments` API; rewarded-moments rule per call site; drift gate (non-MPFR tapes, same reward scaling); gates at k >= 13 and k >= 21; matrices; retire the generated-C++ emitter (R4) with the constant `pmf_from_graph` A/B gate; `use_ffi` deprecation; OI-026, OI-031 | L | not started |
| 5 | Graph state and serialize schema | every record field states its HOME (per `phasic::Graph` object and copied on copy, or shared in `ptd_graph`) and keeps today's `Graph(g)` wrap semantics, with any change listed for approval; the census records each field's CURRENT wrap behaviour (today Python attributes are dropped on a wrap and fall back to getattr defaults, e.g. a log-mode wrap is routed as linear, while C++ fields such as `_weight_formula_src` are copied), so a property over a copied record does not silently change routing; starts with a census of every metadata read in `src/phasic` and in the `tests/` oracle (direct and getattr-default, incl. the inference modules); every field listed as a record field or a named Python-side attribute (§3); per-vertex metadata kept OUT of the FFI structure JSON before phase 11 (it is hashed per batch element, `graph_builder_ffi.cpp:274`, `:1542`), or measured at 600k; at merge the Python attributes become properties over the record, so there is ONE store from then on (not three); strict accessors for every name the inference layer reads; the routing gate; Plan A's flag becomes a C++ function. The serialized formula TAPE stays in the schema until phase 10 (alongside the source); the OI-048 select/pow cases are in the phase-5 gate. One C++-owned record reconciling the Python attributes and the existing C++ fields; carried by clone, copy and serialize. Schema rules: formula SOURCE plus a tape version (phase 10 may change tape semantics); layout as `StateIndexer.to_dict()`; joint and epoch fields may be added in phases 7-8 with a version bump; v0 graph-cache entries are read (or rejected) by an explicit rule. Consumers updated (graph cache, `distributed_scc`, `compute_repository`, GraphBuilder); `from_serialized` hardened and fuzzed; the serialize owner EXTENDS the 0d emitter/reader (D18) and absorbs `phasic::Graph::serialize` and `GraphBuilder::parse_structure` (the public Python `serialize()` dict stays); OI-053, OI-054 | L | not started |
| 6 | Index codec | exact API; OI-077 (C++ codec part); native constructor; codec → StateIndexer direction; error-message parity; the codec handle kept OUT of the instance `__dict__` (`callback_hash.py:149-157`, `:653-720`), so graph-cache keys are unchanged (OI-055 checked); pickle tests; scalar call latency no worse than the phase-1 baseline inside 60k and 600k callback builds | L | not started |
| 7 | add_epoch | port promoted after parity (OI-025), before the joint family so it gates on the final epoch-metadata producer; its refusals of non-linear weight modes and was_dph/aux inputs (`__init__.py:3317`) pinned as fixtures with their messages | M | not started |
| 8a | joint_prob_graph in C++ | index-ordered containers only; its refusals (`_require_linear_weight_mode` `__init__.py:10823`, was_dph/aux inputs `:10824`) pinned with messages; OI-056 (constant edges) decided; OI-077 (C++ `joint_prob_graph`); caps, `reward_only`, trash and t-vertex rules, degenerate merge, multi-property rule. The reward-rates callback has two ABIs (Python objects via codec → StateIndexer; a `std::function`), holds the GIL, runs single-threaded and never inside OpenMP. ~53 test files | L | not started |
| 8b | Epoch family in C++ | `joint_stop_prob_graph`, `joint_sojourn_graph`, `EpochContext`, the epoch forward step, with the FFI using the same code; the July §4a invariants kept | L | not started |
| 8c | Tables and walks | `joint_prob_table`, observation mapping, reward-coverage and absorbing-state walks | M | not started |
| 9 | Cache formats | rev-1/2 loader removed or hardened and `pull_cache` consistent (OI-079; ASan + fuzz corpus); cache invariants in code (OI-087); one cache-path rule, incl. `clear_caches()` honouring `PHASIC_CACHE_DIR` (OI-024); version constants | M | not started |
| 10 | Weight formulas | C++ tape reader; formula gate; C++ compiler as single source; select/pow semantics decided (OI-048, OI-051) | M | not started |
| 11 | FFI compute core and shared tape | one compute core for 9 handlers; content-hash model keys; a C replay API with a per-thread arena and input binding so the read-only tape is shared (requires phase 9's OI-087 invariants; the exact-gradient cores in its gates); failures reported (OI-028); the §4a invariants; a memory-capped gate at 60k with OpenMP threads = cores; the gate exercises every SVGD leaf that `svgd_config` routes to (rules R9, R29-R31) | XL | not started |
| 12 | User C++ models | R13 bytes hand-off for build-once models, using the phase-5 serialize owner (which grew from the 0d buffer); one JIT driver; `pmf_from_cpp` stays in-JIT, rebuilding per process (OI-029, OI-058); R8 example and a test against the equivalent Python model; a memory-capped benchmark of the bytes hand-off (build once, serialize, load: peak RSS and time) at 200k and 600k vertices | M | not started |
| 13 | C-internal unification | one SCC order (OI-045, OI-046, OI-035; tolerance exception if bits change); one conditioned sampler (OI-047); one elimination driver with compile-time dispatch or LTO; benchmark gate; MPFR gate scripts rerun | XL | not started |
| 14 | phasic.c split | translation units, all with the D15 flags; constraints are the JIT source list(s) and the Rcpp example (`docs/pages/popgen/coalescent_with_freqbins.cpp:12`) | L | not started |
| 15 | Python cleanup | delete dead implementations and ports (`profile`, the external-pointer SCC scheme OI-050); split the (smaller) `Graph` class; check `bffg.py` (OI-086) | L | not started |
| 16 | Doctrine and guards | `docs/architecture/boundary.md`; reachability and lock-step CI guards; regenerate API docs (OI-090). Sub-plans cite the July §2 doctrine until then | S | not started |

**Stop points.** If the programme ends after a phase, is the code no worse
than today?

| After | Safe? | Reason |
|---|---|---|
| 0a-0c, 1 | yes | fixes and gates only |
| 0d | yes | one extra (binary) serializer exists until phase 5 absorbs it, but `load_cpp_builder` works (it does not compile today) and no struct crosses copies (the compile fix and the hand-off land together) |
| 2 | yes | dead code removed; the oracle kept under `tests/` |
| 3, 4 | yes | each capability fully delegated within the phase (phase 4 includes the `is_discrete` property, so discreteness has one store) |
| 5 | yes, with a caveat | one record plus the named Python-side attributes; the joint and epoch code still run in Python against it |
| 6 | yes | Python `StateIndexer` delegates to the codec; latency gated |
| 7 | yes | add_epoch fully delegated |
| 8a, 8b | no | the joint family is half in C++ (8a without 8b leaves joint graphs built in C++ but read by Python epoch code); plan 8a-8c back to back |
| 8c | yes | the joint family complete |
| 9, 10, 11, 12 | yes | each self-contained |
| 13 | yes | one driver; the D17 oracle deleted after it |
| 14, 15, 16 | yes | structural and documentation |

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
  (`phasic.h:263-269`) and the C++ members on the standard-library ABI.
  From phase 0d no struct crosses compiled copies (R7: `load_cpp_builder`
  hands over bytes), so neither can corrupt memory.
- **Untrusted bytes** (R13, cache files): fuzzed loaders.
- **Long programme with diversions.** New defects go to `OPEN-ITEMS.md`;
  only blockers of the current phase are fixed in it.

## 7. Out of scope

Plan B and exact PDF gradients (R6; OI-002, OI-005); inference-layer items;
features beyond R8; notebook edits (the user's).

## 8. Open items owned by this plan

- **Phase 0:** OI-015, 019, 020, 022, 023, 033, 060, 061, 062, 067, 088,
  089 (`pmf_from_cpp` parts in 0c; `load_cpp_builder` parts and the
  hand-off in 0d), 110, 111, 113, 116 (0d), 118 (0c).
- **Phase 1:** OI-065. **Phase 2:** OI-044 (as an oracle limitation),
  049, 080, 083. **Phase 3:** OI-017 (check against the builder leak), 030,
  034. **Phase 4:** OI-021, 026, 031, 078. **Phase 5:** OI-053, 054.
  **Phase 6:** OI-055, 077 (codec part). **Phase 7:** OI-025. **Phase 8a:**
  OI-056, 077 (`joint_prob_graph`). **Phase 9:** OI-024, 079, 087.
  **Phase 10:** OI-048, 051. **Phase 11:** OI-028. **Phase 12:** OI-029,
  058. **Phase 13:** OI-035, 045, 046, 047. **Phase 15:** OI-050, 086.
  **Phase 16:** OI-090.
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

**Reviews 5 and 6 (of v2.1) → v2.2** (reports `review5/`, `review6/`):

- **Silent metadata reads** (R5 M1) → §3 row, phase-5 census, strict
  accessors, routing gate.
- **No weight-mode × graph-kind matrix** (R5 M2) → §4 corpus axes and the
  refusal-matrix fixture; Plan A's flag becomes a C query in phase 5.
- **Serialize consumers and callback mode** (R5 M3) → §3 row, the public
  dict kept, callback stays Python-side, its cache round trip pinned.
- **Three metadata stores and stop points** (R5 M4) → phase 5 merges with
  one store, the tape stays until phase 10, the stop-point paragraph.
- **Notebooks** (R5 M5) → §4 notebook scan and execute-smoke, §0 prompt,
  phase-2 cell list.
- **Benchmark families** (R5 M6) → joint, epoch-with-exposure,
  formula-mode, every SVGD leaf in phase 11.
- **Trace liveness wrong** (R6 M1) → §3 corrected, phase-2 removal list.
- **The only elimination oracle** (R6 M2) → pending decision D17.
- **R14 scope** (R6 M3) → phase 0b listed changes, in-process harness.
- **D15 evidence and scope** (R6 M4, R5 m1) → §2 D15.
- **Minors.**
  - R5 m2: the reader rejects unknown versions (0c).
  - R5 m3: the phase-2 list.
  - R5 m4: refusals pinned (phases 7, 8a).
  - R6 1: construction parity incl. the Python wrapper; benchmark reading.
  - R6 2: JSON as an XLA attribute; registered handle.
  - R6 3: OI homes.
  - R6 4: checklist phases 7, 10.
  - R6 5: line references; codec gate relative to the phase-1 baseline.

**Review 7 (of v2.2) → v2.3** (report `review7/`):

- **Refusal matrix vs Plan A** (B1) → the matrix extends Plan A's grid,
  adds a rewards axis, `pmf_from_cpp` / `load_cpp_builder` and Plan A's
  probes; the full list of refusal-moving phases; the C query
  `refuses(record, has_rewards)` behind Plan A's `model_refuses()` seam.
- **Routing gate too late** (B2) → frozen in phase 1, run on phases 3-12.
- **D17 contradictions and oracle limits** (B3) → R12' amended; §3; the
  oracle's limits stated; phase-1 golden fixtures for cyclic and formula
  cells; the phase-2 test-disposition list; OI-044 as a limitation; the
  census includes `tests/`.
- **One store not implementable as written** (B4) → named Python-side
  attributes with their rules; the Python indexer authoritative until
  phase 6, with a consistency assertion.
- **Stop points** (B5) → per-phase table with reasons.
- **Minors.**
  - m1: OI-080 closes in phase 2.
  - m2: a single §8 list.
  - m3: the notebook scan moves to 0a.
  - m4: the strict reader means a cache miss; producers enumerated.
  - m5: per-vertex metadata kept out of the FFI JSON.
  - m6: oracle imports rewritten; the oracle in phase 3's gate.
  - m7: the 600k benchmark capped.
  - m8: the Plan A seam.
  - m9: stale D15 wording removed.

**Review 8 (of v2.3) → v2.4** (report `review8/`):

- **The 0c key does not protect the programme** (M1) → a build-ID key in
  R7, §6 and the 0c row, with a layout-change test.
- **A second discreteness store** (M2) → phase 4 makes `is_discrete` a
  property with a setter; a desync fixture; the stop point is qualified.
- **Minors.**
  - 1: 0c checks against Plan A's grid; the rvp derivation goes through
    the C query.
  - 2: the phase-2 lists are completed.
  - 3: the routing gate runs on every merge after phase 1.
  - 4: the named-attribute list is completed.
  - 5: the `is_cache_disabled` description is corrected; import order.
  - 6: fixed latency numbers dropped.
  - 7: D15's performance cost is measured.
  - 8: a phase-12 bytes hand-off benchmark.
  - The leftover notebook-scan mention in phase 1 is removed.

**Review 9 (of v2.4) → v2.5** (report `review9/`):

- **The build-ID key does not guard against layout mismatch** (M1) →
  R7's layout fingerprint embedded in the extension, checked by both JIT
  routes; the JIT reproduces the layout defines; 0c tests without an
  extension rebuild.
- **The home of the discreteness flag and wrap semantics** (M2) → the
  flag lives per `phasic::Graph` object; wrap semantics are pinned in
  done-criterion 2 and §4; the desync fixture and routing corpus include
  wraps; phase 5 states each record field's home.
- **Minors.**
  - m1: the phase-12 wording; OI-089 rows in the register.
  - m2: pickling names its objects; the `Graph` TypeError pinned.
  - m3: a raw `_Graph` in the routing corpus.
  - m4: the D15 benchmark protocol.
  - m5: the status header.

**Review 10 (of v2.5) → v2.6** (report `review10/`):

- **The layout fingerprint could not work as specified** (M1): private
  structs in `phasic.c`, transitive headers, the standard-library ABI, and
  no sources on a normal install. Replaced by removing the hazard instead
  of detecting it: in phase 0c `load_cpp_builder` hands over plain bytes,
  ahead of the rest of R13 (a timing refinement of D16). Also: the complete
  JIT source set installed, one source-resolution rule with
  `PHASIC_SOURCE_DIR` as override, a staleness key over every compiled
  file, supported install modes stated, `pmf_from_cpp` unchanged (no new
  refusal).
- **Minors.**
  - m1: "C-owned" becomes "C++-owned (per object)"; the query is a C++
    function; C kernels take discreteness as an argument.
  - m2: the 0c test hooks are moot without the fingerprint; new tests are
    named.
  - m3: the phase-5 census records each field's current wrap behaviour.
  - m4: hash canonicalisation is moot.

**Review 11 (of v2.6) → v2.7** (report `review11/`):

- **Per-call cost of the hand-off** (B1, measured) → user decision D18: a
  flat versioned binary buffer rebuilt in C++; a 0c per-call benchmark.
- **The named parser could not do the job, and fields were lost** (B2) →
  the D18 buffer carries every field a C++-built graph carries (incl. the
  theta layout and epoch metadata), stores doubles bitwise, has a version
  constant, is the seed of phase 5's owner, and is gated against Python
  `serialize()` on four graph kinds.
- **Minors.**
  - m1: stale wording.
  - m2: `Graph` vs `_Graph` behaviour listed.
  - m3: the staleness key covers headers and the compiler.
  - m4: edge order preserved; the parallel-edge drop is a known limit.
  - m5: the buffer is freed by the JIT copy.
  - m6: citations.

**Review 12 (of v2.7) → v2.8** (report `review12/`):

- **Formula-mode misrouting; the gate could not catch it** (M1) → the
  wrapper sets the Python metadata from the header; weight mode is
  derived (log not representable); a field-by-field gate with an
  idempotence check; formula and was_dph graphs in the routing corpus.
- **The route does not compile today; no baseline or criterion** (M2,
  measured) → baseline with the compile fixed; bound emit + rebuild <=
  1× build; the hand-off order; the per-call cost exception pending as
  D19.
- **0c too big; no native checks** (M3) → the hand-off is split into a
  new phase 0d (M) with a local ASan, malloc-scribble and fuzz smoke; an
  unbound reader; the stop-point row.
- **Minors.**
  - 1: the phase-5 and phase-12 rows match D18.
  - 2: in-process only, never persisted.
  - 3: the lost `_callback` is listed.
  - 4: phase 3 keeps the 0d binding's exceptions.
  - 5: the `phasic::Vertex` reassignment segfault is OI-118.
  - 6: line references.
  - 7: the status line.

**Review 13 (of v2.8) → v2.9** (report `review13/`):

- **The 0d gate was self-referential** (M1) → an independent oracle: a
  test-only canonical walk of the SOURCE graph in the JIT copy before it
  is freed, compared field by field with the same walk on the rebuilt
  graph; header assertions per test model.
- **Minors.**
  - 1: the compile fix moves into 0d with the hand-off, so the pointer
    route is never re-enabled.
  - 2: an in-lookup-tree bit per vertex; `find_vertex` is tested.
  - 3: field reconstruction rules; current edge weights carried.
  - 4: the new tape getter; do not set `weight_formula = src`.
  - 5: the D19 baseline without `normalize`.
  - 6: native DPH returns `is_discrete=False` until phase 4, listed.
  - 7: 0c's size noted.
  - 8: OI-118 fixed in 0c; 0d avoids `Graph` assignment.
  - Stale "0c" → "0d" references; baselines after 0d; OI-118's home.

**Review 14 (of v2.9) → v2.10** (report `review14/`):

- **0c required `load_cpp_builder` to work, but it only compiles from 0d**
  (M1) → all `load_cpp_builder` work (source rule, cache key, secure
  directory, tests, OI-116) moves into 0d; 0c covers `pmf_from_cpp` only.
- **Minors.**
  - 1: the D19 baseline is bare `build_model` time inside the JIT copy.
  - 2: the oracle walk includes the formula tape, plus a bitwise
    forward evaluation.
  - 3: `-fvisibility=hidden` / `-Bsymbolic` for the JIT library.
  - 4: the native-DPH remedy and an optional `discrete=` keyword.
  - 5: the refusal grid and matrix agree on overlapping cells.
  - 6: the status line; the OI-118 register row.

**Review 15 (of v2.10) → v2.11** (report `review15/`):

- **Epoch metadata not set on the returned graph** (M1) → the wrapper
  sets `_epoch_state_index`, `_n_epochs`, `_base_param_length`; a C++
  epoch test model; a Python `add_epoch` equality test; an epoch graph in
  the routing corpus.
- **Minors.**
  - m1: the bitwise check compares edge weights with matched
    `-ffp-contract`, not an MPFR-sensitive read-out.
  - m2: D19 must be decided before the 0d sub-plan; the alternative is
    named.
  - m3: the D15 JIT scope.
  - m4: the moot routing clause is removed.
  - m5: register rows.
  - m6: `PHASIC_SOURCE_DIR` is developer-only, plus a source hash in the
    header.
