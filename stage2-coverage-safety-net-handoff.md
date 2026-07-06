# Stage 2 — Test Coverage & Refactor Safety Net: Hand-off Note

**Branch:** `stage2-coverage-safety-net` (off `stage1-orphan-removal`; builds on the corpse-free Stage-1 tree).
**Input to Stage 3** (`refactoring-review-prompt.md`): this delivers WS-A (the safety net) — the native
CTest regression net, the bit-identity equivalence gates for the §B duplication pairs, and the coverage &
blast-radius risk report. Stage 3 **consumes** these; it does not rebuild them.

---

## 0. What was delivered

| Deliverable | Status |
|---|---|
| Python coverage measurement (`--cov=src/phasic`) | ✅ measured — see §2 |
| Native C/C++ coverage (gcov via CMake) | ✅ measured — **17.7% line** (see §3) |
| `tests/cpp` wired into CTest + CI | ✅ green (2/2), CI workflow added |
| 7 bit-identity equivalence gates | ✅ all green, backend-asserted, xfails documented (§4–§5) |
| Coverage & risk report, ranked by refactor blast radius | ✅ §6 |
| Discovered issues (captured as xfail/disabled, NOT fixed) | ✅ §7 |

**Constraint honored:** this stage added **tests + build wiring only**. No production logic changed. Every
discovered bug is captured as an xfail (Python) or a documented disabled-assert (native), each pointing at
the Stage-3 question that owns the fix.

---

## 1. Baseline (post-Stage-1, this tree)

- **`tests/pytest` (via `pixi run -- pytest tests/pytest`)**: recorded during the coverage run — see §2 for
  the exact count. The Stage-1 hand-off recorded the notebook-inclusive `pixi run test` baseline as
  **1657 passed, 2 failed, 61 skipped, 12 xfailed**; the known-flaky set is `test_cpp_from_callback` ×5
  (`ptd_err` linkage), `test_model_selection`, `test_svgd_exposure`, `test_daisy_chain` perf-smoke, plus the
  stochastic SVGD/MCMC group (`test_mcmc_accuracy`, `test_nan_observations_correctness`) that flips run-to-run.
- **The 7 new gates add 51 passed / 1 skipped / 13 xfailed** (via `pytest -m equivalence`), all deterministic.
  They are **additions** — no existing test turns red.

Two stray uncommitted cosmetic edits exist in the working tree from before this stage
(`api/cpp/phasiccpp.h` removing `[[maybe_unused]]`, `src/cpp/phasiccpp.cpp` reformatting `wf_*_opcode`).
They are **not** part of Stage 2 and were deliberately left out of the Stage-2 commits.

**⚠ NFS cache-concurrency artifact (important for anyone running the suite here).** `~/.phasic_cache`
lives on an **NFS** filesystem. Several tests (`test_scc_compose_telemetry.py`,
`test_weight_formula_theta_dim.py::test_graph_cache_preserves_theta_dim`) `shutil.rmtree` that cache dir;
when **another process holds a cache file open**, NFS silly-rename leaves `.nfsXXXX` files and the rmtree
raises `OSError: [Errno 16] Device or resource busy`. This fires **only when two pytest processes run
concurrently** (e.g. a coverage run in parallel with a gate run). It is an **environment/harness artifact,
not a regression and not caused by Stage 2** — run the suite single-process (one pytest at a time) and these
pass. This is why an intermediate parallel run showed 8 failures vs the documented ~3-flaky baseline.

**Coverage-tooling defect found:** the repo's blessed coverage command `--cov=src/phasic` measures **0%** on
this **non-editable** install (the running code is in `.pixi/.../site-packages/phasic`, and the `src/phasic`
directory filter drops it; the `.coveragerc` `[paths]` remap only fires at report time, too late for the
source filter). **Use `--cov=phasic`** (the importable package name) — it resolves to the installed location
and the `[paths]` remap then attributes lines back to `src/phasic`. The `/coverage` skill / coverage-analyst
command should be updated accordingly; §2 was measured with `--cov=phasic`.

---

## 2. Python coverage (measured on the corpse-free tree)

Measured with `pytest tests/pytest --cov=phasic` (the gates are additive; they exercise existing
production code). **Overall `src/phasic`: 57.8% (8855/14840 statements).** After-run: **1702 passed, 8 failed,
62 skipped, 25 xfailed** — the 8 failures are exactly the 3 known-flaky + 5 NFS-cache-concurrency artifacts
(§1); **all 7 gates pass in full-suite order**, and no existing test turns red.

Refactor-relevant modules, thinnest first (⚠ = a Stage-3 refactor target that is thinly covered):

| Module | Coverage | Missed | Stage-3 relevance |
|---|---|---|---|
| `plot.py`, `srun_magic.py`, `parallel_utils.py`, `cloud_cache.py` | **0%** | 489 | UX/SLURM/infra — low refactor churn |
| `trace_cache.py` | **11%** | 62 | mostly Stage-1-dead cache helpers |
| `bffg.py` | **18%** | 198 | not a primary target |
| `hierarchical_trace_cache.py` | **30%** ⚠ | 589 | **WS-E trace SSOT (Q1)** reaches deep here |
| `svgd.py` | **45%** ⚠ | 1639 | **WS-C god-object split** (9611 lines) |
| `compute_repository.py` | 54% | 153 | IPFS registry (#14-adjacent) |
| `graph_cache.py` | 57% | 65 | cache API |
| `state_indexing.py` | 63% | 190 | `IndexCodec` extraction (Q12) |
| `mcmc.py` | 71% | 129 | shares shape with SVGD (Q12) |
| `__init__.py` (incl. `class Graph`) | **74%** | 859 | **WS-C god-object split** (12343 lines) |
| `trace_elimination.py` | 77% | 119 | WS-E trace SSOT (#1) — G5 gates the numerics |
| `distributed_scc.py` | 79% | 38 | SCC composition (#9) — but native composer is 0% |
| `ffi_wrappers.py` | 79% | 52 | FFI orchestration (#7) — G1 gates the path |
| `config.py` | 79% | 62 | cross-boundary contracts (WS-G) |
| `method_of_moments.py` | 83% | 34 | — |
| `svgd_config.py` | 88% | 41 | `Graph.svgd` → `SvgdConfig` pipeline (WS-C) |
| `weight_formula.py` | 88% | 30 | #3 — G4 gates it |

The Python suite exercises `__init__.py`/Graph and the parameterized subsystem heavily (via pybind), which
is why their transitive C coverage is far higher than the native-only 17.7% in §3.

---

## 3. Native C/C++ coverage (new — there was none before)

Independent native coverage now exists via gcov. Built standalone (the scikit-build/pip build dir is
ephemeral, so CTest can't run from the install):

```
pixi run coverage-cpp          # cmake -DPHASIC_BUILD_TESTS=ON -DPHASIC_COVERAGE=ON + ctest + gcovr
```

Coverage from the two CTest binaries (`test_c`, `test_cpp`) — **17.7% line, 9.9% branch (1487/8380 lines)**:

| File | Lines | Covered | % |
|---|---|---|---|
| `src/c/phasic.c` | 6073 | 1459 | **24%** |
| `src/cpp/phasiccpp.cpp` | 1135 | 25 | **2%** |
| `src/c/phasic_log.c` | 28 | 3 | 10% |
| `api/cpp/scc_graph.cpp` | 156 | 0 | **0%** |
| `src/c/phasic_hash.c` | 161 | 0 | **0%** |
| `src/c/scc_compose.c` | 254 | 0 | **0%** |
| `src/c/scc_synthetic.c` | 573 | 0 | **0%** |

**Native coverage gap (important for Stage-3 WS-E/F):** the SCC machinery (`scc_compose.c`,
`scc_synthetic.c`, `scc_graph.cpp`), the graph content hasher (`phasic_hash.c`), and the whole C++ facade
(`phasiccpp.cpp` — serialize/discretize/profile/weight-formula-compiler/moment "faithful ports") are
**essentially uncovered by the native tests**. Their behavior is exercised only transitively through the
Python suite (via pybind). Any Stage-3 native refactor of those units is protected only by the Python
equivalence gates, not by an independent native test — factor this into sequencing.

> The native tests are a **thin slice** today: `tester.cpp::main()` runs ~13 of its functions and
> `testcpp.cpp::main()` runs only `test_pmf()`. Expanding `main()` coverage is deferred (many of the
> currently-uncalled functions carry stale pre-migration expectations — see §7).

---

## 4. Equivalence-gate inventory (the WS-A deliverable)

All gates live in `tests/pytest/test_gate_*.py`, are tagged `@pytest.mark.equivalence` (registered in
`pyproject.toml`), and share `tests/pytest/_gate_backend.py`. **Every FFI gate proves the backend it
exercised** — it traces the jitted call with `jax.make_jaxpr` and asserts an `ffi_call` primitive with
`target_name == "ptd_compute_*"` and no `pure_callback` (so an FFI path that silently degraded to the host
callback fails loudly, not vacuously). FFI is compiled in this build
(`get_compute_pmf_ffi_capsule` present).

| Gate | File | Pins (A vs B) | Stage-3 # | Relationship | Backend proof |
|---|---|---|---|---|---|
| **G1** | `test_gate_ffi_vs_pybind.py` | `compute_*_ffi` (FFI capsule) vs `GraphBuilder.compute_*` (pybind) | #7 | `==` (rewards=None) | jaxpr → `@ptd_compute_*` |
| **G2** | `test_gate_moments_3way.py` | `_moments`→`Graph.moments` vs `GraphBuilder.compute_moments` (+ FFI core) | #5 | `==` (diff 0.0) | jaxpr → `ptd_compute_moments` |
| **G3** | `test_gate_serialize_roundtrip.py` | `from_serialized` (Python) vs `GraphBuilder::build` (C++) | #4 | `==` (match graph) | class-module identity |
| **G4** | `test_gate_weight_formula_conformance.py` | `compile_formula`→C-VM vs `eval_tape` oracle, full opcode corpus | #3 | `==` (0 ULP) | `weight_mode=='formula'` |
| **G5** | `test_gate_elimination_bit_identity.py` | Python `EliminationTrace` replay vs C `GraphBuilder` engine | #1 | rtol 1e-6/1e-9 | `_has_param_compute_graph_cache` |
| **G6** | `test_gate_conditioned_samplers.py` | `ptd_..._conditioned` (pybind) vs `..._conditioned_fixed` (FFI) | #11 | distributional (rand≠rand_r) | output form + capsule |
| **G7** | `test_gate_persistent_graph_reuse.py` | persistent thread-local graph reuse vs fresh-build; FFI vs pybind | #7 / Q7 | `==` | jaxpr → `ptd_compute_pmf_and_moments` |

Notes per gate:
- **G1** also has the reward-transform PMF divergence as a strict xfail (Q7.1, see §5).
- **G2** covers impl (a) and (c); impl (b) `phasic::Graph::moments` (`api/cpp/phasiccpp.h:510`, the
  "faithful ports") is **native-only / unbound from Python** → covered only by a future `tests/cpp` test
  (skipped placeholder in the file; recorded as a native-coverage gap).
- **G4** is the reachable 2-of-3 legs. The C++ `wf_*` compiler (`phasiccpp.cpp:720`, "Opcodes MUST match
  weight_formula.OPCODES") is **unreachable from Python** (anonymous-namespace, its only caller
  `phasic::Graph::weight_formula(const std::string&)` is unbound, and `GraphBuilder` accepts only a
  precompiled `weight_formula_tape`, never a source string). The `tape_py == tape_cpp` leg (**G4b**) can be
  closed only by a new `tests/cpp` test — and that additionally needs a C++ tape reader that does not exist
  today. Recorded as an open Stage-3 question (see §7).
- **G5** cyclic + formula graphs are strict xfail (Q1, see §5).
- **G6** is distributional, not per-draw (`rand()` vs `rand_r()` are different glibc generators); it uses a
  trash-escape graph so intermediates have backward-prob < 1 (no premature stop). N=20000, deterministic on
  glibc.

---

## 5. Known-divergence xfails (each → the Stage-3 question that resolves it)

These are **strict** xfails (`xfail(strict=True)`): the day Stage-3 unifies the pair, the assertion starts
passing, the XPASS fails the suite, and the marker must be removed. That is the forcing function.

| xfail test | Divergence | Stage-3 Q |
|---|---|---|
| `test_g1_pmf_and_moments_rewards_pmf_divergence` | FFI computes combined PMF on the **untransformed** graph (`graph_builder_ffi.cpp:492`); pybind uses the reward-transformed graph (`graph_builder.cpp:741`) → PMFs differ | **Q7.1** (new; FFI orchestration, #7) |
| `test_g2_moments_discrete_unbound_xfail` | `Graph.moments(discrete=True)` → `super().moments_discrete` which has **no pybind binding** → `AttributeError` (needs a `.discretize()`d graph to reach it) | **Q5** |
| `test_g3_constant_edges_divergence` | `from_serialized` **drops** `constant_edges`; `GraphBuilder::build` keeps them | **Q6a** |
| `test_g3_vertex_indices_divergence` | `from_serialized` merges duplicate-state vertices; `GraphBuilder` ignores `vertex_indices` and preserves identity | **Q6b** |
| `test_select_dead_arm_oracle_divergence` | `eval_tape` eagerly evaluates untaken `select()` arms (raises); the C residual VM prunes dead arms | **Q-G4-1** (new; weight-formula, #3) |
| `test_pow_negative_base_fractional_divergence` | Python `float.__pow__` → complex for neg base + fractional exp; C `pow` → NaN → reject | **Q-G4-2** (new; weight-formula, #3) |
| `test_g5_*[cyclic_back_edge, cyclic_triangle]` | Python trace refuses self-loops from cycle elimination; C engine handles via 1/(1−q) | **Q1** |
| `test_g5_*[formula_coalescent]` | Python trace refuses `weight_mode='formula'`; C engine evaluates the tape | **Q1** |
| `test_g6_per_draw_seed_identity_xfail` | Impl A `rand()` (no seed arg) vs Impl B `rand_r()` (explicit seed) → per-draw identity impossible | **Q11a** (new; sampler RNG, #11) |

`G7` has **no** xfail today — every persistent-reuse / FFI-vs-pybind comparison is bit-identical on this
build. A ready-to-flip xfail template is left commented in the DPH gate for Stage-3 to activate only if a
real divergence appears (DPH re-normalization is the highest-risk area).

---

## 6. Coverage & risk report, ranked by refactor blast radius

Ranked by **blast radius = (coverage thinness) × (refactor churn) × (gate weakness)**, not raw %. A
thinly-covered god object that WS-C will slice outranks a thinly-covered leaf. "Gate" = the Stage-2
equivalence gate(s) that pin the unit's numerics through the change.

### 🔴 HIGHEST — thin coverage, high churn, no/weak dedicated gate

1. **`svgd.py` / `SVGD` (45% Python, 9611 lines) — WS-C decomposition.** The second-largest god object,
   least-covered of the big units, and **none of the 7 new gates cover it** — its only bit-identity net is
   the *pre-existing* `test_log_prob_unified_bit_identity.py` (pins `_log_prob_unified`) and
   `test_svgd_api_parity.py` (loose, JAX-cache-sensitive). WS-C's `SVGD.__init__` decomposition + the
   `svgd.py` module split move ~5k lines with 1639 uncovered statements underneath. **Recommendation:**
   before WS-C touches `svgd.py`, add bit-identity gates on the extraction seams (constructor →
   `SvgdConfig`, the kernel/optimizer/schedule/preconditioner modules). `svgd_config.py` (88%) is a good
   anchor; the SVGD internals are not.

2. **SCC composition seam (#9) — `distributed_scc.py` 79% Python **but** `scc_compose.c` 0% + `scc_synthetic.c`
   0% native, and NO equivalence gate.** The C composer (`ptd_compose_scc_prcs`, OpenMP) and the Python
   `stitch_scc_traces` "disagree on ordering" (Kahn vs stored Tarjan) per the Stage-3 map — exactly the kind
   of divergence a gate exists to catch, and there is **none**. The native side is entirely uncovered by
   `tests/cpp`. **Recommendation:** WS-E/F must add a topo-order + composition equivalence gate before
   unifying the composer (#9 / Q10); this is the biggest *un-gated* duplication.

3. **`hierarchical_trace_cache.py` (30% Python, 589 uncovered) — WS-E trace SSOT (Q1).** The trace-SSOT
   decision "reaches deep into `hierarchical_trace_cache.py`" (Stage-3 Q1). G5 pins the *elimination
   numerics* (Python `EliminationTrace` vs C engine), but **not** the hierarchical SCC caching/dedup/replay
   logic that Q1 restructures, and that logic is only 30% covered. **Recommendation:** extend G5-style
   gates to the cached/stitched replay path before Q1 lands.

### 🟠 HIGH — big/high-churn, but methods are gated

4. **`class Graph(_Graph)` in `__init__.py` (74% Python, 12343 lines) — WS-C mixin split.** Pure relocations
   (verbatim moves), and the *methods* are pinned by G1 (pmf/moments), G2 (moments), G3 (serialize), G5
   (elimination). Risk is the 859 uncovered statements (validation/dispatch/plotting branches) that no gate
   touches — but WS-C moves them verbatim, so a compile+suite-green check largely suffices.

5. **`PYBIND11_MODULE` in `phasic_pybind.cpp` (5880 lines) — WS-D split + de-dup.** **Not measured by the
   native gcov** (§3 covers `libphasic`/`libphasiccpp`, not the extension) — it is exercised only by the
   Python suite. The moment family (#5) and `as_matrices` (#10) are gated by **G2/G3**; the 4 BFS builders
   and the extracted JIT toolchain are **ungated**. **Recommendation:** the moment-rebind and matrix-unify
   commits are safe behind G2/G3; add a small gate for the BFS builders if WS-D unifies them.

### 🟡 MEDIUM — well-gated by Stage-2 (proceed behind the green gate)

6. **Parameterized engine + FFI orchestration (#1, #7)** — `graph_builder*.cpp`, `graph_builder_ffi.cpp`.
   Gated by **G1** (FFI-vs-pybind bit-identity, incl. the Q7.1 reward-transform xfail), **G5** (elimination),
   **G7** (persistent-graph reuse). Native gcov does not see these (extension-only), but the Python gates
   pin them tightly. `phasic.c` (24% native, ~74%+ transitive) file-split is a **pure move** (header
   unchanged) — lowest behavior risk; kernels pinned by G1/G4/G5/G6.
7. **`serialize()` (#4) → G3** (with the two divergence xfails). **weight-formula (#3) → G4** (+ the G4b
   C++-compiler gap, §7). **moments (#5) → G2**. **conditioned samplers (#11) → G6**.

### 🟢 LOWER — ungated but low churn / late in the plan

8. `as_matrices`/`from_matrices` (#10, ungated — keep the dict shape identical), `discretize` (#8),
   `profile` (#13), the `_mpfr` variant (#8/Q8). These are med/low priority in the Stage-3 sequence; add a
   thin gate only when the specific unification commit is scheduled.

**One-line takeaway for Stage-3 sequencing:** the safety net is strongest exactly where the *native
numeric* duplications are (G1–G7 cover #1/#3/#4/#5/#7/#11). The two blind spots are **`svgd.py` (WS-C,
Python-only, no gate)** and the **SCC composer (#9, native 0%, no gate)** — build a gate for each before its
workstream begins.

---

## 7. Discovered issues (captured, NOT fixed — Stage-3 owns them)

1. **`defect()` / `cdf()` inconsistency (found by wiring the native tests).** For a graph whose IPV puts
   mass directly on an absorbing vertex, `graph.defect()` (and `ptd_defect` after a reward transform)
   returns **0.0** while `cdf()` reflects the instant-absorption mass — internally inconsistent. The two
   native asserts that check this are **disabled with a documented note** (`testcpp.cpp:619`,
   `tester.cpp` `test_rabbit`), the native analog of an xfail. This is a real production-behavior question
   for Stage-3 (defect semantics), not a test-wiring fix.
2. **G4b — the C++ `wf_*` weight-formula compiler is unreachable and untested from anywhere.** No Python
   binding, and `GraphBuilder` never takes a formula source string. A `tape_py == tape_cpp` gate needs a
   new `tests/cpp` test **and** a C++ tape reader accessor (`struct ptd_weight_tape` is opaque). This is a
   coverage gap that a Stage-3 decision (delete the C++ compiler, bind it, or route `GraphBuilder` through a
   source string) will settle. It strengthens the case for deleting the duplicate compiler (#3).
3. **G2 impl (b) native-only gap.** `phasic::Graph::moments`/`expectation`/`variance`/`covariance` in
   `api/cpp/phasiccpp.h` are "faithful ports" bound to nothing; they can only be covered by a `tests/cpp`
   test. They are dead from Python's view — reinforcing the #5 unification (make `phasic::Graph` the SSOT
   and bind member pointers).
4. **A 4th divergent moment surface exists (out of G2 scope).** `Graph.moments_from_graph(use_ffi=False)`
   generates C++ whose higher-moment recurrence is `rewards3[j] = rewards2[j]*pow(rewards2[j], i)` — a
   different (buggy-looking) recurrence than the `_moments`/`compute_moments_impl` family. Flagged for a
   separate gate so nobody "fixes" G2 to match it.

---

## 8. Native CTest / CI wiring — what it took

- **CMakeLists.txt**: replaced the dead test stub (`:408–417`) with `option(PHASIC_BUILD_TESTS OFF)` +
  `option(PHASIC_COVERAGE OFF)`, `enable_testing()`, `add_executable`/`add_test` for `test_c`
  (→ `libphasic`) and `test_cpp` (→ `libphasiccpp`), and `--coverage -O0 -g` on the two libs under
  `PHASIC_COVERAGE`. Both options default **OFF** so the normal scikit-build / conda pip build is unaffected.
- **The native tests did not compile on the current tree** — but **not** because of Stage 1. It was the
  older `ptd_graph_add_edge` API migration (old scalar 3-arg → current coefficient-vector 4-arg). Fixes,
  all **test-only**:
  - `tester.cpp`: a 3-line macro-shim bridges the ~220 old 3-arg call sites to the 4-arg API (a
    single-coefficient edge gets `weight = coeff*1.0` immediately, `phasic.c:4898` — exactly the old scalar
    semantics); `test_reward_parameterized2` (uses the undeclared `ptd_graph_add_edge_parameterized`) is
    `#if 0`-d.
  - `testcpp.cpp`: two raw-C functions (`test_expected_entry_visits`, `test_2pmigTIME`) `#if 0`-d;
    `main()` only runs `test_pmf` (C++ wrappers).
  - The stale relative include `../api/...` (from when the files lived in `tests/`) is resolved by adding
    `tests` to the include path in CMake — no source edit.
- **pixi tasks**: `test-cpp` (build + ctest) and `coverage-cpp` (instrumented + gcovr). `gcovr` added to
  `[tool.pixi.dependencies]` (`lcov`/`gcovr` were missing; `gcov` present).
- **CI**: `.github/workflows/native-tests.yml` (there was **no** test CI before) — job 1 runs CTest +
  gcovr, job 2 runs `pytest -m equivalence` after a real `install-dev` (the FFI-compiled build).

---

## 9. How to run everything

```
pixi run test-cpp                       # native C/C++ CTest (2/2 green)
pixi run coverage-cpp                   # native gcov coverage report -> coverage-native.txt
pixi run -- pytest tests/pytest -m equivalence -v   # the 7 equivalence gates
pixi run -- pytest --cov=src/phasic --cov-report=term-missing tests/pytest   # Python coverage
```
