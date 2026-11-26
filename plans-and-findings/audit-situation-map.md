# SITUATION MAP — Differential Audit of `phasic` Post-Refactor

## 1. BASELINE

**Diff target: `3082ebc6ca960225b5dbee1f65f41e243818156d`** — "Fixed failing tests" (2026-07-04 15:52).

- It is the **direct parent of the first refactoring commit**: `git rev-parse 9b8bc3f6^ == 3082ebc6` (verified), i.e. the last commit before any `refactor:`/`Stage-`/`WS-` work.
- It sits **after** the entire H1–H10 / C1–C3 hardening batch (`05be98af..cd634478`) and after `ec4b80df` "Updated/redid CLAUDE.md" — a green, post-fix tree.
- Crucially it **precedes `c340bedc`** (the sojourn adjoint compute change), so a differential diff against it will surface that behaviour change too.

**Build outlook (from git alone):** expected to build with the current pixi env. The `ptd_graph_add_edge` 3-arg→4-arg API migration that broke the *native* tests is described in `054f6fff` as "older … NOT Stage 1", i.e. it predates this baseline — production source at `3082ebc6` already uses the current 4-arg coefficient-vector API. The only thing that would not compile at this commit is `tests/cpp` (CTest wiring + macro-shim were not added until `054f6fff`), which does not affect the scikit-build/pixi package build. **UNVERIFIED:** no build was actually run (read-only recon).

**Refactoring span (all on master):**
- **First:** `9b8bc3f6` "remove commented-out corpse blocks in phasic.c" (2026-07-05 02:45).
- **Last:** `9de5d61a` "Merge Stage-3 refactoring execution into master" (2026-07-09 09:33); last content commit on the merged branch is `98deb4da` (WS-C serialize extraction), with `62926710` a docs handoff.
- The run of `refactor:`/`test:Stage-2`/`WS-` commits is contiguous walking back from the Stage-3 merge and **stops at `9b8bc3f6`**, whose parent `3082ebc6` is the first non-refactor commit.
- Stage-3 work is on a **merge branch**; everything else in the span is linear on master.

**Not part of the refactoring:** two post-merge commits `57f2f74e` ("added deferred fix file"), `0754c66f` ("Added coverage config"); and the **current branch's** commits above master — `git log master..HEAD` = `d69919f2`, `12a30a78`, `cdda320f` — an unrelated numerical/FD audit layer (see §5).

---

## 2. COMPUTE-PATH ATLAS

Exactness key: **approx-U** = ends in `g.pdf(t,granularity)` discretization/uniformization forward (converges as granularity→∞; api/cpp/phasiccpp.h:1285-1332, api/c/phasic.h:1379) — differential must use a **granularity-dependent tolerance**. **exact** = discrete `g.dph_pmf`, moments via `expected_waiting_time`, `expected_sojourn_time`, `backward_probabilities` (exact elimination-trace quantities). **unknown** = raises / no route.

| Entry point (def) | Flag combo | Backend | Exactness | Evidence |
|---|---|---|---|---|
| **pmf_from_graph** (`__init__.py:3399`) | parameterized & `callback` & continuous | pybind `GraphBuilder.compute_pmf` via `pure_callback` (rebuild per θ) | approx-U | `:3510`, `:3532`; phasiccpp.h:1292-1332 |
| | parameterized & `callback` & discrete | pybind `GraphBuilder.compute_pmf`→`g.dph_pmf` | exact | `:3532`; graph_builder_ffi.cpp:263 |
| | parameterized & {linear,log,formula} & `_use_ffi=True` & continuous | XLA-FFI `ptd_compute_pmf` = `ComputePmfFfiImpl`→`g.pdf` | approx-U | `:3589`,`:3610`; ffi_wrappers.py:589; graph_builder_ffi.cpp:267 |
| | parameterized & {linear,log,formula} & `_use_ffi=True` & discrete | XLA-FFI `ptd_compute_pmf`→`g.dph_pmf(jump)` | exact | graph_builder_ffi.cpp:263; `:3610` |
| | parameterized & `_use_ffi=False` | **NO ROUTE** — raises `PTDBackendError` | **unknown** | `:3591-3603` |
| | non-parameterized, any mode | ctypes-JIT `pmf_from_cpp` on `_generate_cpp_from_graph` (dummy θ=[0.0]) | approx-U (see pmf_from_cpp) | `:3704-3714` |
| **pmf_from_graph_parameterized** (`:3720`) | continuous | ctypes-JIT `compute_pmf_from_arrays` (`g.pdf`) via `pure_callback`, rebuild per θ | approx-U | `:3874`,`:3802`,`:3877` |
| | discrete | ctypes-JIT `compute_dph_pmf_from_arrays`→**`g.normalize()` THEN `g.dph_pmf`** | exact | `:3874`,`:3845`,`:3849` |
| **pmf_from_cpp** (`:3880`) | continuous | ctypes-JIT `lib.compute_pmf`→`g.pdf`, `pure_callback` | approx-U | `:3971`,`:507`,`:526` |
| | discrete | ctypes-JIT `lib.compute_dph_pmf`→`g.dph_pmf` (**NO normalize, by design**) | exact | `:3990`,`:3979-3988` |
| **moments_from_graph** (`:6283`) | `weight_mode != linear` | **NO ROUTE** — raises `ValueError` (generated build is hardcoded linear; would silently linearize) | **unknown** | `:6371-6380` |
| | `linear` (any `use_ffi`) | ctypes-JIT `compute_moments`→repeated `expected_waiting_time` via `pure_callback`; **`use_ffi` selects no FFI** | exact | `:6432`,`:6444`,`:6475`,`:6342` |
| **pmf_and_moments_from_graph** (`:6511`) | `callback` (forces `use_ffi=False`) | pybind `compute_pmf_and_moments` via `pure_callback` | approx-U (moments exact) | `:6612`,`:6629`,`:6764` |
| | not-callback & `use_ffi=True` & `_use_ffi=True` | XLA-FFI `ptd_compute_pmf_and_moments` = `ComputePmfAndMomentsFfiImpl` | approx-U (discrete+moments exact) | `:6779`,`:6788`; ffi_wrappers.py:765 |
| | not-callback & `use_ffi=False` (**METHOD DEFAULT; used by svgd**) | pybind `compute_pmf_and_moments` via `pure_callback` (cached builder) | approx-U (discrete+moments exact) | `:6806-6823`,`:6846` |
| **pmf_and_moments_from_graph_multivariate** (`:7524`) | any (rewards 1D/2D); loops features in Python | delegates to 1D `pmf_and_moments_from_graph` (same 3 sub-routes); **`ptd_compute_pmf_multivariate` FFI NOT used** | approx-U | `:7616`,`:7689`; ffi_wrappers.py:750-752 |
| **pmf_from_graph_joint_index** (`:7100`) | `log` | **NO ROUTE** — raises `ValueError` (sojourn FFI hardcodes linear) | **unknown** | `:7193-7202` |
| | `callback` | pybind `GraphBuilder.build` + `expected_sojourn_time` on subsets, normalized | exact | `:7262`,`:7288`,`:7292` |
| | {linear,formula} & `observed_indices` supplied (baked; SVGD non-exposure) | XLA-FFI `ptd_compute_sojourn_times` = `ComputeSojournTimesFfiImpl` in `custom_vmap` | exact | `:7362`,`:7373`,`:7393`; ffi_wrappers.py:1051 |
| | {linear,formula} & `observed_indices=None` (legacy) | XLA-FFI `ptd_compute_sojourn_times` (two calls, normalized); handler hardcodes `use_log=false` | exact | `:7431-7447`; graph_builder_ffi.cpp:1014-1018,1084-1088 |
| **reward_visit_probability** (`:3179`) | no vertex reward>0 | constant `0.0` short-circuit | exact | `:3232-3241` |
| | θ is jax.Array/Tracer | `compute_reward_visit_probability_ffi`→C `ptd_backward_probabilities` (`BackwardProbabilitiesFfiImpl`) via `pure_callback` + custom_vjp FD | exact | `:3247-3258`; ffi_wrappers.py:1150,1175; graph_builder_ffi.cpp:1127 |
| | θ None/concrete numpy | pybind `Graph.backward_probabilities` (elimination solve) · IPV | exact | `:3265-3270` |
| **daisy_chain_joint_probs** (`:9325`) | graph not from `joint_stop_prob_graph` | **NO ROUTE** — raises `ValueError` | **unknown** | `:9404-9408` |
| | `log` | **NO ROUTE** — raises `ValueError` (daisy FFI hardcodes linear) | **unknown** | `:9414-9421` |
| | `final_read='sojourn'` (**DEFAULT**) & {linear,formula,callback} | XLA-FFI `ptd_daisy_chain_sojourn` = `DaisyChainSojournFfiImpl` (per-epoch `stop_probability` walk + **exact** elimination sojourn final read) | approx-U (**intermediate epochs still `stop_probability(dt)` uniformization**) | `:9554-9560`; ffi_wrappers.py:1468; graph_builder_ffi.cpp:1782,1827 |
| | `final_read='stopprob'` | XLA-FFI `ptd_daisy_chain_joint_probs` = `DaisyChainJointProbsFfiImpl` (`stop_probability(t_eval)` forward, granularity-bound) | approx-U | `:9561-9565`; ffi_wrappers.py:1412; graph_builder_ffi.cpp:1460,1528 |
| **svgd** dispatch (`:4918`) | joint-prob graph & `epoch_starts=None` | → `pmf_from_graph_joint_index` (`joint_index` & `discrete` FORCED True) | exact | `:5782-5786`,`:5630`,`:5679` |
| | joint-prob graph & `epoch_starts!=None` (daisy) | → `_daisy_chain_svgd_model`→`daisy_chain_joint_probs` (`final_read` passthrough, default sojourn) | approx-U | `:5706`,`:5734-5747` |
| | non-joint & rewards 2D | → `pmf_and_moments_from_graph_multivariate(use_ffi=False)` → pybind per-feature | approx-U | `:5791-5797` |
| | non-joint & rewards 1D | → `pmf_and_moments_from_graph` (use_ffi default False) → pybind `pure_callback` | approx-U | `:5798-5803` |
| | non-joint & rewards None | → `pmf_and_moments_from_graph` (use_ffi default False) → pybind `pure_callback` | approx-U | `:5804-5809` |

**Routes flagged "unknown" (no compute; raise/dead):** `pmf_from_graph` parameterized+`_use_ffi=False` (`:3591`); `moments_from_graph` non-linear (`:6371`); `pmf_from_graph_joint_index` log (`:7193`); `daisy_chain_joint_probs` wrong-graph (`:9404`) and log (`:9414`).

**Silent overrides (behaviour not selectable as documented):**
- `pmf_from_graph`: `use_cache` (default True) accepted/documented (`:3422`) but **never read** — symbolic cache removed (`:3483-3485`). FFI path **hardcodes granularity=0** (`:3614`).
- `svgd`: joint-prob → `joint_index` forced True (`:5630`); user `discrete=False` forced True for joint_index models (`:5668-5679`).
- `pmf_and_moments_from_graph`: `use_ffi=True` silently downgraded to False when `_use_ffi=False` (`:6776-6777`).
- `moments_from_graph`: `use_ffi` accepted but selects no FFI (`:6342`,`:6440-6444`).
- `ptd_compute_pmf_multivariate` handler registered (ffi_wrappers.py:241) but **never invoked** — dead route (`:750-752`, `:7616`,`:7689`).
- C sojourn/daisy FFI handlers hardcode `use_log=false` (graph_builder_ffi.cpp:1014-1018,1084-1088,1528,1782,1827): `log` would be silently linearized — caught by Python raises for `log` only; **`formula` passes through** honored via C weight-tape VM (claim read, not test-verified — see §8).

---

## 3. WHAT IS ALREADY GUARDED

12 gate files (11 pytest `test_gate_*` + `_gate_backend.py`, plus native CTest). Of these, **5 do real cross-path comparison**; G6/G7 are partly self-vs-self; 3 are single-path golden/topology pins.

- **`_gate_backend.py`** — infra only (not collected). Proves *which* backend a fn lowered to via `jax.make_jaxpr` walk: FFI custom-call vs `pure_callback` vs pybind-direct (`:76-137`). Asserts `jax_enable_x64` True (`:33`); `FFI_TARGETS` includes pmf/moments/pmf_and_moments/pmf_multivariate/sample_path_conditioned (`:39`). Provides `coalescent_graph(n=4)` 1-param continuous (`:142`), `dph_graph(n=3,p=0.2)` discrete (`:153`). Asserts nothing numeric.

- **G1 `test_gate_ffi_vs_pybind.py`** — **FFI XLA custom-call vs pybind-direct** `GraphBuilder.compute_*` for PMF (cont+disc), moments E[T^k] (nr=3), combined; + discrete-PMF vs closed-form NegBinomial(2,p). Tol: **bit-identity** `assert_array_equal` for rewards=None (`:71,87,100,115`); reward-moments `rtol=1e-12` (`:152`); discrete-vs-closed-form `rtol=1e-12` (`:199`). Coverage: `coalescent_graph(4)` 1-param, `dph_graph(3,0.2)`, `_two_phase_dph`; scalar θ; granularity=100 both sides; **linear only**, theta_dim=1, no vmap.

- **G2 `test_gate_moments_3way.py`** — moments E[T^k], nr=4. **pybind free `_moments` (via `Graph.moments`) vs `GraphBuilder.compute_moments`** (bit-identity `:67`), plus FFI `ptd_compute_moments` == compute_moments (`:81`). **Impl (b) native `phasic::Graph::moments` (phasiccpp.h:510) explicitly SKIPPED** (`:99`) — no cpp test calls it. Single 2-param Erlang, θ=[2.0,3.0]. linear, continuous. discrete=True is xfail(strict) (`:88`).

- **G5 `test_gate_elimination_bit_identity.py`** — the **only** gate touching JAX trace-replay: **Python `EliminationTrace` record/replay vs C `GraphBuilder.compute_pmf/compute_moments`**. Tol NOT bit-identity: pdf `rtol=1e-6 atol=1e-12` (`:178`), mean/var `rtol=1e-9` (`:193-195`). Coverage: acyclic coal_n3/erlang3/branching; cyclic back_edge/triangle (**xfail**); formula `c0*t0` (**xfail**). granularity=100. **Never compares trace to FFI directly.**

- **G3 `test_gate_serialize_roundtrip.py`** — two **parsers**, shared native pdf kernel: Python `Graph.from_serialized`+native pdf vs C++ `GraphBuilder(json).compute_pmf`. Bit-identity (`:105`). 3 tiny graphs: `_g_match`, `_g_const` (constant_edges, **xfail**), `_g_dup` (duplicate-state, **xfail**). θ=[2,3], PDF only.

- **G4 `test_gate_weight_formula_conformance.py`** — **weight level:** Python `compile_formula`→C tape VM vs Python `eval_tape` oracle (bit-identity `:74`); **PMF level:** GraphBuilder formula-tape vs linear reference (`atol=1e-12` `:140`). 18-formula corpus; θ=[0.6,1.1], coeffs=[2.0,0.5]. **C++ `wf_*` compiler UNREACHABLE from Python** (documented gap `:9-15`). dead-select-arm & pow-negative-base **xfail(strict)**.

- **G6 `test_gate_conditioned_samplers.py`** — **distributional only**: pybind `sample_path_conditioned` (C `rand()`, phasic.c:10619) vs FFI `sample_path_conditioned_ffi` (C `rand_r()`, phasic.c:10708). Tol `<0.03` distributional (`:123-124`), mean within 6·SEM (`:128`). Single 6-vertex linear graph, θ=[1.0], N=20000. Per-draw seed-stable identity **xfail(strict)**.

- **G7 `test_gate_persistent_graph_reuse.py`** — combined pmf_and_moments. G7a: shared-persistent vs fresh `GraphBuilder` (**same C core, not a different backend**); G7b/G7c: FFI vs fresh pybind (`assert_ffi_target`). Bit-identity throughout. Threaded (8), K=20; vmap batch 5 in G7b. discrete varies; **no xfail today** (all bit-identical this build).

- **G8/daisy `test_gate_daisy_chain_joint_probs.py`** — **SINGLE-PATH golden pin** (not cross-backend): drives only FFI `compute_daisy_chain_joint_probs_ffi`, pinned to x64 goldens; + jit-vs-eager self-consistency (`:156`). Value `rel=1e-5 abs=1e-9` (`:128`); gradient `rel=5e-2 abs=1e-4` (loose, survives FD→analytic swap, `:153`). One graph, 2 epochs × 3 params, single θ.

- **`test_gate_svgd_seams.py`** — **SINGLE-PATH golden pin**: pure Python/JAX leaf functions (priors, schedules, rbf_kernel, Adam, `SvgdConfig` schema) vs goldens. **Does not force x64** (`:31`) — goldens f32-captured, mode-fragile. No inference loop, no C path.

- **`test_gate_scc_ordering.py`** — **TOPOLOGY only** (no numbers): C++ `SCCGraph::sccs_in_topo_order` (scc_graph.cpp:70-81) vs Python Kahn reference; leveling; `_expected_scc_filenames` vs regex. `sccs_in_topo_order` non-topological **xfail(strict)** (Q10); `_expected_scc_filenames` returns `[]` **xfail(strict)**.

- **Native CTest `testcpp.cpp`/`tester.cpp`** — **single-path native vs closed-form** (pdf==exp(−t), NegBinomial, Erlang). Loose tol ~0.01→1e-4. **`PHASIC_BUILD_TESTS` defaults OFF (CMakeLists.txt:417)** — not built by normal pixi build; `main()` runs **only `test_pmf()`** (rest commented, `:638-644`); `phasic::Graph::moments` never called; `defect()`/cdf inconsistency is a **DISABLED** assertion (testcpp.cpp:623).

**Canonical strict-xfail map (KNOWN, currently-pinned cross-path divergences on master — expected, not bugs; an XPASS = a refactor silently unified the pair):** Q5 discrete moments unbound; Q6a/Q6b serialize constant_edges/vertex_indices; Q-G4-1/2 formula domain; Q1 cyclic+formula trace; Q11a sampler RNG; Q10 SCC topo; Q7.1 reward-PMF (G1).

---

## 4. THE GAP (unguarded routes — the harness work list)

Each item is a (quantity × path-pair × flag × graph-shape × θ) not covered by any non-xfail gate:

1. **JAX trace-replay vs FFI, same quantity — NEVER compared.** G5 pins Python-trace vs C-GraphBuilder; G1 pins FFI vs pybind; **no gate pins trace-replay == FFI**. `pmf_from_graph(use_ffi=False)` trace vs `use_ffi=True` FFI uncovered. **The single largest gap** — the two production JAX paths are never pinned to each other.
2. **ctypes-JIT path never exercised by any gate.** `pmf_from_graph_parameterized`, `pmf_from_cpp`, `moments_from_graph` all compile ctypes-JIT wrappers with **zero equivalence coverage**.
3. **`ptd_compute_pmf_multivariate` never driven** (in FFI_TARGETS `_gate_backend.py:43`; no gate calls it).
4. **`weight_mode='log'` has NO gate anywhere** — G4 only formula, G1/G2/G5 only linear. Combined with the known silent-ignore bug (moments_from_graph & joint_index ignore log), log-mode is untested + known-wrong across every quantity.
5. **Discrete (DPH) cross-path thin:** only G1 (one graph vs closed-form) + G7c (pybind self-vs-self). No discrete in G3/G5/daisy; no discrete FFI-vs-trace PMF; no discrete conditioned sampler; discrete moments xfail.
6. **Cyclic graphs — no numeric cross-path gate.** G5 cyclic xfail (Python trace refuses cycles), G_scc topology-only. FFI-vs-pybind PMF/moments on a cyclic graph (both C paths support cycles) never compared.
7. **Quantities with no cross-path gate at all:** CDF, `expected_sojourn_time`/`expected_waiting_time` (only used as a *probe* in G5, never as a compared value), Laplace transform, plain (non-daisy) `joint_prob_graph`/`joint_index`.
8. **SCC-decomposed / hierarchical-cache numeric equivalence ungated.** `scc_compose.c`/`distributed_scc` stitched pmf/moments never cross-checked vs monolithic elimination (0% native cov); G_scc topology-only; distributed/SLURM fully ungated.
9. **Graph-size regime:** every gate ≤7 vertices. No large-graph, no 100k-vertex trace-replay, no SCC-triggering-size numeric equivalence.
10. **θ regime narrow:** values O(0.3–3.7), dim 1–2 (max 6 daisy). No near-zero, no large, no high-dim, no batch/vmap except G7b, no random sweep except G7a (pybind self-vs-self).
11. **Gradient equivalence across paths ungated.** Only daisy checks grad (single-path golden, loose 5e-2, FD custom_vjp). No FFI-vs-trace gradient equivalence for pmf/moments; the mandated FD→analytic swap has no cross-path oracle.
12. **rewards≠None:** the only reward-PMF gate is xfail(strict) (FFI-uses-untransformed-graph bug). No gate pins a *correct* reward-transformed PMF across FFI/pybind/trace; reward-moments pinned only pybind-vs-FFI at 1e-12 (G1).
13. **Native-only impls ungated:** `phasic::Graph::moments/expectation/variance/covariance` (G2 impl-b) no CTest (verified); C++ `wf_tokenize/WFParser/wf_emit` compiler unreachable + untested.
14. **Serialize round-trip:** only PDF, only the non-divergent graph; moments/joint after round-trip unchecked; both real divergences xfail.
15. **Native CTest dormant:** `PHASIC_BUILD_TESTS` OFF by default; `main()` runs only `test_pmf()` — most native leaf coverage never executed by `pixi run test`.

---

## 5. INTENDED vs ACCIDENTAL

### Intended behaviour changes — differential diffs here are EXPECTED

- **Stage-1 dead-code removals (on master, all `(gated: approved)`):** the *capability is gone* by design — (a) symbolic-elimination subsystem `phasic_symbolic.c` + header/pybind/`__init__` [`cfbf2fc5`]; (b) Channel-3 generated-C++-from-trace [`4ee95276`, deletes `test_trace_rewards.py`/`test_trace_cpp_cache_key.py`]; (c) JAX-replay wrappers `trace_to_jax_fn`/`trace_to_pmf_function` [`deae0acc`]; (d) undefined decls from `api/c/phasic.h` [`ecdef2c3`]; (e) `_create_jax_callback_wrapper` [`eb49a9bd`]; (f) `trace_cache.py` disk helpers [`b73c1872`]; (g) duplicate FFI handlers `ffi_handlers.cpp` [`5f996415`]; (h) `src/c/trace/` shadow dir [`5a44e784`]; (i) corpse blocks in phasic.c [`9b8bc3f6`].
- **Stage-3 WS-B `97604614`:** delete unreachable `if self.cache_trace:` branch in `Graph.covariance()` + 4 orphaned `_*_from_trace` helpers. Claims "never executes; live `super().covariance()` path unchanged." Intended removal, **claimed no observable change** (so a `covariance()` output diff *is* a candidate bug — see §5 preserving).
- **`c340bedc` — `expected_sojourn_time(subset)` adjoint pass.** Reverse-mode adjoint replaces O(n·k) forward dense-matrix replay (avoids ~1.5TB alloc). **Explicitly NOT bit-identical:** "Equivalent to the forward path to summation-order rounding (~1e-15)." Old path kept behind `PHASIC_SOJOURN_FORWARD=1` (src/c/phasic.c:10205-10267). stage3-refactor-plan.md §1 = "reference exemplar." **A diff ≤ ~1e-15 is EXPECTED; a larger diff IS a bug.**
- **Current-branch numerical audit only (`d69919f2`,`12a30a78`,`cdda320f` — NOT on master; a separate later layer):** (a) `moments_from_graph` pow()-recurrence fixed → factorial moments [2,6,24,120] (was E[T²]=10 on Erlang(2,1)); (b) `_compile_wrapper_library` JIT build repaired; (c) `pmf_from_cpp(discrete=True)` no longer calls continuous `normalize()`; (d) raise-only guards: negative-PMF in SVGD + negative-rate rejection in both daisy FFI handlers. `12a30a78` reverted only the relative-FD-step part (restoring absolute eps=1e-7), keeping every other fix.

### Behaviour-preserving claims — diffs here are CANDIDATE BUGS

- **WS-C "pure verbatim relocation"** (stage3-execution-handoff.md:29): `de774f88` (plot/plot_scc_decomp→`_graph_plotting.py`), `1cd75c2d` (pull/push_cache→`_graph_cache_transfer.py`), `0d9ab967` (clear/prewarm_cache→`_graph_cache_mgmt.py`), `a6df2be9` (reward-validation→`_graph_reward_validation.py`), `98deb4da` (serialize/from_serialized→`_graph_serialize.py`). Each commit body: "Extracted verbatim … Pure relocation." **Any observable pre/post diff of these methods is a candidate bug.** (NOT independently diffed line-by-line — the cheap next check.)
- **Stage-3 touched no C/C++/pybind** (handoff:80-86): `git diff master..HEAD = 0 files` under `api/`,`src/c/`,`src/cpp/`; "C/C++ API untouched … native CTest 2/2." Any Stage-3-attributable C/C++ diff contradicts this.
- **Merge `9de5d61a`:** "Pure verbatim relocations … behavior-preserving (1746 passed, only 2 known-flaky)."
- **The 7 equivalence gates are "bit-identity"** (stage2 handoff §0/§4): G1 FFI==pybind (0.0), G2 `_moments`==`Graph.moments`==`compute_moments` (0.0), G4 C-VM==oracle (0 ULP), G5 trace==C engine (1e-6/1e-9), G7 persistent==fresh (==). **Any divergence at a pair NOT already strict-xfailed is a candidate bug.**
- **11 eager-vs-FFI forward-quantity pairs claimed parity** (plan §4a:194-198): pmf (phasic_pybind.cpp:4622), moments (`__init__.py:2199`), pdf/cdf, expected_waiting/sojourn_time, reward_visit_probability (phasiccpp.h:832), backward_probabilities, joint_prob_table, sample_path_conditioned, daisy_chain_joint_probs. **Caveat:** the on-branch FD-gradient audit shows these numeric surfaces are more fragile than the plan assumed.
- **C++ parity ports asserted equivalent but UNBOUND/native-only** (plan §3a): `Graph::moments/expectation/variance/covariance` (phasiccpp.h:510-676), `as_matrices/from_matrices` — untested claim (G2 impl-b / G4b open).

### Boundary moves — what actually happened
Stage-1/2/3 boundary activity was **deletions across the boundary**, not live-behaviour relocation: `cfbf2fc5` removed symbolic kernel from all three layers at once; `4ee95276` removed Channel-3 from Python; `5f996415` removed a duplicate C++ FFI handler set. **WS-C moved Python method *bodies* out of `class Graph` into sibling `_graph_*.py` via class-body assignment** — all stayed Python, but each moved fn's `__module__` changes `'phasic'`→`'phasic._graph_<cluster>'` (handoff:35-61), an **observable metadata change**; the G3 gate assertion was relaxed `== 'phasic'`→`'not phasic_pybind'` in the same commit (`98deb4da`) — a test change. **Any test keying on `.__module__`/`__qualname__` of a moved method sees a changed value.**

**Planned but NOT executed / NOT on master** (do not treat as done): the "complete-then-promote" C++-SSOT promotions of moments/serialize/as_matrices/weight-formula compiler/discretize/profile/add_epoch (plan §3a/b/§8, "highest-risk zone"); `compute_*_core` GIL-free extraction shared by pybind+9 FFI handlers; SCC de-mirroring; the phasic.c 9-way TU split. The 5 flagged behaviour-FIXES (Q5, Q6a/b, Q7.1, moments pow(), defect()/cdf()) are **future** changes still pinned as strict-xfails — **a forward diff at these points is NOT yet expected on master.**

**Sanctioned intentional dual-core (must NOT be collapsed — plan §4a:215-259):** composite algorithms (daisy-chain, joint-prob, sojourn, backward-prob) exist twice on purpose — eager pure-Python mirror + differentiable C-FFI. The **daisy-chain multi-epoch incremental forward** (per-epoch `update_ipv`→`update_weights(θ_epoch)`→`stop_probability(dt)`, graph_builder_ffi.cpp:1469-1476) is the flagged invariant, pinned by the daisy gate.

---

## 6. KNOWN-DEFECT BREADCRUMBS (pre-identified starting points)

- **`defect()`/`cdf()` inconsistency** (054f6fff msg; stage2 handoff:255-260 §7#1; plan §3c): for an IPV putting mass on an absorbing vertex, `defect()` (and `ptd_defect` after reward transform) returns 0.0 while `cdf()` reflects instant-absorption mass — internally inconsistent. Two native asserts DISABLED (testcpp.cpp:619/623, tester.cpp `test_rabbit`). Real production-semantics question, unresolved.
- **C++ `wf_*` weight-formula compiler unreachable/untested** (stage2 §7#2/§4 G4b; phasiccpp.cpp:720; the same compiler added in `6a94deb9`): no Python binding, `GraphBuilder` never takes a source string, `ptd_weight_tape` opaque, no C++ tape reader. G4b "cannot even be built today." A second unexercised weight-formula path beside `weight_formula.py`.
- **G2 impl-(b) native-only** (stage2 §7#3): `phasic::Graph::moments/expectation/variance/covariance` (phasiccpp.h:510) bound to nothing — dead from Python; coverable only by a future tests/cpp test.
- **`moments_from_graph(use_ffi=False)` pow()-recurrence** (stage2 §7#4; plan §3c #5; `__init__.py:7572`): `rewards3[j]=rewards2[j]*pow(rewards2[j], i)` — divergent/buggy recurrence, "live, default, numerically buggy." **FIXED only on current branch (`d69919f2`), NOT on master.** Flagged for a separate gate so no one "fixes" G2 to match it.
- **SCC ordering divergence — "the sharpest risk"** (plan §5, §4#9; stage3 handoff:WS-A): `ptd_compose_scc_prcs` (scc_compose.c:324-373) re-derives sink-first order via Kahn ("stored Tarjan order not topologically valid") while bound `SCCGraph::sccs_in_topo_order` (scc_graph.cpp:70-81) **trusts** stored Tarjan; both bound, different consumers, 0% native coverage — "a real behavioral divergence already exists." `sccs_in_topo_order` non-topological = strict-xfail (Q10).
- **`_expected_scc_filenames` silently broken** (plan §4#8): passes `(graph, metadata)` tuple to `compute_graph_hash` (expects a Graph) → always returns `[]`. Pinned strict-xfail. "A one-line fix."
- **`cache_file_path()`→`ptd_scc_build_synthetic_graph` spurious OOM** under pytest config (plan §4#9) — not root-caused; gate routes around it. 0%-covered SCC synthetic-graph code.
- **Notebook incident** (stage3 handoff:92-96): `git checkout -- docs/` destroyed ~24 uncommitted tutorial-notebook bug fixes (unrecoverable). Open follow-up: re-apply (details lost). Git status still shows 24 modified `.ipynb` — consistent.
- **FD→analytic gradient removal deferred** (stage3 handoff:120; plan §4a:226-245): three FD sites stay — moments use_ffi=False; reward-visit-prob central-diff custom_vjp (ffi_wrappers.py:1278); daisy-chain joint-probs FD custom_vjp (`__init__.py:11293`). **NOTE:** the on-branch audit (`cdda320f`, tagged [R] reproduced) concludes the FD gradient is "broken in BOTH step regimes … no central-difference step can fix this" at mixed parameter scales — a deeper unresolved issue than the deferral assumed.
- **Sojourn escape-hatch** (sojourn-fix.md:302,393-394): `PHASIC_SOJOURN_FORWARD=1` "still allocates the n·k matrix and will OOM on large k by design"; no-arg full `ptd_expected_sojourn_time` still O(n²).
- **SVGD hard crash on one bad particle** (deferred-svgd-divergece-fix.md): a single particle → θ implying rate ~1e31; the granularity error crosses the `jax.pure_callback` boundary and "the whole optimization dies — one bad particle out of 60 takes down the entire run." Repro: `test_model_selection::test_log_likelihood_independent_of_regularization`, `test_svgd_exposure::test_exposure_shifts_posterior_inverse_to_alpha` (the known-flaky pair).
- **SVGD LR-hug / LRT blocker** (deferred-svgd-lr-bug.md): cannot run a nested LRT reusing one fitted model's `SVGD.log_likelihood()` across an all-tied vs coalescent-free epoch daisy-chain joint-prob pair (3 enumerated blockers); `epoch_starts` TypeError in `SVGD.__init__`.
- **Notebook triage** (deferred-notebook-triage.md): docs pipeline (`scripts/docs-run-notebooks.sh`) only replaces outputs on success → a failing notebook silently keeps its last-good version. Open: model-selection.ipynb (2 failures), time-inhomogeneous.ipynb (graphviz `dot -c` env), distributed.ipynb (kernel SIGABRT: libgomp thread-creation failure → pybind11 `dec_ref` with invalid GIL → terminate — a failed thread spawn should raise, not abort).
- **`import phasic` does NOT enable JAX x64** (plan §4#10) — contrary to CLAUDE.md; x64 only flips when the FFI path first runs → SVGD leaf ops are float32 standalone but float64 inside a full run (order/context-dependent precision). The svgd-seams gate must compare with tolerance, not bit-exact.
- **Build-surface landmines** (plan §4#7): `src/phasic/phasic.h` `#include`s `src/c/phasic.c` directly (JIT `load_cpp_builder` path) — the planned phasic.c 9-way split must keep this working. `bffg.py` (612 lines) possible low-grade duplication of sojourn/reward accounting vs the C sampler (plan §4#6) — one confirmation pass recommended.

---

## 7. RECOMMENDED FIRST SWEEP (ranked — highest behaviour-change risk × unguarded)

1. **`expected_sojourn_time(subset)` adjoint vs `PHASIC_SOJOURN_FORWARD=1` legacy — temporal + cross-flag differential.** `c340bedc` is the single highest-risk compute-path change; only ~1e-15 equivalence claimed; reachable via svgd→joint_prob_table→`_get_joint_probs`. **A diff > rounding is a bug.** Pinned only by `test_sojourn_subset_adjoint.py`; no large-k / large-graph regime tested. Also diff the adjoint path against baseline `3082ebc6` (which predates it).
2. **JAX trace-replay vs FFI custom-call, same quantity (pmf & moments).** The largest structural gap (§4.1): the two production JAX paths are never pinned to each other. `pmf_from_graph(use_ffi=False)` trace vs `use_ffi=True` FFI, across acyclic + discrete, using granularity-dependent tolerance for continuous PDF.
3. **The 5 WS-C extracted modules vs their pre-extraction `Graph` methods** (claimed verbatim): plot/plot_scc_decomp, pull/push_cache, clear/prewarm_cache, reward-validation, serialize/from_serialized. Cheap, high-value: any observable pre/post diff is a candidate bug. Includes the `__module__` metadata change as a *known* diff to whitelist.
4. **`Graph.covariance()` before/after WS-B `97604614`** — the dead-branch deletion claims the live path is unchanged; diff the live `super().covariance()`/`covariance_discrete()` output.
5. **The 7 equivalence-gate pairs at inputs NOT already strict-xfailed**, pushed off their tiny fixtures into the unguarded regimes (§4.9–4.12): cyclic graphs (FFI-vs-pybind, both support cycles), larger graphs, near-zero/large/high-dim θ, vmap batches, rewards≠None *correct* transform. An XPASS on a strict-xfail = a refactor silently unified the pair (forcing function), not a bug.
6. **Discrete-DPH normalization tri-divergence** (atlas note): `pmf_from_graph_parameterized` calls `g.normalize()` before `dph_pmf` (`:3845-3849`) while `pmf_from_cpp` and the FFI `ComputePmfFfiImpl` do **not** (graph_builder_ffi.cpp:263; `:3979-3990`). For graphs with row sums ≠ 1 the "same" discrete PMF differs across paths — a concrete unguarded cross-path divergence to pin. (Note the current branch already touched (c); confirm master state.)
7. **The 11 eager-vs-FFI forward-quantity pairs** (claimed parity, plan §4a) — especially the daisy-chain `final_read='sojourn'` default route, where intermediate epochs remain approx-U (`stop_probability(dt)`) while only the final read is exact — a mixed-exactness surface with a loose (5e-2) gradient pin.

---

## 8. CONTRADICTIONS / UNVERIFIED (not papered over)

- **`import phasic` x64 claim.** CLAUDE.md states JAX is forced to 64-bit at import; the intent agent's plan §4#10 says it is **NOT** enabled at import (x64 flips only when FFI first runs). Direct contradiction; the svgd-seams gate behaviour (f32-captured goldens, `:31`) corroborates the plan's version, not CLAUDE.md.
- **Baseline build outlook — UNVERIFIED.** No build was actually run; the "should build with pixi" claim rests on git-log reasoning that the 4-arg `add_edge` migration predates the baseline (from `054f6fff`'s description). Read-only recon.
- **Removal "deadness" NOT independently confirmed.** Risk ratings for the 9 removal commits assume the commit messages ("Two-critic cleared", explicit dead-path arguments) are accurate; the tree was not grepped to confirm deadness. In particular, `5f996415` deleted a compute-path file (`ffi_handlers.cpp`) — verify no build target still expected the deleted symbols (surviving impls are `ComputePmfFfiImpl` etc. in `graph_builder_ffi.cpp`).
- **WS-C "verbatim" NOT diffed line-by-line** — the class-body relocations are claimed verbatim in commit bodies but were not byte-diffed (the cheap next check). `98deb4da` additionally edited a *test* gate assertion (G3), which is a test change, not production.
- **`stage1-orphan-removal-handoff.md` does not exist** in the working tree (only stage2/stage3 present) — Stage-1's record is external; some Stage-1 claims cannot be corroborated from repo docs.
- **`formula` weight_mode end-to-end through sojourn/daisy FFI — claim read, not test-verified.** The atlas agent explicitly did not confirm `formula` is honored via the C weight-tape VM through these handlers (only read the code comment); it also did not open `_daisy_chain_svgd_model` to confirm exposure vs no-exposure sub-routes, nor trace the BFFG `ptd_sample_path_conditioned` handlers.
- **Native `phasic::Graph::moments` parity is an untested claim** (plan §3a marks it ✅ "numerically equivalent") yet it is unbound/native-only with **no CTest** (G2 impl-b) — the ✅ and the "dead from Python's view" open item coexist unreconciled.
- **Daisy-chain gradient correctness not independently verified** — the gradient is FD custom_vjp (`__init__.py:11293`); the daisy gate pins it only against a loose golden (5e-2), and the on-branch FD audit contradicts the plan's assumption that the FD gradient is merely a temporary stand-in ("broken in BOTH step regimes"). Whether this is a defect or an accepted approximation is unresolved between the intent doc and the branch audit.
- **G7 self-vs-self caveat:** G7a/G7c compare pybind persistent-vs-fresh (same C core) — labelled "cross-path" loosely but is **not a different backend**; only G7b (FFI-vs-pybind) is a true cross-path check there.