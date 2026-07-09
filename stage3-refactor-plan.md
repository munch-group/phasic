# Stage 3 — Refactoring Review & Plan (Phase 1 verified, Phase 2 proposed)

**Status:** Phase 1 (adversarial re-verification of the map) is **complete**; Phase 2 (this plan) is
**presented for human sign-off**. No production code has been changed. Execution is blocked on the
decision gate in §9.

**Method:** 33-agent read-only fan-out (1.13M tokens): one investigator + one independent adversarial
critic per live §B target, parallel re-derivation of every god object, a doctrine-vs-live audit, a
lock-step/landmine sweep, and a completeness critic. **≥2-critic agreement was reached on every
target — zero investigator/critic disagreements.** Every line number below was freshly grepped on the
live `master` tree (the starting map's numbers were stale by ~4k lines in `phasic.c` after the Stage-1
corpse removal).

---

## 1. Verified baseline & deltas from the starting map

**Confirmed as given:** Stages 1 & 2 have landed on `master`. The tree is corpse-free at the spot-check
seeds (`ffi_handlers.cpp`, `phasic_symbolic.c`, Channel-3 `trace_to_log_likelihood`,
`save_trace_to_cache_python`, `ptd_instantiate_from_trace` all gone). The dead C trace trio is still
frozen in place. All 7 equivalence gates (`tests/pytest/test_gate_*.py`) are present and green. **Zero
declared-but-undefined build landmines remain** (the completeness critic diffed the whole public C
header).

**Six deltas the review found against the starting map — carry these forward:**

| # | Starting map said | Live reality |
|---|---|---|
| D1 | WS-C follows `h11-prompt.md` "verbatim" | **`h11-prompt.md` does not exist in the tree.** The only "h11" reference is inside this prompt. WS-C's mechanism must be chosen directly (→ Q11 is a hard blocker, not a pointer). |
| D2 | `phasic_pybind.cpp` shrinks; parity commits grew it | **False attribution.** The parity commits (`c64ddff0`/`6a94deb9`/`1948ab30`/`393f1366`) touched **only** `phasiccpp.h`/`.cpp`. `phasic_pybind.cpp` grew 5287→5880 for unrelated reasons and the parity ports are **not bound** there. |
| D3 | #5 moments is **triplicated** | **Quadruplicated**, and the C core has **no** copy at all. The 4th surface (`moments_from_graph`, `__init__.py:7572`) is **live, default, and numerically buggy** (`pow()`-squaring recurrence). |
| D4 | Channel-3 generated-C++ codegen fully deleted | A **second, distinct** generated-C++/ctypes channel survives: `moments_from_graph`/`pmf_and_moments_from_graph` **default to `use_ffi=False`**, compiling per-model `.so`s with **finite-difference gradients** (`__init__.py:7561-7648`). Violates doctrine rules 7 & 8. |
| D5 | #8/#12/Q8: drop `_mpfr` if unused | **Refuted.** `_mpfr` is a **gated, test-covered legacy path** (`#if HAVE_MPFR`, env-reachable via `PHASIC_FORCE_MPFR`/`PHASIC_USE_MPFR_LEGACY`). Keep it; fold it into the unified driver, don't delete. |
| D6 | #9 SCC ordering has 2 Python mirrors | **~5 mirror sites**, and the two divergent providers are **both bound** and consumed by different paths — a **live latent correctness risk**, not just duplication (see §5). |

**Recent commit `c340bedc` (sojourn adjoint fix)** is the reference exemplar of the target doctrine:
numeric kernel in the C core, forward path behind `PHASIC_SOJOURN_FORWARD=1`, pinned by a bit-identity
gate. Cite it in the WS-G doctrine doc.

---

## 2. The delegation doctrine, validated against live code (WS-G spec)

The doctrine (§A of the prompt) is **substantially violated today**, and the recent C++ parity ports
made it worse. Audit result, rule by rule:

| Rule | Verdict | Evidence |
|---|---|---|
| 1. C core owns every numeric kernel | **partial** | True for elimination/PMF/sampling/reward-transform/waiting-sojourn/AVL/hash/weight-tape. **Exception: `moments` never lived in the C core** — the recurrence is in C++/pybind/generated-string only. |
| 2. C++ = thin RAII facade + one JAX adapter | **violated** | `GraphBuilder::compute_moments_impl` reimplements the moment recurrence (`graph_builder.cpp:479`, comment admits it); `phasiccpp.h`/`.cpp` carry ~9 algorithm ports. |
| 3. pybind = mechanical marshalling only | **violated** | `phasic_pybind.cpp:212-408` is a family of numeric free functions (`_moments`/`_variance`/`_covariance`…), the **bound** moment implementation. |
| 4. Python never re-implements a native kernel | **violated** | `moments_from_graph` authors a native moment kernel as a **generated C++ string** + hand-rolled finite-difference gradient (`__init__.py:7561-7648`). |
| 5. One implementation per cross-pybind capability | **violated** | `moments` has **4** implementations; `serialize` has **3**; `as_matrices` has **3**. |
| 6. Cross-boundary contracts have one asserted version constant | **violated** | Trace-op enum defined twice (`phasic.h:723` vs `trace_elimination.py:53`); serialize JSON unversioned. Only `PHASIC_CALLBACK_VERSION` + binary `PTD_PCG_VERSION` exist. |
| 7. Differentiability = the one native engine via `jax.ffi` + `custom_vjp` | **violated** | The per-model generated-`.so` channel is **live and default** for `moments_from_graph` (D4). |
| 8. Fallbacks call the same impl or raise loudly | **violated** | Non-uniform: `pmf_from_graph` raises `PTDBackendError` when FFI is off; `moments_from_graph` **silently** runs the codegen second algorithm. |
| 9. Dead code deleted, not disabled | **violated** | ~9 compiled-but-unbound C++ parity ports, zero callers (the D2 cluster). |

**The doctrine to codify (edited to reality):** the 9 rules stand as the north star, with **three**
clarifications the audit forces: (1) **`moments` needs a single owning layer chosen deliberately** —
either a new `ptd_moments` C primitive (rule-1-pure) or `phasic::Graph::moments` as the typed facade
over the C `expected_waiting_time` leaf (pragmatic); (2) **the generated-C++/ctypes channel is the
last remaining rule-7 violation** and its removal (or explicit demotion) is the concrete deliverable
that makes rule 7 true — *subject to §4a's differentiability precondition*; (3) **the JAX dual-path
carve-out (§4a): goal #3's "one implementation" is qualified to "one source of truth for the numeric
core."** The eager-Python ↔ C-FFI duality is architecturally required by JAX's tracing model, so it is
a *sanctioned* pattern, not a lock-step defect.

---

## 3. Verified unification inventory

All 12 live §B targets re-confirmed on the live tree (critic-agreed). "Kind" drives the commit
strategy: **DELETE** (behavior-preserving dead-code removal), **RELOCATE** (verbatim move behind a
green gate), **BEHAVIOR-FIX** (flagged, own commit, flips a gate xfail).

### 3a. The C++ parity-port cluster — **HUMAN DECISION: Q4 = "C++ is a product → ports become SSOT"**

Added by the `c64ddff0`-era commit spree, **all compiled, none bound in pybind, zero callers today.**
The human chose to make these the single source of truth and have Python delegate — **not** delete them.

| Item | Location | §B row | Parity status vs live Python (⚠ = under-implements) |
|---|---|---|---|
| `wf_*` weight-formula compiler + `Graph::weight_formula` | `phasiccpp.cpp:726-1131` | #3 | ⚠ **no test harness** — anonymous-namespace, no C++ tape reader; G4b gate cannot even be built today |
| `Graph::serialize` / `from_serialized` | `phasiccpp.cpp:1132-1287` | #4 (3rd copy) | ⚠ a **third** impl distinct from both Python `from_serialized` and the reachable `GraphBuilder::parse_structure`; must reconcile which is authoritative |
| `Graph::moments/expectation/variance/covariance` | `phasiccpp.h:510-676` | #5 (copy b) | ✅ numerically equivalent to the bound `_moments` |
| `Graph::discretize` | `phasiccpp.cpp:442-518` | #8 | ⚠ **throws on parameterized graphs** (Python supports them via `_rebuild_with_wider_layout`) and **does not flag DPH** — promoting as-is **regresses the frozen Python API** |
| `Graph::as_matrices/from_matrices` | `phasiccpp.h:757` / `phasiccpp.cpp:326` | #10 (copy b) | ✅ value-equivalent (container types differ only) |
| `Graph::profile` + `GraphProfile::apply_snippet` | `phasiccpp.cpp:519-725` | #13 | ⚠ **partial** — omits the measured dyn-ordering probe present in Python `profile.py` |
| `Graph::add_epoch` + `epoch_rebuild_wider` | `phasiccpp.cpp:1288-1448` | **NEW** | needs equivalence check vs Python `add_epoch` |
| `Graph::extend` (BFS builder port) | `phasiccpp.cpp:410-441` | **NEW** | needs equivalence check vs Python `Graph.extend` |
| `_rebuild_with_wider_layout` C++ | `phasiccpp.cpp:1285` | **NEW** | structural helper for add_epoch/serialize |

**Governing constraint (non-negotiable #2): the public Python API is frozen.** Therefore "ports become
SSOT" **cannot** be a verbatim promotion where a port under-implements Python.

**DECISION (recorded): complete-then-promote, uniform C++ SSOT.** Every capability's source of truth
moves to the C++ facade. For each ⚠ port the promotion is a **three-step sequence**, each step its own
commit: **(1) reach parity** — extend the native port to full behavioral parity with Python (add
parameterized-graph support + DPH flagging to `discretize`; add the dyn-ordering probe to `profile`;
reconcile `serialize`/`from_serialized` into one authoritative C++ owner; build a C++ tape reader and
finish the `wf_*` compiler); **(2) bind + gate** — add the pybind binding and a per-port bit-identity
equivalence gate (green before and after); **(3) delegate** — switch the Python method to call the bound
native SSOT and delete the now-redundant Python/pybind/generated copy. The ✅ ports (`moments`,
`as_matrices`, and `add_epoch`/`extend` after a parity check) skip step 1. **The C API is frozen**
(Q13). This is the **largest native effort** and dominates WS-E/WS-D weight.

### 3b. Live duplications — unify (SSOT + gate)

| # | Capability | Live sites | SSOT / action | Kind | Gate | Blast |
|---|---|---|---|---|---|---|
| #1 | Parameterized elimination | Python `EliminationTrace` vs C `GraphBuilder` engine | C engine authoritative; Python trace scoped to hierarchical caching/JAX; **delete the dead `covariance(cache_trace=True)` branch** (`__init__.py:2401-2412` — unreachable, `cache_trace` forced False at `:1266`) | mixed | **G5** | med |
| #2 | Trace op-tape (dead C trio) | `phasic.c:12331/12563/12672` | DELETE the trio; keep the JSON cache surface; SSOT = Python `trace_elimination.py`. Enum divergence is real but **dormant** (`OpType.SUB`→8; C enum lacks it → default-case error, *not* silent CONST-coercion as the map claimed) | delete | (dead) | low |
| #3 | Weight-formula compiler | Python `compile_formula` (live) vs C++ `wf_*` (dead, →3a) | Python sole compiler; C VM (`phasic.c:5092`) sole evaluator; demote `eval_tape`→labelled oracle | delete+relocate | **G4** | low |
| #4 | serialize schema | Python `from_serialized` vs C++ `GraphBuilder::parse_structure` (the live pair) | One schema owner + asserted `SCHEMA_VERSION`; keep GraphBuilder as the reachable C++ path | mixed | **G3** | med |
| #5 | Moment recurrence | pybind `_moments` (bound) + `GraphBuilder::compute_moments_impl` + generated string `__init__.py:7572` | One recurrence (phasic::Graph or `ptd_moments`); pybind binds member ptrs; GraphBuilder delegates | mixed | **G2** | med |
| #7 | FFI orchestration | `graph_builder_ffi.cpp` 9 handlers re-implement the times-loop/dispatch | Extract GIL-free `compute_*_core` both pybind & FFI call | mixed | **G1/G7** | med |
| #9 | SCC composition/ordering | C Kahn (`scc_compose.c:324-373`) vs C++ trust-Tarjan (`scc_graph.cpp:70-81`), **both bound**; ~5 mirror sites | Single native topo-order provider (C Kahn authoritative); de-mirror level + cache-path | mixed | **NONE → build first** | **high** |
| #10 | Matrix interop | 3 copies (bound lambda / orphan Eigen class / dead port) | phasic::Graph SSOT; bind thinly; delete copies (no behavioral divergence) | relocate | none (add thin) | low |
| #11 | Conditioned samplers | `..._conditioned` vs `..._conditioned_fixed` (`phasic.c:10674` / `:10763`) | One inner loop parameterized by RNG callback + emitter | relocate | **G6** | med |
| #12 | C 5-variant elimination family | `phasic.c` V1-V5 (6873-9718, ~2200 lines) | One driver + (vertex-order policy × arithmetic-sink) strategy hooks; **keep `_mpfr`** (D5) | mixed | **NONE → build first** | **high** |
| #13 | Profiling port | C++ `profile()` (dead, →3a) vs Python `profile.py` | Python sole orchestrator; delete C++ port | delete | none | low |

### 3c. Behavior-fixes (each its own flagged commit; flips a gate xfail)

- **Q5** — `Graph.moments(discrete=True)` → `AttributeError` (`__init__.py:2250` calls unbound
  `moments_discrete`). The **only** AttributeError-on-first-call surface in the whole class (completeness
  critic diffed 40+ `super()` delegations). Flips `test_g2_moments_discrete_unbound_xfail`.
- **Q6a/b** — `from_serialized` drop of `constant_edges` + ignore of `vertex_indices`. Flips two G3 xfails.
- **Q7.1** — FFI computes the combined PMF on the **untransformed** graph (`graph_builder_ffi.cpp:493`)
  vs pybind's reward-transformed graph (`graph_builder.cpp:741`). Flips a G1 xfail.
- **#5 pow() bug** — `moments_from_graph(use_ffi=False)` recurrence `rewards3[j]=rewards2[j]*pow(...)`
  (`__init__.py:7572`) — **live, default, wrong**. Rank ahead of the cosmetic #5 relocations.
- **defect()/cdf() inconsistency** — IPV mass on an absorbing vertex: `defect()` returns 0.0 while
  `cdf()` reflects it (native asserts disabled at `testcpp.cpp:619`). Owns its own semantics decision.

---

## 4. New findings beyond §B (fold into the plan)

1. **Surviving generated-C++/ctypes codegen channel (D4).** `moments_from_graph` /
   `pmf_and_moments_from_graph` default `use_ffi=False` → compile a per-model `.so` with **finite-difference**
   gradients. This is a rule-7 violation and the home of the #5 pow() bug.
   **Re-decided (author-directed): DEFERRED — keep the channel for now to preserve functionality.** Removing
   it depends on a validated analytical moments gradient (§4a), which is a later effort. Not deleted in this
   refactor pass. (The pow() forward bug is a separate validated fix inside the kept channel — see §4a.)
2. **Non-uniform fallback policy (rule 8).** Standardize the `pmf_from_graph*`/`moments_from_graph`
   family on one policy (loud raise when FFI is unavailable).
3. **No cross-boundary version constants (rule 6).** WS-G adds asserted `TRACE_OP_VERSION` and
   `SERIALIZE_SCHEMA_VERSION` checked on both sides.
4. **Extra #9 mirror sites** (completeness critic): a 3rd C++ level mirror inside the dead `profile`
   port (`phasiccpp.cpp:549`), and 3rd/4th cache-path mirrors in `compute_repository.py:293`/`:561`. The
   #9 gate + SSOT must enumerate **all** of them or they drift.
5. **`state_indexing.py` (2266 lines)** is a god-object/split candidate §C under-listed (Python-only;
   the `StateIndexer`→`IndexCodec` extraction of Q12).
6. **`bffg.py` (612 lines)** — unconfirmed possible low-grade duplication of sojourn/reward accounting
   vs the C sampler/reward-transform; one confirmation pass recommended before touching it (not a
   blocker).
7. **Build-surface landmine:** `src/phasic/phasic.h` is a 16-line amalgamation that `#include`s
   `src/c/phasic.c` **directly**, consumed by the `load_cpp_builder` JIT path. The `phasic.c` 9-way
   TU-split (WS-F) **must keep this include working** — split into headers the amalgamation re-includes,
   or update it in the same commit.
8. **(Found while building the #9 gate) `compute_repository._expected_scc_filenames` is silently
   broken** — it passes the `(graph, metadata)` tuple from `as_synthetic_graph()` to
   `compute_graph_hash` (which takes a `Graph`), so it always returns `[]` (a cache-path mirror that has
   already drifted). Pinned by `test_gate_scc_ordering.py` (strict-xfail). A one-line fix, but flag it in
   the #9 cache-path unification.
9. **(Found while building the #9 gate) `cache_file_path()` → `ptd_scc_build_synthetic_graph` raises a
   spurious "out of memory"** under the *project pytest configuration* (works under plain `python` and
   for the toy-model graphs). Native fragility in the 0%-covered SCC synthetic-graph code — more evidence
   #9/#12 native work needs hardening. Not yet root-caused; the gate routes around it.
10. **(Found while building the svgd seam gate) `import phasic` does NOT enable JAX x64** — contrary to
    the CLAUDE.md claim ("JAX is forced into 64-bit mode at import"). After `import phasic`,
    `jax_enable_x64` is still `False` and `jnp.array([1.0]).dtype == float32`; x64 only flips on later
    (when the FFI path first runs). Consequence: SVGD leaf ops (kernel/prior/optimizer/schedule) run
    **float32 standalone but float64 inside a full run** — an order/context-dependent precision. Relevant
    to WS-C (svgd split) and WS-G (fix the CLAUDE.md claim or actually force x64 at import); it also
    means the svgd seam gate must compare with tolerance, not bit-exact, across ambient modes.

---

## 4a. Sanctioned JAX dual paths — do NOT collapse (correction after author review)

The SVGD/JAX inference path deliberately keeps several algorithms **C-side** so that JAX can `jit`/
`grad`/`vmap` through them — you cannot trace through an opaque pybind call, so the composite-model
loops exist **twice on purpose**: an eager pure-Python version and a differentiable C-FFI version.
This is **load-bearing, not a defect**, and the original goal #3 ("no separate Python and C/C++
implementations") must be read as **"one source of truth for the numeric *core*, with sanctioned eager
+ FFI adapters."**

**Governing invariant (author-affirmed): every forward PH-distribution *property* must be computable
eagerly, outside the JAX/SVGD path, for non-inference use.** The eager pybind → `phasic::Graph` → C-core
surface is a **first-class standalone product** (a scientist computing a PMF / moments / sojourn times of
a concrete coalescent model), **not** a probe/helper for inference. **Verified: this invariant already
holds** — all 11 FFI forward quantities have an eager counterpart (`pmf`→`phasic_pybind.cpp:4622`,
`moments`→`__init__.py:2199`, `pdf`/`cdf`, `expected_waiting_time`/`expected_sojourn_time`,
`reward_visit_probability`→`phasiccpp.h:832`, `backward_probabilities`, `joint_prob_table`,
`sample_path_conditioned`, `daisy_chain_joint_probs`). The refactor must **preserve** it.

**Precise delineation (what must / must-not have an eager path):**
- **Forward property *values*** (pmf/pdf/cdf, moments, sojourn/waiting time, sampling, reward transform,
  defect, discretize, joint/reward-visit probabilities, as_matrices/serialize) → **eager path REQUIRED**
  (the invariant above).
- **Gradient / `custom_vjp` / pure-JAX-replay machinery** → legitimately **inference-only** (gradients
  only matter for inference).
- **Inference *constructs*** (log-likelihood, exposure/zero-inflation, the observation-tailored daisy
  model) → **not properties**; inference-only is fine. But the PH *properties* they consume still obey
  the invariant.

**The real defect, restated:** not the dual *adapter* (eager + FFI) — that is sanctioned — but any dual
*core*, where the numeric algorithm is implemented **inside an FFI handler in C++ AND separately
re-mirrored in pure Python**. The end state is **one core in the C layer, two thin adapters** (eager
pybind, inference FFI) both calling it. That makes doctrine **rule 1 the operational test**.

**Three JAX-compat mechanisms, all intentional:**

1. **XLA-FFI handlers** — `compute_pmf_ffi`, `compute_moments_ffi`, `compute_pmf_and_moments_ffi`,
   `compute_daisy_chain_joint_probs_ffi`, `compute_daisy_chain_sojourn_ffi`, `backward_probabilities_ffi`,
   `sample_path_conditioned_ffi` (`ffi_wrappers.py`). The composite algorithms (daisy-chain, joint-prob,
   sojourn) are reimplemented C-side (`graph_builder_ffi.cpp`, 9 handlers) as the differentiable JAX
   entry; the eager Python mirror (e.g. `_resolve_daisy_chain_t_eval`, `__init__.py:11333`) serves setup/
   probe. The `pure_callback` fallback in `pmf_from_graph` is **deliberately disabled** — the JAX path
   *requires* FFI (`__init__.py:4770-4804`; commented wrappers in `ffi_wrappers.py:339-508`).
2. **`pure_callback` + `custom_vjp` + finite/central differences** — differentiability *shims* where a C
   primitive lacks a batched-theta FFI or an analytic gradient (`compute_reward_visit_probability_ffi`,
   `ffi_wrappers.py:1162-1180`, `custom_vjp` at `:1278`; and the `moments_from_graph`/
   `pmf_and_moments_from_graph` `use_ffi=False` paths).
   **⚠ Finite-difference gradients: replacement is DEFERRED (author-directed, supersedes the earlier
   "delete unconditionally").** *Preserving working functionality overrides the "no finite-difference
   gradients" goal.* "Zero finite-difference paths" remains the eventual **target**, but the analytical
   replacements are a **later effort, out of scope for this refactor pass** — the finite-difference paths
   **stay working** wherever removing them would risk functionality (which is all three sites today, since
   none has a validated analytical replacement). The three deferred sites:
   - **(i) moments** — `moments_from_graph`/`pmf_and_moments_from_graph` `use_ffi=False`. Analytical
     eventually via the differentiable trace replay. *Deferred; codegen channel kept for now (see §4 #1).*
   - **(ii) reward-visit-probability** — `compute_reward_visit_probability_ffi` central-difference
     `custom_vjp` (`ffi_wrappers.py:1278`). Analytical eventually via an adjoint through
     `ptd_backward_probabilities`. *Deferred.*
   - **(iii) daisy-chain joint probabilities — the hardest, and the highest-value inference path.**
     `Graph.daisy_chain_joint_probs` finite-difference `custom_vjp` (`__init__.py:11293`) over a
     **multi-epoch sequential** forward (`for epoch in 0..n_epochs-1`: `update_ipv → update_weights(θ_epoch)
     → stop_probability(dt)` incrementally stepping the CTMC to time t, then projecting the surviving
     distribution into the next epoch's IPV; final-epoch read-out — `graph_builder_ffi.cpp:1469-1476`). An
     analytical adjoint would have to backprop through the whole epoch chain incl. inter-epoch IPV coupling.
     *Deferred — do not attempt during this refactor; the finite-difference gradient stays.*

   **⚠⚠ DAISY-CHAIN INVARIANT (author-flagged — must not break; now the PRIMARY concern).** With the
   analytical adjoint deferred, the job here is purely **preserve the working path**: **(a)** the multi-epoch
   incremental forward (per-epoch IPV projection) stays intact — #7's FFI-orchestration unify must **not
   flatten the epoch loop** into a single-shot compute, and Q1's trace-SSOT reroute must **not assume a
   single static-graph trace**; **(b)** the path stays differentiable, gradient flowing through **every**
   epoch (the finite-difference `custom_vjp` is retained). Pinned by the new gate
   **`test_gate_daisy_chain_joint_probs.py`** (value tight; per-epoch gradient-block-norm structural check;
   jit==eager) — in the `equivalence` set, **must be green before #7 / Q1 touch this path.** Its per-epoch
   block-norm assertion fails the instant a refactor disconnects an epoch.
3. **Pure-JAX trace replay** — `evaluate_trace_jax` (`trace_elimination.py:1443`), fully native (no C call
   inside the traced region; full jit/vmap/grad/pmap test suite). **Not dead code** — it is the one
   mechanism that could eventually *eliminate* the dual path, and it is the concrete upside of the Q1
   "export the C op-arrays to JAX" direction.

**Plan corrections this forces:**
- **#7 stays a "share the core, keep both entry adapters" job — never collapse the eager/FFI pair to one.**
  Generalize the shared-`compute_*_core` extraction to the whole daisy-chain/joint-prob/sojourn/
  backward-prob family (which §3b under-listed as just "compute_* orchestration").
- **⚠ Q-CODEGEN — DEFERRED (author-directed: preserve functionality first).** The
  `moments_from_graph`/`pmf_and_moments_from_graph` `use_ffi=False` codegen path computes gradients by
  **finite differences**. The eventual target is an analytical moments gradient (via the JAX-differentiable
  trace replay, the Q1 direction, or a C-engine-backed analytic `custom_vjp`), after which the
  codegen/finite-diff channel is removed — **but this is a later effort, not this refactor pass.** For now
  the codegen/finite-diff moments channel **stays** (it is the working functional path). Do **not** delete
  it before a validated analytical replacement exists. The `pow()`-recurrence forward bug is a **separate**
  question (a wrong forward value, not a gradient issue) — investigate whether it is genuinely wrong and, if
  so, fix it as its own validated behavior-fix commit *inside the kept channel*; do not fold it into a
  relocation, and validate against current SVGD behavior (functionality preservation).
- **`evaluate_trace_jax` is re-classified** from "kept, intended-but-unwired" to **"strategic Q1 lever"**;
  do not consider it for deletion.
- **WS-G guardrail nuance:** the "fail CI on new *lock-step*/*MUST match* comments" grep must **whitelist
  the sanctioned JAX dual-path family** (weight-tape opcode sync, the FFI/eager mirror comments), or it
  will fire on load-bearing code. Distinguish "sanctioned dual path" from "gratuitous port" in the guard.
- **WS-G invariant guard (new):** add a CI check that **every FFI forward-quantity handler has an eager
  pybind/`phasic::Graph` counterpart** (the invariant above), so a future change can't add an
  inference-only forward quantity with no standalone eager path. All 11 pass today; the guard freezes that.

## 5. Why #9 is the sharpest risk (read before sequencing)

`ptd_compose_scc_prcs` (`scc_compose.c:324-373`) **explicitly re-derives** a sink-first topological
order via Kahn's algorithm because *stored Tarjan order is not topologically valid*. The bound C++
`SCCGraph::sccs_in_topo_order` (`scc_graph.cpp:70-81`) **trusts** the stored Tarjan order. Both are
bound (`phasic_pybind.cpp:3535` and `:3565`) and consumed by different downstream paths, with **0%
native coverage and no equivalence gate**. This is the one place where the safety net is thinnest and a
real behavioral divergence already exists. **A gate that pins ordering + level + cache-path parity
across all ~5 provider sites must exist before #9 or #12 is touched, and before #1's SSOT work lands
upstream of it.**

---

## 6. God-object inventory (current, verified line ranges)

| Unit | File | Lines (map) | Split target |
|---|---|---|---|
| `class Graph(_Graph)` | `__init__.py` 1141-11750 | 10,610 (10,665) | 17 clusters → WS-C mixins: construction, weight-config, weights/cache, moments, pmf/pdf/cdf, sampling, discretize/epochs, reward-transform, reward-validation, serialize/matrices, pmf-factory, **inference/svgd (2210)**, moments/pmf-factory (1405), plotting (617), copy/rebuild, **joint-probability (1632)**, dist-cache |
| `phasic.c` | `src/c/phasic.c` | 13,343 (17,507) | 16 concerns (non-contiguous) → 9 TUs; dedup #11/#12 **before** the split so code moves once; preserve the `phasic.h` amalgamation |
| `svgd.py` | `src/phasic/svgd.py` | 9,611 (9,611) | 5 leaf modules — `svgd_priors` (496-1524), `svgd_optimizers` (1782-2484), `svgd_schedules` (1525-1781 + 2485-2875), `svgd_preconditioners` (3334-3745), `svgd_kernels` (3746-4379) — + re-exports; SVGD class 4380-9611: 929-line/27-param `__init__`, plotting (1331+696), diagnostics (506). **Also purge dead commented blocks (2485-2875, 2876-3333).** |
| `phasic_pybind.cpp` | `src/cpp/phasic_pybind.cpp` | 5,880 (5,287) | per-class `register_*()`; push DOWN into phasic::Graph: `_moments` (212-411), matrix copies (117-211), 4 BFS builders (412-676), JIT toolchain `load_cpp_builder` (4920-5086) |
| `phasiccpp.h` + `.cpp` | `api/cpp/…` + `src/cpp/…` | 2,083 + 1,892 | extract context classes + move inline bodies out-of-line; **delete the §3a dead ports** (delete-dominant, not extract) |
| `graph_builder_ffi.cpp` | `src/cpp/parameterized/…` | 1,962 (1,962) | **9 handlers** (map said 4) + 10 `Create*Handler()` → one batch-dispatch template delegating to `GraphBuilder::compute_*` (#7) |
| `state_indexing.py` | `src/phasic/state_indexing.py` | 2,266 | (Q12) extract `IndexCodec`; Python-only, no native mirror |

---

## 7. Workstreams — go/no-go (decisions locked)

- **WS-A (safety net)** — delivered by Stage 2. Consume. **Extend** with: the **#9 SCC
  ordering/level/cache-path parity gate** (all ~5 sites), a **svgd.py seam gate** before WS-C moves it,
  and a **per-port bit-identity gate** for each C++ SSOT promotion (§3a step 2). **GO.**
- **WS-B (dead-code)** — mostly delivered. Residual now smaller (Q4 = promote, not delete): only the
  dead C trace trio stays **frozen** (Q13 = freeze C API), the surviving codegen channel is deleted
  under WS-E (Q-CODEGEN), and the dead `covariance(cache_trace)` branch + svgd.py commented blocks are
  removed under WS-C. **GO.**
- **WS-C (Python god-objects)** — **GO. Mixins, full scope** (Q11): Graph mixin split, `Graph.svgd`→
  `SvgdConfig`, `svgd.py` module split. Parallelizable in a worktree. Adds `IndexCodec` extraction (Q12).
- **WS-D (pybind thinning + moment/matrix promotion)** — **GO.** Split `PYBIND11_MODULE` into
  `register_*()`; promote the ✅ ports (`moments`, `as_matrices`) to bound C++ SSOT; the Q5
  `moments_discrete` fix. Behind G2/G3.
- **WS-E (single parameterized engine + contracts + C++ SSOT promotions)** — **GO, now the largest
  workstream** (Q4 uniform-SSOT). Owns: the complete-then-promote sequence for every ⚠ port (§3a),
  delete the generated-C++ codegen channel + route to FFI (Q-CODEGEN), serialize version constant +
  Q6 fixes, weight-formula C++-compiler promotion (Q3), FFI→`compute_*` core + Q7.1, trace SSOT (Q1),
  de-mirror SCC (#9/Q10). Blocked on the #9 gate.
- **WS-F (C-core decomposition + C-internal dedup)** — **GO** behind the **#9 and #12 gates built
  first**. Keep `_mpfr` (Q8). Preserve the `phasic.h` amalgamation across the `phasic.c` split.
- **WS-G (doctrine + guardrails)** — **GO**; drafted after the gate, finalized after WS-D/E. Adds the
  cross-boundary version constants (rule 6) and the reachability/orphan + lock-step CI guards. The Q12
  **InferenceResult** shared base lands here or in WS-C.

---

## 8. Safest-first commit sequence (each: install-dev + suite green vs baseline; relevant gate green)

*Legend:* ⚠ = flagged behavior-fix (own commit). ★ = C++-SSOT promotion (3-step: reach-parity → bind+gate → delegate+delete).

0. **Build the missing gates** (WS-A): #9 SCC ordering/level/cache-path parity (all ~5 sites); svgd.py
   extraction-seam bit-identity. *(nothing risky proceeds without these)*
1. **WS-B small deletions:** dead `covariance(cache_trace)` branch (unreachable); svgd.py dead commented
   blocks. *(pure deletions; mixins then cut from a clean tree. The C trace trio stays frozen — Q13.)*
2. **WS-C** Graph mixin extractions, one cluster per commit (worktree, parallel).
3. **WS-D** split `PYBIND11_MODULE` into `register_*()` (pure relocation).
4. **WS-D ★** promote `moments` to bound `phasic::Graph` member SSOT; delete pybind free `_moments` +
   `GraphBuilder::compute_moments_impl` delegates to it [G2 green].
5. **⚠ WS-D** `moments_discrete` binding (Q5).
6. **DEFERRED (Q-CODEGEN):** analytical moment gradients + removing the finite-difference/codegen channel
   are a **later effort**, not this pass — the channel stays to preserve functionality (§4a). *If* the pow()
   forward recurrence is confirmed genuinely wrong, fix it as its own validated behavior-fix inside the kept
   channel [G2] — otherwise this step is a no-op for now.
7. **WS-D ★** promote `as_matrices`/`from_matrices` to bound C++ SSOT; delete the 2 in-binding copies.
   Unify the 4 BFS builders; extract the JIT toolchain out of the binding.
8. **WS-C** `Graph.svgd`→`SvgdConfig` decomposition + `svgd.py` module split [seam gate green].
9. **WS-E ★** `serialize`: reconcile the 3 impls to one C++ owner + shared `SCHEMA_VERSION`; bind; then
   **⚠** `constant_edges`/`vertex_indices` fixes (Q6) [G3].
10. **WS-E ★** weight-formula: build the C++ tape reader + G4b gate; finish the `wf_*` compiler; bind it
    as SSOT; Python `compile_formula` delegates; fix the `OpType.SUB` enum divergence [G4].
11. **WS-E ★** `discretize`: extend the native port to parameterized graphs + DPH flagging (reach
    parity); bind; Python delegates [new per-port gate].
12. **WS-E ★** `profile`: add the dyn-ordering probe to the native port (reach parity); bind; Python
    delegates [new per-port gate].
13. **WS-E** FFI → shared `compute_*_core` (persistent-graph, Q7) + **⚠** Q7.1 reward-transform fix [G1/G7].
14. **WS-E** trace SSOT restructuring (Q1) — C engine authoritative, Python trace scoped to caching/JAX.
15. **WS-E/F** de-mirror SCC composition + single topo-order provider (all ~5 sites, Q10) [#9 gate green].
16. **WS-F** unify the two conditioned samplers [G6].
17. **WS-F** unify the 5-variant elimination family into one driver (keep `_mpfr`, Q8) [#12 gate green].
18. **WS-F** split `phasic.c` into 9 TUs; move `phasiccpp.h` bodies out-of-line (preserve the
    `phasic.h` amalgamation for the JIT include).
19. **WS-C/G** Q12 niceties: shared `InferenceResult` base (SVGD+MCMC); `StateIndexer`→`IndexCodec`.
20. **WS-G** doctrine doc (`docs/architecture/boundary.md`) + version constants + reachability/lock-step CI guards.

---

## 9. Decision gate — human answers (recorded)

| Q | Decision | **Answer (recorded)** → consequence |
|---|---|---|
| **Q4** | C++ API product or Python-only? | **"C++ is a product → ports become SSOT."** The §3a ports are **promoted, not deleted**; C API **freezes**. Behavior-sensitive; each ⚠ port needs a reach-parity sub-commit. **Open follow-up fork below.** |
| **Q11** | WS-C mechanism + scope | **Mixins, full scope** — Graph mixin split + `Graph.svgd`→`SvgdConfig` + `svgd.py` module split. |
| **Q-CODEGEN** | Surviving generated-C++ channel (D4) | **DEFERRED (author-directed): keep it to preserve functionality.** Analytical-gradient replacement is a later effort; do not delete finite-diff/codegen this pass. pow() forward bug = separate validated fix inside the kept channel. |
| **Q5/Q6/Q7.1/Q12** | Behavior-fixes & niceties in scope | **All four IN** — `moments_discrete` (Q5), serialize `constant_edges`/`vertex_indices` (Q6), FFI reward-transform (Q7.1), **and Q12 niceties** (InferenceResult base + `IndexCodec`). |
| Q1 | Trace SSOT direction | C engine authoritative; Python trace kept for hierarchical caching/JAX behind G5. (Recommendation; confirm at WS-E.) |
| Q3 | Weight-formula compiler home | **Reframed by Q4:** promote the C++ `wf_*` compiler to SSOT and bind it; Python `compile_formula` delegates. Requires a C++ tape reader to build the G4b gate first. |
| Q7 | FFI graph lifetime | Persistent thread-local + documented single-graph-per-thread contract (G7 green). |
| Q8 | `_mpfr` variant | **Keep** (D5 — gated legacy); fold into the unified #12 driver. |
| Q10 | SCC composer keep both / retire one | Keep both through the refactor; unify onto one topo-order provider (C Kahn authoritative) once the #9 gate exists. |
| Q13 | Prune vestigial public C symbols | **Freeze** (follows from Q4 = C++/C API is a product). |
| Q2, Q9 | Channel-3; C JSON trace cache | Resolved by Stage 1 (deleted / kept-live). Closed. |

### Open follow-up fork (raised by the Q4 answer)

Because several ports **under-implement the frozen Python API** (⚠ in §3a — `discretize` throws on
parameterized graphs, `profile` omits the probe, the C++ `serialize` is a third divergent impl, the C++
weight-formula compiler has no test harness), "ports become SSOT" has two valid readings with very
different cost/risk. **This is the one remaining user-owned decision blocking WS-D/E design** — see the
interactive follow-up.

**Definition of done for this prompt:** ✅ doctrine validated; ✅ inventory verified; ✅ tree corpse-free;
✅ god-object ranges re-derived; ✅ workstream plan + commit sequence; ✅ decision gate answered — with one
follow-up fork (below) that the Q4 answer surfaced. Execution begins once the fork is resolved.
