
# Questions

1. Is graph caching active by default?
2. Can the c path be made to support hierarchical elim/cache


# Todo

- Remove references to python path symbolic trace


The most important thing is to maintain the core functionality: disk-cashing of the O(n^3) 
  computation. The ambition and long term goal is to support hierarchical SCC decomposition,
  distributed computation of SCCs and disc-cashing of the assembled result (as is almost
  implemented with the python Elimination trace path). Would it be a way forward to keep the 
  parameterized reward compute graph as the default path for SVGD and in parallel develop the 
  missing/defunct parts of the Elimination trace path? 



## C-side disk caches in phasic: status and coverage

### What "the C trace cache is dormant" means

The `~/.phasic_cache/traces/` cache is dead-but-not-deleted code on the C side. Specifically:

- **The reader/writer is intact.** `src/c/trace/trace_cache.c`, plus `ptd_load_trace_from_cache` / `ptd_save_trace_to_cache` (`src/c/phasic.c:988+`, `:1054+`), are fully implemented and exported in the public C API. A C caller could still read or write a trace JSON to `~/.phasic_cache/traces/<hash>.json`.
- **But nobody calls them.** The function that would have wrapped elimination + lookup, `ptd_record_elimination_trace` (`src/c/phasic.c:12561+`), is commented out. The note at `src/c/phasic.c:4213–4216` explains it was removed because it caused memory blow-ups on large graphs and was never used downstream.
- **No producer.** Since C never calls `ptd_save_trace_to_cache`, no C path writes those files. The directory is populated only by the Python side (`src/phasic/trace_serialization.py`, driven by `hierarchical_trace_cache.py`) — and that pipeline is itself behind the now-deprecated `Graph.compute_trace`.
- **No consumer either.** The only thing that would call `ptd_load_trace_from_cache` is the commented-out `ptd_record_elimination_trace`, so the C loader never fires.

In short: functional, reachable from a hypothetical C caller, but no current execution path on either the producer or consumer side touches it. Removing the code would not change any observable behavior of phasic today.

### Are the parameterized reward compute graph and the elimination trace both un-cached on disk?

No — those are two different caches with different statuses.

**Parameterized reward compute graph (Stage A2): actively cached to disk.**

- Location: `~/.phasic_cache/parameterized_reward_compute/<hash>.bin`
- Producer: every parameterized eager moment / waiting-time call goes through `ptd_precompute_reward_compute_graph`, which on cache miss runs the O(n³) Gaussian elimination and then calls `ptd_save_parameterized_reward_compute_graph` (`src/c/phasic.c:1949`). On cache hit it calls `ptd_load_parameterized_reward_compute_graph` first and skips elimination.
- Gated only by `PHASIC_DISABLE_CACHE=1`. So for any normal SVGD / `Graph.moments` / `Graph.expectation` workload, this cache fills up automatically.

**Elimination trace (`~/.phasic_cache/traces/`): not actively cached by any C code path.**

- The C reader/writer exists (`ptd_load_trace_from_cache` / `ptd_save_trace_to_cache`) but its sole intended caller `ptd_record_elimination_trace` is commented out, so no C path writes there.
- Python *can* still write there via `hierarchical_trace_cache.py` + `trace_serialization.py`,  but only when invoked through `Graph.compute_trace()` — which is deprecated and emits a  `DeprecationWarning`. The default public moments/expectation path skips it entirely (the trace  branches in `Graph.moments` / `expectation` / `variance` are commented out at `__init__.py:2461+`, `:2523+`, `:2585+`).

So Stage A2 is live and serving real workloads; the trace cache is effectively orphaned — 
populated only if a user explicitly opts into the deprecated trace API.

### Is the Stage A2 cache produced/used by all user-facing methods?

No — it's used by the parameterized eager moments path and only by that path. Concrete coverage:

**Where Stage A2 is hit (producer + consumer)**

`ptd_precompute_reward_compute_graph` is the *only* site that loads/saves `~/.phasic_cache/parameterized_reward_compute/`. It runs whenever the C code needs the symbolic compute graph for a parameterized graph. Reaching it from Python requires:

1. The graph is parameterized.
2. The call asks for a moment / waiting-time quantity.

Concretely:

- `Graph.moments` (`__init__.py:2430`) → `super().moments` → `_moments` → `ptd_expected_waiting_time` → `ptd_precompute_reward_compute_graph` ✓
- `Graph.expectation`, `Graph.variance`, `Graph.covariance` — same chain ✓
- `Graph.expected_sojourn_time(indices)` when invoked on a parameterised graph ✓ (subset variant 
feeds the same precompute)

**Where it is NOT hit**

- **PDF / PMF / CDF paths.** `Graph.pdf`, `Graph.cdf`, `Graph.stop_probability` use uniformization (`ptd_probability_distribution_context_*`), which doesn't need the reward compute graph. No Stage A2 read/write.
- **JAX FFI PMF path.** `Graph.pmf_from_graph` → `compute_pmf_ffi` → `ComputePmfFfiImpl` builds afresh `phasic::Graph` from JSON and runs uniformization. No reward compute graph involved.
- **Joint-prob / sojourn FFI path.** `Graph.pmf_from_graph_joint_index` → `compute_sojourn_times_ffi` calls `ptd_expected_sojourn_time_subset`, which does *not* go through`ptd_precompute_reward_compute_graph`. No Stage A2.
- **Daisy-chain.** `DaisyChainJointProbsFfiImpl` does `update_ipv` / `update_weights` / `stop_probability` per epoch — uniformization only, no Stage A2.
- **`pmf_and_moments_from_graph` (FFI or pybind11 fallback).** The "moments" half routes through `ptd_expected_waiting_time` → so Stage A2 *is* hit per worker thread, but each FFI worker has itsown `phasic::Graph` (Stage A1 builder cache), so each thread's first call pays the Stage A2 read.
- **User-supplied builders.** `Graph.pmf_from_graph_parameterized` and `Graph.pmf_from_cpp` build a Graph per call and only ask for PMF, so no Stage A2.
- **SVGD.** Inherits whichever model factory it dispatched to — most SVGD configurations route through PMF/sojourn paths, not moments. The exception is reward-regularized SVGD via `pmf_and_moments_from_graph`, which does hit Stage A2 inside its moments computation.

**Summary.** Stage A2 disk caching is automatic only for the moment/waiting-time family. PDF/PMF/sojourn/joint-prob/daisy-chain paths — which are the bulk of JAX-traceable workloads — never touch it. So calling it "the universal C-side cache" would be wrong; it's specifically a cache for the symbolic elimination used in moment computation.

### Does `Graph.expectation()` produce/use the disk cache by default?

Yes. `Graph.expectation()` on a **parameterised** graph populates and reads `~/.phasic_cache/parameterized_reward_compute/` automatically by default.

Call chain:

1. `Graph.expectation(rewards)` (`__init__.py:2485`) → `super().expectation(rewards)` (the trace-based branch is commented out at `:2523+`).
2. Pybind11 `expectation` → C `_moments` helper (`phasic_pybind.cpp:212`) → `ptd_expected_waiting_time` (`api/c/phasic.h:324`).
3. `ptd_expected_waiting_time` triggers `ptd_precompute_reward_compute_graph` (`phasic.c:1895+`), which:
    - **Cache miss:** runs the O(n³) symbolic elimination, then call `ptd_save_parameterized_reward_compute_graph` (`phasic.c:1949`) → writes `~/.phasic_cache/parameterized_reward_compute/<hash>.bin`.
    - **Cache hit:** calls `ptd_load_parameterized_reward_compute_graph` (`phasic.c:1913`) and skips elimination.

**Defaults and gating**

- **Default ON.** The cache is enabled unconditionally. There is no opt-in flag.
- **Opt-out:** set `PHASIC_DISABLE_CACHE=1` in the environment — that's the only switch (`phasic.c:1908`, `ptd_pcg_cache_disabled`). Programmatic equivalent: `phasic.cache.is_cache_disabled()` reflects the same env var.
- **Within a process,** Stage A0 (in-memory `parameterized_reward_compute_graph` field on the `ptd_graph`) means subsequent `expectation()` calls on the same graph instance skip the disk lookup entirely — the disk cache only matters at process start, or for fresh graphs sharing the same content hash.

**Caveats**

- **Non-parameterised graphs** take a different branch (`reward_compute_graph`, no Stage A2 cache). Plain constant-weight graphs do not produce disk cache entries.
- **Discrete graphs** (`self.is_discrete`) route through `super().expectation_discrete` (`__init__.py:2541`); for parameterised discrete graphs that path also reaches `ptd_expected_waiting_time`, so Stage A2 is hit there too.

In short: calling `Graph.expectation()` on a parameterised graph silently produces a `.bin` file under `~/.phasic_cache/parameterized_reward_compute/` the first time the graph's content hash is seen, and reuses that file on subsequent runs — unless `PHASIC_DISABLE_CACHE=1` is set.





----



❯ The most important thing is to maintain the core functionality: disk-cashing of the O(n^3)       
    computation. The ambition and long term goal is to support hierarchical SCC decomposition,     
    distributed computation of SCCs and disc-cashing of the assembled result (as is almost         
    implemented with the python Elimination trace path). Would it be a way forward to keep the     
    parameterized reward compute graph as the default path for SVGD and in parallel develop the    
    missing/defunct parts of the Elimination trace path?                                           
  ⎿  ⧉ Selected 6 lines from notes.md in Visual Studio Code

⏺ Yes, that's a sensible split — but only if you're clear-eyed about what each path actually buys
  you, because they're not interchangeable today.

  Where the two paths actually overlap and where they don't

  The parameterized reward compute graph (Stage A2) and the Python EliminationTrace are not drop-in
   replacements for each other — they solve different problems and feed different consumers:

  - Stage A2 caches the symbolic elimination for moment/waiting-time computation. Its consumer is
  ptd_expected_waiting_time, which is reached only by the moments family. SVGD uses it only when
  reward regularisation is on (via pmf_and_moments_from_graph's moments half). The bulk of SVGD
  traffic (PDF/PMF, sojourn-times, daisy-chain) bypasses it entirely and runs uniformization on a
  freshly built phasic::Graph per worker.
  - EliminationTrace records the full elimination as a linear op-list that can be replayed in JAX
  (evaluate_trace_jax) or instantiated back into a concrete graph. It was designed for exactly the
  workload you're describing — record once, evaluate many times under different θ — and the
  SCC-decomposition + stitching logic in hierarchical_trace_cache.py is genuinely the right shape
  for distributed computation. It's also the only path that ever produced an "assembled SCC result"
   cache key.

  So "keep Stage A2 as the SVGD default" is fine for moments/expectation, but SVGD's hot loop today
   is mostly not moments — it's PMF and sojourn-times. Those don't currently disk-cache anything
  substantial: they cache the graph topology (Stage A1, in-memory per worker) but redo the work
  each time θ changes, because the work is θ-dependent. There is no symbolic-elimination shortcut
  on those paths.

  The honest picture of "almost implemented"

  The Python EliminationTrace pipeline has the SCC decomposition, the per-SCC hashing, the parallel
   compute, and the trace stitching written. What's missing or defunct:

  - The C-side reader/writer (ptd_load/save_trace_from_cache) is wired up but its only intended
  caller is commented out, so the C path can't produce the on-disk JSON. Python is the only
  producer.
  - The trace path is no longer connected to any public entry point except the deprecated
  compute_trace. Graph.moments/expectation/variance were explicitly routed away from it (commented
  out at __init__.py:2461+, :2523+, :2585+) because the trace branch was redundant work for the
  moments use case.
  - The trace's main downstream consumer was evaluate_trace_jax and trace_to_log_likelihood,
  neither of which is on any current SVGD model factory. SVGD's JAX-traceable factories all go
  through FFI handlers that take a JSON graph, not a trace.
  - The "memory blow-up on large graphs" comment in phasic.c:4213 is a real signal, not just
  dead-code commentary. The Python trace pipeline allocated a big op-list per graph and was killed
  in production.

  So "almost implemented" is fair structurally, but the gap-to-production is more than wiring —
  it's also performance work and a consumer-side rewrite, because the FFI handlers don't currently
  know what a trace is.

  Is the parallel-track strategy a good idea?

  Yes, with two important framings:

  1. Stop treating Stage A2 as a stopgap. It's not a degraded version of the trace path; it's the
  right tool for moments. Even after the trace path is fully revived, Stage A2 should keep serving
  moments/expectation — moving moments onto the trace path would be net-worse (extra serialisation,
   extra Python boundary, no win on numerics). So "Stage A2 is default for SVGD-with-rewards"
  should be the long-term answer, not a temporary one.
  2. Be specific about what the trace path is for. The real goal you described — "hierarchical SCC
  decomposition + distributed SCC compute + disk-cache the assembled result" — pays off only when
  (a) the same SCCs recur across many graphs you actually build, and (b) the per-SCC work is
  expensive enough to dominate I/O. That's a population-genetics workflow shape (lots of related
  coalescent models), not a generic SVGD shape. So scoping the trace revival as "the path for very
  large structured graphs where SCC reuse is real," not "the path for all SVGD," will make the work
   tractable and the wins measurable.

  Concrete suggestions for sequencing

  If you want to do this without losing ground:

  1. Lock down Stage A2 as the production path for moments/expectation/variance/covariance.
  Document the env var, write an integration test that asserts a .bin lands in the cache dir on
  first call and is reused on the second, and consider exposing cache stats on the Graph object
  (hits/misses) so users can see it working.
  2. Decide what "the trace path's job" is. Pick one concrete use case (e.g. "1000-vertex
  coalescent + SVGD with rewards, where SCC reuse across runs gives ≥5× wallclock improvement").
  That picks the consumer (probably a new compute_pmf_and_moments_from_trace_ffi) and the success
  metric.
  3. Fix the C producer or commit to Python-only production. The current half-state — C reader
  exists, C producer commented out, Python writes — is the worst of both worlds because you can't
  tell from a .json on disk which side wrote it. Either revive ptd_record_elimination_trace (with a
   memory budget) so the C and Python sides agree, or delete the C reader/writer and own that the
  trace cache is a Python artifact.
  4. Address the memory issue before re-enabling. Whatever caused the original blow-up will recur.
  Likely candidates: storing intermediate symbolic results that are never consumed, or recording
  op-lists for every elimination step rather than collapsing them. Worth profiling on a
  known-painful graph (the 67-vertex case the docs mention, then a 500-vertex one) before wiring it
   back to a public method.
  5. Keep them disjoint, not parallel. Don't try to make the trace path serve moments — Stage A2
  wins there. Don't try to make Stage A2 serve PMF/sojourn — it doesn't have the right shape. Two
  narrow paths each owning their workload is more maintainable than one path that mostly works for
  both.

  The main risk in the parallel-track plan is scope drift: "let's also make the trace path do
  moments" or "let's also make Stage A2 distributable." Both are tempting and both are traps.
  Holding the line that Stage A2 is for moments and the trace path is for SCC-reusable PMF
  computation keeps the work bounded.



  DataPrior should not be constructed when graph is joint-prob (vanilla or daisy)

Then if a (non joint-prob) graph does not use reqularization, could it use pmf_from_graph rather than pmf_and_moments_from_graph to avoid the O(n^3) step altogether?