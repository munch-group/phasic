# Differential audit — findings log

Running log of the secure differential process (see `audit-situation-map.md` for the atlas,
baseline, and gap list). Each finding is **verified by execution against an independent
reference** (closed form / cross-backend / the `3082ebc6` baseline), scoped via the atlas, and
dated (pre-existing vs refactoring-introduced) before any fix decision.

Baseline built and verified: `3082ebc6` (parent of first refactor commit `9b8bc3f6`) builds
under the current pixi env and computes `moments(2) = [0.325, 0.16125]` correctly.

---

## F-001 — discrete-PMF `normalize()` divergence in `pmf_from_graph_parameterized`

**Verdict: REAL bug, but LATENT and PRE-EXISTING. Not refactoring-introduced. Do not fix in
isolation — record to fix together with the upstream break that masks it.**

### What the atlas flagged
Three discrete-PMF compute paths, one of which normalizes:
- `pmf_from_graph(discrete=True)` → FFI `ComputePmfFfiImpl`: `dph_pmf`, no normalize.
- `pmf_from_cpp(discrete=True)` → ctypes-JIT: `dph_pmf`, no normalize (the fix, `__init__.py`
  "NO normalize() here").
- `pmf_from_graph_parameterized(discrete=True)` → ctypes-JIT: **`g.normalize(); g.dph_pmf(...)`**
  (`__init__.py`, wrapper string, comment "Normalize for discrete mode (required for DPH)").

In a DPH the edge weights ARE per-step transition probabilities and the deficit (1 − row sum)
is the implicit stay-in-place probability, so the continuous `normalize()` (which rescales each
vertex's out-edges to sum to 1) **collapses the chain to a deterministic walk**. This is the
exact class of bug already fixed in `pmf_from_cpp`.

### Verified by execution (closed-form oracle)
Forced 2-phase DPH `s→v3-(p)->v2-(p)->v1`, each transient vertex one out-edge of weight
`p = 0.3` (row sum 0.3 < 1). Correct DPH: jump count ~ NegBinomial(2, p),
`P(T=n) = (n-1)p²(1-p)^(n-2)`. Normalize would rescale the single out-edge to 1.0 → `P(T=2)=1`.

```
closed form NegBinomial(2,0.3) = [0.09, 0.126, 0.12348, 0.07412]
pmf_from_graph (FFI)                -> [0.09, 0.126, 0.12348, 0.07412]   CORRECT
pmf_from_cpp (ctypes JIT)           -> [0.09, 0.126, 0.12348, 0.07412]   CORRECT
pmf_from_graph_parameterized (JIT)  -> RAISED AttributeError
```

### Why it's LATENT (verified, not assumed)
`pmf_from_graph_parameterized` raises **before** the ctypes wrapper (with the `normalize()`)
ever runs:
```
result_shape_dtypes = jax.ShapeDtypeStruct(times.shape, jnp.float32)
AttributeError: 'NoneType' object has no attribute 'ShapeDtypeStruct'
```
Module-level `jax` is `None` (lazily imported) and the code requests `jnp.float32` where the FFI
requires F64. This is the original hand-off's "bug 5". So the buggy `normalize()` is **dead code
today** — no user can reach it — and the divergence is not observable by any current caller.

### Dated: PRE-EXISTING (not refactoring-introduced)
The `normalize(); dph_pmf` block is **byte-identical** on the baseline `3082ebc6` (which predates
the entire refactoring) and the current tree. The refactoring neither introduced nor touched it.

### Scope & fix
- **Scope:** one path (`pmf_from_graph_parameterized`), one wrapper string. The other two
  discrete-PMF paths are correct. No cross-path divergence is *observable* while bug 5 stands.
- **Fix (deferred):** remove the `g.normalize()` call, mirroring the `pmf_from_cpp` fix — **but
  only once bug 5 (the `jax`-is-None / `float32` break) is fixed**, since it is otherwise dead.
  Fixing dead code in isolation adds risk (a rebuild, a diff) with no observable benefit. Bundle
  it with any `pmf_from_graph_parameterized` revival.
- **Guard needed at that time:** a discrete cross-path gate (`pmf_from_graph_parameterized` ==
  `pmf_from_cpp` == FFI, against NegBinomial) on a row-sum≠1 graph. No existing gate covers this
  (gap §4.2 / §4.5: ctypes-JIT and discrete cross-path are ungated).

### Process note
Reading the code alone would have yielded a false "cross-path divergence bug (HIGH)". Executing
the differential downgraded it correctly to **latent + pre-existing** — the calibrated verdict.
This is the method working: reproduce → find it doesn't run → find *why* → scope → date against
baseline → defer the fix.

---

## F-002 — expected_sojourn_time(subset) adjoint (c340bedc) — CLEARED, no bug

**Verdict: the #1 risk item is CORRECT.** The reverse-mode adjoint that replaced the O(n*k)
forward dense replay agrees with every independent reference to machine precision, including in
the large-k / large-graph / cyclic regime the situation map flagged as untested.

### What was checked (all executed)
Four independent references for ptd_expected_sojourn_time_subset (src/c/phasic.c:10181):
- legacy forward (PHASIC_SOJOURN_FORWARD=1, same trace, dense replay -- the claimed-equal ref);
- independent dense oracle sojourn = alpha @ (-S)^{-1} (validated to 2.8e-16 vs the library;
  does not touch the elimination-trace machinery at all);
- pre-refactor baseline 3082ebc6 (forward-only -- the temporal reference);
- cross-checked on branching, chain, coalescent (n=4..7), and a birth-death graph WITH cycles.

### Results
| regime | adjoint vs forward | adjoint vs oracle | adjoint vs baseline |
|---|---|---|---|
| small (branching / chain / coalescent, <=16 v) | 0 .. 2e-16 | <=1.2e-15 | bit-identical |
| large + cyclic (birth-death, 50 .. 1200 v, k=1200) | <=8e-15 | <=7e-15 | -- |

The claim (~1e-15 summation-order rounding) holds; the adjoint tracks an oracle independent of
the trace to machine precision even at 1200 vertices with cycles and a full-size subset (k=1200,
where the forward computes 1200 columns and the adjoint one reverse pass). Temporal diff:
bit-identical to the baseline on branching and coalescent.

### Not triggered (and why it's not needed)
The genuine forward-path OOM (the ~1.5TB alloc the adjoint exists to avoid) needs n ~ 100k+;
not reproduced. It does not need to be: the adjoint's O(n) memory is a structural property of a
single reverse pass over one length-n vector (verified by reading, src/c/phasic.c:10215-10230),
and its CORRECTNESS is already confirmed against an independent oracle at 1200 vertices. Triggering
a 1.5TB alloc would test the forward path's failure mode, not the adjoint's correctness.

### Process note
Highest-risk change, retired with evidence rather than left as a worry. The independent dense
oracle was the load-bearing reference (the forward path is not independent -- it shares the trace);
it was re-validated (2.8e-16) before use. A permanent gate for this belongs in the gap list
(§4.7: sojourn has no cross-path equivalence gate) -- adjoint == forward-flag == dense-oracle across
small + large + cyclic.

---

## F-003 — WS-C "pure verbatim relocation" (5 module extractions) — CLEARED, no bug

**Verdict: the refactoring's central claim holds. All 5 WS-C extractions are behaviour-
preserving; no accidental change.** These commits each claim "extracted verbatim, pure
relocation" (stage3-execution-handoff.md:29) — the classic place a "behaviour-preserving"
refactor accidentally changes behaviour.

### Commits
de774f88 plotting, 1cd75c2d cache_transfer, 0d9ab967 cache_mgmt,
a6df2be9 reward_validation, 98deb4da serialize/from_serialized (-> _graph_*.py).

### Checks (proportionate to compute risk)
**serialize (crown jewel: every FFI path does serialize()->JSON->GraphBuilder)** — BEHAVIOURAL
differential, baseline `3082ebc6` vs current, byte-for-byte JSON across a 6-graph zoo (linear,
log, formula, cyclic, coalescent, joint_prob): **0 differences**; from_serialized roundtrip
behaviour identical on both (incl. joint_prob raising the same RuntimeError — a pre-existing
limit, not a WS-C change).

**reward_validation (on the compute path; moments/reward_transform call `_validate_rewards`)** —
BEHAVIOURAL differential of `absorbing_state_rewards`, `_starting/_absorbing_vertex_indices`,
`_validate_rewards` across valid / wrong-length / partial-coverage / all-zero / negative rewards:
**0 differences** (return values AND raised errors identical).

**All 5 — TEXTUAL verbatim:** every non-trivial deleted body line reappears as an added line.
Unaccounted lines: **0** for plotting/cache_transfer/cache_mgmt/reward_validation; **1** for
serialize (`@classmethod`) — explained relocation mechanics: `from_serialized` moves to a plain
module function and is re-bound `from_serialized = classmethod(_graph_serialize.from_serialized)`
(`__init__.py:3306`); verified a classmethod at runtime on BOTH baseline and current.

**cache_transfer / cache_mgmt / plotting — SMOKE (reference resolution):** the one way a
textually-verbatim move still breaks is a name that resolved in the class namespace but not the
new module. All six methods resolve cleanly (pull_cache/plot/plot_scc_decomp ran; the others
raised legitimate ARGUMENT/STATE errors — RuntimeError/TypeError from their real logic — not
NameError/ImportError/AttributeError).

### Coverage honesty
serialize + reward_validation got a full baseline-vs-current behavioural differential (the
compute-critical two). cache/plotting got textual-verbatim + reference-resolution smoke, not a
full stateful behavioural differential — proportionate to their I/O/visual (non-compute) risk.

### Process note
Third finding; still no refactoring-introduced bug (F-001 pre-existing, F-002 correct, F-003
verbatim). A good early signal about the refactoring's quality — and the differential is now
demonstrated on temporal (F-002/F-003) and cross-backend (F-001) axes both.

---

## F-004 — trace-vs-FFI gate (gap #1) built; latent use_log footgun found

**Verdict: cross-backend consistency is CLEAN; one LATENT API footgun documented. Permanent
gate added: tests/pytest/test_gate_trace_ffi_equivalence.py (11 tests).**

### The gate (the deliverable)
Closes the single largest cross-path hole (situation map gap #1): the TRACE replay and the
XLA-FFI path were never pinned to each other (G1 = FFI-vs-pybind, G5 = trace-vs-pybind, neither
trace-vs-FFI). The new gate compares TRACE == pybind == FFI for mean, var, and the
matched-granularity pdf, plus discrete-PMF (pybind vs FFI), across the previously-ungated
regimes: **log weight mode, cyclic graphs, discrete DPH, a coalescent(5) graph, and three theta
magnitudes** (incl. mixed-scale 1e-2). FFI lowering is PROVEN via `_gate_backend.assert_ffi_target`
(no pure_callback fallback). Result: all participating backends agree **bit-for-bit** (worst
3e-16). Cyclic + formula skip the trace (the trace engine refuses them: RuntimeError /
NotImplementedError) and compare pybind-vs-FFI only.

### Two methodology traps this exposed (both real, both handled)
1. **Continuous-pdf comparison is confounded by granularity.** The FFI path hardcodes
   granularity=0 (auto); comparing it to a trace/pybind pdf at granularity=1000 shows a spurious
   ~1.7e-3 "divergence". At MATCHED granularity (0) all three agree bit-for-bit. The gate matches
   granularity. (A first draft of the gate did not, and "found" the spurious divergence.)
2. **The use_log footgun (F-004 proper).** `instantiate_from_trace` / `evaluate_trace_jax`
   default `use_log=False`. The trace is recorded with unit weights; log semantics
   (weight = prod(c_i*theta_i)) are applied at REPLAY via use_log. Replaying a log-mode graph's
   trace WITHOUT use_log silently yields the LINEAR answer: E[T] 0.325 instead of 0.333, and the
   pdf diverges and EXPLODES with theta (measured 0.70 -> 1168 -> 3.9e16 rel err at theta
   [1,2]/[1e-2,2]/[5,3]). Verified root cause: the rebuilt graph carried linear rates (8,5) not
   log rates (12,4); passing use_log=True made all backends agree exactly.

### Reachability of the footgun: LATENT (verified)
No production user-facing quantity path replays the trace for pdf/moments:
- `evaluate_trace_jax` has **zero** production callers (grep of src/phasic/, excluding its own
  def/docstring).
- The `instantiate_from_trace`-based pdf/moments code in `__init__.py` (~:2017-2144) is
  **commented out** (dead).
- `pmf_from_graph` for {linear,log,formula} with `use_ffi=False` **raises** PTDBackendError
  (`__init__.py:3591`) -- there is no trace-replay production route.
- Production pdf/moments/sojourn all go through pybind / FFI, which honor log correctly (the
  gate confirms this: log pmf/moments agree across pybind and FFI).
So the footgun is a real trap in the PUBLIC trace primitives, not reachable via any current
compute entry point. **Pre-existing** (the use_log-default-False API predates the refactoring;
reasoned from the API shape, not baseline-diffed -- the reachability fact is decisive on its own).

### Why it still matters (forward-looking)
The deferred exact-AD gradient plan (audit-fd-step-remediation-plan.md, Tier 1) proposes using
`evaluate_trace_jax` as the differentiable backward. On a LOG-mode graph that path MUST thread
use_log or it will silently differentiate the LINEAR function. Recorded here so that work does
not step on it. The gate's `test_trace_requires_use_log` pins the footgun and would become a
forcing function if use_log is ever auto-detected.

### Separate LEAD (not a cross-path bug; note for later)
Auto-granularity (granularity=0) gives a pdf with a FIXED ~6.5e-3 error that does NOT converge
to the closed form, while explicit granularity>=1000 converges first-order (gran=1e5 -> 5.6e-5).
All backends agree at gran=0, so this is not a cross-path divergence -- it is a question of
whether the auto-granularity DEFAULT is accurate enough, or is mis-derived. Worth a separate
look (could be intended coarseness or a real accuracy bug); out of scope for this gate.

---

## F-005 — auto-granularity `max_rate = 512` floor — NOT a bug, pre-existing

**Verdict: the auto-granularity lead is a pre-existing undocumented quirk, not a correctness
bug and not refactoring-introduced. Closed.**

The lead (from F-004): granularity=0 (auto) gives a pdf with a fixed ~6.5e-3 error that does not
converge, while explicit granularity>=1000 converges. Root cause found by instrumentation:

`ptd_probability_distribution_context_create` (src/c/phasic.c:11363) initializes
`double max_rate = 512;` and the loop only raises it if a vertex's total outgoing rate exceeds
512. So for any graph with max rate < 512 (nearly all small graphs) auto-granularity =
`max(512, actual)*2 = 1024` (min-1000 floor never binds). Debug log confirms:
`Auto-selected granularity: 1024 (max_rate=512.00)` on a graph whose real max rate is 8.

Consequences, all benign:
- gran=0 uses granularity **1024**, not 1000 -- which is why gran=0 != explicit gran=1000
  (they differ by ~1e-3 on the oscillating uniformization curve). My "doesn't converge" reading
  was wrong: gran=0 is *fixed* at a coarse 1024, so of course it doesn't improve without an
  explicit higher granularity.
- The pdf at gran=1024 is a valid uniformization approximation (~6.5e-3 rel err) -- a coarse
  DEFAULT, not a wrong answer.

**Pre-existing, not refactoring-introduced:** `max_rate = 512` is byte-identical on baseline
`3082ebc6` and traces to commit `8710f715` ("renamed project to phasic") -- inherited from the
original codebase, long before this refactoring.

Also observed (inherent, not a bug): the uniformization pdf oscillates non-monotonically by
~1e-3 as granularity changes by 1 (gran 999/1000/1001 -> 1.7748/1.7712/1.7727) -- a discretization
artifact of evaluating a discrete-step pdf at a continuous t. Granularity is fixed per call, so
this does not affect gradients; it does mean the default-granularity pdf is only ~1e-3 accurate.

**Worth a doc note, not a fix:** 512 is an undocumented magic number and semantically odd
(initializing a *max* accumulator to 512 rather than 0 with an explicit `fmax(..., 1024)` floor);
users needing better than ~6.5e-3 pdf accuracy must pass an explicit granularity. Neither is a
correctness defect.

---

## F-006 — strict-xfail map INTACT — no silent unification, no regression

**Verdict: all 15 known cross-path divergences are present and unchanged. 0 XPASS (no refactor
silently unified a pair), 0 FAIL (no known-agreeing pair regressed). The refactoring preserved
every pinned cross-path relationship.**

Ran the full `equivalence`-marked gate suite (with PHASIC_SOURCE_DIR): **82 passed, 1 skipped,
15 xfailed, 0 xpassed, 0 failed**. Every entry of the canonical strict-xfail map is still
xfailing exactly as documented on master:

| pin | divergence (still present) |
|---|---|
| Q1 (×6) | Python trace refuses cyclic + formula (trace_elimination.py:442/845); C engine handles them |
| Q5 | `Graph.moments(discrete=True)` -> missing `moments_discrete` pybind binding -> AttributeError |
| Q6a | `from_serialized` DROPS `constant_edges` while GraphBuilder rebuilds them |
| Q6b | `from_serialized` merges duplicate-state vertices; GraphBuilder preserves identity |
| Q7.1 | **FFI combined PMF uses the UNTRANSFORMED graph; pybind uses the reward-transformed graph -> rewards!=None PMFs DIFFER** (a real cross-path correctness divergence, pinned) |
| Q-G4-1/2 | formula dead-select-arm / pow-negative-base: eval-tape oracle vs C VM disagree |
| Q10 | `SCCGraph::sccs_in_topo_order` trusts a stored Tarjan order that is NOT topological |
| #9 | `_expected_scc_filenames` passes a tuple to `compute_graph_hash` -> silently returns [] |
| Q11a | conditioned sampler: `rand()` (no seed) vs `rand_r()` (seeded) -> no per-draw identity |

**Significance.** An XPASS here would mean a refactor silently changed a cross-path relationship
that was supposed to stay divergent (needs investigation); a FAIL would mean a known-agreeing pair
now diverges (a regression). Neither occurred. Combined with the new trace-vs-FFI gate (F-004,
which extended agreement to log/cyclic/discrete/larger/multi-theta), the evidence is consistent:
the refactoring preserved cross-path behaviour.

**Caveat (honest):** these gates run on TINY fixtures (1-7 vertices, 1-2 params), so "82 passed"
proves the known-agreeing pairs still agree ON THOSE FIXTURES, not universally -- which is exactly
why F-004 extended the regimes. The strict-xfail check is a necessary intactness check, not a
sufficiency proof.

**Note for Stage-3 (not this audit's job):** several of these xfails are REAL deferred bugs, not
mere engine-capability gaps -- Q7.1 (reward-PMF cross-path divergence), Q5 (discrete moments
crash), Q6a/b (serialize round-trip data loss), Q10/#9 (SCC ordering). They are documented,
pinned, and owned by Stage-3.

---

## F-007 — Q7.1 reward-PMF cross-path divergence — REAL, opt-in-only, PRE-EXISTING, Stage-3-owned

**Verdict: a genuine reachable correctness bug -- `pmf_and_moments_from_graph(..., use_ffi=True)`
with rewards returns the UNTRANSFORMED PMF instead of the reward-transformed one. But it is
reachable only by an explicit `use_ffi=True` opt-in, no production path hits it, it is
PRE-EXISTING (not refactoring-introduced), and it is already documented and owned by Stage-3
(the Q7.1 strict-xfail). Investigated, not fixed.**

### The divergence (docstring is decisive on which side is correct)
`pmf_and_moments_from_graph_multivariate` docstring: rewards "define the marginal distribution".
So the PMF must be the reward-TRANSFORMED distribution. Two backends:
- pybind `compute_pmf_and_moments` (graph_builder.cpp:741): `g.reward_transform(rewards)` then pdf
  -> **CORRECT**.
- FFI `ComputePmfAndMomentsFfiImpl` (graph_builder_ffi.cpp): pdf on the untransformed graph
  -> **WRONG** (only the PMF; the reward-MOMENTS agree to 1 ulp -- expected_waiting_time(rewards)
  is math-equivalent to reward_transform-then-moments).

### Verified by execution against an independent oracle
Oracle = `g.reward_transform(REW).pdf(t)` (correct) vs `g.pdf(t)` (untransformed), matched
granularity:
```
reward-transformed pdf (CORRECT) = [1.58859, 0.72769, 0.20154]
untransformed pdf     (WRONG)    = [1.77206, 0.55463, 0.08481]   (differ by 1.38 rel)

pmf_and_moments_from_graph(use_ffi=False)  -> TRANSFORMED (correct)   [the DEFAULT]
pmf_and_moments_from_graph(use_ffi=True )  -> UNTRANSFORMED (WRONG)
multivariate (per-feature rewards)         -> TRANSFORMED (correct)
```

### Scope & reachability (the key result)
- **Default is correct.** `use_ffi=False` is the method default; the resolution
  (`__init__.py:6774`: `if not use_ffi: use_ffi=False; else: use_ffi=config._use_ffi`) keeps FFI
  OFF unless the caller explicitly passes `use_ffi=True`. So `config._use_ffi=True` never forces
  the wrong path by itself.
- **No production caller opts in.** Every svgd dispatch (`__init__.py:5793,5800,5806,6043,6048,
  6053`) uses the default; the multivariate path delegates to the default 1D pybind path
  (confirmed CORRECT above). The only `use_ffi=True` in src/phasic is a comment (`:7077`).
- So the wrong PMF is reachable ONLY by a user who explicitly writes
  `pmf_and_moments_from_graph(g, rewards=..., use_ffi=True)`.

### Dated: PRE-EXISTING
The baseline `3082ebc6` `ComputePmfAndMomentsFfiImpl` has **zero** `reward_transform` calls too
-- the FFI handler computed the PMF on the untransformed graph before the refactoring. Not
introduced by it.

### The fix (for Stage-3, one handler)
In `ComputePmfAndMomentsFfiImpl`, when rewards != None, compute the PMF on
`g.reward_transform(rewards)` -- mirroring the pybind path (graph_builder.cpp:741) and the
multivariate FFI handler (graph_builder_ffi.cpp:675, which already does this). The moments path
is already correct and must be left as-is. This is exactly the "Stage-3 Q7 must unify which
graph the combined PMF uses" pin; flipping/removing the Q7.1 strict-xfail is the forcing function.

**Not fixed here:** pre-existing (out of the refactoring-bug scope), pinned as a documented
strict-xfail, and owned by Stage-3. Fixing it in isolation would silently unpin a Stage-3 gate.
