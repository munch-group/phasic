# phasic — open items register

**The one place that tracks what is still open.** Plans, findings and
review files remain the records of design and evidence, but their status
lines are not maintained; this file is. Built 2026-11-25 by reading every
markdown file in the repository root, CLAUDE.md, the xfail markers in
[tests/pytest/](tests/pytest/), and the assistant's memory notes, then re-checking a
sample of items against current master (`a857b112`).

## How to use and maintain it

- One row per item. IDs never change and are never reused.
- **Check** column: **V** = reproduced or confirmed against current
  master on 2026-11-25; **S** = taken from the source record, not
  re-checked (may have been fixed since without the record saying so).
- New defects, follow-ups and notebook edits go here first. A plan may
  describe them in detail but links here for status.
- When an item closes, move its row to "Closed" with the commit hash.
- The August atlas (14 files, `plans-and-findings/exact-fd-atlas-*.md` and `plans-and-findings/plan-feasibility-*.md`, moved from `atlas/` 2026-11-26) was reviewed 2026-11-26. Its seven
  atlas files fed the exact-gradient master plan, and its six feasibility
  studies became Batches A-H and Deferred 1-4. They are history; their
  still-open findings are rows OI-079, OI-085, OI-086, OI-087 (and OI-029,
  OI-031, OI-034, OI-065, already listed).
- [PROJECT-OVERVIEW.md](PROJECT-OVERVIEW.md) stays the narrative and the label decoder; its
  "open items" list (section 1c) is superseded by this file.


---

## Priorities toward goal 1: a stable refactored core engine with the full API in C++ and Python

Set 2026-11-26 at your request. **Plan A Batch 0 (census, step sizes, master baseline) DONE 2026-11-27** on branch `fix/fd-domain-steps`: [plans-and-findings/fd-domain-steps-findings.md](plans-and-findings/fd-domain-steps-findings.md); step size decided 2026-11-27: s = 3e-5 for every family (findings §4.3); all Batch 0 amendments approved and folded into the plan (v2.10). **Batch 1 (helper, builders, crash fallback) DONE 2026-11-27** (findings §10: Opus review signed off, gate passed); **Batch 2 (Graph.svgd and epoch_model declare domains, containment checks, D-NZ docs) DONE 2026-11-27** (findings §11: gate passed after one justified test edit, Opus review signed off); OI-124 drafted as plan §12 amendment A-124, AWAITING YOUR APPROVAL (not built). **Plans written and SIGNED OFF 2026-11-26** (Plan A v2.9 after thirteen
adversarial reviews, the refactor master plan v2.16 after twenty; each phase
still gets its own reviewed sub-plan; decision D19 made: `load_cpp_builder` disabled from phase 0d until phase 12): [plans-and-findings/fd-domain-steps-plan.md](plans-and-findings/fd-domain-steps-plan.md) (Plan A,
done first) and [plans-and-findings/core-engine-ssot-plan.md](plans-and-findings/core-engine-ssot-plan.md) (the
refactor master plan; its phase 0 is Tiers 1-3 below). Scope decided
(OI-009): C/C++ single source for all core computations including the
joint-probability family and a C++ index codec; inference layer stays
Python. Goal 1 is the July refactor plan
([plans-and-findings/stage3-refactor-plan.md](plans-and-findings/stage3-refactor-plan.md)): C++ is a product, each
capability gets one source of truth in the C++ facade, Python delegates
("complete-then-promote": reach parity, bind and gate, delegate and delete).
Done so far: safety-net gates, dead-code removal, part of splitting the
`Graph` class, `discretize()` moved to C++, C++ method-name parity ports
(July), and three of the plan's behaviour fixes (Q5, Q6, Q7.1).

Three rules decided the sorting:

1. **The refactor is checked by bit-identity gates** (old path == new path,
   bit for bit). Anything that makes results non-reproducible must be fixed
   first, or the gates cannot tell refactor drift from existing noise. For
   the same reason, work that deliberately changes numbers (a new forward
   kernel, exact PDF gradients) comes after the refactor, not before.
2. **A port copies behaviour.** Where the right behaviour of a core function
   is still undecided, decide it before that function becomes the C++
   source of truth; otherwise the refactor freezes the wrong contract in two
   languages.
3. **The inference layer (SVGD, MCMC, priors, plots) sits above the core
   engine** and is not part of the C++ API. Its defects can wait, unless you
   need a result from it now.

### Tier 0: decide before anything else

- **OI-009 (new): scope of "full API in C++", and a refresh of the July plan.**
  The plan is four months old: line numbers are stale, and it predates the
  exact-gradient C code, the C++ `discretize`, bug 4's capped discrete step
  and the new graph cache. It also contradicts your goal in places: it keeps
  `StateIndexer` / `IndexCodec` Python-only, deletes the C++ `profile` port,
  and has two opposite answers on which weight-formula compiler wins (its §3a
  and §3b disagree). Decide which parts must exist in C++:
  - graph building with a state indexer;
  - `joint_prob_graph` and the epoch builders;
  - `serialize`;
  - weight formulas;
  - `profile`;
  - the parameterized JAX path.

  Then re-baseline the plan's inventory and sequence against current master.
  OI-077 (C++ StateIndexer, C++ `joint_prob_graph`, theta layout) belongs
  here.
- OI-007 becomes "continue", under the scope set in OI-009.
- OI-001 (approving the found-defects fixes) is answered by this sorting
  if you accept it: the defects in Tiers 2-3 now, the rest later.

### Tier 1: safety net (before the refactor; mostly small)

| ID | Why first |
|---|---|
| OI-062 | CI runs only the equivalence gates. A refactor across C, C++, pybind and Python needs the broad suite run somewhere other than by hand. |
| OI-067 (new) | The suite-gate tooling used for every merge since bug 1 ([gate2.sh](.git/claude-scratch/gate2.sh), [gate_compare.py](.git/claude-scratch/gate_compare.py), [runcap.py](.git/claude-scratch/runcap.py)) lives in [.git/claude-scratch/](.git/claude-scratch/). It is unversioned, so it can be lost and is not in CI. Move it into [scripts/](scripts/). |
| OI-060, OI-061 | Fix the 4 always-failing plot tests and seed the 3 flaky tests, so that "no new failures" is unambiguous at every refactor step. |
| OI-113, OI-111, OI-110 | Commit the file move. Prune the 19 worktrees and 16 stale branches, and push, so the refactor starts from a clean, pushed master. |

### Tier 2: reproducibility (before the refactor)

| ID | Why first |
|---|---|
| OI-023 | A constant model's expectation()/variance() differ in the last bit between instances. That is exactly what a bit-identity gate compares. **Likely root cause (refactor review 3, 2026-11-26):** constant-rate elimination, the reward transform and `_dyn` order edges by memory address (`arr_c_cmp`, [src/c/phasic.c](src/c/phasic.c#L6088)); 6 instances gave 3 distinct bit patterns ([experiments/core_ssot/review3/probe_determinism.py](experiments/core_ssot/review3/probe_determinism.py)). Fix approved as a numeric exception (refactor plan R14, phase 0b). |
| OI-022, OI-015 | The continuous reward transform and 1-D-reward fits are not reproducible run to run. Probably one cause in the C core. |

First check whether the existing gates already see this noise; the fix is
then likely a fixed summation order, as in `8e79ac68`.

### Tier 3: core contracts to settle, and cheap core fixes (before promoting the code they touch)

| ID | What to decide or fix |
|---|---|
| OI-021 | One rule for discrete graphs across all read-outs. Includes `Graph.moments(power)` on a discrete graph silently returning the continuous moment. |
| OI-031 | What `defect()` means for mass that starts on an absorbing vertex. The plan lists this as its own decision. |
| OI-026 | Whether derived graphs (`laplace_transform`, `reward_transform`) stay parameterized; one rule. |
| OI-033 | C++ `discretize` accepts NaN and requires a rate in (0, 1) where Python requires > 0. This is a C++/Python API mismatch, the goal's own territory. |
| OI-019, OI-020 | Discrete reward transform: validation gap, memory leak, misleading error. Small C fixes in code the refactor moves. |
| OI-046 | `_expected_scc_filenames` one-line fix (optional here; otherwise at the SCC step). |
| OI-088 | `Graph.moments(k)` wrong for k >= 13 (32-bit factorial overflow): a small C++ fix, and a wrong result today. |
| OI-089 | `load_cpp_builder`: predictable `/tmp` library path (planting risk), cache key ignores the phasic version (stale-library memory corruption), probably fails to link. Secure directory, for `pmf_from_cpp`: complete installed source set and one source rule (refactor plan phase 0c); for `load_cpp_builder`: disabled with a clear `NotImplementedError` in phase 0d (user decision D19), rebuilt with the compile fix and the plain-bytes hand-off in phase 12; tests for both user-C++ routes. |
| OI-116 | `load_cpp_builder` normalizes every returned graph (wrong distribution; DPH collapse). Remove the call (or make it explicit) before the phase-1 baseline freezes it. Scheduled: refactor plan phase 12 (the route is disabled from phase 0d until then, D19). |
| OI-117 | The exact moments gradient (`exact_moment_grad=True`, the default) disagrees with closed forms near a rate pole, with nothing logged at the default level: on a cycle a<->b with absorption rate t1 = 5e-8 it gives -1.333e16 against the closed form -8e14; on a single-exit chain at t1 = 1e-13 it gives -1e16 against -1e26. Mechanism not investigated (a conditioning gate, MPFR switch or tape precision are candidates). Matters for fits whose rates approach zero. | S (measured by Plan A review 10, `experiments/fd_domain_steps/review10/p4_cycle_pole.log`, `p2_edge_rule.log`; not re-run) | Plan A review 10 |
| OI-118 | C++ API defect: reassigning a `phasic::Vertex` (`prev = v`, the natural chain-building idiom for a C++ user) segfaults in `Graph::operator=`. Found while probing the refactor plan; matters for done-criterion 4 (C++ users build every model). Scheduled: refactor plan phase 0c. | V (measured, `experiments/core_ssot/review12/p_vertex_assign_crash.log`) | refactor plan review 12 |
| OI-116 | `load_cpp_builder` normalizes every graph it returns: the generated wrapper calls `g->normalize()` after the user's `build_model` ([src/cpp/phasic_pybind.cpp](src/cpp/phasic_pybind.cpp#L5182)), which divides each vertex's out-edges by its total rate. So the returned continuous graph has unit exit rates and its documented `pdf` / `moments` describe a different distribution; on a discrete graph it collapses the chain (the same class as F-001, fixed earlier in `pmf_from_cpp`). Scheduled: refactor plan phase 12 (route disabled until then, D19). | V (read from source 2026-11-26) | refactor plan review 4 |
| OI-089 | `load_cpp_builder` (the pybind route for user C++ models) writes and loads its compiled library at a predictable `/tmp/ptd_builder_<hash>.so`, keyed only on the user's source text ([src/cpp/phasic_pybind.cpp](src/cpp/phasic_pybind.cpp#L5164)). Two consequences: (1) a peer on a shared machine can plant a `.so` there; the Python JIT route was hardened against exactly this in July (`_secure_artifact_dir`), this one was not; (2) after any phasic change to the graph's memory layout, a stale library is loaded and memory is silently corrupted; even a fresh compile can mismatch, because it compiles the live source tree without the extension's layout defines (e.g. `PHASIC_B3_VALIDATORS`). Fix: hand the graph over as plain bytes so no struct crosses compiled copies, (refactor plan phase 12; the route is disabled from phase 0d, D19) and install the complete JIT source set (phase 0c). It also compiles only `phasiccpp.cpp` + `phasic.c`, without the SCC sources the Python route says are needed to link, so it is broken today: measured by refactor review 12, it does not even compile (`phasic_log.h` not found, then missing symbols; it builds only when the C files are compiled as C), `experiments/core_ssot/review12/p_today_compile.log`. No tests exist for it or for `pmf_from_cpp`. | S (read from source) | refactor plan review 2 |

### Tier 4: do inside the refactor, at the step that touches the code

Fold these into the refreshed plan (OI-009) as part of the matching step:

- **add_epoch promotion:** OI-025.
- **elimination-cache layer:** OI-079 (delete or harden the old-format
  loader), OI-087 (cache invariants as code, not caller discipline).
- **refactor inventory (OI-009):** OI-086 (bffg duplication).
- **serialize reconciliation:**
  - OI-053: the synthetic marker is lost on serialize;
  - OI-054: `from_serialized` does not restore the current theta;
  - OI-083: the hierarchical cache rebuilds a constant graph as parameterized;
  - OI-079: legacy cache-format hardening.
- **weight-formula compiler:** OI-048, OI-051.
- **pybind thinning:** OI-030 (the dangling `scc_decomposition`) and OI-034 (the abort when an OpenMP thread spawn fails).
- **FFI core consolidation:** OI-028 (failures swallowed as NaN).
- **trace engine unification:** OI-044.
- **SCC ordering:**
  - OI-045: the topological-order divergence;
  - OI-046: if not done in Tier 3;
  - OI-035: the spurious out-of-memory;
  - OI-049: stitching small graphs.
- **sampler unification:** OI-047.
- **dead-code removal:** OI-050 (delete the per-SCC external-pointer scheme).
- **each port's gate:** the coverage gaps of OI-065, one gate per promoted capability.
- **generated-C++ channel, kept for now by the plan's decision:** OI-029, OI-058.

### Tier 5: after the refactor, or independent of it

- **Work that changes numbers by design** (rule 1):
  - OI-002: scale-free forward and gradients;
  - OI-005: exact PDF gradient;
  - OI-040, OI-043: the limits they would close;
  - OI-004: earlier-epoch gradient.

  The scale-free plan's FD-step half (Plan A) is Python-side and could run
  in parallel if the mixed-scale crash blocks your analyses.
- **Gradient programme:** OI-003, OI-008, OI-027, OI-052, OI-070 to OI-074,
  OI-085.
- **Inference layer:**
  - inference defects: OI-041, OI-042, OI-071;
  - MCMC and SVGD bugs: OI-010 to OI-014, OI-016 to OI-018, OI-032;
  - priors and fit conveniences: OI-057, OI-066, OI-084, OI-081, OI-082;
  - overflow observations: OI-006;
  - missing test fixtures: OI-064.
- **Exception to consider now:** OI-010 and OI-018 (MCMC chains that freeze
  silently). A loud error is a few lines of code, if you use MCMC for real
  analyses during the refactor.
- **Other:**
  - OI-024: tests wiping the real caches;
  - OI-017: memory growth per fit;
  - OI-055: bug 5 cache limits;
  - OI-056: constant edges in discrete graphs;
  - OI-063: the two oversized test files;
  - OI-075, OI-076: optional revival and fusion work;
  - OI-078: discretized expectation counts all steps;
  - OI-080: the per-SCC file cache keyed on path.
- **Docs and notebooks:** your notebook edits (OI-091 to OI-099) are
  independent and can happen any time. Regenerate the API docs (OI-090) after
  the refactor, since the API pages will change again.
- **Housekeeping:** OI-112 (trim superseded status text) and OI-114 (delete
  the old compiled-gradient cache files). OI-115 needs nothing.

---

## 1. Decisions waiting for you

| ID | Decision | Source |
|---|---|---|
| OI-001 | Approve (or not) fixing the defects found during the five bug fixes. They are listed one per row in section 2 (OI-010 to OI-026). | [plans-and-findings/found-defects-2026-11-23.md](plans-and-findings/found-defects-2026-11-23.md) |
| OI-002 | **Scale-free likelihood and gradients.** Split 2026-11-26: **Plan A** (domain-aware FD steps, no forward change) is written as [plans-and-findings/fd-domain-steps-plan.md](plans-and-findings/fd-domain-steps-plan.md) and goes before the refactor; **Plan B** (true-uniformization forward, cost ceiling, `advance(dt)`) stays in v5 and goes after the refactor. v5 also absorbs Batches 1 and 4 of [plans-and-findings/epoch-fit-prior-and-fd-floor-plan.md](plans-and-findings/epoch-fit-prior-and-fd-floor-plan.md). | [plans-and-findings/fd-step-and-slow-band-guard-plan.md](plans-and-findings/fd-step-and-slow-band-guard-plan.md) |
| OI-003 | **Exact gradients for big models by splitting them into strongly connected components** (Deferred 1). Implementation plan drafted and reviewed, awaiting your sign-off. It is the only designed route to exact gradients above ~10k vertices. | [plans-and-findings/b3-d1-implementation-plan.md](plans-and-findings/b3-d1-implementation-plan.md), [plans-and-findings/b3-stocktake-2026-08-16.md](plans-and-findings/b3-stocktake-2026-08-16.md) §4 |
| OI-004 | **Exact gradient for the earlier epochs of multi-epoch models** (Deferred 2): build or park. Exploration done; no implementation plan. The main reason to build it: the current backward pass crashes for parameters below ~1e-7 (also covered by OI-002). | [plans-and-findings/deferred-2-daisy-intermediate-epoch-plan.md](plans-and-findings/deferred-2-daisy-intermediate-epoch-plan.md), [plans-and-findings/b3-d2-derisk-findings.md](plans-and-findings/b3-d2-derisk-findings.md) |
| OI-005 | **Exact gradient of the PDF at observed times** (Deferred 3): build or park. Route chosen (Poisson mixture), derivation verified; no implementation plan. It is the only item that makes the default SVGD fit exact, and would also close the 2.5e-3 likelihood bias (OI-040). Overlaps with OI-002's forward replacement. | [plans-and-findings/deferred-3-pdf-gradient-revival-plan.md](plans-and-findings/deferred-3-pdf-gradient-revival-plan.md), [b3-d3-derisk-findings.md](plans-and-findings/b3-d3-derisk-findings.md), [b3-d3-e5-derivation-dossier.md](plans-and-findings/b3-d3-e5-derivation-dossier.md) |
| OI-006 | **Overflow observations** (keep over-cap joint-probability observations as an explicit category, about twice as informative). Deferred by you; design v3 has open decisions D1-D5. | [plans-and-findings/overflow-observations-plan.md](plans-and-findings/overflow-observations-plan.md) |
| OI-007 | **Stage-3 refactor, remaining workstreams**: thinning the pybind layer (WS-D), C++ single-source promotions (WS-E, only discretize done), splitting the C core (WS-F), doctrine document and CI guards (WS-G). Not started; whether to continue is yours. Several pinned divergences in section 3 are owned by it. | [plans-and-findings/stage3-refactor-plan.md](plans-and-findings/stage3-refactor-plan.md), [plans-and-findings/stage3-execution-handoff.md](plans-and-findings/stage3-execution-handoff.md) |
| OI-009 | DECIDED 2026-11-26: C/C++ single source for every core computation (incl. joint-probability family, C++ index codec), inference layer Python; plan [plans-and-findings/core-engine-ssot-plan.md](plans-and-findings/core-engine-ssot-plan.md) (v2.17, SIGNED OFF after twenty adversarial reviews, 2026-11-26; D19: `load_cpp_builder` disabled until phase 12). | [plans-and-findings/stage3-refactor-plan.md](plans-and-findings/stage3-refactor-plan.md) |
| OI-008 | High-precision (MPFR) gradients (Deferred 4). Parked by you 2026-08-15; listed so it is not lost. | [plans-and-findings/deferred-4-mpfr-conditioning-floor-plan.md](plans-and-findings/deferred-4-mpfr-conditioning-floor-plan.md) |

## 2. Defects: wrong results, crashes, freezes

| ID | Defect | Check | Source |
|---|---|---|---|
| OI-010 | MCMC silently freezes a chain whose initial log-probability is non-finite (no warning). 17 of 32 chains frozen in one reviewer's runs. | S | found-defects A |
| OI-011 | Built-in priors hard-code one transform's Jacobian and inverse, so any built-in prior with a custom `param_transform` has a wrong log-prior and wrong initial points. | S | found-defects B; [plans-and-findings/epoch-default-prior-and-truncated-gauss-plan.md](plans-and-findings/epoch-default-prior-and-truncated-gauss-plan.md) §6 |
| OI-012 | `Graph.mcmc` is not reproducible for a fixed seed. | S | found-defects C |
| OI-013 | `MCMC(parallel='processes')` fails for graph-built models (cannot pickle GraphBuilder); its only test is commented out. | S | found-defects D |
| OI-014 | `SVGD.map_estimate_with_optimization` raises TypeError. | S | found-defects E |
| OI-015 | Fits with 1-D rewards are not bit-reproducible run to run (~1e-8). Possibly the same cause as OI-022. | S | found-defects F |
| OI-016 | The preconditioner's moment-matching grid is in inverse-softplus units; on a (0, 1) slot it cannot reach theta below 0.119 (affects scaling only). | S | found-defects G |
| OI-017 | Memory grows ~8-10 MB per fit when refitting the same model in a loop. | S | found-defects H |
| OI-018 | MCMC with positive_params=False, a single BetaPrior and a fixed value outside (0, 1) freezes (OI-010's mechanism). | S | found-defects J |
| OI-019 | `reward_transform_discrete` skips validation of zero-reward vertices, so an invalid row is eliminated away and a clean pdf comes back. | S | found-defects K |
| OI-020 | The C discrete reward transform leaks its vertices array on success, and a negative weight on a zero-reward vertex gives the misleading error "Failed to clone edge". | S | found-defects L |
| OI-021 | Read-outs that do not validate a native discrete graph (row sums above 1): expected_waiting_time, expected_sojourn_time, joint_prob_table, probability_matching, sample_path, normalize(), the sojourn and moments-only FFI paths, trace paths. Also `Graph.moments(power)` without `discrete=True` on a discrete graph returns the continuous-type moment. | S | found-defects N; [plans-and-findings/bug4-discrete-rowsum-plan.md](plans-and-findings/bug4-discrete-rowsum-plan.md) §2.3 |
| OI-022 | The continuous `reward_transform` is not bit-reproducible run to run (~7e-16). **Likely root cause (refactor review 3, 2026-11-26):** constant-rate elimination, the reward transform and `_dyn` order edges by memory address (`arr_c_cmp`, [src/c/phasic.c](src/c/phasic.c#L6088)); 6 instances gave 3 distinct bit patterns ([experiments/core_ssot/review3/probe_determinism.py](experiments/core_ssot/review3/probe_determinism.py)). Fix approved as a numeric exception (refactor plan R14, phase 0b). | S | found-defects M |
| OI-023 | Fresh builds of one constant-rate model differ in the last bit of expectation()/variance() between instances (likely summation order, the class fixed for parameterized models in 8e79ac68). | S | found-defects Q |
| OI-024 | Tests that call `phasic.clear_caches()` wipe your real `~/.phasic_cache` and `~/.jax_cache` (SLURM-only tests); `clear_caches()` itself ignores `PHASIC_CACHE_DIR`. | S | found-defects P |
| OI-025 | `add_epoch` on a graph built with `Graph(cb, theta_dim=1)` raises TypeError (theta_dim is passed to the callback). | V | [plans-and-findings/constant-parameterized-consistency-plan.md](plans-and-findings/constant-parameterized-consistency-plan.md) §11 |
| OI-026 | `laplace_transform` of a parameterized graph stays parameterized while `reward_transform` of one does not (inconsistent). | V | [plans-and-findings/graph-cache-constant-roundtrip-plan.md](plans-and-findings/graph-cache-constant-roundtrip-plan.md) §6 |
| OI-027 | Rate-blowup forward/backward mismatch: when the forward returns its rate-blowup penalty, the exact moments gradient still returns the gradient of the real (never computed) moments. Untested. | S | master plan §16b item 4; CLAUDE.md |
| OI-028 | Multi-epoch FFI handlers turn a failed context creation into a NaN row with no log line; the batched sojourn FFI also NaN-fills out-of-range indices silently. | S | master plan §16b item 8 |
| OI-029 | `pmf_from_cpp`'s callback probably has the same vmap bug fixed elsewhere (no ndim handling); unconfirmed (needs a generated C++ model fixture). | S | master plan §16b item 7 |
| OI-030 | Python `scc_decomposition()` on a temporary graph keeps a dangling reference; composing afterwards returns plausible garbage silently. | S | master plan §16b item 12 |
| OI-031 | `defect()` returns 0 while `cdf()` counts mass that starts on an absorbing vertex (inconsistent); two native asserts are disabled for it. | V (asserts still disabled) | [plans-and-findings/stage2-coverage-safety-net-handoff.md](plans-and-findings/stage2-coverage-safety-net-handoff.md) §7 |
| OI-032 | `method_of_moments` falls back to std = 0.5·abs(theta) when the delta method fails, with a warning only if verbose. | V | [plans-and-findings/epoch-default-prior-and-truncated-gauss-plan.md](plans-and-findings/epoch-default-prior-and-truncated-gauss-plan.md) §6 |
| OI-033 | C++ `Graph::discretize(rate_fn)` lets NaN through, and its scalar overload requires a rate in (0, 1) where Python requires > 0; `joint_prob_graph`'s `mutation_rate` has no finite/non-negative check. | S | [plans-and-findings/constant-parameterized-consistency-plan.md](plans-and-findings/constant-parameterized-consistency-plan.md) §13 |
| OI-034 | When an OpenMP thread spawn fails (distributed notebook, pmap cells), the process aborts in pybind (GIL not held) instead of raising. | S (July) | [plans-and-findings/deferred-notebook-triage.md](plans-and-findings/deferred-notebook-triage.md) §4.2 |
| OI-035 | `cache_file_path()` → `ptd_scc_build_synthetic_graph` reports a spurious out-of-memory under the pytest configuration; not root-caused. | S (July) | [plans-and-findings/audit-situation-map.md](plans-and-findings/audit-situation-map.md) §6 |
| OI-088 | `Graph.moments(k)` is wrong for k >= 13: the bound C++ helper `fac()` returns a 32-bit `int` ([src/cpp/phasic_pybind.cpp](src/cpp/phasic_pybind.cpp#L82)), which overflows at 13!. On Exp(1): 13th moment 1.93e9 instead of 6.23e9, 14th 1.28e9 instead of 8.7e10; correct up to k = 12. Affects `moments`, and any variance/covariance/discrete correction built on high moments. Fix: wide integer or double factorial; gate k >= 13. | V (reproduced 2026-11-26) | refactor plan review 1, [experiments/core_ssot/review1/probe_moments_fac.py](experiments/core_ssot/review1/probe_moments_fac.py) |
| OI-119 | Deferred refusal with rewards: on `pmf_and_moments_from_graph` with 1-D rewards (pybind and FFI paths), a call at a theta that makes an edge weight negative is NOT refused; it silently returns a wrong answer (an Exp(1) density, moments [1, 2] on the test chain), and the NEXT call, at a valid theta, raises the old "non-positive value" error. Valid-theta calls are bitwise consistent, and today's FD backward never probes below zero there (1e-9 clamp), so today's gradients are unaffected; the hazard is direct calls at invalid theta and any rule that probes below zero with rewards. Same class as the fixed add-edge `ptd_err` leak (`test_ptd_err_no_leak.py`), on the reward-transform path. | V (`experiments/fd_domain_steps/batch0/a2_state_leak.log`) | Plan A Batch 0, [plans-and-findings/fd-domain-steps-findings.md](plans-and-findings/fd-domain-steps-findings.md) §3 |
| OI-120 | `Graph.reward_visit_probability` fails on every callback-mode graph ("Error parsing graph structure: Unknown weight_mode: callback" from the GraphBuilder parser). Zero inflation avoids it in callback mode (uses `_cdf_zero_fn`), so only the public method is affected. | V (`experiments/fd_domain_steps/batch0/g1_refusal_grid.jsonl`) | Plan A Batch 0, [plans-and-findings/fd-domain-steps-findings.md](plans-and-findings/fd-domain-steps-findings.md) §3 |
| OI-123 | `pmf_from_cpp` models cannot be vmapped: the `pure_callback` (`vmap_method='expand_dims'`) returns one row for a batch of particles ("Incorrect output shape ... Expected: (2, 3), Actual: (1, 3)"), so `jax.vmap` of the forward, and `jax.vmap(jax.grad(loss))` over SVGD particles, fail. Eager and `jit` work. | V (`experiments/fd_domain_steps/batch1/p_cpp_vmap.py` / `.log`; also on master `ff049679` during the golden capture) | Plan A Batch 1, [plans-and-findings/fd-domain-steps-findings.md](plans-and-findings/fd-domain-steps-findings.md) §10.3 |
| OI-124 | **Amendment A-124 drafted 2026-11-27 (plan §12), awaiting your approval; reachable only through an explicit 'unit' declaration at exactly 0 (SVGD's sigmoid is clipped to [1e-9, 1-1e-9]).** Plan A review m1, DEFERRED to Batch 2 (user decision 2026-11-27): a slot declared `'unit'` at exactly 0 gets today's +-1e-7 probe (plan §4.5: unit outside (0, 1) keeps today's probe), so on a refusing model its gradient crashes, while an undeclared slot at 0 gets the one-sided [0, 1e-7] step. Reachable after Batch 2 via sigmoid underflow or a 'unit' slot fixed at 0; master crashes too. Proposed fix: one-sided step at 0 for 'unit', as for 'positive' (log mode excepted). | V (`experiments/fd_domain_steps/batch1/review/r5_edges.log`) | Plan A Batch 1 review, [plans-and-findings/fd-domain-steps-findings.md](plans-and-findings/fd-domain-steps-findings.md) §10.4 |

## 3. Known limits and pinned divergences (tests mark them xfail)

| ID | Item | Check | Source |
|---|---|---|---|
| OI-040 | The default continuous PDF is only ~2.5e-3 accurate and biases the estimate by the same amount, regardless of sample size. Two strict xfails in [test_likelihood_correctness.py](tests/pytest/test_likelihood_correctness.py). Fix routes: OI-002 or OI-005. | V | CLAUDE.md; [plans-and-findings/b3-mixed-scale-defect-settled.md](plans-and-findings/b3-mixed-scale-defect-settled.md) |
| OI-041 | Three settled inference defects, mechanism unknown: under-dispersed posterior (coalescent), joint-probability 168-vertex model stuck ~54% low, mixed-scale two-locus recombination not recovered. Strict xfails in [test_notebook_model_recovery.py](tests/pytest/test_notebook_model_recovery.py). Candidates: kernel bandwidth, preconditioner, step schedule. | V (xfails active) | [plans-and-findings/b3-budget-vs-defect-findings.md](plans-and-findings/b3-budget-vs-defect-findings.md) |
| OI-042 | Two-island 186-vertex recovery is budget-limited (not a defect); slow-tier xfail. | S | same |
| OI-043 | Multi-epoch FD gradient breaks at mixed scale (step 1e-7 drives a ~1e-8 rate negative and the backward aborts). Strict xfail in [test_fd_gradient_mixed_scale.py](tests/pytest/inference/test_fd_gradient_mixed_scale.py). Fix: OI-002 Batch 1. | V | CLAUDE.md |
| OI-044 | Python trace refuses cyclic graphs and `weight_mode='formula'`; the C engine handles both (Stage-3 Q1). xfails in [test_gate_elimination_bit_identity.py](tests/pytest/test_gate_elimination_bit_identity.py) and [test_trace_vs_direct.py](tests/pytest/inference/test_trace_vs_direct.py). To be closed by deleting the Python trace pipeline (user decision 2026-11-26, refactor plan R12', phase 2). | V | [plans-and-findings/stage3-refactor-plan.md](plans-and-findings/stage3-refactor-plan.md) §9 |
| OI-045 | `SCCGraph::sccs_in_topo_order` trusts a stored order that is not topological (Stage-3 Q10). xfail in [test_gate_scc_ordering.py](tests/pytest/test_gate_scc_ordering.py). | V | same |
| OI-046 | `compute_repository._expected_scc_filenames` always returns [] (passes a tuple where a Graph is expected); "a one-line fix". xfail in [test_gate_scc_ordering.py](tests/pytest/test_gate_scc_ordering.py). | V | [plans-and-findings/audit-situation-map.md](plans-and-findings/audit-situation-map.md) §6 |
| OI-047 | The two conditioned samplers use different RNGs (one unseeded), so per-draw seeded identity is impossible (Stage-3 Q11a). xfail. | V | [test_gate_conditioned_samplers.py](tests/pytest/test_gate_conditioned_samplers.py) |
| OI-048 | Weight-formula oracle and C runtime disagree on dead `select()` arms and on pow of a negative base with a fractional exponent (Q-G4-1, Q-G4-2). xfails. | V | [test_gate_weight_formula_conformance.py](tests/pytest/test_gate_weight_formula_conformance.py) |
| OI-049 | `stitch_scc_traces` gives all-zero rates for graphs below min_size (not reached in production). xfail. To be closed by deleting the Python trace pipeline (R12', phase 2). | V | [inference/test_trace_stitching.py](tests/pytest/inference/test_trace_stitching.py) |
| OI-050 | The external-pointer per-SCC replay scheme (WP-3) does not work and is unused; xfail. Candidate for deletion. | V | [test_scc_prc_external.py](tests/pytest/test_scc_prc_external.py); memory "Distributed SCC architecture" |
| OI-051 | The C++ weight-formula compiler (`wf_tokenize` / `WFParser`) is unreachable and untested; the C++-only moment functions are bound to nothing (Stage-3 Q3, G2). | V (code present) | [plans-and-findings/stage2-coverage-safety-net-handoff.md](plans-and-findings/stage2-coverage-safety-net-handoff.md) §7 |
| OI-052 | Exact-gradient boundaries kept by design, listed so they are not re-investigated: discrete + rewards; non-JAX-native callbacks; joint-index exact gradient for `was_dph` graphs and for formula/callback modes; decoupled formula theta dimensions. | S | CLAUDE.md B3 sections |
| OI-053 | A synthetic SCC graph wrapped and serialized loses its "synthetic" marker (no shipped caller does this). | S | master plan §16b item 14 |
| OI-054 | `from_serialized()` restores structure, not the current theta (weights at theta = 1 until `update_weights`); apparently deliberate. | S | [plans-and-findings/graph-cache-constant-roundtrip-plan.md](plans-and-findings/graph-cache-constant-roundtrip-plan.md) §6 |
| OI-055 | Bug 5 limits: callbacks that take or capture user objects (incl. phasic's own `HexCell` / `HexGrid` dataclasses and a bare `PropertySet`) cannot be cached; recursive nested helpers are refused. Candidate follow-up: accept phasic's own dataclasses. | S | [plans-and-findings/bug5-graph-cache-plan.md](plans-and-findings/bug5-graph-cache-plan.md) §6 |
| OI-056 | Constant (coefficient-less) edges are refused in discretized and joint-probability graphs rather than supported (needs an idempotent per-theta renormalisation in C). | S | [plans-and-findings/discretize-cpp-ssot-plan.md](plans-and-findings/discretize-cpp-ssot-plan.md) §8 |
| OI-057 | `_inverse_softplus` floors at 1e-6, so truncated GaussPrior draws below 1e-6 are still floored; `_transform` is never reset between fits; default priors ignore exposure on non-joint graphs. | S | [plans-and-findings/epoch-default-prior-and-truncated-gauss-plan.md](plans-and-findings/epoch-default-prior-and-truncated-gauss-plan.md) §6 |
| OI-058 | `pmf_from_graph` / `moments_from_graph` JIT paths need the C/C++ source tree on disk ("JIT compilation requires the phasic C/C++ source tree"), so they fail on a plain install. | S | [plans-and-findings/constant-parameterized-consistency-plan.md](plans-and-findings/constant-parameterized-consistency-plan.md) §11 |

## 4. Tests and CI

| ID | Item | Check | Source |
|---|---|---|---|
| OI-060 | Four [inference/test_plot_ci.py](tests/pytest/inference/test_plot_ci.py) tests fail on every run: they read the y-label after a commit moved the text to the title (one will still fail after that fix, since the title now says "KDE HPD"). | V (fail in bug 5 suite run) | [plans-and-findings/hpd-hdr-naming-plan.md](plans-and-findings/hpd-hdr-naming-plan.md) §8 |
| OI-061 | Unseeded, flaky tests (also `test_svgd_api_parity::test_graph_svgd_and_direct_svgd_agree`: data were unseeded, and the fit is chaotic, findings §11.6; seeded since Plan A Batch 2): `test_mcmc_accuracy::test_two_param_recovery` (R-hat threshold, failed once 2026-11-25), `test_svgd_correctness::test_basic_convergence`, `test_reward_validation::test_svgd_runs_silently_on_partial_coverage_with_introspection` (also shows a ~5% upward bias on seeded data, not investigated). Also `inference/test_svgd_correctness::test_positive_constraint` (failed once in the Plan A Batch 1 gate, 3/3 passes in isolation: its data come from `Graph.sample` under `np.random.seed` only, so they differ run to run; it also asserts on `svgd.particles`, which are unconstrained phi, not theta). | V | bug 1 and bug 5 suite runs; epoch-default-prior plan |
| OI-062 | CI runs only the equivalence gates (`-m equivalence`), so almost none of the suite, including all tests added since August, runs in CI. | V | [.github/workflows/native-tests.yml:79](.github/workflows/native-tests.yml#L79) |
| OI-063 | [test_svgd_divergence_robustness.py](tests/pytest/test_svgd_divergence_robustness.py) exceeds 6 GB run alone; [test_notebook_model_recovery.py](tests/pytest/test_notebook_model_recovery.py) takes ~26 min. Both run outside the normal cap. | S | found-defects O; discretize plan §13 |
| OI-064 | No pytest fixture with a trap / deficit-sink vertex, so the infinite-sojourn tangent guard is untested end to end; no exposure + tied combination test for the exact final-epoch gradient. | S | CLAUDE.md (joint-index, Batch H) |
| OI-066 | [tests/pytest/test_mcmc.py](tests/pytest/test_mcmc.py): all 35 tests are commented out (MCMC_TODO.md still claims 27 passing); MCMC is covered only by [inference/test_mcmc_accuracy.py](tests/pytest/inference/test_mcmc_accuracy.py) and the bug 2/3 test files. | V | [plans-and-findings/MCMC_TODO.md](plans-and-findings/MCMC_TODO.md) |
| OI-067 | The suite-gate tooling ([gate2.sh](.git/claude-scratch/gate2.sh), [gate_compare.py](.git/claude-scratch/gate_compare.py), [runcap.py](.git/claude-scratch/runcap.py), [gate_rerun.sh](.git/claude-scratch/gate_rerun.sh)) lives in [.git/claude-scratch/](.git/claude-scratch/): unversioned, not in CI, lost if [.git](.git) is re-cloned. | V | bug series handoff |
| OI-065 | Cross-path test coverage gaps from the July audit: ctypes-JIT paths, multivariate PMF, discrete and cyclic cross-path checks, CDF/sojourn/Laplace with no cross-path gate, SCC numeric equivalence, graph sizes above 7 vertices, gradient equivalence across paths. Trace-vs-FFI was added since (F-004); the rest is unverified. | S (July) | [plans-and-findings/audit-situation-map.md](plans-and-findings/audit-situation-map.md) §4 |
| OI-121 | Flaky last bit: `inference/test_log_prob_unified_bit_identity.py::test_log_prob_unified_bit_identity_zi[theta0]` failed once in the Plan A Batch 0 master baseline (diff 2.8e-14) and passed 3 of 3 isolated reruns. Zero-inflated (rewarded) path; probably the OI-022 / OI-015 non-reproducibility. Rerun in isolation before calling it a regression. | V | Plan A Batch 0 baseline (`.gate_master_fdsteps` in `../phasic-fdsteps`) |

## 5. Smaller follow-ups and improvements

| ID | Item | Check | Source |
|---|---|---|---|
| OI-070 | Two cheap gradient fixes: let the moments gradient skip its conditioning gate (the sojourn family already can), and cache the moments tape between calls (measured 3,062x cost difference). | S | CLAUDE.md resume section; stocktake §2 |
| OI-071 | Recombination recovery from richer observations (multivariate rewards, joint probabilities); not yet made to converge. | S | CLAUDE.md resume section |
| OI-072 | `moments_from_graph` and `method_of_moments` are FD-only (scipy computes its own Jacobian); `pmf_and_moments_from_graph_multivariate` has no `exact_moment_grad` passthrough. | S | master plan §16b item 5; CLAUDE.md |
| OI-073 | Synthetic-graph gradient decline has no latch: re-evaluated and logged on every backward, and the moments family pays a full tape build each time. | S | master plan §16b item 13 |
| OI-074 | Opt-in user-supplied derivative for non-JAX-native weight callbacks (declined as default; available if wanted). | S | master plan §16b item 11 |
| OI-075 | Revive the disabled builder-based `pmf_from_graph_parameterized` (revival checklist in CLAUDE.md). Only if the builder style is wanted; nothing uses it. | S | CLAUDE.md |
| OI-076 | Fuse particles × features into one FFI call for the multivariate path (custom_vmap); do only if measurements show it matters. | S | memory "custom_vmap follow-up" |
| OI-077 | Discretize follow-ups: theta layout for SVGD epoch routes derived from the graph layout; C++ StateIndexer and C++ `joint_prob_graph`; `is_aux` in the C++ serialized graph and the SLURM deserialize path; persisting the theta layout. | S | [plans-and-findings/discretize-cpp-ssot-plan.md](plans-and-findings/discretize-cpp-ssot-plan.md) §8, §13 |
| OI-078 | `expectation()` on a discretized graph counts all steps; the mutation count needs `graph.rewards` passed explicitly. | S | PROJECT-OVERVIEW §1d |
| OI-079 | Old (rev-1/2, `PTDPRMC1`) elimination-cache loader is unhardened, unlike rev-3. Problems: an unchecked memory offset read from the file gives an out-of-bounds WRITE (`ptd_pcg_decode_ptr_impl`, [phasic.c:3254](src/c/phasic.c#L3254)); an unvalidated command type aborts the whole process ([phasic.c:4470](src/c/phasic.c#L4470)); size arithmetic is not overflow-checked. Reachable today only through the private bindings `_load_param_compute_graph` / `_load_synthetic_param_compute_graph_ex` (no production caller). Related: `Graph.pull_cache()` downloads a rev-1 parent artifact that the automatic loader (rev-3 only) silently ignores, contrary to its docstring. Recommended: delete or harden the rev-1/2 path at the refactor's cache/serialize step. | V (decoder unchanged) | [plans-and-findings/exact-fd-atlas-caching.md](plans-and-findings/exact-fd-atlas-caching.md) §2c, §5 (L2-L4); [plans-and-findings/b3-execution-tracker.md](plans-and-findings/b3-execution-tracker.md) |
| OI-080 | `_pmap_file_cache` is keyed on file path only (stale if a work-unit file is rewritten in one process; SLURM path; untested). | S | [plans-and-findings/bug1-compiled-cache-plan.md](plans-and-findings/bug1-compiled-cache-plan.md) §6 |
| OI-081 | Other SVGD warnings not routed through the short-notice mechanism (`logger.warning` in svgd.py, DataPrior messages). | S | [plans-and-findings/svgd-notices-plan.md](plans-and-findings/svgd-notices-plan.md) §6 |
| OI-082 | Cosmetic: `MCMC.plot_posterior(ci_method='percentile')` deprecation warning points at mcmc.py instead of the caller. | S | [plans-and-findings/hpd-hdr-naming-plan.md](plans-and-findings/hpd-hdr-naming-plan.md) §8 |
| OI-084 | MCMC warm restart (continue sampling from where a run stopped); the last open feature of the April MCMC to-do list (plots, vmap chains and thinning are done). | V | [plans-and-findings/MCMC_TODO.md](plans-and-findings/MCMC_TODO.md) |
| OI-085 | `reward_visit_probability` has a finite-difference-only gradient (`custom_vjp` at [ffi_wrappers.py:1278](src/phasic/ffi_wrappers.py#L1278)), a third FD site besides `moments_from_graph` / `method_of_moments`, not in CLAUDE.md's gap lists. | V | [plans-and-findings/exact-fd-atlas-loose-ends-docs.md](plans-and-findings/exact-fd-atlas-loose-ends-docs.md) item 4 |
| OI-086 | [bffg.py](src/phasic/bffg.py) (~600 lines) may duplicate the C sampler's sojourn and reward accounting (unconfirmed); its model calls a raw FFI primitive with no differentiation rule (harmless while only MCMC uses it). A "one implementation per capability" question for the refactor inventory (OI-009). | S | [plans-and-findings/exact-fd-atlas-python-wiring.md](plans-and-findings/exact-fd-atlas-python-wiring.md) §12; [plans-and-findings/audit-situation-map.md](plans-and-findings/audit-situation-map.md) §6 |
| OI-087 | Undocumented invariants in the elimination-tape caches that new code can silently break: (1) `input_specs` is NULL after an mmap load, guarded at one call site only, unguarded at the per-SCC layer (matters for OI-003); (2) the graph-level tape can be stale between `set_was_dph(True)` and the next precompute; (3) buffer ownership (heap vs mmap) differs by construction path; (4) `use_dyn_ordering` is not part of the content hash (probably benign, untested). Fix: shared accessors or asserts at the refactor's cache step. | S | [plans-and-findings/exact-fd-atlas-caching.md](plans-and-findings/exact-fd-atlas-caching.md) §5 (L1, L5-L8) |
| OI-083 | [hierarchical_trace_cache.py](src/phasic/hierarchical_trace_cache.py) serializes with `theta_dim=param_length()`, so a constant graph's work unit is rebuilt parameterized; probably unreachable. | S | [plans-and-findings/graph-cache-constant-roundtrip-plan.md](plans-and-findings/graph-cache-constant-roundtrip-plan.md) §6 |
| OI-122 | The Euler (uniformization) forward's memory appears to grow with its step count: 5.3 GB peak for the multivariate model on a 4-vertex division-formula chain at a rate of ~2e7 (~1e8 steps), and above 6 GB (killed) for a reward-transformed 5-sample coalescent at theta 1e-5 with times up to ~9e5 (~9e8 steps). Cause not investigated. Matters for the unguarded "slow band" (CLAUDE.md) and for population-genetics time scales. | S (measured peaks, cause unverified) | Plan A Batch 0, [plans-and-findings/fd-domain-steps-findings.md](plans-and-findings/fd-domain-steps-findings.md) §3, §4 |

## 6. Docs and notebooks (yours to edit; nothing here is changed by the assistant)

| ID | Item | Source |
|---|---|---|
| OI-090 | Regenerate the API docs with `pixi run api`: docs/api still shows `graph_cache` (Graph, configure, get_config, PhasicConfig) and the pre-rename HPD/HDR method names. It runs a full quarto render that touches your uncommitted [_quarto-default.yml](docs/_quarto-default.yml), so it was left to you. | bug 5; [plans-and-findings/hpd-hdr-naming-plan.md](plans-and-findings/hpd-hdr-naming-plan.md) §8 |
| OI-091 | **Bug 5 (graph cache), cells that now raise:** moran-coalescent-derived 1 and matrix-solve 2 (remove `configure(graph_cache=True)`); old-moran-coalescent-derived 4 (remove `graph_cache=False`); both trace-and-jax-caching notebooks 8, 10, 12 and state-space 36, 38 (`graph_cache=True` → `cache_key="..."`); state-space 45 (pass `model.to_dict()` or drop caching). Text to revise and output-only cells: full list in the plan. | [plans-and-findings/bug5-graph-cache-plan.md](plans-and-findings/bug5-graph-cache-plan.md) §5 |
| OI-092 | **GaussPrior needs `lower=0.0` under positive_params** (raises otherwise). Line scan 2026-11-25 of GaussPrior calls without `lower`: model-selection 7, 13, 26, 30; svgd-basics 42, 49; svgd-joint-prob 54, 56, 61, 62, 66, 71, 84; svgd-learning-rate 22; svgd-multi-feature 5, 15, 30; svgd-multi-param 5; svgd-priors 4, 6, 7, 16, 20 (a call that sets `lower` on a later line would be listed wrongly; cells using positive_params=False are fine). | CLAUDE.md; epoch-default-prior plan |
| OI-093 | model-selection cell 13 also fixes migration at 0, which is now a true zero (was 0.693): its likelihood-ratio results change or the cell raises. | found-defects I; bug 2 plan §4 |
| OI-094 | Joint graphs from an indexer with several properties must pass `reward_only` (two-locus `['loc1','loc2']`, label properties `['descendants']`); earlier numbers there were probably wrong: svgd-joint-prob, model-selection, smc, method-of-moments, coalescent-derived, old-moran-coalescent-derived, possibly time-inhomogeneous. | PROJECT-OVERVIEW §1g (T9) |
| OI-095 | October model-building changes: parametrization 23 (drop the reserved `[rate, 0]` slot); joint-probability 3 and time-inhomogeneous 32 (rates as lists); coalescent-derived 8→26 and old-moran-coalescent-derived 19→37 (`joint_prob_graph` on a discretized graph, already broken). As of 2026-10-15 parametrization and joint-probability looked edited. | PROJECT-OVERVIEW §1d |
| OI-096 | Notebooks whose numbers change on re-run (no edit needed): every fit with a non-round-tripping fixed value (bug 2: svgd-joint-prob, model-selection, method-of-moments, ghost, coalescent-derived, moran-coalescent-derived); epoch fits and continuous `tot_reward_limit` (November normalisation fixes). | bug 2 plan §4; PROJECT-OVERVIEW §1g |
| OI-097 | Environment / docs-pipeline: graphviz in the pixi env needs `dot -c` (inline SVG plots fail); the docs pipeline keeps a notebook's last good output when it fails, so failures are silent. | [plans-and-findings/deferred-notebook-triage.md](plans-and-findings/deferred-notebook-triage.md) |
| OI-098 | Uncommitted notebook fixes lost in the July `git checkout -- docs/` incident were never re-applied (details lost). | [plans-and-findings/stage3-execution-handoff.md](plans-and-findings/stage3-execution-handoff.md) |
| OI-099 | Doc corrections: distributed.ipynb and [profile.py](src/phasic/profile.py) overstate `parallel_elimination`'s benefit for gradient workflows; F-005 note on the 512 auto-granularity floor. | master plan §16b item 1; tracker |

## 7. Housekeeping

| ID | Item |
|---|---|
| OI-110 | 94 commits on master are not pushed (you push). |
| OI-111 | 19 worktrees (`/Users/kmt/phasic-batch*`, `-bug1..5`, `-epochnorm`, `-epochprior`, `-gausslower`, `-totreward`) and 16 stale branches whose work reached master as squash merges (`b3/*`, `derisk/*`, `fd-b3-experiments`, `b3-discrete-theta-adjoint`). Untracked review scratch in the bug 4 and bug 5 worktrees. Removing them is your call. |
| OI-112 | Superseded status text to trim into pointers here: CLAUDE.md "RESUMING THE INFERENCE-CORRECTNESS SESSION (2026-08-17)" and the B3 known-gaps sections; PROJECT-OVERVIEW §1c; the ticket table in [plans-and-findings/joint-prob-followups-plan.md](plans-and-findings/joint-prob-followups-plan.md) (shows merged tickets as "committed"); [plans-and-findings/b3-execution-tracker.md](plans-and-findings/b3-execution-tracker.md). |
| OI-113 | Done 2026-11-26 by you: plans and findings moved to [plans-and-findings/](plans-and-findings/) (staged as renames, not yet committed). Cross-references INSIDE the moved files to root files (e.g. [CLAUDE.md](CLAUDE.md)) still use bare names. |
| OI-114 | Old `compiled_svgd_*.pkl` files in `~/.phasic_cache` are no longer read (bug 1); you may delete them. |
| OI-115 | Five plans referenced by old memory notes were never committed and are lost (`performance-UX-plan.md`, `profiler-plan.md`, `hierarchical-benchmark-plan.md`, `zero-copy-cache-plan.md`, `daisy-chain-fusion-recovery-plan.md`); memory now marks them lost (2026-11-26). Nothing to do unless you have copies. |

---

## Closed (recently confirmed, kept so they are not re-opened)

| Item | Closed by |
|---|---|
| Bugs 1-5 of the November series (compiled-gradient cache, fixed values, BetaPrior transform, discrete row sums, stale graph cache) | merges `974d6422`, `39154082`, `b1203728`, `195ab396`, `96b141fc` |
| Joint-probability tickets T1-T12; T13 not a defect | `31e799b3`, `928bfb6c`, `36edaaa8`, `7b0b7a7c` |
| Epoch likelihood normalisation; total-cap trash edge; thread-exit segfault | `e7282362`, `9cedb4f3`, `038dcc63` |
| Default epoch prior, GaussPrior truncation, theta_mean on parameter scale | `357329ef`, `070728cd`, `cc92bb6e` |
| Parallel elimination treating every weight mode as linear (§16b item 6) | `ccc54e33` |
| Constant models passed to svgd / probability_matching now refused (V) | `e7b7679a` |
| `import phasic` enables JAX 64-bit (V); `moments_from_graph` pow recurrence (V) | — ; `d69919f2` |
| FFI reward-PMF divergence (Q7.1), serialize round-trip divergences (Q6), discrete moments binding (Q5): no longer xfail (V) | `da6e7a6b` and later |
| Likelihood-ratio tests on epoch models; SVGD divergence crash | [plans-and-findings/svgd-lrt-fix-a-plan.md](plans-and-findings/svgd-lrt-fix-a-plan.md); [plans-and-findings/svgd-divergence-robustness-plan.md](plans-and-findings/svgd-divergence-robustness-plan.md) |
| §16b items 2, 3, 9, 10 | Batch H, G.1, G.2 merges |
| August atlas findings since fixed: `moments_from_graph` broke under vmap (Batch D.1, `164e2758`); multivariate path never reached the exact gradient because of rewards (Batches A and G.2); unguarded debug gradient function (Batch D.2); three duplicated gradient skeletons (Batch 0); unwired `ptd_graph_pdf_with_gradient` (deleted, `9ae753c4`); unquantified conditioning floor (characterised by the CC-2 sweep, pinned in [inference/test_d4_conditioning_pin.py](tests/pytest/inference/test_d4_conditioning_pin.py)); only 1 of 5 `Graph.svgd()` leaves reaching an exact path (Batches D.4, G.1, G.2, H) | see each |
