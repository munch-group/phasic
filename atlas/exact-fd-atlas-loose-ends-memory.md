# Cross-session memory audit: open loose ends vs CLAUDE.md, B3/exact-FD relevance

**Scope.** Every file in `/Users/kmt/.claude/projects/-Users-kmt-phasic/memory/`
(32 files: `MEMORY.md` index + 21 `project_*.md` + 11 `feedback_*.md`, confirmed by
`ls`), cross-referenced against the current on-disk
`/Users/kmt/phasic/CLAUDE.md` (read fresh from disk for this audit — it is
**more current** than the CLAUDE.md snapshot pasted into this session's system
prompt: the disk version already has a third B3 subsection, "B3 joint-index
exact sojourn gradient", that the pasted snapshot lacks). "B3" = the initiative
replacing finite-difference gradients with exact analytic θ-adjoint gradients
(moments, sojourn/joint-index, PMF/PDF), wired through JAX for SVGD.

Every code claim below is either **(grounded: file:line, read this session)**
or **(from memory: <name>, <age>)** — I did not re-derive every claim, per
instructions, but I did spot-check the highest-value ones (see §4, "New
findings"), and several of those checks overturned or sharpened the memory's
claim.

---

## 1. Inventory table

| memory | one-line summary | status (per memory) | in CLAUDE.md already? | B3/exact-FD relevance | reason |
|---|---|---|---|---|---|
| `project_b3_analytic_gradient` | Main B3 tracker: continuous/discrete/was_dph/log-mode/joint-index moment & sojourn adjoints shipped; formula/hierarchical still FD; joint-index defaults `exact_grad=False` | Mostly shipped, several flagged-not-fixed gaps, all itemized | **Yes**, extensively (both B3 sections are near-verbatim from this memory) | **likely** | this *is* B3's own tracking memory |
| `project_fd_gradient_b3` | The original FD defect (mixed-scale, absolute eps) + why FD was chosen (trace refuses cycles) + Tier1/2/3 strategy that led to B3 | Superseded by `project_b3_analytic_gradient`'s later batches; one thread (`ptd_graph_pdf_with_gradient` unwired/unverified sign) never resurfaced | Partially — the defect/strategy narrative is folded in; the **unwired `ptd_graph_pdf_with_gradient` lead is NOT mentioned anywhere in CLAUDE.md** | **likely** | direct B3 prehistory; contains a live, ungrounded loose end (see §3.1) |
| `project_svgd_perf_ux` | SVGD-on-parameterized-models perf/UX initiative: caching pivot, per-θ cost (moments vs joint/sojourn path), zero-copy cache | Mixed — several sub-plans done, cache work pivoted/dropped, one hybrid-scheduler follow-up still open | No — CLAUDE.md doesn't discuss per-θ cost tradeoffs or cache economics at all | **likely** | directly informs the *unmeasured per-call cost* CLAUDE.md flags for the new joint-index sojourn gradient (see §3.2) |
| `project_cache_load_cpu_bound` | Reward-compute cache LOAD is CPU-(deserialize)-bound, not I/O; resolved by rev-3 zero-copy mmap; recompute still wins for well-decomposing (small-SCC) models | Resolved (rev-3), but the underlying cost-measurement methodology is exactly what's missing for B3's new sojourn-gradient tape conversion | No | **likely** | same cross-reference as above — methodology transfers directly |
| `project_weight_mode_log_semantics` | `weight_mode='log'` = PRODUCT-in-log-space, strictly positive; `moments_from_graph` and `pmf_from_graph_joint_index` silently ignore `weight_mode` (always linear) | Described as a **silent wrong-answer bug**, no error | Partially — CLAUDE.md's weight-mode explanation (linear/log/formula semantics) matches this memory's correction, but doesn't mention the two silent-ignore sites | **likely** | **STALE — this specific bug is already fixed**, see §3.3 (verified from source) |
| `project_dph_two_representations_was_dph` | Native DPH vs `discretize()`; `was_dph` C-only renorm latch must propagate through FFI/`from_serialized` | Fixed (`2f60ea37` + correction) | **Yes** — `was_dph` semantics directly underpin B3's discrete/was_dph gradient scoping, referenced by name in `project_b3_analytic_gradient` | **likely** | `was_dph` exclusion is load-bearing for two separate B3 exact-gradient functions (moments-dph, joint-index) |
| `project_compute_path_atlas` | phasic computes each quantity via multiple backends (pybind/FFI/ctypes-JIT/trace); routing map at `audit-situation-map.md`; a bug can live in one path and not another | Live navigation instrument, not "done" | Implicitly — CLAUDE.md's architecture section describes the layers but not the "many paths per quantity, gated by flags" framing explicitly | **likely** | exactly the methodology needed to scope any further B3 batch (formula/hierarchical/daisy) correctly |
| `project_epoch_sojourn_initiative` | Epoch/daisy-chain joint-prob: surrogate abandoned; exact granularity-free final-epoch sojourn read (`joint_sojourn_graph`/`DaisyChainSojournFfiImpl`) shipped and made **DEFAULT** (`final_read='sojourn'`) | DONE | No — CLAUDE.md never mentions `daisy_chain_joint_probs`/epoch models in the B3 sections | **likely** | this is the now-default multi-epoch SVGD path, and it is **entirely untouched by B3** (verified, see §3.4) — the single biggest unflagged gap found in this audit |
| `project_custom_vmap_followup` | After the exposure/no-exposure daisy-chain `custom_vmap` fusion shipped, a multivariate retrofit was deferred; lists 6 other `custom_vjp` sites in `__init__.py` as future audit candidates | Item 1 (multivariate fusion) explicitly "not yet attempted" | No | **possible→likely** | the 6 listed `custom_vjp` line numbers are FD-gradient sites; this memory is effectively an independent index of FD sites that B3 could target, never cross-linked to B3 |
| `project_parameterized_path_nondeterminism` | Parameterized elimination wasn't bit-reproducible (pointer-address merge order); FIXED by merging on `->index` | Fixed (uncommitted-at-time-of-writing merge is now on master per `project_svgd_perf_ux`) | No explicit mention, but the fix underlies numerical stability of the same elimination tape B3's reverse-mode adjoint replays | **possible** | B3's C θ-adjoint walks the same "`_off`" elimination tape; worth confirming the fixed determinism code path and B3's new reverse-tape code don't have an analogous unfixed pointer-order issue anywhere they diverge |
| `project_sojourn_solve_memory` | `expected_sojourn_time == -solve(sim.T,ipv)`; memory dominated by cache-write buffer + symbolic trace; `PHASIC_DYN_ORDERING` gives -35% trace / 2.6x speed, gradient-safe (exact reordering) | Progress landed, some pieces uncommitted-at-time-of-writing | No | **possible** | `expected_sojourn_time` is exactly the quantity B3's joint-index adjoint differentiates; ordering/memory levers here are gradient-safe but interact with the same trace the adjoint walks |
| `project_parallel_elimination_scope` | `parallel_elimination=True` (hierarchical SCC composer) only covers `expected_waiting_time`/moments(rewards=None); `expected_sojourn_time` separately parallelized over reward columns (bit-identical) | Landed on branch `parallel-default-theta` (unclear if merged) | No | **possible** | if `parallel_elimination=True`, is the *tape* the B3 gradient functions replay still the same monolithic one, or the SCC-composed one? Not addressed by either memory or CLAUDE.md — ties into the "hierarchical SCC still FD-only" gap (§3.5) |
| `project_distributed_scc_arch` | SLURM path = `precompute_distributed`→`scc_worker`→per-SCC cache; EXTERNAL/`_ex` (WP-3) is dead code | Grounded, settled | No | **possible** | same hierarchical/SCC-and-gradients question as above, at the distributed-cluster scale |
| `project_svgd_lrt_canonical` | `deferred-svgd-lr-bug` closed; canonical `likelihood_ratio_test` handles tied-vs-free epoch pairs; explicitly notes "Issue 2 = FD→analytic gradient... not started" | Resolved for its own scope; the FD→analytic note is now **stale** (B3 did start and largely ship) | N/A (LRT mechanics not covered in CLAUDE.md's B3 sections) | **possible** | LRT's `refine=True`/`likelihood_ratio_test_at` runtime optimization may itself use SVGD-fit gradients; not verified whether it benefits from or is blocked by any B3 gap (e.g. daisy-chain FD-only) |
| `project_mcmc_svgd_consistency` | MCMC and SVGD should handle `observed_data` shape identically | Open goal, not started | No | **none** | **grounded**: `mcmc.py` uses only random-walk Metropolis (`grep` shows no `jax.grad` anywhere) — no gradient path exists to converge with B3 today |
| `project_test_suite_state` | Pre-existing test failures/hangs exist; don't use "full pytest green" as a gate; fixed the GIL-deadlock hang (`216fb558`) | Historical, largely resolved (hang fixed 2026-05-25) | Not explicitly, but CLAUDE.md's Tests section is consistent with it | **possible** | pure process/methodology relevance: governs how any future B3 fix must be verified (targeted subset, not full-suite green) |
| `project_stateindexer_slot_layout` | `StateIndexer` slot-ordering bug broke `joint_prob_graph` epoch offsets after `+`/`append`; fixed via unified `entity_order` | Fixed | No | **possible** | joint-prob/epoch state layout feeds directly into the B3 joint-index sojourn gradient's target-vertex indexing; a *recurrence* of a similar layout bug would silently corrupt gradient inputs, not just PMF values — worth a regression check, not re-derivation |
| `project_hex_grid_design` | `HexGrid` composes row/col `Property`s into `StateIndexer`; pure-function callbacks | Design decision, stable | No | **none** | confirmed — no gradient/inference surface touched |
| `project_docs_cpp_api_reference` | Doxygen+quartodoc C/C++ API docs pipeline; local sidebar quirk | Shipped (PR #31, unmerged at time of writing) | No | **none** | confirmed — pure docs tooling |
| `project_cpp_python_graph_parity` | C++ `phasic::Graph` method-name parity with Python, explicitly **excluding SVGD/inference** | "ALL requested C++ parity work is now complete" | No | **none** | confirmed by the memory's own scope exclusion |
| `project_install_dev_rewrites_deps` | `pixi run install-dev` rewrites `pyproject.toml`/`pixi.lock` as a side effect; exclude from feature commits | Standing gotcha | No (build-doc section doesn't mention it) | **none** | pure build/commit hygiene |

**Feedback memories** (process constraints, not open technical items) are summarized in §5, not the table above.

---

## 2. Note on the existing (non-memory) reachability atlas

While grounding §3.4, I found `/Users/kmt/phasic/atlas/exact-fd-atlas-svgd-reachability.md`
already in the repo (dated 2026-08-01/04, i.e. very recent — the same window as
`project_b3_analytic_gradient`'s last `modified` timestamp). It is **not a
memory file** and is not referenced by `MEMORY.md` or `CLAUDE.md`, but it
independently confirms and sharpens the biggest finding in this audit: **of
`Graph.svgd()`'s five dispatch leaves, only one can reach any B3 exact-gradient
path today**, and that's incidental (a callee default happening to be `True`),
not a deliberate design. Two of its findings go beyond what any memory file
captures:
- The **rewards-bearing SVGD leaves are dynamically excluded from the exact
  path on every single gradient step of the whole run** (not intermittently)
  — `SVGD` fixes `rewards` at construction and passes it unchanged into every
  forward/backward call, and the exact path's own (correct, intentional)
  "decline to FD when rewards present" guard therefore always fires.
- The **daisy-chain leaf has no exact-gradient implementation to select at
  all** — confirmed independently in this session, see §3.4.

Because this document lives outside the memory system, its findings risk being
lost if not folded into a memory or into CLAUDE.md — flagging that risk itself
as a process gap (§6, item 1).

---

## 3. Detailed findings (open content not captured in CLAUDE.md)

### 3.1 `ptd_graph_pdf_with_gradient` — unwired, untested, unverified sign (from `project_fd_gradient_b3`)

The memory flags: *"Suspect UNWIRED lead: `ptd_graph_pdf_with_gradient`
(`phasic.c:11805`, 'minus sign empirically determined' red flag) — re-derive
before trusting."* This was never followed up in any later memory or in
CLAUDE.md. I checked it **(grounded)**:

- The function still exists: `src/c/phasic.c:13090` (`int
  ptd_graph_pdf_with_gradient(...)`), declared in the public header
  `api/c/phasic.h:1467`.
- Its final gradient combination (`src/c/phasic.c` ~13215-13224) still carries
  the exact comment: *"NOTE: Empirically determined that the lambda gradient
  term should be SUBTRACTED. Mathematical analysis suggests this is because
  pmf_grad already accounts for λ dependence through the Poisson gradient
  term, creating a double-counting issue if we naively apply the product
  rule. The minus sign gives correct results."*
- `grep -rn "pdf_with_gradient" src/cpp/ src/phasic/` returns **nothing** —
  it is not called from the C++ layer, pybind, or Python anywhere.
- `grep -rln "pdf_with_gradient" tests/` returns **nothing** — it is untested.

So this is a real, dead, unreferenced C function computing a PDF gradient via
uniformization, whose only justification for a sign choice is "empirically
determined... gives correct results" rather than a derivation — sitting right
next to where B3's actual PDF-gradient work would need to go if the
"unguarded slow uniformization-cost band" gap (already flagged in CLAUDE.md)
or any future PDF-gradient batch is tackled. Not urgent (it's unreachable
dead code), but exactly the kind of landmine `feedback_never_assume_verify_adversarially`
warns about if someone later wires it up assuming it's already correct.

### 3.2 Per-call sojourn-gradient cost is "unmeasured" — but the measurement methodology already exists (from `project_svgd_perf_ux` + `project_cache_load_cpu_bound`)

CLAUDE.md's "B3 joint-index exact sojourn gradient" section says: *"the real
per-call cost/memory profile on production-scale graphs (`n` up to ~7×10^5)
is unmeasured"* for `ptd_sojourn_grad_theta_subset`'s fresh-every-call
`O(commands)` offset-tape conversion. Two memories not cross-referenced there
already did closely analogous measurement:

- `project_cache_load_cpu_bound` measured that deserializing/converting a
  saved trace is **CPU-bound, not I/O-bound**, and quantified load-vs-recompute
  crossovers by SCC size (56→~20-90ms, 224→~23s, 620→did not finish in 5min).
- `project_svgd_perf_ux` measured **per-θ inner-loop cost** directly on the
  joint/sojourn path (`pmf_from_graph_joint_index`→`expected_sojourn_time`)
  vs the moments/forward-PDF path, on the exact production two-locus model
  class this repo targets.

Both used harnesses (`scratch/scc_crossover.py`, `scratch/granularity_lever.py`,
`scratch/io_overlap_probe.py`) that could very plausibly be reused/adapted to
fill in the "unmeasured" cost CLAUDE.md flags, rather than building a new
benchmark from scratch. This connection is not made anywhere.

### 3.3 STALE: `weight_mode` silent-ignore bug is already fixed (from `project_weight_mode_log_semantics`)

The memory (22 days old) states as current fact: *"`moments_from_graph` and
`pmf_from_graph_joint_index` **silently ignore** `weight_mode` entirely and
always compute LINEAR weights — a log/callback/formula graph gets a silently
wrong answer, not an error."*

I checked this **(grounded)** and it is **no longer true as a silent bug**:

- `src/phasic/__init__.py:6682-6691` (`moments_from_graph`): explicit guard —
  `if _wm != 'linear': raise ValueError(...)`, with a comment: *"Silently
  returning linear moments for a 'log'/'callback'/'formula' graph is a WRONG
  ANSWER with no error... Fail loudly instead."*
- `src/phasic/__init__.py:7783-7792` (`pmf_from_graph_joint_index`): explicit
  guard for `weight_mode == 'log'` specifically (formula/callback ARE
  honoured on this path) — raises `ValueError` naming the exact same failure
  mode.
- Both guards were introduced by commit `d69919f2` ("fix(numerical):
  relative, mode-aware FD probe + guards against silent linearisation"),
  2026-07-12 — i.e. **before** this memory's own `modified` timestamp window,
  so the memory is describing a state that had already been fixed by the time
  it (or shortly after it) was written, or the fix landed right around when
  the memory captured the bug.

Net: the *silent-wrong-answer* failure mode is gone (now a loud, correctly
worded `ValueError`, consistent with `feedback_no_silent_fallbacks`). The
underlying **feature gap** — `moments_from_graph` has no non-linear
weight-mode support at all, and `pmf_from_graph_joint_index` has no `'log'`
support — remains open, and is consistent with (implicitly covered by)
CLAUDE.md's statement that the B3 moments adjoint covers only
`weight_mode='linear'`/`'log'` and that `moments_from_graph` is "an entirely
separate FD-only path" from the exact-grad work. This is a case of a memory
being **stale on the specific failure mode**, not on the underlying gap.

### 3.4 The daisy-chain / epoch SVGD path is 100% FD, untouched by B3, and is now the DEFAULT (from `project_epoch_sojourn_initiative`, corroborated by `project_fd_gradient_b3` and the reachability atlas in §2)

This is the single largest gap found in this audit, and it isn't stated
outright by any one memory — it only emerges from combining three of them
plus direct verification.

- `project_epoch_sojourn_initiative` (68 days old) records that the
  "exact, granularity-free final-epoch sojourn read" work landed, and that a
  later follow-up branch made `final_read='sojourn'` the **DEFAULT** for both
  `daisy_chain_joint_probs` and `Graph.svgd` for epoch/daisy-chain models.
- `project_fd_gradient_b3` (4 days old) documents the *original* FD defect
  class this whole B3 initiative exists to fix, and explicitly calls out that
  *"the DAISY path HARD-CRASHES"* under the old absolute-eps central
  difference.
- Neither memory, nor `project_b3_analytic_gradient` (the main B3 tracker),
  nor CLAUDE.md's two/three B3 follow-up sections, ever states whether the
  daisy-chain path got an exact-gradient option once B3 started shipping.

I checked directly **(grounded)**:
- `Graph._daisy_chain_svgd_model` (`src/phasic/__init__.py:4254`) and
  `Graph.daisy_chain_joint_probs` (`:10084`) have **no** `exact_grad`/
  `exact_moment_grad` parameter anywhere in their signatures or bodies
  (`grep -in "exact"` over the full ~750-line `_daisy_chain_svgd_model` body
  returns nothing but an unrelated comment).
- Both branches (`:4668-4692` no-exposure, `:4900-4932` per-obs/exposure) wrap
  their forward in `@jax.custom_vjp` with a hand-written **central-difference
  backward at absolute `eps=1e-7`** (`:10326-10361` for
  `daisy_chain_joint_probs` — comment literally says *"Wrap the forward in a
  custom_vjp so jax.grad works via finite differences... eps=1e-7 matches the
  established pattern"*) — i.e., the exact defect class B3 was created to
  replace, still live, unconditionally, on what is now the default entry
  point for multi-epoch models.

This is independently confirmed by the (non-memory) reachability atlas in §2,
which traced `Graph.svgd()`'s dispatch tree and found the daisy-chain leaf has
"no exact-gradient implementation of any kind to plumb through — it is FD by
construction, full stop." **None of CLAUDE.md's B3 sections mention the
daisy-chain/epoch path at all.**

### 3.5 Hierarchical-SCC gradient status is unclear and unflagged in CLAUDE.md (from `project_b3_analytic_gradient`, `project_parallel_elimination_scope`, `project_distributed_scc_arch`)

`project_b3_analytic_gradient`'s Batch-3 notes explicitly list *"hierarchical
SCC"* as one of three remaining FD-only scopes (alongside log/formula, which
DID ship, and joint-index, which DID ship). I grepped CLAUDE.md: **the word
"hierarchical" never appears in either B3 follow-up section** — only in the
unrelated architecture description of `hierarchical_trace_cache.py`. So a
reader of CLAUDE.md alone would not know that `parallel_elimination=True` /
SCC-composed graphs are excluded from every B3 exact-gradient function (or
whether they are — this was never explicitly tested per the memory, only
listed as a remaining batch). `project_parallel_elimination_scope` and
`project_distributed_scc_arch` add relevant substrate (which functions
parallelize, how the distributed cache populates) but neither addresses
whether the *tape* a parallel/hierarchical build produces is the same one the
B3 reverse-mode adjoint functions expect to walk.

---

## 4. Feedback memories relevant to how future B3 work should proceed

Quick-reference checklist, ordered as they'd apply during a B3 follow-up pass:

1. **`feedback_derisk_and_reevaluate`** — before committing to any new B3
   batch (daisy-chain exact grad, hierarchical, formula/callback), run
   targeted de-risking experiments on a branch first; detail batches from
   findings; re-verify each batch's assumptions against current code before
   starting it (assumptions go stale fast in this codebase — §3.3 is a live
   example).
2. **`feedback_batch_plan`** — divide any such plan into sequential batches,
   each with its own test gate, per the pattern B3 itself already used
   (Batch-0 through Batch-3, MPFR gate, discrete de-risk, log-mode, joint-index).
3. **`feedback_never_assume_verify_adversarially`** — B3's own history is the
   proof case: the `exact_moment_grad` default-flip passed its own green
   regression suite, and still had two real bugs (rewards silently ignored;
   grad-clip median collapse) that only 3 independent adversarial review
   passes caught. Any daisy-chain/hierarchical/formula exact-gradient batch
   must get the same treatment before merging or defaulting on.
4. **`feedback_no_modify_existing`** / **`feedback_no_change_svgd`** — purely
   additive changes only; this is explicitly why the "reverse-tape skeleton
   duplication" refactor (three near-identical C functions) was *not* done
   unilaterally even though a review recommended it — CLAUDE.md itself notes
   this requires asking first. Any daisy-chain exact-gradient work must add a
   new code path, not touch the existing FD `custom_vjp`.
5. **`feedback_no_silent_fallbacks`** — any new exact/FD dispatch must be an
   explicit, loud choice (as `exact_moment_grad`/`exact_grad` already are,
   INFO-logged on fallback) — never an implicit branch on argument presence.
   Directly relevant if `Graph.svgd()` is ever given a top-level exact-grad
   kwarg (per §2's reachability atlas, it currently has none at all).
6. **`feedback_ground_code_claims`**: — state explicitly what's read-from-source
   vs. recollection; this audit found one memory (§3.3) whose central factual
   claim had gone stale, which is exactly the failure mode this discipline
   guards against.
7. **`feedback_avoid_matrix_exp`** — if PDF-gradient work ever revisits
   `ptd_graph_pdf_with_gradient` (§3.1) or the "slow uniformization band" gap,
   stay with graph/elimination/uniformization, not dense `expm`/Krylov.
8. **`feedback_plans_in_repo`** / **`feedback_handoff_plans_self_contained`** —
   any new B3 batch plan goes to `/Users/kmt/phasic/<name>-plan.md` (matching
   the existing `b3-*-plan.md` convention) and must be self-contained enough
   for a fresh session with zero conversation context.
9. **`feedback_ipv_not_optimized`** — if daisy-chain exact-gradient work is
   ever attempted, IPV remains a fixed setter, never a gradient/SVGD
   parameter; `jax.grad` is over `epoch_thetas` only.

---

## 5. Prioritized list: open items most likely to matter for B3/exact-FD work

Ranked by (a) how directly gradient-correctness/coverage is implicated and
(b) how surprising/un-flagged the gap is relative to CLAUDE.md's current
documentation.

1. **Daisy-chain/epoch SVGD path has zero exact-gradient coverage and is now
   the DEFAULT multi-epoch inference path** (§3.4). Highest priority: this is
   exactly the FD defect class (absolute-eps central difference) B3 exists to
   fix, on what appears to be the primary path for this repo's actual
   population-genetics workloads (epoch/migration models), and it is
   completely unmentioned in CLAUDE.md's B3 sections.
2. **In practice, almost none of the SVGD dispatch tree reaches any shipped
   B3 exact path** (§2, corroborating non-memory atlas). Even where exact
   kwargs exist and default to `True`, `Graph.svgd()` never plumbs them, and
   two of five leaves are *dynamically* excluded on every single call
   (rewards present) regardless of default. This means the "exact_moment_grad
   defaults to True" story in CLAUDE.md, while accurate at the function
   level, may be **misleading about real-world impact** unless the caller
   bypasses `Graph.svgd()` and hand-builds the model.
3. **Hierarchical/SCC-parallelized graphs' gradient status is unverified and
   unflagged** (§3.5) — explicitly listed as a remaining B3 scope item in the
   tracking memory, silently dropped from CLAUDE.md's current text.
4. **`ptd_graph_pdf_with_gradient`** (§3.1) — dead, untested C code with an
   "empirically determined" sign in a PDF-gradient computation; low urgency
   (unreachable) but a landmine if ever wired up on the strength of "it's
   already there."
5. **Formula/callback weight modes remain FD-only for moments and (for
   gradients specifically) joint-index** — captured reasonably well in
   CLAUDE.md by omission (only linear/log listed as in-scope), but worth
   restating as an explicit "not supported" rather than inferred.
6. **`pmf_and_moments_from_graph_multivariate` has no `exact_moment_grad`
   passthrough**, and per the reachability atlas its 2D-rewards SVGD leaf is
   *also* dynamically FD-excluded on every call regardless — CLAUDE.md
   flags the missing kwarg but not the compounding dynamic exclusion.
7. **Unmeasured per-call cost of the joint-index sojourn gradient's offset-tape
   conversion** (§3.2) — CLAUDE.md flags it as unmeasured; two other memories
   already built directly reusable measurement harnesses for exactly this
   class of cost.
8. **Reverse-tape skeleton duplication** (three ~150-line near-identical C
   functions) — already well-flagged in CLAUDE.md as deliberately deferred
   pending sign-off; listed here only for completeness since a 4th
   weight-mode variant (formula) or the daisy-chain work above would be a
   natural trigger to finally ask about it.
9. **Parameterized-path non-determinism fix and the B3 reverse-tape adjoint**
   (§ table row) — plausible but unconfirmed overlap; worth a quick check
   that the fixed pointer-order merge and the new B3 C functions don't
   diverge on this point anywhere.
10. **`weight_mode` silent-ignore bug is fixed, not open** (§3.3) — included
    here only as a negative finding: don't re-fix what's already fixed; the
    *feature gap* (no non-linear support in `moments_from_graph`) is the
    actually-open part, and it's already indirectly captured in CLAUDE.md.
