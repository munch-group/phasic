# Feasibility scoping: `rewards` support in the B3 exact moment-gradient adjoint

**Status:** scoping only, read-only investigation. No code changed. All line
numbers are from a fresh read of the current `master` tree
(commit `cadf1ca4` at time of writing) on 2026-08-05, not from the atlas docs
(which were read first as a hypothesis, then re-verified against source; two
of their framings turned out to need correction — see Finding 1 below, the
single most important result of this pass).

## 0. Headline verdict

**Feasible, and still small in code-diff terms, but NOT the one-line
seed-swap the task's own framing hypothesized.** The correct fix requires
inserting the *same* elementwise `× rewards` operation at **two** specific
lines inside the shared stage-0/stage-1 "moment chain" loop that all three
C functions share — not at the `seeds[v]=1.0` initialization line. This was
proven empirically (§1) with the currently-built package: seeding `a_0` with
`rewards` and otherwise leaving the K-stage replay loop untouched reproduces
the reward-weighted **first** moment correctly but is **silently wrong** for
the second (and higher) moment. This is exactly the kind of defect the B3
effort has hit before (rewards silently ignored, commit `315ce9c8`) and the
project's adversarial-review culture exists to catch before it ships again.

Net effort estimate: **~2 new lines × 3 C functions (6 lines) for the core
math**, plus mechanical parameter threading (new `rewards`/`rewards_len` arg
through 3 C signatures, 3 C++ header methods, 3 pybind `.def()`s, and the
Python host-callback + `pure_callback` call site) — call it a half-day to a
day of focused work for the C+C++/pybind layer, plus a comparable amount for
the Python wiring and gates. Low-to-moderate risk, contingent on the
cross-batch sequencing note in §5, which is the most important scheduling
constraint found.

---

## 1. The primal: how rewards flow through `ptd_expected_waiting_time` and `Graph.moments`

`ptd_expected_waiting_time(graph, rewards)` (`src/c/phasic.c:9980-9980+`,
body through ~10202): when `rewards != NULL` it seeds the **same** numeric
elimination replay ("reward compute graph", `graph->reward_compute_graph`)
with `result[j] = rewards[j]` instead of `result[j] = 1` (lines 10049-10056),
then replays the identical command list either way (`result[from] +=
result[to] * multiplier`, lines 10147-10194). So a single call is: **one
linear-solve-via-elimination-replay, seeded by an arbitrary per-vertex
vector.** This part of the task's starting hypothesis is correct.

But `Graph.moments(power, rewards)` — the actual oracle, bound via pybind
`.def("moments", &_moments, ...)` (`src/cpp/phasic_pybind.cpp:1700`, free
function at `phasic_pybind.cpp:212-271`; a byte-identical copy also lives as
`phasic::Graph::moments` in `api/cpp/phasiccpp.h:635-674`, used by the C++
API surface) — does **not** just call `expected_waiting_time` once per
moment order with the previous output threaded straight through. Read
verbatim (`phasiccpp.h:650-671`):

```cpp
std::vector<double> rewards2 = this->expected_waiting_time(rewards);   // a_1 = EWT(rewards)
res[0] = rewards2[0];
for (int i = 1; i < power; ++i) {
    if (!rewards.empty()) {
        for (j) rewards3[j] = rewards2[j] * rewards[j];   // <-- elementwise RE-multiply by the ORIGINAL rewards, every step
    } else {
        rewards3 = rewards2;
    }
    rewards2 = this->expected_waiting_time(rewards3);      // a_{i+1} = EWT(a_i .* rewards)
    res[i] = factorial(i+1) * rewards2[0];
}
```

Rewritten uniformly with `a_0 = ones` (matching the K-stage moment-chain's
own `seeds[v]=1.0` convention exactly): for every stage `j = 1..K`,
`seed_j = a_{j-1} .* rewards`, `a_j = replay(seed_j)`. When `rewards` is the
default all-ones vector, `a_{j-1} .* ones == a_{j-1}`, which is exactly the
no-op the current C functions implement — **the current implementation is
the `rewards≡ones` special case of a more general recursion**, and the
missing general case is precisely the elementwise reweight-by-`rewards` at
*every* stage transition, not just at `a_0`.

**Empirical confirmation** (ran against the already-built package, no
rebuild): a 4-vertex continuous parameterized chain, `theta=[2.0]`,
`rewards=[0, 1, 2, 0.5]`:

```
Graph.moments(2)                          -> [1.0, 1.5]
Graph.moments(2, rewards=rewards)         -> [1.5, 3.5]     <- ground truth
naive "seed a_0=rewards, then chain plain EWT(a_1)" for 2nd moment -> 2.5   <- WRONG
correct "seed_2 = a_1 .* rewards, then EWT(seed_2)" for 2nd moment -> 3.5   <- matches ground truth
```

This settles it: a naive C patch that only changes the `seeds[v]=1.0`
initialization line (as the task's own framing suggested) would be correct
for `nr_moments=1` and **silently wrong for `nr_moments>=2`** — it would pass
any test that only checks the first moment/expectation and fail (without
erroring or declining) on variance/higher moments. This is the single most
important finding of this pass.

Length validation for the primal is NOT done inside the raw C function
(`ptd_expected_waiting_time` just `memcpy`s `vertices_length` doubles out of
whatever pointer it's given — an out-of-bounds read if the caller passes a
short array). It is done one layer up, in C++: `phasiccpp.h:442-450`
(`expected_waiting_time`, inline check) and the `_check_reward_length`
helper (`phasiccpp.h:433-440`, used by `covariance`/etc.), both throwing
`std::runtime_error` on mismatch. `_moments`/`Graph.moments` itself also
validates inline (`phasic_pybind.cpp:214-228`, `phasiccpp.h:636-642`). On the
Python side, `pmf_and_moments_from_graph` has its own independent guard,
`_check_rewards_len` (`src/phasic/__init__.py:7053-7061`), checking that the
last axis of `rewards` equals `serialized['n_vertices']`; called at the top
of every `_compute_pure` variant (`__init__.py:7094`, `7268`, `7342`).

## 2. All three C functions: exactly where `seeds`/`seed` is touched

Read in full: `ptd_moments_grad_theta` (`src/c/phasic.c:10738-10881`),
`ptd_moments_grad_theta_dph` (`:11142-11338`), `ptd_moments_grad_theta_log`
(`:10917-11063`). All three share an identical stage-0 (forward moment
chain + MPFR gate) / stage-1 (per-`outk` reverse chain) skeleton, verified
line-by-line — the in-code comments at `:10883-10889` and `:11121-11127`
explicitly say so, and diffing the three confirms it byte-for-byte outside
the edge→theta contraction step.

Every occurrence of `seeds`/`seed` in each function:

- `seeds[v]=1.0` (linear `:10792`, dph `:11230`, log `:10968`) — the `a_0`
  baseline. **This does not need to change at all** — per §1's derivation,
  `a_0` stays `ones`; the reward vector enters at every subsequent stage
  transition, not here.
- `for (v) out[v]=seed[v];` (linear `:10795`, dph `:11233`, log `:10971`) —
  this is the line that needs to become
  `out[v] = seed[v] * (rewards ? rewards[v] : 1.0);` — the per-stage
  elementwise reweight identified in §1.
- `for (v) bar_out[v]=adj[v];` at the end of each reverse-chain `j`
  iteration, *inside* the per-`outk` loop (linear `:10827`, dph `:11264`,
  log `:11002`) — the adjoint counterpart. Since `out = seed .* rewards` is
  a diagonal (elementwise-scale) linear map, its VJP is multiplication by
  the same `rewards` vector: this line needs to become
  `bar_out[v] = adj[v] * (rewards ? rewards[v] : 1.0);` to keep the
  reverse-mode pass consistent with the now-reweighted forward pass. `rewards`
  is a theta-independent constant, so no new `dm[]`/theta contribution is
  needed from this step — it is a pure rescale, not a new gradient term.

**Nothing else needs to change.** Verified explicitly, reading every line
that touches the tape/theta pipeline in each function:

- `na[]`/`nb[]`/`nm[]` (the extracted from/to/multiplier arrays) are built
  once from `off->commands` *before* the K-stage moment-chain loop even
  starts (linear `:10766-10779`) — purely a function of theta/topology,
  untouched by rewards.
- The MPFR gate, `ptd_dbg_tape_needs_mpfr(nm, nc)` (linear `:10783`, dph
  `:11221`, log `:10960`), only examines `nm[]` — confirmed reward-blind,
  and this matches the primal's own condition-number pre-scan
  (`ptd_expected_waiting_time:10063-10083`, which also only scans
  `graph->reward_compute_graph->commands[j].multiplier`, never `rewards`).
  So the exact-gradient's MPFR decline policy needs **no change** to stay
  consistent with the primal — this is pre-existing, reward-blind scope
  inherited unchanged, not a new defect (though see the residual risk
  flagged in §6).
- Stage-2 (param-tape reverse, producing `binp[]`) and the edge→theta
  contraction (linear `:10850-10870`; dph's was_dph quotient-rule branch
  `:11314-11320` and plain-linear branch `:11321-11323`; log's product-rule
  branch `:11049-11051`) all consume only `binp[k]`/`e->coefficients`/`Sv`/
  `SigmaCv`/`theta[j]` — none of which depend on rewards. Once `dm[]` is
  correctly accumulated across the K reweighted stages (via the two-line
  fix above), everything downstream is unaffected. **No new math needed for
  the was_dph renormalization quotient rule or the log-mode product rule —
  they are orthogonal to the rewards question**, contradicting the
  "does this need new math" concern the task raised as a live possibility;
  it does not, for these two branches specifically.
- The final `isfinite` sweep(s) (linear `:10872`; dph does two, `:11326`
  pre-correction and `:11328` post-correction; log `:11054`) are unaffected
  in structure — they still just check the final `J_out` buffer.
- `ptd_dph_correct_discrete_moment_grad` (`:11094-11119`) is a fixed
  combinatorial linear map (Stirling numbers / binomials / factorials)
  applied to the K-length gradient-vs-theta columns *after* the edge→theta
  contraction. Its own doc comment (`:600-604` in
  `_continuous_to_discrete_moments`, the value-level Python analog) grounds
  its validity in "`U=(I-P)^{-1}` commutes with `P`" — a property of the
  embedded-chain structure, not of what vector is being propagated through
  `U`. This is evidence the correction generalizes to reward-weighted
  moments without new math, but it was **not independently re-derived or
  numerically re-verified against a reward-weighted ground truth** in this
  pass — flagged as a specific, low-but-nonzero-confidence item for the
  `_dph` function's de-risk gate (§6), separate from the linear/log
  functions where no such correction step exists at all.

## 3. Shape/length validation for the new C parameter

Primal precedent (§1): the raw C function does not itself validate length
(it just reads `vertices_length` doubles); validation is enforced one layer
up (C++ `expected_waiting_time`/`_check_reward_length`, throwing) and again
in Python (`_check_rewards_len`, raising `ValueError`). The three exact-grad
C functions, however, already establish a *different*, more defensive
convention for their own new parameters: `ptd_moments_grad_theta_dph` and
`_log` both take `(theta, theta_len)` and explicitly `return -1` on
`theta_len != P` (`:11147`, `:10923`) — a STATIC decline inside the C
function itself, not a throw one layer up.

**Recommendation, matching the existing convention of *this* file rather
than the primal's convention:** add `(rewards, rewards_len)` to all three
signatures and decline (`return -1`) when `rewards_len != 0 &&
rewards_len != graph->vertices_length` — treating `rewards_len==0` /
`rewards==NULL` as "no rewards" (all-ones), mirroring
`ptd_expected_waiting_time`'s own `rewards==NULL` sentinel exactly. This
keeps the new parameter's validation self-contained in C (consistent with
how `theta_len` is already checked in the same functions) while the
Python/C++ layers keep their own independent checks as defense-in-depth
(already-existing `_check_rewards_len`, no change needed there).

## 4. Python wiring (`src/phasic/__init__.py`, `pmf_and_moments_from_graph`, current range 6822-7649)

Read in full. Confirmed current line numbers (shifted only slightly from
the atlas's 2026-08-04 read):

- `_exact_graph = graph.clone()` — `:6992`. **Does not need to be
  "reward-transformed" at construction** — this is a correction to the
  task's own framing. Rewards is not a structural graph property anywhere
  in the primal call chain (§1); it is a plain runtime array passed at
  *call* time to `expected_waiting_time`/the grad functions, exactly like
  `theta` is passed to `update_weights`/`_moments_grad_theta_dph`. There is
  no analog of "reward-transform the clone once" to mirror — the clone only
  ever needs `update_weights(t, ...)` (already done, `:7012`) plus the new
  `rewards` array passed straight through to the C call at evaluation time.
- `_exact_moments_jac_np`'s `_one(t)` closure (`:7004-7033`) calls
  `_exact_graph._moments_grad_theta_dph(_exact_K, t.tolist())` (`:7015`),
  `..._log(_exact_K, t.tolist())` (`:7019`), `..._moments_grad_theta(_exact_K)`
  (`:7022`) — confirmed, none pass rewards, matching the atlas.
- `_rewards_provided` dynamic-decline guard, `model_bwd` (`:7534-7622`):
  `_rewards_provided = rewards is not None and jnp.asarray(rewards).size > 0`
  (`:7559`); `if _exact_grad_enabled and _rewards_provided:` logs at INFO
  and leaves `_exact_tbm = None`, forcing pure FD for the moments term
  (`:7561-7566`). Confirmed exactly as the atlas describes, and confirmed
  live: running `pmf_and_moments_from_graph(..., exact_moment_grad=True)`
  against the already-built package with a concrete `rewards` array produces
  a different (FD) gradient value than the no-rewards call, consistent with
  this dynamic decline actually firing (source-level confirmation is the
  primary grounding here; the INFO log itself did not surface in the quick
  REPL check due to `phasic`'s own logging config, not re-investigated
  further since the branch logic was already confirmed by direct reading).

**What must change:**
1. Thread `rewards` as a genuine new argument down the C→C++→pybind chain:
   `ptd_moments_grad_theta(graph, nr_moments, rewards, rewards_len, J_out)`,
   `phasiccpp.h::moments_grad_theta(int nr_moments, std::vector<double> rewards={})`
   (currently `:560-567`, no rewards param), similarly for `_dph`
   (`:575-583`) and `_log` (`:591-599`) adding `rewards` alongside the
   existing `theta` parameter. Three corresponding `pybind` `.def()` sites
   need a new `py::arg("rewards")=std::vector<double>()`
   (`src/cpp/phasic_pybind.cpp:1915`, `1919`, `1926`).
2. `_exact_moments_jac_np`/`_one` need a `rewards` parameter. Crucially,
   **rewards is a genuine per-call runtime value** (it can differ between
   successive calls to the same built `model`, exactly like `theta` does),
   so it cannot simply be closed over at construction the way `_exact_K`/
   `_exact_is_log` are — it must cross the `jax.pure_callback` boundary as
   an actual array argument, the same pattern already used one function
   over for a different non-theta per-call argument:
   `pmf_from_graph_joint_index`'s `_exact_sojourn_jac_np(theta_np,
   vertex_indices_np)` crossing `theta, _vi_norm` at `:8248-8254`. That is
   a directly reusable, already-in-this-codebase precedent for exactly this
   shape of problem (empty-array sentinel for "no rewards", matching the
   existing `rewards_jax = jnp.array([], dtype=jnp.float64)` idiom already
   used in the callback-mode `_compute_pure` at `:7117-7120`).
3. `model_bwd`'s `_rewards_provided` branch (`:7561-7566`) changes from
   "always decline" to "build the rewards-aware exact callback and use it,
   still gated by the existing `_exact_ok = isfinite` sentinel for per-theta
   declines." The existing 1D `_check_rewards_len` shape guard already
   covers the length check before this point in every call path.
4. **Free side-effect fix:** `Graph.pmf_and_moments_from_graph_multivariate`
   (`:8283-8477`) already loops per-feature calling
   `model_1d(theta, times_j, rewards=reward_j)` with a concrete 1D
   `reward_j` slice every time (`:8408-8422` sparse, `:8437-8452` dense).
   Per the atlas (confirmed unchanged by this pass), this is *why* the
   multivariate wrapper's exact path is dynamically dead today — fixing the
   1D-rewards case in `pmf_and_moments_from_graph` automatically un-breaks
   `pmf_and_moments_from_graph_multivariate`'s reachability too, with no
   separate code change needed in the multivariate wrapper itself.

## 5. Cross-batch conflict check (the single most important scheduling constraint)

**Direct collision with batch (a) (formula-mode + shared-skeleton refactor)
is real, specific, and line-level.** The rewards fix (§2) touches exactly
two lines inside the stage-0/stage-1 skeleton — `for(v) out[v]=seed[v];`
and `for(v) bar_out[v]=adj[v];`, both inside the per-level loop — in **all
three** of `ptd_moments_grad_theta`, `_dph`, `_log`. This is precisely the
~150-line duplicated skeleton CLAUDE.md's "Reverse-tape skeleton
duplication" section already flags as a refactor candidate ("extracting the
shared core into one static helper... before any further weight-mode
variant is added"). Two concrete bad outcomes if these run uncoordinated:

- If batch (a) extracts the shared skeleton into one static helper *while*
  this rewards batch is independently patching the three still-separate
  copies, whichever lands second either merge-conflicts outright or (worse,
  silently) gets applied against source that no longer exists in that
  shape — reproducing exactly the "fix-lineage divergence" pattern already
  documented in `atlas/exact-fd-atlas-c-functions.md` §"Cross-cutting
  observations" #1 (MPFR gate / `coefficients_length` guard / alloc checks
  were each independently fixed in one function and never backported to
  the "superseded" `ptd_moment0_grad_theta`, which still carries both gaps
  today).
- If batch (e) (`weight_mode='callback'` exact adjoint) or batch (a)'s
  formula-mode variant is built by copy-pasting one of the three existing
  functions as a template — exactly how `_dph` and `_log` were each built
  from `ptd_moments_grad_theta` (per their own header comments,
  `:11121-11127`, `:10883-10889`) — then whichever of {rewards-support,
  formula/callback-mode} lands *second* should copy from an
  *already-reward-aware* template, or the new 4th/5th copy will ship
  without rewards support and need the same fix applied a 4th/5th time
  later (the exact repeated-patch pattern the atlas's cross-cutting
  observation #1 already documents happening historically).

**Recommendation:** do not run these two batches as blind-parallel. Either
(i) land rewards-support first (it is small, additive, does not touch
weight-mode dispatch, and does not add a 4th tape variant, so it does not
trip CLAUDE.md's "wait before a 4th variant" caution) and have batch (a)'s
refactor/formula-mode work consolidate the now-reward-aware pattern into
the shared helper, verified via the existing gates (`dr_moments_jac_gate.py`,
`dr_mpfr_gate_test.py`, `dr_dph_moments_jac_gate.py`,
`dr_log_mode_moments_jac_gate.py`, re-run as value-identical checks per
CLAUDE.md's own suggested verification), or (ii) if batch (a)'s refactor is
prioritized first, design the rewards `× reward[v]` hook into the unified
skeleton from the start rather than bolting it on afterward. Either order
is fine; running them uncoordinated on the same three functions is not.

Other four batches, briefly:

- **(b) `Graph.svgd()` plumbing + joint-index baked-mode:** no direct code
  collision (different file regions — `Graph.svgd` is ~`:5241-6120`, this
  batch's changes are `:6822-7649`) but a real *logical* dependency: fixing
  `Graph.svgd`'s two rewards-bearing leaves (atlas Leaf C/D) to actually
  reach the exact path requires this batch to ship first (or in the same
  release) — otherwise (b)'s plumbing work has nothing to plumb to.
- **(c) hierarchical/SCC tape compatibility:** structurally disjoint today
  — the primal's own hierarchical/composer path (`:10004`,
  `PHASIC_HIERAR_ELIMINATION`) is explicitly gated `rewards == NULL`
  (`:9989-9990`, "Reward-transformed cases fall through to the monolithic
  path"), and all three exact-grad functions already always rebuild the
  monolithic tape every call regardless (confirmed, `:10743-10746` /
  `:11181-11184` / `:10924-10927` — never touch the SCC/composer path at
  all). If (c) later adds an SCC-aware branch to these same three
  functions, it will need to decide how rewards interact with that new
  branch (most likely: mirror the primal's own policy and require
  `rewards==NULL` for the composed path, monolithic fallback otherwise) —
  a small, deferred coordination note, not a blocker now.
- **(d) PMF/PDF gradient re-derivation + daisy-chain:** no overlap — a
  different quantity (PMF/PDF via uniformization,
  `ptd_graph_pdf_with_gradient`) and different C functions entirely, already
  ruled out of the B3 `*_grad_theta*` lineage by the c-functions atlas.
  Independent.
- **(e) `weight_mode='callback'` + MPFR-precision "conditioning floor"
  adjoint:** same collision class as (a) if it adds a new
  `ptd_moments_grad_theta_callback` copying the existing skeleton (see
  above); its MPFR-precision-floor work also touches the shared
  `ptd_dbg_tape_needs_mpfr` gate mechanism used by all three functions,
  though that gate is itself reward-blind by design (§2) so there is no
  direct rewards-specific conflict, only the general "many hands editing
  the same skeleton" risk this section is about.

## 6. Risk / unknowns list

1. **(Resolved by this pass, high confidence)** The seed-only framing is
   wrong for `nr_moments>=2`; the correct fix is the two-line
   elementwise-reweight-per-stage change in §2, empirically verified
   against the built package in §1.
2. **(Low-moderate confidence, not verified numerically here)** Whether
   `ptd_dph_correct_discrete_moment_grad`'s combinatorial
   continuous→discrete correction remains valid *unchanged* when applied to
   reward-weighted moment-gradients rather than standard ones. The
   value-level analog (`_continuous_to_discrete_moments`,
   `__init__.py:590-633`) documents its validity via a
   graph/vector-independent commutativity argument (`U` and `P` commute),
   which is suggestive that it generalizes cleanly, but this was not
   independently re-derived or gated in this pass. Recommend a dedicated
   `dr_*.py` gate (reward-weighted discrete moments-gradient, exact vs. FD)
   before shipping the `_dph` function's rewards support specifically —
   the linear and log functions have no such correction step and so carry
   no equivalent risk.
3. **(Pre-existing, inherited, not new)** The MPFR escalation gate (both
   primal and all three exact-grad functions) is blind to reward magnitude
   — it only pre-scans elimination multipliers, never the reward vector
   itself. An extreme-dynamic-range `rewards` vector could, in principle,
   degrade the K-stage moment-chain's numerical precision in ways the
   existing gate would not catch (compounding ill-conditioning across K
   reweight-and-replay stages, invisible to a multiplier-only condition
   check). This is not a regression introduced by adding rewards support —
   the same blind spot already exists for the *primal* `Graph.moments`
   today — but it's worth flagging since rewards support is what would
   newly expose users to it via the exact-gradient path specifically (FD
   has no equivalent precision concern beyond its usual step-size
   trade-off).
4. **Allocation/size-guard debt is unrelated but adjacent.** All three
   functions already have zero NULL-checked allocations and no size guard
   on `L`/`nc`/`n` (confirmed unchanged from the c-functions atlas). Adding
   one more `rewards`-length array to allocate/copy does not make this
   worse, but does not fix it either — out of scope for this specific
   feature, flagged only so it isn't mistaken for something this batch
   should absorb.
5. **Target vertex / IPV.** `target=0` is hardcoded identically in all
   three functions (confirmed, `:10753`, `:11195`, `:10934`) and is
   unaffected by rewards (rewards weight the *seed*, `target` selects the
   *read-out* vertex) — no interaction found, listed here only because the
   task asked whether anything besides the seed line needed inspection.

## 7. Effort/risk verdict

**Small-to-moderate, well-understood once the §1 correction is accounted
for.** Concretely:

- C layer: ~6 lines of real math change (2 lines × 3 functions) plus
  mechanical parameter/decline-check additions (`rewards`, `rewards_len`)
  to all three signatures, each roughly doubling the existing `theta_len`
  check pattern already present in `_dph`/`_log` (`ptd_moments_grad_theta`
  itself takes no `theta` argument today and would gain its first
  additional parameter).
- C++/pybind layer: 3 header method signatures
  (`api/cpp/phasiccpp.h:560-599`) + 3 `.def()` sites
  (`src/cpp/phasic_pybind.cpp:1915-1930`) gain a `rewards` default-arg,
  mirroring `expected_waiting_time`'s own existing pattern.
- Python layer: `_exact_moments_jac_np`/`_one` gain a `rewards` parameter
  threaded across the `pure_callback` boundary (precedented by the
  joint-index function's `vertex_indices` handling), and the
  `_rewards_provided` guard (`:7561-7566`) becomes a real dispatch instead
  of an unconditional decline. `pmf_and_moments_from_graph_multivariate`
  needs no change of its own to benefit.
- Test/gate burden: extend `dr_moments_jac_gate.py`,
  `dr_dph_moments_jac_gate.py`, `dr_log_mode_moments_jac_gate.py` with
  reward-weighted cases (FD vs. exact, matching the existing gate style),
  plus a pytest addition to `tests/pytest/inference/test_exact_grad_*`
  covering `nr_moments>=2` with non-uniform rewards specifically (this is
  the exact case that would have caught the naive-seed bug — no existing
  gate in the current suite exercises reward-weighted moments through the
  exact path, since the path is unconditionally declined today).

No fundamentally new math is needed beyond the elementwise-reweight
identified in §1-2; no interaction was found with the was_dph
renormalization quotient rule, the log-mode product rule, or the MPFR gate.
The one open mathematical question (§6.2, the discrete moment correction
under reward-weighting) is plausible-but-unverified, not a known blocker.

## Recommended batch sequence position

**Independent of (d); should be sequenced explicitly (not blind-parallel)
relative to (a) and (e) per §5 — either strictly before them (preferred:
small, additive, doesn't add a 4th tape variant, gives (a)/(e) an
already-reward-aware template to consolidate/copy from) or strictly after a
coordinated design that bakes the reweight hook into whatever shared
skeleton (a) produces. Should land before or alongside (b), since (b)'s
whole purpose (making `Graph.svgd`'s rewards-bearing branches reach the
exact path) has nothing to plumb to until this batch ships. No dependency
on (c) today; (c) will need a small follow-up coordination note if/when it
adds an SCC-aware branch to these same three functions.**
