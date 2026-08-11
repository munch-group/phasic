# Feasibility: exact theta-adjoint gradients over the hierarchical/SCC elimination path

**Purpose.** Scoping-only investigation (no implementation). Answers whether the four
existing B3 exact-gradient C functions (`ptd_moments_grad_theta`, `_dph`, `_log`,
`ptd_sojourn_grad_theta_subset`, all in `src/c/phasic.c`) work, could trivially be
made to work, or fundamentally cannot work, against the SCC-decomposed/hierarchical
elimination path (`parallel_elimination=True` / `PHASIC_HIERAR_ELIMINATION=1`). All
claims below are grounded in a fresh read of current source this session
(2026-08-05); line numbers are cited and were spot-verified against the live file
(not copied from the two background atlases, which were read first but re-verified,
per the task instruction not to trust them).

**Headline verdict (Q3): NO — `graph->parameterized_reward_compute_graph_off` (and
`parameterized_reward_compute_graph`) is never populated on the parent graph by the
hierarchical/SCC path, for any configuration.** The hierarchical path bypasses
`ptd_precompute_reward_compute_graph` entirely and produces a **pure numeric**
per-vertex result vector via a composition step that overwrites edge weights with
plain `double`s — there is no persistent, differentiable, top-level tape object at
any point. This is not a caching/plumbing gap that could be closed by "just also
populating the field" — the composition step's core operation (a reciprocal of a
downstream SCC's *already-computed numeric result*, wired in as a raw edge-weight
write) has no representation in the existing tape-input encoding
(`PTD_PCG_PTR_MEM`/`PTD_PCG_PTR_EDGE`/`PTD_PCG_PTR_EXTERNAL`) at all. See §2 for the
full mechanism and §4 for why this means new math, not new plumbing.

---

## 1. The monolithic baseline (what the 4 existing functions assume)

`ptd_precompute_reward_compute_graph` (`src/c/phasic.c:1930`) is the sole populator
of `graph->reward_compute_graph` (numeric) / `graph->parameterized_reward_compute_graph`
(raw symbolic) / `graph->parameterized_reward_compute_graph_off` (offset/zero-copy
symbolic). It builds a **single elimination trace over the whole graph** via
`ptd_graph_ex_absorbation_time_comp_graph_parameterized[_dyn]` (chosen by
`graph->use_dyn_ordering`, an unrelated elimination-*order* knob — static vs.
dynamic min-degree pivoting — not a decomposition strategy). Grepping the full body
of this function (`phasic.c:1930-2155`, per the caching atlas's own line-range,
re-verified) for any SCC/hierarchical awareness returns **zero hits** — it has no
knowledge of SCC decomposition at all; it is the monolithic-only path, unconditionally.

Of the four exact-gradient functions:
- **Functions 1-3** (`ptd_moments_grad_theta` `:10738`, `_log` `:10917`, `_dph`
  `:11142`) never call `ptd_precompute_reward_compute_graph` at all — each builds
  its **own private** monolithic raw tape directly via
  `ptd_graph_ex_absorbation_time_comp_graph_parameterized[_dyn]` on every call
  (confirmed: `graph->use_dyn_ordering ? ..._dyn(graph) : ...(graph)` at
  `:10744-10746`, `:10926-10927` [wait: line drifted, see below], `:11183-11184`),
  then converts to offset form locally (`ptd_pcg_convert_to_offset`) — confirmed
  at `:10744-10746` (function 1), `:10925-10927` (function 3, `_log`), and
  `:11183-11184` (function 2, `_dph`). Grepping the full body of all three
  functions (`phasic.c:10738-11338`) for `HIERAR`/`scc`/`compose` returns **zero
  hits**.
- **Function 4** (`ptd_sojourn_grad_theta_subset`, `:11408`) is the only one that
  reuses the graph-level cache: it calls `ptd_precompute_reward_compute_graph(graph)`
  as its first action (`:11415`), then reads `graph->parameterized_reward_compute_graph_off`
  if populated, else converts the raw `graph->parameterized_reward_compute_graph`
  itself (`:11419-11436`). Since `ptd_precompute_reward_compute_graph` (per above)
  never runs the hierarchical path, this function is **also monolithic-only**,
  unconditionally, regardless of `parallel_elimination`.

**Consequence:** none of the four functions has ever, under any configuration,
touched the hierarchical/SCC machinery. `parallel_elimination=True` and
`exact_moment_grad=True` (the shipped default) are **completely orthogonal** at the
C level today — turning on hierarchical mode silently gives the gradient computation
*zero* benefit; it always falls back to full monolithic tape construction every call.
No Python code cross-checks or warns about this combination: grepping
`exact_moment_grad` across `src/phasic/*.py` for any co-occurring
`parallel`/`hierar`/`scc` term returns nothing, and grepping `__init__.py` for
`parallel_elimination` finds only the unrelated graph-size profiler docstring
(`__init__.py:373`, `:8497`).

---

## 2. The hierarchical/SCC mechanism, precisely (Q1-Q3 grounding)

**Entry point.** `ptd_expected_waiting_time` (`phasic.c:9980`) is the *only* C
function with hierarchical dispatch. The gate (`:9999-10041`):

```c
const char *hierar_env = getenv("PHASIC_HIERAR_ELIMINATION");
bool use_hierarchical = (hierar_env != NULL && hierar_env[0]=='1'
                         && hierar_env[1]=='\0' && !ptd_scc_compose_in_progress);
if (use_hierarchical && graph->parameterized && rewards == NULL
    && graph->param_length > 0) {
    ...
    struct ptd_scc_graph *scc_graph = ptd_find_strongly_connected_components(graph);
    double *result = ptd_compose_scc_prcs(graph, scc_graph, compose_theta, graph->param_length);
    ...
    if (result != NULL) { return result; }   // hierarchical success
}
if (ptd_precompute_reward_compute_graph(graph)) return NULL;   // monolithic fallback
```

Confirmed Python wiring: `phasic.configure(parallel_elimination=True)` sets exactly
this env var (`src/phasic/config.py:447`: `'parallel_elimination': ('PHASIC_HIERAR_ELIMINATION', False, False)`).

**Critical restriction already in the gate: `rewards == NULL`.** The moments
primal (`GraphBuilder::compute_moments_impl`, `src/cpp/parameterized/graph_builder.cpp:512-553`)
computes `E[T]` via `g.expected_waiting_time(rewards)` with `rewards` empty
(→ NULL) **only for the first sub-call**; every higher moment
(`k=1..nr_moments-1`) re-invokes `expected_waiting_time` with a concrete,
non-empty `rewards3` vector (`:528-542`), which per the gate **always falls
through to monolithic**. This is independently confirmed by a pytest docstring:
`tests/pytest/test_hierar_elimination_env.py:71-74` — *"variance() reads
expected_waiting_time twice (with reward vector for the second moment) — when
rewards != None, the hierarchical path falls back to monolithic, so this exercises
the fallback path."* **Practical upshot: for any `nr_moments >= 2` (the normal
case — mean+variance at minimum), the hierarchical composer contributes at most
the trivial `E[T]` term; the moments actually used to build `J_out` in
`ptd_moments_grad_theta` never come from it even on the numeric/forward side.**

**The composition mechanism** (`src/c/scc_compose.c`, entry `ptd_compose_scc_prcs`
`:555`, real work in `ptd_compose_scc_prcs_inner` `:273` and per-SCC worker
`ptd_compose_scc_one` `:138`):

1. Decompose into SCCs (`ptd_find_strongly_connected_components`), Kahn-topo-sort
   the condensation, compute per-SCC "levels" (longest path to a sink) so that
   same-level SCCs are mutually independent and safe to run in an OpenMP
   `#pragma omp parallel for` (`:517`).
2. Process levels **sink-first** (so a downstream SCC's numeric result is always
   already available before an upstream SCC that depends on it runs).
3. Per SCC (`ptd_compose_scc_one:138-267`): build (or load from the rev-3 on-disk
   cache) a **synthetic graph** wrapping the SCC's internal vertices plus a
   synthetic source and one "phantom" absorbing vertex, with one extra
   "per-channel absorbing" vertex per external out-edge
   (`ptd_scc_build_synthetic_graph`, `src/c/scc_synthetic.c:515-1115`). This
   synthetic graph is a genuine parameterized `ptd_graph` — its internal edges
   carry real copied coefficients (`add_edge_raw`, `:876-880` copies
   `edge->coefficients`/`coefficients_length` verbatim), so `ptd_synth_get_or_compute_prc`
   (`scc_synthetic.c:1133-1202`) builds/loads a **real**
   `ptd_desc_reward_compute_parameterized[_off]` for it via the identical
   `ptd_graph_ex_absorbation_time_comp_graph_parameterized` used by the monolithic
   path — same struct types, same command tape format the four adjoint functions
   already know how to walk.
4. **Then the composer overwrites two categories of edge weight with plain
   `double`s, numerically, via `ptd_edge_update_weight`** (`scc_compose.c:204-225`):
   - Type-C edge (SCC-internal "downstream-connecting" vertex → its per-channel
     absorbing vertex): weight := `parent_dj->edges[ch->parent_edge_idx]->weight`
     — the **parent's current numeric edge weight** at this theta (itself
     theta-dependent through the parent's own coefficients, but read here as a
     bare number, not as a symbolic contraction).
   - Phantom edge (per-channel absorbing vertex → phantom absorbing): weight :=
     `1.0 / parent_result[parent_target_idx]` — the **reciprocal of a different
     SCC's already-computed numeric result** (`:213-223`). `parent_result` here is
     the running parent-wide result array being filled in sink-first order by
     *earlier* per-SCC calls at lower levels.
5. `ptd_expected_waiting_time(synth, NULL)` is then called **recursively but with
   the hierarchical branch suppressed** (`ptd_scc_compose_in_progress` is bumped
   as a thread-local re-entrancy guard, `scc_compose.c:159`, `:281`) — i.e. this
   inner call runs the **monolithic, purely numeric** executor on the synth
   graph's already-built PRC, producing `synth_result` (a `double*`, `:245`).
6. Per-internal-vertex results are copied into the parent-wide `parent_result`
   array (`:253-259`); the synth graph, its metadata, and (if freshly built) its
   PRC are then destroyed (`out:` label, `:262-264`) — **nothing about this
   per-SCC tape or PRC survives past this one call.**

`ptd_compose_scc_prcs` returns `parent_result` — a flat `double*`, length
`parent->vertices_length`. **`parent` itself is never touched beyond having its
edge weights refreshed from theta** (`ptd_graph_update_weights(parent, ...)`,
`scc_compose.c:315`, purely for reading Type-C weights in step 4) — grepping
`scc_compose.c` and `scc_synthetic.c` for any assignment to
`parent->parameterized_reward_compute_graph`/`_off` or `parent->reward_compute_graph`
finds **none**; the only writes to those fields anywhere in either file target
`synth->...` (the transient per-SCC graph), never `parent`/the top-level graph
passed in by the caller.

**This directly answers Q3.** For a graph flagged for hierarchical elimination:
`graph->parameterized_reward_compute_graph_off` on the *parent* is **never**
populated by this path (it stays whatever the monolithic fallback last set it to,
or NULL if hierarchical always succeeded). The hierarchical path does not "store
its result somewhere else that's still symbolic" — the parent-level artifact it
produces (`parent_result`) is **purely numeric**, and the intermediate symbolic
artifacts (per-SCC PRCs) are real tape objects but are **destroyed after each
composition call**, never assembled into a single whole-graph symbolic structure.

---

## 3. Numeric correctness / maturity of the underlying path (Q5)

The hierarchical composer's **numeric** (non-gradient) output is tested against
the monolithic reference to `rtol=atol=1e-12`:

- `tests/pytest/test_scc_compose.py::test_compose_matches_monolithic` — every toy
  fixture in `toy_model.py` (`BUILDERS`: `build_toy_base`/`_a`/`_b`/`_c_p`/
  `_c_pprime`/`_d`) × 4 theta vectors, via the direct `SCCGraph.compose()` binding.
  Also `test_compose_with_disk_cache` (rev-3 on-disk cache round-trip) and
  determinism/theta-sensitivity smoke tests.
- `tests/pytest/test_hierar_elimination_env.py` — same toy fixtures, through the
  **user-facing** `Graph.expected_waiting_time()`/`.expectation()`/`.variance()`,
  confirming the env-var-gated fallback behaves correctly, **including the explicit
  confirmation that `.variance()` (needs the 2nd moment) exercises the
  `rewards != NULL` monolithic-fallback branch, not the hierarchical one** (see §2).

This is solid parity coverage **but only on a handful of hand-built toy graphs**
(`toy_model.py`'s docstring: 5 canonical variants, chosen to exercise specific
structural edge cases — duplicate states, parallel SCCs, cross-graph SCC reuse —
not scale). No test in `tests/pytest/` combines the hierarchical composer with a
production-scale model (the population-genetics coalescent/two-locus models this
library targets, or the n~10⁴-10⁶ scale cited elsewhere in this repo for
`expected_sojourn_time_subset`). No test combines it with `was_dph`/native-DPH:
grepping `was_dph`/`is_discrete`/`discretize` across `scc_synthetic.c` and
`scc_compose.c` returns **zero hits** — the composer has no discrete-renorm
awareness at all, consistent with (not a gap relative to) its scope, but meaning
DPH support is not just "untested for gradients," it doesn't exist for the
numeric hierarchical path either.

**Net maturity assessment:** the *numeric* hierarchical path is well-gated
(explicit env var, explicit fallback-on-failure at every level) and has a genuine,
if narrow, correctness gate. It is young infrastructure (work-package-numbered
comments — WP-1 through WP-8 — suggest an actively-in-progress branch/initiative)
built for **performance** (parallelism across independent SCCs, hash-deduplication
of repeated substructure across models/graphs), not necessarily for scale the
monolithic path can't already reach — the companion atlas
(`atlas/exact-fd-atlas-c-functions.md:240`) notes monolithic sojourn's *design
target* is n up to ~7×10⁵ (a stated intent, not an empirically measured benchmark:
the same atlas explicitly flags "no measured cost/memory profile at production
scale (n~7×10⁵)" as an open gap). Whether hierarchical decomposition is required
for tractability on any real model, vs. purely a wall-clock speedup, is therefore
not established either way by this investigation — see risk item §5.2.

---

## 4. Why this is new math, not new plumbing (Q4)

Two independent, compounding reasons rule out "point the existing tape-walk
functions at a hierarchically-built tape and it just works":

**(a) There is no single tape to point them at.** As established in §2, the
top-level artifact the hierarchical path produces is a plain numeric array, not a
tape. Even a from-scratch engineering effort to "make the composer *also* leave
behind a persistent, symbolic, whole-graph tape" would have to solve (b) below —
it cannot be done by capturing more state, because the operation performed at the
SCC boundary is not expressible in the existing tape command language at all.

**(b) The cross-SCC linkage is a genuinely nonlinear, cross-tape dependency the
existing command/input encoding cannot represent.** The phantom-edge weight
injected at each SCC boundary is `1.0 / parent_result[target]`
(`scc_compose.c:213-214`) — the reciprocal of a **different SCC's fully-eliminated
numeric output**. The existing tape-input specs (`PTD_PCG_PTR_MEM`/`_EDGE`/
`_EXTERNAL`, `phasic.c:3065-3067`) only ever encode "this tape input is a pointer
to a live memory slot / a specific edge's weight-plus-byte-offset / a rebindable
external anchor for cache reuse" — all three describe a *linear-in-theta* quantity
(a coefficient-times-theta dot product, or a symbolic slot for one). None can
express "the derivative of this input with respect to theta must itself be
obtained by running a *different* elimination's own reverse-mode adjoint and then
applying `d(1/x)/dx = -1/x²`." Differentiating the composed whole-graph result
therefore requires a genuinely new **two-level reverse-mode structure**:
  1. *Inner level* (reusable): differentiate each per-SCC tape w.r.t. its own
     theta-linear edges — this part *can* reuse the existing single-tape-walk
     machinery (functions 1-3's core stage-0/1/2 pattern) as a building block,
     extended to also expose `d(synth_result)/d(channel edge weight)` for the
     handful of injected channel edges (not just `d(.)/dtheta`).
  2. *Outer level* (new): a second reverse-mode pass **over the SCC condensation
     DAG itself**, run in the *opposite* order from value composition (value
     composition is sink-first; the gradient accumulation must run source-first /
     forward-topological, standard reverse-mode-over-a-DAG), chaining each SCC's
     per-channel Jacobian block through the `d(1/x)/dx` reciprocal at every
     boundary. This is new code with no existing analogue anywhere in this
     codebase — it operates at SCC-block granularity, not elimination-command
     granularity.

**A concrete, source-grounded illustration of why the "obvious shortcut" is a
silent-wrong-answer landmine, not just a missing feature:** the per-input
topology guard present in all four functions (e.g.
`ptd_moments_grad_theta:10852-10854`: `if (sp.kind != PTD_PCG_PTR_EDGE || sp.byte
!= 0 || sp.v >= ... || sp.e >= ...) { ok = 0; ... }`) only validates *structural*
consistency (is this a real edge index), not *semantic* consistency (does the
edge's coefficient vector still describe its true weight-vs-theta relationship).
The synthetic SCC graph's channel edges are built with **placeholder
coefficients** (`coefficients[0]=1.0`, rest 0 — `scc_synthetic.c:839-842`, comment
at `:828-838` explicitly: *"the placeholder produces a concrete weight of θ[0]...
in cache form, the placeholder coefficients are encoded as EXTERNAL pointers so
the actual binding... happens at composition time"*) and are of ordinary
`PTD_PCG_PTR_EDGE` kind (not `PTD_PCG_PTR_EXTERNAL` — that kind exists only in
the legacy rev-1/2 *on-disk serialization* format for cross-graph SCC cache reuse,
via `ptd_scc_collect_external_anchors`, `scc_synthetic.c:414-484`, used only by
test-only pybind methods — it is not used by the live in-memory
`ptd_compose_scc_prcs` numeric composition at all). **If someone naively called
one of the four existing exact-gradient functions directly on a per-SCC synthetic
graph *after* composition has overwritten its channel weights, none of the
existing guards would fire** — the function would happily compute
`d(weight)/dtheta[0] = 1` for a channel edge (reading the untouched placeholder
coefficient array) when the true dependency is either an arbitrary linear
combination (Type-C, from a different parent edge's real coefficients) or a
complex nonlinear chain through a different SCC's adjoint (phantom edge) — a
**plausible-looking, silently wrong Jacobian entry**, exactly the class of defect
CLAUDE.md's B3 section explicitly calls out as the thing adversarial review
exists to catch.

**Conclusion: this is a large batch, not a small one.** It is not "the existing
functions likely already work if X is confirmed" — X (a persistent, whole-graph
symbolic tape from the hierarchical path) does not exist and cannot be produced
without solving the cross-SCC nonlinearity, which is new mathematical content, not
an engineering gap.

---

## 5. Risk / unknowns list

1. **No live interaction today = no regression risk, but no free lunch either.**
   Because the two features are currently orthogonal (§1), there is nothing to
   "fix" for correctness right now — but there is also no path by which turning on
   `parallel_elimination=True` speeds up exact-gradient SVGD on a large graph
   today. Any user attempting that combination silently pays full monolithic
   elimination cost per gradient call regardless of the flag — worth flagging to
   users/docs even absent new adjoint code, since nothing currently warns about it.
2. **Scale mismatch is unclear.** It's not established in this investigation
   whether there exist real (non-toy) models where monolithic elimination is
   actually intractable (not just slower) and hierarchical decomposition is
   *required* rather than merely a speed/parallelism optimization. If such models
   exist, they would also be closed off from exact gradients today (the FD
   fallback would have to carry them, and FD's own known defects — see
   `project_fd_gradient_b3` memory: "FD custom_vjp broken at mixed scales" — would
   then apply with no exact-gradient escape hatch). This is worth a follow-up
   check (does any production model in this repo actually *require* hierarchical
   mode to complete at all, vs. using it only for wall-clock speedup) before
   deciding urgency.
3. **`was_dph`/native-DPH is entirely out of scope for the composer today** (§3) —
   any future hierarchical-adjoint work would need to either explicitly exclude
   discrete graphs (mirroring the existing `was_dph` exclusion in
   `ptd_moments_grad_theta_log`/`ptd_sojourn_grad_theta_subset`) or extend the
   composer itself first, which is out of scope for a gradient-only batch.
4. **Test coverage is toy-scale only** (§3) — any future work in this area needs
   new fixtures at least one order of magnitude larger than the 5 canonical toys,
   and ideally a real coalescent/two-locus model, before trusting numeric parity
   at production scale, independent of the gradient question.
5. **MPFR interaction unexplored.** The composer's inner `ptd_expected_waiting_time`
   call on each synth graph goes through the same MPFR auto-escalation logic as
   the monolithic path (`phasic.c:10085-10145`, condition-number-triggered) — but
   whether condition numbers computed *per-SCC* (on a small synthetic subgraph)
   correctly reflect the whole-graph conditioning is unverified; a per-SCC adjoint
   would inherit this same question for its own MPFR gate.
6. **`PHASIC_MAX_PARALLEL_SCCS`/OpenMP nesting.** The composer already runs
   per-SCC eliminations inside an `omp parallel for` over same-level SCCs
   (`scc_compose.c:517`), each of which may itself spawn nested OpenMP work. A
   future per-SCC adjoint added inside `ptd_compose_scc_one` would run under the
   exact same nesting, and would need to be reentrant/thread-safe under that
   regime — untested territory, since nothing gradient-related runs there today.

---

## 6. Cross-batch notes

- **(b) formula-mode + skeleton refactor.** CLAUDE.md's own "Disabled paths /
  follow-ups" section already flags **reverse-tape skeleton duplication**: the
  three existing `ptd_moments_grad_theta[_dph|_log]` functions in `phasic.c` are
  "three near-identical (~150 line) copies of the same stage-0/1/2 skeleton,"
  and recommends extracting a shared core **before a 4th weight-mode variant
  (`'formula'`) is attempted.** A hierarchical/SCC adjoint, per §4, would need to
  reuse that exact stage-0/1/2 inner-level machinery as a building block (for the
  per-SCC contribution) **plus** a wholly new outer SCC-DAG level — i.e. it would
  become not a 4th copy but a *structurally different* consumer of the same core.
  **These two batches should very likely share a design conversation before
  either is implemented**: if the formula-mode work extracts a shared
  stage-0/1/2 helper first, the hierarchical batch's inner level should be built
  against that helper from day one rather than adding a 5th near-duplicate. If
  the hierarchical batch is scoped first without the extraction, it risks locking
  in a 4th copy that then has to be un-duplicated alongside the other three.
  Recommendation: sequence the skeleton extraction (already independently
  motivated by (b) alone) **before** any hierarchical-adjoint implementation
  work, not after.
- **(a) rewards support for the moments adjoint.** Orthogonal — rewards support
  is about `ptd_moments_grad_theta`'s existing monolithic path ignoring caller
  rewards (already fixed per CLAUDE.md, commit `315ce9c8`). No direct
  interaction, except that the composer's own primal call
  (`ptd_expected_waiting_time(synth, NULL)`, `scc_compose.c:245`) *always* passes
  `rewards=NULL` — if a future hierarchical adjoint needs per-vertex rewards
  propagated through SCC composition, that is new plumbing this batch would have
  to add, not something (a) already provides.
- **(c) Graph.svgd() plumbing + joint-index baked-mode.** No direct conflict
  found. `ptd_sojourn_grad_theta_subset` (the joint-index gradient) is
  monolithic-only today (§1) exactly like the other three; a hierarchical
  extension would need this function's own new per-SCC-forward-mode variant,
  independent of (c)'s baked-mode plumbing work, which operates purely on the
  monolithic tape.
- **(d) PMF/PDF gradient re-derivation + daisy-chain.** Not investigated in
  depth this session (out of the requested scope), but worth flagging: if (d)'s
  PDF/PMF re-derivation also assumes a single monolithic tape (as the moments
  adjoints do), it likely has the identical hierarchical-incompatibility
  described here, for the same structural reason (§2/§4) — worth a one-line
  cross-check by whoever scopes (d).
- **(e) weight_mode='callback' + MPFR-precision adjoint.** No direct conflict.
  Callback/formula-mode theta resolution happens per-edge, independent of
  whether that edge sits inside a monolithic or per-SCC synthetic tape — but per
  §4(b), the *channel* edges injected by the composer are always plain
  linear-mode placeholders regardless of the graph's real weight_mode, so a
  future SCC-adjoint's "outer level" (§4) would need to handle the *composer's*
  injected weights (always linear/placeholder) as a mechanism distinct from
  whatever weight_mode the *user's* graph uses — these are two independent axes,
  not something (e)'s callback-mode work would incidentally cover.

---

## 7. Recommended sequence position

**Defer.** Concretely:

1. This batch is large (a new, two-level, SCC-DAG-spanning reverse-mode adjoint —
   not a bug fix, not a guard, not a cache-plumbing fix), not small.
2. It has **no current users being blocked**: nothing today combines
   `parallel_elimination=True` with `exact_moment_grad=True` in any tested or
   documented workflow, and no evidence was found in this session that any
   production model *requires* hierarchical mode to complete (as opposed to using
   it for speed) — meaning there's no "SVGD is currently broken/slow for a
   specific real model" pain point motivating urgency, only a theoretical gap.
3. It has a **real, cheap, non-code-writing prerequisite** that should happen
   first regardless: the skeleton-extraction refactor already flagged by
   CLAUDE.md for the formula-mode batch (b). Attempting the SCC-adjoint's inner
   level before that extraction risks a 4th near-duplicate stage-0/1/2 copy that
   then has to be unwound.
4. It has an **open scale question** (§5.2) that would change its priority
   substantially if answered "yes, some real model needs hierarchical mode to run
   at all" — worth a quick, cheap check (grep production model configs / recent
   SVGD run logs for `parallel_elimination=True` usage) before committing
   engineering time, since that would upgrade this from "nice-to-have parity" to
   "blocking gap."

**Suggested position in the batch queue:** after (b)'s skeleton extraction lands
and is gate-verified (a natural prerequisite, independently motivated), and after
a cheap usage-check answers §5.2 — not before either. If §5.2 comes back "no
production model needs it," this batch can reasonably sit at the back of the
queue behind all five sibling batches; if it comes back "yes," it should jump
ahead of any batch not already blocking a real user.
