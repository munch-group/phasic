# Feasibility assessment: `weight_mode='callback'` exact gradient (Job A) and the MPFR conditioning-floor adjoint (Job B)

All line numbers below were read directly from the current tree on 2026-08-05 (not copied from
the atlas docs, which were used only as a starting map and re-verified against source). Where I
rely on reasoning rather than a direct repro/experiment, I say so explicitly.

---

## Job A — `weight_mode='callback'` exact gradient

### Q1: how is the callback invoked today in the forward pass — plain float/numpy, or JAX array/tracer?

**Plain, concrete numpy — never a JAX tracer.** Confirmed by reading `_apply_weight_callback`
(`src/phasic/__init__.py:735-807`) and all four call sites:

- `_apply_weight_callback` calls `float(callback(theta, coeffs))` (lines 782-786 for regular
  param edges, 793-797 for start edges) — a **hard cast to a Python float**. Even if a callback
  internally used `jax.numpy`, the return value is immediately collapsed to a concrete scalar.
- Every call site invokes `_apply_weight_callback` from **inside the body of a
  `jax.pure_callback`-wrapped host function**, after the crossing back to concrete values has
  already happened:
  - `pmf_from_graph` (`__init__.py:3652-3673`): `_compute_pdf_callback(theta_np, times_np)` is
    called from a `lambda t, tm: _compute_pdf_callback(np.asarray(t, dtype=np.float64), ...)`
    inside `jax.pure_callback(..., vmap_method='sequential')` (3664-3673). `theta_np` is always
    concrete numpy by construction.
  - `pmf_and_moments_from_graph` (`__init__.py:7078-7138`): same pattern — `_cb(t, tm, rw)` does
    `t_np = np.asarray(t, dtype=np.float64)` (7123) before calling `_compute_callback` → `_apply_weight_callback`.
  - Its `cdf_zero` companion (`__init__.py:7148-7168`) and `pmf_from_graph_joint_index`
    (`__init__.py:7968-8003`) follow the identical pattern.

So today the callback receives numpy inputs; nothing about the *invocation* prevents a callback
from being written using `jax.numpy` ops (`jnp.sum`, `jnp.exp`, etc. all accept numpy input and
just return a 0-d JAX array, which `float()` then reads out) — but the current wiring throws away
any JAX structure at the `float()` cast, so **no autodiff information survives the call today**,
regardless of how the callback is written internally.

### Q2: is this purely an architectural choice, or is there a real mathematical/engineering obstacle?

**It is fixable for a JAX-native callback, and the reason is a clean piece of chain-rule
mathematics, not an accident of wiring** — but the "just swap the contraction target" framing in
the task is *slightly* optimistic: getting there needs a small, genuinely new C/C++ exit point, not
a pure-Python change. Here is the reasoning, grounded in the actual C code:

`ptd_moments_grad_theta` (`src/c/phasic.c:10738-10881`, and its `_dph`/`_log` siblings) has an
internal three-stage structure (confirmed by reading the function body and its own in-code
comments, e.g. `phasic.c:10802` "per-output-moment reverse chain + stage-2 + edge->theta
contraction", `10830` "stage-2: reverse the param tape ONCE"):

- **Stage 0** (10766-10779): forward moment chain over the recorded elimination trace — pure
  arithmetic over recorded per-op values, does not reference `theta` or `coefficients` directly.
- **Stage 1** (10811-10829): reverse-mode adjoint of stage 0, producing `dm[c]` — the sensitivity
  of each output moment to each **tape-input command** `c` (which corresponds 1:1 to an edge
  weight).
- **Stage 2** (10830-10870): the *only* stage that mentions `theta` at all. For linear mode it is
  literally:
  ```c
  for (size_t j=0;j<P;++j) J_out[outk*P + j] += binp[k] * e->coefficients[j];   // line 10869
  ```
  i.e. `d(moment)/d(theta_j) = Σ_e (d(moment)/d(w_e)) · (d(w_e)/d(theta_j))`, where
  `e->coefficients[j]` is used **because, for linear mode, `w_e = Σ_j c_j θ_j` so `∂w_e/∂θ_j = c_j`
  is a graph-topology constant** — nothing about stages 0-1 depends on that being constant. The
  `_log` variant (`phasic.c:11017-11031`, product rule `w_e/θ_j`) and `_dph` variant
  (`phasic.c:11290-11322`, was_dph quotient-rule) are the same stage-0/1 skeleton with a *different*
  stage-2 formula — this is exactly the "reverse-tape skeleton duplication" CLAUDE.md already
  flags.

For `weight_mode='callback'`, `w_e = f(θ, c_e)` for an arbitrary (possibly nonlinear) `f`. The
chain rule still holds exactly: `∂w_e/∂θ_j = (∂f/∂θ_j)(θ, c_e)`, evaluated **at the same θ** used
to set the primal edge weight for that call. This is *not* a constant across calls, but it is still
just a number for a given θ, and the same stage-2 formula (`Σ_e adjoint_e · ∂w_e/∂θ_j`) is exactly
correct — reverse-mode composition doesn't care whether the local Jacobian at the interface is a
graph constant or was computed by `jax.grad` a moment earlier. If `f` is JAX-native,
`jax.grad(weight_callback, argnums=0)(theta, coeffs_e)` gives exactly `∂w_e/∂θ` for one edge, and
`jax.vmap` over edges gives it for all edges in one call. **So mathematically, a JAX-native
callback absolutely can get an exact gradient with the existing stage-0/1 machinery unchanged —
this is a real, sound result, not a hopeful analogy.**

What is missing is a way to get that per-edge `∂w_e/∂θ` vector into stage 2's contraction, and here
the picture is more nuanced than "swap `e->coefficients[j]` for `jax.grad(...)`, in Python, for
free":

- There is **no existing pybind/C++ method to overwrite `edge->coefficients[]`** on a live graph
  after construction (checked `src/cpp/phasic_pybind.cpp` for any `coefficients` setter — only a
  read-only `serialize()` getter exists at 3966/3968). Coefficients are set once, at
  `add_edge(..., coefficients=[...])`, and never mutated by any current code path.
- There **is** an existing, directly-reusable mechanism for the **primal** half: `Graph.update_weights(theta, callback=fn)`
  (`__init__.py:1910-1991`, dispatching to the C++ overload
  `phasic::Graph::update_weights_parameterized(scalars, callback)`, `src/cpp/phasiccpp.cpp:1857-1926`)
  sets `edge->weight = callback(theta, coeffs)` directly on the live C graph (line 1921) while
  leaving `graph->parameterized`/`param_length` untouched — so `_exact_graph.update_weights(t,
  callback=weight_callback)` (mirroring the existing `_exact_graph.update_weights(t,
  log=_exact_is_log)` at `__init__.py:7012`) would give a **correct, already-available, zero-new-code**
  way to set the private clone's primal edge weights for stage 0. (`graph.clone()` already
  propagates `_weight_callback` to the clone via `_propagate_weight_state`,
  `__init__.py:678-699`/`8528-8545`, confirmed by reading — no gap there.)
- But that same C++ function reads `edge->coefficients[]` only to **hand it to the callback**
  (`phasiccpp.cpp:1904-1907`) — it never writes them back. So `e->coefficients[]` after this call
  is still whatever the user originally passed to `add_edge(...)` (arbitrary auxiliary data for
  the callback's own use, not `∂w_e/∂θ`, and not even guaranteed to be length `P` — e.g. a
  PSMC-style callback may keep extra auxiliary numbers per edge unrelated to `theta`'s length).
- And `_moments_grad_theta` / `_dph` / `_log` are exposed to Python only as the fully-contracted
  Jacobian (`phasic_pybind.cpp:1915/1919/1926`, thin wrappers in `api/cpp/phasiccpp.h:560-604`
  that call the C function and return `J` directly) — **there is no exit point that returns
  stage 1's pre-contraction adjoint** (`dm[]`/`binp[]` per tape input/edge).

**Conclusion: not purely Python.** A small, genuinely new C exit point is required — either (a) a
new mutator to overwrite `edge->coefficients[]` on a live graph (touches persistent-graph memory
ownership, a part of the codebase the atlas already flags as having zero NULL-checked allocations —
real, if modest, memory-safety surface), or (b) — the cleaner option — a new, small function that
runs the *existing* stage-0/1 code verbatim and returns the per-tape-input adjoint **before**
stage 2, so the theta-contraction can be done in Python as a plain matmul against a
`jax.vmap(jax.grad(weight_callback))`-computed Jacobian matrix. Either way it is dramatically
**smaller** than a full new `_moments_grad_theta_callback` C function, because it reuses the
expensive O(n³) elimination-adjoint machinery (stages 0-1) completely unchanged; only a thin new
"stage-2 exit" is needed.

### Q3: concretely, what would change?

- `pmf_and_moments_from_graph`'s exact-grad gate (`__init__.py:6972-6989`) currently sets
  `_linear_scope_ok = (_wm in (None, 'linear'))` and excludes everything else (including
  `'callback'`) from `_exact_grad_enabled`. A `_callback_scope_ok = (_wm == 'callback' and
  <callback is JAX-native, e.g. an explicit user opt-in flag>)` would be added alongside it.
- `_exact_moments_jac_np`'s `_one(t)` (`__init__.py:7004-7033`) currently does
  `_exact_graph.update_weights(t, log=_exact_is_log)` then calls one of the three existing
  `_moments_grad_theta*` bindings. For callback mode it would instead: (1) call
  `_exact_graph.update_weights(t, callback=weight_callback)` (existing, no changes) to set the
  correct primal weights; (2) call the **new** pre-contraction-adjoint binding to get `dm`/`binp`
  per edge; (3) compute `dw_de_dtheta = jax.vmap(jax.grad(weight_callback, argnums=0),
  in_axes=(None,0))(jnp.asarray(t), coeffs_matrix)` (a new, pure-Python/JAX step, using the
  coefficient matrix already available from `graph.serialize()`); (4) contract
  `J = adjoint_per_edge @ dw_de_dtheta` (`numpy.matmul`, K×n_edges times n_edges×P) in Python.
  Steps 1, 3, 4 are pure Python/JAX; step 2 needs the new C exit point from Q2.
- `pmf_from_graph_joint_index`'s equivalent callback branch (`__init__.py:7968-8003`) is
  architecturally different (forward-mode adjoint over `ptd_sojourn_grad_theta_subset`, a distinct
  C function with its own stage split) — the same idea applies but would need its **own** new exit
  point; it is not automatically covered by fixing the moments path.
- This is exactly the same "extract the shared stage-0/1 core, make stage-2 pluggable" refactor
  CLAUDE.md's "Reverse-tape skeleton duplication" note already recommends doing before formula mode
  (batch b) lands — see Cross-batch notes below.

### Q4: non-JAX-native callbacks

If the callback uses arbitrary numpy/scipy (not JAX-traceable), no general automatic
differentiation is possible through it — `jax.grad` requires the traced computation to be built
from JAX primitives; there is no way to recover a gradient from a black-box Python function that
returns a plain float. The only real alternatives are:

1. **Require the user to also supply an analytic derivative callback**, e.g.
   `weight_callback_grad(theta, coefficients) -> d(weight)/d(theta)` (shape `(P,)`), mirroring
   `weight_callback` itself. This slots into exactly the same stage-2 plug point described in Q3 —
   whether the per-edge Jacobian vector comes from `jax.grad` or from the user's own function makes
   no difference to the C-side exit point needed. This is a legitimate, modest API addition, but it
   pushes correctness risk onto the user (a wrong analytic derivative silently produces a wrong
   "exact" gradient with no way to detect it, unlike FD which is self-checking by construction).
2. **Leave it FD-only permanently.** Given `weight_mode='callback'` is explicitly the
   "arbitrary Python escape hatch" mode (contrasted with `'formula'`, which is restricted precisely
   so it *can* be differentiated), this is a legitimate, honest scope boundary — not every
   escape-hatch API needs an exact-gradient counterpart. I think this is probably the right default
   for the general (non-JAX-native) case, with option 1 offered only as an opt-in for advanced users
   who explicitly want it.

### Job A risks / unknowns

- **Not verified by execution**: I did not run `jax.grad`/`jax.vmap` against a toy callback in this
  environment to empirically confirm zero surprises (e.g. any JAX version-specific quirk in
  differentiating through `jnp.exp`/`jnp.sum` composed this way). The math is standard first-order
  JAX usage with no unusual structure, so I have not treated this as a real risk, but it is untested
  here.
- **Coefficient-length mismatch risk carries over unchanged.** As documented in-code
  (`__init__.py:1343-1358`), callback-mode edges may have coefficients longer/shorter than
  `theta_dim`. The stage-2 contraction (existing linear code, `phasic.c:10869`) already reads
  `e->coefficients[0..P-1]` **without checking `coefficients_length >= P`** — a pre-existing,
  unguarded latent risk (out of scope to fix here, but a new callback-mode contraction would
  inherit it if implemented naively).
- **Which new exit point (mutate coefficients vs. return pre-contraction adjoint) is chosen matters
  a lot for engineering cost and memory safety** — this needs a real design decision, not just
  "small C change," before implementation.
- **`pmf_from_graph_joint_index`'s callback path is a second, separate piece of work** (different C
  function, different stage split) — fixing the moments path does not fix this one for free.
- Determining whether a given user callback *is* JAX-native (so this path is eligible) needs an
  explicit opt-in flag or a runtime probe (e.g. try `jax.grad` and catch a `TracerArrayConversionError`
  or similar) — not free, but small.

---

## Job B — the "conditioning floor": MPFR-precision adjoint feasibility

### Q1: the primal's MPFR escalation path, read in full

Location: inside `ptd_expected_waiting_time` (`src/c/phasic.c:9980-...`), escalation logic at
`10058-10145`.

**Trigger** (10063-10088): a condition number is computed by pre-scanning the already-built
`graph->reward_compute_graph`'s elimination commands (`compute->commands[j].multiplier`, the
double-precision constants baked in by elimination) for `max(|multiplier|) / min(|multiplier|)`
over nonzero, non-infinite multipliers. If this ratio exceeds a threshold (`condition_threshold`,
default `1e12`, overridable via `PHASIC_CONDITION_THRESHOLD` env var), or `PHASIC_FORCE_MPFR` is
set, MPFR is activated (guarded by `#ifdef HAVE_MPFR`; without it, only a warning is logged,
10138-10145).

**Precision selection** (10092-10104): `mpfr_precision = log2(condition_number) + 64` (overridable
via `PHASIC_MPFR_BITS`), clamped to `[128, 1024]` bits.

**What it does differently** (10106-10129): two implementations exist, gated by
`PHASIC_USE_MPFR_LEGACY`:
- **Default ("MPFR-A"), `ptd_expected_waiting_time_mpfr_from_double_pcg`** (`phasic.c:9885-9955`):
  replays the **same already-compiled double-precision `reward_compute_graph`** (a fixed sequence
  of `result[from] += result[to] * multiplier` commands — the same trace-replay architecture
  CLAUDE.md's "trace-based elimination" describes) but does the *accumulation* (`result[]`, one
  `mpfr_t` per **vertex**, i.e. O(n) elevated-precision values) in MPFR: `mpfr_set_d(multiplier,
  cmd.multiplier, ...)` converts the (still-double) per-op constant into an MPFR scalar for one
  multiply-accumulate, then `mpfr_mul`/`mpfr_add` into the vertex accumulator. Only the O(n)
  accumulators and 2 scalar temporaries are ever `mpfr_t`; the O(L) trace itself (its `multiplier`
  constants) stays `double` throughout.
  - **Important asymmetry for Job B**: this economy (upgrade only the O(n) accumulators, not the
    O(L) trace) is what keeps the primal's MPFR path cheap. It is **not** obviously available to
    the adjoint (see Q2).
- **Legacy, `ptd_expected_waiting_time_mpfr`** (`phasic.c:9744-9830`): builds a wholly separate
  MPFR-typed compute graph (`ptd_graph_ex_absorbation_time_comp_graph_mpfr`) and consumes that —
  more expensive, kept only "for comparison / safety-net during transition" per the in-code
  comment.

**Conversion back to double** (both paths): `mpfr_get_d(result[i], MPFR_RNDN)` per vertex, with an
explicit `mpfr_inf_p` check mapped to `INFINITY` (9945-9949) — a simple, final truncation back to
double precision for the caller.

### Q2: could the same pattern apply to the reverse/forward-mode adjoint?

**First, confirming the existing gate is decline-only, as the task assumed.**
`ptd_dbg_tape_needs_mpfr` (`phasic.c:10643-10664`) computes a condition number from `nm[]` (the
theta-adjoint's own recorded elimination multipliers — the in-code comment at 10635-10642 confirms
this "mirrors `ptd_expected_waiting_time`'s gate EXACTLY... nm[] carries the same numeric
multipliers as reward_compute_graph"), compares against the same `1e12`/`PHASIC_CONDITION_THRESHOLD`
default, and simply `return cond > thr;` — called at `phasic.c:10783` (`ptd_moments_grad_theta`),
`10960` (`_log`), `11221` (`_dph`), each time causing an immediate `return -1` (decline → Python
falls back to FD). **It never escalates anything to MPFR itself** — confirmed directly, matching
the task's premise.

**Is a mechanical translation feasible, or is there a real numerical-methods wrinkle?**
I reasoned through this from first principles and checked it against the actual C code
(`ptd_moments_grad_theta`, `phasic.c:10738-10881`), rather than treating it as a guess:

Reverse-mode AD's chain-rule composition (local partial derivative × incoming adjoint, summed) is
itself just more elementary arithmetic — multiply, add, subtract, reciprocal, negate. There is no
step in the mathematics of reverse-mode differentiation that behaves differently under
arbitrary-precision arithmetic than under double precision; precision only affects the accuracy of
each elementary operation's result, uniformly whether the pass is "forward elimination" or
"backward adjoint accumulation." Concretely, every operator used in stages 0-2
(`phasic.c:10766-10870`) — `+=`, `*`, `1.0/(*rf)` (reciprocal, case 2), `1.0 - *rf` (case 4),
`(*rf)/(*rt)` (quotient, case 5), negate (case 4's reverse), and the reverse-mode counterparts at
10838-10846 (`*bf = (*bf)*(-1.0/(s0[i]*s0[i]))` — reciprocal-square derivative;
`*bt += v*s1[i]; *bm += v*s0[i]` — product-rule terms; `*bt += v/(*rt)`-style quotient-rule terms) —
has a direct MPFR equivalent (`mpfr_mul`, `mpfr_add`, `mpfr_sub`, `mpfr_div`, `mpfr_neg`,
`mpfr_ui_div`/`mpfr_sqr`), all already in active use elsewhere in this same file for the primal's
MPFR path. **So: (a), a mechanical translation — no new numerical-methods research needed for the
elementary operations themselves.** I did not find, and do not believe there is, a
"reverse-mode-specific" incompatibility with arbitrary-precision arithmetic; this is consistent
with arbitrary-precision automatic differentiation being a well-established, unremarkable technique
generally (used e.g. in `mpmath`-based AD, high-precision variants of ADOL-C).

**However, "mechanical" is not "cheap," and there is a real asymmetry with the primal's escalation
that the task's framing doesn't fully anticipate:**

1. **Self-consistency requires upgrading stages 0-2 together, not just the accumulator.** The
   primal's MPFR-A economy works by trusting the O(L) trace's `multiplier` constants as already-safe
   doubles and only elevating the O(n) accumulation. But the adjoint's stage 0 (`phasic.c:10766-10779`)
   *records* `s0[i]`/`s1[i]` — per-operation primal snapshots that stage 2's quotient-rule terms
   (case 5, `-s0[i]/(s1[i]*s1[i])`, exactly the kind of expression that catastrophically cancels
   when `s1[i]` is tiny — i.e. exactly the "one rate ~1e-8" scenario the conditioning-floor docs
   describe) directly depend on. If stage 0 is left in double precision, those snapshots are already
   corrupted before stage 1/2 ever runs in MPFR — upgrading only the backward accumulation would not
   fix the root cause. A correct MPFR adjoint would need stage 0 (the forward moment chain) run in
   MPFR too, at minimum for `s0`/`s1`/`mem`, not just the K-moment output accumulators.
2. **This means far more arrays become `mpfr_t` than in the primal's path.** The primal only
   upgrades `result[]` (size n) plus 2 scalars. A self-consistent adjoint would need `mem` (size
   `md`), `s0`/`s1`/`nm` (size `L` — recall CLAUDE.md's own characterization of the trace as
   "O(n³) record"), plus `seeds`/`snaptos`/`dm`/`bar_out`/`adj`/`bmem`/`binp`/`J_out`. Every one of
   these becoming an array of `mpfr_t` (not `double`) means per-element `mpfr_init2`/`mpfr_clear`
   pairs (MPFR's C API has no operator overloading — no cheap "just change the type"), i.e. up to
   `O(L)` individual heap allocations where today there is one `malloc` of `L` doubles. For graphs
   where `L` is large this is a real, possibly severe, memory/performance cost that the primal's
   escalation never pays (it only ever allocates `O(n)` `mpfr_t` values). The atlas's own
   characterization — "moment graphs are typically modest-sized," "tolerable" only in that
   context — makes this a plausible-but-unquantified concern, not a proven blocker.
3. **Memory-safety surface.** The atlas independently found these three functions have **zero
   NULL-checked allocations and no size guard** (unlike the sojourn function, which got both after
   adversarial review). Converting ~14 arrays to `mpfr_t` while preserving correct
   `mpfr_clear`-on-every-early-return-path behavior, in a function family that doesn't currently even
   check `malloc` for NULL, is a substantially larger correctness-engineering burden than "swap
   `double` for `mpfr_t`" suggests — genuine `mpfr_t` leaks (leaked GMP limb buffers) are more
   consequential than a leaked flat `double*` array.
4. **Precision/threshold consistency with the primal is required, not optional.** For the
   gradient to be a locally-consistent derivative of the value actually returned to the caller
   (which may itself be MPFR-computed), an MPFR adjoint would need to trigger under the *same*
   condition and use the *same* chosen precision as whatever the primal did for that θ — otherwise
   you reintroduce exactly the inconsistency `ptd_dbg_tape_needs_mpfr`'s own comment describes
   guarding against today.
5. **An open question the conditioning-floor docs don't resolve**: the existing gate's condition
   number is computed from `nm[]`/`reward_compute_graph`'s elimination multipliers — a proxy tuned
   for the *primal value's* stability. The conditioning-floor finding says the corrupted quantity is
   specifically the **gradient's sub-dominant component**, which is sensitive to the quotient-rule
   terms in stage 2 (`s0[i]/s1[i]²`-type expressions) — a *different* sensitivity than the plain
   multiplier spread the existing gate measures. It is plausible (not proven here) that the existing
   gate's metric under-detects gradient-specific ill-conditioning even when it correctly detects
   primal-value ill-conditioning — i.e. simply "reusing the same metric with a stricter threshold"
   (part of the cheap alternative in Q3) may not, by itself, actually catch the regime the
   conditioning-floor docs found. This would need to be checked against the actual historical
   repro (`b3-experiment-findings.md` DR-A) before committing to either fix.

**Overall answer to (a) vs (b): mostly (a) — mechanical, no new AD-theory wrinkle — but with a
real, non-trivial "how much of the pipeline needs upgrading, and how do you keep memory-safe while
doing it" engineering cost that is materially larger than the primal's own MPFR path, because the
adjoint's stage 0 snapshots feed directly into the cancellation-prone stage-2 terms, unlike the
primal which only ever upgrades an O(n) accumulator.**

### Q3: recommendation

Given: (i) FD is unambiguously worse in this exact regime (2.2e7 relerr vs. ~3 relerr per the
historical de-risking finding) — so this is not an argument for reverting to FD generally; (ii) the
conditioning-floor docs frame this as an "inherent float64 precision floor," not a fixable
implementation bug, for the *current* double-precision adjoint; (iii) no gate anywhere currently
detects or flags this regime at all (confirmed absent from `CLAUDE.md` and, as far as I can tell
from the atlas's own accounting, never turned into a pinned test) — the priority ordering I'd
recommend is:

1. **Do the cheap thing first, independent of whether MPFR-adjoint is ever attempted**: characterize
   the regime (re-run/confirm the historical DR-A repro), and extend/add a decline check so this
   specific regime causes a clean, logged fallback to FD rather than a silent moderately-wrong
   "exact" value. Per the risk in Q2 point 5, this may need a **new**, gradient-specific
   condition-number metric (e.g. over the recorded `s1[i]` magnitudes / the quotient-rule term
   magnitudes) rather than simply lowering the existing `ptd_dbg_tape_needs_mpfr` threshold, since
   that threshold's metric was designed for the primal value's stability, not the gradient's. This
   is small, additive, matches `feedback_no_silent_fallbacks`, and does not block on any of the
   larger work below.
2. **Do not build a full MPFR adjoint as a first move.** The cost (upgrading stage 0 through 2,
   likely `O(L)`-many new `mpfr_t` allocations where the primal only ever needed `O(n)`, in a
   function family with no existing NULL-checks or size guards) is disproportionate to a regime
   the source docs themselves describe as an edge case ("extreme mixed-scale theta"), and the
   payoff is capped by the same "genuine precision floor" framing — i.e. this is exactly the kind
   of investment that should follow, not precede, actually confirming (via the decline gate in
   step 1) how often production models land in this regime at all.
3. **If, after step 1, telemetry/user reports show this regime is hit often enough that "decline to
   FD" is unacceptably lossy in practice**, an MPFR adjoint becomes worth scoping properly — at
   that point it should be built to upgrade stage 0 as well as stages 1-2 (not just the
   accumulators), reuse the primal's existing precision-selection formula for consistency, and get
   the same NULL-check / size-guard treatment the sojourn function already received under
   adversarial review, given this codebase's established pattern of finding real bugs exactly
   this way.

My honest recommendation: **Job B is out of scope for near-term implementation.** The
documentation/decline-gate route (step 1) is the right immediate action and is cheap; the full
MPFR adjoint (steps 2-3) should be deferred pending evidence it's actually needed, not built
speculatively.

### Job B risks / unknowns

- I did not re-run the historical DR-A repro myself to confirm the conditioning-floor finding still
  reproduces on current `master`; I'm relying on the atlas's description of three independent
  historical documents converging on the same result.
- Whether the existing gate's condition-number metric under- or over-detects the gradient-specific
  regime (Q2 point 5) is an open question I could not resolve from static reading alone — would
  need an actual numeric experiment against the DR-A fixture.
- The `O(L)` memory-blowup concern for an MPFR adjoint (Q2 point 2) is reasoned from the code
  structure, not measured — I do not have a concrete number for typical `L` in production
  moment-graphs to say how bad this would actually be.
- `ptd_dbg_tape_needs_mpfr` is also called from `ptd_sojourn_grad_theta_subset`
  (`phasic.c:11529`), but the atlas already establishes (and I have not re-disputed) that this call
  is inert there — the sojourn primal (`ptd_expected_sojourn_time_subset`) has no MPFR path at all,
  so Job B's scope is the three moment functions only, not sojourn.

---

## Cross-batch conflict check

**Job A (callback) vs. batch (b) (formula-mode + skeleton refactor): yes, direct design overlap —
should not be planned independently.**
Both are, at their core, "teach the exact-gradient stage-2 contraction a new way to compute
`∂w_e/∂θ` for a non-constant weight rule." CLAUDE.md's own "Reverse-tape skeleton duplication" note
(already in the repo, not something I'm inferring) explicitly says the shared stage-0/1/2 core
across `ptd_moments_grad_theta`/`_dph`/`_log` should be extracted into one static helper "before a
4th variant (`'formula'`) is attempted." Job A's callback-mode exact gradient would be architecturally
a 5th variant of the exact same shape (a new stage-2 contraction plugged into the same stage-0/1
core; see Job A Q3). **Recommendation: sequence the skeleton-extraction refactor once, then implement
formula-mode's and callback-mode's stage-2 variants on top of the shared core, rather than doing the
extraction twice or skipping it for one of the two.** This is a genuine synergy, not just a
scheduling collision avoidance — refactoring once for both is strictly cheaper than the current
trajectory of "copy stage-0/1, write a new stage-2, fix the same bug three times" that already
happened once (the `coefficients_length==0` NULL-deref fix, per CLAUDE.md, was applied to two
copies but had to be found and fixed independently).

**Job B (MPFR adjoint) vs. batch (a) (rewards support for moments) and batch (b) (formula +
skeleton refactor): yes, all three would touch the same three C functions around the same time —
a real sequencing conflict if attempted concurrently.**
`ptd_moments_grad_theta` (10738-10881), `_dph` (11142-11338), `_log` (10917-11063) currently take
**no `rewards` parameter at all** — this is precisely why `pmf_and_moments_from_graph`'s dynamic
guard (`__init__.py:7559-7566`) force-disables the exact path whenever rewards are passed. Batch
(a)'s "rewards support for moments" almost certainly means adding a rewards array to these same
three functions (seeding `seeds[v] = rewards[v]` instead of `1.0` at stage 0, `phasic.c:10792`, the
same change already made in the primal `ptd_expected_waiting_time`, `phasic.c:10049-10056`). If a
full MPFR adjoint were ever built (Job B step 3), it would rewrite the **entire body** of these same
three functions. And batch (b)'s skeleton-extraction refactor also touches the same functions'
stage-0/1 core. **Given my recommendation to defer Job B (out of scope near-term), the immediate
practical conflict is only between (a) and (b)** — both want to edit the same ~600 lines of C soon;
whichever lands first should structure its change (adding a rewards parameter, or extracting the
shared core) so the other doesn't have to redo work. If both are truly imminent, doing the skeleton
extraction *first* (as CLAUDE.md already recommends) and then adding rewards support as a parameter
on the now-shared core is the lower-total-effort ordering.

---

## Recommended sequence position

- **Job A (callback exact gradient): plausible cheap-ish win, but sequence it *after* (not
  instead of, and not before) the batch-(b) skeleton-extraction refactor.** Concretely: (1) land the
  stage-0/1/2 extraction that batch (b) already needs for formula mode; (2) add a new, small
  "return pre-contraction adjoint" exit point on the extracted core (the only genuinely new C
  needed for Job A); (3) implement callback-mode's Python-side `jax.vmap(jax.grad(...))` contraction
  on top of that exit point, gated behind an explicit opt-in (since not all callbacks are JAX-native
  and there is currently no reliable runtime way to tell without trying). Doing it in this order
  means Job A adds close to zero *additional* C surface beyond what batch (b) is already paying for.
  If batch (b) is not actually happening soon, Job A alone still requires the smaller "new exit
  point" C change described in Q2/Q3 and is a legitimate standalone (if modest) batch.
- **Job B (MPFR adjoint): recommend OUT of scope for near-term work.** Do the cheap,
  independent decline-gate/documentation step now (it's useful regardless of Job B's fate and
  matches `feedback_no_silent_fallbacks`); defer the actual MPFR-precision adjoint rewrite until
  there's evidence (from telemetry/user reports after the gate ships) that the regime is hit often
  enough in practice to justify the cost profiled above. If it is ever greenlit, sequence it after
  both batch (a) and batch (b) have landed on the shared/refactored core (so it's one MPFR rewrite
  of the final shared skeleton, not three separate rewrites of soon-to-change functions).
