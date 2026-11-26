# Feasibility scoping: PDF-at-time-t gradient (`ptd_graph_pdf_with_gradient`) and daisy-chain `stop_probability(dt)` exact gradients

**Status:** scoping only, read-only investigation, one throwaway empirical
probe compiled and run against the currently-installed native library (no
repository files changed; the probe programs live in `/tmp` and are not part
of this deliverable). All line numbers are from a fresh read of the current
tree on 2026-08-05. The two atlas docs named in the task
(`exact-fd-atlas-loose-ends-memory.md` §3.1, `exact-fd-atlas-loose-ends-docs.md`)
were read first as a hypothesis source, then every claim was re-derived from
source or measured directly; see §7 for exactly what carried over and what
didn't.

## 0. Headline verdict

**This is not "verify a sign, add a binding." `ptd_graph_pdf_with_gradient`
is broken in at least four independent, structural ways, one of which means
its *forward* PDF value is wrong by ~72% on a trivial 2-state test graph —
not merely "a gradient with a debatable sign," but a function that does not
compute the phase-type PDF at all as this codebase defines it.** The
"empirically determined minus sign" is the least of its problems. Treating
this dead code as reusable, even after a sign fix, would be a mistake.

Separately, `stop_probability(dt)` (the primal daisy-chain uses) is a
**different algorithm** from what `ptd_graph_pdf_with_gradient` implements —
not a variant, not the same formula with different bookkeeping. The claimed
dependency ("daisy-chain's exact gradient is gated on the PDF/uniformization
batch landing first") is **refuted**: fixing or reusing
`ptd_graph_pdf_with_gradient` would not unlock, simplify, or share one line
of code with a daisy-chain adjoint. If anything the daisy-chain math is
*simpler* once one prerequisite (pin `granularity` to a θ-independent
constant) is accepted, because it never needs a Poisson mixture or a
θ-dependent uniformization rate at all.

Recommended sequencing: **last, and this is a real follow-up initiative, not
a batch.** See §5 for a concrete, de-riskable staging that still extracts a
smaller, genuinely low-risk win (the daisy-chain final epoch) before
attempting the hard part (intermediate-epoch backprop-through-time).

---

## 1. What `g.pdf(t, granularity)` / `stop_probability(t, granularity)` actually compute (the primal, re-derived from source)

This had to be established first, because the task's own working hypothesis
("uniformization = Poisson-weighted mixture over DTMC steps, exact for any
λ ≥ max-rate, only the tail truncation is approximate") is the *textbook*
uniformization method — and it is **not** what this codebase's primal
`pdf`/`stop_probability` implement.

Driver (`api/cpp/phasiccpp.h:1412-1460`, `Graph::pdf`; `:1518-1550`,
`Graph::stop_probability` — **both** call the same pair of C entry points):

- `ptd_probability_distribution_context_create(graph, granularity)`
  (`src/c/phasic.c:12644`): if `granularity == 0`, auto-selects
  `granularity = max(1000, 2 * max(512, max_exit_rate))` (note the **512
  floor** on `max_rate`, `src/c/phasic.c:12648`, before the loop even looks
  at real vertices) — then validates `rate/granularity <= 1.0001` for every
  vertex. This confirms: **`granularity` *is* the uniformization rate λ**
  (not "steps per unit time" in the abstract — it is used directly as the
  divisor in each per-step transition probability).
- The very first `ptd_dph_probability_distribution_step` call inside
  `_ptd_dph_probability_distribution_context_create` (`phasic.c:12453-12572`)
  runs with `priv3 = 1` (**not** `1/granularity`) — a **special,
  unscaled, one-time redistribution** of the starting vertex's edge weights,
  which the comment at `phasic.c:12563-12564` calls "instantaneous
  transition from starting vertex." Only *after* this special step does the
  caller overwrite `priv3 = 1.0/granularity` (`phasic.c:12735,12746`) for all
  subsequent real-time steps. **The starting vertex's outgoing edges are
  literal initial probabilities (an IPV), not CTMC rates**, and the primal
  algorithm treats them as such by construction, exactly once, at t=0.
- Every subsequent `ptd_probability_distribution_step` (`phasic.c:12774-12787`)
  does **one** matrix-vector product against the fixed embedded DTMC
  `P(θ) = I + Q(θ)/λ` (self-loop `1 - exit_rate/λ`, off-diagonal
  `weight/λ`), harvesting absorbed mass into `pmf`/`cdf` and reporting
  `pdf = pmf * granularity` (a probability-mass→density conversion, since
  `Δt = 1/granularity`).
- `Graph::pdf(time, granularity)` (`phasiccpp.h:1451-1459`) then simply
  **steps this DTMC forward until `time >= context->time`** and reads off
  `_pdf[floor(granularity * time)]` — i.e. it evaluates the embedded DTMC at
  **exactly one** fixed step count `k = floor(granularity·t)`, with **no
  Poisson mixture at all**.

So the primal is: *one* application of `P(θ)^k` to the IPV, read out at
`k = floor(λt)`, where `λ = granularity`. This is a first-order
(Euler-style) discretization of the CTMC; as `granularity → ∞`,
`(I + Qt/λ)^{λt} → e^{Qt}` by the standard compounding-limit definition of
the matrix exponential, with `O(1/granularity)` error at any finite
granularity. **The task's framing — "the forward pass itself only converges
as granularity→∞" — is correct for the *primal*, and I confirmed it
numerically** (§3): `g.pdf(1.0, 0) = 0.27067039`,
`g.pdf(1.0, 200000) = 0.27067057`, converging to the closed-form
`θ₁e^{-θ₁} = 0.27067057` for the trivial Exponential(θ₁) test graph built
below.

`stop_probability(time, granularity)` (`phasiccpp.h:1518-1550`) is the
**identical mechanism** — same context-create, same step loop — just reading
out the full `probability_at` vector (state occupancy) instead of harvesting
only the absorbed mass. **Confirmed: daisy-chain's intermediate-epoch
transitions use the exact same primal as `g.pdf`, not a different one**
(task item 4, answered: same).

## 2. What `ptd_graph_pdf_with_gradient` actually computes (re-derived, then measured)

### 2.1 The intended formula (per its own comments)

`compute_pmf_with_gradient` (`phasic.c:12834-13075`, static helper) computes,
per its header comment:

```
PMF(t;θ) = Σ_k Poisson(k; λ(θ)t) · π_k(θ)
```

where `π_k(θ)` is the mass absorbed exactly at DTMC step `k` under
`P(θ) = I + Q(θ)/λ(θ)`, and **`λ(θ) = max_v exit_rate_v(θ)`** — the actual
current max exit rate, computed fresh from `θ` (`ptd_graph_pdf_with_gradient`,
`phasic.c:13103-13138`), with **no 512 floor and no `granularity`
involvement at all**. `ptd_graph_pdf_with_gradient` then sets
`pdf(t;θ) = λ(θ)·PMF(t;θ)`. **This is the textbook exact-uniformization
Poisson mixture** — a genuinely different algorithm from §1's primal, not a
variant of it. If this formula were computed correctly and summed to
convergence, it actually would *not* need `granularity → ∞` at all — for a
FIXED λ = max_exit_rate, the infinite Poisson series is exact; only its
tail truncation (`max_jumps`, chosen by an internal 6σ rule) is
approximate, and the code targets ~1e-15 truncation error. **So the task's
suggested framing ("exact gradient of the granularity-truncated forward,
matching the forward's own convergence") does not describe what this
function is even attempting** — it targets the *different*, classically-exact
Poisson-mixture quantity, decoupled from the primal's Euler-power-iteration
quantity and from `granularity` entirely.

### 2.2 Re-deriving ∂PMF/∂θ by hand

With `λ = λ(θ)`:

```
∂PMF/∂θ_p = Σ_k [ ∂Poisson(k;λt)/∂λ · (∂λ/∂θ_p) · π_k  +  Poisson(k;λt) · ∂π_k/∂θ_p ]
∂Poisson(k;λt)/∂λ = Poisson(k;λt) · (k − λt)/λ            (standard derivative)
```

This matches the code's `poisson_grad_factor = poisson_k*(k-lambda_t)/lambda`
(`phasic.c:13036`) exactly — that piece is correct.

The problem is `∂π_k/∂θ_p`, i.e. `prob_grad`, accumulated through the DP
recursion (`phasic.c:12991-13011`):

```c
next_prob[to_idx]      += prob[v] * weight / lambda;
next_prob_grad[to_idx][i] += prob_grad[v][i]*weight/lambda + prob[v]*weight_grad[i]/lambda;
...
self_prob = (lambda - exit_rate) / lambda;
next_prob_grad[v][i] += prob_grad[v][i]*self_prob + prob[v]*(-exit_rate_grad[i])/lambda;
```

Both updates differentiate `X/λ` (or `(λ-Y)/λ`) **treating λ as a constant**
— there is no `− prob[v]*weight*lambda_grad[i]/lambda²` term for the
off-diagonal transition, and no `+ prob[v]*exit_rate*lambda_grad[i]/lambda²`
term for the self-loop. But `λ = λ(θ)` is **not** constant whenever the
max-rate vertex has any parameterized edges (the common case) — this
recursion runs `k` times, so the omission compounds multiplicatively across
every step, in a way that cannot be repaired by a single global correction
term applied once at the end.

The code's own comment (`phasic.c:13220-13223`) frames the final
`pdf_gradient[i] = pmf_grad[i]*lambda − pmf*lambda_grad[i]` as fixing
"double-counting" of λ. **My derivation shows the opposite direction of
error: the DP recursion *under*-counts λ's θ-dependence (missing terms in
every one of the k steps), not over-counts it.** A single end-of-computation
subtraction cannot correct a per-step, k-compounding omission — the
magnitudes and even the functional form don't match (see §3.2: flipping the
sign to a "+" doesn't fix it either).

### 2.3 A second, independent bug: linear-only, and inconsistent within itself

`compute_pmf_with_gradient` branches per edge on `coefficients_length > 1`
(parameterized) vs `== 1` (constant) (`phasic.c:12941,12980`), while
`ptd_graph_pdf_with_gradient`'s own λ/λ-gradient computation branches on
`coefficients_length >= n_params && n_params > 0` (`phasic.c:13118,13123,13151`).
**These two conditions disagree whenever `n_params == 1`**: a length-1
coefficient array is parameterized by the outer check (`1 >= 1`) but
constant by the inner check (`1 > 1` is false). Confirmed empirically in
§3.3: for a single-parameter graph, the reported gradient is nonzero and
varies with θ, while the reported PDF **value never changes at all** as θ
varies — the DP recursion is silently using the coefficient as a frozen
constant. Separately, neither function has any branch for `weight_mode`
other than the linear dot product — no `log` (product-in-log-space, per
`weight_mode='log'` semantics elsewhere in this codebase), no `formula`,
no `callback`. It hard-codes `weight = Σ c_k θ_k` unconditionally.

### 2.4 A third, independent bug: `granularity` is a dead parameter

`compute_pmf_with_gradient`'s signature accepts `granularity`
(`phasic.c:12839`) but **never reads it** — grepped the full 240-line body:
the only other occurrence of the identifier is inside a log-message string,
not a variable use. `max_jumps` (the Poisson-tail truncation) is derived
solely from a 6σ rule on `lambda*time`, independent of any caller-supplied
granularity. `ptd_graph_pdf_with_gradient` itself only uses its `granularity`
argument (if `0`) to compute a fallback value that is then passed into
`compute_pmf_with_gradient` and ignored there too. **The `granularity`
parameter has zero effect on this function's output, for any value,
including `0`** — directly contradicting its own header doc ("Discretization
granularity (0 = auto-select)"). Confirmed empirically in §3.1.

### 2.5 A fourth, independent bug: the primal's special IPV redistribution is entirely missing

`compute_pmf_with_gradient` seeds `prob[starting_idx] = 1.0`
(`phasic.c:12872`) and then runs the **same** DP loop over the starting
vertex as every other vertex from `k=0` onward — dividing its edge weights
(which represent absolute initial *probabilities*, per §1) by `lambda` at
every step, exactly like a real transition rate. It never performs
§1's special "instantaneous, unscaled" redistribution step. **This means the
initial probability mass leaks out of the starting vertex gradually (at an
effective rate ≈ its own outgoing weight sum, damped by `1/λ` per step)
instead of landing at the correct states instantaneously at t=0** — a
fundamentally different (and wrong) initial condition, not a rounding
difference. Confirmed empirically in §3.2: forward PDF value off by ~72%
against ground truth and against the primal.

## 3. Empirical verification

Verification per `feedback_never_assume_verify_adversarially` / the user's
CLAUDE.md quality bar: rather than resting on the symbolic derivation alone,
I compiled a tiny standalone C harness against the **already-built** native
library (`liblibphasic.dylib` in the pixi env; confirmed the raw C symbols,
including `ptd_graph_pdf_with_gradient`, are exported via `nm -gU`) and
called it directly, bypassing the (nonexistent) Python/pybind path entirely.
Test graph: `start --(w=1·θ₀+0·θ₁)--> A --(w=0·θ₀+1·θ₁)--> B(absorbing)`,
i.e. Exponential(θ₁) with a fixed (θ₀-controlled but always-1.0) IPV, whose
closed form is elementary: `PDF(t;θ) = θ₁e^{-θ₁t}`,
`∂PDF/∂θ₁ = e^{-θ₁t}(1-θ₁t)`, `∂PDF/∂θ₀ = 0`.

### 3.1 Granularity is dead (θ=[1,2], t=1)

```
granularity=0(auto):          pdf=0.465088315870  grad=[0.194417749396, -0.291626624095]
granularity=1000:              pdf=0.465088315870  grad=[0.194417749396, -0.291626624095]
granularity=500000:            pdf=0.465088315870  grad=[0.194417749396, -0.291626624095]
granularity=5:                 pdf=0.465088315870  grad=[0.194417749396, -0.291626624095]
```
Identical to 12 decimal places across a 100,000× range of granularity values.

### 3.2 Forward value is wrong, and the analytic gradient contradicts the function's own finite difference

```
analytic grad (granularity=0)          = [ 0.194417749396, -0.291626624095]
own-FD dPDF/dtheta1 (central, eps=1e-6) =                    0.038126408558
true PDF (closed form)                  = 0.270670566473
true dPDF/dtheta1 (closed form)          = -0.135335283237
primal g.pdf(1.0, granularity=0)         = 0.27067039404886784   (Python, via real phasic.Graph)
primal g.pdf(1.0, granularity=200000)    = 0.2706705664687083    (→ true closed form 0.27067057)
```
`ptd_graph_pdf_with_gradient`'s forward value (0.465088) is **72% too high**
relative to both the closed form and the real primal `g.pdf()` (which
converges correctly as granularity grows, confirming §1). Its reported
analytic gradient for θ₁ (−0.291627) does not match a central finite
difference of **its own** forward output (+0.038126) — not just a different
magnitude, the **opposite sign**. I also checked whether flipping the
"empirically determined" minus sign to a plus would repair this: it does
not (algebraically back out `pmf_grad[1] = -0.029541` from the reported
values; the "+" variant gives `+0.173461`, still matching neither the
function's own FD (0.038126) nor the true gradient (−0.135335)). **This is
not a sign bug — the whole computation is structurally wrong**, consistent
with §2.2's finding of missing per-step chain-rule terms compounded with
§2.5's wrong initial condition.

### 3.3 The n_params==1 inconsistency (§2.3), measured

Single-parameter graph, `A→B` edge coefficient `[1.0]` nominally meaning
"rate = θ₀":
```
theta0=1.0: pdf=0.3678794412  grad=-0.3678794412
theta0=2.0: pdf=0.3678794412  grad=-0.1839397206
theta0=3.0: pdf=0.3678794412  grad=-0.1226264804
theta0=4.0: pdf=0.3678794412  grad=-0.0919698603
```
The reported **PDF value never changes** as θ₀ is varied 1→4 (the DP
recursion is using a frozen constant, exactly as §2.3 predicted), while the
reported **gradient does change** with θ₀ and is nonzero throughout — a
function reporting a nonzero derivative for an input it is provably not
using. This is about as unambiguous a "the gradient is definitionally wrong"
signature as exists.

**Verdict on the sign-convention question (task item 1):** genuinely
suspect, but the sign is not even the primary defect. Fixing only the sign
would still leave a function that (a) ignores its own `granularity`
parameter, (b) implements the wrong initial condition, (c) silently
ignores θ for single-parameter graphs, and (d) supports only linear weight
mode. "Empirically determined... gives correct results" (the code's own
comment) does not hold up under the numbers above; there is no evidence this
function was ever checked against a working oracle.

## 4. Reachability (task item 3)

```
grep -rn "pdf_with_gradient" src/cpp/ api/cpp/     → (no matches)
grep -rn "pdf_with_gradient" src/phasic/           → (no matches)
grep -rln "pdf_with_gradient" tests/               → (no matches)
grep -rln "pdf_with_gradient" . (py/c/cpp/h/md)    → only api/c/phasic.h, src/c/phasic.c,
                                                       and two prior planning docs (b3-derisking-strategy.md,
                                                       exact-fd-atlas-SUMMARY.md)
```
Confirmed independently: **zero callers anywhere** — pybind, C++, Python,
and tests all have no reference. The loose-ends memory's reachability claim
is correct.

**What wiring it up would require (if it were correct, which it is not):**
a pybind binding (`.def("pdf_with_gradient", ...)` or similar on `_Graph`),
a Python-level model builder mirroring `pmf_from_graph`'s pattern (private
clone graph, build once per param_length, `jax.pure_callback` with F64
dtype, decline-to-FD `exact_grad` kwarg with INFO-level logging on
fallback, matching every other B3 batch's established shape), plus a
discrete-vs-continuous dispatch and weight_mode gating (raise, don't
silently linearize, for `log`/`formula`/`callback`). None of this is
worth doing against the current C implementation — see §5.

Also checked: there is no other `_with_gradient` C function
(`grep -n "_with_gradient" api/c/phasic.h` → only this one hit), and no
gradient variant of `ptd_probability_distribution_context_create`/`_step`
(the actual primal `stop_probability`/daisy-chain mechanism) exists at all,
in any form, anywhere in the C layer. There is **no partial work** to build
on for the daisy-chain side; it would be 100% new.

## 5. Daisy-chain `stop_probability(dt)` exact gradient (task item 4)

### 5.1 Same primal mechanism as `g.pdf`, confirmed

§1 already established `stop_probability` and `pdf` share the identical
C driver. `graph_builder_ffi.cpp:1829` (`DaisyChainJointProbsFfiImpl`) and
`:2065` (`DaisyChainSojournFfiImpl`, intermediate epochs) both call
`g.stop_probability(t_step, granularity)` per epoch, after
`ptd_graph_update_ipv` + `ptd_graph_update_weights(..., use_log=false)`
(hard-coded linear; `daisy_chain_joint_probs` explicitly rejects
`weight_mode='log'` at the Python layer, `__init__.py:10173-10180`, for
exactly this reason).

### 5.2 The claimed dependency is refuted

Because `ptd_graph_pdf_with_gradient` implements a **different algorithm**
(Poisson-mixture over a θ-dependent λ, §2.1) than `stop_probability`
(fixed-k Euler power-iteration at a caller-chosen λ=granularity, §1), fixing
or wiring up the former would transfer **zero code and zero formulas** to
the latter. The only thing that legitimately carries over is a *design
pattern*: `compute_pmf_with_gradient`'s forward-mode (tangent) propagation
of `prob_grad` alongside `prob`, discarding history at each step, is a
sound architectural choice — it avoids storing a `k`-step tape (`k` can be
thousands to tens of thousands per epoch) at the cost of scaling with
`n_params` rather than being parameter-count-independent. That pattern is
worth reusing conceptually. Nothing else is.

### 5.3 A load-bearing prerequisite the current default violates: `granularity` must be pinned

`daisy_chain_joint_probs(..., granularity: int = 0)` and
`Graph.svgd(..., daisy_chain_granularity: int = 0)` **default to auto**
(`__init__.py:10092`, `:5069`, `:5274`). Per §1, `granularity=0` resolves to
`(size_t)(2 * max(512, max_rate(θ)))` — an **integer-cast, θ-dependent**
value, re-derived fresh from the *current* θ on every call
(`ptd_probability_distribution_context_create` runs this after
`update_weights` has already set the current epoch's θ). This makes the
embedded DTMC `P(θ)` itself change identity (not just its entries) as θ
moves across integer-granularity thresholds — the forward map
`θ ↦ stop_probability(dt;θ)` is a **piecewise-constant-λ, not even
everywhere-smooth, function of θ under the current default**. Any exact
gradient is meaningless without first **pinning `granularity` to an
explicit, θ-independent constant for the whole SVGD run** — analogous to
(but distinct from) the "IPV is not an SVGD parameter" convention already
established elsewhere in this codebase. This is a **new prerequisite fix**,
not something either existing batch already covers, and it must be decided
(what constant? conservative upper bound over the prior's support? a
user-supplied hyperparameter with no silent default?) before any gradient
derivation makes sense. The current FD-based gradient sidesteps this
because FD only needs local smoothness in an eps-neighborhood, which holds
almost everywhere; an exact/analytic treatment cannot paper over it the
same way.

### 5.4 The IPV-passing wrinkle (task item 4's explicit ask)

Read `Graph._daisy_chain_svgd_model` (`__init__.py:4254-4392`) and
`daisy_chain_joint_probs` (`:10084-10363`), plus the FFI loop
(`graph_builder_ffi.cpp:1762-1882`): each epoch does
`update_ipv(ipv_work) → update_weights(theta_epoch) → stop_probability(dt) →
collapse t-aux pairs → project into next epoch's IPV`. **`ipv_work` is a
vector** (length `n_ipv`, the JSP graph's `_ipv_target_indices` count — not
a scalar), threaded epoch-to-epoch. This means the natural gradient
primitive a correct exact implementation needs per epoch is not "a scalar
gradient" (as every other shipped B3 adjoint produces) but **the full
Jacobian of the collapsed output vector w.r.t. both `ipv_in` (n_ipv×n_ipv)
and `theta_epoch` (n_ipv×n_params)** — because chaining multiple epochs by
hand (rather than via a generic AD system) requires composing these
Jacobians epoch-by-epoch (a `n_ipv × n_ipv` matrix multiply per hop) to
back-propagate a cotangent from the final epoch's output to any earlier
epoch's θ. **This is a strictly larger primitive than anything B3 has built
so far** (moments/sojourn adjoints all produce a scalar-output gradient from
a closed-form elimination solve; nothing else in B3 needs to expose or
chain a Jacobian). It is "chain rule across epochs" in spirit exactly as
the task's framing suggests, but the object being chained is a matrix, not
a number, and that is a genuine, non-cosmetic wrinkle: it changes the cost
model (see §5.5) and the API shape (a per-epoch VJP-style primitive,
`(cotangent_on_output) → (cotangent_on_ipv_in, cotangent_on_theta)`, is the
right unit — matching how the *existing* FD wrapper is already shaped,
encouragingly: `_autodiff_bwd` in `daisy_chain_joint_probs`
(`__init__.py:10347-10361`) already treats the **whole multi-epoch chain**
as one opaque function with a cotangent-dot-product VJP interface. An exact
implementation would slot into the same external shape; the work is
entirely inside `_forward`'s replacement.

### 5.5 Cost/memory: this is backprop-through-time, not a closed-form replay

Every other B3 batch differentiates a **fixed-size elimination tape**
(`O(n)` per replay, no unrolling over time). Daisy-chain's per-epoch
`stop_probability(dt)` is `k = floor(granularity·dt)` repeated applications
of the *same* matrix `P(θ)` — `k` can be in the thousands to tens of
thousands for realistic `dt`/`granularity` combinations. Naive reverse-mode
(store all `k` intermediate `prob` vectors, backprop through the tape) is
`O(k·n_vertices)` memory — for models at the scale CLAUDE.md cites elsewhere
(`n` up to ~7×10^5), this is infeasible without gradient-checkpointing
(store `O(√k)` checkpoints, recompute segments, ~2× compute cost) — a real
design decision with no free lunch, never before needed in this codebase.
Forward-mode (tangent) propagation, matching `compute_pmf_with_gradient`'s
architecture, sidesteps the memory blowup entirely (`O(n_vertices ·
n_seeds)` memory, no history) at the cost of scaling with the number of
seed directions — and per §5.4, the seed count here is `n_params + n_ipv`
per epoch (both θ-directions and IPV-directions must be seeded to build the
Jacobian), not just `n_params`. Total cost order:
`O(n_epochs · (n_params + n_ipv) · k · n_edges)`. Whether this is cheap or
dominates the whole SVGD run depends on `n_ipv` (model-specific — could be
tens to hundreds of IPV-target states for two-locus-style models) and `k`;
**this needs a profiling pass before committing to an approach, not an
assumption**. No existing profiling data covers this (the `project_svgd_perf_ux`/
`project_cache_load_cpu_bound` memories profile the elimination-tape path,
which has no analogue here).

### 5.6 A cheaper, lower-risk partial win exists: the final epoch

`final_read='sojourn'` (the current **default**, `__init__.py:10093`) reads
the final epoch via `joint_sojourn_graph()` — an exact, granularity-free
**elimination** solve (`r_v · expected_sojourn(v) · handoff_mass`), *not*
uniformization at all. But the current code still wraps this in the exact
same bulk-FD `custom_vjp` as `final_read='stopprob'`
(`__init__.py:10312-10363` — `_forward` branches on which FFI call to make;
the surrounding `_autodiff`/`_autodiff_bwd` FD wrapper is identical either
way). **The final epoch's own internal exactness is not currently exploited
by the gradient at all.** Giving the final epoch's contribution an exact
gradient (reusing/extending the already-shipped B3 sojourn adjoint,
`ptd_sojourn_grad_theta_subset` per CLAUDE.md's joint-index section) while
leaving intermediate epochs on FD is a legitimate, much smaller, much
lower-risk staged batch that does not require solving §5.3's
granularity-pinning problem or §5.5's backprop-through-time problem at all
— it only needs the epoch-chain-rule composition (§5.4) applied to
**one** already-exact building block. This should be attempted (and
de-risked, and adversarially reviewed) **before** the intermediate-epoch
uniformization-adjoint work, both because it is smaller and because
succeeding at composing Jacobians across the IPV boundary for one epoch is
a prerequisite skill/pattern for composing it across several.

## 6. Scope/effort estimate (task item 5 — an honest answer, not an optimistic one)

**Standalone `ptd_graph_pdf_with_gradient` wiring: not worth doing as
"wire up the existing function."** A usable version requires, at minimum:
rewriting the forward pass to match §1's actual algorithm (fixed-k
Euler power-iteration with the special instantaneous IPV step) or
consciously choosing to keep the Poisson-mixture variant as a *distinct,
new, correctly-derived* primitive (viable in principle, since a fixed
λ=max_exit_rate Poisson mixture is genuinely exact for the CTMC's true PDF —
but this needs the whole §2.2 chain rule re-derived and re-implemented
correctly, the n_params==1 branch-inconsistency fixed, weight_mode gating
added, and the "is λ ≥ max_rate for the whole θ-range SVGD will explore"
question answered.) Either way this is a **from-scratch C
implementation** with its own MPFR-gate-style numerical de-risking, its own
adversarial review (this codebase's own history: 2 real bugs found in the
*simpler* moments default-flip, by review, not by the original
implementation's own tests), plus the standard C++/pybind/Python/
`pure_callback` wiring every other batch needed. Rough order: **comparable
to or larger than the discrete/was_dph moments-gradient batch** (which
itself needed an MPFR precision gate) — call it **1.5-3 weeks of focused
work**, not the "half-day, fix a sign" the dead code's presence might
suggest to someone who didn't re-derive it.

**Daisy-chain intermediate-epoch exact gradient: a genuinely new B3
sub-initiative.** Novel prerequisites (granularity pinning, §5.3), a novel
primitive shape (Jacobian not gradient, §5.4), a novel cost regime
(backprop-through-time / checkpointing, §5.5) that has no precedent
anywhere else in this codebase and needs its own profiling before a design
can even be chosen. This is **not** a batch-sized unit of work in the sense
the other five parallel batches are. Rough order: **multi-week**, and only
after a dedicated de-risking pass (per `feedback_derisk_and_reevaluate`)
establishes (a) what constant granularity is safe across the SVGD prior's
support for real models, and (b) whether checkpointed reverse-mode or
seeded forward-mode wins on realistic `n_ipv`/`k`/`n_vertices` — an
empirical question, not a judgment call to make on paper.

**The final-epoch partial win (§5.6) is comparatively cheap** — closer to
a normal batch size (days, not weeks), since it reuses an already-shipped,
already-verified adjoint and only needs the epoch-boundary chain-rule
composition for a single hop.

## 7. What carried over from the atlas docs, and what didn't

- The loose-ends memory's core claim — unwired, untested, "empirically
  determined" sign, zero callers — **holds** (§4).
- Its framing of the open question ("re-derive before trusting") was
  correctly cautious, but the actual defect is far larger than a sign
  question: the forward value itself is wrong (§2.5, §3.2), a second
  independent parameter-handling bug exists (§2.3, §3.3), and the
  `granularity` parameter is entirely dead (§2.4, §3.1) — none of which
  either atlas doc mentioned, because neither did the empirical work in §3.
- The task's own working hypothesis about uniformization (classic
  Poisson-mixture, exact for any λ, only tail truncation approximate) is
  correct as a description of what `ptd_graph_pdf_with_gradient` *attempts*,
  but is **not** a correct description of the actual primal `g.pdf`/
  `stop_probability` (§1), which is a different, Euler-power-iteration
  algorithm. This distinction is the basis for refuting the daisy-chain
  dependency claim (§5.2).

## 8. Cross-batch dependency notes (task item 6)

- **(a) rewards support for moments** — orthogonal. Different quantity
  (closed-form elimination solve vs. time-indexed uniformization/DTMC
  power-iteration). No dependency either direction.
- **(b) formula-mode + skeleton refactor** — no code dependency, but a soft
  sequencing argument: if the reverse-tape skeleton-duplication refactor
  (`ptd_moments_grad_theta`/`_dph`/`_log`, flagged in CLAUDE.md, deliberately
  not done unilaterally) happens, doing it **before** this batch avoids
  adding yet another near-duplicate stage-0/stage-1/stage-2 skeleton copy —
  though this batch's actual math (uniformization/DTMC power-iteration) is
  different enough from the elimination-replay skeleton that it may not
  even be a candidate for that shared core. Worth a two-line note when (b)
  is scoped, not a blocker.
- **(c) `Graph.svgd()` plumbing + joint-index baked mode** — defines the
  `exact_grad`-kwarg dispatch shape across `Graph.svgd()`'s leaves
  (per the non-memory reachability atlas cited in the loose-ends memory,
  daisy-chain is one of five leaves, currently with no exact-gradient
  option to select at all). If (c) lands first, this batch should conform
  to whatever uniform interface it establishes rather than inventing its
  own — soft coordination, not a hard blocker in either order.
- **(d) hierarchical/SCC** — orthogonal; the JSP graph daisy-chain builds is
  a monolithic elimination target, not SCC-decomposed, in every code path
  read for this investigation. No dependency.
- **(e) `weight_mode='callback'` + MPFR conditioning-floor adjoint** —
  orthogonal; daisy-chain already explicitly rejects `weight_mode='log'`
  today (`__init__.py:10173-10180`) and would need the same explicit
  rejection (not silent linearization) for `callback` if/when that mode is
  added to the daisy-chain forward path at all — a future scope question
  independent of this batch's gradient work.
- **The specific claimed dependency — "daisy-chain gated on the PDF batch"
  — is refuted** (§5.2): zero code or formula sharing; the only transferable
  content is the forward-mode-tangent architectural pattern, which either
  team could independently rediscover.

## 9. Recommended sequence position

**Last**, and it should not be scoped or committed to as a sixth
same-sized batch alongside (a)-(e). Concretely:

1. If any of it is attempted this cycle, do **§5.6 only** (exact gradient
   for the daisy-chain final epoch, reusing the shipped sojourn adjoint) —
   small, reuses verified machinery, de-riskable in the established
   batch-with-test-gates pattern.
2. Treat the standalone `ptd_graph_pdf_with_gradient` revival and the
   daisy-chain intermediate-epoch backprop-through-time work as a
   **separate follow-up initiative**, requiring its own multi-week
   de-risking pass (branch experiments per `feedback_derisk_and_reevaluate`)
   before any batch plan is written — specifically: (i) decide and validate
   a θ-independent granularity-pinning policy, (ii) profile
   Jacobian-seeded forward-mode vs. checkpointed reverse-mode on a
   realistic model to pick an approach, (iii) decide whether
   `ptd_graph_pdf_with_gradient` is repaired or abandoned in favor of a
   from-scratch implementation (this report's evidence favors abandoning
   it — do not build on it as-is).
3. Do not schedule this ahead of (a)-(e); none of them depend on it, and it
   is the only one of the six whose feasibility work turned up the
   underlying primitive being outright broken rather than merely
   incomplete.
