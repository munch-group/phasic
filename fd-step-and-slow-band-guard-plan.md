# Scale-free continuous likelihood and gradients: plan

**Status: v4, 2026-11-23. v3 (commit `150ec8ca`) went through adversarial
review by three independent agents (maths of the forward; gradient rule;
architecture/rollout/budget). The core idea survived; three blockers and
ten majors changed the design and the order. Every finding is dispositioned
in section 9. NO shipped code changed. Decisions in section 7 need
sign-off.**

Evidence: `experiments/fd_step_guard/` (E1-E12, v2 reviews) and
`experiments/fd_step_guard/v3/` (prototype `unif.py`, X1/X3/X4, and the
three v3 reviews' scripts and logs in `review_math/`, `review_grad/`,
`review_arch/`).

---

## 1. Diagnosis (confirmed by review against source and matrix exponentials)

The continuous forward (`Graph::pdf`/`cdf`/`stop_probability`/
`accumulated_visiting_time`) is an **Euler discretization**:
pi (I + Q/G)^floor(G t), G = max(1000, int(2 x max(512, max_rate)))
(`phasic.c:12969-13041`, `phasiccpp.h:1610-1622`). It has an absolute time
scale and a theta-dependent integer built in. Consequences, all measured:

- **Bias**, first order in 1/G (1.2e-3 at ordinary scales; the documented
  "likelihood accuracy floor"); pdf/cdf are also piecewise constant in t,
  and 100% wrong when G x t < 1 for any model needing two jumps.
- **Unit dependence**: the same distribution costs 7e8 steps (11 GB) at
  rate 1e-6, is refused at 1e-9, and is 11-21% wrong at rates >= 1e2 with
  correspondingly short times.
- **Jumps in theta** above max rate 512 (two-island at rate scale 200:
  log-likelihood off by 2.8e-2, jumps of 4e-2).
- **No finite-difference step works everywhere**: small relative steps
  straddle the jumps (25-34% / garbage), an absolute 1e-7 step goes
  negative below 1e-7 (the reported `NEGATIVE transition rate` crash;
  silent errors up to 4.6e15x on the default joint-graph path and 3.4e3x on
  the main pdf path).

## 2. Principles

1. **The forward computes the model, and its cost depends on one
   dimensionless number.** True uniformization:
   pi e^(Qt) = sum_k Poisson(k; G t) pi P^k, exact for any G >= max rate;
   cost ∝ Lambda = max_rate x t.
2. **A derivative is taken in the coordinate of the parameter's declared
   domain**: log for (0, inf), logit for (0, 1), identity for unconstrained.
   The domain is declared by whoever knows it; undeclared means
   unconstrained, which is exactly today's behaviour.
3. **Cost is known before it is paid and bounded explicitly**: work =
   steps x edges is available when a context is created; exceeding a
   visible, configurable ceiling raises.
4. **Nothing silently changes what is being computed**: no hidden
   per-fit state altering a model's value; method and step rule are bound
   into a model when it is built.

## 3. Design

### 3.1 Forward: true uniformization (review-corrected)

Read-outs (derived independently by the reviewer and checked against
`scipy.linalg.expm` to 1e-13..1e-16 on acyclic, cyclic, stiff, defective
and multi-absorbing models, for G = 1x, 2x, 7x max rate):

- pdf(t) = G sum_k w_k m_{k+1}, with m_k the mass absorbed IN step k (the
  context's existing per-step pmf) — NOT differences of the cumulative
  mass: v3's prototype differenced and lost the right tail (26% error at a
  density of 6e-18, 100% at 1e-26; the per-step form gives 1e-14).
- cdf(t) = sum_k w_k A_k; stop_probability(t) = sum_k w_k (pi P^k);
  accumulated_visiting_time(t) = (1/G) sum_k (1 - PoissonCDF(k; G t)) pi P^k.
- **Weights** by a recursion from the mode, normalised by their sum
  (forward error <= 3.6e-15 for Lambda up to 1e7; the lgamma form loses
  ~1e-9 there).
- **Truncation relative to the result, never by absolute Poisson mass, and
  no left cut**: stop on the right only when the remaining tail bound is
  <= eps x the accumulated sum and not before the first non-zero absorbed
  mass. (Absolute truncation returned exact zeros for deep chains — e.g.
  Erlang-100 at t = 5, true density 1.1e-89 — and a Fox-Glynn left cut gave
  log-densities of -568 for a true -289.)
- **G real-valued, slightly above the max exit rate (x1.02)**, from a new
  context creator: the existing one counts the starting vertex's
  initial-distribution edges as rates, takes an integer granularity and
  seeds the max at 512 (`phasic.c:12999-13054`).
- **Storage and call pattern.** Every native consumer asks for one time
  per call (about 20 call sites) and relies on today's cached per-step
  vectors. Keep one cached scalar sequence m_k per graph (8 B per step,
  and steps are now ~Lambda instead of 1024 x t) and compute the weighted
  sum per requested time; vector read-outs restart from the last
  accumulated vector (semigroup property) so incremental callers
  (`auto_t`, `compute_accumulated_visits_converged`, `add_epoch`) stay
  linear. v3's "no per-step storage" was wrong.
- **The epoch path's intermediate graph has a constant rate-1.0 trap loop
  at every outcome vertex** (`__init__.py:11244`, deliberately theta-
  independent). It forces G >= 1 and so an absolute time scale: at
  theta = (1e-4, 1e-4), T = 1e4 the model needs 97 steps, the loop makes it
  11,030 (still ~1000x fewer than today's 1.0e7). Must be made non-binding
  under the new method (candidates: outcome vertices retain mass with no
  auxiliary edge; or a loop rate tied to the uniformization rate) —
  Batch 2a decides by experiment, with a unit-invariance test on the epoch
  path.
- **The stepping API stays Euler**: `Graph.distribution_context().step()`
  with `update_weights` between steps is the documented time-inhomogeneous
  workflow and cannot be a per-time read-out. Kept, documented as the
  Euler stepper, with tests pinned independent of the method.

Cost, honestly (predicted steps new/shipped at G = 1.02 x max rate R):

| R | t = 1e-3 | 1e-2 | 0.1 | 1 | 100 |
|---|---|---|---|---|---|
| 10 | 20x (41 steps) | 3.7x (41) | 0.45x | 0.07x | 0.01x |
| 512 | 21x (43) | 5.6x (61) | 1.5x (156) | 0.76x | 0.53x |
| 4950 | 6.0x (60) | 1.5x (153) | 0.77x | 0.58x | 0.52x |

Fewer steps for ordinary settings, at most ~2x fewer above rate 512, and
4-20x MORE at very short times (40-60 steps in absolute terms; the tail
constants can be tightened). Vector read-outs add ~10-30% per step.
Per-step cost measured 4.1-6.0 ns per edge on arm64 up to 433k edges;
x86-64 (80-bit long double, a calloc+memcpy of n long doubles per step,
denormals) is unmeasured and is where production runs.

### 3.2 Gradients: the step follows the declared domain

- Each model builder takes per-parameter domains: `(0, inf)`, `(0, 1)` or
  unconstrained. `Graph.svgd`/`mcmc` derive them from the transforms they
  already build (softplus → (0, inf), sigmoid → (0, 1)) and pass them
  down. Direct callers may declare them; **undeclared = unconstrained =
  exactly today's code** (bit-identical, including the clamp sites).
- Central difference in the domain's coordinate, linear in the outputs,
  divided by the actual probe difference:
  log step theta x e^(±s) for (0, inf); logit step for (0, 1) (never
  crosses either bound); ±1e-7 for unconstrained.
- The review showed why sign is the wrong criterion: with a log-step on an
  upper-bounded parameter the unsafe band widens from 1e-7 to s (raises at
  1 - t1 = 5e-5); on an additive parameter (rate = t0 + t1, t1 → 0) the
  theta-space gradient is destroyed (exactly 0 at 1e-13) — harmless only
  when the consumer works in the log coordinate, which is what a declared
  (0, inf) domain means.
- **Error model (not a single number):** in the domain coordinate,
  error ≈ s^2 |F'''|/6 + eps_forward |F| / s. Measured: elimination
  forwards have noise 2.6e-16..6.8e-16 at 168..7,959 vertices and the
  joint-index gradient through the real `jax.grad` is within 6.4e-7 at
  s = 1e-4 (today: 1e2..6e4 at mu = 1e-8); power laws cost (k s)^2/6
  (1.7e-5 at k = 100). A component whose effect is dwarfed by another
  parameter's contribution to the same output is NOT made accurate in
  relative terms by any step (dE[T]/dtheta_0 at [1, 1e-11]: 8.2e-3 at
  s = 1e-4, 5.5e-4 at 1e-3, 100% today); the default exact moment gradient
  covers the moments. s is fixed per forward family in Batch 0
  (starting point 1e-4).
- The p x log(p+/p-) form is dropped (NaN at legitimately-zero outputs,
  not valid for moments/sojourns, needs an extra forward on the epoch
  path, buys nothing at these s).
- **While a model runs on the Euler forward with automatic granularity**,
  a declared-positive parameter uses the capped member of the same family,
  h = min(1e-7, s x theta): it is never larger than today's step, so it
  cannot span more jumps than today, and it removes the negative probe.
  (This is v2's rule, now a transitional case rather than the design.)
  Models on elimination-only forwards, discrete models, and models with an
  explicit granularity have no jumps and use the full rule at once.
- Method and step rule are bound into the model at construction; a later
  `configure()` cannot split a forward from its backward.

### 3.3 Cost bound

- Work = steps x edges is computed where the context is created (the scan
  over all rates already happens there) — measured natively, not predicted
  host-side (the review showed host-side prediction is impossible for
  reward-transformed graphs, formula/callback weights and opaque MCMC
  models).
- An absolute, user-visible ceiling per native call, configurable, default
  generous (1e11 edge-steps ≈ 7-10 minutes on this machine; D4); exceeding
  it raises a specific error stating max rate, time, steps, edges and how
  to raise the ceiling. It ships in the same batch as the new forward, so
  there is never a window without a bound (the new method removes the
  1e9-step cap that today triggers the fail-soft).
- v3's fit-relative budget (K x the initial particles' median) is
  withdrawn: work ∝ max rate, so with a prior that excludes the truth a
  correct posterior 100x above the prior median would be penalised, and a
  penalised particle measurably never returns — it would manufacture bias;
  it also made a model's value depend on hidden per-fit state and could
  not reach the JIT library copies or worker processes.
- An absolute ceiling is a backstop against hours-long evaluations, not a
  detector of "slow for this model". What a fit can honestly do about a
  particle that is merely expensive is report it (D5).

## 4. Batches (each: implement on a branch, test, memory-capped gate, report)

### Batch 1 — domain-aware gradient steps (no forward change; fixes the reported crash)

- `param_domains` on the model builders; svgd/mcmc pass them; a leaf helper
  module (the shared helper cannot live in `__init__.py`: circular import).
- All 15 theta-space FD sites + the 3 preconditioner sites (theta-space
  when no transform is set): full rule on jump-free forwards, capped step
  on Euler/automatic-granularity models; undeclared → today's code.
- Fixed slots are skipped at the four sites that today probe them
  (`pmf_from_graph` x2, `pmf_from_cpp`, `moments_from_graph`,
  reward-visit), where formula/log weight modes currently raise.
- Tests: undeclared domains bit-identical to master; the epoch fit from
  near-floor particles completes; accuracy against exact gradients /
  closed forms per forward family with tolerances from the error model;
  logit steps next to 1; formula and log weight modes; additive and offset
  parameters under each declaration. Known test edits (census in Batch 0
  of this batch): the three mixed-scale pins flip; three tolerance tests
  that replicate the 1e-7 formula or compare exact vs FD at rtol 1e-8
  (`test_exact_grad_daisy_final.py::test_false_path_is_plain_fd`,
  `test_exact_grad_rewards.py` [2], [3]); fix the `log(pmf + 1e-10)`
  regulariser artefact in `test_likelihood_correctness.py`.

### Batch 2 — true uniformization in native code, not yet the default

- 2a (Python, no shipped code): epoch chain and
  `accumulated_visiting_time` prototypes vs the n = 2 two-epoch closed
  forms; the trap-loop decision; right-tail closed-form tests on the log
  scale down to 1e-200.
- 2b: new C entry points (real-valued rate, own context creator, cached
  scalar sequence, relative truncation, recursive weights); a method enum
  read at context creation and included in every context-cache guard
  (today's guards key on granularity and weight version only); the method
  bound per model; the pybind replica of the epoch chain used by the exact
  final-epoch gradient follows the same method; the work ceiling; epoch
  FFIs report native causes via the existing `BatchError` (both variants);
  the try/catch missing in the JIT `pmf_from_cpp` wrapper.
- 2c: census and timing with the new method selected: every test that
  moves, classified; steps and seconds at 45k and production-size models,
  on arm64 here and on x86-64 Linux (hoisted scratch buffer, denormal
  handling). The suite has known pre-existing failures: the gate is "same
  ledger under 'euler'" plus "classified list under 'uniformization'".

### Batch 3 — make it the default (separate decision, D1)

- Version bump and changelog (fitted values shift by ~2.5e-3 at ordinary
  scales, percent above rate 512); Euler pins kept and parametrised with
  the method forced (107 test call sites pass an explicit granularity; the
  strict-xfail bias pins invert); docstrings that already say "exact via
  uniformization" become true; CLAUDE.md rewritten; a list of notebook
  cells that pass `granularity` for the user (not edited by me).

### Later, optional

Exact tangents through uniformization (same iterates); adaptive
uniformization for coalescent-like models, where the top rate C(n,2) is
active only at the start (about n graph steps instead of C(n,2) x t).

## 5. Honest limits

- Stiffness stays expensive (cost ∝ max_rate x time).
- Very short times cost more steps than today (table in 3.1).
- Dominated gradient components stay inaccurate in relative terms under
  any finite difference.
- The stepping API remains Euler; `graph.pdf(t)` and a context stepped to
  t will differ by the Euler error after Batch 3.
- Until Batch 3, continuous models keep the Euler forward's bias and
  jumps; Batch 1 only makes their gradients safe, not more accurate above
  rate 512.
- x86-64 performance is unmeasured.

## 6. Why this is not v2 again

v2 chose a step by threshold and added guards around a forward it left
alone. Here the step is determined by a declared property of the parameter
(its domain), the forward is replaced by the exact method, and the one
remaining threshold-shaped rule (the capped step) is confined to the old
forward and disappears with it. Batch 1 alone already removes the crash
and the silent errors without changing any undeclared caller.

## 7. Decisions

- **D1 — true uniformization.** (a) implement behind a per-model method
  (Batch 2), decide the default flip after the census and timing, with a
  version bump [recommended]; (b) opt-in only, never the default;
  (c) no forward change — stop after Batch 1.
- **D2 — domain declaration.** svgd/mcmc declare domains from their
  transforms; undeclared = unconstrained = today's behaviour
  [recommended] / assume (0, inf) for linear and log weight modes even
  when undeclared.
- **D3 — step s.** Fixed per forward family in Batch 1's de-risking,
  starting from 1e-4 [recommended] / a single value now.
- **D4 — work ceiling.** Absolute, configurable, default 1e11 edge-steps,
  raises [recommended] / other default / none.
- **D5 — expensive or penalised particles in a fit.** Count evaluations
  that hit the existing fail-soft or exceed a multiple of the batch's
  typical work, warn during the fit and flag in `summary()` [recommended]
  / raise / nothing.
- **D6 — tests and docs.** As listed in Batches 1 and 3 [recommended].
- **D7 — exact tangents / adaptive uniformization** as later plans: yes /
  no.

## 8. Carried over (still valid; `git show 0e08065a` for detail)

FD-site inventory and today's behaviour per path and weight mode; SVGD
zeroing non-finite gradient elements; the stuck fail-soft particle
measurement; both epoch FFIs turning native errors into NaN;
`moments_from_graph`'s JIT path not buildable from the pixi install.

## 9. Adversarial review of v3 — dispositions

Maths reviewer (no blocker; diagnosis and all four formulas confirmed):
1. pdf by differencing cumulative mass loses the tail → **accepted**, 3.1.
2. Absolute truncation / left cut wrong for likelihoods → **accepted**, 3.1.
3. One-time-per-call consumers; storage; short-time step counts →
   **accepted**, 3.1 (cached scalar sequence; table).
4. Step counts quoted at 2x Lambda (prototype's integer G) → **accepted**;
   counts restated from the reviewer's table.
5. lgamma weights lose ~1e-9 at Lambda 1e7; recursion fixes it →
   **accepted**, 3.1.
6. Piecewise-constant-in-t symptom → **accepted**, section 1.
7. Context creator constraints; keep G slightly above max rate; platform
   long double → **accepted**, 3.1.
8. Adaptive uniformization → **accepted** as a later option.

Gradient reviewer (no blocker):
1. Sign is the wrong criterion; upper-bounded, offset and additive
   parameters regress → **accepted**: domain declaration, logit step, 3.2.
2. Per-site coupling to the continuous method is unimplementable and left
   the crash unfixed under 'euler' → **accepted**: rule bound per model;
   capped step on Euler; Batch 1 first.
3. s = 1e-4 not uniformly accurate; dominated components; X4's t = 1e8
   row never ran → **accepted**: error model, per-family s, limits stated;
   the unmeasured case is no longer cited.
4. Drop p x log(p+/p-) → **accepted**.
5. Three tolerance tests beyond the pins → **accepted**, Batch 1.
6. Fixed slots still probed at four sites → **accepted**, Batch 1.
7. Host fallback already relative 1e-6 → **accepted**: listed as a site;
   its step changes only under a declared domain.
8. "Mixed-scale pins unnecessary" overstated → **accepted**, 3.2/5.

Architecture reviewer:
B1. Epoch trap loop breaks scale-freeness; X1's "every scale" was the
    same computation rescaled → **accepted** (verified at
    `__init__.py:11244`), 3.1 and Batch 2a.
B2. Windows without a bound and with the crash unfixed → **accepted**:
    Batch 1 first; ceiling ships with the forward.
B3. Fit-relative budget unsound → **accepted**: withdrawn, 3.3.
M1. Call pattern/storage → **accepted** (= maths 3).
M2. Batch 0 could not do native timing or a census → **accepted**:
    restructured into 2a/2b/2c.
M3. Stepping API is an unmentioned consumer → **accepted**, 3.1/5.
M4. Switch design (cache guards, env/atomic reach, pybind replica, int32
    granularity attribute, context creator) → **accepted**, Batch 2b.
M5. Pins invert rather than move; no versioning → **accepted**, Batch 3.
M6. x86 performance, per-step allocation, denormals → **accepted**,
    Batch 2c.
M7. Fail-soft is weaker once the forward is exact → **accepted**: the
    ceiling raises; D5 reports.
m1-m5. Stale docstrings/profile text; alternatives; risks; per-step mass;
    notebook cells passing granularity (uranium_decay cell 17, properties
    cell 63, test_hierarchical_caching cell 14, model-selection cell 26,
    time-inhomogeneous cells 6 and 11, discrete cell 34) → **accepted**.

## 10. Copy-paste prompt for a fresh session

> Read `/Users/kmt/phasic/fd-step-and-slow-band-guard-plan.md` (v4) and
> CLAUDE.md. My decisions on D1-D7 are: <fill in>. Start with Batch 1 on a
> branch: de-risk first (census of affected tests, s per forward family),
> then implement, add the tests, run the gates through
> `.git/claude-scratch/runcap.py`, and report before the next batch. Do
> not change any notebooks.

State at writing: master `150ec8ca` + this file; throwaway prototype
branch `exp/fd-step-prototype` in worktree `/Users/kmt/phasic-epochprior`
(`PHASIC_FD_RULE` = old / rel / default capped, `PHASIC_FD_REL` = c);
Python prototype of the forward in `experiments/fd_step_guard/v3/unif.py`
(note: it differences the cumulative mass and truncates absolutely — both
superseded by 3.1).
