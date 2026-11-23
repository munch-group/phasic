# Scale-free continuous likelihood and gradients: plan

**Status: v3, 2026-11-23. Replaces v2 (commit `0e08065a`: a capped
relative FD step plus a two-part guard), which the user judged ad hoc —
correctly: v2 patched three symptoms of one cause. v3 removes the cause.
Evidence: `experiments/fd_step_guard/` (v1/v2 experiments E1-E12 and the
three v2 reviews) and `experiments/fd_step_guard/v3/` (X1, X3, X4: a Python
prototype built on the existing native step iterator). NO shipped code
changed. Decisions in section 7 need sign-off. v3 has NOT yet been through
adversarial review.**

---

## 1. Diagnosis: one cause, four symptoms

The continuous forward (`Graph::pdf`/`cdf`/`stop_probability`/
`accumulated_visiting_time`, `api/cpp/phasiccpp.h:1569-1740`,
`src/c/phasic.c:12996-13140`) is not uniformization. Read from source: it
steps the discrete chain P = I + Q/G and reads the result at step index
G x t, i.e. it computes pi (I + Q/G)^(G t) — an **Euler discretization**
with step 1/G, where G ("granularity") = max(1000, int(2 x max_rate)) with
max_rate floored at 512. That scheme has an absolute time scale (1/1024)
and an integer that depends on theta built in. Everything below follows:

1. **Bias.** First-order error in 1/G: 1.16e-3 at G = 1024, halving as G
   doubles (X1, vs the hypoexponential closed form). This is the
   documented "likelihood accuracy floor" (CLAUDE.md), which biases
   estimates independent of sample size.
2. **Unit dependence.** The SAME distribution in different time units
   (X1, l1 x t = 0.7 throughout):

   | l1 | shipped error | shipped steps |
   |---|---|---|
   | 1e-9 | refuses (7e11 steps > 1e9 cap) | — |
   | 1e-6 | not run: 7e8 steps, ~11 GB | 7e8 |
   | 1e-3 | 1.2e-6 | 7e5 |
   | 1 | 1.2e-3 | 717 |
   | 1e2 | **11%** | 7 |
   | 1e4, 1e6 | **21%** | 4 |

   Small rates are slow and memory-hungry (the 1000 floor); large rates
   are wrong (two steps per holding time); the "slow band" is the large-
   rate end of the same curve.
3. **Jumps in theta.** Above max rate 512, int(2 x max_rate) makes the
   likelihood a step function of theta. Two-island model at rate scale
   200 (times scaled inversely — the same distribution): log-likelihood
   off by 2.8e-2 with jumps of 4e-2 (X3; E11 measured 12 jumps per ±0.1%).
4. **No usable finite-difference step.** A gradient needs a step that is
   small relative to theta (so probes stay positive and power laws are
   resolved: E1-E8, the user-reported `NEGATIVE transition rate` crash,
   silent errors up to 4.6e15x) but a relative step straddles the jumps
   (25-34% error, E11; garbage, X3). v2's threshold rule was a compromise
   between these two; no compromise is needed once the jumps are gone.

## 2. The principled fix

### 2.1 Forward: true uniformization (scale-free, exact, smooth)

pi e^(Q t) = sum_k Poisson(k; G t) x pi P^k, P = I + Q/G, exact for ANY
G >= max exit rate. The iterates pi P^k are what the existing step function
already computes; only the read-out changes (Poisson-weighted sum instead
of "index G t"):

- cdf(t) = sum_k w_k A_k, pdf(t) = G sum_k w_k (A_{k+1} - A_k), A_k =
  absorbed mass after k steps; stop_probability(t) = sum_k w_k (pi P^k);
  accumulated_visiting_time(t) = (1/G) sum_k (1 - PoissonCDF(k; G t)) pi P^k.
- G is real-valued and proportional to the max exit rate (no 512 floor, no
  1000 floor, no integer cast; the starting vertex's initial-distribution
  edges are not rates). Steps = Lambda + O(sqrt(Lambda)), Lambda = max_rate
  x t: the expected number of jumps — dimensionless, the only quantity cost
  depends on.
- No per-step storage: for several times, one pass with one accumulator per
  requested time. The 16 B/step memory growth (1 GB at 4e7 steps, E12) and
  the cache-order dependence found in review disappear.

Measured with the prototype (same native iterates, Poisson read-out in
Python):

- vs closed form: 2e-13 at G = 1024, and 15 identical digits for
  G = 4, 7, 50 (X1): independent of G, as theory says.
- The unit sweep above: 1e-16 to 4e-16 relative error and 58 steps at
  EVERY scale from 1e-9 to 1e6 (X1).
- Two-island model at scales 1 / 200 / 2000: identical results, no jumps
  (roughness 1.7e-10 = its curvature), 120 steps each (shipped: 2,048 /
  32 / 32 steps, the latter two wrong by 2.8e-2) (X3).

### 2.2 Gradient: one rule, a central step in log(theta)

Rates are positive scale parameters; their natural coordinate is the log.
For theta_i > 0:

    dF/dlog(theta_i) ≈ (F(theta_i e^s) - F(theta_i e^-s)) / (2 s),   dF/dtheta_i = that / theta_i

One constant s for every parameter at every magnitude; probes are positive
by construction; no threshold, no clamp, no "keep the old code above X".
Measured on the new forward (X4, X3), absolute error of d log f / d log
theta against closed forms:

| case | s = 1e-3 | s = 1e-4 |
|---|---|---|
| Erlang-k, k = 3 / 10 / 30, theta = 1e-9 / 1 / 1e6 | 8.3e-8 | <= 8.7e-10 |
| same in the power-law regime (f ∝ theta^k) | <= 1.7e-11 | <= 3.0e-10 |
| hypoexponential at theta = [1, 1e-8], t = 1 and 1e4 | <= 3.1e-8 | <= 1.4e-7 |
| two-island at scale 1 / 200 / 2000 (vs Richardson reference) | 2.0e-7 | 1.9e-9 |

(v2's k-dependent error is gone: a power law is a straight line in log-log
coordinates.) Forward noise grows with Lambda (the prototype's Poisson
weights lose ~Lambda x 1e-16), which sets the useful range of s to about
1e-3..1e-4; recommended s = 1e-4 (D2), to be confirmed with the native
weight recursion in Batch 0.

The same rule applies unchanged at the elimination-based sites
(joint-index, moments, reward-visit, cdf_zero), whose forwards are already
smooth rational functions of theta (E2b: a relative step of 1e-3 there gave
7e-6 where today gives 4.6e15x / NaN).

Parameters that are not positive (theta_i <= 0, possible only with
`positive_params=False` or direct calls) have no intrinsic scale; they keep
today's absolute ±1e-7 step. That is a statement about parameter kind, not
a numerical threshold.

### 2.3 Cost: one number, known before stepping

Work = (Lambda + c sqrt(Lambda) + c') x edges, at the measured ~4 ns per
edge-step (E5, review: 3.8-4.3 ns up to 64k edges; x86 long double
unmeasured). It is intrinsic: a model with max_rate x time = 1e8 needs
~1e8 steps under any uniformization (the X4 case theta = [1, 1e-8], t = 1e8
is exactly this; today that case is refused outright). So a bound is still
needed for diverging particles, but it is now a single dimensionless
budget on Lambda x edges, with no memory component:

- per fit: SVGD/MCMC compute the initial particles' predicted work and
  allow K x their median (K = 100, D4) — relative, because an absolute
  cap cannot tell an expensive model from a diverging particle (v2 review:
  at production size it degenerates to "max rate <= 8,192").
- when exceeded: unchanged from v2's reviewed design — join the existing
  fail-soft on the pdf paths (MCMC rejects; SVGD's divergence handling
  applies), raise with the native cause on the epoch path (which has no
  fail-soft and today reports an anonymous "NaN PMF"), and report
  particles that end an SVGD fit in the penalty (measured: they never come
  back and silently distort theta_mean, E6).

### 2.4 What this makes unnecessary

The 2.5e-3 accuracy floor and its bias; the granularity floor and its
small-rate cost; the 1e9-step memory cap; the FD threshold/clamp logic; the
negative-probe crash and the silent wrong gradients; the mixed-scale FD
defect pins. Exact (tangent) gradients through uniformization become a
straightforward later step — the tangent recursion uses the same iterates —
but are not needed for this plan: FD on the smooth forward is accurate to
~1e-9.

## 3. What stays hard (honest limits)

- **Stiffness.** Cost ∝ max_rate x time. Models mixing very fast and very
  slow rates over long times stay expensive (as today, where they are
  refused or wrong). Not solved here; bounded and reported.
- **Every continuous likelihood value changes**: by ~1e-3 relative at
  ordinary scales (toward the truth), by percent at high rates. Tests with
  stored continuous values or bit-identity will move; fitted results shift
  by about the old bias (CLAUDE.md: 2.5e-3).
- **Discrete models are untouched** (native DPH, `discretize()`,
  `joint_prob_graph(discrete=True)`): no uniformization there.
- **Short times cost slightly more steps** (a ~30-60 step minimum where
  the shipped code took G x t < 30 — and was wrong).
- **Upper-bounded parameters** (BetaPrior's sigmoid, formula weights like
  1 - t): a log-step protects the lower bound only; a probe can cross 1
  within s of it. Documented, as today.
- **Unverified so far:** native speed (the prototype's per-step loop is
  Python), the epoch chain end to end, accumulated_visiting_time, Poisson-
  weight accuracy at Lambda >= 1e6, production-size graphs. Batch 0.

## 4. Batches (each: implement on a branch, test, memory-capped gate, report)

### Batch 0 — de-risk (no shipped code)

1. Prototype `stop_probability` and `accumulated_visiting_time` read-outs
   and the full epoch chain; compare with the n = 2 two-epoch closed forms
   (`test_epoch_normalization.p_n2_two_epoch`) and with the exact
   final-epoch gradient.
2. Poisson weights by a stable recursion (Fox-Glynn style) vs lgamma:
   forward noise as a function of Lambda up to 1e7; confirm s.
3. Log-step FD through the real `jax.grad` paths (reuse the v2 prototype's
   switch): every site in v2's inventory (15 + 3 preconditioner sites),
   linear / formula / log weight modes, including the difference form
   (linear in p vs p x log(p+/p-), which is exact for power laws).
4. Native prototype timing on the 45,704-vertex joint graph and a
   production-size model: steps and seconds, shipped vs new.
5. Impact census: run the full gate with the new forward substituted;
   list every test that moves and why (stored Euler values, bias pins,
   bit-identity).

### Batch 1 — true uniformization in native code

- New read-outs in C/C++ next to the existing ones, selected by
  `phasic.configure(continuous_method='uniformization' | 'euler')`
  (default per D1); real-valued uniformization rate; no per-step vectors.
  An explicit `granularity=` keeps its meaning under 'euler' and is an
  explicit uniformization rate (>= max rate, else error) under the new
  method.
- Tests: closed forms to 1e-10 (exponential, Erlang, hypoexponential,
  n = 2 coalescent epochs); unit invariance (the X1 sweep, 1e-9..1e6);
  smoothness in theta at high rates (the X3 scan); agreement of pybind,
  FFI and epoch paths; cost = predicted steps.

### Batch 2 — the log-step gradient rule

- One helper (leaf module, per v2 review) used at every theta-space FD
  site; log-step for theta_i > 0, today's absolute step otherwise. On
  uniformization-based sites it is active only with
  `continuous_method='uniformization'` (on the Euler forward a relative
  step is harmful above rate 512 — X3); elimination-based sites always.
- Tests: error vs closed forms / exact gradients across 1e-9..1e6 with one
  tolerance; the user-reported epoch crash; formula and log weight modes;
  flip the mixed-scale pins; fix the `log(pmf + 1e-10)` regulariser artefact
  in `test_likelihood_correctness.py` (v2 review) and the CLAUDE.md
  paragraphs built on the old behaviour.

### Batch 3 — the work budget

- Predicted work check before stepping; fit-relative budget installed by
  SVGD/MCMC; fail-soft / epoch error plumbing (both epoch FFIs, the
  existing `BatchError`); the missing try/catch in the JIT `pmf_from_cpp`
  wrapper; SVGD report of penalised particles.

### Later, optional — exact tangents through uniformization (D7)

## 5. Why not smaller alternatives

- **v2 (capped relative step + guards):** works around the jumps instead
  of removing them; leaves the bias, the unit dependence and a k-dependent
  gradient error; adds thresholds and constants.
- **Freeze granularity across an FD pair:** removes the jumps for FD only;
  the likelihood itself stays a step function (MCMC acceptance, model
  comparison) and stays biased; needs per-particle granularity plumbing
  through static FFI attributes.
- **Round granularity to powers of two:** fewer, larger jumps.
- **Exact gradients first:** the derivative of an Euler-biased, jumpy
  forward is still the derivative of the wrong function.

## 6. Risks

- Scope: this changes the core continuous algorithm, not a gradient
  helper. Mitigation: additive implementation behind a switch, the old
  method kept selectable, default flipped only after the Batch 0 census
  and a full gate (D1).
- The epoch path's intermediate handoff and the final sojourn read must
  agree with the new intermediate read-out (Batch 0 step 1).
- Poisson weights at very large Lambda need a careful recursion
  (underflow, tail truncation) — standard, but must be tested to 1e7.
- Hidden consumers of per-step internals (`_pdf`/`_cdf` vectors, the
  context's `time`) — census in Batch 0.

## 7. Decisions

- **D1 — adopt true uniformization as the continuous forward.**
  (a) implement behind a switch, make it the default after the Batch 0
  census and a full gate, keep 'euler' selectable [recommended];
  (b) opt-in only; (c) no — fall back to v2.
- **D2 — gradient rule.** Central log-step, s = 1e-4, at every theta-space
  FD site for positive parameters; absolute step kept for non-positive
  ones [recommended] / s = 1e-3 / other.
- **D3 — tests and docs.** Flip the mixed-scale pins, fix the regulariser
  artefact, rewrite the CLAUDE.md "likelihood accuracy floor" and FD-step
  paragraphs once the default flips [recommended].
- **D4 — work budget.** Fit-relative, K = 100 x the initial particles'
  median predicted work [recommended] / none.
- **D5 — when the budget is exceeded.** Existing fail-soft on pdf paths,
  raise with the cause on the epoch path [recommended] / raise everywhere.
- **D6 — particles ending an SVGD fit in the penalty.** Warning naming
  them + `summary()` flag [recommended] / raise / leave silent.
- **D7 — exact tangents through uniformization** as a later plan: yes / no.
- **D8 — adversarial review of v3 by three independent agents before
  Batch 0** [recommended] / skip.

## 8. Findings carried over from v2 (all still valid, see `git show 0e08065a`)

FD-site inventory (15 theta-space sites + 3 preconditioner sites that are
theta-space when positive_params=False); today's behaviour per path and
weight mode; SVGD zeroing non-finite gradients; the fail-soft stuck-particle
measurement; both epoch FFIs swallowing native errors into NaN; the
`pmf_from_cpp` wrapper without try/catch; `moments_from_graph`'s JIT path
not buildable from the pixi install; the circular-import constraint on a
shared helper.

## 9. Copy-paste prompt for a fresh session

> Read `/Users/kmt/phasic/fd-step-and-slow-band-guard-plan.md` (v3) and
> CLAUDE.md. My decisions on D1-D8 are: <fill in>. Start with Batch 0 on a
> branch (no shipped code), report, then one batch at a time: implement,
> add the tests, run the gates through `.git/claude-scratch/runcap.py`, and
> report before the next batch. Do not change any notebooks.

State at writing: master `0e08065a`; throwaway prototype branch
`exp/fd-step-prototype` in worktree `/Users/kmt/phasic-epochprior`
(v2's rule, switchable by `PHASIC_FD_RULE` / `PHASIC_FD_REL`); the v3
prototype is pure Python in `experiments/fd_step_guard/v3/unif.py`.
