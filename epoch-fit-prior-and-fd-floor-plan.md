# Epoch-fit default prior hang + negative-rate FD error: fix plan

**Status: v1, 2026-11-22. Investigation done, NO code changed. Decisions
D1-D5 (section 5) need sign-off before any batch starts.**

Two user-reported failures of `joint_prob_graph.svgd(..., epoch_starts=...)`
(the daisy-chain epoch path). They have different immediate causes, but
the same underlying weakness: nothing on the epoch path guards against
particles at extreme rates, and two prior-related defects put particles
there.

- **Issue A:** `svgd(epoch_starts=..., prior=None)` hangs.
- **Issue B:** `XlaRuntimeError: DaisyChainSojournFfiImpl: theta produced a
  NEGATIVE transition rate (most negative edge weight: -0.000000)`.

Evidence scripts and logs are in the session scratchpad
(`/private/tmp/claude-501/-Users-kmt-phasic/3666ba56-7958-4c29-a8b2-ec58b11551e2/scratchpad/hang/`:
`repro.py`, `repro2.py`, `repro3.py`, `run.sh`, `*.log`). It may have been
cleared; the numbers below are the record.

---

## 1. Issue A: hang without a prior on epoch fits

### Findings (measured, 4-sample coalescent joint graph, 10,000 observations at coal=5, mu=3, mu fixed)

| call | result |
|---|---|
| non-epoch, `prior=None` | done in 3.1 s (DataPrior builds in 0.1 s; theta_hat 5.09) |
| `epoch_starts=[0, 0.15]`, `prior=None` | killed at 150 s, stack inside the SVGD iterations (`svgd.py:4256`), NOT in prior building |
| same, `prior=None`, `theta_init` near the truth | done in 3.1 s |
| same, explicit `GaussPrior(ci=[-3.2e6, 3.2e6])` | hangs exactly like the default |

### Cause (read from source + measured)

1. `Graph._daisy_chain_svgd_model` (`src/phasic/__init__.py` ~5551-5590)
   builds the default epoch prior by calling `probability_matching(self,
   observed_indices, std_multiplier=sd, verbose=verbose)` **without the
   user's `fixed`** (and without exposure). The non-epoch path (`DataPrior`,
   `svgd.py` ~1512) does pass `fixed`, which is why it works.
2. On a continuous joint graph the normalised joint table depends only on the
   ratio of the rates (time rescaling), so with mu free the overall scale is
   unidentified. Measured: theta_hat = (1.61, 0.95) (right ratio 0.59, wrong
   scale; truth (5, 3)), and the delta-method std = (643,681 ; 379,143) from
   `pinv` of the rank-deficient Jacobian (reasoned; singular values not
   inspected).
3. The epoch builder turns that into `GaussPrior(ci=[m - 5*s, m + 5*s])` =
   about [-3.2e6, 3.2e6].
4. Particles are drawn from it: about half are negative and get clipped to
   1e-6 (see B2), the rest are at rates up to ~1e6. The intermediate epochs
   use uniformization, whose cost grows with rate x dt x granularity — the
   unguarded "slow band" already recorded in CLAUDE.md. Each evaluation
   becomes extremely slow: an apparent hang.

The prior is wrong, not only wide: it is centred at 1.61 when the truth is 5.

### Secondary observations (not causes of the hang)

- The epoch builder reads `m ± sd*s` as a 95% interval (prior std
  = sd*s/1.96), while `probability_matching`'s own priors use std = sd*s.
  The two default priors therefore differ by about 2x in width for the same
  data. To check and make consistent in Batch 2.
- `probability_matching` ignores exposure. Under exposure, the probability
  depends on alpha x theta[exposure_param_index], so even with `fixed`
  passed, a reference fit that ignores alpha is on the wrong scale.

## 2. Issue B: negative-rate error during an epoch fit

### Findings

Direct model evaluation (`repro3.py`; 2 epochs, exposure 1000, mu fixed at
1.2e-4) at epoch-2 coalescence rate c2:

| c2 | forward | gradient |
|---|---|---|
| 1e-6, 2e-7, 1e-7 | ok | ok |
| 5e-8 | ok | **the user's exact error, "most negative edge weight: -0.000000"** |
| 1e-9 (softplus floor) | ok | same error, "-0.000001" |

### Cause (read from source + measured)

- The daisy-chain model's gradient is central finite differences with an
  ABSOLUTE step `eps_local = 1e-7` (`src/phasic/__init__.py`, four copies of
  `tm = theta_flat.at[i].add(-eps_local)`: ~5048, ~5086, ~5389, ~5440). For
  any free component below 1e-7, the minus-probe is negative, and the C++
  FFI (`graph_builder_ffi.cpp:2226`) correctly refuses a negative rate. The
  particle itself was positive; the error text ("some theta component went
  below zero") misleads.
- The positivity transform floors at 1e-9 (`svgd.py` ~5598,
  `max(softplus(phi), 1e-9)`), i.e. 100x below the FD step, so this floor
  does not protect the FD probes.
- **How particles get there (B2):** `GaussPrior.sample` (`svgd.py` 833-839)
  draws in theta space and clips negative draws to 1e-6. With
  `GaussPrior(ci=[1, 20])` (mean 10.5, sd ~4.8) about 1.5% of draws are
  negative, so with 100 particles x 2 free parameters a few start at exactly
  1e-6 (phi = -13.8). There d(theta)/d(phi) ~ 1e-6, so these particles feel
  almost no likelihood or prior gradient. Adam normalises step sizes, so the
  kernel's repulsion can still move them by ~learning_rate per iteration in
  phi. Getting below 1e-7 needs phi < -16.1.

**Not reproduced end to end.** The user's exact notebook cell (cell 49,
same msprime seeds, 10,000 windows, 91.6% all-zero observations) finished
in 6/6 runs (seed unset + seeds 1-5, ~26 s each). The clipped particles
started at phi = -13.8 and stayed above it. Some drifted down again later
(epoch-2 parameter at about -0.01 per iteration after ~125 iterations, reaching
-8 to -11.6 by iteration 300), but none reached the -16.1 threshold. The
user's trajectory evidently differs (other session state, seed or
settings). The failure mechanism is confirmed exactly by the direct
evaluation above.

### How A and B relate

- Separate immediate causes: huge rates → slow uniformization (A); rates
  below the FD step → negative probe (B).
- Shared factor: the epoch path has no guard for extreme particles.
- They compound: issue A's default prior puts ~50% of particles at the
  clipped 1e-6, i.e. 10x above B's threshold. Fixing B alone leaves A; fixing
  A alone leaves B reachable with any wide Gaussian prior.

---

## 3. Fix plan, in batches with test gates

Each batch: de-risk on a branch first, then implement, then gate.
Inference correctness must be tracked by tests in the suite. Every run is
memory-capped (RLIMIT + RSS watchdog). Gates are "no new failures" against
the current baseline, never "full suite green" (pre-existing failures exist).

### Batch 1 — FD probes never leave the positive domain (fixes B; crash-level)

- **Change (per D1):** in the four daisy-chain FD backward copies, use a
  per-component step that keeps both probes positive. Recommended: relative
  step `h_i = 1e-7 * max(|theta_i|, theta_floor)` with central differences
  (both probes positive by construction because h_i < theta_i whenever
  theta_i > 0). This also addresses the pinned mixed-scale FD defect on this
  path (CLAUDE.md: absolute 1e-7 step).
- Keep the change identical across the four copies. Consider a single
  shared helper (one place to fix); that is a refactor of shipped code →
  part of D1's sign-off.
- **De-risk first:** (a) at theta ~ O(1) the new gradient matches the
  current one to FD noise (regression bound); (b) at c2 in {5e-8, 1e-9} it is
  finite and matches a high-precision reference (exact final-epoch gradient
  via `exact_final_grad=True` for final-epoch slots; mpmath / long-step
  Richardson reference for intermediate slots); (c) per-iteration cost is
  unchanged (same number of FFI calls).
- **Tests (new file, additive):** gradient at small components is finite and
  correct; gradient at normal values is unchanged within tolerance; an epoch
  fit with a particle initialised near the floor completes.
- **Optional, same batch:** reword the C++ message to mention that FD probes
  can cause it (only if D1 keeps any path where that is possible).

### Batch 2 — correct default epoch prior (fixes A)

- **2a.** Pass the user's `fixed` into the epoch path's
  `probability_matching` call, converted from per-epoch form (local index;
  scalar or per-epoch list) to the scalar form it takes (per D2).
- **2b.** Exposure in the reference fit (per D3).
- **2c. Identifiability guard** in `probability_matching` (benefits both
  paths): if the Jacobian at the solution is rank-deficient, or any free
  parameter's std exceeds a stated multiple of |theta_hat|, raise a
  `ValueError` naming the cause ("the overall rate scale is not identified;
  fix one rate or pass prior="). No silent fallback.
- **2d.** Make the epoch builder's width convention match
  `probability_matching`/`DataPrior` (or document why it differs).
- **Tests:** epoch `prior=None` with `fixed` gives a prior whose interval
  contains the truth and is finite; it completes in bounded time at small
  n_iterations; `prior=None` with no rate fixed raises the identifiability
  error (both epoch and non-epoch); a recovery test with the default prior,
  marked as such (the "priors must exclude the truth" rule applies to
  user-specified priors in recovery tests, not to this data-informed default;
  confirm with user).

### Batch 3 — no particles clipped onto the floor at initialisation (B2)

- Per D4. Recommended: under a positivity transform, sample Gaussian priors
  truncated to theta > 0 (rejection or inverse-CDF), instead of clipping to
  1e-6. Applies to `GaussPrior` and the other priors with the same clip
  (`svgd.py` ~1154).
- **Tests:** no initial particle equals the clip value; the mean/sd of the
  initial cloud matches the truncated Gaussian.
- This changes initial particles for every fit whose prior puts mass below
  zero → a behaviour change to SVGD, so it needs explicit sign-off (repo
  preference: do not change SVGD unasked).

### Batch 4 (optional) — slow-band guard

- The open CLAUDE.md item. A per-evaluation cost bound for uniformization
  in the daisy chain that raises (or penalises, per D5) instead of running
  for hours. Only after Batches 1-2, which remove the known routes into it.

---

## 4. Order and rationale

1 → 2 → 3 → (4). Batch 1 turns a hard crash into correct behaviour and is
small. Batch 2 fixes the hang and a wrong prior. Batch 3 removes the route
into Batch 1's regime but changes SVGD initialisation for everyone, so it is
last and separately approved.

## 5. Decisions needed

- **D1 — FD step.** (a) relative step `1e-7*|theta_i|` with a floor
  [recommended: fixes B and the mixed-scale defect on this path];
  (b) keep 1e-7 but shrink to theta_i/2 near zero; (c) forward difference
  when theta_i < 1e-7. Also: extract the four copies into one helper (yes/no).
- **D2 — per-epoch fixed values that differ across epochs**, for the
  time-homogeneous reference fit: use the first epoch's value
  [recommended] / use the mean / refuse `prior=None`.
- **D3 — exposure in the default epoch prior:** extend `probability_matching`
  additively with per-observation exposure [recommended] / refuse
  `prior=None` together with exposure.
- **D4 — initialisation:** truncated sampling [recommended] / refuse priors
  with more than x% mass below zero under positive_params / keep clipping
  and add a notice.
- **D5 — slow band:** do Batch 4 now / leave it as an open item.

## 6. Copy-paste prompt for a fresh session

> Read `/Users/kmt/phasic/epoch-fit-prior-and-fd-floor-plan.md` (and
> CLAUDE.md). My decisions on D1-D5 are: <fill in>. Implement Batch 1 on a
> branch: de-risk first as the plan says, then implement, add the tests,
> run the gates memory-capped, and report before merging. Do not start the
> next batch without my go-ahead.

State at writing: master at `d875149d` plus uncommitted work from the same
session (`event_rate_coefficient` rename + plot_* epoch true_theta fix,
with tests `test_event_rate_coefficient.py`, `test_plot_epoch_true_theta.py`).
