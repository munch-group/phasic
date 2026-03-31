# BFFG Inference in Phasic

## Overview

Backward Filtering Forward Guiding (BFFG) enables inference under time-inhomogeneous models using phasic's time-homogeneous phase-type machinery. The core idea: sample paths from a homogeneous proposal that phasic can handle, then reweight those paths to account for the inhomogeneous target.

This document describes the components involved and how they compose.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MCMC loop                            │
│  For each proposed θ (e.g., epoch population sizes):        │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Per locus:                                           │  │
│  │                                                       │  │
│  │  1. Sample M paths from proposal graph                │  │
│  │         ↓ graph.sample_path()                         │  │
│  │                                                       │  │
│  │  2. For each path:                                    │  │
│  │     a. Compute features    → path_to_rewards()        │  │
│  │     b. Observation lik     → model-specific function   │  │
│  │     c. Importance weight   → rate ratio along path    │  │
│  │         ↓                                             │  │
│  │         path_exit_rates()                             │  │
│  │         importance_log_weight_from_rates()            │  │
│  │                                                       │  │
│  │  3. Combine via log-sum-exp                           │  │
│  │         ↓ importance_weighted_log_likelihood()        │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  Sum over loci → total log-likelihood → MH accept/reject    │
└─────────────────────────────────────────────────────────────┘
```

## Components

### General (phasic library)

These are model-agnostic and live in `src/phasic/`.

#### `Graph.sample_path()` — Path sampling

**File**: C layer (`src/c/phasic.c`), exposed via `src/phasic/__init__.py`

Simulates the Markov chain from the starting vertex to absorption, recording every vertex visited and the cumulative time at each entry. Returns a dict with `vertex_indices` and `entry_times`.

This is the forward simulation step. The graph must be parameterized with the proposal rates before sampling.

#### `path_to_rewards(graph, path, rewards)` — Feature extraction from paths

**File**: `src/phasic/bffg.py`

Given a sampled path and a reward matrix, computes the reward-weighted sojourn time at each vertex, summed over the path. This is the per-path decomposition of what `graph.sample(rewards=...)` computes internally.

- With a 1D reward vector: returns a scalar (total weighted time)
- With a 2D reward matrix `(n_features, n_vertices)`: returns a vector of per-feature totals

In the coalescent application, the reward matrix is `graph.states().T`, so each feature corresponds to a descendant class (singletons, doubletons, etc.), and the output is the branch length per class for that specific genealogy.

#### `path_exit_rates(graph, path)` — Exit rates along a path

**File**: `src/phasic/bffg.py`

Extracts the total exit rate (sum of outgoing edge weights) and sojourn time at each transient vertex along the path. These are the proposal rates — needed to compute the importance weight.

#### `importance_log_weight_from_rates(proposal_rates, target_rates, sojourn_times)` — Girsanov weight

**File**: `src/phasic/bffg.py`

Computes the log importance weight for reweighting a path from the proposal to the target process. For a continuous-time Markov chain, the density of a path factorizes as a product over vertices: `rate * exp(-rate * sojourn_time)`. The importance weight is the ratio of target to proposal densities:

```
log w = Σ_k [ log(r_target_k / r_proposal_k) - (r_target_k - r_proposal_k) * s_k ]
```

This is the Girsanov-type change of measure for continuous-time chains. It requires only the exit rates under both models at the visited vertices — not the full transition matrix.

#### `importance_weighted_log_likelihood(log_likelihoods, log_weights)` — Pseudo-marginal estimator

**File**: `src/phasic/bffg.py`

Combines M importance-weighted samples into a single likelihood estimate using the log-sum-exp trick:

```
log P̂(data | target) = logsumexp(log_lik_m + log_w_m) - log(M)
```

This is the standard pseudo-marginal likelihood estimator. When used inside MCMC, it gives a pseudo-marginal Markov chain that targets the correct posterior despite the noisy likelihood estimate (Andrieu and Roberts, 2009).

#### `MCMC` class — Metropolis-Hastings sampler

**File**: `src/phasic/mcmc.py`

The outer loop. Accepts either:
- `model` + `observed_data`: standard phasic interface where the model returns per-observation likelihoods
- `log_prob_fn`: direct log-likelihood function (used for BFFG, where the likelihood involves stochastic sampling and can't be JIT-compiled)

Supports multiple chains, burn-in, thinning, parameter transformations (`positive_params`), fixed parameters, and convergence diagnostics (R-hat, ESS).

### Model-specific (notebook)

These are specific to the coalescent application and defined in the notebook.

#### `extract_waiting_times(graph, path)` — Coalescence event detection

Reads the state vector at each vertex along a path, computes the total lineage count (`sum(state)`), and detects coalescence events wherever the count drops. Returns inter-coalescence waiting times.

#### `compute_target_exit_rates(graph, path, epoch_boundaries, epoch_sizes)` — Inhomogeneous rate scaling

For the coalescent, all rates scale as `1/N(t)`. Given a path sampled under proposal rate `1/N_0`, the target rates are `proposal_rate * (N_0 / N(t))` where `N(t)` is determined by which epoch the vertex's entry time falls in.

#### `poisson_sfs_log_likelihood(observed_counts, branch_lengths, mutation_rate)` — SFS observation model

The Poisson mutation model: mutations on branches subtending `i` descendants follow `Poisson(θ/2 * L_i)` where `L_i` is the branch length for descendant class `i`. This connects the latent genealogy (branch lengths from `path_to_rewards`) to the observed data (SFS counts).

## Data flow for the coalescent example

```
Observed: SFS counts per locus (n_loci × n_classes matrix)
Parameters: epoch population sizes N_1, N_2, ...

graph.update_weights([1/N_0])          Set proposal rate
        │
        ▼
graph.sample_path()                    Sample genealogy from proposal
        │
        ├──► path_to_rewards(graph, path, states.T)
        │           │
        │           ▼
        │    Branch lengths L_1, ..., L_{n-1}
        │           │
        │           ▼
        │    poisson_sfs_log_likelihood(SFS_obs, L, θ)
        │           │
        │           ▼
        │    log P(SFS | genealogy, θ)          ... per-path SFS likelihood
        │
        ├──► path_exit_rates(graph, path)
        │           │
        │           ▼
        │    Proposal rates r_proposal at each vertex
        │           │
        │    compute_target_exit_rates(...)
        │           │
        │           ▼
        │    Target rates r_target = r_proposal * N_0/N(t)
        │           │
        │           ▼
        │    importance_log_weight_from_rates(r_prop, r_target, sojourns)
        │           │
        │           ▼
        │    log w                               ... per-path importance weight
        │
        ▼
importance_weighted_log_likelihood([log_liks], [log_weights])
        │
        ▼
log P̂(SFS | N_1, N_2)                  ... estimated target likelihood

        ▼
MCMC accept/reject
```

## Key properties

**Correctness**: The importance-weighted estimator is unbiased for the target likelihood. Pseudo-marginal MCMC with an unbiased likelihood estimator targets the correct posterior.

**Variance**: The quality of the estimate depends on the overlap between proposal and target. A moment-matched proposal (homogeneous graph whose moments approximate the inhomogeneous target) reduces variance and requires fewer paths per evaluation.

**Computational cost**: Each MCMC iteration requires `n_loci × n_paths` path simulations. Path simulation is fast (C code), but the total cost scales linearly with both. The proposal quality directly controls how many paths are needed.

**Generality**: The library components (`path_to_rewards`, `path_exit_rates`, `importance_log_weight_from_rates`, `importance_weighted_log_likelihood`) work with any phase-type graph, not just coalescents. The model-specific parts (rate scaling formula, observation likelihood) are defined by the user.
