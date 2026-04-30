

If I use accumulated_occupancy(time) to get the expected total time spent at each transient vertex up to time, do I then still need to sample from the joint table to estimate amounts of different kinds of branch lengths?

when sampling is required to obtain a weight, sampling 
n_paths reduce variance

the joint

compute weight for each locus

What if we 

- Build joint graph then for each mcmc step
  - sample 



# The "generic exit-rate-ratio form" used by `bffg_log_prob`

## What the formula is

For each transient vertex `v` along a sampled path, the importance weight contribution per step uses **two factors**:

```
log w_step = (exit-rate-ratio factor)  +  (transition-probability ratio factor)
```

In the actual code (`src/phasic/bffg.py` lines 385-390):

```python
log_w += np.log(r_tgt) - np.log(r_prop) - (r_tgt - r_prop) * s_k
log_w += np.log(p_tgt) - np.log(p_prop)
```

where:

- `s_k` is the sojourn time at vertex `v`
- `r_v = sum over outgoing edges of w_e` is the total exit rate at `v`
- `w_e = coeffs_e · theta` is the edge weight (linear function of the parameter vector)
- `p_e = w_e / r_v` is the conditional probability of taking edge `e` given a jump from `v`
- Subscript `prop` means evaluated at the proposal parameter vector
- Subscript `tgt` means evaluated at the target parameter vector

## What each piece is

### 1. Exit-rate-ratio piece

The marginal density of "stay at vertex `v` for time `s_k` then leave" is `r_v · exp(-r_v · s_k)`. So the ratio of target to proposal sojourn-time densities is:

```
[r_tgt · exp(-r_tgt · s_k)]  /  [r_prop · exp(-r_prop · s_k)]
  = (r_tgt / r_prop) · exp(-(r_tgt - r_prop) · s_k)
```

Taking logs gives the first three terms in the code:

```
log r_tgt - log r_prop - (r_tgt - r_prop) · s_k
```

### 2. Transition-probability ratio piece

Given a jump occurs from `v`, the probability of taking edge `e` is `w_e / r_v`. So the ratio is:

```
log[(w_e_tgt / r_tgt) / (w_e_prop / r_prop)]
  = log p_tgt - log p_prop
```

## Equivalence to a simpler form

Multiplying the two factors out:

```
(r_tgt / r_prop) · exp(-(r_tgt - r_prop) · s_k) · (w_e_tgt · r_prop) / (w_e_prop · r_tgt)
  = (w_e_tgt / w_e_prop) · exp(-(r_tgt - r_prop) · s_k)
```

The `r_tgt / r_prop` factors **cancel exactly**. So the two-factor form is mathematically identical to:

```
log w_step = log w_e_tgt - log w_e_prop  -  (r_tgt - r_prop) · s_k
```

This is the same density ratio I use in `log_iw_corrected`, just decomposed differently.

## Two important things to notice in `bffg_log_prob`

### A. It uses `path['entry_times']` directly as physical sojourn times

Line 357 in the code:

```python
sojourns = np.diff(times)
```

The `times` here are `path['entry_times']` straight from `sample_path_conditioned`. **There is no resampling.**

This is a bug when the graph is a `joint_prob_graph`. Such graphs have `was_dph=True`, which means `update_weights` normalizes the stored edge weights to sum to 1 per vertex. The C sampler uses these stored weights to compute its sojourn rate, so the `entry_times` it returns come from `Exp(1)` and **do not reflect the proposal CTMC's sojourn distribution**. The exit-rate-ratio formula expects physical sojourn times distributed `Exp(r_v_proposal)`, but it gets `Exp(1)` samples instead.

In my `log_iw_corrected` I fix this by resampling sojourn times myself from `Exp(R_v_raw)` where `R_v_raw = sum(coeffs_e · theta_proposal)` is the **un-normalized** exit rate computed from raw edge coefficients.

### B. It does not include the inverse-guiding correction

`sample_path_conditioned` is **not** the unconditional CTMC. It is an h-guided forward sampler: at each transient vertex it picks the next edge with probability proportional to `w_e · h(child_e)` instead of the unconditional `w_e / r_v`, where `h = backward_probabilities([target_t_vertex])`.

So the actual proposal density per step is:

```
proposal density per step = R_v · exp(-R_v · s_k) · w_e · h(child_e) / sum_e' (w_e' · h(child_e'))
```

But the formula treats the proposal as if it were the unconditional CTMC, where the per-step density is just `w_e · exp(-r_v · s_k)`. The missing factor is the guiding correction:

```
guiding correction per step = h(child_e) / sum_e' (w_e' · h(child_e'))     <- proposal includes this
                              w_e / r_v                                    <- formula assumes this
```

These differ unless `h` is uniform across children (which is essentially never the case — that's the whole point of guiding).

In my `log_iw_corrected` I cancel the proposal's actual guiding factor explicitly:

```
- log h(child_e) + log sum_e' (w_e' · h(child_e'))
```

and replace it with the unconditional jump probability `w_e_target / r_v_target` for the target.

## Why these two issues matter for your notebook

Both effects bias the estimator. In your notebook's "estimator variance" comparison cell you will see:

- **Manual `log_iw_corrected`**: stochastic across runs (`sd ≈ a few`) — genuinely random because the rng is fresh each time, but **unbiased** (Test 1 and Test 2 confirm this).
- **Package `bffg_log_prob` correction**: reports `sd = 0.00` because the seeds are baked into the JIT closure (so the same random paths are reused every call), but the value it converges to is **biased** because of the two issues above.

That's why the two MCMC posteriors don't quite agree in your runs — they're not just two noisy estimators of the same posterior, they're targeting slightly different distributions due to the systematic bias in `bffg_log_prob` on `joint_prob_graph` outputs.

## Summary table

| Aspect                          | `bffg_log_prob` | `log_iw_corrected` |
|---------------------------------|-----------------|--------------------|
| Sojourn times                   | uses `entry_times` directly (Exp(1) on a normalized DTMC graph — wrong) | resamples from Exp(R_v_raw) using raw coefficients |
| Inverse-guiding correction      | absent          | included via `-log h(child) + log sum(w·h)` |
| Identity test (target = prop)   | nonzero variance, biased mean | var(log_w) ≈ 0, mean = log h(v_start) exactly |
| Perturbation test               | biased (does not match analytic h_target / h_proposal) | matches within MC error |
| Stochastic across calls         | no (fixed seeds in JIT closure) | yes |
| JIT-compatible                  | yes (`return_model=True`) | no (pure NumPy) |
| Time-inhomogeneity handling     | `theta_target_fn(theta_mcmc, t)` evaluated per step | unnecessary if epochs are encoded via `add_epoch` |
