# Eliminating the dual graph requirement in BFFG

## Current problem

`bffg_log_prob` requires two copies of the `joint_prob_graph`:
- `jg_disc` (discrete): for `joint_prob_table()` to get proposal SFS probabilities `P_proposal(s)`
- `jg_continuous` (with `is_discrete=False`): for `sample_path_conditioned()` with correct exponential sojourn times

This duplication exists because:
- `joint_prob_table()` assumes normalized edge weights (summing to ≤ 1) and gives wrong probabilities on the unnormalized continuous graph
- `sample_path_conditioned()` on the discrete graph draws sojourn times from Exp(1) instead of Exp(actual_rate), giving wrong importance weights

## Proposed fix

Use `backward_probabilities([target_vertex])` on the **continuous** graph to compute `P_proposal(s)`. The backward pass normalizes internally (`weight_i / total_weight`), so it correctly computes transition probabilities regardless of whether edge weights are normalized or not. The value `h[start_vertex]` equals the absorption probability at the target — which is exactly `P(s)`.

This would allow using a single continuous graph for both:
1. `P_proposal(s)` via `backward_probabilities`
2. Conditioned path sampling via `sample_path_conditioned`
3. Importance weights via the edge coefficients

## To verify

Compare `backward_probabilities([target])[0]` on the continuous graph with `joint_prob_table().loc[target, 'prob']` on the discrete graph for all terminal vertices. They should match.

## Impact

- `bffg_log_prob` would take a single graph instead of `jg_disc` + `jg_continuous`
- No need for the user to manage two copies
- Simpler API
