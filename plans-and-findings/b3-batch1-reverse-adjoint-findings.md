# Batch-1 — reverse-mode θ-adjoint over the real _off tape — COMPLETE (tape adjoint)

Ports the verified reference interpreter `experiments/dr_twotier_full_adjoint.py`
(218/218 vs JAX autodiff) into C as `ptd_debug_reverse_grad` /
`Graph._debug_reverse_grad`, operating on the REAL `_off` two-tier elimination
tape (clean pre-execution mem via the Batch-0 stash mechanism). Non-shippable
validator; isolated worktree env; master install untouched.

## What it computes
Reverse-mode `dQ/d(edge weight)` for `Q = E[T] = result[0]` (first moment, seed
all-1, continuous), in three linked stages exactly as the plan/reference:
1. param-tape forward: mutate local mem/inputs copies, snapshot operand primals
   per op, record numeric commands with `m_c = (from==to)? weight-1 : weight`;
2. numeric replay forward: seed all-1, snapshot `result[to]` per command
   (primal skips `m_c==0` / `inf*0` to match native bit-for-bit);
3. stage-1 reverse (`dm_c = adj[a]*snap_to` emit-before-transpose; transpose
   `adj[b]+=adj[a]*m_c`) → stage-2 reverse of the 7 ops with REPLACE/kill
   semantics, gluing `bar[mptr] += dm_c` in-order at each NEW_ADD.

Both Batch-0 findings are honored: the diagonal `-1` is applied to the primal /
snapshot / transpose (the glue is unchanged because `d(m_c)/d(weight)=1`), and
the `m_c==0` command still emits its gradient (`dm_c`) while the transpose is a
no-op.

## GATE: PASS on cyclic fixtures across the regime grid
`experiments/dr_reverse_adjoint_gate.py` (run in the worktree env):

- **2-cycle** `s->A<->B->abs` (4 inputs), θ ∈ {[1,1],[1,.5],[2,1],[1,1e-3],[1,1e-6]}:
  reverse == forward-mode oracle to **machine precision** (|Δ|max 0, 8.9e-16,
  2.2e-16, 2.3e-10), reverse == central-diff, reverse E[T] == native; at θ=[1,1e-6]
  the dominant `reverse[2] = -2e12` == closed-form exactly (CD off 2e-5).
- **3-cycle** `s->A->B->C->A, C->abs` (5 inputs), θ ∈ {[1,1],[1.5,.7],[3,.2],[.5,2]}:
  reverse == forward-mode (|Δ|max 0, 1.1e-15, 7.1e-15, 1.1e-16), == central-diff,
  == native E[T] at every point.

`reverse == forward-mode` component-wise is the dot-product identity
`⟨Jv,u⟩ == ⟨v,Jᵀu⟩` (amendment 7) evaluated at every basis vector — forward-mode
shares NONE of the reverse-interpreter code, so this pins the shipped transpose
(stage-1 + stage-2 REPLACE/kill + emit-before-transpose ordering) and catches
in-place-aliasing bugs a value-only comparison would miss. (The two AD paths are
independent, so they agree to the conditioning floor ~1e-10 rel at extreme mixed
scale, not bit-for-bit; a real transpose bug is O(1) rel — caught at machine
precision on the benign scales.)

## Resolved: amendment 1 (input_specs on the fresh-convert path)
`ptd_pcg_convert_to_offset` sets `off->input_specs = spec` with `{kind, v
(vertex_idx), e (edge_idx), byte}` on the NON-load path (not just heap-load, as
the stale struct comment implied). So the edge→θ map the Batch-2 gradient FFI
needs (input k → `graph->vertices[v]->edges[e]->coefficients` → `∂w_e/∂θ_j = c_j`
for linear mode) is available directly. This closes the plan's amendment-1 risk.

## Scope (Batch-1 = the tape adjoint; edge→θ + wiring = Batch-2)
Delivered: the reverse adjoint over the tape → `dQ/d(edge weight)`, where ALL the
elimination-algebra risk lives (division, cycles, REPLACE/kill, diagonal-1,
mult==0), validated to machine precision. Deferred to Batch-2 (the shippable
wiring, which needs the edge→θ map regardless): the **edge→θ linear contraction**
(feasible per above), the **gradient FFI handler + defvjp swap** replacing the
`eps=1e-7` FD loop (`moments_from_graph` first), the **MPFR gate** (cond>1e12
diverts the primal → FD fallback), and **snapshot buffers captured inside the
locked precompute** (production; the validator already captures clean state).
Still first-moment / continuous / linear / monolithic / no-was_dph; higher
moments, was_dph, log/formula, joint-index, hierarchical → Batch-3.

## Status
- Reference interpreter: verified (218/218 vs JAX).
- Batch-0 forward-mode oracle on the real tape: confirmed (both gates, all scales).
- Batch-1 reverse adjoint on the real tape: CONFIRMED == oracle + central-diff +
  native + closed-form, 2 fixtures × regime grid.
- Next: Batch-2 — edge→θ contraction + gradient FFI + defvjp swap for
  `moments_from_graph`, flip the mixed-scale pin.
