# Atlas: Exact-Gradient (`*_grad_theta*`) C Functions in `src/c/phasic.c`

Confirmed-complete list (grep for `*_grad_theta*` plus the two named validators):

| # | Function | Status | file:line (body) |
|---|---|---|---|
| 1 | `ptd_moments_grad_theta` | **production** | phasic.c:10738-10881 |
| 2 | `ptd_moments_grad_theta_dph` | **production** | phasic.c:11142-11338 |
| 3 | `ptd_moments_grad_theta_log` | **production** | phasic.c:10917-11063 |
| 4 | `ptd_sojourn_grad_theta_subset` | **production** | phasic.c:11408-11610 |
| 5 | `ptd_moment0_grad_theta` | validator, `#ifdef PHASIC_B3_VALIDATORS` (OFF by default) | phasic.c:10678-10725 |
| 6 | `ptd_debug_fwdmode_grad` | validator, `#ifdef PHASIC_B3_VALIDATORS` | phasic.c:10486-10515 |
| 7 | `ptd_debug_reverse_grad` | validator, `#ifdef PHASIC_B3_VALIDATORS` | phasic.c:10614-10632 |

One extra candidate was found and ruled **out of scope**: `ptd_graph_pdf_with_gradient` (phasic.c:13090), a PDF·uniformization gradient w.r.t. `params` directly — different mechanism (no elimination-tape theta-adjoint), not named `*_grad_theta*`, not part of the B3 trace-based lineage described in `CLAUDE.md`. Excluded.

`PHASIC_B3_VALIDATORS` is OFF by default (`CMakeLists.txt:24`), not referenced by any CI workflow, and opted into only manually (`CMAKE_ARGS="-DPHASIC_B3_VALIDATORS=ON" pixi run install-dev`, comment at `CMakeLists.txt:23`). Functions 5-7 are therefore not reachable from the shipped Python package, only from `experiments/dr_*.py` de-risk scripts.

## Summary comparison table

| Function | Computes | Weight mode(s) | Direction | Cost scaling | Caching | MPFR gate present? | MPFR gate *applicable* (primal has MPFR path)? | `coefficients_length==0` guard | `off->input_specs` NULL guard | Alloc NULL-checked | Size guard | Test coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ptd_moments_grad_theta` | moment vector `m_0..m_{K-1}` | linear (continuous) | reverse | O(K·(n+nc+L)), P-independent | **rebuilds tape every call** (10745-10746) | yes (10783) | **yes** — primal `ptd_expected_waiting_time` has a real `#ifdef HAVE_MPFR` branch (10085-10145) | yes (10868, added retroactively) | N/A — never touches mmap-cached `off` | **no** (0/14 allocs) | **no** | `dr_moments_jac_gate.py`, `dr_mpfr_gate_test.py`; pytest via `test_exact_grad_rewards.py`, `test_fd_gradient_mixed_scale.py` |
| `ptd_moments_grad_theta_dph` | discrete-corrected moments | linear, was_dph renorm, native DPH | reverse | O(K·(n+nc+L)), P-independent | **rebuilds every call** (11183-11184) | yes (11221) | yes (same primal) | yes (11312) | N/A | **no** (0/16 allocs, plus `Sv`/`SigmaCv`) | **no** | `dr_dph_moments_jac_gate.py`, `dr_dph_renorm_jacobian.py`; pytest `test_exact_grad_discrete.py` |
| `ptd_moments_grad_theta_log` | moments, log weight mode | log (continuous only) | reverse | O(K·(n+nc+L)), P-independent | **rebuilds every call** (10926-10927) | yes (10960) | yes (same primal) | yes (11048) | N/A | **no** (0/14 allocs) | **no** | `dr_log_mode_moments_jac_gate.py`; pytest `test_exact_grad_log_weight_mode.py` |
| `ptd_sojourn_grad_theta_subset` | sojourn(v) for k requested vertices | linear, native DPH (was_dph excluded) | **forward** | O(P·(L+nc+n)), scales with P | **reuses graph-level cache** via `ptd_precompute_reward_compute_graph` (11415); offset-conversion itself still redone per call | yes (11529) | **no** — primal `ptd_expected_sojourn_time_subset` has **no** MPFR path at all (confirmed: no `#ifdef HAVE_MPFR` anywhere in it, 10206-10345) — gate is pure conservatism here | yes (11545, non-crashing seed-to-0) | **yes** (11432, added after review) | **yes** (11477, 11501-11507) | **yes**, `L > 5e7` (11460) | `dr_sojourn_grad_theta_gate.py`, `dr_sojourn_fwdmode_adjoint.py`; pytest `test_exact_grad_joint_index.py` |
| `ptd_moment0_grad_theta` (validator, K=1 oracle) | E[T] gradient | linear (continuous) | reverse | O(n+nc+L), P-independent | rebuilds every call (10688-10689) | **no — never calls the gate at all** (predates it; introduced 85df40c0 07-30 23:35, gate added b1f7fa0a 07-31 09:54, never retrofitted) | n/a (gate absent) | **no — missing** (10715 dereferences `e->coefficients[j]` unconditionally) | N/A | **no** (0/3 allocs) | **no** | `dr_moment0_theta_gate.py` only |
| `ptd_debug_fwdmode_grad` | dE[T]/d(edge weight), fwd-mode + CD oracle | n/a (edge-weight-level, no theta contraction) | forward | O(ni·(L+n)) | fresh env-var-gated stash per call (`ptd_dbg_acquire_clean_off`) | no | n/a | n/a (no theta contraction) | n/a | **no** (0/5 allocs) | **no** | `dr_realtape_validator.py`, `dr_reverse_adjoint_gate.py` |
| `ptd_debug_reverse_grad` | dE[T]/d(edge weight), reverse-mode oracle | n/a | reverse | O(L+nc+n) | same as above | no | n/a | n/a | n/a | **no** (0/3 allocs) | **no** | `dr_reverse_adjoint_gate.py` |

---

## 1. `ptd_moments_grad_theta` (phasic.c:10738-10881)

**Signature:** `int ptd_moments_grad_theta(struct ptd_graph *graph, int nr_moments, double *J_out)`. Computes the exact reverse-mode Jacobian `d[m_0..m_{K-1}]/dtheta` (row-major `K×P`) for the standard power-moment vector of a **continuous**, `weight_mode='linear'`, monolithic parameterized graph. No `weight_mode` argument — theta is implicit via the graph's current edge weights (must match the caller's last `update_weights(theta)`).

**Decline reasons (exhaustive, line-by-line):**
1. `!graph->parameterized || graph->param_length == 0` → -1 (10740). **STATIC** — graph-level, invariant.
2. `nr_moments < 1` → -1 (10741). Call-argument, not theta/topology; effectively fixed per model in practice.
3. `ptape == NULL` from `ptd_graph_ex_absorbation_time_comp_graph_parameterized[_dyn]` (10747). That builder (8501-9011 monolithic, 9011-9562 `_dyn`) has **no internal `return NULL`** for either variant except an unrelated `ptd_pcg_resolve` switch-default (9574, a different helper) — in practice this path only fails via unchecked-allocation crash, not a clean decline; **dynamic/environmental (OOM)**.
4. `off == NULL` from `ptd_pcg_convert_to_offset(ptape, graph, NULL, 0)` (10750). Two root causes inside that helper (3425-3527): allocation failure (**dynamic/environmental**) or an "unencodable pointer" (3502-3505, **STATIC/topology** — only reachable if the tape references an SCC/EXTERNAL anchor, which can't happen here since `external_anchors=NULL,n_external=0` is passed unconditionally).
5. MPFR gate `ptd_dbg_tape_needs_mpfr(nm, nc)` (10783) → -1 (10787). **DYNAMIC** — genuinely theta-dependent (condition number of the current numeric tape).
6. Per-input topology check inside the per-outk contraction loop: `sp.kind != PTD_PCG_PTR_EDGE || sp.byte!=0 || sp.v>=... || sp.e>=...` → `ok=0` (10852-10854). **STATIC** — tape-input topology is the same on every call for a given graph.
7. Final isfinite sweep over the **entire** `K*P` output (10872) → `ok=0`. **DYNAMIC** (theta-value-dependent) — but unlike sojourn, there is no per-call output-subset concept here (moments has no "indices" argument), so this decline reason does not have the sojourn-style "probe misses a later call's subset" problem.
8. Final `return ok ? 0 : -1;` (10880).

Note: `if (e->coefficients_length == 0) continue;` (10868) is a **guard/skip**, not itself a decline — it makes a coefficient-less tape input (aux back-edges) contribute 0 instead of crashing.

**Caching:** Rebuilds the entire raw parameterized tape from scratch on **every call** (10743-10746) — never calls `ptd_precompute_reward_compute_graph` and never touches `graph->parameterized_reward_compute_graph`/`_off`. The sojourn function's own docstring (11369-11371) explicitly contrasts itself against this: "Unlike `ptd_moments_grad_theta` (which rebuilds the whole O(n^3) parameterized tape from scratch on every call, tolerable there because moment-graphs are typically modest-sized)…". No benchmark number is given for this cost in the repo; it is simply asserted as tolerable given the target graph sizes.

**Allocation safety:** 14 `malloc`/`calloc` calls (`mem, inv, s0, s1, na, nb, nm, seeds, snaptos, dm, bar_out, adj, bmem, binp`), **zero NULL-checked**. **No size guard** on `L`/`nc`/`n` before allocating — unlike the sojourn function's `L > 5e7` guard, this function has never been retrofitted with either safety net after the sojourn review found the gap.

**`coefficients_length==0` / aux edges:** Guarded (10868). This guard was **added retroactively** in commit `c0cb9de1` (2026-07-31), after building the discrete/was_dph extension exposed it as a **real, already-shipped, always-latent segfault**: any continuous parameterized graph built with `add_aux_vertex`/`add_aux_vertex_constant` (e.g., `Graph.joint_stop_prob_graph()`) combined with `exact_moment_grad=True` would dereference a NULL `coefficients` pointer. Confirmed via manual repro in `B3-DISCRETE-MERGE-REVIEW.md` §3.1 — **this repro is not present as a permanent pytest test anywhere** in `tests/pytest/` (only `test_exact_grad_discrete.py` covers `discretize()`'s aux edges, via `ptd_moments_grad_theta_dph`, not this function).

**`off->input_specs` NULL / mmap:** This function is **structurally immune** to the mmap-cache NULL-`input_specs` bug — it builds its own private `off` from a private `ptape` every call and never reads `graph->parameterized_reward_compute_graph_off` (the only field that can hold a mmap-loaded, `input_specs==NULL` descriptor, set at phasic.c:3813-3814). Immunity is a side effect of always-rebuild, not an explicit guard.

**MPFR gate applicability:** **Genuinely applicable.** The moments primal is `ptd_expected_waiting_time` (via `graph_builder.cpp:518` `g.expected_waiting_time(rewards)`), which has a real `#ifdef HAVE_MPFR` auto-escalation branch (phasic.c:10085-10145, condition-number-triggered). A double-precision adjoint really would be inconsistent with an MPFR-computed primal here, unlike the sojourn case.

**Known bugs already fixed:** NULL-deref on `coefficients_length==0` aux edges, commit `c0cb9de1` (2026-07-31), described above.

**Test/gate coverage gaps:** No permanent pytest regression test for the aux-vertex/coefficient-less-edge NULL-deref repro on this specific function (only a one-off manual script in the review doc). No test exercises the `off==NULL`/`ptape==NULL` OOM paths (untestable without fault injection). No allocation-failure or oversized-tape test (no size guard exists to test).

---

## 2. `ptd_moments_grad_theta_dph` (phasic.c:11142-11338)

**Signature:** `int ptd_moments_grad_theta_dph(struct ptd_graph *graph, int nr_moments, const double *theta, size_t theta_len, double *J_out)`. Discrete-corrected moment Jacobian, covering both `was_dph=True` (`Graph.discretize()`, renormalised sibling-coupled edges) and native DPH (`was_dph=False`). Reuses `ptd_moments_grad_theta`'s stage-0/1/2 verbatim (11122-11141 comment); the only new math is the was_dph renorm quotient-rule contraction plus a fixed continuous→discrete moment-space correction applied afterward.

**Decline reasons:**
1. `!graph->parameterized || graph->param_length==0` → -1 (11144). **STATIC.**
2. `nr_moments < 1` → -1 (11145). Call-argument.
3. `theta_len != P` → -1 (11147). Call-argument consistency check.
4. `mixed` — a `was_dph` vertex mixing constant (`coefficients_length==0`) and parameterized out-edges → -1 (11178, after freeing `Sv`/`SigmaCv`). **STATIC** — pure graph topology, only evaluated when `graph->was_dph` (11159-11179); does not vary with theta.
5. `ptape == NULL` → -1 (11185). Environmental/topology, same as function 1.
6. `off == NULL` → -1 (11188-11192). Environmental/topology, same as function 1.
7. MPFR gate (11221) → -1 (11226, frees `Sv`/`SigmaCv` too — correctly). **DYNAMIC.**
8. Per-input topology check in the contraction loop (11290-11292) → `ok=0`. **STATIC.**
9. **First** isfinite sweep, over the raw continuous-moment `K*P` Jacobian, before the discrete correction (11326) → `ok=0`. **DYNAMIC.**
10. `ptd_dph_correct_discrete_moment_grad(J_out, K, P)` applied only if `ok` (11327) — pure fixed linear map (Stirling numbers / binomials / factorials, computed with small integer `K` in practice; no guard against `K` large enough to overflow `ptd_dph_factorial`, but `nr_moments` is always small in every caller — theoretical, unexercised risk).
11. **Second** isfinite sweep, over `J_out` **after** the discrete correction (11328) → `ok=0`. **DYNAMIC** — this is a decline point **not present** in `ptd_moments_grad_theta`/`_log` (they only sweep once); the discrete correction step could in principle introduce new non-finiteness even from a finite continuous Jacobian.
12. Final `return ok ? 0 : -1;` (11337).

Guards (not declines): `coefficients_length==0` skip **and** starting-vertex skip, combined (11312-11313) — both needed here (unlike the linear function) because the was_dph branch divides by `Sv[sp.v]`, and `0 * inf = NaN` (not 0) in IEEE754 if a starting-vertex edge were ever a tape input with `Sv==0`.

**Caching:** Rebuilds tape every call (11183-11184), identical to function 1.

**Allocation safety:** Same 14 core allocations as function 1, plus `Sv`/`SigmaCv` (calloc'd at 11160-11161) — **16 total, zero NULL-checked**. No size guard.

**`coefficients_length==0`:** Guarded (11312), correctly, from initial introduction (commit `c0cb9de1`) — this is the function whose construction is what *exposed* the bug retroactively fixed in function 1.

**`off->input_specs`/mmap:** Structurally immune, same reasoning as function 1 (never reads `graph->parameterized_reward_compute_graph_off`).

**MPFR gate applicability:** Genuinely applicable — same shared primal (`ptd_expected_waiting_time`) as function 1; `graph_builder.cpp`'s discrete path is `continuous_to_discrete_moments(continuous moments)` regardless of `was_dph` (comment 11132-11134), so the same MPFR risk on the underlying continuous elimination applies.

**Known bugs already fixed:** the coefficients-length/starting-vertex fixes documented in `B3-DISCRETE-MERGE-REVIEW.md` §3 (commit `c0cb9de1`) were designed into this function from the start (it's the one that *found* the class of bug).

**Test/gate coverage:** `dr_dph_moments_jac_gate.py` covers `_erlang().discretize(0.5)` (aux back-edges), `_chain(2).discretize(0.3)` (sibling coupling), native DPH, and an MPFR-decline case (`B3-DISCRETE-MERGE-REVIEW.md` §4). `tests/pytest/inference/test_exact_grad_discrete.py` covers `discretize()` fixtures, native DPH, vmap, and MPFR-fallback-stays-finite. **Not covered:** the `mixed` (constant+parameterized sibling out-edges) decline path — no fixture found that triggers it; the second (post-correction) isfinite sweep as a distinct failure mode from the first.

---

## 3. `ptd_moments_grad_theta_log` (phasic.c:10917-11063)

**Signature:** `int ptd_moments_grad_theta_log(struct ptd_graph *graph, int nr_moments, const double *theta, size_t theta_len, double *J_out)`. Moment Jacobian for `weight_mode='log'` (continuous only). Reuses functions 1's stage-0/1/2 verbatim; only the edge→theta contraction differs (product rule: `dw_e/dtheta_j = w_e/theta_j` for **every** `j`, not conditioned on `coefficients[j]`).

**Decline reasons:**
1. `!graph->parameterized || graph->param_length==0` → -1 (10919). **STATIC.**
2. `nr_moments < 1` → -1 (10920). Call-argument.
3. `graph->was_dph` → -1 (10921). **STATIC** (graph-level flag) — but per the in-code comment (10902-10913) this exclusion is **load-bearing, not defensive**: `log`+`discretize()` does not always fail elsewhere (confirmed by direct repro), so removing this check would be a correctness bug, not just redundant safety.
4. `theta_len != P` → -1 (10923). Call-argument.
5. `ptape == NULL` → -1 (10928). Environmental/topology.
6. `off == NULL` → -1 (10931). Environmental/topology.
7. MPFR gate (10960) → -1 (10964). **DYNAMIC.**
8. Per-input topology check (11029-11031) → `ok=0`. **STATIC.**
9. Final isfinite sweep over full `K*P` (11054) → `ok=0`. **DYNAMIC.**
10. Final `return ok ? 0 : -1;` (11062).

Guards (not declines): starting-vertex skip (11046, defensive — empirically this class of input was "NOT found to register as a tape input in practice" per the comment, since `_graph_serialize.py`'s `start_param_edges` branch is dead code, `if False:`) + `coefficients_length==0` skip (11048).

**Caveat noted in the code itself (10906-10914):** `is_discrete` (native DPH) has **no C-level field** on `ptd_graph` — it's Python-only, reaching C++ only via `serialize()`'s JSON. So this function **cannot** check for it; the Python caller (`pmf_and_moments_from_graph`'s gate) **must** exclude `is_discrete` before calling. `was_dph` here is only "an additional safety net for the subset of `is_discrete` graphs that DO set it" — meaning the exclusion enforced by this C function alone is incomplete by design; correctness for `is_discrete` graphs depends entirely on Python-side gating outside this file.

**Caching:** Rebuilds tape every call (10926-10927), identical to function 1.

**Allocation safety:** Same 14 allocations as function 1, zero NULL-checked, no size guard.

**`coefficients_length==0`:** Guarded (11048).

**`off->input_specs`/mmap:** Structurally immune (never reads `graph->parameterized_reward_compute_graph_off`).

**MPFR gate applicability:** Genuinely applicable (same shared `ptd_expected_waiting_time` primal, log mode only changes `w_e` computation, not the elimination arithmetic).

**Note on `theta[j]` safety:** the division `e->weight / theta[j]` (11050) is asserted safe by comment (10893-10896) because `update_weights` requires `c_e[i]*theta[i] > 0` strictly for every log-mode edge — so no `theta[j]` reaching this function can be exactly 0. This is a caller-contract assumption, not independently re-validated inside this function.

**Known bugs already fixed:** None specific to this function found in git log beyond its own introduction/review commits (`c986daf7`, `ccf558c5`); it inherited functions 1/2's fixes by construction since it postdates them.

**Test/gate coverage:** `dr_log_mode_moments_jac_gate.py`, `dr_log_mode_edge_jacobian.py`; pytest `test_exact_grad_log_weight_mode.py` (branching graphs, mixed-scale theta, vmap, `discretize()+log` decline, MPFR-fallback-stays-finite). **Not covered:** no aux-vertex/`coefficients_length==0` fixture in log mode (log-mode test graphs are dense/branching, not aux-edge-bearing) — the guard is present but its live-fire path is untested for this specific function.

---

## 4. `ptd_sojourn_grad_theta_subset` (phasic.c:11408-11610)

**Signature:** `int ptd_sojourn_grad_theta_subset(struct ptd_graph *graph, const size_t *indices, size_t k, double *J_out)`. **Forward**-mode Jacobian `d(sojourn(indices[r]))/dtheta` for `k` requested target vertices — the only B3 gradient function using forward-mode (cost scales with `P`, not `P`-independent like 1-3), because the primal (`ptd_expected_sojourn_time_subset`) itself has a many-outputs/few-inputs shape that reverse-mode would handle badly. `weight_mode='linear'` + native DPH; `was_dph` excluded.

**Decline reasons (exhaustive):**
1. `!graph->parameterized || graph->param_length==0` → -1 (11410). **STATIC.**
2. `graph->was_dph` → -1 (11411). **STATIC.**
3. `k == 0` → **0 (success, no-op)**, not a decline (11413).
4. `ptd_precompute_reward_compute_graph(graph)` fails → -1 (11415). Environmental (build/cache I/O failure).
5. `off->n_inputs > 0 && off->input_specs == NULL` (only reachable via the mmap-loaded branch) → -1 (11432). **STATIC once established for a process/graph** (depends on whether `PHASIC_REWARD_COMPUTE_CACHE` is enabled and the on-disk cache is warm — an environment/config-level property, invariant across calls on the *same* graph after the first `ptd_precompute_reward_compute_graph`, but not a property discoverable from graph topology alone).
6. **`indices[r] >= n` for any r** → -1 (11444). **DYNAMIC / per-call** — depends on the specific `indices` array passed to *this* call, not on theta. This is the reason a construction-time single-index probe (`[0]`) **cannot generalize**: a later call with a different, out-of-range index would still decline, but the probe (run once with `indices=[0]`) can never observe that.
7. `L > 50000000` size guard → -1 (11460). **STATIC** (L is a graph-topology property, fixed once the tape is built).
8. `edge_for_input == NULL` (malloc failure) → -1 (11477-11480). Environmental.
9. Per-input topology check populating `edge_for_input[]` (11483-11487) → -1 (11488-11492). **STATIC/topology.**
10. `mem/inv/s0/s1/na/nb/nm` allocation failure → -1 (11501-11507). Environmental.
11. MPFR gate (11529) → `ok=0` (no early free/return here — falls through to the end, `J_out` never populated since the `for j<P` loop is gated on `ok`). **DYNAMIC.**
12. `mem_dot/inv_dot/mdot/y/y_dot` allocation failure (only checked if `ok`, 11531-11538) → `ok=0`. Environmental.
13. **Final isfinite sweep, but only over `J_out[0..k*P)` — i.e., only the REQUESTED rows** (11598) → `ok=0`. **DYNAMIC / per-call** — same generalizability problem as #6: a trap/deficit-sink vertex at an index *not* requested by the construction-time probe would never be observed by that probe, but could cause a real, later, per-call decline when requested by a subsequent call with different `indices`.
14. Final `return ok ? 0 : -1;` (11609).

**This confirms**: decline reasons #6 and #13 are genuinely per-call (index-set-dependent), not graph-wide/theta-wide, and are NOT generalizable by a single construction-time probe at one reference index.

**Caching:** **Reuses** the graph-level cache via `ptd_precompute_reward_compute_graph(graph)` (11415) — either the mmap-loaded `graph->parameterized_reward_compute_graph_off` directly (11419-11432), or converts the cached **raw** tape `graph->parameterized_reward_compute_graph` to offset form fresh (11433-11436, `owns_off=1`). Per `b3-joint-index-plan.md` D3: this eliminates the catastrophic O(n³)-rebuild-per-call risk, but the O(L) `ptd_pcg_convert_to_offset` step (when the mmap cache isn't warm) is **not itself cached** and is redone every call. Measured: exact is a correctness win always, but a **speed** win only for `P`≳10-20 on a representative 2000-vertex chain (P=2: FD 4× faster; P=50: exact ~2.9× faster); at the joint-index model's native P=2 on a realistic n=39603 graph, FD is ~2.6× faster — which is why `exact_grad` defaults to **`False`** here (unlike every other B3 function, which defaults `True`).

**Allocation safety:** The **only** one of the seven functions with NULL checks on allocation (11477, 11501-11507, 11536-11538) and a size guard (`L > 5e7`, 11460) — both added by the D5 adversarial-review fix (commit `d64c3400`).

**`coefficients_length==0`:** Guarded non-crashingly — coefficient-less edges are seeded to `inv_dot[kk]=0.0` (11545) rather than dereferencing `e->coefficients[j]`.

**`off->input_specs`/mmap:** **The only function of the seven that reads the graph-level cached `off`**, and therefore the only one that needed (and, after `bc071d84`, got) the explicit `off->input_specs == NULL` guard (11432). Fixed 2026-08-04, found "by the adversarial review of the D6 plan, independent of D6 itself" (commit `bc071d84`).

**MPFR gate applicability:** **NOT genuinely applicable.** `ptd_expected_sojourn_time_subset` (10206-10345, the primal) has **zero** `#ifdef HAVE_MPFR` code anywhere in its body — confirmed by direct read. Per `CLAUDE.md`'s "Disabled paths" section and the plan: the gate here is "a pure, build-dependent (inert without `HAVE_MPFR`) conservatism knob… not a correctness necessity," and the in-code comment claiming the same rationale as `ptd_moments_grad_theta`'s gate is acknowledged as **not yet corrected** to reflect this.

**Known bugs already fixed:** (a) tangent-guard `0*inf=NaN` per-summand fix, (b) allocation NULL-checks + size guard, (c) `off->input_specs` mmap NULL-deref (separate later commit `bc071d84`) — all documented in `b3-joint-index-plan.md` D5/D6 and `CLAUDE.md`.

**Test/gate coverage:** `dr_sojourn_grad_theta_gate.py` (D3, all continuous/branching/native-DPH/joint-prob-graph/MPFR/empty-indices cases); `tests/pytest/inference/test_exact_grad_joint_index.py` (unsorted/duplicated/subset indices, vmap, `fixed_mask`, `was_dph` decline, `theta_dim` override, `observed_indices` baked-mode decline). **Confirmed gaps** (explicitly stated in `CLAUDE.md`): no test fixture exercises a trap/deficit-sink vertex through the gradient path (`test_sojourn_subset_adjoint.py` tests the **primal** with deficit-sinks, but never calls the gradient function — confirmed, zero references to `grad_theta`/`exact_grad` in that file); no mmap-cache-warm test exercising the new `off->input_specs` guard; no out-of-range-index test found in the searched files (decline #6 above is asserted by code reading, not confirmed exercised by a test); no measured cost/memory profile at production scale (n~7×10⁵).

---

## 5. `ptd_moment0_grad_theta` (validator, phasic.c:10678-10725)

**Signature:** `int ptd_moment0_grad_theta(struct ptd_graph *graph, double *ewt_out, double *dtheta_out)`. K=1 special case of function 1's math, explicitly documented as "superseded by `ptd_moments_grad_theta`… Kept as a de-risk oracle" (10667-10668). Introduced (commit `85df40c0`, 2026-07-30) **before** both the MPFR gate (`b1f7fa0a`, 2026-07-31) and the coefficients_length fix (`c0cb9de1`, 2026-07-31) existed, and **never retrofitted with either**.

**Decline reasons:**
1. `!graph->parameterized || graph->param_length==0` → -1 (10680). **STATIC.**
2. `ptape == NULL` → -1 (10690). Environmental/topology.
3. `off == NULL` → -1 (10693). Environmental/topology.
4. `rc = ptd_dbg_reverse_tape(...)` then `ok = (rc==0)` (10702, 10704) — **dead branch**: `ptd_dbg_reverse_tape` (10528-10611) has no internal failure path and always `return 0;` unconditionally, so `ok` is always initially `1` here; this "decline reason" can never actually fire in the current code.
5. Per-input topology check (10709-10713) → `ok=0`. **STATIC.**
6. Final isfinite sweep over `dtheta_out[0..P)` (10717) → `ok=0`. **DYNAMIC.**
7. `!isfinite(q)` on the scalar `E[T]` (10718) → `ok=0`. **DYNAMIC.**
8. Final `return ok ? 0 : -1;` (10724).

**Critical finding — a genuine, previously-undocumented latent bug:** The contraction loop (10709-10716) does **`dtheta_out[j] += ge[k] * e->coefficients[j];`** (10715) with **no `coefficients_length==0` guard** — an unconditional dereference of `e->coefficients`, which is NULL for coefficient-less aux edges (as established by `B3-DISCRETE-MERGE-REVIEW.md` §3.1). This is the **exact same bug class** that was found and fixed in `ptd_moments_grad_theta` (the function this one claims to be superseded by) — but the fix was never backported here. **Also missing the MPFR gate entirely** — no call to `ptd_dbg_tape_needs_mpfr` anywhere in the function body, unlike its production successor. Both gaps are consistent with this function predating both fixes and being frozen as a "kept for de-risk" oracle that nobody re-audited afterward. Low real-world severity (validator-only, `PHASIC_B3_VALIDATORS` OFF by default, not exercised by CI), but it is exactly the kind of gap a systematic inventory should catch.

**Caching:** Rebuilds tape every call (10686-10689), same as functions 1-3.

**Allocation safety:** 3 allocations (`mem0, inp0, ge`), zero NULL-checked, no size guard.

**Test/gate coverage:** `dr_moment0_theta_gate.py` only (no pytest — validator-only).

---

## 6. `ptd_debug_fwdmode_grad` (validator, phasic.c:10486-10515)

**Signature:** `int ptd_debug_fwdmode_grad(struct ptd_graph *graph, double *ewt_out, double **fwd_out, double **cd_out, size_t *ni_out)`. Computes `dE[T]/d(edge weight)` for every tape input, by forward-mode AND by self-contained central difference (both replayed on a local copy of the tape) — an **edge-weight-level** oracle, not a theta-adjoint (no coefficient contraction at all; theta doesn't appear).

**Decline reasons:** exactly **one**: `off == NULL` (10490), where `off = ptd_dbg_acquire_clean_off(graph)`. Tracing into that helper (10467-10484): it force-rebuilds via `ptd_precompute_reward_compute_graph` with `PHASIC_PCG_SELFCHECK`/`PHASIC_DBG_STASH_OFF` env vars set, and returns NULL either when that precompute call fails (`pc != 0`) or — silently — when it succeeds but never stashed `graph->_dbg_off_clean` (e.g., a non-parameterized graph, whose precompute path never runs the parameterized-stash hook). The in-code comment "/* stash did not fire (not parameterized?) */" (10490) is thus an accurate but **indirect** signal — this function has no explicit `graph->parameterized` check of its own. **STATIC** (graph-level) in practice.

**No isfinite check at all** on the returned `fwd`/`cd` arrays — unlike every other function in this atlas, it unconditionally returns `0` (success) after computing, even if the result contains NaN/Inf.

**Concurrency note:** `ptd_dbg_acquire_clean_off` uses `setenv`/`unsetenv` (process-wide global state) to signal the stash hook — **not thread-safe** if called concurrently from multiple threads. Low real-world impact (validator-only, single-threaded de-risk scripts), but worth flagging.

**Allocation safety:** 5 allocations (`mem0, inp0, fwd, cd, idot`), zero NULL-checked.

**Caching:** Neither reuses nor "rebuilds an existing" cache in the same sense as 1-4 — it force-invalidates and rebuilds specifically to get a clean pre-execution tape snapshot, every call, by design (it's a differential oracle, not a production path).

**Test/gate coverage:** `dr_realtape_validator.py`, `dr_reverse_adjoint_gate.py`.

---

## 7. `ptd_debug_reverse_grad` (validator, phasic.c:10614-10632)

**Signature:** `int ptd_debug_reverse_grad(struct ptd_graph *graph, double *ewt_out, double **grad_out, size_t *ni_out)`. Reverse-mode counterpart to function 6 — `dE[T]/d(edge weight)` via one backward pass, target=vertex 0 hardcoded.

**Decline reasons:**
1. `off == NULL` (10618) — identical mechanism to function 6.
2. `rc` from `ptd_dbg_reverse_tape(...)` (10626) → -1 if nonzero (10629) — but as noted for function 5, `ptd_dbg_reverse_tape`'s body (10528-10611) has **no internal failure path** and always returns `0`; this decline branch is currently **dead code**.

No isfinite check on `grad`/`q` either (same gap as function 6).

**Allocation safety:** 3 allocations (`mem0, inp0, grad`), zero NULL-checked.

**Test/gate coverage:** `dr_reverse_adjoint_gate.py` (cross-checked against function 6's forward-mode + CD outputs, 79/79 match per the plan comments).

---

## Cross-cutting observations

1. **Fix-lineage divergence.** Every safety fix found by adversarial review (MPFR gate, `coefficients_length==0` guard, alloc NULL-checks, size guard, mmap `input_specs` guard) was applied **only to the function under review at the time**, never backported to earlier or "superseded" siblings still compiled into the codebase. `ptd_moment0_grad_theta` in particular carries two of these gaps simultaneously (missing MPFR gate + missing `coefficients_length` guard) that its stated successor `ptd_moments_grad_theta` had fixed.
2. **The sojourn function's two "dynamic, non-generalizable" decline reasons are real and confirmed by code reading**: `indices[r] >= n` (11444) and the isfinite sweep restricted to `J_out[0..k*P)` (11598) both depend on the specific `indices` argument of a given call, not on theta or graph topology.
3. **MPFR-gate applicability is genuinely bifurcated**: applicable for all three moment functions (shared `ptd_expected_waiting_time` primal has a real MPFR branch, confirmed at phasic.c:10085-10145) but **not applicable** for the sojourn function (its primal `ptd_expected_sojourn_time_subset` has zero MPFR code) — confirmed by direct inspection, matching `CLAUDE.md`'s claim.
4. **Caching is bimodal by construction, not by oversight**: functions 1-3 (and 5-7) always rebuild the raw parameterized tape from scratch; only function 4 reuses `ptd_precompute_reward_compute_graph`'s graph-level cache — a deliberate scope choice tied to sojourn's much larger target graph sizes (n up to ~7×10⁵), not something the other three merely "forgot."
