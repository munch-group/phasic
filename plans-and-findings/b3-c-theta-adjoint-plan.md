# Plan — C reverse-mode θ-adjoint for exact trace-expressible gradients (B3 Tier-3)

> **Adversarial review: SOUND-WITH-GAPS (core math CORRECT) → 9 amendments folded in (v2).**
> The multiplier-gradient math is verified (reviewer hand-example + my synthetic-tape
> check `experiments/dr_tape_adjoint.py`, 15 tapes incl. self-loops/aliasing, machine
> precision). Amendments, most-critical first:
> 1. **Force `_off` on the adjoint path** — the `_off` tape (with MEM/INPUT provenance the
>    edge→θ chain needs) runs ONLY on a rev-3 cache HIT; default runs the RAW tape whose
>    bare `double*` operands carry NO provenance. Must call `ptd_pcg_convert_to_offset`
>    unconditionally when a gradient is requested, and verify a freshly-converted `_off`
>    populates `input_specs` (vertex,edge) on the non-load path (`phasic.c:2069`, `:3716-3725`).
> 2. **Per-op REPLACE/kill semantics** in the stage-2 param reverse: `INV`/`ONE_MINUS`/
>    `DIVIDE` **replace** `bar[fromT]`; `ZERO` sets it 0; `P`/`PP` keep it. A uniform
>    accumulate-transpose leaks adjoint through reused mem slots → wrong.
> 3. **Reverse ordering:** emit `dm_c = adjoint[from]·snapshot` **before** the transpose
>    `adjoint[to]+=adjoint[from]·m_c` — load-bearing for the `from==to` self-multiply
>    commands `add_command` emits with `mult=weight-1` (independently verified in my check).
> 4. **Scope "moments" honestly:** Batch-1 = FIRST moment only (single replay) — which is
>    enough to XPASS the pin (it differentiates `moments[0]` only). Higher moments
>    (`nr_moments≥2`) chain replays whose SEED is θ-dependent → a separate seed-chain
>    sub-batch (`graph_builder.cpp` moment recursion).
> 5. **Runtime MPFR gate in Batch-1/2** (not Batch-3): `expected_waiting_time` diverts to
>    MPFR when condition>1e12 (`phasic.c:10060-10112`) — a θ-dependent switch that can fire
>    mid-SVGD; a double-tape adjoint would mismatch the MPFR primal → FD fallback required.
> 6. **Handle the `m_c==0`/`inf` skip asymmetry:** the forward skips those commands but
>    `dQ/dm_c` is generally nonzero there — decide structural-drop vs θ-reachable-keep.
> 7. **Batch-0 must add a forward/reverse dot-product identity** `⟨Jv,u⟩==⟨v,Jᵀu⟩` on the
>    same C tape — forward-mode shares NONE of the reverse-interpreter code, so it alone
>    doesn't pin the shipped reverse path.
> 8. **Batch-2 target = `moments_from_graph`** (clean full-FD replacement); `pmf_and_moments`
>    is separable (`__init__.py:7459` `theta_bar=grad_pmf+grad_moments`) but its FD loop
>    still runs the pmf forward at θ±eps, so no forward-pass saving there.
> 9. **Reclassify the prototype** as a *principle* oracle (dense rate-matrix elimination,
>    ops `{+,−,×,÷}`; exercises none of `INV`/`ONE_MINUS`/`ZERO`/`NEW_ADD`/two-tier/mem-reuse).
>    Batch-0 on the real tape is the MANDATORY de-risk, not optional.

Branch `fd-b3-experiments`. Scopes the recommended B3 fix: replace the
finite-difference `custom_vjp` backward with an **exact reverse-mode θ-gradient
computed in C**, for the quantities that go through the elimination tape. Grounded
in the C-internals investigation + the validated Python prototype.

## Goal & scope

- **In scope (tape-computed):** `moments`, `expected_waiting_time`,
  `expected_sojourn_time`, joint-index. These replay the numeric
  `reward_compute_graph` (`phasic.c:9955`, `:10122-10169`, `:10181-10264`).
- **Out of scope (stay on FD / separate future adjoint):** pmf/pdf
  (uniformization, `ComputePmfFfiImpl`) and daisy-chain (its own FFI impls; FD VJPs
  at `__init__.py:4649/:4880`). These never touch the tape.
- **Fixes** the mixed-scale gradient defect (`tests/pytest/inference/test_fd_gradient_mixed_scale.py`)
  for the in-scope quantities.

## Foundation (de-risked at the PRINCIPLE level; the real tape is Batch-0)

- **Principle validated** (prototype `159238dd`, `experiments/dr_proto_theta_adjoint.py`):
  reverse-mode θ-adjoint over a division-containing Gaussian-elimination trace is exact,
  cyclic-correct, crushes FD. **Caveat (amendment 9):** the prototype is a DENSE
  rate-matrix elimination with ops `{+,−,×,÷}` — it exercises NONE of `INV`/`ONE_MINUS`/
  `ZERO`/`NEW_ADD`, the two-tier structure, or in-place mem reuse. It validates the math
  PRINCIPLE, not phasic's actual tape.
- **Stage-1 (numeric tape) reverse verified** (`experiments/dr_tape_adjoint_stage1.py`): the
  formula `dQ/dm_c = adjoint[from_c]·result_primal[to_c]` + the seed-walk, with the exact
  timing/ordering rules, match JAX autodiff to machine precision on 15 synthetic tapes
  INCLUDING self-loop (`from==to`) commands and aliased slots (both present in the real tape).
- **FULL two-tier algorithm verified build-free** (`experiments/dr_twotier.py`, Batch-0
  algorithm portion): a faithful Python model of the two-tier tape (7 param ops → numeric tape
  → quantity) + the complete reverse θ-adjoint matches JAX autodiff on 218/218 finite random
  tapes to 2.22e-16, exercising all 7 ops, in-place mem reuse, self-loops, the REPLACE/kill
  reverse (amendment 2), the emit-before-transpose ordering (amendment 3), and the
  param↔numeric glue. **It caught two real bugs** (the `PP`/`DIV` aliasing snapshot and the
  glue timing) — this file is now the verified REFERENCE INTERPRETER to port to C. Still
  unverified: the REAL C executors' behaviour + the `_off` provenance → the Batch-0 build.
- **Tier 2 dead** (DR-D: no FD step works). **Exact AD feasible** (DR-A).
- **Conditioning floor** characterised: near-singular sub-generator corrupts the
  *sub-dominant* gradient at extreme mixed scale, IDENTICALLY for adjoint, trace,
  and the linear-solve oracle — inherent to the algebra, narrow, not a blocker.
- **C tape is a straight-line arithmetic program** (investigation): two tiers —
  the parameterized Wengert tape (ops `P/PP/INV/ONE_MINUS/DIVIDE/ZERO/NEW_ADD`
  over edge weights + scratch, `phasic.c:9599-9704`) emits the numeric
  `{from,to,multiplier}` tape, whose replay yields the quantity. Provenance via the
  `_off` operand `kind ∈ {MEM, INPUT}` (`phasic.c:1882-1900`); INPUT = a live
  `&edge->weight`. **Route (i)** (adjoint over the existing tape) recommended over
  (ii) a new θ-aware elimination and (iii) forward-mode (kept only as the oracle).

## The fix (route i): reverse-mode over the two-tier tape

`edge weights(θ) ──[param tape]──► numeric tape {from,to,m} ──[replay]──► Q`.
The θ-adjoint reverses the whole program in three linked stages:

1. **Reverse the numeric replay** for `dQ/dm_c` per command (seed `adjoint[0]=1`,
   transpose walk as in the sojourn adjoint `phasic.c:10234-10244`). **Ordering rule
   (amendment 3):** at each command c in reverse, FIRST emit
   `dm_c = adjoint[from_c] · result_primal[to_c]`, THEN apply
   `adjoint[to_c] += adjoint[from_c]·m_c` — reading `adjoint[from]` before the transpose
   is load-bearing for `from==to` self-multiply commands (`mult=weight-1`). **Skip
   asymmetry (amendment 6):** the forward `continue`s on `m_c==0` / `isinf`; `dm_c` is
   generally nonzero there — keep+emit unless the command is structurally constant.
   **New recording:** one forward-primal `result[to]` snapshot per numeric command
   (value at c's forward execution). *(Stage-1 verified: `dr_tape_adjoint.py`.)*
2. **Reverse the parameterized tape** to turn `dQ/dm_c` into `dQ/d(edge weight)`: a reverse
   interpreter of the 7 ops with **explicit REPLACE/kill semantics per op (amendment 2)** —
   `bar[·]` = adjoint of a mem/edge slot:
   | op (fwd) | forward | reverse on adjoints |
   |---|---|---|
   | `P` | `fromT += toT·const` | `bar[toT] += bar[fromT]·const`; `bar[fromT]` kept |
   | `PP` | `fromT += toT·*mptr` | `bar[toT]+=bar[fromT]·mptr₀`; `bar[mptr]+=bar[fromT]·toT₀`; `bar[fromT]` kept |
   | `INV` | `fromT = 1/fromT` | `bar[fromT] := bar[fromT]·(−1/fromT₀²)` **(REPLACE)** |
   | `ONE_MINUS` | `fromT = 1−fromT` | `bar[fromT] := −bar[fromT]` **(REPLACE)** |
   | `DIVIDE` | `fromT /= toT` | `bar[toT]+=bar[fromT]·(−fromT₀/toT₀²)`; `bar[fromT] := bar[fromT]/toT₀` **(REPLACE)** |
   | `ZERO` | `fromT = 0` | `bar[fromT] := 0` **(kill)** |
   `·₀` = operand primal snapshotted at execution (**new recording**; the executor
   overwrites in place `:9635/9665/9689`). Reverse the commands in REVERSE order. The 1:1
   `NEW_ADD ↔ numeric-command` map (`:9626-9632`, `command_index++`) glues stage-1's `dm_c`
   onto stage-2's `bar[multiplierptr_operand] += dm_c` — the `multiplierptr` operand may be
   **MEM or a bare INPUT edge** (`add_command_param` `:7105`), so the glue must handle both.
3. **Edge→θ Jacobian:** chain `dQ/d(edge weight)` to `dθ` via `update_weights`
   (linear `∂w_e/∂θ_j = c_j`, `phasic.c:5727-5730`; log `= w_e/θ_j`, `:5703-5724`).
   Uses the edge→coefficient map (the `_off` INPUT spec already carries
   `(vertex_idx, edge_idx)`, `:3716-3725`).

Build against the **`_off`** form (clean MEM/INPUT provenance, one executor).

## Batches (each: de-risk gate → implement → test)

> Build reality: this is C — `pixi run install-dev` after each C change. Every
> batch keeps the FORWARD (primal) bit-identical; the adjoint is purely additive.
> Prefer an opt-in flag (`PHASIC_EXACT_GRAD` or a builder arg) with FD fallback so
> the swap is reversible and per-quantity.

### Batch 0 — DE-RISK: is the C tape a complete, correct differentiable trace? (before ANY adjoint)
- **Algorithm portion: DONE build-free** (`experiments/dr_twotier.py`) — the full two-tier
  reverse θ-adjoint is verified vs JAX autodiff (218/218, 2.22e-16) and CAUGHT TWO BUGS
  (aliasing snapshot, glue timing). Port THIS reference interpreter to C. What remains below
  is the REAL-C-tape confirmation (needs a build).
- **Experiment (investigation #1, highest value, ~1 file):** add a forward-mode
  tangent `mem_dot[]` alongside the existing param+numeric executors — seed INPUT
  slots with `∂w_e/∂θ_j`, propagate the elementary tangent of each op, run the
  numeric replay's tangent → `dQ/dθ_j`. Compare to the Python prototype's analytic
  oracle on a **small cyclic parameterized** graph. (Confirms the REAL executors are
  pure-arithmetic + provenance-complete, which `dr_twotier.py` models but doesn't prove.)
- **Plus #2** (force-convert to `_off` via `ptd_pcg_convert_to_offset` and assert every
  operand resolves to MEM or a bound INPUT edge with a populated `input_specs` (v,e) on the
  NON-load path — amendment 1) and **#3** (assert `NEW_ADD` command_index ↔ emitted numeric
  command is 1:1).
- **Plus a forward/reverse dot-product identity `⟨Jv,u⟩ == ⟨v,Jᵀu⟩` (amendment 7)** on the
  SAME C tape — forward-mode shares none of the reverse-interpreter code, so this is what
  actually pins the shipped transpose (stage-1 + stage-2 REPLACE/kill + ordering) and
  catches in-place aliasing bugs a value-only oracle comparison misses.
- **GATE:** forward-mode `dQ/dθ` == oracle AND the dot-product identity holds to machine
  precision → route (i) is sound and forward-mode becomes the reverse-pass oracle. If
  forward-mode fails → stop; the tape is missing provenance and route (ii) is needed.
- Non-shippable, de-risk only. **Mandatory** first build (amendment 9).

### Batch 1 — the reverse θ-adjoint (FIRST MOMENT, continuous, linear, monolithic, no was_dph)
- **Force `_off`** (amendment 1): `ptd_pcg_convert_to_offset` on the gradient path
  regardless of cache state (the raw tape lacks MEM/INPUT provenance).
- Forward-primal + operand-primal snapshot buffers (in
  `ptd_desc_reward_compute_parameterized_off` `:1906` and `ptd_desc_reward_compute`
  `phasic.h:512`), captured INSIDE the locked precompute (`:1944-1946`) to avoid
  racing a concurrent `update_weights` (landmine 7); re-captured per `update_weights`
  (numeric tape freed `:5790-5794`, param tape kept `:5796`).
- Reverse interpreters (per the stage-1 ordering + stage-2 REPLACE/kill tables above)
  beside `phasic.c:10122/10181` (numeric) and `:9599` (param). Guard NaN trap/deficit
  vertices (`:10231-10233`) for cyclic graphs.
- Edge→θ **linear** Jacobian helper mirroring `:5727-5730`.
- **Runtime MPFR gate (amendment 5):** if `expected_waiting_time` fired the MPFR path
  (condition>1e12, `:10060-10112`), return an FD gradient — a double-tape adjoint would
  mismatch the MPFR primal.
- **Scope (amendment 4): the FIRST moment only** (single θ-independent-seed replay) —
  continuous, `weight_mode==linear`, monolithic (**refuse hierarchical**
  `PHASIC_HIERAR_ELIMINATION`), `was_dph==false`. Higher moments → Batch 3.
- **GATE:** `dθ` matches the Batch-0 forward-mode oracle + the dot-product identity + a
  scale-matched central-difference, on cyclic parameterized fixtures across the regime
  grid — beating FD, with the conditioning-floor regime documented (not required to beat
  FD there, since no float64 method resolves it).

### Batch 2 — binding + `defvjp` replacement (first SVGD-facing target)
- New gradient FFI handler (e.g. `ComputeMomentsGradFfiImpl` / `...SojournGrad...`)
  returning `dθ`, + a `ffi_wrappers.py` wrapper (no gradient handler exists today —
  all are primal-only, factories `graph_builder_ffi.cpp:2177-2239`).
- Wire as the `defvjp` bwd, replacing the `eps=1e-7` FD loop. **Target = `moments_from_graph`
  (`__init__.py:6799`) (amendment 8):** a CLEAN full-FD replacement (pure tape, continuous,
  no was_dph). `pmf_and_moments_from_graph` (`:7414-7465`) is separable
  (`theta_bar=grad_pmf+grad_moments`, `:7459`) so the moments half can be swapped, BUT its
  FD loop still runs the pmf forward at θ±eps (`:7438-7439`) → no forward-pass saving; do it
  after. The pure-sojourn `pmf_from_graph_joint_index` `:7878` is DISCRETE (was_dph) +
  reverse-over-reverse → Batch 3. Batch-1 is first-moment only, which is exactly what the
  pin differentiates (`moments[0]`), so it flips the pin without the seed chain.
- Opt-in flag + FD fallback.
- **GATE:** the mixed-scale pin XPASSes for the wired quantity (except the
  conditioning point); forward parity bit-identical; existing gates green
  (`test_gate_trace_ffi_equivalence`, moments/joint-index tests).

### Batch 3 — extend coverage (each its own de-risk)
- **Higher moments (amendment 4):** `nr_moments≥2` chains replays whose SEED is the
  previous replay's output vector — θ-dependent (`graph_builder.cpp:512-545`). The reverse
  naturally yields the seed-adjoint (`adjoint[v]`); hand it back as replay k−1's
  output-cotangent. De-risk the chain against central-diff before claiming general moments.
- **was_dph** (landmine 1): `update_weights` normalises edges AFTER `coeff·θ`
  (`:5764-5781`), so `∂p_e/∂θ_j` gains a quotient coupling SIBLING edges. De-risk on
  a discrete fixture (adjoint vs central-diff) before wiring joint-index/discrete.
- **log/formula modes** (landmine 2): log Jacobian `w_e/θ_j`; formula is a separate
  bytecode tape (`ptd_weight_tape`, `phasic.c:5132`) needing its own small adjoint.
- **joint-index (structurally distinct, not just a nuance):** its forward IS the
  transpose walk (`:10243`), so the θ-adjoint differentiates a transpose
  (reverse-over-reverse) — a separate derivation from the Batch-1 forward-replay reverse,
  though still straight-line-linear (tractable). Needs its own de-risk.
- **hierarchical SCC** (landmine 3): reverse through the θ-dependent phantom-weight
  stitching (`scc_compose.c:164-257`) — or keep refusing it and document.
- **MPFR** (landmine 5): the cond>1e12 auto-switch (`:10060-10112`) — keep FD
  fallback there (a double adjoint won't cover it).

### Batch 4 — adversarial review of the C diff (sized to complexity).

## Verification / gates
- **Oracles:** Batch-0 forward-mode (on the real C tape) + the Python prototype
  reverse-adjoint + scale-matched central-difference, on cyclic parameterized
  fixtures across a regime grid (uniform / moderate-mixed / extreme).
- **Forward parity is sacred** — the primal must stay bit-identical; the adjoint is
  additive and must not perturb it (diff the forward before/after).
- **The mixed-scale pin** (`test_fd_gradient_mixed_scale.py`) XPASSes for each wired
  quantity; strengthen it with an exact-AD comparison + a documented
  conditioning-floor regime (pinned, not xfail-on-the-fix).
- Reuse `test_gate_trace_ffi_equivalence.py`, `test_sojourn_subset_adjoint.py`.

## Risks / non-negotiables
- **Snapshot timing** (landmine 7): capture primals inside the locked precompute.
- **_off/base sync** (landmine 4): build against `_off`; keep the base executor a
  primal-only fallback.
- **Scope discipline:** continuous + linear + monolithic + no-was_dph FIRST;
  everything else is Batch 3 with its own de-risk. Do not boil the ocean.
- **Conditioning floor:** document the extreme-mixed sub-dominant regime; the fix is
  not required to resolve what no float64 method can.
- **Additive/opt-in + FD fallback** — reversible, per-quantity; respects the
  don't-break-the-primal and don't-modify-existing-behaviour constraints.

## Open decision for the user
First wired target: **continuous moments** (`pmf_and_moments_from_graph` moments-half
or `moments_from_graph`) — no was_dph/log, maximal SVGD reach — is the recommended
Batch-2 target; joint-index/discrete (was_dph) and log/formula follow in Batch 3.
