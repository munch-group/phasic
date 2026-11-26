# Feasibility: `weight_mode='formula'` exact gradient + reverse-tape skeleton refactor

Scoping-only investigation (no code changes). Every claim below is grounded in a
fresh read of the current source (file:line cited) or an actually-executed gate
script, not carried over from CLAUDE.md/memory/atlas without re-verification.
Several claims in CLAUDE.md/memory/the older atlas turned out to be either
stale or in need of a correction — flagged explicitly where found.

## Headline verdict

Adding an exact reverse-mode gradient for `weight_mode='formula'` is
**feasible and well-scoped**, and is architecturally clean: the elimination
tape (`P/PP/INV/OM/DIV/ZERO`, `ptd_pcg_command`) and the weight-formula tape
(`ptd_weight_tape`, opcodes `PUSH_THETA..SELECT`) are **two separate, already
loosely-coupled VMs** — the elimination tape only ever reads an edge's
*current weight* as an opaque free variable (proven already, since `_log` and
`_dph` reuse the same stage-0/1/2 verbatim regardless of how the weight was
computed). Formula-mode's contraction step therefore needs a **new, additive**
autodiff pass over the *formula* tape (not the elimination tape) to produce
per-edge `dw_e/dtheta`, then plugs into the *same* contraction-loop pattern
(`J_out[outk*P+j] += binp[k] * dw_e_dtheta[j]`) the other three already use.
No new op types are needed in the elimination tape; no changes to
`off->input_specs`/tape-input machinery are needed either — it already hands
back `(sp.v, sp.e)` → `struct ptd_edge*`, from which `e->coefficients` /
`e->coefficients_length` (exactly what a formula tape needs as its `c<j>`
inputs) are already available, exactly as linear/dph/log already extract them.

The **reverse-tape skeleton refactor is not a hard technical prerequisite**
(a 4th near-identical ~150-line copy could be written without it, the way
`_dph` and `_log` were), but it is **strongly recommended before formula-mode
lands**, for the same reason CLAUDE.md gives — and this investigation found a
**second, independent reason** beyond CLAUDE.md's original one: the shared
skeleton region is not just a duplication-hygiene concern, it is the
**exact same lines** a separately-planned batch, (a) rewards support, needs to
change (the seed-vector initialization), and — newly found in this pass — a
third batch, (e) MPFR-precision adjoint, would also touch the *same* shared
region (the MPFR-gate call). Doing the extraction now, before formula lands,
means both future batches touch **one** helper instead of **three or four**
near-duplicate call sites — directly reducing the "safety fix applied to only
one sibling, never backported" bug class CLAUDE.md itself documents as having
already happened twice.

The most important cross-batch fact: **this batch's sibling investigations
(`plan-feasibility-hierarchical-scc.md`, `plan-feasibility-svgd-plumbing.md`,
already on disk in this same `atlas/` directory) independently reached the
same conclusion** — the hierarchical/SCC batch (c) explicitly recommends
"sequence the skeleton extraction … before any hierarchical-adjoint
implementation work" and frames formula-mode's extraction as a shared
prerequisite for itself too. All three documents now agree on the same
ordering. See "Cross-batch conflict" below for the one correction this
investigation makes to that shared understanding (the "(a) already fixed"
parenthetical is imprecise).

---

## 1. `weight_formula.py`: the expression language (read in full, 521 lines)

`/Users/kmt/phasic/src/phasic/weight_formula.py` compiles a per-edge formula
string into a flat stack-machine tape: `{ops, consts, n_theta, n_coeff,
stack_depth, src}`.

**Grammar** (recursive-descent, precedence-climbing): comparisons → add/sub →
mul/div → unary → power (right-assoc `**`) → atom (number / `t<i>` /
`c<j>` / function call / parenthesized expr).

**Operations** (23 opcodes, `OPCODES` dict, lines 60-70):
`PUSH_THETA, PUSH_COEFF, PUSH_CONST` (leaves) — `ADD, SUB, MUL, DIV, POW, NEG`
(arithmetic) — `EXP, LOG, SQRT, LOGISTIC` (transcendental, 1-arg) —
`EQ, NE, LT, GT, LE, GE` (comparisons, yield 1.0/0.0) — `AND, OR, NOT`
(boolean on nonzero-is-true) — `SELECT` (branchless ternary,
`(cond!=0)?a:b`). `delta(a,b)` is sugar for `eq`. So: **arithmetic + a fixed
set of transcendentals + boolean/comparison/ternary logic — no loops, no
user-defined functions, no unbounded recursion.** It is not merely
"polynomial combinations of theta and coefficients"; the transcendental and
conditional operators are real and used (confirmed by test fixtures in
`tests/pytest/test_weight_formula_*.py`).

**Intermediate representation:** a flat **stack-based bytecode tape** (list
of ints for opcodes/operand-indices + a deduplicated float constant pool) —
not an AST, not symbolic-expression trees. The tape is built once
(`compile_formula`, called from the `Graph.weight_formula` setter,
`src/phasic/__init__.py:1867-1888`) and evaluated per-edge, per-theta.

**theta-independence guard (critical for differentiability):** the compiler
*statically rejects* (`WeightFormulaError` at assignment time) any formula
where a comparison operand, a `delta`/`and`/`or`/`not` operand, or `select`'s
*condition* references `t<i>`. Only `select`'s two *value* branches may use
theta. This means: **every boolean/comparison/select-condition subtree in a
compiled tape is, by construction, provably theta-independent** — this is the
single most important fact for gradient feasibility (see §2, §5): it means a
differentiation pass never needs sub-gradient handling at a branch/comparison
boundary, because the branch/comparison's *value itself* has zero true
partial derivative w.r.t. theta, and `select`'s runtime-but-theta-invariant
condition means only the *chosen* branch's gradient needs to propagate.

**Evaluator:** `eval_tape` (pure Python, lines 454-521) is the "authoritative
reference semantics" the C executor must match bit-for-bit (confirmed
identical logic in the C executor, §3below) — it is a plain stack-machine
interpreter, no autodiff of any kind exists yet in this file or its C
counterpart.

---

## 2. Does the elimination-tape op set (`P/PP/INV/OM/DIV/ZERO`) generalize to formula? No — they are two separate VMs at two separate levels

Confirmed by direct read (`src/c/phasic.c:3638-3640`, the validator's own
comment): the elimination-tape command types are
**`NEW_ADD=0, P=1, INV=2, PP=3, ONE_MINUS=4, DIVIDE=5, ZERO=6`** (7 types,
not 6 — `NEW_ADD` is a distinct 7th type that registers a *moment-chain edge*
node, separate from the arithmetic ops). Reading the stage-0 forward switch
(identical in all three functions, e.g. `phasic.c:10766-10779`):

- `case 0` (`NEW_ADD`): registers `(from, to, multiplier)` into the parallel
  `na[]/nb[]/nm[]` arrays used by the **separate** forward/reverse *moment
  chain* (the `a_j` seed-propagation loop) — does not touch `mem`/`inv` at all.
- `case 1` (`P`): `*rf += (*rt) * c.multiplier` — scaled accumulate with a
  **compile-time-constant** multiplier baked into the command.
- `case 2` (`INV`): `*rf = 1.0 / (*rf)`.
- `case 3` (`PP`): `*rf += (*rt)*(*rm)` — product-accumulate of two *pointer*
  operands (both resolved via `mem`/`input`).
- `case 4` (`ONE_MINUS`): `*rf = 1.0 - *rf`.
- `case 5` (`DIVIDE`): `*rf = (*rf)/(*rt)`.
- `case 6` (`ZERO`): `*rf = 0.0`.

This is a small, purpose-built VM for **Gaussian-elimination arithmetic over
scratch memory `mem[]`/free-variable inputs `inv[]`** (the graph-reduction
computation), each op reading/writing raw `double*` pointers resolved via
`ptd_pcg_operand.kind` (`MEM`/`INPUT`/`NULL`). It has no notion of "theta" or
"edge coefficients" at all — an edge's current *weight* enters this tape
purely as one of the opaque `off->inputs[k]` free variables (`ni` of them,
looked up via `off->input_specs[k]` → `(sp.v, sp.e)` → `struct ptd_edge*`).

**Formula's tape operates one level up and earlier**: it computes *that very
weight* (`e->weight`) from `(theta, e->coefficients)`, *before* the
elimination tape ever runs. The two tapes never need to share an opcode set,
because they operate on entirely different objects (elimination-tape ops
transform scratch/mem values through the Gaussian-elimination DAG; formula-tape
ops transform `(theta, coeff)` into one scalar per edge). **Targeting
formula's tape "into" the elimination tape's op set is not the right framing
at all** — no new op types (`P/PP/INV/OM/DIV/ZERO`) are needed, and none of
formula's 23 opcodes (arithmetic/transcendental/boolean/select) map onto that
7-type set in any useful way. What formula-mode genuinely needs is a **new,
independent autodiff pass over its own existing tape/VM** (`ptd_weight_tape`,
§3) — this is new code, not a re-target of old code, but it is a
well-precedented *pattern* already used three times in this codebase: snapshot
needed forward operands in a forward pass (as `s0[i]/s1[i]` already do for
`PP`/`DIVIDE` at `phasic.c:10772,10775`), then replay a reverse/adjoint pass
using those snapshots (`phasic.c:10840,10843` — the existing `PP`/`DIVIDE`
reverse cases are literally the product-rule/quotient-rule adjoint already,
just for the *elimination* tape's arithmetic). The same idiom transfers
directly to a new pass over formula's 12 non-boolean opcodes
(`PUSH_THETA/PUSH_COEFF/PUSH_CONST/ADD/SUB/MUL/DIV/POW/NEG/EXP/LOG/SQRT/LOGISTIC`)
— see §5 for the concrete differentiation rules.

---

## 3. The C++ `wf_*` compiler: confirmed genuinely dead — but formula-mode itself is very much alive (correction to the older atlas's framing)

Grep across `src/cpp/phasic_pybind.cpp`, `src/cpp/phasiccpp.cpp`, and
`src/cpp/parameterized/graph_builder.cpp` gives a precise, three-part picture:

1. **`src/cpp/phasiccpp.cpp:720-1122`** contains a full, independent C++
   reimplementation of `weight_formula.py`'s grammar: `wf_tokenize`,
   `struct WFParser`, `wf_enforce_guard`, `wf_emit` → compiles into a
   `ptd_weight_tape` via `Graph::weight_formula(const std::string&)`
   (`phasiccpp.cpp:1094`, declared `api/cpp/phasiccpp.h:1042`).
2. **This C++ method has zero pybind bindings** — confirmed by grep across
   `phasic_pybind.cpp` for `.def("weight_formula"` (or any registration of
   `&phasic::Graph::weight_formula`): none found. It is unreachable from
   Python. **This much of the older atlas's claim is correct and reconfirmed.**
3. **But `weight_mode='formula'` itself is fully live, wired, and tested** —
   via a *different*, parallel path that bypasses the C++ compiler entirely:
   - Python's `Graph.weight_formula` setter (`src/phasic/__init__.py:1867-1888`)
     calls **Python's own** `compile_formula` (`weight_formula.py`), producing
     the `{ops, consts, ...}` tape dict.
   - That dict is shipped straight into C via the pybind method
     **`_set_weight_tape`** (`src/cpp/phasic_pybind.cpp:1461-1478`), which
     calls `ptd_weight_tape_create(...)` + `ptd_graph_set_weight_tape(...)` —
     no C++ parsing involved at all.
   - The FFI/`GraphBuilder` path (JAX-facing) gets the *same* pre-compiled
     `ops`/`consts` arrays via `serialize()`'s JSON (`_graph_serialize.py:239-244,
     708-712`, key `weight_formula_tape`) → `graph_builder.cpp:43-55`
     (`tj.at("ops")`/`tj.at("consts")`) → same `ptd_weight_tape_create` call
     (`graph_builder.cpp:366-374`). Also never touches the C++ WFParser.
   - Both paths evaluate the *same* C executor, `ptd_weight_tape_eval_arrays`
     / `ptd_weight_tape_eval` (`phasic.c:5117-5235`), invoked from
     `ptd_graph_update_weights`'s per-edge loop (`phasic.c:5636-5694`) and from
     `GraphBuilder::compute_weight` (`graph_builder.cpp:459-470`) for IPV/build
     -time weights.
   - Mature test coverage confirms this is production, not experimental:
     `tests/pytest/test_weight_formula_{svgd,residual,theta_dim,kwarg,parser,
     cpath,daisy}.py` + `test_gate_weight_formula_conformance.py`.

**Conclusion for this batch:** the C++ `wf_*`/`WFParser` compiler is a
genuinely separate, unreachable dead end (duplicate implementation, safe to
ignore) — but this must not be read as "formula mode is unreachable." The
*correct* target for any new gradient work is the **live C executor**,
`ptd_weight_tape`/`ptd_weight_tape_eval_arrays` and its opcode enum
`ptd_wf_opcode` (`phasic.c:5085-5093`, `PTD_WF_PUSH_THETA=0 .. PTD_WF_SELECT=22`
— confirmed to mirror `weight_formula.OPCODES` exactly, integer-for-integer).
A useful bonus discovered in this same region: `ptd_weight_tape_specialize`
(`phasic.c:5341-5477`) already builds a **per-edge, theta-only "residual"
tape** (constant-folds every theta-independent subexpression, prunes
untaken `select()` arms, using the *same* provable-theta-independence
invariant from §1) as a forward-eval speed optimization
(`ptd_graph_build_wf_residuals`, `phasic.c:5496-5532`). This residual
mechanism is a **candidate future speed optimization** for a gradient pass
too (it already contains no comparisons/booleans/select once specialized —
only arithmetic/transcendental ops), but is **not required for a correct
first implementation** — see §5, which recommends differentiating the *full*
tape directly using `e->coefficients` (already available at every contraction
call site in all three existing functions), deferring the residual-tape
optimization as a follow-up.

---

## 4. Reverse-tape skeleton: what's actually identical vs different (read all three functions in full, side by side)

Read in full: `ptd_moments_grad_theta` (`phasic.c:10738-10881`),
`ptd_moments_grad_theta_log` (`10917-11063`),
`ptd_moments_grad_theta_dph` (`11142-11338`). Line numbers below are current
(re-verified against the live file, not carried over from the atlas).

### Genuinely identical (byte-for-byte, modulo variable/free-list bookkeeping)

| Block | linear | log | dph |
|---|---|---|---|
| Stage-0 forward tape walk (`case 0..6` switch building `na/nb/nm`, snapshotting `s0/s1`) | 10766-10779 | 10946-10959 | 11207-11220 |
| MPFR gate call (`ptd_dbg_tape_needs_mpfr`) | 10783-10788 | 10960-10965 | 11221-11227 |
| Forward moment chain (`seeds`/`snaptos` build, `a_0=ones` seed) | 10790-10801 | 10966-10977 | 11228-11239 |
| Per-`outk` reverse chain (`bar_out`/`adj`, the `j!` factor injection) | 10811-10829 | 10986-11004 | 11248-11266 |
| Stage-2 param-tape reverse (`bmem`/`binp`, `case 0..6` reverse switch) | 10830-10847 | 11005-11021 | 11267-11283 |

This confirms the core CLAUDE.md claim: **this ~110-line inner block is
genuinely, exactly duplicated three times** and is the legitimate extraction
target.

### Where the three functions genuinely diverge — more than "only the contraction differs"

This investigation confirms CLAUDE.md's claim understates the divergence in
four concrete ways:

1. **Function signatures differ.** `ptd_moments_grad_theta` takes **no**
   `theta`/`theta_len` parameters at all (`phasic.c:10738-10739`) — its
   contraction only needs `e->coefficients[j]`. `_log` and `_dph` both add
   `const double *theta, size_t theta_len` (`10917-10918`, `11142-11143`)
   because their contraction math needs concrete theta values (`e->weight /
   theta[j]` for log; `Sv`/`SigmaCv` built from `theta` for dph). A shared
   helper must accept `theta` unconditionally (linear simply ignores it), or
   the "shared skeleton" needs two overloads/an optional pointer.
2. **`ptd_moments_grad_theta_dph` has an entire extra PRE-pass** not present
   in linear/log at all: the `Sv`/`SigmaCv` precompute + "mixed constant +
   parameterized sibling" decline (`11149-11179`, ~30 lines), executed
   *before* the tape is even built. This is graph-topology + theta dependent,
   one-time-per-call work, not part of the "contraction" step in the sense of
   the per-`outk` loop.
3. **`ptd_moments_grad_theta_dph` has an entire extra POST-pass**: after the
   per-`outk` contraction loop and its own isfinite sweep, a **second**
   isfinite sweep bookends a call to `ptd_dph_correct_discrete_moment_grad`
   (`phasic.c:11094-11119`, a ~26-line Stirling-number/binomial/factorial
   helper applying the continuous→discrete moment-space correction to every
   output column) — `11326-11328`. Neither linear nor log has any equivalent
   of this second stage at all.
4. **The contraction loop's own *guard conditions* differ, not just its
   math.** Linear only skips `coefficients_length==0` (`10868`). Log *and*
   dph additionally skip the starting-vertex edge (`11046`, `11313`) — a
   guard linear's function does not have. Per the in-code comment at
   `11032-11046`, this guard's necessity is asserted "empirically" and its
   unreachability today rests on `_graph_serialize.py`'s `if False:` around
   `start_param_edges` staying dead — i.e. it is a latent asymmetry, not
   fully proven-equivalent across the three functions.

A related, previously-undocumented observation from this read: **linear's C
function (`ptd_moments_grad_theta`) has no `graph->was_dph` guard of its own
at all** (unlike `_log`, which explicitly declines `was_dph`, `10921`). Its
safety today rests entirely on the **Python-side routing** in
`pmf_and_moments_from_graph` (`__init__.py:6965`, `_effective_discrete` routes
to `_moments_grad_theta_dph` instead whenever discrete) never calling the
plain linear C function on a `was_dph` graph. If a future caller ever invoked
`Graph._moments_grad_theta` directly on a `was_dph` graph (bypassing that
Python gate), it would silently compute the wrong answer (contracting via
plain `e->coefficients[j]` against a post-renormalization `e->weight`,
instead of the quotient rule `_dph` uses). This is a pre-existing gap, not
something this batch needs to fix, but worth carrying into the refactor's
risk list since a shared helper is a natural place to *also* close it (e.g.
an explicit `assert !graph->was_dph` inside the linear-mode contraction
callback) — cheap to add opportunistically, not required.

### Net assessment for Task 4

**Mechanical, but not "just extract and done."** The ~110-line stage-0/1/2
inner block extracts cleanly as a static helper. The interface needs a
deliberate design decision: pass `theta` unconditionally (ignored by
linear's contraction), and represent the varying **contraction step** as a
callback/function-pointer parameter (C doesn't have closures, so this means
either a function-pointer + `void *ctx` struct, or an `enum` dispatched
inside one shared contraction function with a `switch` — the latter is
simpler and arguably more idiomatic for this codebase's existing style, e.g.
`ptd_pdf`'s own `discrete`/`is_discrete` dispatch pattern elsewhere). `_dph`'s
extra pre-pass and post-pass are naturally kept **outside** the shared helper
(called by `_dph` before/after invoking it) — they should not be forced into
the shared core. Verification is exactly as CLAUDE.md suggests: re-run
`dr_moments_jac_gate.py`, `dr_dph_moments_jac_gate.py`,
`dr_log_mode_moments_jac_gate.py` (all three **re-run in this session,
current baseline: ALL PASS**, see §8) as a byte-identical-output check before
and after.

---

## 5. Formula-mode gradient function: concrete design sketch

**Shape:** a 4th function, `ptd_moments_grad_theta_formula`, following the
(ideally refactored) shared stage-0/1/2 skeleton, signature matching `_log`/
`_dph` (`graph, nr_moments, const double *theta, size_t theta_len, double
*J_out`) since its contraction needs concrete theta values.

**What's new — a per-edge `dw_e/dtheta` autodiff pass over `ptd_weight_tape`,
not over the elimination tape.** Concretely:

1. **New forward+reverse pass over the formula tape** (new C, mirrors the
   existing snapshot-then-replay idiom already used for `PP`/`DIVIDE` at
   `10772/10840` and `10775/10843`): given `(ops, consts, theta, e->coefficients)`,
   run the tape once recording intermediate stack values (as `eval_tape`/
   `ptd_weight_tape_eval_arrays` already do for the *value*), then run a
   single reverse (adjoint) pass accumulating `d(final)/d(each PUSH_THETA
   leaf)` into a length-`P` output vector. Differentiation rules needed (12
   "real" opcodes; the other 11 need only the rule below):
   - `ADD`/`SUB`: adjoint passes through (±1) to both operands.
   - `MUL`: product rule, `d(ab)=b·da+a·db` (needs the *other* operand's
     forward value snapshotted, exactly like `PP`'s `s0/s1`).
   - `DIV`: quotient rule (needs both forward values snapshotted, exactly
     like the elimination tape's existing `DIVIDE` case).
   - `POW(a,b)`: general rule `d(a^b) = a^b·(b/a·da + ln(a)·db)` — needed in
     general because the grammar allows a theta-dependent exponent
     (`t0 ** t1` is syntactically legal, `_power`'s exponent is a full
     `_unary()`/atom, which can itself be `t<i>` or a parenthesized
     sub-expr) — cannot assume the exponent is theta-independent.
   - `NEG`: `d(-a) = -da`.
   - `EXP`/`LOG`/`SQRT`/`LOGISTIC`: standard 1-arg chain rules
     (`exp(a)·da`, `da/a`, `da/(2·sqrt(a))`, `logistic(a)(1-logistic(a))·da`).
   - `EQ/NE/LT/GT/LE/GE/AND/OR/NOT`: **zero gradient, and — critically —
     no further adjoint needs to propagate into their operand subtrees at
     all**, because the compiler's theta-independence guard (§1) already
     guarantees no `PUSH_THETA` ever appears beneath them. (A defensive
     implementation could instead propagate a zero-adjoint downward and let
     it naturally produce nothing; either is correct, the "stop early"
     version is just cheaper.)
   - `SELECT`: condition is theta-independent by the same guard, so its
     *value* is a plain runtime fact (0/1) with no differentiability
     ambiguity; the chosen branch's adjoint passes straight through, the
     unchosen branch's is dropped — a standard, well-defined
     branchless-ternary-with-invariant-condition derivative, not a
     subgradient/relaxation concern.
2. **Where this runs:** *once per tape-input edge*, **before** the
   per-`outk` loop (mirroring `_dph`'s `Sv`/`SigmaCv` one-time precompute
   pattern, `11149-11179`) — `dw_e/dtheta` does not depend on `outk`, so
   computing it `K` times inside the loop would be wasted work. Store the
   result as an `ni × P` (or sparse, per-edge nonzero-index list) array,
   indexed by the same `k` the contraction loop already iterates.
3. **Contraction step (inside the `outk` loop, same pattern as the other
   three):** `for (k=0;k<ni;++k) { ...resolve e via off->input_specs[k]...;
   if (coefficients_length==0 || starting_vertex) continue; for(j<P)
   J_out[outk*P+j] += binp[k] * dw_dtheta[k][j]; }` — **identical shape** to
   linear's/dph's non-was_dph contraction, just substituting the new
   per-edge derivative vector for `e->coefficients[j]`.

**Does `off->input_specs`/tape-input machinery generalize, or does it need
its own indexing scheme?** **It generalizes cleanly, no new scheme needed.**
`off->input_specs[k]` already resolves to `(sp.v, sp.e)` →
`graph->vertices[sp.v]->edges[sp.e]` → a real `struct ptd_edge*`, from which
`e->coefficients`/`e->coefficients_length` are directly available — exactly
the `(theta, coeff)` pair `ptd_weight_tape_eval_arrays` already consumes at
forward-eval time (`phasic.c:5117-5124`). No new lookup table, no new
per-edge index, no coupling to the *separate* `wf_residuals[]`
array/ordering (which uses a different iteration order — "skip start vertex,
skip zero-coeff edges, in vertex-then-edge order," `phasic.c:5500-5523` —
that does **not** line up with the tape's own `input_specs` order and would
require a nontrivial reverse-mapping to use directly). **Recommendation:
differentiate the full `graph->weight_tape` per edge using `e->coefficients`
directly for the first implementation; treat consulting the pre-specialized,
smaller `wf_residuals[]` tapes as a follow-up speed optimization**, not a
correctness requirement — this avoids introducing a second indexing scheme
in the first pass.

**Decline conditions specific to formula mode (design, not yet implemented):**
mirrors `_log`: `graph->was_dph` should almost certainly be excluded (same
"not guaranteed to fail elsewhere" caution `_log`'s own comment gives for
`discretize()`+`log`, `10902-10913` — needs its own direct repro check before
being asserted, not assumed by analogy). The MPFR gate is inherited unchanged
from the shared skeleton (orthogonal — it's about the *elimination* tape's
conditioning, not the formula tape's own numerics). A **new** failure mode
formula introduces that linear/log/dph don't have: the formula tape's own
evaluation can itself produce non-finite values (e.g. `log` of a non-positive
value, `POW` with an invalid base/exponent combination, `DIV` by zero) —
these are **not new decline logic to write**, they fall through naturally
into the *existing* final `isfinite` sweep over `J_out` (the same pattern
`_log`/`_dph`/linear all already use) as long as the new contraction step's
output flows into `J_out` unconditionally — no special-casing required, just
confirm the new code doesn't accidentally return early on its own before that
sweep runs.

---

## 6. Cross-batch conflict check

The task named five other in-flight batches, (a)-(e). Two of the five
already have their own feasibility documents on disk in this same directory
(`plan-feasibility-hierarchical-scc.md` = batch (c),
`plan-feasibility-svgd-plumbing.md` covers pieces of (a)/(b)/(c)/(d)/(e) as
cross-checks) — both were read and cross-referenced rather than re-derived
from scratch.

### (a) Rewards support for the moments adjoint — direct, confirmed conflict

**Correction to a shared imprecision across CLAUDE.md and the sibling
hierarchical-scc plan doc:** both describe commit `315ce9c8` as having
"fixed" the rewards issue. Reading that commit directly (`git show
315ce9c8`) confirms its actual fix is a **Python-side decline-to-FD guard**
(`_rewards_provided` forces `_exact_grad_enabled=False` whenever `rewards is
not None`, `__init__.py` — the exact C path is *never invoked* when rewards
are present) — **not** an implementation of reward-weighted gradients inside
`ptd_moments_grad_theta`/`_dph`/`_log`. The commit message itself says so
explicitly: *"Proper reward support for the exact path … is a follow-up, not
attempted here."* So **"(a) rewards support" is a real, unimplemented, future
batch**, exactly as this task's framing assumed — not something already done.

Given that, the conflict is real and direct: standard (reward=all-ones)
moments seed the forward moment chain with `for (v<n) seeds[v]=1.0;` — this
exact line is byte-identical across all three functions (`10792`, `10968`,
`11230`, part of the "genuinely identical" block in §4). Reward support would
change this to `seeds[v] = rewards ? rewards[v] : 1.0;` (or equivalent) in
**all three places**. This is precisely the line a skeleton-refactor would
extract into the shared helper.

**Sequencing recommendation:** skeleton-refactor **before** rewards-support.
If the refactor lands first, rewards-support touches **one** helper's seed
init instead of three near-duplicate call sites — strictly lower risk of the
exact "fix applied to only one sibling" bug class CLAUDE.md's own
cross-cutting observations document as having already happened twice (the
`coefficients_length==0` guard, and the MPFR gate/guard set on
`ptd_moment0_grad_theta`). If rewards-support lands first instead, it is not
blocked or broken by the refactor happening later — the refactor would just
need to additionally verify the (by-then triply-duplicated) reward-seed
logic extracts identically across all three, a strictly bigger diff to
review than doing it once, up front.

### (e) weight_mode='callback' + MPFR-precision adjoint — a second, newly-found overlap with the shared region

Not previously connected to the skeleton-refactor question by either sibling
document (the svgd-plumbing doc found (e) has "no direct conflict" with
*its own* scope, which is a different question). If batch (e)'s
"MPFR-precision adjoint" work means replacing today's binary MPFR-gate
decline (`ptd_dbg_tape_needs_mpfr(nm, nc)` → `-1`, `10783-10788` /
`10960-10965` / `11221-11227`, identical across all three, part of the
"genuinely identical" block in §4) with an actual higher-precision adjoint
computation, **that also touches the exact same shared-skeleton region** the
refactor would extract — a third consumer of the same lines, strengthening
(not just repeating) the case for extracting before either (a) or (e) lands.
This was not explicitly asked about by the task but falls directly out of
reading the shared-block boundaries carefully for Task 4.

### (b) `Graph.svgd()` plumbing + joint-index baked-mode

No conflict. Confirmed by `plan-feasibility-svgd-plumbing.md`'s own explicit
cross-check: "Batches (b) formula-mode/reverse-tape refactor, (c)
hierarchical/SCC tape compatibility, and (e) callback+MPFR conditioning-floor
were checked against every leaf/fix in this document and found not to
intersect." The joint-index gradient is a structurally separate C function
(`ptd_sojourn_grad_theta_subset`, forward-mode, not part of the
linear/log/dph reverse-mode trio at all).

### (c) hierarchical/SCC tape compatibility

No conflict, but a **shared prerequisite**, confirmed directly in
`plan-feasibility-hierarchical-scc.md` §6: it recommends the *same*
skeleton-extraction happen first, framing its own per-SCC adjoint as "not a
4th copy but a structurally different consumer of the same core," and
explicitly warns against scoping the hierarchical batch before the
extraction lands (risk of a 4th near-duplicate that then has to be unwound
alongside formula's). Both this document and that one now agree on the same
ordering independently.

### (d) PMF/PDF gradient re-derivation + daisy-chain

No conflict — confirmed by direct read: `ptd_graph_pdf_with_gradient`
(`phasic.c:13090`, declared `api/c/phasic.h:1467`) is a wholly separate,
currently-unwired uniformization-based gradient, mechanically unrelated to
the Gaussian-elimination-tape/moment-chain approach the three moments
functions share. No line-level or mechanism-level overlap found.

---

## 7. Risk/unknowns list

1. **`POW`'s general two-sided differentiation rule is more complex than the
   other opcodes** and is exercised only when a formula actually puts theta
   in an exponent (`t0**t1` style) — worth a dedicated de-risk script
   (`jax.jacobian` of the same expression) before trusting it, exactly the
   style of verification every existing B3 batch has used.
2. **`was_dph` exclusion for formula mode is asserted by analogy to `_log`'s
   own caution, not yet independently confirmed by a direct repro** the way
   `_log`'s own exclusion was (`b3-log-weight-mode-plan.md` D1). Needs its
   own repro before being assumed safe or unsafe.
3. **Whether to build on the full tape or the specialized `wf_residuals[]`
   is an open design choice** — this document recommends the full tape for
   a first, correctness-first implementation (no new indexing scheme needed)
   and defers the residual-tape speed optimization; a future pass could
   revisit if formula-mode gradient cost becomes a measured bottleneck.
4. **No existing FD-defect-at-mixed-scale gate exists yet for formula mode**
   — `test_weight_formula_svgd.py` (grepped, zero mentions of
   `exact_moment_grad`/gradient-quality concerns) currently exercises only
   FD gradients for formula-mode SVGD. This means the *motivating* problem
   (FD is unreliable at mixed parameter scales — `project_fd_gradient_b3`)
   is currently unverified/unquantified specifically for formula mode; a
   concrete repro (mirroring `dr_log_mode_moments_jac_gate.py`'s
   mixed-scale cases) would strengthen the case for prioritizing this work
   and should be an early step of any implementation batch, not an
   afterthought.
5. **The refactor's own interface design (callback vs. switch-dispatched
   contraction) is not yet decided** — this document recommends an
   `enum`-dispatched single contraction function (simpler, no C
   function-pointer/closure plumbing) but this is a design call for
   whoever implements it, not resolved here.
6. **Linear's missing `was_dph` guard (§4) is a latent, pre-existing gap**
   unrelated to this batch's scope, surfaced as a side effect of reading all
   three functions in full — worth a one-line mention in any future
   refactor's PR description even though fixing it isn't this batch's job.

---

## 8. Recommended sequence position

**Skeleton-refactor first, then formula-mode.** Concretely:

1. Extract the shared stage-0/1/2 core (§4) as a standalone, independently
   verifiable step — gate with `dr_moments_jac_gate.py` +
   `dr_dph_moments_jac_gate.py` + `dr_log_mode_moments_jac_gate.py`
   byte-identical-output re-runs (all three **re-run fresh in this session,
   current state: ALL PASS** — a clean, confirmed baseline to diff against).
2. Land formula-mode's new `ptd_weight_tape` autodiff pass (§5) as its own
   batch, against the refactored core, with its own new gate (mirroring
   `dr_log_mode_moments_jac_gate.py`'s structure: closed-form/CD cross-checks,
   mixed-scale cases, a `was_dph`-decline repro, an MPFR-decline repro).
3. This ordering directly de-risks two *separately planned* future batches —
   (a) rewards-support and (e) MPFR-precision-adjoint — both of which this
   investigation confirms would otherwise touch the same shared lines a
   second and third time. It also matches (independently, per
   `plan-feasibility-hierarchical-scc.md` §6-7) what batch (c)'s own
   feasibility pass already concluded it needs.
4. No urgency signal was found suggesting this must happen before the other
   batches for user-facing-breakage reasons (formula-mode's FD gradient
   works today, just with the same unquantified mixed-scale risk every
   other B3-motivated path had before its own exact adjoint shipped) — the
   priority argument here is entirely about avoiding a second/third/fourth
   duplicated-skeleton copy, not about fixing a currently-broken user path.

---

## 9. Gate baseline (executed this session, not assumed)

All three existing exact-gradient gates were re-run fresh against the
current `master` working tree (no code changes made):

```
pixi run python experiments/dr_moments_jac_gate.py          -> ALL PASS
pixi run python experiments/dr_dph_moments_jac_gate.py       -> ALL PASS
pixi run python experiments/dr_log_mode_moments_jac_gate.py  -> ALL PASS
```

This is the clean baseline any future skeleton-refactor or formula-mode
implementation batch should diff against for regressions.
