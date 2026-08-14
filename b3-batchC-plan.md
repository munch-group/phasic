# Batch C — `weight_mode='callback'` exact moment gradient, Job A only (plan v1)

**Master plan:** §5 (signed off 2026-08-11). **Feasibility:**
`atlas/plan-feasibility-callback-and-conditioning-floor.md` Job A (read
in full; Job B = Deferred 4, explicitly NOT this batch). **Process:**
`b3-execution-process.md`. **Base:** master `769fe3a1` (Batch B
close-out; eighth ledger stamp 1978/0/84/24). **Branch/worktree:**
`b3/batchC-callback` in `../phasic-batchC`. **Sequencing:** the
strictly-serial core rule is satisfied — B is merged; C builds on the
4-kind core and adds the 5th consumer.

## 0. What changed since the feasibility doc (it predates Batches 0/A/B)

1. **Batch 0**: the shared core exists; its header comment pre-declares
   "Batch C: its pre-contraction exit" (`src/c/phasic.c` core comment
   block). The feasibility doc's option (b) — return the
   pre-contraction adjoint, never mutate `edge->coefficients[]` — is
   the master-plan-recommended design and this plan adopts it.
2. **Batch A**: the core's reward hooks scale the moment chain that
   PRODUCES `binp`, so the exit inherits 1-D-rewards correctness
   automatically — gated, never assumed.
3. **Batch B**: (i) the core has FOUR kinds — C adds `PTD_B3_BINP_EXIT`
   as the 5th; (ii) the ALIGNED-theta-dimension lesson TRANSFERS:
   callback mode also resolves the model's theta dimension at inference
   time, decoupled from C `param_length` — the same lazy-decoupled
   crash class exists (clone `update_weights(t, callback=...)` should
   length-check t; GROUNDING ITEM + de-risk D-C4) and the same static
   Python gate pattern applies; (iii) B's gate/test disciplines
   (twin-oracle, fresh rewards-bearing goldens, live-filter + spy,
   measured actuals) are the template.

## 1. Scope

- **In:** continuous `weight_mode='callback'` graphs whose callback is
  **JAX-native** (probed at model construction), on
  `pmf_and_moments_from_graph` (and its svgd moments leaves via the
  existing forwarding — no new svgd_config rules). 1-D rewards.
  MPFR/per-theta declines → NaN→FD as elsewhere.
- **Out (static declines, INFO-logged):** non-JAX-native callbacks
  (PERMANENT — the feasibility doc's option 2, adopted as default; the
  analytic-derivative-callback API of option 1 is a ledgered follow-up,
  NOT built); discrete/was_dph × callback; the lazy-decoupled
  theta-dimension class (B's predicate, message adapted); 2-D rewards
  on the 1-D leaf (unchanged).
- **Explicitly NOT bundled** (master §5): the joint-index callback path
  (`ptd_sojourn_grad_theta_subset` needs its OWN exit — follow-up
  sub-item); Job B (MPFR floor, = Deferred 4).

## 2. New C surface (all additive)

1. **`PTD_B3_BINP_EXIT` enum value + ctx** `struct ptd_b3_binp_ctx {
   double *binp_out; /* K x ni, caller-allocated */ }`: the
   "contraction" case is a COPY — `ctx->binp_out[outk*ni + k] =
   binp[k];` (no skip guards: the exit returns EVERY tape input's
   adjoint; the skip logic lives with the consumer that knows the
   weight rule — Python here). The core's final isfinite sweep runs
   over `J_out`, which stays zeroed for this kind → the finiteness
   contract MOVES TO THE CALLER for the exit kind (deliberate,
   documented at the case; Python checks the matrix and NaN→FDs).
   DESIGN NOTE (third-consumer, master §5): Deferred-1's per-SCC VJP
   needs this same exit PLUS caller-supplied cotangent seeding; the
   seeding block remains the comment-marked orthogonal section (Batch-0
   decision re-affirmed) — this exit composes with a future seed
   parameter untouched, and its K×ni shape is exactly the
   pre-contraction surface Deferred-1's plan names. Validated, not
   declined blind.
2. **`ptd_moments_binp_exit(graph, nr_moments, const double *rewards,
   size_t rewards_len, double *binp_out /* K*ni */)`** — thin wrapper
   (owns ptape/off; same decline set shape as the siblings:
   `!parameterized`, `nr_moments < 1`, `was_dph`). NOTE: no theta
   parameter — stages 0-2 replay the CURRENT edge weights (the caller
   must `update_weights(theta, callback=fn)` first, the same
   current-weight-state contract the log/formula wrappers have —
   documented loudly, the A-G4 lesson).
3. **Tape-input identity export**: the Python contraction needs, per
   tape input k, WHICH edge (to evaluate ∂f/∂θ at that edge's
   coefficients). C++/pybind wrapper `_moments_binp_exit(nr_moments,
   rewards=[])` returns a tuple `(binp_flat /* K*ni */, ve_pairs /*
   ni×2: (vertex_idx, edge_idx) */)` — the (v,e) pairs read from
   `off->input_specs` in the SAME k order as binp columns. A second
   pybind helper (or the same call) exposes per-input
   `coefficients` vectors directly (ragged list) so Python needs no
   (v,e)→coefficients re-derivation — GROUNDING ITEM: pick whichever
   the C++ layer can produce cleanly; de-risk D-C3 verifies the
   identity/order contract either way.
4. **CRLF discipline** as always (phasic.c/phasiccpp.h CRLF; phasic.h/
   pybind LF; binary replaces, assert count==1).

## 3. Python changes (`src/phasic/__init__.py`)

- Scope predicate: `_callback_scope_ok = (_wm == 'callback' and not
  _effective_discrete and param_length == int(graph.param_length())
  and _cb_jax_native)`. `_cb_jax_native` = construction-time probe:
  `jax.grad(weight_callback, argnums=0)(jnp.ones(P), jnp.asarray(c0))`
  inside try/except on a representative coefficient vector (c0 from
  the graph; GROUNDING: which attribute holds the callback —
  `graph.weight_callback`? — and what the probe's failure modes are:
  TracerArrayConversionError, ConcretizationTypeError, TypeError...
  catch broad, log the exception class). Probe failure → static INFO
  decline naming non-JAX-native as the cause (permanent, the honest
  scope boundary).
- Static declines (each its own truthful INFO message, the B pattern):
  discrete×callback; lazy-decoupled theta-dim; non-JAX-native.
- Callback arm in `_one(t)` (inside the existing try/except):
  1. `_exact_graph.update_weights(t, callback=weight_callback)`
     (existing API, zero new code — feasibility Q3);
  2. `binp, coeffs_list = <the new exit binding>` (+ rewards threaded);
  3. `W[k] = _cb_grad_jit(t_jnp, coeffs_k)` — the per-edge ∂w/∂θ rows,
     via a construction-time `jax.jit(jax.grad(f, argnums=0))`
     (re-traces per distinct coefficient SHAPE only; ragged lengths →
     one trace per length class; de-risk D-C2 measures);
  4. `J = binp_matrix @ W` (K×ni @ ni×P) + finiteness check → NaN→FD
     with the per-theta log on non-finite.
- Docstrings/messages: pmf_and_moments coverage + FD-causes; svgd
  docstring static-decline clause; R29 comment ("callback precedent"
  becomes "non-JAX-native-callback precedent" — the third rewrite of
  that sentence, A→B→C); kwarg test-file docstring.

## 4. De-risk experiments (branch, BEFORE implementation; recorded in
   `b3-batchC-findings.md`)

- **D-C1 (the feasibility doc's flagged UNVERIFIED risk):**
  `jax.grad`/`jax.vmap`/`jax.jit(jax.grad)` over toy callbacks BY
  EXECUTION — linear-equivalent `jnp.dot(theta, coeffs)`, nonlinear
  `jnp.exp`/`jnp.log`/ratio forms, coefficient vectors longer than P
  (auxiliary data), non-JAX-native (numpy/scipy) → confirm the failure
  mode the probe must catch, and a callback returning a Python float
  from jnp ops (float() collapse inside the callback itself — is
  jax.grad still possible? NO — pin the probe's behavior).
- **D-C2 W-computation strategy + cost:** ragged vs uniform coeffs;
  jitted-grad loop vs pad+vmap; measure per-call cost at ni∈{10,100}
  vs the FD backward it replaces (2P forward calls). Pick and record.
- **D-C3 identity/order contract:** input_specs k-order ↔ (v,e) ↔
  coefficients — build a branchy graph, verify the exported pairs
  resolve to the right edges (cross-check against a linear-mode
  fixture where binp@c_matrix must reproduce `_moments_grad_theta`
  EXACTLY — this is also micro-gate (b1)'s oracle).
- **D-C4 theta-dim + clone contract (B's D-B6 mirrored):** does
  `update_weights(t, callback=)` length-check t vs param_length
  (probe)? The lazy-decoupled callback class end-to-end (model works
  FD today; clone raises?); canonical `set_param_length` class; the
  callback attribute's clone propagation (feasibility says
  `_propagate_weight_state` covers it — verify by probe).
- **D-C5 discrete×callback:** silently computes or fails loudly
  (the exclusion evidence, never analogy — the log/formula D1
  discipline).

**Re-evaluate at v2** with each verdict before implementation.

## 5. Gates

- **G0**: branch base recorded; delta above the 8th stamp docs-only.
- **G1 micro-gates** (`dr_batchC_i1_gate.py`, dump/check, FRESH pre-C
  goldens incl. rewards-bearing linear/log/formula + dph):
  (a) byte-identity on the four existing kinds (the core is edited
  again — same bar B set);
  (b1) **linear-equivalent callback** (`f = jnp.dot(t, c[:P])`) vs the
  LINEAR exact path on a twin graph — FD-independent oracle, benign
  AND mixed θ=[1,1e-8] scale (the B twin-oracle trick, now composing
  exit+matmul end-to-end);
  (b2) nonlinear callback (exp/ratio) vs FD-of-the-primal, K=1..3;
  (b3) callback with AUXILIARY coefficients (len > P) — the
  documented callback-mode idiom — vs FD-of-primal;
  (c) rewards×callback vs primal-FD + all-ones == rewardless BITWISE;
  (d) decline contract: was_dph, non-parameterized, theta-dim
  mismatch (Python), non-JAX-native (Python), rewards_len;
  (e) the exit's finiteness contract: a callback whose gradient is
  non-finite at a probe theta → Python NaN→FD with the per-theta log
  (fixture from D-C1).
- **G2 targeted map**: the moments-core row (3 jac-gates +
  `dr_batchA_i1_gate.py` + `dr_batchB_i1_gate.py` checks vs fresh
  pre-C goldens + `test_gate_moments_3way.py` +
  `inference/test_jax_integration.py` vs ledger +
  `inference/test_exact_grad_{discrete,rewards,log_weight_mode,
  formula_mode}.py` + `test_fd_gradient_mixed_scale.py`) + the svgd
  config/validation row + callback-mode files
  (`test_weight_callback*.py` / wherever callback tests live —
  GROUNDING: enumerate at v2) + `test_multivariate_correctness.py`.
- **G3** chunked vs the 8th stamp (1978/0/84/24), both amendments.
- **G4** two adversarial diff refuters (mandate: independent numeric
  probes incl. an own-oracle contraction; the exit's K×ni memory
  contract; the moved finiteness responsibility; probe-vs-call
  callback-nativeness consistency; fan-out of the SAME coefficient
  vector across edges).
- **G5** merge review + squash-merge + close-out (9th stamp; tracker;
  master §5 banner + §15 Phase-3 COMPLETE; CLAUDE.md; process map;
  memory; the R29-precedent third rewrite verified everywhere).

## 6. Tests (`tests/pytest/inference/test_exact_grad_callback_mode.py`,
   the B file as template)

1. linear-equivalent callback: twin parity vs linear-exact (tight,
   measured) + central-diff + engagement spy (full-size successes).
2. mixed-scale twin parity (FD-independent).
3. nonlinear callback vs central-diff.
4. auxiliary-coefficients callback (len > P) vs central-diff.
5. rewards engage + all-ones bitwise.
6. non-JAX-native callback → static decline log + FD works (the
   PERMANENT boundary, live-filter).
7. discrete×callback decline (construction-time capture).
8. lazy-decoupled decline + FD works.
9. engagement flip: the generic out-of-scope log must be GONE for a
   JAX-native callback (live filter + spy).
10. vmap/jit composition bitwise.
11. svgd front door (real init; spy success-floor ≥ n_particles).
12. multivariate per-feature engagement + hard pin (measured).
Fate table: expected NO existing-test edits except the kwarg-file
docstring precedent line + any file greping the generic message
(enumerate by grep at v2 — B measured zero).

## 7. Shipped-text list (same-commit)

pmf_and_moments docstring (coverage + causes); svgd docstring;
svgd_config R29 comment (precedent example → non-JAX-native callback);
kwarg test-file docstring; CLAUDE.md at G5 (B3 section: callback
covered for JAX-native; the "callback/hierarchical remain FD-only"
memory line).

## 8. Ledger arithmetic

Expected G3 = 1978 + N_new (≈12, pinned at G1 time; fate breaks: none
expected, verified by grep at v2).

## 9. Risks / open questions carried into review

1. The probe-vs-call consistency gap: the construction probe runs ONE
   coefficient vector; a callback could be JAX-native for c0 but not
   for some other edge's coefficients (data-dependent Python
   branching). The per-call W computation sits inside the try/except →
   NaN→FD, so the failure mode is safe-but-logged; refuters judge
   whether per-edge probing is needed.
2. The K×ni exit matrix can be large (ni = tape inputs ≈ edges); no
   size guard planned (the FD alternative is 2P FULL forwards — the
   exit is strictly cheaper; state in plan, let refuters check).
3. `float()`-collapsing callbacks (jnp ops + float() inside) are NOT
   differentiable — D-C1 pins; the probe must catch them.
4. The exit returns binp for ALL K rows in one call (the outk loop
   fills rows) — sweep/ok-flag interaction when a LATER outk row would
   have declined (conditioning is stage-0/global, so per-outk decline
   cannot differ — verify in review).
5. Multivariate/2-D rewards composition inherited from A/B unchanged
   (per-feature 1-D slices) — cell 12 gates it.

## v2 amendment (2026-08-14, post two-refuter plan review — BINDING over v1 where they conflict)

**Verdicts:** design/math refuter SOUND-WITH-CORRECTIONS; completeness/
process refuter SOUND-WITH-CORRECTIONS. Convergent top finding, folded
as the headline scope DECISION below. Probes referenced here were run
by the refuters against the post-B install (jax 0.7.2).

### A. HEADLINE DECISION — the ALIGNED-theta-dim restriction is DROPPED for callback mode (v1 §0.3(ii)/§1/§3 superseded)

v1 transferred B's lazy-decoupled crash rationale. REFUTED by source
and probe: `update_weights(theta, callback=)` deliberately does NOT
length-check theta (`phasiccpp.cpp:1879-1884` names PSMC-style longer
AND shorter theta as intended; len-4/len-1/len-0 all accepted by
probe) — B's raise-inside-pure_callback class cannot occur. And the
binp exit takes NO theta: P enters only through `jax.grad`'s rows
(len(t)) and the final J shape. The exact path is therefore
mathematically valid on decoupled graphs, and v1's predicate would
have statically declined the DOCUMENTED PRIMARY callback idiom (a
lazily-built aux-coefficients graph locks C param_length to the
coefficient length > model theta_dim — probed). **v2: callback mode
has NO theta-dimension predicate.** Scope = continuous + JAX-native
(§B). De-risk D-C4 is REPURPOSED: verify the decoupled composition
end-to-end (lazy aux-coeff class; canonical set_param_length class;
plain aligned; theta LONGER than param_length) — engagement, not
decline, is the expected outcome on all four. G1(b3)'s fixture may be
lazily built (no set_param_length required); ADD a lazy-decoupled
pytest cell asserting engagement (the anti-B cell).

### B. JAX-nativeness probe = the DEPLOYED transform over ALL edges (design H2)

v1's plain-grad-at-c0 probe is systematically lenient: data-dependent
Python branching on theta PASSES plain grad (concrete tracers) but
RAISES TracerBoolConversionError under the deployed
`jax.jit(jax.grad(f))` at EVERY call → a committed model that NaN→FDs
each step with a retrace attempt (~13 ms) per theta. MANDATED:
construction probe = the jitted transform evaluated over all edges'
actual coefficient vectors. Coefficient lengths are provably UNIFORM
in-scope (serialize() raises on inconsistent lengths,
`_graph_serialize.py:99-110`, and every model build serializes) → one
trace + ~60 µs warm evals; jit tracing is value-independent, so with
fixed shapes construction success ⇒ success at EVERY theta — the
per-call raise class vanishes; try/except stays defense-in-depth.
Scope wording everywhere: "JAX-native UNDER JIT" (no Python control
flow on theta/coeffs; jnp.where/lax.cond fine). Probed failure modes
to catch: TracerArrayConversionError (numpy callback),
ConcretizationTypeError (float() inside), TracerBoolConversionError
(branching). Non-finite gradients RETURN inf (no raise) → the J
finiteness check is the right net (G1(e) fixture valid). D-C1 is
thereby pre-resolved; its cells re-encode as gate/pytest fixtures.

### C. C interface corrected (design H1, L8, L9)

v1's caller-allocated `binp_out[K*ni]` is unimplementable (ni exists
only after the wrapper builds off; the core also unconditionally
writes/sweeps J_out). MANDATED single-call form:
`int ptd_moments_binp_exit(graph, nr_moments, const double *rewards,
size_t rewards_len, double **binp_out /* malloc'd K*ni */,
uint32_t **v_out, uint32_t **e_out, uint8_t **frozen_out /* ni each */,
size_t *ni_out)` — C allocates (NULL-checked), caller frees; the
wrapper allocates the dummy K×param_length J_out internally; decline
set matches siblings INCLUDING `param_length == 0` (L8). The pybind
binding returns ONE tuple `(binp K×ni, v, e, frozen)` from ONE off
build (L9: two-call designs would make tape-build determinism
load-bearing; single call removes the dependency). The per-input
`frozen` flag is computed C-SIDE as `starting_vertex ∨
coefficients_length==0` (design M4: the frozen set is
correctness-load-bearing — an unskipped aux-constant edge is the
wrong-but-finite formula-class hazard, and Python cannot compute it
reliably from serialize() indices since input_specs order ≠ serialize
order and sp.v is the C array index).

### D. Python contraction spec completed (design M4/L7/L10/L11)

In `_one(t)`: rows with `frozen[k]` are EXCLUDED from W evaluation and
contribute 0 (never evaluate the callback's grad on a constant edge's
coeffs — the empty/irrelevant-coeffs row could raise or poison);
W rows for engaged k = `_cb_grad_jit(t_jnp, coeffs_k)` where
`_cb_grad_jit = jax.jit(jax.grad(_captured_callback, argnums=0))` is
built ONCE at model construction from the BUILD-TIME-CAPTURED
`graph.weight_callback` (L10: update_weights(callback=) is one-shot
per call and must be passed the same captured callable — mirroring
the forward's capture at model build); coeffs fetched once at
construction via the (v,e) export → the graph's live edge
coefficients (uniform lengths ⇒ a stacked matrix + optional vmap).
Finiteness check on J AFTER the skip-aware matmul (L7 — matches the
shipped C sweep semantics: a non-finite binp on a FROZEN row lawfully
ignored; on an engaged row it provably reaches J).

### E. Cost framing corrected (design M5)

The exact moments term does NOT replace the FD loop: model_bwd's 2P
forwards still run for the pmf FD term; only grad_moments is swapped
(same economics as linear/log/formula). Justification = mixed-scale
accuracy. D-C2 measures the ADD-ON (exit + W + matmul) at ni≈10/100,
simplified per L11 (uniform lengths, one trace).

### F. Pre-C math validation: DONE at plan review (design M6)

The refuter's probe composed FD-binp (marker-perturbing
update_weights(callback=) per edge) with jax.grad-W on a branchy
K=3/P=2 aux-coeff graph: **1.2e-10 vs the shipped
`_moments_grad_theta`** (linear-equivalent callback, primal parity
exact) and **4.1e-10 vs theta-FD** (nonlinear exp/ratio). The
chain-rule composition is validated BEFORE any C. Fold the probe as
D-C3's pre-implementation arm; post-implementation D-C3 re-verifies
with the real exit (binp@C == _moments_grad_theta on the
linear-equivalent callback — expected ~bitwise).

### G. Fate table + G2 corrected (completeness HIGH-2/M1/M2)

- FATE BREAKS EXIST (v1's "none expected" withdrawn):
  `test_exact_grad_discrete.py::test_no_silent_fallback_logs_on_out_of_scope_weight_mode`
  uses a callback-mode graph as THE out-of-scope example with a
  float()-collapsing fixture — post-C it must be REWORKED (docstring +
  fixture stays non-JAX-native so the DECLINE still fires; pin the NEW
  static-decline message text, which must contain "weight_mode" and
  "finite differences" or the test reworked accordingly);
  `test_exact_grad_formula_mode.py:16` module docstring ("callback
  mode remains out of scope") → §7 shipped-text list.
- G2 gains: `test_weight_formula_svgd.py`,
  `test_weight_formula_residual.py`, `test_weight_formula_theta_dim.py`,
  `test_gate_weight_formula_conformance.py` (the process map's B row
  applies in full — the core is edited); and the REAL callback
  inventory (there is no test_weight_callback*.py):
  `test_callback_svgd_kwarg.py`, `test_beta_prior.py`,
  `test_weight_mode_probe_and_guards.py`, `test_joint_index_callback.py`
  (+ note: `inference/test_cpp_from_callback.py` is a
  construction-callback, excluded deliberately).
- The formula-vs-callback equivalence cells in
  `test_weight_formula_svgd.py:110-167` use float() callbacks →
  non-JAX-native → decline → FD: behavior-preserving (the B §H-style
  expectation, stated; G3 watches).

### H. Blast radius + conventions (completeness M3/M4/LOW)

- mcmc: builds these models at default exact_moment_grad → post-C a
  callback graph under mcmc runs the construction probe (INFO logs;
  gradients inert — zero jax.grad in mcmc.py). NOTE mcmc auto-infers
  theta_dim = param_length() WITHOUT _resolve_inference_theta_dim →
  the probe's P there is the coefficient length; harmless (probe
  correctness doesn't depend on P) — one findings line.
- Declared UNTOUCHED: `_compute_callback_cdf_zero`/`_cdf_zero_fn`
  (forward-only), `pmf_from_graph` (FD custom_vjp, Deferred-3),
  `moments_from_graph`/`method_of_moments` (§16b item 5), the
  joint-index callback path (own follow-up; its docstring/CLAUDE.md
  line 259 stay TRUE and must NOT be edited).
- svgd cell 11 precondition stated: callback mode under svgd requires
  explicit theta_dim=/theta_init= (raises otherwise) — the front-door
  cell passes theta_dim.
- G1(a) goldens enumerated (7): lin/log/formula × {plain, rewards} +
  dph_plain (no dph_rw — refuted-permanent).
- Implementation order I1 (C exit + wrapper, micro-gated) → I2
  (header/C++/pybind) → I3 (Python: probe, dispatch, contraction,
  messages) → I4 (tests + shipped text), committing per item; G5 adds
  main-checkout rebuild, tracker §16b snapshot refresh, worktree
  disposition; G3 amendments' operative content: -rf + preserved
  per-chunk outputs; groups enumerated from split output on disk +
  union==collected + per-group output file before tallies.
- Deferred-1 wording softened: the exit is the same pre-contraction
  CLASS Deferred-1 §4-P2 needs (caller-seeded cotangents replace the
  factorial seed — the seeding block stays the orthogonal
  comment-marked section); shapes to be confirmed at Deferred-1's own
  activation, not promised here.
- Stale tracker duplicate C row: deleted (rides in this commit).

### Remaining pre-implementation de-risks

- **D-C4 (repurposed §A):** decoupled-composition end-to-end probes.
- **D-C5:** discrete×callback exclusion evidence (silently computes
  vs fails loudly — never analogy).
- D-C1/D-C2/D-C3-pre: RESOLVED at plan review (refuter probes,
  recorded above); re-encoded as gate/pytest cells.
