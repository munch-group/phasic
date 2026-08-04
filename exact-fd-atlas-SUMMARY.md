# Exact/FD gradient atlas — master synthesis

**Why this exists.** During the joint-index `lax.cond`/`vmap` redesign, a fix was nearly built on a premise (that `Graph.svgd()` would benefit from it) that turned out false — discovered only by chance, late, after real design work had gone in. That was a symptom of a broader pattern this session: each B3 batch discovered real, sometimes serious codebase structure (caching invariants, JAX control-flow semantics, SVGD wiring) *reactively*, through adversarial review, rather than *before* proposing a design. This document is the fix for that pattern: a complete, current picture of the exact-vs-finite-difference (FD) gradient landscape in `phasic`, built via six parallel research passes (2026-08-04), so future B3 planning starts from ground truth instead of assumption.

**How to use this document.** Read this synthesis first — it's the prioritized, cross-referenced summary. Each finding links to one of seven detailed atlas documents in `/Users/kmt/phasic/atlas/` for full file:line citations and verification methodology. Nothing here is guessed; every claim in the source atlases was read from code or confirmed by execution, and is marked accordingly in the source documents.

**The atlas documents:**
| Doc | Covers |
|---|---|
| [`atlas/exact-fd-atlas-c-functions.md`](atlas/exact-fd-atlas-c-functions.md) | Every `*_grad_theta*` C function: decline reasons, caching, safety guards |
| [`atlas/exact-fd-atlas-python-wiring.md`](atlas/exact-fd-atlas-python-wiring.md) | Every Python model-builder entry point: exact kwargs, exclusions, rewards/fixed_mask handling |
| [`atlas/exact-fd-atlas-svgd-reachability.md`](atlas/exact-fd-atlas-svgd-reachability.md) | `Graph.svgd()`'s full dispatch tree: which exact paths are actually reachable |
| [`atlas/exact-fd-atlas-caching.md`](atlas/exact-fd-atlas-caching.md) | Every cache touching the parameterized elimination tape, field-by-field |
| [`atlas/exact-fd-atlas-jax-semantics.md`](atlas/exact-fd-atlas-jax-semantics.md) | `lax.cond`/`vmap`/`pure_callback`/`custom_vjp` rules, quantified, with a design checklist |
| [`atlas/exact-fd-atlas-loose-ends-memory.md`](atlas/exact-fd-atlas-loose-ends-memory.md) | Every persistent-memory file, cross-referenced against CLAUDE.md |
| [`atlas/exact-fd-atlas-loose-ends-docs.md`](atlas/exact-fd-atlas-loose-ends-docs.md) | All 31 repo-root planning docs, cross-referenced against CLAUDE.md |

---

## 1. The headline finding: the exact-gradient landscape is much less reachable than it looks

`phasic` has invested heavily in exact gradients (4 production C functions, 2 wired Python entry points, extensive adversarial review). But **`Graph.svgd()` — the actual, primary way users run inference — has no top-level knob for exact-vs-FD at all**, and of its five dispatch branches, only **one** reaches any shipped exact path today, incidentally rather than by design:

| SVGD branch | Reaches exact path? | Why / why not |
|---|---|---|
| Daisy-chain (`epoch_starts` set) — now the **default** multi-epoch path | **No** | No exact-gradient implementation exists at all. 100% FD (absolute `eps=1e-7` central difference) — the exact defect class B3 exists to fix, still live, unconditionally, on what's now the default entry point for multi-epoch population-genetics models |
| Joint-index, default (`exposure=None`, the common case) | **No** | `exact_grad` never passed by `Graph.svgd`; baked/dedup mode is *statically* excluded from the exact path regardless of the kwarg |
| Joint-index, `exposure` set | **No** (but not structurally blocked) | `exact_grad` never passed; would work if `Graph.svgd` passed it |
| Non-joint, 2D rewards (multivariate) | **No** | No `exact_moment_grad` param exists on the multivariate wrapper at all; every per-feature call passes real rewards, which dynamically forces FD anyway |
| Non-joint, 1D rewards | **No** | `exact_moment_grad` defaults `True`, but `SVGD` fixes `rewards` for the whole run and passes it every call — the exact path's own (correct) "rewards not supported" guard fires on **every single gradient step** |
| Non-joint, no rewards | **Yes** — the only reachable leaf | `exact_moment_grad=True` default applies, nothing disqualifies it |

**Escape hatch**: `SVGD(model=..., ...)` accepts a pre-built model directly, bypassing `Graph.svgd`'s dispatch — a caller can hand-build `pmf_and_moments_from_graph(..., exact_moment_grad=True)` or `pmf_from_graph_joint_index(..., exact_grad=True, observed_indices=None)` and get the exact path, at the cost of losing `Graph.svgd`'s convenience wiring (tied params, daisy-chain, reward validation).

**Implication**: the "exact_moment_grad defaults to True" story in CLAUDE.md, while accurate at the function level, overstates real-world impact for anyone using `Graph.svgd()` directly — which is most usage. → Full detail: [`svgd-reachability.md`](atlas/exact-fd-atlas-svgd-reachability.md).

---

## 2. Critical / high-severity findings (act on these first)

### 2.1 A real, execution-confirmed crash: `Graph.moments_from_graph` breaks under `vmap` entirely

Not a gradient bug — the **forward pass** raises `RuntimeError: Incorrect output shape` under `jax.vmap(moments_fn)(theta_batch)`. Its `pure_callback` uses `vmap_method='expand_dims'` but the underlying ctypes call assumes 1-D `theta` unconditionally, unlike its three siblings (which all correctly detect and loop over a 2-D batch). Confirmed by direct execution, not just code reading. `Graph.pmf_from_cpp` shares the same underlying helper and is a probable (unconfirmed) sibling bug. → [`jax-semantics.md` §3d–3e](atlas/exact-fd-atlas-jax-semantics.md).

### 2.2 Security-adjacent: the legacy on-disk cache format backing the public GitHub trace registry has no input hardening

`Graph.pull_cache()`/`push_cache()` — a real, network-facing production feature — uses a format (`PTDPRMC1`, rev-1/2) that was **never given the bounds-hardening the newer rev-3 format got**: an unbounded pointer-offset decode (out-of-bounds **write** primitive from a crafted `.bin` file) and an unvalidated command-type byte that aborts the whole process on corrupt input. The security comment justifying rev-3's hardening ("pulled from the community registry") is attached to the wrong file format — the registry's "parent" artifact is always saved in the *unhardened* legacy format. Compounding this: that parent artifact is saved in a format the automatic loader can never read (`PTDPRMC3`-only), so `pull_cache()`'s documented promise silently doesn't hold for parent artifacts (only per-SCC artifacts, which are genuinely rev-3, get auto-consumed). → [`caching.md` §2c, L2–L4`](atlas/exact-fd-atlas-caching.md).

### 2.3 Daisy-chain/epoch SVGD — zero exact-gradient coverage, now the default multi-epoch path, unmentioned in CLAUDE.md

Covered in §1 above; restated here because of severity. This is the single largest, most surprising gap found across all six streams — it is exactly the FD defect class (absolute-eps central difference on mixed-scale parameters) B3 was built to eliminate, on what is now the default entry point for this repo's actual population-genetics (multi-epoch/migration) workloads, and it is completely absent from CLAUDE.md's B3 sections. → [`loose-ends-memory.md` §3.4`](atlas/exact-fd-atlas-loose-ends-memory.md), corroborated independently by [`svgd-reachability.md`](atlas/exact-fd-atlas-svgd-reachability.md).

### 2.4 An unquantified accuracy limit on the exact gradients themselves ("the conditioning floor")

Three independent historical documents (predating this session) found the same thing during B3's original de-risking: at extreme mixed-scale θ, the **sub-dominant** gradient component is corrupted **identically** for exact reverse-mode AD, exact forward-mode AD, and an independent linear-solve oracle — a genuine float64 precision floor in the underlying linear algebra (near-singular sub-generator), not a defect of any one gradient method. FD is far worse in the same regime, so this isn't a reason to prefer FD — but all three documents recommended characterizing and pinning this regime as a documented (not `xfail`) limit in `test_fd_gradient_mixed_scale.py`, and there's no evidence that was ever done. It is absent from CLAUDE.md entirely — meaning the exact gradients shipped as the B3 initiative's flagship result have an unquantified accuracy envelope that nobody has written down. → [`loose-ends-docs.md`, "conditioning floor" section`](atlas/exact-fd-atlas-loose-ends-docs.md).

---

## 3. Medium-severity findings

### 3.1 The `input_specs`-NULL-on-mmap bug (already fixed) has close relatives

The bug fixed earlier this session (`ptd_sojourn_grad_theta_subset` segfaulting when reading a mmap-loaded offset tape whose `input_specs` field is unconditionally `NULL`) is a template, not a one-off:
- The **per-SCC on-disk cache** (`scc_synthetic.c`) shares the exact same construction-path shape (mmap load → `input_specs == NULL`), with **zero guard today** — dormant until the first SCC-level exact-gradient C function is written.
- `graph->parameterized_reward_compute_graph_off` can be **transiently stale** (holding pre-`was_dph` structure) in the window between `set_was_dph(True)` and the next cache-populating call — safe today only because the one reader happens to call the right precompute function first, with no type-level enforcement.
- Ownership (heap vs. mmap-backed) of the tape's `commands`/`mem_base` pointers is invisible outside one internal flag; any future code bypassing the existing destroy function risks a leak or corruption depending on which of three construction paths built the object it's holding.

→ [`caching.md` §5 (L1, L5–L7)`](atlas/exact-fd-atlas-caching.md).

### 3.2 Fix-lineage divergence: safety fixes found by review are never backported to sibling functions

Every fix an adversarial review found this session (NULL-pointer guards, allocation checks, size guards, the MPFR gate) was applied only to the specific function under review, never to its near-identical siblings:
- **None of the three moment-gradient C functions** (`ptd_moments_grad_theta`/`_dph`/`_log`) have allocation NULL-checks or a tape-size guard — both safety nets exist only in the sojourn function (added after its own review) and were never backported, despite these three functions rebuilding their entire elimination tape from scratch on *every call* (an O(n³) cost the sojourn function was specifically redesigned to avoid).
- A validator-only function (`ptd_moment0_grad_theta`, gated behind an opt-in build flag, not reachable from the shipped package) is missing both the MPFR gate and the `coefficients_length==0` NULL-guard that its own stated successor (`ptd_moments_grad_theta`) already has — it predates both fixes and was never re-audited. Low real-world severity (unreachable by default), but exactly the kind of gap a systematic inventory should catch.

→ [`c-functions.md`, Cross-cutting observations`](atlas/exact-fd-atlas-c-functions.md).

### 3.3 `lax.cond`/`vmap`: the rule, quantified, and one more real cost implication

Beyond confirming the original finding (both branches execute under a batched `vmap` predicate), this session's follow-up research quantified it with wall-clock timing (a scalar-predicate `cond` gets a genuine 264–346× skip; an all-`True` `vmap` batch costs the same as an all-`False` one, ~0.9–1.0×, and both pay ~8.5–9× a single expensive call) and found it composes the same way through nesting and through a branch's own `custom_vjp`. A design checklist for future JAX wiring in this codebase is now written down. → [`jax-semantics.md` §1, and the checklist at the end`](atlas/exact-fd-atlas-jax-semantics.md).

### 3.4 `pmf_and_moments_from_graph_multivariate`'s exact path is effectively dead in its real use case

Not just "missing a passthrough kwarg" (as CLAUDE.md already notes) — every per-feature call it makes passes real, non-`None` rewards, which the underlying function's own (correct) dynamic reward-exclusion guard catches on **every single call**. So in its actual 1D/2D-rewards workload, this function is FD-only in practice, not merely "inheriting a default" as CLAUDE.md currently characterizes it. → [`python-wiring.md` §9`](atlas/exact-fd-atlas-python-wiring.md).

### 3.5 An unwired, unverified PDF-gradient function is a landmine for future PMF/PDF gradient work

`ptd_graph_pdf_with_gradient` (`src/c/phasic.c:11805`) computes a PDF gradient via uniformization, has zero callers anywhere in the codebase, is untested, and its own comment admits its sign convention was "empirically determined" rather than derived. Flagged in two historical planning documents as something that must be re-verified before any future PMF/PDF gradient work (the one B3 dimension — PMF/PDF, as opposed to moments/sojourn — that remains completely untouched) builds on it. Not urgent (dead code), but a real trap if someone later wires it up assuming it's already correct. → [`loose-ends-memory.md` §3.1`](atlas/exact-fd-atlas-loose-ends-memory.md), [`loose-ends-docs.md`](atlas/exact-fd-atlas-loose-ends-docs.md).

### 3.6 `reward_visit_probability` is a third undocumented FD-only site

Alongside `moments_from_graph`/`method_of_moments` (which CLAUDE.md already lists as FD-only), `reward_visit_probability`'s `custom_vjp` (`ffi_wrappers.py:1278`) is also 100% FD and also unmentioned in CLAUDE.md's B3 gaps — despite being explicitly named as in-scope-in-principle ("analytical eventually via an adjoint through `ptd_backward_probabilities`") in an older planning document. → [`loose-ends-docs.md`](atlas/exact-fd-atlas-loose-ends-docs.md), [`python-wiring.md` §11`](atlas/exact-fd-atlas-python-wiring.md).

### 3.7 The offset-tape "no-caching" gap is broader than CLAUDE.md currently documents

The current `b3-joint-index-plan.md` explicitly notes that the O(commands) offset-tape-conversion cost (not cached, redone every call) also applies to the shipped `ptd_moments_grad_theta`/`_dph`/`_log` functions, and explicitly asks for this to be added to CLAUDE.md — but only the joint-index section (about its own function) carries this caveat; the moment-gradient section, the largest B3 section, doesn't mention it. → [`loose-ends-docs.md`](atlas/exact-fd-atlas-loose-ends-docs.md).

---

## 4. Lower-severity / informational findings

- **`bffg.py`'s `model`** has *no* differentiation rule at all (not FD, not exact — a raw, undifferentiated FFI call). Currently harmless because `mcmc.py` never differentiates it (confirmed by grep: zero `grad` calls anywhere in that file), but would raise, not silently misbehave, if that ever changed. → [`python-wiring.md` §12`](atlas/exact-fd-atlas-python-wiring.md).
- **Half the model-builder entry points never had B3 applied and likely never will need to**: `pmf_from_graph`, `pmf_from_cpp`, `daisy_chain_joint_probs`, `_daisy_chain_svgd_model` are 100% FD by original design, not oversight (some, like `pmf_from_graph`, are simple enough that FD's mixed-scale defect may matter less — not independently assessed here).
- **Hierarchical/SCC-parallelized graphs' gradient status is unverified and unflagged.** Explicitly listed as a remaining B3 scope item in the original tracking memory, silently dropped from CLAUDE.md's current text. Whether the SCC-composed tape is even the same one the B3 reverse-mode adjoint functions expect to walk has never been checked. → [`loose-ends-memory.md` §3.5`](atlas/exact-fd-atlas-loose-ends-memory.md).
- **The `defect()`/`cdf()` semantic inconsistency** (a foundational ambiguity about "probability mass that never leaves the start vertex," disabled native assert, never re-enabled) sits in the same C engine as every reward/moment/gradient computation — a latent trap for future work on start-absorption-adjacent quantities, not gradient-specific but adjacent.
- **`jax.debug.callback` raises propagate identically to `pure_callback` raises** (same `XlaRuntimeError` wrapping) despite "fire-and-forget" describing only the return value — a non-obvious distinction now written down, both patterns found live and used correctly in `svgd.py`.
- **`use_dyn_ordering` is not part of the on-disk cache's content hash** — very likely benign (elimination order shouldn't affect the numeric result) but an undocumented, untested cross-path invariant of exactly the shape that has caused bugs elsewhere in this codebase.

---

## 5. What's already fine (a lot, per the loose-ends audits)

Both loose-ends streams deliberately verified claims against live code rather than trusting old documents at face value, and found that **most historical "open bugs" from the pre-B3 refactoring audits are already fixed** (F-007/Q7.1 reward transform, several silent-linearization bugs, the tied-slave export bug, the SVGD single-particle-crash issue, the log-mode trace-replay footgun, the epoch-model/LRT nested-test blockers). Reporting these as open would have been false positives — this is itself a useful, if less exciting, finding: the codebase's actual defect rate on *already-flagged* items is low; the gaps that matter are the *unflagged* ones enumerated above.

---

## 6. Suggested next step

This document is research, not a plan — no code has been touched as part of this pass (except the two fixes already shipped earlier: the `lax.cond` de-risk script and the mmap `input_specs` segfault). Given the size of what's surfaced, the natural next step is to decide *priority*, not to act on everything at once. Candidates, roughly in the order §2–§3 above rank them:

1. Resume the paused joint-index `lax.cond`/`vmap` redesign (D6), now correctly scoped given everything above.
2. Address the daisy-chain zero-coverage gap (§2.3) — likely the highest-impact item given it's the default multi-epoch path.
3. Fix the confirmed `moments_from_graph` vmap crash (§2.1) — small, isolated, execution-confirmed.
4. Harden or deprecate the legacy on-disk cache format (§2.2) — security-adjacent, touches a public-facing feature.
5. Characterize and pin the conditioning floor (§2.4) — a documentation/testing task, not a code fix, but closes a real trust gap in the B3 initiative's own flagship claim.
6. Something else — your call.